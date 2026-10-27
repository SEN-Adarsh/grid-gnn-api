"""Track C: real CEEW telemetry + injected theft labels, evaluated with the
frozen served models M0-M4.

REAL telemetry (Mathura/Bareilly, CEEW/Harvard, CC0) with SYNTHETIC injected
labels (Zidi et al. 2023 six-type generator + our DT-hook extension). NOT real
theft labels. Virtual DTs, star topology. See
docs/superpowers/specs/2026-10-27-track-c-ceew-injection-design.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data.ceew_adapter import (CEEW_BASE, PREVALENCE, apply_theft,
                                   build_window, inject_window, load_district,
                                   meter_rng, read_manifest, trackc_cfg,
                                   write_manifest)
from src.eval.merged_eval import balance_scores, precision_at_k
from src.features.build import build_features

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'results'
MODELS = ROOT / 'artifacts_merged_demo' / 'models.joblib'


def _load_windows() -> list[dict]:
    return json.loads((CEEW_BASE / 'windows.json').read_text(encoding='utf-8'))['windows']


def run() -> dict:
    manifest = read_manifest()
    types = {k: (int(v.split('_')[1]) if v else None)
             for k, v in manifest['meter_types'].items()}
    models = joblib.load(MODELS)
    windows = _load_windows()

    pool = {name: {'score': [], 'y': [], 'type': []} for name in models}
    per_window, window_rows, skipped_auc, hook_results = [], [], [], []
    for wi, win in enumerate(windows):
        cfg = trackc_cfg(win['n_dts'], win['meters_per_dt'])
        obs, info = build_window(win['district'], win, cfg)
        rep, row = inject_window(obs, types, win, win_index=wi)
        window_rows.append(row)

        # DT-hook detection via balance residual (merged_eval convention)
        if row['hooks']:
            bal = balance_scores(rep)
            hooked = [h['dt_id'] for h in row['hooks']]
            others = [g for g in range(len(bal)) if g not in hooked]
            auc = float(roc_auc_score([1] * len(hooked) + [0] * len(others),
                                      list(bal[hooked]) + list(bal[others]))) \
                if others else None
            rank1 = int(np.argmax(bal) in hooked)
            hook_results.append({'window_id': win['window_id'], 'hooked': hooked,
                                 'balance_auc': auc, 'hooked_in_rank1': rank1})

        feats = build_features(rep)
        y_meta = {p['meter_id']: p for p in row['positives']}
        y = np.array([1 if m in y_meta else 0 for m in info['meters']])
        if y.sum() == 0 or y.sum() == len(y):
            skipped_auc.append({'window_id': win['window_id'],
                                'reason': 'no_class_variation',
                                'n_pos': int(y.sum())})
            continue
        wres = {'window_id': win['window_id'], 'n_meters': int(len(y)),
                'n_pos': int(y.sum()), 'models': {}}
        for name, model in models.items():
            score, _prob = model.predict(feats)
            score = np.asarray(score, float)
            s = pool[name]
            s['score'].append(score)
            s['y'].append(y)
            s['type'].append(np.array([y_meta.get(m, {}).get('type', 'normal')
                                       for m in info['meters']]))
            k5, k10 = 5, min(10, max(1, len(y) // 4))
            wres['models'][name] = {
                'roc_auc': float(roc_auc_score(y, score)),
                'pr_auc': float(average_precision_score(y, score)),
                'precision_at_k': [precision_at_k(score, y, k) for k in (k5, k10)],
            }
        per_window.append(wres)

    # persist window rows (positives + hooks) back into the manifest
    write_manifest(types, window_rows, manifest.get('skipped_windows', []))

    # ---- pooled metrics (meter-window instances; same meter may repeat) ----
    pooled = {}
    for name in models:
        if not pool[name]['y']:
            pooled[name] = {'n_meter_windows': 0, 'prevalence': None,
                            'roc_auc': None, 'pr_auc': None,
                            'mannwhitney_p_greater': None, 'per_type': {}}
            continue
        sc = np.concatenate(pool[name]['score'])
        y = np.concatenate(pool[name]['y'])
        ty = np.concatenate(pool[name]['type'])
        _, p_mwu = mannwhitneyu(sc[y == 1], sc[y == 0], alternative='greater')
        cut = np.quantile(sc, 0.80)
        neg = sc[~(y == 1)]
        per_type = {}
        for tname in [f'theft_{i}' for i in range(1, 7)]:
            mask = ty == tname
            if mask.sum() == 0:
                per_type[tname] = {'positives': 0}
                continue
            per_type[tname] = {'positives': int(mask.sum()),
                               'recall_at_top20pct': float((sc[mask] >= cut).mean()),
                               'mean_score_percentile': float((sc[mask] > neg).mean())}
        pooled[name] = {
            'n_meter_windows': int(len(y)),
            'prevalence': float(y.mean()),
            'roc_auc': float(roc_auc_score(y, sc)),
            'pr_auc': float(average_precision_score(y, sc)),
            'mannwhitney_p_greater': float(p_mwu),
            'per_type': per_type,
        }

    periodicity = periodicity_report(types)
    return {
        'track': 'C — real CEEW telemetry with injected labels',
        'honesty': ('REAL telemetry (CEEW Mathura/Bareilly, CC0) with SYNTHETIC '
                    'injected labels (Zidi et al. 2023 + our DT-hook extension). '
                    'NOT real theft labels. Virtual DTs, star topology, single-phase, '
                    'no weather. No field-performance claim.'),
        'dataset': {'source': 'CEEW/Harvard Dataverse doi:10.7910/DVN/GOCHJH '
                              '(Kaggle mirror, CC0)', 'interval': '15 min (resampled from 3 min)',
                    'districts': ['BR', 'MH'], 'seed': 42, 'prevalence': PREVALENCE},
        'n_windows': len(per_window), 'skipped_windows': skipped_auc,
        'per_window': per_window,
        'pooled': pooled,
        'dt_hook': {'windows_with_hook': len(hook_results),
                    'hooked_rank1_rate': (float(np.mean([h['hooked_in_rank1']
                                                          for h in hook_results]))
                                          if hook_results else None),
                    'mean_balance_auc': (float(np.nanmean([h['balance_auc'] for h in hook_results
                                                           if h['balance_auc'] is not None]))
                                         if hook_results else None),
                    'detail': hook_results,
                    'note': 'DT-hook is our extension beyond Zidi et al.'},
        'periodicity': periodicity,
        'protocol': {'dt_input_from_true_loads': True,
                     'theft_touches_reported_only': True,
                     'frozen_served_models': sorted(models),
                     'features_unchanged': 'src/features/build.py (pinned)'},
    }


def periodicity_report(types: dict) -> dict:
    """Paper-1 lens: week-over-week Pearson correlation of daily totals over the
    full staged history, clean vs injected (theft applied per complete day at
    slot level). Zheng et al. IEEE TII: normal > 0.8, thieves < 0.7."""
    import pandas as pd
    out = {'note': 'WoW-PCC of daily totals (Zheng et al. IEEE TII: '
                   'normal > 0.8, thieves < 0.7)'}
    try:
        stats = {'clean': [], 'injected': []}
        per_type = {}
        for district in ('BR', 'MH'):
            base = load_district(district)
            for mid, t in sorted(types.items()):
                if not mid.startswith(f'CEEW-{district}'):
                    continue
                ser = base[base.meter_id == mid].sort_values('ts')
                if ser.empty:
                    continue
                rng = meter_rng(mid, salt=999)
                clean_d, inj_d = [], []
                for _day, grp in ser.groupby(ser.ts.dt.date):
                    vec = grp.kwh.to_numpy(np.float64)
                    if t and len(vec) == 96:      # complete days only
                        inj = apply_theft(vec, t, rng)
                    else:
                        inj = vec
                    clean_d.append(vec.sum())
                    inj_d.append(inj.sum())
                n = (len(clean_d) // 7) * 7
                if n < 14:
                    continue
                wc = np.asarray(clean_d[:n]).reshape(-1, 7)
                wi = np.asarray(inj_d[:n]).reshape(-1, 7)
                with np.errstate(invalid='ignore'):
                    pcc_c = float(np.nanmean(
                        [np.corrcoef(wc[k], wc[k + 1])[0, 1] for k in range(len(wc) - 1)]))
                    pcc_i = float(np.nanmean(
                        [np.corrcoef(wi[k], wi[k + 1])[0, 1] for k in range(len(wi) - 1)]))
                stats['clean'].append(pcc_c)
                stats['injected'].append(pcc_i)
                if t:
                    per_type.setdefault(f'theft_{t}', {'clean': [], 'injected': []})
                    per_type[f'theft_{t}']['clean'].append(pcc_c)
                    per_type[f'theft_{t}']['injected'].append(pcc_i)
        finite_c = np.array([x for x in stats['clean'] if np.isfinite(x)])
        finite_i = np.array([x for x in stats['injected'] if np.isfinite(x)])
        out['n_meters'] = len(stats['clean'])
        out['mean_pcc_clean'] = float(np.mean(finite_c)) if len(finite_c) else None
        out['mean_pcc_injected'] = float(np.mean(finite_i)) if len(finite_i) else None
        out['share_pcc_gt_0.8_clean'] = float((finite_c > 0.8).mean()) if len(finite_c) else None
        out['share_pcc_gt_0.8_injected'] = float((finite_i > 0.8).mean()) if len(finite_i) else None
        out['per_type'] = {}
        for k, v in per_type.items():
            fc = np.array([x for x in v['clean'] if np.isfinite(x)])
            fi = np.array([x for x in v['injected'] if np.isfinite(x)])
            out['per_type'][k] = {'n': len(v['clean']),
                                  'mean_clean': float(np.mean(fc)) if len(fc) else None,
                                  'mean_injected': float(np.mean(fi)) if len(fi) else None}
    except Exception as exc:                    # never fail the report on the lens
        out['error'] = f'{type(exc).__name__}: {exc}'
    return out


def write_report(result: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / 'track_c_real_up.json').write_text(json.dumps(result, indent=2),
                                                   encoding='utf-8')
    p = result['pooled']
    rows = '\n'.join(
        f"| {name} | {m['roc_auc']:.3f} | {m['pr_auc']:.3f} | "
        f"{m['mannwhitney_p_greater']:.2e} | "
        + ' | '.join(f"{m['per_type'].get(t, {}).get('recall_at_top20pct', float('nan')):.2f}"
                     for t in [f'theft_{i}' for i in range(1, 7)]) + ' |'
        for name, m in p.items() if m['roc_auc'] is not None)
    md = f"""# Track C — real CEEW telemetry with injected theft labels

**{result['honesty']}**

## Setup

- Dataset: {result['dataset']['source']}; {result['dataset']['interval']}; districts {result['dataset']['districts']}
- Injection: Zidi et al. (2023) six-type generator, prevalence {result['dataset']['prevalence']:.0%}, seed {result['dataset']['seed']}; DT-hook = our extension
- Windows: {result['n_windows']} valid 14-day monthly windows (skipped: {len(result['skipped_windows'])})
- Models: frozen served artifacts ({', '.join(result['protocol']['frozen_served_models'])}); features unchanged (pinned build.py)
- Physics rule: dt input from TRUE loads + sweep loss; theft touches reported readings only

## Pooled metrics (meter-window instances)

| Model | ROC-AUC | PR-AUC | MW p | T1 | T2 | T3 | T4 | T5 | T6 |
|---|---|---|---|---|---|---|---|---|---|
{rows}

(T1–T6 = per-type recall at top-20% inspection budget; T6 is the type Zidi et al.'s RF caught at only 3.1%.)

## DT-hook detection (our extension)

- Windows with hook: {result['dt_hook']['windows_with_hook']}
- Hooked DT ranked #1 by balance residual: {result['dt_hook']['hooked_rank1_rate']}
- Mean balance AUC (hooked vs clean DTs): {result['dt_hook']['mean_balance_auc']}

## Periodicity validation (Zheng et al. IEEE TII lens)

- Meters: {result['periodicity'].get('n_meters')}; mean WoW-PCC clean {result['periodicity'].get('mean_pcc_clean')}, injected {result['periodicity'].get('mean_pcc_injected')}
- Share with PCC > 0.8: clean {result['periodicity'].get('share_pcc_gt_0.8_clean')}, injected {result['periodicity'].get('share_pcc_gt_0.8_injected')}

## Limitations

Injected labels are synthetic (not real theft); virtual DTs with star topology;
single-phase; neutral temperature (no weather); meters repeat across windows
(pooled instances are not independent); small cohort.
"""
    (RESULTS / 'track_c_real_up.md').write_text(md, encoding='utf-8')


if __name__ == '__main__':
    write_report(run())
    print('track C report written to results/track_c_real_up.{md,json}')
