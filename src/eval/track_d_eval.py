"""Track D: served vs shadow models on three test beds (offline, not served).

SHADOW models are a declared offline augmentation study: they are NEVER served
and no published number changes. Synthetic data + synthetic labels (Zidi et
al. 2023 six-type family) throughout; train/eval salts are separated so the
generator signature learned at train time cannot trivially re-appear at eval.
Spec: docs/superpowers/specs/2026-10-28-track-d-shadow-augmentation-design.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from src.data.ceew_adapter import (build_window, inject_window,  # noqa: E402
                                   read_manifest, trackc_cfg)
from src.data.derive_shadow import (HOOK_FACTOR, N_INJECTED,  # noqa: E402
                                    TRAIN_SALT)
from src.data.merged_adapter import (DEMO_METERS, EVAL_METERS,  # noqa: E402
                                     build_observations, merged_cfg)
from src.data.theft_inject import (PREVALENCE, apply_theft,  # noqa: E402
                                   assign_types, meter_rng)
from src.eval.merged_eval import precision_at_k  # noqa: E402
from src.eval.track_c_eval import _load_windows  # noqa: E402
from src.features.build import build_features  # noqa: E402

SERVED = ROOT / 'artifacts_merged_demo' / 'models.joblib'
SHADOW = ROOT / 'artifacts_shadow' / 'models.joblib'
EVAL_SALT = 7717
RESULTS = ROOT / 'results'
MODELS_ORDER = ['M0', 'M1', 'M2', 'M3', 'M4']
HONESTY = ('SHADOW models — NOT served; synthetic data + synthetic labels '
           '(Zidi family). All published Track A/B/C numbers refer to the '
           'ORIGINAL served models and are unchanged. Generator-signature '
           'caveat applies (train/eval salts separated).')


def _metrics_block(y, score, types=None):
    y = np.asarray(y, int)
    score = np.asarray(score, float)
    keep = np.isfinite(score)
    y, score = y[keep], score[keep]
    if types is not None:
        types = np.asarray(types, object)[keep]
    ok = bool(len(y)) and bool(y.std() > 0)
    block = {'n': int(len(y)), 'positives': int(y.sum()),
             'prevalence': float(y.mean()) if len(y) else None,
             'roc_auc': float(roc_auc_score(y, score)) if ok else None,
             'pr_auc': float(average_precision_score(y, score)) if ok else None,
             'mannwhitney_p_greater': (
                 float(mannwhitneyu(score[y == 1], score[y == 0],
                                    alternative='greater').pvalue)
                 if ok else None),
             'precision_at': {str(k): precision_at_k(score, y, k)
                              for k in (10, 20, 50)},
             'recall_at_20pct_by_type': {}}
    if types is not None and len(score):
        budget = max(1, int(0.2 * len(score)))
        top = set(np.argsort(-score)[:budget].tolist())
        for t in sorted({t for t in types if t}):
            idx = [i for i, tt in enumerate(types) if tt == t]
            block['recall_at_20pct_by_type'][str(t)] = (
                float(np.mean([i in top for i in idx])) if idx else None)
    return block


def _verdict(g1=None, g2=None, g3=None) -> dict:
    out = {}
    if g1 is not None:
        out['G1'] = bool(g1['max_pr_drop'] <= 0.02 and g1['max_roc_drop'] <= 0.02)
    if g2 is not None:
        out['G2'] = bool(g2['best_t56_recall'] >= 0.50)
    if g3 is not None:
        out['G3'] = bool(g3['m3_delta_pr'] > 0 and g3['m4_delta_pr'] > 0)
    return out


def _score_all(feats, models: dict) -> dict:
    out = {}
    for name, model in models.items():
        raw, prob = model.predict(feats)
        out[name] = {'raw': np.asarray(raw, float),
                     'prob': np.asarray(prob, float)}
    return out


def _assessment_slice(cfg: dict) -> tuple[int, int]:
    """The exact assessment interval indices build_features uses
    (baseline capped window on the left, assessment_days on the right)."""
    t = cfg['days'] * 96
    b = min(cfg['baseline_days'], cfg['days'] - 1) * 96
    start = max(b, t - cfg['assessment_days'] * 96)
    return start, t


def _select_eval_thieves(clean_ids: list[str]) -> dict:
    """Exact-k counts and per-type multiset drawn by assign_types (seed-42
    family), but the meters are re-mapped by an EVAL_SALT ranking so the
    injected SET is an independent draw from the train-time one."""
    raw = assign_types(sorted(clean_ids), prevalence=PREVALENCE)
    types = sorted(t for t in raw.values() if t)
    ranked = sorted(sorted(clean_ids),
                    key=lambda mid: (meter_rng(mid, salt=EVAL_SALT).random(), mid))
    return dict(zip(ranked[:len(types)], types))


def _inject_selected(obs, selected: dict) -> None:
    """Zidi per-day injection on obs.kwh over the feature assessment window
    only; missing slots keep their original values and obs.missing is
    untouched (the matrix-side mirror of inject_readings_frame, which
    NaN-preserves on the raw frame)."""
    start, end = _assessment_slice(obs.cfg)
    assert (end - start) % 96 == 0 and end - start > 0
    n_days = (end - start) // 96
    row_of = {str(m): i for i, m in enumerate(obs.meters.meter_id)}
    for mid, theft_type in selected.items():
        row = row_of[str(mid)]
        rng = meter_rng(mid, salt=EVAL_SALT)
        seg = np.asarray(obs.kwh[row, start:end], np.float64)
        miss = obs.missing[row, start:end]
        new = seg.copy()
        for d in range(n_days):
            a, b = d * 96, (d + 1) * 96
            new[a:b] = apply_theft(seg[a:b], theft_type, rng)
        new[miss] = seg[miss]
        obs.kwh[row, start:end] = new


def track_b_bed(gens: dict) -> tuple[dict, object]:
    """Track B replay: original oracle truth on the untouched eval cohort."""
    obs, extras = build_observations('eval', merged_cfg(60), 0)
    oracle = extras['oracle_labels']
    assert list(oracle.meter_id) == list(obs.meters.meter_id), 'label misalignment'
    y = oracle.is_theft.to_numpy(int)
    feats = build_features(obs)
    blocks = {g: {n: _metrics_block(y, sc['prob'])
                  for n, sc in _score_all(feats, md).items()}
              for g, md in gens.items()}
    return blocks, feats


def track_c_bed(gens: dict) -> tuple[dict, dict]:
    """Track C replay: staged CEEW windows + seed-42 injection, pooled like
    track_c_eval, scored for every generation in gens (shared per-window
    features)."""
    manifest = read_manifest()
    types = {k: (int(v.split('_')[1]) if v else None)
             for k, v in manifest['meter_types'].items()}
    windows = _load_windows()
    pool = {g: {n: {'score': [], 'y': [], 'type': []} for n in md}
            for g, md in gens.items()}
    n_skipped = 0
    for wi, win in enumerate(windows):
        cfg = trackc_cfg(win['n_dts'], win['meters_per_dt'])
        rep, info = build_window(win['district'], win, cfg)
        _rep2, row = inject_window(rep, types, win, win_index=wi)
        feats = build_features(_rep2)
        y_meta = {p['meter_id']: p for p in row['positives']}
        y = np.array([1 if m in y_meta else 0 for m in info['meters']])
        if y.sum() == 0 or y.sum() == len(y):
            n_skipped += 1
            continue
        ty = np.array([y_meta.get(m, {}).get('type', 'normal')
                       for m in info['meters']], dtype=object)
        for g, md in gens.items():
            for n, sc in _score_all(feats, md).items():
                p = pool[g][n]
                p['score'].append(sc['prob'])
                p['y'].append(y)
                p['type'].append(ty)
    blocks = {}
    for g, md in gens.items():
        blocks[g] = {}
        for n in md:
            p = pool[g][n]
            blocks[g][n] = (_metrics_block(np.concatenate(p['y']),
                                           np.concatenate(p['score']),
                                           np.concatenate(p['type']))
                            if p['y'] else {'n': 0, 'positives': 0,
                                            'prevalence': None, 'roc_auc': None,
                                            'pr_auc': None,
                                            'mannwhitney_p_greater': None,
                                            'precision_at': {},
                                            'recall_at_20pct_by_type': {}})
    return blocks, {'windows': len(windows), 'skipped': n_skipped,
                    'pooled_windows': len(windows) - n_skipped}


def track_d_bed(gens: dict) -> tuple[dict, dict]:
    """Track D bed: eval cohort + Zidi injection at eval time (EVAL_SALT,
    assessment window only, truth-clean meters only)."""
    obs, extras = build_observations('eval', merged_cfg(60), 0)
    oracle = extras['oracle_labels']
    assert list(oracle.meter_id) == list(obs.meters.meter_id), 'label misalignment'
    mids = [str(m) for m in oracle.meter_id]
    clean_ids = [str(m) for m, theft in zip(oracle.meter_id, oracle.is_theft)
                 if not theft]
    selected = _select_eval_thieves(clean_ids)
    _inject_selected(obs, selected)
    y_new = (oracle.is_theft.to_numpy(bool)
             | np.isin(mids, list(selected))).astype(int)
    types_arr = np.array(
        [f'zidi_t{selected[m]}_injected' if m in selected else None
         for m in mids], dtype=object)
    feats = build_features(obs)
    keep = np.array([m not in selected for m in mids])
    blocks = {}
    for g, md in gens.items():
        scored = _score_all(feats, md)
        blocks[g] = {
            'injected_plus_oracle': {
                n: _metrics_block(y_new, sc['prob'], types_arr)
                for n, sc in scored.items()},
            'clean_noninjected': {
                n: _metrics_block(y_new[keep], sc['prob'][keep])
                for n, sc in scored.items()}}
    meta = {'n_injected_eval': len(selected),
            'prevalence_requested': PREVALENCE,
            'assessment_slice': _assessment_slice(obs.cfg),
            'y_new_positives': int(y_new.sum()),
            'oracle_positives': int(oracle.is_theft.sum()),
            'n_noninjected': int(keep.sum())}
    return blocks, meta


def _delta(sv: dict, sh: dict) -> dict:
    def d(key):
        a, b = sv.get(key), sh.get(key)
        return float(b - a) if a is not None and b is not None else None
    return {'roc_auc': d('roc_auc'), 'pr_auc': d('pr_auc')}


def _flat_deltas(bed: dict) -> dict:
    return {m: _delta(bed['served'][m], bed['shadow'][m])
            for m in MODELS_ORDER if m in bed.get('served', {})}


def run() -> dict:
    assert not (set(DEMO_METERS) & set(EVAL_METERS)), 'demo/eval overlap'
    gens = {'served': joblib.load(SERVED), 'shadow': joblib.load(SHADOW)}
    assert sorted(gens['served']) == MODELS_ORDER, sorted(gens['served'])
    assert sorted(gens['shadow']) == MODELS_ORDER, sorted(gens['shadow'])

    b_blocks, feats_b = track_b_bed(gens)
    try:
        c_blocks, c_meta = track_c_bed(gens)
        c_error = None
    except Exception as exc:                       # report, never fail silently
        c_blocks, c_meta, c_error = {}, {'windows': None}, f'{type(exc).__name__}: {exc}'
    d_blocks, d_meta = track_d_bed(gens)

    deltas = {'track_b': _flat_deltas(b_blocks),
              'track_c': _flat_deltas(c_blocks) if c_error is None else None,
              'track_d_main': {m: _delta(
                  d_blocks['served']['injected_plus_oracle'][m],
                  d_blocks['shadow']['injected_plus_oracle'][m])
                  for m in MODELS_ORDER},
              'track_d_clean_noninjected': {m: _delta(
                  d_blocks['served']['clean_noninjected'][m],
                  d_blocks['shadow']['clean_noninjected'][m])
                  for m in MODELS_ORDER}}

    # --- G1: Track B regression guard (M2/M3/M4, both metrics, drop <= 0.02)
    g1_models = ['M2', 'M3', 'M4']
    g1 = {'per_model': {m: _delta(b_blocks['served'][m], b_blocks['shadow'][m])
                        for m in g1_models}}
    g1['max_pr_drop'] = float(max(
        (b_blocks['served'][m]['pr_auc'] - b_blocks['shadow'][m]['pr_auc'])
        for m in g1_models))
    g1['max_roc_drop'] = float(max(
        (b_blocks['served'][m]['roc_auc'] - b_blocks['shadow'][m]['roc_auc'])
        for m in g1_models))

    # --- G2: Track C blind-spot lift (T5/T6 recall@20%, any shadow model)
    g2 = None
    if c_error is None:
        per_model = {}
        for m, blk in c_blocks.get('shadow', {}).items():
            r = blk.get('recall_at_20pct_by_type', {})
            per_model[m] = {'T5': r.get('theft_5'), 'T6': r.get('theft_6')}
        vals = [v for d in per_model.values() for v in (d['T5'], d['T6'])
                if v is not None]
        if vals:
            g2 = {'best_t56_recall': float(max(vals)), 'per_model': per_model}

    # --- G3: Track D in-domain sanity (shadow > served PR for M3 and M4)
    d_main = {g: d_blocks[g]['injected_plus_oracle'] for g in gens}
    g3 = {'m3_delta_pr': _delta(d_main['served']['M3'], d_main['shadow']['M3'])['pr_auc'],
          'm4_delta_pr': _delta(d_main['served']['M4'], d_main['shadow']['M4'])['pr_auc']}

    gates = {}
    gates.update(_verdict(g1=g1))
    if g2 is not None:
        gates.update(_verdict(g2=g2))
    else:
        gates['G2'] = 'unavailable (Track C bed error)'
    gates.update(_verdict(g3=g3))

    # --- M0 soft comparison (rule identical; Platt refit on shadow data)
    m0_s = np.asarray(gens['served']['M0'].predict(feats_b)[1], float)
    m0_h = np.asarray(gens['shadow']['M0'].predict(feats_b)[1], float)
    m0_soft = {'served_track_b': b_blocks['served']['M0'],
               'shadow_track_b': b_blocks['shadow']['M0'],
               'prob_corr': float(np.corrcoef(m0_s, m0_h)[0, 1]),
               'note': ('M0 rule formula is identical by construction; only the '
                        'Platt calibration was refit on shadow training data.')}

    sv_sel = json.loads((ROOT / 'artifacts_merged_demo' / 'selected_model.json')
                        .read_text(encoding='utf-8'))
    sh_sel = json.loads((ROOT / 'artifacts_shadow' / 'selected_model.json')
                        .read_text(encoding='utf-8'))
    protocol = {'train_salt': TRAIN_SALT, 'eval_salt': EVAL_SALT,
                'trackc_seed': 42,
                'trackc_windows': c_meta, 'trackc_error': c_error,
                'n_injected_train_pack': N_INJECTED, 'hook_factor': HOOK_FACTOR,
                'hook_note': ('DT-hooks live in the shadow TRAIN pack only; '
                              'the Track D eval bed is meter-level'),
                'track_d_injection': ('eval cohort, truth-clean meters, '
                                      'assessment window only, EVAL_SALT'),
                'demo_eval_disjoint': True,
                'served_config_hash': sv_sel.get('config_hash'),
                'shadow_config_hash': sh_sel.get('config_hash'),
                'served_artifacts_untouched': True}
    limitations = [
        'Shadow models are offline only: never served; published Track A/B/C '
        'numbers unchanged.',
        'All labels are synthetic (generator archetypes / Zidi injection); no '
        'real theft labels are involved.',
        'Generator-signature caveat: train and eval salts are separated, but '
        'the generator family is shared; a family-level match is not proof of '
        'real-theft detection.',
        'Track C bed inherits virtual DTs, star topology, single-phase, no '
        'weather; meters repeat across windows (pooled instances are not '
        'independent).',
        'M0 is a soft comparison row: the rule is identical, only the '
        'calibration differs, so probability agreement is below 1 by design.',
        'Track D bed main labels are oracle OR eval-time injection; the '
        'clean-subset block uses oracle labels on non-injected meters only.',
    ]
    return {
        'track': 'D — shadow-augmented retraining vs served models (offline)',
        'honesty': HONESTY,
        'protocol': protocol,
        'beds': {'track_b': b_blocks,
                 'track_c': c_blocks if c_error is None else {'error': c_error},
                 'track_d': d_blocks},
        'bed_meta': {'track_c': c_meta, 'track_d': d_meta},
        'deltas_shadow_minus_served': deltas,
        'gate_inputs': {'G1': g1, 'G2': g2, 'G3': g3},
        'gates': gates,
        'm0_soft_comparison': m0_soft,
        'limitations': limitations,
    }


def _f(x, nd=3) -> str:
    return '—' if x is None else f'{x:.{nd}f}'


def _paired_table(bed: dict) -> str:
    gens = [g for g in ('served', 'shadow') if g in bed]
    if len(gens) < 2:
        return '_bed unavailable_'
    rows = []
    for m in MODELS_ORDER:
        sv, sh = bed['served'].get(m), bed['shadow'].get(m)
        if sv is None or sh is None:
            continue
        dr = sh['roc_auc'] - sv['roc_auc'] if sv['roc_auc'] is not None \
            and sh['roc_auc'] is not None else None
        dp = sh['pr_auc'] - sv['pr_auc'] if sv['pr_auc'] is not None \
            and sh['pr_auc'] is not None else None
        rows.append(f'| {m} | {_f(sv["roc_auc"])} | {_f(sh["roc_auc"])} | '
                    f'{_f(dr)} | {_f(sv["pr_auc"])} | {_f(sh["pr_auc"])} | '
                    f'{_f(dp)} |')
    return ('| Model | served ROC | shadow ROC | ΔROC | served PR | shadow PR | ΔPR |\n'
            '|---|---|---|---|---|---|---|\n' + '\n'.join(rows))


def write_report(result: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / 'track_d_augmentation.json').write_text(
        json.dumps(result, indent=2), encoding='utf-8')

    b = result['beds']['track_b']
    c = result['beds']['track_c']
    d = result['beds']['track_d']
    gates = result['gates']
    gi = result['gate_inputs']
    m0 = result['m0_soft_comparison']
    c_prev = (b['served']['M0'].get('prevalence'))
    g2 = gi['G2']
    g2_rows = '\n'.join(
        f'| {m} | {_f(v["T5"], 2)} | {_f(v["T6"], 2)} |'
        for m, v in sorted((g2 or {}).get('per_model', {}).items()))
    g1_rows = '\n'.join(
        f'| {m} | {_f(v["pr_auc"])} | {_f(v["roc_auc"])} |'
        for m, v in sorted(gi['G1']['per_model'].items()))
    g3 = gi['G3']
    err = result['protocol'].get('trackc_error')
    track_c_md = (f'Pooled over {result["bed_meta"]["track_c"]["pooled_windows"]} '
                  f'of {result["bed_meta"]["track_c"]["windows"]} windows '
                  f'(skipped {result["bed_meta"]["track_c"]["skipped"]}).\n\n'
                  f'{_paired_table(c)}') if err is None else f'**Error: {err}**'
    md = f"""# Track D — shadow-augmented retraining study (served vs shadow)

**{result['honesty']}**

## Protocol

- Salts: train {result['protocol']['train_salt']}, eval {result['protocol']['eval_salt']}, Track C seed {result['protocol']['trackc_seed']}; demo∩eval disjoint: {result['protocol']['demo_eval_disjoint']}
- Shadow train pack: {result['protocol']['n_injected_train_pack']} injected meters + 2 DT-hooks (factor {result['protocol']['hook_factor']}) — {result['protocol']['hook_note']}
- Track D eval bed: {result['protocol']['track_d_injection']} ({result['bed_meta']['track_d']['n_injected_eval']} meters, {result['bed_meta']['track_d']['y_new_positives']} positives of {result['bed_meta']['track_d']['n_injected_eval'] + result['bed_meta']['track_d']['n_noninjected']})
- Config hashes: served `{result['protocol']['served_config_hash']}` vs shadow `{result['protocol']['shadow_config_hash']}`; served artifacts untouched: {result['protocol']['served_artifacts_untouched']}

## Test bed 1 — Track B replay (original oracle truth; prevalence {c_prev:.4f})

{_paired_table(b)}

## Test bed 2 — Track C replay (real CEEW telemetry, injected labels)

{track_c_md}

## Test bed 3 — Track D (eval cohort + eval-time Zidi injection)

Main (oracle OR injected labels):

{_paired_table({g: d[g]['injected_plus_oracle'] for g in d})}

Clean subset (oracle labels, non-injected meters only):

{_paired_table({g: d[g]['clean_noninjected'] for g in d})}

## Gates (pre-registered)

| Gate | Verdict | Inputs |
|---|---|---|
| G1 regression guard (≤0.02 drop, M2/M3/M4, both metrics) | {gates['G1']} | max PR drop {_f(gi['G1']['max_pr_drop'])}, max ROC drop {_f(gi['G1']['max_roc_drop'])} |
| G2 blind-spot lift (T5/T6 ≥0.50, any shadow model) | {gates['G2']} | best recall {('unavailable' if g2 is None else _f(g2['best_t56_recall'], 2))} |
| G3 in-domain sanity (shadow PR > served, M3 and M4) | {gates['G3']} | M3 Δ {_f(g3['m3_delta_pr'])}, M4 Δ {_f(g3['m4_delta_pr'])} |

G1 per-model drops (served − shadow, positive = regression):

| Model | ΔPR | ΔROC |
|---|---|---|
{g1_rows}

Track C shadow per-type recall@20% (T5/T6):

| Model (shadow) | T5 | T6 |
|---|---|---|
{g2_rows or '_unavailable_'}

## M0 soft comparison

- Track B PR: served {_f(m0['served_track_b']['pr_auc'])} vs shadow {_f(m0['shadow_track_b']['pr_auc'])}; ROC: served {_f(m0['served_track_b']['roc_auc'])} vs shadow {_f(m0['shadow_track_b']['roc_auc'])}; probability correlation {m0['prob_corr']:.3f}
- {m0['note']}

## Limitations

{chr(10).join('- ' + x for x in result['limitations'])}
"""
    lowered = md.lower()
    assert 'accuracy' not in lowered, 'forbidden claim word: accuracy'
    assert 'field-performance' not in lowered, 'forbidden claim: field-performance'
    (RESULTS / 'track_d_augmentation.md').write_text(md, encoding='utf-8')


if __name__ == '__main__':
    write_report(run())
    print('track D report written to results/track_d_augmentation.{md,json}')
