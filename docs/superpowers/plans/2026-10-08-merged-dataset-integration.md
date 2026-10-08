# Merged-Dataset Integration Plan (WP-14)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `synthetic_india_v2` (the merged 2,500-meter dataset) the project's final primary synthetic dataset, integrated into the inherited Grid-GNN stack under the audit pack's protocol, with an inspection-supervised headline evaluation.

**Architecture:** A new adapter module (`src/data/merged_adapter.py`) converts the merged parquet tables into the inherited `Observations` interface so the whole existing stack (features, models, eval, dashboard) runs unchanged. A new eval module (`src/eval/merged_eval.py`) produces the two evaluation tracks (inspection-supervised headline, declared oracle secondary). A new app context module (`app/merged_context.py`) gives the API/dashboard a replay context on the 14-day demo cohort without touching hash-pinned inherited files. The 60-day evaluation uses only the 2,000-meter added cohort (complete coverage); the 500-meter/14-day original cohort is demo-only. The 8 quarantined inspections are excluded from labels. `event_flags` and `feeder_id`/`pole_id` are excluded from predictors per protocol.

**Tech Stack:** Python 3.12 `.venv` at `D:\rodic\inherited\.venv` (numpy 2.5.3, pandas 2.3.3, pyarrow, scikit-learn 1.8.0). PowerShell 5.1 (no `&&`; chain with `;`). Background jobs must redirect child output to files.

**Spec:** `D:\rodic\merged_audit\Merged_Dataset_Audit_Pack\` (README.txt, recommended_protocol.json, merged_audit.json, inspection_rows_to_quarantine.csv, recommended_meter_splits.csv, recommended_dt_splits.csv) + `D:\rodic\Theft_Dataset_Comparison_Final.html` + `D:\rodic\merged_data\README.md`. User decisions (2026-10-08): merged dataset replaces WP-13 synth_v2 packs as primary; headline = inspection-supervised; truth used for declared oracle evaluation only.

## Global Constraints

- Work dir `D:\rodic\inherited`; python is `D:\rodic\inherited\.venv\Scripts\python.exe`.
- Data root `D:\rodic\merged_data\data` (extracted from `C:\Users\haris\Downloads\synth_v2_merged.zip`, SHA-256 `57F8C31B71DF3D45EB872A8BC2A646325A07213BBBA7ABFF32E3B4A9904335E6`).
- `D:\rodic\inherited` is NOT a git repo: no commits; each task ends with a green pytest gate instead.
- Pinned inherited files (`src/sim/simulator.py`, `run_all.py`, `src/common.py`, `app/api.py`, `app/scoring.py`, hash-pinned by `tests/test_synth_v2.py::test_backward_compat`) must NOT be modified. New code goes in new files. If a pinned file seems necessary, stop and re-plan.
- Audit protocol (verbatim obligations): exclude from predictors: `labels_observed.*`, `inspection_log.outcome`, `inspection_log.triggered_by`, `truth/*`, `data_source`, `meter_id_as_number`, `cohort`, `partition`, `event_flags`. Quarantine the 8 inconsistent positive inspections; never relabel them `no_theft`. Never treat uninspected/not_accessible as negatives. Absence after day 14 for the original cohort is outside coverage — never zeros, never "missing packet". Truth directory is oracle-evaluation only; if used for training, declare a separate fully-synthetic supervised experiment (we do NOT — inspection-supervised is the headline).
- Claims wording: no "realistic Indian prevalence/field performance" claims anywhere; every number carries provenance `SYNTHETIC (synthetic_india_v2)` and the caveat that inspections are sparse, selected, and biased.
- WP-13 synth_v2 packs are demoted to "scenario design reference" — documented, not deleted.

---

### Task 0: Recon gates (read-only, before any code)

**Files:** none created; findings recorded in the plan-execution log.

- [ ] **Step 1: Check whether `build_features` consumes `obs.events`**

Grep `src/features/build.py` for `events` and `forbidden`. If events feed any feature, the adapter will supply `events = np.zeros((n, steps), dtype=np.uint8)` (protocol: events are not predictors) and this decision goes in the docs task. Record the exact `meters` DataFrame columns `build_features` requires (expected: `meter_id`, `dt_id`, `category`, `phase`, `three_phase` — verify, don't assume).

- [ ] **Step 2: Check M0 baseline signature**

Read `src/models/baselines.py`: confirm how M0 (own-history residual) is constructed and scored (`fit`/`score` signatures, whether it needs labels). Record the call pattern the eval task will reuse.

- [ ] **Step 3: Check app context format**

Read `app/api.py` + `app/scoring.py`: how the replay context is stored (origin date, customer payload, artifact paths). Record what `app/merged_context.py` must produce. Do not modify these files.

- [ ] **Step 4: Check `Topology` dataclass fields**

Read `src/sim/physics.py` `Topology` (fields + how `build_features`/DT estimator consume `obs.topologies`). Decide: construct per-DT Topology from `topology_edges.parquet` (r/x/length per edge) or reuse default topology with documented caveat. Prefer real edges if fields allow.

### Task 1: `src/data/merged_adapter.py` — loading + audit protocol

**Files:**
- Create: `src/data/__init__.py` (if absent), `src/data/merged_adapter.py`
- Test: `tests/test_merged_adapter.py`

**Interfaces (produced, consumed by Tasks 2–4):**
- `MERGED_BASE: Path` (= `D:/rodic/merged_data/data`), `ZIP_SHA256: str`
- `QUARANTINE_IDS: frozenset[str]` — the 8 meter ids loaded from `D:/rodic/merged_audit/Merged_Dataset_Audit_Pack/inspection_rows_to_quarantine.csv`
- `EVAL_METERS: list[str]` (2,000 added-cohort ids), `DEMO_METERS: list[str]` (500 original-cohort ids) — split by `days_available` from `recommended_meter_splits.csv` (5,760 vs 1,344)
- `load_observable() -> dict[str, pd.DataFrame]` — meters, inspection_log (quarantine rows REMOVED, count asserted == 203), calendar, weather, outage_log_observed, topology_edges
- `load_readings(meter_ids: list[str] | None, columns: list[str] | None) -> pd.DataFrame` — long format; `ts` kept as tz-aware `Asia/Kolkata`
- `ID_MAPS -> tuple[dict[str,int], dict[str,int]]` — meter_id→row idx, gis_dt_id→dt idx (0..49, string-sorted)
- `PARTITION: np.ndarray` — (50,) `train/val/test` by cohort: DTs of batches 1+2 → train, batch 3 → val, batch 4 → test (from `recommended_meter_splits.csv` `partition` column mapped through each DT's meters; cohort 0 = `demo`)

- [ ] **Step 1: Write failing tests**

```python
import numpy as np, pandas as pd, pytest
from src.data.merged_adapter import (MERGED_BASE, QUARANTINE_IDS, EVAL_METERS,
    DEMO_METERS, load_observable, load_readings, PARTITION)

def test_counts_match_audit():
    obs = load_observable()
    assert len(obs['meters']) == 2500 and len(obs['inspection_log']) == 195
    r = load_readings(columns=['meter_id'])
    assert len(r) == 12192000

def test_quarantine_excluded_and_never_relabelled():
    insp = load_observable()['inspection_log']
    assert set(QUARANTINE_IDS).isdisjoint(set(insp.meter_id))
    assert (insp.outcome == 'theft_found').sum() == 6      # 14 found - 8 quarantined
    assert (insp.outcome == 'no_theft').sum() == 175

def test_cohorts_by_coverage():
    assert len(EVAL_METERS) == 2000 and len(DEMO_METERS) == 500
    assert set(EVAL_METERS).isdisjoint(DEMO_METERS)

def test_partition_disjoint_by_dt():
    assert len(PARTITION) == 50
    assert (PARTITION == 'train').sum() == 20 and (PARTITION == 'val').sum() == 10 \
        and (PARTITION == 'test').sum() == 10

def test_readings_grid_and_tz():
    r = load_readings(EVAL_METERS[:10], columns=['meter_id', 'ts', 'kwh', 'is_missing'])
    assert str(r.ts.dtype.tz) == 'Asia/Kolkata'
    assert r.groupby('meter_id').size().nunique() == 1 and r.is_missing.sum() >= 0
```

- [ ] **Step 2: Run tests, verify failure** — `& .venv\Scripts\python.exe -m pytest tests/test_merged_adapter.py -q` → ImportError.

- [ ] **Step 3: Implement the module.** Key points: read `recommended_meter_splits.csv` once; `days_available` 5760→EVAL, 1344→DEMO; quarantine filter = `~inspection_log.meter_id.isin(QUARANTINE_IDS)`; `load_readings` uses `pyarrow.parquet.ParquetFile.read(filters=[('meter_id','in',list)])` (or full read + filter when None) so the eval-cohort pivot in Task 2 never materialises all 12.19M rows with columns it doesn't need; assert `(insp.outcome=='theft_found').sum()==6` inside `load_observable` (fail loud, not silent).

- [ ] **Step 4: Run tests → all green.**

### Task 2: `build_observations()` — inherited `Observations` interface

**Files:**
- Modify: `src/data/merged_adapter.py` (append)
- Test: `tests/test_merged_adapter.py` (append)

**Interfaces:**
- `build_observations(cohort: str, cfg: dict, seed: int = 0) -> tuple[Observations, dict[str, pd.DataFrame]]` — cohort `'eval'` (2,000 m, 40 DTs, 60 days) or `'demo'` (500 m, 10 DTs, 14 days). Returns `(obs, extras)` where `extras` carries `oracle_labels` (truth_meters joined to row idx: `is_theft`, `family`=archetype, `theft_start`), `inspection_labels` (cleaned log → row idx, `outcome`, `inspected_on`), `coverage_ok: bool`.
- Dense arrays: `kwh/kvarh/voltage` (n,steps) float32, `missing` bool, `events` uint8 (zeros — protocol), `dt_kwh/dt_kvarh` from `dt_readings`, `dt_voltage`, `dt_currents` (n_dts, steps, 3 from `i_a/i_b/i_c`), `temperature` from `weather.temperature_c` expanded to steps, `topologies` per Task-0 decision, `split = PARTITION` over the cohort's DTs.
- Missing semantics: eval cohort coverage is complete → `missing` all False (assert). Demo cohort: only 1,344 intervals exist per meter; the dense matrix covers 14 days only (`steps = 1344`), so no post-day-14 zeros ever exist (spec obligation satisfied by construction; assert `steps == 1344`).
- `cfg`: built by `merged_cfg(days)` — starts from `configs/final.yaml` via `load_config`, overrides `n_dts/meters_per_dt/days` to match reality; documented that remaining generator keys are inert for adapter data.

- [ ] **Step 1: Failing tests**

```python
from src.data.merged_adapter import build_observations, merged_cfg
from src.features.build import build_features, FORBIDDEN
from src.common import load_config

def test_eval_observations_shape():
    cfg = merged_cfg(60)
    obs, extras = build_observations('eval', cfg)
    assert obs.kwh.shape == (2000, 5760) and obs.dt_kwh.shape == (40, 5760)
    assert obs.missing.sum() == 0 and obs.events.sum() == 0
    assert (extras['oracle_labels'].is_theft.sum() > 0)

def test_features_run_and_no_forbidden_names():
    cfg = merged_cfg(60)
    obs, extras = build_observations('eval', cfg)
    feats = build_features(obs)
    names = set(feats.columns) if hasattr(feats, 'columns') else {f['name'] for f in feats}
    assert not any(any(tok in str(n).lower() for tok in FORBIDDEN) for n in names)

def test_demo_steps_14d():
    obs, extras = build_observations('demo', merged_cfg(14))
    assert obs.kwh.shape == (500, 1344)
```

- [ ] **Step 2: Run → fail. Step 3: implement** (pivot via `df.pivot_table(index='meter_id', columns='ts', values='kwh')` on the filtered long frame; reindex to sorted unique ts; fill `kwh` 0 where `is_missing` only). **Step 4: green.**

### Task 3: `src/eval/merged_eval.py` — two evaluation tracks

**Files:**
- Create: `src/eval/merged_eval.py`
- Test: `tests/test_merged_eval.py`
- Outputs: `results/merged_eval.json`, `results/merged_eval.md`

**Interfaces:**
- `score_meters(obs) -> np.ndarray` — meter anomaly score reusing the Task-0-recorded M0 pattern (own-history residual); higher = more anomalous.
- `score_dt_balance(obs) -> np.ndarray` — per-DT `(dt_kwh − Σ kwh_over_covered_meters) / dt_kwh` per interval, aggregated (mean of top decile).
- `run_evaluation(obs, extras) -> dict` — inspection track: rank of each of the 6 `theft_found` meters among 2,000, enrichment@25/@50, Mann-Whitney U of scores (found vs 175 no_theft), all with n and caveats; oracle track (declared): ROC-AUC + PR-AUC of `score_meters` vs `extras['oracle_labels'].is_theft`, per-archetype recall@50, DT-balance AUC vs direct_hooking DT membership; DT-level hooking note (hooking is invisible at meters — balance is the only honest signal).
- Both tracks read labels ONLY from `extras` — never from `data/truth` inside the scoring path.

- [ ] **Step 1: Failing tests** — small synthetic Observations built inline (6 DTs × 20 m × 7 days, one meter with a scaled-down residual) asserting: `score_meters` ranks the anomalous meter first; `run_evaluation` returns both tracks with expected keys; json/md files written when `main()` runs against the real cohort (marked `@pytest.mark.slow`, run explicitly).
- [ ] **Step 2: fail → Step 3: implement → Step 4: green.**
- [ ] **Step 5: Run the real evaluation** (`python -m src.eval.merged_eval`), background-safe (redirect to `results/merged_eval_run.log`), verify `results/merged_eval.md` renders both tracks with caveat language and `SYNTHETIC (synthetic_india_v2)` provenance.

### Task 4: `app/merged_context.py` — dashboard/API demo context

**Files:**
- Create: `app/merged_context.py`
- Test: `tests/test_merged_context.py`

**Interfaces:** builds the stored-customer replay context Task-0 documented (origin `2025-10-01 00:00 IST`, demo cohort, int DT map), trains/snapshots the residual artifact on the demo cohort (oracle/inspections never read), exposes the same call signature `app/scoring.py` expects. Smoke test: score 3 demo meters end-to-end without network. Pinned files untouched.

- [ ] Steps: failing smoke test → implement → green → dashboard manual smoke (`streamlit run app/dashboard.py` if present, else API `/score`).

### Task 5: Documentation + claims hygiene

**Files (all edits additive):**
- `docs/HANDOFF_LOG.md` — WP-14 section: merged = final primary dataset (zip SHA-256), audit protocol adopted, quarantine applied, events excluded, WP-13 packs demoted to scenario reference, pinned-file policy held.
- `docs/ASSUMPTIONS.md` — addendum: merged dataset's parameters are third-party SYNTHETIC assumptions; our adopted policies (inspection-supervised headline, oracle declared, coverage semantics, split proposal is not a trained/evaluated split).
- `results/claims_register.csv` — rows: inspection-track enrichment (provenance SYNTHETIC, n=6 caveat), oracle AUC/PR-AUC (declared oracle-only), DT-balance hooking signal; NO field-realism claims.
- `D:\rodic\PROJECT_REPORT.md` — replace the WP-13-primary dataset section with merged-as-final + pointer to `results/merged_eval.md`.

- [ ] Single step: write docs; verify every number quoted in docs appears in `results/merged_eval.json` (no hand-copied numbers without source).

---

## Self-Review

- Spec coverage: audit protocol items → Task 1 (quarantine, splits, exclusions), Task 2 (coverage semantics, events zeroed), Task 3 (inspection track quarantine-aware, oracle declared), Task 5 (claims/wording). Comparison-report "best immediate use" (merged as main benchmark, 500 m demo) → Tasks 2/4. Not covered by user decision: v3 benchmark and multi-seed holdouts — out of scope, noted in docs.
- Placeholders: Task 0 exists precisely to pin the three call patterns (build_features columns, M0 signature, app context) before dependent tasks; steps referencing them instruct "record exact signature" — acceptable because Task 0 gates Tasks 2–4, and its findings are concrete reads, not design TBDs.
- Type consistency: `build_observations(cohort, cfg, seed)` → `(Observations, extras)` used identically in Tasks 2/3; `PARTITION` indexed by dt idx 0..49 in both; `extras['oracle_labels']`/`extras['inspection_labels']` names consistent across tasks.
