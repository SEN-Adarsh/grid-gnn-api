# Handoff log — GLM 5.3 takeover (2026-10-08)

## What was inherited

- `Grid-GNN-PoC.zip` (195,933,159 bytes; SHA-256 `120AE4C3EF0D6F8B9FD3567677CC0A0CC6D1DA77FA0E6147F9EECC58C956F2D0`; 502 entries, 217,850,467 bytes uncompressed). Extracted to `D:\rodic\inherited\`. Pristine reference copy kept at `D:\rodic\inherited_pristine\`. Zip inspected before extraction: no path traversal, no absolute paths, no executables.
- `Grid-GNN-Review.html` at workspace root: byte-identical (SHA-256 `87C1CAAF…F073CF717`) to bundled `results/review.html`. Authentic.
- Present and intact: `README.md`, `results/summary.md`, `results/claims_register.csv`, `results/known_issues.md`, `results/results.json` (config hash `61dcc432e86ee9be`), benchmark code, dashboard/app, 20 seed CSVs, stress/ablation/temporal outputs, docs set, tests.
- Missing (as expected): SGCC raw data (`data/external/sgcc`) — excluded from delivery; acquisition is scripted (`src/eval/real_data.py`, CRC-verified spanned-ZIP from the authors' GitHub).

## Integrity checks performed

- All 34 source files match SHA-256 hashes recorded in `results/code_manifest.json` — inherited code is exactly what produced the recorded results.
- Config hash `61dcc432e86ee9be` confirmed in summary, claims register, app validation and protocol JSON.

## Environment used for this takeover

- Windows (win32), Python 3.12.10 in `.venv` (recorded run used Linux, Python 3.12.14 — patch difference noted).
- All pinned versions installed from `requirements.lock` match the recorded environment exactly (numpy 2.5.3, pandas 2.3.3, scikit-learn 1.8.0, scipy 1.18.1, torch 2.5.1+cpu, pandapower 3.2.1, fastapi 0.115.12, streamlit 1.50.0, plotly 6.1.1).

## Changes made to inherited code (all Windows portability only; no logic changes)

1. `run_all.py`: `import resource` guarded (Unix-only module); `peak_rss_mib` recorded as `0.0` on Windows.
2. UTF-8 `encoding='utf-8'` added to all file writes (`src/common.py`, `src/eval/reporting.py`, `src/eval/real_data.py`, `src/eval/evidence.py`, `src/eval/app_validation.py`, `src/sim/simulator.py`, `run_all.py` ×2). Without this, report generation crashed on Windows (cp1252 cannot encode '→' in the deck outline). Linux default was UTF-8, so inherited files were already UTF-8; no content change.

## Reproduction status

- **Fast path (`configs/fast.yaml`) — reproduced in full on Windows.** All phases passed: integrity/training/statistics tests, pandapower physics gate, phase2/phase3, 5-seed benchmark, ablations, stress, app tests, app validation, reports. Same conclusion as recorded: thesis "not supported", M4 selected on validation. (Console: `results/full_rerun_console.log` keeps the final run; fast run overwrote only `results/fast/*` and two gate files — verified byte-identical or gate-only.)
- **Full path (`configs/final.yaml`) — running at time of writing**, console at `results/full_rerun_console.log`. Status to be appended below.
- Fast-run side effects on inherited files: `results/core_tests.log` and `results/loadflow_validation.json` regenerated (gate outputs; pass); `results/fast/*` regenerated with same config/seeds. Everything else verified byte-identical to pristine.

## Claims status (WP-0d)

- Headline table in `results/summary.md` matches the review brief's Section 3.2 exactly (M0–M4, CIs).
- M0–M4 reconstructed in `docs/MODELS.md` from `src/models/baselines.py`, `src/models/gnn.py`, `src/features/build.py`. Confirmed: **all models share one residual-based DT estimator** (built once in `build_features`); DT metrics are not per-model. WP-2 fix still pending (skipped under time pressure).
- Until the full rerun completes, deck numbers carry provenance "recorded run `61dcc432e86ee9be`; fast path reproduced on independent machine/OS". If the full rerun reproduces the headline table within CIs, numbers are upgraded to `reproduced`.

## Audit notes on the inherited package

- `results/claims_register.csv` mixes rows from the final run and an intermediate `graph_free_correction.json` (M4 PR-AUC 0.625 there vs 0.572 final). The final `results/summary.md`/`results.json` are authoritative; the intermediate rows should be read as superseded development snapshots. Not altered (preserving evidence); flagged here.
- `results['presentation']['team']` in code lists: Dr. Shilpa Choudhary, Dr. Kasarapu Venkataramana, Mohammed Ahmed Raza, K. Indrasena Reddy ("user-supplied application"). Human confirmed use on 2026-10-08.

## Human decisions recorded (2026-10-08)

- Cutoff: today, within hours → **minimal path** (WP-0 audit, WP-1 claims scrub, WP-11 deck+script, WP-9 Responsible AI slide). WP-2/3/4/5/6/7/8/10/12 skipped and reported here.
- Corrected positioning (brief Section 5) accepted; GNN demoted to tested hypothesis.
- Team names approved for the deck.

## Skipped work (time pressure)

- WP-2 per-model DT metrics, WP-3 extended real-data evaluation (only inherited SGCC check), WP-4 tamper-event fusion build, WP-5 missing-data sweep (inherited stress sweep covers dropout), WP-7 formal pilot doc (deck slide summarises protocol), WP-8 IES schema build (deck slide covers fit), WP-10 dashboard additions, WP-12 physics validation (pandapower gate already in pipeline).

## Full rerun outcome

- COMPLETED 2026-10-08. All phases passed (physics gate, 20-seed benchmark, ablations/temporal, stress sweep, SGCC check, app tests + validation, reports). Console: `results/full_rerun_console.log`. Reproduced headline table matches `results/summary.md` within reported CIs — see `results/reproduction_check.md`. Thesis: **not supported**; selected model M4 (validation rule). Inherited headline numbers are therefore marked **reproduced** on an independent machine/OS (Windows, Python 3.12.10 vs Linux 3.12.14).

## WP-13 — synth_v2 generator (started 2026-10-08)

### What the inherited generator already does (WP-13 Section 0.1)

`src/sim/simulator.py` + `src/sim/physics.py`, config `final.yaml` (hash `61dcc432e86ee9be`), CIGRE-derived LV network (40 DTs x 50 meters, 60 days x 15-min):

- **Load model**: class shapes (domestic morning/evening, commercial daytime, agricultural morning; hard-coded mix 85/10/5), weekend uplift, cooling term vs temperature, log-normal customer scale (clipped, low-consumption tail), daily jitter, per-step noise, unlabelled gradual drift.
- **Theft families (meter-visible)**: `partial_bypass` (=T1), `intermittent` (=T2, evening/night window; daytime variant as parameter set B), `slowing` (sinusoidal scale, closest to T4 without typed events), `frozen` (=T7 zero/flatline).
- **Hooking (=T3)**: DT-level pole load upstream of any meter, never in meter labels or `theft_kwh`; visible/invisible NTL split. The T3 rule is structural.
- **Honest confounders**: `vacant` (H1), `seasonal` (H2), `solar` (H5), `ev_onset` (H4 step-up), `honest_meter_fault` (narrow H3: frozen reading only).
- **Outages**: planned (rotating day per DT) + random per-day; zero TRUE and reported load during outage.
- **Data problems**: random + burst dropouts, DT clock skew (+-5 min interp), DT accuracy (+-0.5%), 5% whole-record cross-DT mapping swaps (within partition), energy/voltage noise, voltage quantisation.
- **Event log (primitive)**: uint8 per interval — bit 1 generic tamper-like (sensitivity 0.08/day theft, 0.015/day honest), bit 2 power-off during outages.
- **Interfaces/exports**: `Observations` (public telemetry only) vs `HiddenLabels` (labels, NTL decomposition, true DT); parquet contract incl. `hidden/labels_*.parquet`; `balance_error` physics closure check; customer(DT)-disjoint splits; `export_contract`.
- **Tests present**: split disjointness, leakage (feature-contract + perturbation invariance), seed reproducibility, clean energy balance, causal imputation, pandapower physics validation.

### v2 deviations from the inherited files (documented, additive only)

1. `src/sim/physics.py`: added `_long_radial_topology` + early-return branch for `source='long_radial'` (WP-13 Section 2). Existing `cigre_residential`/`cigre_commercial` paths untouched; pinned by `tests/test_synth_v2.py::test_backward_compat`.
2. `src/features/build.py`: `FORBIDDEN` feature-name tuple extended with `family`, `honest_archetype`, `meter_level_positive` (WP-13 Section 1.3 — extend the input-only contract test). No inherited feature name contains these substrings; inherited tests still pass.
3. `src/sim/simulator.py`, `run_all.py`, `src/common.py`, `app/*`: NOT modified (hash-pinned in `test_backward_compat`; pins are the WP-0-patched working copies that produced `results/reproduction_check.md`).

### New code

- `src/sim/synth_v2.py`: configurable class mix + three-phase meters, annual temperature envelope, families T1-T8 (T8 = same-DT neighbour cluster of T1/T3), per-DT prevalence draws (2-15%) with zero-theft DTs, confounders H1-H6 with four H3 sub-types (stuck/intermittent/drift/spike) as reporting faults, typed event log (7 types) with honest noise + stealth theft, per-meter clock drift, negative/spike glitches, `ground_truth.parquet` (meter-level, incl. true daily kWh) + `ground_truth_dt.parquet`, `events.parquet`, `topology.json`, per-pack `data_card.json`.
- `configs/synth_v2.yaml` (master; extends `final.yaml`), `configs/packs/P1..P7.yaml`, `configs/synth_v2_seasonal.yaml` (365-day).
- `tests/test_synth_v2.py`: the six WP-13 Section 12 tests (all green before any evaluation; energy conservation incl. hooking closes to <1e-5 pre-noise; export byte-identical per seed).
- `scripts/build_synth_packs.py`: deterministic pack builder; eval seeds 4100xx; freeze = configs committed before evaluation.

## WP-14 — merged `synthetic_india_v2` dataset adopted as final primary dataset (2026-10-08)

### What was provided and adopted

- `C:\Users\haris\Downloads\synth_v2_merged.zip` (SHA-256 `57F8C31B71DF3D45EB872A8BC2A646325A07213BBBA7ABFF32E3B4A9904335E6`), extracted to `D:\rodic\merged_data\`: 2,500 meters / 50 DTs / 12,192,000 x 15-min readings from 2025-10-01 (IST). Original 500 m/14-day cohort (seed 20261008) + four 500 m/60-day batches (seeds 20261009-12). Observable + truth directories; generator NOT supplied.
- `Merged_Dataset_Audit_Pack.zip` (extracted to `D:\rodic\merged_audit\`): structural audit over all 12.19M readings, 8 inspection rows to quarantine, recommended whole-cohort splits, curation protocol, source-part hashes.
- `Theft_Dataset_Comparison_Final.html`: third-party comparison of our synth_v2 packs vs merged vs a v3 candidate.
- **User decisions (2026-10-08):** merged dataset = final primary synthetic dataset; headline = inspection-supervised; truth = declared oracle evaluation only. WP-13 synth_v2 packs demoted to scenario-design reference (kept under `data/synth_v2/`).

### Integration (all additive; no pinned inherited file modified)

- `src/data/merged_adapter.py`: loads merged tables, applies the audit protocol (8 quarantined inspections REMOVED never relabelled; eval = 2,000 m/60 d cohort, demo = 500 m/14 d cohort; audit split proposal adopted: batches 1+2 train / 3 val / 4 test; `event_flags` zeroed as predictors; truth oracle-only). Builds inherited `Observations` so the entire existing stack runs unchanged. Uniform per-DT subsample (45/DT eval, 44/DT demo; fixed seed) because the pinned `build_features` requires rectangular DT blocks; excluded ids recorded. The 206 GIS pole-reference conflicts are re-seated on the declared DT's same-position pole (documented assumption; `pole_id` itself is never a predictor). Categories: `agricultural_pumpset`->`agricultural`; `small_industrial` kept as its own peer group. `area_type` constant `synthetic_mixed` (field absent in merged meters).
- `src/eval/merged_eval.py`: two tracks. Track A (headline, inspection-supervised): M0 own-history residual ranking vs quarantined inspection outcomes; uninspected meters never treated as negatives; not_accessible excluded. Track B (declared oracle): ROC/PR-AUC of the same unsupervised score vs hidden truth, per-family ranks, DT-balance vs direct-hooking DTs. Outputs `results/merged_eval.json` + `results/merged_eval.md`.
- `app/merged_context.py` + `configs/merged_demo.yaml`: builds a demo replay context (artifacts_merged_demo) for the pinned `Scorer` via GRID_CONFIG without touching pinned files. M0+M1 fitted unsupervised on demo meters; Platt calibration on quarantined-clean inspection labels only; thresholds typically report target unattainable and the app disables inspection flags (honest behaviour with 6 positives).
- Tests: `tests/test_merged_adapter.py` (11), `tests/test_merged_eval.py` (4), `tests/test_merged_context.py` (1).

### Known limitations / cosmetic notes

- The pinned `Scorer` hard-codes `profile_source: synthetic_fallback` in API responses; the true provenance of the merged context is `synthetic_india_v2` (recorded in selected_model.json, audit log and docs). Left as-is to keep pinned files untouched.
- Merged-dataset defects from the audit are NOT fixed (generator unavailable): inspection timing inconsistency (quarantined, not repaired), event-flag shortcut (excluded as predictor), hooking onset semantics unresolved (Tariff misuse/billing model undocumented). Consequences are visible in results: meter-level M0 cannot see direct hooking (as expected); DT-balance separation of hooking DTs is weak (AUC ~0.54) on this dataset.
- WP-13 holdout/sanity work on synth_v2 packs is superseded by this integration; packs remain as scenario reference only.

### Deployment (2026-10-08)

- Backend LIVE on Render (Docker, free tier): https://grid-gnn-api.onrender.com
  - /health verified: status ok, model M0, config_hash 6e60f7bd16aff2ef (merged demo context)
  - /score verified live: 440 meters / 10 DTs, 2.16 s runtime, rank-1 p 0.999
- Local verification before deploy: RAM 314 MB working set (fits 512 MB free tier);
  score none/vacancy/upstream_hooking all pass on the exact container code path.
- Frontend: grid-gnn-dashboard on Vercel (user-imported); UI overhaul from
  collaborator pushed as f7a4ad6; app.js API_BASE default switched to the Render
  URL (commit after f7a4ad6). ?api= override and localhost:8001 default retained.
- Full guide: D:\rodic\DEPLOYMENT_GUIDE.md

### 2026-10-08 (late) � friend's Rodic redesign UI merged; /health corrected
- Friend's clone had diverged from 9d67d8b: 9 commits (routed multi-page UI, guided demo,
  sample mode, M0/M1 copy, fast-fail). Merged into origin/main via merge commit 26ed50b
  (no force push; both lineages preserved), then fix commit 7b3dcdd.
- Bugs found in the zip and fixed: (1) API_BASE default was a placeholder
  'https://USERNAME-grid-gnn-api.hf.space' -> set to https://grid-gnn-api.onrender.com;
  (2) recommended-model priority preferred M1 -> flipped to M0 (matches selected_model.json
  and every published eval number); (3) sample-mode model label M1 -> M0; assets cache-bust
  v=9 -> v=10. UI consumes h.scoring_window and h.models from /health (both now served).
- Backend 3f4c947: /health now reports profile_source from selected_model.json
  (synthetic_india_v2; was hard-coded 'synthetic_fallback') and adds
  scoring_window {min,max} computed from context cfg ((baseline_days+7)*96 .. total).
  api.py pin updated to 7ceb0c8868bde954416c32c22b43a02bc586fdf642aa62116e64583f0c8e7058;
  hf_space suite 22 passed.
- Live verified: /health {synthetic_india_v2, 1344-1344, M0/M1}; guided-demo-shaped
  POST /score (M0/theft/0.9/1344/tamper) -> 440 meters, 10 DTs, 1.88 s, config 6e60f7bd16aff2ef.
  Note: /score response body still carries the pinned cosmetic profile_source label
  (scoring.py is pinned); UI never displays it (provenance line omits it).
- Frontend still awaits the user's Vercel import; repo main = 7b3dcdd.
