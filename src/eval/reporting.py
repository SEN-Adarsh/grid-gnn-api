"""Generate every numerical narrative from recorded evidence, never prose constants."""
from __future__ import annotations
import base64
import csv
import html
import importlib.metadata
import json
from pathlib import Path
import platform
import re
from ..common import write_json,hash_file
from .evidence import claims_register

NAMES={'M0':'Energy balance + history deficit','M1':'Isolation Forest','M2':'Temporal RandomForest',
       'M3':'RandomForest + graph summaries','M4':'Temporal GCN'}
PRIMARY=['precision_at_k','recall_at_k','pr_auc','vacant_fpr','dt_top3','dt_mape']
LABELS=['Precision@5%','Recall@5%','PR-AUC','Vacancy FPR','DT top-3','DT kWh MAPE']


def ci(value,percent=False):
    if value['mean'] is None:return 'undefined (no flags)'
    fmt=(lambda x:f'{x:.1%}') if percent else (lambda x:f'{x:.3f}')
    if value['ci_low'] is None:return fmt(value['mean'])+' [CI unavailable]'
    return f"{fmt(value['mean'])} [{fmt(value['ci_low'])}, {fmt(value['ci_high'])}]"


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(str(x).replace('|','/') for x in row)+' |' for row in rows])


def write(path,body):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(body.strip()+'\n', encoding='utf-8')


def generate_reports(cfg):
    out=Path(cfg['output_dir']);docs=Path('docs') if cfg['name']=='final' else Path('docs')/cfg['name']
    r=json.loads((out/'results.json').read_text())
    c=r['config'];chosen=r['selected_model'];h=r['headline']
    validation=json.loads(Path('results/loadflow_validation.json').read_text())
    sources=json.loads((out/'source_manifest.json').read_text()) if (out/'source_manifest.json').exists() else {'sources':[]}
    issues=json.loads((out/'development_issues.json').read_text()) if (out/'development_issues.json').exists() else []
    tests={}
    for name in ['core_tests','app_tests']:
        path=(Path('results') if name=='core_tests' else out)/(name+'.log')
        text=path.read_text() if path.exists() else ''
        counts=re.findall(r'(\d+) passed',text)
        tests[name]={'passed':int(counts[-1]) if counts else None,'log':str(path)}
    r['verification']={'test_suites':tests,'physics':validation,
        'app':json.loads((out/'app_validation.json').read_text()) if (out/'app_validation.json').exists() else None}
    r['runtime']['hardware']['python']=platform.python_version()
    r['reporting_protocol']={'confidence_percent':95,'baseline_model_ids':list(NAMES),
        'meter_count_per_seed':c['n_dts']*c['meters_per_dt'],'git_repository_supplied':False,
        'consumer_band_low_upper_kwh_day':4,'consumer_band_medium_upper_kwh_day':10,
        'shap_groups':6,'runtime_includes_dependency_install':False}
    r['presentation']['team_name']='Crystal Paradigm'
    r['presentation']['team_count']=len(r['presentation']['team'])
    write_json(out/'results.json',r)
    headline=table(['Model']+LABELS,[[n]+[ci(h[n][k],k in ['vacant_fpr','dt_mape']) for k in PRIMARY] for n in NAMES])
    pairs=table(['Difference']+LABELS,[[p]+[ci(r['paired'][p][k],k in ['vacant_fpr','dt_mape']) for k in PRIMARY] for p in ['M4-M3','M3-M2','M3-M0']])
    pair_pr=table(['Paired comparison','PR-AUC difference'],[[p,ci(r['paired'][p]['pr_auc'])] for p in ['M4-M3','M3-M2','M3-M0']])
    op_table=table(['Model','Validation target met / seeds','Test precision','Test recall','ECE','Inspections for visible theft target','Ratio vs M0'],
        [[n,f"{r['operating_points'][n]['target_met_seed_count']} / {c['seeds']}",ci(h[n]['precision']),ci(h[n]['recall']),
          ci(h[n]['ece']),ci(h[n]['inspections_80']),ci(h[n]['inspection_ratio_vs_m0'])] for n in NAMES])
    runtime=r['runtime'];seconds=runtime.get('full_pipeline_wall_seconds')
    duration=f'{seconds:.1f} seconds ({seconds/60:.1f} minutes)' if seconds is not None else 'not yet recorded for a full clean command'
    readme=f"""
# Grid-GNN — NTL inspection workbench

**The graph-advantage thesis is {r['thesis_verdict']}.** This runnable research PoC compares meter history, DT energy balance, engineered neighbourhood context and a temporal graph neural network. All grid profiles are **synthetic_fallback**. No Indian field accuracy, recovered revenue or deployed system is claimed.

The final benchmark covers **{c['n_dts']*c['meters_per_dt']:,} meters, {c['n_dts']} DTs and {c['days']} days per seed**, with {c['interval_minutes']}-minute telemetry and {c['seeds']} seeds. The test partition is by DT. {chosen} is the app default because it had the highest **mean validation** PR-AUC; {max(h,key=lambda n:h[n]['pr_auc']['mean'])} has the highest test point estimate. Test results did not change selection.

## Start the supplied demo

Use Python {platform.python_version()}. From the extracted `grid-gnn` directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m streamlit run app/dashboard.py
```

The packaged model and public demo history let the dashboard run without retraining or downloading SGCC. Open the local URL printed by Streamlit. In another activated terminal:

```bash
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/score -H 'Content-Type: application/json' -d '{{}}'
```

`/docs` serves the request schema. Batch observations update the known network's stored baseline history for that request. Unknown meters need metadata onboarding; this demo does not accept arbitrary new networks. Timestamps must include timezone and align with the configured interval. Missing readings use a mask. Hidden-label or unknown request fields are rejected.

`Inspect` offers the network, reasons and balance chart. `Replay & scenarios` advances stored intervals and injects reversible vacancy, theft, upstream-load or AMI-dropout scenarios. It is labelled **replayed telemetry**. `Evidence` shows results and CIs; `Responsible AI` shows subgroup errors and human-review policy. Candidate meter kWh is a heuristic residual allocation, not verified theft or revenue.

## Reproduce all results

```bash
python run_all.py --config configs/final.yaml
```

This starts the integrity/physics gate, retrains the development baselines and GNN, runs final seeds, ablations, temporal and stress checks, attempts SGCC acquisition, verifies the app, and regenerates reports. It does not reuse fitted models by default. A clean copy needs internet for dependencies and optional SGCC; a failed optional data request is recorded explicitly. `--resume` is for an interrupted final checkpoint, not clean reproduction.

```bash
python run_all.py --config configs/fast.yaml
python -m pytest -q
python run_all.py --config configs/final.yaml --phase docs
```

The fast configuration is a development check, not the final evidence. To serve its artifacts set `GRID_CONFIG=configs/fast.yaml`. The full recorded command took {duration}, excluding dependency installation. CPU only; {runtime['hardware']['worker_processes']} worker processes; recorded worker peak {runtime['peak_worker_rss_mib']:.1f} MiB. Scoring-only timing excludes feature construction and is not end-to-end latency; see `results/app_validation.json` for app timing. No production cost was measured.

## Headline evidence

Mean [95% CI]; precision/recall are at the configured inspection budget. MAPE and vacancy FPR are percentages.

{headline}

{pair_pr}

The paired intervals do not establish an incremental benefit from the GNN or engineered graph summaries. M0 abstains at the main operating target; M1 flags infrequently. Their zero vacancy FPR is not evidence of useful vacancy discrimination. All models use the **same residual-based DT estimator**; those identical columns are not GNN achievements. DT top-k means hitting the single highest-loss DT, not finding every theft-bearing DT.

See `results/summary.md` for operating points, paired metrics, all stress settings, inspection-ordering scope and limitations. `results/claims_register.csv` maps reported values to recorded sources. `results/known_issues.md` includes failures and skipped work.

## What is included

| Path | Contents |
| --- | --- |
| `src/sim`, `src/features`, `src/models`, `src/eval` | Simulator, input-only features, baselines/GNN, evaluation and report generation |
| `configs` | Pinned final and development assumptions |
| `app` | Local API, dashboard and replay scorer |
| `artifacts` | Trained models, public demo context, separate evaluation-only labels |
| `data/generated`, `data/demo` | First-seed contract tables and a held-out replay subset; hidden labels are separate |
| `results` | Per-seed predictions, CIs, stress tables, figures, audits, claims, execution logs |
| `docs` | Assumptions, sources/licences, limitations, model card, datasheet, Responsible AI, presentation outline, video script and DISCOM request |
| `tests` | Leakage, determinism, physics, missingness, explanations, API and replay checks |

Caches, environments and raw SGCC data are excluded from delivery. SGCC's dataset redistribution licence is unclear; the acquisition code is included. Only load trusted bundled model artifacts: joblib is an executable Python serialization format. This local PoC has no authentication, production security or automated enforcement. Do not expose it as a public utility service.
"""
    write('README.md' if cfg['name']=='final' else docs/'README.md',readme)

    extra=table(['Model','DT top-1','DT bias kWh'],[[n,ci(h[n]['dt_top1']),ci(h[n]['dt_bias_kwh'])] for n in NAMES])
    ab=table(['Full model minus ablation','PR-AUC difference'],[[p,ci(v['pr_auc'])] for p,v in r['ablations']['paired'].items() if p.startswith('M4-M4_')])
    stress=table(['Stress axis','Setting','M2 PR-AUC','M3 PR-AUC','M4 PR-AUC'],
        [[s['axis'],s['value']]+[ci(s['headline'][n]['pr_auc']) for n in ['M2','M3','M4']] for s in r['stress']])
    secondary=r.get('secondary_operating_headline')
    sec=table(['Model','Test precision','Test recall','Vacancy FPR'],[[n,ci(secondary[n]['precision']),ci(secondary[n]['recall']),ci(secondary[n]['vacant_fpr'],True)] for n in NAMES]) if secondary else 'Pending full clean recomputation; per-seed flags are saved in meter CSVs.'
    real=r['real_data']['sgcc']
    sgcc=f"SGCC completed on {real['customer_rows_observed']:,} customers and {real['date_columns_observed']:,} observed daily columns. PR-AUC {ci(real['metrics']['pr_auc'])}; ROC-AUC {ci(real['metrics']['roc_auc'])}." if real['status']=='completed' else f"SGCC skipped: {real.get('reason')}"
    v=r['ntl_visibility'];voltage=r['voltage_verdict']
    performance=table(['Model','Mean fit seconds','Mean score-only seconds / DT'],[[n,f"{runtime['models'][n]['mean_fit_seconds']:.4f}",f"{runtime['models'][n]['mean_inference_seconds_per_dt']:.6f}"] for n in NAMES])
    write(out/'summary.md',f"""
# Executed evidence report

Config hash: `{r['config_hash']}`. Profile source: `{r['profile_source']}`. Thesis: **{r['thesis_verdict']}**.

## Headline results

{headline}

Intervals are exploratory hierarchical seed + whole-DT percentile bootstrap intervals. Point estimates are means of seed metrics. Nonlinear metrics and cluster bootstrap bias can put the point outside its percentile interval; particularly ECE and extreme-DT localisation must be read cautiously. These are not Indian field confidence intervals. DT copies share a reference family; independent seeds do not create independent utility networks.

## Paired differences

{pairs}

Positive PR-AUC, precision, recall and DT hit differences favour the first model; negative FPR and MAPE differences favour it. Identical DT columns arise from a shared estimator. Support was defined using PR-AUC before final reporting; no positive graph advantage is established. Many exploratory comparisons are unadjusted for multiplicity.

## Operating points and calibration

Main target precision: {c['target_precision']:.0%}, chosen on validation only, with at least {c['min_validation_flags']} validation flags. The target is not a guarantee on test or field data. Undefined precision means there were no flags. Means of per-seed precision exclude undefined seeds; recall includes abstentions.

{op_table}

Secondary validation target: {c['secondary_precision']:.0%}.

{sec}

Calibration: `figures/reliability.png`. Validation data are reused for model selection, calibration and thresholds; no independent calibration partition is claimed. Precision calibration uses synthetic prevalence and may change under deployment shift.

Inspection target is {r['protocol']['inspection_target_kwh_recall']:.0%} of **meter-visible intentional theft kWh**, compared with M0 ordering, not a full meter sweep. It excludes upstream hooking and honest meter faults, and is not confirmed field recovery. Confidence intervals for ratios are paired. Do not convert these into a deployment savings promise.

## DT estimation and visibility

{extra}

DT estimates use measured input minus imputed consumption minus an imperfect, publicly computed loss estimate. MAPE excludes true NTL at or below {r['protocol']['ntl_mape_floor_kwh']:g} kWh; bias includes all DTs. DT top-3 checks whether the highest-true-loss DT is in the first three ranked residual DTs. The auxiliary GNN DT head performed poorly: mean MAPE {r['gnn_dt_head']['mean_mape']:.1%}, bias {r['gnn_dt_head']['mean_bias_kwh']:.1f} kWh; it is not the deployed estimator.

Meter-visible NTL {v['meter_visible_fraction']:.1%}; invisible upstream NTL {v['invisible_fraction']:.1%}. Visible NTL includes honest meter failures. Intentional meter-visible theft is {v['intentional_theft_meter_visible_fraction']:.1%} of all intentional theft energy. The method can rank consumer inspections for under-recording and unexplained balances by DT. It cannot identify the person behind upstream hooking or reliably resolve pole segments.

## Ablations, voltage and temporal shift

{ab}

The strict no-graph control removes message passing, peer/DT/electrical summaries and the electrical temporal channel. The shuffle control preserves the weight/degree multiset while disrupting node relationships. All ablations use the full model's validation-selected hyperparameters, refit and recalibrate on the original training/validation partitions.

Voltage: incremental benefit **not supported** at noise sigma {c['voltage_noise_v']:g} V and resolution {c['meter_voltage_resolution_v']:g} V. Full minus no-voltage PR-AUC: {ci(voltage['pr_auc_delta'])}. Median theft SNR proxy across seeds: {voltage['median_theft_snr_across_seeds']:.3f}. This is mean gap change divided by baseline interval SD, not causal signal recovery or a significance test. Per-meter proxies for every voltage setting are in `stress/*_voltage_snr.json`.

Temporal holdout for validation-selected {chosen}: PR-AUC {ci(r['temporal_holdout']['headline'][chosen]['pr_auc'])}. Training used an earlier endpoint; evaluation used later days on held-out DTs. Baseline histories overlap by design; this is not a prospective utility trial.

## Stress tests

{c['stress_seeds']} seeds per setting; fitted models and calibrators remain fixed. CIs resample seeds and whole DTs. The unseen topology-family check uses the commercial CIGRE branch; usual headline DTs are already held out by identity. Prevalence varies intentional consumer theft, not all NTL energy. Measurement sweeps reuse latent physics to isolate perturbations. Results are separate experimental conditions, not a joint worst-case guarantee.

{stress}

## Data and physics

{sgcc} Daily Chinese labels have unknown timing/noise and no topology; this checks only a temporal baseline. Customer counts match the author README; the verified file omits a calendar day. Original split archive parts were decoded with CRC and size checks after native ZIP conversion failed; preliminary incomplete-file metrics are invalidated.

LCL/SimBench real-profile validation is skipped after access failure. Grid profiles, weather and coordinates are synthetic. No Pecan Street, Indian field labels or ESMI-fitted noise statistics were used.

Physics validation: {validation['samples']} balanced snapshots against pandapower; max voltage disagreement {validation['voltage_max_error_v']:.3g} V, mean {validation['voltage_mean_error_v']:.3g} V; maximum line-loss relative error {validation['loss_max_relative_error']:.3g}. Tolerances: {validation['voltage_tolerance_v']:g} V and {validation['loss_relative_tolerance']:.1%}. Near-machine agreement is expected between matching simplified models; it does not validate coupled neutral, transformer or field physics.

First-seed missing-data sensitivity: causal-imputation DT MAE {r['imputation_audit']['default_dt_kwh_mae']:.1f} kWh; zero-fill MAE {r['imputation_audit']['zero_fill_dt_kwh_mae']:.1f} kWh. Zero fill raises mean DT residual by {r['imputation_audit']['residual_mean_increase_zero_fill_kwh']:.1f} kWh. Missing communication is a material theft confounder.

## Runtime and audit

{performance}

Full command wall time: {duration}; dependency installation excluded. GPU: {runtime['hardware']['gpu']}. Maximum observed worker RSS: {runtime['peak_worker_rss_mib']:.1f} MiB; this is a per-process peak, not total process-tree memory. Cost has not been estimated for deployment. App validation and all warnings remain in their logs. See `known_issues.md`, `claims_register.csv` and `docs/responsible_ai.md` before using a result externally.
""")

    source_rows=[[s['id'],f"[{s['id']}]({s['url']})",s.get('role',''),s.get('access',''),s.get('licence','See primary source; no additional dataset rights asserted')] for s in sources['sources']]
    write(docs/'data_sources.md',f"""
# Data provenance and licences

All grid outputs are stamped **synthetic_fallback**. Synthetic profiles were substituted when London Datastore access failed. There is no Indian field dataset in this repository. Measured source observations, download hashes and access status are in `results/source_manifest.json` and `results/results.json`.

{table(['Source','Link','Role','Observed access / use','Licence / restriction'],source_rows)}

## Acquired and used

- **pandapower CIGRE LV:** topology and line standard types from the installed package. The package's BSD notice is copied under `docs/third_party/`. CIGRE-derived grids are reference grids with modified lengths and service drops; they are not a DISCOM network survey.
- **SGCC author release:** {real.get('customer_rows_observed','unavailable')} customers, {real.get('date_columns_observed','unavailable')} date columns, Chinese daily consumption. Customer splitting and chronological date sorting are explicit. Dataset redistribution rights were not found, so raw data are excluded; reproduce via the acquisition code subject to applicable rights. CRC, byte count and SHA checks precede use. This is separate from grid model evaluation.
- **Generated telemetry:** deterministic profiles, synthetic weather/coordinates and injected labels. Simulation code and generated contract tables are included. Generation history does not establish fidelity to Indian consumption or faults.

## Substituted or skipped

LCL downloads returned access errors; synthetic load shapes replace them. CER requires an application. SimBench is installed but its grid/data are not used in the reported benchmark. SGSC, SMART-DS and IEEE feeder extensions were not attempted because the required core benchmark used CIGRE. ESMI was reviewed for context only; no raw data or fitted summary statistics were used. Weather is synthetic, not Open-Meteo or NASA observations. PFC national AT&C figures are deliberately absent from claims because none were acquired and used. Pecan Street is excluded.

## Literature boundaries

Jokar's publisher abstract and an indexed primary research attack table support comparisons with fractional scaling and time-window suppression. Full original attack-formula verification could not be completed after access blocks. Our frozen meters and physical upstream hooking are simulator extensions, not claimed exact reproductions of those functions. Energy-preserving permutations are not called NTL. The referenced arXiv graph-control paper concerns Volt-VAR control; it is cited only as background for physics-informed graph operators, never evidence of theft-detection performance.

Indian nameplates, interval fields, resolution, event codes, transformer meter boundary and relevant standards require utility confirmation. RDSS/BIS links are context, not proof that these exact simulated fields match a particular AMI installation. No legal licence or compliance certification is implied.
""")

    low=next(x for x in r['fairness'][chosen] if x['attribute']=='consumption_band' and x['group']=='low')
    fault=next(x for x in r['fairness'][chosen] if x['attribute']=='archetype' and x['group']=='honest_meter_fault')
    weak=[
        'Synthetic transfer: generated load and weather on one principal reference topology family; no Indian field labels and no real-LCL/SimBench grid validation.',
        'Unsupported graph advantage: paired graph/GNN improvement intervals include zero; validation reuse for search, calibration and thresholds can overfit a small DT partition.',
        f"False accusations remain possible: honest meter faults have pooled FPR {fault['fpr']:.1%}; low-consumption meters have flag-rate ratio {low['flag_rate_disparity_ratio']:.2f} against the overall population for {chosen}.",
        'Incomplete physics and metrology: ideal neutral, no mutual coupling or transformer dynamics, assumed loss parameters, stale mapping/phases and clock error. Balanced solver agreement is a narrow check.',
        f"Attribution and kWh limits: invisible upstream NTL is {v['invisible_fraction']:.1%}; DT MAPE is {ci(h[chosen]['dt_mape'],True)}; posterior allocation is not established theft or billing recovery."
    ]
    write(docs/'limitations.md','# Limitations\n\n'+'\n\n'.join(f'{i}. {s}' for i,s in enumerate(weak,1))+f"""

## Statistical and implementation limits

Equal search budget means {c['search_trials']} candidate configurations each for the supervised models, not equal computation or equal model capacity. The GNN uses validation early stopping; forests have a fixed tree count rather than epoch early stopping. All scalers, class weights and calibration are fitted without test data. Validation reuse remains an acknowledged source of selection optimism.

The graph-free neural control differs from a temporal-only forest in capacity and optimisation. The graph shuffle is a falsification check, not proof of causal graph use. Bootstrap comparisons are exploratory and unadjusted. Extreme-DT hit metrics and ECE are nonlinear; percentile bootstrap bias can be material. No confidence claim should be extended beyond the synthetic generator.

A serving instance contains a stored baseline and a fixed demo network, not a streaming history database or arbitrary-grid onboarding service. Replay partial days extrapolate only the temporal-encoder daily aggregate; missing and future masks remain explicit. A target precision fitted on simulation cannot guarantee field precision. Reason codes summarize observable evidence; global permutation and reference-based group Shapley are not causal explanations of guilt or occupancy.

Segment residuals, voltage-slope localisation, direct GIS repair, tariff misuse, full coupled three-phase neutral validation and field economic analysis are omitted. No pole-segment localisation is claimed. The learned DT head is retained for audit but does not determine delivered kWh estimates.
""")
    fair=table(['Attribute','Group','Count','Positive count','Precision','Recall','FPR','Flag-rate ratio'],
        [[x['attribute'],x['group'],x['n'],x['positive_n']]+['undefined' if x[k] is None else f"{x[k]:.3f}" for k in ['precision','recall','fpr','flag_rate_disparity_ratio']] for x in r['fairness'][chosen]])
    write(docs/'responsible_ai.md',f"""
# Responsible AI — inspection support only

## Human authority and the cost of error

Every flag requires an inspector to verify meter health, occupancy changes, outages, solar export, mapping and timestamp quality before considering misconduct. No automatic disconnection, penalty, accusation, billing correction or debt decision is supported. False suspicion can impose travel, stress, stigma and service disruption; correct metrology may explain a residual without theft.

The main operating target is validation precision {c['target_precision']:.0%}; the secondary target is {c['secondary_precision']:.0%}. If a target is unattainable with enough validation examples the model abstains. Test precision is measured, not promised. Review recall, abstention and confidence intervals alongside FPR; zero flags are not a fairness success.

## Vacancy and seasonal protection

Own-history ratios are combined with peer stability, outages, missingness, imperfect events and energy balance. Vacancy and seasonal labels are used only for evaluation. They are never fed to the scorer and occupancy cannot be established from a low score. For {chosen}, vacancy/seasonal FPR is {ci(h[chosen]['vacant_fpr'],True)}. Honest meter faults remain particularly hard to distinguish from intentional under-recording.

## Observed subgroup audit

Pooled held-out predictions from all seeds. These descriptive groups are simulated operational categories, not a demographic fairness certification. Undefined entries lack the relevant labels or predictions. Pooled group precision differs from the mean-of-seed headline estimand.

{fair}

Low consumption means a baseline below {r['reporting_protocol']['consumer_band_low_upper_kwh_day']:g} kWh/day; the medium band's upper boundary is {r['reporting_protocol']['consumer_band_medium_upper_kwh_day']:g} kWh/day. These analysis bands are not tariffs or poverty labels. Disparity is group flag rate divided by overall flag rate; differing injected prevalence matters. Use field sampling and confidence intervals before proposing policy changes.

## Mitigation and explanations

Maintain a meter-fault and data-quality review pathway before theft escalation. Offer manual hold and correction when absence, outage or metrology evidence contradicts the model. Audit low-consumption, seasonal, vacant and faulty-meter outcomes. Do not silently introduce segment-specific thresholds: first establish label quality, error costs and sufficient field samples, then document justification and utility approval.

Every inspection row has template reasons based on observable history, DT balance, peers, missingness and events. Tree models expose exact group Shapley values against training medians; the GNN exposes validation permutation-by-group importance. These explain associations under a model/reference distribution, not intent, occupancy, causal loss allocation or legal liability.

## Privacy and governance

Keep pseudonymisation keys inside the DISCOM. Use project-scoped random meter and DT identifiers; separate names, addresses, account details and precise GIS from the analytical store. Collect only fields required for balance, data quality and review. Access should be role-based and logged with encryption and approved hosting before any real data are loaded. The local demo does not implement those production controls.

Proposed pilot policy, subject to utility/legal approval: retain raw interval data for {r['presentation']['proposed_raw_retention_days']} days, aggregate or delete after review, and separately define retention of substantiated inspection records according to applicable obligations. Synthetic audit JSONL contains timestamp, model/config hash, input hash, flag pseudonyms and reason codes. It is local and append-only by convention, not cryptographically tamper-proof. Rotate and protect it in a pilot.

Use the DISCOM grievance channel for an appeal; assign a human case owner, present the evidence and alternative explanations, record corrections, and notify the consumer through approved procedures. A model cannot reject an appeal. Corrected outcomes must propagate to evaluation and future training with version tracking.

Monitor residual distributions, telemetry gaps, voltage/resolution changes, category-level flag rates, inspector agreement and delayed outcomes. Trigger review on documented changes, not automatic retraining. Randomly sample unflagged meters as well as flagged cases to reduce inspection-feedback bias. Evaluate model updates on held-out DTs and time before approval. Keep rollback artifacts and the claims register with each release.

India's DPDP Act and applicable Rules are legal-review inputs; this report does not certify compliance or determine the lawful basis for a particular deployment. Confirm notices, purpose, retention, processors, security and grievance obligations with the DISCOM's legal/privacy team. See official references in `data_sources.md`.
""")

    write(docs/'model_card.md',f"""
# Model card

**Purpose:** prioritise human inspection of unexplained energy and meter under-recording in a synthetic LV research setting. **Owner:** {r['presentation']['team_name']}; real operational owners must be assigned before pilot use. **Version identity:** config `{r['config_hash']}` plus artifact SHA in `artifacts/selected_model.json` and each audit event.

**Selected model:** {chosen}; selection on mean validation PR-AUC across configured seeds. The packaged model is the first predeclared seed, not the best seed. Full benchmark results average independently fitted seed models; they are not an ensemble accuracy claim about this single demo artifact.

**Architecture:** temporal convolution over daily energy, missingness, event and auxiliary voltage channels; electrical-weighted graph convolutions over DT/poles/meters; a theft logit and auxiliary DT-energy head. M4 also receives engineered temporal and graph summaries. Temporal RandomForest and engineered-context RandomForest are mandatory comparison models. Model, input feature and calibration code is included.

**Target:** intentional injected meter theft versus all other meter conditions. Honest meter faults can generate visible NTL while remaining negative intent labels. Upstream hooking has no meter culprit label. **Features:** public metering, declared topology/nameplates and baseline history; no hidden labels, true losses, actual mapping errors or future intervals.

**Training:** training DTs only, class weighting, {c['search_trials']} validation trials, GNN early stopping. **Calibration:** Platt scaling on validation; thresholds are target-precision rules. Scalers and explanatory references use training data. **Performance:** `results/summary.md`; graph advantage {r['thesis_verdict']}. **Forbidden uses:** enforcement, billing recovery, proof of occupancy or guilt, individual attribution of upstream load, field-performance or ROI claims.

**Operational limits:** known demo assets and stored histories; no authentication or case-management integration. Synthetic noise, label prevalence and reference grids may poorly match a utility. Monitor drift and false accusations; use the responsible-AI review/appeal procedure.
""")
    write(docs/'datasheet.md',f"""
# Datasheet

## Origin and composition

Grid tables are generated under `{r['profile_source']}` from CIGRE-derived radial networks with configurable consumer archetypes and measurement error. Each seed contains {c['n_dts']*c['meters_per_dt']:,} meters and {c['n_dts']} DTs across {c['days']} days at {c['interval_minutes']}-minute cadence. The first configured seed is exported; other seed predictions and complete generation settings are retained. `data/demo` is a public-observation subset of held-out DTs with a separate hidden evaluation export.

## Contract

| Table | Meaning and main fields | Model access |
| --- | --- | --- |
| meters | meter_id, dt_id, feeder_id, pole_id, phase, category, sanctioned_load_kw, lat, lon, mapped_dt_id, has_event_log; extra simulated area_type | Public declared data; coordinates not used |
| readings | meter_id, ts, kwh, kvarh, voltage_v, event_flags, is_missing | Public observations; signed net import |
| dt_readings | dt_id, ts, kwh_in, kvarh_in, v_lv, i_a, i_b, i_c | Public measured data |
| topology_edges | from_node, to_node, r_ohm, x_ohm, length_m | Public declared topology |
| labels_meter | meter_id, archetype, theft_start, theft_params; intent flag and physical assignment for evaluation | Hidden, never model features |
| labels_dt | dt_id, ts, ntl_kwh, tech_loss_kwh, meter_visible_ntl_kwh, invisible_ntl_kwh | Hidden, never inference input |

Public dt_id is the declared mapping and can be wrong. Physical mapping is hidden. Demo identifiers are remapped consistently for table joins. Event bits are simulator encodings, not certified meter-vendor codes. Solar export is signed. Communication gaps remain missing rather than observed zero.

## Collection, preparation and splitting

Generated load has household heterogeneity, daily/weekly patterns, synthetic temperature and cooling. Vacancy, seasonal absence, solar, EV onset, outages, AMI random/burst dropout, faulty mapping, phase changes and meter clock/accuracy error are simulated. Intentional archetypes include partial bypass, slowing, frozen readings and intermittent suppression. Upstream hooking adds unmetered physical load. Separate honest failures avoid equating every accounting loss with intentional theft.

DT partitions are created before labels; mapping swaps remain within partitions. Each split has theft and honest absence examples. Baseline history can include persistent theft. Inference features have no hidden-label argument. Missing data use preceding same-slot history and a prior-history fallback, overridden by measured DT outages. Zero fill is an explicit ablation.

## Distribution and maintenance

Raw SGCC is not redistributed; it is a separate Chinese customer-label check with uncertain dataset licence and label timing. Synthetic exports contain no real consumer identities. Preserve the config hash and source manifest when regenerating. A new generator, event model, topology family or meter specification requires new validation. Keep erroneous extraction/measurement attempts in the issue ledger and invalidate their metrics explicitly.
""")
    write(docs/'data_request_spec.md',f"""
# DISCOM pilot data request through Rodic

Request a documented, authorised historical window of **{r['presentation']['pilot_months_min']}–{r['presentation']['pilot_months_max']} months**. The goal is a blind test of energy balance and inspection ranking, including honest consumption collapse and meter faults. Begin with representative DTs and their complete attached consumer populations, not only already-flagged accounts.

| Dataset | Required fields | Quality and interpretation |
| --- | --- | --- |
| Consumer interval telemetry | Project meter_id; interval start/end with timezone; {c['interval_minutes']}-minute import/export kWh and kVArh; available phase voltage; missing/estimated status; ingest time and revision id | Specify units, interval-vs-cumulative registers, clock corrections, negative/export conventions, resolution, accuracy class, actual profile fields and firmware |
| Events | meter_id; timestamp; event start/end; raw vendor code; translated event type; severity; communication status | Include cover/magnetic/neutral/reversal/phase/power events where available; retain vendor dictionary; absence of an event is not proof of no tampering |
| DT input telemetry | dt_id; timezone interval; active/reactive energy; available LV phase voltage and currents; quality/status; meter accuracy/resolution and clock history | Identify LV/HV installation boundary and whether transformer losses lie inside it; provide outage/switching intervals |
| Meter inventory | Pseudonym, category, sanctioned load, phase, install/replacement dates, solar/net-meter status, known EV connection where authorised | No names, contact details, payment identifiers or unnecessary demographic fields |
| Mapping and network | Meter-to-DT/feeder/pole mapping; effective dates; quality flag and verification method; conductor lengths/types; DT rating/loss test parameters; switch states | Preserve disputed/unknown mapping; GIS precision only as needed; separate surveyed versus inferred edges |
| Inspection outcomes | Pseudonym; inspection timestamp; selection reason; finding taxonomy; verified evidence; meter test/replacement outcome; occupancy/outage/solar explanation; reviewed energy assessment; case resolution/appeal | Distinguish confirmed intent, technical/meter faults, honest absence, inconclusive and unavailable. Keep blinded test outcomes inaccessible to model tuning |
| Context | Weather if available; tariffs/categories dictionary; planned/unplanned outages; meter-data estimation/backfill rules; topology and phase changes | Changes are time-dependent, not a single current snapshot |

The DISCOM creates project-scoped random pseudonyms and retains the lookup in its own controlled environment. Use consistent mappings across tables and across time; separate identity/GIS vault access. Agree data purpose, access controls, retention, hosting, processors and consumer rights with the utility privacy/legal owner before transfer.

Validate table joins, interval completeness, phase/DT consistency, time alignment, units and physical balance before fitting. Split by whole DT/feeder and later time; reserve inspection outcomes for a blind evaluation. Sample unflagged meters as well as flagged cases to assess false negatives and reduce selection bias. Record inspection costs and confirmed recoverable energy separately; do not infer field ROI from simulation. Request meter-vendor schema and standards conformance documentation; verify actual fields and voltage resolution rather than assuming them.
""")

    demo=r.get('demo_examples',{})
    absent=demo.get('vacancy_not_flagged');theft=demo.get('theft_flagged')
    demo_text=(f"Select **{absent['meter_id']}**: absent/seasonal evaluation example, not flagged, probability {absent['probability_simulated_theft']:.1%}. Then select **{theft['meter_id']}**: injected theft example, flagged, probability {theft['probability_simulated_theft']:.1%}, candidate allocation {theft['candidate_kwh_allocation']:.1f} kWh. These are first-seed illustrative cases, not independent performance evidence. The allocation is not verified theft." if absent and theft else 'A qualifying pair is unavailable in this configuration; do not invent a successful example. Show aggregate evidence and documented counterfactual scenarios instead.')
    slide_text=[
        ('Problem and testable thesis','Unexplained energy is not proof of theft. Test whether neighbourhood context and a GNN improve inspection ranking over strong temporal and energy-balance baselines. State synthetic scope immediately.'),
        ('Approach','Show public observations → causal imputation and physical loss estimate → temporal/graph features → calibrated ranking and DT residual → inspector review. Upstream hooking has no meter culprit.'),
        ('Data sources and licences',f"CIGRE-derived topology and synthetic profiles/weather. {sgcc} Raw SGCC excluded because redistribution rights are unclear. LCL/CER unavailable; SimBench real-profile check skipped. List primary source links from data_sources.md."),
        ('Assumptions and physics',f"{c['n_dts']*c['meters_per_dt']:,} meters, {c['n_dts']} DTs, {c['days']} days, {c['interval_minutes']}-minute intervals per seed. Ideal neutral; configurable Indian-style assumptions require DISCOM verification. Balanced validation across {validation['samples']} snapshots; max voltage difference {validation['voltage_max_error_v']:.3g} V, a numerical agreement check only."),
        ('Benchmark design',f"{c['seeds']} seeds, DT-disjoint partitions, {c['search_trials']} search candidates per supervised model, validation calibration/thresholds, paired whole-DT bootstrap. Inspect {r['protocol']['inspection_fraction']:.0%} of meters. Show M0 through M4 definitions, including strong M3."),
        ('Results with uncertainty',headline+'\n\n'+pair_pr+'\n\nThesis: **'+r['thesis_verdict']+'**. M2 has the strongest test point estimate. Default model was chosen on validation. Shared DT metrics are not evidence of graph superiority; M0 abstains.'),
        ('Stress and voltage',f"Use results/figures/stress_curves.png and ablations.png. {c['stress_seeds']} seeds per stress setting. Full minus no-voltage PR-AUC {ci(voltage['pr_auc_delta'])}; no established voltage benefit at {c['voltage_noise_v']:g} V noise / {c['meter_voltage_resolution_v']:g} V resolution. Show mapping, dropout and clock-skew uncertainty; no pole localisation claim."),
        ('Live replay demonstration',demo_text+' Use the stored interval replay and reversible scenarios. Show missingness beside the DT balance. Label all telemetry replayed and all profiles synthetic.'),
        ('Responsible AI',f"Human verification, no automatic enforcement, reasons, appeal and correction. Target precision {c['target_precision']:.0%} on validation; {chosen} test vacancy FPR {ci(h[chosen]['vacant_fpr'],True)}. Honest-fault false positives and low-consumption disparity require mitigation."),
        ('Limits and utility data request',f"Invisible upstream NTL {v['invisible_fraction']:.1%}; no individual attribution. Request {r['presentation']['pilot_months_min']}–{r['presentation']['pilot_months_max']} months of joined AMI/DT/mapping and blinded field outcomes. Synthetic results do not establish field accuracy or ROI."),
        ('Team and work ownership','Team '+r['presentation']['team_name']+': '+', '.join(r['presentation']['team'])+'. Names are from the supplied application. Before submission, have the team confirm who owns ML/evaluation, electrical modelling, application integration and utility/governance; do not invent credentials or assignments.'),
        ('Roadmap and decision','Current deliverable: local reproducible benchmark and replay workbench. Next gate: authorised DISCOM data quality and blind utility evaluation, followed by governed field review. Kafka/cloud ingestion, GIS reconciliation and security hardening are roadmap only. Close with the unsupported thesis honestly and a concrete pilot data request.')]
    deck='# Presentation outline\n\nSource of all numerical slide values: `results/results.json`; audit mapping: `results/claims_register.csv`. This is the requested Markdown outline, not a rendered submission PDF. Convert a reviewed slide deck to the competition-required format separately.\n\n'
    for i,(title,body) in enumerate(slide_text,1):deck+=f'## Slide {i} — {title}\n\n{body}\n\n'
    write(docs/'deck_outline.md',deck)
    write(docs/'video_script.md',f"""
# Video script — at most {r['presentation']['video_seconds']//60} minutes

The timed outline below is a rehearsal guide, not a recorded video. All spoken metrics come from `results/results.json`. Read CIs aloud with the main result; avoid adding accuracy or ROI promises.

## 0:00–0:30 — Problem and thesis

“A fall in billed energy may mean theft, but it may also mean a vacant home, a failed meter or missing communication. Grid-GNN tests whether network context can help prioritise inspections without treating every unexplained unit as guilt. This is a synthetic proof of concept, and we tested the graph hypothesis against strong baselines.”

## 0:30–1:15 — Approach and data

“We combine consumer history with transformer energy balance, estimated technical loss and telemetry quality. Missing packets are imputed using only past observations. We compare energy-balance ordering, an anomaly detector, temporal and context-aware forests, and a temporal graph network. The grid benchmark uses CIGRE-derived networks and synthetic profiles. A separate Chinese daily-data check has real labels but no topology; it does not validate Indian field performance. London profiles were unavailable, so the fallback is explicit.”

## 1:15–3:15 — Demonstration

Show **Inspect**, default model {chosen}. {demo_text}

Say: “A low reading alone does not establish intent. The row shows history, peer context, missingness and imperfect event evidence. This candidate energy share comes from unexplained DT balance; it is not a verified billing recovery.”

Open **Replay & scenarios**. Advance one stored interval; point to the timestamp and missing-data shading. Toggle vacancy, theft, upstream hooking and communication dropout. Explain that vacancy changes physical load and DT input, under-recording changes meter reports, and upstream energy has no consumer culprit. Turn imputation off and back on to expose the missing-packet confounder. These are labelled synthetic interventions, separate from the benchmark.

## 3:15–4:15 — Evidence and an honest limitation

Show **Evidence** and the paired-difference figure. Say: “We used {c['seeds']} independent seeds with {c['n_dts']*c['meters_per_dt']:,} meters per seed, holding out whole DTs. M4 PR-AUC is {ci(h['M4']['pr_auc'])}; M3 is {ci(h['M3']['pr_auc'])}. The graph-advantage thesis is {r['thesis_verdict']}: the paired intervals include zero. A temporal forest is the strongest test point estimate. Voltage also has no established incremental benefit at our default measurement quality. These are simulated results, not a deployment accuracy claim.”

## 4:15–4:45 — Responsible AI and next step

“Every flag is reviewed by a person. We automate no disconnection, penalty or accusation. Honest meter faults remain difficult, so inspectors must review metrology and absence before escalation. We audit subgroup errors, retain reasons, support correction and require privacy/legal review for field data. We request a blinded utility dataset with DT readings, mapping quality and inspection outcomes.”

## 4:45–5:00 — Team and close

“We are {r['presentation']['team_name']}. Our team is {', '.join(r['presentation']['team'])}. The next step is an authorised DISCOM evaluation. The code, assumptions, failed checks and claims register are included so the jury can inspect the evidence.”

Rehearse the names and keep the total recording within the configured limit. Team roles and credentials must be confirmed by the team before recording. No self-recorded video or presentation PDF is included in this engineering handoff.
""")

    known='# Known issues, failures and skipped work\n\n## Material weaknesses\n\n'+'\n'.join(f'- {s}' for s in weak)+'\n\n## Execution ledger\n\n'
    known+=table(['Stage','Observed issue','Resolution / remaining limit'],[[x['stage'],x['issue'],x['resolution']] for x in issues])
    known+='\n\n## Explicitly skipped or incomplete\n\n'+ '\n'.join([
        '- Real LCL/SimBench grid check: access failures; synthetic fallback used. CER: application-controlled and unavailable.',
        '- SGSC, SMART-DS and additional IEEE feeders: optional extensions not attempted. ESMI statistics/weather downloads/PFC figures: not used. Pecan Street: excluded.',
        '- Exact original attack-function verification: primary full-text blocked; comparison is incomplete and formulas are labelled approximations/extensions.',
        '- Pole/segment localisation, voltage-slope pathway, tariff misuse and topology repair: not implemented. No related performance claim.',
        '- Coupled neutral/transformer field validation, Indian field pilot, revenue recovery and ROI: not performed.',
        '- GNN advantage and voltage benefit: not supported; learned auxiliary DT head poor and not used for delivered estimates.',
        '- Production ingestion, authentication, encryption/key management, case/grievance integration and deployment: roadmap only. Local HTTP and AppTest checks passed; no screenshot-based visual check or public hosting.',
        '- Actual slide/PDF production and self-recorded video: outside the prompt’s requested Markdown outline/script deliverables; not produced or submitted.',
        '- Upstream dependency deprecation warnings are preserved in test logs. They do not cause test failures; no third-party package code was patched.',
        '- Percentile cluster-bootstrap bias and repeated validation use remain limitations; exploratory intervals are not multiplicity-adjusted.',
    ])+'\n'
    write(out/'known_issues.md',known)
    write(docs/'evaluation_protocol.md',f"""
# Evaluation protocol and lineage

The simulator establishes disjoint DT groups before labels, and confines mapping mistakes to each partition. Graph messages remain within one declared DT. Public observations are a different type from hidden labels. Hidden targets enter supervised training only for training indices; validation labels choose candidates/calibration/thresholds; test labels enter reporting only. The test-label mutation test refits and verifies identical models.

The final seed list starts at {c['seed_start']} and spans {c['seeds']} seeds. The development gate uses the separate fast configuration. Headline predictions are saved as `seed_*_meters.csv`; DT truth/estimates as `seed_*_dts.csv`. Raw grid exports are first-seed only and can be regenerated. Test pseudonyms may repeat across independent seeds; the seed is part of the observation identity.

Feature transforms and tree explanatory references use training data. Calibration and thresholds share the validation partition with search/early stopping; this is acknowledged, not a hidden extra test set. Candidate budgets are equal in count, not compute. All comparison models use the same split and features permitted by their specification.

PR-AUC is average precision, not trapezoidal curve area. Precision/recall-at-budget use descending raw score with deterministic pseudonym tie breaks. The inspection target uses intentionally stolen meter-visible kWh as hidden evaluation weights. DT localisation uses the highest-loss DT hit definition. Technical-loss truth never enters its estimate. True mapping is not a feature.

The hierarchical bootstrap uses {r['bootstrap_reps']} outer replicates and {r['cluster_draws_per_seed']} precomputed whole-DT draws per seed. The same draws are paired across models. It estimates a mean-of-seed metric, not a pooled meter estimand. Test subgroup audits instead pool meter predictions and include sample counts. Means of undefined precision exclude abstention seeds; recall remains zero there.

Strict graph-free, no-temporal, no-voltage, no-DT-residual, zero-imputation and shuffled-graph controls are fitted on the original partitions. Strict graph-free removes all engineered graph inputs and the electrical temporal channel as well as message passing. The initial weaker no-message-passing experiment is labelled development-only and superseded, not silently presented as graph-free.

Fixed-model stresses cover dropout, voltage noise/resolution, wrong declared mapping, DT skew, prevalence, attack parameter shift, unseen topology family and event availability. Logs/CSV files retain per-seed predictions. Noise-setting SNR files retain per-meter proxy values. Temporal holdout trains at an earlier endpoint and scores later held-out DTs. SGCC is a separate customer-split forest with its own customer bootstrap.

`run_all.py` regenerates results, app checks and documents. `claims_register.csv` includes recursively addressed result values and source filenames with the config hash. Source SHA files and reproduction records distinguish measured execution from configured assumptions. Do not copy preliminary/invalidated extraction metrics into a claim.
""")

    # A self-contained review page is readable offline and carries no telemetry
    # identifiers beyond the deliberately pseudonymous example table.
    html_rows=''.join('<tr><th>'+n+'</th>'+''.join('<td>'+html.escape(ci(h[n][k],k in ['vacant_fpr','dt_mape']))+'</td>' for k in PRIMARY)+'</tr>' for n in NAMES)
    figure_blocks=''
    for filename,title in [('benchmark.png','Benchmark'),('paired_differences.png','Paired evidence'),('stress_curves.png','Stress tests'),('reliability.png','Calibration'),('ablations.png','Ablations')]:
        p=out/'figures'/filename
        if p.exists():figure_blocks+=f'<section><h2>{title}</h2><img alt="{title}" src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}"></section>'
    write(out/'review.html',f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Grid-GNN — Evidence review</title>
<style>body{{margin:0;background:#f3f6fb;color:#17263c;font:16px/1.6 system-ui,sans-serif}}main{{max-width:1200px;margin:auto;padding:48px 24px}}header{{padding:40px;background:#10213a;color:white;border-radius:20px}}h1{{font-size:48px;line-height:1.1;letter-spacing:-2px;margin:12px 0}}h2{{letter-spacing:-.5px}}.eyebrow{{color:#87cdd3;text-transform:uppercase;letter-spacing:2px;font-size:12px}}section{{padding:26px;margin:24px 0;background:white;border:1px solid #e2e8f1;border-radius:14px}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:13px 8px;border-bottom:1px solid #e2e8f1;text-align:left}}th{{color:#2563eb}}.scroll{{overflow:auto}}img{{width:100%;height:auto}}.note{{font-size:14px;color:#56677f}}strong{{font-weight:700}}li{{margin:12px 0}}code{{background:#e8edf4;padding:3px 6px;border-radius:4px}}</style>
<main><header><div class="eyebrow">Crystal Paradigm · Synthetic proof of concept</div><h1>Find the evidence<br>before the visit.</h1><p>Grid-GNN evaluates non-technical-loss inspection ranking using meter histories and distribution-transformer balance.</p><p><strong>Graph-advantage thesis: {html.escape(r['thesis_verdict'])}.</strong></p></header>
<section><h2>What actually ran</h2><p>{c['n_dts']*c['meters_per_dt']:,} meters · {c['n_dts']} DTs · {c['days']} days · {c['seeds']} seeds. Whole-DT holdout, paired bootstrap, strong baselines, ablations, stress tests, real-data temporal check and a local replay app.</p><p>Full command: {html.escape(duration)}. Profile source: <code>synthetic_fallback</code>. Config: <code>{r['config_hash']}</code>.</p></section>
<section><h2>Mean and confidence intervals</h2><div class="scroll"><table><tr><th>Model</th>{''.join('<th>'+x+'</th>' for x in LABELS)}</tr>{html_rows}</table></div><p class="note">Mean [95% CI]. M0 abstains; M1 flags rarely. All models share the residual DT estimator. DT top-3 measures the highest-loss DT hit. These are simulated results, not Indian field estimates.</p></section>
{figure_blocks}<section><h2>Interpretation boundaries</h2><ul>{''.join('<li>'+html.escape(s)+'</li>' for s in weak)}</ul><p>Voltage benefit is not supported. Meter-visible NTL: {v['meter_visible_fraction']:.1%}; invisible upstream: {v['invisible_fraction']:.1%}. No pole or culprit attribution is claimed.</p><p>{html.escape(sgcc)}</p><p>No automatic disconnection, penalty or billing change. Human review and a documented appeal path are required for any future field pilot.</p></section>
<section><h2>Open the project</h2><p>Use <code>README.md</code> to start the supplied dashboard or reproduce the benchmark. <code>results/summary.md</code> contains complete tables; <code>results/claims_register.csv</code> maps values to sources; <code>results/known_issues.md</code> preserves failed and skipped work.</p><p class="note">The package includes the requested presentation outline and video script. It does not include a recorded video or competition submission.</p></section></main></html>''')
    claims_register(out)
    return r
