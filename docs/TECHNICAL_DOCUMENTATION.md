# Grid-GNN — Technical Documentation

**Project:** Rodic InfraAI Innovation Challenge 2026 — *AI for India Energy Stack*
**Team:** Crystal Paradigm
**Live MVP:** API `https://grid-gnn-api.onrender.com` · Dashboard `https://grid-gnn-dashboard.vercel.app` (frontend import pending at time of writing)
**Document date:** 8 October 2026

---

## 1. Executive summary

Grid-GNN is a decision-support system that ranks low-voltage (DT) customers by
their likelihood of non-technical loss (NTL — theft, meter tampering, bypassing)
from 15-minute smart-meter telemetry, attributes unexplained energy to
transformer-level anomalies, and exposes everything through a scoring API and an
inspection-planning dashboard.

The detection philosophy is **physics-informed residual analysis**: honest
consumption obeys energy conservation between the distribution-transformer (DT)
input meter and the sum of customer meters (plus network losses). Where
recorded energy is inconsistent with physics or with a customer's own
consumption history, the deficit is quantified and attributed. A graph layer
(GNN) was built and benchmarked; rigorous paired evaluation showed it did not
beat graph-free baselines, so the shipped detector is the evidence-supported
residual rule — every model remains available and swappable through the API.

Everything in this document operates on **100% synthetic telemetry**
(`synthetic_india_v2`, a third-party research dataset with a published audit).
No real customer data exists anywhere in the system. Section 8 covers data
privacy and the protocol for onboarding real DISCOM telemetry.

---

## 2. Approach

### 2.1 Problem

India's distribution grid loses a large share of supplied energy to Aggregate
Technical & Commercial (AT&C) losses. Disaggregating **technical** losses (conductor
impedance, transformer cores) from **non-technical** losses (hooking, meter
bypass, under-registration) at the local feeder/DT level is the core
intelligence problem: DT input energy that exceeds the sum of customer
consumption plus estimated technical loss is unexplained — and unexplained
energy is the only ground truth a DISCOM actually has before an inspection.

### 2.2 Two-level detection

**Meter level — "is this customer under-recording?"**
Each customer is scored against *its own* baseline history (class-matched,
weather-aware expectation), producing a residual rule score (M0), an
IsolationForest anomaly score (M1), and — where labelled data permits training —
supervised RandomForests (M2/M3) and a temporal graph network (M4). Features
include consumption ratios, missing-interval behaviour, voltage-physics
consistency (the customer's *reported* voltage is checked against the voltage
implied by the network physics for its network position and phase), and
DT-context peer aggregates computed leave-self-out.

**DT level — "where is unexplained energy?"**
Per 15-minute interval: `DT input kWh − Σ customer kWh = technical loss + NTL`.
The residual fraction is a physics-bound signal that does not depend on labels
at all. Unexplained energy at a DT is then apportioned to candidate meters
proportionally to their anomaly scores — a *heuristic allocation for inspection
prioritisation*, explicitly not an assertion of theft or recoverable billing.

**Why the two levels matter:** some theft classes are physically invisible at
the meter (direct hooking upstream of the meter is not recorded by the meter
itself) — only the DT balance can see them. Conversely, single-customer
under-registration vanishes inside 50-customer aggregation noise — only the
meter-level score sees it. The system is designed so that neither class is
silently missed, and both limitations are stated in every response.

### 2.3 Inspection-supervised headline (label policy)

The demonstration dataset provides 203 synthetic "inspections" — sparse,
selected and biased, as real DISCOM inspection logs are. Our headline policy:

1. Models never train on hidden ground truth.
2. Inspection outcomes calibrate/evaluate the detectors; uninspected meters are
   **never** treated as negatives; "not accessible" inspections are excluded.
3. Temporally impossible inspection outcomes (theft "found" before the stated
   theft onset — 8 rows flagged by an independent audit) are **quarantined,
   never relabelled**.
4. Hidden synthetic truth is used only in a separately *declared* oracle
   evaluation, clearly labelled as such and never presented as field
   performance.

### 2.4 Evidence honesty

Every number carries provenance and confidence intervals (paired bootstrap over
DTs). The claims register in the repository records what is measured,
what is assumed, and what is *not* claimable. Two headline examples:
the graph model's paired improvement over graph-free baselines includes zero
(−0.041 [−0.106, 0.049] PR-AUC) — so the shipped default is the simpler
detector; direct hooking is invisible at meter level by construction — the
system says so in its own API notes.

---

## 3. Data

### 3.1 Dataset in use

| Property | Value |
|---|---|
| Identity | `synthetic_india_v2` (merged), third-party research prototype |
| Scale | 2,500 meters · 50 distribution transformers · 12,192,000 × 15-min readings |
| Coverage | 2025-10-01 onward (Asia/Kolkata); 500 m × 14 days + 2,000 m × 60 days |
| Side tables | DT readings (kWh, kvarh, V, I×3), topology edges (r, x, length), weather, calendar, outage logs, 203 inspections |
| Hidden truth | 228 theft-ever meters (9.1%) across 8 archetypes — oracle-evaluation only |
| Provenance | 100% synthetic; generator not supplied; published structural audit (12.19M rows: 0 duplicate keys, 0 orphan ids, 0 off-grid timestamps, 0.40% missing intervals) |

The demo deployment serves the 14-day cohort (440 meters after uniform
sub-sampling, 10 DTs); the 60-day / 2,000-meter cohort is the evaluation
benchmark. Scenario packs from our own generator (8 packs, T1–T8 theft
families, honest-fault hard negatives) are retained as scenario-design
reference.

### 3.2 Curation protocol applied

Adopted from the dataset's independent audit pack and enforced in code
(`src/data/merged_adapter.py`):

- 8 inconsistent inspections quarantined (never relabelled);
- whole-cohort train/validation/test split (no DT crosses partitions);
- `event_flags`, `feeder_id`, `pole_id`, truth tables: **excluded from all
  predictors** (the former is an artificial shortcut; the latter leak the true
  mapping);
- the 206 GIS pole-reference conflicts are re-seated on the declared DT's
  topology (documented assumption) so network physics remain coherent;
- absence of readings is treated as *absence of coverage*, never as zero
  consumption.

### 3.3 Interfaces

Internally the data is normalised to an `Observations` object (public telemetry
only: meter/DT arrays, masks, topology, config) with hidden truth kept in a
strictly separate structure. A contract test forbids any feature name derived
from label-bearing columns, and perturbation-invariance tests verify that
tampering with hidden labels cannot change any served feature.

---

## 4. Technical implementation

### 4.1 System architecture

```
 Browser (static dashboard, Vercel)
   │  fetch  ──────────────────────────────┐
   ▼                                       │
 FastAPI (uvicorn, Render Docker)          │
   ├─ GET  /health   → fitted models, config hash, provenance
   ├─ POST /score    → full pipeline (below)
   └─ results/audit_log.jsonl (local decision-support audit trail)
   │
 Scorer (pinned core)
   ├─ demo_context.joblib   (Observations: telemetry + topology + config)
   ├─ models.joblib         (M0…M4 with calibrators + operating thresholds)
   └─ selected_model.json   (selected model, config hash, training note)
   │
 Pipeline per /score request
   ├─ deep-copy stored context → apply submitted readings/DT readings
   ├─ optional reversible scenario counterfactual
   │    (vacancy / theft / upstream_hooking / ami_dropout)
   ├─ build_features: causal imputation → own-history features →
   │    voltage-physics consistency (pandapower backward sweep per DT) →
   │    leave-self-out peer (graph) features → DT residual estimator
   ├─ model.predict → raw score → calibrated probability
   ├─ operating threshold (validation-precision-gated flag)
   ├─ kWh allocation of DT unexplained energy across candidates
   └─ response: ranked meters, DT table, daily balance, reason codes, notes
```

### 4.2 Feature pipeline (per scoring request)

1. **Imputation:** causal (past-only) imputation of missing intervals; imputed
   energy is tracked separately so missingness can never masquerade as theft.
2. **Own-history features:** recorded-to-expected energy ratio, deficit,
   missing fraction, day/night splits, recent-behaviour aggregates.
3. **Voltage physics:** per DT, customer power is placed on the real edge list
   (r, x, length) by phase; a backward/forward sweep yields expected node
   voltages; reported voltages inconsistent with physics contribute evidence
   (bypassed meters still draw voltage).
4. **Peer context:** leave-self-out class-matched aggregates per DT (a meter's
   own anomaly can never define its own peer baseline).
5. **DT estimator:** residual-based unexplained-energy estimate with technical
   loss, missing-energy and hooking terms separated.

### 4.3 Models

| Model | Type | Role |
|---|---|---|
| M0 | Unsupervised residual rule | Shipped default; transparent, reason-coded |
| M1 | IsolationForest (own-history features) | Alternative unsupervised detector |
| M2 | RandomForest (own-history features) | Supervised; requires labels |
| M3 | RandomForest (+ DT-context features) | Supervised |
| M4 | TemporalGraphNet (PyTorch) | Graph model; benchmarked, not defaulted |

All models emit raw scores that are Platt-calibrated on validation data;
operating thresholds are gated on a validation precision target and explicitly
disable inspection flags when the target is unattainable (the honest behaviour
with 6 usable positive inspections). The demo deployment serves **M0 + M1**;
the supervised models require the labelled 60-day context.

### 4.4 API surface

`GET /health` → service status, selected model, **fitted-model list**, training
provenance note, config hash, `automated_enforcement: false`.

`POST /score` — request (all fields validated by strict Pydantic models, extra
fields forbidden, NaN/Inf rejected):

```json
{
  "readings":     [{"meter_id": "M000270", "ts": "2025-10-14T00:00:00+05:30",
                    "kwh": 0.31, "voltage_v": 228.4, "event_flags": 0, "is_missing": false}],
  "dt_readings":  [{"dt_id": 0, "ts": "2025-10-14T00:00:00+05:30", "kwh_in": 4.61}],
  "model": "M0",
  "as_of_interval": 1344,
  "scenario": "none | vacancy | theft | upstream_hooking | ami_dropout",
  "meter_id": "M000270",
  "severity": 0.6,
  "tamper_event": false,
  "imputation": true
}
```

Response (abridged): ranked `meters[]` (rank, calibrated probability,
inspection flag, candidate kWh allocation, missing fraction, recorded-to-
baseline ratio, human-readable reason codes), `dts[]` (unexplained kWh,
allocated/unassigned split, technical-loss estimate, missing fraction,
"DT only for unmetered upstream load; no culprit inference"), daily balance
series, per-interval packets, optional SHAP-style explanation, and a fixed
`notes[]` block stating: synthetic profiles; probabilities not validated for
field use; allocation is a heuristic; a field inspector must verify every flag;
no disconnection or penalty is automated.

Unknown meter ids, out-of-window prefixes, misaligned timestamps and
non-finite values are rejected with 422 (pinned safety behaviour).

### 4.5 Frontend

Static single-page dashboard (HTML/CSS/JS + Plotly 2.35.2 CDN), no build
chain: model/scenario/severity controls, sortable-and-filterable rankings table
(flag-only view, per-column sort), DT balance and explanation charts,
cold-start-aware fetch layer (502/503 retry with 15 s backoff, 60 s per-request
timeout), model dropdown populated from `/health`, config/model provenance
badges. The dashboard holds no state, sends no telemetry anywhere except to the
configured API, and can target any backend via `?api=<url>`.

### 4.6 Quality gates

- 22 tests green in the deployment repository (36 including development-only
  data-dependent suites): energy conservation closure (<1e-5 pre-noise),
  leakage/perturbation invariance, label-isolation, determinism
  (byte-identical exports per seed), API contract and end-to-end scoring smoke.
- Every artefact (models, context, thresholds) is bound to a `config_hash`;
  the API self-reports its hash so a reviewer can tie a response to an exact
  configuration.
- Local pre-deployment verification: container-identical run measured at
  314 MB RSS (fits Render's 512 MB free tier), 0.53 s scoring runtime.

---

## 5. Evaluation and results

Two separated tracks on the 60-day / 1,800-meter evaluation cohort
(full protocol and numbers: `results/merged_eval.json` / `.md` in the repo):

**Track A — inspection-supervised (headline).** M0 ranking vs the 131
quarantined-clean `no_theft` inspections: the 3 usable `theft_found` meters'
scores stochastically exceed controls (one-sided Mann-Whitney p = 0.012);
ranks 127/185/505 of 1,800; no top-50 claim is made. Reported as indicative
only — six usable positives dataset-wide.

**Track B — declared oracle (evaluation only).** M0 vs hidden truth:
**ROC-AUC 0.908, PR-AUC 0.674** at 9.4% prevalence. Per-family: frozen meters
17/18 in top-50; suppression/bypass families well-ranked; direct hooking
invisible at meters (median rank 880) — with the DT-level balance signal
reported honestly as weak on this dataset (AUC 0.542), consistent with the
audit's unresolved hooking-onset semantics.

**Inherited-context benchmark** (20 seeds, DT-disjoint): best supervised model
PR-AUC 0.644 [0.584, 0.707]; graph model M4 paired Δ −0.041 [−0.106, 0.049]
vs graph-free — the shipped default is therefore the simpler detector, and the
GNN remains one API switch away.

**No claim is made or implied** about Indian field prevalence, detection
performance on real AMI data, or revenue recovery. Every such boundary is
stated inside the API responses themselves.

---

## 6. Tools, frameworks and APIs

| Layer | Technology (pinned versions) | Role |
|---|---|---|
| API | FastAPI 0.115.12, Pydantic 2.13, uvicorn 0.34.2 | Typed, validated scoring endpoints |
| Detection | scikit-learn 1.8.0 (IsolationForest, RF, calibration), custom rule | M0–M3 |
| Graph model | PyTorch 2.5.1+cpu (custom TemporalGraphNet) | M4 |
| Physics | pandapower 3.2.1 (backward/forward sweep, CIGRE reference networks) | Voltage consistency, loss estimation |
| Data | numpy 2.5.3, pandas 2.3.3, PyArrow 20 (Parquet), scipy 1.18.1 (statistics) | Pipeline + evaluation |
| Frontend | Vanilla JS + Plotly 2.35.2 (CDN) | Static dashboard |
| Packaging | Docker (python:3.12-slim), pinned `requirements.txt` | Reproducible image |
| Hosting | Render (Docker web service, `$PORT`-aware) · Vercel (static) · GitHub (2 repos) | Deployment |
| Testing | pytest 8.3.5 | Quality gates |

Substitutions vs the original proposal, with reasons: **pandapower** replaces
OpenDSS (Python-native, same LV-network reference models, already the
validated physics gate of the inherited stack); **replay-based scoring**
replaces Kafka streaming (a PoC needs deterministic, reproducible scoring;
the request schema is stream-shaped and accepts per-interval packets, so a
Kafka consumer is an adapter, not a redesign); **Render/Vercel** replaces GCP
(free-tier reproducibility for the challenge; the container is portable to any
cloud).

---

## 7. Deployment topology

- **Backend** — Docker image from `SEN-Adarsh/grid-gnn-api`; the pinned scoring
  core is parameterised by `GRID_CONFIG`, so the served context (dataset, models,
  window) is switched by one environment variable with no code change. Current
  context: merged demo cohort, config hash `6e60f7bd16aff2ef`. Prebuilt
  artefacts are committed so the service never needs the raw dataset at runtime.
  Free-tier footprint: ~314 MB RSS; cold start after ~15 min idle.
- **Frontend** — static files from `SEN-Adarsh/grid-gnn-dashboard` on Vercel;
  auto-deploys on push; API endpoint configurable per deployment or per-URL
  (`?api=`).
- **Audit trail** — every `/score` call appends a local JSONL record (timestamp,
  model, model hash, config hash, input hash, flagged meters,
  `automated_action: "none; decision support only"`).

---

## 8. Data privacy and compliance

### 8.1 Current state: nothing personal is processed

| Property | Status |
|---|---|
| Personal data | **None.** All telemetry is synthetic; meter ids are fictional (`M000270`); no names, addresses, phone numbers, billing or payment data exist in the system |
| Real individuals identified/inferable | No — no real customer exists behind any id; coordinates are synthetic fiction |
| Data subject rights (access/erasure/rectification) | Not applicable to synthetic data; the mechanism still exists trivially (artefacts are rebuildable from source parquet + config hash) |
| Consent | Not applicable (no data subjects); the dataset licence/README terms are respected (truth directory used for oracle evaluation only, redistribution of claims avoided) |
| Cross-border transfer | Not applicable (no personal data); processing occurs on Render/Vercel infrastructure only as compute for synthetic arrays |
| Data residency | Not applicable for personal data; for future Indian utility deployments, DPDP-aligned processing agreements and region selection (e.g. `ap-south-1` equivalents) are the plan |

### 8.2 Client (browser) privacy

- No cookies, no `localStorage`/`sessionStorage`, no analytics, no trackers,
  no fingerprinting (verified by source scan).
- The only third-party resource is the Plotly charting library from its
  official CDN. The dashboard can be self-hosted/served offline by vendoring
  this file — one-line change.
- The browser sends data **only** to the configured backend (`API_BASE`); there
  is no other egress.

### 8.3 Server privacy

- The API is stateless per request; no session store, no auth store (public
  demo, no credentials exist to leak). CORS is intentionally open for the
  public demo; for utility deployment it is one middleware line to restrict.
- The only persisted server artefact is the decision-support audit log
  (JSONL): timestamps, model/config hashes, a SHA-256 of the request payload,
  and flagged **synthetic** meter ids. It contains no raw telemetry and no
  personal data.
- No secrets, tokens or keys exist in either repository (verified by pre-push
  secret scans); the demo requires no credentials.

### 8.4 Onboarding real DISCOM telemetry (the protocol we commit to)

The application anticipates validation on Rodic's anonymised IPDS/RDSS
telemetry. Our commitment before any real data enters this system:

1. **Lawful basis & DPA** — processing under the utility's instructions via a
   data-processing agreement; DPDP Act 2023 principles applied
   (purpose limitation, storage limitation, minimality).
2. **De-identification at source** — meter/account ids tokenised by the
   utility before transfer; no names, addresses, tariff arrears or identity
   documents; only operational fields the pipeline actually consumes
   (interval kWh/kvarh/V, topology ids, outage and inspection outcomes).
3. **Aggregation safeguards** — DT-level statistics reported with minimum
   cohort sizes; per-meter outputs restricted to authorised inspection users
   with role-based access; every access written to the audit log.
4. **Retention & erasure** — time-boxed retention for the engagement,
   deletion on schedule with hash-verified certificates; model artefacts
   retrainable, so erasure does not break the system.
5. **Security** — secrets in a managed vault, TLS everywhere, least-privilege
   service accounts, container image scanning.
6. **No automated adverse action** — the system is architecturally incapable
   of disconnecting or penalising anyone (`automated_enforcement: false` is a
   fixed field); every flag routes to a human field inspection. This is a
   deliberate fairness safeguard, not a limitation.
7. **Fairness monitoring** — flag-rate ratios across consumption segments are
   computed and reported (the inherited evaluation publishes such ratios, e.g.
   low-consumption-meter flag-rate 2.21× overall for one model), so disparate
   impact on small consumers is measurable before any field use.

---

## 9. Limitations (stated, not hidden)

- All quantitative results are on synthetic data; **no Indian field performance
  is claimed** anywhere in the product, reports or this document.
- Direct hooking is invisible to any meter-level detector by physics; the DT
  balance is the only honest signal, and on the current dataset it is weak —
  the dataset's own audit flags unresolved hooking-onset semantics (we
  quarantine the consequences rather than repair them, since the generator is
  not supplied).
- Candidate kWh allocation is a prioritisation heuristic, not an audit finding.
- The supervised models (M2–M4) are only meaningful with a labelled context;
  the demo's inspection-only policy limits them by design.
- Model artefacts are bound to synthetic class mixes; recalibration on real
  utility data is a prerequisite to any pilot, not an afterthought.

---

## 10. Artefact map

| Artefact | Location |
|---|---|
| Backend repository (API, pipeline, tests, protocol, eval results) | `github.com/SEN-Adarsh/grid-gnn-api` |
| Frontend repository (dashboard) | `github.com/SEN-Adarsh/grid-gnn-dashboard` |
| Live API | `https://grid-gnn-api.onrender.com` (`/docs` for OpenAPI) |
| Evaluation report | `results/merged_eval.md` (+ machine-readable `.json`) in the backend repo |
| Claims register | `results/claims_register.csv` — every claim with provenance + CI |
| Assumptions log | `docs/ASSUMPTIONS.md` |
| Engineering handoff log (all deviations, pins, deployment notes) | `docs/HANDOFF_LOG.md` |
| Known issues (dataset defects + mitigations) | `results/known_issues.md` |
| Deployment guide | `DEPLOYMENT_GUIDE.md` (workspace) |
