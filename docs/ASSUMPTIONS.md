# ASSUMPTIONS.md — WP-13 synth_v2 generator

**Status of every value in this file: ASSUMPTION.** None is a measured Indian
value. They come from general knowledge of how theft and metering behave and
from the research summarised in the agent brief. The generator is **not**
calibrated to any real dataset, and no synthetic run predicts Indian field
performance.

Inherited parameters (load shapes, noise levels, outage structure, dropouts,
mapping-error rate, event sensitivity/FP rates) carry over from
`configs/final.yaml` and are likewise assumptions; see the inherited docs.
Only v2-specific parameters are enumerated here.

## Entities and network (Section 2)

| Parameter | Value | Assumption rationale |
|---|---|---|
| `class_mix` | 80% domestic / 15% small commercial / 5% agricultural+other | Typical LV-feeder mix narrative; not measured. |
| `three_phase_prob_by_class` | domestic 3%, commercial 35%, agricultural 60% | Larger connections are more often three-phase. |
| `long_radial_dt_fraction` | 0 (packs opt in) | Variant for topology-shift tests; trunk spans 25–45 m, conductor 0.45–0.90 Ω/km, service drops 8–30 m — generic LV values, no measured Indian feeder. |
| `area_mix` | 40/35/25 urban/semi-urban/rural (DT level) | Descriptive only. |

## Load model / seasonality (Section 3)

| Parameter | Value |
|---|---|
| `temp_mean_c` | 31 °C |
| `temp_seasonal_amplitude_c` | 6 °C (annual sinusoid) |
| `temp_seasonal_phase_days` | +20 d (peak heat late May/June) |
| `temp_daily_amplitude_c` | 3 °C |
| `cooling_threshold_c` | 29 °C; cooling gain per customer U(0, 0.055) kW/°C |

## Honest confounders (Section 4)

| Parameter | Value |
|---|---|
| `confounder_fractions` | vacancy 6%, seasonal occupancy 4%, meter fault 4%, behaviour-up 3%, behaviour-down 3%, solar 6%, disconnect/reconnect 3% |
| `meter_fault_split` | stuck 30% / intermittent zeros 25% / drift 25% / spikes 20% |
| Vacancy remaining load | U(0.02, 0.22); seasonal U(0.08, 0.42) with 40% of weekly windows returning |
| Disconnect/reconnect | 1–2 zero blocks of 2–10 days, then normal |
| Solar capacity | U(0.4, 2.0) kW midday sin-shape; net-metering-like |
| Behaviour-up | +U(1.2, 3.2) kW night block |
| Behaviour-down | remaining load ×U(0.4, 0.7) from onset |
| Meter drift rate | U(0.05, 0.30) reading change per 30 days, either sign |
| Meter spike rate | 0.004 per interval, ×U(3, 15) |

## Theft families (Section 5) — all ranges are assumptions

| Family | Parameters |
|---|---|
| T1 partial bypass | α ~ U(0.2, 0.8) of load bypassed from onset |
| T2 night bypass | α ~ U(0.2, 0.8) in 18:00–02:00 window (11:00–18:00 variant) |
| T3 hooking | draw U(0.3, 2.0) kW × (0.4 + evening shape); meter untouched; DT-level only |
| T4 magnetic | reported scale β ~ U(0.3, 0.7) + magnetic-influence events (p=0.5) |
| T5 neutral/phase | tampered phase reported at U(0.2, 0.5); three-phase customers only; effective meter deficit = (1−factor)×largest phase share |
| T6 slow ramp | α_end ~ U(0.3, 0.7) over U(14, 35) days |
| T7 zero/flat | zero or ~20th-percentile constant from onset |
| T8 cluster | 3–8 neighbours on one DT get T1/T3 together (50/50) |

Prevalence: per-DT U(2%, 15%) with 40% of DTs theft-free (configurable);
family weights t1..t8 = 22/18/15/12/8/10/10/5%. Theft onset in days 30–45 of
60 (15% of theft persistent from day 0). These are demo values, chosen to
exercise the pipeline, not to match any measured prevalence.

## Typed event log (Section 6) — names are UNCONFIRMED against the Indian
smart-meter standard and the DISCOM meter specification; confirm before
anything is described as realistic.

| Parameter | Value |
|---|---|
| Theft-linked linkage | magnetic for T4, neutral disturbance for T5 (p=0.5); cover-open for T1/T2/T7 (p=0.4); rates 1.5/1.2/0.8 per active day |
| Stealth theft | 50% of theft customers get no linked events at all |
| Honest events | 30% of meters carry background cover-open (0.05/day), comm-failure (0.10/day), magnetic (0.02/day) |
| Power failure | one event per meter per outage block |
| Voltage sag/swell | threshold ±10% of nominal, up to 3 meters sampled per DT event |
| Reverse current | solar customers, p=0.5 linkage, 0.5/day |
| Comm failure on dropout | 60% of dropout burst starts |

## Data problems (Section 7)

| Parameter | Value |
|---|---|
| `dropout` / burst | inherited (5% / 8% of meters); pack P5 uses 25%/50% |
| `per_meter_clock_drift_minutes` | ±4 min (uniform, sub-interval interpolation) |
| `glitch_rate` | 0.0005 per interval; multiplier −1 or ×10, left in reported data |
| `mapping_error` | inherited 5% whole-record cross-DT swaps (within partition) |

## Freeze and seeds (Section 1)

- Generator-development seeds: 7000xx (eyeballing only, never reported).
- Evaluation seeds: 4100xx (`seed_eval` in each pack config), untouched by development.
- Config hash (`src.common.config_hash`) recorded in each pack's `metadata.json`
  and `data_card.json` before any model evaluation runs on that pack.
- If the generator changes after results are seen: log in `known_issues.md`
  and rerun everything (WP-13 Section 1.1).

## Sanity-check policy (Section 9)

Synthetic honest-customer summaries may be compared qualitatively (shape
only) with SGCC, Irish CER / Low Carbon London, and Indian aggregate curves
(Karnataka SLDC, Delhi hourly, Kanpur profile). Similar-in-shape is the
strongest claim allowed; see `docs/SYNTH_SANITY.md`.

## WP-14 addendum — merged `synthetic_india_v2` (final primary dataset)

The primary dataset is now the third-party merged archive (`synthetic_india_v2`,
SHA-256 recorded in HANDOFF_LOG WP-14). Its parameters — load profiles, theft
archetype mixes, prevalence (228/2500 = 9.1% theft-ever), confounders, outage
and inspection processes — are the dataset authors' SYNTHETIC ASSUMPTIONS; the
generator was not supplied, so none of them can be independently recomputed.
They are not Indian measurements and the dataset's own README says the same.

Our adopted policies (assumptions we chose, all documented):
- Inspection-supervised headline; hidden truth only for declared oracle
  evaluation; uninspected/not_accessible never negatives.
- 8 temporally inconsistent inspections quarantined (audit), never relabelled.
- Audit split proposal adopted as-is (cohort-level train/val/test).
- Uniform 45/44 meters per DT subsample for interface compatibility.
- 206 GIS pole conflicts re-seated at the declared DT's same-position pole.
- `agricultural_pumpset` mapped to `agricultural`; `small_industrial` kept
  distinct; `area_type` = constant `synthetic_mixed`.
- `event_flags`, `feeder_id`, `pole_id`, truth columns: never predictors.
- Demo cohort `baseline_days` capped at 7 (14-day history).
