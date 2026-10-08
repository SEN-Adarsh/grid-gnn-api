# Known issues, failures and skipped work

## Material weaknesses

- Synthetic transfer: generated load and weather on one principal reference topology family; no Indian field labels and no real-LCL/SimBench grid validation.
- Unsupported graph advantage: paired graph/GNN improvement intervals include zero; validation reuse for search, calibration and thresholds can overfit a small DT partition.
- False accusations remain possible: honest meter faults have pooled FPR 46.2%; low-consumption meters have flag-rate ratio 2.21 against the overall population for M4.
- Incomplete physics and metrology: ideal neutral, no mutual coupling or transformer dynamics, assumed loss parameters, stale mapping/phases and clock error. Balanced solver agreement is a narrow check.
- Attribution and kWh limits: invisible upstream NTL is 15.0%; DT MAPE is 45.7% [36.9%, 55.5%]; posterior allocation is not established theft or billing recovery.

## Execution ledger

| Stage | Observed issue | Resolution / remaining limit |
| --- | --- | --- |
| environment | Initial uv request for the CPU torch index timed out | Retried through pip; CPU torch installed successfully |
| environment | Initial Streamlit and pandapower pins had incompatible packaging requirements | Selected compatible Streamlit pin before installation |
| data | London Datastore page and download access returned HTTP 403 | Explicit synthetic-profile fallback; no LCL validation claimed |
| implementation | Initial simulator compile caught an extra closing parenthesis | Corrected before executing any experiment |
| baseline_gate | Some methods could not meet validation target precision and abstained | Operating point marked unattainable; no-flag precision remains null, not perfect |
| data | Downloaded SGCC file disagrees with repository-advertised row and day counts | Initial row mismatch was traced to incomplete extraction, not to the author release; invalidate the first check and use the CRC-verified rerun. Any remaining date-column discrepancy is reported separately. |
| reproducibility | Parallel RF prediction summation differed at floating-point roundoff in the exact-repeat test | Serial prediction reduction; exact repeat and test-label isolation now pass |
| export audit | Initial exported edge names were not directly joinable to meter identifiers; demo DT renumbering also left pole prefixes unchanged | Use public meter/pole identifiers and remap demo prefixes; data exports repaired; model inputs and metrics unchanged |
| explanations test | A float32 test predictor caused a small grouped-Shapley additivity rounding error | Promote prediction values to float64 before coalition arithmetic; additivity test now passes |
| literature | Original Jokar full-text and later attack-table article fetches were blocked | Indexed primary-source attack table and abstract checked; exact original-function verification remains incomplete and is disclosed |
| stress execution | Forked stress workers stalled after the parent used multithreaded PyTorch; CPU time stopped advancing | Switched both pools to spawn and resumed saved baseline/ablation checkpoints; clean reproduction uses spawn throughout |
| ablation audit | Initial no-graph control removed message passing but retained engineered graph inputs | Strengthened the graph-free control to remove peer, DT, electrical summary inputs and the electrical temporal channel as well; final clean reproduction recomputes this ablation. Headline model and search settings unchanged. |
| real-data integrity | Native split-ZIP conversion failed CRC and left a partial SGCC CSV. The initial temporal check used that incomplete extraction and is invalidated. | Original source parts match GitHub blob hashes. Direct spanned-DEFLATE extraction passes full length and CRC checks; SGCC check rerun from complete data. Only verified rerun metrics appear in final results. |
| app/reporting | Initial local maintenance command used the wrong relative path; no project file was changed by that command. | Corrected working-directory path; cache references now remain inside the repository. |
| clean reproduction comparison | A strict comparison initially flagged ablation energy-weighted recall at sub-float32-precision differences after the earlier CSV-based correction. | Verified identical saved energy weights and rankings. The native pipeline sums float32 weights; earlier CSV reaggregation used float64. Differences are confined to the secondary energy-weighted recall and below one float32 epsilon; report the exception in reproduction_rounding.json. All headline, paired primary, stress, temporal and SGCC metrics match at the default strict tolerance. |

## Explicitly skipped or incomplete

- Real LCL/SimBench grid check: access failures; synthetic fallback used. CER: application-controlled and unavailable.
- SGSC, SMART-DS and additional IEEE feeders: optional extensions not attempted. ESMI statistics/weather downloads/PFC figures: not used. Pecan Street: excluded.
- Exact original attack-function verification: primary full-text blocked; comparison is incomplete and formulas are labelled approximations/extensions.
- Pole/segment localisation, voltage-slope pathway, tariff misuse and topology repair: not implemented. No related performance claim.
- Coupled neutral/transformer field validation, Indian field pilot, revenue recovery and ROI: not performed.
- GNN advantage and voltage benefit: not supported; learned auxiliary DT head poor and not used for delivered estimates.
- Production ingestion, authentication, encryption/key management, case/grievance integration and deployment: roadmap only. Local HTTP and AppTest checks passed; no screenshot-based visual check or public hosting.
- Actual slide/PDF production and self-recorded video: outside the prompt’s requested Markdown outline/script deliverables; not produced or submitted.
- Upstream dependency deprecation warnings are preserved in test logs. They do not cause test failures; no third-party package code was patched.
- Percentile cluster-bootstrap bias and repeated validation use remain limitations; exploratory intervals are not multiplicity-adjusted.

## WP-14 — known defects of the merged `synthetic_india_v2` dataset (adopted as-is)

The merged dataset is the final primary dataset per project decision. The
audit pack's defects are documented, mitigated by protocol, NOT repaired (the
generator was not supplied). Source: `D:\rodic\merged_audit\`,
`results/merged_eval.json`.

- Inspection timing inconsistency: 8 theft_found outcomes precede both stated
  onset and first positive NTL. Mitigation: quarantined from labels; never
  relabelled as no_theft. Remaining positives: 6 dataset-wide.
- Event-flag shortcut: all 26 tamper_assisted cases carry events; zero honest
  meters do; 4 event-bearing meters claim has_event_log=False. Mitigation:
  event_flags excluded from all predictors (zeroed by the adapter).
- Hooking onset contradiction: all 24 direct_hooking meters show NTL before
  stated start. Not reconciled; consequence visible in results — meter-level
  M0 cannot see hooking (median rank 880/1800) and DT-balance separation of
  hooking DTs is weak (AUC 0.542).
- Mapping-stress recovery path: feeder_id/pole_id match hidden truth, letting
  a model undo the 206 GIS conflicts. Mitigation: those fields are never
  predictors; the 206 meters are re-seated on the declared DT's topology
  (documented assumption).
- Nominal-coverage gaps: no negative kWh (no solar export), all stop/resume
  dates null, no honest meter-fault subtype in truth, no customer edges in
  topology (meters attached at poles). Consequence: honest-fault FPR and
  export behaviour cannot be measured on this dataset.
- Target-semantics mixing: tariff_misuse is energy NTL without a documented
  billing-to-energy model; 2 theft-ever meters have no positive NTL rows.
  Mitigation: inspection-supervised headline; truth only as declared oracle.
