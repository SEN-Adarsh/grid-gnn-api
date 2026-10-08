---
title: Grid-GNN Inspection Decision Support
emoji: ⚡
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: true
license: other
short_description: Label-free NTL triage demo - synthetic telemetry, decision support only
---

# Grid-GNN | Inspection Decision Support (API)

FastAPI service exposing the Grid-GNN PoC scoring engine. Interactive Swagger docs at `/docs`.

**Provenance and limits (read before use):**
- All telemetry is **synthetic** (`profile_source: synthetic_fallback`) replayed from packaged demo assets on a reference CIGRE-derived topology. There is **no Indian field dataset** in this system.
- Probabilities are **not validated for field use**. Output is inspection decision support only: a flag is a request to review evidence; a field inspector must verify every flag. No disconnection or penalty is automated.
- The `/score` audit log is written to the container filesystem and **resets on restart** (ephemeral hosting).
- Headline benchmark results and their caveats: PR-AUC of the strongest model (M2, own-history features) 0.644 [0.584, 0.707] on a held-out DT-disjoint test; the graph model (M4) did **not** show a benefit over graph-free baselines (paired difference includes zero). See the project deck for the full evidence chain.

## Endpoints

- `GET /health` - service status, selected model, config hash
- `POST /score` - score the demo network at a chosen interval prefix, model and scenario
- `GET /docs` - interactive OpenAPI documentation

## Example

```bash
curl -s https://<your-space-subdomain>.hf.space/health

curl -s -X POST https://<your-space-subdomain>.hf.space/score \
  -H 'Content-Type: application/json' \
  -d '{"model":"M2","scenario":"theft","severity":0.7,"tamper_event":true}'
```

Scenario names: `none`, `theft`, `vacancy`, `upstream_hooking`, `ami_dropout`.
Models: `M0` rule, `M1` IsolationForest, `M2` RandomForest (own history), `M3` RandomForest (+DT context), `M4` TemporalGraphNet.

Model artifact SHA-256 and config hash are returned with every response; provenance notes are attached to every payload.
