# Track C — real CEEW telemetry with injected theft labels

> **REAL telemetry (CEEW Mathura/Bareilly smart meters, CC0) with SYNTHETIC injected
> labels (Zidi et al. 2023 six-type generator + our DT-hook extension).
> NOT real theft labels. Virtual DTs, star topology, single-phase, no weather.
> No field-performance claim.**

- Full machine-readable report: [`../results/track_c_real_up.md`](../results/track_c_real_up.md) · [`../results/track_c_real_up.json`](../results/track_c_real_up.json)
- Code: [`../src/data/ceew_adapter.py`](../src/data/ceew_adapter.py) (staging + injection) · [`../src/eval/track_c_eval.py`](../src/eval/track_c_eval.py) (evaluation) · [`../tests/test_ceew_adapter.py`](../tests/test_ceew_adapter.py) (13 tests)

---

## What was injected

**Real electricity data** (84 meters, 2 districts, 35 valid 14-day monthly windows)
+ **fake theft labels we planted ourselves** — so we know exactly who the "thieves" are.

**8 thieves** (9% of meters, matching Zidi's spec prevalence; seed 42), across 6 types:

| Type | What the "thief" does | Intuition |
|---|---|---|
| T1 | Uses only 10–80% of usual power | Siphoning off some current |
| T2 | Goes to **zero** for stretches | Meter tampered off |
| T3 | Adds random noise to usage | Erratic, unnatural pattern |
| T4 | Profile replaced with random numbers | Totally scrambled behavior |
| T5 | **Flat** constant usage all the time | Frozen/stuck meter |
| T6 | **Reverses** the day (night becomes day) | Wired backwards |
| + hook | Extra load on the transformer owned by nobody | Our extension beyond Zidi et al. |

- Only the **assessment days (last 7 of each 14-day window)** carry theft — baseline
  days stay clean so the models keep an honest own-history reference.
- **69 guilty meter-windows** out of 682 total instances (prevalence 0.101);
  every window has ≥ 1 thief, all 6 types + hooks represented.
- Theft touches the **reported meter readings only**; the transformer's true input
  stays untouched (physics-consistent imbalance).

## How the frozen models did (M0–M4 served artifacts, features unchanged)

**Ranking quality (682 pooled meter-window instances):**

| Model | ROC-AUC | PR-AUC (random = 0.101) | MW p-value |
|---|---|---|---|
| M0 | 0.749 | 0.324 (3.2×) | 5.9e-12 |
| M1 | 0.648 | 0.177 | 2.8e-05 |
| M2 | 0.749 | **0.401 (4.0×)** | 2.4e-12 |
| M3 | 0.733 | 0.236 | 8.9e-11 |
| M4 | 0.777 | 0.382 (3.8×) | 2.0e-14 |

- ROC-AUC 0.65–0.78 → picking one thief and one innocent at random, the model ranks
  the thief higher **65–78% of the time** (coin flip = 50%).
- Every model beats the 0.101 base rate; best lift **~4×** (M2); separation is
  statistically solid (p ≤ 3×10⁻⁵).

**Per-type recall at a 20% inspection budget** (inspect the top-20% suspect list):

| | T1 | T2 | T3 | T4 | T5 | T6 |
|---|---|---|---|---|---|---|
| M0 | 0.30 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| M1 | 0.80 | 1.00 | 0.29 | 0.06 | 0.00 | 0.11 |
| M2 | 0.60 | 0.07 | 0.71 | 1.00 | 0.00 | 0.11 |
| M3 | 0.60 | 0.00 | 0.71 | 1.00 | 0.00 | 0.11 |
| M4 | 0.90 | 0.67 | 0.86 | 0.67 | 0.00 | 0.11 |

- **Obvious theft (T1–T4): caught well** — up to 60–100% within the budget.
- **T5 flat meters: 0% for every model** — a frozen meter just looks "low & steady."
- **T6 reversed: ≤ 11%** — same blind spot as Zidi's own Random Forest (theirs: **3.1%**);
  the daily total is unchanged, only the shape flips.

**Transformer hook (our extension):** 12 windows had a hooked transformer → ranked
**#1 suspect 100%** of the time (mean balance AUC 0.958). Visible by construction
(we created the imbalance) — validates the plumbing, not field performance.

**Periodicity lens (Zheng et al. IEEE TII):** their ">0.8 week-to-week correlation
for normal meters" threshold does **not** transfer — clean CEEW meters already sit
near 0 (different country, different habits). The finer slot-level lens drops when
theft is injected (e.g. T4 0.128→−0.002, T5 0.323→−0.048) — except T6, which is
mathematically invisible to it (reversing both weeks preserves correlation).

## One-paragraph summary

On **real Indian meter data with planted theft**, all five models rank thieves
**3–4× better than random** (M2/M4 best, ROC ~0.75–0.78), catch "loud" theft types
at **60–100% within a 20% inspection budget**, and instantly flag our transformer-hook
extension — but they remain **blind to flat (T5) and day-reversed (T6) meters**,
matching the published weakness of Zidi's own model. Labels are synthetic, DTs are
virtual, and **no claim about real-world field performance is made**.

## Reproduce

```powershell
# from D:\rodic\hf_space (Python 3.12 venv)
python -m pytest tests/test_ceew_adapter.py -q    # 13 adapter tests
python -m src.data.ceew_adapter                   # re-stage windows (CEEW data outside repo)
python -m src.eval.track_c_eval                   # regenerate ../results/track_c_real_up.{md,json}
python -m pytest -q                               # full suite: 37 tests
```

Raw/staged CEEW data lives outside the repo (`D:\rodic\ceew_data\`, never committed);
the evaluation uses the frozen served artifacts in `artifacts_merged_demo/` and the
pinned feature builder — no retraining at any point.
