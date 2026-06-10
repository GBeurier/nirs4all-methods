# Full-57 staged campaign — benchmark + recall characterization (2026-06-08)

Accuracy-track Phase 1 (benchmark) + Phase 2 (recall characterization). Engine:
`nirs4all-methods` staged AOM/moment campaign (branch `release-readiness-fixes`).
Cohort: `all57_cohort.csv`, 51 OK datasets (2 LUCAS giants skipped by
`--max-train-feature-product 4000000`; QUARTZ missing data). Plan `compact`,
scale-x grid {false,true}, CUDA PLS route. Runtime ~72 min on GPU0 (RTX 4090).

Artifacts:
- results CSV: `aom_staged_real_cohort_full57_20260608.csv`
- diagnostics: `aom_staged_full57_20260608/` (51 `*.diagnostics.json`)
- oracle comparison: `aom_staged_full57_20260608_oracle_summary.md`
- rank audit: `aom_staged_full57_20260608_rank_audit.{csv,md}`

## 1. External competitiveness (vs AOM-PLS / AOM-Ridge / TabPFN oracles)

| Oracle | paired | target wins | median ratio (target/oracle) |
|---|---:|---:|---:|
| AOM-PLS oracle | 49 | 15 | **1.037** |
| AOM-Ridge oracle | 51 | 5 | **1.097** |
| TabPFN oracle | 51 | 15 | **1.109** |

Winner counts across 51: TabPFN 30, AOM-Ridge 19, AOM-PLS 11, **target (n4m
staged) 1** (DarkResp/Rd25_GTtestSite). **Honest verdict: competitive but not
dominant** — trails all three oracles on the median by ~4–11%. This reproduces
the handoff's compact10 conclusion at full-cohort scale.

ECOSIS nuance: Ccar actually *beats* the AOM-PLS oracle (ratio 0.318), while
Chla+b blows up 2–3× — that is the **−999 target-sentinel data pathology**
(neither engine normalizes it), not a pure selection failure.

## 2. Internal recall (selected vs best-candidate-in-pool, scored on test)

This is the *screen-vs-holdout selection* gap that a recall fix targets — distinct
from the external oracle gap above. Production selection stays train-CV only
(`selection_uses_test_set=False` for all rows); the test ranking is audit-only.

- **top-1 recall = 0.216** (the train-CV winner is the test-best on 11/51).
- **top-5 recall = 0.784** (the test-best candidate IS in the retained top-5 on
  78% of datasets). → **It is a SELECTION problem, not a candidate-budget
  problem.** More cartesian budget will not close it.
- median `oracle_gap_ratio` = **0.031**, mean = **0.105** (heavy right tail).
- **13/51 datasets anti-aligned** (negative CV/test Spearman).
- **15/51 datasets head-disagreement** (train-CV selected head ≠ test-oracle head).

### By head pair (mismatch magnitude)

| selected → oracle | count | median gap ratio |
|---|---:|---:|
| ridge → ridge | 25 | **0.026** (selection within ridge is fine) |
| ridge → pls | 11 | **0.121** |
| pls → ridge | 4 | **0.184** |

### The recall-failure cluster (negative Spearman, sorted by gap)

Almost all are **`sel=ridge, oracle=pls`** with a PLS-derivative chain
(`savgol_derivative(11,2,2)`, `detrend_poly`) as the missed test-best:

| dataset | gap | Spearman | selected | oracle |
|---|---:|---:|---|---|
| BISCUIT/Sucrose_40 | +0.599 | −0.452 | pls | ridge |
| ECOSIS/Chla+b_block2deg | +0.577 | −0.738 | ridge | pls (−999 pathology) |
| PHOSPHORUS/V25 | +0.440 | −0.881 | ridge | pls |
| GRAPEVINE/An_ASD | +0.338 | −0.829 | ridge | pls |
| TIC_spxy70 | +0.276 | −0.543 | ridge | pls |
| PLUMS/Firmness | +0.245 | −0.086 | pls | ridge |
| PHOSPHORUS/LP | +0.229 | −0.357 | ridge | pls |
| COLZA/C_woOutlier | +0.158 | −0.119 | ridge | ridge |
| PHOSPHORUS/MP | +0.104 | −0.690 | ridge | pls |
| BEER/YbaseSplit | +0.102 | −0.393 | ridge | pls |
| BEER/KS | +0.064 | −0.536 | ridge | pls |
| GRAPEVINE/An_NeoSpectra | +0.045 | −0.548 | ridge | pls |
| FUSARIUM/Fv_Fm | +0.007 | −0.314 | ridge | ridge |

## 3. Diagnosis

The recall gap is **structural, not budget-bound**:
1. The test-best candidate is usually retained (top-5 recall 0.784).
2. Train-CV **over-prefers ridge** (low CV variance) where **PLS-with-derivatives**
   generalizes better on holdout — 10/13 anti-aligned datasets are exactly this.
3. Failures concentrate at **small n** (Biscuit 40, An 80, MP 169) where the CV
   estimate is noisy and anti-aligned with holdout (winner's curse).

This motivated two Phase-3 recall fixes, both run on audit20 (same bank/params) —
see `nirs4all-lab/moment_sweep_proto/RECALL_FIX_RESULTS.md`:

- **screen-blind nested-CV** (`recall_fix_eval.py`): improved **6/19**, median
  ratio **1.007** — near-neutral median but **high variance** (catastrophic tails:
  C_woOutlier 1.38→2.58, Rd25_GT 1.88×; tiny-n wins: Firmness 0.885, TIC 0.980).
- **head-hedge ensemble** (ridge+pls, `recall_fix_ensemble.py`): improved **7/19**,
  median **1.018** — milder degradation, one big win (Biscuit 0.709).

**Decisive: no fix improves competitiveness vs any reference** — vs AOM-PLS the
standard beats 13/19 (fixes 10–11), vs AOM-Ridge 6/9 (fixes 3–5), vs TabPFN 4/19
(fixes 2–4). **Conclusion: the recall gap is largely irreducible selection
variance**, not a recoverable signal; the test-oracle overstates the achievable
gain. The lever for the oracle gap is **model expressiveness** (e.g. `moment_stack`),
not preprocessing reselection. Port recommendation: **do NOT** add global nested-CV
or head-hedge to the staged campaign's `python.py:2645` selection; keep train-CV
(1-SE). Full analysis + the one narrow win (conditional tiny-n + close-heads hedge)
in `RECALL_FIX_RESULTS.md`.
