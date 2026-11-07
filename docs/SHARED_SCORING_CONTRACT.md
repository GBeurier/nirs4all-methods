# Shared scoring kernel boundary

The DAG-owned [numerical fixture](../parity/fixtures/shared_scoring_numerical.v1.json)
and [owner contract](../../dag-ml/docs/contracts/shared_scoring_numerical.v1.md)
freeze a bounded phase-2 agreement before migration. DAG-ML owns metric meaning,
context, stable-identity joins, per-output observed-cell filtering, reductions
and selection. Methods provides kernels. No alternate implementation or full
Python reference artifact is replaced.

`n4m_metrics_regression_metrics_*` wraps `nirs_metrics.c`. These eight public
scalar APIs consume finite, unweighted one-dimensional arrays. NaN and +/-Inf
inputs return `N4M_ERR_INVALID_ARGUMENT` without changing the caller's output.
Finite inputs can still yield intentional +Inf for degenerate ratios (RPD,
RPIQ, NRMSE). This is an input validation change, not a finite-output policy.

RMSE, MAE and R² have per-output analytical expectations in the shared fixture.
Scalar R² for constant targets is 1 for perfect prediction, 0 otherwise,
including one row. `n4m_stats_regression_scores` is a separate context ABI with
four outputs (RMSE, MAE, R² significand, R² exponent); it intentionally returns
NaN R² for constant targets. Its divergence is preserved and tested. Internal
`compute_regression_metrics` is not this scalar ABI. The optimizer metric enum
has no standalone evaluation ABI and is out of scope.

The native test counts five direct scalar cases separately from two adapter
cases (four output checks). Both direct ABIs are `not_applicable` for any
multi-output or masked case. The separately declared
`methods_scalar_per_output_adapter` and `methods_stats_per_output_adapter`
extract columns (`extract_per_output`) or filter observed cells
(`coordinator_filter_observed_per_output`) before calling kernels. These checks
do not count as full-case ABI parity passes.
Finite hidden sentinels never become NaN transport. The test demonstrates mean
per-output RMSE differs from flattened RMSE. Methods does not implement the
coordinator's identity joins or contextual reduction in these scalar APIs.

Classification accuracy, balanced accuracy, support-weighted F1, label remapping
and probability-column order belong to the DAG contracts. Sample-weighted
regression/classification is refused by DAG built-ins; F1's support weighting
does not imply sample-weight support. Methods scalar classification, weights,
masks and identity joins are explicitly `not_applicable`, never parity passes.
Invalid numeric inputs are labeled `rejects` and exercised for all eight scalar
metrics, both arrays, every fixture position, with unchanged sentinel outputs.

Finite comparisons use the fixture's absolute-plus-relative tolerance:
`1e-12 + 1e-12 * abs(expected)`. These small cases do not claim universal parity,
full numerical-domain coverage or agreement between every alternate metric API.

The [receipt](../parity/fixtures/shared_scoring_numerical.v1.provenance.json)
pins the upstream URL, commit at fixture creation, source path and exact-byte
SHA-256 over exact file bytes, distinct from TCV1 semantic canonicalization. The source fixture is new, uncommitted work above that creation
base commit. Standalone native tests use the pinned copy without a DAG checkout.
Run `python3 parity/check_shared_scoring_fixture.py --dag-ml-repo ../dag-ml`
to verify all four exact copies: canonical source, Methods, R package-local and
crate-local test fixtures. Without an available source the checker explicitly
reports `UNAVAILABLE` and exits 2; SHA mismatch/drift exits 1. It never fetches.

The persisted R regression is
`bindings/r/n4m/tests/testthat/test-shared-scoring.R`. Run it with `R_LIBS`
pointing to a package linked against the changed native build; set
`N4M_SHARED_SCORING_FIXTURE` only for an explicit qualification override. The
package ships the exact shared fixture and its provenance receipt in
`bindings/r/n4m/inst/extdata`; the test defaults to
`system.file("extdata", "shared_scoring_numerical.v1.json", package = "n4m")`
and fails if the fixture is missing. The drift checker verifies this copy's
SHA-256 and provenance locally before the optional sibling comparisons.
Run the installed default with:

```sh
env -u N4M_SHARED_SCORING_FIXTURE R_LIBS=/path/to/installed/library Rscript -e \
  'testthat::test_file("bindings/r/n4m/tests/testthat/test-shared-scoring.R", stop_on_failure=TRUE, reporter="summary")'
```

It uses `n4m_run(n4m_regression_metrics(), X=predicted,
y=true)` and checks all six nonfinite input cases and the five finite scalar
cases, including R².

The raw WASM ABI regression is `bindings/js/test/run_shared_scoring.mjs`.
From `bindings/js`, run `npm run test:shared-scoring` with `N4M_WASM_MODULE`
pointing to the rebuilt Emscripten `n4m.js` and `N4M_WASM_BINARY` to its matching
`n4m.wasm`. Without these variables the test uses `dist`; a stale dist is not
qualified by a changed-build pass. The test checks all eight scalar metrics
against NaN and both infinities in each input, unchanged error-output sentinels,
finite zero RMSE and intentional RPD +Inf. The full `npm test` chain invokes
`test:shared-scoring`; CI builds and stages the current WASM into `dist` before
running that chain.

## Bounded PLS tuning qualification

`bindings/python/tests/test_tuning_scoring_contract.py` exercises the real native
`finetune_estimator(PLS_REGRESSION, ...)` convenience driver for RMSE, MSE, MAE
and R². This is separate from the scalar ABIs above. The ask/tell metric enum
sets objective direction; it does not evaluate predictions. All four
classification objectives (ACCURACY, BALANCED_ACCURACY, F1, LOGLOSS) return
`NOT_IMPLEMENTED` from this regression driver; those refusal probes are not
numerical evaluation passes.

The driver scores the complete OOF n×q matrix, with N=nq, SSE=Σ(ŷ−y)² and one
grand mean μ=Σy/N: RMSE=√(SSE/N), MSE=RMSE², MAE=Σ|ŷ−y|/N and
R²=1−SSE/Σ(y−μ)². R² is neither a macro average of target scores nor a ratio
using separately centered columns. When TSS ≤ DBL_EPSILON (the absolute C++
double machine epsilon, approximately 2.22e-16), R² is 1 if SSE ≤ DBL_EPSILON,
otherwise 0. The threshold is not 1e-12. Solver-admitted near-constant cases
exercise both branches and a TSS between DBL_EPSILON and 1e-12. Constant-zero
targets complete in finetuning with R²=1, but the public PLS CV producer refuses
that cohort with `NUMERICAL_FAILURE`; this limitation supplies no constant OOF
parity pass and no solver guard is relaxed.

The test feeds identical fold IDs and the exact fold count to both APIs, and
passes all four center/scale flags explicitly as true to `pls_cross_validate`,
matching the fixed finetune defaults (NIPALS/REGRESSION). Each producer call has
one component candidate; candidate mapping, native OOF shapes and actual route
counters are asserted before independent test arithmetic over those predictions.
Single-target production uses the moment route; finetuning uses
`cross_validate_regression`/`fit_model`. Their agreement uses an absolute-plus-
relative tolerance of `1e-12 + 1e-10 * abs(expected)` for rounding across native
routes and NumPy reductions. Multi-target production uses the materialized route
and shares the fit engine: it qualifies fold plumbing, OOF placement, aggregation
and selection, not independent numerical fitting.

The 5/11/21-row folds and unequal-scale, offset multi-target cohort distinguish
pooled objectives from unweighted means of fold scores despite `eval_mode=MEAN`.
Fixed-seed RANDOM sampling with 16 trials must actually cover all three component
values. Every completed trial and the AUTO winner are checked: strict minimum
for RMSE/MSE/MAE, strict maximum for R²; the first completed best candidate is
retained on exact ties. Near-constant cases include exact ties across different
components. The result exposes the winning parameters and score, not its trial ID.
This is selection only, with no final refit. The other five eligible estimator
routes, R/WASM/MATLAB tuning execution, masked/weighted contexts and a universal
PLS-fit or tuning certification remain outside this qualification.
