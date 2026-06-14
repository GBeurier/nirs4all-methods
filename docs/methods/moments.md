# moments

Native moment substrate for exact row-additive linear screens.

## ABI

- `n4m_lowlevel_moments_compute(ctx, X, Y, out_result)`
- `n4m_lowlevel_moments_subset_compute(ctx, X, Y, row_indices, n_indices, out_result)`
- `n4m_lowlevel_moments_subtract(ctx, lhs, rhs, out_result)`

All outputs are returned through `n4m_method_result_t`.

## Outputs

Double matrices:

- `x_sum` `(1, p)`, `y_sum` `(1, q)`
- `xtx` `(p, p)`, `xty` `(p, q)`, `yty` `(q, q)` raw moments
- `x_mean` `(1, p)`, `y_mean` `(1, q)`
- `cxx` `(p, p)`, `cxy` `(p, q)`, `cyy` `(q, q)` centered moments

Scalars:

- `n_samples`
- `n_features`
- `n_targets`

## Exact Fold Subtraction

The raw buffers are additive over rows. A train fold can therefore be computed
as:

1. `all = n4m_lowlevel_moments_compute(X, Y)`
2. `heldout = n4m_lowlevel_moments_subset_compute(X, Y, heldout_indices)`
3. `train = n4m_lowlevel_moments_subtract(all, heldout)`

`n4m_lowlevel_moments_subtract` subtracts only raw sums/products and then recomputes
`x_mean`, `y_mean`, `cxx`, `cxy` and `cyy` from the remaining row count. This is
the important train-only centering invariant for PLS/Ridge screening.

## Python

The grouped facade is available as `n4m.moment`. It intentionally uses the
singular name so the historical `n4m.moments(...)` function stays callable.

```python
import n4m.moment as moment

stats = moment.moments(X, y)
screen = moment.sweep_run(X, y, heads=("ridge",))
ridge_model = moment.NativeRidgeRegressor(alpha=0.1).fit(X, y)
y_hat = ridge_model.predict(X_test)
inventory = moment.available_methods()
```

```python
import n4m

stats = n4m.moments(X, y)
subset = n4m.moments(X, y, row_indices=[0, 3, 5])
train = n4m.moments_train_from_heldout(X, y, heldout_indices=[1, 2, 8])
```

`moment.available_methods()` returns JSON-safe metadata for the moment
surfaces: sufficient statistics, `sweep_run`, direct Ridge/CPPLS/continuum/ECR
heads, reusable direct sklearn wrappers, the AOM/moment fast screen-refit
campaign preset and the CPU/CUDA backend recommendation helper. It is an
inventory for tooling and documentation, not a model-selection policy.

For broad strict-linear preprocessing screens from the moment namespace, use
`moment.aom_moment_screen_refit_campaign(...)`. It is the same function as
`n4m.aom_moment_screen_refit_campaign`: a preset wrapper around
`aom_chain_screen_refit_campaign` with strict moment routes, prefix chunking,
split-head Ridge/PLS scoring, PLS GCV-proxy first-pass ranking and exact-CV
refit.

The direct sklearn wrappers are:

- `NativeRidgeRegressor`
- `NativeCPPLSRegressor`
- `NativeContinuumRegressionRegressor`
- `NativeECRRegressor`

They replay the native MethodResult train predictions from input-space
coefficients plus an intercept and can predict on new `X` without re-running a
screen.

The preconfigured AOM/moment screen-refit estimators are also re-exported from
`n4m.moment`:

- `NativeAOMMomentScreenRefitRegressor`
- `NativeAOMMomentPLSScreenRefitRegressor`
- `NativeAOMMomentPLSExactScreenRefitRegressor`
- `NativeAOMMomentRidgeScreenRefitRegressor`

These are the reusable estimator counterparts to
`moment.aom_moment_screen_refit_campaign`: they screen strict-linear
preprocessing chains, exact-refit retained candidates, then build a final-only
winner model with folded input coefficients for prediction on new spectra.

Winner reuse helpers from the same workflow are available without leaving the
moment namespace:

- `moment.aom_screen_refit_candidate_pool(report)`
- `moment.aom_refit_execution_plan(report)`
- `moment.aom_refit_candidates(X, y, rows, ...)`
- `moment.aom_chain_fixed_fit_run(X, y, chain, ...)`
- `moment.NativeAOMFixedCandidateRegressor.from_refit_report(report)`

Audit and report helpers are exposed from the same namespace:

- `moment.build_aom_strict_chain_grid(...)`
- `moment.decode_aom_chains(report)`
- `moment.aom_candidate_table(report)`
- `moment.aom_evaluate_candidates(X_train, y_train, X_eval, y_eval, rows, ...)`
- `moment.aom_candidate_operator_summary(report_or_rows)`
- `moment.aom_candidate_route_summary(report_or_rows)`
- `moment.aom_candidate_rank_diagnostics(report_or_rows)`
- `moment.aom_candidate_report_records(report_or_rows)`
- `moment.aom_save_candidate_report(path, report_or_rows)`
- `moment.aom_load_candidate_report(path)`

`aom_evaluate_candidates` is the explicit holdout audit path: it refits decoded
candidate rows on the caller's train split and reports CV-vs-eval rank fields.
`aom_candidate_route_summary` reports whether retained rows were scored
through dense, banded or structured operator moments, or through the
materialized fallback, by head and by chain. When a campaign/refit report
contains aggregate counters, it also returns `reported_total` so the full
screen/refit coverage can be audited even when only top-k rows are retained.
The save/load helpers preserve decoded chains for offline comparison and later
winner replay. These helpers operate on explicit candidate/report rows and do
not add any dataset-name routing.

## Backend

The default build accumulates in F64 on CPU. In CUDA-enabled builds the Gram
products (`X'X`, `X'Y`, `Y'Y`) use the existing compile-time `linalg` dispatch
to cuBLAS. This is a reusable moment substrate, not yet the fused batched CUDA
operator-chain sweep.

For a route smoke of the Python facades on one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/cross_binding/aom_moment_cuda_facade_smoke.py
```

The smoke uses `n4m.moment` and `n4m.aom` through the CUDA build and asserts
that wide PLS1 screens report CUDA-device moment CV fits rather than host or
materialized PLS CV fits.
