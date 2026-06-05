# sweep_run

Native moment-based preprocessing/model screen substrate.

ABI v1 supports exact Ridge cross-validation and PLS component screening. Folds
with `p <= n_train` use row-additive train moments for Ridge; folds with
`p > n_train` use a precomputed dual Ridge design and reuse `K = XX'` across
lambdas. When a simple cost heuristic predicts a win, wide dual folds also
reuse held-out/train cross-kernels and predict held-out rows directly in dual
space during screening. The wide dual train Gram, held-out cross-kernel,
dual prediction and final coefficient reconstruction use the internal
`linalg::gemm` dispatch, so CUDA builds route those matrix products through
cuBLAS while CPU builds keep the same row-major dispatch abstraction.
Compatible single-target NIPALS/regression PLS1 grids are now scored from
train/held-out moments. Other PLS regimes still fit the
maximum requested component count once per materialized train fold, then
reconstruct coefficient prefixes for smaller component candidates. Fused
batched IKPLS is still a later optimization.

## ABI

```c
n4m_sweep_run(
    ctx, cfg, X, Y,
    cv,
    fold_ids, n_fold_ids,
    ridge_lambdas, n_ridge_lambdas,
    pls_components, n_pls_components,
    heads_mask,
    out_result)
```

`heads_mask` bits:

- `1`: Ridge
- `2`: PLS

When `fold_ids` is `NULL` and `n_fold_ids == 0`, contiguous balanced folds are
generated from `cv`. When `fold_ids` is provided, its length must equal
`n_samples`.

## Outputs

Double matrices:

- `candidate_scores` `(n_candidates, 4)`: `candidate_id`, `head_id`, `param`,
  `cv_rmse`
- `oof_predictions` `(n_samples, n_targets)` for the selected candidate
- `predictions` `(n_samples, n_targets)` from the selected candidate refit on
  all rows
- `coefficients` `(n_features, n_targets)`
- `intercept` `(1, n_targets)`
- `x_mean`, `x_scale`, `y_mean`

Int vectors:

- `fold_ids`

Scalars:

- `selected_candidate_id`
- `selected_head_id`
- `selected_param`
- `selected_cv_rmse`
- `n_candidates`
- `n_pls_moment_candidates`
- `n_pls_moment_cv_fits`
- `n_pls_moment_host_cv_fits`
- `n_pls_moment_cuda_device_cv_fits`
- `n_pls_materialized_cv_fits`
- `n_pls_moment_final_fits`
- `n_pls_moment_host_final_fits`
- `n_pls_moment_cuda_device_final_fits`
- `n_pls_materialized_final_fits`
- `score_only`
- `cv`
- `n_samples`
- `n_features`
- `n_targets`

## Python

```python
import n4m

res = n4m.sweep_run(
    X,
    y,
    cv=5,
    ridge_lambdas=[0.01, 0.1, 1.0, 10.0],
    pls_components=[2, 4, 6],
    heads=("ridge", "pls"),
    scale_x=False,
)
```

With explicit folds:

```python
res = n4m.sweep_run(
    X,
    y,
    fold_ids=fold_ids,
    ridge_lambdas=[0.1, 1.0],
    scale_x=False,
)
```

For broad ranking passes, skip selected-model output buffers:

```python
scores = n4m.sweep_run(
    X,
    y,
    cv=5,
    pls_components=[1, 2, 4, 8],
    heads=("pls",),
    scale_x=False,
    score_only=True,
)
candidate_scores = scores["candidate_scores"]
```

With `score_only=True`, `candidate_scores`, selected ids, `fold_ids`,
fit-cost counters, `n_pls_moment_candidates`, and scalar diagnostics stay
populated, while `oof_predictions`, `predictions`, `coefficients`, and
`intercept` are returned as empty `0 x 0` matrices.
The PLS fit counters remain populated in score-only mode. In particular,
`n_pls_moment_cv_fits` and `n_pls_materialized_cv_fits` count fold-local CV
fits actually performed by the selected scoring route, while
`n_pls_moment_cuda_parallel_fold_batches` and
`n_pls_moment_cuda_parallel_fold_jobs` report optional bounded CUDA
stream-parallel scheduling for exact PLS1 moment jobs when requested, and
`n_pls_moment_final_fits` and `n_pls_materialized_final_fits` stay zero because
the selected final refit is skipped.
For materialized Ridge/PLS fallback cells where a linear coefficient prefix is
already available, score-only mode computes held-out SSE directly from the
fit instead of allocating held-out prediction buffers. Wide dual cross-kernel
Ridge cells also compute held-out SSE directly from `K_cross` in score-only
mode, so they avoid materializing held-out predictions too.

## Backend launch recommendation

For broad exact moment screens, use the measured CPU/CUDA crossover helper
before launching the process that imports `n4m`:

```python
plan = n4m.moment_screen_backend_recommendation(
    X.shape[0],
    X.shape[1],
    head="pls",
    cuda_available=True,
    cuda_pls_min_device_features=512,
    cuda_pls_many_batched=True,
)
print(plan["recommended_backend"])
print(plan["uses_cuda_pls_device_component_loop"])
print(plan["uses_cuda_pls_fold_workspace"])
print(plan["uses_cuda_pls_many_batched"])
```

The helper is source-free: it uses only `n_samples`, `n_features`, `head`, CUDA
availability, the launch crossover threshold, and the explicit CUDA PLS knobs.
It does not inspect dataset name, source metadata, labels, or spectra. Because
the Python binding loads one `libn4m` shared object per process, a CPU/CUDA
change must be done by starting the campaign with the corresponding build
selected up front. The default launch recommendation is conservative and keeps
CPU below `n_samples * n_features = 512 * 512`; pass `min_cuda_product` only for
controlled timing campaigns. For PLS screens,
`uses_cuda_pls_device_component_loop` reports whether the device-resident
component loop is expected to run, and
`uses_cuda_pls_fold_workspace` reports whether exact-CV folds can reuse one
CUDA workspace. Passing `cuda_pls_many_batched=True` also makes the helper
report whether the optional tiled/strided-batched CUDA route is expected to be
active for that PLS shape. The default PLS device-loop threshold is
`p >= 1024`; pass `cuda_pls_min_device_features=256` or another positive value
only when you are explicitly benchmarking medium-width GPU PLS screens.

Sklearn-style native estimator:

```python
from n4m.sklearn import NativeMomentSweepRegressor

model = NativeMomentSweepRegressor(
    cv=5,
    ridge_lambdas=[0.01, 0.1, 1.0, 10.0],
    pls_components=[2, 4, 6],
    heads=("ridge", "pls"),
    scale_x=False,
).fit(X_train, y_train)

y_pred = model.predict(X_test)
```

## Implementation Note

For moment-eligible folds, `n4m_sweep_run` computes train moments as
`all - heldout`, then fits Ridge from the train `CXX/CXY` moments and scores
held-out rows. For spectral shapes where `p > n_train`, it avoids the slow
`p x p` primal solve and instead precomputes centered/scaled train matrices and
the dual kernel once per fold. It also precomputes held-out/train cross-kernels,
when the estimated `O(h*n*p)` cross-kernel setup is cheaper than repeated
feature-space coefficient reconstruction and prediction. In that case each
Ridge lambda only solves the train dual system and predicts as
`K_heldout,train @ alpha + y_mean`. Otherwise it keeps the older dual-beta
scoring path. The C++ test suite compares both moment-eligible and wide dual
Ridge score paths against materialized fold-by-fold `n4m_ridge_fit`
references. The wide-dual matrix products use `linalg::gemm`: `K = X_train
@ X_train.T`, `K_cross = X_heldout @ X_train.T`, held-out predictions, and
`beta = X_train.T @ alpha`.

Compatible PLS1 candidates are scored without fold-local train matrix
materialization: the sweep computes held-out moments, subtracts them from the
all-row moments, fits NIPALS/regression PLS1 prefixes from the train
sufficient statistics, and scores held-out SSE from held-out moments. In CPU
and BLAS builds, the dense PLS1 moment products (`C @ w`, `P.T @ W`,
`W @ inv(P.T @ W)` and the rank-1 covariance deflation) use the shared
`linalg` dispatch. CUDA builds use a scalar host loop for medium-width PLS1
moment screens, because repeated cuBLAS micro-kernel transfers are slower
there. For very wide `p >= 1024` PLS1 moment screens, CUDA builds use an
internal device-resident cuBLAS component loop: `C` and `s` are copied once,
then the per-component `gemv/dot/ger/axpy` deflations stay on device before
`W/P` are copied back in one block for the existing prefix reconstruction.
Multi-fold exact-CV PLS1 screens reuse one CUDA workspace across fold-local
moment designs, avoiding repeated device allocation while preserving the same
per-fold scores. Passing `cuda_pls_parallel_folds=True` to `sweep_run` or
`NativeMomentSweepRegressor` enables bounded stream-parallel batches for those
independent exact PLS1 moment jobs on the selected single GPU; the historical
`N4M_CUDA_PLS_PARALLEL_FOLDS=1` environment variable remains a profiling
override. `N4M_CUDA_PLS_MANY_BATCHED=1` enables an experimental cuBLAS-only
tiled scheduler for the default many-job exact PLS1 moment path; it preserves
scores but remains opt-in because smoke timings did not beat the legacy
sequential-many workspace path. Passing `cuda_pls_many_batched=True` to
`sweep_run` or `NativeMomentSweepRegressor` enables the same route without an
environment variable. `N4M_CUDA_PLS_MANY_LEGACY=1` forces the legacy path and
`N4M_CUDA_PLS_BATCH_MAX_BYTES=<bytes>` caps experimental tile memory.
Passing `cuda_pls_min_device_features=<positive int>` lowers or raises the CUDA
device-route feature threshold from the default 1024 without recompiling;
scores are unchanged, but timings and host/device counters may change. This is
not fused batched IKPLS. The scalar
`n_pls_moment_candidates` reports how many PLS candidates used the moment
route. For
fit-cost auditing, `n_pls_moment_cv_fits` counts one moment-prefix fit per CV
fold and `n_pls_moment_final_fits` counts the selected all-row refit when
model outputs are requested. The host/device split counters
`n_pls_moment_host_cv_fits`, `n_pls_moment_cuda_device_cv_fits`,
`n_pls_moment_cuda_parallel_fold_batches`,
`n_pls_moment_cuda_parallel_fold_jobs`,
`n_pls_moment_host_final_fits`, and
`n_pls_moment_cuda_device_final_fits` report which execution route actually
ran. For materialized fallback routes,
`n_pls_materialized_cv_fits` counts the fold-local max-component or fallback
per-component PLS fits and `n_pls_materialized_final_fits` counts the selected
all-row materialized refit. These counters expose the remaining PLS screen work
that a later batched IKPLS/fused CUDA implementation is expected to remove. For
multi-target or unsupported PLS solver/deflation regimes, the sweep keeps the
materialized prefix path: it fits the existing native PLS model at
`max(pls_components)` on each train fold, reconstructs prefix coefficients from
`W[:,:k]`, `P[:,:k]` and `Q[:,:k]`, predicts the held-out fold and ranks by CV
RMSE. If that max-component fit fails on a fold, the implementation falls back
to separate per-component materialized fits for that fold. The C++ tests compare
single-component and multi-component score tables against explicit fold-by-fold
`n4m_model_fit` and `n4m_model_predict`.

The timing smoke is:

```bash
python3 benchmarks/cross_binding/bench_moment_sweep_timing.py
```

Current ABI 1.20.0 smoke output is stored in:

- `benchmarks/cross_binding/moment_sweep_timing.csv`
- `benchmarks/cross_binding/moment_sweep_timing_cuda_smoke.csv`

The CSVs include `n_pls_moment_cv_fits`,
`n_pls_moment_cuda_parallel_fold_batches`,
`n_pls_moment_cuda_parallel_fold_jobs`, `n_pls_materialized_cv_fits`,
`n_pls_moment_final_fits`, and `n_pls_materialized_final_fits` so timing rows
can be tied to the exact PLS fit work paid by the route. On CUDA builds,
rerun with `--cuda-pls-parallel-folds` to compare bounded stream-parallel
exact PLS1 moment scheduling against the default reusable workspace path. Use
`--cuda-pls-min-device-features 256` or another positive threshold to test
whether medium-width PLS moment screens should enter the CUDA device route.

CPU medians for Ridge sweep were 1.64 ms at 64 x 64, 11.28 ms at 128 x 128,
and 45.70 ms at 192 x 256. The corresponding `score_only=True` rows were
1.42 ms, 10.31 ms, and 38.07 ms. The materialized CV Ridge baseline took
3.41 ms, 18.41 ms, and 57.32 ms. PLS component-grid smoke medians using the
moment route were 0.43 ms, 1.96 ms, and 9.83 ms; `score_only=True` reduced
those to 0.29 ms, 1.91 ms, and 9.93 ms, with `n_pls_moment_candidates=3` and
`n_pls_moment_cv_fits=5` in all three PLS rows.

The CUDA-build native smoke produced the same selected parameters and scores.
Ridge medians were 5.26 ms, 14.34 ms, and 35.88 ms; Ridge score-only medians
were 4.91 ms, 12.82 ms, and 31.32 ms. PLS medians were 2.04 ms, 3.03 ms, and
8.10 ms; PLS score-only medians were 2.17 ms, 2.94 ms, and 7.62 ms. These
are smoke timings for the CUDA-enabled library rather than a fused GPU screen;
the larger wide-dual Ridge row benefits from the GEMM/cuBLAS route, while PLS1
moment stays host-side to avoid many tiny host/device transfers.

This is not yet the full 200k-chain fused CUDA grinder. It is the exact
screening ABI that the batched IKPLS and fused operator-moment layers can build
on.
