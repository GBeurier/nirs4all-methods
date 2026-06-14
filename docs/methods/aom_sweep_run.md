# `aom_sweep_run` - configurable native AOM preprocessing sweep

_Group_: **Diagnostic / AOM** · _ABI_: `n4m_model_selection_aom_sweep_run`

## Description

`aom_sweep_run` applies the native strict-linear AOM preprocessing chain bank,
then delegates Ridge/PLS candidate scoring to `n4m_model_selection_sweep_run`.

It is the configurable product surface for preprocessing campaigns where the
user wants to vary:

- the AOM chain bank profile: `compact` or `wide`;
- the Ridge lambda grid;
- the PLS component grid;
- the active heads: Ridge, PLS, or both;
- explicit fold ids for reproducible CV.
- the AOM moment route policy: `auto`, `materialized`, or `force_moments`.

Native v1 intentionally keeps only shape-preserving strict-linear AOM
operators. Stateful fold-fitted preprocessings such as SNV, MSC, EMSC and
baseline families remain in the Python reference estimator.

## Backend Status

The method is exposed through the C ABI and Python wrapper and builds in both
CPU and CUDA-enabled `libn4m` configurations. Ridge requests use an exact
operator-moment fast path when `p <= n_train`: strict-linear chain operators
are applied to sufficient statistics, and held-out Ridge SSE is computed from
moments. Dense operator-moment transforms are capped at `p <= 48` for
medium-wide Ridge grids with strictly positive lambdas. Shape-preserving local
linear operators (`identity`, Savitzky-Golay smooth/derivative,
Norris-Williams, finite difference, Gaussian and FCK) also have a banded
operator-moment route that avoids building dense chain matrices; it is enabled
up to `p <= 256` for Ridge moment scoring. Chains containing
`detrend_poly` use a structured low-rank moment transform for the polynomial
projection, and Whittaker chains use a structured pentadiagonal solve for
`(I + lambda D2'D2)^-1`. Both structured routes can compose with those local
banded operators under the same wide Ridge guard. In Ridge-only sweeps the
selected chain is materialized once for public predictions. On CPU builds,
`auto` deliberately routes Ridge rows with `p > n_train` through the exact
materialized dual-Ridge scorer because it is cheaper than feature-space moment
Ridge in that geometry. CPU `auto` also routes compatible PLS1 rows through
the exact materialized prefix scorer when `min_train < 4p`. CUDA builds keep
the operator-moment route in those cells.

Single-target PLS1 requests with NIPALS regression deflation now also have an
operator-moment scoring path. Dense transforms use the same medium feature
guard (`p <= n_train` or `p <= 48`), while banded local operators and
structured `detrend_poly` chains are enabled up to `p <= 1024`. The PLS1
NIPALS component grid is fitted from train moments (`Cxx`, `Cxy`, `Y'Y`) and
held-out SSE is computed from held-out moments. Whittaker uses the same
structured pentadiagonal route for compatible PLS1 rows. The selected PLS
chain is then materialized once to expose public OOF/final predictions.
Multi-target PLS, non-NIPALS solvers, and larger unsupported regimes fall back
to the materialized native PLS path, which still reuses one max-component fit
per fold and reconstructs smaller coefficient prefixes.

For operator-moment routes, native sweeps also cache transformed strict-linear
prefix moments when the feature count is small enough for bounded memory use.
This is an exact compute cache for repeated prefixes in cartesian-style chain
grids; it does not change scores or ranking. The MethodResult exposes
`n_moment_prefix_cache_hits` and `n_moment_prefix_cache_misses` so campaigns
can audit whether a grid is actually sharing prefix work.

When a Ridge row takes the materialized route, the Ridge scorer reuses train
dual kernels across lambdas and, when a cost heuristic predicts a win,
held-out/train cross-kernels too. This avoids rebuilding feature-space
coefficients for every fold/lambda in the regimes where that is cheaper. This
is not yet the fused 200k-chain GPU grinder.

The wrapper exposes `moment_policy="auto"` by default. `auto` uses guarded
operator-moment routes when supported and falls back per regime. Use
`moment_policy="materialized"` or `"legacy"` to force the previous
materialized-chain screen. The scores remain the same up to numerical
roundoff; the policy is a compute-route switch for benchmarking and production
guarding.

Use `moment_policy="force_moments"` when the screen must stay strictly inside
the moment substrate. In that mode, any chain/head/regime that would need a
materialized candidate-screen fallback returns `UNSUPPORTED` instead of being
silently scored outside moments. Aliases accepted by Python include
`"moments_only"`, `"operator_moments_only"`, and `"strict_moments"`. This
strictness only applies to candidate scoring: after the winning candidate is
known, the selected chain is still materialized once to expose public OOF/final
predictions and `input_coefficients`.

Use `score_only=True` for large ranking campaigns when only the candidate
table, selected ids, route counters, folds and chain descriptors are needed.
In score-only mode, `predictions`, `oof_predictions`, `coefficients`,
`input_coefficients`, `intercept`, `x_mean`, `x_scale`, and `y_mean` are
returned as empty `0 x 0` matrices and the scalar `score_only` is `1`. This
currently skips the final selected-chain refit/materialization and OOF/model
output buffers in both operator-moment and materialized candidate-screen
routes. Materialized routes still pay their fold-local scoring fits because
they are not batched IKPLS.
PLS fit-cost counters are still populated: `n_pls_moment_cv_fits` and
`n_pls_materialized_cv_fits` count fold-local PLS fits in the screen, while
the corresponding `*_final_fits` counters remain zero in score-only mode.

For very broad PLS-only first-pass screens, set
`pls_score_mode="gcv_proxy"` together with `score_only=True`. This uses a
deterministic PLS1 GCV RMSE proxy from all-sample operator moments instead of
exact fold CV, exposes `n_pls_gcv_proxy_candidates`,
`n_pls_gcv_proxy_fits`, `aom_pls_score_mode=1`, and marks PLS rows with
`score_metric="pls_gcv_proxy_rmse"`. The proxy is moment-only: it fails rather
than materializing fallback chains. Use the default `pls_score_mode="cv"` to
verify/refit retained candidates with exact CV.

The MethodResult also exports `input_coefficients`, which folds the selected
strict-linear chain back into the original spectral feature space. The legacy
`coefficients` matrix remains the selected transformed-space coefficient
matrix. `input_coefficients` enables sklearn-style native estimators to predict
on new spectra without refitting or reapplying Python preprocessing objects.

## Python Usage

```python
import n4m

res = n4m.aom_sweep_run(
    X,
    y,
    profile="compact",
    cv=5,
    ridge_lambdas=[0.01, 0.1, 1.0],
    pls_components=[1, 2, 4],
    heads=("ridge", "pls"),
    scale_x=False,
    moment_policy="auto",
    pls_score_mode="cv",
    score_only=False,
)

print(res["selected_chain_id"], res["selected_head_id"], res["selected_param"])
print(res["candidate_scores"][:5])

chains = n4m.decode_aom_chains(res)
top = n4m.aom_candidate_table(res, sort=True)[:10]
print(top[0]["chain"], top[0]["head"], top[0]["param"], top[0]["cv_rmse"])
```

Sklearn-style native estimator:

```python
from n4m.sklearn import NativeAOMSweepRegressor

model = NativeAOMSweepRegressor(
    profile="compact",
    cv=5,
    ridge_lambdas=[0.01, 0.1, 1.0],
    pls_components=[1, 2, 4],
    heads=("ridge", "pls"),
    scale_x=False,
    moment_policy="auto",
).fit(X_train, y_train)

y_pred = model.predict(X_test)
print(model.get_diagnostics())
```

PLS-only screening does not require dummy Ridge lambdas:

```python
res = n4m.aom_sweep_run(
    X,
    y,
    fold_ids=fold_ids,
    ridge_lambdas=[],
    pls_components=[1, 2, 3],
    heads=("pls",),
)
```

## Outputs

Double matrices:

- `candidate_scores` `(n_candidates, 5)`: `candidate_id`, `chain_id`,
  `head_id`, `param`, `cv_rmse`
- `chain_params` `(1, n_chain_params)`: flat parameter payload for the
  exported chain descriptor
- `oof_predictions` `(n_samples, n_targets)` for the selected candidate
- `predictions` `(n_samples, n_targets)` from the selected final refit
- `coefficients` `(n_features, n_targets)` in the selected transformed space
- `input_coefficients` `(n_features, n_targets)` folded into the original
  input feature space for direct prediction as `X @ input_coefficients +
  intercept`
- `intercept` `(1, n_targets)`
- `x_mean`, `x_scale`, `y_mean`

Int vectors:

- `fold_ids`
- `candidate_routes` `(n_candidates)`: per-candidate scoring route code,
  `0=materialized`, `1=dense_operator_moment`, `2=banded_operator_moment`,
  `3=structured_operator_moment`
- `chain_offsets` `(n_chains + 1)`, `op_kinds` `(n_ops)`, and
  `param_offsets` `(n_ops + 1)`: together with `chain_params`, these reproduce
  the exact strict-linear chain bank used by `chain_id`. Use
  `n4m.decode_aom_chains(res)` or `n4m.aom_candidate_table(res, sort=True)`
  from Python for decoded campaign reports.

Scalars:

- `selected_candidate_id`
- `selected_chain_id`
- `selected_sweep_candidate_id`
- `selected_head_id`
- `selected_param`
- `selected_cv_rmse`
- `n_candidates`
- `n_operator_moment_candidates`
- `n_ridge_operator_moment_candidates`
- `n_pls_operator_moment_candidates`
- `n_banded_operator_moment_candidates`
- `n_structured_operator_moment_candidates`
- `n_dense_operator_moment_candidates`
- `n_materialized_candidates`
- `n_ridge_materialized_candidates`
- `n_pls_materialized_candidates`
- `n_moment_prefix_cache_hits`
- `n_moment_prefix_cache_misses`
- `n_pls_moment_cv_fits`
- `n_pls_materialized_cv_fits`
- `n_pls_moment_final_fits`
- `n_pls_materialized_final_fits`
- `score_only`
- `n_chains`
- `profile`
- `cv`
- `n_samples`
- `n_features`
- `n_targets`

`head_id` is `0` for Ridge and `1` for PLS. `param` is the Ridge lambda for
Ridge rows and `n_components` for PLS rows. The per-head route counters let
large campaigns audit whether Ridge or PLS rows used operator moments or the
materialized fallback. `candidate_routes` provides the same route provenance
per candidate row without changing the stable `candidate_scores` shape; Python
`n4m.aom_candidate_table` exposes it as `score_route_id` and `score_route`.

## Native Profiles

`compact` has 12 chains:

| ID | Chain |
|----|-------|
| 0 | `raw` |
| 1 | `detrend1` |
| 2 | `detrend2` |
| 3 | `savgol_w5_p2_d0` |
| 4 | `savgol_w7_p2_d0` |
| 5 | `savgol_w7_p2_d1` |
| 6 | `savgol_w11_p2_d2` |
| 7 | `nw_s5_g5_d1` |
| 8 | `finite_diff1` |
| 9 | `detrend1_savgol_w7_p2_d1` |
| 10 | `detrend1_nw_s5_g5_d1` |
| 11 | `savgol_w5_p2_d0_finite_diff1` |

`wide` has 31 chains and adds larger Savitzky-Golay windows, more
Norris-Williams variants, finite second difference, Gaussian/FCK variants,
Whittaker smoothing and additional strict-linear compositions.

## Benchmarks

Timing script:

```bash
PYTHONPATH=bindings/python/src \
N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so \
python3 benchmarks/cross_binding/bench_aom_sweep_timing.py
```

CUDA-build smoke:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=bindings/python/src \
N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so \
python3 benchmarks/cross_binding/bench_aom_sweep_timing.py \
  --output benchmarks/cross_binding/aom_sweep_timing_cuda_smoke.csv
```

Current ABI 1.20.0 smoke medians are stored in:

- `benchmarks/cross_binding/aom_sweep_timing.csv`
- `benchmarks/cross_binding/aom_sweep_timing_cuda_smoke.csv`

The CSVs include `moment_policy` plus per-head route counters:
`n_ridge_operator_moment_candidates`, `n_pls_operator_moment_candidates`,
`n_ridge_materialized_candidates`, and `n_pls_materialized_candidates`.
They also include prefix-cache counters and PLS fit-cost counters
(`n_pls_moment_cv_fits`, `n_pls_materialized_cv_fits`,
`n_pls_gcv_proxy_candidates`, `n_pls_gcv_proxy_fits`,
`n_pls_moment_final_fits`, `n_pls_materialized_final_fits`) so a timing row can
separate preprocessing-route choice from fold-local PLS fitting cost or the
explicit proxy screen.
Current CPU smoke rows show why those counters matter: mixed compact/custom
and PLS-only shapes route through exact materialized scorers in `auto` because
the CPU geometry guard prefers materialized scoring there, while Ridge-only
96 x 32 and 160 x 64 rows use operator moments. In the corresponding
CUDA-enabled smoke, `auto` keeps the mixed, PLS-only and Ridge-only rows on
operator-moment routes, with Ridge/PLS route counts matching the candidate
head split.

The exact medians are intentionally kept in the CSVs because they move with
build type, BLAS, GPU and route guards. The current smoke shows CPU compact
mixed `auto` at roughly 7.22 ms / 36.55 ms for 48 x 64 / 80 x 128, and
CUDA compact mixed `auto` at roughly 20.64 ms / 274.04 ms for the same shapes.
These CUDA timings validate the CUDA-enabled route accounting; they do not
claim a fused device-resident grinder yet.

The structured `detrend_poly` and Whittaker routes are exact and expose route
coverage for auditing. The policy switch exists because the current
implementation still transforms dense `p x p` moments on the host for some
paths, so `auto` is not uniformly faster until the fused GPU/batched engine
and a PLS-specific route selector exist.
