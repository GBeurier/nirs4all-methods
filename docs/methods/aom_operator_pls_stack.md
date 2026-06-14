# `aom_operator_pls_stack` - native AOM operator PLS score stack

_Group_: **Diagnostic / AOM** · _ABI_: `n4m_ensemble_aom_operator_pls_stack_fit`

`aom_operator_pls_stack` computes compact/wide strict-linear AOM operator
views, compresses each view through a fold-local PLS1 score projector,
concatenates those scores, and fits a Ridge head on the stacked score matrix.
The selected final stack is linear in the original input spectra, so the
native result also exposes folded coefficients for direct reuse.

The sklearn-style `AOMOperatorPLSStack` reference estimator remains available
as `n4m.AOMOperatorPLSStack` and `n4m.sklearn.AOMOperatorPLSStack` for custom
operator matrices, shuffled/both CV modes, and optional baseline admission
gating.

## Status

- API surface: C ABI, Python function `n4m.aom_operator_pls_stack`, and native
  sklearn wrapper `NativeAOMOperatorPLSStackRegressor`.
- Native ABI: yes, since ABI `1.18.0`.
- Catalog status: `aom_pop.operator_pls_stack`.
- CPU: tested (`n4m_tests` and full Python bindings).
- CUDA: builds and smoke-tests with the CUDA-enabled library; native v1 is not
  a fused batched GPU stack.
- Target shape: single-target `Y` only, matching the PLS1 reference design.
- Dataset/source routing: forbidden. `metadata` passed to `fit` is audit-only.
- Native/default operator bank: fixed strict-linear matrices only. Compact has
  12 operators; wide has 31 operators, including Gaussian, FCK and Whittaker
  variants. Stateful scatter operators such as SNV, MSC, EMSC, OSC, or EPO are
  not included by default.

## Python Function

```python
n4m.aom_operator_pls_stack(
    X,
    y,
    profile="compact",
    cv=5,
    fold_ids=None,
    components=(2, 4, 8),
    alphas=(1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
    std_penalty=0.0,
    gap_penalty=0.0,
    scale_x=False,
)
```

The selection criterion is:

```text
mean_oof_rmse
  + std_penalty * std_oof_rmse
  + gap_penalty * max(0, mean_oof_rmse - mean_train_rmse)
```

If `fold_ids` is omitted, contiguous balanced folds are generated from `cv`.
For reproducible campaigns, pass explicit train-only fold ids.

## Native Example

```python
import n4m

res = n4m.aom_operator_pls_stack(
    X_train,
    y_train,
    profile="compact",
    cv=5,
    fold_ids=fold_ids,
    components=[1, 2, 4],
    alphas=[1e-3, 1e-1, 10.0],
    std_penalty=0.05,
    gap_penalty=0.25,
    scale_x=False,
)

print(res["selected_components"], res["selected_alpha"])
print(res["candidate_scores"])

y_hat = X_train @ res["input_coefficients"] + res["input_intercept"]
```

## Native Outputs

Double matrices:

- `candidate_scores` `(n_specs, 7)`: `spec_id`, `n_components`, `alpha`,
  `mean_oof_rmse`, `std_oof_rmse`, `mean_train_rmse`, `criterion`
- `fold_scores` `(n_specs, cv)`
- `oof_predictions` `(n_samples, 1)`
- `predictions` `(n_samples, 1)`
- `stack_features` `(n_samples, n_operator_features)`
- `coefficients` `(n_operator_features, 1)`: final Ridge head on
  `stack_features`
- `intercept` `(1, 1)`: final Ridge head intercept on `stack_features`
- `input_coefficients` `(n_features, 1)`: selected stack folded into the
  original input feature space
- `input_intercept` `(1, 1)`: folded input-space intercept

Int vectors:

- `fold_ids`
- `operator_feature_offsets`

Scalars include `selected_spec_id`, `selected_components`, `selected_alpha`,
`selected_oof_rmse`, `selected_train_rmse`, `selected_criterion`,
`n_operator_features`, `n_specs`, `n_operators`, `profile`, `cv`, `n_samples`,
`n_features` and `n_targets`.

The direct replay contract is:

```python
pred = X_new @ res["input_coefficients"] + res["input_intercept"]
```

## Python Estimator

`NativeAOMOperatorPLSStackRegressor` is the sklearn-style wrapper over the
native compact/wide ABI. It stores the selected stack diagnostics and predicts
from `input_coefficients`/`input_intercept` on new spectra.

`AOMOperatorPLSStack` remains the flexible Python/sklearn reference layer for
custom operator matrices and baseline gates.

## Constructor

```python
AOMOperatorPLSStack(
    operator_bank=None,
    components=(2, 4, 8),
    alphas=(1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
    cv=5,
    cv_mode="shuffled",
    std_penalty=0.0,
    gap_penalty=0.0,
    baseline_estimator=None,
    min_relative_oof_gain=0.0,
    random_state=2026,
    drop_failed_specs=True,
)
```

`operator_bank` may be a mapping from name to a fixed matrix, a transformer, or
a callable taking `n_features` and returning a matrix. Matrices may be shaped
`(n_features, n_outputs)` or `(n_outputs, n_features)`.

When `operator_bank=None`, the estimator builds a fixed strict-linear bank from
raw, detrend, finite differences, Gaussian smoothing, Savitzky-Golay variants,
and Norris-Williams variants that are valid for the current feature count.

If `baseline_estimator` is provided, the operator stack is admitted only when
its train-only OOF RMSE improves the baseline by at least
`min_relative_oof_gain`.

## Example

```python
import n4m

model = n4m.AOMOperatorPLSStack(
    components=(1, 2, 4, 8),
    alphas=(1e-4, 1e-2, 1.0, 100.0),
    cv=5,
    cv_mode="both",
    std_penalty=0.05,
    gap_penalty=0.25,
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(model.stack_report_)
```

## Outputs

After `fit`, the Python estimator exposes:

- `selected_spec_`: `AOMOperatorPLSSpec(n_components, alpha)`.
- `accepted_operator_stack_`: `False` when an OOF baseline gate rejects it.
- `operator_names_`: fixed operator views used by the stack.
- `n_operator_features_`: concatenated score feature count.
- `cv_scores_`: per-spec train-only CV diagnostics.
- `stack_report_`: JSON-serializable diagnostic report.

## Validation

The native tests cover:

- compact result shapes and selected-spec criterion;
- explicit fold ids;
- final prediction reconstruction from `stack_features`, `coefficients` and
  `intercept`;
- final prediction reconstruction from `input_coefficients` and
  `input_intercept`;
- operator feature offsets;
- rejection of multi-output `Y`;
- CPU and CUDA-enabled C++ test suites.

The Python estimator tests cover:

- custom fixed operators with finite predictions and score transforms;
- false-positive rejection by the OOF baseline gate;
- metadata perturbation invariance;
- default strict-linear operator bank smoke behavior;
- `predict` before `fit` error behavior.

2026-06-04 validation:

- CPU `n4m_tests`: 328 passed, 0 failed.
- CUDA-enabled `n4m_tests` with `CUDA_VISIBLE_DEVICES=0`: 328 passed, 0 failed.
- Targeted wrapper pytest through `N4M_LIB_PATH`: 13 passed.
- Full Python binding pytest against packaged ABI 1.18.0: 254 passed, 4
  existing UVE warnings.
- Catalog/ABI gates: 196 methods, 558/558 method symbols attributed, 123 infra
  symbols, split method files up to date.
- Import smoke without `N4M_LIB_PATH`: ABI `(1, 18, 0)` and
  `n4m.aom_operator_pls_stack` exported.

## Benchmarks

Timing script:

```bash
PYTHONPATH=bindings/python/src \
N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so \
python3 benchmarks/cross_binding/bench_aom_operator_pls_stack_timing.py
```

The CPU timing smoke uses the compact profile with `cv=4`, components
`[1, 2]`, and alphas `[0.01, 1.0]`. The generated CSV records the current ABI,
library path, elapsed medians, replay error, and fit-count telemetry.

CUDA-build smoke:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=bindings/python/src \
N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so \
python3 benchmarks/cross_binding/bench_aom_operator_pls_stack_timing.py \
  --output benchmarks/cross_binding/aom_operator_pls_stack_timing_cuda_smoke.csv \
  --mode both
```

The CUDA smoke CSV includes both the ABI-close function row
`native_aom_operator_pls_stack` and the sklearn replay wrapper row
`native_aom_operator_pls_stack_sklearn`. The wrapper row records
`prediction_replay_max_abs_error` to prove `predict(X)` replays the native
folded input-space stack on the CUDA build path.

The timing rows also expose deterministic cost accounting for the internal
`fit_stack` calls. For the compact smoke grid (`cv=4`, `n_specs=4`,
`n_operators=12`), `n_operator_pls_stack_fit_calls=21`,
`n_operator_pls_stack_pls_fit_calls=252`, and
`n_operator_pls_stack_ridge_fit_calls=21`. The CV/final breakdown is also
recorded as `n_pls_stack_cv_fits=240`, `n_pls_stack_final_fits=12`,
`n_ridge_stack_cv_fits=20`, and `n_ridge_stack_final_fits=1`.
