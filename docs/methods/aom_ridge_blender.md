# `aom_ridge_blender` - native AOM Ridge OOF simplex blender

_Group_: **Diagnostic / AOM** · _ABI_: `n4m_aom_ridge_blender_fit`

`aom_ridge_blender` runs a native strict-linear AOM Ridge candidate pool,
builds out-of-fold predictions for every chain/lambda candidate, solves a
non-negative simplex blend, then refits every candidate on the full training
set for final blended predictions. Because every native candidate is a
strict-linear chain plus Ridge head, the final simplex blend is also exported
as a reusable linear model in the original input feature space.

The sklearn-style `AOMRidgeBlender` reference estimator is still available as
`n4m.AOMRidgeBlender` and `n4m.sklearn.AOMRidgeBlender` for explicit custom
candidate estimators.

## Status

- API surface: C ABI, Python function `n4m.aom_ridge_blender`, and native
  sklearn wrapper `NativeAOMRidgeBlenderRegressor`.
- Native ABI: yes, since ABI `1.17.0`.
- Catalog status: `aom_pop.ridge_blender`.
- CPU: tested.
- CUDA: builds and smoke-tests with the CUDA-enabled library; native v1 is not
  a fused batched GPU blender.
- Candidate scope: fixed compact/wide strict-linear AOM chain banks and
  strictly positive Ridge lambdas. Compact has 12 chains; wide has 31 chains,
  including Gaussian, FCK and Whittaker variants.
- Dataset/source routing: forbidden. `metadata` passed to `fit` is audit-only
  for the Python estimator and does not affect candidates, weights, or
  predictions.

## Python Function

```python
n4m.aom_ridge_blender(
    X,
    y,
    profile="compact",
    cv=5,
    fold_ids=None,
    ridge_lambdas=(1e-4, 1e-2, 1.0, 100.0),
    regularizer=0.01,
    scale_x=False,
)
```

## Example

```python
import n4m

res = n4m.aom_ridge_blender(
    X_train,
    y_train,
    profile="compact",
    cv=5,
    fold_ids=fold_ids,
    ridge_lambdas=[1e-4, 1e-2, 1.0, 100.0],
    regularizer=0.01,
    scale_x=False,
)

print(res["blend_oof_rmse"], res["selected_chain_id"], res["selected_param"])
print(res["weights"])

y_hat = X_train @ res["input_coefficients"] + res["intercept"]
```

## Outputs

The native method returns a `MethodResult` dictionary with:

- `candidate_scores` `(n_candidates, 5)`: `candidate_id`, `chain_id`,
  `lambda`, `cv_rmse`, `weight`
- `weights` `(1, n_candidates)`: non-negative simplex weights
- `oof_predictions` `(n_samples, n_targets)`: blended OOF predictions
- `predictions` `(n_samples, n_targets)`: blended final in-sample predictions
- `input_coefficients` `(n_features, n_targets)`: final blended coefficients
  folded back into the original input feature space
- `intercept` `(1, n_targets)`: final blended intercept
- `oof_candidate_predictions` `(n_samples * n_targets, n_candidates)`
- `candidate_predictions` `(n_samples * n_targets, n_candidates)`
- `fold_ids`

Scalar diagnostics include `selected_candidate_id`, `selected_chain_id`,
`selected_param`, `selected_cv_rmse`, `blend_oof_rmse`, `regularizer`,
`n_candidates`, `n_chains`, `profile`, `cv`, `n_samples`, `n_features` and
`n_targets`.

The selected blend can be replayed on compatible spectra as:

```python
pred = X_new @ res["input_coefficients"] + res["intercept"]
```

## Python Estimator

`NativeAOMRidgeBlenderRegressor` is the sklearn-style wrapper over the native
compact/wide ABI. It stores `coef_`, `intercept_`, `weights_`,
`candidate_scores_`, OOF predictions, and diagnostics while predicting from
the folded native coefficients.

`AOMRidgeBlender` remains the flexible Python/sklearn reference layer. It can
blend explicit estimator/factory candidates or build a default AOM-Ridge pool
from `build_aom_control_chain_bank(profile)`. It supports custom `chains`,
candidate labels, `random_state`, `n_jobs`, and candidate failure reporting.

Use the native function for the catalogued compact/wide strict-linear Ridge
bank. Use the estimator when the candidates are Python objects or when a
scikit-learn-compatible estimator API is needed.

## Validation

The native tests cover:

- compact and wide chain/lambda candidate counts and result shapes;
- simplex non-negativity and unit-sum weights;
- exact reconstruction of blended OOF/final predictions from candidate
  prediction matrices;
- exact replay of final predictions from `input_coefficients` and `intercept`;
- explicit fold ids;
- rejection of non-positive Ridge lambdas;
- CPU and CUDA-enabled C++ test suites.

The Python estimator tests cover:

- simplex weights and exact OOF blend recovery on synthetic candidates;
- fold-local OOF fitting, with the validation fold excluded from each candidate
  fit;
- metadata perturbation invariance;
- default chain+Ridge candidate construction;
- `predict` before `fit` error behavior.

## Benchmarks

Timing script:

```bash
PYTHONPATH=bindings/python/src \
N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so \
python3 benchmarks/cross_binding/bench_aom_ridge_blender_timing.py
```

CUDA-build smoke:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=bindings/python/src \
N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so \
python3 benchmarks/cross_binding/bench_aom_ridge_blender_timing.py \
  --output benchmarks/cross_binding/aom_ridge_blender_timing_cuda_smoke.csv
```

Current smoke CSVs:

- `benchmarks/cross_binding/aom_ridge_blender_timing.csv`
- `benchmarks/cross_binding/aom_ridge_blender_timing_cuda_smoke.csv`

Current ABI 1.17.0 compact 48-candidate medians:

- CPU: 16.32 ms for 32 x 64, 69.39 ms for 64 x 128, 272.69 ms for 96 x 256.
- CUDA-build smoke: 15.63 ms, 81.00 ms, 299.56 ms on the same cells.

On the same synthetic cells, `blend_oof_rmse` is lower than the best single
candidate OOF RMSE, which confirms that the stored simplex weights and
candidate OOF matrix are doing useful blend work in the smoke benchmark.
