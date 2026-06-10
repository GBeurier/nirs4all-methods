# `moment_stack` — OOF stack over native moment heads

_Group_: **Ensemble** · _Registry tolerance_: `n/a`

## Description

`NativeMomentStackRegressor` builds a small linear meta-model over out-of-fold
predictions from native moment-compatible base models:

- Ridge (`NativeRidgeRegressor`)
- PLS via `NativeMomentSweepRegressor(heads=("pls", ...))`
- PCR (`NativePCRRegressor`)
- Continuum regression (`NativeContinuumRegressionRegressor`)
- ECR (`NativeECRRegressor`)
- CPPLS (`NativeCPPLSRegressor`)

The base models are fit only on training folds when producing OOF predictions.
The final reusable estimator refits the same base models on all training rows,
then applies a small Ridge meta-model on their predictions.

This is intentionally still a strict moment-model composition. It does not add
RFF/RBF lifts, trees, neural models, TabPFN, transformed-spectrum stacking, or
dataset-name routing.

## Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `base_models` | sequence[str] | `("ridge", "pls", "pcr", "continuum", "ecr", "cppls")` | Allowed bases. `continuum_regression` aliases `continuum`. |
| `cv` | `int` | `5` | Outer OOF folds used for the meta-model. |
| `fold_ids` | array-like or `None` | `None` | Explicit train-only fold ids. |
| `inner_cv` | `int` | `3` | Inner CV for the PLS sweep base. |
| `meta_alpha` | `float` | `1e-6` | Ridge penalty for the meta-model. |
| `ridge_alpha` | `float` | `0.1` | Ridge base penalty. |
| `n_components` | `int` | `3` | Component count for PCR/continuum/ECR/CPPLS and max PLS grid when `pls_components=None`. |
| `pls_components` | sequence[int] or `None` | `None` | Explicit PLS component grid. |
| `cppls_gamma` | `float` | `0.5` | CPPLS gamma. |
| `continuum_tau` | `float` | `0.5` | Continuum tau. |
| `ecr_alpha` | `float` | `0.5` | ECR alpha. |
| `center_x`, `scale_x`, `center_y`, `scale_y` | bool or `None` | mixed | Forwarded to bases that expose those options. |
| `cuda_pls_parallel_folds`, `cuda_pls_min_device_features`, `cuda_pls_many_batched` | optional | `None` | Forwarded to the PLS sweep base. |

## Usage

```python
import n4m
from n4m.sklearn import NativeMomentStackRegressor

model = n4m.moment_stack(
    X_train,
    y_train,
    base_models=("ridge", "pcr", "pls"),
    cv=5,
    inner_cv=3,
    n_components=3,
    scale_x=False,
)

same_model_type = NativeMomentStackRegressor(
    base_models=("ridge", "pcr", "pls"),
    cv=5,
    inner_cv=3,
    n_components=3,
    scale_x=False,
).fit(X_train, y_train)

y_pred = model.predict(X_test)
diagnostics = model.get_diagnostics()
```

Key fitted attributes:

- `base_model_names_`
- `base_models_`
- `base_oof_predictions_`
- `oof_predictions_`
- `meta_coefficients_`
- `intercept_`
- `oof_rmse_`
- `rmse_`
- `base_oof_diagnostics_`
- `base_final_diagnostics_`

`get_diagnostics()` includes aggregate PLS route counters for both phases, for
example `n_base_oof_pls_moment_cuda_device_cv_fits` and
`n_base_final_pls_moment_cuda_device_cv_fits`. These counters are audit-only;
they do not affect the OOF split, meta-model fit, or production selection.

CUDA smoke used for release readiness:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=bindings/python/src \
N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so \
  /home/delete/.venv/bin/python benchmarks/cross_binding/bench_moment_stack_timing.py \
  --output benchmarks/cross_binding/moment_stack_timing_cuda_smoke.csv \
  --repeats 1 --shapes 80x1024 --base-models pls --cv 4 --inner-cv 4 \
  --n-components 1 --cuda-pls-min-device-features 1 \
  --cuda-pls-parallel-folds
```

The smoke should show nonzero OOF and final
`n_base_*_pls_moment_cuda_device_cv_fits` and zero corresponding host PLS CV
fits.

## Validation

Covered by `bindings/python/tests/test_moment_model_wrappers.py` and
`benchmarks/cross_binding/bench_moment_stack_timing.py`.
