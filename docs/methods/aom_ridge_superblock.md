# `aom_ridge_superblock` - strict-linear AOM Ridge superblock

_Group_: **Diagnostic / AOM** · _Backend_: Python orchestration over native
`aom_preprocess` and native `ridge`

`aom_ridge_superblock` concatenates a bank of strict-linear AOM operator views,
fits one Ridge model on that superblock, and folds the final coefficients back
to the original input feature space. It is the moment-compatible donor
AOM-Ridge superblock route: no branch-global reference-dependent preprocessing,
no MKL/kernel route, no nonlinear lift, and no dataset/source-name routing.

## Status

- API surface: Python function `n4m.aom_ridge_superblock` and sklearn wrapper
  `NativeAOMRidgeSuperblockRegressor`.
- Native ABI: no new ABI; the implementation reuses native `n4m.aom_preprocess`
  and native `n4m.ridge`.
- Catalog status: `aom_pop.ridge_superblock`.
- CPU: tested.
- CUDA: works against CUDA-enabled `libn4m` builds and uses the native Ridge
  binding for every fold/final fit; this is not yet a fused GPU superblock
  grinder.
- Candidate scope: caller-provided strict single operators or the default
  strict AOM selector bank (`identity`, detrend, SavGol, Norris-Williams and
  finite difference variants).

## Python Function

```python
n4m.aom_ridge_superblock(
    X,
    y,
    operators=None,
    alpha=None,
    alphas=(1e-4, 1e-2, 1.0, 100.0),
    cv=5,
    fold_ids=None,
    block_scaling="rms",
    center_x=True,
    center_y=True,
)
```

If `alpha` is supplied, the method fits that value and still returns one
candidate-score row. Otherwise it evaluates `alphas` by fold-local CV. For each
fold, preprocessing is run on train and validation rows independently, then
the train-fold superblock mean and block scales are applied to validation rows.

## Outputs

The function returns:

- `candidate_scores` `(n_alphas, 3)`: `candidate_id`, `alpha`, `cv_rmse`
- `oof_predictions` `(n_samples, n_targets)`
- `predictions` `(n_samples, n_targets)`
- `superblock_coefficients` / `coefficients`
- `input_coefficients` `(n_features, n_targets)`
- `intercept` `(1, n_targets)`
- `operator_outputs`, `operator_kinds`, `block_scales`, `superblock_mean`
- diagnostics including `selected_alpha`, `selected_cv_rmse`,
  `n_operators`, `n_features_superblock`, `cv`, `block_scaling` and
  `ridge_backend="native"`

The final model can be replayed on compatible spectra as:

```python
pred = X_new @ res["input_coefficients"] + res["intercept"]
```

## Python Estimator

`NativeAOMRidgeSuperblockRegressor` wraps the same function and predicts from
the folded `input_coefficients` plus `intercept`.

```python
from n4m.sklearn import NativeAOMRidgeSuperblockRegressor

model = NativeAOMRidgeSuperblockRegressor(
    operators=["identity", ("finite_difference", [1])],
    alphas=[0.01, 0.1, 1.0],
    cv=5,
).fit(X_train, y_train)

y_hat = model.predict(X_test)
```

## Benchmarks

Timing script:

```bash
PYTHONPATH=bindings/python/src \
N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so \
python3 benchmarks/cross_binding/bench_aom_ridge_superblock_timing.py
```

CUDA-build smoke:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=bindings/python/src \
N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so \
python3 benchmarks/cross_binding/bench_aom_ridge_superblock_timing.py \
  --output benchmarks/cross_binding/aom_ridge_superblock_timing_cuda_smoke.csv \
  --repeats 1 --cv 4 --mode both
```
