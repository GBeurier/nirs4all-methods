# `aom_ridge_global` - strict-linear AOM Ridge global selector

_Group_: **Diagnostic / AOM** · _Backend_: native `aom_chain_sweep_run`

`aom_ridge_global` selects one strict-linear AOM operator and one Ridge alpha by
native cross-validation, then returns the final reusable model folded back into
the original input feature space. It is the moment-compatible donor
AOM-Ridge global route. It does not include branch-global reference-dependent
preprocessing, MKL/kernel routing, nonlinear lifts, or dataset/source-name
routing.

## Status

- API surface: Python function `n4m.aom_ridge_global` and sklearn wrapper
  `NativeAOMRidgeGlobalRegressor`.
- Native ABI: no new ABI; it delegates to native `n4m.aom_chain_sweep_run`.
- Catalog status: `aom_pop.ridge_global`.
- CPU: tested.
- CUDA: works against CUDA-enabled `libn4m` builds; this is not a fused GPU
  grinder.
- Candidate scope: caller-provided strict single operators or the default
  strict AOM selector bank.

## Python Function

```python
n4m.aom_ridge_global(
    X,
    y,
    operators=None,
    cv=5,
    fold_ids=None,
    ridge_lambdas=(1e-4, 1e-2, 1.0, 100.0),
    scale_x=False,
    moment_policy="auto",
)
```

`operators` are converted to one-op strict AOM chains. Selection uses the
native Ridge-only AOM chain sweep and `selected_cv_rmse`.

## Outputs

The method returns the native AOM sweep result plus:

- `operators`: the canonical one-op chain bank
- `selected_operator`, `selected_operator_index`, `selected_operator_kind`
- `selection_mode="global"`
- `ridge_backend="native_aom_chain_sweep"`

The final model can be replayed as:

```python
pred = X_new @ res["input_coefficients"] + res["intercept"]
```

## Python Estimator

```python
from n4m.sklearn import NativeAOMRidgeGlobalRegressor

model = NativeAOMRidgeGlobalRegressor(
    operators=["identity", ("finite_difference", [1])],
    ridge_lambdas=[0.01, 0.1, 1.0],
    cv=5,
).fit(X_train, y_train)

y_hat = model.predict(X_test)
```

## Benchmarks

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=bindings/python/src \
N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so \
python3 benchmarks/cross_binding/bench_aom_ridge_global_timing.py \
  --output benchmarks/cross_binding/aom_ridge_global_timing_cuda_smoke.csv \
  --repeats 1 --cv 4 --mode both
```
