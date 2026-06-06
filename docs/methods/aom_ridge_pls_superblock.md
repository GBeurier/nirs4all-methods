# `aom_ridge_pls_superblock` - strict-linear AOM Ridge-PLS superblock

`n4m.aom_ridge_pls_superblock` ports the donor AOM Ridge-PLS superblock idea
inside the n4m moment contract. It builds strict-linear AOM operator views,
concatenates them into one superblock, selects the PLS component count and
Ridge-PLS penalty by train CV, then folds the final superblock coefficients
back to raw input-space `input_coefficients` plus `intercept`.

This is a reusable strict-linear method, not a dataset router. It intentionally
excludes MKL/kernel modes, branch/global routing, row-reference-dependent
preprocessing, nonlinear lifts and TabPFN residuals.

## API

```python
import n4m

res = n4m.aom_ridge_pls_superblock(
    X,
    y,
    operators=["identity", ("finite_difference", [1]), ("savgol_smooth", [5, 2])],
    pls_components=[1, 2, 4],
    ridge_lambdas=[0.0, 0.1, 1.0],
    cv=5,
)

y_hat = X @ res["input_coefficients"] + res["intercept"]
```

The sklearn wrapper is `n4m.sklearn.NativeAOMRidgePLSSuperblockRegressor`.
`predict()` uses only the folded input coefficients and intercept.

## Selection

Each CV fold rebuilds the AOM operator superblock from the fold training rows,
applies fold-local centering and optional block RMS scaling, fits the native
`ridge_pls` head on the fold train design, and scores the validation rows. The
selected candidate minimizes train-CV RMSE over the cartesian grid of
`pls_components` x `ridge_lambdas`.

## Benchmark

```bash
PYTHONPATH=bindings/python/src \
N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so \
python benchmarks/cross_binding/bench_aom_ridge_pls_superblock_timing.py
```

CUDA-enabled builds can run the same smoke by pointing `N4M_LIB_PATH` at
`build/cuda-on/cpp/src/libn4m.so`; this proves CUDA-build compatibility, but the
current Ridge-PLS superblock implementation is not a fused GPU grinder.
