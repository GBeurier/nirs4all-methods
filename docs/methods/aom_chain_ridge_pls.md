# `aom_chain_ridge_pls` - strict-chain AOM Ridge-PLS

`n4m.aom_chain_ridge_pls` ports the strict/raw-base subset of donor FastAOM
SingleChainPLSRidge into `nirs4all-methods`. It applies each candidate
strict-linear AOM chain sequentially, selects the chain, PLS component count
and Ridge-PLS penalty by train CV, then folds the selected final coefficients
back to raw input-space `input_coefficients` plus `intercept`.

This is a reusable linear method. It intentionally excludes SNV, MSC, EMSC,
OSC, row-reference-dependent preprocessing, nonlinear lifts, kernels, trees,
TabPFN residuals and dataset/source routing.

## API

```python
import n4m

chains = [
    [("identity", ())],
    [("savgol_smooth", (5, 2)), ("finite_difference", (1,))],
]

res = n4m.aom_chain_ridge_pls(
    X,
    y,
    chains=chains,
    pls_components=[1, 2, 4],
    ridge_lambdas=[0.0, 0.1, 1.0],
    cv=5,
)

y_hat = X @ res["input_coefficients"] + res["intercept"]
```

If `chains` is omitted, the function builds a strict-chain grid with
`build_aom_strict_chain_grid(profile=...)`.

The sklearn wrapper is `n4m.sklearn.NativeAOMChainRidgePLSRegressor`. Its
`predict()` method uses only the folded input coefficients and intercept.

## Selection

For each candidate `(chain, n_components, ridge_lambda)`, every CV fold applies
the same strict-linear chain to the fold train and validation matrices, fits
native `ridge_pls` on the transformed fold train matrix, and scores validation
RMSE. The final model refits the selected tuple on all calibration rows.

The raw-space replay is exact for strict-linear chains because the selected
chain is also applied to the identity matrix to recover its composed linear
map.

## Benchmark

```bash
PYTHONPATH=bindings/python/src \
N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so \
python benchmarks/cross_binding/bench_aom_chain_ridge_pls_timing.py
```

CUDA-enabled builds can run the same benchmark by pointing `N4M_LIB_PATH` at
`build/cuda-on/cpp/src/libn4m.so`. This is a compatibility smoke, not a fused
many-chain GPU Ridge-PLS grinder.
