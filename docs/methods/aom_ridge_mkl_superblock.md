# `aom_ridge_mkl_superblock` - strict-linear AOM Ridge MKL-light superblock

`n4m.aom_ridge_mkl_superblock` is the moment-compatible subset of donor
AOM-Ridge MKL-light. It learns non-negative train-only KTA weights over a bank
of strict-linear AOM operator views, fits native Ridge on the equivalent
weighted superblock, then folds the final model back to raw input-space
`input_coefficients` plus `intercept`.

This is not a nonlinear kernel route. The combined model is equivalent to a
single linear Ridge model on concatenated weighted operator features, so
`predict()` can replay directly as:

```python
y_hat = X @ res["input_coefficients"] + res["intercept"]
```

It intentionally excludes branch/global preprocessing, row-reference-dependent
preprocessing, local/SNV/MSC branches, nonlinear kernels and TabPFN residuals.

## API

```python
import n4m

res = n4m.aom_ridge_mkl_superblock(
    X,
    y,
    operators=["identity", ("finite_difference", [1]), ("savgol_smooth", [5, 2])],
    alphas=[0.01, 0.1, 1.0],
    mkl_top_k=3,
    cv=5,
)

print(res["mkl_weights"].ravel())
print(res["selected_operator_indices"])
```

The sklearn wrapper is `n4m.sklearn.NativeAOMRidgeMKLSuperblockRegressor`.

## Selection

For each alpha-CV fold, operator weights are learned only from the fold training
rows:

1. Build each strict-linear operator output block.
2. Center the block and target on the fold training rows.
3. Score each block by kernel-target alignment between `Z_b Z_b.T` and
   `Y Y.T`.
4. Keep at most `mkl_top_k` positive-alignment blocks and project weights onto
   the simplex.
5. Fit native Ridge on the weighted superblock and score the validation rows.

The final model relearns weights on the full calibration rows and refits the
selected alpha. Held-out/test rows are never used for production selection.

## Benchmark

```bash
PYTHONPATH=bindings/python/src \
N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so \
python benchmarks/cross_binding/bench_aom_ridge_mkl_superblock_timing.py
```

CUDA-enabled builds can run the same smoke by pointing `N4M_LIB_PATH` at
`build/cuda-on/cpp/src/libn4m.so`; this proves CUDA-build compatibility, but the
current implementation is not a fused GPU weighted-superblock grinder.
