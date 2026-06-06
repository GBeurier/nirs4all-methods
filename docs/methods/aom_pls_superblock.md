# AOM PLS Superblock

`n4m.aom_pls_superblock` is a strict-linear donor-style AOM-PLS method. It
materializes a bank of AOM operator views, concatenates them into one wide
superblock, then fits a PLS head on that superblock.

The fitted model is replayable from the original input spectra:

```python
pred = X @ result["input_coefficients"] + result["intercept"]
```

The sklearn wrapper is `n4m.sklearn.NativeAOMPLSSuperblockRegressor`. The same
surface is exported from `n4m.aom` and `n4m.moment`.

## Selection Protocol

- Operators must be strict-linear AOM operators supported by `aom_preprocess`.
- PLS component selection is train-CV only via `pls_components`.
- Each CV fold builds its superblock from the fold training rows, applies
  train-fold centering and optional block RMS scaling, and scores validation
  rows through that fold-local state.
- The final model fits the selected component count on all calibration rows.
- Test/eval rows are never used for production selection.

## Scope

This ports the donor AOM-PLS `superblock` idea only. It does not include
row-reference preprocessing branches, nonlinear lifts, MKL/kernel weighting, or
dataset/source/name routing.

CUDA builds can route the underlying PLS fit through the existing native PLS
CUDA controls, but this is not yet a fused many-operator GPU superblock grinder.
