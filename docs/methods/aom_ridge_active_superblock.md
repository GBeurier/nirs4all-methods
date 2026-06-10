# AOM Ridge Active Superblock

`n4m.aom_ridge_active_superblock` is a strict-linear donor-style AOM Ridge
method. It starts from a bank of AOM preprocessing operators, screens a small
active subset on each training fold, then fits one Ridge model on the
concatenated active operator outputs.

The fitted model is replayable from the original input spectra:

```python
pred = X @ result["input_coefficients"] + result["intercept"]
```

The sklearn wrapper is `n4m.sklearn.NativeAOMRidgeActiveSuperblockRegressor`.
The same surface is available from `n4m.aom` and `n4m.moment`.

## Selection Protocol

- Alpha selection is train-CV only.
- For every CV fold, active operators are screened from that fold's training
  rows only.
- The final model screens active operators once on the full calibration set,
  then fits Ridge with the selected alpha.
- Test/eval rows are never used for production selection.

The active score is defined on the strict-linear outputs produced by
`n4m.aom_preprocess`, not on donor-private covariance helpers:

```text
score_b = || scale_b * Z_b.T @ y_centered ||_F^2
```

where `Z_b` is the centered output of operator `b`. Optional `kta` and `blend`
score modes use the same train-only operator outputs.

## Scope

This method intentionally excludes donor AOM Ridge modes that are outside the
strict moment porting scope:

- `branch_global` row-reference preprocessing
- MKL/kernel weighting
- nonlinear lifts
- dataset/source/name routing

CUDA builds can run this method through the same native preprocessing and Ridge
bindings, but this is not a fused GPU active-superblock grinder.
