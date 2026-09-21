# AOM calibration contracts (development ABI 2.6)

The production entry points are `n4m.model_selection.AOMPLSRegressor`,
`AOMRidgeRegressor`, `FastAOMPLSRegressor`, `FastAOMRidgeRegressor`, and
`n4m.ensemble.LinearRidgeStackRegressor`. They use the native numerical engine.
The AOM research repository may forward these names through `aom_nirs.calibration`;
the dependency goes from AOM to methods. Methods does not import AOM.
These additions are unreleased; project version 1.0.19 alone does not identify
this working tree. Use ABI 2.6 plus source and native-library hashes.

## Versioned protocols

| Entry point | Selection and deployment contract |
|---|---|
| AOMPLSRegressor | `global-pls-branch-cv-v1`; exhaustive branch/chain and SIMPLS prefix selection, then final refit. |
| AOMRidgeRegressor | `global-ridge-branch-cv-v1`; exhaustive branch/chain and Ridge alpha selection, one eigendecomposition per fold/view. |
| FastAOMPLSRegressor | `fast-pls-branch-cv-v1`; covariance screen on each training fold, validate its leader, then screen/refit on all calibration rows. |
| FastAOMRidgeRegressor | `fast-ridge-branch-cv-v1`; experimental Ridge extension of the same screen. PLS speed/accuracy evidence does not validate this extension. |
| LinearRidgeStackRegressor | `linear-ridge-stack-nested-oof-v1`; strict-linear base views, inner-CV base tuning for each OOF fold, Ridge meta-head on OOF predictions, affine export. |

The `strict10-gaussian-v1` bank is identity; SG smooth (11,2)/(21,3);
SG first derivative (11,2)/(21,3); SG second derivative (11,2); detrend degree
1/2; and Gaussian sigma 1/2. Widths are channel counts. SG uses unit spacing
and zero padding. Gaussian kernels are **unnormalized** `exp(-x*x/(2*sigma*sigma))`,
truncated at four sigmas, with zero padding. The generic native Gaussian operator
still defaults to normalized kernels: its optional third descriptor parameter
is `normalize` (0 or 1). The versioned bank passes `(sigma, 4, 0)` explicitly;
this amplitude convention matters for Ridge alpha selection.

The chain grammar allows each of three roles (smoother, derivative, detrend)
at most once, in every order, with identity a separate singleton. Cumulative
counts are 10/62/206 at depth 1/2/3. Ties follow `(length, signature)` order.
Default branches are raw/SNV/MSC; SNV uses population standard deviation
(`ddof=0`), MSC uses a training-only reference. Centering is fitted per fold.
Default CV is three contiguous folds; pass explicit `fold_ids` to reproduce a
shuffled protocol. PLS tests prefixes 1..25 (unavailable prefixes get infinite
error). Default Ridge grids have 50 log-spaced factors 1e-6..1e6 multiplied by
`trace(Xc @ Xc.T)/n` from raw calibration X. Explicit alphas are absolute.
Scores are the mean of fold RMSEs, not pooled validation RMSE.

Fast screening uses exact covariance numerators and leading-rank covariance
energy denominators (`rank=200` by default). The native implementation computes
the leading exact Gram-SVD factors, not a randomized SVD. Scores with truncated
denominators can exceed one. All screening costs belong in measured fit time.
Only the strict-linear operator is folded. SNV/MSC branches still run at
prediction; a raw-input affine export cannot represent these branches.

## Fully linear Ridge stack

```python
from n4m.ensemble import LinearRidgeStackRegressor

stack = LinearRidgeStackRegressor(cv=3, inner_cv=3).fit(X_train, y_train)
prediction = stack.predict(X_test)  # already compressed
model = stack.export_affine()       # independent coefficients and intercept only
prediction = model.predict(X_test)
print(model.numeric_bytes)
```

For base coefficients B (p by m), base intercepts b, meta weights w and meta
intercept bm, the export is beta = B w and intercept = b w + bm. It preserves
the trained stack's predictions, including all intercepts. `compress_linear_stack`
also supports multiple output columns and can compress an existing affine stack
without retraining. Fixed affine preprocessing must already be in the base
coefficients. Nonlinear/sample-adaptive transforms cannot be folded into raw X.

Base tuning inside each outer OOF fold only sees that fold's training rows.
The base grids are recomputed on those rows. The meta grid uses OOF design
energy. Meta CV is a tuning criterion, not an unbiased generalization estimate;
report accuracy on an untouched external test set. Compression reduces
inference and deployment storage, while the complete OOF/base/meta training
still has to run. `predict_uncompressed` is retained for auditing. The exported
`AffinePredictor` contains no base models, OOF predictions or training data.

## Historical names are distinct contracts

Existing `aom_search` functions and `NativeAOM*` wrappers keep their behavior.
The historical selector defaults (nine operators and three PLS components),
strict-only Ridge global selector (four alphas and CV 5), superblocks, MKL,
simplex Ridge blender, and hard/soft/sparse chain models are not aliases for
these branch-aware protocols. A Ridge meta-model permits signed weights and
an intercept; it is not a simplex blender.

In particular, research `AOMRidgePLS`/`SingleChainPLSRidge` applies Ridge to PLS
scores; native `ridge_pls` applies PLS to an L2-augmented design. Their latent
spaces and tuning grids differ. `aom_chain_ridge_pls` uses the latter and is
not a parity port of the former. See [Ridge-PLS](ridge_pls.md).

## Verification

`bindings/python/tests/test_aom_calibration.py` checks the complete candidate
Ridge curves against independent SciPy operator matrices and sklearn Ridge,
plus branch state, grammar, cloning and serialization. `test_linear_ridge_stack.py`
checks native compression with multiple outputs/intercepts, fixed-alpha OOF
stack parity against sklearn, and export independence from the training object.
Benchmarks should compare to both separate base predictions and a batched
`(X @ B + b) @ w + bm` baseline, with the same test matrix and alternating order.

The stack uses pooled validation RMSE for base and meta alpha selection, matching
the generic native sweep; this differs from the mean-fold RMSE used by the
branch-aware calibration facades. On near-square/wide inputs (`p > n/2`), its
internal single-view adapter reuses one smaller-Gram eigendecomposition per
fold across all alphas; tall inputs keep the moment sweep. The route depends
only on dimensions. This avoids the generic sweep's repeated design-fit
fallback on wide spectra while preserving its selection criterion.

With BLAS builds, `N4M_AOM_WITH_LAPACKE=ON` (default) detects whether the linked
BLAS provides LAPACKE and uses `LAPACKE_dsyevd` for this calibration eigenpath.
Detection failure or an explicit `OFF` retains the portable implementation.
The option is scoped to this new AOM path; other native eigensolvers are unchanged.
Record the build option and backend with timing results.

If the eigensolver fails to converge on a degenerate view, the calibration path falls back to the existing direct native Ridge solver on the same alpha grid. It keeps the candidate and its original CV criterion.

The global native calibration facade currently materializes each candidate view
for its model path, while FastAOM materializes only the screened fold leaders.
Protocol conformance does not imply the same internal execution strategy as the
independent covariance/adjoint Python reference. Runtime results must identify
the actual implementation snapshot; Python-reference timings cannot be relabelled
as timings of these new native facades.
