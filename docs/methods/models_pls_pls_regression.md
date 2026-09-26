# `pls_regression` — n4m.estimators.regression.latent.pls_regression

_Namespace_: **`n4m.estimators.regression.latent`** · _Fully-qualified_: `n4m.estimators.regression.latent.pls_regression` · _Catalog id_: `models.pls.pls_regression`

## API surface

**C ABI (ABI 2):** [`n4m_model_fit`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/n4m.h#L437). Use the linked public header for the exact signature, configuration, and result handles.

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

Wold, Sjöström & Eriksson (2001), *PLS-regression: a basic tool of chemometrics*, Chemometrics and Intelligent Laboratory Systems 58, 109–130, https://doi.org/10.1016/S0169-7439(01)00155-1; de Jong (1993), *SIMPLS: an alternative approach to partial least squares regression*, https://doi.org/10.1016/0169-7439(93)85002-X.

### Mathematical principle

PLS regression extracts $k$ latent components $t_a = X w_a$ whose weights maximize the covariance between the (centered, optionally scaled) predictors and responses, deflates, and regresses $Y$ on the scores. The fitted model is the affine predictor $\hat Y = \bar y + (X-\bar x) B$ with $B = W(P^T W)^{-1} Q^T$; NIPALS, SIMPLS, kernel, SVD and power solvers compute the same subspace up to numerical and single-response conventions.

### Appropriate uses

General multivariate calibration of spectra to one or several responses, as the reference linear model and as a latent-score feature extractor ahead of another estimator.

### Limits and validation

The component count must be chosen by validation on held-out data; solvers can differ at the last digits and SIMPLS deflates only the cross-covariance, so multi-response scores are not identical across solvers. Scaling choices change the fitted subspace.

### Implementation

Generic role estimator `models.pls.pls_regression` (regressor and transformer) over `n4m_model_fit`; state is an N4MM model inside the N4ME format.

### Sources and provenance

https://doi.org/10.1016/S0169-7439(01)00155-1; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/model.cpp

## Catalog note

PLS regression through the N4MM model life cycle (n4m_model_fit) with a selectable solver; the general PLS estimator of the generic role surface.


_See also_: [methods index](index.md).