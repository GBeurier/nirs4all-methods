# `standard_scale` — n4m.transform.scaling.standard_scale

_Namespace_: **`n4m.transform.scaling`** · _Fully-qualified_: `n4m.transform.scaling.standard_scale` · _Catalog id_: `preprocessing.scaling.standard_scale`

## API surface

**C ABI:** no standalone exported symbol is declared for this method.

**Python:** catalog binding is not currently an AST-verified public `n4m` re-export. See the implementation source below.

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

Population standardization uses the arithmetic mean and population variance. The public StandardScaler reference documents the same ddof=0 convention: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html.

### Mathematical principle

For each training column, the native fitter stores its mean and population standard deviation. Transform applies (x-mean)/scale, with centering and scaling independently controlled by with_mean and with_std. A zero-variance column receives scale one. Held-out rows reuse the learned vectors; they never update training statistics.

### Appropriate uses

Standardizing heterogeneous numerical predictors before a fitted encoder or estimator, with a separate scaler fitted on each training fold.

### Limits and validation

Fitting on validation or Test rows leaks their distribution into training. Standardization is sensitive to outliers and does not establish normality. Population variance differs from sample variance; ddof=1 references require an explicit convention adjustment before parity comparisons.

### Implementation

The native transformer preprocessing.scaling.standard_scale is exposed as n4m.roles.StandardScale through the shared n4m_estimator_* lifecycle. cpp/include/n4m/estimator.h defines the shared ABI. cpp/src/core/estimator/fitted_core.cpp owns fitting, transformation and portable learned means/scales; bindings marshal inputs without numerical logic.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/estimator/fitted_core.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/catalog/methods/preprocessing.scaling.standard_scale.yaml

## Catalog note

Native transformer through the shared n4m_estimator_* role API, with no dedicated C exports. Train-only population standardization (ddof=0; zero variance scale=1), with portable native learned means and scales.


_See also_: [methods index](index.md).