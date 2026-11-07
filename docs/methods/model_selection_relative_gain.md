# `paired_relative_gain` — n4m.ranking.paired_relative_gain

_Namespace_: **`n4m.ranking`** · _Fully-qualified_: `n4m.ranking.paired_relative_gain` · _Catalog id_: `model_selection.relative_gain`

## API surface

**C ABI (ABI 2):** [`n4m_ranking_relative_gain_v1`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/ranking.h#L73). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):** `from n4m.ranking import paired_relative_gain`

**Signature:** [`paired_relative_gain(baseline: ArrayLike, candidate: ArrayLike, *, present: ArrayLike | None = None, variances: ArrayLike | None = None)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/ranking.py#L214)

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

### Parameters

| Name | Type | Default |
|---|---|---|
| `baseline` | `ArrayLike` | `required` |
| `candidate` | `ArrayLike` | `required` |
| `present` | `ArrayLike \| None` | `None` |
| `variances` | `ArrayLike \| None` | `None` |

## Explanations

### Bibliographic source

DerSimonian and Laird (1986), Meta-analysis in clinical trials, Controlled Clinical Trials 7, 177-188, https://doi.org/10.1016/0197-2456(86)90046-2. The inverse-variance random-effects assumptions and heterogeneity quantities are described in the Cochrane Handbook, Chapter 10: https://www.cochrane.org/authors/handbooks-and-manuals/handbook/current/chapter-10.

### Mathematical principle

For each explicitly present pair of strictly positive losses, define the relative improvement $e_i=(b_i-c_i)/b_i$. The arithmetic mean and sample SD describe the paired effects. Optional caller-supplied sampling variances v_i define fixed weights w_i=1/v_i and $Q=\sum_i w_i(e_i-\bar e_w)^2$. For m studies, the DerSimonian-Laird estimate is $\tau^2=\max(0,(Q-m+1)/(\sum w_i-\sum w_i^2/\sum w_i))$. The pooled effect uses random-effects weights $u_i=1/(v_i+\tau^2)$, with standard error $1/\sqrt{\sum u_i}$. I-squared is the nonnegative fraction (Q-m+1)/Q when Q is positive, otherwise zero.

### Appropriate uses

Comparing a declared baseline and candidate through matched positive losses; optionally synthesizing independent study-level relative effects when their sampling variances, including paired covariance, are supplied in the same relative-effect units.

### Limits and validation

Losses must be finite and strictly positive on present rows. Raw-loss variances and fold spread are not valid substitutes for relative-effect sampling variances. Repeated or overlapping folds do not become independent studies by passing a variance vector. The sample SD is descriptive dispersion, not a standard error. Without variances or with fewer than two effects, pooled effect, SE, tau-squared, Q and I-squared are unavailable; no confidence interval is fabricated. Finite extremes outside representable intermediate or output ranges are explicitly refused rather than silently discarding positive weights.

### Implementation

The same dependency-free native ranking kernel supplies the n4m_ranking_relative_gain_v1 caller-buffer C ABI and Python paired_relative_gain wrapper. Explicit 0/1 masks control absence; core validation checks positive losses and supplied variances. Positive prefix cross-products evaluate the DL denominator in linear time without subtractive cancellation, and scaled/extended intermediates retain supported variance ranges. Standardized residuals and separately square-rooted SE factors avoid avoidable underflow. Tests use exact rational DL examples, dominant-weight and subnormal-variance counterexamples, and actual WASM C ABI execution. No standalone R, MATLAB or JS relative-gain facade is claimed.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/ranking/ranking.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/ranking.h; https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/tests/test_ranking.py; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/tests/test_ranking.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/js/test/run_ranking.mjs; https://github.com/GBeurier/nirs4all-methods/blob/main/docs/RANKING.md

## Catalog note

Paired strictly-positive loss relative gains and optional DerSimonian-Laird random effects, only with caller-supplied sampling variances of the relative effects. No fold-derived uncertainty.


_See also_: [methods index](index.md).