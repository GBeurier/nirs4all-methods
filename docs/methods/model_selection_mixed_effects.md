# `mixed_effects` — n4m.component_analysis.mixed_effects

_Namespace_: **`n4m.component_analysis`** · _Fully-qualified_: `n4m.component_analysis.mixed_effects` · _Catalog id_: `model_selection.mixed_effects`

## API surface

**C ABI (ABI 2):** [`n4m_mixed_effects_options_init_v1`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/mixed_effects.h#L63) · [`n4m_mixed_effects_backend_available_v1`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/mixed_effects.h#L64) · [`n4m_mixed_effects_fit_v1`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/mixed_effects.h#L96). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):** `from n4m.component_analysis import mixed_effects`

**Signature:** [`mixed_effects(X: ArrayLike, y: ArrayLike, groups: ArrayLike, **options: Any)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/component_analysis.py#L110)

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

### Parameters

| Name | Type | Default |
|---|---|---|
| `X` | `ArrayLike` | `required` |
| `y` | `ArrayLike` | `required` |
| `groups` | `ArrayLike` | `required` |
| `options` | `Any` | `required` |

## Explanations

### Bibliographic source

Bates, Maechler, Bolker and Walker (2015), Fitting Linear Mixed-Effects Models Using lme4, Journal of Statistical Software 67(1): https://doi.org/10.18637/jss.v067.i01. Pinned independent Rust implementation: https://github.com/bbuchsbaum/mixeff-rs/tree/696e5e74623301280142e73cbb6bf7f2650647d7. Development oracle: https://www.statsmodels.org/stable/generated/statsmodels.regression.mixed_linear_model.MixedLM.html.

### Mathematical principle

Fit y=X beta+sum_j Z_j b_j+epsilon with independent zero-mean Gaussian random intercepts b_j and IID residual. ML/REML profile the fixed effects and residual scale while the pinned native TrustBQ optimizer searches nonnegative relative covariance parameters under an explicit evaluation bound. Beta and conditional plug-in covariance preserve the caller's complete X order; variance components preserve group order, residual first. SVD checks fixed rank. A covariance trace Gram in the residual space orthogonal to X rejects structurally confounded variance directions without allocating an n-by-n covariance matrix.

### Appropriate uses

Conditionally adjusted descriptive model/preprocessing parameter contrasts when the host supplies a defensible fixed design, metric scale, dataset and actual evaluation-block grouping. Crossed/nested intercept factors are supported; nested IDs must include parent context. Candidate comparisons require explicit reference coding and are not a universal causal component ranking.

### Limits and validation

Only Gaussian identity-link random-intercept LMMs are exposed. The caller adds any intercept and all contrasts. Rank-deficient, constant/perfectly explained, variance-confounded and nonconverged fits withhold all estimates with typed diagnostics. Fitted zero-variance boundaries remain flagged. Optimizer termination is not model adequacy. Conditional SEs omit variance-parameter uncertainty; no p-values, confidence intervals or Satterthwaite degrees of freedom are supplied. REML log likelihood is restricted and cannot support ordinary fixed-design ML comparisons. Hosts own matching, scale normalization and independence assumptions; repeated folds do not become independent groups merely through identifier encoding. Fixed resource caps are part of the contract.

### Implementation

Optional statically linked MIT mixeff-rs=1.0.0-rc.5 with defaults disabled, no NLopt/R/Python/formula-string runtime. Programmatic AST contains only explicit numeric fixed columns and intercept-only random terms. Stable caller-buffer C ABI preserves feature-off BACKEND_UNAVAILABLE behavior, technical-error atomicity and caught Rust panic/C++ exception boundaries. Real native checks include analytic balanced ML/REML variance/covariance targets, independent statsmodels0.14.6 crossed/nested fixtures, rank and variance-identifiability counterexamples, genuine boundary, budget exhaustion, repeated fits, arbitrary supported strides and output ownership. Actual optimized native/WASM/package size qualification belongs to aggregate gates, not to a successful standalone archive build.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/backends/mixeff/src/lib.rs; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/backends/mixeff/Cargo.lock; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/mixed_effects.h; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_mixed_effects.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/tests/test_mixed_effects.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/parity/fixtures/mixed_effects_reference_v1.json; https://github.com/GBeurier/nirs4all-methods/blob/main/parity/generate_mixed_effects_reference.py; https://github.com/GBeurier/nirs4all-methods/blob/main/docs/MIXED_EFFECTS.md

## Catalog note

Explicit fixed design and crossed or nested group intercepts; pinned static mixeff-rs with no NLopt. ML or REML; rank, identifiability, optimizer and boundary diagnostics; conditional plug-in SE only. Optional compile-time backend, explicit unavailable state.


_See also_: [methods index](index.md).