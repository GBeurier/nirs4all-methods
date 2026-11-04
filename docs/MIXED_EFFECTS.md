# Optional Gaussian mixed-effects engine

The `N4M_WITH_MIXEFF` feature statically links the pinned MIT-licensed
`mixeff-rs = 1.0.0-rc.5` engine with default features disabled. The published crate
records VCS revision `696e5e74623301280142e73cbb6bf7f2650647d7`; Cargo.lock pins
all transitive sources and checksums. No R/Python process, NLopt library, formula
string parser, CSV reader or runtime plugin is used by the adapter. The ordinary
C++17 build leaves this optional feature off and retains the same C ABI with an
explicit backend-unavailable result.

The public declaration is `cpp/include/n4m/mixed_effects.h`; Python and JS use the
shared `n4m.component_analysis` / component-analysis binding surfaces. The kernel
fits the Gaussian model

`y = X beta + sum_j Z_j b_j + epsilon`,

with independent `b_j ~ N(0, tau_j^2 I)` and IID
`epsilon ~ N(0, sigma^2 I)`. Each group column supplies one random-intercept
factor. Crossed factors use separate columns. For nested factors, the caller
must encode the complete parent context in each nested ID; IDs are independent
between columns. The caller also supplies the fixed intercept explicitly, if
wanted. Fixed columns, coefficients, covariance axes and random-variance columns
preserve caller order even when the upstream engine pivots/reorders internally.

## Inputs, limits and outcomes

The C ABI accepts finite F64 strided `X` (n by p), F64 strided `y` (n by 1),
and nonnegative I64 strided group IDs (n by k). It adds no contrasts, centering,
normalization or implicit fixed intercept. The adapter builds an internal AST
from fixed numeric columns and intercept-only random terms; users cannot supply
arbitrary formulas. ML and REML are selectable. Tolerances and a bounded maximum
number of optimizer evaluations are explicit options. The optimizer is native
TrustBQ. The bound concerns optimizer objective evaluations, not rank checks or
matrix construction; no wall-clock cancellation interface is claimed.

Resource limits are n <= 8192, p <= 64, 1 <= k <= 8, n*p <= 262144, summed distinct
group levels <= 1024, and n*sum(levels) <= 4194304. These bounds limit the engine's
blocked dense workspace. Each random term requires at least two levels.
Different memory/runtime limits may be appropriate for different scientific
applications; exceeding the fixed supported envelope returns unsupported.

Technical failures return `N4M_*` errors and do not modify any output, including
the summary. This includes invalid shapes/dtypes/strides, nonfinite values,
negative group IDs, options, overlapping output buffers, incompatible structure
sizes, resource limits, C++ exceptions or caught Rust unwind panics. There is no
cross-language allocation ownership. Rust allocator process aborts cannot be
converted into return codes; finite resource caps reduce the allocation envelope
rather than promising OS out-of-memory recovery.

A technically valid statistical request returns `N4M_OK` and a typed `fit_state`:

| State | Estimates | Meaning |
| --- | --- | --- |
| `FITTED` | supplied | Optimizer reported a genuine convergence termination. |
| `FITTED_BOUNDARY` | supplied, explicitly flagged | A random/residual variance ratio is at or below the declared boundary tolerance. |
| `RANK_DEFICIENT` | all NaN / binding `None` | The requested fixed columns cannot all be estimated; no silent term dropping. |
| `UNIDENTIFIABLE` | all withheld | Constant/perfectly explained response, no residual degrees of freedom, insufficient group levels or confounded covariance directions. |
| `NONCONVERGED` | all withheld | Budget exhaustion, roundoff-limited termination or optimizer failure. |
| `NUMERICAL_REFUSAL` | all withheld | The engine could not produce a finite supported result. |

The summary records the detailed reason, rank, unscaled X condition number,
residual degrees of freedom, group-level count, optimizer termination and
reported evaluation count. Convergence is a termination classification, not a
certificate of model adequacy or a global likelihood optimum. Statistical
refusals withhold **all** estimate buffers atomically, rather than retaining a
best-so-far coefficient as a valid result.

The adapter checks fixed rank with a scaled SVD and also checks the rank the
upstream design actually retains. A separate covariance trace Gram checks linear
independence of residual and group covariance directions after projecting out X.
It detects relabeled duplicate groups, a singleton-per-observation term confounded
with residual noise, and groups fully absorbed by fixed effects, without building
an n by n matrix. This is a conservative structural residual-space check even
for ML; it is not a finite-sample power test or post-fit variance-parameter
information matrix.

## Interpretation of returned estimates

`beta`, row-major `covariance`, `standard_errors` and `variances` are caller-owned
buffers. Variances contain the residual variance followed by random-intercept
variances in original group-column order. Coefficient covariance is the conditional
plug-in GLS covariance at fitted variance components; its square-root diagonal
omits variance-parameter uncertainty. No p-values, confidence intervals,
Satterthwaite degrees of freedom, contrasts, adjusted candidate ranking, random
slopes, generalized outcome families or correlated random-effect terms are
claimed by this limited API.

`objective` is the chosen ML deviance or REML criterion. `log_likelihood` is its
negative half, so it is a **restricted** log likelihood for REML. Do not compare
REML criteria across different fixed designs as ordinary ML likelihoods.
Boundary estimates require scientific scrutiny; a fitted zero variance is not
proof that a grouping factor is irrelevant.

Hosts own qualified observation matching, design/reference coding, outcome scale,
grouping, dependence and provenance. Raw errors on incompatible dataset scales
must not be pooled without a declared, defensible normalization. The companion
native loss-normalization helper performs arithmetic only; it does not choose or
justify a baseline. A descriptive adjusted coefficient is conditional on this
observed design and does not establish a causal preprocessing contribution.

## Numerical evidence and reproducibility

`cargo test --locked --manifest-path cpp/backends/mixeff/Cargo.toml` executes real
fits. Tests include balanced random-intercept closed forms for ML/REML, distinct
variance ordering for crossed/nested groups, repeated fits, fixed-rank and
covariance confounding, actual zero-variance boundaries, saturated fixed designs,
constant response, a one-evaluation budget, wire sizes and a controlled Rust
panic proving caller buffers stay unchanged. `cpp/tests/test_mixed_effects.cpp`
adds C ABI tests for strided/column-major inputs, explicit intercept order,
dtype/shape/stride overflow, output aliasing, validation atomicity, feature-off
behavior and statistical withholding.

The synthetic 20-row balanced reference has five groups with four observations,
group offsets -2 through 2, and within-group deviations -0.3, -0.1, 0.1, 0.3.
With a fixed intercept its REML residual variance is 1/15, random-intercept
variance 149/60, beta 2 and cov(beta) 0.5. ML gives random-intercept variance
119/60 and cov(beta) 0.4. These are analytic sufficient-statistic checks,
independent of another optimizer's result.

Four additional deterministic 96-row fixtures in
`parity/fixtures/mixed_effects_reference_v1.json` come from statsmodels 0.14.6
`MixedLM`/`VCSpec`, with crossed and nested intercept factors, ML/REML and three
fixed columns including an intercept in the second position. The local generator
records its NumPy version. Conditional covariance is independently computed as
`inverse(X.T solve(V, X))`, matching this API's target rather than the full
observed-Hessian parameter coupling in statsmodels' generic covariance output.
The actual native adapter agreed within beta atol 1e-5/rtol 1e-4, variance/covariance
atol 3e-5/rtol 2e-3, and log-likelihood absolute tolerance 1e-4. These tolerances
reflect finite optimizer agreement, not bit-exact identity.

Regenerate with a development-only statsmodels environment:

```bash
python parity/generate_mixed_effects_reference.py --check-native
cargo test --locked --manifest-path cpp/backends/mixeff/Cargo.toml
cargo clippy --locked --manifest-path cpp/backends/mixeff/Cargo.toml --all-targets --no-deps -- -D warnings
```

No scientific package is added to normal runtime dependencies. Relevant primary
sources are the [pinned engine manifest](https://github.com/bbuchsbaum/mixeff-rs/blob/696e5e74623301280142e73cbb6bf7f2650647d7/Cargo.toml),
[statsmodels MixedLM](https://www.statsmodels.org/stable/generated/statsmodels.regression.mixed_linear_model.MixedLM.html)
and the [lme4 model description](https://lme4.github.io/lme4/articles/lmer.pdf).

## Build, notices and size accounting

Root CMake presets own actual native/WASM linkage and packaging. Cargo's release
profile uses optimization level 3, ThinLTO, one codegen unit and unwind panics;
the adapter's exported bridge catches panics before returning to C++. The bridge
is hidden from the public C export surface. Compilation of a Rust archive alone
is not evidence of a working JS/WASM binding or installed artifact size.

`DEPENDENCIES.json` inventories the locked normal/build closure for native Linux
(70 registry crates) and Emscripten (69). Default NLopt, PRIMA and optional Rayon/
faer backends are absent. `generate_notices.py` reproduces the inventory and
`THIRD_PARTY_NOTICES.txt` from the Cargo cache and includes source notices even
when metadata uses `license-file` rather than an SPDX field. The pinned engine's
MIT notice is also retained separately. `RUST_STANDARD_LIBRARY_NOTICES.html`
contains the compiler-distributed standard library's detailed third-party notice
bundle; the locally selected compiler identity is recorded. Retain both registry
and standard-runtime notices when distributing enabled binaries. A different
compiler requires regenerating the compiler-specific notice inventory.

The notices are about 1.14 MB as raw text/HTML before compression. The actual
optimized binary and final compressed/installed increments must be measured from
equivalent complete presets, including notices and package assets; debug archive
size is not an installed-size estimate. Aggregate release/size results are owned
by the combined backend qualification rather than inferred from these unit tests.
