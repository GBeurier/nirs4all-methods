# Pipeline component analysis: external engine feasibility

Historical assessment preceding the approved optional implementations. The earlier
delivery decision below is superseded by [Mixed effects](MIXED_EFFECTS.md) and
[fANOVA](FANOVA.md); current availability is reported by the compiled capabilities.

Assessed on 4 November 2026. This document concerns optional compiled backends for
pipeline component effects, alongside the dependency-free ranking algorithms.
It does not change their API or implementation.

## Historical probe-stage delivery decision and host-visible status

**Historical probe-stage decision, superseded by the implementations linked above:** no external statistical backend was shipped in that earlier batch. The implemented
work here is reproducible source inspection and disposable compilation/linkage
probes. A successful probe establishes technical feasibility for that exact
configuration; it does not establish scientific parity, ABI qualification or
cross-platform support.

| Proposed capability | Status | Stable reason for host display | Preferred next candidate |
| --- | --- | --- | --- |
| Mixed-effects component estimates | `deferred` | `mixed_effects_backend_not_qualified`: native static subset feasible; matrix contract, inference and supported-target qualification pending | MIT `mixeff-rs` without NLopt; compare raw-matrix OpenBLUP before choosing |
| Surrogate functional ANOVA | `deferred` | `fanova_backend_not_qualified`: C++ static marginals feasible; independent decomposition, parameter-domain semantics and parity pending | BSD C++ `rfr` headers; exclude restricted Python `fanova` |

These values are proposed capability metadata, not new executable return fields.
Hosts should omit controls or display the status/reason until the corresponding
native capability is actually registered. They must not advertise these engines
as available or silently substitute another ranking algorithm.

The dependency-free ranking implementation in the companion block covers Elo,
Bradley–Terry, TrueSkill 1, OpenSkill, mean-rank/Friedman, relative-gain and
meta-ranking independently of this decision. A component ranking assembled from run evidence is distinct from
a mixed-model coefficient or a surrogate variance fraction.

## Minimal application configuration considered first

The proposed mixed model consumes a finite oriented response and explicit fixed
columns for model, preprocessing and selected parameters, plus integer grouping
IDs for dataset and repeated evaluation blocks. Start with Gaussian identity-link
LMM, ML/REML selectable, random intercepts, finite iteration limits and explicit
convergence/boundary diagnostics. No free-text formulas, custom families, R
runtime, model selection, bootstrap engine, plots or file readers are required.
Separate crossed grouping blocks are needed when multiple pipelines share the
same dataset/split; folds are not independent experimental replications.

The fANOVA configuration uses `double`, one fixed RSS split/tree family, bounded
numbers of trees/depth/features, explicit seed, explicit numeric bounds and
categorical levels. The initial safe domain is an independent product of declared
parameter domains, with inactive/conditional parameters rejected. Split IDs are
blocking variables, not ordered continuous hyperparameters. The response must
use a declared per-task normalization before pooling incompatible metric scales.
No ConfigSpace, Python/SWIG, plotting or runtime plugins are needed.

These restrictions remove wrapper work, but do not remove confounding: component
coefficients require a connected, identifiable design; fANOVA fractions explain
the fitted surrogate under its declared integration measure. Neither identifies
a causal preprocessing effect from an adaptive search log. Report design support,
fit diagnostics and held-out surrogate quality. The original
[Hutter–Hoos–Leyton-Brown paper](https://proceedings.mlr.press/v32/hutter14.html)
describes forest marginals and variance decomposition for individual parameters
and interactions.

## Pinned candidates and source evidence

Versions below are the sources actually inspected, not version numbers inferred
from search-result snippets. Sources were downloaded into `/tmp`; no upstream
source is vendored into the engine by this work.

| Candidate | Immutable source pin | Minimal-subset assessment and decision |
| --- | --- | --- |
| lme4 | `48e197dca2fc2e74870010fcfedea55ffc9dace9`, DESCRIPTION `2.1-1` | Reference-only for this batch. Matrix inputs do not eliminate R coupling. |
| mixeff-rs | crates.io `1.0.0-rc.5`, VCS `696e5e74623301280142e73cbb6bf7f2650647d7` | Native static probe succeeds with `default-features=false`; optional integration deferred. |
| OpenBLUP | `0dd1f37ffe7f7caf9fda51ecdd4a24eeba7b2ce0`, workspace `0.1.0` | Raw sparse matrices and pure-Rust core are plausible; optional integration deferred after source/dependency assessment, no fit/build claimed. |
| glmm | crates.io `0.8.0`, VCS `f7bd8b22fcaf5ab01c9d7cb6117ad5a66f26e7d7` | Formula-free matrix kernel exists; static redistribution obligations and additional Rust target qualification make it a lower-priority candidate. No fit/build claimed. |
| rfr / pyrfr | `1ba17650e734edd3ea536efc91d32815c4874e72` | C++17 fixed-template static fit/marginal/variance probe succeeds; full fANOVA backend deferred. |
| Python fanova | `46eb005320a0f67fc6b34ed74a91f702c2358c3f`, `2.0.20.dev` | Excluded from embedding because published metadata limits free usage and requests a commercial license. |

### lme4: explicit matrices still cross the R boundary

The pinned [DESCRIPTION](https://github.com/lme4/lme4/blob/48e197dca2fc2e74870010fcfedea55ffc9dace9/DESCRIPTION)
declares GPL (>= 2), R/Matrix dependencies and Rcpp/RcppEigen linkage. Its
[predictor header](https://github.com/lme4/lme4/blob/48e197dca2fc2e74870010fcfedea55ffc9dace9/src/predModule.h)
includes `RcppEigen.h`; the `merPredD` constructor takes R `SEXP` objects and
conditional covariance returns Rcpp objects. The Gaussian response path also uses
Rcpp vector types. Compiling just the predictor include without installing an R
stack fails at `RcppEigen.h` in the recorded probe.

Inference: fixing X/Z and removing the formula layer would still require an
Rcpp/SEXP/CHOLMOD-boundary extraction, replacement object ownership, an optimizer
entry point and review of copied GPL code. This is a numerical fork rather than
a small static-library switch. Use externally generated R fixtures as a reference;
do not add R as a runtime dependency of every binding. GPL permits commercial use
under its terms; it is not a non-commercial license. No alternative commercial
license was established by this assessment.

### mixeff-rs: no-NLopt native subset is real, matrix-only API remains work

The inspected [manifest](https://github.com/bbuchsbaum/mixeff-rs/blob/696e5e74623301280142e73cbb6bf7f2650647d7/Cargo.toml)
uses MIT and Rust 1.85. Default builds enable NLopt; disabling defaults selects
native TrustBQ/fallback optimizers and excludes optional NLopt, PRIMA, Rayon and
faer backends. Core dependencies still include nalgebra, nalgebra-sparse, serde,
CSV/TOML and inference utilities. There is no feature that strips all data/compiler
modules. The dependency-free n4m preset would therefore leave this feature off.

The probe creates a programmatic `Formula` AST and a numeric/categorical frame,
fits 48 synthetic observations with one random intercept, exports one C function
and links a Rust `staticlib` into a C++ executable. It never parses a formula
string and needs no R/Python runtime. It is **not** a raw X/Z adapter: the stable
constructor still compiles the AST into its internal design. Translating explicit
pre-encoded columns and groups into this restricted AST is feasible, but needs a
contract for intercepts, rank deficiency, categorical contrasts and result labels.

The native probe resolves 73 normal/build packages including its own wrapper
(and target-conditional packages; not all necessarily compile). License metadata
is captured in `mixeff-license-summary.json`. `cobyla` uses `license-file` instead
of an SPDX field; its inspected `LICENSE.md` is MIT text. This inventory is not a
completed binary notice audit. No commercial license is needed for MIT upstream
code if its notice obligations are met; transitive obligations still apply.

Inference: this is a credible optional static backend and reduces optimizer and
inference implementation work. The remaining work is qualification, not an
assumption that Rust is inherently too large. The native compile/link probe does
not qualify repeated-fit precision, inference reliability, concurrent contexts,
Windows CRTs, macOS or Emscripten integration.

### OpenBLUP: matrix door exists, default-free does not mean a small kernel

Pinned [workspace dependencies](https://github.com/jakobrichert/openblup/blob/0dd1f37ffe7f7caf9fda51ecdd4a24eeba7b2ce0/Cargo.toml)
and [core manifest](https://github.com/jakobrichert/openblup/blob/0dd1f37ffe7f7caf9fda51ecdd4a24eeba7b2ce0/crates/core/Cargo.toml)
show MIT OR Apache-2.0, nalgebra/faer/sprs, CSV/serde, Rayon and other core
utilities. Core defaults are already empty. Selecting only `plant-breeding-lmm-core`
excludes Python, CLI and wasm-bindgen wrappers, but does not feature-prune genetics,
diagnostics, model-building or IO modules.

The public `MixedModel` contains response, sparse X and Z blocks, variance
structures and iteration controls; constructing it directly can avoid the formula
builder. Start with IID intercept blocks and residuals, not breeding-specific
pedigrees or spatial structures. The standalone small WASM crate has its own
implementation and does not depend on the core outside dev dependencies; its
existence alone is not evidence that the complete core adapter is qualified.
The separate full-engine WASM wrapper does depend on the core and adds browser
`getrandom` configuration. These are promising upstream portability signals,
not results from a local WASM fit.

Inference: a core-only Rust staticlib wrapper could save sparse REML development,
but cannot be advertised as a ready minimal C++ backend. The metadata-only joint consumer inventory reaches 175 normal/build packages
from OpenBLUP (including target-conditional packages and joint feature union).
No compilation, fit, scientific parity or target pass is claimed for OpenBLUP in
this batch.

### glmm: suitable matrix kernel, distinct redistribution trade-off

The pinned [manifest](https://github.com/pawlenartowicz/glmm/blob/f7bd8b22fcaf5ab01c9d7cb6117ad5a66f26e7d7/Cargo.toml)
has a semver-covered matrix interface and allows `default-features=false` to
exclude its regex-based formula frontend. It uses Rust 1.85, faer, bobyqa, pulp,
commonstats and libm; parallelism can remain off. Its wasm-specific faer feature
selection is relevant evidence that a formula-free portable kernel was considered.

The [LGPL-3.0-or-later license](https://github.com/pawlenartowicz/glmm/blob/f7bd8b22fcaf5ab01c9d7cb6117ad5a66f26e7d7/LICENSE)
permits commercial use with conditions. In particular, a combined statically linked
artifact needs an appropriate source/relinking and notice distribution plan;
calling it through a C ABI does not remove those conditions. No permissive or
commercial dual-license offer was established in the inspected README/manifest.
A commercial alternative must not be assumed available. The metadata-only joint
consumer inventory reaches 145 normal/build packages from glmm; this is not a
linked binary count. Defer integration while
permissive candidates are qualified; do not label LGPL code technically unusable.

### rfr: narrow static C++ integration is feasible

The pinned [license](https://github.com/automl/random_forest_run/blob/1ba17650e734edd3ea536efc91d32815c4874e72/LICENSE.txt)
is BSD-3-Clause for AutoML files, with separate notices for bundled dependencies.
The tested header closure includes 15 rfr headers and 48 cereal/embedded headers:
63 upstream headers, 915,226 source bytes. Whole include trees measure roughly
300 KiB rfr and 1.5 MiB cereal on this checkout. Cereal is BSD; embedded RapidJSON
has its own notices, including components not necessarily reached by this probe.
Package only the reviewed closure and required notices, rather than treating the
repository-wide top-level license as the sole obligation.

The probe fixes `double`, `unsigned`, `std::mt19937`, RSS splits and the
`fANOVA_forest` template. It compiles without Python, SWIG, Boost or BLAS, archives
one object and links it into a C++ executable. Four balanced binary rows with
response equal to feature 0 yield prediction 1, marginalized prediction 1 and
integrated variance 0.25. It explicitly calls `precompute_marginals`.
An initial build exposed a missing upstream `<cassert>` include; including it in
the adapter translation unit resolves the failure without editing upstream.

The pinned [forest source](https://github.com/automl/random_forest_run/blob/1ba17650e734edd3ea536efc91d32815c4874e72/include/rfr/forests/fanova_forest.hpp)
appends domain metadata in `fit()` without clearing it. A production adapter
should construct a fresh forest per fit and prohibit refitting that object, or
carry a tested upstream fix. This observation is source inspection, not a
claimed production reproducer. All C++ exceptions must remain inside the adapter.

The [Python fanova metadata](https://github.com/automl/fanova/blob/46eb005320a0f67fc6b34ed74a91f702c2358c3f/setup.py)
limits free use to academic/non-commercial usage and directs commercial users to
the author. Do not copy its decomposition code assuming the BSD rfr license
extends to that separate project. Implement decomposition from the published
mathematics, independently, or obtain the required rights.

Inference: rfr is the shortest candidate for an optional compiled backend. It
saves tree fitting and marginal integration work. Before shipping, independent
variance assembly, declared integration measures, interaction accounting,
conditional-domain rejection, constant-response handling, holdout checks,
resource limits and Emscripten builds still require tests. The four-row probe
proves none of those general properties.

## Local probe results and reproducibility

All builds used disposable `/tmp/n4m-ranking-external`; production CMake presets,
ABI snapshots and package dependencies were not changed. Toolchain: Ubuntu GCC
11.4, Rust 1.96.0. Probe build commands are intentionally standalone and do not
replace the repository's prescribed CMake-preset release workflow.

| Probe | Observed result |
| --- | --- |
| rfr C++17 compile | Pass after adding `<cassert>` to probe; 3.84 s wall, 277,476 KiB peak RSS |
| rfr static archive/link/run | Pass; archive 350,702 bytes, executable 170,608 bytes; prediction/marginal 1, variance 0.25 |
| rfr executable runtime linkage | libstdc++, libm, libgcc_s, libc only |
| lme4 predictor include without R stack | Expected failure: `RcppEigen.h` unavailable; API inspection separately establishes SEXP coupling |
| mixeff Rust staticlib compile/link/run | Pass; slope 0.75; one-job build, debug info disabled: 124.09 s wall, 1,044,932 KiB peak RSS; archive 81,899,562 bytes, executable 29,900,672 bytes |
| mixeff Emscripten-target Rust archive compile | Pass; archive 46,516,462 bytes; 165.34 s wall, 1,169,936 KiB peak RSS; no C++/JS link or WASM execution performed |

`docs/ranking_external_probes/` preserves original probe sources, Cargo lockfile,
pinned downloader, reached-header list and dependency-license inventory. Copy
that directory to `/tmp` before running: never download archives or build outputs
into the source tree. See its `README.md` for exact commands and additional
observations. Archive sizes are unstripped probe artifacts, not production size
estimates. Native C++ symbol/exception containment, Linux version scripts,
macOS exports and Windows symbol snapshots remain qualification requirements.
No probe symbol is added to `libn4m`.
