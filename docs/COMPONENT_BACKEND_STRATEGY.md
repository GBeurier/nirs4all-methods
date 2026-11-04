# Component analysis backend strategy

Decision and implementation assessment, 2026-11-04. Both recommended static backends are implemented and execute through the native C ABI, Python and JS/WASM. Release tooling enables them in full Python wheels and npm artifacts; ordinary source builds keep optional backends disabled by default.

## Recommended combination

Use a pinned, default-feature-free **mixeff-rs static adapter for mixed effects**
and a narrow **rfr C++ static adapter plus independently implemented fANOVA
decomposition**. Ship qualified code in the existing native library and WASM
artifacts. End users should not install Rust, R, compilers, NLopt or an additional
service. Build requirements belong in release tooling. Source builds retain a
dependency-free preset with explicit capability reporting for optional backends.

| Rank | Choice | End-user cost | Implementation and decision |
| --- | --- | --- | --- |
| Mixed 1 | mixeff-rs, no default features, static C ABI adapter | No additional runtime installation in supported prebuilt artifacts; binary increment must be measured | Preferred: reuses REML fitting, optimizer and diagnostics. Restrict the model builder to encoded numeric columns and grouping IDs; do not expose arbitrary formulas. |
| Mixed 2 | Purpose-built C++ Gaussian mixed model | Smallest dependency surface; no Rust in source builds | Fallback if linkage/size/qualification fails. More numerical implementation and maintenance risk; existing one-way variance components cannot substitute for general fixed effects and crossed groups. |
| Mixed 3 | OpenBLUP core, static wrapper | No extra runtime if precompiled; final weight unknown | Matrix API is attractive, but genetics/IO/dependency surface is broader than this analysis needs. Keep as second external candidate. |
| fANOVA 1 | BSD rfr C++ fit/marginals, own published-math decomposition | No additional user runtime; fits the existing C++ toolchain | Preferred: use only the reviewed template/header closure and notices. Create a fresh forest per fit. Implement main and pairwise variance components without copying the separate restricted Python fanova code. |
| fANOVA 2 | New C++ trees and marginal integrator | Potentially smaller payload | More code and qualification. No reusable general regression forest was found in the current methods core; isolation forest is a different algorithm. |
| Both, last | External R/Python analysis process | Additional environment/runtime and process management | References for validation only. lme4 extraction still depends on Rcpp/SEXP; adding a second user environment would undermine the portable engine. |

## What is established and what remains to measure

Both engines execute through optimized native and Emscripten 3.1.74 artifacts.
Python and actual JS/WASM tests fit analytic mixed effects and known fANOVA
main/interaction cases; an installed wheel and installed static CMake consumer
also execute both engines. Rust panic-unwind and C++ use the same native WASM
exception model. Independent statsmodels ML/REML fixtures qualify crossed and
nested group layouts; technical errors preserve outputs and statistical refusals
withhold estimates. Native and Clang WASM compilation pass warnings as errors.

Dead-section elimination and stripping reduce the Linux x86-64 combined wheel
increment to about 0.99 MB and the installed shared library increment to 2.13 MB.
Mixed effects alone adds about 0.93 MB to the wheel and 2.00 MB installed; fANOVA
adds about 64 KB and 139 KB. Full same-source native/WASM measurements and the
reproduction script are `component_analysis_sizes.linux-x86_64.json` and
`scripts/measure_component_artifacts.py`. These are final artifact measurements,
not intermediate archive sizes. Platform CI qualifies other release targets;
Linux measurements do not claim identical cross-platform sizes. Resource caps
bound supported fits; no automatic substitute engine is used.

## Analysis presets and qualification

Start mixed effects with a Gaussian response, ML/REML, fixed model/preprocessing
contrasts and declared interactions, and random intercepts for dataset and
dataset/evaluation-block groups. Preserve grouping dependence; expose rank
deficiency, singular fits, convergence and inference type. Do not infer general
random-slope/correlation support from a one-intercept compile probe.

Start fANOVA with seeded bounded forests, declared numeric/categorical product
domains, and first/second-order components. Importance describes surrogate
variance, not the sign or causal effect of a component. Specify whether results
aggregate per-tree decompositions or decompose the ensemble mean. Require held-out
surrogate diagnostics. Model-specific hyperparameters are analyzed in their
applicable model slice, or in a qualified conditional-domain implementation;
do not integrate impossible model/parameter combinations as a product domain.

Reuse versioned native artifacts, precompute and cache by complete cohort/design/
domain/options identity, and bound resource usage. Routine UI changes should not
repeat training; static sites can consume precomputed results without a server.
Qualify against analytic cases and independent lme4/statsmodels references,
interactions with known variance, all-constant data, confounding and repeated fits.

## Sources and sequencing

- [Pinned mixeff manifest](https://github.com/bbuchsbaum/mixeff-rs/blob/696e5e74623301280142e73cbb6bf7f2650647d7/Cargo.toml): MIT, NLopt is optional; disabling defaults uses native optimizer fallbacks.
- [Pinned rfr license](https://github.com/automl/random_forest_run/blob/1ba17650e734edd3ea536efc91d32815c4874e72/LICENSE.txt), [fANOVA paper](https://proceedings.mlr.press/v32/hutter14.html), [separate Python license metadata](https://github.com/automl/fanova/blob/46eb005320a0f67fc6b34ed74a91f702c2358c3f/setup.py).
- Detailed evidence and reproducible probes: `RANKING_EXTERNAL_DEPENDENCIES.md`.

The native engines, thin bindings, original-observation benchmark orchestration,
shared React views and benchmark controls have been reviewed in separate blocks.
Complete local qualification precedes replacing the superseded PRs with one PR
per project. Never merge or disturb original refactor checkouts.
