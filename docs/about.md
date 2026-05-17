# pls4all documentation

A portable PLS / NIRS engine in C++17 with a stable C ABI and thin bindings.

→ **<a href="./">Interactive cross-binding benchmark dashboard</a>** — landing page generated from the canonical benchmark registry, filterable / sortable / column-toggleable.

## Latest highlights (May 2026)

- **R binding COMPLETE** — `pls4all_method()` dispatcher covers all 33
  MethodResult fits + 24 selectors + 4 diagnostics ; 16 formula+S3
  tier-2 wrappers ; parsnip + mlr3 meta-models that dispatch to all
  16 algorithms via the `algorithm` arg.
- **MATLAB / Octave binding COMPLETE** — single dispatcher MEX exposes
  the same surface ; 18 idiomatic `classdef` tier-2 wrappers + unified
  `pls4all.fit(algo, X, y, …)` factory.
- **Cross-binding benchmark** — generated from the canonical
  `benchmarks.parity_timing.registry` method catalog, with registry-driven
  external reference columns in addition to the language bindings.
  → [`benchmarks/cross_binding.md`](benchmarks/cross_binding.md) ·
  [`methodology`](benchmarks/methodology.md)

## Quick links

- **Architecture** — [overview](architecture/overview.md) · [memory model](architecture/memory_model.md) · [error model](architecture/error_model.md) · [threading](architecture/threading.md) · [serialization](architecture/serialization.md)
- **ABI** — [reference](abi/reference.md) · [stability policy](abi/stability_policy.md) · [changes log](abi/changes_log.md)
- **Bindings** — [Python](bindings/python.md) · [R](bindings/r.md) · [MATLAB](bindings/matlab.md) · [JavaScript / WebAssembly](bindings/js.md) · [Android](bindings/android.md)
- **Parity** — [methodology](parity/methodology.md) · {doc}`tolerances <parity/tolerances>`
- **Benchmarks** — [index](benchmarks/index.md) · [cross-binding parity + timing](benchmarks/cross_binding.md) · [methodology](benchmarks/methodology.md)
- **Development** — [workflow](dev/workflow.md) · [build](dev/build.md) · [testing](dev/testing.md) · [style](dev/style.md) · [release process](dev/release_process.md)
- **Reviews** — Codex transcripts under `docs/reviews/`
- **Roadmap** — `ROADMAP.md`, per-phase plans under `roadmap/`

```{toctree}
:hidden:
:maxdepth: 2

abi/reference
abi/stability_policy
abi/changes_log
architecture/overview
architecture/memory_model
architecture/error_model
architecture/threading
architecture/serialization
benchmarks/index
benchmarks/cross_binding
benchmarks/cross_binding_threads
benchmarks/methodology
bindings/python
bindings/r
bindings/matlab
bindings/js
bindings/android
parity/methodology
parity/tolerances
dev/workflow
dev/build
dev/testing
dev/style
dev/release_process
```

## Current Status

The C ABI surface is feature-complete (96 `p4a_*` symbols). Lifecycle for
context / config / matrix-view / operator-bank / gating-strategy / pipeline
is implemented, and pipeline fit/transform is live for identity, center,
autoscale, Pareto scaling, SNV, MSC, EMSC, polynomial detrending and
Savitzky-Golay smoothing/derivatives, Norris-Williams gap-segment derivatives,
ASLS baseline correction, Haar wavelet denoising and supervised one-component
OSC / EPO. Internal regression metric kernels cover RMSE, MAE, bias, R2/Q2,
observed-vs-predicted slope/intercept, RPD and RPIQ, validation splitters cover
deterministic k-fold, leave-one-out, holdout, external-fold, repeated k-fold,
Monte-Carlo, Kennard-Stone and SPXY plans, and the internal CV engine refits
fold-local regression models to produce out-of-sample predictions and aggregate
metrics. Binary classification metrics cover sensitivity,
specificity, balanced accuracy, precision/F1, MCC and average-rank AUC;
multiclass extensions cover macro/micro averages with one-vs-rest AUC, and
binary calibration curves use fixed probability bins. Variable-importance
kernels compute VIP scores and selectivity ratio from fitted models with stored
scores, and Phase 5a variable-selection rankers expose deterministic top-k
ordering for VIP, original-scale coefficient magnitude and selectivity ratio.
Phase 5b interval-selection kernels scan contiguous feature windows with
deterministic k-fold PLS CV for moving-window / iPLS-style selection.
Phase 5p biPLS kernels perform deterministic backward interval elimination by CV.
Phase 5q siPLS kernels exhaustively score fixed-size interval combinations by CV.
Phase 5c stability-selection kernels compute Monte-Carlo coefficient mean/std
ratios for MCUVE-style feature ranking, Phase 5d UVE kernels threshold
real-feature stability against deterministic artificial variables, and Phase 5n
EMCUVE kernels ensemble MC-UVE members with a deterministic vote rule. Phase 5o
randomization-test kernels compare observed PLS coefficient scores against
deterministic Y permutations with empirical p-values. Phase 5e
SPA kernels seed from PLS coefficient magnitude and add projection-diverse
variables, and Phase 5f CARS kernels run deterministic exponential-retention
subset elimination with k-fold CV scoring. Phase 5g Random Frog kernels run
deterministic subset walks and rank variables by inclusion frequency. Phase 5h
SCARS kernels run deterministic calibration subsampling with stability-weighted
CARS retention. Phase 5i GA-PLS kernels use deterministic population search
with elitism, crossover, mutation and k-fold CV fitness.
Phase 5j shaving kernels recursively eliminate low-score PLS variables while
scoring each retained subset by deterministic k-fold CV.
Phase 5s REP kernels remove a fixed number of weak coefficient-score variables
per recursive step and keep the lowest-CV-error retained subset.
Phase 5t IPW kernels iteratively reweight coefficient scores, expose score and
weight paths, and keep the lowest-CV-error top-k subset.
Phase 5u ST-PLS kernels apply deterministic score thresholds with min-selected
fallbacks and keep the lowest-CV-error threshold subset.
Phase 5k BVE kernels greedily evaluate every one-variable removal by
deterministic k-fold CV RMSE and keep the best backward path/subset.
Phase 5l T2-PLS kernels compute Hotelling T2 on PLS loading weights, apply
alpha-specific UCL thresholds with top-k fallback, and score subsets by k-fold CV.
Phase 5m WVC-PLS kernels compute normalized weighted-variable-contribution
scores from SVD PLS components and expose deterministic top-k selection.
Phase 5r WVC-threshold kernels apply fixed-threshold and factor-of-mean rules
with a minimum-selected fallback.
Component-coefficient kernels expose the
original-scale regression coefficients for each latent prefix. SIMPLS component
prefixes can also be scored by deterministic k-fold CV for component-count
selection, and the internal PLS-LDA and PLS-logistic kernels fit classifier
scores on PLS score spaces. The internal MB-PLS kernel fits block-autoscaled,
block-weighted PLS models and maps coefficients back to original feature space.
The internal LW-PLS kernel performs stable k-nearest-neighbor local-window
refits and records the neighbor plan for every prediction.
Phase 6a adds an internal AOM preprocessing-bank primitive with soft equal
weights and hard first-operator gating; full AOM-PLS parity is explicitly
anchored on `nirs4all/bench/AOM_v0/aompls`.
Phase 6b adds internal global AOM-SIMPLS CV selection against that bench oracle
for the identity/detrend strict-linear tranche.
Phase 6c adds bench-parity strict-linear zero-padded Savitzky-Golay,
finite-difference and Norris-Williams AOM operators and runs them through the
global selector.
Phase 6d extends the strict-linear AOM tranche to Whittaker smoothing and FCK
operators, with direct transform parity and global AOM-SIMPLS selection parity
against the bench oracle.
Phase 6e adds internal POP-PLS per-component SIMPLS covariance selection,
including bench-compatible CV scoring semantics, selected operator sequences
and full-fit prediction parity.
The supported fitted-model path is now live for NIPALS,
orthogonal-scores, SIMPLS, kernel, wide-kernel, SVD, power-iteration and
randomized-SVD PLS regression
(PLS1 / PLS2), PLSCanonical with NIPALS/SVD, PLSSVD direct cross-covariance
scores, PLS-DA dummy-response scores,
OPLS / OPLS-DA common predictive scores with orthogonal corrections, plus PCR: fit, predict,
transform, fitted-array accessors and binary import/export.

Build matrix: Linux × {gcc-12, gcc-13, clang-16}, macOS × clang
(arm64 + universal2), Windows × {MSVC, MinGW}. ASAN / UBSAN / TSAN green.
ABI symbol gate enforced on every PR.
