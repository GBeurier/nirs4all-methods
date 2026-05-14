# pls4all documentation

A portable PLS / NIRS engine in C++17 with a stable C ABI and thin bindings.

## Quick links

- **Architecture** — [overview](architecture/overview.md) · [memory model](architecture/memory_model.md) · [error model](architecture/error_model.md) · [threading](architecture/threading.md) · [serialization](architecture/serialization.md)
- **ABI** — [reference](abi/reference.md) · [stability policy](abi/stability_policy.md) · [changes log](abi/changes_log.md)
- **Bindings** — [Python](bindings/python.md) · [R](bindings/r.md) · [MATLAB](bindings/matlab.md) · [JavaScript / WebAssembly](bindings/js.md) · [Android](bindings/android.md)
- **Parity** — [methodology](parity/methodology.md) · [tolerances](parity/tolerances.md)
- **Development** — [workflow](dev/workflow.md) · [build](dev/build.md) · [testing](dev/testing.md) · [style](dev/style.md) · [release process](dev/release_process.md)
- **Reviews** — Codex transcripts under [`reviews/`](reviews/)
- **Roadmap** — [`ROADMAP.md`](../ROADMAP.md), per-phase plans under [`roadmap/`](../roadmap/)

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
Phase 5k BVE kernels greedily evaluate every one-variable removal by
deterministic k-fold CV RMSE and keep the best backward path/subset.
Phase 5l T2-PLS kernels compute Hotelling T2 on PLS loading weights, apply
alpha-specific UCL thresholds with top-k fallback, and score subsets by k-fold CV.
Phase 5m WVC-PLS kernels compute normalized weighted-variable-contribution
scores from SVD PLS components and expose deterministic top-k selection.
Component-coefficient kernels expose the
original-scale regression coefficients for each latent prefix. SIMPLS component
prefixes can also be scored by deterministic k-fold CV for component-count
selection, and the internal PLS-LDA and PLS-logistic kernels fit classifier
scores on PLS score spaces. The internal MB-PLS kernel fits block-autoscaled,
block-weighted PLS models and maps coefficients back to original feature space.
The internal LW-PLS kernel performs stable k-nearest-neighbor local-window
refits and records the neighbor plan for every prediction.
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
