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
ASLS baseline correction and Haar wavelet denoising. The supported fitted-model path is now live for NIPALS,
orthogonal-scores, SIMPLS, kernel, wide-kernel, SVD, power-iteration and
randomized-SVD PLS regression
(PLS1 / PLS2), PLSCanonical with NIPALS/SVD, PLSSVD direct cross-covariance
scores, PLS-DA dummy-response scores,
OPLS / OPLS-DA common predictive scores with orthogonal corrections, plus PCR: fit, predict,
transform, fitted-array accessors and binary import/export.

Build matrix: Linux × {gcc-12, gcc-13, clang-16}, macOS × clang
(arm64 + universal2), Windows × {MSVC, MinGW}. ASAN / UBSAN / TSAN green.
ABI symbol gate enforced on every PR.
