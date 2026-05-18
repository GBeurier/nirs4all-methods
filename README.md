# chemometrics4all

[![License: CeCILL-2.1](https://img.shields.io/badge/license-CeCILL--2.1-blue.svg)](LICENSE)
[![Status: Phase 0](https://img.shields.io/badge/status-Phase%200%20bootstrap-orange.svg)](ROADMAP.md)

**chemometrics4all** is a portable C++17 library that ships the **NIRS toolchain around PLS**: preprocessings, splitters, filters, augmentations, signal-type detection, transfer metrics, and FCK feature engineering. It is the companion of [pls4all](https://github.com/GBeurier/pls4all) (which provides the PLS regression / classification variants) and gives every binding language the same parity-validated, zero-mandatory-dependency NIRS engine.

> **chemometrics4all does NOT ship any models.** PLS variants live in `pls4all`. This library focuses on the upstream/downstream layer: how you prepare spectra, partition samples, detect outliers, augment training data, convert signal types, and measure transfer-learning alignment.

---

## Scope

| Category | Count | Examples |
|----------|------:|----------|
| Preprocessings | 52 | SNV, MSC, EMSC, SavitzkyGolay, NorrisWilliams, AsLS, AirPLS, ArPLS, SNIP, Wavelet (Haar/db4/sym4/coif1), OSC, EPO, CARS, MCUVE, ToAbsorbance, KubelkaMunk, … |
| Augmentations | 39 | Gaussian/Multiplicative/Spike/Heteroscedastic noise, LinearBaselineDrift, WavelengthShift/Stretch, BandMasking, ChannelDropout, Mixup, ScatterSimulationMSC, BatchEffect, InstrumentalBroadening, DetectorRollOff, Spline_*, … |
| Splitters | 9 | KennardStone, SPXY, SPXYFold, SPXYGFold, KMeans, KBinsStratified, BinnedStratifiedGroupKFold, SystematicCircular, SPlit (data twinning) |
| Filters | 11 methods | YOutlier (iqr/zscore/mad/percentile), XOutlier (mahalanobis/robust_mahalanobis/pca_residual/pca_leverage/isolation_forest/lof), HighLeverage, SpectralQuality, Composite |
| Utilities | — | SignalTypeDetector, TransferMetricsComputer (CKA/Grassmann/RV/Procrustes/trustworthiness/…), Hotelling T², Q-residuals, RPD/RPIQ/SEP/Bias |
| Feature engineering | 1 | FCKStaticTransformer (Fractional Convolutional Kernels) |

All operators are mirrored byte-bit-where-possible against [nirs4all](https://github.com/nirs4all/nirs4all) (Python) plus references from scikit-learn, scipy, prospectr (R), mdatools (R), pywt, and pybaselines.

---

## Architecture (mirrored from pls4all)

- **C++17 core** in `cpp/src/core/` — internal headers + implementations, never directly exposed.
- **Stable C ABI** in `cpp/include/chemometrics4all/c4a.h` — `c4a_*` symbols, opaque handles, error-buffer-per-context.
- **Thin language bindings** (`bindings/`): Python (ctypes + sklearn-compatible classes), R (.Call), MATLAB (MEX), JS/WASM (Emscripten), and tier-1 stubs for Go/Rust/Julia/Ruby/.NET/Lua/Nim/JNI/Octave.
- **Zero mandatory dependencies** — DIY linalg, RNG (PCG64 reimplementation for NumPy bit-exact parity), banded LDLT, FFT-free wavelet filter banks. Optional accelerated backends (BLAS, OpenMP) are gated by CMake presets.
- **CMake ≥ 3.21** with 20+ presets in `CMakePresets.json` (dev-debug, dev-release, ci-*, blas-on, sanitizer-*).
- **Parity-gated** — every operator is validated against pinned reference implementations (numpy 1.26.4, scipy 1.11.4, pywt 1.6.0, pybaselines 1.1.4, prospectr 0.2.7, …) with `parity/fixtures/*.json` + `parity/tolerances.md`.
- **Codex+Opus review workflow** — each phase is reviewed by Codex (pre & post) and Opus (independent post-review); transcripts in `docs/reviews/phase-N/`.

---

## Build (Phase 0 bootstrap)

```bash
cmake --preset dev-release
cmake --build --preset dev-release
ctest --preset dev-release --output-on-failure
./build/dev-release/cpp/cli/chemometrics4all_cli --version
./build/dev-release/cpp/cli/chemometrics4all_cli --selfcheck
```

Requirements: C++17 toolchain (GCC ≥ 11 / Clang ≥ 14 / MSVC 2022), CMake ≥ 3.21, Ninja (recommended).

---

## Phase status

This repository is being built **phase by phase** with full Codex + Opus reviews after each phase. See [ROADMAP.md](ROADMAP.md) for the 27-phase plan.

| Phase | Status | Scope |
|------:|:-------|-------|
| 0 | 🟢 in progress | Bootstrap (clone pls4all template, strip PLS code, smoke build) |
| 1 | ⚪ pending | Common infrastructure (PCG64 RNG, PCA helper, banded solver, wavelet kernels) |
| 2–10 | ⚪ pending | 52 preprocessings across 9 phases |
| 11 | ⚪ pending | 9 splitters |
| 12–14 | ⚪ pending | 5 filter classes / 11 methods |
| 15–18 | ⚪ pending | 39 augmentations across 4 phases |
| 19–21 | ⚪ pending | NIRS utilities, transfer metrics, FCK |
| 22–25 | ⚪ pending | Python / R / MATLAB / JS-WASM bindings |
| 26 | ⚪ pending | Cross-binding benchmarks + GitHub Pages |

---

## License

[CeCILL-2.1](LICENSE) — French OSS license, GPL-compatible, SPDX-listed.

## Citation

See [CITATION.cff](CITATION.cff).
