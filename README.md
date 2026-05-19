# chemometrics4all

[![License: CeCILL-2.1](https://img.shields.io/badge/license-CeCILL--2.1-blue.svg)](LICENSE)
[![Status: Stabilization](https://img.shields.io/badge/status-stabilization-blue.svg)](ROADMAP.md)

**chemometrics4all** is a portable C++17 library for chemometrics / NIRS
operators: preprocessings, splitters, filters, augmentations, signal-type
detection, transfer metrics, diagnostics, and FCK feature engineering. It is the
companion of [pls4all](https://github.com/GBeurier/pls4all), which owns the PLS
regression / classification model family.

> **chemometrics4all does NOT ship any models.** PLS variants live in `pls4all`. This library focuses on the upstream/downstream layer: how you prepare spectra, partition samples, detect outliers, augment training data, convert signal types, and measure transfer-learning alignment.

---

## Scope

| Category | Count | Examples |
|----------|------:|----------|
| Preprocessings | 52 | SNV, MSC, EMSC, SavitzkyGolay, NorrisWilliams, AsLS, AirPLS, ArPLS, SNIP, Wavelet (Haar/db4/sym4/coif1), OSC, EPO, ToAbsorbance, KubelkaMunk, … |
| Augmentations | 39 | Gaussian/Multiplicative/Spike/Heteroscedastic noise, LinearBaselineDrift, WavelengthShift/Stretch, BandMasking, ChannelDropout, Mixup, ScatterSimulationMSC, BatchEffect, InstrumentalBroadening, DetectorRollOff, Spline_*, … |
| Splitters | 9 | KennardStone, SPXY, SPXYFold, SPXYGFold, KMeans, KBinsStratified, BinnedStratifiedGroupKFold, SystematicCircular, SPlit (data twinning) |
| Filters | 11 methods | YOutlier (iqr/zscore/mad/percentile), XOutlier (mahalanobis/robust_mahalanobis/pca_residual/pca_leverage/isolation_forest/lof), HighLeverage, SpectralQuality, Composite |
| Utilities | — | SignalTypeDetector, TransferMetricsComputer (CKA/Grassmann/RV/Procrustes/trustworthiness/…), Hotelling T², Q-residuals, RPD/RPIQ/SEP/Bias |
| Feature engineering | 1 | FCKStaticTransformer (Fractional Convolutional Kernels) |

All operators are mirrored byte-bit-where-possible against [nirs4all](https://github.com/nirs4all/nirs4all) (Python) plus references from scikit-learn, scipy, prospectr (R), mdatools (R), pywt, and pybaselines.

---

## Architecture

- **C++17 core** in `cpp/src/core/` — internal headers + implementations, never directly exposed.
- **Stable C ABI** in `cpp/include/chemometrics4all/c4a.h` — `c4a_*` symbols, opaque handles, error-buffer-per-context.
- **Thin language bindings** (`bindings/`): Python (ctypes + sklearn-compatible classes) and R (.Call) are active; MATLAB and JS/WASM are next in the roadmap.
- **Zero mandatory dependencies** — DIY linalg, RNG (PCG64 reimplementation for NumPy bit-exact parity), banded LDLT, FFT-free wavelet filter banks. Optional accelerated backends (BLAS, OpenMP) are gated by CMake presets.
- **CMake ≥ 3.21** with 20+ presets in `CMakePresets.json` (dev-debug, dev-release, ci-*, blas-on, sanitizer-*).
- **Parity-gated** — every operator is validated against pinned reference implementations (numpy 1.26.4, scipy 1.17.1, scikit-learn 1.8.0, pywt 1.8.0, pybaselines 1.2.1, selected nirs4all operators, …) with `parity/fixtures/*.json` + `parity/tolerances.md`.
- **Codex+Opus review workflow** — each phase is reviewed by Codex (pre & post) and Opus (independent post-review); transcripts in `docs/reviews/phase-N/`.

---

## Build

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
| 0–21 | 🟢 implemented | C++ core operators, fixtures, C ABI, and C++ parity tests |
| 22–23 | 🟢 implemented | Python and R bindings with binding parity |
| 24–25 | ⚪ pending | MATLAB and JS/WASM bindings |
| 26 | 🟡 scaffolded | Benchmark registry, reference lockfile, and placeholder dashboard; timing matrix not generated yet |

---

## License

[CeCILL-2.1](LICENSE) — French OSS license, GPL-compatible, SPDX-listed.

## Citation

See [CITATION.cff](CITATION.cff).
