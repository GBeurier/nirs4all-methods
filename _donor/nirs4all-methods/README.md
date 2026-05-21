# nirs4all-methods

[![License: CeCILL-2.1](https://img.shields.io/badge/license-CeCILL--2.1-blue.svg)](LICENSE)
[![Status: Stabilization](https://img.shields.io/badge/status-stabilization-blue.svg)](ROADMAP.md)

**nirs4all-methods** is a portable C++17 library for chemometrics / NIRS
operators: preprocessings, splitters, filters, augmentations, signal-type
detection, transfer metrics, diagnostics, and FCK feature engineering. It is the
companion of [pls4all](https://github.com/GBeurier/pls4all), which owns the PLS
regression / classification model family.

> **nirs4all-methods does NOT ship any models.** PLS variants live in `pls4all`. This library focuses on the upstream/downstream layer: how you prepare spectra, partition samples, detect outliers, augment training data, convert signal types, and measure transfer-learning alignment.

**Documentation / dashboard**:
[`https://gbeurier.github.io/nirs4all-methods/`](https://gbeurier.github.io/nirs4all-methods/)
· local source under [`docs/`](docs/).

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
- **Stable C ABI** in `cpp/include/n4m/n4m.h` — `n4m_*` symbols, opaque handles, error-buffer-per-context.
- **Thin language bindings** (`bindings/`): Python (ctypes + sklearn-compatible classes) and R (.Call) are active; MATLAB and JS/WASM are next in the roadmap.
- **Zero mandatory dependencies** — DIY linalg, RNG (PCG64 reimplementation for NumPy bit-exact parity), banded LDLT, FFT-free wavelet filter banks. Optional backends (BLAS, OpenMP, FITPACK spline smoothing) are gated by CMake.
- **CMake ≥ 3.21** with 20+ presets in `CMakePresets.json` (dev-debug, dev-release, ci-*, blas-on, sanitizer-*).
- **Parity-gated outside GitHub Actions** — every operator is validated against pinned reference implementations (numpy 1.26.4, scipy 1.17.1, scikit-learn 1.8.0, pywt 1.8.0, pybaselines 1.2.1, selected nirs4all operators, …) with `parity/fixtures/*.json` + `parity/tolerances.md`. The unified parity gate is intended for local or dedicated-server runs where the nirs4all reference checkout/package is controlled.
- **Benchmark + validation docs** — generated method pages and the interactive matrix consume the same committed benchmark payload (`docs/_static/bench-data.json`) and validation contracts (`benchmarks/validation/registry/`).
- **Codex+Opus review workflow** — each phase is reviewed by Codex (pre & post) and Opus (independent post-review); transcripts in `docs/reviews/phase-N/`.

---

## Documentation

- 🌐 **[Sphinx site + interactive benchmark dashboard](https://gbeurier.github.io/nirs4all-methods/)** — public GitHub Pages build from [`docs/`](docs/)
- 📚 **[Method catalogue](docs/methods/index.md)** — generated pages with signatures, math, bibliography, parity contracts, divergences, and timings
- 📊 **[Benchmark overview](docs/benchmarks/overview.md)** — cross-binding timing/parity payload and dashboard data model
- 🔬 **[Benchmark methodology](docs/benchmarks/methodology.md)** — reference policy, tolerances, datasets, and reproducibility notes
- 📜 **[ABI reference](docs/abi/reference.md)** — stable `n4m_*` C ABI, handles, errors, and exported symbols
- 🔌 **Bindings** — [Python](docs/bindings/python.md) · [R](docs/bindings/r.md) · [MATLAB](docs/bindings/matlab.md) · [JS / WASM](docs/bindings/js.md) · [Android](docs/bindings/android.md)

---

## Build

```bash
cmake --preset dev-release
cmake --build --preset dev-release
ctest --preset dev-release --output-on-failure
./build/dev-release/cpp/cli/n4m_cli --version
./build/dev-release/cpp/cli/n4m_cli --selfcheck
```

Requirements: C++17 toolchain (GCC ≥ 11 / Clang ≥ 14 / MSVC 2022), CMake ≥ 3.21, Ninja (recommended). A Fortran compiler is optional: `N4M_WITH_FITPACK=AUTO` enables the vendored FITPACK spline-smoothing backend when the toolchain supports it and keeps builds portable when it does not.

---

## Phase status

This repository is being built **phase by phase** with full Codex + Opus reviews after each phase. See [ROADMAP.md](ROADMAP.md) for the 27-phase plan.

| Phase | Status | Scope |
|------:|:-------|-------|
| 0–21 | 🟢 implemented | C++ core operators, fixtures, C ABI, and C++ parity tests |
| 22–23 | 🟢 implemented | Python and R bindings with binding parity |
| 24–25 | ⚪ pending | MATLAB and JS/WASM bindings |
| 26 | 🟢 implemented | Benchmark registry, validation snapshot, generated method docs, and interactive matrix/dashboard payload |

---

## License

[CeCILL-2.1](LICENSE) — French OSS license, GPL-compatible, SPDX-listed.

## Citation

See [CITATION.cff](CITATION.cff).
