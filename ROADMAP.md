# Roadmap

`pls4all` is built in deliberate phases. Each phase has a self-contained roadmap under [`roadmap/`](roadmap/) that is reviewed by Codex *before* implementation begins, and again *before* the phase is tagged.

The phase order is **depth-first on the core** (C ABI → algorithms → preprocessing → advanced variants) and **breadth-second on bindings**. This reflects a deliberate choice: we'd rather have a small, correct, idiomatic Python/R/etc API once the core is mature than a wide multi-language surface that thrashes for the first six months.

The canonical technical spec is [`Direction_Technique.md`](Direction_Technique.md). The full algorithm taxonomy is in [`Overview.md`](Overview.md).

## Phase 0 — ABI & Build Foundation · **shipped**

Goal: a callable `libp4a.{so,dll,dylib}` with no algorithm — only ABI plumbing, build matrix, parity scaffolding, docs.

- CMake project structure, options module, presets.
- `cpp/include/pls4all/p4a.h` — every enum, struct, lifecycle and config setter Phase 1 will need.
- Status codes / error messages / version functions / ABI compatibility checks.
- Stride-aware matrix views with `_validate`.
- `pls4all_c` shared library exports only `p4a_*` (verified by `nm` / `dumpbin` against checked-in golden lists).
- doctest-based ABI / lifecycle / OOM / exception-safety / fixture-loader tests.
- CI: Linux × {gcc-12, gcc-13, clang-16} × {Release, Debug}, macOS × clang × {Release, Debug}, Windows × {MSVC 2022, MinGW UCRT64} × {Release, Debug}, ASAN + UBSAN, ABI surface gate, coverage.
- Parity fixture schema v1, three synthetic fixtures, Python + R generator skeletons, initial tolerances table.
- Binding skeletons for Python (ctypes), R (`.Call`), MATLAB (MEX), JS (Emscripten), Android (JNI) — each shipping a hello-version smoke test.
- Documentation skeleton (architecture, ABI reference, binding guides, parity methodology).
- Final commit tagged `phase-0`.

## Phase 1 — PLS CPU Reference · **shipped**

Goal: a reliable, serialisable, portable PLS engine.

- Minimal linalg in `cpp/src/core/linalg/` (dot, axpy, scal, norm2, gemv, gemm-small, transpose-view, QR, power iteration).
- NIPALS PLS1 and PLS2 with regression deflation.
- Center / scale fit on training only; transform applied to validation (leakage-safe).
- `predict`, `transform`, coefficient extraction per number of components.
- Binary export / import with versioning, endianness, checksum.
- The three Phase 0 fixtures flip from `P4A_ERR_NOT_IMPLEMENTED` to green; tolerances per `parity/tolerances.md` (`< 1e-9` vs sklearn for PLS1 / PLS2 in well-conditioned cases).

## Phase 3 — Chemometrics / NIRS preprocessing & validation

Goal: useful for real spectroscopy.

- SNV, MSC, EMSC, Detrend, Savitzky-Golay smoothing + 1st/2nd derivatives, ASLS baseline, Norris-Williams derivatives, Haar wavelet denoising, OSC, EPO.
- Stateful `Preprocessor` C++ interface; pipeline-of-preprocessors with explicit `fit` / `transform`.
- Phase 3a shipped as `phase-3a-preprocessing-pipeline`: pipeline fit/transform plus identity, center, autoscale, Pareto scale and SNV.
- Phase 3b shipped as `phase-3b-msc-preprocessing`: multiplicative scatter correction in the same fitted pipeline contract.
- Phase 3c shipped as `phase-3c-detrend-preprocessing`: row-wise polynomial detrending with degree parameter.
- Phase 3d shipped as `phase-3d-savgol-preprocessing`: Savitzky-Golay smoothing plus 1st/2nd derivative operator support.
- Phase 3e shipped as `phase-3e-emsc-preprocessing`: EMSC with fitted mean reference and polynomial baseline terms.
- Phase 3f shipped as `phase-3f-asls-preprocessing`: asymmetric least-squares baseline correction.
- Phase 3g shipped as `phase-3g-norris-williams-preprocessing`: Norris-Williams gap-segment derivatives.
- Phase 3h shipped as `phase-3h-wavelet-preprocessing`: Haar wavelet denoising with soft-thresholded detail coefficients.
- Phase 3i shipped as `phase-3i-osc-preprocessing`: supervised one-component orthogonal signal correction.
- Phase 3j shipped as `phase-3j-epo-preprocessing`: supervised one-component external parameter orthogonalization.
- Phase 3k shipped as `phase-3k-regression-metrics`: internal regression metric kernels for RMSE, MAE, bias, R2/Q2, observed-vs-predicted slope/intercept, RPD and RPIQ.
- Phase 3l shipped as `phase-3l-validation-splits`: deterministic internal k-fold, LOO and holdout split generators.
- Phase 3m shipped as `phase-3m-cross-validation-engine`: internal k-fold regression CV orchestration with out-of-sample predictions and aggregate regression metrics.
- Cross-validation split extensions: Kennard-Stone, SPXY, external folds, repeated k-fold, Monte-Carlo CV.
- Phase 3n shipped as `phase-3n-classification-metrics`: internal binary classification metrics for sensitivity, specificity, balanced accuracy, precision/F1, MCC and AUC.
- Classification extensions: multi-class macro/micro averaging and calibration curves.
- Phase 3o shipped as `phase-3o-variable-importance`: internal VIP and selectivity-ratio kernels for fitted PLS models with stored scores.
- Regression coefficients per component.

(Phase 3 ships before Phase 2 — bindings benefit from having real algorithms to expose.)

## Phase 4 — Advanced PLS variants

- SIMPLS core solver shipped as `phase-4a-simpls`.
- SVD PLS core solver shipped as `phase-4b-svd`.
- PCR baseline shipped as `phase-4c-pcr`.
- Linear kernel-algorithm PLS shipped as `phase-4d-kernel`.
- Wide-kernel PLS shipped as `phase-4e-wide-kernel`.
- Orthogonal-scores PLS shipped as `phase-4f-orthogonal-scores`.
- Power-iteration PLS shipped as `phase-4g-power`.
- Randomized-SVD PLS shipped as `phase-4h-randomized-svd`.
- PLSCanonical with NIPALS/SVD and canonical deflation shipped as `phase-4i-canonical`.
- PLS-DA dummy-response score model shipped as `phase-4j-pls-da`.
- OPLS1 with one predictive component and orthogonal corrections shipped as `phase-4k-opls`.
- Binary OPLS-DA dummy-response score model shipped as `phase-4l-opls-da`.
- Multi-response OPLS and multi-class OPLS-DA common predictive score model shipped as `phase-4m-multiclass-opls-da`.
- PLSSVD cross-covariance score model shipped as `phase-4n-pls-svd`.
- SIMPLS with materialised auto-prefix CV scoring (à la `aompls`).
- PLS-LDA, PLS-logistic.
- MB-PLS with `Block` API; per-block weighting.
- LW-PLS (locally-weighted PLS) with KD-tree neighbours.

## Phase 2 — Language bindings (now with real algorithms behind them)

- Python wheel: `pip install pls4all`. ctypes loader + NumPy zero-copy views.
- R package: CRAN-ready `DESCRIPTION` + `NAMESPACE` + `.Call` gateway.
- MATLAB toolbox: MEX gateway + class wrapper + model import/export.
- JS / WASM npm package: predict-first, `model.load(buffer)` / `model.predict(X)`.
- Android AAR: JNI gateway, Kotlin API, predict-first.

Each binding ships a parity-test suite that loads the JSON fixtures and asserts the documented tolerances.

## Phase 5 — Variable selection

- VIP selector, regression-coefficient selector, selectivity-ratio selector.
- Interval methods: iPLS, biPLS, siPLS, moving-window PLS.
- Monte-Carlo / stability: UVE-PLS, MCUVE-PLS, EMCUVE-PLS, randomisation tests.
- Wrappers / metaheuristics: CARS, SCARS, SPA, Random Frog, GA-PLS, Shaving, BVE-PLS, T2-PLS, WVC-PLS.

## Phase 6 — AOM-PLS & POP-PLS · the scientific differentiator

- Operator bank as composable, stateful `Preprocessor` objects (Identity, SNV, MSC, SG-deriv-1/2, Polynomial Detrend, ASLS, Norris-Williams, Haar Wavelet, OSC, EPO).
- Soft gating (weighted mixture), hard gating (discrete pick), sparse gating (penalised mixture).
- Per-component / per-block / per-target AOM.
- AOM-NIPALS, AOM-SIMPLS, AOM-OPLS, AOM-MB-PLS.
- POP-PLS with PRESS-based per-component operator selection.

## Phase 7 — Accelerated backends

- Optional BLAS backend (cblas, Accelerate, MKL) — selected at runtime via `p4a_context_set_backend`.
- Optional OpenMP for embarrassingly parallel paths (CV folds, bootstrap).
- Batch APIs (`p4a_fit_batch`) for massive workloads: nested CV, bootstrap, MCUVE / CARS permutations, iPLS / moving-window sweeps, AOM operator scans.
- Optional CUDA backend in a separate shared library `libp4a_cuda.so` — same ABI, different `p4a_backend_t` selector. Self-parity: ≤ 1e-9 abs / ≤ 1e-8 rel for fp64 GPU vs reference CPU.

## Beyond v1.0 — under consideration

- Multiway / tensor PLS (N-PLS, Tri-PLS, PARAFAC-PLS, Tucker-PLS).
- Domain adaptation / calibration transfer (di-PLS, EPO-PLS, PDS).
- Dynamic / recursive / online PLS for process monitoring.
- Ensemble methods (bagging-PLS, boosting-PLS, random-subspace-PLS).
- Distribution to vcpkg, conda-forge, Homebrew, Debian.

Anything listed beyond v1.0 is opt-in via separate roadmap proposals.
