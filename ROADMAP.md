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

- SNV, MSC, EMSC, Detrend, Savitzky-Golay smoothing + 1st/2nd derivatives, ASLS baseline.
- Stateful `Preprocessor` C++ interface; pipeline-of-preprocessors with explicit `fit` / `transform`.
- Cross-validation engine: k-fold, LOO, Kennard-Stone, SPXY, holdout, external folds, repeated k-fold, Monte-Carlo CV.
- Metrics: RMSEC, RMSECV, RMSEP, R², Q², bias, RPD, RPIQ, MAE, slope/intercept obs-vs-pred, sensitivity, specificity, balanced accuracy, AUC, MCC.
- VIP, selectivity ratio, regression coefficients per component.

(Phase 3 ships before Phase 2 — bindings benefit from having real algorithms to expose.)

## Phase 4 — Advanced PLS variants

- SIMPLS core solver shipped as `phase-4a-simpls`.
- SVD PLS core solver shipped as `phase-4b-svd`.
- PCR baseline shipped as `phase-4c-pcr`.
- Linear kernel-algorithm PLS shipped as `phase-4d-kernel`.
- Wide-kernel PLS shipped as `phase-4e-wide-kernel`.
- SIMPLS with materialised auto-prefix CV scoring (à la `aompls`).
- Kernel-algorithm PLS for wide matrices (`p >> n`).
- OPLS, OPLS-DA.
- PLS-DA, PLS-LDA, PLS-logistic.
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

- Operator bank as composable, stateful `Preprocessor` objects (Identity, SNV, MSC, SG-deriv-1/2, Polynomial Detrend, ASLS, OSC).
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
