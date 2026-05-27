# Roadmap

> **⚠️ LEGACY (pre-rename).** This is the original phase log from before the
> `pls4all`/`p4a` → `nirs4all-methods`/`n4m` rename (it still says `pls4all`,
> references `Direction_Technique.md`/`Overview.md`, and stops at phase-49 /
> v0.97). It is kept as a historical record of the shipped algorithm phases.
> **The authoritative, current plan is [`REFACTOR_PLAN.md`](REFACTOR_PLAN.md)**
> (phases A–F); do not treat anything below as current.

`pls4all` is built in deliberate phases. Each delivered phase has a
self-contained note under [`roadmap/`](roadmap/). The canonical technical
spec is [`Direction_Technique.md`](Direction_Technique.md). The full algorithm
taxonomy is in [`Overview.md`](Overview.md).

The project rule remains:

- C++17 internal core.
- Public stable C ABI only.
- No mandatory runtime dependency beyond libc/libstdc++/libgcc.
- Thin bindings over the C ABI.
- Every numerical method must have a deterministic parity gate against the
  best available R or Python reference.
- GitHub Actions are parked for now to save runner quota; the authoritative
  gate is the local parity/build/test/sanitizer run.

## Current Checkpoint - 2026-05-16

Latest local tag: `phase-49-vissa-pls` (`0.97.0+abi.1.14.0`).

Last green local gate:

- 97 deterministic parity fixtures.
- 256 C++ ABI/core tests (unchanged at 256; Batch 1 promotes existing
  internal kernels to public ABI rather than adding new ones).
- `pls4all_cli --selfcheck`.
- `pls4all_cli --bench` smoke for every shipped PLS solver.
- Python ctypes smoke: `bindings/python/smoke_aom_pop.py` exercises every
  AOM/POP fixture through the public C ABI.
- Python model smoke: `pls4all.Model.fit / predict / get_array` succeeds.
- ABI symbol diff against `cpp/abi/expected_symbols_linux.txt`: 159
  symbols, all `p4a_*` prefixed.
- `ldd` dependency audit: only libc/libstdc++/libgcc/libm/loader.
- UBSAN.
- ASAN+UBSAN.
- Benchmarks: `python benchmarks/run.py --check` passes for every
  shipped suite (aom_global, pls_regression, matrix).
- Parity gate (`benchmarks/parity_timing/runner.py`): 13 PASS (all
  external refs), 24 paper-only (smoke-only), 0 numpy-mirror.

Current git notes:

- `main` is ahead of `origin/main` by ~25 local commits.
- Untracked files intentionally left out of commits: `Backlog.md`,
  `docs/_bench/`.
- AOM-PLS parity oracle is `nirs4all/bench/AOM_v0/aompls`, not the packaged
  `nirs4all` library surface.

## Status Summary

### Done

- **Phase 0** — ABI / build foundation, fixture schema, CLI selfcheck.
- **Phase 1** — Dependency-free NIPALS PLS regression with fit / predict /
  transform, fitted-array accessors, serialization.
- **Phase 3 (a–r)** — Preprocessing pipeline (identity, center, autoscale,
  Pareto, SNV, MSC, EMSC, Detrend, Savitzky-Golay, Norris-Williams,
  ASLS, Haar wavelet, OSC, EPO), regression + classification metrics +
  calibration, splitters (k-fold, LOO, holdout, external, repeated,
  Monte-Carlo, Kennard-Stone, SPXY), cross-validation engine, VIP and
  selectivity ratio, component-prefix coefficients.
- **Phase 4 (a–s)** — SIMPLS / SVD / PCR / linear kernel / wide-kernel /
  orthogonal-scores / power / randomized-SVD solvers; PLSCanonical
  (NIPALS + SVD); PLSSVD; PLS-DA; OPLS / OPLS-DA / multi-response shared
  predictive score; SIMPLS component-count CV; PLS-LDA; PLS-logistic;
  MB-PLS; LW-PLS.
- **Phase 5 (a–u)** — Variable-selection family: rangers (VIP,
  coefficient, selectivity ratio), interval (moving-window, biPLS,
  siPLS), stability (Monte-Carlo, UVE, EMCUVE, randomization), wrappers
  (SPA, CARS, Random Frog, SCARS, GA, Shaving, REP, IPW, ST, BVE, T²,
  WVC, WVC threshold).
- **Phase 6 (a–f)** — AOM-PLS / POP-PLS: strict-linear operator family
  (identity, polynomial detrend, zero-padded Savitzky-Golay, finite
  difference, Norris-Williams, Whittaker, FCK); global AOM-SIMPLS CV
  selection; POP-PLS per-component covariance selection; public C ABI
  for AOM/POP including the `p4a_validation_plan_t` opaque handle.
- **Phase 7a** — Benchmark foundation: `benchmarks/` directory, Python
  ctypes driver, AOM-PLS global benchmark vs `nirs4all/bench/AOM_v0/aompls`
  oracle, gated accuracy CSV + informational timing CSV.

### Phase 7 - Benchmark Foundation - shipped through 7e

- Phase 7b: Python ctypes `Model` wrapper + NumPy zero-copy
  `MatrixView` constructor. `Model.fit/predict/transform/get_array`
  cover all 9 shipped PLS regression solvers. New
  `benchmarks/runners/pls_regression.py` runner (smoke gated; full
  matrix on demand).
- Phase 7c: minimum-viable R package at `bindings/r/pls4all/` with
  `.Call` gateway and fit/predict wrappers for the same solver list.
  No Rcpp dependency. Documentation for build / install / smoke. R
  toolchain not installed on this host — package builds against the
  C ABI on a future host; matrix runner skips R rows gracefully when
  `Rscript` is absent.
- Phase 7d: `pls4all_cli --bench` subcommand that times native C++
  fit + predict per algo on a deterministic synthetic dataset and
  prints a CSV-parseable row.
- Phase 7e: `benchmarks/runners/matrix.py` orchestrator. For each
  (algo, n_samples, n_features) cell it times native C++ / pls4all-
  Python / pls4all-R / sklearn reference, writes a gated accuracy CSV
  and an informational timing CSV + summary markdown + metadata JSON.
  Smoke set is committed; full matrix (9 algos × 5 n × 3 p = 135
  cells) is parameterized — re-run under varying CPU pinning when the
  host is free.

### Phase 8 - 15 - PLS extensions (§7, §11, §12, §13, §17, §18, §19)

Internal C++ kernels. Public C ABI exposure deferred to the binding
tranche.

- Phase 8 (§7): Sparse SIMPLS via soft-thresholding;
  `cfg.sparsity_lambda`, `P4A_ALGO_SPARSE_PLS` dispatch.
- Phase 9 (§17): T² Hotelling, Q-residuals (SPE), DModX in
  `cpp/src/core/pls_diagnostics.{hpp,cpp}`.
- Phase 10 (§18): one-SE rule on top of the existing component-count
  CV.
- Phase 11 (§19): empirical percentile thresholds for T² and
  Q-residuals (`pls_monitoring_fit` / `pls_monitoring_evaluate`).
- Phase 13 (§13): domain-invariant PLS (`fit_di_pls`).
- Phase 14 (§11): just-in-time PLS — covered by the existing
  `fit_predict_lw_pls` (documented).
- Phase 15 (§12): recursive PLS — moving-window SIMPLS refit
  (`cpp/src/core/recursive_pls.{hpp,cpp}`).

### Phase 16 - 30 - Overview taxonomy completion

Closes every remaining Overview section. All internal C++ kernels.

- Phase 16 (§8): O2PLS — bi-directional OPLS with predictive +
  X-orthogonal + Y-orthogonal components.
- Phase 17 (§8): SO-PLS — sequential + orthogonalized PLS for B
  X-blocks predicting Y.
- Phase 18 (§8): OnPLS — joint + per-block unique components.
- Phase 19 (§8): ROSA — Response-Oriented Sequential Alternation
  (per-component best-block selection).
- Phase 20 (§7): sPLS-DA via dummy encoding + sparse SIMPLS.
- Phase 21 (§7): group sparse and fused sparse PLS variants.
- Phase 22 (§9): N-PLS (Bro's algorithm for 3-way tensors).
- Phase 24 (§10.2): non-linear kernel PLS (RBF / polynomial /
  sigmoid) via Gram-matrix dual NIPALS.
- Phase 25 (§1): CPPLS / powered PLS (continuous gamma).
- Phase 26: weighted PLS, robust PLS (Huber IRLS), ridge PLS,
  continuum regression.
- Phase 27: PLS-GLM, PLS-QDA, PLS-Cox.
- Phase 28: PDS (window-LS) and DS calibration transfer, MIR-PLS,
  missing-aware NIPALS.
- Phase 29 (§18): approximate-PRESS with leverage-inflated residual
  PRESS and component selection.
- Phase 30 (§20): bagging-PLS, boosting-PLS, random-subspace PLS.

All shipped as internal kernels in
`cpp/src/core/{multiblock_extensions,tensor_pls,kernel_pls,extra_pls}.{hpp,cpp}`.

### Next (Phase 6 continuation)

- **Phase 6g** — POP holdout / approximate-PRESS / one-SE variants;
  AOM-NIPALS materialized engine; AOM/POP covariance and adjoint fast
  paths; soft / sparse / superblock AOM selection fixtures; per-block
  and per-target AOM plans.
- **Phase 6h** — Integrate the locally developed AOM-PLS and FCL-PLS work
  as first-class pls4all methods.

### Later (binding + algorithm tracks)

- **Binding roadmap** (after Phase 7c): MATLAB MEX, JS / WebAssembly,
  Julia, Java / Android, plus secondary targets (C#, Rust, Go, Swift).
- **Algorithm taxonomy backlog** — see the
  [Remaining Algorithm Taxonomy](#remaining-algorithm-taxonomy) section
  below.

## Next Agent Prompt

Continue from `/home/delete/nirs4all/pls4all` on `main`, currently tagged
`phase-49-vissa-pls` (`0.97.0+abi.1.14.0`). Do not use
GitHub Actions for now. Keep using the local gate: pinned fixture generator,
dev-release build, C++ tests, CLI selfcheck (`pls4all_cli --selfcheck`),
CLI bench smoke (`pls4all_cli --bench --algo pls_simpls --samples 200
--features 100`), Python smoke (`bindings/python/smoke_aom_pop.py` plus
the version/context/config snippet in `bindings/python/README.md` plus
`pls4all.Model.fit/predict` round-trip), ABI symbol diff, `ldd`,
`git diff --check`, UBSAN and ASAN+UBSAN, and `python benchmarks/run.py
--check` for the benchmark gate. Also run the new external-reference
parity gate `PYTHONPATH=bindings/python/src \
parity/python_generator/.venv/bin/python -m benchmarks.parity_timing.runner`
and re-commit `benchmarks/results/parity_gate/`. Leave untracked
`Backlog.md` and `docs/_bench/` alone. For AOM/POP parity, use
`/home/delete/nirs4all/nirs4all/bench/AOM_v0/aompls` as the oracle.

Algorithm development is now complete for every Overview section. The next
agent should:

1. **Public C ABI exposure** for the new internal kernels shipped in
   `pls4all::core::` over phases 6g – 30. Plan one MINOR ABI bump per
   batch (the §6g AOM/POP policy work, the §17/§18/§19 diagnostics and
   monitoring, the §8/§9/§10.2/§13 algorithm batches, and the §20
   ensembles), each additive only. Update
   `cpp/abi/expected_symbols_linux.txt` and `cpp/include/pls4all/p4a.h`.
2. **Python ctypes bindings** for the newly exposed surface; extend
   `bindings/python/src/pls4all/*` and the smoke driver.
3. **R `.Call` gateway** for the same surface in
   `bindings/r/pls4all/`.
4. **MATLAB MEX, JS-WASM, Julia, Java** — pick up the remaining
   binding targets once the Python / R coverage is solid.
5. **Benchmark matrix expansion**: add new columns to
   `benchmarks/runners/matrix.py` for every shipped method. Run
   `python benchmarks/run.py --benchmark matrix --scale full
   --repeats 5` under `OMP_NUM_THREADS=1|5|10` when the host is free.
6. **Then** start the Acceleration Roadmap (BLAS / OpenMP / CUDA).

For AOM/POP parity, use
`/home/delete/nirs4all/nirs4all/bench/AOM_v0/aompls` as the oracle. The
pinned `parity/python_generator/` venv (sklearn 1.4.2) is still broken
on this host's Python 3.13; new fixture work should either restore that
venv or migrate to a numpy-only reference path.

## Shipped Core (phase log)

### Phase 0 - ABI and Build Foundation - shipped

- CMake project, presets, warning/sanitizer options.
- Public `p4a.h` ABI with opaque handles, matrix views, enums, status codes
  and version/ABI compatibility functions.
- Context/config/matrix/operator-bank/gating/pipeline/model/array lifecycle.
- Shared and static `libp4a`; symbol snapshot gate.
- CLI selfcheck.
- Fixture schema and parity infrastructure.
- Minimal ctypes Python lifecycle/config binding.
- README skeletons for R, MATLAB, JS/WASM and Android.

### Phase 1 - PLS CPU Reference - shipped

- Dependency-free NIPALS PLS1/PLS2 regression.
- Fit/predict/transform.
- Center/scale learned on train and applied leak-free.
- Fitted-array accessors.
- Component-prefix coefficients.
- Binary export/import with versioned serialization.

### Phase 3 - Preprocessing, Validation and Metrics - shipped through 3r

- Pipeline fit/transform.
- Identity, center, autoscale, Pareto, SNV.
- MSC, EMSC.
- Polynomial detrend.
- Savitzky-Golay smoothing and first/second derivatives.
- ASLS baseline.
- Norris-Williams derivatives.
- Haar wavelet denoising.
- OSC and EPO.
- Regression metrics: RMSE, MAE, bias, R2/Q2, slope/intercept, RPD, RPIQ.
- Splitters: k-fold, LOO, holdout, external folds, repeated k-fold,
  Monte-Carlo, Kennard-Stone, SPXY.
- Cross-validation engine.
- Binary and multiclass classification metrics plus calibration bins.
- VIP and selectivity ratio.
- Original-scale component-prefix coefficients.

### Phase 4 - Advanced PLS Variants - shipped through 4s

- SIMPLS.
- SVD PLS.
- PCR.
- Linear kernel-algorithm PLS.
- Wide-kernel PLS.
- Orthogonal-scores PLS.
- Power-method PLS.
- Randomized-SVD PLS.
- PLSCanonical with NIPALS/SVD.
- PLSSVD direct cross-covariance scores.
- PLS-DA.
- OPLS and OPLS-DA.
- Multiclass OPLS-DA/common predictive score model.
- SIMPLS component-count CV.
- PLS-LDA.
- PLS-logistic.
- MB-PLS.
- LW-PLS.

### Phase 5 - Variable Selection - shipped through 5u

- VIP, coefficient-magnitude and selectivity-ratio rankers.
- Moving-window / contiguous interval CV.
- biPLS backward interval selection.
- siPLS interval-combination search.
- Monte-Carlo coefficient stability.
- UVE with artificial variables.
- SPA-PLS.
- CARS-PLS.
- Random Frog PLS.
- SCARS-PLS.
- GA-PLS.
- Shaving-PLS.
- REP-PLS.
- IPW-PLS.
- ST-PLS.
- BVE-PLS.
- T2-PLS.
- WVC-PLS numeric selection.
- WVC threshold/factor rules.
- EMCUVE.
- PLS randomization-test selection.

### Phase 6 - AOM-PLS and POP-PLS - shipped through 6f

- Phase 6a: internal soft/hard AOM preprocessing-bank transform primitive.
- Phase 6b: internal global AOM-SIMPLS CV selection against
  `nirs4all/bench/AOM_v0/aompls` for identity/detrend.
- Phase 6c: strict-linear AOM kernels for zero-padded Savitzky-Golay,
  finite difference and Norris-Williams, plus wider global-selection parity.
- Phase 6d: strict-linear Whittaker and FCK AOM operators, with direct
  transform parity and global-selection parity.
- Phase 6e: internal POP-PLS per-component SIMPLS covariance selector, with
  selected operator sequence, component candidate scores, prefix scores,
  full-fit predictions and bench-compatible CV scoring semantics.
- Phase 6f: public C ABI for validation plans, AOM global and POP per-
  component selection (opaque result handles, typed accessors, hardened
  fold validation). Python ctypes smoke that drives every shipped AOM/POP
  fixture through the new surface.

### Phase 7 - Benchmark Foundation - shipped through 7a

- Phase 7a: `benchmarks/` directory with a deterministic Python driver.
  First runner compares the public C ABI `p4a_aom_global_select` against
  `nirs4all/bench/AOM_v0/aompls` on 4 cases (identity-favoured and
  detrend-favoured). Numerical deltas are gated by
  `python benchmarks/run.py --check`; wall-clock timings are recorded
  separately and informational.

## Active Track

### Phase 6g - POP/AOM Policy Expansion

Goal: complete the policy variants already present in the bench oracle.

Deliverables:

- POP holdout / approximate-PRESS / one-SE variants.
- AOM-NIPALS materialized path.
- AOM/POP covariance and adjoint fast paths where the strict-linear operator
  contract permits it.
- Soft, sparse and superblock AOM selection fixtures.
- Per-block and per-target AOM plans.

Reference:

- `nirs4all/bench/AOM_v0/aompls/selection.py`
- `nirs4all/bench/AOM_v0/aompls/simpls.py`
- `nirs4all/bench/AOM_v0/aompls/nipals.py`
- `nirs4all/bench/AOM_v0/aompls/banks.py`

### Phase 6h - Local AOM/FCL Integration

Goal: integrate the locally developed AOM-PLS and FCL-PLS work as first-class
pls4all methods.

Deliverables:

- Identify the local source-of-truth implementations and freeze parity
  fixtures before porting.
- Port FCL-PLS kernels behind internal C++ APIs.
- Add fixtures for AOM/FCL edge cases, not only synthetic happy paths.
- Document numerical conventions and deviations from the local prototypes.

## Binding Roadmap

Phase 2 is intentionally delayed until the core stabilizes. It is now active
in parallel with Phase 7 because the benchmark matrix needs every shipped
language binding.

### Python (Phase 7b — in progress)

- Expand ctypes binding beyond lifecycle/config and AOM/POP.
- NumPy zero-copy matrix views (`p4a_matrix_view_t` from
  `numpy.ndarray.ctypes.data`, ascontiguousarray fallback).
- Fit/predict/transform wrappers for every shipped solver (NIPALS,
  orthogonal-scores, SIMPLS, kernel, wide-kernel, SVD, power,
  randomized-SVD, plus PCR).
- sklearn-compatible `BaseEstimator` / `RegressorMixin` /
  `TransformerMixin` classes (deferred to a later Python tranche).
- Fixture-driven Python parity tests.

### R (Phase 7c — in progress)

- `.Call` gateway over the C ABI.
- R matrices to `p4a_matrix_view_t` without unnecessary copies when
  possible (via `REAL()` on a contiguous matrix).
- Parity against R `pls`, `ropls`, `mixOmics`, `plsVarSel` where
  applicable.
- CRAN-compatible package skeleton and smoke tests.

### MATLAB

- MEX gateway.
- Class wrapper for model handles.
- Model import/export.
- Prediction-first examples.

### JavaScript / WebAssembly

- Emscripten build of the C ABI.
- Predict-first npm package.
- TypedArray matrix views.
- Model load/predict smoke tests.

### Julia

- Thin `ccall` wrapper.
- Matrix view helpers.
- Parity fixture runner.

### Java / Android

- JNI bridge and Kotlin API.
- Android AAR.
- Predict-first model loading.
- On-device smoke tests.

Other bindings to consider after these are stable: C#, Rust, Go and Swift.

## Benchmark Roadmap

Benchmarks ship under `benchmarks/` and follow a strict split:

- **Accuracy CSVs** are committed and gated by `python benchmarks/run.py
  --check`. They contain only deterministic numerical deltas (operator
  match, component-count match, RMSE / coefficient deltas).
- **Timing CSVs** are committed for traceability but NOT gated. They are
  platform-dependent, recorded with explicit host info (Python version,
  platform, processor, logical cores, environment variables such as
  `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS`).
- **Summary markdown** is regenerated on every run with a status table.

### Shipped

- **7a** — AOM-PLS global selection vs `nirs4all/bench/AOM_v0/aompls`,
  4 cases (9x6, 12x8, 16x10, 14x9), Python ctypes path.

### Active

- **7b** — PLS regression benchmark matrix: 9 solvers
  (NIPALS / orthogonal-scores / SIMPLS / kernel-algorithm / wide-kernel /
  SVD / power / randomized-SVD / PCR) × 5 sample sizes
  (200, 500, 1000, 2000, 10000) × 3 feature counts
  (100, 1000, 10000), pls4all-Python vs scikit-learn `PLSRegression`.
  CPU-count parameterization through environment variables (run later
  under 1 / 5 / 10 cores when the host machine is free).
- **7c** — R-side numbers added to the same matrix via the minimum
  R binding shipped in the binding roadmap.
- **7d** — Native C++ baseline column via `pls4all_cli --bench`.
- **7e** — Orchestrator that runs the full matrix end-to-end, splits
  outputs per language and per core count, regenerates the summary
  markdown.

### Later

- POP per-component benchmark column.
- AOM-PLS benchmark expanded to the wider operator bank (SG, Norris-
  Williams, Whittaker, FCK).
- Preprocessing throughput vs NumPy / SciPy references.
- Variable-selection runtime vs Python / R references.
- LW-PLS and MB-PLS scaling.
- Batch CV, bootstrap, CARS / MCUVE / AOM sweeps.
- Memory footprint and dependency audit per build.
- GPU / OpenMP / BLAS backend self-parity once the Acceleration Roadmap
  starts.

Benchmark outputs live under `benchmarks/results/`; only curated CSVs and
summaries are committed.

## Acceleration Roadmap

Acceleration remains optional and must not change the C ABI.

- BLAS backend.
- OpenMP for fold/bootstrap/operator scans.
- Batch APIs for large validation and selection workloads.
- CUDA backend in a separate optional shared library.
- Later: WebGPU/WASM SIMD if the JS target needs it.

Every accelerated backend needs self-parity against reference CPU.

## Remaining Algorithm Taxonomy

These are not yet first-class shipped methods and should be split into small
reviewed phases with parity references:

- CPPLS / powered PLS.
- Weighted and sample-weighted PLS.
- Robust PLS.
- Ridge/regularized/penalized PLS.
- Continuum regression.
- MIR-PLS.
- Sparse PLS and sparse PLS-DA.
- PLS-QDA, PLS-GLM, PLS-Cox and survival PLS.
- O2PLS, DOSC-PLS, OnPLS.
- Missing-aware NIPALS.
- Calibration transfer and domain adaptation: PDS, DS, di-PLS and related
  methods.
- Multiway/tensor PLS: N-PLS, Tri-PLS, PARAFAC-PLS, Tucker-PLS.
- Dynamic, recursive and online PLS.
- Ensemble PLS methods: bagging, boosting and random-subspace PLS.

## Release Discipline

For every phase:

1. Add or update a focused `roadmap/phase-*.md`.
2. Implement the smallest coherent tranche.
3. Generate parity fixtures with the pinned generator venv.
4. Build with `/home/delete/.venv/bin/cmake --build --preset dev-release --parallel`.
5. Run `/home/delete/.venv/bin/ctest --preset dev-release --output-on-failure`.
6. Run `build/dev-release/cpp/tests/pls4all_tests`.
7. Run CLI selfcheck and Python smoke.
8. Run ABI symbol diff and `ldd` dependency audit.
9. Run `git diff --check`.
10. Run UBSAN and ASAN+UBSAN local builds.
11. For benchmark phases, run `python benchmarks/run.py --check`.
12. Commit as `release(phase-X): version-topic`.
13. Tag as `phase-X-topic`.

Do not push unless explicitly asked.
