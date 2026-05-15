# Roadmap

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

## Current Checkpoint - 2026-05-15

Latest local tag: `phase-6f-public-aom-pop-abi` (`0.66.0+abi.1.1.0`).

Last green local gate:

- 97 deterministic parity fixtures.
- 209 C++ ABI/core tests.
- `pls4all_cli --selfcheck`.
- Python ctypes smoke: `bindings/python/smoke_aom_pop.py` exercises every
  AOM/POP fixture through the public C ABI.
- ABI symbol diff against `cpp/abi/expected_symbols_linux.txt` (additive
  only: 28 new `p4a_*` symbols for AOM/POP/validation-plan).
- `ldd` dependency audit: only libc/libstdc++/libgcc/libm/loader.
- UBSAN.
- ASAN+UBSAN.

Current git notes:

- `main` is ahead of `origin/main` by ~24 local commits.
- Untracked files intentionally left out of commits: `Backlog.md`,
  `docs/_bench/`.
- AOM-PLS parity oracle is `nirs4all/bench/AOM_v0/aompls`, not the packaged
  `nirs4all` library surface.

## Next Agent Prompt

Continue from `/home/delete/nirs4all/pls4all` on `main`, currently tagged
`phase-6f-public-aom-pop-abi` (`0.66.0+abi.1.1.0`). Do not use GitHub Actions
for now. Keep using the local gate: pinned fixture generator, dev-release
build, C++ tests, CLI selfcheck, Python smoke (`bindings/python/smoke_aom_pop.py`
plus the legacy version/context/config snippet in
`bindings/python/README.md`), ABI symbol diff, `ldd`, `git diff --check`,
UBSAN and ASAN+UBSAN. Leave untracked `Backlog.md` and `docs/_bench/` alone.
For AOM/POP parity, use `/home/delete/nirs4all/nirs4all/bench/AOM_v0/aompls`
as the oracle. Next recommended tranche: start Phase 6g (POP holdout /
approximate-PRESS / one-SE variants + AOM-NIPALS materialized engine) and
land the first batch of `benchmarks/` against shipped methods.

## Shipped Core

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

Phase 2 is intentionally delayed until the core stabilizes. It now needs to
become active in parallel with Phase 6f.

### Python

- Expand ctypes binding beyond lifecycle/config.
- NumPy zero-copy matrix views.
- Fit/predict/transform wrappers for shipped solvers.
- sklearn-compatible `BaseEstimator` / `RegressorMixin` /
  `TransformerMixin` classes.
- AOM/POP wrapper surface after Phase 6f.
- Fixture-driven Python parity tests.

### R

- `.Call` gateway over the C ABI.
- R matrices to `p4a_matrix_view_t` without unnecessary copies when possible.
- Parity against R `pls`, `ropls`, `mixOmics`, `plsVarSel` where applicable.
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

Benchmarks should start once AOM/POP has a public ABI and Python wrappers.

- NIPALS/SIMPLS/SVD/kernel/wide-kernel vs sklearn, R `pls` and NumPy ports.
- AOM/POP vs `nirs4all/bench/AOM_v0/aompls`.
- Preprocessing throughput vs NumPy/SciPy references.
- Variable-selection runtime vs Python/R references.
- LW-PLS and MB-PLS scaling.
- Batch CV, bootstrap, CARS/MCUVE/AOM sweeps.
- Memory footprint and dependency audit per build.

Benchmark outputs should live under `benchmarks/` or `docs/_bench/`, but only
commit curated summaries and reproducible scripts.

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
11. Commit as `release(phase-X): version-topic`.
12. Tag as `phase-X-topic`.

Do not push unless explicitly asked.
