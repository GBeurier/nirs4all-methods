# pls4all

> A portable PLS / NIRS engine with a stable C ABI and thin first-class bindings for Python, R, MATLAB, JavaScript / WebAssembly and Android.

**Status — Phase 1 shipped, Phase 3r preprocessing / validation core live, PLSCanonical, PLSSVD, PLS-DA, PLS-LDA, PLS-logistic, MB-PLS, LW-PLS, OPLS / OPLS-DA, orthogonal-scores, SIMPLS, kernel, wide-kernel, SVD, power, randomized-SVD, PCR and Phase 5u variable-selection core live · API unstable until v1.0 · Not yet on PyPI / CRAN / npm.**

`pls4all` reimplements the full Partial Least Squares family in C++17 behind a small, stable C ABI. The same numerical core powers every binding, so a model trained in Python predicts byte-for-byte the same way in R, MATLAB, a browser or on Android.

## Why pls4all?

- **One implementation, many languages.** A typed C ABI is the only public surface. Bindings translate native objects (NumPy arrays, R matrices, MATLAB `mxArray`, `TypedArray`, JNI buffers) into the same `p4a_matrix_view_t` — no per-language reimplementation.
- **Zero mandatory dependencies.** No Eigen, no Boost, no BLAS, no pybind11, no Rcpp on the install path. Optional accelerated backends (BLAS, OpenMP, CUDA) plug in without changing the ABI.
- **Research-grade rigor, production deployment.** Cross-checked against scikit-learn, R `pls` / `ropls` / `mixOmics` / `plsVarSel`, and `nirs4all`. Parity tolerances are documented per implementation pair in [`parity/tolerances.md`](parity/tolerances.md).
- **Composable engine, not 150 classes.** Algorithm × Solver × Deflation × Preprocessing × Selection × Validation are configurable axes, not separate APIs.
- **AOM-PLS & POP-PLS as first-class citizens.** Adaptive Operator-Mixture PLS and Per-Operator-Per-Component PLS are the scientific differentiator (Phase 6).

See [`Overview.md`](Overview.md) for the full taxonomy of targeted PLS variants, and [`Direction_Technique.md`](Direction_Technique.md) for the canonical technical spec.

## Roadmap

`pls4all` is built in deliberate phases. Each phase has its own roadmap under [`roadmap/`](roadmap/) and a Codex-reviewed implementation log under [`docs/reviews/`](docs/reviews/).

| Phase | Status | Goal |
| --- | --- | --- |
| 0 — ABI & build foundation | shipped (`phase-0`) | Callable `libp4a` with stable C ABI, full CI matrix, parity scaffolding |
| 1 — PLS CPU reference | shipped (`phase-1`) | NIPALS PLS1 / PLS2, predict, transform, binary export/import |
| 3 — NIRS preprocessing & validation | in progress | pipeline fit/transform · identity · center · autoscale · Pareto · SNV · MSC · EMSC · detrend · SG · ASLS · Norris-Williams · Haar wavelet · OSC · EPO · regression metrics · validation splits · k-fold CV engine · advanced splitters · binary/multiclass classification metrics · calibration · VIP/selectivity ratio · coefficients by component |
| 4 — Advanced PLS variants | in progress | SIMPLS · SVD · PCR · kernel PLS · PLSCanonical · PLSSVD · OPLS · PLS-DA · PLS-LDA · PLS-logistic · MB-PLS · LW-PLS · component CV |
| 2 — Language bindings (real wheels / CRAN / AAR) | planned | ctypes (Python) · `.Call` (R) · MEX (MATLAB) · WASM (JS) · JNI (Android) |
| 5 — Variable selection | in progress | VIP/coefficient/selectivity-ratio rankers shipped · interval/moving-window CV scan shipped · biPLS backward-interval selector shipped · siPLS interval-combination selector shipped · coefficient-stability selector shipped · UVE artificial-variable selector shipped · EMCUVE ensemble selector shipped · randomization-test selector shipped · SPA selector shipped · CARS selector shipped · Random Frog selector shipped · SCARS selector shipped · GA-PLS selector shipped · shaving selector shipped · REP-PLS selector shipped · IPW-PLS selector shipped · ST-PLS selector shipped · BVE-PLS selector shipped · T2-PLS selector shipped · WVC-PLS selector shipped · thresholded/factor WVC selector shipped |
| 6 — AOM-PLS & POP-PLS | planned | Operator bank · soft / hard / sparse gating · per-component AOM |
| 7 — Accelerated backends | planned | BLAS · OpenMP · CUDA · batch APIs for CV / bootstrap / MCUVE / AOM sweeps |

Full roadmap: [`ROADMAP.md`](ROADMAP.md). Architecture rationale: [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Quick Build

```bash
git clone https://github.com/GBeurier/pls4all.git
cd pls4all
cmake --preset dev-release
cmake --build --preset dev-release --parallel
ctest --preset dev-release --output-on-failure

# Inspect the library:
./build/dev-release/cpp/cli/pls4all_cli --version
./build/dev-release/cpp/cli/pls4all_cli --abi-info
```

The core currently implements NIPALS, orthogonal-scores, SIMPLS, kernel, wide-kernel, SVD, power-iteration and randomized-SVD PLS regression, PLSCanonical, PLSSVD, PLS-DA, PLS-LDA, PLS-logistic, MB-PLS, LW-PLS, OPLS / OPLS-DA and PCR
behind the stable C ABI: `p4a_model_fit`, `predict`, `transform`, fitted-array
accessors and binary import/export are live for `P4A_ALGO_PLS_REGRESSION` with
`P4A_SOLVER_NIPALS`, `P4A_SOLVER_ORTHOGONAL_SCORES`, `P4A_SOLVER_SIMPLS`,
`P4A_SOLVER_KERNEL_ALGORITHM`, `P4A_SOLVER_WIDE_KERNEL`, `P4A_SOLVER_SVD`,
`P4A_SOLVER_POWER` or `P4A_SOLVER_RANDOMIZED_SVD`, for
`P4A_ALGO_PLS_CANONICAL` with `P4A_SOLVER_NIPALS` or `P4A_SOLVER_SVD` and
`P4A_DEFLATION_CANONICAL`, for `P4A_ALGO_PLS_SVD` with `P4A_SOLVER_SVD` and
`P4A_DEFLATION_CANONICAL`, for `P4A_ALGO_PLS_DA` with dummy-coded `Y` and
`P4A_DEFLATION_REGRESSION`, for `P4A_ALGO_OPLS` and `P4A_ALGO_OPLS_DA`
with one or more response columns, `P4A_SOLVER_NIPALS` and
`P4A_DEFLATION_ORTHOGONAL`, and for `P4A_ALGO_PCR` with `P4A_SOLVER_SVD`.
The preprocessing pipeline ABI is live for `P4A_OP_IDENTITY`, `P4A_OP_CENTER`,
`P4A_OP_AUTOSCALE`, `P4A_OP_PARETO_SCALE`, `P4A_OP_SNV`, `P4A_OP_MSC`,
`P4A_OP_EMSC`, `P4A_OP_DETREND_POLY`, `P4A_OP_SAVGOL_SMOOTH` and
`P4A_OP_SAVGOL_DERIVATIVE`, `P4A_OP_NORRIS_WILLIAMS`,
`P4A_OP_ASLS_BASELINE`, `P4A_OP_WAVELET_DENOISE`, `P4A_OP_OSC` and
`P4A_OP_EPO`, with explicit
fit/transform separation for leakage-safe downstream CV. For PLSSVD,
`transform` returns direct cross-covariance SVD X scores and
`predict` uses the deterministic latent projection `X @ W @ V.T` scaled back to
the response space.
The internal validation core now computes NumPy-parity regression metrics:
RMSE, MAE, bias, R2/Q2, observed-vs-predicted slope/intercept, RPD and RPIQ.
It also generates deterministic k-fold, leave-one-out, holdout, external-fold,
repeated k-fold, Monte-Carlo, Kennard-Stone and SPXY split plans, and can run
sklearn-parity out-of-sample k-fold regression cross-validation by refitting the
selected model on each fold and scoring the stitched predictions.
Binary classification metric kernels cover sensitivity, specificity, balanced
accuracy, accuracy, precision, F1, MCC and average-rank AUC for PLS-DA style
scores. Multiclass metric kernels add macro/micro averaging plus one-vs-rest
AUC, and calibration kernels produce fixed-bin binary reliability curves.
Variable-importance kernels compute VIP scores and selectivity ratio from fitted
models with stored scores, and Phase 5a variable-selection rankers expose
deterministic top-k ordering for VIP, original-scale coefficient magnitude and
selectivity-ratio scores. Phase 5b interval-selection kernels scan contiguous
feature windows with deterministic k-fold PLS CV for moving-window / iPLS-style
selection, and Phase 5p biPLS kernels perform deterministic backward
interval elimination by CV. Phase 5q siPLS kernels exhaustively score
fixed-size interval combinations by CV and retain the best synergistic subset.
Phase 5c stability-selection kernels compute Monte-Carlo
coefficient mean/std ratios for MCUVE-style ranking, Phase 5d UVE kernels
threshold real-variable stability against deterministic artificial variables,
and Phase 5n EMCUVE kernels ensemble MC-UVE members with a deterministic vote
rule. Phase 5o randomization-test kernels compare observed PLS coefficient
scores against deterministic Y permutations with empirical p-values. Phase 5e
SPA kernels seed selection from original-scale PLS coefficients and
then add projection-diverse variables. Phase 5f CARS kernels run deterministic
competitive adaptive reweighted sampling with exponential retention and
k-fold CV subset scoring. Phase 5g Random Frog kernels sample deterministic
add/remove/swap subset walks, accept by CV RMSE, and expose inclusion
frequencies. Phase 5h SCARS kernels combine deterministic calibration
subsampling, stability-weighted CARS retention and k-fold CV subset scoring.
Phase 5i GA-PLS kernels evolve deterministic subset populations with elitism,
crossover, mutation and k-fold CV fitness.
Phase 5j shaving kernels recursively eliminate low-score PLS variables and
score every retained subset by k-fold CV.
Phase 5s REP kernels remove a fixed number of weak coefficient-score variables
per recursive step and keep the lowest-CV-error retained subset.
Phase 5t IPW kernels iteratively reweight coefficient scores, expose score and
weight paths, and keep the lowest-CV-error top-k subset.
Phase 5u ST-PLS kernels apply deterministic score thresholds with min-selected
fallbacks and keep the lowest-CV-error threshold subset.
Phase 5k BVE kernels greedily test one-variable removals by k-fold CV RMSE and
keep the best backward path/subset.
Phase 5l T2-PLS kernels compute Hotelling T2 on PLS loading weights, apply
alpha-specific UCL thresholds with top-k fallback, and score subsets by k-fold CV.
Phase 5m WVC-PLS kernels compute normalized weighted-variable-contribution
scores from SVD PLS components and expose deterministic top-k selection.
Phase 5r WVC-threshold kernels apply fixed-threshold and factor-of-mean rules
with a minimum-selected fallback over the WVC score ranking.
Coefficient kernels materialise the original-scale
regression coefficients for every component prefix. Internal model-selection
kernels can score SIMPLS component prefixes by deterministic k-fold CV, and an
internal PLS-LDA and PLS-logistic kernels fit deterministic classifier scores
on PLS score spaces. The internal MB-PLS kernel fits block-autoscaled,
block-weighted PLS models and maps coefficients back to the original feature
space. The internal LW-PLS kernel performs stable k-nearest-neighbor local
window refits and records the neighbor plan used for every prediction.

## Project layout

```
pls4all/
├── cpp/                  # C++17 internal core + C ABI wrapper + tests + CLI
├── bindings/             # python · r · matlab · js · android  (each a thin FFI shim)
├── parity/               # JSON fixture schema + Python and R reference generators
├── cmake/                # Reusable CMake modules and toolchain helpers
├── packaging/            # PyPI wheel · CRAN package · npm · conda · vcpkg metadata
├── docs/                 # Architecture · ABI reference · binding guides · parity methodology
├── examples/             # Minimal C / Python / R hello-version programs
└── roadmap/              # Per-phase roadmaps reviewed by Codex before implementation
```

## Citation

If you use `pls4all` in academic work, please cite:

```
@software{pls4all,
  author  = {Beurier, Grégory and contributors},
  title   = {pls4all: A portable Partial Least Squares engine with a stable C ABI},
  year    = {2026},
  url     = {https://github.com/GBeurier/pls4all},
  version = {0.60.0}
}
```

A machine-readable [`CITATION.cff`](CITATION.cff) is provided. We also list the foundational PLS papers (Wold, de Jong, Trygg, Bylesjö, Rosipal, …) and the AOM-PLS / POP-PLS source publications there.

## License

CeCILL-2.1 — see [`LICENSE`](LICENSE). Compatible with the GPL family of licenses and recognised by French law.

## Acknowledgements

`pls4all` builds on lessons learned from the existing experimental [`aompls`](https://github.com/GBeurier/aompls) library and on the rich PLS ecosystem in scikit-learn, the R `pls` / `ropls` / `mixOmics` / `plsVarSel` packages, and `nirs4all`.
