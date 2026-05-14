# pls4all

> A portable PLS / NIRS engine with a stable C ABI and thin first-class bindings for Python, R, MATLAB, JavaScript / WebAssembly and Android.

**Status — Phase 1 shipped, Phase 3i preprocessing pipeline live, PLSCanonical, PLSSVD, PLS-DA, OPLS / OPLS-DA, orthogonal-scores, SIMPLS, kernel, wide-kernel, SVD, power, randomized-SVD and PCR core live · API unstable until v1.0 · Not yet on PyPI / CRAN / npm.**

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
| 3 — NIRS preprocessing & validation | in progress | pipeline fit/transform · identity · center · autoscale · Pareto · SNV · MSC · EMSC · detrend · SG · ASLS · Norris-Williams · Haar wavelet · OSC |
| 4 — Advanced PLS variants | in progress | SIMPLS · SVD · PCR · kernel PLS · PLSCanonical · PLSSVD · OPLS · PLS-DA · MB-PLS · LW-PLS |
| 2 — Language bindings (real wheels / CRAN / AAR) | planned | ctypes (Python) · `.Call` (R) · MEX (MATLAB) · WASM (JS) · JNI (Android) |
| 5 — Variable selection | planned | VIP · iPLS · MCUVE · CARS · SPA · Random Frog · GA-PLS |
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

The core currently implements NIPALS, orthogonal-scores, SIMPLS, kernel, wide-kernel, SVD, power-iteration and randomized-SVD PLS regression, PLSCanonical, PLSSVD, PLS-DA, OPLS / OPLS-DA and PCR
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
`P4A_OP_ASLS_BASELINE`, `P4A_OP_WAVELET_DENOISE` and `P4A_OP_OSC`, with explicit
fit/transform separation for leakage-safe downstream CV. For PLSSVD,
`transform` returns direct cross-covariance SVD X scores and
`predict` uses the deterministic latent projection `X @ W @ V.T` scaled back to
the response space.

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
  version = {0.25.0}
}
```

A machine-readable [`CITATION.cff`](CITATION.cff) is provided. We also list the foundational PLS papers (Wold, de Jong, Trygg, Bylesjö, Rosipal, …) and the AOM-PLS / POP-PLS source publications there.

## License

CeCILL-2.1 — see [`LICENSE`](LICENSE). Compatible with the GPL family of licenses and recognised by French law.

## Acknowledgements

`pls4all` builds on lessons learned from the existing experimental [`aompls`](https://github.com/GBeurier/aompls) library and on the rich PLS ecosystem in scikit-learn, the R `pls` / `ropls` / `mixOmics` / `plsVarSel` packages, and `nirs4all`.
