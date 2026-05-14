# pls4all

> A portable PLS / NIRS engine with a stable C ABI and thin first-class bindings for Python, R, MATLAB, JavaScript / WebAssembly and Android.

**Status — Phase 1 shipped, SIMPLS, kernel, wide-kernel, SVD and PCR core live · API unstable until v1.0 · Not yet on PyPI / CRAN / npm.**

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
| 3 — NIRS preprocessing & validation | planned | SNV / MSC / SG / Detrend / ASLS · leakage-safe CV · VIP · RMSEC/RMSEP/Q²/RPD/RPIQ |
| 4 — Advanced PLS variants | in progress | SIMPLS · SVD · PCR · kernel PLS · OPLS · PLS-DA · MB-PLS · LW-PLS |
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

The core currently implements NIPALS, SIMPLS, kernel, wide-kernel and SVD PLS regression plus PCR
behind the stable C ABI: `p4a_model_fit`, `predict`, `transform`, fitted-array
accessors and binary import/export are live for `P4A_ALGO_PLS_REGRESSION` with
`P4A_SOLVER_NIPALS`, `P4A_SOLVER_SIMPLS`, `P4A_SOLVER_KERNEL_ALGORITHM`,
`P4A_SOLVER_WIDE_KERNEL` or `P4A_SOLVER_SVD`, and for `P4A_ALGO_PCR` with
`P4A_SOLVER_SVD`.

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
  version = {0.7.0}
}
```

A machine-readable [`CITATION.cff`](CITATION.cff) is provided. We also list the foundational PLS papers (Wold, de Jong, Trygg, Bylesjö, Rosipal, …) and the AOM-PLS / POP-PLS source publications there.

## License

CeCILL-2.1 — see [`LICENSE`](LICENSE). Compatible with the GPL family of licenses and recognised by French law.

## Acknowledgements

`pls4all` builds on lessons learned from the existing experimental [`aompls`](https://github.com/GBeurier/aompls) library and on the rich PLS ecosystem in scikit-learn, the R `pls` / `ropls` / `mixOmics` / `plsVarSel` packages, and `nirs4all`.
