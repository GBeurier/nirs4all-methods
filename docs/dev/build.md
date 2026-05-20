# Development — Build

Placeholder. The canonical guidance is in `CONTRIBUTING.md`; this page accumulates details (toolchain matrices, sanitizer use, release checklist) as the project stabilises.

## Optional FITPACK Spline Backend

`N4M_WITH_FITPACK` controls the vendored FITPACK backend used by
`aug_spline_smooth`.

* `AUTO` (default): enable FITPACK when a compatible Fortran compiler can be
  configured; otherwise build the ABI without a Fortran dependency.
* `ON`: require FITPACK and fail configuration if Fortran is unavailable or the
  generator is unsupported.
* `OFF`: compile without FITPACK. The public operator remains present for ABI
  stability and returns the input unchanged.

Linux CI exercises the FITPACK path. macOS and MSVC builds are allowed to use
the dependency-free fallback so wheel and ABI jobs do not require a Fortran
toolchain. The R package vendors the same sources and defines
`N4M_HAVE_FITPACK=1` through `Makevars`, relying on the Fortran compiler shipped
with the R build toolchain.

## Optional Numerical Backends: Eigen And Armadillo

The public C ABI should remain dependency-light and row-major. Eigen and
Armadillo are useful candidates for future internal kernels, but they should be
introduced as optional build-time backends rather than as public ABI types.

Eigen is the best fit for small and medium dense linear algebra kernels that
need to stay easy to distribute. It is header-only, has no runtime dependency,
and gives robust QR/SVD/eigenvalue implementations for code paths such as PCA,
MSC/EMSC least-squares fits, leverage diagnostics, Q residuals, and
calibration-transfer operators (direct standardisation, piecewise direct
standardisation, score-augmented projection standardisation, slope-bias
correction). It is a good candidate for a `N4M_ENABLE_EIGEN` backend when the
goal is portability, local optimisation, or replacing handwritten loops without
adding a BLAS/LAPACK requirement.

Armadillo is attractive when a higher-level MATLAB-like expression API or
BLAS/LAPACK-backed dense operations would speed up development. It is useful for
prototype kernels, reference-oriented development backends, and larger matrix
operations where the deployment target already provides BLAS/LAPACK. The tradeoff
is a heavier dependency surface and solver behaviour that may differ from the
canonical N4M contract; for example, normal-equation based least-squares paths
can drift on ill-conditioned local MSC windows. Armadillo should therefore be
parity-gated before being used as an optimised backend.

Policy for both backends:

* keep all Eigen/Armadillo objects behind internal translation-unit boundaries;
* expose only the existing C ABI buffers and status codes;
* make backend selection explicit through build flags;
* benchmark each enabled backend in the cross-binding dashboard;
* require binding parity and external-reference parity before changing the
  default numerical path;
* document any expected divergence caused by solver choice, component sign, RNG
  streams, or boundary handling.

Priority candidates are PCA/SVD projections, Hotelling T2 and Q residuals,
calibration-transfer least-squares kernels, EMSC/MSC variants, and wavelet/PCA
hybrid operators. Stochastic augmenters are lower priority unless the backend is
only used for deterministic substeps.

Each enabled backend must surface in the cross-binding dashboard with a
clearly labelled column (for example `n4m.cpp.<build>+eigen` or
`n4m.cpp.<build>+armadillo`) so divergence against the canonical
reference is visible alongside the default native column; the docs generators
in `docs/_extras/build_landing.py` and `docs/_extras/build_methods.py` recognise
any backend slug that follows the existing `n4m.cpp.*` /
`ref.<library>` conventions, so no manual table edit is required when a new
backend lands.

## Validation Registry And Docs Refresh

Parity and benchmark metadata are generated from a deterministic validation
registry. Refresh it before rebuilding the docs when `benchmarks/cross_binding`
or `benchmarks/benchmark_registry.json` changes:

```bash
python -m benchmarks.validation.export_registry --write
python -m benchmarks.validation.validate_registry
PYTHONPATH=. python -m pytest benchmarks/validation/tests -q
PYTHONPATH=docs/_extras:. python docs/_extras/build_landing.py --results benchmarks/cross_binding/results --out docs/_static/bench-data.json
PYTHONPATH=docs/_extras:. python docs/_extras/build_methods.py --strict
PYTHONPATH=docs/_extras:. python docs/_extras/method_benchmark_tables.py
sphinx-build -b html docs docs/_build/html
```

`benchmarks/validation/registry/*.json` is the stable hand-off between the
validation framework, the dashboard matrix, and the generated method
documentation. Method pages render a `Validation contract` section from that
snapshot, while the landing dashboard embeds the same data under the top-level
`validation` key in `docs/_static/bench-data.json`.
