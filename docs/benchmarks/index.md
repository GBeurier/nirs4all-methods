# Benchmarks

pls4all is benchmarked across three axes. The benchmark matrix carries
two separate parity gates: **binding parity** checks pls4all bindings
against the native C++ row, while **reference parity** checks every
successful implementation against the registry-declared external oracle.

1. **Cross-binding parity + timing** — same algorithm, same data,
   pls4all bindings and external references in one matrix. For each
   `(algo, n, p, threads)` cell we report adaptive wall-clock time and
   the relevant visible parity verdict: reference parity for C++/external
   rows, binding parity for internal binding rows.

2. **Accelerated backend timing** — same SIMPLS, libp4a built five
   ways (`ref`, `blas`, `omp`, `blas_omp`, `cuda`). Shows the speedup
   stack at the C++ layer.

3. **Algorithm matrix** — pls4all's ~60 algorithms timed at
   representative sizes, with accuracy verified against the
   reference implementation per algorithm.

## Pages

- [Cross-binding parity + timing](cross_binding.md)
- [Overview](overview.md)
- [Methodology](methodology.md)

## Reproducing the benchmarks

```bash
# Canonical method/reference matrix, including build + docs render.
# Existing cells in results/full_matrix.csv are skipped by default.
benchmarks/cross_binding/run_overnight.sh

# Exhaustive stress matrix with registry-declared references.
FULL_MATRIX=1 REFERENCE_BACKENDS=registry benchmarks/cross_binding/run_overnight.sh

# Legacy fixed/all external-reference audit. NOT_IMPLEMENTED rows are expected.
FULL_MATRIX=1 REFERENCE_BACKENDS=all benchmarks/cross_binding/run_overnight.sh

# Include CUDA too when that preset is available.
FULL_MATRIX=1 LIBP4A_BUILD=all benchmarks/cross_binding/run_overnight.sh

# On the Pages branch (main), also publish the refreshed dashboard.
PUBLISH_WEB=1 benchmarks/cross_binding/run_overnight.sh

# Exhaustive run, then publish the refreshed dashboard from main.
FULL_MATRIX=1 PUBLISH_WEB=1 benchmarks/cross_binding/run_overnight.sh

# From a work branch, commit/push dashboard sources without live deploy.
PUBLISH_WEB=1 DEPLOY_PAGES=0 benchmarks/cross_binding/run_overnight.sh

# Render an existing CSV only.
python benchmarks/cross_binding/combine_and_render.py \
  --csvs benchmarks/cross_binding/results/full_matrix.csv \
  --out docs/benchmarks/cross_binding.md
```

See [overview.md](overview.md) for the gate semantics and
[methodology.md](methodology.md) for run mechanics, tolerances,
threading conventions and hardware context.
