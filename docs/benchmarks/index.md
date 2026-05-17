# Benchmarks

pls4all is benchmarked across three axes:

1. **Cross-binding parity + timing** — same algorithm, same data,
   pls4all vs the leading external implementation in each language
   (sklearn, pls::plsr, plsregress, ropls, mixOmics, ikpls). For each
   `(algo, n, p, threads)` cell we report the median wall-clock time
   in milliseconds AND the max absolute difference of predictions
   vs a reference backend.

2. **Accelerated backend timing** — same SIMPLS, libp4a built five
   ways (`ref`, `blas`, `omp`, `blas_omp`, `cuda`). Shows the speedup
   stack at the C++ layer.

3. **Algorithm matrix** — pls4all's ~60 algorithms timed at
   representative sizes, with accuracy verified against the
   reference implementation per algorithm.

## Pages

- [Cross-binding parity + timing](cross_binding.md)
- [Methodology](methodology.md)

## Reproducing the benchmarks

```bash
# Canonical method/reference matrix, including build + docs render.
# Existing cells in results/full_matrix.csv are skipped by default.
benchmarks/cross_binding/run_overnight.sh

# Exhaustive stress matrix: all pls4all bindings/tiers, all CPU libp4a
# backend variants, all default dataset sizes, threads 1/3/10, and all
# fixed + registry Python/R/MATLAB references.
FULL_MATRIX=1 benchmarks/cross_binding/run_overnight.sh

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

See [methodology.md](methodology.md) for the reference policy, parity
tolerances, threading conventions and hardware context.
