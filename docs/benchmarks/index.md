# Benchmarks

pls4all is benchmarked across three axes:

1. **Cross-binding parity + timing** — same algorithm, same data,
   pls4all vs the leading external implementation in each language
   (sklearn, pls::plsr, plsregress, ropls, mixOmics, ikpls). For each
   `(algo, n, p, threads)` cell we report the median wall-clock time
   AND the max-abs-diff of predictions vs a reference backend.
   This is the **"plaquette de pub"**: same result, often faster.

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
# Niveau A — PLS SIMPLS, headline matrix (11 sizes × 3 threads)
python benchmarks/cross_binding/orchestrator.py \
  --algorithms pls --threads 1 3 10 --n-runs 5 \
  --libp4a-build blas-omp \
  --out-csv benchmarks/cross_binding/results/niveau_A_pls.csv

# Render to markdown
python benchmarks/cross_binding/render_docs.py \
  --csv benchmarks/cross_binding/results/niveau_A_pls.csv \
  --out docs/benchmarks/cross_binding.md
```

See [methodology.md](methodology.md) for the reference policy, parity
tolerances, threading conventions and hardware context.
