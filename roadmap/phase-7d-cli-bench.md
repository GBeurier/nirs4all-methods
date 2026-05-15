# Phase 7d - CLI native bench mode

Status: shipped locally as `phase-7-comprehensive-benchmark`.

Goal: give the matrix runner a clean way to read pure C++ kernel
timings without any binding overhead. Add a `--bench` subcommand to
`pls4all_cli` that generates a deterministic synthetic dataset
internally, runs `p4a_model_fit + p4a_model_predict`, and prints a
single parseable CSV row.

Delivered:

- New CLI subcommand: `pls4all_cli --bench --algo NAME --samples N
  --features P [--components K] [--repeats R] [--seed S]`.
- Internal SplitMix64 PRNG so the CLI has no random dependency.
- Latent-factor synthetic dataset matching the Python benchmark
  runner's structure (same seed yields equivalent statistical
  properties; the actual sample values differ because the C SplitMix64
  is independent of NumPy's RNG, but the timing characteristics are
  representative).
- Algorithm names mirror the Python benchmark runner's identifiers:
  `pls_nipals`, `pls_orthogonal_scores`, `pls_simpls`,
  `pls_kernel_algorithm`, `pls_wide_kernel`, `pls_svd`, `pls_power`,
  `pls_randomized_svd`, `pcr_svd`.
- Output format: a `HEAD,...` header row followed by a `DATA,...`
  row. The matrix runner greps for `DATA,` and parses the median /
  min fit + predict times.

Local sanity check (1000 samples × 100 features × 5 components):

| algo                  | fit ms (median) | predict ms (median) |
|-----------------------|-----------------|---------------------|
| pls_nipals            | 6.93            | 0.49                |
| pls_orthogonal_scores | 7.20            | 0.50                |
| pls_simpls            | 6.71            | 0.51                |
| pls_kernel_algorithm  | 226.9           | 0.35                |
| pls_wide_kernel       | 149.0           | 0.23                |
| pls_svd               | 6.81            | 0.51                |
| pls_power             | 1.70            | 0.35                |
| pls_randomized_svd    | 2.21            | 0.47                |
| pcr_svd               | 499.9           | 0.39                |

All solvers produce the same RMSE (2.266649e-02) confirming they
converge to the same predictor.
