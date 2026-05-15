# PLS regression benchmark

## Metadata

- `pls4all_version`: `0.67.0+abi.1.1.0`
- `sklearn_version`: `1.8.0`
- `numpy_version`: `2.4.4`
- `host_threads`: `24`

## Accuracy (gated)

```
+----------------+-------------------------------------+-----------+----------------+----------+---------+------+
| benchmark      | case                                | rmse_abs  | best_score_abs | op_match | k_match | pass |
+----------------+-------------------------------------+-----------+----------------+----------+---------+------+
| pls_regression | smoke-200x100-pls_nipals            | 0.000e+00 | 0.000e+00      | y        | y       | y    |
| pls_regression | smoke-200x100-pls_orthogonal_scores | 0.000e+00 | 0.000e+00      | y        | y       | y    |
| pls_regression | smoke-200x100-pls_simpls            | 0.000e+00 | 0.000e+00      | y        | y       | y    |
| pls_regression | smoke-200x100-pls_kernel_algorithm  | 0.000e+00 | 0.000e+00      | y        | y       | y    |
| pls_regression | smoke-200x100-pls_wide_kernel       | 0.000e+00 | 0.000e+00      | y        | y       | y    |
| pls_regression | smoke-200x100-pls_svd               | 0.000e+00 | 0.000e+00      | y        | y       | y    |
| pls_regression | smoke-200x100-pls_power             | 0.000e+00 | 0.000e+00      | y        | y       | y    |
| pls_regression | smoke-200x100-pls_randomized_svd    | 0.000e+00 | 0.000e+00      | y        | y       | y    |
| pls_regression | smoke-200x100-pcr_svd               | 2.923e-05 | 1.534e-06      | y        | y       | y    |
| pls_regression | smoke-500x100-pls_nipals            | 0.000e+00 | 0.000e+00      | y        | y       | y    |
| pls_regression | smoke-500x100-pls_orthogonal_scores | 0.000e+00 | 0.000e+00      | y        | y       | y    |
| pls_regression | smoke-500x100-pls_simpls            | 0.000e+00 | 0.000e+00      | y        | y       | y    |
| pls_regression | smoke-500x100-pls_kernel_algorithm  | 0.000e+00 | 0.000e+00      | y        | y       | y    |
| pls_regression | smoke-500x100-pls_wide_kernel       | 0.000e+00 | 0.000e+00      | y        | y       | y    |
| pls_regression | smoke-500x100-pls_svd               | 0.000e+00 | 0.000e+00      | y        | y       | y    |
| pls_regression | smoke-500x100-pls_power             | 0.000e+00 | 0.000e+00      | y        | y       | y    |
| pls_regression | smoke-500x100-pls_randomized_svd    | 0.000e+00 | 0.000e+00      | y        | y       | y    |
| pls_regression | smoke-500x100-pcr_svd               | 1.913e-06 | 6.146e-07      | y        | y       | y    |
+----------------+-------------------------------------+-----------+----------------+----------+---------+------+
```

## Timing (informational, NOT gated)

- python: `3.13.11`
- platform: `Linux-6.6.114.1-microsoft-standard-WSL2-x86_64-with-glibc2.35`
- processor: `x86_64`
- machine: `x86_64`
- logical cores: `24`

| Case | pls4all (ms, median / min) | reference (ms, median / min) | speedup |
|------|----------------------------|------------------------------|---------|
| smoke-200x100-pls_nipals | 0.678 / 0.678 | 3.089 / 3.089 | 4.555 |
| smoke-200x100-pls_orthogonal_scores | 0.577 / 0.577 | 3.089 / 3.089 | 5.357 |
| smoke-200x100-pls_simpls | 0.345 / 0.345 | 3.089 / 3.089 | 8.954 |
| smoke-200x100-pls_kernel_algorithm | 4.261 / 4.261 | 3.089 / 3.089 | 0.725 |
| smoke-200x100-pls_wide_kernel | 2.893 / 2.893 | 3.089 / 3.089 | 1.068 |
| smoke-200x100-pls_svd | 0.219 / 0.219 | 3.089 / 3.089 | 14.094 |
| smoke-200x100-pls_power | 0.206 / 0.206 | 3.089 / 3.089 | 15.021 |
| smoke-200x100-pls_randomized_svd | 0.188 / 0.188 | 3.089 / 3.089 | 16.429 |
| smoke-200x100-pcr_svd | 250.939 / 250.939 | 3.089 / 3.089 | 0.012 |
| smoke-500x100-pls_nipals | 0.912 / 0.912 | 1.340 / 1.340 | 1.469 |
| smoke-500x100-pls_orthogonal_scores | 0.478 / 0.478 | 1.340 / 1.340 | 2.803 |
| smoke-500x100-pls_simpls | 0.343 / 0.343 | 1.340 / 1.340 | 3.910 |
| smoke-500x100-pls_kernel_algorithm | 20.415 / 20.415 | 1.340 / 1.340 | 0.066 |
| smoke-500x100-pls_wide_kernel | 17.964 / 17.964 | 1.340 / 1.340 | 0.075 |
| smoke-500x100-pls_svd | 0.511 / 0.511 | 1.340 / 1.340 | 2.621 |
| smoke-500x100-pls_power | 0.499 / 0.499 | 1.340 / 1.340 | 2.682 |
| smoke-500x100-pls_randomized_svd | 0.510 / 0.510 | 1.340 / 1.340 | 2.629 |
| smoke-500x100-pcr_svd | 268.782 / 268.782 | 1.340 / 1.340 | 0.005 |
