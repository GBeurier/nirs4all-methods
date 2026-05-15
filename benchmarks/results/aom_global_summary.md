# AOM-PLS global selection benchmark

Deterministic accuracy comparison between the pls4all public C ABI and
the local `nirs4all/bench/AOM_v0/aompls` oracle. Same data, same folds.

- `pls4all` version: `0.66.0+abi.1.1.0`
- bench oracle version: `0.1.0`

## Accuracy (gated)

```
+------------+----------------------------------+-----------+----------------+----------+---------+------+
| benchmark  | case                             | rmse_abs  | best_score_abs | op_match | k_match | pass |
+------------+----------------------------------+-----------+----------------+----------+---------+------+
| aom_global | aom-global-9x6-ncomp3-cv3        | 0.000e+00 | 0.000e+00      | y        | y       | y    |
| aom_global | aom-global-12x8-ncomp4-cv3       | 0.000e+00 | 0.000e+00      | y        | y       | y    |
| aom_global | aom-global-16x10-ncomp4-cv4      | 0.000e+00 | 0.000e+00      | y        | y       | y    |
| aom_global | aom-global-detrend-favoured-14x9 | 0.000e+00 | 0.000e+00      | y        | y       | y    |
+------------+----------------------------------+-----------+----------------+----------+---------+------+
```

## Timing (informational, NOT gated)

- python: `3.13.11`
- platform: `Linux-6.6.114.1-microsoft-standard-WSL2-x86_64-with-glibc2.35`
- processor: `x86_64`
- machine: `x86_64`
- logical cores: `24`

Wall-clock median is the wall time of one full `fit + predict` call;
pls4all's number includes Python list <-> ctypes marshalling overhead.

| Case | pls4all (ms, median / min) | bench (ms, median / min) | speedup |
|------|----------------------------|--------------------------|---------|
| aom-global-9x6-ncomp3-cv3 | 0.070 / 0.058 | 1.009 / 0.971 | 14.316 |
| aom-global-12x8-ncomp4-cv3 | 0.130 / 0.114 | 2.198 / 1.280 | 16.951 |
| aom-global-16x10-ncomp4-cv4 | 0.197 / 0.146 | 1.545 / 1.465 | 7.836 |
| aom-global-detrend-favoured-14x9 | 0.085 / 0.077 | 1.584 / 1.488 | 18.642 |
