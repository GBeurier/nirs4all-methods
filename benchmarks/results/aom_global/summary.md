# AOM-PLS global selection benchmark

## Metadata

- `pls4all_version`: `0.67.0+abi.1.1.0`
- `bench_version`: `0.1.0`

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

| Case | pls4all (ms, median / min) | reference (ms, median / min) | speedup |
|------|----------------------------|------------------------------|---------|
| aom-global-9x6-ncomp3-cv3 | 0.297 / 0.297 | 2.516 / 2.516 | 8.472 |
| aom-global-12x8-ncomp4-cv3 | 0.308 / 0.308 | 2.624 / 2.624 | 8.519 |
| aom-global-16x10-ncomp4-cv4 | 0.450 / 0.450 | 3.535 / 3.535 | 7.857 |
| aom-global-detrend-favoured-14x9 | 0.358 / 0.358 | 1.801 / 1.801 | 5.032 |
