# Performance matrix

Algorithm × dataset × language path. Accuracy compares the
pls4all language paths against scikit-learn; timing is
informational and depends on host CPU pinning + BLAS threads.

## Metadata

- pls4all: `0.67.0+abi.1.1.0`
- scale: `smoke`
- repeats: `1`
- R available: `False`
- CLI available: `True`

Host:
- python: `3.13.11`
- platform: `Linux-6.6.114.1-microsoft-standard-WSL2-x86_64-with-glibc2.35`
- processor: `x86_64`
- logical cores: `24`

Thread pinning environment:
- `OMP_NUM_THREADS`: `(unset)`
- `OPENBLAS_NUM_THREADS`: `(unset)`
- `MKL_NUM_THREADS`: `(unset)`
- `PLS4ALL_BENCH_THREADS`: `(unset)`

## Timing (ms, median across repeats)

| Case | Algo | Native C++ | Python | R | sklearn |
|------|------|------------|--------|---|---------|
| matrix-200x100 | pls_simpls | 0.444 | 0.774 | (r_not_available) | 7.453 |
| matrix-200x100 | pls_svd | 0.641 | 0.848 | (r_not_available) | 7.534 |
| matrix-500x100 | pls_simpls | 1.312 | 0.8 | (r_not_available) | 10.809 |
| matrix-500x100 | pls_svd | 3.541 | 1.123 | (r_not_available) | 6.625 |

## Accuracy gate

| Case | Algo | Python pass | R pass | RMSE abs (Python vs sklearn) |
|------|------|-------------|--------|------------------------------|
| matrix-200x100 | pls_simpls | y | n/a | 0.0 |
| matrix-200x100 | pls_svd | y | n/a | 0.0 |
| matrix-500x100 | pls_simpls | y | n/a | 0.0 |
| matrix-500x100 | pls_svd | y | n/a | 0.0 |
