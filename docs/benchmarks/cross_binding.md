# Cross-binding benchmark — parity + timing

> **Plaquette de pub** : pour chaque algorithme, on prouve que pls4all produit *exactement* le même résultat que la référence externe (parity), souvent plus vite (timing).

_Generated: 2026-05-17 13:36:06 UTC_  
_Host: Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.35_  
_CSV: `niveau_A_pls.csv` (429 cells)_


## Legend

- ✓  parity OK (max abs diff ≤ tolerance, default 1e-6)
- ⚠  parity within 1e-3 but > tolerance (algorithmic drift)
- ✗  parity mismatch (typically different preprocessing convention)
- ⏳  cell timed out (300s wall-clock)
- —  not implemented / skipped

Each cell: `<median_ms> <icon>`. Median is over 5 runs (warmup discarded).


## pls  —  1 thread

| size | cpp | python_tier1 | python_tier2 | sklearn | ikpls | r_tier1 | r_tier2 | r_pls | r_ropls | r_mixomics | matlab_tier1 | matlab_tier2 | matlab_pls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 0.92 ✓ | 0.92 ✓ | 1.05 ✓ | 1.40 ✓ | 1.16 ✗ | 2.50 ✓ | 6.00 ✗ | 6.50 ✓ | — — | 8.50 ✓ | 1.59 ✓ | 5.23 ✗ | 2.38 ✓ |
| 100×500 | 8.30 ✓ | 7.96 ✓ | 8.22 ✓ | 8.76 ✓ | 8.09 ✗ | 39.5 ✓ | 74.5 ✗ | 78.5 ✓ | — — | 67.5 ✓ | 12.9 ✓ | 17.2 ✗ | 14.2 ✓ |
| 100×2500 | 39.8 ✓ | 40.1 ✓ | 38.9 ✓ | 40.6 ✓ | 38.8 ⚠ | 323.0 ✓ | 645.0 ⚠ | 635.0 ✓ | — — | 442.5 ✓ | 65.0 ✓ | 68.0 ⚠ | 66.7 ✓ |
| 500×50 | 4.05 ✓ | 4.16 ✓ | 4.30 ✓ | 4.31 ✓ | 4.14 ✗ | 11.5 ✓ | 16.0 ✗ | 16.0 ✓ | — — | 27.5 ✓ | 6.85 ✓ | 10.2 ✗ | 7.33 ✓ |
| 500×500 | 39.7 ✓ | 38.4 ✓ | 38.2 ✓ | 40.6 ✓ | 38.3 ✗ | 201.0 ✓ | 244.0 ✗ | 245.0 ✓ | — — | 295.5 ✓ | 63.4 ✓ | 69.1 ✗ | 65.6 ✓ |
| 500×2500 | 191.2 ✓ | 191.0 ✓ | 182.1 ✓ | 192.2 ✓ | 181.5 ✗ | 1.4s ✓ | 1.8s ✗ | 1.7s ✓ | — — | 1.5s ✓ | 321.0 ✓ | 317.4 ✗ | 355.1 ✓ |
| 2500×50 | 20.0 ✓ | 19.7 ✓ | 19.5 ✓ | 20.0 ✓ | 18.9 ⚠ | 72.5 ✓ | 74.5 ⚠ | 83.5 ✓ | — — | 105.5 ✓ | 30.6 ✓ | 34.4 ⚠ | 33.7 ✓ |
| 2500×500 | 219.2 ✓ | 213.2 ✓ | 207.5 ✓ | 189.8 ✓ | 177.6 ✗ | 1.3s ✓ | 1.4s ✗ | 1.4s ✓ | — — | 1.3s ✓ | 334.9 ✓ | 350.7 ✗ | 335.5 ✓ |
| 2500×2500 | 1.1s ✓ | 1.1s ✓ | 1.1s ✓ | 1.0s ✓ | 930.1 ✗ | 7.6s ✓ | 9.1s ✗ | 8.3s ✓ | — — | 7.9s ✓ | 1.9s ✓ | 1.9s ✗ | 2.0s ✓ |
| 10000×50 | 78.0 ✓ | 77.2 ✓ | 76.4 ✓ | 77.5 ✓ | 73.9 ⚠ | 418.5 ✓ | 428.5 ⚠ | 385.5 ✓ | — — | 530.5 ✓ | 120.3 ✓ | 128.0 ⚠ | 129.9 ✓ |
| 10000×500 | 890.4 ✓ | 896.0 ✓ | 876.4 ✓ | 832.9 ✓ | 759.2 ⚠ | 6.3s ✓ | 6.5s ⚠ | 6.5s ✓ | — — | 6.7s ✓ | 1.4s ✓ | 1.4s ⚠ | 1.6s ✓ |


## pls  —  3 threads

| size | cpp | python_tier1 | python_tier2 | sklearn | ikpls | r_tier1 | r_tier2 | r_pls | r_ropls | r_mixomics | matlab_tier1 | matlab_tier2 | matlab_pls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 0.90 ✓ | 0.86 ✓ | 1.04 ✓ | 1.34 ✓ | 1.10 ✗ | 2.50 ✓ | 6.00 ✗ | 6.50 ✓ | — — | 8.00 ✓ | 1.64 ✓ | 5.18 ✗ | 2.20 ✓ |
| 100×500 | 7.90 ✓ | 7.76 ✓ | 8.13 ✓ | 9.03 ✓ | 8.24 ✗ | 38.5 ✓ | 76.5 ✗ | 78.0 ✓ | — — | 56.0 ✓ | 13.1 ✓ | 16.3 ✗ | 14.1 ✓ |
| 100×2500 | 39.3 ✓ | 38.6 ✓ | 38.5 ✓ | 39.4 ✓ | 38.1 ⚠ | 326.5 ✓ | 632.5 ⚠ | 626.0 ✓ | — — | 471.0 ✓ | 63.4 ✓ | 69.8 ⚠ | 67.1 ✓ |
| 500×50 | 4.19 ✓ | 4.13 ✓ | 4.03 ✓ | 4.28 ✓ | 4.09 ✗ | 12.5 ✓ | 15.0 ✗ | 17.5 ✓ | — — | 26.0 ✓ | 6.65 ✓ | 9.57 ✗ | 7.90 ✓ |
| 500×500 | 39.1 ✓ | 39.3 ✓ | 38.6 ✓ | 38.4 ✓ | 36.9 ✗ | 186.5 ✓ | 248.0 ✗ | 238.5 ✓ | — — | 317.0 ✓ | 63.9 ✓ | 69.0 ✗ | 67.6 ✓ |
| 500×2500 | 199.8 ✓ | 195.2 ✓ | 188.6 ✓ | 197.7 ✓ | 180.2 ✗ | 1.5s ✓ | 1.7s ✗ | 1.7s ✓ | — — | 1.4s ✓ | 319.0 ✓ | 320.6 ✗ | 352.6 ✓ |
| 2500×50 | 20.1 ✓ | 19.6 ✓ | 19.2 ✓ | 20.1 ✓ | 18.4 ⚠ | 72.0 ✓ | 77.5 ⚠ | 82.5 ✓ | — — | 107.5 ✓ | 31.0 ✓ | 34.7 ⚠ | 34.9 ✓ |
| 2500×500 | 219.8 ✓ | 217.7 ✓ | 211.8 ✓ | 196.2 ✓ | 189.2 ✗ | 1.3s ✓ | 1.4s ✗ | 1.4s ✓ | — — | 1.4s ✓ | 342.8 ✓ | 361.2 ✗ | 327.2 ✓ |
| 2500×2500 | 1.1s ✓ | 1.1s ✓ | 1.1s ✓ | 1.0s ✓ | 967.7 ✗ | 7.9s ✓ | 9.3s ✗ | 8.9s ✓ | — — | 9.6s ✓ | 2.2s ✓ | 2.1s ✗ | 2.5s ✓ |
| 10000×50 | 78.1 ✓ | 75.3 ✓ | 80.0 ✓ | 84.6 ✓ | 78.5 ⚠ | 438.0 ✓ | 425.5 ⚠ | 387.5 ✓ | — — | 554.5 ✓ | 119.5 ✓ | 132.6 ⚠ | 138.8 ✓ |
| 10000×500 | 896.7 ✓ | 899.9 ✓ | 904.9 ✓ | 829.4 ✓ | 752.5 ⚠ | 6.3s ✓ | 6.6s ⚠ | 6.6s ✓ | — — | 7.1s ✓ | 1.5s ✓ | 1.5s ⚠ | 1.6s ✓ |


## pls  —  10 threads

| size | cpp | python_tier1 | python_tier2 | sklearn | ikpls | r_tier1 | r_tier2 | r_pls | r_ropls | r_mixomics | matlab_tier1 | matlab_tier2 | matlab_pls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 0.86 ✓ | 0.89 ✓ | 1.06 ✓ | 1.46 ✓ | 1.12 ✗ | 3.00 ✓ | 5.50 ✗ | 5.50 ✓ | — — | 9.00 ✓ | 1.66 ✓ | 5.68 ✗ | 2.32 ✓ |
| 100×500 | 7.76 ✓ | 7.87 ✓ | 7.93 ✓ | 8.72 ✓ | 8.05 ✗ | 37.0 ✓ | 75.5 ✗ | 77.0 ✓ | — — | 59.0 ✓ | 13.4 ✓ | 16.7 ✗ | 14.1 ✓ |
| 100×2500 | 39.6 ✓ | 37.9 ✓ | 38.5 ✓ | 39.6 ✓ | 38.0 ⚠ | 331.5 ✓ | 648.0 ⚠ | 623.5 ✓ | — — | 467.5 ✓ | 64.8 ✓ | 70.4 ⚠ | 68.4 ✓ |
| 500×50 | 4.06 ✓ | 4.07 ✓ | 3.90 ✓ | 4.31 ✓ | 4.10 ✗ | 10.5 ✓ | 15.5 ✗ | 17.5 ✓ | — — | 27.0 ✓ | 6.71 ✓ | 9.56 ✗ | 7.86 ✓ |
| 500×500 | 39.8 ✓ | 39.1 ✓ | 37.7 ✓ | 40.7 ✓ | 39.3 ✗ | 203.0 ✓ | 247.0 ✗ | 253.0 ✓ | — — | 277.5 ✓ | 59.9 ✓ | 68.1 ✗ | 66.1 ✓ |
| 500×2500 | 190.7 ✓ | 187.1 ✓ | 178.1 ✓ | 241.5 ✓ | 221.1 ✗ | 1.5s ✓ | 1.7s ✗ | 1.8s ✓ | — — | 1.6s ✓ | 320.9 ✓ | 355.1 ✗ | 404.2 ✓ |
| 2500×50 | 20.2 ✓ | 19.8 ✓ | 20.7 ✓ | 21.0 ✓ | 19.0 ⚠ | 70.5 ✓ | 79.5 ⚠ | 81.5 ✓ | — — | 108.0 ✓ | 31.6 ✓ | 34.8 ⚠ | 33.6 ✓ |
| 2500×500 | 220.1 ✓ | 218.6 ✓ | 208.6 ✓ | 235.9 ✓ | 222.3 ✗ | 1.3s ✓ | 1.4s ✗ | 1.5s ✓ | — — | 1.4s ✓ | 329.9 ✓ | 383.2 ✗ | 361.9 ✓ |
| 2500×2500 | 1.4s ✓ | 1.4s ✓ | 1.2s ✓ | 1.1s ✓ | 1.0s ✗ | 8.0s ✓ | 10.0s ✗ | 9.1s ✓ | — — | 8.1s ✓ | 1.9s ✓ | 1.9s ✗ | 2.2s ✓ |
| 10000×50 | 76.8 ✓ | 78.3 ✓ | 76.4 ✓ | 124.8 ✓ | 112.6 ⚠ | 431.5 ✓ | 418.5 ⚠ | 446.0 ✓ | — — | 648.5 ✓ | 119.9 ✓ | 168.8 ⚠ | 169.1 ✓ |
| 10000×500 | 963.2 ✓ | 980.0 ✓ | 932.3 ✓ | 943.2 ✓ | 823.9 ⚠ | 6.8s ✓ | 7.8s ⚠ | 7.9s ✓ | — — | 7.4s ✓ | 1.5s ✓ | 1.7s ⚠ | 2.0s ✓ |


## Backend versions

| backend | versions |
|---|---|
| cpp | `language=C++ (via ctypes bridge) (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1; libopenblas 0.3.33 blas=1; libomp ? openmp=1`; `libp4a=0.94.0+abi.1.13.0`; `numpy=2.3.5` |
| python_tier1 | `language=Python 3.13.11`; `blas=libscipy_openblas 0.3.30 blas=1`; `pls4all=0.97.0`; `numpy=2.3.5` |
| python_tier2 | `language=Python 3.13.11`; `blas=libscipy_openblas 0.3.30 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `pls4all=0.97.0`; `numpy=2.3.5` |
| sklearn | `language=Python 3.13.11`; `blas=libscipy_openblas 0.3.30 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `sklearn=1.8.0`; `numpy=2.3.5` |
| ikpls | `language=Python 3.13.11`; `blas=libscipy_openblas 0.3.30 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `ikpls=4.0.1.post1`; `numpy=2.3.5` |
| r_tier1 | `language=R 4.3.3`; `pls4all=0.68.0`; `blas=linked-BLAS (RhpcBLASctl not installed)` |
| r_tier2 | `language=R 4.3.3`; `pls4all=0.68.0`; `blas=linked-BLAS` |
| r_pls | `language=R 4.3.3`; `pls=2.8.5` |
| r_ropls | `language=R 4.3.3`; `ropls=1.34.0` |
| r_mixomics | `language=R 4.3.3`; `mixOmics=6.26.0` |
| matlab_tier1 | `language=Octave 10.3.0`; `pls4all=from libp4a-linked MEX`; `blas=linked-BLAS` |
| matlab_tier2 | `language=Octave 10.3.0`; `pls4all=from libp4a-linked MEX + classdefs`; `blas=linked-BLAS` |
| matlab_pls | `language=Octave 10.3.0`; `statistics=Octave statistics pkg (plsregress)`; `blas=linked-BLAS` |

## Methodology

- Reference: `cpp` cell at 1 thread (libp4a via ctypes), or `python_tier1` when `cpp` is unavailable for an algorithm
- Parity tolerance: 1e-6 (per-algo overrides possible)
- All backends read the **same** orchestrator-generated CSV (`benchmarks/cross_binding/data/data_<n>x<p>_seed<seed>.csv`) so input data is bit-identical across languages
- 5 runs per cell, first discarded as warmup, median reported
- Per-cell timeout: 300 s
- Thread control via `OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = BLIS_NUM_THREADS` set in the subprocess env, plus `Context.num_threads` for Python pls4all and `maxNumCompThreads()` for Octave
- pls4all libp4a build: `build/blas-omp/cpp/src/libp4a.so` (BLAS + OpenMP enabled)
