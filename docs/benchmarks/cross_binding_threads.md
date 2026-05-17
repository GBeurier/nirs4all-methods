# Cross-binding benchmark — thread sweep

Same matrix as the [main page](cross_binding.md), but with thread counts 1, 3 and 10 shown in separate per-(algorithm, thread) sections. **External libraries (`sklearn`, `pls`, `ropls`, `mixOmics`, `plsregress`, `ikpls`) typically don't accelerate their inner loops with thread count** — only their linked BLAS does, and that helps only when the algo is GEMM-bound. pls4all bindings use OpenMP at the C kernel level on top of the BLAS, so multi-thread wins are visible here.

_Generated: 2026-05-17 20:35:40 UTC_  
_CSV: `tmpi2r2__e5.csv` (4550 cells)_


## Host

| | |
|---|---|
| **CPU**            | 12th Gen Intel(R) Core(TM) i9-12900K |
| **Cores**          | 12 physical / 24 logical |
| **Frequency**  | 3187 MHz nominal |
| **L3 cache**   | 30720 KB |
| **RAM**            | 62.7 GB |
| **OS / kernel**    | Linux 6.6.87.2-microsoft-standard-WSL2 (x86_64) · kernel 6.6.87.2-microsoft-standard-WSL2 |
| **Python**         | 3.13.11 |


## How to read a cell

Each cell shows `<median_ms> <parity icon>`. The icon is the parity verdict vs the reference backend (`pls4all.cpp` at 1 thread, dev-release build):

- ✓ **bit-exact** (max abs diff ≤ 1e-6 vs ref) — this backend produces the *same* predictions as the reference
- ⚠ **close** (≤ 1e-3 but > 1e-6) — algorithmic drift (different convergence path), answer agrees in practice
- ✗ **divergent** (> 1e-3) — different preprocessing convention (e.g. tier-2 wrappers default to `scale=True`, tier-1 / externals default to `scale=False`); the backend works, it just isn't an apples-to-apples comparison
- ⏳ cell timed out (300 s wall-clock)
- `—` see *Why a cell is empty* below

**Bold** = fastest cell on the row, **counted only among ✓ cells**. ⚠ / ✗ / — cells never carry the bold (comparing a non-bit-exact result against bit-exact ones would be misleading).

Timing is the **median of 5 runs**; the first run is discarded as warmup. All backends in a single cell read the SAME orchestrator-generated CSV so cross-language input bytes are bit-identical. See [methodology.md](methodology.md) for the full details.


## Why a cell is empty (`—`)

An empty cell means the backend **did not produce a timing for that (algorithm, size) combination**. Four distinct reasons, all reported as `—` in the matrix:

1. **External library doesn't implement the algorithm.** Example: `sklearn` has no native CPPLS or sparse SIMPLS; `plsregress` (Octave) only does plain PLS; `ikpls` is plain PLS only; `ropls` only does OPLS; `mixOmics` covers PLS / sparse PLS / PLS-DA. Filling this column requires the external maintainer to add the algorithm — out of our control.
2. **pls4all wrapper missing for that algorithm.** Example: `pls4all.sklearn` (tier 2) doesn't yet ship a class for `continuum_regression` or `kernel_pls`; `pls4all.R.formula` doesn't have a formula wrapper for every algorithm yet. This is the *chemin à parcourir* on our side — visible TODO.
3. **Bench script missing the dispatch case.** A handful of niveau C cells failed because the bench script (e.g. `bench_matlab_tier1.m`) doesn't yet route a specific algorithm to its underlying call. Pure tooling gap, no library impact.
4. **Run too slow / OOM / crashed.** The cell timed out (`⏳`) or hit a runtime error. Rare in this matrix (300 s timeout is generous for the included sizes).

Each `—` represents one of these four reasons. The CSV (`results/full_matrix.csv`) carries a `reason` column that tells you exactly which one for any given cell.


Column names: anything starting with `pls4all.` is one of this project's bindings; the others use their real package name (`sklearn`, `ikpls`, `pls`, `ropls`, `mixOmics`, `plsregress`). **Every algorithm table keeps every column** — `—` cells are intentional and show *where coverage is still missing*, either on our side (an algorithm not yet wrapped in a tier-2 class) or on the external side (e.g. `sklearn` doesn't implement CPPLS / OPLS, `plsregress` only does plain PLS, `ikpls` only does plain PLS). Full per-column description: see [Backend definitions](#backend-definitions) at the bottom of this page.


## bagging_pls  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×500 | — | — | — | — | — | — | — | 42.5 ms ✓ | 74.5 ms ✓ | — | **25.3 ms ✓** | — | — | — | — | — | — |
| 500×500 | — | — | — | — | — | — | — | 244.5 ms ✓ | 294.5 ms ✓ | — | **130.6 ms ✓** | — | — | — | — | — | — |
| 2500×500 | — | — | — | — | — | — | — | 3.0 s ✓ | 2.9 s ✓ | — | **2.5 s ✓** | — | — | — | — | — | — |
| 2500×2500 | — | — | — | — | — | — | — | 16.6 s ✓ | 17.6 s ✓ | — | **12.3 s ✓** | — | — | — | — | — | — |


## bagging_pls  —  10 threads

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×500 | — | — | — | — | — | — | — | 43.5 ms ✓ | 74.0 ms ✓ | — | **26.8 ms ✓** | — | — | — | — | — | — |
| 500×500 | — | — | — | — | — | — | — | 253.0 ms ✓ | 299.0 ms ✓ | — | **134.3 ms ✓** | — | — | — | — | — | — |
| 2500×500 | — | — | — | — | — | — | — | 3.3 s ✓ | 2.9 s ✓ | — | **2.1 s ✓** | — | — | — | — | — | — |
| 2500×2500 | — | — | — | — | — | — | — | 16.5 s ✓ | 17.9 s ✓ | — | **12.4 s ✓** | — | — | — | — | — | — |


## boosting_pls  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×500 | — | — | — | — | — | — | — | 45.5 ms ✓ | 74.5 ms ✓ | — | **25.3 ms ✓** | — | — | — | — | — | — |
| 500×500 | — | — | — | — | — | — | — | 241.0 ms ✓ | 297.0 ms ✓ | — | **128.1 ms ✓** | — | — | — | — | — | — |
| 2500×500 | — | — | — | — | — | — | — | 2.9 s ✓ | 2.9 s ✓ | — | **2.1 s ✓** | — | — | — | — | — | — |
| 2500×2500 | — | — | — | — | — | — | — | 16.6 s ✓ | 18.0 s ✓ | — | **12.2 s ✓** | — | — | — | — | — | — |


## boosting_pls  —  10 threads

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×500 | — | — | — | — | — | — | — | 44.0 ms ✓ | 76.0 ms ✓ | — | **25.6 ms ✓** | — | — | — | — | — | — |
| 500×500 | — | — | — | — | — | — | — | 240.5 ms ✓ | 289.0 ms ✓ | — | **127.6 ms ✓** | — | — | — | — | — | — |
| 2500×500 | — | — | — | — | — | — | — | 2.9 s ✓ | 3.4 s ✓ | — | **2.2 s ✓** | — | — | — | — | — | — |
| 2500×2500 | — | — | — | — | — | — | — | 16.6 s ✓ | 18.2 s ✓ | — | **11.9 s ✓** | — | — | — | — | — | — |


## continuum_regression  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×500 | — | — | — | — | — | — | — | 36.0 ms ✓ | 63.0 ms ✓ | — | **15.0 ms ✓** | — | — | — | — | — | — |
| 500×500 | — | — | — | — | — | — | — | 179.0 ms ✓ | 243.5 ms ✓ | — | **63.6 ms ✓** | — | — | — | — | — | — |
| 2500×500 | — | — | — | — | — | — | — | 1.6 s ✓ | 1.5 s ✓ | — | **362.3 ms ✓** | — | — | — | — | — | — |
| 2500×2500 | — | — | — | — | — | — | — | 8.1 s ✓ | 9.2 s ✓ | — | **1.9 s ✓** | — | — | — | — | — | — |


## continuum_regression  —  10 threads

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×500 | — | — | — | — | — | — | — | 31.0 ms ✓ | 65.0 ms ✓ | — | **14.4 ms ✓** | — | — | — | — | — | — |
| 500×500 | — | — | — | — | — | — | — | 188.0 ms ✓ | 244.5 ms ✓ | — | **69.0 ms ✓** | — | — | — | — | — | — |
| 2500×500 | — | — | — | — | — | — | — | 1.5 s ✓ | 1.4 s ✓ | — | **384.0 ms ✓** | — | — | — | — | — | — |
| 2500×2500 | — | — | — | — | — | — | — | 7.7 s ✓ | 8.7 s ✓ | — | **1.9 s ✓** | — | — | — | — | — | — |


## cppls  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | — | — | **0.87 ms ✓** | 1.09 ms ✓ | 2.00 ms ✓ | 5.00 ms ✓ | 1.59 ms ✓ | 3.52 ms ✓ | — | — | 6.50 ms ✗ | — | — | — |
| 100×500 | — | — | — | — | — | 7.51 ms ✓ | **7.36 ms ✓** | 34.0 ms ✓ | 61.0 ms ✓ | 11.9 ms ✓ | 13.7 ms ✓ | — | — | 69.5 ms ✗ | — | — | — |
| 100×2500 | — | — | — | — | — | **38.7 ms ✓** | 39.1 ms ✓ | 305.5 ms ✓ | 589.5 ms ✓ | 62.9 ms ✓ | 63.2 ms ✓ | — | — | 606.5 ms ⚠ | — | — | — |
| 500×50 | — | — | — | — | — | **4.06 ms ✓** | 4.14 ms ✓ | 11.5 ms ✓ | 15.0 ms ✓ | 6.39 ms ✓ | 8.50 ms ✓ | — | — | 18.5 ms ✗ | — | — | — |
| 500×500 | — | — | — | — | — | **37.7 ms ✓** | 38.1 ms ✓ | 177.0 ms ✓ | 229.0 ms ✓ | 65.5 ms ✓ | 65.0 ms ✓ | — | — | 241.0 ms ✗ | — | — | — |
| 500×2500 | — | — | — | — | — | 191.3 ms ✓ | **190.9 ms ✓** | 1.5 s ✓ | 1.8 s ✓ | 336.1 ms ✓ | 335.9 ms ✓ | — | — | 1.9 s ✗ | — | — | — |
| 2500×50 | — | — | — | — | — | **18.7 ms ✓** | 19.0 ms ✓ | 63.0 ms ✓ | 86.0 ms ✓ | 30.0 ms ✓ | 33.1 ms ✓ | — | — | 68.0 ms ⚠ | — | — | — |
| 2500×500 | — | — | — | — | — | 226.3 ms ✓ | **225.1 ms ✓** | 1.3 s ✓ | 1.4 s ✓ | 343.9 ms ✓ | 356.3 ms ✓ | — | — | 1.4 s ✗ | — | — | — |
| 2500×2500 | — | — | — | — | — | **1.1 s ✓** | 1.2 s ✓ | 7.7 s ✓ | 8.6 s ✓ | 1.9 s ✓ | 1.9 s ✓ | — | — | 8.6 s ✗ | — | — | — |
| 10000×50 | — | — | — | — | — | **77.5 ms ✓** | 78.6 ms ✓ | 436.0 ms ✓ | 483.0 ms ✓ | 117.7 ms ✓ | 128.9 ms ✓ | — | — | 427.0 ms ⚠ | — | — | — |
| 10000×500 | — | — | — | — | — | 947.1 ms ✓ | **921.4 ms ✓** | 6.5 s ✓ | 6.8 s ✓ | 1.4 s ✓ | 1.4 s ✓ | — | — | 6.9 s ⚠ | — | — | — |


## cppls  —  3 threads

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | — | — | **0.84 ms ✓** | 1.01 ms ✓ | 2.50 ms ✓ | 5.50 ms ✓ | 1.58 ms ✓ | 3.59 ms ✓ | — | — | 6.50 ms ✗ | — | — | — |
| 100×500 | — | — | — | — | — | **7.61 ms ✓** | 7.77 ms ✓ | 31.5 ms ✓ | 69.0 ms ✓ | 12.7 ms ✓ | 14.6 ms ✓ | — | — | 68.5 ms ✗ | — | — | — |
| 100×2500 | — | — | — | — | — | **37.1 ms ✓** | 38.7 ms ✓ | 312.0 ms ✓ | 622.5 ms ✓ | 60.8 ms ✓ | 64.5 ms ✓ | — | — | 606.0 ms ⚠ | — | — | — |
| 500×50 | — | — | — | — | — | 3.99 ms ✓ | **3.97 ms ✓** | 10.5 ms ✓ | 15.5 ms ✓ | 6.17 ms ✓ | 8.69 ms ✓ | — | — | 19.0 ms ✗ | — | — | — |
| 500×500 | — | — | — | — | — | 41.3 ms ✓ | **38.8 ms ✓** | 172.5 ms ✓ | 232.0 ms ✓ | 61.7 ms ✓ | 62.7 ms ✓ | — | — | 241.0 ms ✗ | — | — | — |
| 500×2500 | — | — | — | — | — | **196.7 ms ✓** | 202.3 ms ✓ | 1.4 s ✓ | 1.8 s ✓ | 323.1 ms ✓ | 330.0 ms ✓ | — | — | 1.9 s ✗ | — | — | — |
| 2500×50 | — | — | — | — | — | **18.9 ms ✓** | 19.7 ms ✓ | 68.0 ms ✓ | 74.5 ms ✓ | 30.2 ms ✓ | 33.6 ms ✓ | — | — | 70.5 ms ⚠ | — | — | — |
| 2500×500 | — | — | — | — | — | 237.5 ms ✓ | **230.2 ms ✓** | 1.3 s ✓ | 1.4 s ✓ | 342.4 ms ✓ | 355.2 ms ✓ | — | — | 1.4 s ✗ | — | — | — |
| 2500×2500 | — | — | — | — | — | 1.2 s ✓ | **1.2 s ✓** | 7.8 s ✓ | 8.6 s ✓ | 1.9 s ✓ | 1.9 s ✓ | — | — | 8.9 s ✗ | — | — | — |
| 10000×50 | — | — | — | — | — | **78.9 ms ✓** | 81.8 ms ✓ | 440.0 ms ✓ | 498.5 ms ✓ | 119.7 ms ✓ | 134.3 ms ✓ | — | — | 432.0 ms ⚠ | — | — | — |
| 10000×500 | — | — | — | — | — | 933.9 ms ✓ | **924.7 ms ✓** | 6.4 s ✓ | 6.8 s ✓ | 1.4 s ✓ | 1.5 s ✓ | — | — | 7.1 s ⚠ | — | — | — |


## cppls  —  10 threads

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | — | — | **0.87 ms ✓** | 0.99 ms ✓ | 2.00 ms ✓ | 5.00 ms ✓ | 1.56 ms ✓ | 3.41 ms ✓ | — | — | 7.50 ms ✗ | — | — | — |
| 100×500 | — | — | — | — | — | 7.96 ms ✓ | **7.61 ms ✓** | 32.5 ms ✓ | 66.0 ms ✓ | 12.3 ms ✓ | 14.4 ms ✓ | — | — | 70.0 ms ✗ | — | — | — |
| 100×2500 | — | — | — | — | — | 37.5 ms ✓ | **37.4 ms ✓** | 304.5 ms ✓ | 596.0 ms ✓ | 62.3 ms ✓ | 62.5 ms ✓ | — | — | 601.0 ms ⚠ | — | — | — |
| 500×50 | — | — | — | — | — | 4.16 ms ✓ | **4.09 ms ✓** | 10.5 ms ✓ | 14.0 ms ✓ | 6.19 ms ✓ | 8.86 ms ✓ | — | — | 17.0 ms ✗ | — | — | — |
| 500×500 | — | — | — | — | — | 39.3 ms ✓ | **38.3 ms ✓** | 181.0 ms ✓ | 223.0 ms ✓ | 61.3 ms ✓ | 65.0 ms ✓ | — | — | 232.5 ms ✗ | — | — | — |
| 500×2500 | — | — | — | — | — | **195.7 ms ✓** | 233.2 ms ✓ | 1.4 s ✓ | 1.8 s ✓ | 320.9 ms ✓ | 362.9 ms ✓ | — | — | 2.0 s ✗ | — | — | — |
| 2500×50 | — | — | — | — | — | **19.1 ms ✓** | 19.1 ms ✓ | 63.5 ms ✓ | 89.5 ms ✓ | 31.7 ms ✓ | 34.6 ms ✓ | — | — | 68.5 ms ⚠ | — | — | — |
| 2500×500 | — | — | — | — | — | **233.5 ms ✓** | 262.4 ms ✓ | 1.3 s ✓ | 1.4 s ✓ | 343.7 ms ✓ | 389.2 ms ✓ | — | — | 1.5 s ✗ | — | — | — |
| 2500×2500 | — | — | — | — | — | **1.2 s ✓** | 1.2 s ✓ | 7.7 s ✓ | 8.7 s ✓ | 1.9 s ✓ | 1.9 s ✓ | — | — | 8.9 s ✗ | — | — | — |
| 10000×50 | — | — | — | — | — | **78.1 ms ✓** | 109.5 ms ✓ | 435.5 ms ✓ | 524.0 ms ✓ | 126.3 ms ✓ | 159.2 ms ✓ | — | — | 504.0 ms ⚠ | — | — | — |
| 10000×500 | — | — | — | — | — | **947.2 ms ✓** | 966.4 ms ✓ | 6.5 s ✓ | 6.8 s ✓ | 1.5 s ✓ | 1.5 s ✓ | — | — | 7.1 s ⚠ | — | — | — |


## ds  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2500×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2500×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## ds  —  10 threads

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2500×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2500×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## ecr  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | — | — | **1.08 ms ✓** | 1.20 ms ✓ | 3.00 ms ✓ | 5.50 ms ✓ | 1.81 ms ✓ | 3.92 ms ✓ | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | 56.9 ms ✓ | **56.6 ms ✓** | 79.0 ms ✓ | 115.0 ms ✓ | 58.7 ms ✓ | 61.6 ms ✓ | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | 5.0 s ✓ | **4.5 s ✓** | 4.7 s ✓ | 4.9 s ✓ | 5.1 s ✓ | 4.9 s ✓ | — | — | — | — | — | — |
| 500×50 | — | — | — | — | — | **4.97 ms ✓** | 4.99 ms ✓ | 11.0 ms ✓ | 16.0 ms ✓ | 7.47 ms ✓ | 9.27 ms ✓ | — | — | — | — | — | — |
| 500×500 | — | — | — | — | — | 187.6 ms ✓ | **185.5 ms ✓** | 339.5 ms ✓ | 377.0 ms ✓ | 200.5 ms ✓ | 201.6 ms ✓ | — | — | — | — | — | — |
| 500×2500 | — | — | — | — | — | 7.5 s ✓ | **7.1 s ✓** | 7.7 s ✓ | 8.3 s ✓ | 7.6 s ✓ | 7.4 s ✓ | — | — | — | — | — | — |
| 2500×50 | — | — | — | — | — | **22.7 ms ✓** | 23.4 ms ✓ | 67.5 ms ✓ | 81.5 ms ✓ | 34.1 ms ✓ | 37.4 ms ✓ | — | — | — | — | — | — |
| 2500×500 | — | — | — | — | — | **866.0 ms ✓** | 873.7 ms ✓ | 2.0 s ✓ | 2.0 s ✓ | 997.0 ms ✓ | 1.0 s ✓ | — | — | — | — | — | — |
| 2500×2500 | — | — | — | — | — | 33.3 s ✓ | **32.5 s ✓** | 39.0 s ✓ | 40.0 s ✓ | 33.7 s ✓ | 34.5 s ✓ | — | — | — | — | — | — |
| 10000×50 | — | — | — | — | — | 93.4 ms ✓ | **92.6 ms ✓** | 465.5 ms ✓ | 426.0 ms ✓ | 133.4 ms ✓ | 137.1 ms ✓ | — | — | — | — | — | — |
| 10000×500 | — | — | — | — | — | 3.6 s ✓ | **3.6 s ✓** | 8.9 s ✓ | 9.2 s ✓ | 4.1 s ✓ | 4.1 s ✓ | — | — | — | — | — | — |


## ecr  —  3 threads

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | — | — | **1.17 ms ✓** | 1.26 ms ✓ | 3.00 ms ✓ | 5.50 ms ✓ | 1.79 ms ✓ | 3.70 ms ✓ | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | 58.9 ms ✓ | **57.1 ms ✓** | 78.5 ms ✓ | 121.0 ms ✓ | 61.3 ms ✓ | 62.5 ms ✓ | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | 4.9 s ✓ | **4.6 s ✓** | 4.7 s ✓ | 4.8 s ✓ | 5.0 s ✓ | 5.1 s ✓ | — | — | — | — | — | — |
| 500×50 | — | — | — | — | — | **4.96 ms ✓** | 5.12 ms ✓ | 11.5 ms ✓ | 18.0 ms ✓ | 7.48 ms ✓ | 9.11 ms ✓ | — | — | — | — | — | — |
| 500×500 | — | — | — | — | — | **178.2 ms ✓** | 179.1 ms ✓ | 320.0 ms ✓ | 363.5 ms ✓ | 201.7 ms ✓ | 206.4 ms ✓ | — | — | — | — | — | — |
| 500×2500 | — | — | — | — | — | 7.4 s ✓ | **7.0 s ✓** | 7.8 s ✓ | 8.1 s ✓ | 7.5 s ✓ | 7.4 s ✓ | — | — | — | — | — | — |
| 2500×50 | — | — | — | — | — | 23.6 ms ✓ | **23.6 ms ✓** | 68.0 ms ✓ | 82.0 ms ✓ | 34.7 ms ✓ | 37.2 ms ✓ | — | — | — | — | — | — |
| 2500×500 | — | — | — | — | — | **920.0 ms ✓** | 924.2 ms ✓ | 2.0 s ✓ | 2.0 s ✓ | 996.7 ms ✓ | 1.0 s ✓ | — | — | — | — | — | — |
| 2500×2500 | — | — | — | — | — | 36.3 s ✓ | **35.6 s ✓** | 39.6 s ✓ | 45.3 s ✓ | 37.1 s ✓ | 35.8 s ✓ | — | — | — | — | — | — |
| 10000×50 | — | — | — | — | — | **91.7 ms ✓** | 95.2 ms ✓ | 459.0 ms ✓ | 422.0 ms ✓ | 138.3 ms ✓ | 150.9 ms ✓ | — | — | — | — | — | — |
| 10000×500 | — | — | — | — | — | 3.6 s ✓ | **3.6 s ✓** | 9.2 s ✓ | 9.2 s ✓ | 4.1 s ✓ | 4.1 s ✓ | — | — | — | — | — | — |


## ecr  —  10 threads

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | — | — | **1.08 ms ✓** | 1.29 ms ✓ | 2.50 ms ✓ | 5.50 ms ✓ | 1.81 ms ✓ | 3.71 ms ✓ | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | **58.1 ms ✓** | 59.7 ms ✓ | 80.5 ms ✓ | 116.5 ms ✓ | 63.6 ms ✓ | 61.8 ms ✓ | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | 5.0 s ✓ | **4.5 s ✓** | 4.6 s ✓ | 4.9 s ✓ | 4.9 s ✓ | 4.9 s ✓ | — | — | — | — | — | — |
| 500×50 | — | — | — | — | — | 5.23 ms ✓ | **5.15 ms ✓** | 11.0 ms ✓ | 15.5 ms ✓ | 7.35 ms ✓ | 9.09 ms ✓ | — | — | — | — | — | — |
| 500×500 | — | — | — | — | — | **190.0 ms ✓** | 191.5 ms ✓ | 318.0 ms ✓ | 368.5 ms ✓ | 198.8 ms ✓ | 208.1 ms ✓ | — | — | — | — | — | — |
| 500×2500 | — | — | — | — | — | 7.4 s ✓ | **6.9 s ✓** | 7.6 s ✓ | 8.3 s ✓ | 7.3 s ✓ | 7.4 s ✓ | — | — | — | — | — | — |
| 2500×50 | — | — | — | — | — | **24.1 ms ✓** | 24.5 ms ✓ | 68.5 ms ✓ | 78.5 ms ✓ | 34.0 ms ✓ | 38.2 ms ✓ | — | — | — | — | — | — |
| 2500×500 | — | — | — | — | — | **882.0 ms ✓** | 917.9 ms ✓ | 2.0 s ✓ | 2.0 s ✓ | 995.0 ms ✓ | 1.1 s ✓ | — | — | — | — | — | — |
| 2500×2500 | — | — | — | — | — | 35.5 s ✓ | **33.4 s ✓** | 39.9 s ✓ | 40.6 s ✓ | 34.7 s ✓ | 35.3 s ✓ | — | — | — | — | — | — |
| 10000×50 | — | — | — | — | — | **92.3 ms ✓** | 112.9 ms ✓ | 470.5 ms ✓ | 469.0 ms ✓ | 140.2 ms ✓ | 178.2 ms ✓ | — | — | — | — | — | — |
| 10000×500 | — | — | — | — | — | 3.6 s ✓ | **3.6 s ✓** | 9.2 s ✓ | 9.4 s ✓ | 4.1 s ✓ | 4.2 s ✓ | — | — | — | — | — | — |


## fused_sparse_pls  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×500 | — | — | — | — | — | — | — | **30.0 ms ✓** | — | — | — | — | — | — | — | — | — |
| 500×500 | — | — | — | — | — | — | — | **180.0 ms ✓** | — | — | — | — | — | — | — | — | — |
| 2500×500 | — | — | — | — | — | — | — | **1.5 s ✓** | — | — | — | — | — | — | — | — | — |
| 2500×2500 | — | — | — | — | — | — | — | **8.1 s ✓** | — | — | — | — | — | — | — | — | — |


## fused_sparse_pls  —  10 threads

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×500 | — | — | — | — | — | — | — | **32.0 ms ✓** | — | — | — | — | — | — | — | — | — |
| 500×500 | — | — | — | — | — | — | — | **182.5 ms ✓** | — | — | — | — | — | — | — | — | — |
| 2500×500 | — | — | — | — | — | — | — | **1.5 s ✓** | — | — | — | — | — | — | — | — | — |
| 2500×2500 | — | — | — | — | — | — | — | **7.6 s ✓** | — | — | — | — | — | — | — | — | — |


## gpr_pls  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×500 | — | — | — | — | — | — | — | **34.0 ms ✓** | — | — | — | — | — | — | — | — | — |
| 500×500 | — | — | — | — | — | — | — | **218.0 ms ✓** | — | — | — | — | — | — | — | — | — |
| 2500×500 | — | — | — | — | — | — | — | **6.5 s ✓** | — | — | — | — | — | — | — | — | — |
| 2500×2500 | — | — | — | — | — | — | — | **13.3 s ✓** | — | — | — | — | — | — | — | — | — |


## gpr_pls  —  10 threads

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×500 | — | — | — | — | — | — | — | **33.5 ms ✓** | — | — | — | — | — | — | — | — | — |
| 500×500 | — | — | — | — | — | — | — | **232.5 ms ✓** | — | — | — | — | — | — | — | — | — |
| 2500×500 | — | — | — | — | — | — | — | **6.6 s ✓** | — | — | — | — | — | — | — | — | — |
| 2500×2500 | — | — | — | — | — | — | — | **13.8 s ✓** | — | — | — | — | — | — | — | — | — |


## group_sparse_pls  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2500×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2500×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## group_sparse_pls  —  10 threads

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2500×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2500×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## kernel_pls  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×500 | — | — | — | — | — | — | — | **34.5 ms ✓** | — | — | — | — | — | — | — | — | — |
| 500×500 | — | — | — | — | — | — | — | **270.5 ms ✓** | — | — | — | — | — | — | — | — | — |
| 2500×500 | — | — | — | — | — | — | — | **3.8 s ✓** | — | — | — | — | — | — | — | — | — |
| 2500×2500 | — | — | — | — | — | — | — | **23.6 s ✓** | — | — | — | — | — | — | — | — | — |


## kernel_pls  —  10 threads

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×500 | — | — | — | — | — | — | — | **34.0 ms ✓** | — | — | — | — | — | — | — | — | — |
| 500×500 | — | — | — | — | — | — | — | **275.5 ms ✓** | — | — | — | — | — | — | — | — | — |
| 2500×500 | — | — | — | — | — | — | — | **4.3 s ✓** | — | — | — | — | — | — | — | — | — |
| 2500×2500 | — | — | — | — | — | — | — | **23.4 s ✓** | — | — | — | — | — | — | — | — | — |


## lw_pls  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×500 | — | — | — | — | — | — | — | **45.5 ms ✓** | — | — | — | — | — | — | — | — | — |
| 500×500 | — | — | — | — | — | — | — | **290.0 ms ✓** | — | — | — | — | — | — | — | — | — |
| 2500×500 | — | — | — | — | — | — | — | **3.1 s ✓** | — | — | — | — | — | — | — | — | — |
| 2500×2500 | — | — | — | — | — | — | — | **17.4 s ✓** | — | — | — | — | — | — | — | — | — |


## lw_pls  —  10 threads

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×500 | — | — | — | — | — | — | — | **46.0 ms ✓** | — | — | — | — | — | — | — | — | — |
| 500×500 | — | — | — | — | — | — | — | **286.5 ms ✓** | — | — | — | — | — | — | — | — | — |
| 2500×500 | — | — | — | — | — | — | — | **3.2 s ✓** | — | — | — | — | — | — | — | — | — |
| 2500×2500 | — | — | — | — | — | — | — | **17.9 s ✓** | — | — | — | — | — | — | — | — | — |


## mir_pls  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | 0.89 ms ✓ | — | **0.84 ms ✓** | 0.99 ms ✓ | 2.00 ms ✓ | 5.50 ms ✓ | 1.43 ms ✓ | 3.37 ms ✓ | — | — | — | — | — | — |
| 100×500 | — | — | — | 7.56 ms ✓ | — | 7.36 ms ✓ | **7.21 ms ✓** | 31.5 ms ✓ | 64.5 ms ✓ | 11.5 ms ✓ | 13.4 ms ✓ | — | — | — | — | — | — |
| 100×2500 | — | — | — | 35.3 ms ✓ | — | **34.8 ms ✓** | 35.4 ms ✓ | 293.0 ms ✓ | 584.0 ms ✓ | 58.1 ms ✓ | 60.2 ms ✓ | — | — | — | — | — | — |
| 500×50 | — | — | — | 3.81 ms ✓ | — | 3.75 ms ✓ | **3.74 ms ✓** | 10.5 ms ✓ | 14.0 ms ✓ | 6.13 ms ✓ | 7.79 ms ✓ | — | — | — | — | — | — |
| 500×500 | — | — | — | **35.4 ms ✓** | — | 36.2 ms ✓ | 36.5 ms ✓ | 173.5 ms ✓ | 227.0 ms ✓ | 57.4 ms ✓ | 60.8 ms ✓ | — | — | — | — | — | — |
| 500×2500 | — | — | — | 179.7 ms ✓ | — | **179.0 ms ✓** | 183.8 ms ✓ | 1.4 s ✓ | 1.7 s ✓ | 308.4 ms ✓ | 313.3 ms ✓ | — | — | — | — | — | — |
| 2500×50 | — | — | — | 18.0 ms ✓ | — | **17.8 ms ✓** | 17.9 ms ✓ | 63.0 ms ✓ | 79.0 ms ✓ | 29.1 ms ✓ | 31.4 ms ✓ | — | — | — | — | — | — |
| 2500×500 | — | — | — | 190.4 ms ✓ | — | 192.4 ms ✓ | **187.2 ms ✓** | 1.3 s ✓ | 1.3 s ✓ | 301.1 ms ✓ | 302.7 ms ✓ | — | — | — | — | — | — |
| 2500×2500 | — | — | — | **940.8 ms ✓** | — | 942.0 ms ✓ | 956.3 ms ✓ | 7.9 s ✓ | 8.3 s ✓ | 1.7 s ✓ | 1.7 s ✓ | — | — | — | — | — | — |
| 10000×50 | — | — | — | **72.7 ms ✓** | — | 73.7 ms ✓ | 72.8 ms ✓ | 441.0 ms ✓ | 484.5 ms ✓ | 115.2 ms ✓ | 120.2 ms ✓ | — | — | — | — | — | — |
| 10000×500 | — | — | — | 779.3 ms ✓ | — | **767.3 ms ✓** | 777.0 ms ✓ | 6.3 s ✓ | 6.7 s ✓ | 1.3 s ✓ | 1.3 s ✓ | — | — | — | — | — | — |


## mir_pls  —  3 threads

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | **0.78 ms ✓** | — | 0.83 ms ✓ | 0.96 ms ✓ | 2.00 ms ✓ | 5.50 ms ✓ | 1.59 ms ✓ | 3.45 ms ✓ | — | — | — | — | — | — |
| 100×500 | — | — | — | 7.63 ms ✓ | — | **7.12 ms ✓** | 7.20 ms ✓ | 32.0 ms ✓ | 61.5 ms ✓ | 11.6 ms ✓ | 13.2 ms ✓ | — | — | — | — | — | — |
| 100×2500 | — | — | — | **35.4 ms ✓** | — | 36.4 ms ✓ | 35.6 ms ✓ | 298.0 ms ✓ | 585.5 ms ✓ | 57.9 ms ✓ | 61.2 ms ✓ | — | — | — | — | — | — |
| 500×50 | — | — | — | **3.71 ms ✓** | — | 3.92 ms ✓ | 3.77 ms ✓ | 10.5 ms ✓ | 14.5 ms ✓ | 5.99 ms ✓ | 8.20 ms ✓ | — | — | — | — | — | — |
| 500×500 | — | — | — | 36.4 ms ✓ | — | **35.2 ms ✓** | 36.0 ms ✓ | 184.0 ms ✓ | 236.5 ms ✓ | 57.1 ms ✓ | 61.9 ms ✓ | — | — | — | — | — | — |
| 500×2500 | — | — | — | 181.8 ms ✓ | — | **177.1 ms ✓** | 191.3 ms ✓ | 1.4 s ✓ | 1.8 s ✓ | 308.5 ms ✓ | 316.1 ms ✓ | — | — | — | — | — | — |
| 2500×50 | — | — | — | 18.3 ms ✓ | — | **18.0 ms ✓** | 18.1 ms ✓ | 64.5 ms ✓ | 77.5 ms ✓ | 28.8 ms ✓ | 31.1 ms ✓ | — | — | — | — | — | — |
| 2500×500 | — | — | — | 189.2 ms ✓ | — | **187.2 ms ✓** | 194.5 ms ✓ | 1.3 s ✓ | 1.3 s ✓ | 300.2 ms ✓ | 318.8 ms ✓ | — | — | — | — | — | — |
| 2500×2500 | — | — | — | **946.2 ms ✓** | — | 949.7 ms ✓ | 954.8 ms ✓ | 8.0 s ✓ | 8.3 s ✓ | 1.7 s ✓ | 1.8 s ✓ | — | — | — | — | — | — |
| 10000×50 | — | — | — | 73.9 ms ✓ | — | **72.3 ms ✓** | 73.9 ms ✓ | 440.0 ms ✓ | 493.0 ms ✓ | 117.0 ms ✓ | 125.5 ms ✓ | — | — | — | — | — | — |
| 10000×500 | — | — | — | **766.1 ms ✓** | — | 766.7 ms ✓ | 780.4 ms ✓ | 6.3 s ✓ | 6.7 s ✓ | 1.3 s ✓ | 1.3 s ✓ | — | — | — | — | — | — |


## mir_pls  —  10 threads

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | **0.80 ms ✓** | — | 0.87 ms ✓ | 0.96 ms ✓ | 2.50 ms ✓ | 5.50 ms ✓ | 1.53 ms ✓ | 3.58 ms ✓ | — | — | — | — | — | — |
| 100×500 | — | — | — | 7.46 ms ✓ | — | **6.99 ms ✓** | 7.46 ms ✓ | 31.5 ms ✓ | 63.5 ms ✓ | 11.8 ms ✓ | 13.8 ms ✓ | — | — | — | — | — | — |
| 100×2500 | — | — | — | 35.7 ms ✓ | — | **34.9 ms ✓** | 36.1 ms ✓ | 300.5 ms ✓ | 584.0 ms ✓ | 58.9 ms ✓ | 61.1 ms ✓ | — | — | — | — | — | — |
| 500×50 | — | — | — | 3.71 ms ✓ | — | 3.73 ms ✓ | **3.71 ms ✓** | 11.0 ms ✓ | 14.5 ms ✓ | 6.06 ms ✓ | 7.81 ms ✓ | — | — | — | — | — | — |
| 500×500 | — | — | — | **36.0 ms ✓** | — | 36.6 ms ✓ | 37.4 ms ✓ | 171.5 ms ✓ | 233.0 ms ✓ | 56.7 ms ✓ | 62.1 ms ✓ | — | — | — | — | — | — |
| 500×2500 | — | — | — | **177.9 ms ✓** | — | 179.6 ms ✓ | 220.0 ms ✓ | 1.4 s ✓ | 1.8 s ✓ | 309.3 ms ✓ | 350.2 ms ✓ | — | — | — | — | — | — |
| 2500×50 | — | — | — | **18.2 ms ✓** | — | 19.0 ms ✓ | 18.3 ms ✓ | 65.5 ms ✓ | 77.0 ms ✓ | 29.1 ms ✓ | 31.4 ms ✓ | — | — | — | — | — | — |
| 2500×500 | — | — | — | 194.1 ms ✓ | — | **192.5 ms ✓** | 223.5 ms ✓ | 1.3 s ✓ | 1.4 s ✓ | 309.5 ms ✓ | 354.0 ms ✓ | — | — | — | — | — | — |
| 2500×2500 | — | — | — | 943.3 ms ✓ | — | **935.9 ms ✓** | 986.1 ms ✓ | 7.5 s ✓ | 8.4 s ✓ | 1.7 s ✓ | 1.8 s ✓ | — | — | — | — | — | — |
| 10000×50 | — | — | — | **74.2 ms ✓** | — | 75.0 ms ✓ | 104.9 ms ✓ | 448.5 ms ✓ | 539.0 ms ✓ | 118.0 ms ✓ | 149.2 ms ✓ | — | — | — | — | — | — |
| 10000×500 | — | — | — | **774.7 ms ✓** | — | 775.2 ms ✓ | 826.4 ms ✓ | 6.8 s ✓ | 6.7 s ✓ | 1.3 s ✓ | 1.3 s ✓ | — | — | — | — | — | — |


## missing_aware_nipals  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | **0.87 ms ✓** | — | 0.89 ms ✓ | 0.93 ms ✓ | 2.00 ms ✓ | 5.00 ms ✓ | 1.51 ms ✓ | 3.58 ms ✓ | — | — | — | — | — | — |
| 100×500 | — | — | — | 8.01 ms ✓ | — | 7.71 ms ✓ | **7.40 ms ✓** | 31.0 ms ✓ | 66.0 ms ✓ | 12.2 ms ✓ | 14.3 ms ✓ | — | — | — | — | — | — |
| 100×2500 | — | — | — | 38.1 ms ✓ | — | **37.2 ms ✓** | 37.8 ms ✓ | 329.5 ms ✓ | 607.0 ms ✓ | 61.4 ms ✓ | 64.7 ms ✓ | — | — | — | — | — | — |
| 500×50 | — | — | — | 4.16 ms ✓ | — | **3.82 ms ✓** | 3.87 ms ✓ | 12.0 ms ✓ | 16.5 ms ✓ | 6.45 ms ✓ | 8.81 ms ✓ | — | — | — | — | — | — |
| 500×500 | — | — | — | 44.2 ms ✓ | — | **39.6 ms ✓** | 41.8 ms ✓ | 204.5 ms ✓ | 255.5 ms ✓ | 62.1 ms ✓ | 63.7 ms ✓ | — | — | — | — | — | — |
| 500×2500 | — | — | — | 196.2 ms ✓ | — | **191.7 ms ✓** | 196.7 ms ✓ | 1.2 s ✓ | 1.9 s ✓ | 328.5 ms ✓ | 331.3 ms ✓ | — | — | — | — | — | — |
| 2500×50 | — | — | — | 20.0 ms ✓ | — | **18.8 ms ✓** | 19.7 ms ✓ | 73.0 ms ✓ | 89.0 ms ✓ | 29.7 ms ✓ | 32.5 ms ✓ | — | — | — | — | — | — |
| 2500×500 | — | — | — | **221.8 ms ✓** | — | 224.3 ms ✓ | 225.5 ms ✓ | 1.5 s ✓ | 1.4 s ✓ | 339.6 ms ✓ | 348.6 ms ✓ | — | — | — | — | — | — |
| 2500×2500 | — | — | — | 1.1 s ✓ | — | **1.1 s ✓** | 1.1 s ✓ | 8.5 s ✓ | 9.2 s ✓ | 1.8 s ✓ | 1.9 s ✓ | — | — | — | — | — | — |
| 10000×50 | — | — | — | 75.4 ms ✓ | — | **74.2 ms ✓** | 74.7 ms ✓ | 447.5 ms ✓ | 565.0 ms ✓ | 118.2 ms ✓ | 119.1 ms ✓ | — | — | — | — | — | — |
| 10000×500 | — | — | — | 875.8 ms ✓ | — | 875.7 ms ✓ | **870.1 ms ✓** | 6.4 s ✓ | 6.1 s ✓ | 1.4 s ✓ | 1.4 s ✓ | — | — | — | — | — | — |


## missing_aware_nipals  —  3 threads

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | **0.86 ms ✓** | — | 1.15 ms ✓ | 0.98 ms ✓ | 2.50 ms ✓ | 5.50 ms ✓ | 1.65 ms ✓ | 3.44 ms ✓ | — | — | — | — | — | — |
| 100×500 | — | — | — | 7.53 ms ✓ | — | 7.53 ms ✓ | **7.53 ms ✓** | 35.0 ms ✓ | 64.5 ms ✓ | 12.2 ms ✓ | 14.6 ms ✓ | — | — | — | — | — | — |
| 100×2500 | — | — | — | **35.3 ms ✓** | — | 37.0 ms ✓ | 38.4 ms ✓ | 309.5 ms ✓ | 595.0 ms ✓ | 60.6 ms ✓ | 63.1 ms ✓ | — | — | — | — | — | — |
| 500×50 | — | — | — | **3.95 ms ✓** | — | 4.17 ms ✓ | 4.03 ms ✓ | 12.0 ms ✓ | 16.0 ms ✓ | 6.72 ms ✓ | 8.89 ms ✓ | — | — | — | — | — | — |
| 500×500 | — | — | — | 38.6 ms ✓ | — | **38.2 ms ✓** | 38.4 ms ✓ | 193.5 ms ✓ | 240.0 ms ✓ | 58.6 ms ✓ | 62.8 ms ✓ | — | — | — | — | — | — |
| 500×2500 | — | — | — | **187.7 ms ✓** | — | 189.5 ms ✓ | 195.4 ms ✓ | 1.2 s ✓ | 1.9 s ✓ | 323.9 ms ✓ | 333.1 ms ✓ | — | — | — | — | — | — |
| 2500×50 | — | — | — | **19.1 ms ✓** | — | 19.7 ms ✓ | 19.6 ms ✓ | 80.5 ms ✓ | 86.5 ms ✓ | 31.1 ms ✓ | 32.5 ms ✓ | — | — | — | — | — | — |
| 2500×500 | — | — | — | 223.4 ms ✓ | — | **216.9 ms ✓** | 217.8 ms ✓ | 1.6 s ✓ | 1.4 s ✓ | 330.7 ms ✓ | 345.4 ms ✓ | — | — | — | — | — | — |
| 2500×2500 | — | — | — | **1.1 s ✓** | — | 1.1 s ✓ | 1.1 s ✓ | 7.7 s ✓ | 9.0 s ✓ | 1.8 s ✓ | 1.8 s ✓ | — | — | — | — | — | — |
| 10000×50 | — | — | — | 75.3 ms ✓ | — | **74.4 ms ✓** | 75.3 ms ✓ | 438.5 ms ✓ | 555.0 ms ✓ | 117.2 ms ✓ | 128.7 ms ✓ | — | — | — | — | — | — |
| 10000×500 | — | — | — | 882.5 ms ✓ | — | 896.2 ms ✓ | **863.3 ms ✓** | 6.7 s ✓ | 6.3 s ✓ | 1.4 s ✓ | 1.4 s ✓ | — | — | — | — | — | — |


## missing_aware_nipals  —  10 threads

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | **0.88 ms ✓** | — | 0.89 ms ✓ | 1.02 ms ✓ | 2.50 ms ✓ | 5.50 ms ✓ | 1.81 ms ✓ | 3.56 ms ✓ | — | — | — | — | — | — |
| 100×500 | — | — | — | 7.62 ms ✓ | — | 7.59 ms ✓ | **7.39 ms ✓** | 33.5 ms ✓ | 63.0 ms ✓ | 12.3 ms ✓ | 14.8 ms ✓ | — | — | — | — | — | — |
| 100×2500 | — | — | — | 37.9 ms ✓ | — | **37.6 ms ✓** | 38.5 ms ✓ | 319.5 ms ✓ | 597.0 ms ✓ | 63.8 ms ✓ | 67.6 ms ✓ | — | — | — | — | — | — |
| 500×50 | — | — | — | 4.09 ms ✓ | — | **4.09 ms ✓** | 4.14 ms ✓ | 11.0 ms ✓ | 15.0 ms ✓ | 6.48 ms ✓ | 8.72 ms ✓ | — | — | — | — | — | — |
| 500×500 | — | — | — | **37.6 ms ✓** | — | 38.9 ms ✓ | 38.1 ms ✓ | 200.5 ms ✓ | 238.0 ms ✓ | 61.0 ms ✓ | 64.4 ms ✓ | — | — | — | — | — | — |
| 500×2500 | — | — | — | **190.1 ms ✓** | — | 194.4 ms ✓ | 196.6 ms ✓ | 1.2 s ✓ | 1.9 s ✓ | 324.5 ms ✓ | 358.6 ms ✓ | — | — | — | — | — | — |
| 2500×50 | — | — | — | **19.0 ms ✓** | — | 19.4 ms ✓ | 20.0 ms ✓ | 74.0 ms ✓ | 88.0 ms ✓ | 30.2 ms ✓ | 33.3 ms ✓ | — | — | — | — | — | — |
| 2500×500 | — | — | — | **217.6 ms ✓** | — | 227.5 ms ✓ | 220.4 ms ✓ | 1.6 s ✓ | 1.4 s ✓ | 339.2 ms ✓ | 379.7 ms ✓ | — | — | — | — | — | — |
| 2500×2500 | — | — | — | 1.1 s ✓ | — | **1.1 s ✓** | 1.1 s ✓ | 7.7 s ✓ | 9.0 s ✓ | 1.8 s ✓ | 1.9 s ✓ | — | — | — | — | — | — |
| 10000×50 | — | — | — | 76.9 ms ✓ | — | 78.8 ms ✓ | **74.9 ms ✓** | 463.5 ms ✓ | 586.0 ms ✓ | 118.1 ms ✓ | 156.4 ms ✓ | — | — | — | — | — | — |
| 10000×500 | — | — | — | 867.6 ms ✓ | — | 884.6 ms ✓ | **853.1 ms ✓** | 6.3 s ✓ | 6.5 s ✓ | 1.3 s ✓ | 1.4 s ✓ | — | — | — | — | — | — |


## o2pls  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×500 | — | — | — | — | — | — | — | **30.5 ms ✓** | 60.5 ms ✓ | — | — | — | — | — | — | — | — |
| 500×500 | — | — | — | — | — | — | — | **177.0 ms ✓** | 232.5 ms ✓ | — | — | — | — | — | — | — | — |
| 2500×500 | — | — | — | — | — | — | — | 1.5 s ✓ | **1.4 s ✓** | — | — | — | — | — | — | — | — |
| 2500×2500 | — | — | — | — | — | — | — | **8.2 s ✓** | 9.3 s ✓ | — | — | — | — | — | — | — | — |


## o2pls  —  10 threads

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×500 | — | — | — | — | — | — | — | **31.5 ms ✓** | 62.5 ms ✓ | — | — | — | — | — | — | — | — |
| 500×500 | — | — | — | — | — | — | — | **178.0 ms ✓** | 229.0 ms ✓ | — | — | — | — | — | — | — | — |
| 2500×500 | — | — | — | — | — | — | — | 1.5 s ✓ | **1.4 s ✓** | — | — | — | — | — | — | — | — |
| 2500×2500 | — | — | — | — | — | — | — | **8.1 s ✓** | 9.2 s ✓ | — | — | — | — | — | — | — | — |


## pds  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2500×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2500×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pds  —  10 threads

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2500×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2500×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 0.92 ms ✓ | 0.96 ms ✓ | 0.96 ms ✓ | **0.92 ms ✓** | 0.93 ms ✓ | 0.92 ms ✓ | 1.05 ms ✓ | 2.50 ms ✓ | 6.00 ms ✗ | 1.59 ms ✓ | 5.23 ms ✗ | 1.40 ms ✓ | 1.16 ms ✗ | 6.50 ms ✓ | — | 8.50 ms ✓ | 2.38 ms ✓ |
| 100×500 | 8.16 ms ✓ | 7.97 ms ✓ | **7.78 ms ✓** | 8.30 ms ✓ | 7.97 ms ✓ | 7.96 ms ✓ | 8.22 ms ✓ | 39.5 ms ✓ | 74.5 ms ✗ | 12.9 ms ✓ | 17.2 ms ✗ | 8.76 ms ✓ | 8.09 ms ✗ | 78.5 ms ✓ | — | 67.5 ms ✓ | 14.2 ms ✓ |
| 100×2500 | 40.7 ms ✓ | 40.0 ms ✓ | **38.4 ms ✓** | 39.8 ms ✓ | 40.1 ms ✓ | 40.1 ms ✓ | 38.9 ms ✓ | 323.0 ms ✓ | 645.0 ms ⚠ | 65.0 ms ✓ | 68.0 ms ⚠ | 40.6 ms ✓ | 38.8 ms ⚠ | 635.0 ms ✓ | — | 442.5 ms ✓ | 66.7 ms ✓ |
| 500×50 | 4.20 ms ✓ | 4.12 ms ✓ | 4.10 ms ✓ | **4.05 ms ✓** | 4.24 ms ✓ | 4.16 ms ✓ | 4.30 ms ✓ | 11.5 ms ✓ | 16.0 ms ✗ | 6.85 ms ✓ | 10.2 ms ✗ | 4.31 ms ✓ | 4.14 ms ✗ | 16.0 ms ✓ | — | 27.5 ms ✓ | 7.33 ms ✓ |
| 500×500 | 41.6 ms ✓ | 41.3 ms ✓ | 40.4 ms ✓ | 39.7 ms ✓ | 40.5 ms ✓ | 38.4 ms ✓ | **38.2 ms ✓** | 201.0 ms ✓ | 244.0 ms ✗ | 63.4 ms ✓ | 69.1 ms ✗ | 40.6 ms ✓ | 38.3 ms ✗ | 245.0 ms ✓ | — | 295.5 ms ✓ | 65.6 ms ✓ |
| 500×2500 | 206.9 ms ✓ | 200.5 ms ✓ | 202.9 ms ✓ | 191.2 ms ✓ | 203.3 ms ✓ | 191.0 ms ✓ | **182.1 ms ✓** | 1.4 s ✓ | 1.8 s ✗ | 321.0 ms ✓ | 317.4 ms ✗ | 192.2 ms ✓ | 181.5 ms ✗ | 1.7 s ✓ | — | 1.5 s ✓ | 355.1 ms ✓ |
| 2500×50 | 20.3 ms ✓ | 20.3 ms ✓ | 20.2 ms ✓ | 20.0 ms ✓ | 20.0 ms ✓ | 19.7 ms ✓ | **19.5 ms ✓** | 72.5 ms ✓ | 74.5 ms ⚠ | 30.6 ms ✓ | 34.4 ms ⚠ | 20.0 ms ✓ | 18.9 ms ⚠ | 83.5 ms ✓ | — | 105.5 ms ✓ | 33.7 ms ✓ |
| 2500×500 | 225.0 ms ✓ | 222.9 ms ✓ | 225.5 ms ✓ | 219.2 ms ✓ | 220.1 ms ✓ | 213.2 ms ✓ | 207.5 ms ✓ | 1.3 s ✓ | 1.4 s ✗ | 334.9 ms ✓ | 350.7 ms ✗ | **189.8 ms ✓** | 177.6 ms ✗ | 1.4 s ✓ | — | 1.3 s ✓ | 335.5 ms ✓ |
| 2500×2500 | 1.2 s ✓ | 1.2 s ✓ | 1.3 s ✓ | 1.1 s ✓ | 1.2 s ✓ | 1.1 s ✓ | 1.1 s ✓ | 7.6 s ✓ | 9.1 s ✗ | 1.9 s ✓ | 1.9 s ✗ | **1.0 s ✓** | 930.1 ms ✗ | 8.3 s ✓ | — | 7.9 s ✓ | 2.0 s ✓ |
| 10000×50 | 82.8 ms ✓ | 81.1 ms ✓ | 80.5 ms ✓ | 78.0 ms ✓ | 82.5 ms ✓ | 77.2 ms ✓ | **76.4 ms ✓** | 418.5 ms ✓ | 428.5 ms ⚠ | 120.3 ms ✓ | 128.0 ms ⚠ | 77.5 ms ✓ | 73.9 ms ⚠ | 385.5 ms ✓ | — | 530.5 ms ✓ | 129.9 ms ✓ |
| 10000×500 | 935.9 ms ✓ | 935.8 ms ✓ | 933.5 ms ✓ | 890.4 ms ✓ | 925.0 ms ✓ | 896.0 ms ✓ | 876.4 ms ✓ | 6.3 s ✓ | 6.5 s ⚠ | 1.4 s ✓ | 1.4 s ⚠ | **832.9 ms ✓** | 759.2 ms ⚠ | 6.5 s ✓ | — | 6.7 s ✓ | 1.6 s ✓ |


## pls  —  3 threads

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | 0.90 ms ✓ | — | **0.86 ms ✓** | 1.04 ms ✓ | 2.50 ms ✓ | 6.00 ms ✗ | 1.64 ms ✓ | 5.18 ms ✗ | 1.34 ms ✓ | 1.10 ms ✗ | 6.50 ms ✓ | — | 8.00 ms ✓ | 2.20 ms ✓ |
| 100×500 | — | — | — | 7.90 ms ✓ | — | **7.76 ms ✓** | 8.13 ms ✓ | 38.5 ms ✓ | 76.5 ms ✗ | 13.1 ms ✓ | 16.3 ms ✗ | 9.03 ms ✓ | 8.24 ms ✗ | 78.0 ms ✓ | — | 56.0 ms ✓ | 14.1 ms ✓ |
| 100×2500 | — | — | — | 39.3 ms ✓ | — | 38.6 ms ✓ | **38.5 ms ✓** | 326.5 ms ✓ | 632.5 ms ⚠ | 63.4 ms ✓ | 69.8 ms ⚠ | 39.4 ms ✓ | 38.1 ms ⚠ | 626.0 ms ✓ | — | 471.0 ms ✓ | 67.1 ms ✓ |
| 500×50 | — | — | — | 4.19 ms ✓ | — | 4.13 ms ✓ | **4.03 ms ✓** | 12.5 ms ✓ | 15.0 ms ✗ | 6.65 ms ✓ | 9.57 ms ✗ | 4.28 ms ✓ | 4.09 ms ✗ | 17.5 ms ✓ | — | 26.0 ms ✓ | 7.90 ms ✓ |
| 500×500 | — | — | — | 39.1 ms ✓ | — | 39.3 ms ✓ | 38.6 ms ✓ | 186.5 ms ✓ | 248.0 ms ✗ | 63.9 ms ✓ | 69.0 ms ✗ | **38.4 ms ✓** | 36.9 ms ✗ | 238.5 ms ✓ | — | 317.0 ms ✓ | 67.6 ms ✓ |
| 500×2500 | — | — | — | 199.8 ms ✓ | — | 195.2 ms ✓ | **188.6 ms ✓** | 1.5 s ✓ | 1.7 s ✗ | 319.0 ms ✓ | 320.6 ms ✗ | 197.7 ms ✓ | 180.2 ms ✗ | 1.7 s ✓ | — | 1.4 s ✓ | 352.6 ms ✓ |
| 2500×50 | — | — | — | 20.1 ms ✓ | — | 19.6 ms ✓ | **19.2 ms ✓** | 72.0 ms ✓ | 77.5 ms ⚠ | 31.0 ms ✓ | 34.7 ms ⚠ | 20.1 ms ✓ | 18.4 ms ⚠ | 82.5 ms ✓ | — | 107.5 ms ✓ | 34.9 ms ✓ |
| 2500×500 | — | — | — | 219.8 ms ✓ | — | 217.7 ms ✓ | 211.8 ms ✓ | 1.3 s ✓ | 1.4 s ✗ | 342.8 ms ✓ | 361.2 ms ✗ | **196.2 ms ✓** | 189.2 ms ✗ | 1.4 s ✓ | — | 1.4 s ✓ | 327.2 ms ✓ |
| 2500×2500 | — | — | — | 1.1 s ✓ | — | 1.1 s ✓ | 1.1 s ✓ | 7.9 s ✓ | 9.3 s ✗ | 2.2 s ✓ | 2.1 s ✗ | **1.0 s ✓** | 967.7 ms ✗ | 8.9 s ✓ | — | 9.6 s ✓ | 2.5 s ✓ |
| 10000×50 | — | — | — | 78.1 ms ✓ | — | **75.3 ms ✓** | 80.0 ms ✓ | 438.0 ms ✓ | 425.5 ms ⚠ | 119.5 ms ✓ | 132.6 ms ⚠ | 84.6 ms ✓ | 78.5 ms ⚠ | 387.5 ms ✓ | — | 554.5 ms ✓ | 138.8 ms ✓ |
| 10000×500 | — | — | — | 896.7 ms ✓ | — | 899.9 ms ✓ | 904.9 ms ✓ | 6.3 s ✓ | 6.6 s ⚠ | 1.5 s ✓ | 1.5 s ⚠ | **829.4 ms ✓** | 752.5 ms ⚠ | 6.6 s ✓ | — | 7.1 s ✓ | 1.6 s ✓ |


## pls  —  10 threads

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | **0.86 ms ✓** | — | 0.89 ms ✓ | 1.06 ms ✓ | 3.00 ms ✓ | 5.50 ms ✗ | 1.66 ms ✓ | 5.68 ms ✗ | 1.46 ms ✓ | 1.12 ms ✗ | 5.50 ms ✓ | — | 9.00 ms ✓ | 2.32 ms ✓ |
| 100×500 | — | — | — | **7.76 ms ✓** | — | 7.87 ms ✓ | 7.93 ms ✓ | 37.0 ms ✓ | 75.5 ms ✗ | 13.4 ms ✓ | 16.7 ms ✗ | 8.72 ms ✓ | 8.05 ms ✗ | 77.0 ms ✓ | — | 59.0 ms ✓ | 14.1 ms ✓ |
| 100×2500 | — | — | — | 39.6 ms ✓ | — | **37.9 ms ✓** | 38.5 ms ✓ | 331.5 ms ✓ | 648.0 ms ⚠ | 64.8 ms ✓ | 70.4 ms ⚠ | 39.6 ms ✓ | 38.0 ms ⚠ | 623.5 ms ✓ | — | 467.5 ms ✓ | 68.4 ms ✓ |
| 500×50 | — | — | — | 4.06 ms ✓ | — | 4.07 ms ✓ | **3.90 ms ✓** | 10.5 ms ✓ | 15.5 ms ✗ | 6.71 ms ✓ | 9.56 ms ✗ | 4.31 ms ✓ | 4.10 ms ✗ | 17.5 ms ✓ | — | 27.0 ms ✓ | 7.86 ms ✓ |
| 500×500 | — | — | — | 39.8 ms ✓ | — | 39.1 ms ✓ | **37.7 ms ✓** | 203.0 ms ✓ | 247.0 ms ✗ | 59.9 ms ✓ | 68.1 ms ✗ | 40.7 ms ✓ | 39.3 ms ✗ | 253.0 ms ✓ | — | 277.5 ms ✓ | 66.1 ms ✓ |
| 500×2500 | — | — | — | 190.7 ms ✓ | — | 187.1 ms ✓ | **178.1 ms ✓** | 1.5 s ✓ | 1.7 s ✗ | 320.9 ms ✓ | 355.1 ms ✗ | 241.5 ms ✓ | 221.1 ms ✗ | 1.8 s ✓ | — | 1.6 s ✓ | 404.2 ms ✓ |
| 2500×50 | — | — | — | 20.2 ms ✓ | — | **19.8 ms ✓** | 20.7 ms ✓ | 70.5 ms ✓ | 79.5 ms ⚠ | 31.6 ms ✓ | 34.8 ms ⚠ | 21.0 ms ✓ | 19.0 ms ⚠ | 81.5 ms ✓ | — | 108.0 ms ✓ | 33.6 ms ✓ |
| 2500×500 | — | — | — | 220.1 ms ✓ | — | 218.6 ms ✓ | **208.6 ms ✓** | 1.3 s ✓ | 1.4 s ✗ | 329.9 ms ✓ | 383.2 ms ✗ | 235.9 ms ✓ | 222.3 ms ✗ | 1.5 s ✓ | — | 1.4 s ✓ | 361.9 ms ✓ |
| 2500×2500 | — | — | — | 1.4 s ✓ | — | 1.4 s ✓ | 1.2 s ✓ | 8.0 s ✓ | 10.0 s ✗ | 1.9 s ✓ | 1.9 s ✗ | **1.1 s ✓** | 1.0 s ✗ | 9.1 s ✓ | — | 8.1 s ✓ | 2.2 s ✓ |
| 10000×50 | — | — | — | 76.8 ms ✓ | — | 78.3 ms ✓ | **76.4 ms ✓** | 431.5 ms ✓ | 418.5 ms ⚠ | 119.9 ms ✓ | 168.8 ms ⚠ | 124.8 ms ✓ | 112.6 ms ⚠ | 446.0 ms ✓ | — | 648.5 ms ✓ | 169.1 ms ✓ |
| 10000×500 | — | — | — | 963.2 ms ✓ | — | 980.0 ms ✓ | **932.3 ms ✓** | 6.8 s ✓ | 7.8 s ⚠ | 1.5 s ✓ | 1.7 s ⚠ | 943.2 ms ✓ | 823.9 ms ⚠ | 7.9 s ✓ | — | 7.4 s ✓ | 2.0 s ✓ |


## pls_cox  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2500×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2500×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_cox  —  10 threads

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2500×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2500×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_glm  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×500 | — | — | — | 7.38 ms ✓ | — | **7.29 ms ✓** | — | 33.5 ms ✓ | — | — | — | — | — | — | — | — | — |
| 500×500 | — | — | — | **35.5 ms ✓** | — | 36.3 ms ✓ | — | 171.5 ms ✓ | — | — | — | — | — | — | — | — | — |
| 2500×500 | — | — | — | 220.3 ms ✓ | — | **219.8 ms ✓** | — | 1.5 s ✓ | — | — | — | — | — | — | — | — | — |
| 2500×2500 | — | — | — | 1.1 s ✓ | — | **1.1 s ✓** | — | 8.0 s ✓ | — | — | — | — | — | — | — | — | — |


## pls_glm  —  10 threads

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×500 | — | — | — | 7.58 ms ✓ | — | **7.38 ms ✓** | — | 30.5 ms ✓ | — | — | — | — | — | — | — | — | — |
| 500×500 | — | — | — | 36.5 ms ✓ | — | **36.2 ms ✓** | — | 175.0 ms ✓ | — | — | — | — | — | — | — | — | — |
| 2500×500 | — | — | — | 225.7 ms ✓ | — | **216.5 ms ✓** | — | 1.5 s ✓ | — | — | — | — | — | — | — | — | — |
| 2500×2500 | — | — | — | 1.1 s ✓ | — | **1.1 s ✓** | — | 8.7 s ✓ | — | — | — | — | — | — | — | — | — |


## pls_lda  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2500×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2500×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_lda  —  10 threads

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2500×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2500×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_logistic  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2500×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2500×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_logistic  —  10 threads

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2500×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2500×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_qda  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2500×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2500×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_qda  —  10 threads

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2500×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2500×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## ridge_pls  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | — | — | **0.90 ms ✓** | 0.94 ms ✓ | 2.50 ms ✓ | 5.50 ms ✓ | 1.49 ms ✓ | 3.48 ms ✓ | 1.93 ms ✗ | — | — | — | — | — |
| 100×500 | — | — | — | — | — | **9.44 ms ✓** | 9.88 ms ✓ | 34.5 ms ✓ | 63.0 ms ✓ | 14.3 ms ✓ | 16.4 ms ✓ | 9.91 ms ✗ | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | 271.9 ms ✓ | **271.8 ms ✓** | 530.0 ms ✓ | 814.5 ms ✓ | 296.0 ms ✓ | 298.2 ms ✓ | 41.4 ms ✗ | — | — | — | — | — |
| 500×50 | — | — | — | — | — | 4.12 ms ✓ | **4.05 ms ✓** | 10.5 ms ✓ | 15.5 ms ✓ | 6.13 ms ✓ | 8.16 ms ✓ | 4.69 ms ✗ | — | — | — | — | — |
| 500×500 | — | — | — | — | — | **40.4 ms ✓** | 42.0 ms ✓ | 208.5 ms ✓ | 260.5 ms ✓ | 64.4 ms ✓ | 68.1 ms ✓ | 77.0 ms ✗ | — | — | — | — | — |
| 500×2500 | — | — | — | — | — | **479.1 ms ✓** | 481.5 ms ✓ | 1.5 s ✓ | 2.3 s ✓ | 654.2 ms ✓ | 663.3 ms ✓ | 200.9 ms ✗ | — | — | — | — | — |
| 2500×50 | — | — | — | — | — | **20.0 ms ✓** | 20.8 ms ✓ | 78.0 ms ✓ | 91.0 ms ✓ | 32.3 ms ✓ | 34.6 ms ✓ | 20.6 ms ✗ | — | — | — | — | — |
| 2500×500 | — | — | — | — | — | 237.5 ms ✓ | **231.6 ms ✓** | 1.6 s ✓ | 1.5 s ✓ | 365.8 ms ✓ | 368.6 ms ✓ | 204.2 ms ✗ | — | — | — | — | — |
| 2500×2500 | — | — | — | — | — | 1.4 s ✓ | **1.4 s ✓** | 8.5 s ✓ | 9.7 s ✓ | 2.2 s ✓ | 2.2 s ✓ | 1.1 s ✗ | — | — | — | — | — |
| 10000×50 | — | — | — | — | — | **79.4 ms ✓** | 96.8 ms ✓ | 458.5 ms ✓ | 476.0 ms ✓ | 124.5 ms ✓ | 133.3 ms ✓ | 74.1 ms ✗ | — | — | — | — | — |
| 10000×500 | — | — | — | — | — | **924.9 ms ✓** | 931.7 ms ✓ | 6.9 s ✓ | 7.0 s ✓ | 1.4 s ✓ | 1.4 s ✓ | 794.5 ms ✗ | — | — | — | — | — |


## ridge_pls  —  3 threads

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | — | — | **0.89 ms ✓** | 0.92 ms ✓ | 2.50 ms ✓ | 5.50 ms ✓ | 1.47 ms ✓ | 3.69 ms ✓ | 1.93 ms ✗ | — | — | — | — | — |
| 100×500 | — | — | — | — | — | **9.61 ms ✓** | 9.83 ms ✓ | 35.0 ms ✓ | 67.0 ms ✓ | 14.1 ms ✓ | 17.2 ms ✓ | 12.2 ms ✗ | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | 267.8 ms ✓ | **259.7 ms ✓** | 519.5 ms ✓ | 814.5 ms ✓ | 294.3 ms ✓ | 298.4 ms ✓ | 40.4 ms ✗ | — | — | — | — | — |
| 500×50 | — | — | — | — | — | 4.06 ms ✓ | **3.86 ms ✓** | 10.5 ms ✓ | 15.5 ms ✓ | 6.39 ms ✓ | 8.60 ms ✓ | 4.84 ms ✗ | — | — | — | — | — |
| 500×500 | — | — | — | — | — | 42.2 ms ✓ | **41.6 ms ✓** | 203.0 ms ✓ | 258.0 ms ✓ | 65.1 ms ✓ | 68.4 ms ✓ | 71.0 ms ✗ | — | — | — | — | — |
| 500×2500 | — | — | — | — | — | 523.6 ms ✓ | **486.1 ms ✓** | 1.4 s ✓ | 2.2 s ✓ | 774.7 ms ✓ | 739.3 ms ✓ | 199.8 ms ✗ | — | — | — | — | — |
| 2500×50 | — | — | — | — | — | **19.6 ms ✓** | 19.9 ms ✓ | 76.5 ms ✓ | 92.5 ms ✓ | 31.2 ms ✓ | 34.6 ms ✓ | 20.4 ms ✗ | — | — | — | — | — |
| 2500×500 | — | — | — | — | — | 241.7 ms ✓ | **234.0 ms ✓** | 1.6 s ✓ | 1.5 s ✓ | 374.2 ms ✓ | 380.1 ms ✓ | 205.1 ms ✗ | — | — | — | — | — |
| 2500×2500 | — | — | — | — | — | 1.4 s ✓ | **1.4 s ✓** | 8.7 s ✓ | 9.6 s ✓ | 2.1 s ✓ | 2.2 s ✓ | 997.3 ms ✗ | — | — | — | — | — |
| 10000×50 | — | — | — | — | — | **78.8 ms ✓** | 80.1 ms ✓ | 518.0 ms ✓ | 456.5 ms ✓ | 126.6 ms ✓ | 134.6 ms ✓ | 75.6 ms ✗ | — | — | — | — | — |
| 10000×500 | — | — | — | — | — | **910.2 ms ✓** | 920.9 ms ✓ | 7.1 s ✓ | 7.0 s ✓ | 1.5 s ✓ | 1.5 s ✓ | 772.0 ms ✗ | — | — | — | — | — |


## ridge_pls  —  10 threads

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | — | — | **0.89 ms ✓** | 0.95 ms ✓ | 2.50 ms ✓ | 5.00 ms ✓ | 1.70 ms ✓ | 3.54 ms ✓ | 3.51 ms ✗ | — | — | — | — | — |
| 100×500 | — | — | — | — | — | 10.0 ms ✓ | **9.65 ms ✓** | 35.5 ms ✓ | 68.0 ms ✓ | 14.6 ms ✓ | 16.0 ms ✓ | 20.5 ms ✗ | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | 272.5 ms ✓ | **261.5 ms ✓** | 525.5 ms ✓ | 807.5 ms ✓ | 298.0 ms ✓ | 299.5 ms ✓ | 77.5 ms ✗ | — | — | — | — | — |
| 500×50 | — | — | — | — | — | 3.99 ms ✓ | **3.98 ms ✓** | 11.0 ms ✓ | 15.5 ms ✓ | 6.79 ms ✓ | 8.69 ms ✓ | 4.65 ms ✗ | — | — | — | — | — |
| 500×500 | — | — | — | — | — | 41.0 ms ✓ | **40.2 ms ✓** | 190.0 ms ✓ | 248.0 ms ✓ | 63.1 ms ✓ | 69.0 ms ✓ | 103.0 ms ✗ | — | — | — | — | — |
| 500×2500 | — | — | — | — | — | 496.6 ms ✓ | **487.0 ms ✓** | 1.6 s ✓ | 2.7 s ✓ | 773.2 ms ✓ | 679.3 ms ✓ | 241.2 ms ✗ | — | — | — | — | — |
| 2500×50 | — | — | — | — | — | **19.5 ms ✓** | 20.3 ms ✓ | 75.5 ms ✓ | 77.0 ms ✓ | 31.7 ms ✓ | 35.4 ms ✓ | 19.4 ms ✗ | — | — | — | — | — |
| 2500×500 | — | — | — | — | — | 250.3 ms ✓ | **237.0 ms ✓** | 1.8 s ✓ | 1.6 s ✓ | 360.1 ms ✓ | 396.7 ms ✓ | 233.7 ms ✗ | — | — | — | — | — |
| 2500×2500 | — | — | — | — | — | 1.4 s ✓ | **1.4 s ✓** | 8.6 s ✓ | 10.7 s ✓ | 2.4 s ✓ | 2.4 s ✓ | 1.0 s ✗ | — | — | — | — | — |
| 10000×50 | — | — | — | — | — | **79.7 ms ✓** | 82.5 ms ✓ | 480.0 ms ✓ | 472.0 ms ✓ | 126.3 ms ✓ | 168.2 ms ✓ | 99.3 ms ✗ | — | — | — | — | — |
| 10000×500 | — | — | — | — | — | **937.6 ms ✓** | 989.7 ms ✓ | 6.7 s ✓ | 7.2 s ✓ | 1.5 s ✓ | 1.5 s ✓ | 803.5 ms ✗ | — | — | — | — | — |


## robust_pls  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | — | — | **1.35 ms ✓** | 1.60 ms ✓ | 3.00 ms ✓ | 6.50 ms ✓ | 2.11 ms ✓ | 4.11 ms ✓ | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | 13.9 ms ✓ | **12.8 ms ✓** | 38.0 ms ✓ | 68.0 ms ✓ | 17.8 ms ✓ | 21.5 ms ✓ | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | **70.8 ms ✓** | 80.1 ms ✓ | 358.5 ms ✓ | 667.0 ms ✓ | 93.7 ms ✓ | 93.5 ms ✓ | — | — | — | — | — | — |
| 500×50 | — | — | — | — | — | 6.69 ms ✓ | **6.24 ms ✓** | 13.0 ms ✓ | 17.0 ms ✓ | 9.53 ms ✓ | 10.9 ms ✓ | — | — | — | — | — | — |
| 500×500 | — | — | — | — | — | 90.1 ms ✓ | **87.7 ms ✓** | 252.0 ms ✓ | 282.5 ms ✓ | 92.8 ms ✓ | 94.5 ms ✓ | — | — | — | — | — | — |
| 500×2500 | — | — | — | — | — | 499.7 ms ✓ | **493.8 ms ✓** | 1.4 s ✓ | 2.2 s ✓ | 541.8 ms ✓ | 552.4 ms ✓ | — | — | — | — | — | — |
| 2500×50 | — | — | — | — | — | **37.7 ms ✓** | 42.5 ms ✓ | 104.5 ms ✓ | 115.5 ms ✓ | 46.3 ms ✓ | 49.7 ms ✓ | — | — | — | — | — | — |
| 2500×500 | — | — | — | — | — | 1.3 s ✓ | **1.1 s ✓** | 2.3 s ✓ | 2.5 s ✓ | 1.3 s ✓ | 1.3 s ✓ | — | — | — | — | — | — |
| 2500×2500 | — | — | — | — | — | 6.6 s ✓ | **5.9 s ✓** | 12.4 s ✓ | 13.3 s ✓ | 6.9 s ✓ | 8.4 s ✓ | — | — | — | — | — | — |
| 10000×50 | — | — | — | — | — | 204.8 ms ✓ | **192.6 ms ✓** | 559.5 ms ✓ | 533.0 ms ✓ | 206.4 ms ✓ | 278.2 ms ✓ | — | — | — | — | — | — |
| 10000×500 | — | — | — | — | — | 4.4 s ✓ | **4.4 s ✓** | 10.4 s ✓ | 10.1 s ✓ | 4.8 s ✓ | 4.9 s ✓ | — | — | — | — | — | — |


## robust_pls  —  3 threads

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | — | — | **1.40 ms ✓** | 1.49 ms ✓ | 3.00 ms ✓ | 6.00 ms ✓ | 1.99 ms ✓ | 4.39 ms ✓ | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | 14.1 ms ✓ | **13.1 ms ✓** | 42.5 ms ✓ | 67.0 ms ✓ | 18.1 ms ✓ | 20.9 ms ✓ | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | **69.8 ms ✓** | 95.5 ms ✓ | 394.5 ms ✓ | 693.0 ms ✓ | 91.1 ms ✓ | 93.9 ms ✓ | — | — | — | — | — | — |
| 500×50 | — | — | — | — | — | 6.92 ms ✓ | **6.39 ms ✓** | 14.0 ms ✓ | 18.0 ms ✓ | 9.51 ms ✓ | 11.3 ms ✓ | — | — | — | — | — | — |
| 500×500 | — | — | — | — | — | 89.5 ms ✓ | **88.3 ms ✓** | 248.5 ms ✓ | 292.5 ms ✓ | 92.4 ms ✓ | 102.0 ms ✓ | — | — | — | — | — | — |
| 500×2500 | — | — | — | — | — | 520.5 ms ✓ | **509.1 ms ✓** | 1.5 s ✓ | 2.2 s ✓ | 555.2 ms ✓ | 635.6 ms ✓ | — | — | — | — | — | — |
| 2500×50 | — | — | — | — | — | **39.3 ms ✓** | 42.3 ms ✓ | 101.5 ms ✓ | 115.0 ms ✓ | 68.2 ms ✓ | 51.3 ms ✓ | — | — | — | — | — | — |
| 2500×500 | — | — | — | — | — | 1.2 s ✓ | **1.2 s ✓** | 2.5 s ✓ | 2.8 s ✓ | 1.3 s ✓ | 1.3 s ✓ | — | — | — | — | — | — |
| 2500×2500 | — | — | — | — | — | 6.1 s ✓ | **5.3 s ✓** | 12.2 s ✓ | 14.2 s ✓ | 6.8 s ✓ | 7.0 s ✓ | — | — | — | — | — | — |
| 10000×50 | — | — | — | — | — | 193.9 ms ✓ | **187.2 ms ✓** | 708.0 ms ✓ | 562.5 ms ✓ | 199.6 ms ✓ | 220.9 ms ✓ | — | — | — | — | — | — |
| 10000×500 | — | — | — | — | — | **4.3 s ✓** | 4.5 s ✓ | 9.9 s ✓ | 10.1 s ✓ | 5.2 s ✓ | 5.0 s ✓ | — | — | — | — | — | — |


## robust_pls  —  10 threads

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | — | — | **1.33 ms ✓** | 1.58 ms ✓ | 3.00 ms ✓ | 6.00 ms ✓ | 2.12 ms ✓ | 4.75 ms ✓ | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | 14.5 ms ✓ | **13.1 ms ✓** | 40.5 ms ✓ | 68.5 ms ✓ | 17.4 ms ✓ | 20.8 ms ✓ | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | **70.9 ms ✓** | 77.5 ms ✓ | 355.0 ms ✓ | 661.0 ms ✓ | 93.5 ms ✓ | 97.6 ms ✓ | — | — | — | — | — | — |
| 500×50 | — | — | — | — | — | 6.75 ms ✓ | **6.71 ms ✓** | 13.5 ms ✓ | 17.5 ms ✓ | 8.87 ms ✓ | 11.1 ms ✓ | — | — | — | — | — | — |
| 500×500 | — | — | — | — | — | **87.3 ms ✓** | 90.7 ms ✓ | 243.0 ms ✓ | 347.0 ms ✓ | 90.3 ms ✓ | 93.9 ms ✓ | — | — | — | — | — | — |
| 500×2500 | — | — | — | — | — | **544.8 ms ✓** | 554.8 ms ✓ | 1.6 s ✓ | 2.3 s ✓ | 591.7 ms ✓ | 711.7 ms ✓ | — | — | — | — | — | — |
| 2500×50 | — | — | — | — | — | **41.7 ms ✓** | 42.4 ms ✓ | 103.5 ms ✓ | 117.0 ms ✓ | 45.7 ms ✓ | 50.3 ms ✓ | — | — | — | — | — | — |
| 2500×500 | — | — | — | — | — | 1.2 s ✓ | **1.1 s ✓** | 2.2 s ✓ | 2.3 s ✓ | 1.3 s ✓ | 1.4 s ✓ | — | — | — | — | — | — |
| 2500×2500 | — | — | — | — | — | 6.1 s ✓ | **5.6 s ✓** | 11.9 s ✓ | 13.1 s ✓ | 6.5 s ✓ | 6.3 s ✓ | — | — | — | — | — | — |
| 10000×50 | — | — | — | — | — | 196.7 ms ✓ | **183.1 ms ✓** | 553.5 ms ✓ | 613.0 ms ✓ | 203.2 ms ✓ | 238.9 ms ✓ | — | — | — | — | — | — |
| 10000×500 | — | — | — | — | — | 5.2 s ✓ | **4.6 s ✓** | 11.0 s ✓ | 10.7 s ✓ | 5.7 s ✓ | 5.8 s ✓ | — | — | — | — | — | — |


## sparse_pls_da  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2500×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2500×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## sparse_pls_da  —  10 threads

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2500×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2500×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## sparse_simpls  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | **0.87 ms ✓** | — | 0.92 ms ✓ | 1.10 ms ✓ | 2.50 ms ✓ | 6.50 ms ✓ | 1.66 ms ✓ | 3.78 ms ✓ | — | — | — | — | 8.50 ms ✗ | — |
| 100×500 | — | — | — | **8.18 ms ✓** | — | 8.20 ms ✓ | 8.25 ms ✓ | 38.0 ms ✓ | 70.0 ms ✓ | 14.0 ms ✓ | 15.6 ms ✓ | — | — | — | — | 67.5 ms ✗ | — |
| 100×2500 | — | — | — | 39.6 ms ✓ | — | **37.4 ms ✓** | 42.2 ms ✓ | 354.0 ms ✓ | 674.0 ms ✓ | 65.0 ms ✓ | 68.7 ms ✓ | — | — | — | — | 503.0 ms ✗ | — |
| 500×50 | — | — | — | **4.22 ms ✓** | — | 4.30 ms ✓ | 4.29 ms ✓ | 11.0 ms ✓ | 17.5 ms ✓ | 6.78 ms ✓ | 8.97 ms ✓ | — | — | — | — | 31.5 ms ✗ | — |
| 500×500 | — | — | — | 40.8 ms ✓ | — | **39.4 ms ✓** | 40.7 ms ✓ | 214.0 ms ✓ | 268.0 ms ✓ | 64.4 ms ✓ | 70.2 ms ✓ | — | — | — | — | 371.0 ms ✗ | — |
| 500×2500 | — | — | — | 203.4 ms ✓ | — | **202.7 ms ✓** | 203.3 ms ✓ | 1.5 s ✓ | 1.9 s ✓ | 339.4 ms ✓ | 345.6 ms ✓ | — | — | — | — | 1.5 s ✗ | — |
| 2500×50 | — | — | — | 20.7 ms ✓ | — | **20.5 ms ✓** | 20.7 ms ✓ | 80.0 ms ✓ | 107.5 ms ✓ | 36.5 ms ✓ | 34.0 ms ✓ | — | — | — | — | 134.0 ms ✗ | — |
| 2500×500 | — | — | — | 247.8 ms ✓ | — | 246.1 ms ✓ | **231.7 ms ✓** | 1.5 s ✓ | 1.5 s ✓ | 403.3 ms ✓ | 397.1 ms ✓ | — | — | — | — | 1.7 s ✗ | — |
| 2500×2500 | — | — | — | 1.3 s ✓ | — | 1.2 s ✓ | **1.2 s ✓** | 8.4 s ✓ | 8.9 s ✓ | 2.0 s ✓ | 2.0 s ✓ | — | — | — | — | 9.2 s ✗ | — |
| 10000×50 | — | — | — | 77.4 ms ✓ | — | **76.4 ms ✓** | 79.0 ms ✓ | 446.0 ms ✓ | 510.0 ms ✓ | 120.2 ms ✓ | 123.7 ms ✓ | — | — | — | — | 569.5 ms ✗ | — |
| 10000×500 | — | — | — | 954.7 ms ✓ | — | 966.3 ms ✓ | **926.1 ms ✓** | 6.5 s ✓ | 6.9 s ✓ | 1.5 s ✓ | 1.4 s ✓ | — | — | — | — | 7.1 s ✗ | — |


## sparse_simpls  —  3 threads

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | **0.87 ms ✓** | — | 0.90 ms ✓ | 1.16 ms ✓ | 2.50 ms ✓ | 6.00 ms ✓ | 1.62 ms ✓ | 3.85 ms ✓ | — | — | — | — | 9.50 ms ✗ | — |
| 100×500 | — | — | — | 8.11 ms ✓ | — | **7.87 ms ✓** | 8.02 ms ✓ | 39.0 ms ✓ | 81.5 ms ✓ | 13.4 ms ✓ | 15.8 ms ✓ | — | — | — | — | 60.5 ms ✗ | — |
| 100×2500 | — | — | — | 39.3 ms ✓ | — | **38.4 ms ✓** | 40.1 ms ✓ | 345.5 ms ✓ | 670.5 ms ✓ | 65.5 ms ✓ | 68.2 ms ✓ | — | — | — | — | 484.5 ms ✗ | — |
| 500×50 | — | — | — | **4.16 ms ✓** | — | 4.34 ms ✓ | 4.40 ms ✓ | 12.0 ms ✓ | 18.0 ms ✓ | 6.72 ms ✓ | 8.84 ms ✓ | — | — | — | — | 29.0 ms ✗ | — |
| 500×500 | — | — | — | 40.8 ms ✓ | — | 48.8 ms ✓ | **40.3 ms ✓** | 199.5 ms ✓ | 261.0 ms ✓ | 62.9 ms ✓ | 66.5 ms ✓ | — | — | — | — | 331.5 ms ✗ | — |
| 500×2500 | — | — | — | 199.3 ms ✓ | — | **197.4 ms ✓** | 206.5 ms ✓ | 1.5 s ✓ | 2.0 s ✓ | 348.3 ms ✓ | 347.0 ms ✓ | — | — | — | — | 1.6 s ✗ | — |
| 2500×50 | — | — | — | 20.0 ms ✓ | — | 20.4 ms ✓ | **19.9 ms ✓** | 76.5 ms ✓ | 112.5 ms ✓ | 31.6 ms ✓ | 34.7 ms ✓ | — | — | — | — | 136.0 ms ✗ | — |
| 2500×500 | — | — | — | 251.7 ms ✓ | — | 241.4 ms ✓ | **233.6 ms ✓** | 1.5 s ✓ | 1.4 s ✓ | 361.1 ms ✓ | 367.3 ms ✓ | — | — | — | — | 1.5 s ✗ | — |
| 2500×2500 | — | — | — | 1.2 s ✓ | — | **1.2 s ✓** | 1.3 s ✓ | 8.4 s ✓ | 9.7 s ✓ | 1.9 s ✓ | 2.0 s ✓ | — | — | — | — | 8.9 s ✗ | — |
| 10000×50 | — | — | — | 75.2 ms ✓ | — | **74.8 ms ✓** | 79.3 ms ✓ | 438.5 ms ✓ | 488.0 ms ✓ | 119.1 ms ✓ | 133.2 ms ✓ | — | — | — | — | 592.5 ms ✗ | — |
| 10000×500 | — | — | — | 944.5 ms ✓ | — | 942.7 ms ✓ | **919.6 ms ✓** | 6.4 s ✓ | 6.8 s ✓ | 1.4 s ✓ | 1.5 s ✓ | — | — | — | — | 6.9 s ✗ | — |


## sparse_simpls  —  10 threads

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | 0.97 ms ✓ | — | **0.94 ms ✓** | 1.09 ms ✓ | 2.50 ms ✓ | 6.00 ms ✓ | 1.66 ms ✓ | 6.67 ms ✓ | — | — | — | — | 8.50 ms ✗ | — |
| 100×500 | — | — | — | 9.12 ms ✓ | — | **7.89 ms ✓** | 8.06 ms ✓ | 40.0 ms ✓ | 76.0 ms ✓ | 13.4 ms ✓ | 15.8 ms ✓ | — | — | — | — | 60.0 ms ✗ | — |
| 100×2500 | — | — | — | 39.7 ms ✓ | — | 40.3 ms ✓ | **39.0 ms ✓** | 334.5 ms ✓ | 659.5 ms ✓ | 68.1 ms ✓ | 72.8 ms ✓ | — | — | — | — | 482.0 ms ✗ | — |
| 500×50 | — | — | — | **4.09 ms ✓** | — | 4.41 ms ✓ | 4.45 ms ✓ | 12.5 ms ✓ | 17.5 ms ✓ | 6.61 ms ✓ | 8.68 ms ✓ | — | — | — | — | 32.0 ms ✗ | — |
| 500×500 | — | — | — | 40.3 ms ✓ | — | **39.4 ms ✓** | 40.6 ms ✓ | 205.5 ms ✓ | 262.0 ms ✓ | 64.7 ms ✓ | 68.5 ms ✓ | — | — | — | — | 346.0 ms ✗ | — |
| 500×2500 | — | — | — | **198.5 ms ✓** | — | 200.5 ms ✓ | 236.6 ms ✓ | 1.7 s ✓ | 2.0 s ✓ | 363.2 ms ✓ | 373.5 ms ✓ | — | — | — | — | 1.6 s ✗ | — |
| 2500×50 | — | — | — | 20.0 ms ✓ | — | **20.0 ms ✓** | 20.2 ms ✓ | 96.5 ms ✓ | 94.5 ms ✓ | 35.1 ms ✓ | 34.9 ms ✓ | — | — | — | — | 140.5 ms ✗ | — |
| 2500×500 | — | — | — | 241.2 ms ✓ | — | **238.8 ms ✓** | 260.5 ms ✓ | 1.4 s ✓ | 1.5 s ✓ | 356.7 ms ✓ | 399.6 ms ✓ | — | — | — | — | 1.7 s ✗ | — |
| 2500×2500 | — | — | — | **1.2 s ✓** | — | 1.3 s ✓ | 1.3 s ✓ | 8.2 s ✓ | 8.9 s ✓ | 1.9 s ✓ | 2.0 s ✓ | — | — | — | — | 8.9 s ✗ | — |
| 10000×50 | — | — | — | **76.3 ms ✓** | — | 80.9 ms ✓ | 114.3 ms ✓ | 434.0 ms ✓ | 516.0 ms ✓ | 120.2 ms ✓ | 163.2 ms ✓ | — | — | — | — | 643.0 ms ✗ | — |
| 10000×500 | — | — | — | 969.8 ms ✓ | — | **958.2 ms ✓** | 962.9 ms ✓ | 6.5 s ✓ | 7.0 s ✓ | 1.4 s ✓ | 1.5 s ✓ | — | — | — | — | 7.1 s ✗ | — |


## weighted_pls  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2500×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2500×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## weighted_pls  —  10 threads

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.cpp.cuda | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2500×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2500×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## Backend definitions

Each column in the per-algorithm tables above is one of the entries below. Columns are named `owner.language[.variant]`: `pls4all.*` are this project's own bindings; everything else is an external library shipped by its own maintainers.

| Column | Language | Tier | What it actually runs |
|---|---|---|---|
| `pls4all.cpp.ref` | C++ | pls4all reference (single-thread) | libp4a built with `PLS4ALL_WITH_BLAS=OFF, OPENMP=OFF` — pure scalar reference loops, no acceleration. The parity baseline. |
| `pls4all.cpp.blas` | C++ | pls4all + BLAS | libp4a built with `PLS4ALL_WITH_BLAS=ON` only — links system BLAS (OpenBLAS in this env), benefits from BLAS thread parallelism. |
| `pls4all.cpp.omp` | C++ | pls4all + OpenMP | libp4a built with `PLS4ALL_WITH_OPENMP=ON` only — OpenMP parallelism in the C kernel loops, no BLAS. |
| `pls4all.cpp.blas+omp` | C++ | pls4all + BLAS + OpenMP | libp4a built with both `PLS4ALL_WITH_BLAS=ON` and `PLS4ALL_WITH_OPENMP=ON` — the recommended production config. |
| `pls4all.cpp.cuda` | C++ | pls4all + CUDA | libp4a built with `PLS4ALL_WITH_CUDA=ON` — GEMM kernels offloaded to GPU via cuBLAS. Overhead-dominated at small matrix sizes; wins at large ones. |
| `pls4all.python` | Python | pls4all raw | `pls4all._methods.<algo>_fit(ctx, cfg, X, y, …)` — direct FFI binding |
| `pls4all.sklearn` | Python | pls4all idiomatic | `pls4all.sklearn.<Class>` — sklearn-style BaseEstimator with `.fit() / .predict()` |
| `pls4all.R` | R | pls4all raw | `pls4all_method(algo, X, y, ...)` — unified dispatcher (33 fits + 24 selectors + 4 diagnostics) |
| `pls4all.R.formula` | R | pls4all idiomatic | `pls(y ~ ., data)`, `cppls(...)`, `sparse_pls(...)`, … — base R formula+S3 wrappers |
| `pls4all.matlab` | MATLAB/Octave | pls4all raw | `pls4all.<algo>(X, y, ...)` — single dispatcher MEX |
| `pls4all.matlab.classdef` | MATLAB/Octave | pls4all idiomatic | `pls4all.fit(algo, X, y, ...)` factory + per-algorithm classdefs |
| `sklearn` | Python | external | `sklearn.cross_decomposition.PLSRegression`, `sklearn.decomposition.PCA + LinearRegression / Ridge / GaussianProcessRegressor` (proxies) |
| `ikpls` | Python | external | `ikpls.numpy_ikpls.PLS` — Improved Kernel PLS (covers plain PLS only) |
| `pls` | R | external | CRAN `pls` package — `pls::plsr / pls::cppls / pls::pcr` |
| `ropls` | R | external | Bioconductor `ropls` — `ropls::opls` (covers OPLS only) |
| `mixOmics` | R | external | Bioconductor `mixOmics` — `pls / spls / plsda / splsda` |
| `plsregress` | MATLAB/Octave | external | Octave statistics `plsregress` (SIMPLS, plain PLS only) |


## Versions per backend

| Column | Versions |
|---|---|
| `pls4all.cpp.ref` | `language=C++ (via ctypes bridge) (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `libp4a=0.97.0+abi.1.16.0`; `numpy=2.3.5` |
| `pls4all.cpp.blas` | `language=C++ (via ctypes bridge) (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1; libopenblas 0.3.33 blas=1`; `libp4a=0.94.0+abi.1.13.0`; `numpy=2.3.5` |
| `pls4all.cpp.omp` | `language=C++ (via ctypes bridge) (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1; libomp ? openmp=1`; `libp4a=0.94.0+abi.1.13.0`; `numpy=2.3.5` |
| `pls4all.cpp.blas+omp` | `language=C++ (via ctypes bridge) (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1; libopenblas 0.3.33 blas=1; libomp ? openmp=1`; `libp4a=0.94.0+abi.1.13.0`; `numpy=2.3.5` |
| `pls4all.cpp.cuda` | `language=C++ (via ctypes bridge) (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `libp4a=0.96.0+abi.1.13.0`; `numpy=2.3.5` |
| `pls4all.python` | `language=Python 3.13.11`; `blas=libscipy_openblas 0.3.30 blas=1`; `pls4all=0.97.0`; `numpy=2.3.5` |
| `pls4all.sklearn` | `language=Python 3.13.11`; `blas=libscipy_openblas 0.3.30 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `pls4all=0.97.0`; `numpy=2.3.5` |
| `pls4all.R` | `language=R 4.3.3`; `pls4all=0.68.0`; `blas=linked-BLAS (RhpcBLASctl not installed)` |
| `pls4all.R.formula` | `language=R 4.3.3`; `pls4all=0.68.0`; `blas=linked-BLAS` |
| `pls4all.matlab` | `language=Octave 10.3.0`; `pls4all=from libp4a-linked MEX`; `blas=linked-BLAS` |
| `pls4all.matlab.classdef` | `language=Octave 10.3.0`; `pls4all=from libp4a-linked MEX + classdefs`; `blas=linked-BLAS` |
| `sklearn` | `language=Python 3.13.11`; `blas=libscipy_openblas 0.3.30 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `sklearn=1.8.0`; `numpy=2.3.5` |
| `ikpls` | `language=Python 3.13.11`; `blas=libscipy_openblas 0.3.30 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `ikpls=4.0.1.post1`; `numpy=2.3.5` |
| `pls` | `language=R 4.3.3`; `pls=2.8.5` |
| `ropls` | `language=R 4.3.3`; `ropls=1.34.0` |
| `mixOmics` | `language=R 4.3.3`; `mixOmics=6.26.0` |
| `plsregress` | `language=Octave 10.3.0`; `statistics=Octave statistics pkg (plsregress)`; `blas=linked-BLAS` |

## Methodology

- Reference: `cpp` cell at 1 thread (libp4a via ctypes), or `python_tier1` when `cpp` is unavailable for an algorithm
- Parity tolerance: 1e-6 (per-algo overrides possible)
- All backends read the **same** orchestrator-generated CSV (`benchmarks/cross_binding/data/data_<n>x<p>_seed<seed>.csv`) so input data is bit-identical across languages
- 5 runs per cell, first discarded as warmup, median reported
- Per-cell timeout: 300 s
- Thread control via `OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = BLIS_NUM_THREADS` set in the subprocess env, plus `Context.num_threads` for Python pls4all and `maxNumCompThreads()` for Octave
- pls4all libp4a build: `build/blas-omp/cpp/src/libp4a.so` (BLAS + OpenMP enabled)
