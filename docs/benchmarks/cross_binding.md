# Cross-binding benchmark — parity + timing (1 thread)

Single-thread numbers — the cleanest cross-language comparison. **Most external libraries don't honour OMP_NUM_THREADS at the algorithm level** (sklearn, pls::plsr, plsregress, ikpls all run the PLS algo serially even when BLAS is multi-threaded), so a multi-thread comparison would compare pls4all's OpenMP path against everyone else's single-threaded loop. That's a real, useful number — see the [thread sweep page](cross_binding_threads.md) — but this main page sticks to 1 thread for honest apples-to-apples timing.

_Generated: 2026-05-18 07:52:21 UTC_
_CSV: `tmpkqk1ctbp.csv` (5625 cells)_


## Host

| | |
|---|---|
| **CPU**            | 12th Gen Intel(R) Core(TM) i9-12900K |
| **Cores**          | 12 physical / 24 logical |
| **Frequency**  | 3187 MHz nominal |
| **L3 cache**   | 30720 KB |
| **RAM**            | 62.7 GB |
| **OS / kernel**    | Linux 6.6.87.2-microsoft-standard-WSL2 (x86_64) · kernel 6.6.87.2-microsoft-standard-WSL2 |
| **Python**         | 3.14.4 |


## How to read a cell

Each cell shows `<median_ms> <parity icon>`. The icon is the parity verdict vs the reference backend (`pls4all.cpp` at 1 thread, dev-release build):

- ✓ **bit-exact** (max abs diff ≤ 1e-6 vs ref) — this backend produces the *same* predictions as the reference
- ⚠ **close** (≤ 1e-3 but > 1e-6) — algorithmic drift (different convergence path), answer agrees in practice
- ✗ **divergent** (> 1e-3) — different preprocessing convention (e.g. tier-2 wrappers default to `scale=True`, tier-1 / externals default to `scale=False`); the backend works, it just isn't an apples-to-apples comparison
- ⏳ cell timed out (300 s wall-clock)
- `—` see *Why a cell is empty* below

**Bold** = fastest cell on the row, **counted only among ✓ cells**. ⚠ / ✗ / — cells never carry the bold (comparing a non-bit-exact result against bit-exact ones would be misleading).

Timing is the **median of 4, 5 run(s)**; the first run is discarded as warmup when `n_runs >= 3`. All backends in a single cell read the SAME orchestrator-generated CSV so cross-language input bytes are bit-identical. See [methodology.md](methodology.md) for the full details.


## Why a cell is empty (`—`)

An empty cell means the backend **did not produce a timing for that (algorithm, size) combination**. Four distinct reasons, all reported as `—` in the matrix:

1. **External library doesn't implement the algorithm.** Example: `sklearn` has no native CPPLS or sparse SIMPLS; `plsregress` (Octave) only does plain PLS; `ikpls` is plain PLS only; `ropls` only does OPLS; `mixOmics` covers PLS / sparse PLS / PLS-DA. Filling this column requires the external maintainer to add the algorithm — out of our control.
2. **pls4all wrapper missing for that algorithm.** Example: `pls4all.sklearn` (tier 2) doesn't yet ship a class for `continuum_regression` or `kernel_pls`; `pls4all.R.formula` doesn't have a formula wrapper for every algorithm yet. This is the *chemin à parcourir* on our side — visible TODO.
3. **Bench script missing the dispatch case.** A few cells can fail because a binding bench script doesn't yet route a specific algorithm to its underlying call. Pure tooling gap, no library impact.
4. **Run too slow / OOM / crashed.** The cell timed out (`⏳`) or hit a runtime error. Rare in this matrix (300 s timeout is generous for the included sizes).

Each `—` represents one of these four reasons. The CSV (`results/full_matrix.csv`) carries a `reason` column that tells you exactly which one for any given cell.


Column names: anything starting with `pls4all.` is one of this project's bindings; the others use their real package name (`sklearn`, `ikpls`, `pls`, `ropls`, `mixOmics`, `plsregress`). **Every algorithm table keeps every column** — `—` cells are intentional and show *where coverage is still missing*, either on our side (an algorithm not yet wrapped in a tier-2 class) or on the external side (e.g. `sklearn` doesn't implement CPPLS / OPLS, `plsregress` only does plain PLS, `ikpls` only does plain PLS). Full per-column description: see [Backend definitions](#backend-definitions) at the bottom of this page.


## aom_preprocess  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | — | **1.26 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×50 | — | — | — | — | **4.76 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## approximate_press  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | — | **1.27 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 214.4 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×50 | — | — | — | — | **5.30 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 721.1 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## bagging_pls  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.96 ms ✓ | 1.96 ms ✓ | **1.91 ms ✓** | 1.98 ms ✓ | 1.47 ms ✗ | 3.82 ms ✓ | — | 5.00 ms ✓ | 9.00 ms ✓ | — | 5.42 ms ✓ | — | — | — | — | — | — | 12.1 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×50 | 10.1 ms ✓ | 11.1 ms ✓ | 9.83 ms ✓ | 10.1 ms ✓ | 5.81 ms ✗ | **9.72 ms ✓** | — | 21.0 ms ✓ | 28.5 ms ✓ | — | 17.3 ms ✓ | — | — | — | — | — | — | 17.4 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## bipls_select  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | — | **4.05 ms ✓** | — | — | 4.50 ms ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.3 s ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×50 | — | — | — | — | **18.2 ms ✓** | — | — | 26.0 ms ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.6 s ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## boosting_pls  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 2.37 ms ✓ | 2.09 ms ✓ | 2.05 ms ✓ | **2.01 ms ✓** | 2.17 ms ✗ | 2.92 ms ✓ | — | 4.00 ms ✓ | 9.50 ms ✓ | — | 5.58 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 806.8 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×50 | 9.56 ms ✓ | 11.4 ms ✓ | 10.1 ms ✓ | **9.35 ms ✓** | 7.83 ms ✗ | 9.90 ms ✓ | — | 21.5 ms ✓ | 32.5 ms ✓ | — | 15.4 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 860.6 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## bve_select  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | — | **19.3 ms ✓** | — | — | 54.5 ms ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 511.1 ms ✗ | — | — | — | — | — | — | — |
| 500×50 | — | — | — | — | **75.5 ms ✓** | — | — | 254.5 ms ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 787.0 ms ✗ | — | — | — | — | — | — | — |


## cars_select  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | — | **1.85 ms ✓** | — | — | 9.50 ms ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 935.3 ms ✗ | — | — | — | — | — | — |
| 500×50 | — | — | — | — | **8.53 ms ✓** | — | — | 38.5 ms ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 968.9 ms ✗ | — | — | — | — | — | — |


## continuum_regression  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.10 ms ✓ | 1.00 ms ✓ | 1.09 ms ✓ | **0.92 ms ✓** | 1.24 ms ✓ | 1.60 ms ✓ | — | 4.00 ms ✓ | 7.50 ms ✓ | — | 4.50 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | 742.7 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×50 | 4.54 ms ✓ | 4.94 ms ✓ | 4.94 ms ✓ | **4.50 ms ✓** | 5.05 ms ✓ | 4.84 ms ✓ | — | 19.5 ms ✓ | 29.0 ms ✓ | — | 10.5 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | 818.5 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## cppls  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | **0.88 ms ✓** | 0.93 ms ✓ | 0.91 ms ✓ | 0.93 ms ✓ | 1.31 ms ✓ | 0.91 ms ✓ | 1.13 ms ✓ | 3.00 ms ✓ | 9.00 ms ✓ | 1.68 ms ✓ | 5.18 ms ✓ | — | — | 11.5 ms ✗ | — | — | — | — | — | 204.4 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×50 | 6.32 ms ✓ | **4.43 ms ✓** | 7.71 ms ✓ | 4.97 ms ✓ | 4.62 ms ✓ | 4.49 ms ✓ | 5.55 ms ✓ | 16.5 ms ✓ | 28.5 ms ✓ | 7.16 ms ✓ | 11.9 ms ✓ | — | — | 29.5 ms ✗ | — | — | — | — | — | 203.7 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## di_pls  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | — | **1.21 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 3.36 ms ✗ | — | — |
| 500×50 | — | — | — | — | **5.01 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 8.67 ms ✗ | — | — |


## ds  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.41 ms ✓ | **1.39 ms ✓** | 2.71 ms ✓ | 1.89 ms ✓ | 1.60 ms ✓ | 1.69 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 174.9 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×50 | 6.74 ms ✓ | 7.06 ms ✓ | 6.94 ms ✓ | **6.16 ms ✓** | 6.83 ms ✓ | 6.60 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 260.8 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — |


## ecr  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.18 ms ✓ | 1.25 ms ✓ | 1.69 ms ✓ | 1.31 ms ✓ | 1.42 ms ✓ | **1.12 ms ✓** | 1.77 ms ✓ | 3.50 ms ✓ | 7.00 ms ✓ | 1.95 ms ✓ | 5.67 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 17.7 ms ⚠ | — | — | — | — | — |
| 500×50 | 5.48 ms ✓ | **5.41 ms ✓** | 6.80 ms ✓ | 5.90 ms ✓ | 6.01 ms ✓ | 6.75 ms ✓ | 7.01 ms ✓ | 18.0 ms ✓ | 26.0 ms ✓ | 8.71 ms ✓ | 11.8 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 21.5 ms ⚠ | — | — | — | — | — |


## emcuve_select  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | — | **1.92 ms ✓** | — | — | 5.00 ms ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 898.4 ms ✗ | — | — | — | — | — | — | — |
| 500×50 | — | — | — | — | **8.15 ms ✓** | — | — | 20.5 ms ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 7.7 s ✗ | — | — | — | — | — | — | — |


## fused_sparse_pls  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.75 ms ✓ | 1.01 ms ✓ | 0.95 ms ✓ | 0.99 ms ✓ | 1.18 ms ✗ | **0.89 ms ✓** | — | 3.00 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×50 | 4.56 ms ✓ | 5.91 ms ✓ | **4.23 ms ✓** | 4.53 ms ✓ | 5.00 ms ✗ | 4.28 ms ✓ | — | 20.0 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## ga_select  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | — | **3.54 ms ✓** | — | — | 114.5 ms ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 716.8 ms ✗ | — | — | — | — | — | — | — |
| 500×50 | — | — | — | — | **13.3 ms ✓** | — | — | 658.0 ms ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 875.0 ms ✗ | — | — | — | — | — | — | — |


## gpr_pls  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | — | **1.44 ms ✓** | — | — | 3.50 ms ✗ | — | — | — | 10.8 ms ✗ | — | — | — | — | — | 2.78 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×50 | — | — | — | — | **40.8 ms ✓** | — | — | 57.0 ms ✗ | — | — | — | 593.2 ms ✗ | — | — | — | — | — | 53.2 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## group_sparse_pls  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 0.99 ms ✓ | 1.02 ms ✓ | 1.69 ms ✓ | 0.92 ms ✓ | 1.06 ms ✓ | **0.89 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.5 s ✗ | — | — | — | — | — | — | — | — | — | — | — |
| 500×50 | 5.68 ms ✓ | 4.66 ms ✓ | 4.92 ms ✓ | 4.68 ms ✓ | **4.55 ms ✓** | 4.91 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.5 s ✗ | — | — | — | — | — | — | — | — | — | — | — |


## interval_select  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | — | **2.07 ms ✓** | — | — | 7.00 ms ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 906.4 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×50 | — | — | — | — | **7.27 ms ✓** | — | — | 26.5 ms ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.1 s ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## ipw_select  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | — | **1.85 ms ✓** | — | — | 4.00 ms ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 455.4 ms ✗ | — | — | — | — | — | — | — |
| 500×50 | — | — | — | — | **8.14 ms ✓** | — | — | 21.0 ms ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 517.5 ms ✗ | — | — | — | — | — | — | — |


## irf_select  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | — | **2.84 ms ✓** | — | — | 7.50 ms ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.0 s ✗ | — | — | — | — | — |
| 500×50 | — | — | — | — | **16.6 ms ✓** | — | — | 38.5 ms ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.0 s ✗ | — | — | — | — | — |


## iriv_select  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | — | **264.1 ms ✓** | — | — | 385.5 ms ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 11.2 s ✗ | — | — | — | — | — |
| 500×50 | — | — | — | — | **1.2 s ✓** | — | — | 1.9 s ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 13.3 s ✗ | — | — | — | — | — |


## kernel_pls_rbf  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | — | **1.66 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 488.9 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×50 | — | — | — | — | **23.4 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 583.8 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## lw_pls  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 2.88 ms ✓ | **2.29 ms ✓** | 2.85 ms ✓ | 2.40 ms ✓ | 3.04 ms ✓ | 2.38 ms ✓ | — | 5.00 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 11.8 ms ✗ | — | — | — | — | — | — | — | — | — |
| 500×50 | 20.3 ms ✓ | 20.5 ms ✓ | 21.3 ms ✓ | 22.5 ms ✓ | 21.5 ms ✓ | **20.0 ms ✓** | — | 36.5 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 147.3 ms ✗ | — | — | — | — | — | — | — | — | — |


## mb_pls  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | — | **1.10 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 2.07 ms ✗ | — | — | — | — | — | — | — | — | — | — |
| 500×50 | — | — | — | — | **5.12 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 7.68 ms ✗ | — | — | — | — | — | — | — | — | — | — |


## mir_pls  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 0.94 ms ✓ | 0.91 ms ✓ | 0.90 ms ✓ | **0.90 ms ✓** | 1.29 ms ✓ | 1.02 ms ✓ | 1.26 ms ✓ | 3.00 ms ✓ | 8.00 ms ✓ | 1.71 ms ✓ | 5.48 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×50 | 6.26 ms ✓ | 4.98 ms ✓ | 7.44 ms ✓ | 5.56 ms ✓ | 4.62 ms ✓ | 5.43 ms ✓ | **4.32 ms ✓** | 16.5 ms ✓ | 24.5 ms ✓ | 7.19 ms ✓ | 10.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## missing_aware_nipals  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.05 ms ✓ | 1.14 ms ✓ | 1.03 ms ✓ | **1.02 ms ✓** | 1.06 ms ✓ | 1.76 ms ✓ | 1.08 ms ✓ | 3.00 ms ✓ | 8.50 ms ✓ | 1.62 ms ✓ | 4.78 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 749.7 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×50 | **4.27 ms ✓** | 4.58 ms ✓ | 6.26 ms ✓ | 4.36 ms ✓ | 4.56 ms ✓ | 4.50 ms ✓ | 5.89 ms ✓ | 14.5 ms ✓ | 23.0 ms ✓ | 7.54 ms ✓ | 10.3 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 796.8 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — |


## n_pls  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | — | **1.36 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 16.9 ms ✗ | — | — | — |
| 500×50 | — | — | — | — | **4.80 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 39.0 ms ✗ | — | — | — |


## o2pls  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.31 ms ✓ | 0.98 ms ✓ | 1.56 ms ✓ | 1.07 ms ✓ | 1.60 ms ⚠ | **0.92 ms ✓** | — | 3.50 ms ✓ | 10.0 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.2 s ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×50 | 6.12 ms ✓ | 5.05 ms ✓ | 7.30 ms ✓ | 4.99 ms ✓ | 6.02 ms ⚠ | **4.48 ms ✓** | — | 17.0 ms ✓ | 22.5 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.3 s ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## on_pls  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | — | **1.37 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 10.9 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×50 | — | — | — | — | **5.41 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 24.5 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## one_se_rule  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | — | **1.11 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 193.8 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×50 | — | — | — | — | **4.85 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 201.0 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## opls  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.41 ms ✓ | 1.91 ms ✓ | 1.84 ms ✓ | 1.08 ms ✓ | 1.21 ms ✓ | **0.91 ms ✓** | 1.27 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 4.1 s ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×50 | 5.16 ms ✓ | 4.72 ms ✓ | 4.76 ms ✓ | 4.53 ms ✓ | 4.72 ms ✓ | **4.51 ms ✓** | 4.94 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 4.5 s ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pcr  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 28.1 ms ✓ | 28.0 ms ✓ | 28.2 ms ✓ | 29.4 ms ✓ | 30.6 ms ✓ | 28.1 ms ✓ | 30.1 ms ✓ | 30.0 ms ✓ | 37.5 ms ✗ | — | — | **2.20 ms ✓** | — | 7.50 ms ✓ | — | — | — | 3.28 ms ✓ | — | 175.4 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — ⏳ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×50 | 32.8 ms ✓ | 34.5 ms ✓ | 33.2 ms ✓ | 32.9 ms ✓ | 34.5 ms ✓ | 32.8 ms ✓ | 34.0 ms ✓ | 47.0 ms ✓ | 59.0 ms ✗ | — | — | 6.91 ms ✓ | — | 26.0 ms ✓ | — | — | — | **5.92 ms ✓** | — | 209.5 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pds  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.51 ms ✓ | 1.20 ms ✓ | 1.14 ms ✓ | 1.14 ms ✓ | 1.16 ms ✓ | **1.05 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 174.3 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×50 | **5.00 ms ✓** | 5.14 ms ✓ | 5.32 ms ✓ | 5.60 ms ✓ | 5.15 ms ✓ | 5.94 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 250.7 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | **0.91 ms ✓** | 1.63 ms ✓ | 1.14 ms ✓ | 0.92 ms ✓ | 1.97 ms ✓ | 0.96 ms ✓ | 1.17 ms ✓ | 3.50 ms ✓ | 7.00 ms ✗ | 1.65 ms ✓ | 6.08 ms ✗ | 1.92 ms ✓ | 2.61 ms ✗ | 8.50 ms ✓ | — | 12.5 ms ✓ | 4.10 ms ✓ | 2.12 ms ✓ | 1.48 ms ✗ | 166.5 ms ✓ | 1.4 s ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | 8.73 ms ⚠ | 10.3 ms ⚠ | 10.7 ms ⚠ | 13.0 ms ⚠ | 14.2 ms ⚠ | 9.97 ms ⚠ | 11.7 ms ⚠ | 64.5 ms ⚠ | 101.0 ms ⚠ | 14.3 ms ⚠ | 20.5 ms ⚠ | 13.9 ms ⚠ | 11.1 ms ⚠ | 116.0 ms ⚠ | — | 75.0 ms ⚠ | 21.0 ms ⚠ | 13.4 ms ⚠ | 11.1 ms ⚠ | 363.6 ms ⚠ | 1.8 s ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×50 | **4.20 ms ✓** | 4.48 ms ✓ | 4.83 ms ✓ | 5.50 ms ✓ | 5.05 ms ✓ | 4.25 ms ✓ | 4.58 ms ✓ | 22.5 ms ✓ | 29.5 ms ✗ | 7.28 ms ✓ | 13.0 ms ✗ | 6.80 ms ✓ | 7.75 ms ✗ | 23.0 ms ✓ | — | 35.5 ms ✓ | 10.8 ms ✓ | 6.63 ms ✓ | 6.30 ms ✗ | 206.1 ms ✓ | 1.4 s ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×500 | 56.3 ms ⚠ | 44.6 ms ⚠ | 43.7 ms ⚠ | 44.9 ms ⚠ | 66.1 ms ⚠ | 50.0 ms ⚠ | 47.2 ms ⚠ | 372.0 ms ⚠ | 477.5 ms ⚠ | 85.9 ms ⚠ | 95.3 ms ⚠ | 55.5 ms ⚠ | 52.1 ms ⚠ | 441.0 ms ⚠ | — | 466.5 ms ⚠ | 96.5 ms ⚠ | 53.1 ms ⚠ | 57.6 ms ⚠ | 1.2 s ⚠ | 3.1 s ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_cox  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.00 ms ✓ | 1.02 ms ✓ | 1.10 ms ✓ | **0.99 ms ✓** | 1.07 ms ✗ | 1.47 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 2.6 s ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×50 | 4.95 ms ✓ | 5.14 ms ✓ | 4.83 ms ✓ | 5.34 ms ✓ | 4.73 ms ✗ | **4.62 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 2.6 s ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_diagnostic_dmodx  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | — | **1.58 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 522.3 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×50 | — | — | — | — | **6.17 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 557.5 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_diagnostic_q  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | — | **1.14 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 487.5 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×50 | — | — | — | — | **5.39 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 536.9 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_diagnostic_t2  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | — | **1.16 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 532.3 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×50 | — | — | — | — | **5.35 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 538.4 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_glm  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.04 ms ✓ | 1.08 ms ✓ | 1.11 ms ✓ | 0.97 ms ✓ | 1.26 ms ⚠ | **0.88 ms ✓** | — | 3.50 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.5 s ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×50 | 5.09 ms ✓ | 4.47 ms ✓ | **4.44 ms ✓** | 4.54 ms ✓ | 4.62 ms ⚠ | 4.50 ms ✓ | — | 16.0 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.7 s ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_lda  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | — | **1.07 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | 2.46 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×50 | — | — | — | — | **4.41 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | 6.20 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_logistic  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | — | **1.19 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | 2.47 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×50 | — | — | — | — | **4.87 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | 7.58 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_monitoring  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | — | **1.18 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 483.1 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×50 | — | — | — | — | **5.48 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 544.0 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_qda  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 0.95 ms ✓ | 1.62 ms ✓ | 1.73 ms ✓ | 1.22 ms ✓ | 1.72 ms ⚠ | **0.91 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | 2.58 ms ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×50 | 4.41 ms ✓ | 4.72 ms ✓ | 5.11 ms ✓ | 4.50 ms ✓ | 7.41 ms ⚠ | **4.38 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | 7.53 ms ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pso_select  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | — | **6.41 ms ✓** | — | — | 102.5 ms ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 191.7 ms ✗ |
| 500×50 | — | — | — | — | **27.5 ms ✓** | — | — | 481.5 ms ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 230.5 ms ✗ |


## random_frog_select  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | — | **1.73 ms ✓** | — | — | 12.0 ms ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 98.6 ms ✗ | — | — | — | — | — |
| 500×50 | — | — | — | — | **8.15 ms ✓** | — | — | 59.0 ms ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 82.7 ms ✗ | — | — | — | — | — |


## random_subspace_pls  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.61 ms ✓ | 2.22 ms ✓ | 2.48 ms ✓ | **1.15 ms ✓** | 1.76 ms ✗ | 1.21 ms ✓ | — | 4.00 ms ✓ | — | — | — | — | — | — | — | — | — | 13.1 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×50 | 5.94 ms ✓ | 6.10 ms ✓ | 6.49 ms ✓ | **5.37 ms ✓** | 8.01 ms ✗ | 5.83 ms ✓ | — | 20.0 ms ✓ | — | — | — | — | — | — | — | — | — | 16.1 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## randomization_select  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | — | **3.10 ms ✓** | — | — | 6.00 ms ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 248.9 ms ✓ | — | — | — | — |
| 500×50 | — | — | — | — | **15.0 ms ✓** | — | — | 29.0 ms ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 322.1 ms ✓ | — | — | — | — |


## recursive_pls  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | — | **3.92 ms ✓** | — | — | 4.00 ms ✗ | — | — | — | — | — | — | — | — | — | 24.6 ms ✗ | — | 251.7 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×50 | — | — | — | — | **13.8 ms ✓** | — | — | 24.0 ms ✗ | — | — | — | — | — | — | — | — | — | 252.2 ms ✗ | — | 812.2 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## rep_select  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | — | **2.07 ms ✓** | — | — | 7.00 ms ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 723.2 ms ✗ | — | — | — | — | — | — | — |
| 500×50 | — | — | — | — | **8.36 ms ✓** | — | — | 24.0 ms ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.3 s ✗ | — | — | — | — | — | — | — |


## ridge_pls  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | **0.89 ms ✓** | 1.08 ms ✓ | 0.97 ms ✓ | 0.89 ms ✓ | 1.15 ms ✗ | 1.01 ms ✓ | 1.06 ms ✓ | 4.00 ms ✓ | 6.50 ms ✓ | 1.96 ms ✓ | 4.17 ms ✓ | 2.95 ms ✗ | — | — | — | — | — | 2.19 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×50 | 4.35 ms ✓ | 4.39 ms ✓ | 5.22 ms ✓ | 4.30 ms ✓ | 4.78 ms ✗ | **4.29 ms ✓** | 4.53 ms ✓ | 19.5 ms ✓ | 23.5 ms ✓ | 7.20 ms ✓ | 9.77 ms ✓ | 5.90 ms ✗ | — | — | — | — | — | 5.80 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## robust_pls  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 3.08 ms ✓ | 2.34 ms ✓ | 2.17 ms ✓ | **1.42 ms ✓** | 1.25 ms ✗ | 1.44 ms ✓ | 1.71 ms ✓ | 3.00 ms ✓ | 6.50 ms ✓ | 2.35 ms ✓ | 5.18 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | 225.3 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×50 | **6.79 ms ✓** | 6.87 ms ✓ | 6.96 ms ✓ | 6.94 ms ✓ | 5.14 ms ⚠ | 7.04 ms ✓ | 7.34 ms ✓ | 22.5 ms ✓ | 23.0 ms ✓ | 10.5 ms ✓ | 13.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | 264.0 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## rosa  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | — | **1.17 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.2 s ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×50 | — | — | — | — | **5.84 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.2 s ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## scars_select  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | — | **1.93 ms ✓** | — | — | 7.50 ms ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×50 | — | — | — | — | **7.32 ms ✓** | — | — | 37.5 ms ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## shaving_select  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | — | **2.28 ms ✓** | — | — | 5.50 ms ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 507.7 ms ✗ | — | — | — | — | — | — | — |
| 500×50 | — | — | — | — | **6.85 ms ✓** | — | — | 21.5 ms ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 526.1 ms ✗ | — | — | — | — | — | — | — |


## sipls_select  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | — | **2.77 ms ✓** | — | — | 6.50 ms ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×50 | — | — | — | — | **10.6 ms ✓** | — | — | 17.0 ms ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## so_pls  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | — | **1.42 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.2 s ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×50 | — | — | — | — | **6.20 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.5 s ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## spa_select  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | — | **1.37 ms ✓** | — | — | 4.00 ms ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 518.2 ms ✗ | — | — | — | — | — | — | — |
| 500×50 | — | — | — | — | **6.98 ms ✓** | — | — | 17.0 ms ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 851.4 ms ✗ | — | — | — | — | — | — | — |


## sparse_pls_da  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.13 ms ✓ | 0.91 ms ✓ | 1.07 ms ✓ | **0.88 ms ✓** | 1.24 ms ⚠ | 1.70 ms ✓ | — | — | — | — | — | — | — | — | — | 49.5 ms ⚠ | — | — | — | — | — | — | 202.7 ms ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×50 | 4.48 ms ✓ | 4.58 ms ✓ | 4.49 ms ✓ | 4.75 ms ✓ | 4.46 ms ⚠ | **4.39 ms ✓** | — | — | — | — | — | — | — | — | — | 247.5 ms ⚠ | — | — | — | — | — | — | 225.3 ms ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## sparse_simpls  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 0.88 ms ✓ | 1.07 ms ✓ | 1.14 ms ✓ | 0.94 ms ✓ | 1.09 ms ✗ | **0.86 ms ✓** | 1.14 ms ✓ | 4.00 ms ✓ | 6.00 ms ✓ | 1.76 ms ✓ | 4.86 ms ✓ | — | — | — | — | 10.0 ms ✗ | — | — | — | — | — | — | 172.5 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×50 | 4.28 ms ✓ | 5.71 ms ✓ | 4.40 ms ✓ | **4.25 ms ✓** | 5.35 ms ✗ | 4.48 ms ✓ | 4.45 ms ✓ | 14.5 ms ✓ | 26.0 ms ✓ | 7.31 ms ✓ | 10.5 ms ✓ | — | — | — | — | 36.5 ms ✗ | — | — | — | — | — | — | 198.4 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## st_select  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | — | **1.52 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 484.0 ms ✗ | — | — | — | — | — | — | — |
| 500×50 | — | — | — | — | **6.36 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 558.0 ms ✗ | — | — | — | — | — | — | — |


## stability_select  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | — | **1.38 ms ✓** | — | — | 3.50 ms ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 589.8 ms ✗ | — | — | — | — | — | — | — |
| 500×50 | — | — | — | — | **6.10 ms ✓** | — | — | 17.5 ms ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 2.0 s ✗ | — | — | — | — | — | — | — |


## t2_select  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **480.5 ms ✓** | — | — | — | — | — | — | — |
| 500×50 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **568.0 ms ✓** | — | — | — | — | — | — | — |


## uve_select  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | — | **1.53 ms ✓** | — | — | 4.00 ms ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 571.6 ms ✗ | — | — | — | — | — | — | — |
| 500×50 | — | — | — | — | **8.64 ms ✓** | — | — | 18.0 ms ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 2.0 s ✗ | — | — | — | — | — | — | — |


## variable_select_coef  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | — | **1.99 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 402.5 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×50 | — | — | — | — | **5.17 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 438.9 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## variable_select_sr  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | — | **1.31 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 409.9 ms ✗ | — | — | — | — | — | — | — |
| 500×50 | — | — | — | — | **4.60 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 427.1 ms ✗ | — | — | — | — | — | — | — |


## variable_select_vip  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | — | **1.26 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 374.8 ms ✗ | — | — | — | — | — | — | — |
| 500×50 | — | — | — | — | **8.12 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 432.3 ms ✓ | — | — | — | — | — | — | — |


## vip_spa_select  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | — | **1.10 ms ✓** | — | — | 3.50 ms ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 156.9 ms ✗ | — |
| 500×50 | — | — | — | — | **4.70 ms ✓** | — | — | 19.0 ms ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 97.0 ms ✗ | — |


## vissa_select  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | — | **25.7 ms ✓** | — | — | 106.0 ms ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.8 s ✗ | — |
| 500×50 | — | — | — | — | **124.9 ms ✓** | — | — | 531.5 ms ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 2.1 s ✗ | — |


## weighted_pls  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.47 ms ✓ | 0.99 ms ✓ | 1.01 ms ✓ | **0.91 ms ✓** | 1.04 ms ✗ | 1.66 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | 1.75 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 500×50 | 4.97 ms ✓ | 4.74 ms ✓ | **4.23 ms ✓** | 4.47 ms ✓ | 4.43 ms ✗ | 4.39 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | 5.98 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## wvc_select  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | — | **1.60 ms ✓** | — | — | 3.00 ms ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 459.4 ms ✗ | — | — | — | — | — | — | — |
| 500×50 | — | — | — | — | **4.88 ms ✓** | — | — | 19.0 ms ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 463.7 ms ✓ | — | — | — | — | — | — | — |


## wvc_threshold_select  —  1 thread

| samples × features | pls4all.cpp.ref | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.python_ikpls | ref.r_pls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.r_chemometrics | ref.r_jico | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_tensorly | ref.python_diplslib | ref.python_auswahl | ref.python_pyswarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | — | — | — | — | **1.43 ms ✓** | — | — | 3.00 ms ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 420.8 ms ✗ | — | — | — | — | — | — | — |
| 500×50 | — | — | — | — | **6.40 ms ✓** | — | — | 16.0 ms ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 473.5 ms ✗ | — | — | — | — | — | — | — |


## Backend definitions

Each column in the per-algorithm tables above is one of the entries below. Columns are named `owner.language[.variant]`: `pls4all.*` are this project's own bindings; everything else is an external library shipped by its own maintainers.

| Column | Language | Tier | What it actually runs |
|---|---|---|---|
| `pls4all.cpp.ref` | C++ | pls4all reference (single-thread) | libp4a built with `PLS4ALL_WITH_BLAS=OFF, OPENMP=OFF` — pure scalar reference loops, no acceleration. The parity baseline. |
| `pls4all.cpp.blas` | C++ | pls4all + BLAS | libp4a built with `PLS4ALL_WITH_BLAS=ON` only — links system BLAS (OpenBLAS in this env), benefits from BLAS thread parallelism. |
| `pls4all.cpp.omp` | C++ | pls4all + OpenMP | libp4a built with `PLS4ALL_WITH_OPENMP=ON` only — OpenMP parallelism in the C kernel loops, no BLAS. |
| `pls4all.cpp.blas+omp` | C++ | pls4all + BLAS + OpenMP | libp4a built with both `PLS4ALL_WITH_BLAS=ON` and `PLS4ALL_WITH_OPENMP=ON` — the recommended production config. |
| `pls4all.registry` | Python | pls4all canonical | `benchmarks.parity_timing.registry.MethodSpec.pls4all_fn` — the canonical per-method pls4all entry point |
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
| `ref.python_scikit_learn` | external | reference | registry-declared external reference library |
| `ref.python_ikpls` | external | reference | registry-declared external reference library |
| `ref.r_pls` | external | reference | registry-declared external reference library |
| `ref.r_mixomics` | external | reference | registry-declared external reference library |
| `ref.r_ropls` | external | reference | registry-declared external reference library |
| `ref.r_spls` | external | reference | registry-declared external reference library |
| `ref.r_chemometrics` | external | reference | registry-declared external reference library |
| `ref.r_jico` | external | reference | registry-declared external reference library |
| `ref.r_kernlab_pls` | external | reference | registry-declared external reference library |
| `ref.r_omicspls` | external | reference | registry-declared external reference library |
| `ref.r_mdatools` | external | reference | registry-declared external reference library |
| `ref.r_multiblock` | external | reference | registry-declared external reference library |
| `ref.python_onpls` | external | reference | registry-declared external reference library |
| `ref.r_mboost` | external | reference | registry-declared external reference library |
| `ref.r_plsrglm` | external | reference | registry-declared external reference library |
| `ref.r_plsrcox` | external | reference | registry-declared external reference library |
| `ref.r_base` | external | reference | registry-declared external reference library |
| `ref.r_softimpute` | external | reference | registry-declared external reference library |
| `ref.r_sgpls` | external | reference | registry-declared external reference library |
| `ref.python_nirs4all_operators_models_sklearn_mbpls` | external | reference | registry-declared external reference library |
| `ref.python_nirs4all_operators_models_sklearn_lwpls` | external | reference | registry-declared external reference library |
| `ref.python_nirs4all_bench_aom_v0_aompls` | external | reference | registry-declared external reference library |
| `ref.r_plsvarsel` | external | reference | registry-declared external reference library |
| `ref.r_enpls` | external | reference | registry-declared external reference library |
| `ref.matlab_libpls` | external | reference | registry-declared external reference library |
| `ref.r_pls_stats` | external | reference | registry-declared external reference library |
| `ref.python_tensorly` | external | reference | registry-declared external reference library |
| `ref.python_diplslib` | external | reference | registry-declared external reference library |
| `ref.python_auswahl` | external | reference | registry-declared external reference library |
| `ref.python_pyswarms` | external | reference | registry-declared external reference library |


## Versions per backend

| Column | Versions |
|---|---|
| `pls4all.cpp.ref` | `language=C++ (via ctypes bridge) (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `libp4a=0.97.0+abi.1.16.0`; `numpy=2.3.5` |
| `pls4all.cpp.blas` | `language=C++ (via ctypes bridge) (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1; libopenblas 0.3.30 blas=1`; `libp4a=0.97.0+abi.1.16.0`; `numpy=2.3.5` |
| `pls4all.cpp.omp` | `language=C++ (via ctypes bridge) (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1; libomp ? openmp=1`; `libp4a=0.97.0+abi.1.16.0`; `numpy=2.3.5` |
| `pls4all.cpp.blas+omp` | `language=C++ (via ctypes bridge) (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1; libopenblas 0.3.30 blas=1; libgomp ? openmp=1`; `libp4a=0.97.0+abi.1.16.0`; `numpy=2.3.5` |
| `pls4all.registry` | `language=Python 3.13.11`; `blas=libscipy_openblas 0.3.30 blas=1`; `pls4all=0.97.0`; `numpy=2.3.5`; `registry_method=pls` |
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
| `ref.python_scikit_learn` | `language=Python 3.13.11`; `blas=libscipy_openblas 0.3.30 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `numpy=2.3.5`; `reference_id=python_scikit_learn`; `reference_library=scikit-learn`; `reference_version=1.8.0` |
| `ref.python_ikpls` | `language=Python 3.13.11`; `blas=libscipy_openblas 0.3.30 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `numpy=2.3.5`; `reference_id=python_ikpls`; `reference_library=ikpls`; `reference_version=4.0.1.post1` |
| `ref.r_pls` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_pls`; `reference_library=pls`; `reference_version=2.8.5` |
| `ref.r_mixomics` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_mixomics`; `reference_library=mixOmics`; `reference_version=6.26.0` |
| `ref.r_ropls` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_ropls`; `reference_library=ropls`; `reference_version=Bioc` |
| `ref.r_spls` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_spls`; `reference_library=spls`; `reference_version=2.3.2` |
| `ref.r_chemometrics` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_chemometrics`; `reference_library=chemometrics`; `reference_version=0.7.x` |
| `ref.r_jico` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_jico`; `reference_library=JICO`; `reference_version=0.0` |
| `ref.r_kernlab_pls` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_kernlab_pls`; `reference_library=kernlab+pls`; `reference_version=0.9.33+2.8.5` |
| `ref.r_omicspls` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_omicspls`; `reference_library=OmicsPLS`; `reference_version=2.1.0` |
| `ref.r_mdatools` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_mdatools`; `reference_library=mdatools`; `reference_version=0.15.0` |
| `ref.r_multiblock` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_multiblock`; `reference_library=multiblock`; `reference_version=0.8.10` |
| `ref.python_onpls` | `language=Python 3.13.11`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=python_onpls`; `reference_library=OnPLS`; `reference_version=github tomlof/OnPLS` |
| `ref.r_mboost` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_mboost`; `reference_library=mboost`; `reference_version=2.9-11` |
| `ref.r_plsrglm` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_plsrglm`; `reference_library=plsRglm`; `reference_version=1.5.1` |
| `ref.r_plsrcox` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_plsrcox`; `reference_library=plsRcox`; `reference_version=1.8.2` |
| `ref.r_base` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_base`; `reference_library=base`; `reference_version=R 4.3.3` |
| `ref.r_softimpute` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_softimpute`; `reference_library=softImpute`; `reference_version=1.4-1` |
| `ref.r_sgpls` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_sgpls`; `reference_library=sgPLS`; `reference_version=1.8.1` |
| `ref.python_nirs4all_operators_models_sklearn_mbpls` | `language=Python 3.13.11`; `blas=libscipy_openblas 0.3.30 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `numpy=2.3.5`; `reference_id=python_nirs4all_operators_models_sklearn_mbpls`; `reference_library=nirs4all.operators.models.sklearn.mbpls`; `reference_version=in-tree` |
| `ref.python_nirs4all_operators_models_sklearn_lwpls` | `language=Python 3.13.11`; `blas=libscipy_openblas 0.3.30 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `numpy=2.3.5`; `reference_id=python_nirs4all_operators_models_sklearn_lwpls`; `reference_library=nirs4all.operators.models.sklearn.lwpls`; `reference_version=in-tree` |
| `ref.python_nirs4all_bench_aom_v0_aompls` | — |
| `ref.r_plsvarsel` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_plsvarsel`; `reference_library=plsVarSel`; `reference_version=0.10.0` |
| `ref.r_enpls` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_enpls`; `reference_library=enpls`; `reference_version=6.1` |
| `ref.matlab_libpls` | `language=matlab (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=matlab_libpls`; `reference_library=libPLS`; `reference_version=1.95` |
| `ref.r_pls_stats` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_pls_stats`; `reference_library=pls+stats`; `reference_version=R 4.3.3` |
| `ref.python_tensorly` | `language=Python 3.13.11`; `blas=libscipy_openblas 0.3.30 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `numpy=2.3.5`; `reference_id=python_tensorly`; `reference_library=tensorly`; `reference_version=0.9.0` |
| `ref.python_diplslib` | `language=Python 3.13.11`; `blas=libscipy_openblas 0.3.30 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `numpy=2.3.5`; `reference_id=python_diplslib`; `reference_library=diPLSlib`; `reference_version=2.5.0` |
| `ref.python_auswahl` | `language=Python 3.13.11`; `blas=libscipy_openblas 0.3.30 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `numpy=2.3.5`; `reference_id=python_auswahl`; `reference_library=auswahl`; `reference_version=0.9.0` |
| `ref.python_pyswarms` | `language=Python 3.13.11`; `blas=libscipy_openblas 0.3.30 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `numpy=2.3.5`; `reference_id=python_pyswarms`; `reference_library=pyswarms`; `reference_version=1.3.0` |

## Methodology

- Reference: `cpp` cell at 1 thread (libp4a via ctypes), or `python_tier1` when `cpp` is unavailable for an algorithm
- Parity tolerance: 1e-6 (per-algo overrides possible)
- All backends read the **same** orchestrator-generated CSV (`benchmarks/cross_binding/data/data_<n>x<p>_seed<seed>.csv`) so input data is bit-identical across languages
- 4, 5 run(s) per cell, first discarded as warmup when `n_runs >= 3`, median reported
- Per-cell timeout: 300 s
- Thread control via `OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = BLIS_NUM_THREADS` set in the subprocess env, plus `Context.num_threads` for Python pls4all and `maxNumCompThreads()` for Octave
- pls4all libp4a build: `build/blas-omp/cpp/src/libp4a.so` (BLAS + OpenMP enabled)
