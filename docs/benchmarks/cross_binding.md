# Cross-binding benchmark — parity + timing (1 thread)

Single-thread numbers — the cleanest cross-language comparison. **Most external libraries don't honour OMP_NUM_THREADS at the algorithm level** (sklearn, pls::plsr, plsregress, ikpls all run the PLS algo serially even when BLAS is multi-threaded), so a multi-thread comparison would compare pls4all's OpenMP path against everyone else's single-threaded loop. That's a real, useful number — see the [thread sweep page](cross_binding_threads.md) — but this main page sticks to 1 thread for honest apples-to-apples timing.

_Generated: 2026-05-17 22:22:14 UTC_
_CSV: `tmp067uz5rc.csv` (142 cells)_


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

Timing is the **median of 1 run(s)**; the first run is discarded as warmup when `n_runs >= 3`. All backends in a single cell read the SAME orchestrator-generated CSV so cross-language input bytes are bit-identical. See [methodology.md](methodology.md) for the full details.


## Why a cell is empty (`—`)

An empty cell means the backend **did not produce a timing for that (algorithm, size) combination**. Four distinct reasons, all reported as `—` in the matrix:

1. **External library doesn't implement the algorithm.** Example: `sklearn` has no native CPPLS or sparse SIMPLS; `plsregress` (Octave) only does plain PLS; `ikpls` is plain PLS only; `ropls` only does OPLS; `mixOmics` covers PLS / sparse PLS / PLS-DA. Filling this column requires the external maintainer to add the algorithm — out of our control.
2. **pls4all wrapper missing for that algorithm.** Example: `pls4all.sklearn` (tier 2) doesn't yet ship a class for `continuum_regression` or `kernel_pls`; `pls4all.R.formula` doesn't have a formula wrapper for every algorithm yet. This is the *chemin à parcourir* on our side — visible TODO.
3. **Bench script missing the dispatch case.** A few cells can fail because a binding bench script doesn't yet route a specific algorithm to its underlying call. Pure tooling gap, no library impact.
4. **Run too slow / OOM / crashed.** The cell timed out (`⏳`) or hit a runtime error. Rare in this matrix (300 s timeout is generous for the included sizes).

Each `—` represents one of these four reasons. The CSV (`results/full_matrix.csv`) carries a `reason` column that tells you exactly which one for any given cell.


Column names: anything starting with `pls4all.` is one of this project's bindings; the others use their real package name (`sklearn`, `ikpls`, `pls`, `ropls`, `mixOmics`, `plsregress`). **Every algorithm table keeps every column** — `—` cells are intentional and show *where coverage is still missing*, either on our side (an algorithm not yet wrapped in a tier-2 class) or on the external side (e.g. `sklearn` doesn't implement CPPLS / OPLS, `plsregress` only does plain PLS, `ikpls` only does plain PLS). Full per-column description: see [Backend definitions](#backend-definitions) at the bottom of this page.


## aom_preprocess  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **210.3 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## approximate_press  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **213.2 ms ✓** | — | 191.9 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## bagging_pls  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **170.1 ms ✓** | 611.4 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## bipls_select  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **172.8 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | 702.9 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## boosting_pls  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **177.6 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 636.4 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — |


## bve_select  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **205.3 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 377.1 ms ✗ | — | — | — |


## cars_select  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **165.9 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 662.9 ms ✗ | — | — |


## continuum_regression  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **167.8 ms ✓** | — | — | — | — | — | — | — | — | 578.5 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## cppls  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **171.8 ms ✓** | — | 143.1 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## di_pls  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **202.6 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## ds  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | 166.4 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **139.2 ms ✓** | — | — | — | — | — | — | — | — | — |


## ecr  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **176.8 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## emcuve_select  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **189.6 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.2 s ✗ | — | — | — |


## fused_sparse_pls  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **167.2 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## ga_select  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **169.0 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 508.3 ms ✗ | — | — | — |


## gpr_pls  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 120×25 | **191.1 ms ✓** | 736.7 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## group_sparse_pls  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **185.9 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.1 s ✗ | — | — | — | — | — | — | — |


## interval_select  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **165.6 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | 606.3 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## ipw_select  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **191.1 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 332.1 ms ✗ | — | — | — |


## irf_select  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 120×30 | **173.3 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## iriv_select  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 80×25 | **201.1 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## kernel_pls_rbf  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **174.3 ms ✓** | — | — | — | — | — | — | — | — | — | — | 330.0 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## lw_pls  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **170.1 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 596.1 ms ✗ | — | — | — | — | — |


## mb_pls  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×60 | **199.7 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 576.4 ms ✗ | — | — | — | — | — | — |


## mir_pls  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **168.5 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## missing_aware_nipals  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **174.3 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 582.7 ms ✓ | — | — | — | — | — | — | — | — |


## n_pls  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×48 | **167.7 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## o2pls  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **170.2 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | 927.0 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## on_pls  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **177.4 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 24.4 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## one_se_rule  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **171.3 ms ✓** | — | 134.2 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## opls  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **165.4 ms ✓** | — | — | — | — | 3.1 s ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pcr  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | 200.3 ms ✓ | 552.7 ms ✓ | **129.1 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pds  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | 165.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **139.4 ms ✓** | — | — | — | — | — | — | — | — | — |


## pls  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | 200.3 ms ✓ | 546.2 ms ✓ | **130.6 ms ✓** | 853.6 ms ✗ | 1.1 s ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_cox  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **168.0 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 2.0 s ✗ | — | — | — | — | — | — | — | — | — | — |


## pls_diagnostic_dmodx  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **171.0 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | 348.7 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_diagnostic_q  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **193.8 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | 352.4 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_diagnostic_t2  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **165.1 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | 364.7 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_glm  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **167.2 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.1 s ✗ | — | — | — | — | — | — | — | — | — | — | — |


## pls_lda  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **185.5 ms ✓** | 565.0 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_logistic  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **168.0 ms ✓** | 560.8 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_monitoring  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **181.6 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | 369.4 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_qda  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **164.7 ms ✓** | 584.5 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pso_select  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 80×25 | **168.6 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## random_frog_select  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **177.8 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## random_subspace_pls  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **166.6 ms ✓** | 598.1 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## randomization_select  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **171.7 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 180.1 ms ✓ |


## recursive_pls  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **177.3 ms ✓** | 566.1 ms ✗ | 256.4 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## rep_select  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **190.3 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 572.9 ms ✗ | — | — | — |


## ridge_pls  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **167.2 ms ✓** | 529.6 ms ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## robust_pls  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **186.1 ms ✓** | — | — | — | — | — | — | — | 217.5 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## rosa  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **169.8 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | 903.4 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## scars_select  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **168.8 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## shaving_select  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **173.4 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 357.1 ms ✗ | — | — | — |


## sipls_select  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **169.8 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## so_pls  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **170.1 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.1 s ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## spa_select  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **170.1 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 416.3 ms ✗ | — | — | — |


## sparse_pls_da  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **168.0 ms ✓** | — | — | — | — | — | 147.0 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## sparse_simpls  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **165.1 ms ✓** | — | — | — | — | — | 135.3 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## st_select  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **201.6 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 351.1 ms ✗ | — | — | — |


## stability_select  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **174.3 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 500.4 ms ✗ | — | — | — |


## t2_select  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **171.0 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 353.7 ms ✗ | — | — | — |


## uve_select  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **169.3 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 505.3 ms ✗ | — | — | — |


## variable_select_coef  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **677.6 ms ✓** | — | 294.0 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## variable_select_sr  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **182.7 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 288.9 ms ✗ | — | — | — |


## variable_select_vip  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **164.9 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 290.3 ms ✗ | — | — | — |


## vip_spa_select  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 80×40 | **172.8 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## vissa_select  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 80×25 | **182.0 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## weighted_pls  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **174.2 ms ✓** | 528.9 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## wvc_select  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **183.4 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 298.6 ms ✓ | — | — | — |


## wvc_threshold_select  —  1 thread

| samples × features | pls4all.registry | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **169.2 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 295.3 ms ✗ | — | — | — |


## Backend definitions

Each column in the per-algorithm tables above is one of the entries below. Columns are named `owner.language[.variant]`: `pls4all.*` are this project's own bindings; everything else is an external library shipped by its own maintainers.

| Column | Language | Tier | What it actually runs |
|---|---|---|---|
| `pls4all.registry` | Python | pls4all canonical | `benchmarks.parity_timing.registry.MethodSpec.pls4all_fn` — the canonical per-method pls4all entry point |
| `ref.python_scikit_learn` | external | reference | registry-declared external reference library |
| `ref.r_pls` | external | reference | registry-declared external reference library |
| `ref.python_ikpls` | external | reference | registry-declared external reference library |
| `ref.r_mixomics` | external | reference | registry-declared external reference library |
| `ref.r_ropls` | external | reference | registry-declared external reference library |
| `ref.r_spls` | external | reference | registry-declared external reference library |
| `ref.python_diplslib` | external | reference | registry-declared external reference library |
| `ref.r_chemometrics` | external | reference | registry-declared external reference library |
| `ref.r_jico` | external | reference | registry-declared external reference library |
| `ref.python_tensorly` | external | reference | registry-declared external reference library |
| `ref.r_kernlab_pls` | external | reference | registry-declared external reference library |
| `ref.r_omicspls` | external | reference | registry-declared external reference library |
| `ref.r_mdatools` | external | reference | registry-declared external reference library |
| `ref.r_multiblock` | external | reference | registry-declared external reference library |
| `ref.python_onpls` | external | reference | registry-declared external reference library |
| `ref.python_auswahl` | external | reference | registry-declared external reference library |
| `ref.python_pyswarms` | external | reference | registry-declared external reference library |
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


## Versions per backend

| Column | Versions |
|---|---|
| `pls4all.registry` | `language=Python 3.13.11`; `blas=libscipy_openblas 0.3.30 blas=1`; `pls4all=0.97.0`; `numpy=2.3.5`; `registry_method=pls` |
| `ref.python_scikit_learn` | `language=Python 3.13.11`; `blas=libscipy_openblas 0.3.30 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `numpy=2.3.5`; `reference_id=python_scikit_learn`; `reference_library=scikit-learn`; `reference_version=1.8.0` |
| `ref.r_pls` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_pls`; `reference_library=pls`; `reference_version=2.8.5` |
| `ref.python_ikpls` | `language=Python 3.13.11`; `blas=libscipy_openblas 0.3.30 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `numpy=2.3.5`; `reference_id=python_ikpls`; `reference_library=ikpls`; `reference_version=4.0.1.post1` |
| `ref.r_mixomics` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_mixomics`; `reference_library=mixOmics`; `reference_version=6.26.0` |
| `ref.r_ropls` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_ropls`; `reference_library=ropls`; `reference_version=Bioc` |
| `ref.r_spls` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_spls`; `reference_library=spls`; `reference_version=2.3.2` |
| `ref.python_diplslib` | — |
| `ref.r_chemometrics` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_chemometrics`; `reference_library=chemometrics`; `reference_version=0.7.x` |
| `ref.r_jico` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_jico`; `reference_library=JICO`; `reference_version=0.0` |
| `ref.python_tensorly` | — |
| `ref.r_kernlab_pls` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_kernlab_pls`; `reference_library=kernlab+pls`; `reference_version=0.9.33+2.8.5` |
| `ref.r_omicspls` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_omicspls`; `reference_library=OmicsPLS`; `reference_version=2.1.0` |
| `ref.r_mdatools` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_mdatools`; `reference_library=mdatools`; `reference_version=0.15.0` |
| `ref.r_multiblock` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_multiblock`; `reference_library=multiblock`; `reference_version=0.8.10` |
| `ref.python_onpls` | `language=Python 3.13.11`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=python_onpls`; `reference_library=OnPLS`; `reference_version=github tomlof/OnPLS` |
| `ref.python_auswahl` | — |
| `ref.python_pyswarms` | — |
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
| `ref.matlab_libpls` | — |
| `ref.r_pls_stats` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_pls_stats`; `reference_library=pls+stats`; `reference_version=R 4.3.3` |

## Methodology

- Reference: `cpp` cell at 1 thread (libp4a via ctypes), or `python_tier1` when `cpp` is unavailable for an algorithm
- Parity tolerance: 1e-6 (per-algo overrides possible)
- All backends read the **same** orchestrator-generated CSV (`benchmarks/cross_binding/data/data_<n>x<p>_seed<seed>.csv`) so input data is bit-identical across languages
- 1 run(s) per cell, first discarded as warmup when `n_runs >= 3`, median reported
- Per-cell timeout: 300 s
- Thread control via `OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = BLIS_NUM_THREADS` set in the subprocess env, plus `Context.num_threads` for Python pls4all and `maxNumCompThreads()` for Octave
- pls4all libp4a build: `build/blas-omp/cpp/src/libp4a.so` (BLAS + OpenMP enabled)
