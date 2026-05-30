# Cross-binding benchmark — thread sweep

Same matrix as the [main page](cross_binding.md), but with thread counts 1, 3 and 10 shown in separate per-(algorithm, thread) sections. **External libraries (`sklearn`, `pls`, `ropls`, `mixOmics`, `plsregress`, `ikpls`) typically don't accelerate their inner loops with thread count** — only their linked BLAS does, and that helps only when the algo is GEMM-bound. pls4all bindings use OpenMP at the C kernel level on top of the BLAS, so multi-thread wins are visible here.

_Generated: 2026-05-30 10:15:30 UTC_
_CSV: `tmpnxk259td.csv` (1992 cells)_


## Host

| | |
|---|---|
| **CPU**            | 12th Gen Intel(R) Core(TM) i9-12900K |
| **Cores**          | 12 physical / 24 logical |
| **Frequency**  | 3187 MHz nominal |
| **L3 cache**   | 30720 KB |
| **RAM**            | 62.7 GB |
| **OS / kernel**    | Linux 6.6.87.2-microsoft-standard-WSL2 (x86_64) · kernel 6.6.87.2-microsoft-standard-WSL2 |
| **Python**         | 3.11.15 |


## How to read a cell

Each cell shows `<reported_ms> <gate>`. C++ and external library cells use Gate 2, the reference parity against the method's canonical oracle. Internal binding cells use Gate 1, the binding parity against the native C++ baseline:

- ✓ **exact/pass** — gate-specific tolerance passed
- ⚠ **drift** — finite mismatch, below 10× the method's gate-2 tolerance or below the binding drift band
- ✗ **divergent** — outside the gate's tolerance band
- ⏳ cell hit the 24 h wall-clock guard
- `—` see *Why a cell is empty* below

**Bold** = fastest cell on the row, **counted only among cells whose relevant gate is ✓**. For internal bindings that is Gate 1; for C++ and external libraries it is Gate 2. Drift / divergent / empty cells never carry the bold.

Timing uses the adaptive protocol: run #1 is the warmstart; if run #1 exceeds 5 min it is reported and the cell stops. Otherwise run #2 is the first scored sample and the warmstart is excluded. A 2-run cell reports run #2 alone; a 3-run cell reports the mean of runs #2-#3; 10/20/40-run cells report the median of runs after the warmstart. All backends in a single cell read the same orchestrator-generated CSV dataset. See [methodology.md](methodology.md) for the full details.


## Why a cell is empty (`—`)

An empty cell means the backend **did not produce a timing for that (algorithm, size) combination**. Four distinct reasons, all reported as `—` in the matrix:

1. **External library doesn't implement the algorithm.** Example: `sklearn` has no native CPPLS or sparse SIMPLS; `plsregress` (Octave) only does plain PLS; `ikpls` is plain PLS only; `ropls` only does OPLS; `mixOmics` covers PLS / sparse PLS / PLS-DA. Filling this column requires the external maintainer to add the algorithm — out of our control.
2. **pls4all wrapper missing for that algorithm.** Example: `pls4all.sklearn` (tier 2) doesn't yet ship a class for `continuum_regression` or `kernel_pls`; `pls4all.R.formula` doesn't have a formula wrapper for every algorithm yet. This is the *chemin à parcourir* on our side — visible TODO.
3. **Bench script missing the dispatch case.** A few cells can fail because a binding bench script doesn't yet route a specific algorithm to its underlying call. Pure tooling gap, no library impact.
4. **Run too slow / OOM / crashed.** The cell hit the 24 h guard (`⏳`) or raised a runtime error.

Each `—` represents one of these four reasons. The CSV (`results/full_matrix.csv`) carries a `reason` column that tells you exactly which one for any given cell.


Column names: anything starting with `pls4all.` is one of this project's bindings; the others use their real package name (`sklearn`, `ikpls`, `pls`, `ropls`, `mixOmics`, `plsregress`). **Every algorithm table keeps every column** — `—` cells are intentional and show *where coverage is still missing*, either on our side (an algorithm not yet wrapped in a tier-2 class) or on the external side (e.g. `sklearn` doesn't implement CPPLS / OPLS, `plsregress` only does plain PLS, `ikpls` only does plain PLS). Full per-column description: see [Backend definitions](#backend-definitions) at the bottom of this page.


## aom_pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 4.20 ms ✓ | 4.30 ms ✓ | 4.16 ms ✓ | **4.15 ms ✓** | 7.84 ms ✓ | 9.43 ms ✓ | 9.87 ms ✓ | 17.6 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 41.9 ms ✓ | — | — | — | — | — | — |


## aom_pls  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 14.6 ms ✓ | 25.3 ms ✓ | **8.48 ms ✓** | 14.7 ms ✓ | 27.0 ms ✓ | 40.4 ms ✓ | 47.9 ms ✓ | 50.0 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 59.7 ms ✓ | — | — | — | — | — | — |


## aom_pls  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 23.7 ms ✓ | 20.5 ms ✓ | 23.7 ms ✓ | 15.5 ms ✓ | 13.4 ms ✓ | 16.4 ms ✓ | **12.4 ms ✓** | 12.9 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 16.6 ms ✓ | — | — | — | — | — | — |


## aom_preprocess  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 1.85 ms ✓ | 2.52 ms ✓ | 1.76 ms ✓ | **1.75 ms ✓** | 6.07 ms ✓ | 7.96 ms ✓ | 7.33 ms ✓ | 6.56 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 2.13 ms ✓ | — | — | — | — | — | — |


## aom_preprocess  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 2.95 ms ✓ | 2.04 ms ✓ | **1.74 ms ✓** | 3.37 ms ✓ | 10.4 ms ✓ | 35.7 ms ✓ | 32.9 ms ✓ | 29.5 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 11.5 ms ✓ | — | — | — | — | — | — |


## aom_preprocess  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 3.32 ms ✓ | 5.02 ms ✓ | 7.59 ms ✓ | 4.10 ms ✓ | 9.94 ms ✓ | 9.00 ms ✓ | 8.32 ms ✓ | 8.97 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **1.87 ms ✓** | — | — | — | — | — | — |


## approximate_press  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | 28.4 ms ✓ | 28.0 ms ✓ | 28.1 ms ✓ | 28.1 ms ✓ | **24.7 ms ✓** | 25.5 ms ✓ | 25.7 ms ✓ | 25.3 ms ✓ | — | 69.5 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## approximate_press  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | 28.9 ms ✓ | 28.8 ms ✓ | 28.8 ms ✓ | 28.8 ms ✓ | **24.9 ms ✓** | 25.2 ms ✓ | 25.4 ms ✓ | 25.5 ms ✓ | — | 69.2 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## approximate_press  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | 29.4 ms ✓ | 28.9 ms ✓ | 28.6 ms ✓ | 28.9 ms ✓ | **24.5 ms ✓** | 25.0 ms ✓ | 25.2 ms ✓ | 24.9 ms ✓ | — | 67.7 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## bagging_pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | 9.82 ms ✓ | 10.2 ms ✓ | **9.30 ms ✓** | 1.69 ms ✗ | 4.62 ms ✗ | 6.41 ms ✗ | 4.93 ms ✗ | 5.11 ms ✗ | 11.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## bagging_pls  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **21.7 ms ✓** | 31.0 ms ✓ | 60.8 ms ✓ | 11.4 ms ✗ | 25.2 ms ✗ | 40.6 ms ✗ | 45.0 ms ✗ | 46.3 ms ✗ | 56.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## bagging_pls  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | 27.8 ms ✓ | 24.9 ms ✓ | **12.7 ms ✓** | 2.54 ms ✗ | 5.73 ms ✗ | 6.42 ms ✗ | 7.41 ms ✗ | 7.63 ms ✗ | 14.2 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## bipls_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 3.83 ms ✓ | 4.36 ms ✓ | **3.60 ms ✓** | 3.69 ms ✓ | 8.26 ms ✓ | 11.7 ms ✓ | 8.83 ms ✓ | 9.90 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 365.6 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## bipls_select  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **3.49 ms ✓** | 3.49 ms ✓ | 3.58 ms ✓ | 3.74 ms ✓ | 9.93 ms ✓ | 25.5 ms ✓ | 24.8 ms ✓ | 15.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.0 s ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## bipls_select  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 14.3 ms ✓ | 17.7 ms ✓ | 13.4 ms ✓ | **5.97 ms ✓** | 23.9 ms ✓ | 35.4 ms ✓ | 21.7 ms ✓ | 16.3 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 333.9 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## boosting_pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **1.23 ms ✓** | 1.23 ms ✓ | 1.52 ms ✓ | 2.07 ms ✗ | 5.05 ms ✗ | 4.95 ms ✗ | 6.27 ms ✗ | 6.92 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 18.5 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — |


## boosting_pls  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **1.25 ms ✓** | 1.26 ms ✓ | 1.25 ms ✓ | 1.79 ms ✗ | 4.47 ms ✗ | 4.80 ms ✗ | 5.37 ms ✗ | 6.14 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 16.4 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — |


## boosting_pls  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | 1.27 ms ✓ | **1.27 ms ✓** | 1.70 ms ✓ | 2.85 ms ✗ | 9.14 ms ✗ | 8.90 ms ✗ | 14.2 ms ✗ | 11.2 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 30.5 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — |


## bve_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 510.0 ms ✓ | 443.3 ms ✓ | 563.8 ms ✓ | 42.4 ms ✗ | 58.7 ms ✗ | 66.5 ms ✗ | 141.0 ms ✗ | 152.1 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **46.9 ms ✓** | — | — | — | — | — |


## bve_select  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 491.1 ms ✓ | 481.4 ms ✓ | 491.1 ms ✓ | 43.0 ms ✗ | 61.4 ms ✗ | 62.6 ms ✗ | 61.3 ms ✗ | 62.1 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **47.3 ms ✓** | — | — | — | — | — |


## bve_select  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 507.2 ms ✓ | 1.1 s ✓ | 479.8 ms ✓ | 43.9 ms ✗ | 61.3 ms ✗ | 62.1 ms ✗ | 60.9 ms ✗ | 109.1 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **45.8 ms ✓** | — | — | — | — | — |


## cars_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 780.7 ms ✓ | 752.8 ms ✓ | 772.6 ms ✓ | — | 13.3 ms ✗ | 10.2 ms ✗ | 13.3 ms ✗ | 16.4 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **188.9 ms ✓** | — | — | — | — |


## cars_select  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 767.0 ms ✓ | 824.3 ms ✓ | 761.2 ms ✓ | — | 5.13 ms ✗ | 5.78 ms ✗ | 5.70 ms ✗ | 5.88 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **62.9 ms ✓** | — | — | — | — |


## cars_select  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 881.2 ms ✓ | 1.9 s ✓ | 802.5 ms ✓ | — | 4.98 ms ✗ | 5.70 ms ✗ | 5.76 ms ✗ | 5.89 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **63.6 ms ✓** | — | — | — | — |


## continuum_regression  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | 2.73 ms ✓ | 2.76 ms ✓ | 2.63 ms ✓ | 2.77 ms ✓ | 5.55 ms ✓ | 6.02 ms ✓ | 6.62 ms ✓ | 6.62 ms ✓ | — | — | — | — | — | — | — | — | — | **2.27 ms ✓** | 31.6 ms — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## continuum_regression  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | 2.70 ms ✓ | 2.65 ms ✓ | 2.65 ms ✓ | 2.83 ms ✓ | 5.33 ms ✓ | 6.41 ms ✓ | 6.57 ms ✓ | 6.49 ms ✓ | — | — | — | — | — | — | — | — | — | **2.39 ms ✓** | 31.6 ms — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## continuum_regression  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **2.70 ms ✓** | 2.71 ms ✓ | 2.76 ms ✓ | 2.93 ms ✓ | 6.21 ms ✓ | 6.41 ms ✓ | 6.57 ms ✓ | 7.10 ms ✓ | — | — | — | — | — | — | — | — | — | 4.06 ms ✓ | 49.0 ms — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## cppls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | 1.88 ms ✓ | **1.79 ms ✓** | 1.87 ms ✓ | 1.97 ms ✓ | 4.57 ms ✓ | 5.65 ms ✓ | 10.6 ms ✗ | 6.07 ms ✗ | — | 9.91 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## cppls  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | 1.78 ms ✓ | **1.76 ms ✓** | 1.79 ms ✓ | 1.97 ms ✓ | 4.38 ms ✓ | 5.49 ms ✓ | 10.6 ms ✗ | 6.44 ms ✗ | — | 9.85 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## cppls  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **1.69 ms ✓** | 1.86 ms ✓ | 1.80 ms ✓ | 1.95 ms ✓ | 4.57 ms ✓ | 5.38 ms ✓ | 11.1 ms ✗ | 6.59 ms ✗ | — | 10.5 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## di_pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | 2.58 ms ✓ | **2.57 ms ✓** | 2.74 ms ✓ | 2.82 ms ✓ | 9.69 ms ✓ | 12.4 ms ✓ | 12.4 ms ✓ | 12.8 ms ✓ | — | — | — | — | — | — | — | 3.78 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## di_pls  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | 2.80 ms ✓ | **2.69 ms ✓** | 2.70 ms ✓ | 2.98 ms ✓ | 9.91 ms ✓ | 12.2 ms ✓ | 11.2 ms ✓ | 12.9 ms ✓ | — | — | — | — | — | — | — | 3.79 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## di_pls  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **2.74 ms ✓** | 2.81 ms ✓ | 2.82 ms ✓ | 2.95 ms ✓ | 11.1 ms ✓ | 14.0 ms ✓ | 12.4 ms ✓ | 13.0 ms ✓ | — | — | — | — | — | — | — | 3.92 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## ds  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **1.64 ms ✓** | 1.73 ms ✓ | 2.09 ms ✓ | 2.27 ms ✓ | 10.2 ms ✓ | 14.0 ms ✓ | 12.6 ms ✓ | 11.4 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 31.0 ms ✓ | — | — | — | — | — | — | — | — | — | — |


## ds  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | 3.32 ms ✓ | **1.44 ms ✓** | 1.57 ms ✓ | 1.72 ms ✓ | 11.1 ms ✓ | 11.8 ms ✓ | 13.0 ms ✓ | 10.7 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 27.8 ms ✓ | — | — | — | — | — | — | — | — | — | — |


## ds  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **1.42 ms ✓** | 1.94 ms ✓ | 2.13 ms ✓ | 1.72 ms ✓ | 8.95 ms ✓ | 9.70 ms ✓ | 10.1 ms ✓ | 9.43 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 28.0 ms ✓ | — | — | — | — | — | — | — | — | — | — |


## ecr  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | 2.16 ms ✓ | **2.15 ms ✓** | 2.24 ms ✓ | 2.37 ms ✓ | 5.88 ms ✓ | 6.96 ms ✓ | 8.61 ms ✓ | 7.41 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 62.4 ms ✓ | — |


## ecr  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | 2.19 ms ✓ | 2.19 ms ✓ | **2.14 ms ✓** | 2.32 ms ✓ | 5.03 ms ✓ | 6.42 ms ✓ | 6.08 ms ✓ | 6.40 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 61.5 ms ✓ | — |


## ecr  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **2.11 ms ✓** | 2.24 ms ✓ | 2.16 ms ✓ | 2.27 ms ✓ | 5.42 ms ✓ | 7.74 ms ✓ | 6.36 ms ✓ | 6.53 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 92.2 ms ✓ | — |


## emcuve_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 1.3 s ✓ | 1.3 s ✓ | 1.3 s ✓ | **1.98 ms ✓** | 4.51 ms ✓ | 5.44 ms ✓ | 5.57 ms ✓ | 5.25 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 922.4 ms ✓ | — | — | — | — | — |


## emcuve_select  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 1.3 s ✓ | 1.3 s ✓ | 1.3 s ✓ | **2.00 ms ✓** | 4.69 ms ✓ | 5.25 ms ✓ | 6.05 ms ✓ | 5.99 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 918.1 ms ✓ | — | — | — | — | — |


## emcuve_select  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 1.3 s ✓ | 1.3 s ✓ | 1.3 s ✓ | **1.90 ms ✓** | 4.22 ms ✓ | 5.03 ms ✓ | 5.01 ms ✓ | 4.95 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 907.4 ms ✓ | — | — | — | — | — |


## fused_sparse_pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | 1.93 ms — | **1.86 ms ✓** | 1.95 ms ✓ | 2.13 ms ✓ | 6.13 ms ✓ | 7.70 ms ✓ | 7.80 ms ✓ | 7.55 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## fused_sparse_pls  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | 1.88 ms — | **1.87 ms ✓** | 1.88 ms ✓ | 2.09 ms ✓ | 6.14 ms ✓ | 7.73 ms ✓ | 7.53 ms ✓ | 8.04 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## fused_sparse_pls  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | 1.87 ms — | **1.95 ms ✓** | 1.97 ms ✓ | 2.02 ms ✓ | 6.56 ms ✓ | 8.64 ms ✓ | 8.65 ms ✓ | 8.07 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## ga_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 607.6 ms ✓ | 625.8 ms ✓ | 756.5 ms ✓ | 3.94 ms ✗ | 7.38 ms ✗ | 8.59 ms ✗ | 9.76 ms ✗ | 16.0 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **218.7 ms ✓** | — | — | — | — | — |


## ga_select  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 628.1 ms ✓ | 829.6 ms ✓ | 600.8 ms ✓ | 3.94 ms ✗ | 7.17 ms ✗ | 7.95 ms ✗ | 8.03 ms ✗ | 8.10 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **425.9 ms ✓** | — | — | — | — | — |


## ga_select  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 626.6 ms ✓ | 690.0 ms ✓ | 1.8 s ✓ | 3.88 ms ✗ | 7.01 ms ✗ | 7.86 ms ✗ | 7.89 ms ✗ | 7.57 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **226.4 ms ✓** | — | — | — | — | — |


## gpr_pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 120×25 | 5.91 ms ✓ | 6.75 ms ✓ | 6.17 ms ✓ | 2.75 ms ✓ | 20.9 ms ✓ | 12.6 ms ✓ | 14.2 ms ✓ | 8.10 ms ✓ | **2.25 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## gpr_pls  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 120×25 | **1.14 ms ✓** | 1.30 ms ✓ | 2.43 ms ✓ | 2.72 ms ✓ | 4.45 ms ✓ | 5.61 ms ✓ | 6.64 ms ✓ | 6.10 ms ✓ | 2.75 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## gpr_pls  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 120×25 | **1.14 ms ✓** | 1.86 ms ✓ | 1.80 ms ✓ | 1.39 ms ✓ | 2.53 ms ✓ | 3.11 ms ✓ | 3.10 ms ✓ | 3.24 ms ✓ | 2.29 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## group_sparse_pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | 8.36 ms ✓ | 4.69 ms ✓ | 7.86 ms ✓ | 3.79 ms ✗ | 9.70 ms ✗ | 11.0 ms ✗ | 11.0 ms ✗ | 10.2 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **2.98 ms ✓** | — | — | — | — | — | — | — | 33.7 ms — | — | — | — | — | — | — | — |


## group_sparse_pls  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **2.62 ms ✓** | 3.88 ms ✓ | 2.76 ms ✓ | 2.28 ms ✗ | 8.78 ms ✗ | 9.80 ms ✗ | 10.5 ms ✗ | 11.9 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 3.11 ms ✓ | — | — | — | — | — | — | — | 31.2 ms — | — | — | — | — | — | — | — |


## group_sparse_pls  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | 2.91 ms ✓ | 2.82 ms ✓ | 2.69 ms ✓ | 2.53 ms ✗ | 7.45 ms ✗ | 9.19 ms ✗ | 8.50 ms ✗ | 11.3 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **2.66 ms ✓** | — | — | — | — | — | — | — | 26.7 ms — | — | — | — | — | — | — | — |


## interval_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 1.5 s ✓ | 2.0 s ✓ | 1.5 s ✓ | 3.75 ms ✗ | 10.9 ms ✗ | 11.7 ms ✗ | 10.3 ms ✗ | 13.9 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **632.3 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## interval_select  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 2.0 s ✓ | 968.0 ms ✓ | 2.0 s ✓ | 3.95 ms ✗ | 8.51 ms ✗ | 9.33 ms ✗ | 9.36 ms ✗ | 7.09 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **706.6 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## interval_select  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 1.1 s ✓ | 1.0 s ✓ | 994.2 ms ✓ | 2.26 ms ✗ | 5.63 ms ✗ | 6.68 ms ✗ | 7.03 ms ✗ | 6.61 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **507.0 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## ipw_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 360.4 ms ✓ | 358.1 ms ✓ | 366.9 ms ✓ | 1.81 ms ✗ | 4.86 ms ✗ | 5.51 ms ✗ | 5.86 ms ✗ | 5.15 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **22.0 ms ✓** | — | — | — | — | — |


## ipw_select  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 368.1 ms ✓ | 372.0 ms ✓ | 358.9 ms ✓ | 1.82 ms ✗ | 4.32 ms ✗ | 5.47 ms ✗ | 5.37 ms ✗ | 5.38 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **20.7 ms ✓** | — | — | — | — | — |


## ipw_select  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 392.3 ms ✓ | 388.3 ms ✓ | 392.5 ms ✓ | 1.86 ms ✗ | 4.39 ms ✗ | 5.60 ms ✗ | 5.70 ms ✗ | 5.39 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **26.2 ms ✓** | — | — | — | — | — |


## irf_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 120×30 | — | — | — | **2.02 ms ✓** | 2.92 ms ✓ | 3.44 ms ✓ | 3.91 ms ✓ | 3.37 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## irf_select  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 120×30 | — | — | — | **1.53 ms ✓** | 2.93 ms ✓ | 3.29 ms ✓ | 3.55 ms ✓ | 3.58 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## irf_select  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 120×30 | — | — | — | **1.53 ms ✓** | 2.56 ms ✓ | 3.34 ms ✓ | 3.57 ms ✓ | 3.55 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## iriv_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 80×25 | **268.0 ms ✓** | 269.9 ms ✓ | 270.2 ms ✓ | 25.3 ms ✗ | 29.4 ms ✗ | 30.0 ms ✗ | 30.5 ms ✗ | 30.8 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 274.8 ms ✓ |


## iriv_select  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 80×25 | 273.2 ms ✓ | **265.7 ms ✓** | 267.2 ms ✓ | 25.1 ms ✗ | 29.8 ms ✗ | 30.8 ms ✗ | 30.3 ms ✗ | 30.6 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 275.1 ms ✓ |


## iriv_select  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 80×25 | 274.2 ms ✓ | **267.6 ms ✓** | 275.4 ms ✓ | 25.4 ms ✗ | 29.9 ms ✗ | 30.7 ms ✗ | 30.7 ms ✗ | 31.0 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 270.7 ms ✓ |


## kernel_pls_rbf  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | 3.07 ms ✓ | **3.02 ms ✓** | 3.23 ms ✓ | 3.23 ms ✓ | 6.34 ms ✓ | 7.05 ms ✓ | 7.21 ms ✓ | 6.83 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | 28.6 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## kernel_pls_rbf  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **3.12 ms ✓** | 3.15 ms ✓ | 3.17 ms ✓ | 3.29 ms ✓ | 5.98 ms ✓ | 7.13 ms ✓ | 7.20 ms ✓ | 7.16 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | 27.3 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## kernel_pls_rbf  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | 3.18 ms ✓ | 3.18 ms ✓ | **3.14 ms ✓** | 3.22 ms ✓ | 6.09 ms ✓ | 7.28 ms ✓ | 7.41 ms ✓ | 7.22 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | 40.0 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## lw_pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 25.6 ms ✓ | **13.3 ms ✓** | 37.2 ms ✓ | 9.19 ms ✗ | 20.3 ms ✗ | 23.2 ms ✗ | 23.2 ms ✗ | 24.6 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 23.9 ms ✓ | — | — | — | — | — | — |


## lw_pls  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 10.8 ms ✓ | **10.0 ms ✓** | 10.3 ms ✓ | 4.44 ms ✗ | 9.02 ms ✗ | 10.0 ms ✗ | 11.1 ms ✗ | 10.8 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 23.7 ms ✓ | — | — | — | — | — | — |


## lw_pls  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **11.8 ms ✓** | 11.9 ms ✓ | 12.9 ms ✓ | 5.14 ms ✗ | 9.85 ms ✗ | 9.03 ms ✗ | 12.1 ms ✗ | 11.2 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 29.0 ms ✓ | — | — | — | — | — | — |


## mb_pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×60 | **2.37 ms ✓** | 2.73 ms ✓ | 2.85 ms ✓ | 2.59 ms ✓ | 7.82 ms ✓ | 10.9 ms ✓ | 10.2 ms ✓ | 11.5 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 2.99 ms ✓ | — | — | — | — | — | — |


## mb_pls  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×60 | 2.33 ms ✓ | 2.82 ms ✓ | **2.24 ms ✓** | 2.55 ms ✓ | 7.24 ms ✓ | 10.4 ms ✓ | 9.18 ms ✓ | 9.38 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 7.12 ms ✓ | — | — | — | — | — | — |


## mb_pls  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×60 | 13.6 ms ✓ | **8.23 ms ✓** | 8.63 ms ✓ | 9.61 ms ✓ | 30.4 ms ✓ | 46.5 ms ✓ | 41.0 ms ✓ | 48.5 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 9.70 ms ✓ | — | — | — | — | — | — |


## mir_pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | 1.19 ms — | **1.21 ms ✓** | 1.27 ms ✓ | 1.47 ms ✓ | 3.98 ms ✓ | 4.72 ms ✓ | 5.04 ms ✓ | 5.11 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## mir_pls  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | 1.33 ms — | **1.28 ms ✓** | 1.29 ms ✓ | 1.43 ms ✓ | 4.49 ms ✓ | 5.07 ms ✓ | 4.94 ms ✓ | 5.39 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## mir_pls  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | 1.22 ms — | 2.14 ms ✓ | 1.99 ms ✓ | **1.41 ms ✓** | 4.09 ms ✓ | 5.05 ms ✓ | 4.83 ms ✓ | 4.73 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## missing_aware_nipals  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | 1.27 ms ✓ | **1.24 ms ✓** | 1.25 ms ✓ | 1.39 ms ✓ | 4.25 ms ✓ | 4.88 ms ✓ | 4.85 ms ✓ | 5.41 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 13.6 ms ✓ | — | — | — | — | — | — | — | — | — |


## missing_aware_nipals  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **1.22 ms ✓** | 1.24 ms ✓ | 1.36 ms ✓ | 1.37 ms ✓ | 4.02 ms ✓ | 4.84 ms ✓ | 4.63 ms ✓ | 4.88 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 14.9 ms ✓ | — | — | — | — | — | — | — | — | — |


## missing_aware_nipals  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **1.34 ms ✓** | 1.92 ms ✓ | 1.79 ms ✓ | 3.36 ms ✓ | 12.2 ms ✓ | 16.6 ms ✓ | 18.2 ms ✓ | 12.2 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 29.9 ms ✓ | — | — | — | — | — | — | — | — | — |


## n_pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×48 | 19.5 ms ✓ | 19.3 ms ✓ | **19.0 ms ✓** | 2.03 ms ✗ | 4.51 ms ✗ | 5.67 ms ✗ | 5.68 ms ✗ | 5.37 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | 19.3 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## n_pls  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×48 | 19.1 ms ✓ | **18.9 ms ✓** | 20.3 ms ✓ | 1.97 ms ✗ | 6.01 ms ✗ | 6.25 ms ✗ | 6.14 ms ✗ | 5.93 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | 19.3 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## n_pls  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×48 | 18.9 ms ✓ | 20.0 ms ✓ | 19.4 ms ✓ | 1.87 ms ✗ | 4.84 ms ✗ | 5.52 ms ✗ | 5.46 ms ✗ | 5.17 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | **18.8 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## o2pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | 1.21 ms ✓ | 1.20 ms ✓ | **1.19 ms ✓** | 1.37 ms ✓ | 2.69 ms ✓ | 3.44 ms ✓ | 3.58 ms ✓ | 3.42 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | 7.21 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## o2pls  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **1.22 ms ✓** | 1.24 ms ✓ | 1.24 ms ✓ | 1.44 ms ✓ | 2.78 ms ✓ | 3.52 ms ✓ | 3.68 ms ✓ | 3.53 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | 7.41 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## o2pls  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **1.24 ms ✓** | 1.33 ms ✓ | 2.09 ms ✓ | 1.45 ms ✓ | 3.09 ms ✓ | 3.52 ms ✓ | 3.25 ms ✓ | 3.49 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | 7.31 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## on_pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | 6.20 ms ✓ | 7.71 ms ✓ | 3.02 ms ✓ | **2.69 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 12.2 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## on_pls  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **1.81 ms ✓** | 1.91 ms ✓ | 1.91 ms ✓ | 1.84 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 9.46 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## on_pls  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | 1.91 ms ✓ | **1.90 ms ✓** | 2.00 ms ✓ | 16.0 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 13.4 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## one_se_rule  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | 2.10 ms ✓ | **2.07 ms ✓** | 2.08 ms ✓ | 1.13 ms ✗ | 3.25 ms ✗ | 3.87 ms ✗ | 4.01 ms ✗ | 3.96 ms ✗ | — | 11.6 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## one_se_rule  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | 7.61 ms ✓ | 3.24 ms ✓ | **2.19 ms ✓** | 1.23 ms ✗ | 3.28 ms ✗ | 4.07 ms ✗ | 4.00 ms ✗ | 4.07 ms ✗ | — | 11.6 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## one_se_rule  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | 2.06 ms ✓ | **2.06 ms ✓** | 2.14 ms ✓ | 1.13 ms ✗ | 2.80 ms ✗ | 5.64 ms ✗ | 4.60 ms ✗ | 4.37 ms ✗ | — | 11.5 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## opls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **1.87 ms ✓** | 1.93 ms ✓ | 1.88 ms ✓ | 2.12 ms ✓ | 4.35 ms ✓ | 5.13 ms ✓ | 5.40 ms ✓ | 5.23 ms ✓ | — | — | — | — | 15.4 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## opls  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | 1.96 ms ✓ | 1.80 ms ✓ | **1.79 ms ✓** | 2.14 ms ✓ | 6.22 ms ✓ | 5.57 ms ✓ | 7.88 ms ✓ | 7.12 ms ✓ | — | — | — | — | 16.0 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## opls  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **1.73 ms ✓** | 1.92 ms ✓ | 1.81 ms ✓ | 1.94 ms ✓ | 4.17 ms ✓ | 5.09 ms ✓ | 5.17 ms ✓ | 5.39 ms ✓ | — | — | — | — | 16.0 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pcr  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | 2.92 ms ✓ | 2.84 ms ✓ | 2.95 ms ✓ | **2.68 ms ✓** | 5.15 ms ✓ | 6.41 ms ✓ | 11.8 ms ✓ | 6.21 ms ✓ | 2.71 ms ✓ | 7.99 ms — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pcr  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | 2.52 ms ✓ | 2.50 ms ✓ | **2.50 ms ✓** | 2.72 ms ✓ | 5.09 ms ✓ | 6.36 ms ✓ | 10.4 ms ✓ | 6.27 ms ✓ | 2.85 ms ✓ | 7.83 ms — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pcr  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **2.49 ms ✓** | 2.70 ms ✓ | 2.73 ms ✓ | 2.67 ms ✓ | 4.86 ms ✓ | 5.87 ms ✓ | 10.7 ms ✓ | 6.40 ms ✓ | 4.94 ms ✓ | 14.0 ms — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pds  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **1.37 ms ✓** | 1.55 ms ✓ | 1.44 ms ✓ | 1.86 ms ✓ | 10.6 ms ✓ | 10.1 ms ✓ | 14.2 ms ✓ | 13.7 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 43.1 ms ✓ | — | — | — | — | — | — | — | — | — | — |


## pds  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **2.54 ms ✓** | 6.79 ms ✓ | 2.67 ms ✓ | 7.45 ms ✓ | 34.6 ms ✓ | 34.8 ms ✓ | 49.1 ms ✓ | 42.9 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 112.2 ms ✓ | — | — | — | — | — | — | — | — | — | — |


## pds  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | 6.67 ms ✓ | **2.94 ms ✓** | 3.05 ms ✓ | 3.01 ms ✓ | 15.7 ms ✓ | 18.9 ms ✓ | 10.5 ms ✓ | 15.4 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 35.6 ms ✓ | — | — | — | — | — | — | — | — | — | — |


## pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | 1.70 ms ✓ | 1.73 ms ✓ | **1.69 ms ✓** | 1.97 ms ✓ | 4.79 ms ✓ | 5.26 ms ✓ | 9.99 ms ✓ | 5.92 ms ✓ | 2.16 ms ✓ | 8.01 ms — | 1.92 ms — | 9.72 ms — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | 1.79 ms ✓ | **1.74 ms ✓** | 1.76 ms ✓ | 1.95 ms ✓ | 4.43 ms ✓ | 5.76 ms ✓ | 10.3 ms ✓ | 6.29 ms ✓ | 2.16 ms ✓ | 7.50 ms — | 1.90 ms — | 9.82 ms — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | 1.79 ms ✓ | 1.82 ms ✓ | **1.78 ms ✓** | 1.93 ms ✓ | 4.69 ms ✓ | 5.49 ms ✓ | 10.4 ms ✓ | 5.59 ms ✓ | 2.17 ms ✓ | 8.15 ms — | 2.04 ms — | 9.91 ms — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_cox  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | 1.79 ms ✓ | 1.84 ms ✓ | **1.75 ms ✓** | 2.13 ms ✗ | 4.52 ms ✗ | 5.52 ms ✗ | 5.43 ms ✗ | 5.51 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 2.03 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_cox  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | 1.92 ms ✓ | **1.92 ms ✓** | 1.97 ms ✓ | 1.45 ms ✗ | 4.59 ms ✗ | 20.3 ms ✗ | 14.0 ms ✗ | 14.8 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 6.01 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_cox  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | 6.51 ms ✓ | 7.34 ms ✓ | 4.69 ms ✓ | 3.91 ms ✗ | 9.54 ms ✗ | 17.4 ms ✗ | 17.2 ms ✗ | 12.8 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **2.02 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_diagnostic_dmodx  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | 1.25 ms ✓ | 1.30 ms ✓ | **1.25 ms ✓** | 1.41 ms ✓ | 4.16 ms ✓ | 4.79 ms ✓ | 4.47 ms ✓ | 5.71 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 19.8 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_diagnostic_dmodx  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | 1.28 ms ✓ | 1.31 ms ✓ | **1.25 ms ✓** | 1.41 ms ✓ | 6.86 ms ✓ | 20.8 ms ✓ | 18.2 ms ✓ | 14.5 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 43.6 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_diagnostic_dmodx  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | 6.12 ms ✓ | 2.44 ms ✓ | 2.29 ms ✓ | **1.45 ms ✓** | 3.74 ms ✓ | 4.94 ms ✓ | 4.64 ms ✓ | 5.32 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 21.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_diagnostic_q  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | 1.22 ms ✓ | **1.19 ms ✓** | 1.19 ms ✓ | 1.30 ms ✓ | 3.10 ms ✓ | 3.57 ms ✓ | 3.47 ms ✓ | 3.96 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 13.8 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_diagnostic_q  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | 1.21 ms ✓ | **1.20 ms ✓** | 1.21 ms ✓ | 1.27 ms ✓ | 3.11 ms ✓ | 3.59 ms ✓ | 3.64 ms ✓ | 3.52 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 14.9 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_diagnostic_q  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | 1.23 ms ✓ | 2.04 ms ✓ | **1.22 ms ✓** | 1.36 ms ✓ | 3.28 ms ✓ | 3.83 ms ✓ | 3.74 ms ✓ | 3.89 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 15.5 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_diagnostic_t2  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **1.19 ms ✓** | 1.22 ms ✓ | 1.22 ms ✓ | 1.31 ms ✓ | 2.67 ms ✓ | 3.38 ms ✓ | 4.11 ms ✓ | 3.70 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 14.2 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_diagnostic_t2  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | 1.19 ms ✓ | **1.18 ms ✓** | 1.19 ms ✓ | 1.37 ms ✓ | 2.74 ms ✓ | 3.29 ms ✓ | 3.70 ms ✓ | 3.43 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 13.5 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_diagnostic_t2  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **1.21 ms ✓** | 1.32 ms ✓ | 2.06 ms ✓ | 1.34 ms ✓ | 2.78 ms ✓ | 3.45 ms ✓ | 3.45 ms ✓ | 3.65 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 13.5 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_glm  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **3.57 ms ✓** | 6.25 ms ✓ | 7.99 ms ✓ | 2.74 ms ✗ | 7.97 ms ✗ | 22.7 ms ✗ | 18.4 ms ✗ | 25.8 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 240.6 ms ✓ | — | — | — | — | — | — | — | — | — | — | — |


## pls_glm  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **3.56 ms ✓** | 13.1 ms ✓ | 8.04 ms ✓ | 1.50 ms ✗ | 4.56 ms ✗ | 5.18 ms ✗ | 5.24 ms ✗ | 5.40 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 138.8 ms ✓ | — | — | — | — | — | — | — | — | — | — | — |


## pls_glm  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | 2.07 ms ✓ | 2.35 ms ✓ | **2.05 ms ✓** | 1.46 ms ✗ | 3.78 ms ✗ | 5.59 ms ✗ | 4.78 ms ✗ | 4.89 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 134.0 ms ✓ | — | — | — | — | — | — | — | — | — | — | — |


## pls_lda  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **3.14 ms ✓** | 7.18 ms ✓ | 7.48 ms ✓ | — | — | — | — | — | 7.32 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_lda  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 6.86 ms ✓ | 7.83 ms ✓ | **4.05 ms ✓** | — | — | — | — | — | 4.54 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_lda  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **1.95 ms ✓** | 2.56 ms ✓ | 2.67 ms ✓ | — | — | — | — | — | 3.59 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_logistic  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **3.14 ms ✓** | 4.04 ms ✓ | 3.34 ms ✓ | — | — | — | — | — | 3.44 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_logistic  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **3.20 ms ✓** | 3.51 ms ✓ | 3.81 ms ✓ | — | — | — | — | — | 23.2 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_logistic  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 4.74 ms ✓ | **4.34 ms ✓** | 4.36 ms ✓ | — | — | — | — | — | 5.41 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_monitoring  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | 1.23 ms ✓ | 1.23 ms ✓ | **1.18 ms ✓** | 1.32 ms ✓ | 2.90 ms ✓ | 3.54 ms ✓ | 3.74 ms ✓ | 3.81 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 14.2 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_monitoring  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **1.19 ms ✓** | 1.21 ms ✓ | 2.29 ms ✓ | 1.41 ms ✓ | 3.19 ms ✓ | 4.18 ms ✓ | 4.17 ms ✓ | 4.00 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 14.3 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_monitoring  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **1.26 ms ✓** | 2.12 ms ✓ | 1.61 ms ✓ | 1.34 ms ✓ | 5.85 ms ✓ | 9.79 ms ✓ | 3.84 ms ✓ | 7.99 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 14.4 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_qda  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | 2.65 ms ✓ | **2.09 ms ✓** | 2.12 ms ✓ | 2.11 ms ✗ | 4.06 ms ✗ | 4.67 ms ✗ | 4.99 ms ✗ | 5.46 ms ✗ | 2.36 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_qda  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | 1.94 ms ✓ | **1.90 ms ✓** | 1.92 ms ✓ | 1.64 ms ✗ | 8.84 ms ✗ | 18.1 ms ✗ | 13.5 ms ✗ | 18.2 ms ✗ | 6.90 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_qda  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | 5.99 ms ✓ | 6.01 ms ✓ | 4.32 ms ✓ | 1.49 ms ✗ | 3.90 ms ✗ | 5.11 ms ✗ | 5.08 ms ✗ | 5.10 ms ✗ | **2.39 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pop_pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 6.24 ms ✓ | **5.68 ms ✓** | 7.23 ms ✓ | 6.51 ms ✓ | 12.0 ms ✓ | 12.5 ms ✓ | 13.2 ms ✓ | 24.7 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 47.8 ms ✓ | — | — | — | — | — | — |


## pop_pls  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 7.34 ms ✓ | 11.0 ms ✓ | 7.46 ms ✓ | **6.77 ms ✓** | 13.2 ms ✓ | 12.9 ms ✓ | 12.4 ms ✓ | 10.0 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 36.4 ms ✓ | — | — | — | — | — | — |


## pop_pls  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 15.3 ms ✓ | **6.13 ms ✓** | 11.1 ms ✓ | 7.06 ms ✓ | 34.2 ms ✓ | 22.8 ms ✓ | 12.5 ms ✓ | 13.5 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 38.6 ms ✓ | — | — | — | — | — | — |


## pso_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 80×25 | 219.8 ms ✓ | 384.2 ms ✓ | 219.3 ms ✓ | 4.05 ms ✗ | 6.11 ms ✗ | 6.29 ms ✗ | 7.12 ms ✗ | 7.33 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **163.8 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pso_select  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 80×25 | 169.0 ms ✓ | **146.5 ms ✓** | 298.6 ms ✓ | 3.17 ms ✗ | 20.9 ms ✗ | 20.3 ms ✗ | 24.6 ms ✗ | 16.0 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 217.8 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pso_select  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 80×25 | 165.6 ms ✓ | 176.8 ms ✓ | 181.1 ms ✓ | 2.80 ms ✗ | 4.19 ms ✗ | 4.78 ms ✗ | 4.74 ms ✗ | 4.76 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **152.3 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — |


## random_frog_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | — | — | — | **2.19 ms ✓** | 4.98 ms ✓ | 5.67 ms ✓ | 5.94 ms ✓ | 5.96 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## random_frog_select  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | — | — | — | **2.19 ms ✓** | 4.83 ms ✓ | 5.64 ms ✓ | 5.73 ms ✓ | 5.77 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## random_frog_select  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | — | — | — | **2.19 ms ✓** | 5.15 ms ✓ | 5.73 ms ✓ | 5.71 ms ✓ | 5.76 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## random_subspace_pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | 13.5 ms ✓ | 25.2 ms ✓ | 13.2 ms ✓ | 1.54 ms ✗ | 5.11 ms ✗ | 5.82 ms ✗ | 5.56 ms ✗ | 6.22 ms ✗ | **13.0 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## random_subspace_pls  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | 12.1 ms ✓ | 14.6 ms ✓ | 13.6 ms ✓ | 2.67 ms ✗ | 4.41 ms ✗ | 5.82 ms ✗ | 6.01 ms ✗ | 6.51 ms ✗ | **10.4 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## random_subspace_pls  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **9.97 ms ✓** | 10.5 ms ✓ | 11.1 ms ✓ | 1.65 ms ✗ | 4.27 ms ✗ | 5.12 ms ✗ | 5.33 ms ✗ | 7.43 ms ✗ | 26.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## randomization_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 181.1 ms ✓ | 182.3 ms ✓ | 182.5 ms ✓ | **2.67 ms ✓** | 6.34 ms ✓ | 5.94 ms ✓ | 6.80 ms ✓ | 6.29 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 54.5 ms ✓ | — | — |


## randomization_select  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 187.4 ms ✓ | 192.4 ms ✓ | 190.6 ms ✓ | **2.80 ms ✓** | 6.12 ms ✓ | 6.58 ms ✓ | 6.95 ms ✓ | 6.46 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 57.0 ms ✓ | — | — |


## randomization_select  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 222.2 ms ✓ | 218.9 ms ✓ | 218.8 ms ✓ | **2.78 ms ✓** | 6.42 ms ✓ | 6.83 ms ✓ | 7.31 ms ✓ | 7.18 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 56.8 ms ✓ | — | — |


## recursive_pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | 2.92 ms ✓ | 2.82 ms ✓ | **2.79 ms ✓** | 3.10 ms ✓ | 5.74 ms ✓ | 6.55 ms ✓ | 7.36 ms ✓ | 6.99 ms ✓ | 48.1 ms ✓ | 141.6 ms — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## recursive_pls  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **2.87 ms ✓** | 2.96 ms ✓ | 2.94 ms ✓ | 3.11 ms ✓ | 5.97 ms ✓ | 6.97 ms ✓ | 6.92 ms ✓ | 7.91 ms ✓ | 46.3 ms ✓ | 144.8 ms — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## recursive_pls  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | 2.95 ms ✓ | **2.87 ms ✓** | 3.02 ms ✓ | 3.06 ms ✓ | 5.98 ms ✓ | 7.42 ms ✓ | 7.32 ms ✓ | 6.94 ms ✓ | 48.7 ms ✓ | 147.3 ms — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## rep_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 605.7 ms ✓ | 592.6 ms ✓ | 588.4 ms ✓ | 3.35 ms ✗ | 5.16 ms ✗ | 6.34 ms ✗ | 6.07 ms ✗ | 6.53 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **241.7 ms ✓** | — | — | — | — | — |


## rep_select  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 589.6 ms ✓ | 596.3 ms ✓ | 594.0 ms ✓ | 2.18 ms ✗ | 5.01 ms ✗ | 5.61 ms ✗ | 5.87 ms ✗ | 5.95 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **239.4 ms ✓** | — | — | — | — | — |


## rep_select  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 611.9 ms ✓ | 615.5 ms ✓ | 630.6 ms ✓ | 2.24 ms ✗ | 4.92 ms ✗ | 6.09 ms ✗ | 5.79 ms ✗ | 5.95 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **246.5 ms ✓** | — | — | — | — | — |


## ridge_pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | 1.75 ms ✓ | 2.14 ms ✓ | 1.83 ms ✓ | **1.75 ms ✓** | 4.62 ms ✓ | 6.20 ms ✓ | 5.59 ms ✓ | 5.56 ms ✓ | 2.19 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## ridge_pls  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **1.80 ms ✓** | 1.87 ms ✓ | 1.80 ms ✓ | 1.82 ms ✓ | 4.67 ms ✓ | 5.85 ms ✓ | 5.74 ms ✓ | 5.48 ms ✓ | 2.20 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## ridge_pls  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **1.83 ms ✓** | 1.86 ms ✓ | 1.87 ms ✓ | 1.85 ms ✓ | 4.60 ms ✓ | 5.61 ms ✓ | 6.40 ms ✓ | 5.92 ms ✓ | 2.20 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## robust_pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | 2.07 ms ✓ | **1.95 ms ✓** | 1.95 ms ✓ | 2.15 ms ✓ | 4.38 ms ✓ | 5.55 ms ✓ | 6.62 ms ✓ | 5.77 ms ✓ | — | — | — | — | — | — | — | — | 17.9 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## robust_pls  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | 2.02 ms ✓ | 2.02 ms ✓ | **2.00 ms ✓** | 2.10 ms ✓ | 5.43 ms ✓ | 6.63 ms ✓ | 5.99 ms ✓ | 5.84 ms ✓ | — | — | — | — | — | — | — | — | 17.0 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## robust_pls  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | 2.05 ms ✓ | 2.07 ms ✓ | **1.98 ms ✓** | 2.25 ms ✓ | 4.94 ms ✓ | 7.07 ms ✓ | 5.68 ms ✓ | 5.59 ms ✓ | — | — | — | — | — | — | — | — | 17.5 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## rosa  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | 7.07 ms ✓ | 2.04 ms ✓ | 2.41 ms ✓ | 2.56 ms ✓ | 5.46 ms ✓ | 6.38 ms ✓ | 6.68 ms ✓ | 6.80 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 2.9 s — | — | **1.43 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## rosa  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **1.24 ms ✓** | 1.83 ms ✓ | 1.36 ms ✓ | 1.42 ms ✓ | 3.82 ms ✓ | 4.80 ms ✓ | 4.86 ms ✓ | 5.17 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.8 s — | — | 6.48 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## rosa  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **1.33 ms ✓** | 2.05 ms ✓ | 1.99 ms ✓ | 1.42 ms ✓ | 4.13 ms ✓ | 4.17 ms ✓ | 15.6 ms ✓ | 23.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 2.2 s — | — | 6.60 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## scars_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 7.65 ms ✓ | 6.52 ms ✓ | 10.2 ms ✓ | 4.14 ms ✗ | 4.85 ms ✗ | 5.55 ms ✗ | 5.72 ms ✗ | 5.92 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **3.61 ms ✓** | — | — | — |


## scars_select  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 3.71 ms ✓ | **3.59 ms ✓** | 3.88 ms ✓ | 2.29 ms ✗ | 5.06 ms ✗ | 7.16 ms ✗ | 6.17 ms ✗ | 6.22 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 3.67 ms ✓ | — | — | — |


## scars_select  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **3.66 ms ✓** | 3.73 ms ✓ | 3.91 ms ✓ | 2.22 ms ✗ | 4.89 ms ✗ | 14.1 ms ✗ | 12.6 ms ✗ | 17.6 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 9.38 ms ✓ | — | — | — |


## shaving_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 1.0 s ✓ | 411.6 ms ✓ | 440.9 ms ✓ | 2.32 ms ✗ | 5.59 ms ✗ | 7.94 ms ✗ | 6.29 ms ✗ | 6.31 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **38.2 ms ✓** | — | — | — | — | — |


## shaving_select  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 424.1 ms ✓ | 773.6 ms ✓ | 445.2 ms ✓ | 2.24 ms ✗ | 5.24 ms ✗ | 5.90 ms ✗ | 5.99 ms ✗ | 6.06 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **94.0 ms ✓** | — | — | — | — | — |


## shaving_select  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 454.0 ms ✓ | 1.0 s ✓ | 443.2 ms ✓ | 2.26 ms ✗ | 5.07 ms ✗ | 5.85 ms ✗ | 5.97 ms ✗ | 5.96 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **35.1 ms ✓** | — | — | — | — | — |


## sipls_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 2.52 ms — | 3.21 ms ✓ | 3.18 ms ✓ | **2.82 ms ✓** | 7.86 ms ✓ | 8.15 ms ✓ | 8.12 ms ✓ | 8.80 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## sipls_select  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 2.55 ms — | **2.52 ms ✓** | 2.97 ms ✓ | 3.11 ms ✓ | 6.18 ms ✓ | 7.15 ms ✓ | 7.56 ms ✓ | 7.20 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## sipls_select  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 2.55 ms — | **2.54 ms ✓** | 2.57 ms ✓ | 2.73 ms ✓ | 6.76 ms ✓ | 7.43 ms ✓ | 13.1 ms ✓ | 18.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## so_pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | 1.62 ms ✓ | **1.62 ms ✓** | 1.67 ms ✓ | 1.77 ms ✓ | 3.26 ms ✓ | 3.99 ms ✓ | 4.11 ms ✓ | 4.00 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 981.7 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## so_pls  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | 1.66 ms ✓ | **1.63 ms ✓** | 1.69 ms ✓ | 2.44 ms ✓ | 22.4 ms ✓ | 31.5 ms ✓ | 34.4 ms ✓ | 31.4 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.5 s ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## so_pls  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | 7.16 ms ✓ | **3.23 ms ✓** | 11.3 ms ✓ | 11.9 ms ✓ | 11.3 ms ✓ | 25.9 ms ✓ | 9.15 ms ✓ | 10.3 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 4.0 s ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## spa_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 499.6 ms ✓ | 536.6 ms ✓ | 515.0 ms ✓ | 1.97 ms ✗ | 4.72 ms ✗ | 6.23 ms ✗ | 5.75 ms ✗ | 5.86 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **104.0 ms ✓** | — | — | — | — | — |


## spa_select  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 819.9 ms ✓ | 554.9 ms ✓ | 530.6 ms ✓ | 1.90 ms ✗ | 5.33 ms ✗ | 5.81 ms ✗ | 6.22 ms ✗ | 10.9 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **164.5 ms ✓** | — | — | — | — | — |


## spa_select  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 557.0 ms ✓ | 586.9 ms ✓ | 514.9 ms ✓ | 1.89 ms ✗ | 4.70 ms ✗ | 5.57 ms ✗ | 5.17 ms ✗ | 5.52 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **179.5 ms ✓** | — | — | — | — | — |


## sparse_pls_da  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | 7.43 ms ✓ | 11.1 ms ✓ | 4.50 ms ✓ | 2.45 ms ✗ | 7.20 ms ✓ | 8.68 ms ✓ | 8.53 ms ✓ | 8.43 ms ✓ | — | — | — | — | — | — | 82.9 ms — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **3.58 ms ✓** | — | — | — | — | — | — | — | — |


## sparse_pls_da  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | 12.7 ms ✓ | 13.4 ms ✓ | **8.19 ms ✓** | 5.55 ms ✗ | 23.8 ms ✓ | 31.7 ms ✓ | 15.7 ms ✓ | 37.4 ms ✓ | — | — | — | — | — | — | 117.9 ms — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 14.4 ms ✓ | — | — | — | — | — | — | — | — |


## sparse_pls_da  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | 7.83 ms ✓ | 10.6 ms ✓ | 7.59 ms ✓ | 3.94 ms ✗ | 12.2 ms ✓ | 47.2 ms ✓ | 32.6 ms ✓ | 26.4 ms ✓ | — | — | — | — | — | — | 91.2 ms — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **6.94 ms ✓** | — | — | — | — | — | — | — | — |


## sparse_simpls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | 1.89 ms ✓ | **1.82 ms ✓** | 1.93 ms ✓ | 1.88 ms ✓ | 4.60 ms ✓ | 5.64 ms ✓ | 5.36 ms ✓ | 6.30 ms ✓ | — | — | — | — | — | 3.55 ms ✓ | 13.4 ms — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## sparse_simpls  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | 1.93 ms ✓ | 1.90 ms ✓ | **1.87 ms ✓** | 1.92 ms ✓ | 4.97 ms ✓ | 6.05 ms ✓ | 5.77 ms ✓ | 6.32 ms ✓ | — | — | — | — | — | 3.54 ms ✓ | 12.8 ms — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## sparse_simpls  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **1.91 ms ✓** | 1.96 ms ✓ | 1.98 ms ✓ | 1.95 ms ✓ | 4.83 ms ✓ | 5.49 ms ✓ | 5.74 ms ✓ | 6.00 ms ✓ | — | — | — | — | — | 3.58 ms ✓ | 12.0 ms — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## st_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 403.7 ms ✓ | 409.2 ms ✓ | 418.3 ms ✓ | 1.82 ms ✗ | 6.05 ms ✗ | 5.27 ms ✗ | 5.46 ms ✗ | 5.45 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **17.8 ms ✓** | — | — | — | — | — |


## st_select  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 399.6 ms ✓ | 401.0 ms ✓ | 400.5 ms ✓ | 1.95 ms ✗ | 4.89 ms ✗ | 6.39 ms ✗ | 5.32 ms ✗ | 8.78 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **17.5 ms ✓** | — | — | — | — | — |


## st_select  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 429.2 ms ✓ | 416.4 ms ✓ | 435.3 ms ✓ | 1.78 ms ✗ | 4.31 ms ✗ | 6.27 ms ✗ | 5.49 ms ✗ | 6.77 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **19.4 ms ✓** | — | — | — | — | — |


## stability_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 690.4 ms ✓ | 820.9 ms ✓ | 1.7 s ✓ | 3.70 ms ✗ | 15.9 ms ✗ | 6.82 ms ✗ | 8.38 ms ✗ | 8.71 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **492.8 ms ✓** | — | — | — | — | — |


## stability_select  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 727.7 ms ✓ | 1.3 s ✓ | 698.1 ms ✓ | 1.79 ms ✗ | 6.01 ms ✗ | 5.91 ms ✗ | 8.89 ms ✗ | 6.92 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **259.7 ms ✓** | — | — | — | — | — |


## stability_select  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 727.3 ms ✓ | 685.8 ms ✓ | 1.5 s ✓ | 1.92 ms ✗ | 5.39 ms ✗ | 6.52 ms ✗ | 6.46 ms ✗ | 7.32 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **230.9 ms ✓** | — | — | — | — | — |


## t2_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 1.67 ms — | **1.67 ms ✓** | 1.69 ms ✓ | 1.83 ms ✓ | 4.21 ms ✓ | 6.34 ms ✓ | 5.80 ms ✓ | 6.75 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 37.0 ms ✓ | — | — | — | — | — |


## t2_select  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 1.67 ms — | **1.69 ms ✓** | 1.69 ms ✓ | 1.89 ms ✓ | 5.03 ms ✓ | 5.50 ms ✓ | 5.34 ms ✓ | 5.35 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 35.9 ms ✓ | — | — | — | — | — |


## t2_select  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 1.66 ms — | 1.80 ms ✓ | **1.76 ms ✓** | 1.83 ms ✓ | 4.63 ms ✓ | 5.24 ms ✓ | 5.58 ms ✓ | 5.78 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 35.4 ms ✓ | — | — | — | — | — |


## uve_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 1.5 s ✓ | 1.5 s ✓ | 1.7 s ✓ | 3.28 ms ✗ | 10.3 ms ✗ | 7.89 ms ✗ | 6.77 ms ✗ | 6.20 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **222.0 ms ✓** | — | — | — | — | — |


## uve_select  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 607.0 ms ✓ | 680.0 ms ✓ | 640.4 ms ✓ | 1.81 ms ✗ | 4.61 ms ✗ | 5.51 ms ✗ | 5.70 ms ✗ | 5.45 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **397.3 ms ✓** | — | — | — | — | — |


## uve_select  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 737.2 ms ✓ | 1.5 s ✓ | 1.5 s ✓ | 2.82 ms ✗ | 9.76 ms ✗ | 9.70 ms ✗ | 6.21 ms ✗ | 5.84 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **230.2 ms ✓** | — | — | — | — | — |


## variable_select_coef  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **1.56 ms ✓** | 1.56 ms ✓ | 1.61 ms ✓ | 2.87 ms ✓ | 15.0 ms ✓ | 9.19 ms ✓ | 17.2 ms ✓ | 18.1 ms ✓ | — | 50.9 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## variable_select_coef  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 11.3 ms ✓ | 6.96 ms ✓ | **6.88 ms ✓** | 7.62 ms ✓ | 16.9 ms ✓ | 18.4 ms ✓ | 17.5 ms ✓ | 9.25 ms ✓ | — | 16.3 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## variable_select_coef  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 1.87 ms ✓ | 2.01 ms ✓ | 2.60 ms ✓ | **1.86 ms ✓** | 5.27 ms ✓ | 6.03 ms ✓ | 6.23 ms ✓ | 6.17 ms ✓ | — | 16.5 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## variable_select_sr  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 542.8 ms ✓ | 1.5 s ✓ | 1.2 s ✓ | 8.28 ms ✗ | 24.1 ms ✗ | 11.2 ms ✗ | 31.6 ms ✗ | 22.3 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **18.9 ms ✓** | — | — | — | — | — |


## variable_select_sr  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 780.0 ms ✓ | 471.2 ms ✓ | 377.8 ms ✓ | 2.81 ms ✗ | 20.2 ms ✗ | 7.53 ms ✗ | 8.23 ms ✗ | 8.89 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **11.4 ms ✓** | — | — | — | — | — |


## variable_select_sr  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 692.5 ms ✓ | 464.1 ms ✓ | 1.3 s ✓ | 5.57 ms ✗ | 6.01 ms ✗ | 8.76 ms ✗ | 6.89 ms ✗ | 6.96 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **12.7 ms ✓** | — | — | — | — | — |


## variable_select_vip  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 2.55 ms ✓ | 2.70 ms ✓ | 2.47 ms ✓ | **1.97 ms ✓** | 6.19 ms ✓ | 6.51 ms ✓ | 5.86 ms ✓ | 6.61 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 16.2 ms ✓ | — | — | — | — | — |


## variable_select_vip  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 8.22 ms ✓ | 7.00 ms ✓ | 7.47 ms ✓ | **2.74 ms ✓** | 25.1 ms ✓ | 11.1 ms ✓ | 17.2 ms ✓ | 9.03 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 18.5 ms ✓ | — | — | — | — | — |


## variable_select_vip  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 4.22 ms ✓ | 4.46 ms ✓ | 4.15 ms ✓ | **3.66 ms ✓** | 5.24 ms ✓ | 6.31 ms ✓ | 6.29 ms ✓ | 6.76 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 12.5 ms ✓ | — | — | — | — | — |


## vip_spa_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 80×40 | — | — | — | **1.34 ms ✓** | 2.19 ms ✓ | 2.80 ms ✓ | 2.83 ms ✓ | 2.87 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## vip_spa_select  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 80×40 | — | — | — | **0.86 ms ✓** | 2.00 ms ✓ | 3.05 ms ✓ | 3.05 ms ✓ | 2.84 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## vip_spa_select  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 80×40 | — | — | — | **0.85 ms ✓** | 1.92 ms ✓ | 2.79 ms ✓ | 3.11 ms ✓ | 2.95 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## vissa_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 80×25 | — | — | — | **10.8 ms ✓** | 11.2 ms ✓ | 11.1 ms ✓ | 21.4 ms ✓ | 49.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## vissa_select  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 80×25 | — | — | — | 13.0 ms ✓ | 13.9 ms ✓ | 14.6 ms ✓ | 11.4 ms ✓ | **11.4 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## vissa_select  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 80×25 | — | — | — | **8.85 ms ✓** | 11.8 ms ✓ | 12.5 ms ✓ | 11.2 ms ✓ | 13.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## weighted_pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | 1.93 ms ✓ | 1.82 ms ✓ | **1.77 ms ✓** | 1.92 ms ✓ | 4.57 ms ✓ | 5.88 ms ✓ | 5.23 ms ✓ | 5.95 ms ✓ | 2.12 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## weighted_pls  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | 1.81 ms ✓ | **1.78 ms ✓** | 1.87 ms ✓ | 1.84 ms ✓ | 5.06 ms ✓ | 5.37 ms ✓ | 6.02 ms ✓ | 5.52 ms ✓ | 2.23 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## weighted_pls  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **1.82 ms ✓** | 1.86 ms ✓ | 1.91 ms ✓ | 1.91 ms ✓ | 4.82 ms ✓ | 5.68 ms ✓ | 6.10 ms ✓ | 5.35 ms ✓ | 2.18 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## wvc_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 1.49 ms ✓ | **1.47 ms ✓** | 1.48 ms ✓ | 1.63 ms ✓ | 4.23 ms ✓ | 4.97 ms ✓ | 5.72 ms ✓ | 5.30 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 14.4 ms ✓ | — | — | — | — | — |


## wvc_select  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 1.55 ms ✓ | 1.57 ms ✓ | **1.53 ms ✓** | 1.65 ms ✓ | 4.24 ms ✓ | 4.92 ms ✓ | 5.28 ms ✓ | 5.18 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 13.0 ms ✓ | — | — | — | — | — |


## wvc_select  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **1.50 ms ✓** | 1.59 ms ✓ | 2.65 ms ✓ | 1.68 ms ✓ | 4.88 ms ✓ | 6.13 ms ✓ | 5.46 ms ✓ | 5.79 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 13.8 ms ✓ | — | — | — | — | — |


## wvc_threshold_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 336.1 ms ✓ | 344.6 ms ✓ | 344.7 ms ✓ | 1.67 ms ✗ | 4.07 ms ✗ | 4.66 ms ✗ | 5.51 ms ✗ | 4.59 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **13.0 ms ✓** | — | — | — | — | — |


## wvc_threshold_select  —  3 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 304.2 ms ✓ | 305.7 ms ✓ | 301.0 ms ✓ | 1.57 ms ✗ | 4.31 ms ✗ | 4.42 ms ✗ | 4.40 ms ✗ | 4.35 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **12.7 ms ✓** | — | — | — | — | — |


## wvc_threshold_select  —  10 threads

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 340.2 ms ✓ | 334.4 ms ✓ | 329.0 ms ✓ | 1.61 ms ✗ | 4.06 ms ✗ | 4.70 ms ✗ | 5.45 ms ✗ | 5.14 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **13.5 ms ✓** | — | — | — | — | — |


## Backend definitions

Each column in the per-algorithm tables above is one of the entries below. Columns are named `owner.language[.variant]`: `pls4all.*` are this project's own bindings; everything else is an external library shipped by its own maintainers.

| Column | Language | Tier | What it actually runs |
|---|---|---|---|
| `pls4all.cpp.blas+omp` | C++ | pls4all + BLAS + OpenMP | libn4m built with both `PLS4ALL_WITH_BLAS=ON` and `PLS4ALL_WITH_OPENMP=ON` — the recommended production config. |
| `pls4all.registry` | Python | pls4all canonical | `benchmarks.parity_timing.registry.MethodSpec.pls4all_fn` — the canonical per-method pls4all entry point |
| `pls4all.python` | Python | pls4all raw | `pls4all._methods.<algo>_fit(ctx, cfg, X, y, …)` — direct FFI binding |
| `pls4all.sklearn` | Python | pls4all idiomatic | `pls4all.sklearn.<Class>` — sklearn-style BaseEstimator with `.fit() / .predict()` |
| `pls4all.R` | R | pls4all raw | `pls4all_method(algo, X, y, ...)` — unified dispatcher (33 fits + 24 selectors + 4 diagnostics) |
| `pls4all.R.formula` | R | pls4all idiomatic | `pls(y ~ ., data)`, `cppls(...)`, `sparse_pls(...)`, … — base R formula+S3 wrappers |
| `pls4all.R.pls_compat` | R | pls4all pls-compatible | `plsr()` / `pcr()` for PLS/PCR/CPPLS, formula-built dispatcher path for the rest of the method matrix |
| `pls4all.R.mdatools_compat` | R | pls4all mdatools-compatible | `pls(x, y, ...)` for PLS/PCR/CPPLS, matrix dispatcher path for the rest of the method matrix |
| `ref.python_scikit_learn` | external | reference | registry-declared external reference library |
| `ref.r_pls` | external | reference | registry-declared external reference library |
| `ref.python_ikpls` | external | reference | registry-declared external reference library |
| `ref.r_mixomics` | external | reference | registry-declared external reference library |
| `ref.r_ropls` | external | reference | registry-declared external reference library |
| `ref.python_chun_keles_spls` | external | reference | registry-declared external reference library |
| `ref.r_spls` | external | reference | registry-declared external reference library |
| `ref.python_diplslib` | external | reference | registry-declared external reference library |
| `ref.r_chemometrics` | external | reference | registry-declared external reference library |
| `ref.python_stone_brooks_1990_py` | external | reference | registry-declared external reference library |
| `ref.r_jico` | external | reference | registry-declared external reference library |
| `ref.python_tensorly` | external | reference | registry-declared external reference library |
| `ref.r_kernlab_pls` | external | reference | registry-declared external reference library |
| `ref.r_omicspls` | external | reference | registry-declared external reference library |
| `ref.r_mdatools` | external | reference | registry-declared external reference library |
| `ref.r_multiblock` | external | reference | registry-declared external reference library |
| `ref.python_onpls` | external | reference | registry-declared external reference library |
| `ref.python_numpy` | external | reference | registry-declared external reference library |
| `ref.python_auswahl` | external | reference | registry-declared external reference library |
| `ref.python_pyswarms` | external | reference | registry-declared external reference library |
| `ref.r_mboost` | external | reference | registry-declared external reference library |
| `ref.r_plsrglm` | external | reference | registry-declared external reference library |
| `ref.r_base` | external | reference | registry-declared external reference library |
| `ref.r_softimpute` | external | reference | registry-declared external reference library |
| `ref.python_chun_keles_splsda` | external | reference | registry-declared external reference library |
| `ref.r_sgpls` | external | reference | registry-declared external reference library |
| `ref.python_nirs4all` | external | reference | registry-declared external reference library |
| `ref.r_plsvarsel` | external | reference | registry-declared external reference library |
| `ref.r_enpls` | external | reference | registry-declared external reference library |
| `ref.python_scars_numpy_port` | external | reference | registry-declared external reference library |
| `ref.r_pls_stats` | external | reference | registry-declared external reference library |
| `libPLS` | external | reference | registry-declared external reference library |
| `ref.python_iriv_numpy_port` | external | reference | registry-declared external reference library |


## Versions per backend

| Column | Versions |
|---|---|
| `pls4all.cpp.blas+omp` | `language=C++ (via ctypes bridge) (host Linux x86_64)`; `blas=libscipy_openblas 0.3.31.188.0 blas=1; libopenblas 0.3.33 blas=1; libomp ? openmp=1`; `libn4m=0.98.0+abi.1.9.0`; `numpy=2.4.6`; `registry_method=pls`; `timing_schema=adaptive-v1` |
| `pls4all.registry` | `language=Python 3.11.15`; `blas=libscipy_openblas 0.3.31.188.0 blas=1; libopenblas 0.3.33 blas=1; libomp ? openmp=1`; `pls4all=0.98.0`; `numpy=2.4.6`; `registry_method=pls`; `timing_schema=adaptive-v1` |
| `pls4all.python` | `language=Python 3.11.15`; `blas=libscipy_openblas 0.3.31.188.0 blas=1; libopenblas 0.3.33 blas=1; libomp ? openmp=1`; `pls4all=0.98.0`; `numpy=2.4.6`; `registry_method=pls`; `timing_schema=adaptive-v1` |
| `pls4all.sklearn` | `language=Python 3.11.15`; `blas=libscipy_openblas 0.3.31.188.0 blas=1; libopenblas 0.3.33 blas=1; libomp ? openmp=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `pls4all=0.98.0`; `numpy=2.4.6`; `sklearn_class=PLSRegression`; `timing_schema=adaptive-v1` |
| `pls4all.R` | `language=R 4.3.3`; `pls4all=0.98.0`; `registry_method=pls`; `blas=linked-BLAS`; `timing_schema=adaptive-v1` |
| `pls4all.R.formula` | `language=R 4.3.3`; `pls4all=0.98.0`; `registry_method=pls`; `formula_facade=True`; `blas=linked-BLAS`; `timing_schema=adaptive-v1` |
| `pls4all.R.pls_compat` | `language=R 4.3.3`; `pls4all=0.98.0`; `registry_method=pls`; `facade=pls_compat`; `formula_facade=True`; `blas=linked-BLAS`; `timing_schema=adaptive-v1` |
| `pls4all.R.mdatools_compat` | `language=R 4.3.3`; `pls4all=0.98.0`; `registry_method=pls`; `facade=mdatools_compat`; `matrix_facade=True`; `blas=linked-BLAS`; `timing_schema=adaptive-v1` |
| `ref.python_scikit_learn` | `language=Python 3.11.15`; `blas=libscipy_openblas 0.3.31.188.0 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `numpy=2.4.6`; `reference_id=python_scikit_learn`; `reference_library=scikit-learn`; `reference_version=1.8.0`; `timing_schema=adaptive-v1` |
| `ref.r_pls` | `language=R 4.3.3`; `pls=2.8.5`; `timing_schema=adaptive-v1` |
| `ref.python_ikpls` | `language=Python 3.11.15`; `blas=libscipy_openblas 0.3.31.188.0 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `numpy=2.4.6`; `reference_id=python_ikpls`; `reference_library=ikpls`; `reference_version=4.0.1.post1`; `timing_schema=adaptive-v1` |
| `ref.r_mixomics` | `language=R 4.3.3`; `mixOmics=6.26.0`; `timing_schema=adaptive-v1` |
| `ref.r_ropls` | `language=R 4.3.3`; `ropls=1.34.0`; `timing_schema=adaptive-v1` |
| `ref.python_chun_keles_spls` | `language=Python 3.11.15`; `blas=libscipy_openblas 0.3.31.188.0 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `numpy=2.4.6`; `reference_id=python_chun_keles_spls`; `reference_library=chun_keles_spls`; `reference_version=1.0`; `timing_schema=adaptive-v1` |
| `ref.r_spls` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.31.188.0 blas=1`; `numpy=2.4.6`; `reference_id=r_spls`; `reference_library=spls`; `reference_version=2.3.2`; `timing_driver=rscript-batch`; `timing_schema=adaptive-v1` |
| `ref.python_diplslib` | `language=Python 3.11.15`; `blas=libscipy_openblas 0.3.31.188.0 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `numpy=2.4.6`; `reference_id=python_diplslib`; `reference_library=diPLSlib`; `reference_version=2.5.0`; `timing_schema=adaptive-v1` |
| `ref.r_chemometrics` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.31.188.0 blas=1`; `numpy=2.4.6`; `reference_id=r_chemometrics`; `reference_library=chemometrics`; `reference_version=0.7.x`; `timing_driver=rscript-batch`; `timing_schema=adaptive-v1` |
| `ref.python_stone_brooks_1990_py` | `language=Python 3.11.15`; `blas=libscipy_openblas 0.3.31.188.0 blas=1`; `numpy=2.4.6`; `reference_id=python_stone_brooks_1990_py`; `reference_library=stone-brooks-1990-py`; `reference_version=1.0`; `timing_schema=adaptive-v1` |
| `ref.r_jico` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.31.188.0 blas=1`; `numpy=2.4.6`; `reference_id=r_jico`; `reference_library=JICO`; `reference_version=0.1`; `timing_driver=rscript-batch`; `timing_schema=adaptive-v1` |
| `ref.python_tensorly` | `language=Python 3.11.15`; `blas=libscipy_openblas 0.3.31.188.0 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `numpy=2.4.6`; `reference_id=python_tensorly`; `reference_library=tensorly`; `reference_version=0.9.0`; `timing_schema=adaptive-v1` |
| `ref.r_kernlab_pls` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.31.188.0 blas=1`; `numpy=2.4.6`; `reference_id=r_kernlab_pls`; `reference_library=kernlab+pls`; `reference_version=0.9.33+2.8.5`; `timing_driver=rscript-batch`; `timing_schema=adaptive-v1` |
| `ref.r_omicspls` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.31.188.0 blas=1`; `numpy=2.4.6`; `reference_id=r_omicspls`; `reference_library=OmicsPLS`; `reference_version=2.1.0`; `timing_driver=rscript-batch`; `timing_schema=adaptive-v1` |
| `ref.r_mdatools` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.31.188.0 blas=1`; `numpy=2.4.6`; `reference_id=r_mdatools`; `reference_library=mdatools`; `reference_version=0.15.0`; `timing_driver=rscript-batch`; `timing_schema=adaptive-v1` |
| `ref.r_multiblock` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.31.188.0 blas=1`; `numpy=2.4.6`; `reference_id=r_multiblock`; `reference_library=multiblock`; `reference_version=0.8.10`; `timing_schema=adaptive-v1` |
| `ref.python_onpls` | `language=Python 3.11.15`; `blas=libscipy_openblas 0.3.31.188.0 blas=1`; `numpy=2.4.6`; `reference_id=python_onpls`; `reference_library=OnPLS`; `reference_version=github tomlof/OnPLS`; `timing_schema=adaptive-v1` |
| `ref.python_numpy` | `language=Python 3.11.15`; `blas=libscipy_openblas 0.3.31.188.0 blas=1`; `numpy=2.4.6`; `reference_id=python_numpy`; `reference_library=numpy`; `reference_version=2.4.6`; `timing_schema=adaptive-v1` |
| `ref.python_auswahl` | — |
| `ref.python_pyswarms` | `language=Python 3.11.15`; `blas=libscipy_openblas 0.3.31.188.0 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `numpy=2.4.6`; `reference_id=python_pyswarms`; `reference_library=pyswarms`; `reference_version=1.3.0`; `timing_schema=adaptive-v1` |
| `ref.r_mboost` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.31.188.0 blas=1`; `numpy=2.4.6`; `reference_id=r_mboost`; `reference_library=mboost`; `reference_version=2.9-11`; `timing_driver=rscript-batch`; `timing_schema=adaptive-v1` |
| `ref.r_plsrglm` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.31.188.0 blas=1`; `numpy=2.4.6`; `reference_id=r_plsrglm`; `reference_library=plsRglm`; `reference_version=1.5.1`; `timing_driver=rscript-batch`; `timing_schema=adaptive-v1` |
| `ref.r_base` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.31.188.0 blas=1`; `numpy=2.4.6`; `reference_id=r_base`; `reference_library=base`; `reference_version=R 4.3.3`; `timing_driver=rscript-batch`; `timing_schema=adaptive-v1` |
| `ref.r_softimpute` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.31.188.0 blas=1`; `numpy=2.4.6`; `reference_id=r_softimpute`; `reference_library=softImpute`; `reference_version=1.4-1`; `timing_driver=rscript-batch`; `timing_schema=adaptive-v1` |
| `ref.python_chun_keles_splsda` | `language=Python 3.11.15`; `blas=libscipy_openblas 0.3.31.188.0 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `numpy=2.4.6`; `reference_id=python_chun_keles_splsda`; `reference_library=chun_keles_splsda`; `reference_version=1.0`; `timing_schema=adaptive-v1` |
| `ref.r_sgpls` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.31.188.0 blas=1`; `numpy=2.4.6`; `reference_id=r_sgpls`; `reference_library=sgPLS`; `reference_version=1.8.1`; `timing_driver=rscript-batch`; `timing_schema=adaptive-v1` |
| `ref.python_nirs4all` | `language=Python 3.11.15`; `blas=libscipy_openblas 0.3.31.188.0 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `numpy=2.4.6`; `reference_id=python_nirs4all`; `reference_library=nirs4all`; `reference_version=in-tree`; `timing_schema=adaptive-v1` |
| `ref.r_plsvarsel` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.31.188.0 blas=1`; `numpy=2.4.6`; `reference_id=r_plsvarsel`; `reference_library=plsVarSel`; `reference_version=0.10.0`; `timing_driver=rscript-batch`; `timing_schema=adaptive-v1` |
| `ref.r_enpls` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.31.188.0 blas=1`; `numpy=2.4.6`; `reference_id=r_enpls`; `reference_library=enpls`; `reference_version=6.1`; `timing_driver=rscript-batch`; `timing_schema=adaptive-v1` |
| `ref.python_scars_numpy_port` | `language=Python 3.11.15`; `blas=libscipy_openblas 0.3.31.188.0 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `numpy=2.4.6`; `reference_id=python_scars_numpy_port`; `reference_library=scars_numpy_port`; `reference_version=1.0.0`; `timing_schema=adaptive-v1` |
| `ref.r_pls_stats` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.31.188.0 blas=1`; `numpy=2.4.6`; `reference_id=r_pls_stats`; `reference_library=pls+stats`; `reference_version=R 4.3.3`; `timing_driver=rscript-batch`; `timing_schema=adaptive-v1` |
| `libPLS` | `language=matlab (host Linux x86_64)`; `blas=libscipy_openblas 0.3.31.188.0 blas=1`; `numpy=2.4.6`; `reference_id=matlab_libpls`; `reference_library=libPLS`; `reference_version=1.95`; `timing_schema=adaptive-v1` |
| `ref.python_iriv_numpy_port` | `language=Python 3.11.15`; `blas=libscipy_openblas 0.3.31.188.0 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `numpy=2.4.6`; `reference_id=python_iriv_numpy_port`; `reference_library=iriv_numpy_port`; `reference_version=1.0.0`; `timing_schema=adaptive-v1` |

## Methodology

- Gate 1 reference: `cpp` cell at 1 thread (libn4m via ctypes), or `python_tier1` when `cpp` is unavailable for an algorithm
- Gate 2 reference: the registry-declared canonical external reference for the method
- Gate 1 tolerance: 1e-6 max-abs-diff; Gate 2 tolerance: `MethodSpec.rmse_rel_tol` (also emitted as `reference_parity_tolerance` by newer CSVs)
- All backends read the same orchestrator-generated CSV dataset (`benchmarks/cross_binding/data/data_<n>x<p>_seed<seed>.csv`)
- Adaptive timing: warmstart run #1; if there is any subsequent run, run #1 is excluded from the score. 2 total runs report run #2 alone; 3 total runs report the mean of runs #2-#3; 10/20/40 total runs report the median after the warmstart.
- Per-cell timeout guard: 24 h
- Thread control via `OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = BLIS_NUM_THREADS` set in the subprocess env, plus `Context.num_threads` for Python pls4all and `maxNumCompThreads()` for Octave
- pls4all libn4m build: `build/blas-omp/cpp/src/libn4m.so` (BLAS + OpenMP enabled)
