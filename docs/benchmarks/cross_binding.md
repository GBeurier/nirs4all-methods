# Cross-binding benchmark — parity + timing (1 thread)

Single-thread numbers — the cleanest cross-language comparison. **Most external libraries don't honour OMP_NUM_THREADS at the algorithm level** (sklearn, pls::plsr, plsregress, ikpls all run the PLS algo serially even when BLAS is multi-threaded), so a multi-thread comparison would compare pls4all's OpenMP path against everyone else's single-threaded loop. That's a real, useful number — see the [thread sweep page](cross_binding_threads.md) — but this main page sticks to 1 thread for honest apples-to-apples timing.

_Generated: 2026-05-30 10:08:14 UTC_
_CSV: `tmp2610y4wz.csv` (810 cells)_


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

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 4.20 ms ✓ | 4.30 ms ✓ | 4.16 ms ✓ | **4.15 ms ✓** | 7.84 ms ✓ | 9.43 ms ✓ | 9.87 ms ✓ | 17.6 ms ✓ | 25.1 ms ✗ | 35.5 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 41.9 ms ✓ | — | — | — | — | — | — |


## aom_preprocess  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 1.85 ms ✓ | 2.52 ms ✓ | 1.76 ms ✓ | **1.75 ms ✓** | 6.07 ms ✓ | 7.96 ms ✓ | 7.33 ms ✓ | 6.56 ms ✓ | 2.74 ms ✗ | 3.37 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 2.13 ms ✓ | — | — | — | — | — | — |


## approximate_press  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | 28.4 ms ✓ | 28.0 ms ✓ | 28.1 ms ✓ | 28.1 ms ✓ | **24.7 ms ✓** | 25.5 ms ✓ | 25.7 ms ✓ | 25.3 ms ✓ | 29.9 ms ✗ | 30.1 ms ✗ | — | 69.5 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## bagging_pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | 9.82 ms ✓ | 10.2 ms ✓ | **9.30 ms ✓** | 1.69 ms ✗ | 4.62 ms ✗ | 6.41 ms ✗ | 4.93 ms ✗ | 5.11 ms ✗ | 2.86 ms ✗ | 3.73 ms ✗ | 11.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## bipls_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 3.83 ms ✓ | 4.36 ms ✓ | **3.60 ms ✓** | 3.69 ms ✓ | 8.26 ms ✓ | 11.7 ms ✓ | 8.83 ms ✓ | 9.90 ms ✓ | 4.33 ms ✓ | 11.7 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 365.6 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## boosting_pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **1.23 ms ✓** | 1.23 ms ✓ | 1.52 ms ✓ | 2.07 ms ✗ | 5.05 ms ✗ | 4.95 ms ✗ | 6.27 ms ✗ | 6.92 ms ✗ | 2.48 ms ✗ | 3.08 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 18.5 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — |


## bve_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 510.0 ms ✓ | 443.3 ms ✓ | 563.8 ms ✓ | 42.4 ms ✗ | 58.7 ms ✗ | 66.5 ms ✗ | 141.0 ms ✗ | 152.1 ms ✗ | 110.8 ms ✗ | 48.6 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **46.9 ms ✓** | — | — | — | — | — |


## cars_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 780.7 ms ✓ | 752.8 ms ✓ | 772.6 ms ✓ | — | 13.3 ms ✗ | 10.2 ms ✗ | 13.3 ms ✗ | 16.4 ms ✗ | 9.44 ms ✗ | 6.89 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **188.9 ms ✓** | — | — | — | — |


## continuum_regression  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | 2.73 ms ✓ | 2.76 ms ✓ | 2.63 ms ✓ | 2.77 ms ✓ | 5.55 ms ✓ | 6.02 ms ✓ | 6.62 ms ✓ | 6.62 ms ✓ | 3.73 ms ✗ | 4.11 ms ✗ | — | — | — | — | — | — | — | — | — | **2.27 ms ✓** | 31.6 ms — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## cppls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | 1.88 ms ✓ | **1.79 ms ✓** | 1.87 ms ✓ | 1.97 ms ✓ | 4.57 ms ✓ | 5.65 ms ✓ | 10.6 ms ✗ | 6.07 ms ✗ | 2.81 ms ✗ | 2.82 ms ✗ | — | 9.91 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## di_pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | 2.58 ms ✓ | **2.57 ms ✓** | 2.74 ms ✓ | 2.82 ms ✓ | 9.69 ms ✓ | 12.4 ms ✓ | 12.4 ms ✓ | 12.8 ms ✓ | 6.23 ms ✗ | 6.86 ms ✗ | — | — | — | — | — | — | — | 3.78 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## ds  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **1.64 ms ✓** | 1.73 ms ✓ | 2.09 ms ✓ | 2.27 ms ✓ | 10.2 ms ✓ | 14.0 ms ✓ | 12.6 ms ✓ | 11.4 ms ✓ | 6.32 ms ✗ | 6.63 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 31.0 ms ✓ | — | — | — | — | — | — | — | — | — | — |


## ecr  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | 2.16 ms ✓ | **2.15 ms ✓** | 2.24 ms ✓ | 2.37 ms ✓ | 5.88 ms ✓ | 6.96 ms ✓ | 8.61 ms ✓ | 7.41 ms ✓ | 3.34 ms ✗ | 3.82 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 62.4 ms ✓ | — |


## emcuve_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 1.3 s ✓ | 1.3 s ✓ | 1.3 s ✓ | **1.98 ms ✓** | 4.51 ms ✓ | 5.44 ms ✓ | 5.57 ms ✓ | 5.25 ms ✓ | 2.67 ms ✗ | 2.92 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 922.4 ms ✓ | — | — | — | — | — |


## fused_sparse_pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | 1.93 ms — | **1.86 ms ✓** | 1.95 ms ✓ | 2.13 ms ✓ | 6.13 ms ✓ | 7.70 ms ✓ | 7.80 ms ✓ | 7.55 ms ✓ | 3.13 ms ✗ | 3.52 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## ga_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 607.6 ms ✓ | 625.8 ms ✓ | 756.5 ms ✓ | 3.94 ms ✗ | 7.38 ms ✗ | 8.59 ms ✗ | 9.76 ms ✗ | 16.0 ms ✗ | 4.75 ms ✗ | 5.24 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **218.7 ms ✓** | — | — | — | — | — |


## gpr_pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 120×25 | 5.91 ms ✓ | 6.75 ms ✓ | 6.17 ms ✓ | 2.75 ms ✓ | 20.9 ms ✓ | 12.6 ms ✓ | 14.2 ms ✓ | 8.10 ms ✓ | 3.55 ms ✗ | 9.81 ms ✗ | **2.25 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## group_sparse_pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | 8.36 ms ✓ | 4.69 ms ✓ | 7.86 ms ✓ | 3.79 ms ✗ | 9.70 ms ✗ | 11.0 ms ✗ | 11.0 ms ✗ | 10.2 ms ✗ | 4.32 ms ✗ | 6.93 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **2.98 ms ✓** | — | — | — | — | — | — | — | 33.7 ms — | — | — | — | — | — | — | — |


## interval_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 1.5 s ✓ | 2.0 s ✓ | 1.5 s ✓ | 3.75 ms ✗ | 10.9 ms ✗ | 11.7 ms ✗ | 10.3 ms ✗ | 13.9 ms ✗ | 3.88 ms ✗ | 5.05 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **632.3 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## ipw_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 360.4 ms ✓ | 358.1 ms ✓ | 366.9 ms ✓ | 1.81 ms ✗ | 4.86 ms ✗ | 5.51 ms ✗ | 5.86 ms ✗ | 5.15 ms ✗ | 2.67 ms ✗ | 3.32 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **22.0 ms ✓** | — | — | — | — | — |


## irf_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 120×30 | — | — | — | **2.02 ms ✓** | 2.92 ms ✓ | 3.44 ms ✓ | 3.91 ms ✓ | 3.37 ms ✓ | 1.90 ms ✗ | 2.27 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## iriv_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 80×25 | **268.0 ms ✓** | 269.9 ms ✓ | 270.2 ms ✓ | 25.3 ms ✗ | 29.4 ms ✗ | 30.0 ms ✗ | 30.5 ms ✗ | 30.8 ms ✗ | 25.0 ms ✗ | 25.8 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 274.8 ms ✓ |


## kernel_pls_rbf  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | 3.07 ms ✓ | **3.02 ms ✓** | 3.23 ms ✓ | 3.23 ms ✓ | 6.34 ms ✓ | 7.05 ms ✓ | 7.21 ms ✓ | 6.83 ms ✓ | 4.33 ms ✗ | 4.75 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | 28.6 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## lw_pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 25.6 ms ✓ | **13.3 ms ✓** | 37.2 ms ✓ | 9.19 ms ✗ | 20.3 ms ✗ | 23.2 ms ✗ | 23.2 ms ✗ | 24.6 ms ✗ | 8.59 ms ✗ | 13.8 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 23.9 ms ✓ | — | — | — | — | — | — |


## mb_pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×60 | **2.37 ms ✓** | 2.73 ms ✓ | 2.85 ms ✓ | 2.59 ms ✓ | 7.82 ms ✓ | 10.9 ms ✓ | 10.2 ms ✓ | 11.5 ms ✓ | 3.64 ms ✗ | 4.29 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 2.99 ms ✓ | — | — | — | — | — | — |


## mir_pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | 1.19 ms — | **1.21 ms ✓** | 1.27 ms ✓ | 1.47 ms ✓ | 3.98 ms ✓ | 4.72 ms ✓ | 5.04 ms ✓ | 5.11 ms ✓ | 2.05 ms ✗ | 2.74 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## missing_aware_nipals  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | 1.27 ms ✓ | **1.24 ms ✓** | 1.25 ms ✓ | 1.39 ms ✓ | 4.25 ms ✓ | 4.88 ms ✓ | 4.85 ms ✓ | 5.41 ms ✓ | 2.15 ms ✗ | 2.56 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 13.6 ms ✓ | — | — | — | — | — | — | — | — | — |


## n_pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×48 | 19.5 ms ✓ | 19.3 ms ✓ | **19.0 ms ✓** | 2.03 ms ✗ | 4.51 ms ✗ | 5.67 ms ✗ | 5.68 ms ✗ | 5.37 ms ✗ | 2.88 ms ✗ | 3.22 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | 19.3 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## o2pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | 1.21 ms ✓ | 1.20 ms ✓ | **1.19 ms ✓** | 1.37 ms ✓ | 2.69 ms ✓ | 3.44 ms ✓ | 3.58 ms ✓ | 3.42 ms ✓ | 1.98 ms ✗ | 2.46 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | 7.21 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## on_pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | 6.20 ms ✓ | 7.71 ms ✓ | 3.02 ms ✓ | **2.69 ms ✓** | — | — | — | — | 3.98 ms ⚠ | 4.57 ms ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 12.2 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## one_se_rule  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | 2.10 ms ✓ | **2.07 ms ✓** | 2.08 ms ✓ | 1.13 ms ✗ | 3.25 ms ✗ | 3.87 ms ✗ | 4.01 ms ✗ | 3.96 ms ✗ | 1.95 ms ✗ | 2.27 ms ✗ | — | 11.6 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## opls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **1.87 ms ✓** | 1.93 ms ✓ | 1.88 ms ✓ | 2.12 ms ✓ | 4.35 ms ✓ | 5.13 ms ✓ | 5.40 ms ✓ | 5.23 ms ✓ | 2.85 ms ✗ | 3.04 ms ✗ | — | — | — | — | 15.4 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pcr  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | 2.92 ms ✓ | 2.84 ms ✓ | 2.95 ms ✓ | **2.68 ms ✓** | 5.15 ms ✓ | 6.41 ms ✓ | 11.8 ms ✓ | 6.21 ms ✓ | 3.55 ms ✗ | 3.83 ms ✗ | 2.71 ms ✓ | 7.99 ms — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pds  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **1.37 ms ✓** | 1.55 ms ✓ | 1.44 ms ✓ | 1.86 ms ✓ | 10.6 ms ✓ | 10.1 ms ✓ | 14.2 ms ✓ | 13.7 ms ✓ | 5.67 ms ✗ | 6.35 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 43.1 ms ✓ | — | — | — | — | — | — | — | — | — | — |


## pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | 1.70 ms ✓ | 1.73 ms ✓ | **1.69 ms ✓** | 1.97 ms ✓ | 4.79 ms ✓ | 5.26 ms ✓ | 9.99 ms ✓ | 5.92 ms ✓ | 2.82 ms ✗ | 3.17 ms ✗ | 2.16 ms ✓ | 8.01 ms — | 1.92 ms — | 9.72 ms — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_cox  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | 1.79 ms ✓ | 1.84 ms ✓ | **1.75 ms ✓** | 2.13 ms ✗ | 4.52 ms ✗ | 5.52 ms ✗ | 5.43 ms ✗ | 5.51 ms ✗ | 2.19 ms ✗ | 2.81 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 2.03 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_diagnostic_dmodx  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | 1.25 ms ✓ | 1.30 ms ✓ | **1.25 ms ✓** | 1.41 ms ✓ | 4.16 ms ✓ | 4.79 ms ✓ | 4.47 ms ✓ | 5.71 ms ✓ | 2.19 ms ✗ | 2.65 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 19.8 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_diagnostic_q  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | 1.22 ms ✓ | **1.19 ms ✓** | 1.19 ms ✓ | 1.30 ms ✓ | 3.10 ms ✓ | 3.57 ms ✓ | 3.47 ms ✓ | 3.96 ms ✓ | 1.90 ms ✗ | 2.28 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 13.8 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_diagnostic_t2  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **1.19 ms ✓** | 1.22 ms ✓ | 1.22 ms ✓ | 1.31 ms ✓ | 2.67 ms ✓ | 3.38 ms ✓ | 4.11 ms ✓ | 3.70 ms ✓ | 2.02 ms ✗ | 2.23 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 14.2 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_glm  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **3.57 ms ✓** | 6.25 ms ✓ | 7.99 ms ✓ | 2.74 ms ✗ | 7.97 ms ✗ | 22.7 ms ✗ | 18.4 ms ✗ | 25.8 ms ✗ | 4.42 ms ✗ | 8.59 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 240.6 ms ✓ | — | — | — | — | — | — | — | — | — | — | — |


## pls_lda  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **3.14 ms ✓** | 7.18 ms ✓ | 7.48 ms ✓ | — | — | — | — | — | — | — | 7.32 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_logistic  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **3.14 ms ✓** | 4.04 ms ✓ | 3.34 ms ✓ | — | — | — | — | — | — | — | 3.44 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_monitoring  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | 1.23 ms ✓ | 1.23 ms ✓ | **1.18 ms ✓** | 1.32 ms ✓ | 2.90 ms ✓ | 3.54 ms ✓ | 3.74 ms ✓ | 3.81 ms ✓ | 1.94 ms ✗ | 2.28 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 14.2 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_qda  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | 2.65 ms ✓ | **2.09 ms ✓** | 2.12 ms ✓ | 2.11 ms ✗ | 4.06 ms ✗ | 4.67 ms ✗ | 4.99 ms ✗ | 5.46 ms ✗ | 2.83 ms ✗ | 2.49 ms ✗ | 2.36 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pop_pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 6.24 ms ✓ | **5.68 ms ✓** | 7.23 ms ✓ | 6.51 ms ✓ | 12.0 ms ✓ | 12.5 ms ✓ | 13.2 ms ✓ | 24.7 ms ✓ | 10.2 ms ✗ | 29.0 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 47.8 ms ✓ | — | — | — | — | — | — |


## pso_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 80×25 | 219.8 ms ✓ | 384.2 ms ✓ | 219.3 ms ✓ | 4.05 ms ✗ | 6.11 ms ✗ | 6.29 ms ✗ | 7.12 ms ✗ | 7.33 ms ✗ | 3.65 ms ✗ | 3.75 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **163.8 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — |


## random_frog_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | — | — | — | **2.19 ms ✓** | 4.98 ms ✓ | 5.67 ms ✓ | 5.94 ms ✓ | 5.96 ms ✓ | 2.95 ms ✗ | 3.38 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## random_subspace_pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | 13.5 ms ✓ | 25.2 ms ✓ | 13.2 ms ✓ | 1.54 ms ✗ | 5.11 ms ✗ | 5.82 ms ✗ | 5.56 ms ✗ | 6.22 ms ✗ | 2.99 ms ✗ | 3.74 ms ✗ | **13.0 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## randomization_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 181.1 ms ✓ | 182.3 ms ✓ | 182.5 ms ✓ | **2.67 ms ✓** | 6.34 ms ✓ | 5.94 ms ✓ | 6.80 ms ✓ | 6.29 ms ✓ | 3.48 ms ✓ | 3.93 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 54.5 ms ✓ | — | — |


## recursive_pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | 2.92 ms ✓ | 2.82 ms ✓ | **2.79 ms ✓** | 3.10 ms ✓ | 5.74 ms ✓ | 6.55 ms ✓ | 7.36 ms ✓ | 6.99 ms ✓ | 3.93 ms ✗ | 4.21 ms ✗ | 48.1 ms ✓ | 141.6 ms — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## rep_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 605.7 ms ✓ | 592.6 ms ✓ | 588.4 ms ✓ | 3.35 ms ✗ | 5.16 ms ✗ | 6.34 ms ✗ | 6.07 ms ✗ | 6.53 ms ✗ | 2.97 ms ✗ | 3.28 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **241.7 ms ✓** | — | — | — | — | — |


## ridge_pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | 1.75 ms ✓ | 2.14 ms ✓ | 1.83 ms ✓ | **1.75 ms ✓** | 4.62 ms ✓ | 6.20 ms ✓ | 5.59 ms ✓ | 5.56 ms ✓ | 2.85 ms ✗ | 2.78 ms ✗ | 2.19 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## robust_pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | 2.07 ms ✓ | **1.95 ms ✓** | 1.95 ms ✓ | 2.15 ms ✓ | 4.38 ms ✓ | 5.55 ms ✓ | 6.62 ms ✓ | 5.77 ms ✓ | 3.05 ms ✗ | 3.39 ms ✗ | — | — | — | — | — | — | — | — | 17.9 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## rosa  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | 7.07 ms ✓ | 2.04 ms ✓ | 2.41 ms ✓ | 2.56 ms ✓ | 5.46 ms ✓ | 6.38 ms ✓ | 6.68 ms ✓ | 6.80 ms ✓ | 2.07 ms ✗ | 2.38 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 2.9 s — | — | **1.43 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## scars_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 7.65 ms ✓ | 6.52 ms ✓ | 10.2 ms ✓ | 4.14 ms ✗ | 4.85 ms ✗ | 5.55 ms ✗ | 5.72 ms ✗ | 5.92 ms ✗ | 2.96 ms ✗ | 3.27 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **3.61 ms ✓** | — | — | — |


## shaving_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 1.0 s ✓ | 411.6 ms ✓ | 440.9 ms ✓ | 2.32 ms ✗ | 5.59 ms ✗ | 7.94 ms ✗ | 6.29 ms ✗ | 6.31 ms ✗ | 2.98 ms ✗ | 3.36 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **38.2 ms ✓** | — | — | — | — | — |


## sipls_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 2.52 ms — | 3.21 ms ✓ | 3.18 ms ✓ | **2.82 ms ✓** | 7.86 ms ✓ | 8.15 ms ✓ | 8.12 ms ✓ | 8.80 ms ✓ | 3.76 ms ✓ | 4.07 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## so_pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | 1.62 ms ✓ | **1.62 ms ✓** | 1.67 ms ✓ | 1.77 ms ✓ | 3.26 ms ✓ | 3.99 ms ✓ | 4.11 ms ✓ | 4.00 ms ✓ | 2.33 ms ✗ | 2.67 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 981.7 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## spa_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 499.6 ms ✓ | 536.6 ms ✓ | 515.0 ms ✓ | 1.97 ms ✗ | 4.72 ms ✗ | 6.23 ms ✗ | 5.75 ms ✗ | 5.86 ms ✗ | 2.78 ms ✗ | 3.26 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **104.0 ms ✓** | — | — | — | — | — |


## sparse_pls_da  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | 7.43 ms ✓ | 11.1 ms ✓ | 4.50 ms ✓ | 2.45 ms ✗ | 7.20 ms ✓ | 8.68 ms ✓ | 8.53 ms ✓ | 8.43 ms ✓ | 3.58 ms ✗ | 3.98 ms ✗ | — | — | — | — | — | — | 82.9 ms — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **3.58 ms ✓** | — | — | — | — | — | — | — | — |


## sparse_simpls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | 1.89 ms ✓ | **1.82 ms ✓** | 1.93 ms ✓ | 1.88 ms ✓ | 4.60 ms ✓ | 5.64 ms ✓ | 5.36 ms ✓ | 6.30 ms ✓ | 2.97 ms ✗ | 2.97 ms ✗ | — | — | — | — | — | 3.55 ms ✓ | 13.4 ms — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## st_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 403.7 ms ✓ | 409.2 ms ✓ | 418.3 ms ✓ | 1.82 ms ✗ | 6.05 ms ✗ | 5.27 ms ✗ | 5.46 ms ✗ | 5.45 ms ✗ | 2.49 ms ✗ | 2.92 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **17.8 ms ✓** | — | — | — | — | — |


## stability_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 690.4 ms ✓ | 820.9 ms ✓ | 1.7 s ✓ | 3.70 ms ✗ | 15.9 ms ✗ | 6.82 ms ✗ | 8.38 ms ✗ | 8.71 ms ✗ | 3.57 ms ✗ | 4.43 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **492.8 ms ✓** | — | — | — | — | — |


## t2_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 1.67 ms — | **1.67 ms ✓** | 1.69 ms ✓ | 1.83 ms ✓ | 4.21 ms ✓ | 6.34 ms ✓ | 5.80 ms ✓ | 6.75 ms ✓ | 2.62 ms ✗ | 3.02 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 37.0 ms ✓ | — | — | — | — | — |


## uve_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 1.5 s ✓ | 1.5 s ✓ | 1.7 s ✓ | 3.28 ms ✗ | 10.3 ms ✗ | 7.89 ms ✗ | 6.77 ms ✗ | 6.20 ms ✗ | 2.70 ms ✗ | 3.09 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **222.0 ms ✓** | — | — | — | — | — |


## variable_select_coef  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **1.56 ms ✓** | 1.56 ms ✓ | 1.61 ms ✓ | 2.87 ms ✓ | 15.0 ms ✓ | 9.19 ms ✓ | 17.2 ms ✓ | 18.1 ms ✓ | 12.4 ms ✗ | 6.42 ms ✗ | — | 50.9 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## variable_select_sr  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 542.8 ms ✓ | 1.5 s ✓ | 1.2 s ✓ | 8.28 ms ✗ | 24.1 ms ✗ | 11.2 ms ✗ | 31.6 ms ✗ | 22.3 ms ✗ | 8.80 ms ✗ | 18.9 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **18.9 ms ✓** | — | — | — | — | — |


## variable_select_vip  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 2.55 ms ✓ | 2.70 ms ✓ | 2.47 ms ✓ | **1.97 ms ✓** | 6.19 ms ✓ | 6.51 ms ✓ | 5.86 ms ✓ | 6.61 ms ✓ | 2.76 ms ✗ | 3.29 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 16.2 ms ✓ | — | — | — | — | — |


## vip_spa_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 80×40 | — | — | — | **1.34 ms ✓** | 2.19 ms ✓ | 2.80 ms ✓ | 2.83 ms ✓ | 2.87 ms ✓ | 1.24 ms ✗ | 1.56 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## vissa_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 80×25 | — | — | — | **10.8 ms ✓** | 11.2 ms ✓ | 11.1 ms ✓ | 21.4 ms ✓ | 49.1 ms ✓ | 62.9 ms ✗ | 75.0 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## weighted_pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | 1.93 ms ✓ | 1.82 ms ✓ | **1.77 ms ✓** | 1.92 ms ✓ | 4.57 ms ✓ | 5.88 ms ✓ | 5.23 ms ✓ | 5.95 ms ✓ | 2.70 ms ✗ | 3.22 ms ✗ | 2.12 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## wvc_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 1.49 ms ✓ | **1.47 ms ✓** | 1.48 ms ✓ | 1.63 ms ✓ | 4.23 ms ✓ | 4.97 ms ✓ | 5.72 ms ✓ | 5.30 ms ✓ | 2.45 ms ✗ | 2.73 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 14.4 ms ✓ | — | — | — | — | — |


## wvc_threshold_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.python_chun_keles_spls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.python_stone_brooks_1990_py | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_numpy | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_base | ref.r_softimpute | ref.python_chun_keles_splsda | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | ref.python_scars_numpy_port | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 336.1 ms ✓ | 344.6 ms ✓ | 344.7 ms ✓ | 1.67 ms ✗ | 4.07 ms ✗ | 4.66 ms ✗ | 5.51 ms ✗ | 4.59 ms ✗ | 2.41 ms ✗ | 2.81 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **13.0 ms ✓** | — | — | — | — | — |


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
| `pls4all.matlab` | MATLAB/Octave | pls4all raw | `pls4all.<algo>(X, y, ...)` — single dispatcher MEX |
| `pls4all.matlab.classdef` | MATLAB/Octave | pls4all idiomatic | `pls4all.fit(algo, X, y, ...)` factory + per-algorithm classdefs |
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
| `pls4all.matlab` | `language=Octave 10.3.0`; `pls4all=from libn4m-linked MEX`; `registry_method=pls`; `blas=linked-BLAS`; `timing_schema=adaptive-v1` |
| `pls4all.matlab.classdef` | `language=Octave 10.3.0`; `pls4all=from libn4m-linked MEX + classdefs`; `registry_method=pls`; `blas=linked-BLAS`; `timing_schema=adaptive-v1` |
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
