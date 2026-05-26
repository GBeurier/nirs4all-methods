# Cross-binding benchmark — thread sweep

Same matrix as the [main page](cross_binding.md), but with thread counts 1, 3 and 10 shown in separate per-(algorithm, thread) sections. **External libraries (`sklearn`, `pls`, `ropls`, `mixOmics`, `plsregress`, `ikpls`) typically don't accelerate their inner loops with thread count** — only their linked BLAS does, and that helps only when the algo is GEMM-bound. pls4all bindings use OpenMP at the C kernel level on top of the BLAS, so multi-thread wins are visible here.

_Generated: 2026-05-25 11:52:44 UTC_
_CSV: `tmp489rmt74.csv` (459 cells)_


## Host

| | |
|---|---|
| **CPU**            | 12th Gen Intel(R) Core(TM) i9-12900K |
| **Cores**          | 12 physical / 24 logical |
| **Frequency**  | 3187 MHz nominal |
| **L3 cache**   | 30720 KB |
| **RAM**            | 62.7 GB |
| **OS / kernel**    | Linux 6.6.87.2-microsoft-standard-WSL2 (x86_64) · kernel 6.6.87.2-microsoft-standard-WSL2 |
| **Python**         | 3.13.13 |


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

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **5.31 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 12.4 ms ✓ | — | — | — | — | — | — | — | — | — |


## aom_pls  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **4.43 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 13.7 ms ✓ | — | — | — | — | — | — | — | — | — |


## aom_pls  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **3.87 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 11.1 ms ✓ | — | — | — | — | — | — | — | — | — |


## aom_preprocess  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **1.89 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.95 ms ✓ | — | — | — | — | — | — | — | — | — |


## aom_preprocess  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **1.68 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.97 ms ✓ | — | — | — | — | — | — | — | — | — |


## aom_preprocess  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 2.35 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **1.74 ms ✓** | — | — | — | — | — | — | — | — | — |


## approximate_press  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **29.4 ms ✓** | 85.7 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## approximate_press  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **33.7 ms ✓** | 87.8 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## approximate_press  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **25.4 ms ✓** | 71.3 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## bagging_pls  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **10.0 ms ✓** | — | — | 10.8 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## bagging_pls  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **10.2 ms ✓** | — | — | 11.0 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## bagging_pls  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | 9.60 ms ✓ | — | — | **9.52 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## bipls_select  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **3.40 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | 313.9 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## bipls_select  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **3.55 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | 271.3 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## bipls_select  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **3.22 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | 244.9 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## boosting_pls  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **2.28 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 15.4 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## boosting_pls  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **1.26 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 16.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## boosting_pls  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **1.20 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 11.5 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## bve_select  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 572.0 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **53.9 ms ✓** | — | — | — | — | — | — | — |


## bve_select  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 504.2 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **47.5 ms ✓** | — | — | — | — | — | — | — |


## bve_select  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 431.9 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **45.7 ms ✓** | — | — | — | — | — | — | — |


## cars_select  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 987.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **78.9 ms ✓** | — | — | — |


## cars_select  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 876.5 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **69.8 ms ✓** | — | — | — |


## cars_select  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 715.0 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **58.2 ms ✓** | — | — | — |


## continuum_regression  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | 2.92 ms ✓ | — | — | — | — | — | — | **2.57 ms ✓** | — | — | — | — | — | — | — | 44.8 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## continuum_regression  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | 4.05 ms ✓ | — | — | — | — | — | — | **2.76 ms ✓** | — | — | — | — | — | — | — | 44.3 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## continuum_regression  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **4.21 ms ✓** | — | — | — | — | — | — | 4.43 ms ✓ | — | — | — | — | — | — | — | 46.2 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## cppls  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **1.90 ms ✓** | 14.6 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## cppls  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **1.85 ms ✓** | 13.6 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## cppls  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **1.85 ms ✓** | 11.6 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## di_pls  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **3.11 ms ✓** | — | — | — | — | — | 4.93 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## di_pls  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **2.88 ms ✓** | — | — | — | — | — | 4.55 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## di_pls  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **2.71 ms ✓** | — | — | — | — | — | 4.15 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## ds  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **1.48 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 28.9 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — |


## ds  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **2.15 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 32.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — |


## ds  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **2.26 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 23.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — |


## ecr  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **2.32 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 16.4 ms ✓ | — |


## ecr  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **2.23 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 13.3 ms ✓ | — |


## ecr  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **2.13 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 9.23 ms ✓ | — |


## emcuve_select  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 1.6 s ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **1.1 s ✓** | — | — | — | — | — | — | — |


## emcuve_select  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 1.5 s ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **1.0 s ✓** | — | — | — | — | — | — | — |


## emcuve_select  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 1.3 s ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **895.4 ms ✓** | — | — | — | — | — | — | — |


## fused_sparse_pls  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **1.83 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## fused_sparse_pls  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **2.25 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## fused_sparse_pls  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **1.84 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## ga_select  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 786.8 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **285.0 ms ✓** | — | — | — | — | — | — | — |


## ga_select  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 708.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **247.7 ms ✓** | — | — | — | — | — | — | — |


## ga_select  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 609.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **201.8 ms ✓** | — | — | — | — | — | — | — |


## gpr_pls  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 120×25 | **1.22 ms ✓** | — | — | 2.29 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## gpr_pls  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 120×25 | **1.14 ms ✓** | — | — | 2.39 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## gpr_pls  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 120×25 | **1.64 ms ✓** | — | — | 1.99 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## group_sparse_pls  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **2.64 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 2.68 ms ✓ | — | — | — | — | — | — | — | — | 32.4 ms ✓ | — | — | — | — | — | — | — | — |


## group_sparse_pls  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | 2.68 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **2.65 ms ✓** | — | — | — | — | — | — | — | — | 33.3 ms ✓ | — | — | — | — | — | — | — | — |


## group_sparse_pls  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | 2.39 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **2.31 ms ✓** | — | — | — | — | — | — | — | — | 22.0 ms ✓ | — | — | — | — | — | — | — | — |


## interval_select  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 951.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | **454.0 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## interval_select  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 824.5 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | **383.1 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## interval_select  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 689.9 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | **317.9 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## ipw_select  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 491.3 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **25.0 ms ✓** | — | — | — | — | — | — | — |


## ipw_select  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 438.7 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **24.5 ms ✓** | — | — | — | — | — | — | — |


## ipw_select  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 383.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **20.3 ms ✓** | — | — | — | — | — | — | — |


## irf_select  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 120×30 | 138.6 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **129.1 ms ✓** | — | — | — | — | — |


## irf_select  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 120×30 | 128.4 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **122.7 ms ✓** | — | — | — | — | — |


## irf_select  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 120×30 | 104.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **103.8 ms ✓** | — | — | — | — | — |


## iriv_select  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 80×25 | 347.5 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **335.9 ms ✓** |


## iriv_select  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 80×25 | 311.5 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **306.4 ms ✓** |


## iriv_select  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 80×25 | **274.7 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 284.0 ms ✓ |


## kernel_pls_rbf  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **3.44 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | 42.2 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## kernel_pls_rbf  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **3.43 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | 37.9 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## kernel_pls_rbf  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **3.13 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | 41.4 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## lw_pls  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **10.6 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 18.2 ms ✓ | — | — | — | — | — | — | — | — | — |


## lw_pls  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **9.50 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 20.4 ms ✓ | — | — | — | — | — | — | — | — | — |


## lw_pls  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **9.11 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 16.9 ms ✓ | — | — | — | — | — | — | — | — | — |


## mb_pls  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×60 | **2.45 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 2.98 ms ✓ | — | — | — | — | — | — | — | — | — |


## mb_pls  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×60 | **2.24 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 3.10 ms ✓ | — | — | — | — | — | — | — | — | — |


## mb_pls  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×60 | **2.14 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 2.59 ms ✓ | — | — | — | — | — | — | — | — | — |


## mir_pls  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **1.30 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## mir_pls  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **1.32 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## mir_pls  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **1.12 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## missing_aware_nipals  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **1.25 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 16.7 ms ✓ | — | — | — | — | — | — | — | — | — | — | — |


## missing_aware_nipals  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **1.99 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 17.8 ms ✓ | — | — | — | — | — | — | — | — | — | — | — |


## missing_aware_nipals  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **2.09 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 10.6 ms ✓ | — | — | — | — | — | — | — | — | — | — | — |


## n_pls  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×48 | **24.2 ms ✓** | — | — | — | — | — | — | — | — | — | 25.2 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## n_pls  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×48 | **23.6 ms ✓** | — | — | — | — | — | — | — | — | — | 24.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## n_pls  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×48 | 28.3 ms ✓ | — | — | — | — | — | — | — | — | — | **19.1 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## o2pls  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **1.28 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | 11.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## o2pls  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **1.29 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | 12.0 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## o2pls  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **2.13 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | 7.33 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## on_pls  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **1.93 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 11.7 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## on_pls  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **1.97 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 11.6 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## on_pls  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **1.92 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 10.2 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## one_se_rule  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **2.41 ms ✓** | 15.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## one_se_rule  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **2.16 ms ✓** | 13.7 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## one_se_rule  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **2.04 ms ✓** | 12.4 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## opls  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **1.89 ms ✓** | — | — | — | — | — | — | — | — | — | — | 21.7 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## opls  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **2.00 ms ✓** | — | — | — | — | — | — | — | — | — | — | 22.0 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## opls  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **1.89 ms ✓** | — | — | — | — | — | — | — | — | — | — | 17.3 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pcr  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **3.14 ms ✓** | 12.6 ms ✗ | — | 3.28 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pcr  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **2.63 ms ✓** | 12.6 ms ✗ | — | 3.45 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pcr  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **2.55 ms ✓** | 13.6 ms ✗ | — | 5.56 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pds  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **1.41 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 31.2 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pds  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **1.38 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 29.2 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pds  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **2.19 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 21.7 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **1.87 ms ✓** | 12.7 ms ✓ | — | 3.76 ms ✓ | 2.27 ms ✓ | 14.9 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | 2.93 ms ✓ | 11.9 ms ✓ | — | 2.59 ms ✓ | **2.30 ms ✓** | 13.7 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | 2.81 ms ✓ | 8.49 ms ✓ | — | 2.21 ms ✓ | **1.89 ms ✓** | 12.7 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_cox  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **1.86 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 2.85 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_cox  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **1.87 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 2.34 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_cox  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | 1.78 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **1.66 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_diagnostic_dmodx  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **1.33 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | 21.8 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_diagnostic_dmodx  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **1.31 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | 22.4 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_diagnostic_dmodx  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **1.25 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | 17.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_diagnostic_q  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **2.21 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | 19.6 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_diagnostic_q  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **1.25 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | 19.8 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_diagnostic_q  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **1.26 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | 15.9 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_diagnostic_t2  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **1.28 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | 18.6 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_diagnostic_t2  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **1.79 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | 18.6 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_diagnostic_t2  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **2.07 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | 13.6 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_glm  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **2.06 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 127.9 ms ✓ | — | — | — | — | — | — | — | — | — | — |


## pls_glm  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **2.89 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 122.5 ms ✓ | — | — | — | — | — | — | — | — | — | — |


## pls_glm  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **1.91 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 97.2 ms ✓ | — | — | — | — | — | — | — | — | — | — |


## pls_lda  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **1.51 ms ✓** | — | — | 3.08 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_lda  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **1.48 ms ✓** | — | — | 2.68 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_lda  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **1.40 ms ✓** | — | — | 2.30 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_logistic  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **2.83 ms ✓** | — | — | 2.98 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_logistic  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 3.84 ms ✓ | — | — | **3.10 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_logistic  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **2.24 ms ✓** | — | — | 2.51 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_monitoring  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **1.40 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | 21.4 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_monitoring  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **1.27 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | 19.0 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_monitoring  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **1.23 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | 16.4 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_qda  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **1.96 ms ✓** | — | — | 2.39 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_qda  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **2.06 ms ✓** | — | — | 3.81 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_qda  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **1.74 ms ✓** | — | — | 2.23 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pop_pls  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **7.08 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 35.7 ms ✓ | — | — | — | — | — | — | — | — | — |


## pop_pls  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **5.02 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 31.5 ms ✓ | — | — | — | — | — | — | — | — | — |


## pop_pls  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **4.24 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 24.8 ms ✓ | — | — | — | — | — | — | — | — | — |


## pso_select  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 80×25 | 174.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **166.6 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pso_select  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 80×25 | **160.6 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 172.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pso_select  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 80×25 | **131.8 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 135.9 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## random_frog_select  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 103.4 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **101.4 ms ✓** | — | — | — | — | — |


## random_frog_select  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **85.1 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 87.2 ms ✓ | — | — | — | — | — |


## random_frog_select  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **74.3 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 75.2 ms ✓ | — | — | — | — | — |


## random_subspace_pls  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **10.0 ms ✓** | — | — | 10.8 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## random_subspace_pls  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **10.5 ms ✓** | — | — | 10.9 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## random_subspace_pls  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **9.21 ms ✓** | — | — | 9.30 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## randomization_select  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 268.9 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **78.2 ms ✓** | — | — |


## randomization_select  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 254.3 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **65.0 ms ✓** | — | — |


## randomization_select  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 207.0 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **51.2 ms ✓** | — | — |


## recursive_pls  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **3.15 ms ✓** | 216.4 ms ✓ | — | 70.0 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## recursive_pls  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **4.74 ms ✓** | 195.9 ms ✓ | — | 60.4 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## recursive_pls  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **2.90 ms ✓** | 147.7 ms ✓ | — | 52.4 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## rep_select  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 779.8 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **283.8 ms ✓** | — | — | — | — | — | — | — |


## rep_select  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 677.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **254.1 ms ✓** | — | — | — | — | — | — | — |


## rep_select  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 597.4 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **223.7 ms ✓** | — | — | — | — | — | — | — |


## ridge_pls  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **2.15 ms ✓** | — | — | 2.97 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## ridge_pls  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **1.81 ms ✓** | — | — | 2.42 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## ridge_pls  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **1.81 ms ✓** | — | — | 2.19 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## robust_pls  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **2.39 ms ✓** | — | — | — | — | — | — | — | — | 25.6 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## robust_pls  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **2.20 ms ✓** | — | — | — | — | — | — | — | — | 24.3 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## robust_pls  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **2.06 ms ✓** | — | — | — | — | — | — | — | — | 18.6 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## rosa  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **1.33 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.58 ms ✓ | — | — | — | — | — | — | — | — | — | — | 1.3 s ✓ | — | — | — | — | — | — |


## rosa  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **1.22 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.60 ms ✓ | — | — | — | — | — | — | — | — | — | — | 1.3 s ✓ | — | — | — | — | — | — |


## rosa  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **1.26 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.36 ms ✓ | — | — | — | — | — | — | — | — | — | — | 1.1 s ✓ | — | — | — | — | — | — |


## scars_select  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **4.31 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 4.76 ms ✓ | — | — | — | — |


## scars_select  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 3.99 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **3.92 ms ✓** | — | — | — | — |


## scars_select  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 3.67 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **3.63 ms ✓** | — | — | — | — |


## shaving_select  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 541.7 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **46.7 ms ✓** | — | — | — | — | — | — | — |


## shaving_select  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 490.5 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **38.7 ms ✓** | — | — | — | — | — | — | — |


## shaving_select  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 416.3 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **32.8 ms ✓** | — | — | — | — | — | — | — |


## sipls_select  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **2.62 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## sipls_select  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **2.56 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## sipls_select  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **2.32 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## so_pls  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **1.89 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.4 s ✓ | — | — | — | — | — | — |


## so_pls  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **1.73 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.3 s ✓ | — | — | — | — | — | — |


## so_pls  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×30 | **1.71 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.0 s ✓ | — | — | — | — | — | — |


## spa_select  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 633.4 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **126.8 ms ✓** | — | — | — | — | — | — | — |


## spa_select  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 556.4 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **98.8 ms ✓** | — | — | — | — | — | — | — |


## spa_select  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 462.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **88.5 ms ✓** | — | — | — | — | — | — | — |


## sparse_pls_da  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **2.37 ms ✓** | — | — | — | — | — | — | — | 32.8 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | 4.20 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — |


## sparse_pls_da  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **2.19 ms ✓** | — | — | — | — | — | — | — | 32.2 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | 4.69 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — |


## sparse_pls_da  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **1.94 ms ✓** | — | — | — | — | — | — | — | 22.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | 3.31 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — |


## sparse_simpls  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **1.96 ms ✓** | — | 4.04 ms ✓ | — | — | — | — | — | 21.8 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## sparse_simpls  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **2.00 ms ✓** | — | 3.90 ms ✓ | — | — | — | — | — | 18.4 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## sparse_simpls  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **2.13 ms ✓** | — | 3.52 ms ✓ | — | — | — | — | — | 13.2 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## st_select  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 477.9 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **23.1 ms ✓** | — | — | — | — | — | — | — |


## st_select  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 456.6 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **18.4 ms ✓** | — | — | — | — | — | — | — |


## st_select  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 404.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **17.0 ms ✓** | — | — | — | — | — | — | — |


## stability_select  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 728.2 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **243.0 ms ✓** | — | — | — | — | — | — | — |


## stability_select  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 632.9 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **215.1 ms ✓** | — | — | — | — | — | — | — |


## stability_select  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 555.0 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **206.0 ms ✓** | — | — | — | — | — | — | — |


## t2_select  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **1.90 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 44.6 ms ✓ | — | — | — | — | — | — | — |


## t2_select  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **1.78 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 40.6 ms ✓ | — | — | — | — | — | — | — |


## t2_select  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **2.13 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 32.9 ms ✓ | — | — | — | — | — | — | — |


## uve_select  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 716.2 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **225.9 ms ✓** | — | — | — | — | — | — | — |


## uve_select  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 639.5 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **199.6 ms ✓** | — | — | — | — | — | — | — |


## uve_select  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 574.9 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **202.7 ms ✓** | — | — | — | — | — | — | — |


## variable_select_coef  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **1.66 ms ✓** | 17.6 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## variable_select_coef  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **1.63 ms ✓** | 12.9 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## variable_select_coef  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **1.88 ms ✓** | 10.4 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## variable_select_sr  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 457.0 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **16.7 ms ✓** | — | — | — | — | — | — | — |


## variable_select_sr  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 396.0 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **11.5 ms ✓** | — | — | — | — | — | — | — |


## variable_select_sr  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 338.8 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **10.4 ms ✓** | — | — | — | — | — | — | — |


## variable_select_vip  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **2.42 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 17.0 ms ✓ | — | — | — | — | — | — | — |


## variable_select_vip  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **2.80 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 15.7 ms ✓ | — | — | — | — | — | — | — |


## variable_select_vip  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **2.93 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 9.47 ms ✓ | — | — | — | — | — | — | — |


## vip_spa_select  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 80×40 | 133.7 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **131.0 ms ✓** | — | — | — | — | — |


## vip_spa_select  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 80×40 | 53.9 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **53.6 ms ✓** | — | — | — | — | — |


## vip_spa_select  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 80×40 | 37.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **25.9 ms ✓** | — | — | — | — | — |


## vissa_select  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 80×25 | 1.6 s ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **1.6 s ✓** | — | — | — | — | — |


## vissa_select  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 80×25 | 802.5 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **782.0 ms ✓** | — | — | — | — | — |


## vissa_select  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 80×25 | 231.8 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **230.4 ms ✓** | — | — | — | — | — |


## weighted_pls  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **2.31 ms ✓** | — | — | 2.56 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## weighted_pls  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **2.08 ms ✓** | — | — | 2.30 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## weighted_pls  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×50 | **1.99 ms ✓** | — | — | 2.21 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## wvc_select  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **1.69 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 17.2 ms ✓ | — | — | — | — | — | — | — |


## wvc_select  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **1.58 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 15.6 ms ✓ | — | — | — | — | — | — | — |


## wvc_select  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | **1.49 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 12.0 ms ✓ | — | — | — | — | — | — | — |


## wvc_threshold_select  —  1 thread

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 459.3 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **18.7 ms ✓** | — | — | — | — | — | — | — |


## wvc_threshold_select  —  3 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 404.6 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **15.2 ms ✓** | — | — | — | — | — | — | — |


## wvc_threshold_select  —  10 threads

| samples × features | pls4all.registry | ref.r_pls | ref.python_chun_keles_spls | ref.python_scikit_learn | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.python_stone_brooks_1990_py | ref.r_spls | ref.r_chemometrics | ref.python_tensorly | ref.r_ropls | ref.r_omicspls | ref.r_kernlab_pls | ref.r_mdatools | ref.r_jico | ref.python_numpy | ref.python_onpls | ref.r_mboost | ref.python_pyswarms | ref.r_base | ref.python_chun_keles_splsda | ref.r_softimpute | ref.r_plsrglm | ref.python_nirs4all | ref.r_sgpls | ref.r_plsvarsel | ref.r_multiblock | ref.python_auswahl | ref.python_scars_numpy_port | ref.r_enpls | ref.r_pls_stats | libPLS | ref.python_iriv_numpy_port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200×40 | 330.9 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **12.6 ms ✓** | — | — | — | — | — | — | — |


## Backend definitions

Each column in the per-algorithm tables above is one of the entries below. Columns are named `owner.language[.variant]`: `pls4all.*` are this project's own bindings; everything else is an external library shipped by its own maintainers.

| Column | Language | Tier | What it actually runs |
|---|---|---|---|
| `pls4all.registry` | Python | pls4all canonical | `benchmarks.parity_timing.registry.MethodSpec.pls4all_fn` — the canonical per-method pls4all entry point |
| `ref.r_pls` | external | reference | registry-declared external reference library |
| `ref.python_chun_keles_spls` | external | reference | registry-declared external reference library |
| `ref.python_scikit_learn` | external | reference | registry-declared external reference library |
| `ref.python_ikpls` | external | reference | registry-declared external reference library |
| `ref.r_mixomics` | external | reference | registry-declared external reference library |
| `ref.python_diplslib` | external | reference | registry-declared external reference library |
| `ref.python_stone_brooks_1990_py` | external | reference | registry-declared external reference library |
| `ref.r_spls` | external | reference | registry-declared external reference library |
| `ref.r_chemometrics` | external | reference | registry-declared external reference library |
| `ref.python_tensorly` | external | reference | registry-declared external reference library |
| `ref.r_ropls` | external | reference | registry-declared external reference library |
| `ref.r_omicspls` | external | reference | registry-declared external reference library |
| `ref.r_kernlab_pls` | external | reference | registry-declared external reference library |
| `ref.r_mdatools` | external | reference | registry-declared external reference library |
| `ref.r_jico` | external | reference | registry-declared external reference library |
| `ref.python_numpy` | external | reference | registry-declared external reference library |
| `ref.python_onpls` | external | reference | registry-declared external reference library |
| `ref.r_mboost` | external | reference | registry-declared external reference library |
| `ref.python_pyswarms` | external | reference | registry-declared external reference library |
| `ref.r_base` | external | reference | registry-declared external reference library |
| `ref.python_chun_keles_splsda` | external | reference | registry-declared external reference library |
| `ref.r_softimpute` | external | reference | registry-declared external reference library |
| `ref.r_plsrglm` | external | reference | registry-declared external reference library |
| `ref.python_nirs4all` | external | reference | registry-declared external reference library |
| `ref.r_sgpls` | external | reference | registry-declared external reference library |
| `ref.r_plsvarsel` | external | reference | registry-declared external reference library |
| `ref.r_multiblock` | external | reference | registry-declared external reference library |
| `ref.python_auswahl` | external | reference | registry-declared external reference library |
| `ref.python_scars_numpy_port` | external | reference | registry-declared external reference library |
| `ref.r_enpls` | external | reference | registry-declared external reference library |
| `ref.r_pls_stats` | external | reference | registry-declared external reference library |
| `libPLS` | external | reference | registry-declared external reference library |
| `ref.python_iriv_numpy_port` | external | reference | registry-declared external reference library |


## Versions per backend

| Column | Versions |
|---|---|
| `pls4all.registry` | `language=Python 3.13.13`; `blas=libscipy_openblas 0.3.30 blas=1; libopenblas 0.3.33 blas=1; libomp ? openmp=1`; `pls4all=0.1.0`; `numpy=2.3.5`; `registry_method=pls`; `timing_schema=adaptive-v1` |
| `ref.r_pls` | `language=R 4.3.3`; `pls=2.8.5`; `timing_schema=adaptive-v1` |
| `ref.python_chun_keles_spls` | `language=Python 3.13.13`; `blas=libscipy_openblas 0.3.30 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `numpy=2.3.5`; `reference_id=python_chun_keles_spls`; `reference_library=chun_keles_spls`; `reference_version=1.0`; `timing_schema=adaptive-v1` |
| `ref.python_scikit_learn` | `language=Python 3.13.13`; `blas=libscipy_openblas 0.3.30 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `numpy=2.3.5`; `reference_id=python_scikit_learn`; `reference_library=scikit-learn`; `reference_version=1.8.0`; `timing_schema=adaptive-v1` |
| `ref.python_ikpls` | `language=Python 3.13.13`; `blas=libscipy_openblas 0.3.30 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `numpy=2.3.5`; `reference_id=python_ikpls`; `reference_library=ikpls`; `reference_version=4.0.1.post1`; `timing_schema=adaptive-v1` |
| `ref.r_mixomics` | `language=R 4.3.3`; `mixOmics=6.26.0`; `timing_schema=adaptive-v1` |
| `ref.python_diplslib` | `language=Python 3.13.13`; `blas=libscipy_openblas 0.3.30 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `numpy=2.3.5`; `reference_id=python_diplslib`; `reference_library=diPLSlib`; `reference_version=2.5.0`; `timing_schema=adaptive-v1` |
| `ref.python_stone_brooks_1990_py` | `language=Python 3.13.13`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=python_stone_brooks_1990_py`; `reference_library=stone-brooks-1990-py`; `reference_version=1.0`; `timing_schema=adaptive-v1` |
| `ref.r_spls` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_spls`; `reference_library=spls`; `reference_version=2.3.2`; `timing_driver=rscript-batch`; `timing_schema=adaptive-v1` |
| `ref.r_chemometrics` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_chemometrics`; `reference_library=chemometrics`; `reference_version=0.7.x`; `timing_driver=rscript-batch`; `timing_schema=adaptive-v1` |
| `ref.python_tensorly` | `language=Python 3.13.13`; `blas=libscipy_openblas 0.3.30 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `numpy=2.3.5`; `reference_id=python_tensorly`; `reference_library=tensorly`; `reference_version=0.9.0`; `timing_schema=adaptive-v1` |
| `ref.r_ropls` | `language=R 4.3.3`; `ropls=1.34.0`; `timing_schema=adaptive-v1` |
| `ref.r_omicspls` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_omicspls`; `reference_library=OmicsPLS`; `reference_version=2.1.0`; `timing_driver=rscript-batch`; `timing_schema=adaptive-v1` |
| `ref.r_kernlab_pls` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_kernlab_pls`; `reference_library=kernlab+pls`; `reference_version=0.9.33+2.8.5`; `timing_driver=rscript-batch`; `timing_schema=adaptive-v1` |
| `ref.r_mdatools` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_mdatools`; `reference_library=mdatools`; `reference_version=0.15.0`; `timing_driver=rscript-batch`; `timing_schema=adaptive-v1` |
| `ref.r_jico` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_jico`; `reference_library=JICO`; `reference_version=0.1`; `timing_driver=rscript-batch`; `timing_schema=adaptive-v1` |
| `ref.python_numpy` | `language=Python 3.13.13`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=python_numpy`; `reference_library=numpy`; `reference_version=2.3.5`; `timing_schema=adaptive-v1` |
| `ref.python_onpls` | `language=Python 3.13.13`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=python_onpls`; `reference_library=OnPLS`; `reference_version=github tomlof/OnPLS`; `timing_schema=adaptive-v1` |
| `ref.r_mboost` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_mboost`; `reference_library=mboost`; `reference_version=2.9-11`; `timing_driver=rscript-batch`; `timing_schema=adaptive-v1` |
| `ref.python_pyswarms` | `language=Python 3.13.13`; `blas=libscipy_openblas 0.3.30 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `numpy=2.3.5`; `reference_id=python_pyswarms`; `reference_library=pyswarms`; `reference_version=1.3.0`; `timing_schema=adaptive-v1` |
| `ref.r_base` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_base`; `reference_library=base`; `reference_version=R 4.3.3`; `timing_driver=rscript-batch`; `timing_schema=adaptive-v1` |
| `ref.python_chun_keles_splsda` | `language=Python 3.13.13`; `blas=libscipy_openblas 0.3.30 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `numpy=2.3.5`; `reference_id=python_chun_keles_splsda`; `reference_library=chun_keles_splsda`; `reference_version=1.0`; `timing_schema=adaptive-v1` |
| `ref.r_softimpute` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_softimpute`; `reference_library=softImpute`; `reference_version=1.4-1`; `timing_driver=rscript-batch`; `timing_schema=adaptive-v1` |
| `ref.r_plsrglm` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_plsrglm`; `reference_library=plsRglm`; `reference_version=1.5.1`; `timing_driver=rscript-batch`; `timing_schema=adaptive-v1` |
| `ref.python_nirs4all` | `language=Python 3.13.13`; `blas=libscipy_openblas 0.3.30 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `numpy=2.3.5`; `reference_id=python_nirs4all`; `reference_library=nirs4all`; `reference_version=in-tree`; `timing_schema=adaptive-v1` |
| `ref.r_sgpls` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_sgpls`; `reference_library=sgPLS`; `reference_version=1.8.1`; `timing_driver=rscript-batch`; `timing_schema=adaptive-v1` |
| `ref.r_plsvarsel` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_plsvarsel`; `reference_library=plsVarSel`; `reference_version=0.10.0`; `timing_driver=rscript-batch`; `timing_schema=adaptive-v1` |
| `ref.r_multiblock` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_multiblock`; `reference_library=multiblock`; `reference_version=0.8.10`; `timing_schema=adaptive-v1` |
| `ref.python_auswahl` | `language=Python 3.13.13`; `blas=libscipy_openblas 0.3.30 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `numpy=2.3.5`; `reference_id=python_auswahl`; `reference_library=auswahl`; `reference_version=0.9.0`; `timing_schema=adaptive-v1` |
| `ref.python_scars_numpy_port` | `language=Python 3.13.13`; `blas=libscipy_openblas 0.3.30 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `numpy=2.3.5`; `reference_id=python_scars_numpy_port`; `reference_library=scars_numpy_port`; `reference_version=1.0.0`; `timing_schema=adaptive-v1` |
| `ref.r_enpls` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_enpls`; `reference_library=enpls`; `reference_version=6.1`; `timing_driver=rscript-batch`; `timing_schema=adaptive-v1` |
| `ref.r_pls_stats` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_pls_stats`; `reference_library=pls+stats`; `reference_version=R 4.3.3`; `timing_driver=rscript-batch`; `timing_schema=adaptive-v1` |
| `libPLS` | `language=matlab (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=matlab_libpls`; `reference_library=libPLS`; `reference_version=1.95`; `timing_schema=adaptive-v1` |
| `ref.python_iriv_numpy_port` | `language=Python 3.13.13`; `blas=libscipy_openblas 0.3.30 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `numpy=2.3.5`; `reference_id=python_iriv_numpy_port`; `reference_library=iriv_numpy_port`; `reference_version=1.0.0`; `timing_schema=adaptive-v1` |

## Methodology

- Gate 1 reference: `cpp` cell at 1 thread (libn4m via ctypes), or `python_tier1` when `cpp` is unavailable for an algorithm
- Gate 2 reference: the registry-declared canonical external reference for the method
- Gate 1 tolerance: 1e-6 max-abs-diff; Gate 2 tolerance: `MethodSpec.rmse_rel_tol` (also emitted as `reference_parity_tolerance` by newer CSVs)
- All backends read the same orchestrator-generated CSV dataset (`benchmarks/cross_binding/data/data_<n>x<p>_seed<seed>.csv`)
- Adaptive timing: warmstart run #1; if there is any subsequent run, run #1 is excluded from the score. 2 total runs report run #2 alone; 3 total runs report the mean of runs #2-#3; 10/20/40 total runs report the median after the warmstart.
- Per-cell timeout guard: 24 h
- Thread control via `OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = BLIS_NUM_THREADS` set in the subprocess env, plus `Context.num_threads` for Python pls4all and `maxNumCompThreads()` for Octave
- pls4all libn4m build: `build/blas-omp/cpp/src/libn4m.so` (BLAS + OpenMP enabled)
