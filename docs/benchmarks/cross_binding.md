# Cross-binding benchmark — parity + timing

For every (algorithm × backend × dataset size × thread count) cell, we report the **adaptive wall-clock time** and one visible **parity verdict**. C++/external columns show reference parity; internal bindings show binding parity. Every backend reads the same orchestrator-generated CSV dataset.

_Generated: 2026-05-20 13:44:47 UTC_
_CSV: `parity_30x30_strict.csv` (3155 cells)_


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

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 1.49 ms ✓ | 1.64 ms ✓ | **1.39 ms ✓** | 1.50 ms ✓ | 1.65 ms ✓ | 1.62 ms ✓ | 1.50 ms ✓ | 55.2 ms ✓ | 53.3 ms ✓ | 62.8 ms ✓ | 62.9 ms ✓ | 3.48 ms ✓ | 6.19 ms ✓ | — | — | — | — | — | — | — | 8.05 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | 1.1 s ✓ | — | — | — | — | 785.3 ms ✓ | — | — | — | — |


## aom_preprocess  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 1.21 ms ✓ | 1.25 ms ✓ | 1.32 ms ✓ | **1.08 ms ✓** | 1.11 ms ✓ | 1.31 ms ✓ | 1.17 ms ✓ | 59.7 ms ✓ | 57.9 ms ✓ | 59.8 ms ✓ | 57.7 ms ✓ | 3.14 ms ✓ | 4.60 ms ✓ | — | — | — | — | — | — | — | 4.18 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | 1.1 s ✓ | — | — | — | — | — | — | — | — | — |


## approximate_press  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 1.05 ms ✗ | 1.45 ms ✗ | 1.04 ms ✗ | 0.94 ms ✗ | 1.17 ms ✓ | **0.97 ms ✓** | 0.98 ms ✓ | 57.7 ms ✓ | 59.9 ms ✓ | 65.0 ms ✓ | 66.6 ms ✓ | 2.35 ms ✓ | 4.46 ms ✓ | — | 56.7 ms ✓ | — | — | — | — | — | 3.26 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## bagging_pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 1.02 ms ✗ | 1.05 ms ✗ | 1.15 ms ✗ | 1.14 ms ✗ | **0.95 ms ✓** | 1.02 ms ✓ | 1.43 ms ✓ | 60.2 ms ✓ | 60.6 ms ✓ | 67.8 ms ✓ | 64.2 ms ✓ | 2.77 ms ✓ | 5.79 ms ✓ | 433.4 ms ✓ | — | — | — | — | — | — | 3.84 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 645.9 ms ✓ | — | — | — | — |


## bipls_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 1.69 ms ✓ | 1.65 ms ✓ | **1.47 ms ✓** | 1.57 ms ✓ | 1.50 ms ✓ | 1.79 ms ✓ | 1.73 ms ✓ | 57.8 ms ✓ | 58.6 ms ✓ | 59.4 ms ✓ | 67.4 ms ✓ | 3.27 ms ✓ | 5.34 ms ✓ | — | — | — | — | — | — | — | 10.6 ms ✓ | — | — | — | — | — | 419.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## boosting_pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 0.99 ms ✗ | 1.01 ms ✗ | 1.08 ms ✗ | 1.02 ms ✗ | **0.98 ms ✓** | 1.00 ms ✓ | 1.73 ms ✓ | 55.3 ms ✓ | 62.5 ms ✓ | 67.0 ms ✓ | 65.3 ms ✓ | 2.63 ms ✓ | 4.90 ms ✓ | — | — | — | — | — | — | — | 3.75 ms ✓ | — | — | — | — | — | — | — | — | 654.3 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — |


## bve_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 1.27 ms ✗ | 1.40 ms ✗ | 1.33 ms ✗ | 1.24 ms ✗ | **1.24 ms ✓** | 1.47 ms ✓ | 1.50 ms ✓ | 55.0 ms ✓ | 58.4 ms ✓ | 63.6 ms ✓ | 64.1 ms ✓ | 2.95 ms ✓ | 5.53 ms ✓ | — | — | — | — | — | — | — | 7.94 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | 255.5 ms ✓ | — | — | — | — | — | — | — | — |


## cars_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 1.32 ms ✗ | 1.48 ms ✗ | 1.31 ms ✗ | 1.48 ms ✗ | **1.41 ms ✓** | 1.89 ms ✓ | 1.81 ms ✓ | 59.8 ms ✓ | 55.5 ms ✓ | 70.5 ms ✓ | 69.4 ms ✓ | 3.34 ms ✓ | 6.68 ms ✓ | — | — | — | — | — | — | — | 8.38 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 614.0 ms ✓ | — | — | — | — | — | — | — |


## continuum_regression  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 0.99 ms ✗ | 0.95 ms ✗ | 1.19 ms ✗ | 0.93 ms ✗ | **0.99 ms ✓** | 1.04 ms ✓ | 1.16 ms ✓ | 58.6 ms ✓ | 64.0 ms ✓ | 63.7 ms ✓ | 66.3 ms ✓ | 2.84 ms ✓ | 5.81 ms ✓ | — | — | — | — | — | — | — | 3.47 ms ✓ | — | 527.6 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## cppls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 0.94 ms ✗ | 1.02 ms ✗ | 0.94 ms ✗ | 0.97 ms ✗ | 1.05 ms ✓ | **0.95 ms ✓** | 1.24 ms ✓ | 56.5 ms ✓ | 58.6 ms ✓ | 56.7 ms ✓ | 51.8 ms ✓ | 2.42 ms ✓ | 4.92 ms ✓ | — | 13.0 ms ✓ | — | — | — | — | — | 3.28 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## di_pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 3.89 ms ✓ | 4.62 ms ✓ | 4.55 ms ✓ | 4.00 ms ✓ | **3.70 ms ✓** | 3.85 ms ✓ | 4.12 ms ✓ | 57.2 ms ✓ | 63.6 ms ✓ | 67.4 ms ✓ | 64.8 ms ✓ | 7.33 ms ✓ | 8.29 ms ✓ | — | — | — | — | — | — | 559.3 ms ✓ | 12.5 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 638.1 ms ✓ | — | — | — | — |


## ds  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 1.73 ms ✓ | 1.09 ms ✓ | 1.00 ms ✓ | **0.98 ms ✓** | 1.03 ms ✓ | 1.05 ms ✓ | 1.17 ms ✓ | 58.0 ms ✓ | 56.3 ms ✓ | 62.5 ms ✓ | 61.2 ms ✓ | 3.56 ms ✓ | 5.02 ms ✓ | — | — | — | — | — | — | — | 4.04 ms ✓ | — | — | — | — | — | — | — | — | — | 47.3 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — |


## ecr  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 1.20 ms ✓ | **1.00 ms ✓** | 1.09 ms ✓ | 1.14 ms ✓ | 1.06 ms ✓ | 1.12 ms ✓ | 1.59 ms ✓ | 61.1 ms ✓ | 62.7 ms ✓ | 71.3 ms ✓ | 61.9 ms ✓ | 2.86 ms ✓ | 5.14 ms ✓ | — | — | — | — | — | — | — | 4.68 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 18.2 ms ✓ | — | — | — | — | — | — |


## emcuve_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 1.78 ms ✗ | 1.43 ms ✗ | 1.43 ms ✗ | 1.25 ms ✗ | **1.32 ms ✓** | 1.48 ms ✓ | 1.53 ms ✓ | 57.5 ms ✓ | 61.5 ms ✓ | 62.1 ms ✓ | 65.5 ms ✓ | 3.02 ms ✓ | 5.66 ms ✓ | — | — | — | — | — | — | — | 8.10 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | 349.0 ms ✓ | — | — | — | — | — | — | — | — |


## fused_sparse_pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 1.11 ms — | 0.96 ms — | 1.23 ms — | 1.19 ms — | **1.01 ms ✓** | 1.01 ms ✓ | 1.43 ms ✓ | 60.4 ms ✓ | 63.0 ms ✓ | 62.6 ms ✓ | 66.8 ms ✓ | 2.53 ms ✓ | 5.11 ms ✓ | — | — | — | — | — | — | — | 3.69 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## ga_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 2.08 ms ✗ | 2.24 ms ✗ | 2.10 ms ✗ | 2.32 ms ✗ | 2.32 ms ✓ | **2.02 ms ✓** | 2.06 ms ✓ | 58.2 ms ✓ | 58.9 ms ✓ | 65.1 ms ✓ | 64.4 ms ✓ | 3.61 ms ✓ | 5.55 ms ✓ | — | — | — | — | — | — | — | 9.98 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | 443.9 ms ✓ | — | — | — | — | — | — | — | — |


## gpr_pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 1.00 ms ✓ | 1.01 ms ✓ | 1.11 ms ✓ | 1.11 ms ✓ | **0.96 ms ✓** | 1.30 ms ✓ | 1.40 ms ✓ | 55.3 ms ✓ | 57.3 ms ✓ | 62.4 ms ✓ | 96.1 ms ✓ | 2.98 ms ✓ | 4.33 ms ✓ | 601.9 ms ✓ | — | — | — | — | — | — | 6.14 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## group_sparse_pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 1.04 ms ✗ | 1.30 ms ✗ | 1.06 ms ✗ | 0.91 ms ✗ | 1.18 ms ✓ | **1.04 ms ✓** | 1.37 ms ✓ | 66.3 ms ✓ | 64.0 ms ✓ | 70.0 ms ✓ | 63.0 ms ✓ | 2.79 ms ✓ | 4.57 ms ✓ | — | — | — | — | — | — | — | 4.82 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | 1.1 s ✓ | — | — | — | — | — | — | — | — | — | — |


## interval_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 1.52 ms ✗ | 1.36 ms ✗ | 1.21 ms ✗ | 1.29 ms ✗ | **1.24 ms ✓** | 1.45 ms ✓ | 1.59 ms ✓ | 56.3 ms ✓ | 63.0 ms ✓ | 68.8 ms ✓ | 68.7 ms ✓ | 3.25 ms ✓ | 5.45 ms ✓ | — | — | — | — | — | — | — | 6.93 ms ✓ | — | — | — | — | — | 580.2 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## ipw_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 1.43 ms ✗ | 1.47 ms ✗ | 1.17 ms ✗ | 1.42 ms ✗ | 1.64 ms ✓ | **1.13 ms ✓** | 1.35 ms ✓ | 60.7 ms ✓ | 62.2 ms ✓ | 62.4 ms ✓ | 62.2 ms ✓ | 2.70 ms ✓ | 4.50 ms ✓ | — | — | — | — | — | — | — | 7.05 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | 335.0 ms ✓ | — | — | — | — | — | — | — | — |


## irf_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 1.59 ms ✗ | 1.68 ms ✗ | 1.71 ms ✗ | 1.60 ms ✗ | 1.73 ms ✓ | **1.52 ms ✓** | 1.60 ms ✓ | 53.1 ms ✓ | 57.1 ms ✓ | 66.8 ms ✓ | 60.8 ms ✓ | 3.47 ms ✓ | 4.96 ms ✓ | — | — | — | — | — | — | — | 8.58 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 755.8 ms ✓ | — | — | — | — | — | — |


## iriv_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 23.8 ms ✓ | 22.5 ms ✓ | **21.2 ms ✓** | 22.6 ms ✓ | 23.1 ms ✓ | 24.0 ms ✓ | 24.7 ms ✓ | 86.0 ms ✓ | 77.0 ms ✓ | 83.3 ms ✓ | 79.5 ms ✓ | 24.2 ms ✓ | 28.3 ms ✓ | — | — | — | — | — | — | — | 46.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 129.2 s ✓ | — | — | — | — | — | — |


## kernel_pls_rbf  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 1.04 ms ✗ | 1.10 ms ✗ | 0.98 ms ✗ | 0.99 ms ✗ | **1.05 ms ✓** | 1.07 ms ✓ | 1.33 ms ✓ | 57.5 ms ✓ | 60.8 ms ✓ | 60.7 ms ✓ | 58.8 ms ✓ | 2.43 ms ✓ | 7.72 ms ✓ | — | — | — | — | — | — | — | 3.88 ms ✓ | — | — | — | 262.7 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## lw_pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 1.13 ms ✓ | **1.08 ms ✓** | 1.36 ms ✓ | 1.13 ms ✓ | 1.18 ms ✓ | 1.17 ms ✓ | 1.31 ms ✓ | 56.6 ms ✓ | 57.8 ms ✓ | 61.1 ms ✓ | 60.1 ms ✓ | 2.80 ms ✓ | 5.32 ms ✓ | — | — | — | — | — | — | — | 8.38 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | 374.9 ms ✓ | — | — | — | — | 637.9 ms ✓ | — | — | — | — |


## mb_pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 1.04 ms ✓ | 0.93 ms ✓ | **0.90 ms ✓** | 0.96 ms ✓ | 0.97 ms ✓ | 0.95 ms ✓ | 1.45 ms ✓ | 53.2 ms ✓ | 55.9 ms ✓ | 60.8 ms ✓ | 60.0 ms ✓ | 2.68 ms ✓ | 4.80 ms ✓ | — | — | — | — | — | — | — | 3.71 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | 361.7 ms ✓ | — | — | — | — | 664.2 ms ✓ | — | — | — | — |


## mir_pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 0.92 ms — | 1.25 ms — | 1.13 ms — | 0.98 ms — | 1.16 ms ✓ | **0.95 ms ✓** | 1.31 ms ✓ | 60.7 ms ✓ | 64.3 ms ✓ | 70.3 ms ✓ | 62.5 ms ✓ | 2.77 ms ✓ | 5.43 ms ✓ | — | — | — | — | — | — | — | 3.38 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## missing_aware_nipals  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 1.20 ms ✓ | **0.97 ms ✓** | 1.03 ms ✓ | 1.09 ms ✓ | 0.99 ms ✓ | 1.42 ms ✓ | 1.30 ms ✓ | 56.9 ms ✓ | 58.2 ms ✓ | 67.0 ms ✓ | 61.1 ms ✓ | 2.74 ms ✓ | 5.15 ms ✓ | — | — | — | — | — | — | — | 3.47 ms ✓ | — | — | — | — | — | — | — | — | — | — | 603.6 ms ✓ | — | — | — | — | — | — | — | — | — | — | — |


## n_pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 1.02 ms ✗ | 0.96 ms ✗ | 0.97 ms ✗ | 1.03 ms ✗ | 1.10 ms ✓ | **1.09 ms ✓** | 1.53 ms ✓ | 61.7 ms ✓ | 60.3 ms ✓ | 62.0 ms ✓ | 59.6 ms ✓ | 2.41 ms ✓ | 5.37 ms ✓ | — | — | — | — | — | — | — | 3.68 ms ✓ | — | — | 446.8 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## o2pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 1.08 ms ✗ | 1.02 ms ✗ | 0.98 ms ✗ | 1.15 ms ✗ | **1.01 ms ✓** | 1.04 ms ✓ | 1.44 ms ✓ | 56.8 ms ✓ | 61.8 ms ✓ | 62.2 ms ✓ | 62.7 ms ✓ | 2.51 ms ✓ | 5.54 ms ✓ | — | — | — | — | — | — | — | 3.70 ms ✓ | — | — | — | — | 872.3 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## on_pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 1.25 ms ✓ | 1.36 ms ✓ | 1.34 ms ✓ | 1.26 ms ✓ | **1.22 ms ✓** | 1.26 ms ✓ | 1.23 ms ✓ | 59.3 ms ✓ | 58.0 ms ✓ | 63.9 ms ✓ | 61.6 ms ✓ | 2.76 ms ✓ | 4.71 ms ✓ | — | — | — | — | — | — | — | 5.54 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 830.8 ms ✓ | — | 6.30 ms ✓ | — | — |


## one_se_rule  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 1.03 ms ✗ | 0.95 ms ✗ | 0.98 ms ✗ | 0.94 ms ✗ | **0.93 ms ✓** | 1.04 ms ✓ | 1.03 ms ✓ | 54.8 ms ✓ | 57.3 ms ✓ | 63.0 ms ✓ | 64.1 ms ✓ | 2.65 ms ✓ | 4.55 ms ✓ | — | 47.9 ms ✓ | — | — | — | — | — | 3.01 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## opls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | **0.97 ms ✓** | 1.11 ms ✓ | 1.08 ms ✓ | 1.00 ms ✓ | 1.00 ms ✓ | 1.21 ms ✓ | 1.35 ms ✓ | 47.2 ms ✓ | 50.0 ms ✓ | 51.3 ms ✓ | 50.4 ms ✓ | 4.67 ms ✓ | 4.46 ms ✓ | — | — | — | — | 25.7 ms ✓ | — | — | 5.33 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pcr  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 1.23 ms ✓ | 1.15 ms ✓ | **1.11 ms ✓** | 1.21 ms ✓ | 1.28 ms ✓ | 1.16 ms ✓ | 1.70 ms ✓ | 48.1 ms ✓ | 51.5 ms ✓ | 60.5 ms ✓ | 53.6 ms ✓ | 3.26 ms ✓ | 4.34 ms ✓ | 420.8 ms ✓ | 12.0 ms ✓ | — | — | — | — | — | 6.59 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pds  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 1.13 ms ✓ | 0.95 ms ✓ | 1.00 ms ✓ | **0.93 ms ✓** | 1.03 ms ✓ | 1.12 ms ✓ | 1.22 ms ✓ | 59.5 ms ✓ | 61.1 ms ✓ | 64.6 ms ✓ | 63.8 ms ✓ | 3.13 ms ✓ | 6.05 ms ✓ | — | — | — | — | — | — | — | 4.08 ms ✓ | — | — | — | — | — | — | — | — | — | 50.0 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — |


## pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | **0.91 ms ✓** | 0.96 ms ✓ | 1.10 ms ✓ | 1.03 ms ✓ | 1.39 ms ✓ | 0.99 ms ✓ | 1.40 ms ✓ | 55.1 ms ✓ | 57.2 ms ✓ | 56.9 ms ✓ | 53.8 ms ✓ | 2.77 ms ✓ | 4.91 ms ✓ | 371.9 ms ✓ | 11.2 ms ✓ | 548.4 ms ✗ | 11.0 ms ✓ | — | — | — | 5.22 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_cox  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 1.02 ms ✗ | 1.13 ms ✗ | 0.93 ms ✗ | 0.88 ms ✗ | **0.89 ms ✓** | 0.98 ms ✓ | 1.17 ms ✓ | 58.5 ms ✓ | 60.1 ms ✓ | 63.1 ms ✓ | 63.3 ms ✓ | 2.58 ms ✓ | 4.57 ms ✓ | — | — | — | — | — | — | — | 3.80 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 697.6 ms ✓ | — | — | — | 2.0 s ✓ |


## pls_diagnostic_dmodx  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 1.13 ms ✓ | 1.08 ms ✓ | **1.05 ms ✓** | 1.09 ms ✓ | 1.44 ms ✓ | 1.22 ms ✓ | 1.50 ms ✓ | 55.3 ms ✓ | 60.1 ms ✓ | 63.4 ms ✓ | 68.7 ms ✓ | 2.98 ms ✓ | 5.11 ms ✓ | — | — | — | — | — | — | — | 7.75 ms ✓ | — | — | — | — | — | 292.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_diagnostic_q  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 1.01 ms ✓ | 1.06 ms ✓ | 1.12 ms ✓ | 1.04 ms ✓ | 1.08 ms ✓ | **0.98 ms ✓** | 1.34 ms ✓ | 54.6 ms ✓ | 58.0 ms ✓ | 62.5 ms ✓ | 63.3 ms ✓ | 2.52 ms ✓ | 4.81 ms ✓ | — | — | — | — | — | — | — | 6.45 ms ✓ | — | — | — | — | — | 313.4 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_diagnostic_t2  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 1.06 ms ✓ | 1.18 ms ✓ | **1.01 ms ✓** | 1.08 ms ✓ | 1.13 ms ✓ | 1.03 ms ✓ | 1.29 ms ✓ | 62.1 ms ✓ | 62.3 ms ✓ | 64.6 ms ✓ | 60.9 ms ✓ | 2.35 ms ✓ | 4.91 ms ✓ | — | — | — | — | — | — | — | 6.82 ms ✓ | — | — | — | — | — | 297.8 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_glm  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 0.97 ms ✗ | 1.14 ms ✗ | 1.01 ms ✗ | 1.42 ms ✗ | **0.92 ms ✓** | 1.03 ms ✓ | 1.33 ms ✓ | 53.3 ms ✓ | 56.9 ms ✓ | 64.9 ms ✓ | 65.7 ms ✓ | 2.38 ms ✓ | 6.54 ms ✓ | — | — | — | — | — | — | — | 6.18 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 720.9 ms ✓ | — | — | 988.9 ms ✓ | — |


## pls_lda  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 0.99 ms ✗ | 0.96 ms ✗ | 0.88 ms ✗ | 0.86 ms ✗ | **0.87 ms ✓** | 0.97 ms ✓ | 1.16 ms ✓ | 51.8 ms ✓ | 56.5 ms ✓ | 61.1 ms ✓ | 60.1 ms ✓ | 2.40 ms ✓ | 4.36 ms ✓ | 385.9 ms ✓ | — | — | — | — | — | — | 6.18 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 677.4 ms ✓ | — | — | — | — |


## pls_logistic  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 0.88 ms ✗ | 0.94 ms ✗ | 0.94 ms ✗ | 1.21 ms ✗ | 0.97 ms ✓ | **0.91 ms ✓** | 1.40 ms ✓ | 53.4 ms ✓ | 60.1 ms ✓ | 61.3 ms ✓ | 64.4 ms ✓ | 2.62 ms ✓ | 4.55 ms ✓ | 411.3 ms ✓ | — | — | — | — | — | — | 5.88 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 730.1 ms ✓ | — | — | — | — |


## pls_monitoring  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 1.11 ms ✓ | 1.08 ms ✓ | 1.08 ms ✓ | 1.07 ms ✓ | **1.05 ms ✓** | 1.19 ms ✓ | 1.36 ms ✓ | 56.8 ms ✓ | 60.1 ms ✓ | 66.0 ms ✓ | 60.4 ms ✓ | 2.50 ms ✓ | 4.52 ms ✓ | — | — | — | — | — | — | — | 8.42 ms ✓ | — | — | — | — | — | 339.5 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_qda  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 0.86 ms ✓ | 0.85 ms ✓ | **0.83 ms ✓** | 0.88 ms ✓ | 1.20 ms ✓ | 0.87 ms ✓ | 1.18 ms ✓ | 47.9 ms ✓ | 50.5 ms ✓ | 51.5 ms ✓ | 53.3 ms ✓ | 2.68 ms ✓ | 4.12 ms ✓ | 376.5 ms ✓ | — | — | — | — | — | — | 3.73 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 590.7 ms ✓ | — | — | — | — |


## pop_pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 1.57 ms ✓ | **1.53 ms ✓** | 1.56 ms ✓ | 1.76 ms ✓ | 1.79 ms ✓ | 1.67 ms ✓ | 1.56 ms ✓ | 51.7 ms ✓ | 56.4 ms ✓ | 66.2 ms ✓ | 60.3 ms ✓ | 2.86 ms ✓ | 4.95 ms ✓ | — | — | — | — | — | — | — | 5.74 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | 1.1 s ✓ | — | — | — | — | 786.4 ms ✓ | — | — | — | — |


## pso_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 3.26 ms ✗ | 2.90 ms ✗ | 2.79 ms ✗ | 2.86 ms ✗ | **2.96 ms ✓** | 3.10 ms ✓ | 3.32 ms ✓ | 63.2 ms ✓ | 62.2 ms ✓ | 64.5 ms ✓ | 64.2 ms ✓ | 4.66 ms ✓ | 6.81 ms ✓ | — | — | — | — | — | — | — | 13.4 ms ✓ | — | — | — | — | — | — | — | 571.7 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## random_frog_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 1.37 ms ✗ | 1.65 ms ✗ | 1.69 ms ✗ | 1.71 ms ✗ | **1.33 ms ✓** | 1.51 ms ✓ | 1.54 ms ✓ | 62.0 ms ✓ | 61.4 ms ✓ | 59.1 ms ✓ | 57.1 ms ✓ | 3.02 ms ✓ | 4.66 ms ✓ | — | — | — | — | — | — | — | 8.71 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 75.4 ms ✓ | — | — | — | — | — | — |


## random_subspace_pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 0.95 ms ✗ | 0.93 ms ✗ | 0.95 ms ✗ | 1.10 ms ✗ | **0.99 ms ✓** | 1.02 ms ✓ | 1.17 ms ✓ | 58.8 ms ✓ | 62.5 ms ✓ | 69.2 ms ✓ | 68.1 ms ✓ | 2.47 ms ✓ | 5.14 ms ✓ | 464.9 ms ✓ | — | — | — | — | — | — | 3.60 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 646.6 ms ✓ | — | — | — | — |


## randomization_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 1.59 ms ✓ | **1.33 ms ✓** | 1.67 ms ✓ | 1.47 ms ✓ | 1.36 ms ✓ | 1.69 ms ✓ | 2.05 ms ✓ | 56.3 ms ✓ | 64.0 ms ✓ | 67.2 ms ✓ | 70.3 ms ✓ | 3.63 ms ✓ | 5.38 ms ✓ | — | — | — | — | — | — | — | 8.85 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 85.0 ms ✓ | — | — | — | — | — |


## recursive_pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 1.39 ms ✓ | 1.15 ms ✓ | 1.32 ms ✓ | 1.40 ms ✓ | **1.07 ms ✓** | 1.15 ms ✓ | 1.39 ms ✓ | 64.5 ms ✓ | 58.0 ms ✓ | 61.3 ms ✓ | 63.0 ms ✓ | 2.79 ms ✓ | 6.72 ms ✓ | 378.5 ms ✓ | 50.0 ms ✓ | — | — | — | — | — | 7.02 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 639.1 ms ✓ | — | — | — | — |


## rep_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 1.38 ms ✗ | 1.36 ms ✗ | 1.67 ms ✗ | 1.37 ms ✗ | 1.47 ms ✓ | **1.44 ms ✓** | 1.56 ms ✓ | 60.1 ms ✓ | 62.5 ms ✓ | 66.3 ms ✓ | 61.2 ms ✓ | 3.01 ms ✓ | 4.45 ms ✓ | — | — | — | — | — | — | — | 9.36 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | 376.0 ms ✓ | — | — | — | — | — | — | — | — |


## ridge_pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 0.40 ms ⚠ | 0.36 ms ⚠ | 0.39 ms ⚠ | 0.41 ms ⚠ | 0.38 ms ✓ | 0.38 ms ✓ | 0.60 ms ✓ | 0.91 ms ✓ | 2.53 ms ✓ | 1.62 ms ✓ | 1.66 ms ✓ | 0.64 ms ✓ | 1.53 ms ✓ | 383.8 ms ✓ | — | — | — | — | — | — | 0.30 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **0.29 ms ✓** | — | — | — | — |


## robust_pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 0.96 ms ✗ | 0.96 ms ✗ | 0.96 ms ✗ | 1.00 ms ✗ | **0.94 ms ✓** | 1.16 ms ✓ | 2.09 ms ✓ | 57.0 ms ✓ | 57.9 ms ✓ | 65.2 ms ✓ | 63.6 ms ✓ | 2.68 ms ✓ | 4.78 ms ✓ | — | — | — | — | — | — | — | 3.49 ms ✓ | 103.2 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## rosa  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 1.01 ms ✗ | 1.11 ms ✗ | 1.01 ms ✗ | 1.05 ms ✗ | 1.07 ms ✓ | **1.05 ms ✓** | 1.81 ms ✓ | 52.4 ms ✓ | 53.3 ms ✓ | 69.0 ms ✓ | 57.8 ms ✓ | 2.77 ms ✓ | 5.58 ms ✓ | — | — | — | — | — | — | — | 3.69 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 810.8 ms ✓ | 946.8 ms ✓ | — | — | — |


## scars_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 1.74 ms — | 1.61 ms — | 1.42 ms — | 1.47 ms — | 1.72 ms ✓ | **1.39 ms ✓** | 1.53 ms ✓ | 59.6 ms ✓ | 56.7 ms ✓ | 60.2 ms ✓ | 62.7 ms ✓ | 2.60 ms ✓ | 4.82 ms ✓ | — | — | — | — | — | — | — | 8.71 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## shaving_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 1.83 ms ✓ | 1.66 ms ✓ | 1.65 ms ✓ | **1.62 ms ✓** | 1.74 ms ✓ | 1.89 ms ✓ | 1.72 ms ✓ | 53.9 ms ✓ | 55.7 ms ✓ | 61.1 ms ✓ | 64.2 ms ✓ | 3.06 ms ✓ | 5.88 ms ✓ | — | — | — | — | — | — | — | 7.51 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | 298.6 ms ✓ | — | — | — | — | — | — | — | — |


## sipls_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 1.27 ms — | 1.47 ms — | 1.25 ms — | 1.35 ms — | **1.35 ms ✓** | 1.43 ms ✓ | 1.65 ms ✓ | 58.0 ms ✓ | 53.0 ms ✓ | 61.4 ms ✓ | 62.6 ms ✓ | 3.01 ms ✓ | 5.98 ms ✓ | — | — | — | — | — | — | — | 8.91 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## so_pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | **1.36 ms ✓** | 1.45 ms ✓ | 1.46 ms ✓ | 1.44 ms ✓ | 1.89 ms ✓ | 1.60 ms ✓ | 1.77 ms ✓ | 55.5 ms ✓ | 57.4 ms ✓ | 63.1 ms ✓ | 67.6 ms ✓ | 2.82 ms ✓ | 4.82 ms ✓ | — | — | — | — | — | — | — | 4.76 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 824.2 ms ✓ | 964.0 ms ✓ | — | — | — |


## spa_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 1.22 ms ✗ | 1.17 ms ✗ | 1.20 ms ✗ | 1.10 ms ✗ | **1.10 ms ✓** | 1.21 ms ✓ | 1.40 ms ✓ | 55.5 ms ✓ | 62.0 ms ✓ | 62.9 ms ✓ | 68.9 ms ✓ | 3.20 ms ✓ | 5.41 ms ✓ | — | — | — | — | — | — | — | 7.38 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | 295.1 ms ✓ | — | — | — | — | — | — | — | — |


## sparse_pls_da  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 0.93 ms ✗ | 1.00 ms ✗ | 0.92 ms ✗ | 0.91 ms ✗ | **0.89 ms ✓** | 0.90 ms ✓ | 1.30 ms ✓ | 57.1 ms ✓ | 52.8 ms ✓ | 58.6 ms ✓ | 61.2 ms ✓ | 2.45 ms ✓ | 5.02 ms ✓ | — | — | — | — | — | 48.9 ms ✓ | — | 6.04 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 705.2 ms ✓ | — | — | — | — |


## sparse_simpls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 0.42 ms ⚠ | 0.42 ms ⚠ | 0.37 ms ⚠ | 0.37 ms ⚠ | 0.41 ms ✓ | 0.38 ms ✓ | 0.55 ms ✓ | 0.93 ms ✓ | 1.96 ms ✓ | 1.72 ms ✓ | 1.92 ms ✓ | 0.64 ms ✓ | 1.15 ms ✓ | — | — | — | — | — | 33.3 ms ✓ | — | 0.26 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **0.21 ms ✓** | — | — | — | — |


## st_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 1.59 ms ✗ | 1.29 ms ✗ | 1.27 ms ✗ | 1.30 ms ✗ | **1.42 ms ✓** | 1.53 ms ✓ | 2.27 ms ✓ | 57.7 ms ✓ | 60.9 ms ✓ | 60.6 ms ✓ | 67.4 ms ✓ | 4.45 ms ✓ | 5.29 ms ✓ | — | — | — | — | — | — | — | 8.41 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | 298.0 ms ✓ | — | — | — | — | — | — | — | — |


## stability_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 1.12 ms ✗ | 1.18 ms ✗ | 1.10 ms ✗ | 1.19 ms ✗ | 1.41 ms ✓ | **1.31 ms ✓** | 1.46 ms ✓ | 73.0 ms ✓ | 66.3 ms ✓ | 66.2 ms ✓ | 62.1 ms ✓ | 2.53 ms ✓ | 4.97 ms ✓ | — | — | — | — | — | — | — | 7.28 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | 315.2 ms ✓ | — | — | — | — | — | — | — | — |


## t2_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 1.29 ms ✓ | 1.56 ms ✓ | 1.40 ms ✓ | 1.50 ms ✓ | **1.17 ms ✓** | 1.38 ms ✓ | 1.61 ms ✓ | 63.7 ms ✓ | 63.8 ms ✓ | 69.2 ms ✓ | 58.0 ms ✓ | 2.67 ms ✓ | 5.18 ms ✓ | — | — | — | — | — | — | — | 9.03 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | 321.2 ms ✓ | — | — | — | — | — | — | — | — |


## uve_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 1.10 ms ✗ | 1.34 ms ✗ | 1.38 ms ✗ | 1.16 ms ✗ | 1.32 ms ✓ | **1.30 ms ✓** | 1.36 ms ✓ | 57.3 ms ✓ | 57.2 ms ✓ | 60.9 ms ✓ | 64.0 ms ✓ | 2.57 ms ✓ | 5.83 ms ✓ | — | — | — | — | — | — | — | 7.59 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | 266.4 ms ✓ | — | — | — | — | — | — | — | — |


## variable_select_coef  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 1.17 ms ✓ | 1.45 ms ✓ | 1.11 ms ✓ | **1.05 ms ✓** | 1.13 ms ✓ | 1.07 ms ✓ | 1.25 ms ✓ | 62.9 ms ✓ | 61.4 ms ✓ | 65.0 ms ✓ | 68.6 ms ✓ | 2.88 ms ✓ | 4.83 ms ✓ | — | 308.5 ms ✓ | — | — | — | — | — | 6.80 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## variable_select_sr  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 1.32 ms ✗ | 1.24 ms ✗ | 1.08 ms ✗ | 1.16 ms ✗ | **1.21 ms ✓** | 1.25 ms ✓ | 1.25 ms ✓ | 56.1 ms ✓ | 57.1 ms ✓ | 61.3 ms ✓ | 60.3 ms ✓ | 2.51 ms ✓ | 5.53 ms ✓ | — | — | — | — | — | — | — | 6.74 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | 301.6 ms ✓ | — | — | — | — | — | — | — | — |


## variable_select_vip  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 1.22 ms ✗ | 1.08 ms ✗ | 1.07 ms ✗ | 1.12 ms ✗ | **1.08 ms ✓** | 1.21 ms ✓ | 1.25 ms ✓ | 56.5 ms ✓ | 58.1 ms ✓ | 62.4 ms ✓ | 60.8 ms ✓ | 5.69 ms ✓ | 5.11 ms ✓ | — | — | — | — | — | — | — | 6.75 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | 297.5 ms ✓ | — | — | — | — | — | — | — | — |


## vip_spa_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 1.48 ms ✗ | 1.12 ms ✗ | 1.01 ms ✗ | 1.00 ms ✗ | 1.39 ms ✓ | **1.22 ms ✓** | 1.70 ms ✓ | 56.5 ms ✓ | 66.1 ms ✓ | 59.8 ms ✓ | 70.2 ms ✓ | 3.29 ms ✓ | 5.67 ms ✓ | — | — | — | — | — | — | — | 7.07 ms ✓ | — | — | — | — | — | — | 543.3 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## vissa_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 7.65 ms ✗ | 7.88 ms ✗ | 7.58 ms ✗ | 9.12 ms ✗ | **7.67 ms ✓** | 8.00 ms ✓ | 7.80 ms ✓ | 71.4 ms ✓ | 66.7 ms ✓ | 74.6 ms ✓ | 71.3 ms ✓ | 9.73 ms ✓ | 13.3 ms ✓ | — | — | — | — | — | — | — | 25.5 ms ✓ | — | — | — | — | — | — | 1.9 s ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## weighted_pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 0.38 ms ⚠ | 0.42 ms ⚠ | 0.58 ms ⚠ | 0.38 ms ⚠ | 0.41 ms ✓ | 0.44 ms ✓ | 0.63 ms ✓ | 0.85 ms ✓ | 1.74 ms ✓ | 2.85 ms ✓ | 2.32 ms ✓ | 0.75 ms ✓ | 1.04 ms ✓ | 400.3 ms ✓ | — | — | — | — | — | — | 0.28 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **0.20 ms ✓** | — | — | — | — |


## wvc_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 1.00 ms ✓ | 1.27 ms ✓ | 0.97 ms ✓ | 1.07 ms ✓ | **0.89 ms ✓** | 0.94 ms ✓ | 1.72 ms ✓ | 54.3 ms ✓ | 55.9 ms ✓ | 62.1 ms ✓ | 61.9 ms ✓ | 2.38 ms ✓ | 5.78 ms ✓ | — | — | — | — | — | — | — | 4.08 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | 280.2 ms ✓ | — | — | — | 910.4 ms ✓ | — | — | — | — |


## wvc_threshold_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | js_wasm | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | ref.r_enpls | libPLS | ref.r_pls_stats | julia | ref.r_multiblock | ref.python_onpls | ref.r_plsrglm | ref.r_plsrcox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30×30 | 0.95 ms ✓ | **0.91 ms ✓** | 1.06 ms ✓ | 1.09 ms ✓ | 1.20 ms ✓ | 1.34 ms ✓ | 1.21 ms ✓ | 61.5 ms ✓ | 63.1 ms ✓ | 63.6 ms ✓ | 63.8 ms ✓ | 2.27 ms ✓ | 5.29 ms ✓ | — | — | — | — | — | — | — | 3.98 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | 309.9 ms ✓ | — | — | — | 911.5 ms ✓ | — | — | — | — |


## Backend definitions

Each column in the per-algorithm tables above is one of the entries below. Columns are named `owner.language[.variant]`: `pls4all.*` are this project's own bindings; everything else is an external library shipped by its own maintainers.

| Column | Language | Tier | What it actually runs |
|---|---|---|---|
| `pls4all.cpp.native` | C++ | pls4all native scalar | libp4a built with `PLS4ALL_WITH_BLAS=OFF, OPENMP=OFF` — pure scalar native C++ loops, no acceleration. Used as one C++ implementation column; binding parity still uses cpp @ blas-omp when available. |
| `pls4all.cpp.blas` | C++ | pls4all + BLAS | libp4a built with `PLS4ALL_WITH_BLAS=ON` only — links system BLAS (OpenBLAS in this env), benefits from BLAS thread parallelism. |
| `pls4all.cpp.omp` | C++ | pls4all + OpenMP | libp4a built with `PLS4ALL_WITH_OPENMP=ON` only — OpenMP parallelism in the C kernel loops, no BLAS. |
| `pls4all.cpp.blas+omp` | C++ | pls4all + BLAS + OpenMP | libp4a built with both `PLS4ALL_WITH_BLAS=ON` and `PLS4ALL_WITH_OPENMP=ON` — the recommended production config. |
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
| `ref.r_spls` | external | reference | registry-declared external reference library |
| `ref.python_diplslib` | external | reference | registry-declared external reference library |
| `ref.r_chemometrics` | external | reference | registry-declared external reference library |
| `ref.r_jico` | external | reference | registry-declared external reference library |
| `ref.python_tensorly` | external | reference | registry-declared external reference library |
| `ref.r_kernlab_pls` | external | reference | registry-declared external reference library |
| `ref.r_omicspls` | external | reference | registry-declared external reference library |
| `ref.r_mdatools` | external | reference | registry-declared external reference library |
| `ref.python_auswahl` | external | reference | registry-declared external reference library |
| `ref.python_pyswarms` | external | reference | registry-declared external reference library |
| `ref.r_mboost` | external | reference | registry-declared external reference library |
| `ref.r_base` | external | reference | registry-declared external reference library |
| `ref.r_softimpute` | external | reference | registry-declared external reference library |
| `ref.r_sgpls` | external | reference | registry-declared external reference library |
| `ref.python_nirs4all` | external | reference | registry-declared external reference library |
| `ref.r_plsvarsel` | external | reference | registry-declared external reference library |
| `ref.r_enpls` | external | reference | registry-declared external reference library |
| `libPLS` | external | reference | registry-declared external reference library |
| `ref.r_pls_stats` | external | reference | registry-declared external reference library |
| `ref.r_multiblock` | external | reference | registry-declared external reference library |
| `ref.python_onpls` | external | reference | registry-declared external reference library |
| `ref.r_plsrglm` | external | reference | registry-declared external reference library |
| `ref.r_plsrcox` | external | reference | registry-declared external reference library |


## Versions per backend

| Column | Versions |
|---|---|
| `pls4all.cpp.native` | `language=C++ (via ctypes bridge) (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `libp4a=0.97.3+abi.1.16.0`; `numpy=2.3.5`; `registry_method=pls`; `timing_schema=adaptive-v1` |
| `pls4all.cpp.blas` | `language=C++ (via ctypes bridge) (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1; libopenblas 0.3.30 blas=1`; `libp4a=0.97.3+abi.1.16.0`; `numpy=2.3.5`; `registry_method=pls`; `timing_schema=adaptive-v1` |
| `pls4all.cpp.omp` | `language=C++ (via ctypes bridge) (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1; libomp ? openmp=1`; `libp4a=0.97.3+abi.1.16.0`; `numpy=2.3.5`; `registry_method=pls`; `timing_schema=adaptive-v1` |
| `pls4all.cpp.blas+omp` | `language=C++ (via ctypes bridge) (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1; libopenblas 0.3.33 blas=1; libomp ? openmp=1`; `libp4a=0.97.3+abi.1.16.0`; `numpy=2.3.5`; `registry_method=pls`; `timing_schema=adaptive-v1` |
| `pls4all.registry` | `language=Python 3.13.13`; `blas=libscipy_openblas 0.3.30 blas=1`; `pls4all=0.97.3`; `numpy=2.3.5`; `registry_method=pls`; `timing_schema=adaptive-v1` |
| `pls4all.python` | `language=Python 3.13.13`; `blas=libscipy_openblas 0.3.30 blas=1`; `pls4all=0.97.3`; `numpy=2.3.5`; `registry_method=pls`; `timing_schema=adaptive-v1` |
| `pls4all.sklearn` | `language=Python 3.13.13`; `blas=libscipy_openblas 0.3.30 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `pls4all=0.97.3`; `numpy=2.3.5`; `sklearn_class=PLSRegression`; `timing_schema=adaptive-v1` |
| `pls4all.R` | `language=R 4.3.3`; `pls4all=0.97.0`; `registry_method=pls`; `blas=linked-BLAS`; `timing_schema=adaptive-v1` |
| `pls4all.R.formula` | `language=R 4.3.3`; `pls4all=0.97.0`; `registry_method=pls`; `formula_facade=True`; `blas=linked-BLAS`; `timing_schema=adaptive-v1` |
| `pls4all.R.pls_compat` | `language=R 4.3.3`; `pls4all=0.97.0`; `registry_method=pls`; `facade=pls_compat`; `formula_facade=True`; `blas=linked-BLAS`; `timing_schema=adaptive-v1` |
| `pls4all.R.mdatools_compat` | `language=R 4.3.3`; `pls4all=0.97.0`; `registry_method=pls`; `facade=mdatools_compat`; `matrix_facade=True`; `blas=linked-BLAS`; `timing_schema=adaptive-v1` |
| `pls4all.matlab` | `language=Octave 10.3.0`; `pls4all=from libp4a-linked MEX`; `registry_method=pls`; `blas=linked-BLAS`; `timing_schema=adaptive-v1` |
| `pls4all.matlab.classdef` | `language=Octave 10.3.0`; `pls4all=from libp4a-linked MEX + classdefs`; `registry_method=pls`; `blas=linked-BLAS`; `timing_schema=adaptive-v1` |
| `ref.python_scikit_learn` | `language=Python 3.13.13`; `blas=libscipy_openblas 0.3.30 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `numpy=2.3.5`; `reference_id=python_scikit_learn`; `reference_library=scikit-learn`; `reference_version=1.8.0`; `timing_schema=adaptive-v1` |
| `ref.r_pls` | `language=R 4.3.3`; `pls=2.8.5`; `timing_schema=adaptive-v1` |
| `ref.python_ikpls` | `language=Python 3.13.13`; `blas=libscipy_openblas 0.3.30 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `numpy=2.3.5`; `reference_id=python_ikpls`; `reference_library=ikpls`; `reference_version=4.0.1.post1`; `timing_schema=adaptive-v1` |
| `ref.r_mixomics` | `language=R 4.3.3`; `mixOmics=6.26.0`; `timing_schema=adaptive-v1` |
| `ref.r_ropls` | `language=R 4.3.3`; `ropls=1.34.0`; `timing_schema=adaptive-v1` |
| `ref.r_spls` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_spls`; `reference_library=spls`; `reference_version=2.3.2`; `timing_driver=rscript-batch`; `timing_schema=adaptive-v1` |
| `ref.python_diplslib` | `language=Python 3.13.13`; `blas=libscipy_openblas 0.3.30 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `numpy=2.3.5`; `reference_id=python_diplslib`; `reference_library=diPLSlib`; `reference_version=2.5.0`; `timing_schema=adaptive-v1` |
| `js_wasm` | `language=Node v22.21.1`; `pls4all=0.85.0+abi.1.13.0`; `registry_method=cppls`; `wasm=emscripten`; `timing_schema=adaptive-v1` |
| `ref.r_chemometrics` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_chemometrics`; `reference_library=chemometrics`; `reference_version=0.7.x`; `timing_driver=rscript-batch`; `timing_schema=adaptive-v1` |
| `ref.r_jico` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_jico`; `reference_library=JICO`; `reference_version=0.1`; `timing_driver=rscript-batch`; `timing_schema=adaptive-v1` |
| `ref.python_tensorly` | `language=Python 3.13.13`; `blas=libscipy_openblas 0.3.30 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `numpy=2.3.5`; `reference_id=python_tensorly`; `reference_library=tensorly`; `reference_version=0.9.0`; `timing_schema=adaptive-v1` |
| `ref.r_kernlab_pls` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_kernlab_pls`; `reference_library=kernlab+pls`; `reference_version=0.9.33+2.8.5`; `timing_driver=rscript-batch`; `timing_schema=adaptive-v1` |
| `ref.r_omicspls` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_omicspls`; `reference_library=OmicsPLS`; `reference_version=2.1.0`; `timing_driver=rscript-batch`; `timing_schema=adaptive-v1` |
| `ref.r_mdatools` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_mdatools`; `reference_library=mdatools`; `reference_version=0.15.0`; `timing_driver=rscript-batch`; `timing_schema=adaptive-v1` |
| `ref.python_auswahl` | `language=Python 3.13.13`; `blas=libscipy_openblas 0.3.30 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `numpy=2.3.5`; `reference_id=python_auswahl`; `reference_library=auswahl`; `reference_version=0.9.0`; `timing_schema=adaptive-v1` |
| `ref.python_pyswarms` | `language=Python 3.13.13`; `blas=libscipy_openblas 0.3.30 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `numpy=2.3.5`; `reference_id=python_pyswarms`; `reference_library=pyswarms`; `reference_version=1.3.0`; `timing_schema=adaptive-v1` |
| `ref.r_mboost` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_mboost`; `reference_library=mboost`; `reference_version=2.9-11`; `timing_driver=rscript-batch`; `timing_schema=adaptive-v1` |
| `ref.r_base` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_base`; `reference_library=base`; `reference_version=R 4.3.3`; `timing_driver=rscript-batch`; `timing_schema=adaptive-v1` |
| `ref.r_softimpute` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_softimpute`; `reference_library=softImpute`; `reference_version=1.4-1`; `timing_driver=rscript-batch`; `timing_schema=adaptive-v1` |
| `ref.r_sgpls` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_sgpls`; `reference_library=sgPLS`; `reference_version=1.8.1`; `timing_driver=rscript-batch`; `timing_schema=adaptive-v1` |
| `ref.python_nirs4all` | `language=Python 3.13.13`; `blas=libscipy_openblas 0.3.30 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `numpy=2.3.5`; `reference_id=python_nirs4all`; `reference_library=nirs4all`; `reference_version=in-tree`; `timing_schema=adaptive-v1` |
| `ref.r_plsvarsel` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_plsvarsel`; `reference_library=plsVarSel`; `reference_version=0.10.0`; `timing_driver=rscript-batch`; `timing_schema=adaptive-v1` |
| `ref.r_enpls` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_enpls`; `reference_library=enpls`; `reference_version=6.1`; `timing_driver=rscript-batch`; `timing_schema=adaptive-v1` |
| `libPLS` | `language=matlab (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=matlab_libpls`; `reference_library=libPLS`; `reference_version=1.95`; `timing_schema=adaptive-v1` |
| `ref.r_pls_stats` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_pls_stats`; `reference_library=pls+stats`; `reference_version=R 4.3.3`; `timing_driver=rscript-batch`; `timing_schema=adaptive-v1` |
| `julia` | `language=Julia 1.12.6`; `registry_method=mb_pls`; `pls4all=0.97.3+abi.1.16.0`; `timing_schema=adaptive-v1` |
| `ref.r_multiblock` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_multiblock`; `reference_library=multiblock`; `reference_version=0.8.10`; `timing_schema=adaptive-v1` |
| `ref.python_onpls` | `language=Python 3.13.13`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=python_onpls`; `reference_library=OnPLS`; `reference_version=github tomlof/OnPLS`; `timing_schema=adaptive-v1` |
| `ref.r_plsrglm` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_plsrglm`; `reference_library=plsRglm`; `reference_version=1.5.1`; `timing_driver=rscript-batch`; `timing_schema=adaptive-v1` |
| `ref.r_plsrcox` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_plsrcox`; `reference_library=plsRcox`; `reference_version=1.8.2`; `timing_driver=rscript-batch`; `timing_schema=adaptive-v1` |

## Methodology

- Gate 1 reference: `cpp` cell at 1 thread (libp4a via ctypes), or `python_tier1` when `cpp` is unavailable for an algorithm
- Gate 2 reference: the registry-declared canonical external reference for the method
- Gate 1 tolerance: 1e-6 max-abs-diff
- Gate 2 release tolerance: strict rmse_rel <= 1e-3 (<= 1e-6 is displayed as exact). `MethodSpec.rmse_rel_tol` is only a diagnostic/variant budget and must not turn a strict reference divergence into a passing release gate.
- All backends read the same orchestrator-generated CSV dataset (`benchmarks/cross_binding/data/data_<n>x<p>_seed<seed>.csv`)
- Adaptive timing: warmstart run #1; if there is any subsequent run, run #1 is excluded from the score. 2 total runs report run #2 alone; 3 total runs report the mean of runs #2-#3; 10/20/40 total runs report the median after the warmstart.
- Per-cell timeout guard: 24 h
- Thread control via `OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = BLIS_NUM_THREADS` set in the subprocess env, plus `Context.num_threads` for Python pls4all and `maxNumCompThreads()` for Octave
- pls4all libp4a build: `build/blas-omp/cpp/src/libp4a.so` (BLAS + OpenMP enabled)
