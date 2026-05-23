# Cross-binding benchmark — thread sweep

Same matrix as the [main page](cross_binding.md), but with thread counts 1, 3 and 10 shown in separate per-(algorithm, thread) sections. **External libraries (`sklearn`, `pls`, `ropls`, `mixOmics`, `plsregress`, `ikpls`) typically don't accelerate their inner loops with thread count** — only their linked BLAS does, and that helps only when the algo is GEMM-bound. pls4all bindings use OpenMP at the C kernel level on top of the BLAS, so multi-thread wins are visible here.

_Generated: 2026-05-19 23:08:33 UTC_
_CSV: `tmpcm42ijos.csv` (27456 cells)_


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

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 2.85 ms ✓ | 2.71 ms ✓ | 2.92 ms ✓ | **2.68 ms ✓** | 2.79 ms ✓ | 2.72 ms ✓ | 2.86 ms ✓ | 5.45 ms ✓ | 6.86 ms ✓ | 6.35 ms ✓ | 6.84 ms ✓ | 3.04 ms ✓ | 3.60 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 12.0 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | 23.5 ms ✓ | **22.5 ms ✓** | 24.3 ms ✓ | 22.7 ms ✓ | 23.5 ms ✓ | 23.6 ms ✓ | 24.1 ms ✓ | 60.0 ms ✓ | 83.9 ms ✓ | 85.9 ms ✓ | 78.7 ms ✓ | 24.9 ms ✓ | 25.5 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 59.5 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | 307.6 ms ✓ | 295.7 ms ✓ | 309.0 ms ✓ | 301.3 ms ✓ | 318.6 ms ✓ | 298.5 ms ✓ | 307.7 ms ✓ | — | — | — | — | 372.2 ms ✓ | 363.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **263.2 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## aom_pls  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 2.80 ms ✓ | 2.70 ms ✓ | 2.85 ms ✓ | **2.65 ms ✓** | 2.70 ms ✓ | 2.78 ms ✓ | 2.73 ms ✓ | 4.94 ms ✓ | 6.26 ms ✓ | 6.05 ms ✓ | 6.14 ms ✓ | 3.04 ms ✓ | 3.25 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 11.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## aom_preprocess  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.01 ms ⚠ | 1.01 ms ⚠ | 1.07 ms ⚠ | 1.13 ms ✓ | **1.04 ms ✓** | 1.36 ms ✓ | 1.06 ms ✓ | 3.41 ms ✓ | 4.90 ms ✓ | 5.30 ms ✓ | 5.32 ms ✓ | 2.79 ms ✓ | 2.25 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.34 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | 8.88 ms ✓ | 8.73 ms ✓ | 8.91 ms ✓ | **8.70 ms ✓** | 8.74 ms ✓ | 8.91 ms ✓ | 8.78 ms ✓ | 50.2 ms ✓ | 66.6 ms ✓ | 64.5 ms ✓ | 72.7 ms ✓ | 14.0 ms ✓ | 15.4 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 9.45 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | **85.5 ms ✓** | 87.3 ms ⚠ | 86.7 ms ✓ | 87.5 ms ✓ | 87.1 ms ✓ | 87.5 ms ✓ | 93.1 ms ✓ | 566.4 ms ✓ | 591.4 ms ✓ | 702.8 ms ✓ | 714.3 ms ✓ | 141.5 ms ✓ | 142.8 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 90.3 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## aom_preprocess  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.03 ms ✓ | 1.04 ms ✓ | 1.05 ms ✓ | 1.03 ms ✓ | 1.08 ms ✓ | 1.04 ms ✓ | **1.02 ms ✓** | 3.14 ms ✓ | 4.26 ms ✓ | 4.58 ms ✓ | 4.53 ms ✓ | 1.70 ms ✓ | 2.08 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.40 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## approximate_press  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.08 ms ✓ | 1.09 ms ✓ | **1.06 ms ✓** | 1.19 ms ✓ | 1.15 ms ✓ | 1.10 ms ✓ | 1.11 ms ✓ | 3.43 ms ✓ | 4.57 ms ✓ | 5.03 ms ✓ | 4.75 ms ✓ | 1.89 ms ✓ | 2.13 ms ✓ | — | — | — | — | — | — | — | 35.9 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | 9.12 ms ⚠ | 9.05 ms ⚠ | 9.13 ms ⚠ | 9.25 ms ⚠ | **9.17 ms ✓** | 9.40 ms ✓ | 9.34 ms ✓ | 45.9 ms ✓ | 60.9 ms ✓ | 66.3 ms ✓ | 63.5 ms ✓ | 14.3 ms ✓ | 14.5 ms ✓ | — | — | — | — | — | — | — | 227.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | 99.4 ms ⚠ | 97.6 ms ⚠ | 96.6 ms ⚠ | 97.2 ms ⚠ | 99.4 ms ✓ | 112.8 ms ✓ | **98.0 ms ✓** | 549.1 ms ✓ | 677.1 ms ✓ | 652.0 ms ✓ | 628.7 ms ✓ | 149.2 ms ✓ | 150.9 ms ✓ | — | — | — | — | — | — | — | 17.0 s ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## approximate_press  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.12 ms ✓ | 1.09 ms ✓ | **1.06 ms ✓** | 1.11 ms ✓ | 1.09 ms ✓ | 1.14 ms ✓ | 1.13 ms ✓ | 3.51 ms ✓ | 4.82 ms ✓ | 4.63 ms ✓ | 4.77 ms ✓ | 1.80 ms ✓ | 2.18 ms ✓ | — | — | — | — | — | — | — | 35.9 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## bagging_pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | **1.19 ms ✓** | 1.21 ms ✓ | 1.21 ms ✓ | 1.22 ms ✓ | 1.20 ms ✓ | 1.67 ms ✓ | 1.32 ms ✓ | 3.00 ms ✓ | 4.24 ms ✓ | 4.49 ms ✓ | 4.22 ms ✓ | 1.89 ms ✓ | 2.26 ms ✓ | — | — | — | — | — | — | 9.38 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | 10.5 ms ✓ | **10.2 ms ✓** | 10.3 ms ✓ | 10.5 ms ✓ | 10.4 ms ✓ | 10.5 ms ✓ | 10.8 ms ✓ | 43.8 ms ✓ | 66.4 ms ✓ | 70.3 ms ✓ | 60.8 ms ✓ | 15.5 ms ✓ | 16.3 ms ✓ | — | — | — | — | — | — | 21.7 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | 121.0 ms ✓ | **116.9 ms ✓** | 118.8 ms ✓ | 119.3 ms ✓ | 120.4 ms ✓ | 122.8 ms ✓ | 118.6 ms ✓ | 503.5 ms ✓ | 701.4 ms ✓ | 675.9 ms ✓ | 760.3 ms ✓ | 173.0 ms ✓ | 176.7 ms ✓ | — | — | — | — | — | — | 171.7 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## bagging_pls  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.19 ms ✓ | 1.18 ms ✓ | **1.17 ms ✓** | 1.21 ms ✓ | 1.22 ms ✓ | 1.26 ms ✓ | 1.36 ms ✓ | 3.28 ms ✓ | 4.65 ms ✓ | 4.75 ms ✓ | 4.39 ms ✓ | 1.88 ms ✓ | 2.42 ms ✓ | — | — | — | — | — | — | 9.16 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## bipls_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 3.67 ms ⚠ | 2.94 ms ⚠ | 3.42 ms ⚠ | **3.18 ms ✓** | 3.65 ms ✓ | 3.62 ms ✓ | 4.05 ms ✓ | 6.05 ms ✓ | 7.32 ms ✓ | 7.41 ms ✓ | 7.26 ms ✓ | 4.30 ms ✓ | 4.89 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | 415.7 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## bipls_select  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 3.49 ms ⚠ | 2.93 ms ⚠ | 3.62 ms ⚠ | 3.01 ms ⚠ | 3.54 ms ✓ | **3.51 ms ✓** | 3.70 ms ✓ | 5.94 ms ✓ | 7.08 ms ✓ | 7.24 ms ✓ | 7.13 ms ✓ | 4.12 ms ✓ | 4.66 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | 372.7 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## boosting_pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.29 ms ✓ | 1.25 ms ✓ | **1.25 ms ✓** | 1.30 ms ✓ | 1.33 ms ✓ | 1.72 ms ✓ | 2.42 ms ✓ | 4.88 ms ✓ | 5.66 ms ✓ | 5.96 ms ✓ | 5.41 ms ✓ | 2.14 ms ✓ | 3.75 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 17.5 ms ✓ | — | — | — | — | — | — |
| 100×500 | 13.6 ms ✓ | 11.0 ms ✓ | 13.8 ms ✓ | 11.4 ms ✓ | **10.9 ms ✓** | 11.3 ms ✓ | 11.2 ms ✓ | 50.0 ms ✓ | 67.9 ms ✓ | 63.1 ms ✓ | 63.8 ms ✓ | 16.0 ms ✓ | 16.8 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 113.3 ms ✓ | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | 121.7 ms ✓ | **119.4 ms ✓** | 119.9 ms ✓ | 122.1 ms ✓ | 120.7 ms ✓ | 120.5 ms ✓ | 124.7 ms ✓ | 552.8 ms ✓ | 724.0 ms ✓ | 698.7 ms ✓ | 703.6 ms ✓ | 164.9 ms ✓ | 174.4 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 927.1 ms ✓ | — | — | — | — | — | — |


## boosting_pls  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.23 ms ✓ | **1.17 ms ✓** | 1.17 ms ✓ | 1.21 ms ✓ | 1.24 ms ✓ | 1.18 ms ✓ | 1.38 ms ✓ | 3.51 ms ✓ | 4.65 ms ✓ | 4.63 ms ✓ | 4.84 ms ✓ | 2.03 ms ✓ | 2.67 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 13.2 ms ✓ | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## bve_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 46.0 ms ⚠ | 35.7 ms ⚠ | 44.0 ms ⚠ | 37.9 ms ⚠ | 45.6 ms ✓ | 46.0 ms ✓ | 46.4 ms ✓ | 48.8 ms ✓ | 50.4 ms ✓ | 50.9 ms ✓ | 50.9 ms ✓ | 46.7 ms ✓ | 46.8 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **27.3 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## bve_select  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 45.9 ms ⚠ | 35.8 ms ⚠ | 43.6 ms ⚠ | 37.9 ms ⚠ | 45.6 ms ✓ | 45.7 ms ✓ | 46.1 ms ✓ | 50.5 ms ✓ | 51.9 ms ✓ | 52.2 ms ✓ | 51.2 ms ✓ | 46.2 ms ✓ | 46.3 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **27.6 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## cars_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.57 ms ⚠ | 1.42 ms ⚠ | 1.53 ms ⚠ | **1.49 ms ✓** | 1.59 ms ✓ | 1.56 ms ✓ | 1.85 ms ✓ | 4.05 ms ✓ | 5.01 ms ✓ | 5.35 ms ✓ | 5.46 ms ✓ | 2.17 ms ✓ | 2.59 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 75.0 ms ✓ |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## cars_select  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.51 ms ✓ | 1.44 ms ✓ | 1.45 ms ✓ | **1.40 ms ✓** | 1.46 ms ✓ | 1.49 ms ✓ | 1.68 ms ✓ | 3.73 ms ✓ | 5.65 ms ✓ | 5.07 ms ✓ | 5.62 ms ✓ | 2.11 ms ✓ | 2.53 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 67.8 ms ✓ |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## continuum_regression  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.13 ms ✓ | 1.05 ms ✓ | 1.06 ms ✓ | 1.07 ms ✓ | **1.02 ms ✓** | 1.04 ms ✓ | 1.19 ms ✓ | 2.87 ms ✓ | 4.52 ms ✓ | 4.39 ms ✓ | 4.77 ms ✓ | 1.72 ms ✓ | 2.39 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 32.9 ms ✓ | — | — | — | — | — | — | — |
| 100×500 | 8.60 ms ⚠ | 8.75 ms ⚠ | 8.73 ms ⚠ | 8.64 ms ⚠ | **8.60 ms ✓** | 8.62 ms ✓ | 8.86 ms ✓ | 40.3 ms ✓ | 57.0 ms ✓ | 56.9 ms ✓ | 63.5 ms ✓ | 13.7 ms ✓ | 14.5 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 150.5 ms ✓ | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | 87.9 ms ⚠ | 86.8 ms ⚠ | 87.0 ms ⚠ | 86.7 ms ⚠ | **86.6 ms ✓** | 88.7 ms ✓ | 88.6 ms ✓ | 469.4 ms ✓ | 591.6 ms ✓ | 540.7 ms ✓ | 618.4 ms ✓ | 133.6 ms ✓ | 137.3 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 3.2 s ✓ | — | — | — | — | — | — | — |


## continuum_regression  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.09 ms ✓ | 1.13 ms ✓ | 1.05 ms ✓ | 1.10 ms ✓ | **1.05 ms ✓** | 1.05 ms ✓ | 1.21 ms ✓ | 3.34 ms ✓ | 4.87 ms ✓ | 4.84 ms ✓ | 4.97 ms ✓ | 1.79 ms ✓ | 2.13 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 45.3 ms ✓ | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## cppls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.54 ms ⚠ | 1.03 ms ⚠ | 1.03 ms ⚠ | 1.90 ms ✓ | 1.60 ms ✓ | **1.10 ms ✓** | 1.25 ms ✓ | 4.00 ms ✓ | 5.29 ms ✓ | 11.5 ms ✓ | 5.70 ms ✓ | 1.80 ms ✓ | 2.16 ms ✓ | — | — | 9.80 ms ✓ | — | — | — | — | 9.32 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | 8.76 ms ✓ | 8.41 ms ✓ | 8.62 ms ✓ | 8.52 ms ✓ | **8.40 ms ✓** | 8.48 ms ✓ | 9.22 ms ✓ | 48.8 ms ✓ | 68.6 ms ✓ | 105.5 ms ✓ | 76.7 ms ✓ | 13.5 ms ✓ | 13.9 ms ✓ | — | — | 97.9 ms ✓ | — | — | — | — | 93.8 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | 43.8 ms ⚠ | 44.1 ms ⚠ | 43.1 ms ⚠ | 42.8 ms ⚠ | 44.2 ms ✓ | **43.2 ms ✓** | 45.1 ms ✓ | 392.1 ms ✓ | 580.4 ms ✓ | 980.6 ms ✓ | 580.3 ms ✓ | 67.1 ms ✓ | 68.1 ms ✓ | — | — | 774.7 ms ✓ | — | — | — | — | 753.0 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | 88.2 ms ⚠ | 87.4 ms ⚠ | 88.1 ms ⚠ | 87.7 ms ⚠ | 88.9 ms ✓ | **87.2 ms ✓** | 89.8 ms ✓ | 500.7 ms ✓ | 649.2 ms ✓ | 703.0 ms ✓ | 482.8 ms ✓ | 131.8 ms ✓ | 133.3 ms ✓ | — | — | 589.6 ms ✓ | — | — | — | — | 562.7 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## cppls  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.01 ms ✓ | 1.02 ms ✓ | 1.01 ms ✓ | **0.99 ms ✓** | 1.03 ms ✓ | 1.02 ms ✓ | 1.22 ms ✓ | 3.59 ms ✓ | 4.74 ms ✓ | 9.11 ms ✓ | 4.61 ms ✓ | 1.66 ms ✓ | 2.13 ms ✓ | — | — | 8.23 ms ✓ | — | — | — | — | 8.51 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## di_pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.04 ms ⚠ | 1.02 ms ⚠ | 1.02 ms ⚠ | 1.02 ms ⚠ | 1.07 ms ✓ | **1.05 ms ✓** | 1.44 ms ✓ | 7.08 ms ✓ | 8.09 ms ✓ | 11.0 ms ✓ | 9.37 ms ✓ | 3.52 ms ✓ | 3.74 ms ✓ | — | — | — | — | — | — | — | — | — | — | 4.69 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | 8.30 ms ⚠ | 9.29 ms ⚠ | 8.30 ms ⚠ | 8.68 ms ⚠ | **8.30 ms ✓** | 8.50 ms ✓ | 8.78 ms ✓ | 104.0 ms ✓ | 110.6 ms ✓ | 123.0 ms ✓ | 121.8 ms ✓ | 26.6 ms ✓ | 26.6 ms ✓ | — | — | — | — | — | — | — | — | — | — | 198.7 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | 43.3 ms ⚠ | 42.1 ms ⚠ | 42.6 ms ⚠ | 43.2 ms ⚠ | 43.2 ms ✓ | **43.0 ms ✓** | 44.0 ms ✓ | 956.1 ms ✓ | 1.1 s ✓ | 1.2 s ✓ | 1.1 s ✓ | 132.1 ms ✓ | 132.0 ms ✓ | — | — | — | — | — | — | — | — | — | — | 50.7 s ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | 84.2 ms ⚠ | 83.7 ms ⚠ | 85.1 ms ⚠ | 83.9 ms ⚠ | 85.0 ms ✓ | **84.6 ms ✓** | 85.4 ms ✓ | 1.0 s ✓ | 985.4 ms ✓ | 1.1 s ✓ | 955.9 ms ✓ | 252.1 ms ✓ | 254.7 ms ✓ | — | — | — | — | — | — | — | — | — | — | 359.5 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## di_pls  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.15 ms ⚠ | 1.15 ms ⚠ | 1.05 ms ⚠ | 1.06 ms ⚠ | **1.05 ms ✓** | 1.12 ms ✓ | 1.35 ms ✓ | 7.78 ms ✓ | 9.06 ms ✓ | 8.90 ms ✓ | 9.00 ms ✓ | 3.41 ms ✓ | 4.06 ms ✓ | — | — | — | — | — | — | — | — | — | — | 3.61 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | 8.39 ms ⚠ | 8.24 ms ⚠ | 8.54 ms ⚠ | 8.25 ms ⚠ | 8.47 ms ✓ | **8.42 ms ✓** | 8.77 ms ✓ | 98.1 ms ✓ | 128.5 ms ✓ | 116.1 ms ✓ | 126.3 ms ✓ | 26.8 ms ✓ | 27.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | 1.3 s ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | 87.7 ms ⚠ | 111.0 ms ✓ | 89.5 ms ⚠ | 105.2 ms ✓ | 88.4 ms ✓ | **88.3 ms ✓** | 115.4 ms ✓ | 1.2 s ✓ | 1.1 s ✓ | 1.2 s ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## ds  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.31 ms ✓ | **1.29 ms ✓** | 1.33 ms ✓ | 1.33 ms ✓ | 1.30 ms ✓ | 1.32 ms ✓ | 1.49 ms ✓ | 7.91 ms ✓ | 9.82 ms ✓ | 8.87 ms ✓ | 9.48 ms ✓ | 3.62 ms ✓ | 4.26 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 24.5 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | 92.2 ms ⚠ | 92.8 ms ⚠ | 90.3 ms ⚠ | 91.4 ms ⚠ | **93.2 ms ✓** | 93.7 ms ✓ | 97.3 ms ✓ | 202.5 ms ✓ | 228.5 ms ✓ | 220.5 ms ✓ | 215.4 ms ✓ | 111.8 ms ✓ | 112.2 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 227.7 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | 868.5 ms ⚠ | 872.4 ms ⚠ | 837.5 ms ⚠ | 853.7 ms ⚠ | **848.4 ms ✓** | 871.3 ms ✓ | 873.3 ms ✓ | 2.0 s ✓ | 1.9 s ✓ | 1.8 s ✓ | 1.9 s ✓ | 962.1 ms ✓ | 1.0 s ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## ds  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.31 ms ✓ | 1.31 ms ✓ | 1.30 ms ✓ | 1.34 ms ✓ | **1.27 ms ✓** | 1.36 ms ✓ | 1.47 ms ✓ | 7.87 ms ✓ | 9.43 ms ✓ | 9.13 ms ✓ | 8.44 ms ✓ | 3.82 ms ✓ | 4.25 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 24.5 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## ecr  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.31 ms ⚠ | 1.24 ms ⚠ | 1.20 ms ⚠ | **1.23 ms ✓** | 1.46 ms ✓ | 1.34 ms ✓ | 1.53 ms ✓ | 3.30 ms ✓ | 4.99 ms ✓ | 4.80 ms ✓ | 4.44 ms ✓ | 2.02 ms ✓ | 2.37 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 13.0 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## ecr  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.31 ms ✓ | **1.20 ms ✓** | 1.24 ms ✓ | 1.23 ms ✓ | 1.35 ms ✓ | 1.36 ms ✓ | 1.52 ms ✓ | 3.40 ms ✓ | 4.40 ms ✓ | 4.46 ms ✓ | 4.50 ms ✓ | 2.01 ms ✓ | 2.40 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 13.4 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## emcuve_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.38 ms ⚠ | 1.34 ms ⚠ | 1.36 ms ⚠ | 1.31 ms ⚠ | **1.36 ms ✓** | 1.37 ms ✓ | 1.50 ms ✓ | 4.32 ms ✓ | 4.74 ms ✓ | 4.78 ms ✓ | 4.52 ms ✓ | 2.36 ms ✓ | 2.44 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 401.7 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## emcuve_select  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.42 ms ⚠ | 1.29 ms ⚠ | 1.37 ms ⚠ | 1.28 ms ⚠ | 1.39 ms ✓ | **1.36 ms ✓** | 1.53 ms ✓ | 3.59 ms ✓ | 5.01 ms ✓ | 5.20 ms ✓ | 4.75 ms ✓ | 2.00 ms ✓ | 2.31 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 398.6 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## fused_sparse_pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 0.97 ms — | 1.04 ms — | 1.01 ms — | 1.01 ms — | 1.05 ms ✓ | **1.01 ms ✓** | 1.19 ms ✓ | 3.40 ms ✓ | 4.80 ms ✓ | 5.25 ms ✓ | 5.39 ms ✓ | 1.91 ms ✓ | 1.99 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | 8.34 ms — | 8.60 ms — | 8.30 ms — | 8.20 ms — | **8.31 ms ✓** | 8.37 ms ✓ | 8.70 ms ✓ | 45.3 ms ✓ | 60.7 ms ✓ | 67.4 ms ✓ | 56.1 ms ✓ | 13.1 ms ✓ | 13.8 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | 83.1 ms — | 85.4 ms — | 86.4 ms — | 87.1 ms — | **85.0 ms ✓** | 85.5 ms ✓ | 87.5 ms ✓ | — | — | — | — | 130.1 ms ✓ | 134.2 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## fused_sparse_pls  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.00 ms — | 1.02 ms — | 1.01 ms — | 1.02 ms — | 1.04 ms ✓ | **1.02 ms ✓** | 1.22 ms ✓ | 3.79 ms ✓ | 4.72 ms ✓ | 4.97 ms ✓ | 4.90 ms ✓ | 1.75 ms ✓ | 2.27 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## ga_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 2.89 ms ⚠ | 2.82 ms ⚠ | 2.77 ms ⚠ | **2.78 ms ✓** | 2.96 ms ✓ | 3.05 ms ✓ | 3.09 ms ✓ | 5.46 ms ✓ | 6.65 ms ✓ | 6.65 ms ✓ | 7.19 ms ✓ | 3.41 ms ✓ | 3.89 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 217.4 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## ga_select  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 2.86 ms ⚠ | 2.54 ms ⚠ | 2.77 ms ⚠ | 2.72 ms ⚠ | 2.85 ms ✓ | **2.81 ms ✓** | 3.03 ms ✓ | 5.15 ms ✓ | 7.21 ms ✓ | 6.75 ms ✓ | 6.46 ms ✓ | 3.43 ms ✓ | 3.95 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 173.6 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## gpr_pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.32 ms ✓ | 1.30 ms ✓ | 2.06 ms ✓ | **1.29 ms ✓** | 1.32 ms ✓ | 1.36 ms ✓ | 1.48 ms ✓ | 3.44 ms ✓ | 5.51 ms ✓ | 5.44 ms ✓ | 4.84 ms ✓ | 2.11 ms ✓ | 2.41 ms ✓ | 4.70 ms ✗ | — | — | — | — | — | 2.29 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | 8.56 ms ✓ | 8.55 ms ✓ | 8.68 ms ✓ | **8.38 ms ✓** | 9.06 ms ✓ | 8.62 ms ✓ | 8.92 ms ✓ | 42.2 ms ✓ | 61.3 ms ✓ | 69.8 ms ✓ | 70.4 ms ✓ | 13.6 ms ✓ | 14.2 ms ✓ | 14.4 ms ✗ | — | — | — | — | — | 10.2 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | 385.9 ms ✓ | 388.1 ms ✓ | 386.8 ms ✓ | 382.6 ms ✓ | 389.1 ms ✓ | 388.0 ms ✓ | **375.7 ms ✓** | 912.8 ms ✓ | 974.5 ms ✓ | 908.4 ms ✓ | 930.3 ms ✓ | 414.8 ms ✓ | 420.1 ms ✓ | 1.8 s ⚠ | — | — | — | — | — | 414.3 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## gpr_pls  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.27 ms ✓ | 1.27 ms ✓ | **1.26 ms ✓** | 1.27 ms ✓ | 1.31 ms ✓ | 1.29 ms ✓ | 1.44 ms ✓ | 3.71 ms ✓ | 4.70 ms ✓ | 4.69 ms ✓ | 4.53 ms ✓ | 1.89 ms ✓ | 2.31 ms ✓ | 6.93 ms ✗ | — | — | — | — | — | 2.12 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## group_sparse_pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.04 ms ✓ | 1.00 ms ✓ | **0.98 ms ✓** | 0.99 ms ✓ | 1.00 ms ✓ | 1.03 ms ✓ | 1.17 ms ✓ | 3.73 ms ✓ | 4.76 ms ✓ | 4.88 ms ✓ | 4.78 ms ✓ | 1.88 ms ✓ | 2.08 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 17.6 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | 8.48 ms ⚠ | 8.44 ms ⚠ | 8.27 ms ⚠ | 8.25 ms ⚠ | **8.29 ms ✓** | 8.32 ms ✓ | 8.67 ms ✓ | 48.2 ms ✓ | 70.0 ms ✓ | 65.9 ms ✓ | 65.0 ms ✓ | 13.2 ms ✓ | 13.9 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 113.3 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | 83.7 ms ⚠ | 86.0 ms ⚠ | 84.9 ms ⚠ | 85.3 ms ⚠ | **83.7 ms ✓** | 84.3 ms ✓ | 84.4 ms ✓ | — | — | — | — | 133.8 ms ✓ | 137.6 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## group_sparse_pls  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.03 ms ✓ | **1.01 ms ✓** | 1.07 ms ✓ | 1.04 ms ✓ | 1.09 ms ✓ | 1.06 ms ✓ | 1.22 ms ✓ | 3.56 ms ✓ | 5.15 ms ✓ | 5.19 ms ✓ | 5.09 ms ✓ | 1.71 ms ✓ | 2.05 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 18.3 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## interval_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 2.06 ms ✓ | **1.42 ms ✓** | 1.42 ms ✓ | 1.47 ms ✓ | 2.17 ms ✓ | 1.49 ms ✓ | 1.72 ms ✓ | 3.86 ms ✓ | 5.25 ms ✓ | 5.40 ms ✓ | 5.12 ms ✓ | 2.15 ms ✓ | 2.61 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | 482.7 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## interval_select  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.51 ms ⚠ | 1.47 ms ⚠ | 1.63 ms ⚠ | 1.58 ms ⚠ | 1.48 ms ✓ | **1.45 ms ✓** | 1.63 ms ✓ | 3.51 ms ✓ | 4.78 ms ✓ | 5.10 ms ✓ | 4.88 ms ✓ | 2.12 ms ✓ | 2.48 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | 428.3 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## ipw_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.18 ms ⚠ | 1.16 ms ⚠ | 1.14 ms ⚠ | **1.20 ms ✓** | 2.09 ms ✓ | 1.26 ms ✓ | 1.32 ms ✓ | 3.46 ms ✓ | 5.84 ms ✓ | 4.90 ms ✓ | 4.87 ms ✓ | 1.90 ms ✓ | 2.18 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 18.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## ipw_select  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.13 ms ✓ | 1.15 ms ✓ | **1.12 ms ✓** | 1.13 ms ✓ | 1.16 ms ✓ | 1.13 ms ✓ | 1.37 ms ✓ | 2.99 ms ✓ | 4.96 ms ✓ | 4.56 ms ✓ | 5.05 ms ✓ | 1.76 ms ✓ | 2.00 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 16.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## irf_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 2.76 ms ⚠ | 2.26 ms ⚠ | 2.45 ms ⚠ | **2.26 ms ✓** | 2.56 ms ✓ | 2.53 ms ✓ | 3.15 ms ✓ | 5.11 ms ✓ | 6.21 ms ✓ | 6.50 ms ✓ | 6.92 ms ✓ | 3.25 ms ✓ | 3.71 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 980.6 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## irf_select  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 2.47 ms ⚠ | 2.18 ms ⚠ | 2.34 ms ⚠ | 2.20 ms ⚠ | **2.42 ms ✓** | 2.46 ms ✓ | 2.65 ms ✓ | 4.75 ms ✓ | 6.54 ms ✓ | 6.62 ms ✓ | 6.12 ms ✓ | 3.14 ms ✓ | 3.55 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 870.4 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## iriv_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 202.3 ms ✓ | 167.5 ms ✓ | 184.5 ms ✓ | **162.5 ms ✓** | 210.1 ms ✓ | 201.4 ms ✓ | 211.2 ms ✓ | 212.6 ms ✓ | 208.8 ms ✓ | 216.3 ms ✓ | 222.8 ms ✓ | 206.3 ms ✓ | 206.7 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 13.3 s ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## iriv_select  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 197.1 ms ⚠ | 154.4 ms ⚠ | 182.3 ms ⚠ | 160.3 ms ⚠ | **198.0 ms ✓** | 199.7 ms ✓ | 198.4 ms ✓ | 210.3 ms ✓ | 208.4 ms ✓ | 209.9 ms ✓ | 209.4 ms ✓ | 202.3 ms ✓ | 201.8 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 11.9 s ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## kernel_pls_rbf  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.43 ms ✓ | **1.40 ms ✓** | 1.44 ms ✓ | 1.42 ms ✓ | 1.49 ms ✓ | 1.50 ms ✓ | 1.61 ms ✓ | 3.67 ms ✓ | 5.22 ms ✓ | 5.48 ms ✓ | 4.95 ms ✓ | 2.22 ms ✓ | 2.80 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | 20.9 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | 12.1 ms ✓ | 12.2 ms ✓ | 12.1 ms ✓ | **12.0 ms ✓** | 12.2 ms ✓ | 12.0 ms ✓ | 12.6 ms ✓ | 44.5 ms ✓ | 76.6 ms ✓ | 82.1 ms ✓ | 75.1 ms ✓ | 17.1 ms ✓ | 18.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | 113.5 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | 518.6 ms ⚠ | 514.5 ms ⚠ | 520.7 ms ⚠ | 519.5 ms ⚠ | 526.8 ms ✓ | **492.7 ms ✓** | 501.7 ms ✓ | 1.0 s ✓ | 1.1 s ✓ | 1.1 s ✓ | 1.1 s ✓ | 581.1 ms ✓ | 573.6 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | 1.3 s ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## kernel_pls_rbf  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.43 ms ✓ | 1.46 ms ✓ | 1.44 ms ✓ | 1.45 ms ✓ | **1.39 ms ✓** | 1.44 ms ✓ | 1.62 ms ✓ | 3.70 ms ✓ | 5.58 ms ✓ | 5.46 ms ✓ | 5.27 ms ✓ | 2.14 ms ✓ | 2.72 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | 27.0 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## lw_pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 2.49 ms ✓ | **2.13 ms ✓** | 2.39 ms ✓ | 2.21 ms ✓ | 2.59 ms ✓ | 2.52 ms ✓ | 2.78 ms ✓ | 4.88 ms ✓ | 6.44 ms ✓ | 6.72 ms ✓ | 7.62 ms ✓ | 3.23 ms ✓ | 3.77 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 11.2 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | 21.7 ms ✓ | **18.0 ms ✓** | 21.6 ms ✓ | 18.3 ms ✓ | 22.0 ms ✓ | 21.7 ms ✓ | 22.4 ms ✓ | 58.4 ms ✓ | 77.0 ms ✓ | 79.0 ms ✓ | 81.3 ms ✓ | 27.9 ms ✓ | 27.3 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 47.7 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | 441.5 ms ⚠ | 394.0 ms ⚠ | 456.1 ms ⚠ | 422.4 ms ⚠ | 464.4 ms ✓ | 461.3 ms ✓ | **448.4 ms ✓** | — | — | — | — | 513.5 ms ✓ | 508.0 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 5.3 s ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## lw_pls  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 2.63 ms ✓ | **2.20 ms ✓** | 2.50 ms ✓ | 2.31 ms ✓ | 2.56 ms ✓ | 2.61 ms ✓ | 2.84 ms ✓ | 4.88 ms ✓ | 6.34 ms ✓ | 6.36 ms ✓ | 6.63 ms ✓ | 3.63 ms ✓ | 3.75 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 11.5 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## mb_pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.04 ms ✓ | 1.04 ms ✓ | 1.03 ms ✓ | 1.03 ms ✓ | 1.69 ms ✓ | **1.02 ms ✓** | 1.27 ms ✓ | 3.28 ms ✓ | 4.70 ms ✓ | 4.74 ms ✓ | 4.98 ms ✓ | 1.80 ms ✓ | 3.10 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.61 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | 8.64 ms ✓ | 8.50 ms ✓ | **8.43 ms ✓** | 8.51 ms ✓ | 8.47 ms ✓ | 8.67 ms ✓ | 8.74 ms ✓ | 40.7 ms ✓ | 61.2 ms ✓ | 66.1 ms ✓ | 67.5 ms ✓ | 13.0 ms ✓ | 13.6 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 9.30 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | 86.4 ms ✓ | **85.1 ms ✓** | 86.2 ms ✓ | 86.0 ms ✓ | 86.3 ms ✓ | 87.7 ms ✓ | 88.7 ms ✓ | 456.6 ms ✓ | 639.4 ms ✓ | 563.3 ms ✓ | 631.7 ms ✓ | 134.8 ms ✓ | 135.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 85.8 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## mb_pls  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.13 ms ✓ | 1.05 ms ✓ | 1.12 ms ✓ | 1.04 ms ✓ | 1.04 ms ✓ | **1.03 ms ✓** | 1.24 ms ✓ | 3.36 ms ✓ | 4.85 ms ✓ | 4.76 ms ✓ | 4.98 ms ✓ | 1.71 ms ✓ | 2.16 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.73 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## mir_pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.12 ms — | 1.01 ms — | 1.11 ms — | 0.99 ms — | 1.60 ms ✓ | **1.00 ms ✓** | 1.23 ms ✓ | 3.93 ms ✓ | 4.96 ms ✓ | 5.08 ms ✓ | 5.23 ms ✓ | 1.69 ms ✓ | 2.14 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | 8.15 ms — | 8.39 ms — | 8.45 ms — | 8.16 ms — | 9.60 ms ✓ | **8.30 ms ✓** | 8.50 ms ✓ | 51.2 ms ✓ | 68.6 ms ✓ | 73.3 ms ✓ | 71.2 ms ✓ | 13.1 ms ✓ | 13.7 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | 83.4 ms — | 83.3 ms — | 82.1 ms — | 83.1 ms — | 83.6 ms ✓ | **82.9 ms ✓** | 86.1 ms ✓ | 552.0 ms ✓ | 706.3 ms ✓ | 723.4 ms ✓ | 657.2 ms ✓ | 129.1 ms ✓ | 135.6 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## mir_pls  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.04 ms — | 1.01 ms — | 1.01 ms — | 1.01 ms — | **1.04 ms ✓** | 1.07 ms ✓ | 1.24 ms ✓ | 3.28 ms ✓ | 4.93 ms ✓ | 5.03 ms ✓ | 5.14 ms ✓ | 1.67 ms ✓ | 2.16 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## missing_aware_nipals  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.01 ms ✓ | 1.00 ms ✓ | 1.01 ms ✓ | 1.03 ms ✓ | **1.00 ms ✓** | 1.01 ms ✓ | 1.19 ms ✓ | 4.02 ms ✓ | 5.16 ms ✓ | 4.99 ms ✓ | 4.75 ms ✓ | 1.67 ms ✓ | 2.31 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 12.8 ms ✓ | — | — | — | — |
| 100×500 | 8.33 ms ✓ | **8.31 ms ✓** | 8.35 ms ✓ | 8.60 ms ✓ | 8.58 ms ✓ | 8.36 ms ✓ | 8.71 ms ✓ | 55.1 ms ✓ | 77.7 ms ✓ | 65.0 ms ✓ | 67.3 ms ✓ | 13.3 ms ✓ | 13.9 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 112.2 ms ✓ | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | 84.9 ms ⚠ | 84.3 ms ⚠ | 84.7 ms ⚠ | 87.2 ms ⚠ | 86.2 ms ✓ | **84.7 ms ✓** | 86.7 ms ✓ | 496.8 ms ✓ | 662.0 ms ✓ | 649.9 ms ✓ | 670.5 ms ✓ | 132.7 ms ✓ | 134.6 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 891.7 ms ✓ | — | — | — | — |


## missing_aware_nipals  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | **0.99 ms ✓** | 1.17 ms ✓ | 1.02 ms ✓ | 1.02 ms ✓ | 1.05 ms ✓ | 1.00 ms ✓ | 1.22 ms ✓ | 3.46 ms ✓ | 4.83 ms ✓ | 5.27 ms ✓ | 5.00 ms ✓ | 1.69 ms ✓ | 2.27 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 12.5 ms ✓ | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## n_pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.04 ms ✓ | 1.02 ms ✓ | 1.06 ms ✓ | 1.04 ms ✓ | 1.05 ms ✓ | **1.02 ms ✓** | 1.20 ms ✓ | 3.18 ms ✓ | 4.38 ms ✓ | 5.05 ms ✓ | 4.36 ms ✓ | 1.68 ms ✓ | 2.19 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 16.7 ms ✓ | — | — |
| 100×500 | 8.47 ms ✓ | 8.43 ms ✓ | 8.73 ms ✓ | **8.43 ms ✓** | 8.49 ms ✓ | 8.45 ms ✓ | 9.28 ms ✓ | 39.4 ms ✓ | 57.6 ms ✓ | 61.2 ms ✓ | 55.3 ms ✓ | 13.2 ms ✓ | 14.0 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 36.6 ms ✓ | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | 84.7 ms ⚠ | 84.4 ms ⚠ | 84.4 ms ⚠ | 86.5 ms ⚠ | 84.9 ms ✓ | **84.7 ms ✓** | 86.1 ms ✓ | 472.7 ms ✓ | 585.7 ms ✓ | 617.9 ms ✓ | 618.4 ms ✓ | 132.5 ms ✓ | 134.9 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 271.8 ms ✓ | — | — |


## n_pls  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.08 ms ✓ | 1.03 ms ✓ | **1.03 ms ✓** | 1.08 ms ✓ | 1.06 ms ✓ | 1.05 ms ✓ | 1.25 ms ✓ | 3.42 ms ✓ | 4.65 ms ✓ | 4.53 ms ✓ | 5.04 ms ✓ | 1.80 ms ✓ | 2.19 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 16.3 ms ✓ | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## o2pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.16 ms ✓ | 1.13 ms ✓ | **1.10 ms ✓** | 1.14 ms ✓ | 1.14 ms ✓ | 1.10 ms ✓ | 2.07 ms ✓ | 3.66 ms ✓ | 5.78 ms ✓ | 5.07 ms ✓ | 5.84 ms ✓ | 2.23 ms ✓ | 2.89 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 12.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | 9.59 ms ✓ | **9.32 ms ✓** | 9.98 ms ✓ | 9.47 ms ✓ | 11.3 ms ✓ | 12.6 ms ✓ | 12.1 ms ✓ | 47.4 ms ✓ | 74.6 ms ✓ | 85.9 ms ✓ | 73.0 ms ✓ | 14.7 ms ✓ | 15.6 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 110.6 ms ✓ | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | 96.4 ms ⚠ | 95.3 ms ⚠ | 94.9 ms ⚠ | 96.3 ms ⚠ | 96.6 ms ✓ | **95.8 ms ✓** | 99.2 ms ✓ | 524.1 ms ✓ | 587.0 ms ✓ | 642.2 ms ✓ | 612.3 ms ✓ | 146.2 ms ✓ | 155.7 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.1 s ✓ | — | — | — | — | — | — | — | — | — | — | — |


## o2pls  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.15 ms ✓ | 1.10 ms ✓ | **1.09 ms ✓** | 1.14 ms ✓ | 1.15 ms ✓ | 1.16 ms ✓ | 1.35 ms ✓ | 3.91 ms ✓ | 4.91 ms ✓ | 4.74 ms ✓ | 5.05 ms ✓ | 1.92 ms ✓ | 2.55 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 7.94 ms ✓ | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## on_pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.12 ms ✓ | 1.16 ms ✓ | 1.12 ms ✓ | **1.11 ms ✓** | 1.22 ms ✓ | 1.14 ms ✓ | 1.21 ms ✓ | 3.22 ms ✓ | 4.90 ms ✓ | 4.73 ms ✓ | 4.70 ms ✓ | 1.82 ms ✓ | 2.24 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 9.00 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | 19.6 ms ✓ | 18.4 ms ✓ | **18.3 ms ✓** | 19.5 ms ✓ | 19.3 ms ✓ | 19.5 ms ✓ | 19.5 ms ✓ | 53.4 ms ✓ | 84.5 ms ✓ | 78.9 ms ✓ | 87.5 ms ✓ | 24.3 ms ✓ | 24.8 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 50.8 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | 172.5 ms ⚠ | 170.5 ms ⚠ | 177.5 ms ⚠ | 173.8 ms ⚠ | **171.7 ms ✓** | 173.4 ms ✓ | 174.9 ms ✓ | 628.3 ms ✓ | 809.2 ms ✓ | 799.5 ms ✓ | 808.0 ms ✓ | 222.0 ms ✓ | 218.3 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 800.3 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## on_pls  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.11 ms ✓ | 1.10 ms ✓ | **1.09 ms ✓** | 1.13 ms ✓ | 1.10 ms ✓ | 1.13 ms ✓ | 1.18 ms ✓ | 3.28 ms ✓ | 4.99 ms ✓ | 5.42 ms ✓ | 4.97 ms ✓ | 1.87 ms ✓ | 2.18 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 9.34 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## one_se_rule  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | **0.97 ms ✓** | 0.98 ms ✓ | 0.97 ms ✓ | 0.99 ms ✓ | 0.97 ms ✓ | 1.00 ms ✓ | 1.00 ms ✓ | 3.21 ms ✓ | 4.36 ms ✓ | 4.53 ms ✓ | 4.46 ms ✓ | 1.70 ms ✓ | 2.09 ms ✓ | — | — | — | — | — | — | — | 13.2 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | **7.81 ms ✓** | 8.47 ms ✓ | 8.06 ms ✓ | 7.99 ms ✓ | 7.84 ms ✓ | 8.00 ms ✓ | 7.89 ms ✓ | 57.6 ms ✓ | 71.1 ms ✓ | 76.5 ms ✓ | 69.6 ms ✓ | 12.9 ms ✓ | 13.5 ms ✓ | — | — | — | — | — | — | — | 78.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | 78.2 ms ⚠ | 77.6 ms ⚠ | 79.2 ms ⚠ | 78.0 ms ⚠ | 88.7 ms ✓ | **78.0 ms ✓** | 79.5 ms ✓ | 492.2 ms ✓ | 690.2 ms ✓ | 711.3 ms ✓ | 674.7 ms ✓ | 129.7 ms ✓ | 129.3 ms ✓ | — | — | — | — | — | — | — | 708.6 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## one_se_rule  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.00 ms ✓ | 1.11 ms ✓ | 1.02 ms ✓ | 1.10 ms ✓ | **0.93 ms ✓** | 0.98 ms ✓ | 1.00 ms ✓ | 3.20 ms ✓ | 4.81 ms ✓ | 4.74 ms ✓ | 4.96 ms ✓ | 1.64 ms ✓ | 2.06 ms ✓ | — | — | — | — | — | — | — | 11.8 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## opls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.02 ms ⚠ | 1.05 ms ⚠ | 1.09 ms ⚠ | **1.05 ms ✓** | 1.05 ms ✓ | 1.13 ms ✓ | 1.25 ms ✓ | 2.97 ms ✓ | 4.58 ms ✓ | 4.80 ms ✓ | 4.33 ms ✓ | 1.78 ms ✓ | 2.09 ms ✓ | — | — | — | 13.2 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 14.6 ms ✓ | — | — | — | — | — | — | — | — |
| 100×500 | 9.38 ms ✓ | 9.33 ms ✓ | 9.32 ms ✓ | **9.11 ms ✓** | 9.68 ms ✓ | 9.29 ms ✓ | 9.59 ms ✓ | 46.2 ms ✓ | 65.7 ms ✓ | 56.5 ms ✓ | 64.9 ms ✓ | 14.3 ms ✓ | 14.7 ms ✓ | — | — | — | 80.6 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 83.9 ms ✓ | — | — | — | — | — | — | — | — |
| 100×2500 | 91.5 ms ⚠ | 91.4 ms ⚠ | 91.8 ms ⚠ | 91.4 ms ⚠ | 94.2 ms ✓ | **92.8 ms ✓** | 94.0 ms ✓ | 428.1 ms ✓ | 612.4 ms ✓ | 613.7 ms ✓ | 611.9 ms ✓ | 125.2 ms ✓ | 131.5 ms ✓ | — | — | — | 592.6 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 598.7 ms ✓ | — | — | — | — | — | — | — | — |
| 1000×500 | 89.0 ms ⚠ | 88.1 ms ⚠ | 89.5 ms ⚠ | **89.0 ms ✓** | 90.4 ms ✓ | 89.7 ms ✓ | 150.8 ms ✓ | 605.0 ms ✓ | 580.6 ms ✓ | 582.6 ms ✓ | 574.6 ms ✓ | 139.2 ms ✓ | 139.1 ms ✓ | — | — | — | 936.4 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 933.6 ms ✓ | — | — | — | — | — | — | — | — |


## opls  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.03 ms ✓ | 1.01 ms ✓ | 1.07 ms ✓ | **1.00 ms ✓** | 1.06 ms ✓ | 1.02 ms ✓ | 1.25 ms ✓ | 3.49 ms ✓ | 4.29 ms ✓ | 4.30 ms ✓ | 4.67 ms ✓ | 1.77 ms ✓ | 2.03 ms ✓ | — | — | — | 13.4 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 12.4 ms ✓ | — | — | — | — | — | — | — | — |
| 100×500 | 9.66 ms ✓ | **9.21 ms ✓** | 9.41 ms ✓ | 9.27 ms ✓ | 9.34 ms ✓ | 9.58 ms ✓ | 9.73 ms ✓ | 44.2 ms ✓ | 66.2 ms ✓ | 59.7 ms ✓ | 61.5 ms ✓ | 14.4 ms ✓ | 14.8 ms ✓ | — | — | — | 85.6 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 85.9 ms ✓ | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | 88.9 ms ⚠ | 120.9 ms ✓ | 89.2 ms ⚠ | 110.9 ms ✓ | 89.0 ms ✓ | **87.7 ms ✓** | 88.9 ms ✓ | 564.8 ms ✓ | 578.4 ms ✓ | 600.8 ms ✓ | 629.0 ms ✓ | 162.8 ms ✓ | 160.1 ms ✓ | — | — | — | 1.0 s ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 954.9 ms ✓ | — | — | — | — | — | — | — | — |


## pcr  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.94 ms ✓ | 1.85 ms ✓ | **1.79 ms ✓** | 1.89 ms ✓ | 1.89 ms ✓ | 1.95 ms ✓ | 2.19 ms ✓ | 22.0 ms ✓ | 23.0 ms ✓ | 29.0 ms ✓ | 23.8 ms ✓ | 2.75 ms ✓ | 2.79 ms ✓ | 2.01 ms ⚠ | — | 7.40 ms ✓ | — | — | — | 2.12 ms ✓ | 8.13 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | 1.0 s ⚠ | 992.7 ms ⚠ | 1.1 s ⚠ | 1.0 s ⚠ | 1.0 s ✓ | 1.0 s ✓ | 996.9 ms ✓ | 43.8 s ✓ | 44.1 s ✓ | 44.5 s ✓ | 44.7 s ✓ | 1.0 s ✓ | 1.0 s ✓ | **14.9 ms ✓** | — | — | — | — | — | 15.9 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | 2.0 s ⚠ | 1.9 s ⚠ | 1.9 s ⚠ | 1.9 s ⚠ | 1.9 s ✓ | 2.0 s ✓ | 1.9 s ✓ | — | — | — | — | 2.0 s ✓ | 2.0 s ✓ | 86.7 ms ✗ | — | — | — | — | — | **109.1 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pcr  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.93 ms ✓ | 1.98 ms ✓ | 1.95 ms ✓ | 1.94 ms ✓ | **1.91 ms ✓** | 2.01 ms ✓ | 2.23 ms ✓ | 39.7 ms ✓ | 23.1 ms ✓ | 30.3 ms ✓ | 24.0 ms ✓ | 2.83 ms ✓ | 2.87 ms ✓ | 3.86 ms ✓ | — | 10.8 ms ✓ | — | — | — | 3.35 ms ✓ | 10.5 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | 1.1 s ⚠ | 1.1 s ⚠ | 1.1 s ⚠ | 1.0 s ⚠ | 1.1 s ✓ | 1.1 s ✓ | 1.1 s ✓ | 44.6 s ✓ | — | — | — | 1.0 s ✓ | 1.0 s ✓ | **17.2 ms ✓** | — | — | — | — | — | 18.0 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | 2.0 s ⚠ | 2.0 s ⚠ | 2.0 s ⚠ | 1.9 s ⚠ | 1.9 s ✓ | 1.9 s ✓ | 1.9 s ✓ | — | — | — | — | 2.1 s ✓ | 2.0 s ✓ | 115.9 ms ✗ | — | — | — | — | — | **122.8 ms ✓** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pds  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | **1.11 ms ✓** | 1.12 ms ✓ | 1.20 ms ✓ | 1.12 ms ✓ | 1.12 ms ✓ | 1.18 ms ✓ | 1.32 ms ✓ | 7.98 ms ✓ | 9.16 ms ✓ | 9.47 ms ✓ | 9.45 ms ✓ | 3.56 ms ✓ | 4.11 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 23.8 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | 20.5 ms ⚠ | 20.4 ms ⚠ | 20.4 ms ⚠ | 20.6 ms ⚠ | **20.6 ms ✓** | 20.7 ms ✓ | 22.9 ms ✓ | 115.4 ms ✓ | 140.4 ms ✓ | 162.9 ms ✓ | 145.0 ms ✓ | 38.6 ms ✓ | 39.0 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 219.3 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | 192.1 ms ⚠ | 183.6 ms ⚠ | 192.7 ms ⚠ | 189.6 ms ⚠ | **189.6 ms ✓** | 195.7 ms ✓ | 203.2 ms ✓ | 1.3 s ✓ | 1.3 s ✓ | 1.3 s ✓ | 1.3 s ✓ | 377.4 ms ✓ | 381.3 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 2.0 s ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pds  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.19 ms ✓ | **1.11 ms ✓** | 1.14 ms ✓ | 1.17 ms ✓ | 1.16 ms ✓ | 1.14 ms ✓ | 1.40 ms ✓ | 7.47 ms ✓ | 9.09 ms ✓ | 10.3 ms ✓ | 8.94 ms ✓ | 3.44 ms ✓ | 3.88 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 25.4 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | **0.98 ms ✓** | 0.99 ms ✓ | 0.99 ms ✓ | 0.99 ms ✓ | 1.00 ms ✓ | 1.61 ms ✓ | 1.20 ms ✓ | 2.91 ms ✓ | 4.83 ms ✓ | 10.3 ms ✓ | 5.07 ms ✓ | 1.77 ms ✓ | 2.10 ms ✓ | 1.35 ms ✓ | 1.12 ms ⚠ | 7.57 ms ✓ | — | 8.63 ms ✓ | 2.29 ms ✓ | 1.55 ms ✓ | 6.69 ms ✓ | 1.54 ms ✓ | 8.73 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | 8.24 ms ✓ | 8.35 ms ✓ | 8.36 ms ✓ | 8.07 ms ✓ | 7.97 ms ✓ | 8.18 ms ✓ | 8.74 ms ✓ | 47.6 ms ✓ | 67.2 ms ✓ | 100.6 ms ✓ | 59.4 ms ✓ | 12.9 ms ✓ | 13.6 ms ✓ | 8.66 ms ✓ | **7.82 ms ✓** | 90.8 ms ⚠ | — | 55.5 ms ✓ | 14.4 ms ✓ | 8.82 ms ✓ | 88.0 ms ✓ | 8.31 ms ✓ | 56.7 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | 41.0 ms ✓ | 40.3 ms ✓ | 40.8 ms ✓ | 40.3 ms ✓ | 41.1 ms ✓ | 41.3 ms ✓ | 42.0 ms ✓ | 394.9 ms ✓ | 556.5 ms ✓ | 960.5 ms ✓ | 588.2 ms ✓ | 65.9 ms ✓ | 66.5 ms ✓ | 41.3 ms ✓ | **39.1 ms ✓** | 700.0 ms ⚠ | — | 519.9 ms ⚠ | 71.0 ms ✓ | 46.0 ms ✓ | 724.1 ms ⚠ | 42.6 ms ✓ | 519.5 ms ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | 83.1 ms ✓ | 182.3 ms ✓ | 81.8 ms ✓ | 99.6 ms ✓ | 83.1 ms ✓ | 82.5 ms ✓ | 97.3 ms ✓ | 461.6 ms ✓ | 621.1 ms ✓ | 660.1 ms ✓ | 569.6 ms ✓ | 128.0 ms ✓ | 161.3 ms ✓ | 102.8 ms ✓ | **77.6 ms ✓** | 613.5 ms ✓ | — | 766.5 ms ✓ | 177.2 ms ✓ | 84.7 ms ✓ | 651.7 ms ✓ | 82.8 ms ✓ | 770.5 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.00 ms ✓ | **0.98 ms ✓** | 0.99 ms ✓ | 1.02 ms ✓ | 0.98 ms ✓ | 0.98 ms ✓ | 1.23 ms ✓ | 3.19 ms ✓ | 4.61 ms ✓ | 9.24 ms ✓ | 4.85 ms ✓ | 1.66 ms ✓ | 2.10 ms ✓ | 1.30 ms ✓ | 1.14 ms ✓ | 5.78 ms ✓ | — | 8.35 ms ✓ | 2.50 ms ✓ | 1.51 ms ✓ | 6.76 ms ✓ | 1.32 ms ✓ | 8.48 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | 8.14 ms ✓ | 7.97 ms ✓ | 8.06 ms ✓ | 8.25 ms ✓ | 8.06 ms ✓ | 7.99 ms ✓ | 8.44 ms ✓ | 44.9 ms ✓ | 59.7 ms ✓ | 101.1 ms ✓ | 58.5 ms ✓ | 13.0 ms ✓ | 13.5 ms ✓ | 8.44 ms ✓ | **7.92 ms ✓** | 83.5 ms ✓ | — | 57.2 ms ✓ | 14.1 ms ✓ | 8.89 ms ✓ | 82.3 ms ✓ | 8.42 ms ✓ | 56.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | 41.1 ms ✓ | 40.4 ms ✓ | 40.7 ms ✓ | 41.0 ms ✓ | 40.7 ms ✓ | 41.8 ms ✓ | 41.9 ms ✓ | 383.0 ms ✓ | 561.9 ms ✓ | 950.2 ms ✓ | 577.9 ms ✓ | 65.9 ms ✓ | 66.4 ms ✓ | 41.0 ms ✓ | **39.3 ms ✓** | 706.3 ms ⚠ | — | 522.6 ms ⚠ | 75.4 ms ✓ | 42.1 ms ✓ | 710.4 ms ⚠ | 42.7 ms ✓ | 536.0 ms ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | 83.0 ms ✓ | 243.2 ms ✓ | 95.1 ms ✓ | 103.7 ms ✓ | 82.5 ms ✓ | 82.2 ms ✓ | **80.8 ms ✓** | 468.3 ms ✓ | 619.2 ms ✓ | 686.6 ms ✓ | 543.8 ms ✓ | 129.3 ms ✓ | 155.4 ms ✓ | 111.4 ms ✓ | 104.8 ms ✓ | 642.8 ms ✓ | — | 777.7 ms ✓ | 169.5 ms ✓ | 114.5 ms ✓ | 649.4 ms ✓ | 107.1 ms ✓ | 771.8 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_cox  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.55 ms ✓ | 1.01 ms ✓ | **0.99 ms ✓** | 1.01 ms ✓ | 1.07 ms ✓ | 1.02 ms ✓ | 1.19 ms ✓ | 3.64 ms ✓ | 4.88 ms ✓ | 5.15 ms ✓ | 4.97 ms ✓ | 1.79 ms ✓ | 2.21 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 34.7 ms ✓ | — | — | — | — | — |
| 100×500 | 8.60 ms ⚠ | 8.29 ms ⚠ | 8.29 ms ⚠ | 8.47 ms ⚠ | 8.48 ms ✓ | **8.35 ms ✓** | 8.74 ms ✓ | 43.1 ms ✓ | 67.0 ms ✓ | 72.2 ms ✓ | 63.8 ms ✓ | 13.3 ms ✓ | 14.0 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 142.0 ms ✓ | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | 85.6 ms ⚠ | 85.2 ms ⚠ | 84.7 ms ⚠ | 87.6 ms ⚠ | **85.0 ms ✓** | 86.2 ms ✓ | 87.2 ms ✓ | 536.6 ms ✓ | 664.9 ms ✓ | 529.0 ms ✓ | 693.0 ms ✓ | 133.8 ms ✓ | 133.0 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.3 s ✓ | — | — | — | — | — |


## pls_cox  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.06 ms ✓ | 1.03 ms ✓ | 1.02 ms ✓ | **1.02 ms ✓** | 1.03 ms ✓ | 1.03 ms ✓ | 1.20 ms ✓ | 3.58 ms ✓ | 5.16 ms ✓ | 5.16 ms ✓ | 5.17 ms ✓ | 1.79 ms ✓ | 2.19 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 35.6 ms ✓ | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_diagnostic_dmodx  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | **1.02 ms ✓** | 1.04 ms ✓ | 1.03 ms ✓ | 1.03 ms ✓ | 1.07 ms ✓ | 1.03 ms ✓ | 1.17 ms ✓ | 2.99 ms ✓ | 4.33 ms ✓ | 4.38 ms ✓ | 4.87 ms ✓ | 1.68 ms ✓ | 2.06 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | 16.0 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | 8.69 ms ✓ | **8.52 ms ✓** | 8.89 ms ✓ | 8.65 ms ✓ | 9.92 ms ✓ | 8.68 ms ✓ | 8.96 ms ✓ | 47.9 ms ✓ | 71.7 ms ✓ | 62.2 ms ✓ | 67.2 ms ✓ | 13.6 ms ✓ | 14.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | 114.3 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | 87.6 ms ⚠ | 86.6 ms ⚠ | 87.3 ms ⚠ | 87.1 ms ⚠ | 87.7 ms ✓ | 86.7 ms ✓ | **84.2 ms ✓** | 566.1 ms ✓ | 540.2 ms ✓ | 541.1 ms ✓ | 548.9 ms ✓ | 137.9 ms ✓ | 137.6 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.2 s ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_diagnostic_dmodx  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.07 ms ✓ | **1.06 ms ✓** | 1.14 ms ✓ | 1.08 ms ✓ | 1.08 ms ✓ | 1.14 ms ✓ | 1.33 ms ✓ | 3.63 ms ✓ | 4.51 ms ✓ | 4.82 ms ✓ | 5.25 ms ✓ | 1.81 ms ✓ | 2.15 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | 16.5 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_diagnostic_q  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | **1.05 ms ✓** | 1.12 ms ✓ | 1.10 ms ✓ | 1.21 ms ✓ | 1.10 ms ✓ | 1.11 ms ✓ | 1.25 ms ✓ | 3.72 ms ✓ | 5.05 ms ✓ | 4.68 ms ✓ | 5.15 ms ✓ | 1.85 ms ✓ | 2.18 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | 15.5 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | 10.9 ms ✓ | 11.3 ms ✓ | 10.1 ms ✓ | 13.1 ms ✓ | **8.84 ms ✓** | 8.99 ms ✓ | 9.39 ms ✓ | 53.8 ms ✓ | 70.9 ms ✓ | 73.8 ms ✓ | 85.8 ms ✓ | 14.6 ms ✓ | 14.9 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | 119.9 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | 89.9 ms ⚠ | 89.7 ms ⚠ | 89.5 ms ⚠ | 88.9 ms ⚠ | 90.2 ms ✓ | 90.4 ms ✓ | **85.2 ms ✓** | 564.1 ms ✓ | 520.7 ms ✓ | 537.4 ms ✓ | 533.5 ms ✓ | 137.0 ms ✓ | 138.0 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.2 s ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_diagnostic_q  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.08 ms ✓ | 1.08 ms ✓ | 1.10 ms ✓ | 1.05 ms ✓ | 1.08 ms ✓ | **1.04 ms ✓** | 1.25 ms ✓ | 3.61 ms ✓ | 4.65 ms ✓ | 6.26 ms ✓ | 5.61 ms ✓ | 1.84 ms ✓ | 2.16 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | 15.9 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_diagnostic_t2  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.05 ms ✓ | 1.09 ms ✓ | 1.27 ms ✓ | 1.09 ms ✓ | **1.02 ms ✓** | 1.05 ms ✓ | 1.15 ms ✓ | 3.72 ms ✓ | 4.59 ms ✓ | 4.30 ms ✓ | 4.69 ms ✓ | 1.79 ms ✓ | 1.98 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | 15.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | 8.65 ms ⚠ | 8.67 ms ⚠ | 8.50 ms ⚠ | 8.64 ms ⚠ | 8.60 ms ✓ | **8.53 ms ✓** | 9.04 ms ✓ | 46.9 ms ✓ | 60.8 ms ✓ | 57.7 ms ✓ | 71.0 ms ✓ | 13.9 ms ✓ | 13.9 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | 108.5 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | 87.6 ms ⚠ | 87.4 ms ⚠ | 87.9 ms ⚠ | 87.2 ms ⚠ | 87.6 ms ✓ | 88.5 ms ✓ | **85.6 ms ✓** | 562.1 ms ✓ | 478.9 ms ✓ | 560.9 ms ✓ | 493.6 ms ✓ | 133.8 ms ✓ | 134.8 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.1 s ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_diagnostic_t2  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | **1.05 ms ✓** | 1.07 ms ✓ | 1.05 ms ✓ | 1.11 ms ✓ | 1.05 ms ✓ | 1.08 ms ✓ | 1.18 ms ✓ | 3.54 ms ✓ | 4.76 ms ✓ | 4.89 ms ✓ | 4.57 ms ✓ | 1.70 ms ✓ | 2.07 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | 15.4 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_glm  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.01 ms ⚠ | 1.02 ms ⚠ | 1.00 ms ⚠ | 1.05 ms ⚠ | 1.03 ms ✓ | **1.02 ms ✓** | 1.23 ms ✓ | 3.22 ms ✓ | 4.83 ms ✓ | 4.51 ms ✓ | 4.24 ms ✓ | 1.67 ms ✓ | 2.13 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 135.0 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | 8.44 ms ⚠ | 8.32 ms ⚠ | 8.36 ms ⚠ | 8.39 ms ⚠ | **8.46 ms ✓** | 8.56 ms ✓ | 8.82 ms ✓ | 44.5 ms ✓ | 66.4 ms ✓ | 68.4 ms ✓ | 70.7 ms ✓ | 13.5 ms ✓ | 14.5 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.3 s ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | 84.9 ms ⚠ | 86.2 ms ⚠ | 88.5 ms ⚠ | 86.7 ms ⚠ | 86.0 ms ✓ | **84.9 ms ✓** | 89.1 ms ✓ | 528.6 ms ✓ | 589.4 ms ✓ | 617.7 ms ✓ | 587.2 ms ✓ | 137.7 ms ✓ | 137.2 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 5.2 s ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_glm  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.08 ms ⚠ | 1.02 ms ⚠ | 1.02 ms ⚠ | 1.06 ms ⚠ | 1.04 ms ✓ | **1.03 ms ✓** | 1.25 ms ✓ | 3.73 ms ✓ | 5.03 ms ✓ | 5.18 ms ✓ | 4.72 ms ✓ | 1.72 ms ✓ | 2.39 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 140.3 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_lda  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.00 ms ✓ | 1.01 ms ✓ | 0.94 ms ✓ | 1.01 ms ✓ | 0.97 ms ✓ | **0.94 ms ✓** | 1.20 ms ✓ | 3.24 ms ✓ | 4.89 ms ✓ | 4.31 ms ✓ | 4.44 ms ✓ | 1.68 ms ✓ | 2.04 ms ✓ | — | — | — | — | — | — | 1.88 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | 8.40 ms ✓ | 8.06 ms ✓ | 8.01 ms ✓ | 7.96 ms ✓ | 8.05 ms ✓ | **7.96 ms ✓** | 8.27 ms ✓ | 39.3 ms ✓ | 66.0 ms ✓ | 56.1 ms ✓ | 55.1 ms ✓ | 13.1 ms ✓ | 13.5 ms ✓ | — | — | — | — | — | — | 9.60 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | 81.2 ms ✓ | 82.3 ms ✓ | **81.1 ms ✓** | 81.3 ms ✓ | 82.6 ms ✓ | 82.1 ms ✓ | 82.4 ms ✓ | — | — | — | — | 127.7 ms ✓ | 129.5 ms ✓ | — | — | — | — | — | — | 82.5 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_lda  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 0.99 ms ✓ | 1.11 ms ✓ | 0.99 ms ✓ | 1.05 ms ✓ | **0.98 ms ✓** | 0.99 ms ✓ | 1.29 ms ✓ | 3.13 ms ✓ | 4.81 ms ✓ | 4.64 ms ✓ | 4.84 ms ✓ | 1.70 ms ✓ | 2.03 ms ✓ | — | — | — | — | — | — | 1.93 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_logistic  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.14 ms ✓ | 1.08 ms ✓ | 1.09 ms ✓ | 1.12 ms ✓ | **1.05 ms ✓** | 1.11 ms ✓ | 1.44 ms ✓ | 3.39 ms ✓ | 4.94 ms ✓ | 5.07 ms ✓ | 5.54 ms ✓ | 1.96 ms ✓ | 2.58 ms ✓ | — | — | — | — | — | — | 2.40 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | 8.40 ms ✓ | **8.30 ms ✓** | 8.45 ms ✓ | 8.35 ms ✓ | 8.75 ms ✓ | 8.52 ms ✓ | 8.88 ms ✓ | 48.0 ms ✓ | 69.6 ms ✓ | 71.6 ms ✓ | 71.4 ms ✓ | 13.4 ms ✓ | 14.1 ms ✓ | — | — | — | — | — | — | 10.0 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | 85.5 ms ✓ | 85.2 ms ✓ | 84.2 ms ✓ | 85.3 ms ⚠ | **84.1 ms ✓** | 85.0 ms ✓ | 87.3 ms ✓ | 535.8 ms ✓ | 726.1 ms ✓ | 655.0 ms ✓ | 733.8 ms ✓ | 130.3 ms ✓ | 130.4 ms ✓ | — | — | — | — | — | — | 86.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_logistic  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.08 ms ✓ | **1.08 ms ✓** | 1.08 ms ✓ | 1.09 ms ✓ | 1.12 ms ✓ | 1.09 ms ✓ | 1.30 ms ✓ | 3.34 ms ✓ | 4.59 ms ✓ | 4.71 ms ✓ | 4.54 ms ✓ | 1.80 ms ✓ | 2.07 ms ✓ | — | — | — | — | — | — | 2.25 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_monitoring  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.08 ms ✓ | 1.06 ms ✓ | 1.05 ms ✓ | 1.06 ms ✓ | 1.06 ms ✓ | **1.05 ms ✓** | 1.24 ms ✓ | 3.48 ms ✓ | 4.83 ms ✓ | 5.38 ms ✓ | 5.05 ms ✓ | 2.28 ms ✓ | 2.05 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | 17.6 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | 8.52 ms ✓ | 8.55 ms ✓ | **8.51 ms ✓** | 8.55 ms ✓ | 8.53 ms ✓ | 8.80 ms ✓ | 8.80 ms ✓ | 55.7 ms ✓ | 71.8 ms ✓ | 61.6 ms ✓ | 67.7 ms ✓ | 13.5 ms ✓ | 14.4 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | 74.6 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | 84.3 ms ✓ | 84.3 ms ✓ | 87.1 ms ✓ | 84.3 ms ⚠ | 85.2 ms ✓ | 83.7 ms ✓ | **83.5 ms ✓** | 608.7 ms ✓ | 653.4 ms ✓ | 496.6 ms ✓ | 680.6 ms ✓ | 137.7 ms ✓ | 132.9 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | 701.7 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_monitoring  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.05 ms ✓ | 1.04 ms ✓ | **1.04 ms ✓** | 1.09 ms ✓ | 1.17 ms ✓ | 1.07 ms ✓ | 1.28 ms ✓ | 3.75 ms ✓ | 4.94 ms ✓ | 4.94 ms ✓ | 4.65 ms ✓ | 1.75 ms ✓ | 2.16 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | 17.2 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_qda  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.04 ms ✓ | 1.01 ms ✓ | 1.40 ms ✓ | **1.00 ms ✓** | 1.01 ms ✓ | 1.13 ms ✓ | 1.23 ms ✓ | 3.53 ms ✓ | 5.67 ms ✓ | 5.07 ms ✓ | 5.00 ms ✓ | 1.76 ms ✓ | 2.28 ms ✓ | — | — | — | — | — | — | 2.15 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | 8.50 ms ✓ | 8.52 ms ✓ | 8.49 ms ✓ | 8.85 ms ✓ | **8.33 ms ✓** | 8.48 ms ✓ | 8.83 ms ✓ | 46.6 ms ✓ | 69.9 ms ✓ | 66.3 ms ✓ | 79.4 ms ✓ | 13.8 ms ✓ | 14.4 ms ✓ | — | — | — | — | — | — | 10.4 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | 85.7 ms ✓ | **84.1 ms ✓** | 98.3 ms ✓ | 85.5 ms ✓ | 86.0 ms ✓ | 85.8 ms ✓ | 86.3 ms ✓ | 573.7 ms ✓ | 716.1 ms ✓ | 684.1 ms ✓ | 731.5 ms ✓ | 133.0 ms ✓ | 131.7 ms ✓ | — | — | — | — | — | — | 98.0 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_qda  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.01 ms ✓ | 1.02 ms ✓ | **0.98 ms ✓** | 1.02 ms ✓ | 0.99 ms ✓ | 0.98 ms ✓ | 1.38 ms ✓ | 3.40 ms ✓ | 5.16 ms ✓ | 4.98 ms ✓ | 5.21 ms ✓ | 1.79 ms ✓ | 2.29 ms ✓ | — | — | — | — | — | — | 2.21 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pop_pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 3.18 ms ✓ | **2.87 ms ✓** | 2.88 ms ✓ | 3.20 ms ✓ | 3.13 ms ✓ | 3.11 ms ✓ | 3.21 ms ✓ | 5.32 ms ✓ | 6.69 ms ✓ | 7.02 ms ✓ | 6.91 ms ✓ | 3.45 ms ✓ | 4.03 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 26.5 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | 97.3 ms ✓ | 94.8 ms ✓ | 90.9 ms ⚠ | 96.1 ms ⚠ | **91.9 ms ✓** | 98.6 ms ✓ | 92.9 ms ✓ | 127.2 ms ✓ | 155.1 ms ✓ | 141.6 ms ✓ | 152.8 ms ✓ | 98.7 ms ✓ | 93.2 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 109.2 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | 366.6 ms ✓ | 352.5 ms ✓ | **348.5 ms ✓** | 376.4 ms ✓ | 371.3 ms ✓ | 374.3 ms ✓ | 359.8 ms ✓ | — | — | — | — | 379.2 ms ✓ | 390.0 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 374.5 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pop_pls  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 3.21 ms ✓ | **2.84 ms ✓** | 2.87 ms ✓ | 3.27 ms ✓ | 3.13 ms ✓ | 3.18 ms ✓ | 3.22 ms ✓ | 5.01 ms ✓ | 6.19 ms ✓ | 6.67 ms ✓ | 6.50 ms ✓ | 3.38 ms ✓ | 3.74 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 26.6 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pso_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 6.31 ms ✓ | **5.21 ms ✓** | 5.89 ms ✓ | 5.44 ms ✓ | 6.30 ms ✓ | 6.20 ms ✓ | 6.60 ms ✓ | 9.21 ms ✓ | 10.6 ms ✓ | 10.4 ms ✓ | 10.3 ms ✓ | 6.90 ms ✓ | 7.53 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 204.3 ms ✓ | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pso_select  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 6.01 ms ⚠ | 4.83 ms ⚠ | 5.67 ms ⚠ | 5.11 ms ⚠ | **5.99 ms ✓** | 6.02 ms ✓ | 6.14 ms ✓ | 8.43 ms ✓ | 9.54 ms ✓ | 9.80 ms ✓ | 9.70 ms ✓ | 6.60 ms ✓ | 7.16 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 164.4 ms ✓ | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## random_frog_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.44 ms ✓ | 1.39 ms ✓ | 1.38 ms ✓ | 1.41 ms ✓ | 1.81 ms ✓ | **1.37 ms ✓** | 1.93 ms ✓ | 3.47 ms ✓ | 4.79 ms ✓ | 5.42 ms ✓ | 5.18 ms ✓ | 2.12 ms ✓ | 2.49 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 66.6 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## random_frog_select  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.49 ms ✓ | 1.48 ms ✓ | 1.41 ms ✓ | 1.61 ms ✓ | 1.51 ms ✓ | **1.41 ms ✓** | 1.59 ms ✓ | 3.68 ms ✓ | 7.48 ms ✓ | 5.38 ms ✓ | 5.30 ms ✓ | 2.17 ms ✓ | 2.48 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 64.6 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## random_subspace_pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.08 ms ✓ | 1.71 ms ✓ | **1.08 ms ✓** | 1.16 ms ✓ | 1.12 ms ✓ | 1.11 ms ✓ | 1.34 ms ✓ | 3.41 ms ✓ | 5.49 ms ✓ | 5.68 ms ✓ | 5.40 ms ✓ | 1.88 ms ✓ | 2.43 ms ✓ | — | — | — | — | — | — | 10.4 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | 8.24 ms ✓ | 8.30 ms ✓ | **8.20 ms ✓** | 8.44 ms ✓ | 8.31 ms ✓ | 8.23 ms ✓ | 8.78 ms ✓ | 47.8 ms ✓ | 71.1 ms ✓ | 70.4 ms ✓ | 67.5 ms ✓ | 13.0 ms ✓ | 13.5 ms ✓ | — | — | — | — | — | — | 17.5 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | 81.9 ms ✓ | 82.0 ms ✓ | **81.8 ms ✓** | 83.3 ms ✓ | 83.0 ms ✓ | 82.0 ms ✓ | 84.4 ms ✓ | 529.7 ms ✓ | 794.6 ms ✓ | 654.3 ms ✓ | 688.7 ms ✓ | 129.1 ms ✓ | 128.9 ms ✓ | — | — | — | — | — | — | 93.9 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## random_subspace_pls  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.11 ms ✓ | 1.08 ms ✓ | **1.06 ms ✓** | 1.12 ms ✓ | 1.08 ms ✓ | 1.08 ms ✓ | 1.31 ms ✓ | 3.46 ms ✓ | 4.73 ms ✓ | 5.11 ms ✓ | 5.39 ms ✓ | 1.77 ms ✓ | 2.30 ms ✓ | — | — | — | — | — | — | 9.87 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## randomization_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 2.37 ms ⚠ | 1.85 ms ⚠ | 2.21 ms ⚠ | **1.89 ms ✓** | 2.38 ms ✓ | 2.39 ms ✓ | 2.41 ms ✓ | 5.18 ms ✓ | 6.40 ms ✓ | 5.66 ms ✓ | 5.96 ms ✓ | 3.31 ms ✓ | 3.58 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 48.9 ms ✓ | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## randomization_select  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 2.31 ms ✓ | 1.92 ms ✓ | 2.25 ms ✓ | **1.90 ms ✓** | 2.27 ms ✓ | 2.29 ms ✓ | 2.52 ms ✓ | 4.82 ms ✓ | 5.62 ms ✓ | 5.94 ms ✓ | 5.75 ms ✓ | 2.95 ms ✓ | 3.22 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 49.8 ms ✓ | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## recursive_pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.70 ms ⚠ | 1.44 ms ⚠ | 1.62 ms ⚠ | **1.46 ms ✓** | 1.66 ms ✓ | 1.64 ms ✓ | 1.79 ms ✓ | 4.04 ms ✓ | 6.37 ms ✓ | 5.25 ms ✓ | 5.32 ms ✓ | 2.39 ms ✓ | 2.64 ms ✓ | — | — | — | — | — | — | 18.4 ms ✓ | 64.7 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | 14.6 ms ✓ | **11.9 ms ✓** | 14.6 ms ✓ | 12.3 ms ✓ | 14.8 ms ✓ | 14.7 ms ✓ | 15.3 ms ✓ | 48.5 ms ✓ | 66.6 ms ✓ | 72.6 ms ✓ | 62.6 ms ✓ | 19.7 ms ✓ | 20.3 ms ✓ | — | — | — | — | — | — | 36.0 ms ✓ | 223.0 ms ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | 246.6 ms ✓ | 196.2 ms ✓ | 233.2 ms ✓ | **193.7 ms ✓** | 237.1 ms ✓ | 261.5 ms ✓ | 240.7 ms ✓ | 625.0 ms ✓ | 834.5 ms ✓ | 767.8 ms ✓ | 892.7 ms ✓ | 309.7 ms ✓ | 314.7 ms ✓ | — | — | — | — | — | — | 746.1 ms ✓ | 3.3 s ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## recursive_pls  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.69 ms ✓ | **1.42 ms ✓** | 1.62 ms ✓ | 1.50 ms ✓ | 1.70 ms ✓ | 1.67 ms ✓ | 1.86 ms ✓ | 4.19 ms ✓ | 5.53 ms ✓ | 5.43 ms ✓ | 5.72 ms ✓ | 2.41 ms ✓ | 2.75 ms ✓ | — | — | — | — | — | — | 19.7 ms ✓ | 61.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## rep_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 2.05 ms ⚠ | 1.86 ms ⚠ | 1.50 ms ⚠ | **1.40 ms ✓** | 1.48 ms ✓ | 1.52 ms ✓ | 1.65 ms ✓ | 3.76 ms ✓ | 6.02 ms ✓ | 5.32 ms ✓ | 5.16 ms ✓ | 2.16 ms ✓ | 2.47 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 231.5 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## rep_select  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.48 ms ⚠ | 1.40 ms ⚠ | 1.50 ms ⚠ | 1.44 ms ⚠ | 1.49 ms ✓ | **1.49 ms ✓** | 1.64 ms ✓ | 3.61 ms ✓ | 4.72 ms ✓ | 4.81 ms ✓ | 4.88 ms ✓ | 2.26 ms ✓ | 2.47 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 206.8 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## ridge_pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.07 ms ✓ | 1.17 ms ✓ | **1.01 ms ✓** | 1.06 ms ✓ | 1.02 ms ✓ | 1.03 ms ✓ | 1.21 ms ✓ | 3.70 ms ✓ | 5.59 ms ✓ | 6.25 ms ✓ | 6.14 ms ✓ | 1.79 ms ✓ | 3.16 ms ✓ | 2.05 ms ⚠ | — | — | — | — | — | 1.59 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | 11.8 ms ✓ | 15.3 ms ✓ | 11.7 ms ✓ | 12.3 ms ✓ | **11.6 ms ✓** | 14.8 ms ✓ | 12.9 ms ✓ | 48.8 ms ✓ | 70.0 ms ✓ | 64.2 ms ✓ | 65.9 ms ✓ | 17.2 ms ✓ | 17.8 ms ✓ | 11.2 ms ⚠ | — | — | — | — | — | 14.8 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | 88.2 ms ⚠ | 87.0 ms ✓ | **86.9 ms ✓** | 88.9 ms ✓ | 87.4 ms ✓ | 88.1 ms ✓ | 91.7 ms ✓ | 533.2 ms ✓ | 725.0 ms ✓ | 692.7 ms ✓ | 684.1 ms ✓ | 137.6 ms ✓ | 140.1 ms ✓ | 87.1 ms ⚠ | — | — | — | — | — | 90.4 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## ridge_pls  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.06 ms ✓ | **1.01 ms ✓** | 1.03 ms ✓ | 1.01 ms ✓ | 1.04 ms ✓ | 1.02 ms ✓ | 1.20 ms ✓ | 3.40 ms ✓ | 5.02 ms ✓ | 4.70 ms ✓ | 4.61 ms ✓ | 1.70 ms ✓ | 2.09 ms ✓ | 3.21 ms ⚠ | — | — | — | — | — | 1.63 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## robust_pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.14 ms ✓ | **1.09 ms ✓** | 1.10 ms ✓ | 1.13 ms ✓ | 1.15 ms ✓ | 1.15 ms ✓ | 1.44 ms ✓ | 3.40 ms ✓ | 4.19 ms ✓ | 4.49 ms ✓ | 4.27 ms ✓ | 1.76 ms ✓ | 2.78 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | 13.9 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | 9.36 ms ⚠ | 9.49 ms ⚠ | 9.52 ms ⚠ | 9.76 ms ⚠ | 9.56 ms ✓ | **9.43 ms ✓** | 9.76 ms ✓ | 39.5 ms ✓ | 65.4 ms ✓ | 56.3 ms ✓ | 65.9 ms ✓ | 14.4 ms ✓ | 21.3 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | 152.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | 100.7 ms ⚠ | 104.5 ms ⚠ | 104.4 ms ⚠ | 147.5 ms ⚠ | **110.5 ms ✓** | 111.2 ms ✓ | 117.6 ms ✓ | 495.3 ms ✓ | 653.6 ms ✓ | 578.1 ms ✓ | 687.5 ms ✓ | 153.0 ms ✓ | 163.9 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | 1.1 s ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## robust_pls  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.13 ms ✓ | 1.16 ms ✓ | **1.13 ms ✓** | 1.16 ms ✓ | 1.16 ms ✓ | 1.14 ms ✓ | 1.33 ms ✓ | 3.35 ms ✓ | 4.98 ms ✓ | 4.65 ms ✓ | 4.88 ms ✓ | 1.85 ms ✓ | 2.29 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | 15.2 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## rosa  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.08 ms ⚠ | 1.05 ms ⚠ | 1.12 ms ⚠ | 1.04 ms ⚠ | 1.09 ms ✓ | **1.06 ms ✓** | 1.33 ms ✓ | 3.41 ms ✓ | 5.14 ms ✓ | 4.57 ms ✓ | 4.86 ms ✓ | 1.89 ms ✓ | 2.18 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.1 s ✓ | — | — | — | — | — | — | — | — | — | — |
| 100×500 | 8.68 ms ⚠ | 8.38 ms ⚠ | 8.24 ms ⚠ | 8.48 ms ⚠ | **8.41 ms ✓** | 8.48 ms ✓ | 9.11 ms ✓ | 47.4 ms ✓ | 66.8 ms ✓ | 69.7 ms ✓ | 62.0 ms ✓ | 13.5 ms ✓ | 13.9 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.2 s ✓ | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | 91.9 ms ⚠ | 85.1 ms ⚠ | 84.7 ms ⚠ | 85.0 ms ⚠ | **85.5 ms ✓** | 85.6 ms ✓ | 86.7 ms ✓ | 534.9 ms ✓ | 687.6 ms ✓ | 643.4 ms ✓ | 729.3 ms ✓ | 129.5 ms ✓ | 130.7 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 2.0 s ✓ | — | — | — | — | — | — | — | — | — | — |


## rosa  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.07 ms ⚠ | 1.06 ms ⚠ | 1.07 ms ⚠ | 1.09 ms ⚠ | 1.07 ms ✓ | **1.05 ms ✓** | 1.19 ms ✓ | 3.30 ms ✓ | 4.45 ms ✓ | 4.58 ms ✓ | 4.52 ms ✓ | 1.76 ms ✓ | 2.03 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.0 s ✓ | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## scars_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.46 ms — | 1.42 ms — | 1.52 ms — | 1.67 ms — | **1.47 ms ✓** | 1.57 ms ✓ | 1.68 ms ✓ | 3.94 ms ✓ | 5.80 ms ✓ | 5.46 ms ✓ | 4.95 ms ✓ | 2.13 ms ✓ | 2.49 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## scars_select  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.49 ms — | 1.49 ms — | 1.44 ms — | 1.47 ms — | 1.49 ms ✓ | **1.47 ms ✓** | 1.80 ms ✓ | 3.89 ms ✓ | 5.24 ms ✓ | 5.74 ms ✓ | 5.72 ms ✓ | 2.28 ms ✓ | 2.84 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## shaving_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.80 ms ⚠ | 1.48 ms ⚠ | 1.65 ms ⚠ | 1.80 ms ⚠ | 1.58 ms ✓ | **1.51 ms ✓** | 1.74 ms ✓ | 4.20 ms ✓ | 5.62 ms ✓ | 5.62 ms ✓ | 5.77 ms ✓ | 2.20 ms ✓ | 2.71 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 32.5 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## shaving_select  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.67 ms ⚠ | 1.44 ms ⚠ | 1.65 ms ⚠ | 1.55 ms ⚠ | **1.59 ms ✓** | 1.65 ms ✓ | 1.65 ms ✓ | 4.02 ms ✓ | 5.46 ms ✓ | 5.61 ms ✓ | 5.47 ms ✓ | 2.12 ms ✓ | 2.72 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 32.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## sipls_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 2.17 ms — | 2.05 ms — | 2.10 ms — | 2.04 ms — | 2.26 ms ✓ | **2.08 ms ✓** | 2.24 ms ✓ | 4.83 ms ✓ | 6.09 ms ✓ | 7.60 ms ✓ | 6.11 ms ✓ | 2.87 ms ✓ | 3.19 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## sipls_select  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 2.25 ms — | 1.89 ms — | 1.95 ms — | 2.12 ms — | 2.20 ms ✓ | **2.06 ms ✓** | 2.18 ms ✓ | 3.90 ms ✓ | 5.58 ms ✓ | 5.29 ms ✓ | 5.73 ms ✓ | 2.63 ms ✓ | 3.00 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## so_pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.70 ms ⚠ | 1.45 ms ⚠ | 1.70 ms ⚠ | 1.40 ms ⚠ | **1.50 ms ✓** | 1.53 ms ✓ | 1.74 ms ✓ | 3.59 ms ✓ | 5.19 ms ✓ | 4.80 ms ✓ | 5.10 ms ✓ | 2.08 ms ✓ | 2.66 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.0 s ✓ | — | — | — | — | — | — | — | — | — | — |
| 100×500 | 12.9 ms ⚠ | 12.0 ms ⚠ | 12.0 ms ⚠ | 12.8 ms ⚠ | **12.9 ms ✓** | 13.0 ms ✓ | 13.7 ms ✓ | 55.0 ms ✓ | 61.6 ms ✓ | 72.3 ms ✓ | 75.5 ms ✓ | 18.5 ms ✓ | 18.9 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.2 s ✓ | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | 143.2 ms ⚠ | 140.2 ms ⚠ | 137.2 ms ⚠ | 147.8 ms ⚠ | **146.5 ms ✓** | 146.8 ms ✓ | 152.3 ms ✓ | 548.1 ms ✓ | 748.4 ms ✓ | 799.9 ms ✓ | 752.1 ms ✓ | 187.9 ms ✓ | 196.2 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 3.2 s ✓ | — | — | — | — | — | — | — | — | — | — |


## so_pls  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.43 ms ⚠ | 1.35 ms ⚠ | 1.33 ms ⚠ | 1.40 ms ⚠ | **1.45 ms ✓** | 1.49 ms ✓ | 1.63 ms ✓ | 3.68 ms ✓ | 5.00 ms ✓ | 4.79 ms ✓ | 5.16 ms ✓ | 2.14 ms ✓ | 2.52 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.1 s ✓ | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## spa_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.30 ms ⚠ | 1.42 ms ⚠ | 1.24 ms ⚠ | **1.18 ms ✓** | 1.22 ms ✓ | 1.32 ms ✓ | 1.39 ms ✓ | 3.56 ms ✓ | 5.05 ms ✓ | 4.95 ms ✓ | 5.54 ms ✓ | 2.35 ms ✓ | 2.34 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 60.5 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## spa_select  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.17 ms ✓ | 1.21 ms ✓ | 1.19 ms ✓ | 1.25 ms ✓ | **1.16 ms ✓** | 1.21 ms ✓ | 1.34 ms ✓ | 3.91 ms ✓ | 5.05 ms ✓ | 4.59 ms ✓ | 4.82 ms ✓ | 1.89 ms ✓ | 2.17 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 54.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## sparse_pls_da  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.01 ms ✓ | **0.96 ms ✓** | 1.00 ms ✓ | 1.00 ms ✓ | 0.98 ms ✓ | 1.02 ms ✓ | 1.39 ms ✓ | 3.16 ms ✓ | 5.02 ms ✓ | 4.74 ms ✓ | 5.05 ms ✓ | 2.64 ms ✓ | 2.12 ms ✓ | — | — | — | — | 19.8 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 19.5 ms ✓ | — | — | — |
| 100×500 | 8.33 ms ⚠ | 8.33 ms ⚠ | 8.38 ms ⚠ | 8.35 ms ⚠ | 8.48 ms ✓ | **8.41 ms ✓** | 9.22 ms ✓ | 46.2 ms ✓ | 65.3 ms ✓ | 69.2 ms ✓ | 60.1 ms ✓ | 13.3 ms ✓ | 13.9 ms ✓ | — | — | — | — | 73.1 ms ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 160.5 ms ✓ | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | 84.7 ms ⚠ | 83.3 ms ⚠ | 83.8 ms ⚠ | 83.7 ms ⚠ | **85.2 ms ✓** | 85.2 ms ✓ | 86.1 ms ✓ | 530.7 ms ✓ | 699.1 ms ✓ | 625.0 ms ✓ | 678.7 ms ✓ | 138.1 ms ✓ | 130.3 ms ✓ | — | — | — | — | 994.6 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.0 s ✓ | — | — | — |


## sparse_pls_da  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.01 ms ✓ | 1.00 ms ✓ | 1.02 ms ✓ | 1.04 ms ✓ | 1.02 ms ✓ | **0.99 ms ✓** | 1.28 ms ✓ | 3.72 ms ✓ | 4.71 ms ✓ | 4.77 ms ✓ | 5.24 ms ✓ | 1.80 ms ✓ | 2.27 ms ✓ | — | — | — | — | 19.7 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 21.0 ms ✓ | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## sparse_simpls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.00 ms ⚠ | 1.00 ms ⚠ | 1.02 ms ⚠ | 1.46 ms ✓ | **1.04 ms ✓** | 1.13 ms ✓ | 1.23 ms ✓ | 2.98 ms ✓ | 4.81 ms ✓ | 4.99 ms ✓ | 4.38 ms ✓ | 1.66 ms ✓ | 2.16 ms ✓ | — | — | — | — | 9.79 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 10.5 ms ✓ | — | — | — |
| 100×500 | 8.32 ms ✓ | 8.36 ms ✓ | 8.23 ms ✓ | 8.12 ms ✓ | **7.98 ms ✓** | 8.09 ms ✓ | 8.46 ms ✓ | 46.9 ms ✓ | 69.4 ms ✓ | 57.6 ms ✓ | 55.3 ms ✓ | 13.0 ms ✓ | 13.5 ms ✓ | — | — | — | — | 60.3 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 104.8 ms ✓ | — | — | — |
| 100×2500 | 40.8 ms ⚠ | 40.6 ms ⚠ | 40.6 ms ⚠ | 40.8 ms ⚠ | 41.1 ms ✓ | **41.0 ms ✓** | 41.9 ms ✓ | 405.2 ms ✓ | 554.1 ms ✓ | 549.9 ms ✓ | 558.6 ms ✓ | 66.4 ms ✓ | 66.4 ms ✓ | — | — | — | — | 500.9 ms ⚠ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.1 s ✓ | — | — | — |
| 1000×500 | 83.4 ms ⚠ | 83.1 ms ⚠ | 83.4 ms ⚠ | 82.6 ms ⚠ | **83.1 ms ✓** | 84.3 ms ✓ | 84.7 ms ✓ | 503.1 ms ✓ | 645.0 ms ✓ | 630.9 ms ✓ | 631.5 ms ✓ | 127.5 ms ✓ | 132.3 ms ✓ | — | — | — | — | 705.2 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.0 s ✓ | — | — | — |


## sparse_simpls  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.06 ms ✓ | **1.01 ms ✓** | 1.05 ms ✓ | 1.02 ms ✓ | 1.03 ms ✓ | 1.03 ms ✓ | 1.25 ms ✓ | 3.35 ms ✓ | 4.72 ms ✓ | 4.63 ms ✓ | 4.63 ms ✓ | 1.72 ms ✓ | 2.13 ms ✓ | — | — | — | — | 9.89 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 10.2 ms ✓ | — | — | — |
| 100×500 | 8.33 ms ✓ | 8.23 ms ✓ | 8.20 ms ✓ | 8.11 ms ✓ | 8.17 ms ✓ | **8.06 ms ✓** | 8.53 ms ✓ | 37.6 ms ✓ | 55.0 ms ✓ | 57.5 ms ✓ | 59.3 ms ✓ | 13.6 ms ✓ | 13.6 ms ✓ | — | — | — | — | 62.7 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 108.4 ms ✓ | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | 82.8 ms ⚠ | 110.2 ms ✓ | 83.2 ms ⚠ | 105.2 ms ✓ | **82.6 ms ✓** | 85.2 ms ✓ | 111.7 ms ✓ | 493.6 ms ✓ | 667.7 ms ✓ | 612.8 ms ✓ | 694.0 ms ✓ | 129.0 ms ✓ | 156.4 ms ✓ | — | — | — | — | 836.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.1 s ✓ | — | — | — |


## st_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | **1.12 ms ✓** | 1.13 ms ✓ | 1.14 ms ✓ | 1.14 ms ✓ | 1.15 ms ✓ | 1.19 ms ✓ | 1.34 ms ✓ | 3.04 ms ✓ | 5.13 ms ✓ | 5.17 ms ✓ | 4.40 ms ✓ | 1.91 ms ✓ | 2.23 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 16.0 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## st_select  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.16 ms ✓ | **1.14 ms ✓** | 1.15 ms ✓ | 1.16 ms ✓ | 1.18 ms ✓ | 1.16 ms ✓ | 1.33 ms ✓ | 3.45 ms ✓ | 5.24 ms ✓ | 4.98 ms ✓ | 4.59 ms ✓ | 1.78 ms ✓ | 2.05 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 16.2 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## stability_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.06 ms ✓ | 1.04 ms ✓ | **1.04 ms ✓** | 1.10 ms ✓ | 1.07 ms ✓ | 1.07 ms ✓ | 1.26 ms ✓ | 3.21 ms ✓ | 4.62 ms ✓ | 4.32 ms ✓ | 4.50 ms ✓ | 1.78 ms ✓ | 2.06 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 83.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## stability_select  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.10 ms ✓ | 1.09 ms ✓ | 1.08 ms ✓ | **1.06 ms ✓** | 1.09 ms ✓ | 1.08 ms ✓ | 1.23 ms ✓ | 2.87 ms ✓ | 4.40 ms ✓ | 4.97 ms ✓ | 4.97 ms ✓ | 1.74 ms ✓ | 2.03 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 82.6 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## t2_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.27 ms ✓ | 1.51 ms ✓ | 1.20 ms ✓ | 1.25 ms ✓ | **1.18 ms ✓** | 1.22 ms ✓ | 1.75 ms ✓ | 3.61 ms ✓ | 5.32 ms ✓ | 5.36 ms ✓ | 5.02 ms ✓ | 1.90 ms ✓ | 2.38 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 50.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## t2_select  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.21 ms ✓ | 1.21 ms ✓ | 1.24 ms ✓ | 1.24 ms ✓ | 1.31 ms ✓ | **1.17 ms ✓** | 1.56 ms ✓ | 3.61 ms ✓ | 5.41 ms ✓ | 5.31 ms ✓ | 4.92 ms ✓ | 1.85 ms ✓ | 2.18 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 40.3 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## uve_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.89 ms ⚠ | 1.12 ms ⚠ | 1.19 ms ⚠ | 1.10 ms ⚠ | **1.12 ms ✓** | 1.20 ms ✓ | 1.26 ms ✓ | 3.46 ms ✓ | 5.18 ms ✓ | 4.84 ms ✓ | 4.88 ms ✓ | 1.69 ms ✓ | 2.11 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 54.2 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## uve_select  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.14 ms ⚠ | 1.07 ms ⚠ | 1.14 ms ⚠ | 1.10 ms ⚠ | **1.07 ms ✓** | 1.10 ms ✓ | 1.23 ms ✓ | 3.46 ms ✓ | 5.00 ms ✓ | 5.06 ms ✓ | 4.93 ms ✓ | 1.74 ms ✓ | 2.11 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 52.0 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## variable_select_coef  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.03 ms ⚠ | 1.09 ms ⚠ | 1.01 ms ⚠ | **1.04 ms ✓** | 1.25 ms ✓ | 1.09 ms ✓ | 1.22 ms ✓ | 3.57 ms ✓ | 5.24 ms ✓ | 6.87 ms ✓ | 5.06 ms ✓ | 1.74 ms ✓ | 2.11 ms ✓ | — | — | — | — | — | — | — | 10.5 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## variable_select_coef  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.00 ms ✓ | 1.00 ms ✓ | 1.01 ms ✓ | 1.02 ms ✓ | **1.00 ms ✓** | 1.00 ms ✓ | 1.18 ms ✓ | 3.52 ms ✓ | 4.55 ms ✓ | 4.32 ms ✓ | 4.44 ms ✓ | 1.68 ms ✓ | 1.93 ms ✓ | — | — | — | — | — | — | — | 9.81 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## variable_select_sr  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.08 ms ✓ | 1.05 ms ✓ | 1.04 ms ✓ | 1.15 ms ✓ | 1.14 ms ✓ | **1.01 ms ✓** | 1.20 ms ✓ | 3.84 ms ✓ | 4.12 ms ✓ | 3.93 ms ✓ | 4.16 ms ✓ | 1.64 ms ✓ | 1.92 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 8.35 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## variable_select_sr  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | **1.01 ms ✓** | 1.11 ms ✓ | 1.02 ms ✓ | 1.07 ms ✓ | 1.09 ms ✓ | 1.05 ms ✓ | 1.19 ms ✓ | 3.46 ms ✓ | 4.75 ms ✓ | 4.66 ms ✓ | 4.64 ms ✓ | 1.69 ms ✓ | 2.03 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 9.60 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## variable_select_vip  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.10 ms ⚠ | 1.06 ms ⚠ | 1.04 ms ⚠ | 1.11 ms ✓ | **1.02 ms ✓** | 1.09 ms ✓ | 1.21 ms ✓ | 3.65 ms ✓ | 5.39 ms ✓ | 6.62 ms ✓ | 4.95 ms ✓ | 1.79 ms ✓ | 2.13 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 11.6 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## variable_select_vip  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.02 ms ✓ | 1.02 ms ✓ | 1.04 ms ✓ | **0.99 ms ✓** | 1.00 ms ✓ | 1.05 ms ✓ | 1.19 ms ✓ | 3.12 ms ✓ | 4.64 ms ✓ | 4.40 ms ✓ | 4.57 ms ✓ | 1.64 ms ✓ | 1.92 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 9.28 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## vip_spa_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.03 ms ⚠ | 1.00 ms ⚠ | 1.02 ms ⚠ | 1.00 ms ✓ | 1.03 ms ✓ | **0.99 ms ✓** | 1.17 ms ✓ | 2.90 ms ✓ | 4.18 ms ✓ | 4.04 ms ✓ | 5.03 ms ✓ | 1.67 ms ✓ | 2.07 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 154.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## vip_spa_select  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.03 ms ⚠ | 1.03 ms ⚠ | 1.01 ms ⚠ | 1.00 ms ⚠ | 1.03 ms ✓ | **1.01 ms ✓** | 1.21 ms ✓ | 3.36 ms ✓ | 4.41 ms ✓ | 4.65 ms ✓ | 4.59 ms ✓ | 1.69 ms ✓ | 1.99 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 151.5 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## vissa_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 19.9 ms ✓ | **16.4 ms ✓** | 18.4 ms ✓ | 16.8 ms ✓ | 19.3 ms ✓ | 19.5 ms ✓ | 19.6 ms ✓ | 22.7 ms ✓ | 24.3 ms ✓ | 24.5 ms ✓ | 25.3 ms ✓ | 20.1 ms ✓ | 21.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 2.2 s ✓ | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## vissa_select  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 19.0 ms ⚠ | 15.9 ms ⚠ | 17.8 ms ⚠ | 16.6 ms ⚠ | 19.4 ms ✓ | **19.1 ms ✓** | 19.3 ms ✓ | 22.1 ms ✓ | 23.2 ms ✓ | 23.3 ms ✓ | 23.2 ms ✓ | 19.7 ms ✓ | 20.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.9 s ✓ | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## weighted_pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.04 ms ✓ | 1.03 ms ✓ | 1.00 ms ✓ | 1.06 ms ✓ | 1.05 ms ✓ | **1.00 ms ✓** | 1.23 ms ✓ | 3.18 ms ✓ | 4.92 ms ✓ | 5.92 ms ✓ | 5.02 ms ✓ | 1.75 ms ✓ | 2.37 ms ✓ | — | — | — | — | — | — | 1.58 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | 8.48 ms ✓ | 8.58 ms ✓ | 11.3 ms ✓ | 8.42 ms ✓ | 8.67 ms ✓ | **8.34 ms ✓** | 10.4 ms ✓ | 51.5 ms ✓ | 62.7 ms ✓ | 71.3 ms ✓ | 65.7 ms ✓ | 13.4 ms ✓ | 13.8 ms ✓ | — | — | — | — | — | — | 9.08 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | 44.1 ms ✓ | **41.2 ms ✓** | 42.9 ms ✓ | 42.4 ms ✓ | 41.7 ms ✓ | 42.3 ms ✓ | 43.6 ms ✓ | 377.3 ms ✓ | 580.2 ms ✓ | 560.6 ms ✓ | 566.1 ms ✓ | 65.7 ms ✓ | 65.9 ms ✓ | — | — | — | — | — | — | 41.9 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | 86.5 ms ✓ | 84.5 ms ✓ | 85.5 ms ✓ | 85.9 ms ✓ | **83.3 ms ✓** | 85.3 ms ✓ | 87.2 ms ✓ | 509.5 ms ✓ | 642.3 ms ✓ | 693.6 ms ✓ | 653.6 ms ✓ | 131.3 ms ✓ | 132.0 ms ✓ | — | — | — | — | — | — | 83.4 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## weighted_pls  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | **1.00 ms ✓** | 1.00 ms ✓ | 1.11 ms ✓ | 1.02 ms ✓ | 1.04 ms ✓ | 1.00 ms ✓ | 1.21 ms ✓ | 3.51 ms ✓ | 4.57 ms ✓ | 4.90 ms ✓ | 4.96 ms ✓ | 1.73 ms ✓ | 2.24 ms ✓ | — | — | — | — | — | — | 1.52 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## wvc_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.05 ms ⚠ | 1.00 ms ⚠ | 1.04 ms ⚠ | 1.04 ms ✓ | 1.07 ms ✓ | **1.04 ms ✓** | 1.16 ms ✓ | 3.69 ms ✓ | 4.80 ms ✓ | 5.04 ms ✓ | 4.89 ms ✓ | 1.69 ms ✓ | 2.17 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 12.9 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## wvc_select  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.07 ms ✓ | 1.02 ms ✓ | **1.02 ms ✓** | 1.06 ms ✓ | 1.03 ms ✓ | 1.11 ms ✓ | 1.19 ms ✓ | 3.14 ms ✓ | 4.54 ms ✓ | 4.36 ms ✓ | 4.97 ms ✓ | 1.64 ms ✓ | 1.98 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 12.2 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## wvc_threshold_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.08 ms ⚠ | 1.04 ms ⚠ | 1.03 ms ⚠ | 1.09 ms ✓ | **1.03 ms ✓** | 1.15 ms ✓ | 1.22 ms ✓ | 3.51 ms ✓ | 5.18 ms ✓ | 4.84 ms ✓ | 4.77 ms ✓ | 1.66 ms ✓ | 2.11 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 13.5 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## wvc_threshold_select  —  8 threads

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.python_diplslib | ref.r_chemometrics | ref.r_kernlab_pls | ref.r_mdatools | ref.python_onpls | ref.r_plsrglm | ref.r_base | ref.r_sgpls | ref.python_nirs4all | ref.r_plsvarsel | libPLS | ref.python_auswahl | ref.r_omicspls | ref.r_multiblock | ref.r_pls_stats | ref.r_ropls | ref.r_jico | ref.r_mboost | ref.r_plsrcox | ref.r_softimpute | ref.r_spls | ref.python_tensorly | ref.python_pyswarms | ref.r_enpls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.00 ms ✓ | **0.99 ms ✓** | 1.04 ms ✓ | 1.01 ms ✓ | 1.01 ms ✓ | 1.01 ms ✓ | 1.20 ms ✓ | 2.91 ms ✓ | 4.14 ms ✓ | 4.18 ms ✓ | 4.65 ms ✓ | 1.71 ms ✓ | 2.02 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 12.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 100×2500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 1000×500 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## Backend definitions

Each column in the per-algorithm tables above is one of the entries below. Columns are named `owner.language[.variant]`: `pls4all.*` are this project's own bindings; everything else is an external library shipped by its own maintainers.

| Column | Language | Tier | What it actually runs |
|---|---|---|---|
| `pls4all.cpp.native` | C++ | pls4all native scalar | libn4m built with `PLS4ALL_WITH_BLAS=OFF, OPENMP=OFF` — pure scalar native C++ loops, no acceleration. Used as one C++ implementation column; binding parity still uses cpp @ blas-omp when available. |
| `pls4all.cpp.blas` | C++ | pls4all + BLAS | libn4m built with `PLS4ALL_WITH_BLAS=ON` only — links system BLAS (OpenBLAS in this env), benefits from BLAS thread parallelism. |
| `pls4all.cpp.omp` | C++ | pls4all + OpenMP | libn4m built with `PLS4ALL_WITH_OPENMP=ON` only — OpenMP parallelism in the C kernel loops, no BLAS. |
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
| `sklearn` | Python | external | `sklearn.cross_decomposition.PLSRegression`, `sklearn.decomposition.PCA + LinearRegression / Ridge / GaussianProcessRegressor` (proxies) |
| `ikpls` | Python | external | `ikpls.numpy_ikpls.PLS` — Improved Kernel PLS (covers plain PLS only) |
| `pls` | R | external | CRAN `pls` package — `pls::plsr / pls::cppls / pls::pcr` |
| `ropls` | R | external | Bioconductor `ropls` — `ropls::opls` (covers OPLS only) |
| `mixOmics` | R | external | Bioconductor `mixOmics` — `pls / spls / plsda / splsda` |
| `plsregress` | MATLAB/Octave | external | Octave statistics `plsregress` (SIMPLS, plain PLS only) |
| `ref.python_scikit_learn` | external | reference | registry-declared external reference library |
| `ref.r_pls` | external | reference | registry-declared external reference library |
| `ref.python_ikpls` | external | reference | registry-declared external reference library |
| `ref.r_mixomics` | external | reference | registry-declared external reference library |
| `ref.python_diplslib` | external | reference | registry-declared external reference library |
| `ref.r_chemometrics` | external | reference | registry-declared external reference library |
| `ref.r_kernlab_pls` | external | reference | registry-declared external reference library |
| `ref.r_mdatools` | external | reference | registry-declared external reference library |
| `ref.python_onpls` | external | reference | registry-declared external reference library |
| `ref.r_plsrglm` | external | reference | registry-declared external reference library |
| `ref.r_base` | external | reference | registry-declared external reference library |
| `ref.r_sgpls` | external | reference | registry-declared external reference library |
| `ref.python_nirs4all` | external | reference | registry-declared external reference library |
| `ref.r_plsvarsel` | external | reference | registry-declared external reference library |
| `libPLS` | external | reference | registry-declared external reference library |
| `ref.python_auswahl` | external | reference | registry-declared external reference library |
| `ref.r_omicspls` | external | reference | registry-declared external reference library |
| `ref.r_multiblock` | external | reference | registry-declared external reference library |
| `ref.r_pls_stats` | external | reference | registry-declared external reference library |
| `ref.r_ropls` | external | reference | registry-declared external reference library |
| `ref.r_jico` | external | reference | registry-declared external reference library |
| `ref.r_mboost` | external | reference | registry-declared external reference library |
| `ref.r_plsrcox` | external | reference | registry-declared external reference library |
| `ref.r_softimpute` | external | reference | registry-declared external reference library |
| `ref.r_spls` | external | reference | registry-declared external reference library |
| `ref.python_tensorly` | external | reference | registry-declared external reference library |
| `ref.python_pyswarms` | external | reference | registry-declared external reference library |
| `ref.r_enpls` | external | reference | registry-declared external reference library |


## Versions per backend

| Column | Versions |
|---|---|
| `pls4all.cpp.native` | `language=C++ (via ctypes bridge) (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `libn4m=0.97.0+abi.1.16.0`; `numpy=2.3.5`; `registry_method=pls`; `timing_schema=adaptive-v1` |
| `pls4all.cpp.blas` | `language=C++ (via ctypes bridge) (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1; libopenblas 0.3.30 blas=1`; `libn4m=0.97.0+abi.1.16.0`; `numpy=2.3.5`; `registry_method=pls`; `timing_schema=adaptive-v1` |
| `pls4all.cpp.omp` | `language=C++ (via ctypes bridge) (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1; libomp ? openmp=1`; `libn4m=0.97.0+abi.1.16.0`; `numpy=2.3.5`; `registry_method=pls`; `timing_schema=adaptive-v1` |
| `pls4all.cpp.blas+omp` | `language=C++ (via ctypes bridge) (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1; libopenblas 0.3.33 blas=1; libomp ? openmp=1`; `libn4m=0.97.0+abi.1.16.0`; `numpy=2.3.5`; `registry_method=pls`; `timing_schema=adaptive-v1` |
| `pls4all.registry` | `language=Python 3.13.13`; `blas=libscipy_openblas 0.3.30 blas=1`; `pls4all=0.97.0`; `numpy=2.3.5`; `registry_method=pls`; `timing_schema=adaptive-v1` |
| `pls4all.python` | `language=Python 3.13.13`; `blas=libscipy_openblas 0.3.30 blas=1`; `pls4all=0.97.0`; `numpy=2.3.5`; `registry_method=pls`; `timing_schema=adaptive-v1` |
| `pls4all.sklearn` | `language=Python 3.13.13`; `blas=libscipy_openblas 0.3.30 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `pls4all=0.97.0`; `numpy=2.3.5`; `sklearn_class=PLSRegression`; `timing_schema=adaptive-v1` |
| `pls4all.R` | `language=R 4.3.3`; `pls4all=0.97.0`; `registry_method=pls`; `blas=linked-BLAS`; `timing_schema=adaptive-v1` |
| `pls4all.R.formula` | `language=R 4.3.3`; `pls4all=0.97.0`; `registry_method=pls`; `formula_facade=True`; `blas=linked-BLAS`; `timing_schema=adaptive-v1` |
| `pls4all.R.pls_compat` | `language=R 4.3.3`; `pls4all=0.97.0`; `registry_method=pls`; `facade=pls_compat`; `formula_facade=True`; `blas=linked-BLAS`; `timing_schema=adaptive-v1` |
| `pls4all.R.mdatools_compat` | `language=R 4.3.3`; `pls4all=0.97.0`; `registry_method=pls`; `facade=mdatools_compat`; `matrix_facade=True`; `blas=linked-BLAS`; `timing_schema=adaptive-v1` |
| `pls4all.matlab` | `language=Octave 10.3.0`; `pls4all=from libn4m-linked MEX`; `registry_method=pls`; `blas=linked-BLAS`; `timing_schema=adaptive-v1` |
| `pls4all.matlab.classdef` | `language=Octave 10.3.0`; `pls4all=from libn4m-linked MEX + classdefs`; `registry_method=pls`; `blas=linked-BLAS`; `timing_schema=adaptive-v1` |
| `sklearn` | `language=Python 3.13.13`; `blas=libscipy_openblas 0.3.30 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `sklearn=1.8.0`; `numpy=2.3.5`; `timing_schema=adaptive-v1` |
| `ikpls` | `language=Python 3.13.13`; `blas=libscipy_openblas 0.3.30 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `ikpls=4.0.1.post1`; `numpy=2.3.5`; `timing_schema=adaptive-v1` |
| `pls` | `language=R 4.3.3`; `pls=2.8.5`; `timing_schema=adaptive-v1` |
| `ropls` | `language=R 4.3.3`; `ropls=1.34.0`; `timing_schema=adaptive-v1` |
| `mixOmics` | `language=R 4.3.3`; `mixOmics=6.26.0`; `timing_schema=adaptive-v1` |
| `plsregress` | `language=Octave 10.3.0`; `statistics=Octave statistics pkg (plsregress)`; `blas=linked-BLAS`; `timing_schema=adaptive-v1` |
| `ref.python_scikit_learn` | `language=Python 3.13.13`; `blas=libscipy_openblas 0.3.30 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `numpy=2.3.5`; `reference_id=python_scikit_learn`; `reference_library=scikit-learn`; `reference_version=1.8.0`; `timing_schema=adaptive-v1` |
| `ref.r_pls` | `language=R 4.3.3`; `pls=2.8.5`; `timing_schema=adaptive-v1` |
| `ref.python_ikpls` | `language=Python 3.13.13`; `blas=libscipy_openblas 0.3.30 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `numpy=2.3.5`; `reference_id=python_ikpls`; `reference_library=ikpls`; `reference_version=4.0.1.post1`; `timing_schema=adaptive-v1` |
| `ref.r_mixomics` | `language=R 4.3.3`; `mixOmics=6.26.0`; `timing_schema=adaptive-v1` |
| `ref.python_diplslib` | `language=Python 3.13.13`; `blas=libscipy_openblas 0.3.30 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `numpy=2.3.5`; `reference_id=python_diplslib`; `reference_library=diPLSlib`; `reference_version=2.5.0`; `timing_schema=adaptive-v1` |
| `ref.r_chemometrics` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_chemometrics`; `reference_library=chemometrics`; `reference_version=0.7.x`; `timing_driver=rscript-batch`; `timing_schema=adaptive-v1` |
| `ref.r_kernlab_pls` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_kernlab_pls`; `reference_library=kernlab+pls`; `reference_version=0.9.33+2.8.5`; `timing_driver=rscript-batch`; `timing_schema=adaptive-v1` |
| `ref.r_mdatools` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_mdatools`; `reference_library=mdatools`; `reference_version=0.15.0`; `timing_driver=rscript-batch`; `timing_schema=adaptive-v1` |
| `ref.python_onpls` | `language=Python 3.13.13`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=python_onpls`; `reference_library=OnPLS`; `reference_version=github tomlof/OnPLS`; `timing_schema=adaptive-v1` |
| `ref.r_plsrglm` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_plsrglm`; `reference_library=plsRglm`; `reference_version=1.5.1`; `timing_driver=rscript-batch`; `timing_schema=adaptive-v1` |
| `ref.r_base` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_base`; `reference_library=base`; `reference_version=R 4.3.3`; `timing_driver=rscript-batch`; `timing_schema=adaptive-v1` |
| `ref.r_sgpls` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_sgpls`; `reference_library=sgPLS`; `reference_version=1.8.1`; `timing_driver=rscript-batch`; `timing_schema=adaptive-v1` |
| `ref.python_nirs4all` | `language=Python 3.13.13`; `blas=libscipy_openblas 0.3.30 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `numpy=2.3.5`; `reference_id=python_nirs4all`; `reference_library=nirs4all`; `reference_version=in-tree`; `timing_schema=adaptive-v1` |
| `ref.r_plsvarsel` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_plsvarsel`; `reference_library=plsVarSel`; `reference_version=0.10.0`; `timing_driver=rscript-batch`; `timing_schema=adaptive-v1` |
| `libPLS` | `language=matlab (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=matlab_libpls`; `reference_library=libPLS`; `reference_version=1.95`; `timing_schema=adaptive-v1` |
| `ref.python_auswahl` | `language=Python 3.13.13`; `blas=libscipy_openblas 0.3.30 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `numpy=2.3.5`; `reference_id=python_auswahl`; `reference_library=auswahl`; `reference_version=0.9.0`; `timing_schema=adaptive-v1` |
| `ref.r_omicspls` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_omicspls`; `reference_library=OmicsPLS`; `reference_version=2.1.0`; `timing_driver=rscript-batch`; `timing_schema=adaptive-v1` |
| `ref.r_multiblock` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_multiblock`; `reference_library=multiblock`; `reference_version=0.8.10`; `timing_schema=adaptive-v1` |
| `ref.r_pls_stats` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_pls_stats`; `reference_library=pls+stats`; `reference_version=R 4.3.3`; `timing_driver=rscript-batch`; `timing_schema=adaptive-v1` |
| `ref.r_ropls` | `language=R 4.3.3`; `ropls=1.34.0`; `timing_schema=adaptive-v1` |
| `ref.r_jico` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_jico`; `reference_library=JICO`; `reference_version=0.1`; `timing_driver=rscript-batch`; `timing_schema=adaptive-v1` |
| `ref.r_mboost` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_mboost`; `reference_library=mboost`; `reference_version=2.9-11`; `timing_driver=rscript-batch`; `timing_schema=adaptive-v1` |
| `ref.r_plsrcox` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_plsrcox`; `reference_library=plsRcox`; `reference_version=1.8.2`; `timing_driver=rscript-batch`; `timing_schema=adaptive-v1` |
| `ref.r_softimpute` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_softimpute`; `reference_library=softImpute`; `reference_version=1.4-1`; `timing_driver=rscript-batch`; `timing_schema=adaptive-v1` |
| `ref.r_spls` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_spls`; `reference_library=spls`; `reference_version=2.3.2`; `timing_driver=rscript-batch`; `timing_schema=adaptive-v1` |
| `ref.python_tensorly` | `language=Python 3.13.13`; `blas=libscipy_openblas 0.3.30 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `numpy=2.3.5`; `reference_id=python_tensorly`; `reference_library=tensorly`; `reference_version=0.9.0`; `timing_schema=adaptive-v1` |
| `ref.python_pyswarms` | `language=Python 3.13.13`; `blas=libscipy_openblas 0.3.30 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `numpy=2.3.5`; `reference_id=python_pyswarms`; `reference_library=pyswarms`; `reference_version=1.3.0`; `timing_schema=adaptive-v1` |
| `ref.r_enpls` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_enpls`; `reference_library=enpls`; `reference_version=6.1`; `timing_driver=rscript-batch`; `timing_schema=adaptive-v1` |

## Methodology

- Gate 1 reference: `cpp` cell at 1 thread (libn4m via ctypes), or `python_tier1` when `cpp` is unavailable for an algorithm
- Gate 2 reference: the registry-declared canonical external reference for the method
- Gate 1 tolerance: 1e-6 max-abs-diff; Gate 2 tolerance: `MethodSpec.rmse_rel_tol` (also emitted as `reference_parity_tolerance` by newer CSVs)
- All backends read the same orchestrator-generated CSV dataset (`benchmarks/cross_binding/data/data_<n>x<p>_seed<seed>.csv`)
- Adaptive timing: warmstart run #1; if there is any subsequent run, run #1 is excluded from the score. 2 total runs report run #2 alone; 3 total runs report the mean of runs #2-#3; 10/20/40 total runs report the median after the warmstart.
- Per-cell timeout guard: 24 h
- Thread control via `OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = BLIS_NUM_THREADS` set in the subprocess env, plus `Context.num_threads` for Python pls4all and `maxNumCompThreads()` for Octave
- pls4all libn4m build: `build/blas-omp/cpp/src/libn4m.so` (BLAS + OpenMP enabled)
