# Cross-binding benchmark — thread sweep

Same matrix as the [main page](cross_binding.md), but with thread counts 1, 3 and 10 shown in separate per-(algorithm, thread) sections. **External libraries (`sklearn`, `pls`, `ropls`, `mixOmics`, `plsregress`, `ikpls`) typically don't accelerate their inner loops with thread count** — only their linked BLAS does, and that helps only when the algo is GEMM-bound. pls4all bindings use OpenMP at the C kernel level on top of the BLAS, so multi-thread wins are visible here.

_Generated: 2026-05-19 10:02:56 UTC_
_CSV: `tmp2ki77ite.csv` (3420 cells)_


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

Each cell shows `<median_ms> <gate>`. C++ and external library cells use Gate 2, the reference parity against the method's canonical oracle. Internal binding cells use Gate 1, the binding parity against the native C++ baseline:

- ✓ **exact/pass** — gate-specific tolerance passed
- ⚠ **drift** — finite mismatch, below 10× the method's gate-2 tolerance or below the binding drift band
- ✗ **divergent** — outside the gate's tolerance band
- ⏳ cell timed out (300 s wall-clock)
- `—` see *Why a cell is empty* below

**Bold** = fastest cell on the row, **counted only among cells whose relevant gate is ✓**. For internal bindings that is Gate 1; for C++ and external libraries it is Gate 2. Drift / divergent / empty cells never carry the bold.

Timing is the **median of 4 timed run(s)** after up to three unmeasured warmups. All backends in a single cell read the same orchestrator-generated CSV dataset. See [methodology.md](methodology.md) for the full details.


## Why a cell is empty (`—`)

An empty cell means the backend **did not produce a timing for that (algorithm, size) combination**. Four distinct reasons, all reported as `—` in the matrix:

1. **External library doesn't implement the algorithm.** Example: `sklearn` has no native CPPLS or sparse SIMPLS; `plsregress` (Octave) only does plain PLS; `ikpls` is plain PLS only; `ropls` only does OPLS; `mixOmics` covers PLS / sparse PLS / PLS-DA. Filling this column requires the external maintainer to add the algorithm — out of our control.
2. **pls4all wrapper missing for that algorithm.** Example: `pls4all.sklearn` (tier 2) doesn't yet ship a class for `continuum_regression` or `kernel_pls`; `pls4all.R.formula` doesn't have a formula wrapper for every algorithm yet. This is the *chemin à parcourir* on our side — visible TODO.
3. **Bench script missing the dispatch case.** A few cells can fail because a binding bench script doesn't yet route a specific algorithm to its underlying call. Pure tooling gap, no library impact.
4. **Run too slow / OOM / crashed.** The cell timed out (`⏳`) or hit a runtime error. Rare in this matrix (300 s timeout is generous for the included sizes).

Each `—` represents one of these four reasons. The CSV (`results/full_matrix.csv`) carries a `reason` column that tells you exactly which one for any given cell.


Column names: anything starting with `pls4all.` is one of this project's bindings; the others use their real package name (`sklearn`, `ikpls`, `pls`, `ropls`, `mixOmics`, `plsregress`). **Every algorithm table keeps every column** — `—` cells are intentional and show *where coverage is still missing*, either on our side (an algorithm not yet wrapped in a tier-2 class) or on the external side (e.g. `sklearn` doesn't implement CPPLS / OPLS, `plsregress` only does plain PLS, `ikpls` only does plain PLS). Full per-column description: see [Backend definitions](#backend-definitions) at the bottom of this page.


## aom_pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 2.64 ms ✓ | 2.60 ms ✓ | 2.63 ms ✓ | 2.64 ms ✓ | 2.68 ms ✓ | **2.55 ms ✓** | 2.60 ms ✓ | 3.77 ms ✓ | 5.06 ms ✓ | 4.91 ms ✓ | 5.20 ms ✓ | 2.89 ms ✓ | 3.08 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 10.8 ms ✓ |


## aom_preprocess  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | **1.01 ms ✓** | 1.02 ms ✓ | 1.08 ms ✓ | 1.03 ms ✓ | 1.19 ms ✓ | 1.10 ms ✓ | — | 3.50 ms ✓ | — | 3.71 ms ✓ | 3.47 ms ✓ | 1.60 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## approximate_press  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.12 ms ✓ | **1.05 ms ✓** | 1.06 ms ✓ | 1.20 ms ✓ | 1.05 ms ✓ | 1.13 ms ✓ | — | — | — | 3.62 ms ✓ | 3.57 ms ✓ | — | — | — | — | — | — | — | — | — | 146.9 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## bagging_pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.25 ms ✓ | 1.17 ms ✓ | 1.20 ms ✓ | **1.13 ms ✓** | 1.37 ms ✓ | 1.16 ms ✓ | 1.35 ms ✓ | 3.00 ms ✓ | 6.00 ms ✓ | 3.54 ms ✓ | 3.45 ms ✓ | 1.72 ms ✓ | 3.73 ms ✓ | — | — | — | — | — | — | 9.04 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## bipls_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 3.47 ms ✓ | 2.92 ms ✓ | 3.24 ms ✓ | **2.81 ms ✓** | 3.71 ms ✓ | 3.63 ms ✓ | 4.35 ms ✓ | 6.00 ms ✓ | — | 5.38 ms ✓ | 5.22 ms ✓ | 3.38 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 876.7 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## boosting_pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.14 ms ✓ | 1.44 ms ✓ | 1.26 ms ✓ | **1.12 ms ✓** | 1.38 ms ✓ | 1.18 ms ✓ | 1.28 ms ✓ | 3.00 ms ✓ | 5.50 ms ✓ | 3.52 ms ✓ | 3.58 ms ✓ | 1.79 ms ✓ | 3.79 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 541.9 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — |


## bve_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 12.9 ms ✓ | 9.09 ms ✓ | 11.8 ms ✓ | **9.04 ms ✓** | 13.2 ms ✓ | 13.7 ms ✓ | 17.6 ms ✓ | 15.0 ms ✓ | — | 48.7 ms ✓ | 49.2 ms ✓ | 9.14 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 337.2 ms ✓ | — | — | — | — |


## cars_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.49 ms ✓ | 1.60 ms ✓ | 1.45 ms ✓ | **1.41 ms ✓** | 1.54 ms ✓ | 1.53 ms ✓ | 1.65 ms ✗ | 3.50 ms ✓ | — | 3.66 ms ✓ | 3.70 ms ✓ | 1.88 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 620.4 ms ✓ | — | — | — |


## continuum_regression  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 0.99 ms ✓ | **0.99 ms ✓** | 1.01 ms ✓ | 1.00 ms ✓ | 1.04 ms ✓ | 1.10 ms ✓ | 1.36 ms ✓ | 3.00 ms ✓ | 5.00 ms ✓ | 3.42 ms ✓ | 3.36 ms ✓ | 1.59 ms ✓ | 3.51 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 502.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## cppls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.04 ms ✓ | **1.00 ms ✓** | 1.03 ms ✓ | 1.07 ms ✓ | 1.02 ms ✓ | 1.05 ms ✓ | 1.14 ms ✓ | 3.50 ms ✓ | 5.50 ms ✓ | 7.13 ms ✓ | 3.72 ms ✓ | 1.64 ms ✓ | 3.55 ms ✓ | — | — | 8.00 ms ✓ | — | — | — | — | 124.0 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## di_pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.24 ms ✓ | 1.02 ms ✓ | **1.00 ms ✓** | 1.02 ms ✓ | 1.03 ms ✓ | 1.44 ms ✓ | 1.20 ms ✓ | — | — | 6.34 ms ✓ | 6.41 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 3.59 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## ds  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | **1.21 ms ✓** | 1.26 ms ✓ | 1.57 ms ✓ | 1.25 ms ✓ | 1.24 ms ✓ | 1.33 ms ✓ | 1.42 ms ✓ | — | — | 6.36 ms ✓ | 6.00 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 136.1 ms ✓ | — | — | — | — | — | — | — | — | — | — |


## ecr  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.32 ms ✓ | **1.16 ms ✓** | 1.20 ms ✓ | 1.26 ms ✓ | 1.37 ms ✓ | 1.32 ms ✓ | 1.48 ms ✓ | 3.50 ms ✓ | 5.50 ms ✓ | 3.54 ms ✓ | 4.06 ms ✓ | 1.74 ms ✓ | 3.66 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 9.85 ms ✓ | — | — |


## emcuve_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.37 ms ✓ | 1.38 ms ✓ | 1.34 ms ✓ | **1.26 ms ✓** | 1.31 ms ✓ | 1.49 ms ✓ | 1.60 ms ✗ | 3.50 ms ✗ | — | 3.98 ms ✓ | 3.81 ms ✓ | 1.90 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 659.8 ms ✓ | — | — | — | — |


## fused_sparse_pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.07 ms — | 0.99 ms — | 1.04 ms — | 1.01 ms — | **1.01 ms ✓** | 1.06 ms ✓ | 1.24 ms ✓ | 3.00 ms ✓ | — | 3.78 ms ✓ | 3.52 ms ✓ | 1.64 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## ga_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 2.72 ms ✓ | 2.56 ms ✓ | 2.75 ms ✓ | **2.54 ms ✓** | 2.89 ms ✓ | 2.76 ms ✓ | 3.60 ms ✗ | 4.50 ms ✗ | — | 4.89 ms ✓ | 4.63 ms ✓ | 2.96 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 473.8 ms ✓ | — | — | — | — |


## gpr_pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.22 ms ✓ | 1.31 ms ✓ | 1.22 ms ✓ | 1.26 ms ✓ | **1.20 ms ✓** | 1.30 ms ✓ | — | 3.50 ms ✓ | — | 3.72 ms ✓ | 3.92 ms ✓ | 1.98 ms ✓ | — | 4.96 ms ✗ | — | — | — | — | — | 2.05 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## group_sparse_pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.05 ms ✓ | 1.05 ms ✓ | **0.98 ms ✓** | 1.07 ms ✓ | 1.09 ms ✓ | 1.03 ms ✓ | 1.15 ms ✓ | 3.00 ms ✓ | — | 3.60 ms ✓ | 3.55 ms ✓ | 1.59 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.0 s ✓ | — | — | — | — | — | — | — | — |


## interval_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.44 ms ✓ | 1.51 ms ✓ | 1.62 ms ✓ | 1.52 ms ✓ | **1.43 ms ✓** | 1.52 ms ✓ | 1.51 ms ✓ | 3.50 ms ✓ | — | 3.98 ms ✓ | 3.67 ms ✓ | 1.97 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 593.7 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## ipw_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.28 ms ⚠ | 1.15 ms ⚠ | 1.22 ms ⚠ | **1.29 ms ✓** | 1.19 ms ✗ | 1.20 ms ✗ | 1.51 ms ✗ | 3.50 ms ✗ | — | 3.93 ms ✓ | 3.85 ms ✓ | 1.64 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 310.0 ms ✓ | — | — | — | — |


## irf_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 2.52 ms ✓ | **2.13 ms ✓** | 2.33 ms ✓ | 2.26 ms ✓ | 2.55 ms ✓ | 2.38 ms ✓ | 3.11 ms ✓ | 5.00 ms ✗ | — | 4.27 ms ✓ | 4.43 ms ✓ | 2.65 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 759.8 ms ✓ | — | — |


## iriv_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 191.6 ms ✓ | **145.1 ms ✓** | 173.5 ms ✓ | 146.4 ms ✓ | 192.4 ms ✓ | 192.1 ms ✓ | — | 190.0 ms ✓ | — | 147.7 ms ✓ | 149.6 ms ✓ | 145.3 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 10.1 s ✓ | — | — |


## kernel_pls_rbf  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.33 ms ✓ | 1.36 ms ✓ | 1.41 ms ✓ | 1.39 ms ✓ | 1.44 ms ✓ | **1.32 ms ✓** | 1.52 ms ✓ | 3.50 ms ✓ | — | 3.85 ms ✓ | 3.70 ms ✓ | 1.90 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 299.6 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## lw_pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 2.55 ms ✓ | 2.52 ms ✓ | 2.31 ms ✓ | **2.17 ms ✓** | 2.47 ms ✓ | 2.74 ms ✓ | 2.64 ms ✓ | 5.00 ms ✓ | — | 4.42 ms ✓ | 4.43 ms ✓ | 2.80 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 10.4 ms ✓ | — | — | — | — | — | — |


## mb_pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.19 ms ✓ | 1.10 ms ✓ | 1.24 ms ✓ | 1.05 ms ✓ | **1.00 ms ✓** | 1.05 ms ✓ | 1.20 ms ✓ | 3.50 ms ✓ | 5.50 ms ✓ | 3.32 ms ✓ | 3.45 ms ✓ | 1.61 ms ✓ | 3.82 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.74 ms ✓ | — | — | — | — | — | — | — |


## mir_pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 0.99 ms — | 0.99 ms — | 1.12 ms — | 1.09 ms — | **0.98 ms ✓** | 1.05 ms ✓ | 1.17 ms ✓ | 2.00 ms ✓ | 5.50 ms ✓ | 3.63 ms ✓ | 3.68 ms ✓ | 1.66 ms ✓ | 3.70 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## missing_aware_nipals  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.05 ms ✓ | 1.06 ms ✓ | 1.03 ms ✓ | 1.03 ms ✓ | 1.13 ms ✓ | **1.02 ms ✓** | 1.20 ms ✓ | 2.00 ms ✓ | 4.50 ms ✓ | 3.63 ms ✓ | 3.40 ms ✓ | 1.65 ms ✓ | 3.62 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 548.3 ms ✓ | — | — | — | — | — | — | — | — | — |


## n_pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.07 ms ✓ | 1.04 ms ✓ | 1.03 ms ✓ | **1.00 ms ✓** | 1.01 ms ✓ | 1.07 ms ✓ | 1.25 ms ✓ | 3.00 ms ✓ | — | 3.65 ms ✓ | 3.73 ms ✓ | 1.59 ms ✓ | 3.62 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 15.0 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## o2pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.12 ms ✓ | 1.19 ms ✓ | 1.13 ms ✓ | 1.09 ms ✓ | 1.11 ms ✓ | **1.08 ms ✓** | 1.30 ms ✓ | 3.50 ms ✓ | — | 3.75 ms ✓ | 3.95 ms ✓ | 1.70 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 832.7 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## on_pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.18 ms ✓ | **1.05 ms ✓** | 1.12 ms ✓ | 1.13 ms ✓ | 1.10 ms ✓ | 1.13 ms ✓ | — | — | — | 3.55 ms ✓ | 3.76 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 8.95 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## one_se_rule  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | **0.95 ms ✓** | 0.97 ms ✓ | 0.98 ms ✓ | 0.97 ms ✓ | 0.98 ms ✗ | 1.06 ms ✗ | — | — | — | 3.37 ms ✓ | 3.64 ms ✓ | — | — | — | — | — | — | — | — | — | 121.8 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## opls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.03 ms ✓ | 1.13 ms ✓ | **1.03 ms ✓** | 1.11 ms ✓ | 1.07 ms ✓ | 1.03 ms ✓ | 1.24 ms ✓ | 2.50 ms ✓ | — | 3.16 ms ✓ | 3.22 ms ✓ | 1.73 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | 3.0 s ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pcr  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 20.0 ms ✓ | 27.0 ms ✓ | 27.5 ms ✓ | 27.3 ms ✓ | 20.2 ms ✓ | 20.2 ms ✓ | 20.2 ms ✓ | 19.5 ms ✓ | 23.5 ms ✓ | 14.2 ms ✓ | 11.4 ms ✓ | 27.2 ms ✓ | — | **2.03 ms ✓** | — | 6.50 ms ✓ | — | — | — | 2.06 ms ✓ | 109.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pds  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.27 ms ✓ | **1.08 ms ✓** | 1.15 ms ✓ | 1.13 ms ✓ | 1.16 ms ✓ | 1.16 ms ✓ | 1.23 ms ✓ | — | — | 6.86 ms ✓ | 6.58 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 136.6 ms ✓ | — | — | — | — | — | — | — | — | — | — |


## pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.04 ms ✓ | **1.02 ms ✓** | 1.07 ms ✓ | 1.03 ms ✓ | 1.06 ms ✓ | 1.05 ms ✓ | 1.19 ms ✓ | 2.50 ms ✓ | 6.00 ms ✓ | 6.83 ms ✓ | 3.66 ms ✓ | 1.56 ms ✓ | 3.66 ms ✓ | 1.33 ms ✓ | 1.08 ms ✓ | 5.50 ms ✓ | — | 7.50 ms ✓ | 2.21 ms ✓ | 1.77 ms ✓ | 108.7 ms ✓ | 1.22 ms ✓ | 988.9 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_cox  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.04 ms ✓ | 1.07 ms ✓ | **1.00 ms ✓** | 1.11 ms ✓ | 1.00 ms ✓ | 1.13 ms ✓ | 1.06 ms ✓ | — | — | 3.71 ms ✓ | 4.00 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.9 s ✓ | — | — | — | — | — | — | — | — | — | — | — |


## pls_diagnostic_dmodx  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.14 ms ✓ | 1.12 ms ✓ | 1.07 ms ✓ | 1.09 ms ✓ | 1.05 ms ✓ | **1.00 ms ✓** | — | — | — | 3.79 ms ✓ | 3.42 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 334.3 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_diagnostic_q  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.06 ms ✓ | **1.04 ms ✓** | 1.22 ms ✓ | 1.07 ms ✓ | 1.07 ms ✓ | 1.07 ms ✓ | — | — | — | 3.48 ms ✓ | 3.34 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 328.6 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_diagnostic_t2  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.05 ms ✓ | 1.12 ms ✓ | 1.09 ms ✓ | 1.08 ms ✓ | 1.05 ms ✓ | **1.01 ms ✓** | — | — | — | 3.81 ms ✓ | 3.66 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 317.0 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_glm  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.03 ms ✓ | 1.06 ms ✓ | 1.01 ms ✓ | 1.01 ms ✓ | 1.08 ms ✓ | **0.98 ms ✓** | 1.26 ms ✓ | 3.00 ms ✓ | — | 3.41 ms ✓ | 3.51 ms ✓ | 1.72 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.1 s ✓ | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_lda  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 0.98 ms ✓ | **0.95 ms ✓** | 1.04 ms ✓ | 0.99 ms ✓ | 1.01 ms ✓ | 0.98 ms ✓ | 1.21 ms ✓ | 3.50 ms ✓ | — | 3.81 ms ✓ | 3.49 ms ✓ | 1.63 ms ✓ | — | — | — | — | — | — | — | 1.94 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_logistic  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.19 ms ✓ | **1.01 ms ✓** | 1.10 ms ✓ | 1.15 ms ✓ | 1.18 ms ✓ | 1.20 ms ✓ | 1.32 ms ✓ | 3.50 ms ✓ | — | 3.76 ms ✓ | 3.84 ms ✓ | 1.64 ms ✓ | — | — | — | — | — | — | — | 1.95 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_monitoring  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.03 ms ✓ | **1.01 ms ✓** | 1.09 ms ✓ | 1.09 ms ✓ | 1.06 ms ✓ | 1.13 ms ✓ | — | — | — | 3.33 ms ✓ | 3.70 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 348.9 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pls_qda  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 0.98 ms ✓ | **0.98 ms ✓** | 1.12 ms ✓ | 1.06 ms ✓ | 1.10 ms ✓ | 1.08 ms ✓ | 1.22 ms ✓ | 3.50 ms ✓ | — | 3.90 ms ✓ | 3.32 ms ✓ | 1.65 ms ✓ | — | — | — | — | — | — | — | 2.06 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## pop_pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 3.03 ms ✓ | 2.80 ms ✓ | **2.79 ms ✓** | 3.05 ms ✓ | 3.10 ms ✓ | 3.05 ms ✓ | 2.98 ms ✓ | 3.73 ms ✓ | 4.99 ms ✓ | 4.97 ms ✓ | 4.87 ms ✓ | 3.18 ms ✓ | 3.63 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 26.1 ms ✓ |


## pso_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 5.86 ms ✓ | 4.81 ms ✓ | 5.20 ms ✓ | **4.70 ms ✓** | 6.07 ms ✓ | 5.73 ms ✓ | 7.21 ms ✓ | 7.50 ms ✓ | — | 7.14 ms ✓ | 6.70 ms ✓ | 5.09 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 169.2 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## random_frog_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.53 ms ✓ | 1.78 ms ✓ | 1.46 ms ✓ | 1.46 ms ✓ | 1.44 ms ✓ | **1.42 ms ✓** | 1.75 ms ✗ | 4.00 ms ✓ | — | 3.65 ms ✓ | 3.85 ms ✓ | 1.96 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 63.4 ms ✓ | — | — |


## random_subspace_pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.05 ms ✓ | 1.05 ms ✓ | 1.08 ms ✓ | **1.02 ms ✓** | 1.03 ms ✓ | 1.09 ms ✓ | 1.17 ms ✓ | 3.50 ms ✓ | 5.50 ms ✓ | 3.47 ms ✓ | 3.67 ms ✓ | 1.61 ms ✓ | — | — | — | — | — | — | — | 8.13 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## randomization_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 2.32 ms ✓ | 2.04 ms ✓ | 2.24 ms ✓ | **1.92 ms ✓** | 2.24 ms ✓ | 2.44 ms ✓ | 2.98 ms ✓ | 4.50 ms ✓ | — | 4.57 ms ✓ | 4.14 ms ✓ | 2.40 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 167.7 ms ✓ | — |


## recursive_pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.66 ms ✓ | 1.48 ms ✓ | 1.68 ms ✓ | **1.42 ms ✓** | 1.75 ms ✓ | 1.73 ms ✓ | 1.97 ms ✓ | 3.50 ms ✓ | — | 3.87 ms ✓ | 4.17 ms ✓ | 2.01 ms ✓ | 3.89 ms ✓ | — | — | — | — | — | — | 18.6 ms ✓ | 154.4 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## rep_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.65 ms ✓ | **1.47 ms ✓** | 1.51 ms ✓ | 1.51 ms ✓ | 1.57 ms ✗ | 1.77 ms ✗ | 1.87 ms ✗ | 4.00 ms ✗ | — | 4.18 ms ✓ | 4.41 ms ✓ | 2.12 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 471.5 ms ✓ | — | — | — | — |


## ridge_pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.02 ms ✓ | **0.99 ms ✓** | 1.06 ms ✓ | 1.11 ms ✓ | 1.10 ms ✓ | 1.02 ms ✓ | 1.14 ms ✓ | 3.00 ms ✓ | 5.50 ms ✓ | 4.12 ms ✓ | 3.39 ms ✓ | 1.67 ms ✓ | 3.58 ms ✓ | 1.85 ms ⚠ | — | — | — | — | — | 1.64 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## robust_pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.14 ms ✓ | **1.09 ms ✓** | 1.25 ms ✓ | 1.25 ms ✓ | 1.11 ms ✓ | 1.17 ms ✓ | 1.26 ms ✓ | 3.50 ms ✓ | 5.50 ms ✓ | 3.48 ms ✓ | 3.59 ms ✓ | 1.65 ms ✓ | 4.02 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | 162.0 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## rosa  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.06 ms ✓ | 1.01 ms ✓ | 1.02 ms ✓ | **0.99 ms ✓** | 1.14 ms ✓ | 1.09 ms ✓ | 1.22 ms ✓ | 4.00 ms ✓ | — | 3.42 ms ✓ | 3.44 ms ✓ | 1.66 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 877.9 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## scars_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.46 ms — | 1.44 ms — | 1.53 ms — | 1.39 ms — | **1.57 ms ✓** | 1.61 ms ✓ | 1.85 ms ✗ | 4.50 ms ✓ | — | 3.70 ms ✓ | 3.80 ms ✓ | 1.91 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## shaving_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.41 ms ⚠ | 1.26 ms ⚠ | 1.32 ms ⚠ | **1.30 ms ✓** | 1.49 ms ✗ | 1.37 ms ✗ | 1.57 ms ✗ | 4.00 ms ✗ | — | 3.83 ms ✓ | 3.60 ms ✓ | 1.88 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 345.0 ms ✓ | — | — | — | — |


## sipls_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 2.03 ms — | 1.88 ms — | 1.89 ms — | 2.07 ms — | 2.13 ms ✓ | **2.06 ms ✓** | 2.70 ms ✓ | 4.50 ms ✓ | — | 4.16 ms ✓ | 4.07 ms ✓ | 2.53 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## so_pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.56 ms ✓ | 1.40 ms ✓ | 1.38 ms ✓ | **1.37 ms ✓** | 1.54 ms ✓ | 1.61 ms ✓ | 1.57 ms ✓ | 3.50 ms ✓ | — | 3.99 ms ✓ | 3.40 ms ✓ | 1.87 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 878.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## spa_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.18 ms ✓ | 1.17 ms ✓ | 1.48 ms ✓ | 1.20 ms ✓ | 1.25 ms ✓ | **1.11 ms ✓** | 1.26 ms ✓ | 3.50 ms ✓ | — | 3.82 ms ✓ | 3.63 ms ✓ | 1.70 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 365.3 ms ✓ | — | — | — | — |


## sparse_pls_da  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.06 ms ✓ | 1.02 ms ✓ | **0.97 ms ✓** | 1.00 ms ✓ | 0.99 ms ✓ | 1.00 ms ✓ | 1.36 ms ✓ | 3.00 ms ✓ | — | 3.50 ms ✓ | 3.94 ms ✓ | 1.69 ms ✓ | — | — | — | — | — | 41.5 ms ⚠ | — | — | — | — | — | — | 128.1 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## sparse_simpls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.01 ms ✓ | 0.98 ms ✓ | 0.97 ms ✓ | 1.09 ms ✓ | **0.96 ms ✓** | 1.04 ms ✓ | 1.22 ms ✓ | 3.00 ms ✓ | 6.50 ms ✓ | 3.44 ms ✓ | 3.99 ms ✓ | 1.55 ms ✓ | 3.59 ms ✓ | — | — | — | — | 9.00 ms ✓ | — | — | — | — | — | — | 113.0 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## st_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.25 ms ✓ | 1.19 ms ✓ | 1.20 ms ✓ | 1.29 ms ✓ | 1.24 ms ✓ | **1.16 ms ✓** | 1.34 ms ✗ | 3.00 ms ✓ | — | 3.67 ms ✓ | 3.58 ms ✓ | 1.69 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 332.2 ms ✓ | — | — | — | — |


## stability_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.13 ms ✓ | 1.13 ms ✓ | 1.16 ms ✓ | **1.05 ms ✓** | 1.09 ms ✓ | 1.10 ms ✓ | 1.29 ms ✗ | 3.50 ms ✗ | — | 3.41 ms ✓ | 3.59 ms ✓ | 1.66 ms ✗ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 393.1 ms ✓ | — | — | — | — |


## t2_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.30 ms ✓ | 1.30 ms ✓ | 1.16 ms ✓ | 1.29 ms ✓ | 1.18 ms ✓ | **1.14 ms ✓** | 1.34 ms ✗ | 3.50 ms ✓ | — | 3.49 ms ✓ | 4.04 ms ✓ | 1.72 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 333.0 ms ✓ | — | — | — | — |


## uve_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.17 ms ✓ | 1.12 ms ✓ | 1.15 ms ✓ | **1.10 ms ✓** | 1.11 ms ✓ | 1.20 ms ✓ | 1.25 ms ✓ | 3.50 ms ✓ | — | 3.66 ms ✓ | 3.39 ms ✓ | 1.79 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 388.4 ms ✓ | — | — | — | — |


## variable_select_coef  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.03 ms ✓ | 1.05 ms ✓ | 1.06 ms ✓ | 1.05 ms ✓ | **1.02 ms ✓** | 1.03 ms ✓ | 1.12 ms ✓ | — | — | 3.16 ms ✓ | 3.59 ms ✓ | — | — | — | — | — | — | — | — | — | 276.5 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## variable_select_sr  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.02 ms ✓ | 1.04 ms ✓ | 1.03 ms ✓ | 1.06 ms ✓ | **1.00 ms ✓** | 1.01 ms ✓ | 1.14 ms ✓ | — | — | 3.12 ms ✓ | 3.51 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 282.6 ms ✓ | — | — | — | — |


## variable_select_vip  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | **0.98 ms ✓** | 1.03 ms ✓ | 1.09 ms ✓ | 1.00 ms ✓ | 0.99 ms ✓ | 1.14 ms ✓ | 1.44 ms ✗ | — | — | 3.10 ms ✓ | 3.82 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 272.5 ms ✓ | — | — | — | — |


## vip_spa_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.04 ms ✓ | 0.99 ms ✓ | **0.97 ms ✓** | 1.00 ms ✓ | 0.99 ms ✓ | 1.02 ms ✓ | 1.15 ms ✗ | 3.00 ms ✓ | — | 4.04 ms ✓ | 3.83 ms ✓ | 1.55 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 126.4 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## vissa_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 18.2 ms ✓ | **15.1 ms ✓** | 16.9 ms ✓ | 15.6 ms ✓ | 18.2 ms ✓ | 18.2 ms ✓ | 23.8 ms ✓ | 20.5 ms ✓ | — | 16.2 ms ✓ | 16.1 ms ✓ | 15.3 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.7 s ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## weighted_pls  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.02 ms ✓ | 1.02 ms ✓ | **1.01 ms ✓** | 1.05 ms ✓ | 1.03 ms ✓ | 1.05 ms ✓ | 1.11 ms ✓ | 2.50 ms ✓ | 5.50 ms ✓ | 3.45 ms ✓ | 3.50 ms ✓ | 1.54 ms ✓ | 3.55 ms ✓ | — | — | — | — | — | — | 1.42 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


## wvc_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.07 ms ✓ | 1.22 ms ✓ | 1.08 ms ✓ | **1.03 ms ✓** | 1.13 ms ✓ | 1.10 ms ✓ | 1.15 ms ✓ | 3.50 ms ✓ | — | 3.53 ms ✓ | 3.34 ms ✓ | 1.53 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 315.0 ms ✓ | — | — | — | — |


## wvc_threshold_select  —  1 thread

| samples × features | pls4all.cpp.native | pls4all.cpp.blas | pls4all.cpp.omp | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.R.pls_compat | pls4all.R.mdatools_compat | pls4all.matlab | pls4all.matlab.classdef | sklearn | ikpls | pls | ropls | mixOmics | plsregress | ref.python_scikit_learn | ref.r_pls | ref.python_ikpls | ref.r_mixomics | ref.r_ropls | ref.r_spls | ref.python_diplslib | ref.r_chemometrics | ref.r_jico | ref.python_tensorly | ref.r_kernlab_pls | ref.r_omicspls | ref.r_mdatools | ref.r_multiblock | ref.python_onpls | ref.python_auswahl | ref.python_pyswarms | ref.r_mboost | ref.r_plsrglm | ref.r_plsrcox | ref.r_base | ref.r_softimpute | ref.r_sgpls | ref.python_nirs4all_operators_models_sklearn_mbpls | ref.python_nirs4all_operators_models_sklearn_lwpls | ref.python_nirs4all_bench_aom_v0_aompls | ref.r_plsvarsel | ref.r_enpls | ref.matlab_libpls | ref.r_pls_stats | ref.python_nirs4all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100×50 | 1.12 ms ✓ | 1.09 ms ✓ | **1.06 ms ✓** | 1.07 ms ✓ | 1.13 ms ✓ | 1.13 ms ✓ | 1.15 ms ✓ | 3.50 ms ✓ | — | 3.81 ms ✓ | 3.65 ms ✓ | 1.59 ms ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 300.7 ms ✓ | — | — | — | — |


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
| `ref.python_nirs4all` | external | reference | registry-declared external reference library |


## Versions per backend

| Column | Versions |
|---|---|
| `pls4all.cpp.native` | `language=C++ (via ctypes bridge) (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `libp4a=0.97.0+abi.1.16.0`; `numpy=2.3.5`; `registry_method=pls` |
| `pls4all.cpp.blas` | `language=C++ (via ctypes bridge) (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1; libopenblas 0.3.30 blas=1`; `libp4a=0.97.0+abi.1.16.0`; `numpy=2.3.5`; `registry_method=pls` |
| `pls4all.cpp.omp` | `language=C++ (via ctypes bridge) (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1; libomp ? openmp=1`; `libp4a=0.97.0+abi.1.16.0`; `numpy=2.3.5`; `registry_method=pls` |
| `pls4all.cpp.blas+omp` | `language=C++ (via ctypes bridge) (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1; libopenblas 0.3.30 blas=1; libgomp ? openmp=1`; `libp4a=0.97.0+abi.1.16.0`; `numpy=2.3.5`; `registry_method=pls` |
| `pls4all.registry` | `language=Python 3.13.11`; `blas=libscipy_openblas 0.3.30 blas=1`; `pls4all=0.97.0`; `numpy=2.3.5`; `registry_method=pls` |
| `pls4all.python` | `language=Python 3.13.11`; `blas=libscipy_openblas 0.3.30 blas=1`; `pls4all=0.97.0`; `numpy=2.3.5`; `registry_method=pls` |
| `pls4all.sklearn` | `language=Python 3.13.11`; `blas=libscipy_openblas 0.3.30 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `pls4all=0.97.0`; `numpy=2.3.5`; `sklearn_class=PLSRegression` |
| `pls4all.R` | `language=R 4.3.3`; `pls4all=0.97.0`; `registry_method=pls`; `blas=linked-BLAS` |
| `pls4all.R.formula` | `language=R 4.3.3`; `pls4all=0.97.0`; `registry_method=pls`; `blas=linked-BLAS` |
| `pls4all.R.pls_compat` | `language=R 4.3.3`; `pls4all=0.97.0`; `registry_method=pls`; `facade=pls_compat`; `formula_facade=True`; `blas=linked-BLAS`; `timing_schema=warmup-v3` |
| `pls4all.R.mdatools_compat` | `language=R 4.3.3`; `pls4all=0.97.0`; `registry_method=pls`; `facade=mdatools_compat`; `matrix_facade=True`; `blas=linked-BLAS`; `timing_schema=warmup-v3` |
| `pls4all.matlab` | `language=Octave 10.3.0`; `pls4all=from libp4a-linked MEX`; `registry_method=pls`; `blas=linked-BLAS` |
| `pls4all.matlab.classdef` | `language=Octave 10.3.0`; `pls4all=from libp4a-linked MEX + classdefs`; `registry_method=pls`; `blas=linked-BLAS` |
| `sklearn` | `language=Python 3.13.11`; `blas=libscipy_openblas 0.3.30 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `sklearn=1.8.0`; `numpy=2.3.5` |
| `ikpls` | `language=Python 3.13.11`; `blas=libscipy_openblas 0.3.30 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `ikpls=4.0.1.post1`; `numpy=2.3.5` |
| `pls` | `language=R 4.3.3`; `pls=2.8.5` |
| `ropls` | `language=R 4.3.3`; `ropls=1.34.0` |
| `mixOmics` | `language=R 4.3.3`; `mixOmics=6.26.0` |
| `plsregress` | `language=Octave 10.3.0`; `statistics=Octave statistics pkg (plsregress)`; `blas=linked-BLAS` |
| `ref.python_scikit_learn` | `language=Python 3.13.11`; `blas=libscipy_openblas 0.3.30 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `numpy=2.3.5`; `reference_id=python_scikit_learn`; `reference_library=scikit-learn`; `reference_version=1.8.0` |
| `ref.r_pls` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_pls`; `reference_library=pls`; `reference_version=2.8.5` |
| `ref.python_ikpls` | `language=Python 3.13.11`; `blas=libscipy_openblas 0.3.30 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `numpy=2.3.5`; `reference_id=python_ikpls`; `reference_library=ikpls`; `reference_version=4.0.1.post1` |
| `ref.r_mixomics` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_mixomics`; `reference_library=mixOmics`; `reference_version=6.26.0` |
| `ref.r_ropls` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_ropls`; `reference_library=ropls`; `reference_version=Bioc` |
| `ref.r_spls` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_spls`; `reference_library=spls`; `reference_version=2.3.2` |
| `ref.python_diplslib` | `language=Python 3.13.11`; `blas=libscipy_openblas 0.3.30 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `numpy=2.3.5`; `reference_id=python_diplslib`; `reference_library=diPLSlib`; `reference_version=2.5.0` |
| `ref.r_chemometrics` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_chemometrics`; `reference_library=chemometrics`; `reference_version=0.7.x` |
| `ref.r_jico` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_jico`; `reference_library=JICO`; `reference_version=0.0` |
| `ref.python_tensorly` | `language=Python 3.13.11`; `blas=libscipy_openblas 0.3.30 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `numpy=2.3.5`; `reference_id=python_tensorly`; `reference_library=tensorly`; `reference_version=0.9.0` |
| `ref.r_kernlab_pls` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_kernlab_pls`; `reference_library=kernlab+pls`; `reference_version=0.9.33+2.8.5` |
| `ref.r_omicspls` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_omicspls`; `reference_library=OmicsPLS`; `reference_version=2.1.0` |
| `ref.r_mdatools` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_mdatools`; `reference_library=mdatools`; `reference_version=0.15.0` |
| `ref.r_multiblock` | `language=R (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=r_multiblock`; `reference_library=multiblock`; `reference_version=0.8.10` |
| `ref.python_onpls` | `language=Python 3.13.11`; `blas=libscipy_openblas 0.3.30 blas=1`; `numpy=2.3.5`; `reference_id=python_onpls`; `reference_library=OnPLS`; `reference_version=github tomlof/OnPLS` |
| `ref.python_auswahl` | `language=Python 3.13.11`; `blas=libscipy_openblas 0.3.30 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `numpy=2.3.5`; `reference_id=python_auswahl`; `reference_library=auswahl`; `reference_version=0.9.0` |
| `ref.python_pyswarms` | `language=Python 3.13.11`; `blas=libscipy_openblas 0.3.30 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `numpy=2.3.5`; `reference_id=python_pyswarms`; `reference_library=pyswarms`; `reference_version=1.3.0` |
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
| `ref.python_nirs4all` | `language=Python 3.13.13`; `blas=libscipy_openblas 0.3.30 blas=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `numpy=2.3.5`; `reference_id=python_nirs4all`; `reference_library=nirs4all`; `reference_version=in-tree`; `timing_schema=warmup-v3` |

## Methodology

- Gate 1 reference: `cpp` cell at 1 thread (libp4a via ctypes), or `python_tier1` when `cpp` is unavailable for an algorithm
- Gate 2 reference: the registry-declared canonical external reference for the method
- Gate 1 tolerance: 1e-6 max-abs-diff; Gate 2 tolerance: `MethodSpec.rmse_rel_tol` (also emitted as `reference_parity_tolerance` by newer CSVs)
- All backends read the same orchestrator-generated CSV dataset (`benchmarks/cross_binding/data/data_<n>x<p>_seed<seed>.csv`)
- 4 timed run(s) per cell after one unmeasured warmup, median reported
- Per-cell timeout: 300 s
- Thread control via `OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = BLIS_NUM_THREADS` set in the subprocess env, plus `Context.num_threads` for Python pls4all and `maxNumCompThreads()` for Octave
- pls4all libp4a build: `build/blas-omp/cpp/src/libp4a.so` (BLAS + OpenMP enabled)
