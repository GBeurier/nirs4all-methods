# Cross-binding benchmark — thread sweep

Same matrix as the [main page](cross_binding.md), but with thread counts 1, 3 and 10 shown in separate per-(algorithm, thread) sections. **External libraries (`sklearn`, `pls`, `ropls`, `mixOmics`, `plsregress`, `ikpls`) typically don't accelerate their inner loops with thread count** — only their linked BLAS does, and that helps only when the algo is GEMM-bound. pls4all bindings use OpenMP at the C kernel level on top of the BLAS, so multi-thread wins are visible here.

_Generated: 2026-05-18 17:55:53 UTC_
_CSV: `tmpth5k8sjt.csv` (568 cells)_


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

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 154.2 ms ✓ | 160.0 ms ✓ | 156.7 ms ✓ | — | 32.0 ms ✓ | — | **3.32 ms ✓** | — |


## approximate_press  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | **159.7 ms ✓** | 164.5 ms ✓ | 209.6 ms ✓ | — | — | — | — | — |


## bagging_pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 212.0 ms ✓ | 209.2 ms ✓ | 162.5 ms ✓ | **2.63 ms ✓** | 31.0 ms ✓ | 23.0 ms ✓ | 3.50 ms ✓ | 10.3 ms ✓ |


## bipls_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 166.9 ms ✓ | 155.0 ms ✓ | 161.2 ms ✓ | **3.95 ms ✓** | 32.0 ms ✓ | — | 5.11 ms ✓ | — |


## boosting_pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 172.4 ms ✓ | 163.7 ms ✓ | 162.5 ms ✓ | **2.00 ms ✓** | 34.0 ms ✓ | 19.0 ms ✓ | 3.37 ms ✓ | 10.4 ms ✓ |


## bve_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 162.9 ms ✓ | 231.3 ms ✓ | **162.3 ms ✓** | 12.5 ms ✗ | 39.0 ms ✗ | — | 11.2 ms ✗ | — |


## cars_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 157.8 ms ✓ | 156.2 ms ✓ | 236.8 ms ✓ | 2.22 ms ✗ | 30.0 ms ✓ | — | **3.55 ms ✓** | — |


## continuum_regression  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 227.8 ms ✓ | 223.9 ms ✓ | 248.0 ms ✓ | **1.98 ms ✓** | 35.0 ms ✓ | 20.0 ms ✓ | 3.36 ms ✓ | 10.1 ms ✓ |


## cppls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 158.0 ms ✓ | 175.1 ms ✓ | 157.3 ms ✓ | **1.72 ms ✓** | 32.0 ms ✓ | 18.0 ms ✓ | 3.34 ms ✓ | 9.56 ms ✓ |


## di_pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 175.3 ms ✓ | 223.6 ms ✓ | 197.6 ms ✓ | **1.79 ms ✓** | — | — | — | — |


## ds  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 233.0 ms ✓ | 226.5 ms ✓ | 164.2 ms ✓ | **1.93 ms ✓** | — | — | — | — |


## ecr  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 212.7 ms ✓ | 264.9 ms ✓ | 229.4 ms ✓ | **2.15 ms ✓** | 32.0 ms ✓ | 20.0 ms ✓ | 3.54 ms ✓ | 9.95 ms ✓ |


## emcuve_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 160.0 ms ✓ | **156.8 ms ✓** | 167.9 ms ✓ | 2.35 ms ✗ | 30.0 ms ✗ | — | 3.76 ms ✗ | — |


## fused_sparse_pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 212.3 ms ✓ | 227.3 ms ✓ | 151.0 ms ✓ | **1.62 ms ✓** | 29.0 ms ✓ | — | 3.50 ms ✓ | — |


## ga_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 163.5 ms ✓ | 222.1 ms ✓ | **157.8 ms ✓** | 3.73 ms ✗ | 32.0 ms ✗ | — | 4.67 ms ✗ | — |


## gpr_pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 157.9 ms ✓ | 164.4 ms ✓ | 163.1 ms ✓ | — | 31.0 ms ✓ | — | **3.71 ms ✓** | — |


## group_sparse_pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 166.9 ms ✓ | 158.1 ms ✓ | 160.6 ms ✓ | **1.68 ms ✓** | 31.0 ms ✓ | — | 3.16 ms ✓ | — |


## interval_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 219.1 ms ✓ | 158.9 ms ✓ | 158.9 ms ✓ | 1.92 ms ✗ | 28.0 ms ✓ | — | **3.71 ms ✓** | — |


## ipw_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 150.0 ms ✓ | 158.7 ms ✓ | 213.0 ms ✓ | 1.86 ms ✗ | 29.0 ms ✓ | — | **3.33 ms ✓** | — |


## irf_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 151.6 ms ✓ | 155.1 ms ✓ | 159.2 ms ✓ | **3.17 ms ✓** | 31.0 ms ✓ | — | 4.36 ms ✓ | — |


## iriv_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 362.1 ms ✓ | **362.1 ms ✓** | 411.6 ms ✓ | 352.8 ms ✗ | 218.0 ms ✗ | — | 150.0 ms ✗ | — |


## kernel_pls_rbf  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 170.4 ms ✓ | 166.2 ms ✓ | 190.9 ms ✓ | **2.10 ms ✓** | 33.0 ms ✓ | — | 3.72 ms ✓ | — |


## lw_pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 215.5 ms ✓ | 157.4 ms ✓ | 158.1 ms ✓ | **2.77 ms ✓** | 32.0 ms ✓ | — | 4.31 ms ✓ | — |


## mb_pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 154.2 ms ✓ | 153.5 ms ✓ | 197.4 ms ✓ | **1.76 ms ✓** | 30.0 ms ✓ | 17.0 ms ✓ | 3.29 ms ✓ | 9.88 ms ✓ |


## mir_pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 149.7 ms ✓ | 159.6 ms ✓ | 158.8 ms ✓ | **1.77 ms ✓** | 31.0 ms ✓ | 19.0 ms ✓ | 3.62 ms ✓ | 9.99 ms ✓ |


## missing_aware_nipals  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 148.3 ms ✓ | 154.1 ms ✓ | 197.1 ms ✓ | **1.65 ms ✓** | 29.0 ms ✓ | 19.0 ms ✓ | 3.11 ms ✓ | 9.64 ms ✓ |


## n_pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 151.7 ms ✓ | 200.5 ms ✓ | 150.3 ms ✓ | **1.75 ms ✓** | 33.0 ms ✓ | — | 3.38 ms ✓ | 9.49 ms ✓ |


## o2pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 210.5 ms ✓ | 166.4 ms ✓ | 171.1 ms ✓ | **2.45 ms ✓** | 33.0 ms ✓ | — | 3.52 ms ✓ | — |


## on_pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 229.3 ms ✓ | 219.0 ms ✓ | **214.8 ms ✓** | — | — | — | — | — |


## one_se_rule  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 214.8 ms ✓ | 208.6 ms ✓ | **174.2 ms ✓** | — | — | — | — | — |


## opls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 161.0 ms ✓ | 193.3 ms ✓ | 159.6 ms ✓ | **1.87 ms ✓** | 33.0 ms ✓ | — | 3.87 ms ✓ | — |


## pcr  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 271.1 ms ✓ | 224.3 ms ✓ | 255.7 ms ✓ | **28.3 ms ✓** | 50.0 ms ✓ | 39.0 ms ✗ | 35.8 ms ✓ | — |


## pds  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 167.3 ms ✓ | 159.0 ms ✓ | 188.4 ms ✓ | **1.72 ms ✓** | — | — | — | — |


## pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 228.2 ms ✓ | 222.6 ms ✓ | 667.8 ms ✓ | **1.94 ms ✓** | 34.0 ms ✓ | 20.0 ms ✗ | 3.31 ms ✓ | 9.78 ms ✗ |


## pls_cox  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 152.9 ms ✓ | 165.9 ms ✓ | 160.7 ms ✓ | **1.57 ms ✓** | — | — | — | — |


## pls_diagnostic_dmodx  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 154.4 ms ✓ | 162.5 ms ✓ | **154.0 ms ✓** | — | — | — | — | — |


## pls_diagnostic_q  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | **153.4 ms ✓** | 161.5 ms ✓ | 158.5 ms ✓ | — | — | — | — | — |


## pls_diagnostic_t2  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | **157.1 ms ✓** | 163.3 ms ✓ | 162.8 ms ✓ | — | — | — | — | — |


## pls_glm  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 232.3 ms ✓ | 207.2 ms ✓ | 221.4 ms ✓ | **1.97 ms ✓** | 30.0 ms ✓ | — | 3.27 ms ✓ | — |


## pls_lda  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 158.8 ms ✓ | 157.5 ms ✓ | 158.7 ms ✓ | **1.80 ms ✓** | 28.0 ms ✓ | — | 3.80 ms ✓ | — |


## pls_logistic  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 155.3 ms ✓ | 155.3 ms ✓ | 231.5 ms ✓ | **2.22 ms ✓** | 32.0 ms ✓ | — | 4.19 ms ✓ | — |


## pls_monitoring  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | **153.7 ms ✓** | 223.2 ms ✓ | 170.4 ms ✓ | — | — | — | — | — |


## pls_qda  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 154.9 ms ✓ | 183.3 ms ✓ | 169.0 ms ✓ | **1.84 ms ✓** | 29.0 ms ✓ | — | 3.61 ms ✓ | — |


## pso_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | **161.0 ms ✓** | 200.3 ms ✓ | 231.3 ms ✓ | 6.86 ms ✗ | 35.0 ms ✗ | — | 7.04 ms ✗ | — |


## random_frog_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 159.9 ms ✓ | 165.6 ms ✓ | **157.9 ms ✓** | 2.28 ms ✗ | 30.0 ms ✗ | — | 3.80 ms ✗ | — |


## random_subspace_pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 156.2 ms ✓ | 164.7 ms ✓ | 167.7 ms ✓ | **1.83 ms ✓** | 32.0 ms ✓ | 21.0 ms ✓ | 3.42 ms ✓ | — |


## randomization_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 650.0 ms ✓ | 155.6 ms ✓ | 256.0 ms ✓ | **3.31 ms ✓** | 29.0 ms ✓ | — | 4.22 ms ✓ | — |


## recursive_pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 162.5 ms ✓ | 159.6 ms ✓ | 198.7 ms ✓ | **2.11 ms ✓** | 31.0 ms ✓ | — | 3.87 ms ✓ | 10.7 ms ✓ |


## rep_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 165.2 ms ✓ | 171.6 ms ✓ | 163.0 ms ✓ | 2.48 ms ✗ | 29.0 ms ✓ | — | **3.62 ms ✓** | — |


## ridge_pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 174.8 ms ✓ | 173.4 ms ✓ | 162.0 ms ✓ | **1.80 ms ✓** | 32.0 ms ✓ | 20.0 ms ✓ | 3.54 ms ✓ | 10.2 ms ✓ |


## robust_pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 159.2 ms ✓ | 159.4 ms ✓ | 224.9 ms ✓ | **1.74 ms ✓** | 32.0 ms ✓ | 21.0 ms ✓ | 3.42 ms ✓ | 10.1 ms ✓ |


## rosa  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 162.7 ms ✓ | 177.6 ms ✓ | 168.0 ms ✓ | **1.73 ms ✓** | 29.0 ms ✓ | — | 4.35 ms ✓ | — |


## scars_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 167.2 ms ✓ | 154.6 ms ✓ | 183.5 ms ✓ | 2.31 ms ✗ | 31.0 ms ✓ | — | **3.85 ms ✓** | — |


## shaving_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 152.1 ms ✓ | 154.2 ms ✓ | 158.7 ms ✓ | 1.98 ms ✗ | 31.0 ms ✓ | — | **3.55 ms ✓** | — |


## sipls_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 154.4 ms ✓ | 161.1 ms ✓ | 178.2 ms ✓ | **2.90 ms ✓** | 30.0 ms ✓ | — | 4.08 ms ✓ | — |


## so_pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 163.6 ms ✓ | 162.3 ms ✓ | 157.6 ms ✓ | **1.92 ms ✓** | 29.0 ms ✓ | — | 3.67 ms ✓ | — |


## spa_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 153.5 ms ✓ | 167.6 ms ✓ | 153.0 ms ✓ | **1.87 ms ✓** | 30.0 ms ✓ | — | 3.47 ms ✓ | — |


## sparse_pls_da  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 155.9 ms ✓ | 325.7 ms ✓ | 155.2 ms ✓ | **2.34 ms ✓** | 29.0 ms ✓ | — | 3.20 ms ✓ | — |


## sparse_simpls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 175.8 ms ✓ | 156.1 ms ✓ | 169.1 ms ✓ | **1.75 ms ✓** | 32.0 ms ✓ | 19.0 ms ✓ | 3.14 ms ✓ | 10.3 ms ✓ |


## st_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 224.5 ms ✓ | 158.9 ms ✓ | 225.5 ms ✓ | 2.02 ms ✗ | 32.0 ms ✓ | — | **3.55 ms ✓** | — |


## stability_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | **152.9 ms ✓** | 158.2 ms ✓ | 157.0 ms ✓ | 1.82 ms ✗ | 28.0 ms ✗ | — | 3.26 ms ✗ | — |


## t2_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 153.3 ms ✓ | 186.1 ms ✓ | 159.1 ms ✓ | 2.00 ms ✗ | 31.0 ms ✓ | — | **3.46 ms ✓** | — |


## uve_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | **156.4 ms ✓** | 157.9 ms ✓ | 206.6 ms ✓ | 1.88 ms ✗ | 29.0 ms ✗ | — | 3.34 ms ✗ | — |


## variable_select_coef  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 150.7 ms ✓ | 165.4 ms ✓ | 161.8 ms ✓ | **1.74 ms ✓** | — | — | — | — |


## variable_select_sr  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 195.9 ms ✓ | 160.0 ms ✓ | 158.3 ms ✓ | **1.64 ms ✓** | — | — | — | — |


## variable_select_vip  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 160.2 ms ✓ | 154.5 ms ✓ | **154.1 ms ✓** | 1.66 ms ✗ | — | — | — | — |


## vip_spa_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 152.6 ms ✓ | 158.5 ms ✓ | 161.9 ms ✓ | 1.68 ms ✗ | 31.0 ms ✓ | — | **3.83 ms ✓** | — |


## vissa_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 246.7 ms ✓ | 251.0 ms ✓ | **243.5 ms ✓** | 22.1 ms ✗ | 51.0 ms ✗ | — | 18.2 ms ✗ | — |


## weighted_pls  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 158.8 ms ✓ | 180.1 ms ✓ | 225.1 ms ✓ | **1.61 ms ✓** | 33.0 ms ✓ | 18.0 ms ✓ | 3.35 ms ✓ | 9.42 ms ✓ |


## wvc_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 154.5 ms ✓ | 152.8 ms ✓ | 159.9 ms ✓ | **1.65 ms ✓** | 30.0 ms ✓ | — | 3.14 ms ✓ | — |


## wvc_threshold_select  —  1 thread

| samples × features | pls4all.cpp.blas+omp | pls4all.registry | pls4all.python | pls4all.sklearn | pls4all.R | pls4all.R.formula | pls4all.matlab | pls4all.matlab.classdef |
|---|---|---|---|---|---|---|---|---|
| 100×50 | 154.2 ms ✓ | 153.4 ms ✓ | 236.2 ms ✓ | **1.55 ms ✓** | 30.0 ms ✓ | — | 3.31 ms ✓ | — |


## Backend definitions

Each column in the per-algorithm tables above is one of the entries below. Columns are named `owner.language[.variant]`: `pls4all.*` are this project's own bindings; everything else is an external library shipped by its own maintainers.

| Column | Language | Tier | What it actually runs |
|---|---|---|---|
| `pls4all.cpp.blas+omp` | C++ | pls4all + BLAS + OpenMP | libp4a built with both `PLS4ALL_WITH_BLAS=ON` and `PLS4ALL_WITH_OPENMP=ON` — the recommended production config. |
| `pls4all.registry` | Python | pls4all canonical | `benchmarks.parity_timing.registry.MethodSpec.pls4all_fn` — the canonical per-method pls4all entry point |
| `pls4all.python` | Python | pls4all raw | `pls4all._methods.<algo>_fit(ctx, cfg, X, y, …)` — direct FFI binding |
| `pls4all.sklearn` | Python | pls4all idiomatic | `pls4all.sklearn.<Class>` — sklearn-style BaseEstimator with `.fit() / .predict()` |
| `pls4all.R` | R | pls4all raw | `pls4all_method(algo, X, y, ...)` — unified dispatcher (33 fits + 24 selectors + 4 diagnostics) |
| `pls4all.R.formula` | R | pls4all idiomatic | `pls(y ~ ., data)`, `cppls(...)`, `sparse_pls(...)`, … — base R formula+S3 wrappers |
| `pls4all.matlab` | MATLAB/Octave | pls4all raw | `pls4all.<algo>(X, y, ...)` — single dispatcher MEX |
| `pls4all.matlab.classdef` | MATLAB/Octave | pls4all idiomatic | `pls4all.fit(algo, X, y, ...)` factory + per-algorithm classdefs |


## Versions per backend

| Column | Versions |
|---|---|
| `pls4all.cpp.blas+omp` | `language=C++ (via ctypes bridge) (host Linux x86_64)`; `blas=libscipy_openblas 0.3.30 blas=1; libopenblas 0.3.30 blas=1; libgomp ? openmp=1`; `libp4a=0.97.0+abi.1.16.0`; `numpy=2.3.5`; `registry_method=pls` |
| `pls4all.registry` | `language=Python 3.13.11`; `blas=libscipy_openblas 0.3.30 blas=1; libopenblas 0.3.30 blas=1; libgomp ? openmp=1`; `pls4all=0.97.0`; `numpy=2.3.5`; `registry_method=pls` |
| `pls4all.python` | `language=Python 3.13.11`; `blas=libscipy_openblas 0.3.30 blas=1; libopenblas 0.3.30 blas=1; libgomp ? openmp=1`; `pls4all=0.97.0`; `numpy=2.3.5`; `registry_method=pls` |
| `pls4all.sklearn` | `language=Python 3.13.11`; `blas=libscipy_openblas 0.3.30 blas=1; libopenblas 0.3.30 blas=1; libgomp ? openmp=1; libscipy_openblas 0.3.30 blas=1; libgomp ? openmp=1`; `pls4all=0.97.0`; `numpy=2.3.5`; `sklearn_class=PLSRegression` |
| `pls4all.R` | `language=R 4.3.3`; `pls4all=0.97.0`; `registry_method=pls`; `blas=linked-BLAS` |
| `pls4all.R.formula` | `language=R 4.3.3`; `pls4all=0.97.0`; `registry_method=pls`; `blas=linked-BLAS` |
| `pls4all.matlab` | `language=Octave 10.3.0`; `pls4all=from libp4a-linked MEX`; `registry_method=pls`; `blas=linked-BLAS` |
| `pls4all.matlab.classdef` | `language=Octave 10.3.0`; `pls4all=from libp4a-linked MEX + classdefs`; `registry_method=pls`; `blas=linked-BLAS` |

## Methodology

- Reference: `cpp` cell at 1 thread (libp4a via ctypes), or `python_tier1` when `cpp` is unavailable for an algorithm
- Parity tolerance: 1e-6 (per-algo overrides possible)
- All backends read the **same** orchestrator-generated CSV (`benchmarks/cross_binding/data/data_<n>x<p>_seed<seed>.csv`) so input data is bit-identical across languages
- 1 run(s) per cell, first discarded as warmup when `n_runs >= 3`, median reported
- Per-cell timeout: 300 s
- Thread control via `OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = BLIS_NUM_THREADS` set in the subprocess env, plus `Context.num_threads` for Python pls4all and `maxNumCompThreads()` for Octave
- pls4all libp4a build: `build/blas-omp/cpp/src/libp4a.so` (BLAS + OpenMP enabled)
