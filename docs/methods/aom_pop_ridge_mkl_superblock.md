# `ridge_mkl_superblock` — n4m.compose.aom_superblock.ridge_mkl_superblock

_Namespace_: **`n4m.compose.aom_superblock`** · _Fully-qualified_: `n4m.compose.aom_superblock.ridge_mkl_superblock` · _Catalog id_: `aom_pop.ridge_mkl_superblock`

_C ABI symbols_ (ABI 2.0): `n` · `o` · `n` · `e`

_Python_: `from n4m.compose.aom_superblock import aom_ridge_mkl_superblock`

Python-backed donor-style AOM Ridge MKL-light superblock constrained to strict-linear single-operator AOM views. It learns non-negative train-only KTA weights over operator blocks inside every alpha-CV fold, refits weights on the full calibration set, fits native Ridge on the equivalent weighted superblock, and folds final coefficients back to original-input input_coefficients plus intercept. It intentionally excludes donor branch_global, row-reference-dependent preprocessing, nonlinear kernels and nonlinear AOM Ridge modes; native v1 builds in CUDA-enabled configurations but this is not yet a fused GPU weighted-superblock grinder.

_Timing benchmark_: `benchmarks/cross_binding/bench_aom_ridge_mkl_superblock_timing.py`


_See also_: [methods index](index.md).