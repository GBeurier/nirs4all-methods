# `ridge_active_superblock` — n4m.compose.aom_superblock.ridge_active_superblock

_Namespace_: **`n4m.compose.aom_superblock`** · _Fully-qualified_: `n4m.compose.aom_superblock.ridge_active_superblock` · _Catalog id_: `aom_pop.ridge_active_superblock`

_C ABI symbols_ (ABI 2.0): `n` · `o` · `n` · `e`

_Python_: `from n4m.compose.aom_superblock import aom_ridge_active_superblock`

Python-backed donor-style AOM Ridge active-superblock constrained to strict-linear single-operator AOM views. It screens operators fold-locally during alpha CV using response signatures computed from native aom_preprocess outputs, refits the active subset on the full calibration set, and folds final superblock coefficients back to original-input input_coefficients plus intercept. It intentionally excludes donor branch_global, MKL/kernel, row-reference-dependent preprocessing and nonlinear AOM Ridge modes; native v1 builds in CUDA-enabled configurations but this is not yet a fused GPU active-superblock grinder.

_Timing benchmark_: `benchmarks/cross_binding/bench_aom_ridge_active_superblock_timing.py`


_See also_: [methods index](index.md).