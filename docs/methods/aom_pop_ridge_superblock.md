# `ridge_superblock` — n4m.compose.aom_superblock.ridge_superblock

_Namespace_: **`n4m.compose.aom_superblock`** · _Fully-qualified_: `n4m.compose.aom_superblock.ridge_superblock` · _Catalog id_: `aom_pop.ridge_superblock`

_C ABI symbols_ (ABI 2.0): `n` · `o` · `n` · `e`

_Python_: `from n4m.compose.aom_superblock import aom_ridge_superblock`

Python-backed donor-style AOM Ridge superblock constrained to strict-linear single-operator AOM views. It builds operator outputs through native aom_preprocess, selects/fits Ridge alphas fold-locally through the native Ridge binding, applies train-fold centering and optional block RMS scaling to validation folds, and folds final superblock coefficients back to original-input input_coefficients plus intercept. It intentionally excludes donor branch_global, MKL/kernel, row-reference-dependent preprocessing and nonlinear AOM Ridge modes; native v1 builds in CUDA-enabled configurations but this is not yet a fused GPU superblock grinder.

_Timing benchmark_: `benchmarks/cross_binding/bench_aom_ridge_superblock_timing.py`


_See also_: [methods index](index.md).