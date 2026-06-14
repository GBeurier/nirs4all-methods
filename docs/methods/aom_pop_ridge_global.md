# `ridge_global` — n4m.model_selection.aom_search.ridge_global

_Namespace_: **`n4m.model_selection.aom_search`** · _Fully-qualified_: `n4m.model_selection.aom_search.ridge_global` · _Catalog id_: `aom_pop.ridge_global`

_C ABI symbols_ (ABI 2.0): `n` · `o` · `n` · `e`

_Python_: `from n4m.model_selection.aom_search import aom_ridge_global`

Python-backed donor-style AOM Ridge global selector constrained to strict-linear single-operator AOM views. It delegates scoring and final fit to the native aom_chain_sweep_run Ridge-only path, selects one operator plus one positive Ridge alpha by train CV, and returns folded input_coefficients plus intercept for replay through NativeAOMRidgeGlobalRegressor. It intentionally excludes donor branch_global, MKL/kernel, row-reference-dependent preprocessing and nonlinear AOM Ridge modes; native v1 builds in CUDA-enabled configurations but this is not yet a fused GPU grinder.

_Timing benchmark_: `benchmarks/cross_binding/bench_aom_ridge_global_timing.py`


_See also_: [methods index](index.md).