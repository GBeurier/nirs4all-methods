**A. Verdict: NO-GO**

Two residual required fixes remain.

1. Selector implementations are real native bindings, but plan semantics drift from `pls4all`. `_fold_plan` shuffles folds when seed is nonzero in [native.py](/home/delete/nirs4all/nirs4all-methods/bindings/python/src/n4m/_impl/native.py:9379), while `pls4all` explicitly uses contiguous, non-shuffled folds and says seed must not affect plan composition in [_selection.py](/home/delete/nirs4all/nirs4all-methods/bindings/python/src/pls4all/sklearn/_selection.py:38). R matches that contiguous-plan contract in [r_dispatch.c](/home/delete/nirs4all/nirs4all-methods/bindings/r/n4m/src/r_dispatch.c:329). This affects `UVE`, `CARS`, `Stability`, etc. Fix `_fold_plan` to mirror the `pls4all`/R/MATLAB default plan exactly.

2. Cross-binding executable scripts still call removed top-level `n4m.*` APIs. Top-level re-exports are intentionally absent in [__init__.py](/home/delete/nirs4all/nirs4all-methods/bindings/python/src/n4m/__init__.py:13), but live calls remain at [bench_aom_chain_ridge_pls_timing.py](/home/delete/nirs4all/nirs4all-methods/benchmarks/cross_binding/bench_aom_chain_ridge_pls_timing.py:51), [bench_aom_robust_hpo_timing.py](/home/delete/nirs4all/nirs4all-methods/benchmarks/cross_binding/bench_aom_robust_hpo_timing.py:49), [bench_aom_ridge_blender_timing.py](/home/delete/nirs4all/nirs4all-methods/benchmarks/cross_binding/bench_aom_ridge_blender_timing.py:45), and [bench_aom_screen_refit_scaling.py](/home/delete/nirs4all/nirs4all-methods/benchmarks/cross_binding/bench_aom_screen_refit_scaling.py:94).

Confirmed resolved: wrapper exports all 24 names; `SPA`/`UVE`/`CARS`/`VariableSelect` make real ABI-2 calls; `nirs_metrics` uses `n4m_metrics_regression_metrics_*` and smoked successfully; catalog validation, strict ABI coverage, and split check passed; empty leaves have no catalog Python bindings; R `n4m_regression_ridge` alias/export/man/test exists.

**B. CI-Bound Gates**

Still mandatory before merge, after the above fixes:

- R: `R CMD INSTALL` for R packages, testthat, parity, and ABI `c(2, 0)` assertion.
- JS/WASM: Emscripten build/staging plus `npm test`.
- MATLAB: MEX/build in CI and `runtests`.

I did not modify files.