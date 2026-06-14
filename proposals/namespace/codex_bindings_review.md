**A. Verdict: NO-GO**

Core ABI rename usage is mostly sound, but the bindings merge unit still has binding-scope blockers.

Required fixes:

1. Python public namespace is incomplete: [feature_selection/wrapper.py](/home/delete/nirs4all/nirs4all-methods/bindings/python/src/n4m/feature_selection/wrapper.py:1) exports nothing, so the locked API example `from n4m.feature_selection.wrapper import CARS` fails.

2. Python still has an ABI-1 dynamic lookup path: [native.py](/home/delete/nirs4all/nirs4all-methods/bindings/python/src/n4m/_impl/native.py:9635) builds `n4m_metric_<name>`, while ABI 2 has `n4m_metrics_regression_metrics_<name>`. `n4m.metrics.scoring.nirs_metrics` is therefore broken.

3. Catalog source is stale: split `catalog/methods/<id>.yaml` files were updated, but `catalog/methods.yaml` still has 30 `python.module: n4m.python` entries. Per repo rules, update the source YAML and regenerate splits.

4. Cross-binding executable scripts are not fully ported. The 9 touched files are in-scope, but more scripts still import removed surfaces such as top-level `Native*`, `n4m.sklearn`, `n4m.aom`, `n4m.moment`, or `n4m.python`, e.g. [bench_aom_robust_hpo_timing.py](/home/delete/nirs4all/nirs4all-methods/benchmarks/cross_binding/bench_aom_robust_hpo_timing.py:22) and [donor_ops.py](/home/delete/nirs4all/nirs4all-methods/benchmarks/cross_binding/donor_ops.py:607).

Positive checks: `scripts/generate_ffi_decls.py --check` passes; `_ffi_decls.py` has exactly 702 unique declarations matching `expected_symbols_linux.txt` minus `N4M_2`; [n4m/_ffi.py](/home/delete/nirs4all/nirs4all-methods/bindings/python/src/n4m/_ffi.py:45) is ABI `2.0.0`. R, JS, and MATLAB active call-site scans had zero exact old ABI symbols.

**B. R Role-Naming Ruling**

Accept the partial R role-name scope for this merge unit. Do not require a complete estimator/regression role scheme now; the full R naming model is underspecified, and the spec explicitly preserves idiomatic formula/S3 wrappers.

But the partial scope must actually include all locked names. I do not find `n4m_regression_ridge`, so add that alias/export/docs/test before merge. The existing selector/transform/split aliases are acceptable as Phase 3 scope.

**C. Required Gates Before Merge**

Must pass before merge:

- `python3 scripts/generate_ffi_decls.py --check`
- Python tests for `bindings/python/tests`, plus smoke imports for locked APIs: `SNV`, `Ridge`, `CARS`, `KennardStone`, ensemble classes, and `nirs_metrics`
- Catalog validation after updating `catalog/methods.yaml`: `python catalog/scripts/selftest.py && python catalog/scripts/validate.py`
- Cross-binding scripts import smoke after namespace changes
- R: `R CMD INSTALL` for both R packages, testthat, parity test, and an explicit ABI assertion for major/minor `c(2, 0)`
- JS/WASM: Emscripten build, `npm run build`, WASM staging, `npm test`
- MATLAB/Octave: MEX build and runtests in the CI environments that own those bindings

Static rename completeness plus a clean TypeScript compile is enough to accept JS/MATLAB rename work on the branch, but not enough to merge without their CI runtime gates.

**D. Top Risks**

- Python tests miss locked public import examples.
- Dynamic ABI symbol construction can bypass rename-map scans; `nirs_metrics` proves this risk is real.
- Catalog source/split drift will regress generated metadata.
- Bench scripts are executable consumers, not docs; stale imports will break parity tooling.
- MATLAB/Octave and JS/WASM still need real link/runtime validation.
- The known `docs/methods/index.md` 208 vs catalog 209 failure is Phase 4 docs scope, not a binding defect. Static dashboard/bench-data old IDs are also Phase 4 unless used by executable binding gates.