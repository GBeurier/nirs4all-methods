**(A) Overall**

GO-WITH-CHANGES.

The prior ML-table change-list was folded in correctly: regression split, GLM/survival split, EPO move, AOM redistribution, splitters rename, leaf renames, and dissolved utils grab-bag are all present in `_target_ml_table.tsv`. The roadmap direction is sound, but it needs a few hard corrections before code starts.

**(B) Required Changes**

1. Split WVC before migration.
`selection.wvc` currently hides two real public surfaces: `n4m_wvc_select` and `n4m_wvc_threshold_select`, while docs/registry/Python expose both. Make two catalog rows:
`feature_selection.wrapper.wvc` and `feature_selection.wrapper.wvc_threshold`.
Do not carry this ambiguity into ABI 2.0.

2. Do not treat `_target_ml_table.tsv`’s `abi_symbol` column as final ABI truth.
It is an inventory seed. Some AOM rows point at result/helper symbols, and 10 rows are blank. Phase R must derive final C symbols from reconciled catalog `abi_symbols`, not this single-column snapshot.

3. Open namespace decisions:
- Keep `lowlevel.moments`; it is a real sufficient-statistics substrate with multiple ABI functions. Add `lowlevel.h` to the Phase 2 header plan.
- Remove top-level `utils`; move `signal_type_detector` to `transform.signal_conversion.signal_type_detector`.
- Flatten `ensemble.aom` into `ensemble` unless there is a strong future-growth reason. Use leaves like `aom_ridge_blender` and `aom_operator_pls_stack`.
- Confirm `estimators.regression.glm` and `estimators.survival` as singletons. They are acceptable growth points.

4. ABI 2.0 plan must explicitly update:
`N4M_ABI_VERSION_*`, Python ABI constants, SONAME expectations, `.github/workflows/abi-check.yml`, and `cpp/src/c_api/n4m_linux.map` from `N4M_1` to `N4M_2`. `scripts/bump_version.sh` only syncs project semver; it does not bump ABI semver.

5. Treat core ABI + in-tree bindings as one merge unit.
It is fine to implement in subwaves, but do not merge after Phase 2 alone. Once old exports are deleted, Python/R/Octave/JS/WASM are broken until updated.

6. Catalog as single source of truth is right, but keep stable legacy `method_id`s where possible. Add `namespace`, `leaf`, final `fq_name`, `c_surface`, and `legacy_ids` fields rather than renaming every method ID and forcing fixture/doc churn.

**(C) Required Green Gates**

Phase R:
```bash
python proposals/namespace/_build_target_ml.py
python catalog/scripts/selftest.py
python catalog/scripts/split_legacy_methods.py --check
python catalog/scripts/validate.py --strict-abi --check-references
python catalog/scripts/reconcile_abi.py --check
python -m benchmarks.parity_timing.lockfile --check
```

Phase 2, core ABI checkpoint:
```bash
cmake --preset dev-release
cmake --build --preset dev-release --parallel
ctest --preset dev-release --output-on-failure
./build/dev-release/cpp/cli/n4m_cli --selfcheck
./build/dev-release/cpp/cli/n4m_cli --abi-info
scripts/regen_abi_snapshots.sh --check --lib "$(find build/dev-release -name 'libn4m.so.*' ! -type l | head -n1)"
scripts/bump_version.sh --check
python catalog/scripts/validate.py --strict-abi --check-references
```

Phase 3, Python:
```bash
PYTHONPATH=bindings/python/src N4M_LIB_PATH="$PWD/build/dev-release/cpp/src/libn4m.so" python -m pytest bindings/python/tests -q
python -m pytest bindings/python_nirs4all_methods/tests bindings/python_pls4all/tests -q
ruff format --check bindings/python
ruff check bindings/python
python -m pytest benchmarks/cross_binding/tests/test_donor_binding_specs.py -q
```

Phase 4, other in-tree bindings:
```bash
R CMD INSTALL --no-multiarch --no-staged-install bindings/r/n4m
Rscript bindings/r/test_parity.R
octave --no-gui --no-history --eval "cd bindings/matlab; build_mex"
octave --no-gui --no-history --eval "addpath('$PWD/bindings/matlab'); cd bindings/matlab/test; test_parity"
cmake --preset emscripten
cmake --build --preset emscripten --target pls4all_wasm
npm --prefix bindings/js install
npm --prefix bindings/js run build
npm --prefix bindings/js run stage:wasm
npm --prefix bindings/js test
```
Before merging the ABI branch, require CI green for `ci.yml`, `abi-check.yml`, `catalog-validate.yml`, `version-sync.yml`, `parity-gate.yml`, `cross-binding-parity.yml`, and `docs.yml`.

Phase 5, docs/dashboard:
```bash
python docs/_extras/build_methods.py
python docs/_extras/build_landing.py --results benchmarks/cross_binding/results --out docs/_static/bench-data.json
python -m pytest benchmarks/cross_binding/tests/test_dashboard_contract.py benchmarks/cross_binding/tests/test_raw_manifest_reconciliation.py -q
sphinx-build -b html docs docs/_build/html --keep-going
```

Final pre-downstream parity sweep:
```bash
BENCH_SKLEARN_N_JOBS=8 python benchmarks/cross_binding/orchestrator.py --algorithms all --registry-cells --threads 8 --workers 1 --libn4m-build blas-omp --n-runs 2 --canonical-pls4all-only --reference-backends registry --timeout 180 --out-csv /tmp/n4m_full_parity.csv --force --flush-each-cell
```

Downstream order is right: `nirs4all-lite` first, then `nirs4all-web`, then light repos. Keep `nirs4all` main last. For `lite`, require its documented `make test`, `cargo test --workspace`, Python unittest, WASM npm test, R check, and Octave smoke where toolchains exist. For `web`, require `npm run typecheck`, `npm run test`, `npm run validate:catalog`, `npm run check:lite-shim`, `npm run build`, and `npm run build:single`.

**(D) Under-Estimated**

The big under-estimate is lockstep FFI churn: C symbols, Python `n4m`, Python `pls4all`, R, MEX, JS/WASM exports, ABI snapshots, SONAME, and CI expectations all move together.

Also under-estimated: WVC is not a naming cleanup; it is a real catalog cardinality problem. The 10 zero-ABI rows need explicit `c_surface: none` or real symbols. Bench/dashboard artifacts will churn even if numerics do not. Parity fixtures should not be numerically regenerated unless an actual algorithm changed.