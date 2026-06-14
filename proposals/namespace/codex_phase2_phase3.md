**A. Phase 2 Core**
GO.

I found no Phase 2 core blocker. Verified:

- `_rename_map.tsv`: 566 mappings, 566 unique old, 566 unique new, 0 collisions; convention `n4m_<top_level_role>_<leaf><tail>` holds for every row. The four irregulars are correct, including `n4m_pls_fit_simple -> n4m_estimators_pls_fit` and the three no-tail utility symbols.
- Worked examples matched, including Ridge, SNV, EPO, CARS, WVC threshold, AOM result accessors, transfer metrics, Hotelling T2, RMSE, and moments.
- Header split is correct: umbrella + 12 role headers + `transform` 10, `estimators` 4, `augmentation` 8, plus `lowlevel.h`. `pls.h` and the 10 old stubs are absent, with no compat include.
- The 10 `c_surface: none` rows have no header declaration and no snapshot/export pattern.
- ABI mechanics are right: [n4m_version.h](/home/delete/nirs4all/nirs4all-methods/cpp/include/n4m/n4m_version.h:20) is ABI `2.0.0`; [n4m_linux.map](/home/delete/nirs4all/nirs4all-methods/cpp/src/c_api/n4m_linux.map:12) uses `N4M_2`; [abi-check.yml](/home/delete/nirs4all/nirs4all-methods/.github/workflows/abi-check.yml:50) expects `libn4m.so.2`; [regen_abi_snapshots.sh](/home/delete/nirs4all/nirs4all-methods/scripts/regen_abi_snapshots.sh:14) is generic over `N4M_<n>`.
- Built `libn4m.so.2.0.0` exports exactly the Linux snapshot; SONAME is `libn4m.so.2`. Linux has 702 `n4m_*` symbols plus the Linux-only `N4M_2` node; macOS/Windows snapshots are exactly Linux minus that node.
- `scripts/bump_version.sh --check` reports only [bindings/python/src/n4m/_ffi.py](/home/delete/nirs4all/nirs4all-methods/bindings/python/src/n4m/_ffi.py:45) ABI constants `1.22.0 -> 2.0.0`. That is Phase 3 scope, not a missed core file.

I could not run `regen_abi_snapshots.sh --check` directly because this sandbox has read-only `/tmp`; I reproduced its `nm | diff` manually and it matched.

**B. Phase 3 Locked Plan**
Python:

- Bump canonical `bindings/python/src/n4m/_ffi.py` ABI constants to `2.0.0`; regenerate mirrors, do not hand-edit mirror constants.
- There is no usable checked-in FFI generator: `_ffi_decls.py` references `scripts/generate_ffi_decls.py`, but that file is absent ([ref](/home/delete/nirs4all/nirs4all-methods/bindings/python/src/n4m/_ffi_decls.py:6)).
- Add/recover `scripts/generate_ffi_decls.py`. It must parse public `N4M_API` prototypes from `cpp/include/n4m/**/*.h`, map C types to existing ctypes structs, emit `_ffi_decls.py`, and validate names exactly against `cpp/abi/expected_symbols_linux.txt` minus `N4M_2`. Use `_rename_map.tsv` only as migration validation/backstop; do not derive signatures from the `.so`.
- Replace the current flat/top-level compatibility surface ([current old shape](/home/delete/nirs4all/nirs4all-methods/bindings/python/src/n4m/__init__.py:4)) with role packages. `n4m.__init__` exposes only metadata/helpers plus subpackages, not classes/functions and not `Native*`.

Target Python layout:

```text
n4m/
  transform/{scatter,baseline,smoothing,wavelet,signal_conversion,resampling,scaling,alignment,orthogonalization,specialized}
  augmentation/{noise,drift,wavelength,spectral,scattering,instrument,splines,mixup}
  estimators/regression/{latent,regularized,robust,kernel,sparse,tensor,online,local,glm}
  estimators/{classification,multiblock,survival}
  feature_selection/{wrapper,filter,interval,ranking}
  model_selection/{splitters,aom_search,aom_campaign,sweep}
  domain_adaptation/{standardization,invariant,metrics,orthogonalization}
  outlier_detection/
  ensemble/
  compose/aom_superblock/
  metrics/{scoring,diagnostics}
  decomposition/
  lowlevel/moments.py
```

Public names are role names, not implementation names: `Ridge`, `RidgePLS`, `PLS`, `PCR`, `SNV`, `MSC`, `SavitzkyGolay`, `EPO`, `CARS`, `KennardStone`, `AOMRidgeBlender`, `AOMRidgePLSSuperblock`. No public `NativeRidgeRegressor`, no `n4m.python`.

`pls4all` remains the slim historical package per subset contract, but every underlying C call must use ABI-2 names. `bindings/python_nirs4all_methods` and `bindings/python_pls4all` are generated mirrors via [make_python_package.py](/home/delete/nirs4all/nirs4all-methods/bindings/python/scripts/make_python_package.py:4).

Python gates:

```bash
python3 scripts/generate_ffi_decls.py --check
N4M_LIB_PATH=$PWD/build/dev-release/cpp/src/libn4m.so.2.0.0 PYTHONPATH=bindings/python/src python3 -m pytest bindings/python/tests -q
python3 bindings/python/scripts/make_python_package.py --all
N4M_LIB_PATH=$PWD/build/dev-release/cpp/src/libn4m.so.2.0.0 PYTHONPATH=bindings/python_nirs4all_methods/src python3 -m pytest bindings/python_nirs4all_methods/tests -q
N4M_LIB_PATH=$PWD/build/dev-release/cpp/src/libn4m.so.2.0.0 PYTHONPATH=bindings/python_pls4all/src python3 -m pytest bindings/python_pls4all/tests -q
ruff format --check bindings/python bindings/python_nirs4all_methods bindings/python_pls4all
ruff check bindings/python bindings/python_nirs4all_methods bindings/python_pls4all
scripts/bump_version.sh --check
```

Other bindings:

- R: replace all old C calls in `bindings/r/n4m/src` and `bindings/r/pls4all/src` using `_rename_map.tsv`; keep `.Call` names internal, update public R wrappers/docs to the locked R role naming. No compile-time ABI constant found; gate with `n4m_abi_version()[1:2] == c(2,0)`.
- MATLAB/Octave: update MEX C calls, e.g. [n4m_pls_fit_simple](/home/delete/nirs4all/nirs4all-methods/bindings/matlab/mex/n4m_pls_fit_mex.c:115) becomes `n4m_estimators_pls_fit`. MEX filenames can remain MATLAB internals unless public API files are renamed.
- JS/WASM: `EXPORTED_FUNCTIONS` is already snapshot-driven; update direct `ccall/cwrap` names, e.g. [model.ts](/home/delete/nirs4all/nirs4all-methods/bindings/js/src/model.ts:62), and all direct calls in `wasm_entry.c`; regenerate `dist`.
- JNI/Android: archived only under `bindings/_archive`, no active merge gate.

Binding gates:

```bash
PLS4ALL_LIB_DIR=$PWD/build/dev-release/cpp/src PLS4ALL_INCLUDE_DIR=$PWD/cpp/include PLS4ALL_GENERATED_DIR=$PWD/build/dev-release/generated R CMD INSTALL bindings/r/n4m
Rscript -e 'testthat::test_dir("bindings/r/n4m/tests/testthat", reporter="summary")'
Rscript bindings/r/test_parity.R

PLS4ALL_LIB_DIR=$PWD/build/dev-release/cpp/src PLS4ALL_INCLUDE_DIR=$PWD/cpp/include PLS4ALL_GENERATED_DIR=$PWD/build/dev-release/generated R CMD INSTALL bindings/r/pls4all
Rscript -e 'testthat::test_dir("bindings/r/pls4all/tests/testthat", reporter="summary")'

matlab -batch "cd('bindings/matlab'); build_mex; results=runtests('test'); assertSuccess(results)"
octave --quiet --eval "cd('bindings/matlab'); build_mex; run('test/test_parity.m'); run('test/test_matlab_tier2.m')"

cmake --preset emscripten
cmake --build --preset emscripten --target pls4all_wasm --parallel
(cd bindings/js && npm run build && npm run stage:wasm && npm test)
```

MATLAB and Emscripten are CI/env-bound if the local runner lacks those toolchains.

**C. Top Risks**
- Missing FFI generator is the biggest Phase 3 risk; do not hand-edit `_ffi_decls.py`.
- Stale generated/binary artifacts can shadow tests; always use `N4M_LIB_PATH` for Python.
- Python public API migration is a clean break: no top-level compatibility re-exports and no public `Native*`.
- The 10 `c_surface: none` methods must stay Python-only; no placeholder C calls.
- R/MATLAB/JS have many old symbol call sites; use `_rename_map.tsv` mechanically, then compile gates catch signatures.