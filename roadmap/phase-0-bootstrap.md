# Phase 0 — Bootstrap

## Goal

Stand up nirs4all-methods by cloning the [pls4all](https://github.com/GBeurier/pls4all) template, renaming all symbols/paths from `p4a/pls4all` to `n4m/nirs4all-methods`, stripping every PLS-algorithm-specific file, and verifying a clean smoke build with `n4m_cli --selfcheck` passing 7 unit tests.

## Out of scope

- Implementing any preprocessing, splitter, filter, augmentation, or NIRS utility (Phases 2+).
- RNG, PCA helper, banded solver, wavelet kernels (Phase 1).
- Regenerating any parity fixtures (Phases 2+).
- Updating CI workflows (will be re-enabled in Phase 1 once the C ABI surface is wired).

## Files created / modified

| Path | Action |
|------|--------|
| `cpp/include/n4m/n4m.h` | renamed from `p4a.h`; declares Phase 0 surface (status codes, dtype/backend enums, matrix view, context, version queries). Full operator surface comes phase by phase. |
| `cpp/include/n4m/n4m_version.h` | reset to project 0.1.0, ABI 1.0.0 |
| `cpp/include/n4m/n4m_export.h.in` | renamed, s/P4A/N4M/g |
| `cpp/src/core/{config,context,matrix_view,status,version}.cpp` | renamed namespace `pls4all` → `nirs4all-methods`. All PLS-specific core files deleted. |
| `cpp/src/c_api/{c_api_context,c_api_matrix,c_api_version}.cpp` | kept (only the version/context/matrix wrappers). All PLS C-API wrappers deleted. |
| `cpp/src/CMakeLists.txt` | gutted to compile only the 5 core files + 3 c_api files. |
| `cpp/cli/n4m_cli.cpp` | rewritten as minimal `--version`/`--abi-info`/`--selfcheck`. PLS pipeline smoke removed. |
| `cpp/tests/{main.cpp,harness.hpp,CMakeLists.txt}` | rewritten as Phase 0 smoke harness with 7 tests. `cpp/tests/{abi,fixtures}/` deleted. |
| `cpp/abi/expected_symbols_{linux,macos,windows}.txt` | emptied (no exported symbols yet; Phase 1 will repopulate). |
| `CMakeLists.txt`, `CMakePresets.json` | all `pls4all`/`Pls4all`/`PLS4ALL` → `nirs4all-methods`/`N4M`/`N4M`. |
| `cmake/{C4aOptions,CompilerWarnings,SanitizersToolchain,C4aConfig.cmake.in}` | renamed from `Pls4all*`. |
| `ROADMAP.md`, `ARCHITECTURE.md`, `CHANGELOG.md`, `README.md` | written fresh for nirs4all-methods (replaces pls4all originals). |
| `bindings/python/src/n4m/{_ffi,_context,_errors,_types,__init__}.py` | renamed module, content sed-renamed. `_aom.py`, `_config.py`, `_methods.py`, `_model.py`, `sklearn/*` (except `_base.py`) deleted. |
| `bindings/{r,matlab,go,rust,lua,nim,julia,ruby,dotnet}/` | path renames `pls4all → nirs4all-methods`, content sed-renamed. |
| `parity/python_generator/src/n4m_parity/` | renamed module. Adapters to be rewritten in Phases 2+. |

## Strip list (deleted)

- `cpp/src/core/`: `pls_*`, `simpls*`, `opls*`, `aom_*`, `*_selection.{cpp,hpp}`, `model.{cpp,hpp}`, `pipeline.{cpp,hpp}`, `metrics.{cpp,hpp}`, `classification_metrics.{cpp,hpp}`, `cross_validation.{cpp,hpp}`, `validation.{cpp,hpp}`, `variable_importance.{cpp,hpp}`, `variable_selection.{cpp,hpp}`, `extra_pls.{cpp,hpp}`, `kernel_pls.{cpp,hpp}`, `lw_pls.{cpp,hpp}`, `mb_pls.{cpp,hpp}`, `recursive_pls.{cpp,hpp}`, `tensor_pls.{cpp,hpp}`, `gpr_pls.{cpp,hpp}`, `multiblock_extensions.{cpp,hpp}`, `pls_diagnostics.{cpp,hpp}`, `pls_monitoring.{cpp,hpp}`, `pls_lda.{cpp,hpp}`, `pls_logistic.{cpp,hpp}`, `ecr.{cpp,hpp}`, `gating_strategy.{cpp,hpp}`, `operator_bank.{cpp,hpp}`, `model_selection.{cpp,hpp}`, `component_coefficients.{cpp,hpp}`, `cuda_dispatch.{cpp,hpp}`, `method_result.hpp`, `operator_entry.hpp`.
- `cpp/src/c_api/`: `c_api_aom_selection.cpp`, `c_api_config.cpp`, `c_api_gating_strategy.cpp`, `c_api_method_result.cpp`, `c_api_model.cpp`, `c_api_operator_bank.cpp`, `c_api_pipeline.cpp`, `c_api_validation_plan.cpp`, `p4a_linux.map`.
- `cpp/tests/abi/*.cpp`, `cpp/tests/fixtures/*.hpp` — every PLS test.
- `parity/fixtures/*.json` — every pls4all fixture (regenerated phase by phase).
- `roadmap/phase-*.md` — every pls4all phase plan.
- `Overview.md`, `Direction_Technique.md`, `DISTRIBUTION.md`, `Backlog.md`, `CHANGELOG.md` (old).
- `bindings/python/src/pls4all/{_aom,_config,_methods,_model}.py`, `bindings/python/src/pls4all/sklearn/{_classification,_classifiers_extras,_diagnostics,_in_sample,_method_result,_method_result_regressors,_regression,_selection,_transformers}.py`.
- All leaked build artefacts: `build/`, `bindings/rust/target/`, `bindings/dotnet/*/bin`, `bindings/dotnet/*/obj`, `bindings/matlab/+n4m/*.mex`, `benchmarks/cross_binding/data`, `benchmarks/results/`, `benchmarks/parity_timing/results/`.

## ABI surface (Phase 0)

```
(empty — n4m_get_*, n4m_status_to_string, n4m_backend_*, n4m_dtype_*, n4m_check_abi_compatibility,
         n4m_context_{create,destroy,set_seed,get_seed,set_backend,get_backend,set_num_threads,get_num_threads,last_error,clear_error,set_user_data,get_user_data},
         n4m_matrix_view_{init_rowmajor,init_colmajor,init_strided,validate}
 — all already declared in n4m.h, all already compiled into libn4m, NO new operators yet)
```

The expected_symbols_*.txt files are empty at Phase 0 by design. They are repopulated in Phase 1 with the actual symbols once `nm -D libn4m.so` is run on a clean build and the list is committed.

## Acceptance criteria

- ✅ `cmake --preset dev-debug` configures cleanly.
- ✅ `cmake --build --preset dev-debug` builds `libn4m.so.1.0.0`, `libn4m_static.a`, `n4m_cli`, `n4m_tests` with zero warnings.
- ✅ `./build/dev-debug/cpp/cli/n4m_cli --version` prints `nirs4all-methods 0.1.0+abi.1.0.0 (ABI 1.0.0)`.
- ✅ `./build/dev-debug/cpp/cli/n4m_cli --selfcheck` exits 0.
- ✅ `ctest --preset dev-debug` reports `100% tests passed, 0 tests failed out of 1`.
- ✅ Repo pushed to `github.com/GBeurier/nirs4all-methods` (public).
- ✅ Opus-max independent review transcript committed at `docs/reviews/phase-0/opus-post.md`.

## Verification

```bash
cd /home/delete/nirs4all/nirs4all-methods
cmake --preset dev-release
cmake --build --preset dev-release
./build/dev-release/cpp/cli/n4m_cli --selfcheck
ctest --preset dev-release --output-on-failure
```

All four commands must exit 0.

## Next phase

[Phase 1 — Common infrastructure](phase-1-common.md): PCG64 RNG, PCA helper, banded LDLT solver, wavelet filter banks, B-spline.
