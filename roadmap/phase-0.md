# Phase 0 — ABI & Build Foundation

> Status: revision 3 · post-Codex-re-review · 2026-05-14
>
> Canonical spec: [`Direction_Technique.md`](../Direction_Technique.md).
> Algorithm taxonomy: [`Overview.md`](../Overview.md).
> Codex review transcripts: [`docs/reviews/phase-0/0001-roadmap-review.md`](../docs/reviews/phase-0/0001-roadmap-review.md) (rev 1 review) · [`docs/reviews/phase-0/0002-roadmap-rereview.md`](../docs/reviews/phase-0/0002-roadmap-rereview.md) (rev 2 re-review).
> Master plan: `/home/delete/.claude/plans/je-veux-fournir-une-binary-music.md`.

## 1. Objective

Ship a callable `libp4a.{so,dll,dylib}` with **no PLS algorithm yet**. Phase 0 is pure plumbing: the C ABI surface, the build matrix, the parity infrastructure, the bindings skeletons, the docs. Phase 1 must plug NIPALS into this chassis without changing a single public symbol — and Phase 1's surface is therefore declared (as stubs) at Phase 0.

Phase 0 acceptance criteria — objective and CI-enforced:

1. `cmake --preset ci-<name> && cmake --build --preset ci-<name> && ctest --preset ci-<name>` exits 0 on **all 10 CI matrix cells** (see §5.1) plus the macOS universal2 cell. Sanitizer jobs (ASAN, UBSAN, TSAN) are also green on Linux.
2. The shared library exports **only** `p4a_*` symbols (plus platform-required `_init`, `_fini`, `_edata`, `_end`, `__bss_start` on ELF). The list is diff-checked against checked-in golden files `cpp/abi/expected_symbols_{linux,macos,windows}.txt`. Any drift fails the PR.
3. The runtime-dependency audit step (`ldd` / `otool -L` / `dumpbin /DEPENDENTS`) reports no link against OpenBLAS, MKL, Boost, pybind11, Rcpp, embind, nlohmann-json, yaml-cpp or any other forbidden dependency.
4. `pls4all_cli --version`, `--abi-info`, `--selfcheck` exit 0 and print the expected strings.
5. Each binding skeleton (Python, R, MATLAB, JS, Android) compiles a hello-version test that loads `libp4a` and prints the same version string as the C CLI.
6. The three canonical fixtures (`synthetic_small_pls1_v1`, `synthetic_small_pls2_v1`, `synthetic_tiny_centered_v1`) round-trip bit-exactly through **both** the Python and R generators (SHA-256 in `manifest.json` matches checked-in values).
7. `test_fixture_loader.cpp` asserts that calling `p4a_model_fit` against any Phase 0 fixture returns `P4A_ERR_NOT_IMPLEMENTED` — this is the test Phase 1 will invert.
8. Codex's final review on the `phase-0` tag returns no blocking objections; the transcript is checked into `docs/reviews/phase-0/`.

## 2. Repository tree

```
pls4all/
├── cpp/
│   ├── CMakeLists.txt
│   ├── include/pls4all/
│   │   ├── p4a.h                             # public C ABI — only header users see
│   │   ├── p4a_version.h                     # compile-time ABI/project version macros
│   │   └── p4a_export.h.in                   # P4A_API visibility macro (configured by CMake)
│   ├── src/
│   │   ├── CMakeLists.txt
│   │   ├── core/                             # internal C++17 — never exposed
│   │   │   ├── version.{cpp,hpp}
│   │   │   ├── status.{cpp,hpp}
│   │   │   ├── context.{cpp,hpp}
│   │   │   ├── config.{cpp,hpp}
│   │   │   ├── matrix_view.{cpp,hpp}
│   │   │   ├── error_buffer.{cpp,hpp}
│   │   │   ├── operator_bank.{cpp,hpp}       # Phase 0 surface only
│   │   │   ├── pipeline.{cpp,hpp}            # Phase 0 surface only
│   │   │   ├── gating_strategy.{cpp,hpp}     # Phase 0 surface only
│   │   │   ├── fault_injection.{cpp,hpp}     # test-only, PLS4ALL_ENABLE_FAULT_INJECTION
│   │   │   ├── linalg/.gitkeep               # Phase 1
│   │   │   ├── pls/.gitkeep                  # Phase 1
│   │   │   ├── preprocessing/.gitkeep        # Phase 3
│   │   │   ├── selection/.gitkeep            # Phase 5
│   │   │   ├── validation/.gitkeep           # Phase 3
│   │   │   └── serialization/.gitkeep        # Phase 1
│   │   └── c_api/                            # extern "C" wrappers — every fn is noexcept
│   │       ├── c_api_version.cpp
│   │       ├── c_api_context.cpp
│   │       ├── c_api_config.cpp
│   │       ├── c_api_matrix.cpp
│   │       ├── c_api_error.cpp
│   │       ├── c_api_operator_bank_stub.cpp
│   │       ├── c_api_pipeline_stub.cpp
│   │       ├── c_api_model_stub.cpp          # fit/predict/transform/serialize → NOT_IMPLEMENTED
│   │       └── pls4all.def                   # MSVC export gate
│   ├── tests/
│   │   ├── CMakeLists.txt
│   │   ├── doctest/
│   │   │   ├── doctest.h                     # vendored v2.4.11 (MIT, SHA-256 pinned)
│   │   │   └── json.hpp                      # vendored nlohmann/json v3.11.3 (MIT) — TEST ONLY
│   │   ├── abi/
│   │   │   ├── test_lifecycle.cpp
│   │   │   ├── test_config_setters.cpp
│   │   │   ├── test_status_codes.cpp
│   │   │   ├── test_version.cpp
│   │   │   ├── test_matrix_view.cpp
│   │   │   ├── test_error_messages.cpp
│   │   │   ├── test_exception_safety.cpp     # exception → C status translation
│   │   │   └── test_oom_paths.cpp
│   │   ├── concurrency/
│   │   │   ├── test_multi_context_threads.cpp
│   │   │   └── test_fork_smoke.cpp           # POSIX only
│   │   ├── fuzz/                             # libFuzzer harnesses
│   │   │   ├── fuzz_config_setters.cpp
│   │   │   ├── fuzz_matrix_view_validate.cpp
│   │   │   ├── fuzz_error_buffer_format.cpp
│   │   │   └── fuzz_fixture_loader.cpp
│   │   ├── fixture/
│   │   │   └── test_fixture_loader.cpp
│   │   ├── numerical/.gitkeep                # Phase 1
│   │   └── property/.gitkeep                 # Phase 1
│   ├── cli/
│   │   ├── CMakeLists.txt
│   │   └── p4a_cli.cpp                       # --version / --abi-info / --selfcheck
│   └── abi/
│       ├── expected_symbols_linux.txt
│       ├── expected_symbols_macos.txt
│       └── expected_symbols_windows.txt
│
├── bindings/                                 # see master plan; each ships hello-version test
│   ├── python/   {pyproject.toml, pls4all/{__init__,_ffi,_types,_errors,_version}.py, tests/}
│   ├── r/        {pls4all/{DESCRIPTION,NAMESPACE,R/, src/, tests/testthat/}}
│   ├── matlab/   {+pls4all/version.m, mex/{p4a_version_mex.cpp,build_mex.m}, tests/}
│   ├── js/       {package.json, tsconfig.json, emscripten_build.sh, src/, test/}
│   └── android/  {settings.gradle.kts, pls4all-android/{build.gradle.kts,src/{cpp,kotlin}/}}
│
├── parity/
│   ├── README.md
│   ├── tolerances.md
│   ├── schema/ {fixture_schema_v1.json, fixture_schema_v1.md}
│   ├── fixtures/ {manifest.json, synthetic_small_pls1_v1.json,
│   │              synthetic_small_pls2_v1.json, synthetic_tiny_centered_v1.json}
│   ├── python_generator/
│   │   ├── pyproject.toml, requirements-lock.txt
│   │   ├── generate_fixtures.py
│   │   ├── suites/{synthetic.py, nirs_real.py}
│   │   ├── adapters/{sklearn_adapter.py, nirs4all_adapter.py}
│   │   └── tests/test_generator_determinism.py
│   └── r_generator/
│       ├── DESCRIPTION, renv.lock
│       ├── generate_fixtures.R                # produces all 3 canonical fixtures
│       └── adapters/{pls_adapter.R, ropls_adapter.R, mixOmics_adapter.R}
│
├── cmake/  {Pls4allConfig.cmake.in, Pls4allTargets.cmake.in, CompilerWarnings.cmake,
│           SanitizersToolchain.cmake, Pls4allOptions.cmake, modules/CodeCoverage.cmake}
├── packaging/  {python/, r/, js/, conda/meta.yaml.in, vcpkg/portfile.cmake}
├── examples/   {c/, python/, r/}
├── bench/cpp/  {CMakeLists.txt, bench_noop.cpp}
├── docs/       {index.md, architecture/, abi/, bindings/, parity/, dev/, reviews/phase-0/}
├── CMakeLists.txt        # top-level
├── CMakePresets.json
└── .github/    {workflows/, ISSUE_TEMPLATE/, PULL_REQUEST_TEMPLATE.md, CODEOWNERS, dependabot.yml}
```

## 3. CMake structure

### 3.1 Top-level (`CMakeLists.txt`)

```cmake
cmake_minimum_required(VERSION 3.21)
project(pls4all
    VERSION 0.1.0
    DESCRIPTION "Portable PLS/NIRS engine with stable C ABI"
    LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_C_VISIBILITY_PRESET   hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(Pls4allOptions)
include(CompilerWarnings)        # pls4all_add_warnings(target)
include(SanitizersToolchain)

add_subdirectory(cpp/src)
if(PLS4ALL_BUILD_TESTS)
    enable_testing()
    add_subdirectory(cpp/tests)
endif()
if(PLS4ALL_BUILD_CLI)
    add_subdirectory(cpp/cli)
endif()
if(PLS4ALL_BUILD_EXAMPLES)
    add_subdirectory(examples/c)
endif()
if(PLS4ALL_BUILD_BENCH)
    add_subdirectory(bench/cpp)
endif()
if(PLS4ALL_BUILD_BINDINGS_R)
    add_subdirectory(bindings/r/pls4all)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Install rules are scoped to the targets that are actually built.
set(_pls4all_install_targets "")
if(PLS4ALL_BUILD_SHARED)
    list(APPEND _pls4all_install_targets pls4all_c)
endif()
if(PLS4ALL_BUILD_STATIC)
    list(APPEND _pls4all_install_targets pls4all_c_static)
endif()
if(PLS4ALL_BUILD_CLI)
    list(APPEND _pls4all_install_targets pls4all_cli)
endif()
if(_pls4all_install_targets)
    install(TARGETS ${_pls4all_install_targets}
            EXPORT pls4all_targets
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
install(DIRECTORY cpp/include/pls4all DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

configure_package_config_file(
    cmake/Pls4allConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/Pls4allConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pls4all)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/Pls4allConfigVersion.cmake
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/Pls4allConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/Pls4allConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pls4all)
install(EXPORT pls4all_targets
        FILE Pls4allTargets.cmake
        NAMESPACE pls4all::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pls4all)
```

### 3.2 Options (`cmake/Pls4allOptions.cmake`)

```
PLS4ALL_BUILD_SHARED        [ON]    pls4all_c shared library
PLS4ALL_BUILD_STATIC        [ON]    pls4all_c_static (for Android predict-only embedders)
PLS4ALL_BUILD_TESTS         [ON]    doctest runner + fuzz harnesses
PLS4ALL_BUILD_FUZZ          [OFF]   libFuzzer harnesses (requires clang)
PLS4ALL_BUILD_CLI           [ON]    pls4all_cli executable
PLS4ALL_BUILD_EXAMPLES      [OFF]
PLS4ALL_BUILD_BENCH         [OFF]
PLS4ALL_BUILD_BINDINGS_PYTHON   [OFF]
PLS4ALL_BUILD_BINDINGS_R        [OFF]
PLS4ALL_BUILD_BINDINGS_MATLAB   [OFF]
PLS4ALL_BUILD_BINDINGS_JS       [OFF]   # ON when CMAKE_TOOLCHAIN_FILE is Emscripten
PLS4ALL_BUILD_BINDINGS_ANDROID  [OFF]   # ON when CMAKE_TOOLCHAIN_FILE is Android NDK
PLS4ALL_WITH_BLAS               [OFF]   # Phase 7
PLS4ALL_WITH_OPENMP             [OFF]   # Phase 7
PLS4ALL_WITH_CUDA               [OFF]   # Phase 7
PLS4ALL_ENABLE_ASAN             [OFF]
PLS4ALL_ENABLE_UBSAN            [OFF]
PLS4ALL_ENABLE_TSAN             [OFF]
PLS4ALL_ENABLE_COVERAGE         [OFF]
PLS4ALL_ENABLE_LTO              [OFF]
PLS4ALL_ENABLE_FAULT_INJECTION  [OFF]
PLS4ALL_WARNINGS_AS_ERRORS      [OFF]   # ON in CI
```

### 3.3 `cpp/src/CMakeLists.txt`

```cmake
add_library(pls4all_core STATIC
    core/version.cpp
    core/status.cpp
    core/context.cpp
    core/config.cpp
    core/matrix_view.cpp
    core/error_buffer.cpp
    core/operator_bank.cpp
    core/pipeline.cpp
    core/gating_strategy.cpp)
target_include_directories(pls4all_core
    PUBLIC  ${CMAKE_SOURCE_DIR}/cpp/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(pls4all_core PUBLIC cxx_std_17)
target_compile_definitions(pls4all_core PRIVATE PLS4ALL_BUILDING_CORE=1)
if(PLS4ALL_ENABLE_FAULT_INJECTION)
    target_sources(pls4all_core PRIVATE core/fault_injection.cpp)
    target_compile_definitions(pls4all_core PUBLIC PLS4ALL_ENABLE_FAULT_INJECTION=1)
endif()
pls4all_add_warnings(pls4all_core)

# Dead-code stripping in non-Debug configs (multi-config-safe via generator
# expressions — CMAKE_BUILD_TYPE is empty under MSVC / Xcode multi-config).
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(pls4all_core PRIVATE
        $<$<NOT:$<CONFIG:Debug>>:-ffunction-sections>
        $<$<NOT:$<CONFIG:Debug>>:-fdata-sections>)
endif()

configure_file(
    ${CMAKE_SOURCE_DIR}/cpp/include/pls4all/p4a_export.h.in
    ${CMAKE_BINARY_DIR}/generated/pls4all/p4a_export.h
    @ONLY)

set(PLS4ALL_C_SOURCES
    c_api/c_api_version.cpp
    c_api/c_api_context.cpp
    c_api/c_api_config.cpp
    c_api/c_api_matrix.cpp
    c_api/c_api_error.cpp
    c_api/c_api_operator_bank_stub.cpp
    c_api/c_api_pipeline_stub.cpp
    c_api/c_api_model_stub.cpp)

function(pls4all_apply_c_target target)
    target_link_libraries(${target} PRIVATE pls4all_core)
    target_include_directories(${target}
        PUBLIC  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/cpp/include>
                $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/generated>
                $<INSTALL_INTERFACE:include>)
    # Apply dead-code stripping to the C API sources as well as the core.
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE
            $<$<NOT:$<CONFIG:Debug>>:-ffunction-sections>
            $<$<NOT:$<CONFIG:Debug>>:-fdata-sections>)
    endif()
endfunction()

if(PLS4ALL_BUILD_SHARED)
    add_library(pls4all_c SHARED ${PLS4ALL_C_SOURCES})
    pls4all_apply_c_target(pls4all_c)
    set_target_properties(pls4all_c PROPERTIES
        OUTPUT_NAME p4a
        SOVERSION  ${PROJECT_VERSION_MAJOR}
        VERSION    ${PROJECT_VERSION})
    target_compile_definitions(pls4all_c PRIVATE PLS4ALL_BUILDING_DLL=1)

    # Undefined-symbol enforcement + dead-code stripping at link time.
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        target_link_options(pls4all_c PRIVATE
            -Wl,--no-undefined
            $<$<NOT:$<CONFIG:Debug>>:-Wl,--gc-sections>)
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        target_link_options(pls4all_c PRIVATE
            -Wl,-undefined,error
            $<$<NOT:$<CONFIG:Debug>>:-Wl,-dead_strip>)
    elseif(MSVC)
        target_link_options(pls4all_c PRIVATE
            $<$<NOT:$<CONFIG:Debug>>:/OPT:REF>
            $<$<NOT:$<CONFIG:Debug>>:/OPT:ICF>)
        target_sources(pls4all_c PRIVATE c_api/pls4all.def)
    endif()
endif()

if(PLS4ALL_BUILD_STATIC)
    add_library(pls4all_c_static STATIC ${PLS4ALL_C_SOURCES})
    pls4all_apply_c_target(pls4all_c_static)
    set_target_properties(pls4all_c_static PROPERTIES OUTPUT_NAME p4a_static)
endif()

# Install the generated p4a_export.h alongside the source headers so consumers
# find it under <prefix>/include/pls4all/p4a_export.h.
install(FILES ${CMAKE_BINARY_DIR}/generated/pls4all/p4a_export.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/pls4all)
```

### 3.4 Visibility macro template (`p4a_export.h.in`)

Same as revision 1 (`__attribute__((visibility("default")))` / `__declspec(dllexport)` etc).

### 3.5 `CMakePresets.json`

Preset names match the CI matrix labels in §5.1 exactly, so the CI YAML can be written as `cmake --preset ci-<label>` with no translation layer.

- Dev: `dev-debug`, `dev-release`
- CI build matrix: `ci-linux-gcc12-release`, `ci-linux-gcc12-debug`, `ci-linux-gcc13-release`, `ci-linux-clang16-release`, `ci-linux-clang16-debug`, `ci-macos-clang-release`, `ci-macos-clang-debug`, `ci-windows-msvc-release`, `ci-windows-msvc-debug`, `ci-windows-mingw-release`, `ci-macos-universal2`
- Sanitizers: `ci-asan`, `ci-ubsan`, `ci-tsan`, `ci-asan_ubsan`
- Coverage: `ci-coverage`
- Cross-compile (Phase 2): `emscripten`, `android-arm64`, `android-x86_64`

## 4. C ABI surface

Every Phase 1+ surface is declared as a stub at Phase 0. The Phase 0 implementation answers correctly for lifecycle / config / version / matrix-view / operator-bank/pipeline lifecycle; any function that needs a fitted model or a real algorithm returns `P4A_ERR_NOT_IMPLEMENTED`.

### 4.0 Prelude — universal rules

**Every `P4A_API` function is `noexcept`.** Status-returning wrappers translate every exception:

```cpp
try {
    // C++ internal call
    return P4A_OK;
} catch (const std::bad_alloc&) {
    set_last_error(ctx, "out of memory in <fn>");
    return P4A_ERR_OUT_OF_MEMORY;
} catch (const std::exception& e) {
    set_last_error(ctx, "%s in <fn>", e.what());
    return P4A_ERR_INTERNAL;
} catch (...) {
    set_last_error(ctx, "non-standard exception in <fn>");
    return P4A_ERR_INTERNAL;
}
```

`void` wrappers (destroy/free) catch and **swallow** after best-effort cleanup — never propagate, never crash.

**Forbidden** in the public header: `std::*`, `Eigen::*`, any C++ class, template, exception, reference, function-pointer-with-noexcept-qualifier (which is C++17, not C).

### 4.1 Status codes

```c
typedef enum p4a_status_t {
    P4A_OK                        =   0,
    P4A_ERR_INVALID_ARGUMENT      =   1,
    P4A_ERR_NULL_POINTER          =   2,
    P4A_ERR_SHAPE_MISMATCH        =   3,
    P4A_ERR_DTYPE_MISMATCH        =   4,
    P4A_ERR_STRIDE_INVALID        =   5,
    P4A_ERR_NOT_FITTED            =   6,
    P4A_ERR_NUMERICAL_FAILURE     =   7,
    P4A_ERR_CONVERGENCE_FAILED    =   8,
    P4A_ERR_OUT_OF_MEMORY         =   9,
    P4A_ERR_UNSUPPORTED           =  10,
    P4A_ERR_NOT_IMPLEMENTED       =  11,   /* Phase 0 stubs return this */
    P4A_ERR_ABI_MISMATCH          =  12,
    P4A_ERR_IO                    =  13,
    P4A_ERR_CORRUPT_BUFFER        =  14,
    P4A_ERR_VERSION_INCOMPATIBLE  =  15,
    P4A_ERR_BACKEND_UNAVAILABLE   =  16,
    P4A_ERR_CANCELLED             =  17,
    P4A_ERR_INTERNAL              = 255
} p4a_status_t;

P4A_API const char* p4a_status_to_string(p4a_status_t s);
```

### 4.2 Backend / dtype / matrix view

```c
typedef enum p4a_backend_t {
    P4A_BACKEND_AUTO            = 0,
    P4A_BACKEND_REFERENCE_CPU   = 1,
    P4A_BACKEND_NATIVE_CPU      = 2,
    P4A_BACKEND_BLAS            = 3,
    P4A_BACKEND_OPENMP          = 4,
    P4A_BACKEND_CUDA            = 5,
    P4A_BACKEND_OPENCL          = 6,
    P4A_BACKEND_METAL           = 7
} p4a_backend_t;

P4A_API int          p4a_backend_is_available(p4a_backend_t backend);
P4A_API const char*  p4a_backend_to_string(p4a_backend_t backend);

typedef enum p4a_dtype_t {
    P4A_DTYPE_UNKNOWN = 0,
    P4A_DTYPE_F64     = 1,
    P4A_DTYPE_F32     = 2,
    P4A_DTYPE_I32     = 3,
    P4A_DTYPE_I64     = 4
} p4a_dtype_t;

P4A_API size_t       p4a_dtype_size(p4a_dtype_t dtype);
P4A_API const char*  p4a_dtype_to_string(p4a_dtype_t dtype);

typedef struct p4a_matrix_view_t {
    void*        data;
    int64_t      rows;
    int64_t      cols;
    int64_t      row_stride;   /* in *elements*, not bytes */
    int64_t      col_stride;
    p4a_dtype_t  dtype;
    int32_t      reserved0;    /* padding — 48 bytes total on LP64; not used on ILP32 */
} p4a_matrix_view_t;

P4A_API p4a_status_t p4a_matrix_view_init_rowmajor(
    p4a_matrix_view_t* out, void* data,
    int64_t rows, int64_t cols, p4a_dtype_t dtype);
P4A_API p4a_status_t p4a_matrix_view_init_colmajor(
    p4a_matrix_view_t* out, void* data,
    int64_t rows, int64_t cols, p4a_dtype_t dtype);
P4A_API p4a_status_t p4a_matrix_view_init_strided(
    p4a_matrix_view_t* out, void* data,
    int64_t rows, int64_t cols,
    int64_t row_stride, int64_t col_stride,
    p4a_dtype_t dtype);
P4A_API p4a_status_t p4a_matrix_view_validate(const p4a_matrix_view_t* v);
```

`_validate` accepts positive strides (transposed and slice views are valid) and **rejects negative strides** (Phase 0 design choice; revisited only if a binding genuinely needs them — none does currently).

### 4.3 Opaque handles

```c
typedef struct p4a_context_s         p4a_context_t;
typedef struct p4a_config_s          p4a_config_t;
typedef struct p4a_operator_bank_s   p4a_operator_bank_t;
typedef struct p4a_gating_strategy_s p4a_gating_strategy_t;
typedef struct p4a_pipeline_s        p4a_pipeline_t;
typedef struct p4a_model_s           p4a_model_t;
typedef struct p4a_array_s           p4a_array_t;
```

### 4.4 Context lifecycle (unchanged from rev 1)

```c
P4A_API p4a_status_t p4a_context_create(p4a_context_t** out_ctx);
P4A_API void         p4a_context_destroy(p4a_context_t* ctx);

P4A_API p4a_status_t p4a_context_set_seed(p4a_context_t* ctx, uint64_t seed);
P4A_API p4a_status_t p4a_context_get_seed(const p4a_context_t* ctx, uint64_t* out_seed);

P4A_API p4a_status_t p4a_context_set_backend(p4a_context_t* ctx, p4a_backend_t backend);
P4A_API p4a_status_t p4a_context_get_backend(const p4a_context_t* ctx, p4a_backend_t* out);

P4A_API p4a_status_t p4a_context_set_num_threads(p4a_context_t* ctx, int32_t n);
P4A_API p4a_status_t p4a_context_get_num_threads(const p4a_context_t* ctx, int32_t* out);

P4A_API const char*  p4a_context_last_error(const p4a_context_t* ctx);  /* never NULL */
P4A_API void         p4a_context_clear_error(p4a_context_t* ctx);

P4A_API p4a_status_t p4a_context_set_user_data(p4a_context_t* ctx, void* user);
P4A_API void*        p4a_context_get_user_data(const p4a_context_t* ctx);
```

### 4.5 Config lifecycle (composable algorithm × solver × pipeline × bank × gating)

```c
typedef enum p4a_algorithm_t {
    P4A_ALGO_PLS_REGRESSION = 0,
    P4A_ALGO_PLS_CANONICAL  = 1,
    P4A_ALGO_PLS_SVD        = 2,
    P4A_ALGO_PLS_DA         = 3,
    P4A_ALGO_OPLS           = 4,
    P4A_ALGO_OPLS_DA        = 5,
    P4A_ALGO_SPARSE_PLS     = 6,
    P4A_ALGO_MB_PLS         = 7,
    P4A_ALGO_LW_PLS         = 8,
    P4A_ALGO_AOM_PLS        = 9,
    P4A_ALGO_PCR            = 10    /* baseline */
    /* Nonlinear kernel PLS (RBF / polynomial / sigmoid) lands in Phase 4 along
     * with the necessary kernel-type enum + setters. Deliberately not declared
     * at Phase 0 so we don't lock in a kernel-type surface before it's designed.
     * SIMPLS / kernel-algorithm / wide-kernel are SOLVER choices (see below),
     * not algorithms. POP is a GATING-STRATEGY mode, not an algorithm. */
} p4a_algorithm_t;

typedef enum p4a_solver_t {
    P4A_SOLVER_NIPALS            = 0,
    P4A_SOLVER_SIMPLS            = 1,
    P4A_SOLVER_ORTHOGONAL_SCORES = 2,
    P4A_SOLVER_KERNEL_ALGORITHM  = 3,   /* linear kernel-algorithm PLS (Lindgren/Wold) */
    P4A_SOLVER_WIDE_KERNEL       = 4,
    P4A_SOLVER_SVD               = 5,
    P4A_SOLVER_POWER             = 6,
    P4A_SOLVER_RANDOMIZED_SVD    = 7
} p4a_solver_t;

typedef enum p4a_deflation_t {
    P4A_DEFLATION_REGRESSION  = 0,
    P4A_DEFLATION_CANONICAL   = 1,
    P4A_DEFLATION_X_ONLY      = 2,
    P4A_DEFLATION_XY          = 3,
    P4A_DEFLATION_ORTHOGONAL  = 4
} p4a_deflation_t;

P4A_API p4a_status_t p4a_config_create(p4a_config_t** out_cfg);
P4A_API void         p4a_config_destroy(p4a_config_t* cfg);
P4A_API p4a_status_t p4a_config_clone(const p4a_config_t* src, p4a_config_t** out_dst);

/* Setters — every PLS knob from §22 phases is declared NOW so the surface
 * doesn't churn when Phase 1 / 4 / 6 land. Setters validate inputs and are
 * idempotent on rejection (state unchanged). */
P4A_API p4a_status_t p4a_config_set_algorithm        (p4a_config_t*, p4a_algorithm_t);
P4A_API p4a_status_t p4a_config_set_solver           (p4a_config_t*, p4a_solver_t);
P4A_API p4a_status_t p4a_config_set_deflation       (p4a_config_t*, p4a_deflation_t);
P4A_API p4a_status_t p4a_config_set_n_components    (p4a_config_t*, int32_t);
P4A_API p4a_status_t p4a_config_set_center_x        (p4a_config_t*, int32_t enabled);
P4A_API p4a_status_t p4a_config_set_scale_x         (p4a_config_t*, int32_t enabled);
P4A_API p4a_status_t p4a_config_set_center_y        (p4a_config_t*, int32_t enabled);
P4A_API p4a_status_t p4a_config_set_scale_y         (p4a_config_t*, int32_t enabled);
P4A_API p4a_status_t p4a_config_set_tol             (p4a_config_t*, double);
P4A_API p4a_status_t p4a_config_set_max_iter        (p4a_config_t*, int32_t);
P4A_API p4a_status_t p4a_config_set_store_scores    (p4a_config_t*, int32_t enabled);
P4A_API p4a_status_t p4a_config_set_store_diagnostics(p4a_config_t*, int32_t enabled);
P4A_API p4a_status_t p4a_config_set_dtype           (p4a_config_t*, p4a_dtype_t);

/* Composability hooks — non-owning pointers; lifetime managed by the caller. */
P4A_API p4a_status_t p4a_config_set_pipeline         (p4a_config_t*,
                                                       const p4a_pipeline_t* pipe);
P4A_API p4a_status_t p4a_config_set_operator_bank    (p4a_config_t*,
                                                       const p4a_operator_bank_t* bank);
P4A_API p4a_status_t p4a_config_set_gating_strategy  (p4a_config_t*,
                                                       const p4a_gating_strategy_t* gate);

/* Getters — exhaustive list. Every setter has a matching getter. */
P4A_API p4a_status_t p4a_config_get_algorithm        (const p4a_config_t*, p4a_algorithm_t*);
P4A_API p4a_status_t p4a_config_get_solver           (const p4a_config_t*, p4a_solver_t*);
P4A_API p4a_status_t p4a_config_get_deflation        (const p4a_config_t*, p4a_deflation_t*);
P4A_API p4a_status_t p4a_config_get_n_components     (const p4a_config_t*, int32_t*);
P4A_API p4a_status_t p4a_config_get_center_x         (const p4a_config_t*, int32_t*);
P4A_API p4a_status_t p4a_config_get_scale_x          (const p4a_config_t*, int32_t*);
P4A_API p4a_status_t p4a_config_get_center_y         (const p4a_config_t*, int32_t*);
P4A_API p4a_status_t p4a_config_get_scale_y          (const p4a_config_t*, int32_t*);
P4A_API p4a_status_t p4a_config_get_tol              (const p4a_config_t*, double*);
P4A_API p4a_status_t p4a_config_get_max_iter         (const p4a_config_t*, int32_t*);
P4A_API p4a_status_t p4a_config_get_store_scores     (const p4a_config_t*, int32_t*);
P4A_API p4a_status_t p4a_config_get_store_diagnostics(const p4a_config_t*, int32_t*);
P4A_API p4a_status_t p4a_config_get_dtype            (const p4a_config_t*, p4a_dtype_t*);
/* Pipeline / operator-bank / gating-strategy getters return the pointer set by
 * the corresponding setter, or NULL if none was set. */
P4A_API p4a_status_t p4a_config_get_pipeline         (const p4a_config_t*,
                                                       const p4a_pipeline_t** out);
P4A_API p4a_status_t p4a_config_get_operator_bank    (const p4a_config_t*,
                                                       const p4a_operator_bank_t** out);
P4A_API p4a_status_t p4a_config_get_gating_strategy  (const p4a_config_t*,
                                                       const p4a_gating_strategy_t** out);
```

### 4.6 Operator bank · gating strategy · pipeline (Phase 0 stubs, Phase 3/6 user)

```c
/* --- Operator bank (Phase 3 fills with SNV, MSC, SG-deriv-1/2, …) --- */
typedef enum p4a_operator_kind_t {
    P4A_OP_IDENTITY            = 0,
    P4A_OP_CENTER              = 1,
    P4A_OP_AUTOSCALE           = 2,
    P4A_OP_PARETO_SCALE        = 3,
    P4A_OP_SNV                 = 4,
    P4A_OP_MSC                 = 5,
    P4A_OP_EMSC                = 6,
    P4A_OP_DETREND_POLY        = 7,
    P4A_OP_SAVGOL_SMOOTH       = 8,
    P4A_OP_SAVGOL_DERIVATIVE   = 9,
    P4A_OP_NORRIS_WILLIAMS     = 10,
    P4A_OP_ASLS_BASELINE       = 11,
    P4A_OP_OSC                 = 12,
    P4A_OP_EPO                 = 13,
    P4A_OP_WAVELET_DENOISE     = 14
} p4a_operator_kind_t;

P4A_API p4a_status_t p4a_operator_bank_create (p4a_operator_bank_t** out);
P4A_API void         p4a_operator_bank_destroy(p4a_operator_bank_t* bank);
P4A_API p4a_status_t p4a_operator_bank_add    (p4a_operator_bank_t* bank,
                                                p4a_operator_kind_t kind,
                                                const double* params, int32_t n_params);
P4A_API p4a_status_t p4a_operator_bank_size   (const p4a_operator_bank_t* bank,
                                                int32_t* out_size);

/* --- Gating strategy (Phase 6) --- */
typedef enum p4a_gating_mode_t {
    P4A_GATING_HARD            = 0,    /* discrete operator pick */
    P4A_GATING_SOFT            = 1,    /* weighted mixture */
    P4A_GATING_SPARSE          = 2,    /* penalty-driven sparse mixture */
    P4A_GATING_PER_COMPONENT   = 3,    /* different operator per component (POP-PLS) */
    P4A_GATING_PER_BLOCK       = 4,
    P4A_GATING_PER_TARGET      = 5
} p4a_gating_mode_t;

P4A_API p4a_status_t p4a_gating_strategy_create (p4a_gating_strategy_t** out,
                                                  p4a_gating_mode_t mode);
P4A_API void         p4a_gating_strategy_destroy(p4a_gating_strategy_t* gs);

/* --- Preprocessing pipeline (Phase 3) ---
 * Pipeline = ordered list of operators that share `fit` / `transform`.
 * Critical for leakage-safe CV: fit on train fold, transform validation fold.
 */
P4A_API p4a_status_t p4a_pipeline_create        (p4a_pipeline_t** out);
P4A_API void         p4a_pipeline_destroy       (p4a_pipeline_t* pipe);
P4A_API p4a_status_t p4a_pipeline_clone         (const p4a_pipeline_t* src,
                                                  p4a_pipeline_t** out_dst);
P4A_API p4a_status_t p4a_pipeline_add_operator  (p4a_pipeline_t* pipe,
                                                  p4a_operator_kind_t kind,
                                                  const double* params, int32_t n_params);
P4A_API p4a_status_t p4a_pipeline_size          (const p4a_pipeline_t* pipe,
                                                  int32_t* out_size);
P4A_API p4a_status_t p4a_pipeline_fit           (p4a_context_t* ctx,
                                                  p4a_pipeline_t* pipe,
                                                  const p4a_matrix_view_t* X,
                                                  const p4a_matrix_view_t* Y);
P4A_API p4a_status_t p4a_pipeline_transform     (p4a_context_t* ctx,
                                                  const p4a_pipeline_t* pipe,
                                                  const p4a_matrix_view_t* X,
                                                  p4a_matrix_view_t* out);
P4A_API p4a_status_t p4a_pipeline_transform_alloc(p4a_context_t* ctx,
                                                  const p4a_pipeline_t* pipe,
                                                  const p4a_matrix_view_t* X,
                                                  p4a_array_t** out);
```

All Phase 0 implementations: `_create/_destroy/_clone/_add/_size` are real; `_fit/_transform/_transform_alloc` return `P4A_ERR_NOT_IMPLEMENTED`.

### 4.7 Model lifecycle (Phase 0 stubs; Phase 1 implements)

```c
P4A_API p4a_status_t p4a_model_fit(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    p4a_model_t** out_model);   /* Phase 0: P4A_ERR_NOT_IMPLEMENTED */

P4A_API void         p4a_model_destroy(p4a_model_t* model);

/* Caller-provided buffer variants (high-performance path). */
P4A_API p4a_status_t p4a_model_predict(
    p4a_context_t* ctx, const p4a_model_t* model,
    const p4a_matrix_view_t* X, p4a_matrix_view_t* out);

P4A_API p4a_status_t p4a_model_transform(
    p4a_context_t* ctx, const p4a_model_t* model,
    const p4a_matrix_view_t* X, p4a_matrix_view_t* out_scores);

/* Core-allocated variants (convenience path). */
P4A_API p4a_status_t p4a_model_predict_alloc(
    p4a_context_t* ctx, const p4a_model_t* model,
    const p4a_matrix_view_t* X, p4a_array_t** out);

P4A_API p4a_status_t p4a_model_transform_alloc(
    p4a_context_t* ctx, const p4a_model_t* model,
    const p4a_matrix_view_t* X, p4a_array_t** out_scores);

/* Tagged accessor for fitted-model arrays. */
typedef enum p4a_model_array_t {
    P4A_MODEL_COEFFICIENTS = 0,    /* (n_features, n_targets) */
    P4A_MODEL_INTERCEPT    = 1,    /* (n_targets,) */
    P4A_MODEL_X_MEAN       = 2,
    P4A_MODEL_X_SCALE      = 3,
    P4A_MODEL_Y_MEAN       = 4,
    P4A_MODEL_Y_SCALE      = 5,
    P4A_MODEL_WEIGHTS_W    = 6,    /* (n_features, n_components) */
    P4A_MODEL_LOADINGS_P   = 7,
    P4A_MODEL_Y_LOADINGS_Q = 8,
    P4A_MODEL_ROTATIONS_R  = 9,
    P4A_MODEL_SCORES_T     = 10,   /* optional, store_scores=1 */
    P4A_MODEL_Y_SCORES_U   = 11    /* optional */
} p4a_model_array_t;

P4A_API p4a_status_t p4a_model_get_array(
    p4a_context_t* ctx, const p4a_model_t* model,
    p4a_model_array_t which, p4a_array_t** out);

P4A_API p4a_status_t p4a_model_get_n_components(const p4a_model_t* m, int32_t* out);
P4A_API p4a_status_t p4a_model_get_n_features  (const p4a_model_t* m, int32_t* out);
P4A_API p4a_status_t p4a_model_get_n_targets   (const p4a_model_t* m, int32_t* out);
```

### 4.8 Serialization (Phase 0 declares, Phase 1 implements)

```c
#define P4A_SERIALIZATION_MAGIC          "P4AM"   /* 4 bytes — file/buffer magic */
#define P4A_SERIALIZATION_FORMAT_VERSION 1u       /* bumps on incompatible format change */
/* The serialized buffer also embeds the producing library's ABI version
 * (major.minor.patch) and project version string. The consumer rejects on
 * P4A_SERIALIZATION_FORMAT_VERSION mismatch (status P4A_ERR_VERSION_INCOMPATIBLE)
 * and warns via last_error if the ABI version differs but format is compatible.
 * All multi-byte integers are little-endian; all doubles are IEEE 754 binary64 LE. */

P4A_API p4a_status_t p4a_model_export_size(
    const p4a_model_t* model, size_t* out_size);

P4A_API p4a_status_t p4a_model_export_to_buffer(
    const p4a_model_t* model, void* buffer, size_t buffer_size,
    size_t* out_written);

P4A_API p4a_status_t p4a_model_import_from_buffer(
    p4a_context_t* ctx, const void* buffer, size_t buffer_size,
    p4a_model_t** out_model);

/* Inspect a buffer's header without fully importing it. Useful for compatibility
 * checks before allocating model memory. */
P4A_API p4a_status_t p4a_serialization_inspect(
    const void* buffer, size_t buffer_size,
    uint32_t* out_format_version,
    uint32_t* out_writer_abi_major,
    uint32_t* out_writer_abi_minor,
    uint32_t* out_writer_abi_patch);
```

### 4.9 ABI version + build info

```c
P4A_API uint32_t     p4a_get_abi_version_major(void);
P4A_API uint32_t     p4a_get_abi_version_minor(void);
P4A_API uint32_t     p4a_get_abi_version_patch(void);
P4A_API uint32_t     p4a_get_abi_version_int(void);   /* MAJOR*10000 + MINOR*100 + PATCH */
P4A_API const char*  p4a_get_version_string(void);    /* "0.1.0+abi.1.0.0" */
P4A_API const char*  p4a_get_build_info(void);
P4A_API const char*  p4a_get_git_revision(void);
P4A_API p4a_status_t p4a_check_abi_compatibility(uint32_t hdr_major, uint32_t hdr_minor);
```

### 4.10 Output array helper

```c
P4A_API p4a_status_t p4a_array_dtype(const p4a_array_t* arr, p4a_dtype_t* out);
P4A_API p4a_status_t p4a_array_shape(const p4a_array_t* arr,
                                      int64_t* rows, int64_t* cols);
P4A_API p4a_status_t p4a_array_view (const p4a_array_t* arr, p4a_matrix_view_t* out);
P4A_API void         p4a_array_free (p4a_array_t* arr);
```

### 4.11 Error-message ownership rule (canonical)

- `p4a_context_t` owns a thread-safe, per-context, fixed-capacity **4096-byte** UTF-8 buffer (`P4A_ERROR_BUFFER_BYTES = 4096`, immutable, never bumped — this is an ABI commitment).
- Every `p4a_*` call that fails and is given a non-NULL context writes a message into that buffer (truncating safely).
- `p4a_context_last_error(ctx)` returns a pointer into the buffer; it is invalidated by the next call on the same context — bindings must copy if they retain it.
- No `p4a_free_string`, no `char** out_msg`. Period.

### 4.12 Compile-time version header (`p4a_version.h`)

Unchanged from rev 1.

## 5. CI workflows

### 5.1 `ci.yml` — main build matrix · **10 cells** + 1 universal2

The cells are explicitly chosen to maximise coverage / cost. Combinations not listed are deliberately omitted (rationale in comments).

| # | Label | OS | C/C++ | Build type | Notes |
|---|---|---|---|---|---|
| 1 | linux-gcc12-release   | ubuntu-22.04 | gcc-12   | Release | LTS baseline |
| 2 | linux-gcc12-debug     | ubuntu-22.04 | gcc-12   | Debug   | Debug coverage on the LTS toolchain |
| 3 | linux-gcc13-release   | ubuntu-24.04 | gcc-13   | Release | Newer libstdc++; Debug omitted (gcc-12-debug covers debug paths) |
| 4 | linux-clang16-release | ubuntu-22.04 | clang-16 | Release | UB catches libstdc++ ↔ libc++ surprises in Release LTO |
| 5 | linux-clang16-debug   | ubuntu-22.04 | clang-16 | Debug   | |
| 6 | macos-clang-release   | macos-14 (arm64) | Apple clang | Release | |
| 7 | macos-clang-debug     | macos-14 (arm64) | Apple clang | Debug   | |
| 8 | windows-msvc-release  | windows-2022 | MSVC 17  | Release | |
| 9 | windows-msvc-debug    | windows-2022 | MSVC 17  | Debug   | |
| 10| windows-mingw-release | windows-2022 | MinGW UCRT64 | Release | MinGW-debug omitted (MSYS2 debug builds are slow; LD-level checks are covered by MSVC-debug) |
| 11| macos-universal2      | macos-14 | Apple clang | Release | `CMAKE_OSX_ARCHITECTURES="x86_64;arm64"` |

Per-cell steps: `cmake --preset ci-<label>` → `cmake --build --preset ci-<label>` → `ctest --preset ci-<label> --output-on-failure`. Release jobs upload `libp4a*` + symbol-export snapshots + `--build-info` output as artefacts. `-DPLS4ALL_WARNINGS_AS_ERRORS=ON` on every cell.

### 5.2 `sanitizers.yml`

Linux × clang-16, three jobs: ASAN, UBSAN, **TSAN** (added). Each runs the full doctest binary plus the concurrency stress tests in `cpp/tests/concurrency/`.

### 5.3 `abi-check.yml`

Steps per OS:

1. Build `pls4all_c` in Release.
2. Extract exported symbols:
   - Linux: `nm -D --defined-only libp4a.so | awk '{print $3}' | sort -u`
   - macOS: `nm -gU libp4a.dylib | awk '{print $3}' | sed 's/^_//' | sort -u`
   - Windows: `dumpbin /EXPORTS p4a.dll` parsed via PowerShell.
3. Assert every symbol matches `^(p4a_|_init$|_fini$|_edata$|_end$|__bss_start$)`.
4. **Runtime dependency audit** (forbidden list — applied identically on all three OSes; case-insensitive on Windows):
   - `libopenblas` (and `openblas` on Windows), `libmkl`, `libpython`, `libR` (R runtime), `libRcpp`, `libboost`, `libeigen` (Eigen if it ever forms a runtime dep — it should not), `libpybind11`, `libembind` (Emscripten host runtime, not allowed in CPython wheels), `libnlohmann_json`, `nlohmann-json`, `libyaml-cpp`, `yaml-cpp`.
   - Tool per OS: Linux `ldd libp4a.so`, macOS `otool -L libp4a.dylib`, Windows `dumpbin /DEPENDENTS p4a.dll` (PowerShell). Any match fails the job. The same list is also referenced from §1 acceptance criterion 3 to keep them in lockstep.
5. Diff exported-symbol list against `cpp/abi/expected_symbols_{linux,macos,windows}.txt`. Any drift fails.

### 5.4 `coverage.yml`

gcc-12 `--coverage` + codecov upload. Informational at Phase 0 (no merge gate).

### 5.5 `parity-gate.yml`

- Set up Python (locked from `parity/python_generator/requirements-lock.txt`).
- Set up R (locked from `parity/r_generator/renv.lock`).
- Run `python parity/python_generator/generate_fixtures.py --suite synthetic --check`: re-derives the three canonical fixtures, asserts SHA-256 in `manifest.json` matches.
- Run `Rscript parity/r_generator/generate_fixtures.R --check`: same for R-side companion fixtures.
- Build `pls4all_c` + tests; run `ctest -L fixture`.
- At Phase 0 the loader test asserts `P4A_ERR_NOT_IMPLEMENTED` (red by design on comparison, green on workflow).

### 5.6 `docs.yml`

mkdocs build on push to `main`; deploy to gh-pages on tag.

### 5.7 `release.yml`

Placeholder; wires up Phase 2.

### 5.8 Branch protection (configured manually after `phase-0` tag)

Required checks for merge to `main`:
- All `ci.yml` cells (10 + universal2).
- `sanitizers.yml` (ASAN, UBSAN, TSAN).
- `abi-check.yml` (Linux + macOS + Windows).
- `parity-gate.yml` — added to the required set after PR 17 (`parity-gate-workflow`) lands.

## 6. Parity-gate scaffolding

### 6.1 Schema (`parity/schema/fixture_schema_v1.json`)

Unchanged in shape from rev 1, with one clarification: `comparison_policy.tolerance_table_row` indexes into `parity/tolerances.md`; `sign_invariant: true` triggers the `max_abs_element_positive` resolver in the comparator (Phase 1+).

### 6.2 Python generator

`parity/python_generator/`:

- `pyproject.toml` declares the project + console-script `generate-fixtures`.
- `requirements-lock.txt` pins exact patch versions: `scikit-learn==1.4.2`, `numpy==1.26.4`, `nirs4all==0.8.5`, `scipy==1.11.4`, `joblib==1.4.0`, plus their full transitive closure (committed lockfile, not regenerated in CI).
- Adapters: `sklearn_adapter.py`, `nirs4all_adapter.py` (the 21 variants — most `pytest.skip("variant deferred to Phase N")` at Phase 0; symbols are imported to verify the import graph).
- Suites: `synthetic.py` (seeded RNG), `nirs_real.py` (BEER / CORN / ALPINE loaders wired up but skipped at Phase 0 — file paths are externalised via env vars).
- CLI: `python generate_fixtures.py --suite synthetic --variant pls1|pls2|tiny_centered --out parity/fixtures/`.
- Determinism tests: `tests/test_generator_determinism.py` regenerates, hashes, diffs against `manifest.json`.

Three canonical fixtures shipped at Phase 0 by both generators:
- `synthetic_small_pls1_v1`: seeded 50×20 X, 50×1 Y, sklearn `PLSRegression(n_components=3)`.
- `synthetic_small_pls2_v1`: seeded 50×20 X, 50×4 Y, sklearn `PLSRegression(n_components=3)`.
- `synthetic_tiny_centered_v1`: hand-crafted 10×3 numerical edge case.

### 6.3 R generator

`parity/r_generator/`:

- `DESCRIPTION` (bare script project, not a CRAN package).
- `renv.lock` pins exact versions of `pls`, `ropls`, `mixOmics`, `plsVarSel`, plus their transitive closure.
- `generate_fixtures.R --suite synthetic --variant <…> --out parity/fixtures/` produces the **same three canonical fixtures** as Python, using `pls::plsr(method='kernelpls'|'simpls')` and (for the canonical case) `mixOmics::pls`.
- Adapters: `pls_adapter.R`, `ropls_adapter.R`, `mixOmics_adapter.R`. `plsVarSel_adapter.R` is wired up but skipped at Phase 0 (used in Phase 5).
- Determinism test: `tests/test_generator_determinism.R`.

### 6.4 Tolerance table (`parity/tolerances.md`)

Cross-algorithm comparisons gate on **predictions, coefficients, intercept, x/y means and scales only**. Raw latent matrices (W, P, Q, R, T, U) are *diagnostic* unless both sides use the same algorithm + solver + deflation; sign-invariance is applied via `max_abs_element_positive` resolver.

| A | B | Variant | Algorithm | abs_tol | rel_tol | Comparison set | Notes |
|---|---|---------|-----------|---------|---------|----------------|-------|
| sklearn 1.4 PLSRegression | sklearn (round-trip) | regression | NIPALS | 1e-15 | 1e-15 | all | sanity |
| sklearn 1.4 PLSRegression | R pls kernelpls | regression | NIPALS vs kernel-algo | 5e-7 | 1e-6 | predictions, coefficients, intercept, x_mean, x_scale, y_mean, y_scale | cross-algorithm — raw latent arrays excluded |
| sklearn 1.4 PLSRegression | R pls simpls | regression | NIPALS vs SIMPLS | 1e-5 | 1e-4 | predictions, coefficients, intercept, x_mean, x_scale, y_mean, y_scale | cross-algorithm — raw latent arrays excluded |
| sklearn 1.4 PLSRegression | mixOmics 6.28 pls | regression | NIPALS | **1e-7** | **1e-7** | predictions, coefficients, intercept, x_mean, x_scale, y_mean, y_scale | cross-impl — mixOmics scaling convention differs; latent arrays excluded |
| sklearn 1.4 PLSRegression | nirs4all SIMPLS | regression | SIMPLS | 1e-6 | 1e-5 | predictions, coefficients, intercept, x_mean, x_scale, y_mean, y_scale | cross-algorithm |
| sklearn 1.4 PLSCanonical | mixOmics pls(mode='canonical') | canonical | NIPALS | 1e-7 | 1e-7 | predictions, coefficients, intercept, x_mean, x_scale, y_mean, y_scale | cross-impl |
| ropls 1.34 opls | nirs4all OPLS | orthogonal | OPLS-NIPALS | 1e-5 | 1e-4 | predictions only | cross-impl — orthogonal-component count + sign + ordering all vary |
| nirs4all AOM_PLS | future pls4all AOM-PLS | AOM | SIMPLS+bank | **TBD** | **TBD** | predictions | freezes when pls4all AOM-PLS reference is frozen (Phase 6) |
| pls4all reference_cpu | pls4all BLAS backend | any | any | **1e-10** | **1e-9** | all | same algorithm, different SGEMM path; fixed reduction order + no FMA |
| pls4all reference_cpu | pls4all CUDA backend | any | any | 1e-9 | 1e-8 | predictions | fp64 GPU drift envelope |
| pls4all reference_cpu (Linux x86_64) | pls4all reference_cpu (macOS arm64) | any | any | **1e-11** | **1e-11** | all | x86_64 vs arm64 floating-point drift |

Numbers in **bold** are the relaxations Codex recommended over the initial draft.

## 7. Phase 0 tests

### 7.1 ABI lifecycle / config / status / version / matrix-view / errors

| File | Coverage |
|------|----------|
| `cpp/tests/abi/test_lifecycle.cpp` | create / destroy happy + NULL no-op + 10k churn under ASAN. **Double-destroy of a non-NULL handle is documented as UB and not tested.** |
| `cpp/tests/abi/test_config_setters.cpp` | every setter: defaults, accept/reject, idempotent rejection, clone-deep-copy |
| `cpp/tests/abi/test_status_codes.cpp` | `_to_string` non-empty for every status; pinned strings |
| `cpp/tests/abi/test_version.cpp` | runtime vs compile-time consistency; `_check_abi_compatibility` truth table |
| `cpp/tests/abi/test_matrix_view.cpp` | rowmajor / colmajor / strided init; `_validate` reject negative strides, accept transposed and zero-sized; sub-stride slice views |
| `cpp/tests/abi/test_error_messages.cpp` | last_error non-empty after failure; overwrite-not-append; clear; 4097-byte truncation; per-context isolation across threads |
| `cpp/tests/abi/test_exception_safety.cpp` | inject `bad_alloc / runtime_error / non-std-exception` at C boundary (via `PLS4ALL_ENABLE_FAULT_INJECTION`) → correct status + message. **Every `P4A_API` is verified `noexcept` via a static assert table.** |
| `cpp/tests/abi/test_oom_paths.cpp` | budget-limited allocator wrapper; create/destroy paths leak-clean under ASAN |

### 7.2 Concurrency / fork (POSIX)

| File | Coverage |
|------|----------|
| `cpp/tests/concurrency/test_multi_context_threads.cpp` | two `p4a_context_t*` on two threads, each running its own failing-call sequence → independent `last_error` buffers, no cross-talk; TSAN-clean |
| `cpp/tests/concurrency/test_fork_smoke.cpp` | POSIX `fork()` smoke: parent context survives, child can create a fresh context. Documents that **no** `p4a_*` function is async-signal-safe — including the read-only ones — and that the safe pattern is "drain the error string before forking and let the child create its own context". |

### 7.3 Fuzzing (libFuzzer; PLS4ALL_BUILD_FUZZ=ON)

| File | Target |
|------|--------|
| `cpp/tests/fuzz/fuzz_config_setters.cpp` | random byte stream → choice of setter + payload; expect either OK or known-status, never crash |
| `cpp/tests/fuzz/fuzz_matrix_view_validate.cpp` | random `p4a_matrix_view_t` fields → `_validate` must terminate, never crash |
| `cpp/tests/fuzz/fuzz_error_buffer_format.cpp` | random message lengths → 4096-byte buffer never overruns, NUL-terminated |
| `cpp/tests/fuzz/fuzz_fixture_loader.cpp` | random JSON-like bytes → loader rejects safely |

A future `fuzz_serialization_import.cpp` is reserved for Phase 1.

### 7.4 Fixture loader

| File | Coverage |
|------|----------|
| `cpp/tests/fixture/test_fixture_loader.cpp` | loads `synthetic_small_pls1_v1.json` via vendored `nlohmann/json`, validates schema + SHA-256 (checked against `manifest.json`), asserts `p4a_model_fit` returns `P4A_ERR_NOT_IMPLEMENTED` |

### 7.5 NaN / Inf / async-signal / enum-sentinel policy

Documented in `docs/architecture/error_model.md` and `docs/architecture/threading.md`:

- **NaN/Inf inputs to scalar `set_*` config setters** (`set_tol`): rejected with `P4A_ERR_INVALID_ARGUMENT`. Tests in `test_config_setters.cpp` enforce this; the matrix-view inits take integers (rows / cols / strides / dtype tag) and have no floating-point inputs.
- **NaN/Inf inside `p4a_matrix_view_t.data`**: not inspected at `validate` time (would require scanning O(n×m) doubles for every call). Phase 1 algorithms decide per-algorithm whether to reject (`P4A_ERR_INVALID_ARGUMENT`) or propagate. The default policy will be "reject in NIPALS/SIMPLS, propagate in transforms".
- **Async-signal-safety**: **no** `p4a_*` function is documented async-signal-safe — including the inert helpers (`p4a_status_to_string`, `p4a_dtype_size`, `p4a_get_version_string`). Do not call any `p4a_*` from a POSIX signal handler. This is single source of truth; `test_fork_smoke.cpp` carries the same statement in its file header so contributors run into it during reviews.
- **Enum sentinels**: There are **no** `_COUNT` sentinels in the public C header. Compile-time bounds-checking lives in internal C++ (`core/status.cpp` etc.) via `static_assert` against the highest declared enum value. Adding a new enum value is therefore an ABI-additive change and never breaks consumers compiled against an older header — the public header's enum tail is implicitly open.

## 8. Documentation skeleton

- `README.md` — already shipped at Step 0.
- `docs/architecture/overview.md` — ABI rationale.
- `docs/architecture/memory_model.md` — who allocates, who frees; `p4a_array_free` ownership.
- `docs/architecture/error_model.md` — status table, error buffer, NaN/Inf policy.
- `docs/architecture/threading.md` — single-context-per-thread default; multi-context concurrency; async-signal policy.
- `docs/architecture/serialization.md` — binary format v1 spec (Phase 1 implements; surface at Phase 0).
- `docs/abi/reference.md` — Doxygen-extracted from `p4a.h`.
- `docs/abi/stability_policy.md` — semver of ABI; `_COUNT` non-ABI note.
- `docs/abi/changes_log.md` — per-release ABI deltas.
- `docs/bindings/{python,r,matlab,js,android}.md` — install + hello-version.
- `docs/parity/methodology.md` — multi-reference cross-check policy.
- `docs/parity/tolerances.md` — symlink/include of `parity/tolerances.md`.
- `docs/dev/workflow.md` — roadmap → Codex review → implement → Codex review → tag.
- `docs/dev/build.md`, `testing.md` (incl. fuzzing + TSAN + doctest pin SHA-256), `style.md`, `release_process.md`.

## 9. Order of operations — 18 atomic PRs

Each PR is small, keeps CI green, gets its own Codex review and lands on `main` only when all required checks pass.

1. **`bootstrap`** ✅ — already landed (commit `6545895`).
2. **`cmake-skeleton`** — top-level `CMakeLists.txt`, `Pls4allOptions.cmake`, `CompilerWarnings.cmake`, `SanitizersToolchain.cmake`, `CMakePresets.json`. Empty `pls4all_c` shared lib compiles. Dead-code / undefined-symbol link flags configured.
3. **`p4a-header`** — `p4a.h` (every Phase 0 declaration, all rev-2 additions), `p4a_version.h`, `p4a_export.h.in`.
4. **`status-version-context-config`** — string tables; version functions; `p4a_context_*` and `p4a_config_*` implementations with full input validation; corresponding `extern "C"` wrappers with the `noexcept` translation pattern.
5. **`matrix-view-error-buffer`** — `p4a_matrix_view_init_{rowmajor,colmajor,strided}` + `_validate`; per-context 4096-byte thread-safe error buffer; status → message integration.
6. **`operator-bank-pipeline-gating`** — `p4a_operator_bank_*` (real lifecycle, no algorithm bodies), `p4a_gating_strategy_*` (real lifecycle), `p4a_pipeline_*` lifecycle + `add_operator` (real), `_fit`/`_transform` stubbed to `P4A_ERR_NOT_IMPLEMENTED`.
7. **`abi-stubs`** — `p4a_model_*` stubs (fit/predict/transform/get_array/serialization), `pls4all_cli`, first snapshot of `expected_symbols_{linux,macos,windows}.txt`.
8. **`tests-doctest-abi`** — vendor `doctest.h` (v2.4.11 + SHA-256 in `cpp/tests/doctest/README.md`); lifecycle, config_setters, status, version, matrix_view tests.
9. **`tests-advanced`** — error_messages, exception_safety, oom_paths, fixture_loader (with vendored `nlohmann/json.hpp`).
10. **`tests-concurrency-fuzz`** — multi-context thread test, fork smoke (POSIX), 4 libFuzzer harnesses.
11. **`ci-linux`** — `ci.yml` Linux cells (1–5); `abi-check.yml` Linux only (incl. `ldd` audit).
12. **`ci-mac-windows`** — extend `ci.yml` (cells 6–10, plus universal2 cell 11) and `abi-check.yml` (macOS + Windows incl. `otool -L`, `dumpbin /DEPENDENTS`).
13. **`ci-sanitizers-coverage`** — `sanitizers.yml` (ASAN/UBSAN/TSAN), `coverage.yml`, clang-tidy enforcement step.
14. **`parity-schema-and-fixtures`** — `fixture_schema_v1.{json,md}`, three synthetic fixtures, `manifest.json`, revised `tolerances.md`.
15. **`parity-python-generator`** — pinned `pyproject.toml` + `requirements-lock.txt`, suites, sklearn + nirs4all adapters, determinism tests.
16. **`parity-r-generator`** — `renv.lock`, `pls` + `ropls` + `mixOmics` adapters, R determinism tests, all 3 canonical fixtures regenerable on R side.
17. **`parity-gate-workflow`** — `parity-gate.yml`; bindings skeletons (Python ctypes, R `.Call`, MATLAB MEX, JS Emscripten, Android JNI); docs skeleton; `docs.yml`.
18. **`release-tag`** — bump CHANGELOG to `0.1.0-phase0`, tag `phase-0`, final Codex review, push. Configure branch protection.

## 10. Technical risks

1. **`p4a_matrix_view_t` struct layout on ILP32**: The view has an `int64_t row_stride` + `int64_t col_stride` + `int32_t reserved0`. On ILP32 platforms (notably Android `armeabi-v7a`) the trailing `int32_t` pads differently. **Decision**: hard-exclude `armeabi-v7a` from Phase 0 and Phase 2. Android targets are `arm64-v8a` + `x86_64` (the emulator) only. Documented in §11. Revisit only if user demand surfaces.
2. **Tolerances frozen too early**: §6.4 numbers are best-current-estimates; Phase 1 will measure them and may need to relax some further. The table is versioned and lives in git.
3. **Fuzzing requires clang**: libFuzzer is clang-specific. The `PLS4ALL_BUILD_FUZZ` option fails fast if the compiler isn't clang.
4. **TSAN false positives in doctest**: doctest uses some lock-free counters internally; TSAN may complain. Mitigation: gate TSAN to the concurrency tests only, not the whole suite.
5. **Universal2 macOS build cross-link complications**: arm64 + x86_64 fat binary requires both arch slices to compile cleanly; arm64-only CI cells (6, 7) catch arm64 issues; universal2 (cell 11) catches cross-compilation issues.
6. **`nlohmann/json` is large**: ~25 k lines. Test-only — never linked into `libp4a`. Vendored at `cpp/tests/doctest/json.hpp` (path is historical; will move under `cpp/tests/vendored/` in a Phase 0 cleanup PR if needed).

## 11. Operations notes (non-technical)

These are project-operations decisions that affect Phase 0 but aren't architecture risks. They live here rather than in §10 to keep the technical risk register focused.

- **Repository visibility**: created `--private` at Step 0 (auto-mode classifier preferred caution). `gh repo edit GBeurier/pls4all --visibility public` can flip it at any time; Phase 0 implementation doesn't depend on it.
- **Codex MCP**: registered project-scoped via `.mcp.json`. Becomes callable as MCP tools after Claude reloads the session; until then, `codex exec` via Bash is the workflow.
- **Doctest vendoring**: pinned to upstream tag `v2.4.11` at <https://github.com/doctest/doctest>. SHA-256 of the single `doctest.h` recorded in `cpp/tests/doctest/README.md` before vendor commit. Bump procedure documented in `docs/dev/testing.md`.
- **`nlohmann/json` vendoring**: pinned to upstream tag `v3.11.3` (single-include release) at <https://github.com/nlohmann/json>. Same SHA-256 hygiene; same bump procedure.

## 12. Codex review checklist (for the next round)

When sending the implementation diffs (one per atomic PR §9.2–9.18) to Codex, explicitly ask:

- Any violation of `Direction_Technique.md` decisions.
- Any deviation from this revision of the roadmap that wasn't agreed.
- Any missing test case for the surface introduced in this PR.
- Any CI gap that would let a regression slip through.
- Any non-`p4a_*` symbol that escaped the export gate.
- Any `extern "C"` wrapper that is **not** `noexcept`.
- Any allocation / free that crosses the language boundary.
- Any STL or Eigen type leaking into the public header.
