# cpp/src/CMakeLists.txt — pls4all_core (internal OBJECT) + pls4all_c (C ABI
# shared) + pls4all_c_static (C ABI static, for embedders).
#
# At PR 2 (cmake-skeleton) both targets compile from placeholder TUs only;
# PRs 3–7 replace the placeholders with the real version / status / context /
# config / matrix-view / error-buffer / pipeline / operator-bank / gating /
# model-stub implementations.

# ---------------------------------------------------------------------------
# Generated visibility macro header
# ---------------------------------------------------------------------------
configure_file(
    ${PROJECT_SOURCE_DIR}/cpp/include/pls4all/p4a_export.h.in
    ${PROJECT_BINARY_DIR}/generated/pls4all/p4a_export.h
    @ONLY)

# ---------------------------------------------------------------------------
# pls4all_core — internal OBJECT library. Embedded inside pls4all_c and
# pls4all_c_static via $<TARGET_OBJECTS:pls4all_core>. Never installed.
# Object libraries solve the "static lib that depends on an internal static
# lib" install-set problem by simply not being archive files in their own
# right.
# ---------------------------------------------------------------------------
add_library(pls4all_core OBJECT
    core/aom_operators.cpp
    core/aom_preprocessing.cpp
    core/aom_selection.cpp
    core/bipls_selection.cpp
    core/bve_selection.cpp
    core/emcuve_selection.cpp
    core/sipls_selection.cpp
    core/t2_selection.cpp
    core/wvc_selection.cpp
    core/status.cpp
    core/version.cpp
    core/cars_selection.cpp
    core/classification_metrics.cpp
    core/component_coefficients.cpp
    core/context.cpp
    core/ga_selection.cpp
    core/gpr_pls.cpp
    core/ipw_selection.cpp
    core/variable_importance.cpp
    core/variable_selection.cpp
    core/config.cpp
    core/cross_validation.cpp
    core/matrix_view.cpp
    core/mb_pls.cpp
    core/metrics.cpp
    core/model.cpp
    core/model_selection.cpp
    core/operator_bank.cpp
    core/gating_strategy.cpp
    core/extra_pls.cpp
    core/kernel_pls.cpp
    core/multiblock_extensions.cpp
    core/pls_diagnostics.cpp
    core/pls_monitoring.cpp
    core/recursive_pls.cpp
    core/tensor_pls.cpp
    core/interval_selection.cpp
    core/lw_pls.cpp
    core/pls_lda.cpp
    core/pls_logistic.cpp
    core/pipeline.cpp
    core/pso_selection.cpp
    core/randomization_selection.cpp
    core/random_frog_selection.cpp
    core/rep_selection.cpp
    core/scars_selection.cpp
    core/shaving_selection.cpp
    core/spa_selection.cpp
    core/stability_selection.cpp
    core/st_selection.cpp
    core/uve_selection.cpp
    core/validation.cpp
    core/vissa_selection.cpp)
if(PLS4ALL_WITH_CUDA)
    target_sources(pls4all_core PRIVATE core/cuda_dispatch.cpp)
endif()
target_include_directories(pls4all_core
    PUBLIC  ${PROJECT_SOURCE_DIR}/cpp/include
            ${PROJECT_BINARY_DIR}/generated
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(pls4all_core PUBLIC cxx_std_17)
target_compile_definitions(pls4all_core PRIVATE PLS4ALL_BUILDING_CORE=1)
set_target_properties(pls4all_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(PLS4ALL_ENABLE_FAULT_INJECTION)
    target_compile_definitions(pls4all_core PUBLIC PLS4ALL_ENABLE_FAULT_INJECTION=1)
endif()
set(_pls4all_sanitizer_build OFF)
if(PLS4ALL_ENABLE_ASAN OR PLS4ALL_ENABLE_UBSAN OR PLS4ALL_ENABLE_TSAN)
    set(_pls4all_sanitizer_build ON)
endif()
# Dead-code stripping in non-Debug configs (multi-config-safe).
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(pls4all_core PRIVATE
        $<$<NOT:$<CONFIG:Debug>>:-ffunction-sections>
        $<$<NOT:$<CONFIG:Debug>>:-fdata-sections>)
endif()
pls4all_add_warnings(pls4all_core)

# ---------------------------------------------------------------------------
# Optional BLAS backend (Phase 43). Default OFF — when ON we route the hot
# linear-algebra kernels in cpp/src/core/model.cpp through CBLAS via the
# pls4all::linalg dispatch layer in cpp/src/core/linalg.hpp. The reference
# scalar loops stay as the !P4A_USE_BLAS path so the default build is
# bit-identical to pre-Phase-43.
# ---------------------------------------------------------------------------
if(PLS4ALL_WITH_BLAS)
    find_package(BLAS)
    if(NOT BLAS_FOUND)
        message(FATAL_ERROR
            "PLS4ALL_WITH_BLAS=ON but no BLAS implementation was found.\n"
            "Install OpenBLAS (libopenblas-dev), MKL, or Apple Accelerate, "
            "then reconfigure.\n"
            "Alternatively set PLS4ALL_WITH_BLAS=OFF to keep the reference "
            "scalar loops.")
    endif()
    # Verify CBLAS header reachability. find_package(BLAS) locates the
    # Fortran/C link libraries but does not always know where cblas.h
    # lives; on Apple it's <Accelerate/cblas.h>, on conda-forge / Debian
    # it's <cblas.h> under the BLAS include dir.
    include(CheckIncludeFileCXX)
    set(CMAKE_REQUIRED_INCLUDES_BACKUP ${CMAKE_REQUIRED_INCLUDES})
    if(BLAS_INCLUDE_DIRS)
        list(APPEND CMAKE_REQUIRED_INCLUDES ${BLAS_INCLUDE_DIRS})
    endif()
    check_include_file_cxx(cblas.h PLS4ALL_HAS_CBLAS_HEADER)
    set(CMAKE_REQUIRED_INCLUDES ${CMAKE_REQUIRED_INCLUDES_BACKUP})
    if(NOT PLS4ALL_HAS_CBLAS_HEADER)
        message(FATAL_ERROR
            "PLS4ALL_WITH_BLAS=ON but <cblas.h> was not found in the "
            "compiler's include path. Install the CBLAS dev package "
            "(e.g. libopenblas-dev on Debian/Ubuntu, "
            "openblas-devel on RHEL/Fedora, or Apple Accelerate on macOS) "
            "or pass -DBLAS_INCLUDE_DIRS=/path/to/include.")
    endif()
    target_compile_definitions(pls4all_core PRIVATE P4A_USE_BLAS=1)
    if(BLAS_LINKER_FLAGS)
        target_link_options(pls4all_core PRIVATE ${BLAS_LINKER_FLAGS})
    endif()
    if(BLAS_INCLUDE_DIRS)
        target_include_directories(pls4all_core PRIVATE ${BLAS_INCLUDE_DIRS})
    endif()
    # The actual link to ${BLAS_LIBRARIES} happens inside
    # pls4all_apply_c_target() for pls4all_c and pls4all_c_static.
    # Linking it here on the OBJECT library is inert because PRIVATE
    # link deps do not propagate through $<TARGET_OBJECTS:>.
    message(STATUS "  BLAS backend      : ON  (${BLAS_LIBRARIES})")
else()
    message(STATUS "  BLAS backend      : OFF (reference scalar loops)")
endif()

# ---------------------------------------------------------------------------
# Optional OpenMP backend (Phase 44). Default OFF — when ON we annotate
# the hot inner loops in model.cpp with #pragma omp parallel for via the
# P4A_PARALLEL_FOR_STATIC macro in cpp/src/core/parallel.hpp. The OFF
# build is bit-identical to pre-Phase-44.
#
# Coexistence note: when both PLS4ALL_WITH_BLAS=ON and
# PLS4ALL_WITH_OPENMP=ON, OpenBLAS will spawn its own pthread pool
# inside cblas_* calls AND OpenMP will fork threads for the annotated
# outer loops. To avoid oversubscription, users should set
# OPENBLAS_NUM_THREADS=1 (or the MKL equivalent) before running.
# ---------------------------------------------------------------------------
if(PLS4ALL_WITH_OPENMP)
    # OpenMP::OpenMP_CXX is created by find_package(OpenMP) in the
    # top-level CMakeLists.txt so both cpp/src/ and cpp/tests/ see it.
    target_compile_definitions(pls4all_core PRIVATE P4A_USE_OPENMP=1)
    target_link_libraries(pls4all_core PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "  OpenMP backend    : ON  (${OpenMP_CXX_FLAGS})")
    if(PLS4ALL_WITH_BLAS)
        message(WARNING
            "PLS4ALL_WITH_BLAS=ON together with PLS4ALL_WITH_OPENMP=ON: "
            "OpenBLAS distributions built with USE_OPENMP=1 share the same "
            "libgomp runtime as our pragmas, which can deadlock or "
            "oversubscribe cores. Set OPENBLAS_NUM_THREADS=1 (or the MKL "
            "equivalent) in the consumer environment before running.")
    endif()
    if(PLS4ALL_ENABLE_TSAN)
        message(WARNING
            "PLS4ALL_WITH_OPENMP=ON together with PLS4ALL_ENABLE_TSAN=ON: "
            "libgomp is not TSAN-instrumented and will produce false "
            "positives. Use a separate non-TSAN build for OpenMP testing.")
    endif()
else()
    message(STATUS "  OpenMP backend    : OFF (serial inner loops)")
endif()

# ---------------------------------------------------------------------------
# Optional CUDA backend (Phase 45). Default OFF — when ON we route the
# linalg dispatch through cuBLAS via cpp/src/core/cuda_dispatch.cpp.
# Phase 45 uses host-side cuBLAS only, no custom CUDA kernels, so we do
# not call enable_language(CUDA) — the .cpp file compiles with the host
# C++ compiler and the CUDA include path. CUDA::cudart and CUDA::cublas
# carry the link dependencies.
#
# Runtime safety: a build with PLS4ALL_WITH_CUDA=ON still loads cleanly
# on a machine without a GPU. cuda_dispatch::cuda_runtime_available()
# returns false in that case, and p4a_backend_is_available + set_backend
# both report unavailability cleanly.
# ---------------------------------------------------------------------------
if(PLS4ALL_WITH_CUDA)
    target_compile_definitions(pls4all_core PRIVATE P4A_USE_CUDA=1)
    target_include_directories(pls4all_core PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    message(STATUS "  CUDA backend      : ON  (toolkit ${CUDAToolkit_VERSION})")
    if(PLS4ALL_WITH_BLAS)
        message(WARNING
            "PLS4ALL_WITH_CUDA=ON together with PLS4ALL_WITH_BLAS=ON: "
            "linalg dispatch prefers CUDA when both are present. The "
            "BLAS link is kept but the BLAS code paths in linalg.hpp "
            "are not compiled (guarded by #elif on P4A_USE_CUDA).")
    endif()
else()
    message(STATUS "  CUDA backend      : OFF (no GPU dispatch)")
endif()

# ---------------------------------------------------------------------------
# pls4all_c — public C ABI shared library (libp4a.{so,dll,dylib}).
# pls4all_c_static is the same surface, statically linked.
# ---------------------------------------------------------------------------
set(_pls4all_c_sources
    c_api/c_api_version.cpp
    c_api/c_api_context.cpp
    c_api/c_api_config.cpp
    c_api/c_api_matrix.cpp
    c_api/c_api_operator_bank.cpp
    c_api/c_api_gating_strategy.cpp
    c_api/c_api_pipeline.cpp
    c_api/c_api_model.cpp
    c_api/c_api_validation_plan.cpp
    c_api/c_api_aom_selection.cpp
    c_api/c_api_method_result.cpp)

function(pls4all_apply_c_target target)
    # Embed all pls4all_core objects directly into the C-API target so that
    # neither pls4all_c nor pls4all_c_static needs pls4all_core at install /
    # link time on the consumer side.
    target_sources(${target} PRIVATE $<TARGET_OBJECTS:pls4all_core>)
    target_include_directories(${target}
        PUBLIC  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/cpp/include>
                $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/generated>
                $<INSTALL_INTERFACE:include>
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})  # for core/*.hpp internal headers
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(${target} PRIVATE
            $<$<NOT:$<CONFIG:Debug>>:-ffunction-sections>
            $<$<NOT:$<CONFIG:Debug>>:-fdata-sections>)
    endif()
    if(PLS4ALL_WITH_BLAS)
        # Mirrors the define applied to pls4all_core, so the C-API
        # translation units (e.g. c_api_version.cpp) see the same BLAS
        # availability flag.
        target_compile_definitions(${target} PRIVATE P4A_USE_BLAS=1)
        # OBJECT libraries do not propagate PRIVATE link deps through
        # $<TARGET_OBJECTS:>, so the consuming target must link BLAS too.
        target_link_libraries(${target} PRIVATE ${BLAS_LIBRARIES})
        if(BLAS_LINKER_FLAGS)
            target_link_options(${target} PRIVATE ${BLAS_LINKER_FLAGS})
        endif()
    endif()
    if(PLS4ALL_WITH_OPENMP)
        target_compile_definitions(${target} PRIVATE P4A_USE_OPENMP=1)
        # OBJECT libraries do not propagate PRIVATE link deps through
        # $<TARGET_OBJECTS:>, so the consuming target must link the
        # OpenMP runtime as well.
        target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
    endif()
    if(PLS4ALL_WITH_CUDA)
        target_compile_definitions(${target} PRIVATE P4A_USE_CUDA=1)
        target_include_directories(${target} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
        # OBJECT libraries do not propagate PRIVATE link deps through
        # $<TARGET_OBJECTS:>, so the consuming target must link cudart
        # and cuBLAS as well.
        target_link_libraries(${target} PRIVATE CUDA::cudart CUDA::cublas)
    endif()
    pls4all_add_warnings(${target})
endfunction()

if(PLS4ALL_BUILD_SHARED)
    add_library(pls4all_c SHARED ${_pls4all_c_sources})
    pls4all_apply_c_target(pls4all_c)
    set_target_properties(pls4all_c PROPERTIES
        OUTPUT_NAME p4a
        SOVERSION   ${PROJECT_VERSION_MAJOR}
        VERSION     ${PROJECT_VERSION})
    target_compile_definitions(pls4all_c PRIVATE PLS4ALL_BUILDING_DLL=1)

    # Undefined-symbol enforcement + dead-code stripping at link time.
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        if(NOT _pls4all_sanitizer_build)
            target_link_options(pls4all_c PRIVATE -Wl,--no-undefined)
        endif()
        target_link_options(pls4all_c PRIVATE
            -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/c_api/p4a_linux.map
            $<$<NOT:$<CONFIG:Debug>>:-Wl,--gc-sections>)
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        target_link_options(pls4all_c PRIVATE
            -Wl,-undefined,error
            $<$<NOT:$<CONFIG:Debug>>:-Wl,-dead_strip>)
    elseif(MSVC)
        target_link_options(pls4all_c PRIVATE
            $<$<NOT:$<CONFIG:Debug>>:/OPT:REF>
            $<$<NOT:$<CONFIG:Debug>>:/OPT:ICF>)
    endif()
    # MinGW and MSVC both honour the .def export gate once it exists (PR 7).
    if((MSVC OR MINGW) AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/c_api/pls4all.def")
        target_sources(pls4all_c PRIVATE c_api/pls4all.def)
    endif()
endif()

if(PLS4ALL_BUILD_STATIC)
    add_library(pls4all_c_static STATIC ${_pls4all_c_sources})
    pls4all_apply_c_target(pls4all_c_static)
    set_target_properties(pls4all_c_static PROPERTIES OUTPUT_NAME p4a_static)
    target_compile_definitions(pls4all_c_static PUBLIC PLS4ALL_STATIC=1)
endif()
