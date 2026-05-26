# cpp/src/CMakeLists.txt — n4m_core (internal OBJECT) + n4m_c (C ABI
# shared) + n4m_c_static (C ABI static, for embedders).
#
# At PR 2 (cmake-skeleton) both targets compile from placeholder TUs only;
# PRs 3–7 replace the placeholders with the real version / status / context /
# config / matrix-view / error-buffer / pipeline / operator-bank / gating /
# model-stub implementations.

# ---------------------------------------------------------------------------
# Generated visibility macro header
# ---------------------------------------------------------------------------
configure_file(
    ${PROJECT_SOURCE_DIR}/cpp/include/n4m/n4m_export.h.in
    ${PROJECT_BINARY_DIR}/generated/n4m/n4m_export.h
    @ONLY)

# ---------------------------------------------------------------------------
# n4m_core — internal OBJECT library. Embedded inside n4m_c and
# n4m_c_static via $<TARGET_OBJECTS:n4m_core>. Never installed.
# Object libraries solve the "static lib that depends on an internal static
# lib" install-set problem by simply not being archive files in their own
# right.
# ---------------------------------------------------------------------------
add_library(n4m_core OBJECT
    core/aom_operators.cpp
    core/aom_preprocessing.cpp
    core/aom_selection.cpp
    core/bipls_selection.cpp
    core/bve_selection.cpp
    core/emcuve_selection.cpp
    core/sipls_selection.cpp
    core/t2_selection.cpp
    core/wvc_selection.cpp
    core/cars_selection.cpp
    core/classification_metrics.cpp
    core/component_coefficients.cpp
    core/ecr.cpp
    core/ga_selection.cpp
    core/gpr_pls.cpp
    core/ipw_selection.cpp
    core/irf_selection.cpp
    core/iriv_selection.cpp
    core/variable_importance.cpp
    core/variable_selection.cpp
    core/config.cpp
    core/cross_validation.cpp
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
    core/vip_spa_selection.cpp
    core/vissa_selection.cpp)
if(N4M_WITH_CUDA)
    target_sources(n4m_core PRIVATE core/cuda_dispatch.cpp)
endif()
target_include_directories(n4m_core
    PUBLIC  ${PROJECT_SOURCE_DIR}/cpp/include
            ${PROJECT_BINARY_DIR}/generated
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(n4m_core PUBLIC cxx_std_17)
target_compile_definitions(n4m_core PRIVATE N4M_BUILDING_CORE=1)
set_target_properties(n4m_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(N4M_ENABLE_FAULT_INJECTION)
    target_compile_definitions(n4m_core PUBLIC N4M_ENABLE_FAULT_INJECTION=1)
endif()
set(_n4m_sanitizer_build OFF)
if(N4M_ENABLE_ASAN OR N4M_ENABLE_UBSAN OR N4M_ENABLE_TSAN)
    set(_n4m_sanitizer_build ON)
endif()
# Dead-code stripping in non-Debug configs (multi-config-safe).
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(n4m_core PRIVATE
        $<$<NOT:$<CONFIG:Debug>>:-ffunction-sections>
        $<$<NOT:$<CONFIG:Debug>>:-fdata-sections>)
endif()
n4m_add_warnings(n4m_core)

# ---------------------------------------------------------------------------
# Optional BLAS backend (Phase 43). Default OFF — when ON we route the hot
# linear-algebra kernels in cpp/src/core/model.cpp through CBLAS via the
# n4m::linalg dispatch layer in cpp/src/core/linalg.hpp. The reference
# scalar loops stay as the !N4M_USE_BLAS path so the default build is
# bit-identical to pre-Phase-43.
# ---------------------------------------------------------------------------
if(N4M_WITH_BLAS)
    find_package(BLAS)
    if(NOT BLAS_FOUND)
        message(FATAL_ERROR
            "N4M_WITH_BLAS=ON but no BLAS implementation was found.\n"
            "Install OpenBLAS (libopenblas-dev), MKL, or Apple Accelerate, "
            "then reconfigure.\n"
            "Alternatively set N4M_WITH_BLAS=OFF to keep the reference "
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
    check_include_file_cxx(cblas.h N4M_HAS_CBLAS_HEADER)
    set(CMAKE_REQUIRED_INCLUDES ${CMAKE_REQUIRED_INCLUDES_BACKUP})
    if(NOT N4M_HAS_CBLAS_HEADER)
        message(FATAL_ERROR
            "N4M_WITH_BLAS=ON but <cblas.h> was not found in the "
            "compiler's include path. Install the CBLAS dev package "
            "(e.g. libopenblas-dev on Debian/Ubuntu, "
            "openblas-devel on RHEL/Fedora, or Apple Accelerate on macOS) "
            "or pass -DBLAS_INCLUDE_DIRS=/path/to/include.")
    endif()
    target_compile_definitions(n4m_core PRIVATE N4M_USE_BLAS=1)
    if(BLAS_LINKER_FLAGS)
        target_link_options(n4m_core PRIVATE ${BLAS_LINKER_FLAGS})
    endif()
    if(BLAS_INCLUDE_DIRS)
        target_include_directories(n4m_core PRIVATE ${BLAS_INCLUDE_DIRS})
    endif()
    # The actual link to ${BLAS_LIBRARIES} happens inside
    # n4m_apply_c_target() for n4m_c and n4m_c_static.
    # Linking it here on the OBJECT library is inert because PRIVATE
    # link deps do not propagate through $<TARGET_OBJECTS:>.
    message(STATUS "  BLAS backend      : ON  (${BLAS_LIBRARIES})")
else()
    message(STATUS "  BLAS backend      : OFF (reference scalar loops)")
endif()

# ---------------------------------------------------------------------------
# Optional OpenMP backend (Phase 44). Default OFF — when ON we annotate
# the hot inner loops in model.cpp with #pragma omp parallel for via the
# N4M_PARALLEL_FOR_STATIC macro in cpp/src/core/parallel.hpp. The OFF
# build is bit-identical to pre-Phase-44.
#
# Coexistence note: when both N4M_WITH_BLAS=ON and
# N4M_WITH_OPENMP=ON, OpenBLAS will spawn its own pthread pool
# inside cblas_* calls AND OpenMP will fork threads for the annotated
# outer loops. To avoid oversubscription, users should set
# OPENBLAS_NUM_THREADS=1 (or the MKL equivalent) before running.
# ---------------------------------------------------------------------------
if(N4M_WITH_OPENMP)
    # OpenMP::OpenMP_CXX is created by find_package(OpenMP) in the
    # top-level CMakeLists.txt so both cpp/src/ and cpp/tests/ see it.
    target_compile_definitions(n4m_core PRIVATE N4M_USE_OPENMP=1)
    target_link_libraries(n4m_core PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "  OpenMP backend    : ON  (${OpenMP_CXX_FLAGS})")
    if(N4M_WITH_BLAS)
        message(WARNING
            "N4M_WITH_BLAS=ON together with N4M_WITH_OPENMP=ON: "
            "OpenBLAS distributions built with USE_OPENMP=1 share the same "
            "libgomp runtime as our pragmas, which can deadlock or "
            "oversubscribe cores. Set OPENBLAS_NUM_THREADS=1 (or the MKL "
            "equivalent) in the consumer environment before running.")
    endif()
    if(N4M_ENABLE_TSAN)
        message(WARNING
            "N4M_WITH_OPENMP=ON together with N4M_ENABLE_TSAN=ON: "
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
# Runtime safety: a build with N4M_WITH_CUDA=ON still loads cleanly
# on a machine without a GPU. cuda_dispatch::cuda_runtime_available()
# returns false in that case, and n4m_backend_is_available + set_backend
# both report unavailability cleanly.
# ---------------------------------------------------------------------------
if(N4M_WITH_CUDA)
    target_compile_definitions(n4m_core PRIVATE N4M_USE_CUDA=1)
    target_include_directories(n4m_core PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    message(STATUS "  CUDA backend      : ON  (toolkit ${CUDAToolkit_VERSION})")
    if(N4M_WITH_BLAS)
        message(WARNING
            "N4M_WITH_CUDA=ON together with N4M_WITH_BLAS=ON: "
            "linalg dispatch prefers CUDA when both are present. The "
            "BLAS link is kept but the BLAS code paths in linalg.hpp "
            "are not compiled (guarded by #elif on N4M_USE_CUDA).")
    endif()
else()
    message(STATUS "  CUDA backend      : OFF (no GPU dispatch)")
endif()

# ---------------------------------------------------------------------------
# n4m_c — public C ABI shared library (libn4m.{so,dll,dylib}).
# n4m_c_static is the same surface, statically linked.
#
# Both targets are defined in n4m_targets.cmake from a single c_api/ source
# glob — there is no longer a separate libpls4all.so / libn4m.so split.
# ---------------------------------------------------------------------------
include(${CMAKE_CURRENT_SOURCE_DIR}/n4m_targets.cmake)

if(N4M_BUILD_STATIC)
    # Static counterpart — same surface, same sources, statically linked.
    # Intended for Android predict-only embedders that ship a single AAR.
    file(GLOB _N4M_C_API_STATIC_CPP
        CONFIGURE_DEPENDS
        "${CMAKE_SOURCE_DIR}/cpp/src/c_api/*.cpp"
    )
    add_library(n4m_c_static STATIC
        ${_N4M_C_API_STATIC_CPP}
        $<TARGET_OBJECTS:n4m_core>)
    target_include_directories(n4m_c_static
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/cpp/include>
            $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/generated>
            $<INSTALL_INTERFACE:include>
        PRIVATE
            ${CMAKE_SOURCE_DIR}/cpp/src
    )
    target_compile_features(n4m_c_static PUBLIC cxx_std_17)
    target_compile_definitions(n4m_c_static PRIVATE N4M_STATIC=1)
    set_target_properties(n4m_c_static PROPERTIES
        OUTPUT_NAME "n4m_static"
        POSITION_INDEPENDENT_CODE ON
    )
    if(N4M_WITH_OPENMP AND TARGET OpenMP::OpenMP_CXX)
        target_link_libraries(n4m_c_static PRIVATE OpenMP::OpenMP_CXX)
    endif()
    if(N4M_WITH_BLAS AND DEFINED BLAS_LIBRARIES)
        target_link_libraries(n4m_c_static PRIVATE ${BLAS_LIBRARIES})
    endif()
    n4m_add_warnings(n4m_c_static)
endif()



