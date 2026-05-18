# cpp/src/CMakeLists.txt — chemometrics4all_core (internal OBJECT) +
# chemometrics4all_c (C ABI shared) + chemometrics4all_c_static (C ABI static).
#
# Phase 0 (bootstrap): only the structural plumbing compiles. Each subsequent
# phase appends its files to the lists below.

# ---------------------------------------------------------------------------
# Generated visibility macro header
# ---------------------------------------------------------------------------
configure_file(
    ${PROJECT_SOURCE_DIR}/cpp/include/chemometrics4all/c4a_export.h.in
    ${PROJECT_BINARY_DIR}/generated/chemometrics4all/c4a_export.h
    @ONLY)

# ---------------------------------------------------------------------------
# chemometrics4all_core — internal OBJECT library. Embedded inside
# chemometrics4all_c and chemometrics4all_c_static via $<TARGET_OBJECTS:...>.
# Never installed.
# ---------------------------------------------------------------------------
add_library(chemometrics4all_core OBJECT
    core/context.cpp
    core/matrix_view.cpp
    core/status.cpp
    core/version.cpp
    core/common/banded_solver.c
    core/common/fck_kernel.c
    core/common/linalg.c
    core/common/padding.c
    core/common/rng_pcg64.c
    core/common/svd.c
    core/common/ziggurat.c
    core/filters/composite.c
    core/filters/high_leverage.c
    core/filters/spectral_quality.c
    core/filters/y_outlier.c
    core/preprocessing/baselines/airpls.c
    core/preprocessing/baselines/arpls.c
    core/preprocessing/baselines/asls.c
    core/preprocessing/baselines/beads.c
    core/preprocessing/baselines/detrend.c
    core/preprocessing/baselines/iasls.c
    core/preprocessing/baselines/imodpoly.c
    core/preprocessing/baselines/modpoly.c
    core/preprocessing/baselines/rolling_ball.c
    core/preprocessing/baselines/snip.c
    core/preprocessing/scatter/snv.c
    core/preprocessing/scatter/local_snv.c
    core/preprocessing/scatter/robust_snv.c
    core/preprocessing/scatter/area_normalization.c
    core/preprocessing/scatter/msc.c
    core/preprocessing/scatter/emsc.c
    core/preprocessing/scaling/normalize.c
    core/preprocessing/scaling/simple_scale.c
    core/preprocessing/scaling/log_transform.c
    core/preprocessing/scaling/baseline.c
    core/preprocessing/derivatives/derivate.c
    core/preprocessing/derivatives/first_derivative.c
    core/preprocessing/derivatives/norris_williams.c
    core/preprocessing/derivatives/savitzky_golay.c
    core/preprocessing/derivatives/second_derivative.c
    core/preprocessing/smoothing/gaussian.c
    core/preprocessing/signal_conversion/fraction_to_percent.c
    core/preprocessing/signal_conversion/from_absorbance.c
    core/preprocessing/signal_conversion/kubelka_munk.c
    core/preprocessing/signal_conversion/percent_to_fraction.c
    core/preprocessing/signal_conversion/to_absorbance.c
    core/preprocessing/orthogonalization/epo.c
    core/preprocessing/orthogonalization/osc.c
    core/preprocessing/feature_selection/flexible_pca.c
    core/preprocessing/feature_selection/flexible_svd.c
    core/preprocessing/resampling/crop.c
    core/preprocessing/resampling/kbins_discretizer.c
    core/preprocessing/resampling/range_discretizer.c
    core/preprocessing/resampling/resample_transformer.c
    core/preprocessing/resampling/resampler.c
    core/preprocessing/specialized/fck_static.c
    core/splitters/binned_strat_group_kfold.c
    core/splitters/kbins_stratified.c
    core/splitters/kennard_stone.c
    core/splitters/kmeans.c
    core/splitters/split_common.c
    core/splitters/split_splitter.c
    core/splitters/spxy.c
    core/splitters/spxy_fold.c
    core/splitters/spxy_g_fold.c
    core/splitters/systematic_circular.c
    core/utilities/hotelling_t2.c
    core/utilities/nirs_metrics.c
    core/utilities/q_residuals.c
    core/utilities/signal_type_detector.c
    core/utilities/transfer_metrics.c)
target_include_directories(chemometrics4all_core
    PUBLIC  ${PROJECT_SOURCE_DIR}/cpp/include
            ${PROJECT_BINARY_DIR}/generated
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(chemometrics4all_core PUBLIC cxx_std_17)
target_compile_definitions(chemometrics4all_core PRIVATE CHEMOMETRICS4ALL_BUILDING_CORE=1)
set_target_properties(chemometrics4all_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(CHEMOMETRICS4ALL_ENABLE_FAULT_INJECTION)
    target_compile_definitions(chemometrics4all_core PUBLIC CHEMOMETRICS4ALL_ENABLE_FAULT_INJECTION=1)
endif()
set(_chemometrics4all_sanitizer_build OFF)
if(CHEMOMETRICS4ALL_ENABLE_ASAN OR CHEMOMETRICS4ALL_ENABLE_UBSAN OR CHEMOMETRICS4ALL_ENABLE_TSAN)
    set(_chemometrics4all_sanitizer_build ON)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(chemometrics4all_core PRIVATE
        $<$<NOT:$<CONFIG:Debug>>:-ffunction-sections>
        $<$<NOT:$<CONFIG:Debug>>:-fdata-sections>)
endif()
chemometrics4all_add_warnings(chemometrics4all_core)

# Optional backends (BLAS / OpenMP / CUDA): re-introduced phase-by-phase.
# Phase 0 keeps the build dependency-free.

# ---------------------------------------------------------------------------
# chemometrics4all_c — public C ABI shared library (libc4a.{so,dll,dylib}).
# chemometrics4all_c_static is the same surface, statically linked.
# ---------------------------------------------------------------------------
set(_chemometrics4all_c_sources
    c_api/c_api_context.cpp
    c_api/c_api_fck.cpp
    c_api/c_api_feature_selection.cpp
    c_api/c_api_filters.cpp
    c_api/c_api_filters_meta.cpp
    c_api/c_api_matrix.cpp
    c_api/c_api_metrics.cpp
    c_api/c_api_orthogonalization.cpp
    c_api/c_api_preprocessing.cpp
    c_api/c_api_resampling.cpp
    c_api/c_api_rng.cpp
    c_api/c_api_signal_conversion.cpp
    c_api/c_api_signal_type.cpp
    c_api/c_api_splitters.cpp
    c_api/c_api_transfer_metrics.cpp
    c_api/c_api_version.cpp)

function(chemometrics4all_apply_c_target target)
    target_sources(${target} PRIVATE $<TARGET_OBJECTS:chemometrics4all_core>)
    target_include_directories(${target}
        PUBLIC  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/cpp/include>
                $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/generated>
                $<INSTALL_INTERFACE:include>
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(${target} PRIVATE
            $<$<NOT:$<CONFIG:Debug>>:-ffunction-sections>
            $<$<NOT:$<CONFIG:Debug>>:-fdata-sections>)
    endif()
    chemometrics4all_add_warnings(${target})
endfunction()

if(CHEMOMETRICS4ALL_BUILD_SHARED)
    add_library(chemometrics4all_c SHARED ${_chemometrics4all_c_sources})
    chemometrics4all_apply_c_target(chemometrics4all_c)
    set_target_properties(chemometrics4all_c PROPERTIES
        OUTPUT_NAME c4a
        SOVERSION   ${C4A_ABI_VERSION_MAJOR}
        VERSION     ${C4A_ABI_VERSION})
    target_compile_definitions(chemometrics4all_c PRIVATE CHEMOMETRICS4ALL_BUILDING_DLL=1)

    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        if(NOT _chemometrics4all_sanitizer_build)
            target_link_options(chemometrics4all_c PRIVATE -Wl,--no-undefined)
        endif()
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/c_api/c4a_linux.map")
            target_link_options(chemometrics4all_c PRIVATE
                -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/c_api/c4a_linux.map
                $<$<NOT:$<CONFIG:Debug>>:-Wl,--gc-sections>)
        endif()
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        target_link_options(chemometrics4all_c PRIVATE
            -Wl,-undefined,error
            $<$<NOT:$<CONFIG:Debug>>:-Wl,-dead_strip>)
    elseif(MSVC)
        target_link_options(chemometrics4all_c PRIVATE
            $<$<NOT:$<CONFIG:Debug>>:/OPT:REF>
            $<$<NOT:$<CONFIG:Debug>>:/OPT:ICF>)
    endif()
    if((MSVC OR MINGW) AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/c_api/chemometrics4all.def")
        target_sources(chemometrics4all_c PRIVATE c_api/chemometrics4all.def)
    endif()
endif()

if(CHEMOMETRICS4ALL_BUILD_STATIC)
    add_library(chemometrics4all_c_static STATIC ${_chemometrics4all_c_sources})
    chemometrics4all_apply_c_target(chemometrics4all_c_static)
    set_target_properties(chemometrics4all_c_static PROPERTIES OUTPUT_NAME c4a_static)
    target_compile_definitions(chemometrics4all_c_static PUBLIC CHEMOMETRICS4ALL_STATIC=1)
endif()
