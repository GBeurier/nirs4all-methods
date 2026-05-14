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
    core/status.cpp
    core/version.cpp
    core/cars_selection.cpp
    core/classification_metrics.cpp
    core/component_coefficients.cpp
    core/context.cpp
    core/ga_selection.cpp
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
    core/interval_selection.cpp
    core/lw_pls.cpp
    core/pls_lda.cpp
    core/pls_logistic.cpp
    core/pipeline.cpp
    core/random_frog_selection.cpp
    core/scars_selection.cpp
    core/spa_selection.cpp
    core/stability_selection.cpp
    core/uve_selection.cpp
    core/validation.cpp)
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
    c_api/c_api_model.cpp)

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
