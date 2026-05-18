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
    core/version.cpp)
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
    c_api/c_api_matrix.cpp
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
