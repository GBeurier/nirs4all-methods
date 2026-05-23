# n4m_targets.cmake — wiring for libn4m + n4m_tests + n4m_cli
#
# After Phase A:
#   * single libn4m.{so,dll,dylib} target (`n4m_c`)
#   * single c_api/ surface (pls + donor wrappers merged in A3)
#   * cli at cpp/cli/, tests at cpp/tests/
#
# This file owns the donor-side core sources (preprocessing,
# augmentation, splitters, filters, utilities, common) lifted out of
# `_donor/` by M3 and the c_api wrappers under cpp/src/c_api/. The
# pls-side core lives in cpp/src/core/{aom_pop, diagnostics, selection,
# ...}/ and is wired by cpp/src/CMakeLists.txt — both source sets feed
# into the same n4m_core OBJECT library.

include_guard(GLOBAL)

# ---------------------------------------------------------------------------
# Generated visibility macro header for libn4m
# ---------------------------------------------------------------------------
configure_file(
    ${PROJECT_SOURCE_DIR}/cpp/include/n4m/n4m_export.h.in
    ${PROJECT_BINARY_DIR}/generated/n4m/n4m_export.h
    @ONLY)

# ---------------------------------------------------------------------------
# Donor-side core sources — added to the n4m_core OBJECT library defined
# in cpp/src/CMakeLists.txt with target_sources().
# ---------------------------------------------------------------------------
file(GLOB_RECURSE _N4M_DONOR_CORE_C
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/core/preprocessing/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/augmentation/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/splitters/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/filters/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/utilities/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/common/*.c"
)
file(GLOB _N4M_DONOR_CORE_CPP
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/core/common/context.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/common/matrix_view.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/common/status.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/common/version.cpp"
)
# FITPACK vendored Fortran .c files are gated under their own flag below.
list(FILTER _N4M_DONOR_CORE_C EXCLUDE REGEX "_vendored/fitpack")

# Optional FITPACK Fortran sources
file(GLOB _N4M_FITPACK_FORTRAN
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/core/common/_vendored/fitpack/*.f"
)

set(_N4M_FITPACK_ENABLED OFF)
if(_N4M_FITPACK_FORTRAN)
    include(CheckLanguage)
    check_language(Fortran)
    if(CMAKE_Fortran_COMPILER)
        enable_language(Fortran)
        set(_N4M_FITPACK_ENABLED ON)
    endif()
endif()

# Merge donor sources into n4m_core (defined by cpp/src/CMakeLists.txt
# before this file is included).
if(TARGET n4m_core)
    target_sources(n4m_core PRIVATE ${_N4M_DONOR_CORE_C} ${_N4M_DONOR_CORE_CPP})
    set_target_properties(n4m_core PROPERTIES
        C_STANDARD 11
        C_STANDARD_REQUIRED ON
    )
    if(_N4M_FITPACK_ENABLED)
        add_library(n4m_fitpack OBJECT ${_N4M_FITPACK_FORTRAN})
        target_compile_options(n4m_fitpack PRIVATE -std=legacy)
        set_target_properties(n4m_fitpack PROPERTIES POSITION_INDEPENDENT_CODE ON)
        target_compile_definitions(n4m_core PRIVATE N4M_HAVE_FITPACK=1)
    else()
        target_compile_definitions(n4m_core PRIVATE N4M_HAVE_FITPACK=0)
    endif()
endif()

# ---------------------------------------------------------------------------
# n4m_c — single shared library (libn4m.{so,dll,dylib})
# ---------------------------------------------------------------------------
file(GLOB _N4M_C_API_CPP
    CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/cpp/src/c_api/*.cpp"
)

set(_N4M_LINK_OBJS $<TARGET_OBJECTS:n4m_core>)
if(_N4M_FITPACK_ENABLED)
    list(APPEND _N4M_LINK_OBJS $<TARGET_OBJECTS:n4m_fitpack>)
endif()

add_library(n4m_c SHARED ${_N4M_C_API_CPP} ${_N4M_LINK_OBJS})
target_include_directories(n4m_c
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/cpp/include>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/generated>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_SOURCE_DIR}/cpp/src
)
target_compile_features(n4m_c PUBLIC cxx_std_17)
target_compile_definitions(n4m_c PRIVATE N4M_BUILDING_C_DLL=1)
set_target_properties(n4m_c PROPERTIES
    OUTPUT_NAME "n4m"
    SOVERSION ${N4M_ABI_VERSION_MAJOR}
    VERSION ${N4M_ABI_VERSION}
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# Apply the n4m_linux.map version script on Linux so libn4m.so exports
# only n4m_* C ABI symbols and hides all C++ class member implementations.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND EXISTS "${CMAKE_SOURCE_DIR}/cpp/src/c_api/n4m_linux.map")
    target_link_options(n4m_c PRIVATE
        -Wl,--version-script=${CMAKE_SOURCE_DIR}/cpp/src/c_api/n4m_linux.map)
endif()
n4m_add_warnings(n4m_c)

if(_N4M_FITPACK_ENABLED)
    target_link_libraries(n4m_c PRIVATE ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES})
endif()

# OBJECT-library transitive link deps don't propagate through TARGET_OBJECTS,
# so the consuming shared library has to re-link OpenMP / BLAS itself.
if(N4M_WITH_OPENMP AND TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(n4m_c PRIVATE OpenMP::OpenMP_CXX)
endif()
if(N4M_WITH_BLAS AND DEFINED BLAS_LIBRARIES)
    target_link_libraries(n4m_c PRIVATE ${BLAS_LIBRARIES})
endif()

# ---------------------------------------------------------------------------
# n4m_cli — small CLI for ABI introspection / selfcheck
# ---------------------------------------------------------------------------
if(EXISTS "${CMAKE_SOURCE_DIR}/cpp/cli/CMakeLists.txt")
    add_subdirectory(${CMAKE_SOURCE_DIR}/cpp/cli
                     ${CMAKE_BINARY_DIR}/cpp/cli)
endif()

# ---------------------------------------------------------------------------
# n4m_tests — internal harness against the C ABI surface
# ---------------------------------------------------------------------------
if(EXISTS "${CMAKE_SOURCE_DIR}/cpp/tests/CMakeLists.txt")
    add_subdirectory(${CMAKE_SOURCE_DIR}/cpp/tests
                     ${CMAKE_BINARY_DIR}/cpp/tests)
endif()
