# n4m_targets.cmake — wiring for libn4m + n4m_tests + n4m_cli
#
# Added in the post-M7 session that closes the "donor methods not in
# benchmark registry" gap. Builds the donor sources (lifted to
# cpp/src/core/{preprocessing,augmentation,splitters,filters,utilities,common}/
# by M3, with the common-core unified in M2.5) into a separate shared
# library libn4m.so that lives alongside libn4m.so.
#
# n4m_tests and n4m_tests remain disjoint test binaries — each links
# its own library and tests its own ABI surface. Eventually (M5+M6 paired
# session) the two ABIs collapse into a single n4m_* surface and the two
# binaries merge; until then they coexist.

include_guard(GLOBAL)

# ---------------------------------------------------------------------------
# Generated visibility macro header for libn4m
# ---------------------------------------------------------------------------
configure_file(
    ${PROJECT_SOURCE_DIR}/cpp/include/n4m/n4m_export.h.in
    ${PROJECT_BINARY_DIR}/generated/n4m/n4m_export.h
    @ONLY)

# ---------------------------------------------------------------------------
# Source list — donor sources lifted out of _donor/ by M3 + common from M2.5
# ---------------------------------------------------------------------------
file(GLOB_RECURSE _N4M_CORE_C
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/core/preprocessing/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/augmentation/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/splitters/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/filters/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/utilities/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/common/*.c"
)
file(GLOB _N4M_CORE_CPP
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/core/common/context.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/common/matrix_view.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/common/status.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/common/version.cpp"
)
# Skip only the FITPACK vendored Fortran .c files (none here today; FITPACK
# is .f) at this level — keep the C-level vendored kernels under
# filters/_vendored (isolation_forest, lof, min_cov_det) since they are
# called from the filter c_api wrappers.
list(FILTER _N4M_CORE_C   EXCLUDE REGEX "_vendored/fitpack")

# Optional FITPACK Fortran sources
file(GLOB _N4M_FITPACK_FORTRAN
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/core/common/_vendored/fitpack/*.f"
)

set(_N4M_FITPACK_ENABLED OFF)
if(_N4M_FITPACK_FORTRAN)
    # The donor's FITPACK path needs gfortran. Detect and gate.
    include(CheckLanguage)
    check_language(Fortran)
    if(CMAKE_Fortran_COMPILER)
        enable_language(Fortran)
        set(_N4M_FITPACK_ENABLED ON)
    endif()
endif()

# ---------------------------------------------------------------------------
# n4m_core — internal OBJECT library
# ---------------------------------------------------------------------------
add_library(n4m_core OBJECT ${_N4M_CORE_C} ${_N4M_CORE_CPP})
target_include_directories(n4m_core
    PUBLIC
        ${CMAKE_SOURCE_DIR}/cpp/include
        ${PROJECT_BINARY_DIR}/generated
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_features(n4m_core PUBLIC cxx_std_17)
set_target_properties(n4m_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
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
target_compile_definitions(n4m_core PRIVATE N4M_BUILDING_CORE=1)

# Use the same warning baseline as n4m_core. The donor codebase has
# been tested under those warnings.
include(${CMAKE_SOURCE_DIR}/cmake/CompilerWarnings.cmake)
pls4all_add_warnings(n4m_core)

# ---------------------------------------------------------------------------
# n4m_c — shared library (libn4m.so / libn4m.dll / libn4m.dylib)
# ---------------------------------------------------------------------------
file(GLOB _N4M_C_API_CPP
    CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/cpp/src/c_api_n4m/*.cpp"
)
set(_N4M_LINK_OBJS $<TARGET_OBJECTS:n4m_core>)
if(_N4M_FITPACK_ENABLED)
    list(APPEND _N4M_LINK_OBJS $<TARGET_OBJECTS:n4m_fitpack>)
endif()
# Pls4all c_api wrappers (PLS-side public ABI) — merged into libn4m.so
file(GLOB _PLS_C_API_CPP
    CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/cpp/src/c_api/*.cpp"
)
add_library(n4m_c SHARED ${_N4M_C_API_CPP} ${_PLS_C_API_CPP} ${_N4M_LINK_OBJS} $<TARGET_OBJECTS:pls4all_core>)
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
pls4all_add_warnings(n4m_c)

if(_N4M_FITPACK_ENABLED)
    target_link_libraries(n4m_c PRIVATE ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES})
endif()

# pls4all_core is an OBJECT library, so its transitive link-deps (OpenMP,
# BLAS) don't propagate via $<TARGET_OBJECTS:>. Re-link them explicitly so
# the n4m_c shared library resolves omp_/cblas_ symbols at load time.
if(PLS4ALL_WITH_OPENMP AND TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(n4m_c PRIVATE OpenMP::OpenMP_CXX)
endif()
if(PLS4ALL_WITH_BLAS AND DEFINED BLAS_LIBRARIES)
    target_link_libraries(n4m_c PRIVATE ${BLAS_LIBRARIES})
endif()

# ---------------------------------------------------------------------------
# n4m_cli — small CLI for ABI introspection / selfcheck
# ---------------------------------------------------------------------------
if(EXISTS "${CMAKE_SOURCE_DIR}/cpp/cli_n4m/n4m_cli.cpp")
    add_executable(n4m_cli ${CMAKE_SOURCE_DIR}/cpp/cli_n4m/n4m_cli.cpp)
    target_link_libraries(n4m_cli PRIVATE n4m_c)
    target_include_directories(n4m_cli PRIVATE ${PROJECT_BINARY_DIR}/generated)
    pls4all_add_warnings(n4m_cli)
endif()

# ---------------------------------------------------------------------------
# n4m_tests — internal harness against the C ABI surface
# ---------------------------------------------------------------------------
if(EXISTS "${CMAKE_SOURCE_DIR}/cpp/tests_n4m/CMakeLists.txt")
    add_subdirectory(${CMAKE_SOURCE_DIR}/cpp/tests_n4m
                     ${CMAKE_BINARY_DIR}/cpp/tests_n4m)
endif()
