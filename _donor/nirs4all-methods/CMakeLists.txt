cmake_minimum_required(VERSION 3.21)

# ============================================================================
# Single source of truth: cpp/include/n4m/n4m_version.h
# ============================================================================
# Both the project semver (N4M_PROJECT_VERSION_*) and the ABI semver
# (N4M_ABI_VERSION_*) live in n4m_version.h. We parse the header at configure
# time so:
#   - PROJECT_VERSION is derived from N4M_PROJECT_VERSION_*  (used for
#     find_package(), packaging, install metadata).
#   - The shared library SONAME (cpp/src/CMakeLists.txt) is derived from
#     N4M_ABI_VERSION_MAJOR — independent of project semver so the ABI can
#     stay at 1.x while project moves from 0.x to 1.x.
#
# scripts/bump_version.sh edits n4m_version.h and rewrites every downstream
# manifest. The version-sync.yml workflow fails any PR where a manifest has
# drifted from the header.

set(_n4m_version_h "${CMAKE_CURRENT_SOURCE_DIR}/cpp/include/n4m/n4m_version.h")
if(NOT EXISTS "${_n4m_version_h}")
    message(FATAL_ERROR
        "nirs4all-methods: canonical version header not found at ${_n4m_version_h}")
endif()
file(READ "${_n4m_version_h}" _n4m_version_h_content)

foreach(_macro IN ITEMS
        N4M_ABI_VERSION_MAJOR     N4M_ABI_VERSION_MINOR     N4M_ABI_VERSION_PATCH
        N4M_PROJECT_VERSION_MAJOR N4M_PROJECT_VERSION_MINOR N4M_PROJECT_VERSION_PATCH)
    # Anchor to a real preprocessor directive (allowing optional whitespace
    # between '#' and 'define') so commented-out lines like '// #define FOO'
    # are not matched.
    if(NOT _n4m_version_h_content MATCHES
            "(^|\n)[ \t]*#[ \t]*define[ \t]+${_macro}[ \t]+([0-9]+)")
        message(FATAL_ERROR
            "nirs4all-methods: ${_macro} not found in n4m_version.h — header drift?")
    endif()
    set(${_macro} "${CMAKE_MATCH_2}")
endforeach()

set(N4M_ABI_VERSION
    "${N4M_ABI_VERSION_MAJOR}.${N4M_ABI_VERSION_MINOR}.${N4M_ABI_VERSION_PATCH}")
set(N4M_PROJECT_VERSION
    "${N4M_PROJECT_VERSION_MAJOR}.${N4M_PROJECT_VERSION_MINOR}.${N4M_PROJECT_VERSION_PATCH}")

# Cross-check the N4M_PROJECT_VERSION_STRING literal matches the numeric macros.
# Same line-anchored pattern as above (allows '# define' but rejects comments).
if(NOT _n4m_version_h_content MATCHES
        "(^|\n)[ \t]*#[ \t]*define[ \t]+N4M_PROJECT_VERSION_STRING[ \t]+\"([^\"]+)\"")
    message(FATAL_ERROR
        "nirs4all-methods: N4M_PROJECT_VERSION_STRING not found in n4m_version.h")
endif()
if(NOT "${CMAKE_MATCH_2}" STREQUAL "${N4M_PROJECT_VERSION}")
    message(FATAL_ERROR
        "nirs4all-methods: N4M_PROJECT_VERSION_STRING (\"${CMAKE_MATCH_2}\") disagrees "
        "with numeric macros (${N4M_PROJECT_VERSION}). Run "
        "scripts/bump_version.sh to re-sync.")
endif()

project(nirs4all-methods
    VERSION ${N4M_PROJECT_VERSION}
    DESCRIPTION "Portable NIRS chemometrics engine: preprocessings, splitters, filters, augmentations, transfer metrics"
    HOMEPAGE_URL "https://github.com/GBeurier/nirs4all-methods"
    LANGUAGES C CXX)

# ============================================================================
# Language and visibility defaults
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Hidden by default — only N4M_API-decorated symbols escape n4m_c.
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_C_VISIBILITY_PRESET   hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

# Out-of-source builds only.
if(CMAKE_BINARY_DIR STREQUAL PROJECT_SOURCE_DIR)
    message(FATAL_ERROR
        "nirs4all-methods does not support in-source builds. "
        "Run `cmake --preset dev-debug` (or dev-release / a ci-* preset).")
endif()

# Toplevel detection so consumers using find_package(N4M) or
# add_subdirectory() don't accidentally compile our tests.
if(PROJECT_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(N4M_IS_TOPLEVEL ON)
else()
    set(N4M_IS_TOPLEVEL OFF)
endif()

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(N4MOptions)
include(CompilerWarnings)
include(SanitizersToolchain)

set(N4M_HAVE_FITPACK OFF)
if(N4M_WITH_FITPACK STREQUAL "ON"
        OR N4M_WITH_FITPACK STREQUAL "AUTO")
    set(_n4m_skip_fitpack_fortran OFF)
    if(WIN32 AND (MSVC OR CMAKE_GENERATOR MATCHES "Visual Studio"))
        set(_n4m_skip_fitpack_fortran ON)
    endif()

    if(_n4m_skip_fitpack_fortran)
        if(N4M_WITH_FITPACK STREQUAL "ON")
            message(FATAL_ERROR
                "N4M_WITH_FITPACK=ON is not supported with the "
                "Visual Studio generator. Use AUTO/OFF or a non-MSVC toolchain.")
        endif()
    else()
        if(NOT DEFINED CMAKE_Fortran_COMPILER)
            find_program(_n4m_fortran_compiler
                NAMES gfortran x86_64-conda-linux-gnu-gfortran flang ifort ifx)
            if(_n4m_fortran_compiler)
                set(CMAKE_Fortran_COMPILER "${_n4m_fortran_compiler}" CACHE FILEPATH
                    "Fortran compiler used for the vendored FITPACK spline smoother")
            endif()
        endif()
        include(CheckLanguage)
        check_language(Fortran)
        if(CMAKE_Fortran_COMPILER)
            enable_language(Fortran)
            set(N4M_HAVE_FITPACK ON)
        elseif(N4M_WITH_FITPACK STREQUAL "ON")
            message(FATAL_ERROR
                "N4M_WITH_FITPACK=ON requires a working Fortran compiler.")
        endif()
    endif()
endif()

# The vendored FITPACK subset is plain numerical Fortran 77 and does not call
# the Fortran runtime. Keeping the implicit gfortran runtime off avoids leaking
# a compiler-runtime dependency into libn4m for this tiny source subset.
if(N4M_HAVE_FITPACK AND CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
    set(CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES "")
    set(CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES "")
    set(CMAKE_Fortran_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")
    set(CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES "" CACHE STRING "" FORCE)
    set(CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES "" CACHE STRING "" FORCE)
    set(CMAKE_Fortran_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "" CACHE STRING "" FORCE)
endif()

# OpenMP must be discovered at the top-level so its imported target
# (OpenMP::OpenMP_CXX) is visible in both cpp/src/ and cpp/tests/.
# When N4M_WITH_OPENMP=OFF this is a no-op.
if(N4M_WITH_OPENMP)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()

# Same pattern for CUDA: hoist find_package(CUDAToolkit) so the
# CUDA::cudart / CUDA::cublas imported targets are visible everywhere.
# Phase 45 uses host-side cuBLAS only, no .cu compilation, so we do
# NOT enable_language(CUDA).
if(N4M_WITH_CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

# nirs4all-methods needs at least one library output to be useful.
if(NOT N4M_BUILD_SHARED AND NOT N4M_BUILD_STATIC)
    message(FATAL_ERROR
        "N4M_BUILD_SHARED and N4M_BUILD_STATIC are both OFF — "
        "nothing to build. Enable at least one.")
endif()

# ============================================================================
# Subdirectories — only those that exist in this revision are added. Later
# Later PRs flip optional examples, bench targets and full bindings on.
# ============================================================================
add_subdirectory(cpp/src)

if(N4M_BUILD_TESTS AND EXISTS "${PROJECT_SOURCE_DIR}/cpp/tests/CMakeLists.txt")
    enable_testing()
    add_subdirectory(cpp/tests)
endif()
if(N4M_BUILD_CLI AND EXISTS "${PROJECT_SOURCE_DIR}/cpp/cli/CMakeLists.txt")
    add_subdirectory(cpp/cli)
endif()

# Add this so install rules below detect n4m_cli.
# (Empty placeholder kept to preserve commit context.)
if(N4M_BUILD_EXAMPLES AND EXISTS "${PROJECT_SOURCE_DIR}/examples/c/CMakeLists.txt")
    add_subdirectory(examples/c)
endif()
if(N4M_BUILD_BENCH AND EXISTS "${PROJECT_SOURCE_DIR}/bench/cpp/CMakeLists.txt")
    add_subdirectory(bench/cpp)
endif()
if(N4M_BUILD_BINDINGS_R AND EXISTS "${PROJECT_SOURCE_DIR}/bindings/r/n4m/CMakeLists.txt")
    add_subdirectory(bindings/r/n4m)
endif()
if(N4M_BUILD_BINDINGS_JS AND EXISTS "${PROJECT_SOURCE_DIR}/bindings/js/CMakeLists.txt")
    add_subdirectory(bindings/js)
endif()

# ============================================================================
# Install + export — wired up so downstream `find_package(N4M)` works
# once at least one target is built.
# ============================================================================
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(_n4m_install_targets "")
if(N4M_BUILD_SHARED AND TARGET n4m_c)
    list(APPEND _n4m_install_targets n4m_c)
endif()
if(N4M_BUILD_STATIC AND TARGET n4m_c_static)
    list(APPEND _n4m_install_targets n4m_c_static)
endif()
if(N4M_BUILD_CLI AND TARGET n4m_cli)
    list(APPEND _n4m_install_targets n4m_cli)
endif()
# n4m_core is an OBJECT library, so it is consumed via $<TARGET_OBJECTS:...>
# inside n4m_c / n4m_c_static and never installed on its own.

if(_n4m_install_targets)
    install(TARGETS ${_n4m_install_targets}
            EXPORT  n4m_targets
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

    install(DIRECTORY cpp/include/n4m
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
            FILES_MATCHING PATTERN "*.h")

    # Generated visibility macro header — installed alongside the source headers
    # so consumers find it under <prefix>/include/n4m/n4m_export.h.
    install(FILES ${PROJECT_BINARY_DIR}/generated/n4m/n4m_export.h
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/n4m)

    configure_package_config_file(
        cmake/N4MConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/N4MConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/N4M)
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/N4MConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion)
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/N4MConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/N4MConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/N4M)
    install(EXPORT n4m_targets
            FILE N4MTargets.cmake
            NAMESPACE n4m::
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/N4M)
endif()

# ============================================================================
# Configuration summary — printed at the end of configure on the top level.
# ============================================================================
if(N4M_IS_TOPLEVEL)
    message(STATUS "")
    message(STATUS "nirs4all-methods ${PROJECT_VERSION}")
    message(STATUS "  CMake             : ${CMAKE_VERSION}")
    message(STATUS "  Build type        : ${CMAKE_BUILD_TYPE}")
    message(STATUS "  C compiler        : ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
    message(STATUS "  C++ compiler      : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
    message(STATUS "  Build shared      : ${N4M_BUILD_SHARED}")
    message(STATUS "  Build static      : ${N4M_BUILD_STATIC}")
    message(STATUS "  Build tests       : ${N4M_BUILD_TESTS}")
    message(STATUS "  Build CLI         : ${N4M_BUILD_CLI}")
    message(STATUS "  FITPACK smoothing : ${N4M_HAVE_FITPACK}")
    message(STATUS "  Sanitizers (A/U/T): ${N4M_ENABLE_ASAN}/${N4M_ENABLE_UBSAN}/${N4M_ENABLE_TSAN}")
    message(STATUS "  Warnings as errs  : ${N4M_WARNINGS_AS_ERRORS}")
    message(STATUS "")
endif()
