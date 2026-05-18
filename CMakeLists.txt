cmake_minimum_required(VERSION 3.21)

# ============================================================================
# Single source of truth: cpp/include/chemometrics4all/c4a_version.h
# ============================================================================
# Both the project semver (C4A_PROJECT_VERSION_*) and the ABI semver
# (C4A_ABI_VERSION_*) live in c4a_version.h. We parse the header at configure
# time so:
#   - PROJECT_VERSION is derived from C4A_PROJECT_VERSION_*  (used for
#     find_package(), packaging, install metadata).
#   - The shared library SONAME (cpp/src/CMakeLists.txt) is derived from
#     C4A_ABI_VERSION_MAJOR — independent of project semver so the ABI can
#     stay at 1.x while project moves from 0.x to 1.x.
#
# scripts/bump_version.sh edits c4a_version.h and rewrites every downstream
# manifest. The version-sync.yml workflow fails any PR where a manifest has
# drifted from the header.

set(_c4a_version_h "${CMAKE_CURRENT_SOURCE_DIR}/cpp/include/chemometrics4all/c4a_version.h")
if(NOT EXISTS "${_c4a_version_h}")
    message(FATAL_ERROR
        "chemometrics4all: canonical version header not found at ${_c4a_version_h}")
endif()
file(READ "${_c4a_version_h}" _c4a_version_h_content)

foreach(_macro IN ITEMS
        C4A_ABI_VERSION_MAJOR     C4A_ABI_VERSION_MINOR     C4A_ABI_VERSION_PATCH
        C4A_PROJECT_VERSION_MAJOR C4A_PROJECT_VERSION_MINOR C4A_PROJECT_VERSION_PATCH)
    # Anchor to a real preprocessor directive (allowing optional whitespace
    # between '#' and 'define') so commented-out lines like '// #define FOO'
    # are not matched.
    if(NOT _c4a_version_h_content MATCHES
            "(^|\n)[ \t]*#[ \t]*define[ \t]+${_macro}[ \t]+([0-9]+)")
        message(FATAL_ERROR
            "chemometrics4all: ${_macro} not found in c4a_version.h — header drift?")
    endif()
    set(${_macro} "${CMAKE_MATCH_2}")
endforeach()

set(C4A_ABI_VERSION
    "${C4A_ABI_VERSION_MAJOR}.${C4A_ABI_VERSION_MINOR}.${C4A_ABI_VERSION_PATCH}")
set(C4A_PROJECT_VERSION
    "${C4A_PROJECT_VERSION_MAJOR}.${C4A_PROJECT_VERSION_MINOR}.${C4A_PROJECT_VERSION_PATCH}")

# Cross-check the C4A_PROJECT_VERSION_STRING literal matches the numeric macros.
# Same line-anchored pattern as above (allows '# define' but rejects comments).
if(NOT _c4a_version_h_content MATCHES
        "(^|\n)[ \t]*#[ \t]*define[ \t]+C4A_PROJECT_VERSION_STRING[ \t]+\"([^\"]+)\"")
    message(FATAL_ERROR
        "chemometrics4all: C4A_PROJECT_VERSION_STRING not found in c4a_version.h")
endif()
if(NOT "${CMAKE_MATCH_2}" STREQUAL "${C4A_PROJECT_VERSION}")
    message(FATAL_ERROR
        "chemometrics4all: C4A_PROJECT_VERSION_STRING (\"${CMAKE_MATCH_2}\") disagrees "
        "with numeric macros (${C4A_PROJECT_VERSION}). Run "
        "scripts/bump_version.sh to re-sync.")
endif()

project(chemometrics4all
    VERSION ${C4A_PROJECT_VERSION}
    DESCRIPTION "Portable NIRS chemometrics engine: preprocessings, splitters, filters, augmentations, transfer metrics"
    HOMEPAGE_URL "https://github.com/GBeurier/chemometrics4all"
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

# Hidden by default — only C4A_API-decorated symbols escape chemometrics4all_c.
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_C_VISIBILITY_PRESET   hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

# Out-of-source builds only.
if(CMAKE_BINARY_DIR STREQUAL PROJECT_SOURCE_DIR)
    message(FATAL_ERROR
        "chemometrics4all does not support in-source builds. "
        "Run `cmake --preset dev-debug` (or dev-release / a ci-* preset).")
endif()

# Toplevel detection so consumers using find_package(chemometrics4all) or
# add_subdirectory() don't accidentally compile our tests.
if(PROJECT_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(CHEMOMETRICS4ALL_IS_TOPLEVEL ON)
else()
    set(CHEMOMETRICS4ALL_IS_TOPLEVEL OFF)
endif()

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(Chemometrics4allOptions)
include(CompilerWarnings)
include(SanitizersToolchain)

# OpenMP must be discovered at the top-level so its imported target
# (OpenMP::OpenMP_CXX) is visible in both cpp/src/ and cpp/tests/.
# When CHEMOMETRICS4ALL_WITH_OPENMP=OFF this is a no-op.
if(CHEMOMETRICS4ALL_WITH_OPENMP)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()

# Same pattern for CUDA: hoist find_package(CUDAToolkit) so the
# CUDA::cudart / CUDA::cublas imported targets are visible everywhere.
# Phase 45 uses host-side cuBLAS only, no .cu compilation, so we do
# NOT enable_language(CUDA).
if(CHEMOMETRICS4ALL_WITH_CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

# chemometrics4all needs at least one library output to be useful.
if(NOT CHEMOMETRICS4ALL_BUILD_SHARED AND NOT CHEMOMETRICS4ALL_BUILD_STATIC)
    message(FATAL_ERROR
        "CHEMOMETRICS4ALL_BUILD_SHARED and CHEMOMETRICS4ALL_BUILD_STATIC are both OFF — "
        "nothing to build. Enable at least one.")
endif()

# ============================================================================
# Subdirectories — only those that exist in this revision are added. Later
# Later PRs flip optional examples, bench targets and full bindings on.
# ============================================================================
add_subdirectory(cpp/src)

if(CHEMOMETRICS4ALL_BUILD_TESTS AND EXISTS "${PROJECT_SOURCE_DIR}/cpp/tests/CMakeLists.txt")
    enable_testing()
    add_subdirectory(cpp/tests)
endif()
if(CHEMOMETRICS4ALL_BUILD_CLI AND EXISTS "${PROJECT_SOURCE_DIR}/cpp/cli/CMakeLists.txt")
    add_subdirectory(cpp/cli)
endif()

# Add this so install rules below detect chemometrics4all_cli.
# (Empty placeholder kept to preserve commit context.)
if(CHEMOMETRICS4ALL_BUILD_EXAMPLES AND EXISTS "${PROJECT_SOURCE_DIR}/examples/c/CMakeLists.txt")
    add_subdirectory(examples/c)
endif()
if(CHEMOMETRICS4ALL_BUILD_BENCH AND EXISTS "${PROJECT_SOURCE_DIR}/bench/cpp/CMakeLists.txt")
    add_subdirectory(bench/cpp)
endif()
if(CHEMOMETRICS4ALL_BUILD_BINDINGS_R AND EXISTS "${PROJECT_SOURCE_DIR}/bindings/r/chemometrics4all/CMakeLists.txt")
    add_subdirectory(bindings/r/chemometrics4all)
endif()
if(CHEMOMETRICS4ALL_BUILD_BINDINGS_JS AND EXISTS "${PROJECT_SOURCE_DIR}/bindings/js/CMakeLists.txt")
    add_subdirectory(bindings/js)
endif()

# ============================================================================
# Install + export — wired up so downstream `find_package(chemometrics4all)` works
# once at least one target is built.
# ============================================================================
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(_chemometrics4all_install_targets "")
if(CHEMOMETRICS4ALL_BUILD_SHARED AND TARGET chemometrics4all_c)
    list(APPEND _chemometrics4all_install_targets chemometrics4all_c)
endif()
if(CHEMOMETRICS4ALL_BUILD_STATIC AND TARGET chemometrics4all_c_static)
    list(APPEND _chemometrics4all_install_targets chemometrics4all_c_static)
endif()
if(CHEMOMETRICS4ALL_BUILD_CLI AND TARGET chemometrics4all_cli)
    list(APPEND _chemometrics4all_install_targets chemometrics4all_cli)
endif()
# chemometrics4all_core is an OBJECT library, so it is consumed via $<TARGET_OBJECTS:...>
# inside chemometrics4all_c / chemometrics4all_c_static and never installed on its own.

if(_chemometrics4all_install_targets)
    install(TARGETS ${_chemometrics4all_install_targets}
            EXPORT  chemometrics4all_targets
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

    install(DIRECTORY cpp/include/chemometrics4all
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
            FILES_MATCHING PATTERN "*.h")

    # Generated visibility macro header — installed alongside the source headers
    # so consumers find it under <prefix>/include/chemometrics4all/c4a_export.h.
    install(FILES ${PROJECT_BINARY_DIR}/generated/chemometrics4all/c4a_export.h
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/chemometrics4all)

    configure_package_config_file(
        cmake/Chemometrics4allConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/Chemometrics4allConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/chemometrics4all)
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/Chemometrics4allConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion)
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/Chemometrics4allConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/Chemometrics4allConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/chemometrics4all)
    install(EXPORT chemometrics4all_targets
            FILE Chemometrics4allTargets.cmake
            NAMESPACE chemometrics4all::
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/chemometrics4all)
endif()

# ============================================================================
# Configuration summary — printed at the end of configure on the top level.
# ============================================================================
if(CHEMOMETRICS4ALL_IS_TOPLEVEL)
    message(STATUS "")
    message(STATUS "chemometrics4all ${PROJECT_VERSION}")
    message(STATUS "  CMake             : ${CMAKE_VERSION}")
    message(STATUS "  Build type        : ${CMAKE_BUILD_TYPE}")
    message(STATUS "  C compiler        : ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
    message(STATUS "  C++ compiler      : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
    message(STATUS "  Build shared      : ${CHEMOMETRICS4ALL_BUILD_SHARED}")
    message(STATUS "  Build static      : ${CHEMOMETRICS4ALL_BUILD_STATIC}")
    message(STATUS "  Build tests       : ${CHEMOMETRICS4ALL_BUILD_TESTS}")
    message(STATUS "  Build CLI         : ${CHEMOMETRICS4ALL_BUILD_CLI}")
    message(STATUS "  Sanitizers (A/U/T): ${CHEMOMETRICS4ALL_ENABLE_ASAN}/${CHEMOMETRICS4ALL_ENABLE_UBSAN}/${CHEMOMETRICS4ALL_ENABLE_TSAN}")
    message(STATUS "  Warnings as errs  : ${CHEMOMETRICS4ALL_WARNINGS_AS_ERRORS}")
    message(STATUS "")
endif()
