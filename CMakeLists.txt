cmake_minimum_required(VERSION 3.21)

# Read the project version from CHANGELOG / single source of truth would be
# nice, but for now the version lives here and in cpp/include/pls4all/p4a_version.h
# (the header file is the canonical version for ABI / runtime queries — this
# CMake project version is used only for installation / find_package).
project(pls4all
    VERSION 0.67.0
    DESCRIPTION "Portable PLS/NIRS engine with stable C ABI"
    HOMEPAGE_URL "https://github.com/GBeurier/pls4all"
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

# Hidden by default — only P4A_API-decorated symbols escape pls4all_c.
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_C_VISIBILITY_PRESET   hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

# Out-of-source builds only.
if(CMAKE_BINARY_DIR STREQUAL PROJECT_SOURCE_DIR)
    message(FATAL_ERROR
        "pls4all does not support in-source builds. "
        "Run `cmake --preset dev-debug` (or dev-release / a ci-* preset).")
endif()

# Toplevel detection so consumers using find_package(pls4all) or
# add_subdirectory() don't accidentally compile our tests.
if(PROJECT_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(PLS4ALL_IS_TOPLEVEL ON)
else()
    set(PLS4ALL_IS_TOPLEVEL OFF)
endif()

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(Pls4allOptions)
include(CompilerWarnings)
include(SanitizersToolchain)

# pls4all needs at least one library output to be useful.
if(NOT PLS4ALL_BUILD_SHARED AND NOT PLS4ALL_BUILD_STATIC)
    message(FATAL_ERROR
        "PLS4ALL_BUILD_SHARED and PLS4ALL_BUILD_STATIC are both OFF — "
        "nothing to build. Enable at least one.")
endif()

# ============================================================================
# Subdirectories — only those that exist in this revision are added. Later
# Later PRs flip optional examples, bench targets and full bindings on.
# ============================================================================
add_subdirectory(cpp/src)

if(PLS4ALL_BUILD_TESTS AND EXISTS "${PROJECT_SOURCE_DIR}/cpp/tests/CMakeLists.txt")
    enable_testing()
    add_subdirectory(cpp/tests)
endif()
if(PLS4ALL_BUILD_CLI AND EXISTS "${PROJECT_SOURCE_DIR}/cpp/cli/CMakeLists.txt")
    add_subdirectory(cpp/cli)
endif()

# Add this so install rules below detect pls4all_cli.
# (Empty placeholder kept to preserve commit context.)
if(PLS4ALL_BUILD_EXAMPLES AND EXISTS "${PROJECT_SOURCE_DIR}/examples/c/CMakeLists.txt")
    add_subdirectory(examples/c)
endif()
if(PLS4ALL_BUILD_BENCH AND EXISTS "${PROJECT_SOURCE_DIR}/bench/cpp/CMakeLists.txt")
    add_subdirectory(bench/cpp)
endif()
if(PLS4ALL_BUILD_BINDINGS_R AND EXISTS "${PROJECT_SOURCE_DIR}/bindings/r/pls4all/CMakeLists.txt")
    add_subdirectory(bindings/r/pls4all)
endif()

# ============================================================================
# Install + export — wired up so downstream `find_package(pls4all)` works
# once at least one target is built.
# ============================================================================
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(_pls4all_install_targets "")
if(PLS4ALL_BUILD_SHARED AND TARGET pls4all_c)
    list(APPEND _pls4all_install_targets pls4all_c)
endif()
if(PLS4ALL_BUILD_STATIC AND TARGET pls4all_c_static)
    list(APPEND _pls4all_install_targets pls4all_c_static)
endif()
if(PLS4ALL_BUILD_CLI AND TARGET pls4all_cli)
    list(APPEND _pls4all_install_targets pls4all_cli)
endif()
# pls4all_core is an OBJECT library, so it is consumed via $<TARGET_OBJECTS:...>
# inside pls4all_c / pls4all_c_static and never installed on its own.

if(_pls4all_install_targets)
    install(TARGETS ${_pls4all_install_targets}
            EXPORT  pls4all_targets
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

    install(DIRECTORY cpp/include/pls4all
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
            FILES_MATCHING PATTERN "*.h")

    # Generated visibility macro header — installed alongside the source headers
    # so consumers find it under <prefix>/include/pls4all/p4a_export.h.
    install(FILES ${PROJECT_BINARY_DIR}/generated/pls4all/p4a_export.h
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/pls4all)

    configure_package_config_file(
        cmake/Pls4allConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/Pls4allConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pls4all)
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/Pls4allConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion)
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/Pls4allConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/Pls4allConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pls4all)
    install(EXPORT pls4all_targets
            FILE Pls4allTargets.cmake
            NAMESPACE pls4all::
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pls4all)
endif()

# ============================================================================
# Configuration summary — printed at the end of configure on the top level.
# ============================================================================
if(PLS4ALL_IS_TOPLEVEL)
    message(STATUS "")
    message(STATUS "pls4all ${PROJECT_VERSION}")
    message(STATUS "  CMake             : ${CMAKE_VERSION}")
    message(STATUS "  Build type        : ${CMAKE_BUILD_TYPE}")
    message(STATUS "  C compiler        : ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
    message(STATUS "  C++ compiler      : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
    message(STATUS "  Build shared      : ${PLS4ALL_BUILD_SHARED}")
    message(STATUS "  Build static      : ${PLS4ALL_BUILD_STATIC}")
    message(STATUS "  Build tests       : ${PLS4ALL_BUILD_TESTS}")
    message(STATUS "  Build CLI         : ${PLS4ALL_BUILD_CLI}")
    message(STATUS "  Sanitizers (A/U/T): ${PLS4ALL_ENABLE_ASAN}/${PLS4ALL_ENABLE_UBSAN}/${PLS4ALL_ENABLE_TSAN}")
    message(STATUS "  Warnings as errs  : ${PLS4ALL_WARNINGS_AS_ERRORS}")
    message(STATUS "")
endif()
