# cmake/Chemometrics4allOptions.cmake
# Centralised declaration of every CHEMOMETRICS4ALL_* configure-time option.
# Including this module is idempotent.

if(DEFINED _CHEMOMETRICS4ALL_OPTIONS_INCLUDED)
    return()
endif()
set(_CHEMOMETRICS4ALL_OPTIONS_INCLUDED ON)

# ---- Build targets ---------------------------------------------------------
option(CHEMOMETRICS4ALL_BUILD_SHARED        "Build chemometrics4all_c shared library (libc4a.so/.dll/.dylib)" ON)
option(CHEMOMETRICS4ALL_BUILD_STATIC        "Build chemometrics4all_c_static (for Android predict-only embedders)" ON)
option(CHEMOMETRICS4ALL_BUILD_TESTS         "Build the doctest-based unit/ABI test suite"            ON)
option(CHEMOMETRICS4ALL_BUILD_FUZZ          "Build libFuzzer harnesses (requires clang)"             OFF)
option(CHEMOMETRICS4ALL_BUILD_CLI           "Build the chemometrics4all_cli executable"                       ON)
option(CHEMOMETRICS4ALL_BUILD_EXAMPLES      "Build examples/c/*"                                     OFF)
option(CHEMOMETRICS4ALL_BUILD_BENCH         "Build bench/cpp/*"                                      OFF)

# Bindings — disabled by default; enabled by the binding's own build scripts.
option(CHEMOMETRICS4ALL_BUILD_BINDINGS_PYTHON   "Build the Python ctypes binding wheel"     OFF)
option(CHEMOMETRICS4ALL_BUILD_BINDINGS_R        "Build the R .Call package"                 OFF)
option(CHEMOMETRICS4ALL_BUILD_BINDINGS_MATLAB   "Build the MATLAB MEX wrapper"              OFF)
option(CHEMOMETRICS4ALL_BUILD_BINDINGS_JS       "Build the JS/WASM bundle (Emscripten)"     OFF)
option(CHEMOMETRICS4ALL_BUILD_BINDINGS_ANDROID  "Build the Android AAR (NDK)"               OFF)

# ---- Accelerated backends (Phase 7) ---------------------------------------
option(CHEMOMETRICS4ALL_WITH_BLAS    "Build optional BLAS backend"   OFF)
option(CHEMOMETRICS4ALL_WITH_OPENMP  "Build optional OpenMP backend" OFF)
option(CHEMOMETRICS4ALL_WITH_CUDA    "Build optional CUDA backend"   OFF)
set(CHEMOMETRICS4ALL_WITH_FITPACK "AUTO" CACHE STRING
    "Build the vendored FITPACK spline smoother when a compatible Fortran compiler is available (AUTO, ON, OFF)")
set_property(CACHE CHEMOMETRICS4ALL_WITH_FITPACK PROPERTY STRINGS AUTO ON OFF)

# ---- Diagnostics ----------------------------------------------------------
option(CHEMOMETRICS4ALL_ENABLE_ASAN             "Enable AddressSanitizer (clang/gcc only)"            OFF)
option(CHEMOMETRICS4ALL_ENABLE_UBSAN            "Enable UndefinedBehaviorSanitizer"                   OFF)
option(CHEMOMETRICS4ALL_ENABLE_TSAN             "Enable ThreadSanitizer (mutually exclusive with ASAN)" OFF)
option(CHEMOMETRICS4ALL_ENABLE_COVERAGE         "Enable gcov / llvm-cov coverage instrumentation"     OFF)
option(CHEMOMETRICS4ALL_ENABLE_LTO              "Enable Link-Time Optimisation (only if toolchain supports IPO)" OFF)
option(CHEMOMETRICS4ALL_ENABLE_FAULT_INJECTION  "Expose test-only hooks to inject bad_alloc, etc."    OFF)
option(CHEMOMETRICS4ALL_WARNINGS_AS_ERRORS      "Treat compile warnings as errors (CI default)"       OFF)

# ---- Option validation ----------------------------------------------------
if(CHEMOMETRICS4ALL_ENABLE_ASAN AND CHEMOMETRICS4ALL_ENABLE_TSAN)
    message(FATAL_ERROR
        "CHEMOMETRICS4ALL_ENABLE_ASAN and CHEMOMETRICS4ALL_ENABLE_TSAN are mutually exclusive — "
        "TSAN cannot run under AddressSanitizer.")
endif()

if(NOT CHEMOMETRICS4ALL_WITH_FITPACK STREQUAL "AUTO"
        AND NOT CHEMOMETRICS4ALL_WITH_FITPACK STREQUAL "ON"
        AND NOT CHEMOMETRICS4ALL_WITH_FITPACK STREQUAL "OFF")
    message(FATAL_ERROR
        "CHEMOMETRICS4ALL_WITH_FITPACK must be AUTO, ON, or OFF "
        "(got '${CHEMOMETRICS4ALL_WITH_FITPACK}').")
endif()

if(CHEMOMETRICS4ALL_BUILD_FUZZ AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang")
    message(WARNING
        "CHEMOMETRICS4ALL_BUILD_FUZZ requires clang for libFuzzer support; "
        "disabling fuzz targets for this build.")
    set(CHEMOMETRICS4ALL_BUILD_FUZZ OFF CACHE BOOL "" FORCE)
endif()

# LTO is opt-in: only applied when CHEMOMETRICS4ALL_ENABLE_LTO=ON AND the toolchain
# supports IPO AND we are not in a Debug build. Default is OFF.
if(CHEMOMETRICS4ALL_ENABLE_LTO AND NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    include(CheckIPOSupported)
    check_ipo_supported(RESULT _ipo_supported OUTPUT _ipo_msg LANGUAGES C CXX)
    if(_ipo_supported)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    else()
        message(WARNING
            "CHEMOMETRICS4ALL_ENABLE_LTO=ON but the toolchain does not support IPO: ${_ipo_msg}. "
            "Continuing without LTO.")
    endif()
endif()
