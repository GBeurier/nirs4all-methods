# cmake/Pls4allOptions.cmake
# Centralised declaration of every PLS4ALL_* configure-time option.
# Including this module is idempotent.

if(DEFINED _PLS4ALL_OPTIONS_INCLUDED)
    return()
endif()
set(_PLS4ALL_OPTIONS_INCLUDED ON)

# ---- Build targets ---------------------------------------------------------
option(PLS4ALL_BUILD_SHARED        "Build pls4all_c shared library (libp4a.so/.dll/.dylib)" ON)
option(PLS4ALL_BUILD_STATIC        "Build pls4all_c_static (for Android predict-only embedders)" ON)
option(PLS4ALL_BUILD_TESTS         "Build the doctest-based unit/ABI test suite"            ON)
option(PLS4ALL_BUILD_FUZZ          "Build libFuzzer harnesses (requires clang)"             OFF)
option(PLS4ALL_BUILD_CLI           "Build the pls4all_cli executable"                       ON)
option(PLS4ALL_BUILD_EXAMPLES      "Build examples/c/*"                                     OFF)
option(PLS4ALL_BUILD_BENCH         "Build bench/cpp/*"                                      OFF)

# Bindings — disabled by default; enabled by the binding's own build scripts.
option(PLS4ALL_BUILD_BINDINGS_PYTHON   "Build the Python ctypes binding wheel"     OFF)
option(PLS4ALL_BUILD_BINDINGS_R        "Build the R .Call package"                 OFF)
option(PLS4ALL_BUILD_BINDINGS_MATLAB   "Build the MATLAB MEX wrapper"              OFF)
option(PLS4ALL_BUILD_BINDINGS_JS       "Build the JS/WASM bundle (Emscripten)"     OFF)
option(PLS4ALL_BUILD_BINDINGS_ANDROID  "Build the Android AAR (NDK)"               OFF)

# ---- Accelerated backends (Phase 7) ---------------------------------------
option(PLS4ALL_WITH_BLAS    "Build optional BLAS backend"   OFF)
option(PLS4ALL_WITH_OPENMP  "Build optional OpenMP backend" OFF)
option(PLS4ALL_WITH_CUDA    "Build optional CUDA backend"   OFF)

# ---- Diagnostics ----------------------------------------------------------
option(PLS4ALL_ENABLE_ASAN             "Enable AddressSanitizer (clang/gcc only)"            OFF)
option(PLS4ALL_ENABLE_UBSAN            "Enable UndefinedBehaviorSanitizer"                   OFF)
option(PLS4ALL_ENABLE_TSAN             "Enable ThreadSanitizer (mutually exclusive with ASAN)" OFF)
option(PLS4ALL_ENABLE_COVERAGE         "Enable gcov / llvm-cov coverage instrumentation"     OFF)
option(PLS4ALL_ENABLE_LTO              "Enable Link-Time Optimisation (only if toolchain supports IPO)" OFF)
option(PLS4ALL_ENABLE_FAULT_INJECTION  "Expose test-only hooks to inject bad_alloc, etc."    OFF)
option(PLS4ALL_WARNINGS_AS_ERRORS      "Treat compile warnings as errors (CI default)"       OFF)

# ---- Option validation ----------------------------------------------------
if(PLS4ALL_ENABLE_ASAN AND PLS4ALL_ENABLE_TSAN)
    message(FATAL_ERROR
        "PLS4ALL_ENABLE_ASAN and PLS4ALL_ENABLE_TSAN are mutually exclusive — "
        "TSAN cannot run under AddressSanitizer.")
endif()

if(PLS4ALL_BUILD_FUZZ AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang")
    message(WARNING
        "PLS4ALL_BUILD_FUZZ requires clang for libFuzzer support; "
        "disabling fuzz targets for this build.")
    set(PLS4ALL_BUILD_FUZZ OFF CACHE BOOL "" FORCE)
endif()

# LTO is opt-in: only applied when PLS4ALL_ENABLE_LTO=ON AND the toolchain
# supports IPO AND we are not in a Debug build. Default is OFF.
if(PLS4ALL_ENABLE_LTO AND NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    include(CheckIPOSupported)
    check_ipo_supported(RESULT _ipo_supported OUTPUT _ipo_msg LANGUAGES C CXX)
    if(_ipo_supported)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    else()
        message(WARNING
            "PLS4ALL_ENABLE_LTO=ON but the toolchain does not support IPO: ${_ipo_msg}. "
            "Continuing without LTO.")
    endif()
endif()
