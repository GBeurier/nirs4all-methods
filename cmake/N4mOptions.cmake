# cmake/N4mOptions.cmake
# Centralised declaration of every N4M_* configure-time option.
# Including this module is idempotent.

if(DEFINED _N4M_OPTIONS_INCLUDED)
    return()
endif()
set(_N4M_OPTIONS_INCLUDED ON)

# ---- Build targets ---------------------------------------------------------
option(N4M_BUILD_SHARED        "Build n4m_c shared library (libn4m.so/.dll/.dylib)" ON)
option(N4M_BUILD_STATIC        "Build n4m_c_static (for Android predict-only embedders)" ON)
option(N4M_BUILD_TESTS         "Build the doctest-based unit/ABI test suite"            ON)
option(N4M_BUILD_FUZZ          "Build libFuzzer harnesses (requires clang)"             OFF)
option(N4M_BUILD_CLI           "Build the n4m_cli executable"                       ON)
option(N4M_BUILD_EXAMPLES      "Build examples/c/*"                                     OFF)
option(N4M_BUILD_BENCH         "Build bench/cpp/*"                                      OFF)

# Bindings — disabled by default; enabled by the binding's own build scripts.
option(N4M_BUILD_BINDINGS_PYTHON   "Build the Python ctypes binding wheel"     OFF)
option(N4M_BUILD_BINDINGS_R        "Build the R .Call package"                 OFF)
option(N4M_BUILD_BINDINGS_MATLAB   "Build the MATLAB MEX wrapper"              OFF)
option(N4M_BUILD_BINDINGS_JS       "Build the JS/WASM bundle (Emscripten)"     OFF)
option(N4M_BUILD_BINDINGS_ANDROID  "Build the Android AAR (NDK)"               OFF)

# ---- Accelerated backends (Phase 7) ---------------------------------------
option(N4M_WITH_BLAS    "Build optional BLAS backend"   OFF)
option(N4M_WITH_OPENMP  "Build optional OpenMP backend" OFF)
option(N4M_WITH_CUDA    "Build optional CUDA backend"   OFF)

# ---- Diagnostics ----------------------------------------------------------
option(N4M_ENABLE_ASAN             "Enable AddressSanitizer (clang/gcc only)"            OFF)
option(N4M_ENABLE_UBSAN            "Enable UndefinedBehaviorSanitizer"                   OFF)
option(N4M_ENABLE_TSAN             "Enable ThreadSanitizer (mutually exclusive with ASAN)" OFF)
option(N4M_ENABLE_COVERAGE         "Enable gcov / llvm-cov coverage instrumentation"     OFF)
option(N4M_ENABLE_LTO              "Enable Link-Time Optimisation (only if toolchain supports IPO)" OFF)
option(N4M_ENABLE_FAULT_INJECTION  "Expose test-only hooks to inject bad_alloc, etc."    OFF)
option(N4M_WARNINGS_AS_ERRORS      "Treat compile warnings as errors (CI default)"       OFF)

# ---- Option validation ----------------------------------------------------
if(N4M_ENABLE_ASAN AND N4M_ENABLE_TSAN)
    message(FATAL_ERROR
        "N4M_ENABLE_ASAN and N4M_ENABLE_TSAN are mutually exclusive — "
        "TSAN cannot run under AddressSanitizer.")
endif()

if(N4M_BUILD_FUZZ AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang")
    message(WARNING
        "N4M_BUILD_FUZZ requires clang for libFuzzer support; "
        "disabling fuzz targets for this build.")
    set(N4M_BUILD_FUZZ OFF CACHE BOOL "" FORCE)
endif()

# LTO is opt-in: only applied when N4M_ENABLE_LTO=ON AND the toolchain
# supports IPO AND we are not in a Debug build. Default is OFF.
if(N4M_ENABLE_LTO AND NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    include(CheckIPOSupported)
    check_ipo_supported(RESULT _ipo_supported OUTPUT _ipo_msg LANGUAGES C CXX)
    if(_ipo_supported)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    else()
        message(WARNING
            "N4M_ENABLE_LTO=ON but the toolchain does not support IPO: ${_ipo_msg}. "
            "Continuing without LTO.")
    endif()
endif()
