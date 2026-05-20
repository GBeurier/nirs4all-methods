# cmake/CompilerWarnings.cmake
# Provides chemometrics4all_add_warnings(<target>) — applies the canonical warning set.

if(DEFINED _CHEMOMETRICS4ALL_COMPILER_WARNINGS_INCLUDED)
    return()
endif()
set(_CHEMOMETRICS4ALL_COMPILER_WARNINGS_INCLUDED ON)

# Common (C + C++) warning flags.
set(_chemometrics4all_gnu_clang_warnings_common
    -Wall
    -Wextra
    -Wpedantic
    -Wcast-align
    -Wcast-qual
    -Wconversion
    -Wdouble-promotion
    -Wformat=2
    -Wmissing-declarations
    -Wmissing-include-dirs
    -Wnull-dereference
    -Wshadow
    -Wsign-conversion
    -Wundef
    -Wuninitialized
    -Wunused
)

# C++-only warning flags. Wrapped in $<COMPILE_LANGUAGE:CXX> so they do not
# trigger "command-line option '-W*' is valid for C++/ObjC++ but not for C"
# diagnostics when chemometrics4all_core compiles its C TUs (rng_pcg64.c,
# ziggurat.c, future BLAS/banded solvers, etc.).
set(_chemometrics4all_gnu_clang_warnings_cxx_only
    -Wnon-virtual-dtor
    -Wold-style-cast
    -Woverloaded-virtual
    -Wzero-as-null-pointer-constant
)

# Selectively disable a few warnings that are unreasonably noisy on
# generated code (CMake-configured headers) or in C-friendly hot paths.
set(_chemometrics4all_gnu_clang_disables
    -Wno-unknown-pragmas
)

set(_chemometrics4all_msvc_warnings
    /W4
    /permissive-
    /Zc:__cplusplus
    /Zc:preprocessor
    /utf-8
    /wd4068  # unknown pragma — equivalent to -Wno-unknown-pragmas
    /wd4324  # struct padded due to alignas — expected in matrix view
)

function(chemometrics4all_add_warnings target)
    if(NOT TARGET ${target})
        message(FATAL_ERROR "chemometrics4all_add_warnings: '${target}' is not a target.")
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(${target} PRIVATE
            "$<$<COMPILE_LANGUAGE:C,CXX>:${_chemometrics4all_gnu_clang_warnings_common}>"
            "$<$<COMPILE_LANGUAGE:CXX>:${_chemometrics4all_gnu_clang_warnings_cxx_only}>"
            "$<$<COMPILE_LANGUAGE:C,CXX>:${_chemometrics4all_gnu_clang_disables}>")
    elseif(MSVC)
        target_compile_options(${target} PRIVATE ${_chemometrics4all_msvc_warnings})
    endif()

    if(CHEMOMETRICS4ALL_WARNINGS_AS_ERRORS)
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
            target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:C,CXX>:-Werror>")
        elseif(MSVC)
            target_compile_options(${target} PRIVATE /WX)
        endif()
    endif()
endfunction()
