# cmake/CompilerWarnings.cmake
# Provides n4m_add_warnings(<target>) — applies the canonical warning set.

if(DEFINED _N4M_COMPILER_WARNINGS_INCLUDED)
    return()
endif()
set(_N4M_COMPILER_WARNINGS_INCLUDED ON)

set(_n4m_gnu_clang_warnings
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
    -Wnon-virtual-dtor
    -Wnull-dereference
    -Wold-style-cast
    -Woverloaded-virtual
    -Wshadow
    -Wsign-conversion
    -Wundef
    -Wuninitialized
    -Wunused
    -Wzero-as-null-pointer-constant
)

# Selectively disable a few warnings that are unreasonably noisy on
# generated code (CMake-configured headers) or in C-friendly hot paths.
set(_n4m_gnu_clang_disables
    -Wno-unknown-pragmas
)

set(_n4m_msvc_warnings
    /W4
    /permissive-
    /Zc:__cplusplus
    /Zc:preprocessor
    /utf-8
    /wd4068  # unknown pragma — equivalent to -Wno-unknown-pragmas
    /wd4324  # struct padded due to alignas — expected in matrix view
)

function(n4m_add_warnings target)
    if(NOT TARGET ${target})
        message(FATAL_ERROR "n4m_add_warnings: '${target}' is not a target.")
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(${target} PRIVATE
            ${_n4m_gnu_clang_warnings}
            ${_n4m_gnu_clang_disables})
    elseif(MSVC)
        target_compile_options(${target} PRIVATE ${_n4m_msvc_warnings})
    endif()

    if(N4M_WARNINGS_AS_ERRORS)
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
            target_compile_options(${target} PRIVATE -Werror)
        elseif(MSVC)
            target_compile_options(${target} PRIVATE /WX)
        endif()
    endif()
endfunction()
