# cmake/SanitizersToolchain.cmake
# Adds the sanitizer flags requested via PLS4ALL_ENABLE_{ASAN,UBSAN,TSAN}.
# Coverage instrumentation is also wired here for the same reason — it shares
# the "global compile/link options applied to every pls4all target" pattern.

if(DEFINED _PLS4ALL_SANITIZERS_INCLUDED)
    return()
endif()
set(_PLS4ALL_SANITIZERS_INCLUDED ON)

set(_pls4all_san_flags "")
set(_pls4all_san_link  "")

if(PLS4ALL_ENABLE_ASAN)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        list(APPEND _pls4all_san_flags
            -fsanitize=address
            -fno-omit-frame-pointer
            -fno-optimize-sibling-calls)
        list(APPEND _pls4all_san_link -fsanitize=address)
    else()
        message(WARNING "PLS4ALL_ENABLE_ASAN: AddressSanitizer not supported on ${CMAKE_CXX_COMPILER_ID}.")
    endif()
endif()

if(PLS4ALL_ENABLE_UBSAN)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        list(APPEND _pls4all_san_flags
            -fsanitize=undefined
            -fno-sanitize-recover=all
            -fno-omit-frame-pointer)
        list(APPEND _pls4all_san_link -fsanitize=undefined)
    else()
        message(WARNING "PLS4ALL_ENABLE_UBSAN: UndefinedBehaviorSanitizer not supported on ${CMAKE_CXX_COMPILER_ID}.")
    endif()
endif()

if(PLS4ALL_ENABLE_TSAN)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        list(APPEND _pls4all_san_flags
            -fsanitize=thread
            -fno-omit-frame-pointer)
        list(APPEND _pls4all_san_link -fsanitize=thread)
    else()
        message(WARNING "PLS4ALL_ENABLE_TSAN: ThreadSanitizer not supported on ${CMAKE_CXX_COMPILER_ID}.")
    endif()
endif()

if(PLS4ALL_ENABLE_COVERAGE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        list(APPEND _pls4all_san_flags --coverage -fprofile-arcs -ftest-coverage)
        list(APPEND _pls4all_san_link  --coverage)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang")
        list(APPEND _pls4all_san_flags -fprofile-instr-generate -fcoverage-mapping)
        list(APPEND _pls4all_san_link  -fprofile-instr-generate)
    else()
        message(WARNING "PLS4ALL_ENABLE_COVERAGE: not supported on ${CMAKE_CXX_COMPILER_ID}.")
    endif()
endif()

if(_pls4all_san_flags)
    add_compile_options(${_pls4all_san_flags})
endif()
if(_pls4all_san_link)
    add_link_options(${_pls4all_san_link})
endif()
