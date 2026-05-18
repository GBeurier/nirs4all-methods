# cmake/SanitizersToolchain.cmake
# Adds the sanitizer flags requested via CHEMOMETRICS4ALL_ENABLE_{ASAN,UBSAN,TSAN}.
# Coverage instrumentation is also wired here for the same reason — it shares
# the "global compile/link options applied to every chemometrics4all target" pattern.

if(DEFINED _CHEMOMETRICS4ALL_SANITIZERS_INCLUDED)
    return()
endif()
set(_CHEMOMETRICS4ALL_SANITIZERS_INCLUDED ON)

set(_chemometrics4all_san_flags "")
set(_chemometrics4all_san_link  "")

if(CHEMOMETRICS4ALL_ENABLE_ASAN)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        list(APPEND _chemometrics4all_san_flags
            -fsanitize=address
            -fno-omit-frame-pointer
            -fno-optimize-sibling-calls)
        list(APPEND _chemometrics4all_san_link -fsanitize=address)
    else()
        message(WARNING "CHEMOMETRICS4ALL_ENABLE_ASAN: AddressSanitizer not supported on ${CMAKE_CXX_COMPILER_ID}.")
    endif()
endif()

if(CHEMOMETRICS4ALL_ENABLE_UBSAN)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        list(APPEND _chemometrics4all_san_flags
            -fsanitize=undefined
            -fno-sanitize-recover=all
            -fno-omit-frame-pointer)
        list(APPEND _chemometrics4all_san_link -fsanitize=undefined)
    else()
        message(WARNING "CHEMOMETRICS4ALL_ENABLE_UBSAN: UndefinedBehaviorSanitizer not supported on ${CMAKE_CXX_COMPILER_ID}.")
    endif()
endif()

if(CHEMOMETRICS4ALL_ENABLE_TSAN)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        list(APPEND _chemometrics4all_san_flags
            -fsanitize=thread
            -fno-omit-frame-pointer)
        list(APPEND _chemometrics4all_san_link -fsanitize=thread)
    else()
        message(WARNING "CHEMOMETRICS4ALL_ENABLE_TSAN: ThreadSanitizer not supported on ${CMAKE_CXX_COMPILER_ID}.")
    endif()
endif()

if(CHEMOMETRICS4ALL_ENABLE_COVERAGE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        list(APPEND _chemometrics4all_san_flags --coverage -fprofile-arcs -ftest-coverage)
        list(APPEND _chemometrics4all_san_link  --coverage)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang")
        list(APPEND _chemometrics4all_san_flags -fprofile-instr-generate -fcoverage-mapping)
        list(APPEND _chemometrics4all_san_link  -fprofile-instr-generate)
    else()
        message(WARNING "CHEMOMETRICS4ALL_ENABLE_COVERAGE: not supported on ${CMAKE_CXX_COMPILER_ID}.")
    endif()
endif()

if(_chemometrics4all_san_flags)
    add_compile_options(${_chemometrics4all_san_flags})
endif()
if(_chemometrics4all_san_link)
    add_link_options(${_chemometrics4all_san_link})
endif()
