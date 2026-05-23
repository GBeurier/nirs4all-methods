# cmake/SanitizersToolchain.cmake
# Adds the sanitizer flags requested via N4M_ENABLE_{ASAN,UBSAN,TSAN}.
# Coverage instrumentation is also wired here for the same reason — it shares
# the "global compile/link options applied to every n4m target" pattern.

if(DEFINED _N4M_SANITIZERS_INCLUDED)
    return()
endif()
set(_N4M_SANITIZERS_INCLUDED ON)

set(_n4m_san_flags "")
set(_n4m_san_link  "")

if(N4M_ENABLE_ASAN)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        list(APPEND _n4m_san_flags
            -fsanitize=address
            -fno-omit-frame-pointer
            -fno-optimize-sibling-calls)
        list(APPEND _n4m_san_link -fsanitize=address)
    else()
        message(WARNING "N4M_ENABLE_ASAN: AddressSanitizer not supported on ${CMAKE_CXX_COMPILER_ID}.")
    endif()
endif()

if(N4M_ENABLE_UBSAN)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        list(APPEND _n4m_san_flags
            -fsanitize=undefined
            -fno-sanitize-recover=all
            -fno-omit-frame-pointer)
        list(APPEND _n4m_san_link -fsanitize=undefined)
    else()
        message(WARNING "N4M_ENABLE_UBSAN: UndefinedBehaviorSanitizer not supported on ${CMAKE_CXX_COMPILER_ID}.")
    endif()
endif()

if(N4M_ENABLE_TSAN)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        list(APPEND _n4m_san_flags
            -fsanitize=thread
            -fno-omit-frame-pointer)
        list(APPEND _n4m_san_link -fsanitize=thread)
    else()
        message(WARNING "N4M_ENABLE_TSAN: ThreadSanitizer not supported on ${CMAKE_CXX_COMPILER_ID}.")
    endif()
endif()

if(N4M_ENABLE_COVERAGE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        list(APPEND _n4m_san_flags --coverage -fprofile-arcs -ftest-coverage)
        list(APPEND _n4m_san_link  --coverage)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang")
        list(APPEND _n4m_san_flags -fprofile-instr-generate -fcoverage-mapping)
        list(APPEND _n4m_san_link  -fprofile-instr-generate)
    else()
        message(WARNING "N4M_ENABLE_COVERAGE: not supported on ${CMAKE_CXX_COMPILER_ID}.")
    endif()
endif()

if(_n4m_san_flags)
    add_compile_options(${_n4m_san_flags})
endif()
if(_n4m_san_link)
    add_link_options(${_n4m_san_link})
endif()
