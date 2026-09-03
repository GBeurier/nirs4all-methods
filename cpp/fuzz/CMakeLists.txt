# Bounded N4MM hostile-input harness. The portable smoke target keeps the
# public-ABI call contract buildable without requiring a local libFuzzer.

add_library(n4m_fuzz_n4mm_driver OBJECT fuzz_n4mm.cpp)
target_include_directories(n4m_fuzz_n4mm_driver PRIVATE
    ${CMAKE_SOURCE_DIR}/cpp/include
    ${PROJECT_BINARY_DIR}/generated)
n4m_add_warnings(n4m_fuzz_n4mm_driver)

add_executable(n4m_fuzz_n4mm_smoke
    fuzz_n4mm_smoke.cpp
    $<TARGET_OBJECTS:n4m_fuzz_n4mm_driver>)
target_link_libraries(n4m_fuzz_n4mm_smoke PRIVATE n4m_c)
target_include_directories(n4m_fuzz_n4mm_smoke PRIVATE
    ${CMAKE_SOURCE_DIR}/cpp/include
    ${PROJECT_BINARY_DIR}/generated)
n4m_add_warnings(n4m_fuzz_n4mm_smoke)

if(N4M_BUILD_TESTS)
    add_test(NAME n4m_fuzz_n4mm_smoke COMMAND n4m_fuzz_n4mm_smoke)
    set_tests_properties(n4m_fuzz_n4mm_smoke PROPERTIES
        LABELS "abi;security"
        TIMEOUT 10)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang")
    add_executable(n4m_fuzz_n4mm fuzz_n4mm.cpp)
    target_link_libraries(n4m_fuzz_n4mm PRIVATE n4m_c)
    target_include_directories(n4m_fuzz_n4mm PRIVATE
        ${CMAKE_SOURCE_DIR}/cpp/include
        ${PROJECT_BINARY_DIR}/generated)
    target_compile_options(n4m_fuzz_n4mm PRIVATE -fsanitize=fuzzer-no-link)
    target_link_options(n4m_fuzz_n4mm PRIVATE -fsanitize=fuzzer)
    n4m_add_warnings(n4m_fuzz_n4mm)
endif()
