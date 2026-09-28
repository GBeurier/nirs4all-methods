# Bounded N4MM hostile-input harness. Link the fuzzer to the instrumented
# static C ABI so coverage from inspectors and the importer reaches libFuzzer.

if(NOT TARGET n4m_c_static)
    message(FATAL_ERROR "N4M_BUILD_FUZZ requires N4M_BUILD_STATIC=ON")
endif()

add_library(n4m_fuzz_n4mm_driver OBJECT fuzz_n4mm.cpp)
target_include_directories(n4m_fuzz_n4mm_driver PRIVATE
    ${CMAKE_SOURCE_DIR}/cpp/include
    ${PROJECT_BINARY_DIR}/generated)
n4m_add_warnings(n4m_fuzz_n4mm_driver)

add_executable(n4m_fuzz_n4mm_seed generate_seed.cpp)
target_link_libraries(n4m_fuzz_n4mm_seed PRIVATE n4m_c_static)
target_compile_definitions(n4m_fuzz_n4mm_seed PRIVATE N4M_STATIC=1)
target_include_directories(n4m_fuzz_n4mm_seed PRIVATE
    ${CMAKE_SOURCE_DIR}/cpp/include
    ${PROJECT_BINARY_DIR}/generated)
n4m_add_warnings(n4m_fuzz_n4mm_seed)

if(N4M_BUILD_TESTS)
    add_executable(n4m_fuzz_n4mm_smoke
        fuzz_n4mm_smoke.cpp
        $<TARGET_OBJECTS:n4m_fuzz_n4mm_driver>)
    target_link_libraries(n4m_fuzz_n4mm_smoke PRIVATE n4m_c_static)
    target_compile_definitions(n4m_fuzz_n4mm_smoke PRIVATE N4M_STATIC=1)
    target_include_directories(n4m_fuzz_n4mm_smoke PRIVATE
        ${CMAKE_SOURCE_DIR}/cpp/include
        ${PROJECT_BINARY_DIR}/generated)
    n4m_add_warnings(n4m_fuzz_n4mm_smoke)
    add_test(NAME n4m_fuzz_n4mm_smoke COMMAND n4m_fuzz_n4mm_smoke
        "${CMAKE_SOURCE_DIR}/cpp/fuzz/corpus/n4mm/valid-affine-1.n4mm"
        "${CMAKE_SOURCE_DIR}/cpp/fuzz/corpus/n4mm/valid-affine-2.n4mm")
    set_tests_properties(n4m_fuzz_n4mm_smoke PROPERTIES
        LABELS "abi;security"
        TIMEOUT 10)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang")
    target_compile_options(n4m_core PRIVATE
        $<$<COMPILE_LANGUAGE:C,CXX>:-fsanitize=fuzzer-no-link>)
    target_compile_options(n4m_c_static PRIVATE
        $<$<COMPILE_LANGUAGE:C,CXX>:-fsanitize=fuzzer-no-link>)
    add_executable(n4m_fuzz_n4mm fuzz_n4mm.cpp)
    target_link_libraries(n4m_fuzz_n4mm PRIVATE n4m_c_static)
    target_compile_definitions(n4m_fuzz_n4mm PRIVATE N4M_STATIC=1)
    target_include_directories(n4m_fuzz_n4mm PRIVATE
        ${CMAKE_SOURCE_DIR}/cpp/include
        ${PROJECT_BINARY_DIR}/generated)
    target_compile_options(n4m_fuzz_n4mm PRIVATE -fsanitize=fuzzer-no-link)
    target_link_options(n4m_fuzz_n4mm PRIVATE -fsanitize=fuzzer)
    n4m_add_warnings(n4m_fuzz_n4mm)
endif()
