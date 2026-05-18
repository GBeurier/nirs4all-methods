# cpp/tests/CMakeLists.txt — Phase 0 test runner.
#
# Hand-rolled zero-dependency harness in harness.hpp + main.cpp.
# Each subsequent phase appends its test files to the source list.

add_executable(chemometrics4all_tests main.cpp)
target_link_libraries(chemometrics4all_tests PRIVATE chemometrics4all_c)
target_include_directories(chemometrics4all_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_BINARY_DIR}/generated)
chemometrics4all_add_warnings(chemometrics4all_tests)

add_test(NAME chemometrics4all_tests COMMAND chemometrics4all_tests)
set_tests_properties(chemometrics4all_tests PROPERTIES
    LABELS "smoke;abi"
    TIMEOUT 30)
