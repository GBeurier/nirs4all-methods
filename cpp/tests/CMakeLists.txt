# cpp/tests/CMakeLists.txt — Phase 0 test runner.
#
# Hand-rolled zero-dependency harness in harness.hpp + main.cpp.
# Each subsequent phase appends its test files to the source list.

add_executable(chemometrics4all_tests
    main.cpp
    test_rng_pcg64.cpp
    test_preprocessing_stateless.cpp
    test_preprocessing_stateful.cpp
    test_preprocessing_smoothing.cpp
    test_preprocessing_baselines.cpp
    test_preprocessing_signal_conversion.cpp
    test_preprocessing_orthogonalization.cpp
    test_preprocessing_feature_selection.cpp
    test_preprocessing_resampling.cpp
    test_splitters.cpp
    test_filters_y.cpp
    test_filters_meta.cpp
    test_signal_type.cpp
    test_metrics.cpp
    test_transfer_metrics.cpp
    test_fck.cpp)
target_link_libraries(chemometrics4all_tests PRIVATE chemometrics4all_c)
target_include_directories(chemometrics4all_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_BINARY_DIR}/generated)
# Pass the parity fixture directory so test_rng_pcg64.cpp can load the JSON
# without relying on a working directory. Quote-wrapping survives the CMake
# generator and ends up as a stringised C++ token at compile time.
target_compile_definitions(chemometrics4all_tests PRIVATE
    C4A_PARITY_FIXTURE_DIR="${PROJECT_SOURCE_DIR}/parity/fixtures")
chemometrics4all_add_warnings(chemometrics4all_tests)

add_test(NAME chemometrics4all_tests COMMAND chemometrics4all_tests)
set_tests_properties(chemometrics4all_tests PROPERTIES
    LABELS "smoke;abi;parity"
    TIMEOUT 60)
