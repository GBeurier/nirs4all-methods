# cpp/tests/CMakeLists.txt — Phase 0 test runner.
#
# Hand-rolled zero-dependency harness in harness.hpp + main.cpp.
# Each subsequent phase appends its test files to the source list.

add_executable(n4m_tests
    main.cpp
    test_augmenters_noise_drift.cpp
    test_augmenters_phase17.cpp
    test_augmenters_edge_splines_random.cpp
    test_augmenters_wavelength_spectral.cpp
    test_fck.cpp
    test_filter_x_outlier.cpp
    test_filters_meta.cpp
    test_filters_y.cpp
    test_metrics.cpp
    test_preprocessing_baselines.cpp
    test_preprocessing_feature_selection.cpp
    test_preprocessing_orthogonalization.cpp
    test_preprocessing_resampling.cpp
    test_preprocessing_signal_conversion.cpp
    test_preprocessing_smoothing.cpp
    test_preprocessing_stateful.cpp
    test_preprocessing_stateless.cpp
    test_preprocessing_wavelets.cpp
    test_rng_pcg64.cpp
    test_signal_type.cpp
    test_splitters.cpp
    test_transfer_metrics.cpp)
target_link_libraries(n4m_tests PRIVATE n4m_c)
target_include_directories(n4m_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_BINARY_DIR}/generated)
# Pass the parity fixture directory so test_rng_pcg64.cpp can load the JSON
# without relying on a working directory. Quote-wrapping survives the CMake
# generator and ends up as a stringised C++ token at compile time.
target_compile_definitions(n4m_tests PRIVATE
    N4M_PARITY_FIXTURE_DIR="${PROJECT_SOURCE_DIR}/parity/fixtures")
pls4all_add_warnings(n4m_tests)

if(WIN32)
    add_custom_command(TARGET n4m_tests POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_RUNTIME_DLLS:n4m_tests>
            $<TARGET_FILE_DIR:n4m_tests>
        COMMAND_EXPAND_LISTS)
endif()

add_test(NAME n4m_tests COMMAND n4m_tests)
set_tests_properties(n4m_tests PROPERTIES
    LABELS "smoke;abi;parity"
    TIMEOUT 60)
