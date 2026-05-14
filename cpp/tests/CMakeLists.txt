# cpp/tests/CMakeLists.txt — Phase 0 test runner.
#
# Uses a hand-rolled zero-dependency harness (cpp/tests/harness.hpp +
# main.cpp) so we honour the zero-mandatory-deps promise. doctest can be
# vendored in a future PR if richer ergonomics become worth the policy
# departure.

add_executable(pls4all_tests
    main.cpp
    abi/test_lifecycle.cpp
    abi/test_classification_extensions.cpp
    abi/test_classification_metrics.cpp
    abi/test_component_coefficients.cpp
    abi/test_config_setters.cpp
    abi/test_cross_validation.cpp
    abi/test_status_codes.cpp
    abi/test_variable_importance.cpp
    abi/test_version.cpp
    abi/test_matrix_view.cpp
    abi/test_error_messages.cpp
    abi/test_oom_paths.cpp
    abi/test_metrics.cpp
    abi/test_model_selection.cpp
    abi/test_pipeline.cpp
    abi/test_pls_lda.cpp
    abi/test_pls_logistic.cpp
    abi/test_validation.cpp
    abi/test_validation_advanced.cpp
    abi/test_model_stubs.cpp)
target_link_libraries(pls4all_tests PRIVATE pls4all_c pls4all_core)
target_include_directories(pls4all_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/cpp/src
        ${PROJECT_BINARY_DIR}/generated)
pls4all_add_warnings(pls4all_tests)

if(WIN32 AND TARGET pls4all_c)
    add_custom_command(TARGET pls4all_tests POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:pls4all_c>
                $<TARGET_FILE_DIR:pls4all_tests>)
endif()

add_test(NAME pls4all_tests
         COMMAND pls4all_tests)
set_tests_properties(pls4all_tests PROPERTIES TIMEOUT 60)
