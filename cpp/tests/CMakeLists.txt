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
    test_aom_operator_pls_stack.cpp
    test_aom_ridge_blender.cpp
    test_aom_robust_hpo.cpp
    test_aom_selection.cpp
    test_aom_sweep.cpp
    test_fck.cpp
    test_filter_x_outlier.cpp
    test_filters_meta.cpp
    test_filters_y.cpp
    test_metrics.cpp
    test_models_pls.cpp
    test_models_extra.cpp
    test_moments.cpp
    test_ridge.cpp
    test_sweep.cpp
    test_c_abi_memory.cpp
    test_threading.cpp
    test_preprocessing_baselines.cpp
    test_preprocessing_feature_selection.cpp
    test_spectral_encoding.cpp
    test_preprocessing_orthogonalization.cpp
    test_preprocessing_resampling.cpp
    test_preprocessing_signal_conversion.cpp
    test_preprocessing_smoothing.cpp
    test_preprocessing_stateful.cpp
    test_preprocessing_stateless.cpp
    test_preprocessing_wavelets.cpp
    test_rng_pcg64.cpp
    test_rng_mt_r.cpp
    test_rng_engine.cpp
    # The RNG engines are internal core modules (hidden visibility in libn4m),
    # so the parity/golden tests compile them directly rather than linking
    # through n4m_c. rng_engine pulls pcg64 + ziggurat (its PCG64 dispatch arm).
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/core/common/rng_mt_r.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/core/common/rng_numpy_mt.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/core/common/rng_engine.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/core/common/rng_pcg64.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/core/common/ziggurat.c
    test_signal_type.cpp
    test_splitters.cpp
    test_optimization.cpp
    include_check_optimization.cpp
    include_check_optimization.c
    test_transfer_metrics.cpp)
target_link_libraries(n4m_tests PRIVATE n4m_c)
target_include_directories(n4m_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/cpp/src
    ${PROJECT_BINARY_DIR}/generated)
# Pass the parity fixture directory so test_rng_pcg64.cpp can load the JSON
# without relying on a working directory. Quote-wrapping survives the CMake
# generator and ends up as a stringised C++ token at compile time.
target_compile_definitions(n4m_tests PRIVATE
    N4M_PARITY_FIXTURE_DIR="${PROJECT_SOURCE_DIR}/parity/fixtures"
    $<$<BOOL:${N4M_ENABLE_UBSAN}>:N4M_UBSAN_BUILD=1>)
n4m_add_warnings(n4m_tests)

if(WIN32)
    add_custom_command(TARGET n4m_tests POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_RUNTIME_DLLS:n4m_tests>
            $<TARGET_FILE_DIR:n4m_tests>
        COMMAND_EXPAND_LISTS)
endif()

add_test(NAME n4m_tests COMMAND n4m_tests)
add_executable(n4m_estimator_tests test_estimator.cpp)
target_link_libraries(n4m_estimator_tests PRIVATE n4m_c)
add_test(NAME n4m_estimator_tests COMMAND n4m_estimator_tests)
add_executable(n4m_role_pipeline_tests test_role_pipeline.cpp)
target_link_libraries(n4m_role_pipeline_tests PRIVATE n4m_c)
add_test(NAME n4m_role_pipeline_tests COMMAND n4m_role_pipeline_tests)
add_executable(n4m_multimodal_pipeline_tests test_multimodal_pipeline.cpp)
target_link_libraries(n4m_multimodal_pipeline_tests PRIVATE n4m_c)
n4m_add_warnings(n4m_multimodal_pipeline_tests)
add_test(NAME n4m_multimodal_pipeline_tests COMMAND n4m_multimodal_pipeline_tests)
set_tests_properties(n4m_multimodal_pipeline_tests PROPERTIES LABELS "abi;parity" TIMEOUT 30)
add_executable(n4m_multimodal_classifier_tests test_multimodal_classifier.cpp)
target_link_libraries(n4m_multimodal_classifier_tests PRIVATE n4m_c)
target_include_directories(n4m_multimodal_classifier_tests PRIVATE ${CMAKE_SOURCE_DIR}/cpp/src)
n4m_add_warnings(n4m_multimodal_classifier_tests)
add_test(NAME n4m_multimodal_classifier_tests COMMAND n4m_multimodal_classifier_tests)
set_tests_properties(n4m_multimodal_classifier_tests PROPERTIES LABELS "abi;parity;classification" TIMEOUT 60)
# Procedure results are compared field by field through the internal
# MethodResult layout.
add_executable(n4m_procedure_tests test_procedure.cpp)
target_link_libraries(n4m_procedure_tests PRIVATE n4m_c)
target_include_directories(n4m_procedure_tests PRIVATE ${CMAKE_SOURCE_DIR}/cpp/src)
add_test(NAME n4m_procedure_tests COMMAND n4m_procedure_tests)
set_tests_properties(n4m_tests PROPERTIES
    LABELS "smoke;abi;parity"
    TIMEOUT 60)

# Focused N4MM v2 model-pipeline contract. Kept separate so format changes can
# be gated without rerunning the complete native suite.
add_executable(n4m_model_pipeline_tests test_model_pipeline.cpp)
target_link_libraries(n4m_model_pipeline_tests PRIVATE n4m_c)
target_include_directories(n4m_model_pipeline_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/cpp/src
    ${PROJECT_BINARY_DIR}/generated)
n4m_add_warnings(n4m_model_pipeline_tests)
add_test(NAME n4m_model_pipeline_tests COMMAND n4m_model_pipeline_tests)
set_tests_properties(n4m_model_pipeline_tests PROPERTIES
    LABELS "abi;parity"
    TIMEOUT 30)

# Standalone fitted N4MP preprocessing wire; distinct from model N4MM.
add_executable(n4m_preprocess_wire_tests test_preprocess_wire.cpp)
target_link_libraries(n4m_preprocess_wire_tests PRIVATE n4m_c)
target_include_directories(n4m_preprocess_wire_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/cpp/src
    ${PROJECT_BINARY_DIR}/generated)
n4m_add_warnings(n4m_preprocess_wire_tests)
add_test(NAME n4m_preprocess_wire_tests COMMAND n4m_preprocess_wire_tests)
set_tests_properties(n4m_preprocess_wire_tests PROPERTIES LABELS "abi;parity" TIMEOUT 30)

# Focused additive ABI inspector contract. It constructs minimal canonical
# N4MM bytes directly, so this gate does not rerun model pipeline parity.
add_executable(n4m_serialized_pipeline_inspect_tests
    test_serialized_pipeline_inspect.cpp)
target_link_libraries(n4m_serialized_pipeline_inspect_tests PRIVATE n4m_c)
target_include_directories(n4m_serialized_pipeline_inspect_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/cpp/src
    ${PROJECT_BINARY_DIR}/generated)
n4m_add_warnings(n4m_serialized_pipeline_inspect_tests)
add_test(NAME n4m_serialized_pipeline_inspect_tests
    COMMAND n4m_serialized_pipeline_inspect_tests)
set_tests_properties(n4m_serialized_pipeline_inspect_tests PROPERTIES
    LABELS "abi"
    TIMEOUT 30)

# Internal-primitive tests. The PCG64 bounded-integer + choice-without-
# replacement primitives are hidden in libn4m, so they are unreachable from
# n4m_tests (which links the shared lib). Link the static archive instead — a
# static link resolves hidden symbols and (per n4m_targets.cmake) already
# carries the optional OpenMP/BLAS/CUDA deps.
if(TARGET n4m_c_static)
    add_executable(n4m_internal_tests
        test_internal_rng_choice.cpp
        test_uve_r_exact.cpp
        test_internal_cv.cpp
        test_internal_optimization.cpp
        test_internal_sweep.cpp
        test_internal_linalg.cpp)
    target_link_libraries(n4m_internal_tests PRIVATE n4m_c_static)
    target_include_directories(n4m_internal_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/cpp/src
        ${PROJECT_BINARY_DIR}/generated)
    target_compile_definitions(n4m_internal_tests PRIVATE
        $<$<BOOL:${N4M_WITH_CUDA}>:N4M_USE_CUDA=1>)
    n4m_add_warnings(n4m_internal_tests)
    add_test(NAME n4m_internal_tests COMMAND n4m_internal_tests)
    set_tests_properties(n4m_internal_tests PROPERTIES
        LABELS "parity"
        TIMEOUT 60)
endif()

# Existing public build-info ABI must report the numerical core's actual flags.
add_executable(n4m_build_capabilities_tests test_build_capabilities.cpp)
target_link_libraries(n4m_build_capabilities_tests PRIVATE n4m_c)
target_include_directories(n4m_build_capabilities_tests PRIVATE ${CMAKE_SOURCE_DIR}/cpp/src)
target_compile_definitions(n4m_build_capabilities_tests PRIVATE
    $<TARGET_PROPERTY:n4m_core,COMPILE_DEFINITIONS>)
n4m_add_warnings(n4m_build_capabilities_tests)
add_test(NAME n4m_build_capabilities_tests COMMAND n4m_build_capabilities_tests)
set_tests_properties(n4m_build_capabilities_tests PROPERTIES LABELS "abi" TIMEOUT 30)

# Compile only the metadata TU for every backend combination. This proves
# compiled CUDA is reported even on hosts without a toolkit or visible GPU;
# these executables neither link nor run numerical acceleration libraries.
foreach(_profile RANGE 0 7)
    set(_target n4m_build_profile_${_profile}_tests)
    add_executable(${_target} test_build_capabilities.cpp ../src/core/common/version.cpp)
    target_include_directories(${_target} PRIVATE
        ${CMAKE_SOURCE_DIR}/cpp/include ${CMAKE_SOURCE_DIR}/cpp/src
        ${PROJECT_BINARY_DIR}/generated)
    target_compile_definitions(${_target} PRIVATE
        N4M_TEST_BUILD_INFO_CORE=1 N4M_BUILD_INFO="profile matrix descriptive text")
    math(EXPR _blas "${_profile} & 1")
    math(EXPR _openmp "${_profile} & 2")
    math(EXPR _cuda "${_profile} & 4")
    if(_blas)
        target_compile_definitions(${_target} PRIVATE N4M_USE_BLAS=1)
    endif()
    if(_openmp)
        target_compile_definitions(${_target} PRIVATE N4M_USE_OPENMP=1)
    endif()
    if(_cuda)
        target_compile_definitions(${_target} PRIVATE N4M_USE_CUDA=1)
    endif()
    n4m_add_warnings(${_target})
    add_test(NAME ${_target} COMMAND ${_target})
    set_tests_properties(${_target} PROPERTIES LABELS "abi" TIMEOUT 30)
endforeach()

# A real externally supplied compiler flag enables _OPENMP even when the
# configured N4M OpenMP backend is disabled. Compile the metadata TU only:
# no numerical acceleration is linked or executed by this witness.
include(CheckCXXCompilerFlag)
if(MSVC)
    set(_external_openmp_flag "/openmp")
else()
    set(_external_openmp_flag "-fopenmp")
endif()
check_cxx_compiler_flag("${_external_openmp_flag}" N4M_TEST_HAS_EXTERNAL_OPENMP_FLAG)
if(N4M_TEST_HAS_EXTERNAL_OPENMP_FLAG)
    add_executable(n4m_build_external_openmp_tests
        test_build_capabilities.cpp ../src/core/common/version.cpp)
    target_include_directories(n4m_build_external_openmp_tests PRIVATE
        ${CMAKE_SOURCE_DIR}/cpp/include ${CMAKE_SOURCE_DIR}/cpp/src
        ${PROJECT_BINARY_DIR}/generated)
    target_compile_definitions(n4m_build_external_openmp_tests PRIVATE
        N4M_TEST_BUILD_INFO_CORE=1 N4M_TEST_REQUIRE_EXTERNAL_OPENMP=1)
    target_compile_options(n4m_build_external_openmp_tests PRIVATE "${_external_openmp_flag}")
    n4m_add_warnings(n4m_build_external_openmp_tests)
    add_test(NAME n4m_build_external_openmp_tests COMMAND n4m_build_external_openmp_tests)
    set_tests_properties(n4m_build_external_openmp_tests PROPERTIES LABELS "abi" TIMEOUT 30)
else()
    message(STATUS "External compiler OpenMP capability witness unavailable on this toolchain")
endif()

# Explicit test instrumentation only. Product libn4m sources/exports are untouched.
# ctypes releases the GIL across each CDLL probe and the real linked native FIT.
add_library(n4m_fit_witness SHARED test_native_fit_witness.cpp)
target_link_libraries(n4m_fit_witness PRIVATE n4m_c)
target_compile_features(n4m_fit_witness PRIVATE cxx_std_17)
set_target_properties(n4m_fit_witness PROPERTIES
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
n4m_add_warnings(n4m_fit_witness)
