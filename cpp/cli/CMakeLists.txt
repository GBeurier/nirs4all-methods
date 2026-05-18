# cpp/cli/CMakeLists.txt — chemometrics4all_cli executable. Links libc4a; the CLI
# itself contains no algorithm logic, only ABI introspection.

add_executable(chemometrics4all_cli c4a_cli.cpp)
target_link_libraries(chemometrics4all_cli PRIVATE chemometrics4all_c)
target_include_directories(chemometrics4all_cli PRIVATE ${PROJECT_BINARY_DIR}/generated)
chemometrics4all_add_warnings(chemometrics4all_cli)
set_target_properties(chemometrics4all_cli PROPERTIES OUTPUT_NAME chemometrics4all_cli)

# The selfcheck uses `#if defined(C4A_USE_CUDA)` to pick its expected
# return value from c4a_context_set_backend(C4A_BACKEND_CUDA). The
# define is PRIVATE on chemometrics4all_core / chemometrics4all_c and does not
# transitively propagate, so we mirror it here.
if(CHEMOMETRICS4ALL_WITH_CUDA)
    target_compile_definitions(chemometrics4all_cli PRIVATE C4A_USE_CUDA=1)
endif()
