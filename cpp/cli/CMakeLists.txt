# cpp/cli/CMakeLists.txt — pls4all_cli executable. Links libp4a; the CLI
# itself contains no algorithm logic, only ABI introspection.

add_executable(pls4all_cli p4a_cli.cpp)
target_link_libraries(pls4all_cli PRIVATE pls4all_c)
target_include_directories(pls4all_cli PRIVATE ${PROJECT_BINARY_DIR}/generated)
pls4all_add_warnings(pls4all_cli)
set_target_properties(pls4all_cli PROPERTIES OUTPUT_NAME pls4all_cli)

# The selfcheck uses `#if defined(P4A_USE_CUDA)` to pick its expected
# return value from p4a_context_set_backend(P4A_BACKEND_CUDA). The
# define is PRIVATE on pls4all_core / pls4all_c and does not
# transitively propagate, so we mirror it here.
if(PLS4ALL_WITH_CUDA)
    target_compile_definitions(pls4all_cli PRIVATE P4A_USE_CUDA=1)
endif()
