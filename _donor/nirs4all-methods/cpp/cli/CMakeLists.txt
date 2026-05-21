# cpp/cli/CMakeLists.txt — n4m_cli executable. Links libn4m; the CLI
# itself contains no algorithm logic, only ABI introspection.

add_executable(n4m_cli n4m_cli.cpp)
target_link_libraries(n4m_cli PRIVATE n4m_c)
target_include_directories(n4m_cli PRIVATE ${PROJECT_BINARY_DIR}/generated)
n4m_add_warnings(n4m_cli)
set_target_properties(n4m_cli PROPERTIES OUTPUT_NAME n4m_cli)

# The selfcheck uses `#if defined(N4M_USE_CUDA)` to pick its expected
# return value from n4m_context_set_backend(N4M_BACKEND_CUDA). The
# define is PRIVATE on n4m_core / n4m_c and does not
# transitively propagate, so we mirror it here.
if(N4M_WITH_CUDA)
    target_compile_definitions(n4m_cli PRIVATE N4M_USE_CUDA=1)
endif()
