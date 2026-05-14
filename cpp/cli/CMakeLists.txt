# cpp/cli/CMakeLists.txt — pls4all_cli executable. Links libp4a; the CLI
# itself contains no algorithm logic, only ABI introspection.

add_executable(pls4all_cli p4a_cli.cpp)
target_link_libraries(pls4all_cli PRIVATE pls4all_c)
target_include_directories(pls4all_cli PRIVATE ${PROJECT_BINARY_DIR}/generated)
pls4all_add_warnings(pls4all_cli)
set_target_properties(pls4all_cli PROPERTIES OUTPUT_NAME pls4all_cli)
