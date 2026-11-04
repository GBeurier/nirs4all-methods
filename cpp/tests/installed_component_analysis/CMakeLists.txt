cmake_minimum_required(VERSION 3.21)
project(n4m_installed_component_analysis LANGUAGES C CXX)

# Deliberately consume only the installed package. No source/build include or
# archive path may mask an incomplete exported static dependency closure.
find_package(N4m CONFIG REQUIRED)
add_executable(n4m_installed_component_analysis main.c)
set_property(TARGET n4m_installed_component_analysis PROPERTY C_STANDARD 11)
target_link_libraries(n4m_installed_component_analysis PRIVATE n4m::n4m_c_static)
enable_testing()
add_test(NAME installed_component_analysis COMMAND n4m_installed_component_analysis)
