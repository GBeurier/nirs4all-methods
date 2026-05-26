# bench/cpp/CMakeLists.txt — n4m_donor_bench executable.
#
# Times the donor operators (augmentation / preprocessing / filters /
# splitters) through the public C ABI (`n4m_c`), with no high-level binding
# in between. Emits one adaptive-timing JSON line per (method, size) cell —
# the same adaptive-v1 schema the cross-binding orchestrator's Python bench
# scripts produce — so the donor pipeline can feed it straight into the
# dashboard CSV. Links the public shared ABI (not n4m_core) so it exercises
# exactly what every other binding sees and never affects the ABI snapshot.
#
# Enabled via -DN4M_BUILD_BENCH=ON (declared in cmake/N4mOptions.cmake).
add_executable(n4m_donor_bench n4m_donor_bench.cpp)
target_link_libraries(n4m_donor_bench PRIVATE n4m_c)
target_include_directories(n4m_donor_bench PRIVATE ${PROJECT_BINARY_DIR}/generated)
n4m_add_warnings(n4m_donor_bench)
set_target_properties(n4m_donor_bench PROPERTIES OUTPUT_NAME n4m_donor_bench)
