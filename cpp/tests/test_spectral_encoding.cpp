// SPDX-License-Identifier: CECILL-2.1
#include <cmath>
#include <limits>
#include <vector>

#include "harness.hpp"
#include "n4m/n4m.h"

namespace {
n4m_matrix_view_t view(double* data, int64_t n, int64_t p, bool column_major = false) {
    n4m_matrix_view_t result{};
    result.data = data;
    result.rows = n;
    result.cols = p;
    result.row_stride = column_major ? 1 : p;
    result.col_stride = column_major ? n : 1;
    result.dtype = N4M_DTYPE_F64;
    return result;
}
void lvse_strides_and_affine() {
    constexpr int64_t n = 12, p = 7;
    std::vector<double> x(n * p), column(n * p);
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < p; ++j) {
            const auto index = static_cast<std::size_t>(i * p + j);
            x[index] = std::sin(static_cast<double>(i + j * 3)) + static_cast<double>(j);
            column[static_cast<std::size_t>(j * n + i)] = x[index];
        }
    n4m_spectral_encoder_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_decomposition_spectral_create(0, 4, 2, .5, 1, 0, 60, .001, &h) == N4M_OK);
    int64_t q = 0;
    N4M_TEST_REQUIRE(n4m_decomposition_spectral_output_cols(h, &q) != N4M_OK);
    N4M_TEST_REQUIRE(n4m_decomposition_spectral_fit(h, view(column.data(), n, p, true)) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_decomposition_spectral_output_cols(h, &q) == N4M_OK);
    N4M_TEST_REQUIRE(q == 6);
    std::vector<double> z(static_cast<std::size_t>(n * q)), op(static_cast<std::size_t>(q * p)),
        b(static_cast<std::size_t>(q));
    N4M_TEST_REQUIRE(
        n4m_decomposition_spectral_transform(h, view(x.data(), n, p), view(z.data(), n, q, true))
        == N4M_OK);
    N4M_TEST_REQUIRE(n4m_decomposition_spectral_export_affine(
                         h, view(op.data(), q, p, true), view(b.data(), 1, q))
                     == N4M_OK);
    for (int64_t i = 0; i < n; ++i)
        for (int64_t a = 0; a < q; ++a) {
            double expected = b[static_cast<std::size_t>(a)];
            for (int64_t j = 0; j < p; ++j)
                expected += x[static_cast<std::size_t>(i * p + j)]
                            * op[static_cast<std::size_t>(j * q + a)];
            N4M_TEST_REQUIRE(std::abs(expected - z[static_cast<std::size_t>(a * n + i)]) < 1e-11);
        }
    x[0] = std::numeric_limits<double>::quiet_NaN();
    N4M_TEST_REQUIRE(n4m_decomposition_spectral_fit(h, view(x.data(), n, p))
                     == N4M_ERR_INVALID_ARGUMENT);
    std::vector<double> after(z.size());
    N4M_TEST_REQUIRE(n4m_decomposition_spectral_transform(
                         h, view(column.data(), n, p, true), view(after.data(), n, q, true))
                     == N4M_OK);
    N4M_TEST_REQUIRE(after == z);
    n4m_decomposition_spectral_destroy(h);
}
void gcu_nonnegative_and_nonaffine() {
    std::vector<double> x{1, 2, 5, 2, 4, 1, 3, 2, 1, 4, 1, 3};
    n4m_spectral_encoder_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_decomposition_spectral_create(1, 64, 2, 0, 1, 0, 60, .001, &h) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_decomposition_spectral_fit(h, view(x.data(), 4, 3)) == N4M_OK);
    std::vector<double> z(8), op(6), b(2);
    N4M_TEST_REQUIRE(
        n4m_decomposition_spectral_transform(h, view(x.data(), 4, 3), view(z.data(), 4, 2))
        == N4M_OK);
    for (double value : z)
        N4M_TEST_REQUIRE(std::isfinite(value) && value >= 0);
    N4M_TEST_REQUIRE(
        n4m_decomposition_spectral_export_affine(h, view(op.data(), 2, 3), view(b.data(), 1, 2))
        == N4M_ERR_INVALID_ARGUMENT);
    n4m_decomposition_spectral_destroy(h);
}
}  // namespace
void register_spectral_encoding_tests(n4m_testing::Runner& r);
void register_spectral_encoding_tests(n4m_testing::Runner& r) {
    r.run("lvse_strides_affine_atomic_refit", lvse_strides_and_affine);
    r.run("gcu_nonnegative_nonaffine", gcu_nonnegative_and_nonaffine);
}
