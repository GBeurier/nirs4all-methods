// SPDX-License-Identifier: CECILL-2.1
#include "core/linear_stack.hpp"

#include <array>
#include <cmath>

#include "core/common/matrix_view.hpp"
namespace n4m::core {
n4m_status_t compress_linear_stack(const n4m_matrix_view_t& base,
                                   const n4m_matrix_view_t& bias,
                                   const n4m_matrix_view_t& weights,
                                   const n4m_matrix_view_t& meta_bias,
                                   n4m_matrix_view_t& coefficients,
                                   n4m_matrix_view_t& intercept) {
    for (const auto* v : std::array<const n4m_matrix_view_t*, 6>{
             &base, &bias, &weights, &meta_bias, &coefficients, &intercept}) {
        if (validate_nonnull_view(*v) != N4M_OK)
            return N4M_ERR_INVALID_ARGUMENT;
        if (v->dtype != N4M_DTYPE_F64)
            return N4M_ERR_DTYPE_MISMATCH;
    }
    if (base.rows < 1 || base.cols < 1 || weights.cols < 1 || weights.rows != base.cols
        || bias.rows != 1 || bias.cols != base.cols || meta_bias.rows != 1
        || meta_bias.cols != weights.cols || coefficients.rows != base.rows
        || coefficients.cols != weights.cols || intercept.rows != 1
        || intercept.cols != weights.cols)
        return N4M_ERR_SHAPE_MISMATCH;
    auto read = [](const n4m_matrix_view_t& v, int64_t i, int64_t j) {
        return static_cast<const double*>(v.data)[i * v.row_stride + j * v.col_stride];
    };
    for (const auto* v :
         std::array<const n4m_matrix_view_t*, 4>{&base, &bias, &weights, &meta_bias})
        for (int64_t i = 0; i < v->rows; ++i)
            for (int64_t j = 0; j < v->cols; ++j)
                if (!std::isfinite(read(*v, i, j)))
                    return N4M_ERR_INVALID_ARGUMENT;
    for (int64_t k = 0; k < weights.cols; ++k) {
        double b = read(meta_bias, 0, k);
        for (int64_t j = 0; j < base.cols; ++j)
            b += read(bias, 0, j) * read(weights, j, k);
        static_cast<double*>(intercept.data)[k * intercept.col_stride] = b;
        for (int64_t i = 0; i < base.rows; ++i) {
            double value = 0;
            for (int64_t j = 0; j < base.cols; ++j)
                value += read(base, i, j) * read(weights, j, k);
            static_cast<double*>(
                coefficients.data)[i * coefficients.row_stride + k * coefficients.col_stride] =
                value;
        }
    }
    return N4M_OK;
}
}  // namespace n4m::core
