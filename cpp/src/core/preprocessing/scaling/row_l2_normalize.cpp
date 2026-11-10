// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
#include "core/preprocessing/scaling/row_l2_normalize.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace n4m::core {

n4m_status_t row_l2_validate_input(const n4m_matrix_view_t& X) noexcept {
    const auto* values = static_cast<const double*>(X.data);
    for (std::int64_t row = 0; row < X.rows; ++row) {
        for (std::int64_t col = 0; col < X.cols; ++col) {
            if (!std::isfinite(values[row * X.row_stride + col * X.col_stride])) {
                return N4M_ERR_INVALID_ARGUMENT;
            }
        }
    }
    return N4M_OK;
}

n4m_status_t row_l2_normalize(const n4m_matrix_view_t& X, n4m_matrix_view_t& out) noexcept {
    const auto* values = static_cast<const double*>(X.data);
    auto* result = static_cast<double*>(out.data);
    for (std::int64_t row = 0; row < X.rows; ++row) {
        double scale = 0.0;
        for (std::int64_t col = 0; col < X.cols; ++col) {
            scale = std::max(scale, std::abs(values[row * X.row_stride + col * X.col_stride]));
        }
        double sum = 0.0;
        if (scale != 0.0) {
            for (std::int64_t col = 0; col < X.cols; ++col) {
                const double value = values[row * X.row_stride + col * X.col_stride] / scale;
                sum += value * value;
            }
        }
        const double norm = std::sqrt(sum);
        for (std::int64_t col = 0; col < X.cols; ++col) {
            const double value = values[row * X.row_stride + col * X.col_stride];
            // Divide in two stages: scale * norm can overflow for finite rows.
            result[row * out.row_stride + col * out.col_stride] =
                scale == 0.0 ? value : (value / scale) / norm;
        }
    }
    return N4M_OK;
}

}  // namespace n4m::core
