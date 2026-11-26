// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
#include "core/preprocessing/signal_conversion/reference_conversions.hpp"

#include <cmath>
#include <limits>

namespace n4m::core {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

}  // namespace

n4m_status_t reflectance_from_references(const n4m_matrix_view_t& X,
                                         const double* white,
                                         const double* dark,
                                         double scale,
                                         n4m_matrix_view_t& out) noexcept {
    const auto* values = static_cast<const double*>(X.data);
    auto* result = static_cast<double*>(out.data);
    for (std::int64_t col = 0; col < X.cols; ++col) {
        const double offset = dark == nullptr ? 0.0 : dark[col];
        const double range = white[col] - offset;
        const bool valid = std::isfinite(range) && range > 0.0;
        for (std::int64_t row = 0; row < X.rows; ++row) {
            const double value = values[row * X.row_stride + col * X.col_stride];
            result[row * out.row_stride + col * out.col_stride] =
                valid ? scale * ((value - offset) / range) : kNaN;
        }
    }
    return N4M_OK;
}

n4m_status_t
from_kubelka_munk(const n4m_matrix_view_t& X, double scale, n4m_matrix_view_t& out) noexcept {
    const auto* values = static_cast<const double*>(X.data);
    auto* result = static_cast<double*>(out.data);
    for (std::int64_t row = 0; row < X.rows; ++row) {
        for (std::int64_t col = 0; col < X.cols; ++col) {
            const double f = values[row * X.row_stride + col * X.col_stride];
            const double r = f >= 0.0 ? 1.0 / (1.0 + f + std::sqrt(f * (f + 2.0))) : kNaN;
            result[row * out.row_stride + col * out.col_stride] = scale * r;
        }
    }
    return N4M_OK;
}

}  // namespace n4m::core
