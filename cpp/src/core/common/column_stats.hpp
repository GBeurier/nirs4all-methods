// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace n4m::core::detail {

// Autoscaling guard shared by every standardization (column standard
// deviation used as a divisor: PLS-family scale_x / scale_y, Ridge,
// StandardScale, sweeps, preprocessing pipelines). A scale that is not
// finite or below 10 * DBL_EPSILON is replaced by 1, the absolute rule of
// scikit-learn's _handle_zeros_in_scale (sklearn.preprocessing.scale,
// MinMaxScaler, MaxAbsScaler, RobustScaler). Such a column is constant up to
// rounding (for example the exactly-zero edge column of a mirror-mode
// Savitzky-Golay derivative, computed as +-1e-17): dividing by its spread
// would turn rounding noise into a unit-variance variable. The rule is
// absolute, as in scikit-learn, so a genuine variable whose spread is below
// 2.2e-15 in its own units is left unscaled. (scikit-learn's PLS and
// StandardScaler only map an exactly zero spread, or a spread small
// relative to the column mean, to 1; Methods applies the stronger rule.)
constexpr double kMinAutoscale = 10.0 * std::numeric_limits<double>::epsilon();

inline double guarded_scale(double scale) noexcept {
    return (std::isfinite(scale) && scale >= kMinAutoscale) ? scale : 1.0;
}

// Shared native statistics; existing callers retain population scaling (ddof=0).
inline void column_means(const std::vector<double>& mat, std::size_t rows,
                         std::size_t cols, std::vector<double>& means) {
    means.assign(cols, 0.0);
    if (rows == 0) return;
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) means[c] += mat[r * cols + c];
    }
    const double inv = 1.0 / static_cast<double>(rows);
    for (double& m : means) m *= inv;
}

// Nonzero ddof is admitted by the fitted scaler boundary: ddof < rows.
inline void column_scales(const std::vector<double>& centered, std::size_t rows,
                          std::size_t cols, std::vector<double>& scales, std::size_t ddof = 0) {
    scales.assign(cols, 1.0);
    if (rows == 0) return;
    for (std::size_t c = 0; c < cols; ++c) {
        double ss = 0.0;
        for (std::size_t r = 0; r < rows; ++r) {
            const double v = centered[r * cols + c];
            ss += v * v;
        }
        const double std_dev = std::sqrt(ss / static_cast<double>(rows - ddof));
        scales[c] = guarded_scale(std_dev);
    }
}

}  // namespace n4m::core::detail
