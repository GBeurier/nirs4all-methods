// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

namespace n4m::core::detail {

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
        scales[c] = (std_dev > 0.0) ? std_dev : 1.0;
    }
}

}  // namespace n4m::core::detail
