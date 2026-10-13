// SPDX-License-Identifier: CECILL-2.1
// Exponent-managed centring shared by the statistics kernels: exact
// power-of-two rescaling and translation-stable centring, so a scale is
// always chosen from centred variation, never from raw magnitudes.
#ifndef N4M_CORE_COMMON_CENTRING_HPP
#define N4M_CORE_COMMON_CENTRING_HPP

#include <algorithm>
#include <cmath>
#include <vector>

namespace n4m::core {

// Exponent e of a power of two bounding max|v|: max|v| * 2^-e in [1/2, 1).
// 0 when every value is zero.
inline int bounding_exponent(const std::vector<double>& values) noexcept {
    double largest = 0.0;
    for (double v : values) largest = std::max(largest, std::abs(v));
    int exponent = 0;
    if (largest > 0.0) std::frexp(largest, &exponent);
    return exponent;
}

// In-place centering without cancellation of a large common offset: shift
// by the first value (exact for nearby values), subtract the mean, then the
// residual mean.
inline void center_in_place(std::vector<double>& v) noexcept {
    if (v.empty()) return;
    const double origin = v[0];
    double mean = 0.0;
    for (double& x : v) {
        x -= origin;
        mean += x;
    }
    mean /= static_cast<double>(v.size());
    double residual = 0.0;
    for (double& x : v) {
        x -= mean;
        residual += x;
    }
    residual /= static_cast<double>(v.size());
    for (double& x : v) x -= residual;
}

}  // namespace n4m::core

#endif  // N4M_CORE_COMMON_CENTRING_HPP
