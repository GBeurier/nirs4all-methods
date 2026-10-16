// SPDX-License-Identifier: CECILL-2.1
// Supported numerical domain D of the Preflight and statistics kernels (lead
// rules 13 and 15; public constants in n4m/metrics.h): finite values, every
// nonzero |x| in [2^-256, 2^256] and, where a statistic depends on differences
// between values, max nonzero |x| / min nonzero |x| <= 2^32 inside the array
// (or slice) that is differenced. Exact zeros are always allowed. Every kernel
// checks D once, in an O(n) pre-pass before computing, and refuses inputs
// outside it with N4M_ERR_UNSUPPORTED and one of the messages below.
#ifndef N4M_CORE_COMMON_NUMERICAL_DOMAIN_HPP
#define N4M_CORE_COMMON_NUMERICAL_DOMAIN_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

#include "n4m/metrics.h"

namespace n4m::core {

static_assert(N4M_DOMAIN_MAX_MAGNITUDE_LOG2 == 256 && N4M_DOMAIN_MAX_SPREAD_LOG2 == 32,
              "update the domain constants and messages below");
inline constexpr double domain_low = 0x1p-256;
inline constexpr double domain_high = 0x1p256;
inline constexpr double domain_spread = 0x1p32;
inline constexpr const char* domain_message =
    "input outside the supported numerical domain (nonzero |x| in [2^-256, 2^256], "
    "max/min nonzero |x| <= 2^32)";
// Summaries and nearest neighbours: magnitude bounds only (rule 15 exemption).
inline constexpr const char* magnitude_domain_message =
    "input outside the supported numerical domain (nonzero |x| in [2^-256, 2^256])";

// Smallest and largest nonzero |x| of the values added; a nonfinite value is
// outside D.
struct DomainRange {
    double low = std::numeric_limits<double>::infinity();
    double high = 0.0;
    bool finite = true;
    void add(double x) noexcept {
        if (!std::isfinite(x)) {
            finite = false;
            return;
        }
        const double a = std::abs(x);
        if (a == 0.0) return;
        low = std::min(low, a);
        high = std::max(high, a);
    }
    bool magnitudes_inside() const noexcept {
        return finite && (high == 0.0 || (low >= domain_low && high <= domain_high));
    }
    bool inside() const noexcept {
        return magnitudes_inside() && (high == 0.0 || high <= low * domain_spread);
    }
};

inline bool in_numerical_domain(const double* values, std::size_t n) noexcept {
    DomainRange range;
    for (std::size_t i = 0; i < n; ++i) range.add(values[i]);
    return range.inside();
}

}  // namespace n4m::core

#endif  // N4M_CORE_COMMON_NUMERICAL_DOMAIN_HPP
