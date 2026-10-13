// SPDX-License-Identifier: CECILL-2.1
//
// Exact sums of doubles as nonoverlapping floating-point expansions
// (Shewchuk, "Adaptive Precision Floating-Point Arithmetic and Fast Robust
// Geometric Predicates", Discrete & Computational Geometry 18, 1997). An
// expansion is a list of doubles in increasing magnitude whose exact sum is
// the value; TwoSum is error-free for every finite addition (also with
// subnormals), so sums and signs are exact as long as no component overflows.
#ifndef N4M_CORE_COMMON_EXPANSION_HPP
#define N4M_CORE_COMMON_EXPANSION_HPP

#include <vector>

namespace n4m::core::expansion {

using Expansion = std::vector<double>;

// x + y == a + b exactly, x = fl(a + b).
inline void two_sum(double a, double b, double& x, double& y) noexcept {
    x = a + b;
    const double bv = x - a;
    const double av = x - bv;
    y = (a - av) + (b - bv);
}

// Grow-Expansion with zero elimination: e + b.
inline Expansion grow(const Expansion& e, double b) {
    Expansion h;
    h.reserve(e.size() + 1);
    double q = b;
    for (const double component : e) {
        double x = 0.0, y = 0.0;
        two_sum(q, component, x, y);
        if (y != 0.0) h.push_back(y);
        q = x;
    }
    if (q != 0.0) h.push_back(q);
    return h;
}

inline Expansion add(const Expansion& e, const Expansion& f) {
    Expansion h = e;
    for (const double component : f) h = grow(h, component);
    return h;
}

inline Expansion negate(Expansion e) {
    for (double& component : e) component = -component;
    return e;
}

inline Expansion subtract(const Expansion& e, const Expansion& f) { return add(e, negate(f)); }

// Sign of the exact value: the largest component decides.
inline int sign(const Expansion& e) noexcept {
    return e.empty() ? 0 : (e.back() > 0.0 ? 1 : -1);
}

}  // namespace n4m::core::expansion

#endif  // N4M_CORE_COMMON_EXPANSION_HPP
