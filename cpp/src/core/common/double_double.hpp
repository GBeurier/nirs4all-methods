// SPDX-License-Identifier: CECILL-2.1
// Minimal double-double arithmetic (about 106 significant bits) built from
// error-free transformations (TwoSum, fma TwoProduct), after Dekker (1971)
// and the QD library algorithms of Hida, Li and Bailey (2001). Used where a
// statistic needs coefficients or sums more accurate than binary64, e.g.
// Savitzky-Golay weights applied to values with a large dynamic range.
// Relative error of add/mul: below 2^-104; div: below 2^-102 (finite,
// non-overflowing operands, results above 2^-969).
#ifndef N4M_CORE_COMMON_DOUBLE_DOUBLE_HPP
#define N4M_CORE_COMMON_DOUBLE_DOUBLE_HPP

#include <cmath>

namespace n4m::core {

struct DD {
    double hi = 0.0;
    double lo = 0.0;
};

inline DD dd_two_sum(double a, double b) {
    const double s = a + b, bb = s - a;
    return {s, (a - (s - bb)) + (b - bb)};
}
inline DD dd_quick_two_sum(double a, double b) {
    const double s = a + b;
    return {s, b - (s - a)};
}
inline DD dd_add(DD a, DD b) {
    DD s = dd_two_sum(a.hi, b.hi);
    const DD t = dd_two_sum(a.lo, b.lo);
    s.lo += t.hi;
    s = dd_quick_two_sum(s.hi, s.lo);
    s.lo += t.lo;
    return dd_quick_two_sum(s.hi, s.lo);
}
inline DD dd_neg(DD a) { return {-a.hi, -a.lo}; }
inline DD dd_mul(DD a, DD b) {
    const double p = a.hi * b.hi;
    double e = std::fma(a.hi, b.hi, -p);
    e += a.hi * b.lo + a.lo * b.hi;
    return dd_quick_two_sum(p, e);
}
inline DD dd_div(DD a, DD b) {
    const double q1 = a.hi / b.hi;
    DD r = dd_add(a, dd_neg(dd_mul(b, {q1, 0.0})));
    const double q2 = r.hi / b.hi;
    r = dd_add(r, dd_neg(dd_mul(b, {q2, 0.0})));
    const double q3 = r.hi / b.hi;
    return dd_add(dd_quick_two_sum(q1, q2), {q3, 0.0});
}

}  // namespace n4m::core

#endif  // N4M_CORE_COMMON_DOUBLE_DOUBLE_HPP
