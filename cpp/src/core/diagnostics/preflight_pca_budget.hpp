// SPDX-License-Identifier: CECILL-2.1
//
// Guaranteed work and memory bounds of n4m_preflight_pca_compute{,_resolved}
// (cpp/src/c_api/c_api_preflight.cpp), checked before any allocation. Kept
// in a header so the counts are tested directly (cpp/tests/test_preflight.cpp)
// without allocating the inputs they refuse.
//
// Unit of work: one long-double multiply-add (software binary128 in
// WebAssembly, measured at least 16 times slower than a double operation
// there); every other operation counts 1/16. With s = min(n,p), l = max(n,p)
// and the Gram matrix on the smaller side, the bound adds, as upper bounds of
// what the code executes:
//  - long double: sum of squares n p, Gram s(s+1)/2 * l, trace(C^2) s(s+1),
//    projections (scores or loadings) k s l and their division k l;
//  - double, at 1/16: input copy, domain scan and centring 32 n p; Gram
//    storage s^2; eigensolver scaling and copies 4 s^2 + s; Householder
//    tridiagonalisation with accumulation 4 s^3 / 3 + 3 s^2 + 10 s; the QL
//    iteration at its CEILING, the solver's cap of 30 s sweeps (svd.c), each
//    at most s - 1 rotations of 4 s + 32 operations plus a 2 s convergence
//    scan; the eigenvalue sort 2 s (log2 s + 2); for n <= p the loading
//    re-orthonormalisation, Householder QR and completion 3 k^2 p + 7 k p
//    (for n > p the loading copy k p); scores, sign pass and output writes
//    2 k p + 2 n k.
// The QL ceiling is never reached in practice (QL converges in about two
// sweeps per eigenvalue), so the typical cost is far below the bound: for a
// square s x s input it is about 0.5 s^3 + 0.6 s^3 against 8 s^3 for the
// bound. The limit 2.4e9 is the smallest round value that keeps every shape a
// Preflight request can produce (n p <= 10^7, k <= 10) and that Methods 1.3.6
// (Gram-only s^2 l <= 2.5e8) or the previous whole-computation estimate
// accepted: their worst case under this bound is 658 x 658, k = 1, 2.33e9
// (enumerated in cpp/tests/test_preflight.cpp).
//
// Memory: the doubles simultaneously live at the peak, all counted together
// (an upper bound of the peak): the centred copy n p, the column exponents p
// ints, the Gram matrix, the eigenvectors and the eigensolver's work copy
// 3 s^2, eigenvalues, off-diagonal and sort order 3 s, loadings and their
// Householder copy 2 k p, scores n k, reflector coefficients and flags 2 k;
// limit 1 GiB.
#ifndef N4M_CORE_DIAGNOSTICS_PREFLIGHT_PCA_BUDGET_HPP
#define N4M_CORE_DIAGNOSTICS_PREFLIGHT_PCA_BUDGET_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace n4m::core {

constexpr long double preflight_pca_max_work = 2.4e9L;
constexpr long double preflight_pca_max_doubles = 134217728.0L;  // 1 GiB

struct PreflightPcaBudget {
    long double work;     // long-double multiply-add equivalents (upper bound)
    long double doubles;  // peak live buffers, in doubles (upper bound)
};

inline PreflightPcaBudget preflight_pca_budget(std::int64_t n, std::int64_t p,
                                               std::int64_t k) noexcept {
    const long double N = static_cast<long double>(n), P = static_cast<long double>(p);
    const long double K = static_cast<long double>(k);
    const long double s = std::min(N, P), l = std::max(N, P);
    const bool wide = n <= p;
    const long double extended =
        N * P + s * (s + 1.0L) / 2.0L * l + s * (s + 1.0L) + K * s * l + K * l;
    const long double ql_ceiling = 30.0L * s * ((s - 1.0L) * (4.0L * s + 32.0L) + 2.0L * s);
    const long double dbl =
        32.0L * N * P + s * s + (4.0L * s * s + s) +
        (4.0L * s * s * s / 3.0L + 3.0L * s * s + 10.0L * s) + ql_ceiling +
        2.0L * s * (std::log2(s) + 2.0L) + (wide ? 3.0L * K * K * P + 7.0L * K * P : K * P) +
        2.0L * K * P + 2.0L * N * K;
    const long double doubles = N * P + std::ceil(P / 2.0L) + 3.0L * s * s + 3.0L * s +
                                2.0L * K * P + N * K + 2.0L * K;
    return {extended + dbl / 16.0L, doubles};
}

}  // namespace n4m::core

#endif  // N4M_CORE_DIAGNOSTICS_PREFLIGHT_PCA_BUDGET_HPP
