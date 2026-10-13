// SPDX-License-Identifier: BSD-3-Clause
//
// Rectangular linear sum assignment — C++17 port of SciPy 1.18.1
// scipy/optimize/rectangular_lsap/rectangular_lsap.cpp (author PM Larsen),
// the shortest augmenting path algorithm of Crouse (2016). See
// LICENSE.scipy.txt in this directory for the upstream licence.
//
// Frozen semantics (scipy.optimize.linear_sum_assignment, evaluated in exact
// arithmetic: duals and path costs are exact sums of the input doubles, so
// the result is an exact optimum):
//   * a tall matrix (cols < rows) is solved on its transpose and the pairs are
//     returned sorted by row; maximize negates the costs first;
//   * NaN or -inf costs (after negation) are invalid; +inf marks a forbidden
//     pair and an assignment of every row (or column) without one is
//     infeasible;
//   * ties follow the upstream order: columns are scanned from the last to the
//     first, a strictly smaller reduced cost wins, and on equal reduced cost
//     an unassigned column wins over an assigned one (a constant matrix gives
//     the identity assignment).
#ifndef N4M_CORE_OPTIMIZATION_VENDORED_RECTANGULAR_LSAP_HPP
#define N4M_CORE_OPTIMIZATION_VENDORED_RECTANGULAR_LSAP_HPP

#include <cstddef>
#include <cstdint>

namespace n4m::core::lsap {

enum class Status : std::int32_t { ok = 0, invalid = 1, infeasible = 2 };

// cost is row-major rows x cols. On ok, row_ind/col_ind receive
// min(rows, cols) pairs in increasing row order.
Status solve(std::size_t rows, std::size_t cols, const double* cost, bool maximize,
             std::int64_t* row_ind, std::int64_t* col_ind);

}  // namespace n4m::core::lsap

#endif  // N4M_CORE_OPTIMIZATION_VENDORED_RECTANGULAR_LSAP_HPP
