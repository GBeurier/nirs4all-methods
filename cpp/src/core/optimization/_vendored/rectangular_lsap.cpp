// SPDX-License-Identifier: BSD-3-Clause
//
// Port of SciPy 1.18.1 scipy/optimize/rectangular_lsap/rectangular_lsap.cpp.
// Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers.
// Author of the upstream implementation: PM Larsen. The algorithm is the
// shortest augmenting path method on pages 1685-1686 of
//     DF Crouse. On implementing 2D rectangular assignment algorithms.
//     IEEE Transactions on Aerospace and Electronic Systems
//     52(4):1679-1696, August 2016. doi: 10.1109/TAES.2016.140952
// Redistribution and use are subject to the BSD-3-Clause conditions in
// LICENSE.scipy.txt in this directory. Changes from upstream: unsigned index
// types with a `none` sentinel in place of -1, a const input copied once, an
// enum status, and exact arithmetic: every finite cost is an integer multiple
// of 2^E (E = binary exponent of the last mantissa bit of the smallest
// nonzero |cost|), and the dual variables, reduced costs and path costs are
// kept as signed 192-bit integers in units of 2^E, so every comparison is
// decided on exact values. SciPy's double arithmetic can lose small cost
// differences next to large ones and return a suboptimal assignment; this
// port returns an exact optimum. The control flow and the tie rules are
// unchanged.
#include "core/optimization/_vendored/rectangular_lsap.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

namespace n4m::core::lsap {
namespace {

constexpr std::size_t none = std::numeric_limits<std::size_t>::max();
constexpr double infinity = std::numeric_limits<double>::infinity();

std::vector<std::size_t> argsort_iter(const std::vector<std::size_t>& values) {
    std::vector<std::size_t> index(values.size());
    std::iota(index.begin(), index.end(), std::size_t{0});
    std::sort(index.begin(), index.end(),
              [&values](std::size_t i, std::size_t j) { return values[i] < values[j]; });
    return index;
}

// Signed 192-bit two's-complement integer (three little-endian limbs).
struct Fixed {
    std::uint64_t w[3] = {0, 0, 0};
};

Fixed add(const Fixed& a, const Fixed& b, bool& overflow) {
    Fixed r;
    unsigned carry = 0;
    for (int k = 0; k < 3; ++k) {
        const std::uint64_t partial = a.w[k] + b.w[k];
        const unsigned c1 = partial < a.w[k] ? 1u : 0u;
        r.w[k] = partial + carry;
        carry = c1 + (r.w[k] < partial ? 1u : 0u);
    }
    // Values stay far below 2^189 (see solve()); flag any growth beyond it.
    const std::uint64_t top = r.w[2] >> 61;
    overflow = overflow || (top != 0 && top != 7);
    return r;
}

Fixed negate(const Fixed& a) {
    Fixed inverted;
    for (int k = 0; k < 3; ++k) inverted.w[k] = ~a.w[k];
    bool ignored = false;
    Fixed one;
    one.w[0] = 1;
    return add(inverted, one, ignored);
}

int compare_fixed(const Fixed& a, const Fixed& b) {
    const bool negative_a = (a.w[2] >> 63) != 0, negative_b = (b.w[2] >> 63) != 0;
    if (negative_a != negative_b) return negative_a ? -1 : 1;
    for (int k = 2; k >= 0; --k)
        if (a.w[k] != b.w[k]) return a.w[k] < b.w[k] ? -1 : 1;
    return 0;
}

// An exact value: +infinity (a forbidden pair) or an integer in units 2^E.
struct Exact {
    bool infinite = false;
    Fixed value;
};

struct Arithmetic {
    int unit_exponent = 0;
    bool overflow = false;

    Exact exact(double v) const {
        if (std::isinf(v)) return {true, {}};
        Exact r;
        if (v == 0.0) return r;
        int exponent = 0;
        const double fraction = std::frexp(std::fabs(v), &exponent);
        const auto mantissa = static_cast<std::uint64_t>(std::ldexp(fraction, 53));
        const int shift = exponent - 53 - unit_exponent;  // 0 <= shift <= 66
        const int limb = shift / 64, offset = shift % 64;
        r.value.w[limb] = mantissa << offset;
        if (offset != 0 && limb < 2) r.value.w[limb + 1] = mantissa >> (64 - offset);
        if (v < 0.0) r.value = negate(r.value);
        return r;
    }
    Exact plus(const Exact& a, const Exact& b) {
        if (a.infinite || b.infinite) return {true, {}};
        return {false, add(a.value, b.value, overflow)};
    }
    // a - b for a finite b.
    Exact minus(const Exact& a, const Exact& b) {
        if (a.infinite) return {true, {}};
        return {false, add(a.value, negate(b.value), overflow)};
    }
};

int compare(const Exact& a, const Exact& b) {
    if (a.infinite || b.infinite) return a.infinite == b.infinite ? 0 : (a.infinite ? 1 : -1);
    return compare_fixed(a.value, b.value);
}

std::size_t augmenting_path(Arithmetic& arithmetic, std::size_t nc, const double* cost,
                            std::vector<Exact>& u,
                            std::vector<Exact>& v, std::vector<std::size_t>& path,
                            std::vector<std::size_t>& row4col,
                            std::vector<Exact>& shortest_path_costs, std::size_t i,
                            std::vector<bool>& SR, std::vector<bool>& SC,
                            std::vector<std::size_t>& remaining, Exact* p_min_val) {
    Exact min_val;

    // Crouse's pseudocode uses set complements to keep track of remaining
    // nodes; a vector is more efficient.
    std::size_t num_remaining = nc;
    for (std::size_t it = 0; it < nc; it++) {
        // Filling this up in reverse order ensures that the solution of a
        // constant cost matrix is the identity matrix (scipy #11602).
        remaining[it] = nc - it - 1;
    }

    std::fill(SR.begin(), SR.end(), false);
    std::fill(SC.begin(), SC.end(), false);
    std::fill(shortest_path_costs.begin(), shortest_path_costs.end(), Exact{true, {}});

    // Find the shortest augmenting path.
    std::size_t sink = none;
    while (sink == none) {
        std::size_t index = none;
        Exact lowest{true, {}};
        SR[i] = true;

        for (std::size_t it = 0; it < num_remaining; it++) {
            const std::size_t j = remaining[it];

            const Exact r = arithmetic.minus(
                arithmetic.plus(min_val, arithmetic.exact(cost[i * nc + j])),
                arithmetic.plus(u[i], v[j]));
            if (compare(r, shortest_path_costs[j]) < 0) {
                path[j] = i;
                shortest_path_costs[j] = r;
            }

            // When multiple nodes have the minimum cost, select one which
            // gives a new sink node (important for small integer costs).
            const int order = compare(shortest_path_costs[j], lowest);
            if (order < 0 || (order == 0 && row4col[j] == none)) {
                lowest = shortest_path_costs[j];
                index = it;
            }
        }

        min_val = lowest;
        if (min_val.infinite) {  // infeasible cost matrix
            return none;
        }

        const std::size_t j = remaining[index];
        if (row4col[j] == none) {
            sink = j;
        } else {
            i = row4col[j];
        }

        SC[j] = true;
        remaining[index] = remaining[--num_remaining];
    }

    *p_min_val = min_val;
    return sink;
}

}  // namespace

Status solve(std::size_t nr, std::size_t nc, const double* input_cost, bool maximize,
             std::int64_t* a, std::int64_t* b) {
    // Handle trivial inputs.
    if (nr == 0 || nc == 0) {
        return Status::ok;
    }

    // A tall rectangular cost matrix must be transposed.
    const bool transpose = nc < nr;

    std::vector<double> cost(nr * nc);
    if (transpose) {
        for (std::size_t i = 0; i < nr; i++) {
            for (std::size_t j = 0; j < nc; j++) {
                cost[j * nr + i] = input_cost[i * nc + j];
            }
        }
        std::swap(nr, nc);
    } else {
        std::copy(input_cost, input_cost + nr * nc, cost.begin());
    }

    // Negate the cost matrix for maximization.
    if (maximize) {
        for (double& value : cost) {
            value = -value;
        }
    }

    // Test for NaN and -inf entries.
    for (const double value : cost) {
        if (value != value || value == -infinity) {
            return Status::invalid;
        }
    }

    // Exact integer units: 2^E with E the exponent of the last mantissa bit
    // of the smallest nonzero |cost|. In the supported domain (nonzero |cost|
    // in [2^-256, 2^256], spread <= 2^64) every cost is below 2^119 units.
    // Duals and path costs are differences of sums of at most
    // 4 * (rows + 1) costs, far below the 2^189 overflow guard for every size
    // the C API admits; a guard breach is reported, never wrapped.
    Arithmetic arithmetic;
    int smallest = std::numeric_limits<int>::max(), largest = std::numeric_limits<int>::min();
    for (const double value : cost) {
        if (value == 0.0 || std::isinf(value)) continue;
        int exponent = 0;
        std::frexp(value, &exponent);
        smallest = std::min(smallest, exponent);
        largest = std::max(largest, exponent);
    }
    if (largest != std::numeric_limits<int>::min() && largest - smallest > 120) {
        return Status::overflow;  // outside the domain the caller guarantees
    }
    arithmetic.unit_exponent = smallest == std::numeric_limits<int>::max() ? 0 : smallest - 53;

    // Initialize variables.
    std::vector<Exact> u(nr);
    std::vector<Exact> v(nc);
    std::vector<Exact> shortest_path_costs(nc);
    std::vector<std::size_t> path(nc, none);
    std::vector<std::size_t> col4row(nr, none);
    std::vector<std::size_t> row4col(nc, none);
    std::vector<bool> SR(nr);
    std::vector<bool> SC(nc);
    std::vector<std::size_t> remaining(nc);

    // Iteratively build the solution.
    for (std::size_t cur_row = 0; cur_row < nr; cur_row++) {
        Exact min_val;
        const std::size_t sink = augmenting_path(arithmetic, nc, cost.data(), u, v, path, row4col,
                                                 shortest_path_costs, cur_row, SR, SC,
                                                 remaining, &min_val);
        if (sink == none) {
            return Status::infeasible;
        }

        // Update dual variables.
        u[cur_row] = arithmetic.plus(u[cur_row], min_val);
        for (std::size_t i = 0; i < nr; i++) {
            if (SR[i] && i != cur_row) {
                u[i] = arithmetic.plus(
                    u[i], arithmetic.minus(min_val, shortest_path_costs[col4row[i]]));
            }
        }

        for (std::size_t j = 0; j < nc; j++) {
            if (SC[j]) {
                v[j] = arithmetic.minus(v[j], arithmetic.minus(min_val, shortest_path_costs[j]));
            }
        }
        if (arithmetic.overflow) {
            return Status::overflow;
        }

        // Augment the previous solution.
        std::size_t j = sink;
        while (true) {
            const std::size_t i = path[j];
            row4col[j] = i;
            std::swap(col4row[i], j);
            if (i == cur_row) {
                break;
            }
        }
    }

    if (transpose) {
        std::size_t i = 0;
        for (const std::size_t column : argsort_iter(col4row)) {
            a[i] = static_cast<std::int64_t>(col4row[column]);
            b[i] = static_cast<std::int64_t>(column);
            i++;
        }
    } else {
        for (std::size_t i = 0; i < nr; i++) {
            a[i] = static_cast<std::int64_t>(i);
            b[i] = static_cast<std::int64_t>(col4row[i]);
        }
    }

    return Status::ok;
}

}  // namespace n4m::core::lsap
