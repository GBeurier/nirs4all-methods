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
// types with a `none` sentinel in place of -1, a const input copied once, and
// an enum status; the control flow and operation order are unchanged.
#include "core/optimization/_vendored/rectangular_lsap.hpp"

#include <algorithm>
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

std::size_t augmenting_path(std::size_t nc, const double* cost, std::vector<double>& u,
                            std::vector<double>& v, std::vector<std::size_t>& path,
                            std::vector<std::size_t>& row4col,
                            std::vector<double>& shortest_path_costs, std::size_t i,
                            std::vector<bool>& SR, std::vector<bool>& SC,
                            std::vector<std::size_t>& remaining, double* p_min_val) {
    double min_val = 0;

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
    std::fill(shortest_path_costs.begin(), shortest_path_costs.end(), infinity);

    // Find the shortest augmenting path.
    std::size_t sink = none;
    while (sink == none) {
        std::size_t index = none;
        double lowest = infinity;
        SR[i] = true;

        for (std::size_t it = 0; it < num_remaining; it++) {
            const std::size_t j = remaining[it];

            const double r = min_val + cost[i * nc + j] - u[i] - v[j];
            if (r < shortest_path_costs[j]) {
                path[j] = i;
                shortest_path_costs[j] = r;
            }

            // When multiple nodes have the minimum cost, select one which
            // gives a new sink node (important for small integer costs).
            if (shortest_path_costs[j] < lowest ||
                (shortest_path_costs[j] == lowest && row4col[j] == none)) {
                lowest = shortest_path_costs[j];
                index = it;
            }
        }

        min_val = lowest;
        if (min_val == infinity) {  // infeasible cost matrix
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

    // Initialize variables.
    std::vector<double> u(nr, 0);
    std::vector<double> v(nc, 0);
    std::vector<double> shortest_path_costs(nc);
    std::vector<std::size_t> path(nc, none);
    std::vector<std::size_t> col4row(nr, none);
    std::vector<std::size_t> row4col(nc, none);
    std::vector<bool> SR(nr);
    std::vector<bool> SC(nc);
    std::vector<std::size_t> remaining(nc);

    // Iteratively build the solution.
    for (std::size_t cur_row = 0; cur_row < nr; cur_row++) {
        double min_val = 0;
        const std::size_t sink = augmenting_path(nc, cost.data(), u, v, path, row4col,
                                                 shortest_path_costs, cur_row, SR, SC,
                                                 remaining, &min_val);
        if (sink == none) {
            return Status::infeasible;
        }

        // Update dual variables.
        u[cur_row] += min_val;
        for (std::size_t i = 0; i < nr; i++) {
            if (SR[i] && i != cur_row) {
                u[i] += min_val - shortest_path_costs[col4row[i]];
            }
        }

        for (std::size_t j = 0; j < nc; j++) {
            if (SC[j]) {
                v[j] -= min_val - shortest_path_costs[j];
            }
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
