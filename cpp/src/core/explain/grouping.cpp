// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
//
// Feature-grouping and attribution-summary helpers.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

#include "core/explain/explain_internal.hpp"

namespace n4m::explain {

n4m_status_t groups_fixed_width(int64_t n_features, int64_t width, int64_t* feature_group,
                                int64_t feature_group_size, int64_t* out_n_groups) {
    if (feature_group == nullptr || out_n_groups == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    if (n_features < 1 || width < 1) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (feature_group_size < n_features) {
        return N4M_ERR_SHAPE_MISMATCH;
    }
    for (int64_t j = 0; j < n_features; ++j) {
        feature_group[j] = j / width;
    }
    *out_n_groups = n_features / width + (n_features % width != 0 ? 1 : 0);
    return N4M_OK;
}

n4m_status_t groups_from_ranges(int64_t n_features, const int64_t* ranges, int64_t n_ranges,
                                int64_t* feature_group, int64_t feature_group_size,
                                int64_t* range_of_group, int64_t range_of_group_size,
                                int64_t* out_n_groups) {
    if (feature_group == nullptr || out_n_groups == nullptr
        || (n_ranges > 0 && ranges == nullptr)) {
        return N4M_ERR_NULL_POINTER;
    }
    if (n_features < 1 || n_ranges < 0 || n_ranges > n_features) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (feature_group_size < n_features) {
        return N4M_ERR_SHAPE_MISMATCH;
    }
    const auto n_r = static_cast<std::size_t>(n_ranges);
    for (std::size_t r = 0; r < n_r; ++r) {
        const int64_t start = ranges[2 * r];
        const int64_t end = ranges[2 * r + 1];
        if (start < 0 || start >= end || end > n_features) {
            return N4M_ERR_INVALID_ARGUMENT;
        }
    }
    std::vector<std::size_t> order(n_r);
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(),
              [ranges](std::size_t a, std::size_t b) { return ranges[2 * a] < ranges[2 * b]; });
    for (std::size_t k = 1; k < n_r; ++k) {
        if (ranges[2 * order[k]] < ranges[2 * order[k - 1] + 1]) {
            return N4M_ERR_INVALID_ARGUMENT;  // overlapping ranges
        }
    }
    const auto F = static_cast<std::size_t>(n_features);
    std::vector<int64_t> groups(F, -1);
    std::vector<int64_t> source(n_r);
    for (std::size_t r = 0; r < n_r; ++r) {
        source[r] = static_cast<int64_t>(r);
        for (int64_t j = ranges[2 * r]; j < ranges[2 * r + 1]; ++j) {
            groups[static_cast<std::size_t>(j)] = static_cast<int64_t>(r);
        }
    }
    int64_t next = n_ranges;
    for (std::size_t j = 0; j < F; ++j) {
        if (groups[j] >= 0) {
            continue;
        }
        if (j > 0 && groups[j - 1] >= n_ranges) {
            groups[j] = groups[j - 1];  // continue the current uncovered run
        } else {
            groups[j] = next++;
            source.push_back(-1);
        }
    }
    if (range_of_group != nullptr && range_of_group_size < next) {
        return N4M_ERR_SHAPE_MISMATCH;
    }
    std::copy(groups.begin(), groups.end(), feature_group);
    if (range_of_group != nullptr) {
        std::copy(source.begin(), source.end(), range_of_group);
    }
    *out_n_groups = next;
    return N4M_OK;
}

n4m_status_t summarize(const double* attributions, int64_t n_rows, int64_t n_groups,
                       int64_t n_outputs, int64_t output_index, double* mean_abs,
                       double* mean_signed, int64_t summary_size) {
    if (attributions == nullptr || mean_abs == nullptr || mean_signed == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    int64_t total = 0;
    if (n_rows < 1 || n_groups < 1 || n_outputs < 1 || output_index < 0
        || output_index >= n_outputs || !checked_mul(n_rows, n_groups, total)
        || !checked_mul(total, n_outputs, total)) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (summary_size < n_groups) {
        return N4M_ERR_SHAPE_MISMATCH;
    }
    const auto R = static_cast<std::size_t>(n_rows);
    const auto G = static_cast<std::size_t>(n_groups);
    const auto O = static_cast<std::size_t>(n_outputs);
    const auto o = static_cast<std::size_t>(output_index);
    std::vector<double> abs_sum(G, 0.0), signed_sum(G, 0.0);
    for (std::size_t r = 0; r < R; ++r) {
        for (std::size_t g = 0; g < G; ++g) {
            const double value = attributions[(r * G + g) * O + o];
            if (!std::isfinite(value)) {
                return N4M_ERR_INVALID_ARGUMENT;
            }
            abs_sum[g] += std::fabs(value);
            signed_sum[g] += value;
        }
    }
    for (std::size_t g = 0; g < G; ++g) {
        mean_abs[g] = abs_sum[g] / static_cast<double>(R);
        mean_signed[g] = signed_sum[g] / static_cast<double>(R);
    }
    return N4M_OK;
}

}  // namespace n4m::explain
