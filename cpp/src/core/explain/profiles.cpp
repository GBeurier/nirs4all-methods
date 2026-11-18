// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
//
// Importance-profile helpers (n4m/explain.h section 4, ABI 2.24): the
// Monte-Carlo standard error of a row-mean attribution, aggregation of a
// per-feature profile over a feature partition, and pairwise agreement of
// profiles (Spearman through the library's correlation primitive, top-K
// Jaccard overlap).

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

#include "core/explain/explain_internal.hpp"
#include "n4m/metrics.h"

namespace n4m::explain {

n4m_status_t summarize_std_error(const double* std_error, int64_t n_rows, int64_t n_groups,
                                 int64_t n_outputs, int64_t output_index, double* out,
                                 int64_t out_size) {
    if (std_error == nullptr || out == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    int64_t total = 0;
    if (n_rows < 1 || n_groups < 1 || n_outputs < 1 || output_index < 0
        || output_index >= n_outputs || !checked_mul(n_rows, n_groups, total)
        || !checked_mul(total, n_outputs, total)) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (out_size < n_groups) {
        return N4M_ERR_SHAPE_MISMATCH;
    }
    const auto R = static_cast<std::size_t>(n_rows);
    const auto G = static_cast<std::size_t>(n_groups);
    const auto O = static_cast<std::size_t>(n_outputs);
    const auto o = static_cast<std::size_t>(output_index);
    // Independent rows: Var(mean_r phi_r) = sum_r se_r^2 / R^2.
    std::vector<double> sum(G, 0.0);
    for (std::size_t r = 0; r < R; ++r) {
        for (std::size_t g = 0; g < G; ++g) {
            const double se = std_error[(r * G + g) * O + o];
            sum[g] += se * se;
        }
    }
    for (std::size_t g = 0; g < G; ++g) {
        out[g] = std::sqrt(sum[g]) / static_cast<double>(R);
    }
    return N4M_OK;
}

n4m_status_t aggregate_profile(n4m_context_t* ctx, const double* values, int64_t n_features,
                               const int64_t* feature_group, int64_t n_groups, int32_t rule,
                               double* out, int64_t out_size) {
    if (values == nullptr || out == nullptr) {
        return fail(ctx, N4M_ERR_NULL_POINTER, "values and out must be non-NULL");
    }
    if (n_features < 1) {
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "n_features must be >= 1");
    }
    if (rule < N4M_PROFILE_SUM || rule > N4M_PROFILE_MAX_ABS) {
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "unknown profile aggregation rule");
    }
    std::vector<std::size_t> group_of_feature;
    const n4m_status_t status =
        validate_groups(ctx, feature_group, n_features, n_groups, group_of_feature);
    if (status != N4M_OK) {
        return status;
    }
    if (out_size < n_groups) {
        return fail(ctx, N4M_ERR_SHAPE_MISMATCH, "out_size is smaller than n_groups");
    }
    const auto G = static_cast<std::size_t>(n_groups);
    for (std::size_t j = 0; j < group_of_feature.size(); ++j) {
        if (std::isnan(values[j])) {
            return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "profile values must not be NaN");
        }
    }
    std::vector<double> acc(G, 0.0);
    std::vector<std::size_t> count(G, 0);
    for (std::size_t j = 0; j < group_of_feature.size(); ++j) {
        const std::size_t g = group_of_feature[j];
        const double v = rule == N4M_PROFILE_SUM || rule == N4M_PROFILE_MEAN
                             ? values[j]
                             : std::fabs(values[j]);
        acc[g] = rule == N4M_PROFILE_MAX_ABS ? std::max(acc[g], v) : acc[g] + v;
        ++count[g];
    }
    for (std::size_t g = 0; g < G; ++g) {
        out[g] = rule == N4M_PROFILE_MEAN || rule == N4M_PROFILE_MEAN_ABS
                     ? acc[g] / static_cast<double>(count[g])
                     : acc[g];
    }
    return N4M_OK;
}

namespace {

// Dense ranks (1, 2, ... over the distinct values, ties equal): an order-
// and tie-preserving finite recoding, so Spearman's average ranks are those
// of the original values (+-inf included).
std::vector<double> dense_ranks(const std::vector<double>& values) {
    std::vector<std::size_t> order(values.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t a, std::size_t b) { return values[a] < values[b]; });
    std::vector<double> ranks(values.size(), 0.0);
    double rank = 0.0;
    for (std::size_t k = 0; k < order.size(); ++k) {
        if (k == 0 || values[order[k]] != values[order[k - 1]]) {
            rank += 1.0;
        }
        ranks[order[k]] = rank;
    }
    return ranks;
}

// The K entries of largest |value|, lowest index on ties.
std::vector<unsigned char> top_k_mask(const std::vector<double>& values, std::size_t k) {
    std::vector<std::size_t> order(values.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return std::fabs(values[a]) > std::fabs(values[b]);
    });
    std::vector<unsigned char> mask(values.size(), 0);
    for (std::size_t i = 0; i < k; ++i) {
        mask[order[i]] = 1;
    }
    return mask;
}

}  // namespace

n4m_status_t compare_profiles(n4m_context_t* ctx, const n4m_matrix_view_t* profiles,
                              int64_t top_k, double* spearman, int64_t spearman_size,
                              double* jaccard, int64_t jaccard_size) {
    n4m_status_t status = validate_f64_view(ctx, profiles, "profiles", 1);
    if (status != N4M_OK) {
        return status;
    }
    const int64_t m = profiles->rows;
    const int64_t G = profiles->cols;
    if (G < 2) {
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "profiles need at least 2 entries (columns)");
    }
    if (top_k < 1) {
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "top_k must be >= 1");
    }
    int64_t mm = 0;
    if (!checked_mul(m, m, mm)) {
        return fail(ctx, N4M_ERR_UNSUPPORTED, "profile count overflows");
    }
    if ((spearman != nullptr && spearman_size < mm) || (jaccard != nullptr && jaccard_size < mm)) {
        return fail(ctx, N4M_ERR_SHAPE_MISMATCH, "output matrices need m * m entries");
    }
    const auto M = static_cast<std::size_t>(m);
    const auto F = static_cast<std::size_t>(G);
    std::vector<std::vector<double>> rows(M, std::vector<double>(F));
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < F; ++j) {
            const double v = view_at(*profiles, static_cast<int64_t>(i), static_cast<int64_t>(j));
            if (std::isnan(v)) {
                return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "profiles must not contain NaN");
            }
            rows[i][j] = v;
        }
    }
    if (spearman != nullptr) {
        std::vector<std::vector<double>> ranks(M);
        for (std::size_t i = 0; i < M; ++i) {
            ranks[i] = dense_ranks(rows[i]);
        }
        std::vector<double> result(M * M, 0.0);
        for (std::size_t i = 0; i < M; ++i) {
            for (std::size_t k = i; k < M; ++k) {
                double five[5];
                int64_t two[2];
                status = n4m_stats_correlation_test(ctx, ranks[i].data(), ranks[k].data(), G, 1,
                                                    0, 0, five, two);
                if (status != N4M_OK) {
                    return status;
                }
                result[i * M + k] = five[0];
                result[k * M + i] = five[0];
            }
        }
        std::copy(result.begin(), result.end(), spearman);
    }
    if (jaccard != nullptr) {
        const auto K = static_cast<std::size_t>(std::min(top_k, G));
        std::vector<std::vector<unsigned char>> masks(M);
        for (std::size_t i = 0; i < M; ++i) {
            masks[i] = top_k_mask(rows[i], K);
        }
        for (std::size_t i = 0; i < M; ++i) {
            for (std::size_t k = 0; k < M; ++k) {
                std::size_t inter = 0;
                for (std::size_t j = 0; j < F; ++j) {
                    inter += static_cast<std::size_t>(masks[i][j] & masks[k][j]);
                }
                jaccard[i * M + k] =
                    static_cast<double>(inter) / static_cast<double>(2 * K - inter);
            }
        }
    }
    return N4M_OK;
}

}  // namespace n4m::explain
