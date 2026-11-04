// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
#include "core/component_analysis/fanova.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

#if N4M_WITH_RFR
#include "rfr/data_containers/default_data_container.hpp"
#include "rfr/forests/fanova_forest.hpp"
#include "rfr/splits/binary_split_one_feature_rss_loss.hpp"
#endif

namespace n4m::fanova {
#if N4M_WITH_RFR
namespace {
constexpr double nan = std::numeric_limits<double>::quiet_NaN();
using Rng = std::mt19937_64;
using Split = rfr::splits::binary_split_one_feature_rss_loss<double, double, unsigned, Rng>;
using Forest = rfr::forests::fANOVA_forest<Split, double, double, unsigned, Rng>;
using Data = rfr::data_containers::default_container<double, double, unsigned>;

double at(const n4m_matrix_view_t& v, int64_t i, int64_t j) {
    double value;
    const auto offset = static_cast<size_t>(i * v.row_stride + j * v.col_stride) * sizeof(double);
    std::memcpy(&value, static_cast<const unsigned char*>(v.data) + offset, sizeof(value));
    return value;
}

// Weighted, centered population moments. Long double avoids cancellation when
// a marginal mean is almost constant compared with its absolute level.
struct Moments {
    long double weight = 0, mean = 0, m2 = 0;
    void push(double value, double w) {
        const long double value_ld = static_cast<long double>(value);
        const long double w_ld = static_cast<long double>(w);
        const long double next = weight + w_ld;
        const long double delta = value_ld - mean;
        mean += delta * w_ld / next;
        m2 += w_ld * delta * (value_ld - mean);
        weight = next;
    }
    double variance() const { return static_cast<double>(m2 / weight); }
};
struct Cell { double value, weight; };

std::vector<Cell> grid(std::vector<double> cuts, uint32_t type) {
    std::vector<Cell> result;
    if (type) {
        for (uint32_t k = 0; k < type; ++k)
            result.push_back({static_cast<double>(k), 1.0 / type});
    } else {
        cuts.push_back(0);
        cuts.push_back(1);
        std::sort(cuts.begin(), cuts.end());
        cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
        for (size_t k = 1; k < cuts.size(); ++k) {
            const double width = cuts[k] - cuts[k - 1];
            if (width > 0)
                result.push_back({cuts[k - 1] + width / 2, width});
        }
    }
    return result;
}

double fraction(double value, double variance, bool& valid) {
    const double ratio = value / variance;
    if (!std::isfinite(ratio) || ratio < -1e-8 || ratio > 1 + 1e-8) {
        valid = false;
        return nan;
    }
    return std::clamp(ratio, 0.0, 1.0);
}

void aggregate(const std::vector<double>& values, double* output) {
    Moments stat;
    for (double x : values)
        stat.push(x, 1);
    output[0] = static_cast<double>(stat.mean);
    output[1] = values.size() > 1
        ? std::sqrt(stat.variance() * static_cast<double>(values.size())
            / static_cast<double>(values.size() - 1)) : nan;
}
}  // namespace
#endif

n4m_status_t fit(const n4m_matrix_view_t& X, const n4m_matrix_view_t& y,
                 const n4m_matrix_view_t& bounds, const uint32_t* types,
                 const n4m_matrix_view_t* X_validation, const n4m_matrix_view_t* y_validation,
                 const n4m_fanova_options_v1_t& options, Result& result) {
#if !N4M_WITH_RFR
    (void)X; (void)y; (void)bounds; (void)types; (void)X_validation;
    (void)y_validation; (void)options; (void)result;
    return N4M_ERR_BACKEND_UNAVAILABLE;
#else
    const size_t p = static_cast<size_t>(X.cols);
    result.main_effects.assign(p * 2, nan);
    result.pair_effects.assign(p * p * 2, nan);
    result.summary = {sizeof(n4m_fanova_summary_v1_t), N4M_FANOVA_CONSTANT_RESPONSE, 0,
                      X.rows, X.cols, 0, 0, 0, nan, nan, nan, nan, nan};
    auto& summary = result.summary;
    long double ymin = static_cast<long double>(at(y, 0, 0)), ymax = ymin;
    for (int64_t i = 1; i < y.rows; ++i) {
        ymin = std::min(ymin, static_cast<long double>(at(y, i, 0)));
        ymax = std::max(ymax, static_cast<long double>(at(y, i, 0)));
    }
    if (ymin == ymax)
        return N4M_OK;
    const long double center = ymin / 2 + ymax / 2;
    const long double scale = ymax / 2 - ymin / 2;
    if (!std::isfinite(scale) || scale <= 0) {
        summary.fit_state = N4M_FANOVA_NUMERICAL_REFUSAL;
        return N4M_OK;
    }
    for (size_t j = 0; j < p; ++j) {
        if (types[j] == 0) {
            const long double width = static_cast<long double>(at(bounds, static_cast<int64_t>(j), 1))
                - static_cast<long double>(at(bounds, static_cast<int64_t>(j), 0));
            if (!std::isfinite(width) || width <= 0) {
                summary.fit_state = N4M_FANOVA_NUMERICAL_REFUSAL;
                return N4M_OK;
            }
        }
    }
    // Affine changes of coordinates preserve the declared uniform product measure.
    // Unit continuous domains and centered/scaled y protect upstream split and
    // subspace arithmetic from overflowing on valid finite user data.
    auto features = [&](const n4m_matrix_view_t& v, int64_t i) {
        std::vector<double> row(p);
        for (size_t j = 0; j < p; ++j) {
            if (types[j])
                row[j] = at(v, i, static_cast<int64_t>(j));
            else {
                const long double lo = static_cast<long double>(at(bounds, static_cast<int64_t>(j), 0));
                const long double hi = static_cast<long double>(at(bounds, static_cast<int64_t>(j), 1));
                row[j] = static_cast<double>((static_cast<long double>(at(v, i, static_cast<int64_t>(j))) - lo)
                                             / (hi - lo));
            }
        }
        return row;
    };
    auto response = [&](double v) { return static_cast<double>((static_cast<long double>(v) - center) / scale); };
    Data data(static_cast<unsigned>(p));
    for (size_t j = 0; j < p; ++j) {
        if (types[j])
            data.set_type_of_feature(static_cast<unsigned>(j), types[j]);
        else
            data.set_bounds_of_feature(static_cast<unsigned>(j), 0, 1);
    }
    for (int64_t i = 0; i < X.rows; ++i)
        data.add_data_point(features(X, i), response(at(y, i, 0)));
    // A new instance on every call is essential: upstream fANOVA caches domains.
    Forest forest;
    forest.options.num_trees = static_cast<unsigned>(options.num_trees);
    forest.options.num_data_points_per_tree = static_cast<unsigned>(X.rows);
    forest.options.do_bootstrapping = true;
    forest.options.compute_oob_error = false;
    auto& tree = forest.options.tree_opts;
    tree.max_features = options.max_features ? static_cast<unsigned>(options.max_features)
                                              : static_cast<unsigned>(p);
    tree.max_depth = static_cast<unsigned>(options.max_depth);
    tree.min_samples_in_leaf = static_cast<unsigned>(options.min_samples_leaf);
    tree.min_weight_in_leaf = options.min_samples_leaf;
    tree.min_samples_to_split = static_cast<unsigned>(2 * options.min_samples_leaf);
    tree.min_weight_to_split = 2 * options.min_samples_leaf;
    tree.epsilon_purity = 0;
    Rng rng(options.seed);
    forest.fit(data, rng);
    forest.precompute_marginals();
    const auto variances = forest.get_trees_total_variances();
    const auto splits = forest.all_split_values();
    summary.trees = static_cast<int64_t>(variances.size());
    long double total_variance = 0;
    bool valid = true;
    std::vector<std::vector<double>> mains(p), pairs(p * p);
    std::vector<double> remainders;
    for (size_t t = 0; t < variances.size(); ++t) {
        const double variance = variances[t];
        if (!std::isfinite(variance) || variance < 0) {
            valid = false;
            break;
        }
        total_variance += static_cast<long double>(variance);
        if (variance == 0)
            continue;
        ++summary.informative_trees;
        std::vector<std::vector<Cell>> grids(p);
        int64_t needed = 0;
        for (size_t j = 0; j < p; ++j) {
            grids[j] = grid(splits[t][j], types[j]);
            if (!splits[t][j].empty())
                needed += static_cast<int64_t>(grids[j].size());
        }
        if (options.max_order == 2) {
            for (size_t j = 0; j < p; ++j)
                for (size_t k = j + 1; k < p; ++k)
                    if (!splits[t][j].empty() && !splits[t][k].empty())
                        needed += static_cast<int64_t>(grids[j].size() * grids[k].size());
        }
        if (needed > options.max_marginal_evaluations - summary.marginal_evaluations)
            return N4M_ERR_UNSUPPORTED;
        std::vector<double> query(p, nan), main_variance(p, 0);
        double accounted = 0;
        for (size_t j = 0; j < p; ++j) {
            Moments moment;
            if (!splits[t][j].empty()) {
                for (const auto& cell : grids[j]) {
                    query[j] = cell.value;
                    const double mean = forest.marginal_prediction_stat_of_tree(
                        static_cast<unsigned>(t), query).mean();
                    ++summary.marginal_evaluations;
                    moment.push(mean, cell.weight);
                }
                query[j] = nan;
                main_variance[j] = moment.variance();
            }
            const double f = fraction(main_variance[j], variance, valid);
            mains[j].push_back(f);
            accounted += f;
        }
        if (options.max_order == 2) {
            for (size_t j = 0; j < p; ++j) {
                for (size_t k = j + 1; k < p; ++k) {
                    double pair_variance = 0;
                    if (!splits[t][j].empty() && !splits[t][k].empty()) {
                        Moments moment;
                        for (const auto& a : grids[j]) {
                            query[j] = a.value;
                            for (const auto& b : grids[k]) {
                                query[k] = b.value;
                                const double mean = forest.marginal_prediction_stat_of_tree(
                                    static_cast<unsigned>(t), query).mean();
                                ++summary.marginal_evaluations;
                                moment.push(mean, a.weight * b.weight);
                            }
                        }
                        query[j] = query[k] = nan;
                        pair_variance = moment.variance() - main_variance[j] - main_variance[k];
                    }
                    const double f = fraction(pair_variance, variance, valid);
                    pairs[j * p + k].push_back(f);
                    accounted += f;
                }
            }
        }
        if (!std::isfinite(accounted) || accounted > 1 + 1e-7)
            valid = false;
        remainders.push_back(std::clamp(1 - accounted, 0.0, 1.0));
    }
    summary.mean_tree_variance = static_cast<double>(
        total_variance / static_cast<long double>(variances.size()) * scale * scale);
    if (!std::isfinite(summary.mean_tree_variance)
        || (total_variance > 0 && summary.mean_tree_variance == 0)) {
        summary.mean_tree_variance = nan;
        valid = false;
    }
    if (!valid) {
        summary.fit_state = N4M_FANOVA_NUMERICAL_REFUSAL;
        return N4M_OK;
    }
    if (!summary.informative_trees) {
        summary.fit_state = N4M_FANOVA_NO_INFORMATIVE_TREES;
        return N4M_OK;
    }
    auto rmse = [&](const n4m_matrix_view_t& vx, const n4m_matrix_view_t& vy,
                    double* r2) {
        long double squared_error = 0;
        Moments truth;
        for (int64_t i = 0; i < vx.rows; ++i) {
            const double actual = response(at(vy, i, 0));
            const double predicted = forest.predict(features(vx, i));
            const long double residual = static_cast<long double>(actual) - static_cast<long double>(predicted);
            squared_error += residual * residual;
            truth.push(actual, 1);
        }
        if (r2 && truth.m2 > 0)
            *r2 = static_cast<double>(1 - squared_error / truth.m2);
        return static_cast<double>(std::sqrt(squared_error / static_cast<long double>(vx.rows)) * scale);
    };
    summary.train_rmse = rmse(X, y, nullptr);
    if (X_validation)
        summary.validation_rmse = rmse(*X_validation, *y_validation, &summary.validation_r2);
    if (!std::isfinite(summary.train_rmse)
        || (X_validation && !std::isfinite(summary.validation_rmse))) {
        summary.fit_state = N4M_FANOVA_NUMERICAL_REFUSAL;
        return N4M_OK;
    }
    for (size_t j = 0; j < p; ++j)
        aggregate(mains[j], result.main_effects.data() + j * 2);
    if (options.max_order == 2) {
        for (size_t j = 0; j < p; ++j) {
            for (size_t k = j + 1; k < p; ++k) {
                aggregate(pairs[j * p + k], result.pair_effects.data() + (j * p + k) * 2);
                std::copy_n(result.pair_effects.data() + (j * p + k) * 2, 2,
                            result.pair_effects.data() + (k * p + j) * 2);
            }
        }
    }
    summary.remainder_fraction = std::accumulate(remainders.begin(), remainders.end(), 0.0)
        / static_cast<double>(remainders.size());
    summary.fit_state = N4M_FANOVA_FITTED;
    return N4M_OK;
#endif
}
}  // namespace n4m::fanova
