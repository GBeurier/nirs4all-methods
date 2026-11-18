// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
//
// Ask/tell model-agnostic explainers: exact, permutation and Kernel SHAP of
// the group game, LIME over group masks and occlusion. See
// explain_session.hpp for the plan layout and docs/EXPLAIN.md for the
// estimators and their evaluation counts.

#include "core/explain/explain_session.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <utility>
#include <vector>

#include "core/common/context.hpp"
#include "core/explain/explain_internal.hpp"

namespace n4m::explain {
namespace {

constexpr int64_t kMaxExactGroups = 16;
constexpr int64_t kKernelAutoExtra = 2048;
constexpr int64_t kLimeAutoSamples = 1000;

std::size_t popcount(uint64_t bits) noexcept {
    std::size_t count = 0;
    while (bits != 0) {
        bits &= bits - 1;
        ++count;
    }
    return count;
}

double binomial(std::size_t n, std::size_t k) noexcept {
    if (k > n) {
        return 0.0;
    }
    double result = 1.0;
    for (std::size_t i = 1; i <= k; ++i) {
        result = result * static_cast<double>(n - k + i) / static_cast<double>(i);
    }
    return result;
}

std::size_t to_size(int64_t value) noexcept {
    return static_cast<std::size_t>(value);
}

}  // namespace

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

n4m_status_t Session::create(n4m_context_t* ctx, const n4m_matrix_view_t* X,
                             const n4m_matrix_view_t* background, const int64_t* feature_group,
                             int64_t n_groups, const n4m_explain_options_v1_t* options,
                             n4m_explain_session_t*& out) {
    out = nullptr;
    if (options == nullptr) {
        return fail(ctx, N4M_ERR_NULL_POINTER, "options is NULL");
    }
    if (options->struct_size != sizeof(n4m_explain_options_v1_t)) {
        return fail(ctx, N4M_ERR_ABI_MISMATCH,
                    "options.struct_size does not match n4m_explain_options_v1_t");
    }
    const int32_t method = options->method;
    const int32_t baseline = options->baseline;
    if (method < N4M_EXPLAIN_EXACT_SHAP || method > N4M_EXPLAIN_GRADIENT) {
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "unknown explanation method");
    }
    if (baseline < N4M_EXPLAIN_BASELINE_BACKGROUND
        || baseline > N4M_EXPLAIN_BASELINE_INTERPOLATE) {
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "unknown explanation baseline");
    }
    if (method == N4M_EXPLAIN_GRADIENT && baseline == N4M_EXPLAIN_BASELINE_INTERPOLATE) {
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT,
                    "GRADIENT uses the background mean as reference: choose the BACKGROUND or "
                    "MEAN baseline");
    }
    n4m_status_t status = validate_f64_view(ctx, X, "X", 1);
    if (status != N4M_OK) {
        return status;
    }
    const int64_t n_features = X->cols;
    if (options->n_outputs < 1) {
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "options.n_outputs must be >= 1");
    }
    if (options->max_batch_rows < 1) {
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "options.max_batch_rows must be >= 1");
    }
    const bool uses_background = baseline != N4M_EXPLAIN_BASELINE_INTERPOLATE;
    if (uses_background) {
        status = validate_f64_view(ctx, background, "background", 1);
        if (status != N4M_OK) {
            return status;
        }
        if (background->cols != n_features) {
            return fail(ctx, N4M_ERR_SHAPE_MISMATCH,
                        "background must have the same number of columns as X");
        }
    }
    std::vector<std::size_t> group_of_feature;
    status = validate_groups(ctx, feature_group, n_features, n_groups, group_of_feature);
    if (status != N4M_OK) {
        return status;
    }
    const auto G = to_size(n_groups);
    if (method == N4M_EXPLAIN_EXACT_SHAP && n_groups > kMaxExactGroups) {
        return fail(ctx, N4M_ERR_UNSUPPORTED, "EXACT_SHAP is limited to n_groups <= 16");
    }
    if (method == N4M_EXPLAIN_PERMUTATION_SHAP && options->n_permutations < 1) {
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "options.n_permutations must be >= 1");
    }
    if ((method == N4M_EXPLAIN_KERNEL_SHAP || method == N4M_EXPLAIN_LIME)
        && options->n_coalitions < 0) {
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "options.n_coalitions must be >= 0");
    }
    if (method == N4M_EXPLAIN_LIME
        && (!std::isfinite(options->lime_ridge) || options->lime_ridge < 0.0
            || !std::isfinite(options->lime_kernel_width))) {
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT,
                    "options.lime_ridge must be finite and >= 0; lime_kernel_width finite");
    }
    if (baseline == N4M_EXPLAIN_BASELINE_INTERPOLATE) {
        std::vector<std::size_t> lo(G, std::numeric_limits<std::size_t>::max());
        std::vector<std::size_t> hi(G, 0);
        std::vector<std::size_t> count(G, 0);
        for (std::size_t j = 0; j < group_of_feature.size(); ++j) {
            const std::size_t g = group_of_feature[j];
            lo[g] = std::min(lo[g], j);
            hi[g] = std::max(hi[g], j);
            ++count[g];
        }
        for (std::size_t g = 0; g < G; ++g) {
            if (hi[g] - lo[g] + 1 != count[g]) {
                return fail(ctx, N4M_ERR_INVALID_ARGUMENT,
                            "INTERPOLATE requires every group to be a contiguous feature range");
            }
        }
    }

    auto session = std::make_unique<n4m_explain_session_s>();
    Session& s = *session;
    s.method_ = method;
    s.baseline_ = baseline;
    s.n_rows_ = to_size(X->rows);
    s.n_features_ = to_size(n_features);
    s.n_groups_ = G;
    s.n_outputs_ = to_size(options->n_outputs);
    s.max_batch_rows_ = to_size(options->max_batch_rows);
    s.group_of_feature_ = std::move(group_of_feature);
    if (!copy_finite_rowmajor(*X, s.x_)) {
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "X must contain only finite values");
    }
    if (uses_background) {
        std::vector<double> bg;
        if (!copy_finite_rowmajor(*background, bg)) {
            return fail(ctx, N4M_ERR_INVALID_ARGUMENT,
                        "background must contain only finite values");
        }
        const auto n_bg = to_size(background->rows);
        if (method == N4M_EXPLAIN_GRADIENT) {
            // Mean reference r (both background baselines), and the
            // per-feature scale of the finite-difference steps.
            const std::size_t F = s.n_features_;
            s.baseline_ = N4M_EXPLAIN_BASELINE_MEAN;
            s.reference_.assign(F, 0.0);
            s.gradient_scale_.assign(F, 0.0);
            for (std::size_t b = 0; b < n_bg; ++b) {
                for (std::size_t j = 0; j < F; ++j) {
                    s.reference_[j] += bg[b * F + j];
                }
            }
            for (double& v : s.reference_) {
                v /= static_cast<double>(n_bg);
            }
            for (std::size_t b = 0; b < n_bg; ++b) {
                for (std::size_t j = 0; j < F; ++j) {
                    const double d = bg[b * F + j] - s.reference_[j];
                    s.gradient_scale_[j] += d * d;
                }
            }
            for (std::size_t j = 0; j < F; ++j) {
                double scale = std::sqrt(s.gradient_scale_[j] / static_cast<double>(n_bg));
                if (!(scale > 0.0)) {
                    scale = s.reference_[j] != 0.0 ? std::fabs(s.reference_[j]) : 1.0;
                }
                s.gradient_scale_[j] = scale;
            }
            s.update_gradient_steps(s.gradient_step_);
            s.n_base_ = 1;
        } else if (baseline == N4M_EXPLAIN_BASELINE_BACKGROUND) {
            s.background_ = std::move(bg);
            s.n_base_ = n_bg;
        } else {
            s.reference_.assign(s.n_features_, 0.0);
            for (std::size_t b = 0; b < n_bg; ++b) {
                for (std::size_t j = 0; j < s.n_features_; ++j) {
                    s.reference_[j] += bg[b * s.n_features_ + j];
                }
            }
            for (double& v : s.reference_) {
                v /= static_cast<double>(n_bg);
            }
            s.n_base_ = 1;
        }
    }

    const bool needs_empty = method != N4M_EXPLAIN_LIME;
    s.global_empty_ = needs_empty && uses_background;
    s.row_empty_ = needs_empty && !uses_background;
    s.first_method_coalition_ = s.row_empty_ ? 2 : 1;

    const int64_t g64 = n_groups;
    int64_t n_method = 0;
    int64_t per_row_mask_bytes = 0;
    switch (method) {
        case N4M_EXPLAIN_EXACT_SHAP:
            n_method = (int64_t{1} << g64) - 2;
            break;
        case N4M_EXPLAIN_PERMUTATION_SHAP:
            s.n_permutations_ = to_size(options->n_permutations);
            if (!checked_mul(2, g64 - 1, n_method)
                || !checked_mul(options->n_permutations, n_method, n_method)
                || !checked_mul(options->n_permutations, g64, per_row_mask_bytes)) {
                return fail(ctx, N4M_ERR_UNSUPPORTED, "permutation plan size overflows");
            }
            break;
        case N4M_EXPLAIN_KERNEL_SHAP: {
            const int64_t full_enum = g64 <= 62 ? (int64_t{1} << g64) - 2
                                                : std::numeric_limits<int64_t>::max();
            int64_t requested = options->n_coalitions;
            if (requested == 0) {
                int64_t budget = 0;
                if (!checked_mul(2, g64, budget)
                    || !checked_add(budget, kKernelAutoExtra, budget)) {
                    return fail(ctx, N4M_ERR_UNSUPPORTED, "kernel plan size overflows");
                }
                requested = std::min(full_enum, budget);
            }
            if (requested >= full_enum) {
                s.kernel_enumerate_ = true;
                n_method = full_enum;
            } else {
                n_method = 2 * (requested / 2 + requested % 2);
                if (!checked_mul(n_method, g64, per_row_mask_bytes)) {
                    return fail(ctx, N4M_ERR_UNSUPPORTED, "kernel plan size overflows");
                }
                // Total Shapley-kernel weight of coalition size k: (G-1)/(k(G-k)).
                double total = 0.0;
                for (std::size_t k = 1; k < G; ++k) {
                    total += static_cast<double>(G - 1)
                             / (static_cast<double>(k) * static_cast<double>(G - k));
                    s.kernel_size_cdf_.push_back(total);
                }
            }
            break;
        }
        case N4M_EXPLAIN_LIME: {
            const int64_t samples =
                options->n_coalitions == 0 ? kLimeAutoSamples : options->n_coalitions;
            n_method = samples - 1;
            if (!checked_mul(n_method, g64, per_row_mask_bytes)) {
                return fail(ctx, N4M_ERR_UNSUPPORTED, "LIME plan size overflows");
            }
            s.lime_width_ = options->lime_kernel_width > 0.0
                                ? options->lime_kernel_width
                                : 0.75 * std::sqrt(static_cast<double>(G));
            s.lime_ridge_ = options->lime_ridge;
            break;
        }
        case N4M_EXPLAIN_GRADIENT:
            // Coalition 1 + 2j: x + h_j e_j; coalition 2 + 2j: x - h_j e_j.
            if (!checked_mul(2, n_features, n_method)) {
                return fail(ctx, N4M_ERR_UNSUPPORTED, "gradient plan size overflows");
            }
            break;
        default:  // N4M_EXPLAIN_OCCLUSION
            n_method = g64;
            break;
    }
    (void)per_row_mask_bytes;  // overflow-checked above; masks are allocated per row
    s.n_method_coalitions_ = to_size(n_method);
    int64_t coalitions = 0;
    int64_t values = 0;
    int64_t per_row = 0;
    int64_t total = 0;
    int64_t output_size = 0;
    const auto n_base = static_cast<int64_t>(s.n_base_);
    if (!checked_add(static_cast<int64_t>(s.first_method_coalition_), n_method, coalitions)
        || !checked_mul(coalitions, options->n_outputs, values)
        || !checked_mul(coalitions - 1, n_base, per_row) || !checked_add(per_row, 1, per_row)
        || !checked_mul(per_row, X->rows, total)
        || !checked_add(total, s.global_empty_ ? n_base : 0, total)
        || !checked_mul(X->rows, g64, output_size)
        || !checked_mul(output_size, options->n_outputs, output_size)) {
        return fail(ctx, N4M_ERR_UNSUPPORTED, "explanation plan size overflows int64");
    }
    s.coalitions_per_row_ = to_size(coalitions);
    s.total_evaluations_ = total;
    s.attributions_.assign(to_size(output_size), 0.0);
    s.expected_.assign(s.n_rows_ * s.n_outputs_, 0.0);
    s.predictions_.assign(s.n_rows_ * s.n_outputs_, 0.0);
    if (method == N4M_EXPLAIN_PERMUTATION_SHAP) {
        s.std_error_.assign(to_size(output_size), 0.0);
    }
    if (method == N4M_EXPLAIN_GRADIENT) {
        int64_t gradient_size = 0;
        if (!checked_mul(X->rows, n_features, gradient_size)
            || !checked_mul(gradient_size, options->n_outputs, gradient_size)) {
            return fail(ctx, N4M_ERR_UNSUPPORTED, "gradient result size overflows int64");
        }
        s.gradients_.assign(to_size(gradient_size), 0.0);
    }
    s.global_empty_value_.assign(s.n_outputs_, 0.0);
    s.group_mask_.assign(G, 0);
    s.present_.assign(s.n_features_, 0);
    s.subset_scratch_.resize(G);
    for (std::size_t g = 0; g < G; ++g) {
        s.subset_scratch_[g] = g;
    }
    n4m_pcg64_engine_seed(&s.rng_, options->seed);
    s.emit_ = s.start_cursor();
    s.state_ = State::kReady;
    out = session.release();
    return N4M_OK;
}

// ---------------------------------------------------------------------------
// Plan walk
// ---------------------------------------------------------------------------

Session::Cursor Session::start_cursor() const noexcept {
    Cursor cursor;
    cursor.row = global_empty_ ? -1 : 0;
    return cursor;
}

bool Session::at_end(const Cursor& cursor) const noexcept {
    return cursor.row >= static_cast<int64_t>(n_rows_);
}

void Session::advance(Cursor& cursor) const noexcept {
    if (cursor.row < 0) {
        if (++cursor.base >= n_base_) {
            cursor.row = 0;
            cursor.coalition = 0;
            cursor.base = 0;
        }
        return;
    }
    const std::size_t evaluations = cursor.coalition == 0 ? 1 : n_base_;
    if (++cursor.base < evaluations) {
        return;
    }
    cursor.base = 0;
    if (++cursor.coalition < coalitions_per_row_) {
        return;
    }
    cursor.coalition = 0;
    ++cursor.row;
}

void Session::sample_subset(std::size_t size, unsigned char* mask, unsigned char value) {
    // Partial Fisher-Yates over a persistent permutation of the group indices.
    const std::size_t G = n_groups_;
    for (std::size_t i = 0; i < size; ++i) {
        const auto offset =
            static_cast<std::size_t>(n4m_pcg64_engine_interval(&rng_, G - 1 - i));
        std::swap(subset_scratch_[i], subset_scratch_[i + offset]);
        mask[subset_scratch_[i]] = value;
    }
}

Session::RowPlan Session::make_plan() {
    RowPlan plan;
    plan.values.assign(coalitions_per_row_ * n_outputs_, 0.0);
    const std::size_t G = n_groups_;
    if (method_ == N4M_EXPLAIN_PERMUTATION_SHAP) {
        plan.perms.resize(n_permutations_ * G);
        std::vector<int64_t> perm(G);
        for (std::size_t d = 0; d < n_permutations_; ++d) {
            n4m_pcg64_engine_permutation(&rng_, perm.data(), static_cast<int64_t>(G));
            for (std::size_t g = 0; g < G; ++g) {
                plan.perms[d * G + g] = to_size(perm[g]);
            }
        }
    } else if (method_ == N4M_EXPLAIN_KERNEL_SHAP && !kernel_enumerate_) {
        plan.masks.assign(n_method_coalitions_ * G, 0);
        const double total = kernel_size_cdf_.back();
        for (std::size_t p = 0; p < n_method_coalitions_ / 2; ++p) {
            const double u = n4m_pcg64_engine_next_double(&rng_) * total;
            const auto it = std::upper_bound(kernel_size_cdf_.begin(), kernel_size_cdf_.end(), u);
            std::size_t size = static_cast<std::size_t>(it - kernel_size_cdf_.begin()) + 1;
            size = std::min(size, G - 1);
            unsigned char* mask = &plan.masks[2 * p * G];
            unsigned char* complement = mask + G;
            sample_subset(size, mask, 1);
            for (std::size_t g = 0; g < G; ++g) {
                complement[g] = static_cast<unsigned char>(1 - mask[g]);
            }
        }
    } else if (method_ == N4M_EXPLAIN_LIME) {
        plan.masks.assign(n_method_coalitions_ * G, 1);
        for (std::size_t i = 0; i < n_method_coalitions_; ++i) {
            const auto absent =
                1 + static_cast<std::size_t>(n4m_pcg64_engine_interval(&rng_, G - 1));
            sample_subset(absent, &plan.masks[i * G], 0);
        }
    }
    return plan;
}

void Session::coalition_mask(const RowPlan& plan, std::size_t coalition,
                             unsigned char* mask) const {
    const std::size_t G = n_groups_;
    if (coalition == 0) {
        std::fill(mask, mask + G, static_cast<unsigned char>(1));
        return;
    }
    if (row_empty_ && coalition == 1) {
        std::fill(mask, mask + G, static_cast<unsigned char>(0));
        return;
    }
    const std::size_t i = coalition - first_method_coalition_;
    const bool by_bits = method_ == N4M_EXPLAIN_EXACT_SHAP
                         || (method_ == N4M_EXPLAIN_KERNEL_SHAP && kernel_enumerate_);
    if (by_bits) {
        const auto bits = static_cast<uint64_t>(i) + 1;
        for (std::size_t g = 0; g < G; ++g) {
            mask[g] = static_cast<unsigned char>((bits >> g) & 1U);
        }
        return;
    }
    switch (method_) {
        case N4M_EXPLAIN_PERMUTATION_SHAP: {
            const std::size_t chain = G - 1;
            const std::size_t draw = i / (2 * chain);
            const std::size_t rem = i % (2 * chain);
            const bool reverse = rem >= chain;
            const std::size_t k = rem % chain + 1;
            const std::size_t* perm = &plan.perms[draw * G];
            std::fill(mask, mask + G, static_cast<unsigned char>(reverse ? 1 : 0));
            for (std::size_t t = 0; t < k; ++t) {
                mask[perm[t]] = static_cast<unsigned char>(reverse ? 0 : 1);
            }
            return;
        }
        case N4M_EXPLAIN_OCCLUSION:
            std::fill(mask, mask + G, static_cast<unsigned char>(1));
            mask[i] = 0;
            return;
        default:  // sampled KERNEL, LIME
            std::memcpy(mask, &plan.masks[i * G], G);
            return;
    }
}

void Session::interpolate(const double* x, const unsigned char* present, double* out) const {
    const std::size_t F = n_features_;
    std::size_t j = 0;
    while (j < F) {
        if (present[j] != 0) {
            out[j] = x[j];
            ++j;
            continue;
        }
        const std::size_t a = j;
        while (j < F && present[j] == 0) {
            ++j;
        }
        const std::size_t e = j;  // maximal absent run [a, e)
        if (a == 0 && e == F) {
            if (F == 1) {
                out[0] = x[0];
            } else {
                const double slope = (x[F - 1] - x[0]) / static_cast<double>(F - 1);
                for (std::size_t k = 0; k < F; ++k) {
                    out[k] = x[0] + slope * static_cast<double>(k);
                }
            }
        } else if (a == 0) {
            std::fill(out, out + e, x[e]);
        } else if (e == F) {
            std::fill(out + a, out + F, x[a - 1]);
        } else {
            const double lo = x[a - 1];
            const double span = static_cast<double>(e - a + 1);
            const double step = (x[e] - lo) / span;
            for (std::size_t k = a; k < e; ++k) {
                out[k] = lo + step * static_cast<double>(k - a + 1);
            }
        }
    }
}

void Session::fill_row(const Cursor& cursor, const RowPlan* plan, double* out) {
    const std::size_t F = n_features_;
    if (cursor.row < 0) {
        const double* source = baseline_ == N4M_EXPLAIN_BASELINE_BACKGROUND
                                   ? &background_[cursor.base * F]
                                   : reference_.data();
        std::copy(source, source + F, out);
        return;
    }
    const double* x = &x_[to_size(cursor.row) * F];
    if (cursor.coalition == 0) {
        std::copy(x, x + F, out);
        return;
    }
    if (method_ == N4M_EXPLAIN_GRADIENT) {
        const std::size_t k = cursor.coalition - first_method_coalition_;
        const std::size_t j = k / 2;
        std::copy(x, x + F, out);
        out[j] = k % 2 == 0 ? x[j] + gradient_steps_[j] : x[j] - gradient_steps_[j];
        return;
    }
    if (cached_row_ != cursor.row || cached_coalition_ != cursor.coalition) {
        coalition_mask(*plan, cursor.coalition, group_mask_.data());
        for (std::size_t j = 0; j < F; ++j) {
            present_[j] = group_mask_[group_of_feature_[j]];
        }
        cached_row_ = cursor.row;
        cached_coalition_ = cursor.coalition;
    }
    fill_present(x, cursor.base, out);
}

void Session::fill_present(const double* x, std::size_t base, double* out) const {
    const std::size_t F = n_features_;
    switch (baseline_) {
        case N4M_EXPLAIN_BASELINE_BACKGROUND: {
            const double* b = &background_[base * F];
            for (std::size_t j = 0; j < F; ++j) {
                out[j] = present_[j] != 0 ? x[j] : b[j];
            }
            return;
        }
        case N4M_EXPLAIN_BASELINE_MEAN:
            for (std::size_t j = 0; j < F; ++j) {
                out[j] = present_[j] != 0 ? x[j] : reference_[j];
            }
            return;
        default:
            interpolate(x, present_.data(), out);
            return;
    }
}

// ---------------------------------------------------------------------------
// Ask / tell
// ---------------------------------------------------------------------------

void Session::info(int64_t* n_rows, int64_t* n_features, int64_t* n_groups, int64_t* n_outputs,
                   int64_t* total_evaluations) const noexcept {
    if (n_rows != nullptr) {
        *n_rows = static_cast<int64_t>(n_rows_);
    }
    if (n_features != nullptr) {
        *n_features = static_cast<int64_t>(n_features_);
    }
    if (n_groups != nullptr) {
        *n_groups = static_cast<int64_t>(n_groups_);
    }
    if (n_outputs != nullptr) {
        *n_outputs = static_cast<int64_t>(n_outputs_);
    }
    if (total_evaluations != nullptr) {
        *total_evaluations = total_evaluations_;
    }
}

n4m_status_t Session::next_batch(n4m_context_t* ctx, double* out_rows, int64_t capacity_rows,
                                 int64_t* out_rows_written) {
    if (state_ == State::kPoisoned) {
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT,
                    "explanation session was poisoned by an earlier failure");
    }
    if (out_rows_written == nullptr) {
        return fail(ctx, N4M_ERR_NULL_POINTER, "out_rows_written is NULL");
    }
    if (state_ == State::kPending) {
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT,
                    "submit the pending batch before requesting another one");
    }
    if (capacity_rows < 1) {
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "capacity_rows must be >= 1");
    }
    if (state_ == State::kComplete || at_end(emit_)) {
        *out_rows_written = 0;
        return N4M_OK;
    }
    if (out_rows == nullptr) {
        return fail(ctx, N4M_ERR_NULL_POINTER, "out_rows is NULL");
    }
    const std::size_t limit = std::min(to_size(capacity_rows), max_batch_rows_);
    try {
        batch_start_ = emit_;
        std::size_t written = 0;
        while (written < limit && !at_end(emit_)) {
            if (emit_.row >= 0 && emit_.coalition == 0 && emit_.base == 0) {
                plans_.push_back(make_plan());
            }
            const RowPlan* plan = emit_.row >= 0 ? &plans_.back() : nullptr;
            fill_row(emit_, plan, out_rows + written * n_features_);
            advance(emit_);
            ++written;
        }
        pending_rows_ = written;
        state_ = State::kPending;
        *out_rows_written = static_cast<int64_t>(written);
        return N4M_OK;
    } catch (...) {
        state_ = State::kPoisoned;
        throw;
    }
}

n4m_status_t Session::submit(n4m_context_t* ctx, const double* predictions, int64_t rows) {
    if (state_ == State::kPoisoned) {
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT,
                    "explanation session was poisoned by an earlier failure");
    }
    if (state_ != State::kPending) {
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "no pending batch to submit");
    }
    if (rows < 0 || to_size(rows) != pending_rows_) {
        return fail(ctx, N4M_ERR_SHAPE_MISMATCH,
                    "submitted row count differs from the pending batch");
    }
    if (predictions == nullptr) {
        return fail(ctx, N4M_ERR_NULL_POINTER, "predictions is NULL");
    }
    const std::size_t O = n_outputs_;
    const std::size_t count = pending_rows_ * O;
    for (std::size_t i = 0; i < count; ++i) {
        if (!std::isfinite(predictions[i])) {
            state_ = State::kPoisoned;
            return fail(ctx, N4M_ERR_NUMERICAL_FAILURE,
                        "nonfinite prediction submitted; the session is poisoned");
        }
    }
    try {
        Cursor cursor = batch_start_;
        for (std::size_t k = 0; k < pending_rows_; ++k) {
            const double* p = predictions + k * O;
            if (cursor.row < 0) {
                for (std::size_t o = 0; o < O; ++o) {
                    global_empty_value_[o] += p[o];
                }
            } else {
                double* v = &plans_.front().values[cursor.coalition * O];
                for (std::size_t o = 0; o < O; ++o) {
                    v[o] += p[o];
                }
            }
            const int64_t row = cursor.row;
            advance(cursor);
            if (cursor.row == row) {
                continue;
            }
            if (row < 0) {
                for (double& value : global_empty_value_) {
                    value /= static_cast<double>(n_base_);
                }
                continue;
            }
            const n4m_status_t status = finalize_row(ctx, to_size(row), plans_.front());
            plans_.pop_front();
            if (status != N4M_OK) {
                state_ = State::kPoisoned;
                return status;
            }
        }
        pending_rows_ = 0;
        state_ = at_end(cursor) ? State::kComplete : State::kReady;
        return N4M_OK;
    } catch (...) {
        state_ = State::kPoisoned;
        throw;
    }
}

n4m_status_t Session::result(n4m_context_t* ctx, double* attributions,
                             int64_t attributions_size, double* expected, int64_t expected_size,
                             double* predictions, int64_t predictions_size, double* std_error,
                             int64_t std_error_size) const {
    if (state_ == State::kPoisoned) {
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT,
                    "explanation session was poisoned by an earlier failure");
    }
    if (state_ != State::kComplete) {
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT,
                    "explanation plan is not complete; keep calling next_batch/submit");
    }
    if (attributions == nullptr || expected == nullptr || predictions == nullptr) {
        return fail(ctx, N4M_ERR_NULL_POINTER,
                    "attributions, expected and predictions must be non-NULL");
    }
    const auto n_attr = static_cast<int64_t>(attributions_.size());
    const auto n_out = static_cast<int64_t>(expected_.size());
    if (attributions_size < n_attr || expected_size < n_out || predictions_size < n_out
        || (std_error != nullptr && std_error_size < n_attr)) {
        return fail(ctx, N4M_ERR_SHAPE_MISMATCH,
                    "result buffers are smaller than n_rows * n_groups * n_outputs "
                    "(attributions, std_error) or n_rows * n_outputs");
    }
    std::copy(attributions_.begin(), attributions_.end(), attributions);
    std::copy(expected_.begin(), expected_.end(), expected);
    std::copy(predictions_.begin(), predictions_.end(), predictions);
    if (std_error != nullptr) {
        if (std_error_.empty()) {
            std::fill(std_error, std_error + n_attr, std::numeric_limits<double>::quiet_NaN());
        } else {
            std::copy(std_error_.begin(), std_error_.end(), std_error);
        }
    }
    return N4M_OK;
}

// ---------------------------------------------------------------------------
// Per-row estimators
// ---------------------------------------------------------------------------

double Session::value(const RowPlan& plan, std::size_t coalition,
                      std::size_t output) const noexcept {
    const double sum = plan.values[coalition * n_outputs_ + output];
    return coalition == 0 ? sum : sum / static_cast<double>(n_base_);
}

double Session::empty_value(const RowPlan& plan, std::size_t output) const noexcept {
    return row_empty_ ? value(plan, 1, output) : global_empty_value_[output];
}

n4m_status_t Session::finalize_row(n4m_context_t* ctx, std::size_t row, const RowPlan& plan) {
    for (std::size_t o = 0; o < n_outputs_; ++o) {
        predictions_[row * n_outputs_ + o] = value(plan, 0, o);
    }
    switch (method_) {
        case N4M_EXPLAIN_EXACT_SHAP:
            finalize_exact(row, plan);
            return N4M_OK;
        case N4M_EXPLAIN_PERMUTATION_SHAP:
            finalize_permutation(row, plan);
            return N4M_OK;
        case N4M_EXPLAIN_KERNEL_SHAP:
            return finalize_kernel(ctx, row, plan);
        case N4M_EXPLAIN_LIME:
            return finalize_lime(ctx, row, plan);
        case N4M_EXPLAIN_GRADIENT:
            finalize_gradient(row, plan);
            return N4M_OK;
        default:
            finalize_occlusion(row, plan);
            return N4M_OK;
    }
}

void Session::finalize_exact(std::size_t row, const RowPlan& plan) {
    const std::size_t G = n_groups_;
    const std::size_t O = n_outputs_;
    const std::size_t n_masks = std::size_t{1} << G;
    std::vector<double> v(n_masks * O, 0.0);
    for (std::size_t o = 0; o < O; ++o) {
        v[o] = empty_value(plan, o);
        v[(n_masks - 1) * O + o] = value(plan, 0, o);
    }
    for (std::size_t m = 1; m + 1 < n_masks; ++m) {
        const std::size_t coalition = first_method_coalition_ + m - 1;
        for (std::size_t o = 0; o < O; ++o) {
            v[m * O + o] = value(plan, coalition, o);
        }
    }
    // Shapley weight |S|! (G - |S| - 1)! / G! = 1 / (G * C(G - 1, |S|)).
    std::vector<double> weight(G, 0.0);
    for (std::size_t s = 0; s < G; ++s) {
        weight[s] = 1.0 / (static_cast<double>(G) * binomial(G - 1, s));
    }
    double* phi = &attributions_[row * G * O];
    for (std::size_t g = 0; g < G; ++g) {
        const std::size_t bit = std::size_t{1} << g;
        for (std::size_t m = 0; m < n_masks; ++m) {
            if ((m & bit) != 0) {
                continue;
            }
            const double w = weight[popcount(m)];
            for (std::size_t o = 0; o < O; ++o) {
                phi[g * O + o] += w * (v[(m | bit) * O + o] - v[m * O + o]);
            }
        }
    }
    for (std::size_t o = 0; o < O; ++o) {
        expected_[row * O + o] = v[o];
    }
}

void Session::finalize_permutation(std::size_t row, const RowPlan& plan) {
    const std::size_t G = n_groups_;
    const std::size_t O = n_outputs_;
    const std::size_t chain = G - 1;
    std::vector<double> full(O), empty(O), prev(O);
    for (std::size_t o = 0; o < O; ++o) {
        full[o] = value(plan, 0, o);
        empty[o] = empty_value(plan, o);
    }
    std::vector<double> sum(G * O, 0.0), mean(G * O, 0.0), m2(G * O, 0.0), est(G * O, 0.0);
    for (std::size_t d = 0; d < n_permutations_; ++d) {
        const std::size_t* perm = &plan.perms[d * G];
        const std::size_t forward = first_method_coalition_ + d * 2 * chain;
        const std::size_t reverse = forward + chain;
        std::fill(est.begin(), est.end(), 0.0);
        // Forward chain: add perm[0], perm[1], ... from the empty coalition.
        prev = empty;
        for (std::size_t k = 1; k <= chain; ++k) {
            const std::size_t g = perm[k - 1];
            for (std::size_t o = 0; o < O; ++o) {
                const double current = value(plan, forward + k - 1, o);
                est[g * O + o] += current - prev[o];
                prev[o] = current;
            }
        }
        for (std::size_t o = 0; o < O; ++o) {
            est[perm[G - 1] * O + o] += full[o] - prev[o];
        }
        // Reverse chain: remove perm[0], perm[1], ... from the full coalition.
        prev = full;
        for (std::size_t k = 1; k <= chain; ++k) {
            const std::size_t g = perm[k - 1];
            for (std::size_t o = 0; o < O; ++o) {
                const double current = value(plan, reverse + k - 1, o);
                est[g * O + o] += prev[o] - current;
                prev[o] = current;
            }
        }
        for (std::size_t o = 0; o < O; ++o) {
            est[perm[G - 1] * O + o] += prev[o] - empty[o];
        }
        const double n = static_cast<double>(d + 1);
        for (std::size_t i = 0; i < G * O; ++i) {
            const double e = 0.5 * est[i];
            sum[i] += e;
            const double delta = e - mean[i];
            mean[i] += delta / n;
            m2[i] += delta * (e - mean[i]);
        }
    }
    const double draws = static_cast<double>(n_permutations_);
    double* phi = &attributions_[row * G * O];
    double* se = &std_error_[row * G * O];
    for (std::size_t i = 0; i < G * O; ++i) {
        phi[i] = sum[i] / draws;
        se[i] = n_permutations_ > 1 ? std::sqrt(m2[i] / (draws - 1.0) / draws)
                                    : std::numeric_limits<double>::quiet_NaN();
    }
    for (std::size_t o = 0; o < O; ++o) {
        expected_[row * O + o] = empty[o];
    }
}

n4m_status_t Session::finalize_kernel(n4m_context_t* ctx, std::size_t row, const RowPlan& plan) {
    const std::size_t G = n_groups_;
    const std::size_t O = n_outputs_;
    std::vector<double> empty(O), delta(O);
    for (std::size_t o = 0; o < O; ++o) {
        empty[o] = empty_value(plan, o);
        delta[o] = value(plan, 0, o) - empty[o];
        expected_[row * O + o] = empty[o];
    }
    double* phi = &attributions_[row * G * O];
    if (G == 1) {
        for (std::size_t o = 0; o < O; ++o) {
            phi[o] = delta[o];
        }
        return N4M_OK;
    }
    // Efficiency constraint sum(phi) = delta eliminates the last group:
    // v(S) - v(empty) - z_last * delta = sum_{j < last} (z_j - z_last) phi_j.
    const std::size_t n = G - 1;
    std::vector<double> size_weight(G, 1.0);
    if (kernel_enumerate_) {
        for (std::size_t s = 1; s < G; ++s) {
            size_weight[s] = static_cast<double>(G - 1)
                             / (binomial(G, s) * static_cast<double>(s)
                                * static_cast<double>(G - s));
        }
    }
    std::vector<double> normal(n * n, 0.0), rhs(n * O, 0.0), a(n), t(O);
    std::vector<unsigned char> mask(G);
    for (std::size_t c = first_method_coalition_; c < coalitions_per_row_; ++c) {
        coalition_mask(plan, c, mask.data());
        std::size_t size = 0;
        for (std::size_t g = 0; g < G; ++g) {
            size += mask[g];
        }
        const double w = size_weight[size];
        const double z_last = mask[G - 1];
        for (std::size_t i = 0; i < n; ++i) {
            a[i] = static_cast<double>(mask[i]) - z_last;
        }
        for (std::size_t o = 0; o < O; ++o) {
            t[o] = value(plan, c, o) - empty[o] - z_last * delta[o];
        }
        for (std::size_t i = 0; i < n; ++i) {
            if (a[i] == 0.0) {
                continue;
            }
            const double wa = w * a[i];
            for (std::size_t j = 0; j < n; ++j) {
                normal[i * n + j] += wa * a[j];
            }
            for (std::size_t o = 0; o < O; ++o) {
                rhs[i * O + o] += wa * t[o];
            }
        }
    }
    std::vector<double> solution;
    bool ridged = false;
    if (solve_spd(std::move(normal), n, rhs, O, solution, ridged) != N4M_OK) {
        return fail(ctx, N4M_ERR_NUMERICAL_FAILURE,
                    "Kernel SHAP weighted least squares is singular");
    }
    for (std::size_t o = 0; o < O; ++o) {
        double partial = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            phi[i * O + o] = solution[i * O + o];
            partial += solution[i * O + o];
        }
        phi[n * O + o] = delta[o] - partial;
    }
    return N4M_OK;
}

n4m_status_t Session::finalize_lime(n4m_context_t* ctx, std::size_t row, const RowPlan& plan) {
    const std::size_t G = n_groups_;
    const std::size_t O = n_outputs_;
    const double width_sq = lime_width_ * lime_width_;
    std::vector<unsigned char> mask(G);
    // Data points: the full coalition (coalition 0) and every sampled mask.
    auto kernel = [&](const unsigned char* z) {
        std::size_t present = 0;
        for (std::size_t g = 0; g < G; ++g) {
            present += z[g];
        }
        const double distance =
            1.0 - std::sqrt(static_cast<double>(present) / static_cast<double>(G));
        return std::sqrt(std::exp(-(distance * distance) / width_sq));
    };
    double weight_sum = 0.0;
    std::vector<double> z_mean(G, 0.0), y_mean(O, 0.0);
    for (std::size_t c = 0; c < coalitions_per_row_; ++c) {
        coalition_mask(plan, c, mask.data());
        const double w = kernel(mask.data());
        weight_sum += w;
        for (std::size_t g = 0; g < G; ++g) {
            z_mean[g] += w * mask[g];
        }
        for (std::size_t o = 0; o < O; ++o) {
            y_mean[o] += w * value(plan, c, o);
        }
    }
    for (double& v : z_mean) {
        v /= weight_sum;
    }
    for (double& v : y_mean) {
        v /= weight_sum;
    }
    std::vector<double> normal(G * G, 0.0), rhs(G * O, 0.0), zc(G), yc(O);
    for (std::size_t c = 0; c < coalitions_per_row_; ++c) {
        coalition_mask(plan, c, mask.data());
        const double w = kernel(mask.data());
        for (std::size_t g = 0; g < G; ++g) {
            zc[g] = static_cast<double>(mask[g]) - z_mean[g];
        }
        for (std::size_t o = 0; o < O; ++o) {
            yc[o] = value(plan, c, o) - y_mean[o];
        }
        for (std::size_t i = 0; i < G; ++i) {
            const double wz = w * zc[i];
            for (std::size_t j = 0; j < G; ++j) {
                normal[i * G + j] += wz * zc[j];
            }
            for (std::size_t o = 0; o < O; ++o) {
                rhs[i * O + o] += wz * yc[o];
            }
        }
    }
    for (std::size_t g = 0; g < G; ++g) {
        normal[g * G + g] += lime_ridge_;
    }
    std::vector<double> beta;
    bool ridged = false;
    if (solve_spd(std::move(normal), G, rhs, O, beta, ridged) != N4M_OK) {
        return fail(ctx, N4M_ERR_NUMERICAL_FAILURE, "LIME weighted ridge system is singular");
    }
    double* phi = &attributions_[row * G * O];
    for (std::size_t o = 0; o < O; ++o) {
        double intercept = y_mean[o];
        for (std::size_t g = 0; g < G; ++g) {
            phi[g * O + o] = beta[g * O + o];
            intercept -= z_mean[g] * beta[g * O + o];
        }
        expected_[row * O + o] = intercept;
    }
    return N4M_OK;
}

void Session::finalize_occlusion(std::size_t row, const RowPlan& plan) {
    const std::size_t G = n_groups_;
    const std::size_t O = n_outputs_;
    double* phi = &attributions_[row * G * O];
    for (std::size_t g = 0; g < G; ++g) {
        for (std::size_t o = 0; o < O; ++o) {
            phi[g * O + o] = value(plan, 0, o) - value(plan, first_method_coalition_ + g, o);
        }
    }
    for (std::size_t o = 0; o < O; ++o) {
        expected_[row * O + o] = empty_value(plan, o);
    }
}

void Session::finalize_gradient(std::size_t row, const RowPlan& plan) {
    const std::size_t F = n_features_;
    const std::size_t O = n_outputs_;
    const double* x = &x_[row * F];
    double* phi = &attributions_[row * n_groups_ * O];
    double* grad = &gradients_[row * F * O];
    for (std::size_t j = 0; j < F; ++j) {
        // The step actually taken (x_j + h) - (x_j - h), not 2h: rounding of
        // the perturbed coordinates enters the quotient exactly.
        const double span = (x[j] + gradient_steps_[j]) - (x[j] - gradient_steps_[j]);
        const std::size_t plus = first_method_coalition_ + 2 * j;
        const double offset = x[j] - reference_[j];
        double* phi_g = &phi[group_of_feature_[j] * O];
        for (std::size_t o = 0; o < O; ++o) {
            const double g = (value(plan, plus, o) - value(plan, plus + 1, o)) / span;
            grad[j * O + o] = g;
            phi_g[o] += g * offset;
        }
    }
    for (std::size_t o = 0; o < O; ++o) {
        expected_[row * O + o] = global_empty_value_[o];
    }
}

void Session::update_gradient_steps(double relative_step) {
    gradient_step_ = relative_step;
    gradient_steps_.resize(gradient_scale_.size());
    for (std::size_t j = 0; j < gradient_scale_.size(); ++j) {
        gradient_steps_[j] = relative_step * gradient_scale_[j];
    }
}

n4m_status_t Session::set_gradient_step(n4m_context_t* ctx, double relative_step) {
    if (method_ != N4M_EXPLAIN_GRADIENT) {
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT,
                    "the gradient step applies to GRADIENT sessions");
    }
    const Cursor start = start_cursor();
    if (state_ != State::kReady || emit_.row != start.row || emit_.coalition != 0
        || emit_.base != 0) {
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT,
                    "the gradient step must be set before the first next_batch");
    }
    if (!std::isfinite(relative_step) || !(relative_step > 0.0)) {
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "the gradient step must be finite and > 0");
    }
    for (std::size_t j = 0; j < gradient_scale_.size(); ++j) {
        const double h = relative_step * gradient_scale_[j];
        const double* x = &x_[j];
        for (std::size_t r = 0; r < n_rows_; ++r) {
            if (!std::isfinite(x[r * n_features_] + h) || !std::isfinite(x[r * n_features_] - h)
                || !(h > 0.0)) {
                return fail(ctx, N4M_ERR_INVALID_ARGUMENT,
                            "the gradient step overflows or underflows a feature");
            }
        }
    }
    update_gradient_steps(relative_step);
    return N4M_OK;
}

n4m_status_t Session::gradients(n4m_context_t* ctx, double* gradients, int64_t gradients_size,
                                double* mean_gradient, int64_t mean_gradient_size,
                                double* steps, int64_t steps_size) const {
    if (method_ != N4M_EXPLAIN_GRADIENT) {
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT,
                    "gradients are available for GRADIENT sessions");
    }
    if (state_ == State::kPoisoned) {
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT,
                    "explanation session was poisoned by an earlier failure");
    }
    if (state_ != State::kComplete) {
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT,
                    "explanation plan is not complete; keep calling next_batch/submit");
    }
    const std::size_t F = n_features_;
    const std::size_t O = n_outputs_;
    if ((gradients != nullptr
         && (gradients_size < 0 || to_size(gradients_size) < gradients_.size()))
        || (mean_gradient != nullptr
            && (mean_gradient_size < 0 || to_size(mean_gradient_size) < F * O))
        || (steps != nullptr && (steps_size < 0 || to_size(steps_size) < F))) {
        return fail(ctx, N4M_ERR_SHAPE_MISMATCH,
                    "gradient buffers are smaller than n_rows * n_features * n_outputs, "
                    "n_features * n_outputs or n_features");
    }
    if (gradients != nullptr) {
        std::copy(gradients_.begin(), gradients_.end(), gradients);
    }
    if (mean_gradient != nullptr) {
        std::fill(mean_gradient, mean_gradient + F * O, 0.0);
        for (std::size_t r = 0; r < n_rows_; ++r) {
            for (std::size_t k = 0; k < F * O; ++k) {
                mean_gradient[k] += gradients_[r * F * O + k];
            }
        }
        for (std::size_t k = 0; k < F * O; ++k) {
            mean_gradient[k] /= static_cast<double>(n_rows_);
        }
    }
    if (steps != nullptr) {
        std::copy(gradient_steps_.begin(), gradient_steps_.end(), steps);
    }
    return N4M_OK;
}

n4m_status_t Session::perturb_rows(n4m_context_t* ctx, const n4m_matrix_view_t* X,
                                   const n4m_matrix_view_t* background,
                                   const int64_t* feature_group, int64_t n_groups,
                                   int32_t baseline, const int64_t* absent_groups,
                                   int64_t n_absent, double* out_rows,
                                   int64_t out_rows_capacity, int64_t* out_n_rows) {
    if (out_n_rows == nullptr) {
        return fail(ctx, N4M_ERR_NULL_POINTER, "out_n_rows is NULL");
    }
    *out_n_rows = 0;
    // The session constructor validates the inputs exactly as for an
    // explanation; occlusion keeps the plan small.
    n4m_explain_options_v1_t options{};
    options.struct_size = sizeof(options);
    options.method = N4M_EXPLAIN_OCCLUSION;
    options.baseline = baseline;
    options.n_outputs = 1;
    options.n_permutations = 1;
    options.lime_ridge = 1.0;
    options.max_batch_rows = 1;
    n4m_explain_session_t* raw = nullptr;
    n4m_status_t status = create(ctx, X, background, feature_group, n_groups, &options, raw);
    if (status != N4M_OK) {
        return status;
    }
    std::unique_ptr<n4m_explain_session_s> session(raw);
    Session& s = *session;
    if (n_absent < 0 || (n_absent > 0 && absent_groups == nullptr)) {
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "absent_groups must list n_absent >= 0 groups");
    }
    std::fill(s.group_mask_.begin(), s.group_mask_.end(), static_cast<unsigned char>(1));
    for (int64_t k = 0; k < n_absent; ++k) {
        const int64_t g = absent_groups[k];
        if (g < 0 || g >= n_groups || s.group_mask_[to_size(g)] == 0) {
            return fail(ctx, N4M_ERR_INVALID_ARGUMENT,
                        "absent_groups must hold distinct groups in [0, n_groups)");
        }
        s.group_mask_[to_size(g)] = 0;
    }
    for (std::size_t j = 0; j < s.n_features_; ++j) {
        s.present_[j] = s.group_mask_[s.group_of_feature_[j]];
    }
    int64_t total = 0;
    if (!checked_mul(X->rows, static_cast<int64_t>(s.n_base_), total)) {
        return fail(ctx, N4M_ERR_UNSUPPORTED, "perturbed row count overflows int64");
    }
    *out_n_rows = total;
    if (out_rows == nullptr && out_rows_capacity == 0) {
        return N4M_OK;
    }
    if (out_rows == nullptr) {
        return fail(ctx, N4M_ERR_NULL_POINTER, "out_rows is NULL");
    }
    if (out_rows_capacity < total) {
        return fail(ctx, N4M_ERR_SHAPE_MISMATCH, "out_rows_capacity is smaller than the row count");
    }
    const std::size_t F = s.n_features_;
    std::size_t written = 0;
    for (std::size_t i = 0; i < s.n_rows_; ++i) {
        for (std::size_t b = 0; b < s.n_base_; ++b) {
            s.fill_present(&s.x_[i * F], b, out_rows + written * F);
            ++written;
        }
    }
    return N4M_OK;
}

}  // namespace n4m::explain

