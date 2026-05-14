// SPDX-License-Identifier: CeCILL-2.1

#include "core/classification_metrics.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <utility>
#include <vector>

#include "core/matrix_view.hpp"
#include "core/status.hpp"

namespace {

constexpr double kEps = std::numeric_limits<double>::epsilon();

[[nodiscard]] unsigned long long ull(std::size_t value) noexcept {
    return static_cast<unsigned long long>(value);
}

[[nodiscard]] bool checked_element_count(std::int64_t rows,
                                         std::int64_t cols,
                                         std::size_t& out) noexcept {
    if (rows < 0 || cols < 0) {
        return false;
    }
    const auto urows = static_cast<std::size_t>(rows);
    const auto ucols = static_cast<std::size_t>(cols);
    if (ucols != 0U &&
        urows > std::numeric_limits<std::size_t>::max() / ucols) {
        return false;
    }
    out = urows * ucols;
    return true;
}

[[nodiscard]] double read_numeric(const p4a_matrix_view_t& view,
                                  std::size_t row,
                                  std::size_t col) noexcept {
    const std::int64_t off =
        static_cast<std::int64_t>(row) * view.row_stride +
        static_cast<std::int64_t>(col) * view.col_stride;
    const auto uoff = static_cast<std::size_t>(off);
    switch (view.dtype) {
        case P4A_DTYPE_F64: {
            const auto* ptr = static_cast<const double*>(view.data);
            return ptr[uoff];
        }
        case P4A_DTYPE_F32: {
            const auto* ptr = static_cast<const float*>(view.data);
            return static_cast<double>(ptr[uoff]);
        }
        case P4A_DTYPE_I32: {
            const auto* ptr = static_cast<const std::int32_t*>(view.data);
            return static_cast<double>(ptr[uoff]);
        }
        case P4A_DTYPE_I64: {
            const auto* ptr = static_cast<const std::int64_t*>(view.data);
            return static_cast<double>(ptr[uoff]);
        }
        case P4A_DTYPE_UNKNOWN:
            return 0.0;
    }
    return 0.0;
}

[[nodiscard]] bool is_label_dtype(p4a_dtype_t dtype) noexcept {
    return dtype == P4A_DTYPE_F64 || dtype == P4A_DTYPE_F32 ||
           dtype == P4A_DTYPE_I32 || dtype == P4A_DTYPE_I64;
}

[[nodiscard]] bool is_score_dtype(p4a_dtype_t dtype) noexcept {
    return dtype == P4A_DTYPE_F64 || dtype == P4A_DTYPE_F32;
}

[[nodiscard]] p4a_status_t validate_view(::pls4all::core::Context& ctx,
                                         const p4a_matrix_view_t& view,
                                         const char* name,
                                         bool score) noexcept {
    const p4a_status_t status = ::pls4all::core::validate_nonnull_view(view);
    if (status != P4A_OK) {
        ctx.set_errorf("%s matrix view is invalid: %s",
                       name,
                       ::pls4all::core::status_to_string(status));
        return status;
    }
    if (view.rows == 0 || view.cols == 0) {
        ctx.set_errorf("%s must contain at least one value", name);
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (score ? !is_score_dtype(view.dtype) : !is_label_dtype(view.dtype)) {
        ctx.set_errorf("%s dtype is unsupported for binary classification metrics", name);
        return P4A_ERR_DTYPE_MISMATCH;
    }
    return P4A_OK;
}

[[nodiscard]] p4a_status_t copy_inputs(::pls4all::core::Context& ctx,
                                       const p4a_matrix_view_t& labels,
                                       const p4a_matrix_view_t& scores,
                                       std::vector<std::int32_t>& out_labels,
                                       std::vector<double>& out_scores) {
    std::size_t n_values = 0;
    if (!checked_element_count(labels.rows, labels.cols, n_values)) {
        ctx.set_error("classification metric input shape is too large");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    out_labels.clear();
    out_scores.clear();
    out_labels.resize(n_values);
    out_scores.resize(n_values);

    const auto rows = static_cast<std::size_t>(labels.rows);
    const auto cols = static_cast<std::size_t>(labels.cols);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            const double label_value = read_numeric(labels, row, col);
            const double score_value = read_numeric(scores, row, col);
            if (!std::isfinite(label_value)) {
                ctx.set_errorf("labels contains NaN or Inf at row %llu col %llu",
                               ull(row),
                               ull(col));
                return P4A_ERR_INVALID_ARGUMENT;
            }
            if (label_value != 0.0 && label_value != 1.0) {
                ctx.set_errorf("labels must be binary 0/1; got %.17g at row %llu col %llu",
                               label_value,
                               ull(row),
                               ull(col));
                return P4A_ERR_INVALID_ARGUMENT;
            }
            if (!std::isfinite(score_value)) {
                ctx.set_errorf("scores contains NaN or Inf at row %llu col %llu",
                               ull(row),
                               ull(col));
                return P4A_ERR_INVALID_ARGUMENT;
            }
            const std::size_t flat = row * cols + col;
            out_labels[flat] = label_value == 1.0 ? 1 : 0;
            out_scores[flat] = score_value;
        }
    }
    return P4A_OK;
}

[[nodiscard]] double auc_with_average_ranks(const std::vector<std::int32_t>& labels,
                                            const std::vector<double>& scores,
                                            std::int64_t positives,
                                            std::int64_t negatives) {
    std::vector<std::pair<double, std::int32_t>> ranked;
    ranked.reserve(scores.size());
    for (std::size_t i = 0; i < scores.size(); ++i) {
        ranked.emplace_back(scores[i], labels[i]);
    }
    std::sort(ranked.begin(), ranked.end(),
              [](const auto& a, const auto& b) {
                  return a.first < b.first;
              });

    double positive_rank_sum = 0.0;
    std::size_t start = 0;
    while (start < ranked.size()) {
        std::size_t stop = start + 1U;
        while (stop < ranked.size() && ranked[stop].first == ranked[start].first) {
            ++stop;
        }
        const double average_rank =
            (static_cast<double>(start + 1U) + static_cast<double>(stop)) * 0.5;
        for (std::size_t i = start; i < stop; ++i) {
            if (ranked[i].second == 1) {
                positive_rank_sum += average_rank;
            }
        }
        start = stop;
    }

    const double pos = static_cast<double>(positives);
    const double neg = static_cast<double>(negatives);
    return (positive_rank_sum - pos * (pos + 1.0) * 0.5) / (pos * neg);
}

}  // namespace

namespace pls4all::core {

p4a_status_t compute_binary_classification_metrics(
    Context& ctx,
    const p4a_matrix_view_t& labels,
    const p4a_matrix_view_t& scores,
    double threshold,
    BinaryClassificationMetrics& out) {
    try {
        out = BinaryClassificationMetrics{};
        out.threshold = threshold;

        if (!std::isfinite(threshold)) {
            ctx.set_error("classification threshold must be finite");
            return P4A_ERR_INVALID_ARGUMENT;
        }

        p4a_status_t status = validate_view(ctx, labels, "labels", false);
        if (status != P4A_OK) {
            return status;
        }
        status = validate_view(ctx, scores, "scores", true);
        if (status != P4A_OK) {
            return status;
        }
        if (labels.rows != scores.rows || labels.cols != scores.cols) {
            ctx.set_errorf("labels shape (%lld, %lld) must match scores shape (%lld, %lld)",
                           static_cast<long long>(labels.rows),
                           static_cast<long long>(labels.cols),
                           static_cast<long long>(scores.rows),
                           static_cast<long long>(scores.cols));
            return P4A_ERR_SHAPE_MISMATCH;
        }

        std::vector<std::int32_t> y_true;
        std::vector<double> y_score;
        status = copy_inputs(ctx, labels, scores, y_true, y_score);
        if (status != P4A_OK) {
            return status;
        }

        for (std::size_t i = 0; i < y_true.size(); ++i) {
            const bool actual = y_true[i] == 1;
            const bool predicted = y_score[i] >= threshold;
            if (actual) {
                out.positives += 1;
                if (predicted) {
                    out.true_positive += 1;
                } else {
                    out.false_negative += 1;
                }
            } else {
                out.negatives += 1;
                if (predicted) {
                    out.false_positive += 1;
                } else {
                    out.true_negative += 1;
                }
            }
        }
        if (out.positives == 0 || out.negatives == 0) {
            ctx.set_error("binary classification metrics require both positive and negative labels");
            out = BinaryClassificationMetrics{};
            return P4A_ERR_INVALID_ARGUMENT;
        }

        const double tp = static_cast<double>(out.true_positive);
        const double tn = static_cast<double>(out.true_negative);
        const double fp = static_cast<double>(out.false_positive);
        const double fn = static_cast<double>(out.false_negative);
        const double pos = static_cast<double>(out.positives);
        const double neg = static_cast<double>(out.negatives);
        const double n = static_cast<double>(y_true.size());

        out.count = static_cast<std::int64_t>(y_true.size());
        out.sensitivity = tp / pos;
        out.specificity = tn / neg;
        out.balanced_accuracy = 0.5 * (out.sensitivity + out.specificity);
        out.accuracy = (tp + tn) / n;
        out.precision = (tp + fp) <= kEps ? 0.0 : tp / (tp + fp);
        out.f1 = (out.precision + out.sensitivity) <= kEps
            ? 0.0
            : 2.0 * out.precision * out.sensitivity / (out.precision + out.sensitivity);
        const double mcc_denominator = std::sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
        out.mcc = mcc_denominator <= kEps
            ? 0.0
            : ((tp * tn) - (fp * fn)) / mcc_denominator;
        out.auc = auc_with_average_ranks(y_true, y_score, out.positives, out.negatives);

        ctx.clear_error();
        return P4A_OK;
    } catch (const std::bad_alloc&) {
        ctx.set_error("out of memory while computing binary classification metrics");
        out = BinaryClassificationMetrics{};
        return P4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        ctx.set_error("unexpected exception while computing binary classification metrics");
        out = BinaryClassificationMetrics{};
        return P4A_ERR_INTERNAL;
    }
}

}  // namespace pls4all::core
