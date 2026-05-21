// SPDX-License-Identifier: CECILL-2.1

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

[[nodiscard]] double read_numeric(const n4m_matrix_view_t& view,
                                  std::size_t row,
                                  std::size_t col) noexcept {
    const std::int64_t off =
        static_cast<std::int64_t>(row) * view.row_stride +
        static_cast<std::int64_t>(col) * view.col_stride;
    const auto uoff = static_cast<std::size_t>(off);
    switch (view.dtype) {
        case N4M_DTYPE_F64: {
            const auto* ptr = static_cast<const double*>(view.data);
            return ptr[uoff];
        }
        case N4M_DTYPE_F32: {
            const auto* ptr = static_cast<const float*>(view.data);
            return static_cast<double>(ptr[uoff]);
        }
        case N4M_DTYPE_I32: {
            const auto* ptr = static_cast<const std::int32_t*>(view.data);
            return static_cast<double>(ptr[uoff]);
        }
        case N4M_DTYPE_I64: {
            const auto* ptr = static_cast<const std::int64_t*>(view.data);
            return static_cast<double>(ptr[uoff]);
        }
        case N4M_DTYPE_UNKNOWN:
            return 0.0;
    }
    return 0.0;
}

[[nodiscard]] bool is_label_dtype(n4m_dtype_t dtype) noexcept {
    return dtype == N4M_DTYPE_F64 || dtype == N4M_DTYPE_F32 ||
           dtype == N4M_DTYPE_I32 || dtype == N4M_DTYPE_I64;
}

[[nodiscard]] bool is_score_dtype(n4m_dtype_t dtype) noexcept {
    return dtype == N4M_DTYPE_F64 || dtype == N4M_DTYPE_F32;
}

[[nodiscard]] n4m_status_t validate_view(::n4m::core::Context& ctx,
                                         const n4m_matrix_view_t& view,
                                         const char* name,
                                         bool score) noexcept {
    const n4m_status_t status = ::n4m::core::validate_nonnull_view(view);
    if (status != N4M_OK) {
        ctx.set_errorf("%s matrix view is invalid: %s",
                       name,
                       ::n4m::core::status_to_string(status));
        return status;
    }
    if (view.rows == 0 || view.cols == 0) {
        ctx.set_errorf("%s must contain at least one value", name);
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (score ? !is_score_dtype(view.dtype) : !is_label_dtype(view.dtype)) {
        ctx.set_errorf("%s dtype is unsupported for binary classification metrics", name);
        return N4M_ERR_DTYPE_MISMATCH;
    }
    return N4M_OK;
}

[[nodiscard]] n4m_status_t copy_inputs(::n4m::core::Context& ctx,
                                       const n4m_matrix_view_t& labels,
                                       const n4m_matrix_view_t& scores,
                                       std::vector<std::int32_t>& out_labels,
                                       std::vector<double>& out_scores) {
    std::size_t n_values = 0;
    if (!checked_element_count(labels.rows, labels.cols, n_values)) {
        ctx.set_error("classification metric input shape is too large");
        return N4M_ERR_INVALID_ARGUMENT;
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
                return N4M_ERR_INVALID_ARGUMENT;
            }
            if (label_value != 0.0 && label_value != 1.0) {
                ctx.set_errorf("labels must be binary 0/1; got %.17g at row %llu col %llu",
                               label_value,
                               ull(row),
                               ull(col));
                return N4M_ERR_INVALID_ARGUMENT;
            }
            if (!std::isfinite(score_value)) {
                ctx.set_errorf("scores contains NaN or Inf at row %llu col %llu",
                               ull(row),
                               ull(col));
                return N4M_ERR_INVALID_ARGUMENT;
            }
            const std::size_t flat = row * cols + col;
            out_labels[flat] = label_value == 1.0 ? 1 : 0;
            out_scores[flat] = score_value;
        }
    }
    return N4M_OK;
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

[[nodiscard]] n4m_status_t copy_multiclass_labels(::n4m::core::Context& ctx,
                                                  const n4m_matrix_view_t& labels,
                                                  std::int32_t n_classes,
                                                  std::vector<std::int32_t>& out) {
    std::size_t n_values = 0;
    if (!checked_element_count(labels.rows, labels.cols, n_values)) {
        ctx.set_error("multiclass label input shape is too large");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    out.clear();
    out.reserve(n_values);
    const auto rows = static_cast<std::size_t>(labels.rows);
    const auto cols = static_cast<std::size_t>(labels.cols);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            const double label_value = read_numeric(labels, row, col);
            if (!std::isfinite(label_value)) {
                ctx.set_errorf("labels contains NaN or Inf at row %llu col %llu",
                               ull(row),
                               ull(col));
                return N4M_ERR_INVALID_ARGUMENT;
            }
            if (std::floor(label_value) != label_value ||
                label_value < 0.0 ||
                label_value >= static_cast<double>(n_classes)) {
                ctx.set_errorf("labels must be integer class ids in [0, %d); got %.17g",
                               static_cast<int>(n_classes),
                               label_value);
                return N4M_ERR_INVALID_ARGUMENT;
            }
            out.push_back(static_cast<std::int32_t>(label_value));
        }
    }
    return N4M_OK;
}

[[nodiscard]] double mean_or_zero(const std::vector<double>& values) noexcept {
    if (values.empty()) {
        return 0.0;
    }
    double total = 0.0;
    for (const double value : values) {
        total += value;
    }
    return total / static_cast<double>(values.size());
}

}  // namespace

namespace n4m::core {

n4m_status_t compute_binary_classification_metrics(
    Context& ctx,
    const n4m_matrix_view_t& labels,
    const n4m_matrix_view_t& scores,
    double threshold,
    BinaryClassificationMetrics& out) {
    try {
        out = BinaryClassificationMetrics{};
        out.threshold = threshold;

        if (!std::isfinite(threshold)) {
            ctx.set_error("classification threshold must be finite");
            return N4M_ERR_INVALID_ARGUMENT;
        }

        n4m_status_t status = validate_view(ctx, labels, "labels", false);
        if (status != N4M_OK) {
            return status;
        }
        status = validate_view(ctx, scores, "scores", true);
        if (status != N4M_OK) {
            return status;
        }
        if (labels.rows != scores.rows || labels.cols != scores.cols) {
            ctx.set_errorf("labels shape (%lld, %lld) must match scores shape (%lld, %lld)",
                           static_cast<long long>(labels.rows),
                           static_cast<long long>(labels.cols),
                           static_cast<long long>(scores.rows),
                           static_cast<long long>(scores.cols));
            return N4M_ERR_SHAPE_MISMATCH;
        }

        std::vector<std::int32_t> y_true;
        std::vector<double> y_score;
        status = copy_inputs(ctx, labels, scores, y_true, y_score);
        if (status != N4M_OK) {
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
            return N4M_ERR_INVALID_ARGUMENT;
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
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        ctx.set_error("out of memory while computing binary classification metrics");
        out = BinaryClassificationMetrics{};
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        ctx.set_error("unexpected exception while computing binary classification metrics");
        out = BinaryClassificationMetrics{};
        return N4M_ERR_INTERNAL;
    }
}

n4m_status_t compute_multiclass_classification_metrics(
    Context& ctx,
    const n4m_matrix_view_t& labels,
    const n4m_matrix_view_t& scores,
    std::int32_t n_classes,
    MulticlassClassificationMetrics& out) {
    try {
        out = MulticlassClassificationMetrics{};
        if (n_classes < 2) {
            ctx.set_errorf("n_classes must be >= 2; got %d", static_cast<int>(n_classes));
            return N4M_ERR_INVALID_ARGUMENT;
        }

        n4m_status_t status = validate_view(ctx, labels, "labels", false);
        if (status != N4M_OK) {
            return status;
        }
        status = validate_view(ctx, scores, "scores", true);
        if (status != N4M_OK) {
            return status;
        }
        if (scores.cols != n_classes) {
            ctx.set_errorf("scores cols (%lld) must equal n_classes (%d)",
                           static_cast<long long>(scores.cols),
                           static_cast<int>(n_classes));
            return N4M_ERR_SHAPE_MISMATCH;
        }

        std::vector<std::int32_t> y_true;
        status = copy_multiclass_labels(ctx, labels, n_classes, y_true);
        if (status != N4M_OK) {
            return status;
        }
        if (static_cast<std::int64_t>(y_true.size()) != scores.rows) {
            ctx.set_errorf("label count (%llu) must match score rows (%lld)",
                           ull(y_true.size()),
                           static_cast<long long>(scores.rows));
            return N4M_ERR_SHAPE_MISMATCH;
        }

        const auto n_samples = static_cast<std::size_t>(scores.rows);
        const auto n_cls = static_cast<std::size_t>(n_classes);
        std::vector<double> score_values(n_samples * n_cls, 0.0);
        std::vector<std::int64_t> class_counts(n_cls, 0);
        for (std::size_t row = 0; row < n_samples; ++row) {
            class_counts[static_cast<std::size_t>(y_true[row])] += 1;
            for (std::size_t cls = 0; cls < n_cls; ++cls) {
                const double value = read_numeric(scores, row, cls);
                if (!std::isfinite(value)) {
                    ctx.set_errorf("scores contains NaN or Inf at row %llu col %llu",
                                   ull(row),
                                   ull(cls));
                    return N4M_ERR_INVALID_ARGUMENT;
                }
                score_values[row * n_cls + cls] = value;
            }
        }
        for (std::size_t cls = 0; cls < n_cls; ++cls) {
            if (class_counts[cls] == 0) {
                ctx.set_errorf("class %llu has no labels", ull(cls));
                return N4M_ERR_INVALID_ARGUMENT;
            }
        }

        out.count = static_cast<std::int64_t>(n_samples);
        out.n_classes = n_classes;
        out.confusion_matrix.assign(n_cls * n_cls, 0);
        out.sensitivity.assign(n_cls, 0.0);
        out.specificity.assign(n_cls, 0.0);
        out.precision.assign(n_cls, 0.0);
        out.f1.assign(n_cls, 0.0);
        out.auc_ovr.assign(n_cls, 0.0);

        std::int64_t correct = 0;
        for (std::size_t row = 0; row < n_samples; ++row) {
            std::size_t predicted = 0;
            double best_score = score_values[row * n_cls];
            for (std::size_t cls = 1; cls < n_cls; ++cls) {
                const double value = score_values[row * n_cls + cls];
                if (value > best_score) {
                    best_score = value;
                    predicted = cls;
                }
            }
            const auto actual = static_cast<std::size_t>(y_true[row]);
            out.confusion_matrix[actual * n_cls + predicted] += 1;
            if (actual == predicted) {
                correct += 1;
            }
        }

        std::int64_t sum_tp = 0;
        std::int64_t sum_fp = 0;
        std::int64_t sum_fn = 0;
        for (std::size_t cls = 0; cls < n_cls; ++cls) {
            const std::int64_t tp = out.confusion_matrix[cls * n_cls + cls];
            std::int64_t row_total = 0;
            std::int64_t col_total = 0;
            for (std::size_t other = 0; other < n_cls; ++other) {
                row_total += out.confusion_matrix[cls * n_cls + other];
                col_total += out.confusion_matrix[other * n_cls + cls];
            }
            const std::int64_t fn = row_total - tp;
            const std::int64_t fp = col_total - tp;
            const std::int64_t tn = out.count - tp - fn - fp;
            sum_tp += tp;
            sum_fp += fp;
            sum_fn += fn;

            const double dtp = static_cast<double>(tp);
            const double dfn = static_cast<double>(fn);
            const double dfp = static_cast<double>(fp);
            const double dtn = static_cast<double>(tn);
            out.sensitivity[cls] = (dtp + dfn) <= kEps ? 0.0 : dtp / (dtp + dfn);
            out.specificity[cls] = (dtn + dfp) <= kEps ? 0.0 : dtn / (dtn + dfp);
            out.precision[cls] = (dtp + dfp) <= kEps ? 0.0 : dtp / (dtp + dfp);
            out.f1[cls] = (out.precision[cls] + out.sensitivity[cls]) <= kEps
                ? 0.0
                : 2.0 * out.precision[cls] * out.sensitivity[cls] /
                    (out.precision[cls] + out.sensitivity[cls]);

            std::vector<std::int32_t> binary_labels(n_samples, 0);
            std::vector<double> class_scores(n_samples, 0.0);
            for (std::size_t row = 0; row < n_samples; ++row) {
                binary_labels[row] = y_true[row] == static_cast<std::int32_t>(cls) ? 1 : 0;
                class_scores[row] = score_values[row * n_cls + cls];
            }
            out.auc_ovr[cls] = auc_with_average_ranks(binary_labels,
                                                      class_scores,
                                                      class_counts[cls],
                                                      out.count - class_counts[cls]);
        }

        out.accuracy = static_cast<double>(correct) / static_cast<double>(out.count);
        out.macro_sensitivity = mean_or_zero(out.sensitivity);
        out.macro_specificity = mean_or_zero(out.specificity);
        out.macro_precision = mean_or_zero(out.precision);
        out.macro_f1 = mean_or_zero(out.f1);
        out.macro_auc_ovr = mean_or_zero(out.auc_ovr);
        const double micro_tp = static_cast<double>(sum_tp);
        const double micro_fp = static_cast<double>(sum_fp);
        const double micro_fn = static_cast<double>(sum_fn);
        out.micro_precision = (micro_tp + micro_fp) <= kEps ? 0.0 : micro_tp / (micro_tp + micro_fp);
        out.micro_recall = (micro_tp + micro_fn) <= kEps ? 0.0 : micro_tp / (micro_tp + micro_fn);
        out.micro_f1 = (out.micro_precision + out.micro_recall) <= kEps
            ? 0.0
            : 2.0 * out.micro_precision * out.micro_recall /
                (out.micro_precision + out.micro_recall);

        ctx.clear_error();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        ctx.set_error("out of memory while computing multiclass classification metrics");
        out = MulticlassClassificationMetrics{};
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        ctx.set_error("unexpected exception while computing multiclass classification metrics");
        out = MulticlassClassificationMetrics{};
        return N4M_ERR_INTERNAL;
    }
}

n4m_status_t compute_binary_calibration_curve(
    Context& ctx,
    const n4m_matrix_view_t& labels,
    const n4m_matrix_view_t& scores,
    std::int32_t n_bins,
    BinaryCalibrationCurve& out) {
    try {
        out = BinaryCalibrationCurve{};
        if (n_bins < 1) {
            ctx.set_errorf("n_bins must be >= 1; got %d", static_cast<int>(n_bins));
            return N4M_ERR_INVALID_ARGUMENT;
        }

        n4m_status_t status = validate_view(ctx, labels, "labels", false);
        if (status != N4M_OK) {
            return status;
        }
        status = validate_view(ctx, scores, "scores", true);
        if (status != N4M_OK) {
            return status;
        }
        if (labels.rows != scores.rows || labels.cols != scores.cols) {
            ctx.set_errorf("labels shape (%lld, %lld) must match scores shape (%lld, %lld)",
                           static_cast<long long>(labels.rows),
                           static_cast<long long>(labels.cols),
                           static_cast<long long>(scores.rows),
                           static_cast<long long>(scores.cols));
            return N4M_ERR_SHAPE_MISMATCH;
        }

        std::vector<std::int32_t> y_true;
        std::vector<double> y_score;
        status = copy_inputs(ctx, labels, scores, y_true, y_score);
        if (status != N4M_OK) {
            return status;
        }

        const auto bins = static_cast<std::size_t>(n_bins);
        out.n_bins = n_bins;
        out.counts.assign(bins, 0);
        out.bin_lower.assign(bins, 0.0);
        out.bin_upper.assign(bins, 0.0);
        out.mean_score.assign(bins, 0.0);
        out.positive_rate.assign(bins, 0.0);
        std::vector<double> score_sums(bins, 0.0);
        std::vector<double> positive_sums(bins, 0.0);

        for (std::size_t bin = 0; bin < bins; ++bin) {
            out.bin_lower[bin] = static_cast<double>(bin) / static_cast<double>(bins);
            out.bin_upper[bin] = static_cast<double>(bin + 1U) / static_cast<double>(bins);
        }
        for (std::size_t i = 0; i < y_score.size(); ++i) {
            const double score = y_score[i];
            if (score < 0.0 || score > 1.0) {
                ctx.set_errorf("calibration scores must be in [0, 1]; got %.17g", score);
                out = BinaryCalibrationCurve{};
                return N4M_ERR_INVALID_ARGUMENT;
            }
            std::size_t bin = static_cast<std::size_t>(std::floor(score * static_cast<double>(bins)));
            if (bin >= bins) {
                bin = bins - 1U;
            }
            out.counts[bin] += 1;
            score_sums[bin] += score;
            positive_sums[bin] += static_cast<double>(y_true[i]);
        }
        for (std::size_t bin = 0; bin < bins; ++bin) {
            if (out.counts[bin] > 0) {
                const double count = static_cast<double>(out.counts[bin]);
                out.mean_score[bin] = score_sums[bin] / count;
                out.positive_rate[bin] = positive_sums[bin] / count;
            }
        }

        ctx.clear_error();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        ctx.set_error("out of memory while computing binary calibration curve");
        out = BinaryCalibrationCurve{};
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        ctx.set_error("unexpected exception while computing binary calibration curve");
        out = BinaryCalibrationCurve{};
        return N4M_ERR_INTERNAL;
    }
}

}  // namespace n4m::core
