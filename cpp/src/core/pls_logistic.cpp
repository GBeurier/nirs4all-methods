// SPDX-License-Identifier: CECILL-2.1

#include "core/pls_logistic.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <vector>

#include "core/common/matrix_view.hpp"
#include "core/model.hpp"
#include "core/common/status.hpp"

namespace {

constexpr double kLogisticRidge = 1e-4;
constexpr double kLogisticTol = 1e-10;
constexpr int kLineSearchSteps = 32;

[[nodiscard]] std::size_t idx(std::size_t row, std::size_t cols, std::size_t col) noexcept {
    return row * cols + col;
}

[[nodiscard]] std::size_t param_idx(std::size_t cls,
                                    std::size_t n_terms,
                                    std::size_t term) noexcept {
    return cls * n_terms + term;
}

[[nodiscard]] bool checked_mul_i64(std::int64_t a,
                                   std::int64_t b,
                                   std::int64_t& out) noexcept {
    if (a < 0 || b < 0) {
        return false;
    }
    if (a != 0 && b > std::numeric_limits<std::int64_t>::max() / a) {
        return false;
    }
    out = a * b;
    return true;
}

[[nodiscard]] bool checked_mul_size(std::size_t a,
                                    std::size_t b,
                                    std::size_t& out) noexcept {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        return false;
    }
    out = a * b;
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

void resize_fill(std::vector<double>& values, std::size_t n, double fill) {
    values.clear();
    values.resize(n);
    std::fill(values.begin(), values.end(), fill);
}

[[nodiscard]] n4m_matrix_view_t rowmajor_f64_view(std::vector<double>& values,
                                                  std::int64_t rows,
                                                  std::int64_t cols) noexcept {
    n4m_matrix_view_t view{};
    view.data = values.data();
    view.rows = rows;
    view.cols = cols;
    view.row_stride = cols > 0 ? cols : 1;
    view.col_stride = 1;
    view.dtype = N4M_DTYPE_F64;
    return view;
}

[[nodiscard]] n4m_status_t validate_labels(::n4m::core::Context& ctx,
                                           const n4m_matrix_view_t& labels,
                                           std::int64_t n_samples,
                                           std::int32_t n_classes,
                                           std::vector<std::int32_t>& out) {
    const n4m_status_t status = ::n4m::core::validate_nonnull_view(labels);
    if (status != N4M_OK) {
        ctx.set_errorf("labels matrix view is invalid: %s",
                       ::n4m::core::status_to_string(status));
        return status;
    }
    std::int64_t label_count = 0;
    if (!checked_mul_i64(labels.rows, labels.cols, label_count)) {
        ctx.set_error("label count overflows int64");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (label_count != n_samples) {
        ctx.set_error("label count must match X rows");
        return N4M_ERR_SHAPE_MISMATCH;
    }
    out.clear();
    out.reserve(static_cast<std::size_t>(label_count));
    for (std::int64_t row = 0; row < labels.rows; ++row) {
        for (std::int64_t col = 0; col < labels.cols; ++col) {
            const double value = read_numeric(labels,
                                              static_cast<std::size_t>(row),
                                              static_cast<std::size_t>(col));
            if (!std::isfinite(value) || std::floor(value) != value ||
                value < 0.0 || value >= static_cast<double>(n_classes)) {
                ctx.set_errorf("labels must be integer class ids in [0, %d)",
                               static_cast<int>(n_classes));
                return N4M_ERR_INVALID_ARGUMENT;
            }
            out.push_back(static_cast<std::int32_t>(value));
        }
    }
    return N4M_OK;
}

[[nodiscard]] bool solve_linear_system(std::vector<double> a,
                                       std::vector<double> b,
                                       std::size_t n,
                                       std::vector<double>& x) {
    for (std::size_t col = 0; col < n; ++col) {
        std::size_t pivot = col;
        double pivot_abs = std::fabs(a[idx(col, n, col)]);
        for (std::size_t row = col + 1U; row < n; ++row) {
            const double candidate = std::fabs(a[idx(row, n, col)]);
            if (candidate > pivot_abs) {
                pivot = row;
                pivot_abs = candidate;
            }
        }
        if (pivot_abs <= std::numeric_limits<double>::epsilon()) {
            return false;
        }
        if (pivot != col) {
            for (std::size_t j = col; j < n; ++j) {
                std::swap(a[idx(col, n, j)], a[idx(pivot, n, j)]);
            }
            std::swap(b[col], b[pivot]);
        }
        const double diag = a[idx(col, n, col)];
        for (std::size_t row = col + 1U; row < n; ++row) {
            const double factor = a[idx(row, n, col)] / diag;
            if (factor == 0.0) {
                continue;
            }
            a[idx(row, n, col)] = 0.0;
            for (std::size_t j = col + 1U; j < n; ++j) {
                a[idx(row, n, j)] -= factor * a[idx(col, n, j)];
            }
            b[row] -= factor * b[col];
        }
    }

    resize_fill(x, n, 0.0);
    for (std::size_t rev = 0; rev < n; ++rev) {
        const std::size_t row = n - 1U - rev;
        double sum = b[row];
        for (std::size_t col = row + 1U; col < n; ++col) {
            sum -= a[idx(row, n, col)] * x[col];
        }
        const double diag = a[idx(row, n, row)];
        if (std::fabs(diag) <= std::numeric_limits<double>::epsilon()) {
            return false;
        }
        x[row] = sum / diag;
    }
    return true;
}

[[nodiscard]] double linear_predict(const std::vector<double>& beta,
                                    const std::vector<double>& design,
                                    std::size_t row,
                                    std::size_t n_terms,
                                    std::size_t cls) noexcept {
    double eta = 0.0;
    for (std::size_t term = 0; term < n_terms; ++term) {
        eta += beta[param_idx(cls, n_terms, term)] * design[idx(row, n_terms, term)];
    }
    return eta;
}

void compute_row_probabilities(const std::vector<double>& beta,
                               const std::vector<double>& design,
                               std::size_t row,
                               std::size_t n_terms,
                               std::size_t n_classes,
                               std::vector<double>& logits,
                               std::vector<double>& probabilities) {
    logits.assign(n_classes, 0.0);
    probabilities.assign(n_classes, 0.0);
    double max_logit = 0.0;
    for (std::size_t cls = 1; cls < n_classes; ++cls) {
        const double eta = linear_predict(beta, design, row, n_terms, cls - 1U);
        logits[cls] = eta;
        max_logit = std::max(max_logit, eta);
    }
    double denom = std::exp(-max_logit);
    for (std::size_t cls = 1; cls < n_classes; ++cls) {
        denom += std::exp(logits[cls] - max_logit);
    }
    probabilities[0] = std::exp(-max_logit) / denom;
    for (std::size_t cls = 1; cls < n_classes; ++cls) {
        probabilities[cls] = std::exp(logits[cls] - max_logit) / denom;
    }
}

[[nodiscard]] double objective(const std::vector<double>& beta,
                               const std::vector<double>& design,
                               const std::vector<std::int32_t>& labels,
                               std::size_t n_samples,
                               std::size_t n_terms,
                               std::size_t n_classes) {
    std::vector<double> logits;
    std::vector<double> probabilities;
    double loss = 0.0;
    for (std::size_t row = 0; row < n_samples; ++row) {
        compute_row_probabilities(beta, design, row, n_terms, n_classes, logits, probabilities);
        double max_logit = 0.0;
        for (std::size_t cls = 1; cls < n_classes; ++cls) {
            max_logit = std::max(max_logit, logits[cls]);
        }
        double denom = std::exp(-max_logit);
        for (std::size_t cls = 1; cls < n_classes; ++cls) {
            denom += std::exp(logits[cls] - max_logit);
        }
        loss += max_logit + std::log(denom) -
                logits[static_cast<std::size_t>(labels[row])];
    }
    for (std::size_t cls = 1; cls < n_classes; ++cls) {
        for (std::size_t term = 1; term < n_terms; ++term) {
            const double value = beta[param_idx(cls - 1U, n_terms, term)];
            loss += 0.5 * kLogisticRidge * value * value;
        }
    }
    return loss;
}

[[nodiscard]] n4m_status_t fit_baseline_logistic(::n4m::core::Context& ctx,
                                                 const std::vector<double>& scores,
                                                 const std::vector<std::int32_t>& labels,
                                                 std::size_t n_samples,
                                                 std::size_t n_components,
                                                 std::size_t n_classes,
                                                 std::int32_t max_iter,
                                                 ::n4m::core::PlsLogisticResult& out) {
    const std::size_t n_terms = n_components + 1U;
    const std::size_t n_tail_classes = n_classes - 1U;
    std::size_t design_size = 0;
    std::size_t param_count = 0;
    std::size_t hessian_size = 0;
    if (!checked_mul_size(n_samples, n_terms, design_size) ||
        !checked_mul_size(n_tail_classes, n_terms, param_count) ||
        !checked_mul_size(param_count, param_count, hessian_size)) {
        ctx.set_error("PLS-logistic working size overflows size_t");
        return N4M_ERR_INVALID_ARGUMENT;
    }

    std::vector<double> design(design_size, 0.0);
    for (std::size_t row = 0; row < n_samples; ++row) {
        design[idx(row, n_terms, 0)] = 1.0;
        for (std::size_t comp = 0; comp < n_components; ++comp) {
            design[idx(row, n_terms, comp + 1U)] =
                scores[idx(row, n_components, comp)];
        }
    }

    std::vector<double> beta(param_count, 0.0);
    std::vector<double> gradient;
    std::vector<double> hessian;
    std::vector<double> step;
    std::vector<double> logits;
    std::vector<double> probabilities;
    const int iterations = std::max(1, static_cast<int>(max_iter));
    for (int iter = 0; iter < iterations; ++iter) {
        resize_fill(gradient, param_count, 0.0);
        resize_fill(hessian, hessian_size, 0.0);
        for (std::size_t row = 0; row < n_samples; ++row) {
            compute_row_probabilities(beta,
                                      design,
                                      row,
                                      n_terms,
                                      n_classes,
                                      logits,
                                      probabilities);
            for (std::size_t cls_j = 0; cls_j < n_tail_classes; ++cls_j) {
                const std::size_t actual_j = cls_j + 1U;
                const double pj = probabilities[actual_j];
                const double yj =
                    labels[row] == static_cast<std::int32_t>(actual_j) ? 1.0 : 0.0;
                for (std::size_t term = 0; term < n_terms; ++term) {
                    gradient[param_idx(cls_j, n_terms, term)] +=
                        (pj - yj) * design[idx(row, n_terms, term)];
                }
                for (std::size_t cls_l = 0; cls_l < n_tail_classes; ++cls_l) {
                    const std::size_t actual_l = cls_l + 1U;
                    const double pl = probabilities[actual_l];
                    const double weight = pj * ((cls_j == cls_l ? 1.0 : 0.0) - pl);
                    for (std::size_t term_a = 0; term_a < n_terms; ++term_a) {
                        const double za = design[idx(row, n_terms, term_a)];
                        const std::size_t hrow = param_idx(cls_j, n_terms, term_a);
                        for (std::size_t term_b = 0; term_b < n_terms; ++term_b) {
                            const double zb = design[idx(row, n_terms, term_b)];
                            const std::size_t hcol = param_idx(cls_l, n_terms, term_b);
                            hessian[idx(hrow, param_count, hcol)] += weight * za * zb;
                        }
                    }
                }
            }
        }

        for (std::size_t cls = 0; cls < n_tail_classes; ++cls) {
            for (std::size_t term = 1; term < n_terms; ++term) {
                const std::size_t offset = param_idx(cls, n_terms, term);
                gradient[offset] += kLogisticRidge * beta[offset];
                hessian[idx(offset, param_count, offset)] += kLogisticRidge;
            }
        }

        if (!solve_linear_system(hessian, gradient, param_count, step)) {
            ctx.set_error("failed to solve PLS-logistic Newton system");
            return N4M_ERR_NUMERICAL_FAILURE;
        }
        double max_step = 0.0;
        for (const double value : step) {
            max_step = std::max(max_step, std::fabs(value));
        }
        const double current = objective(beta,
                                         design,
                                         labels,
                                         n_samples,
                                         n_terms,
                                         n_classes);
        double alpha = 1.0;
        bool accepted = false;
        std::vector<double> candidate(param_count, 0.0);
        for (int trial = 0; trial < kLineSearchSteps; ++trial) {
            for (std::size_t param = 0; param < param_count; ++param) {
                candidate[param] = beta[param] - alpha * step[param];
            }
            const double trial_objective = objective(candidate,
                                                    design,
                                                    labels,
                                                    n_samples,
                                                    n_terms,
                                                    n_classes);
            if (std::isfinite(trial_objective) &&
                (trial_objective <= current || alpha * max_step <= kLogisticTol)) {
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }
        if (!accepted) {
            ctx.set_error("PLS-logistic Newton line search failed");
            return N4M_ERR_CONVERGENCE_FAILED;
        }
        beta = candidate;
        if (alpha * max_step <= kLogisticTol) {
            break;
        }
    }

    out.intercepts.assign(n_tail_classes, 0.0);
    out.coefficients.assign(n_tail_classes * n_components, 0.0);
    for (std::size_t cls = 0; cls < n_tail_classes; ++cls) {
        out.intercepts[cls] = beta[param_idx(cls, n_terms, 0)];
        for (std::size_t comp = 0; comp < n_components; ++comp) {
            out.coefficients[idx(cls, n_components, comp)] =
                beta[param_idx(cls, n_terms, comp + 1U)];
        }
    }

    out.predictions.assign(n_samples, 0);
    out.decision_scores.assign(n_samples * n_classes, 0.0);
    out.probabilities.assign(n_samples * n_classes, 0.0);
    for (std::size_t row = 0; row < n_samples; ++row) {
        compute_row_probabilities(beta,
                                  design,
                                  row,
                                  n_terms,
                                  n_classes,
                                  logits,
                                  probabilities);
        std::size_t best_cls = 0;
        double best_score = logits[0];
        for (std::size_t cls = 0; cls < n_classes; ++cls) {
            out.decision_scores[idx(row, n_classes, cls)] = logits[cls];
            out.probabilities[idx(row, n_classes, cls)] = probabilities[cls];
            if (logits[cls] > best_score) {
                best_score = logits[cls];
                best_cls = cls;
            }
        }
        out.predictions[row] = static_cast<std::int32_t>(best_cls);
    }
    return N4M_OK;
}

}  // namespace

namespace n4m::core {

n4m_status_t fit_predict_pls_logistic(Context& ctx,
                                     const Config& cfg,
                                     const n4m_matrix_view_t& X,
                                     const n4m_matrix_view_t& labels,
                                     std::int32_t n_classes,
                                     PlsLogisticResult& out) {
    try {
        out = PlsLogisticResult{};
        const n4m_status_t x_status = validate_nonnull_view(X);
        if (x_status != N4M_OK) {
            ctx.set_errorf("X matrix view is invalid: %s", status_to_string(x_status));
            return x_status;
        }
        if (X.dtype != N4M_DTYPE_F64 && X.dtype != N4M_DTYPE_F32) {
            ctx.set_error("X dtype must be f64 or f32");
            return N4M_ERR_DTYPE_MISMATCH;
        }
        if (X.rows <= 0 || X.cols <= 0) {
            ctx.set_error("X matrix must be non-empty");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        if (n_classes < 2) {
            ctx.set_errorf("n_classes must be >= 2; got %d", static_cast<int>(n_classes));
            return N4M_ERR_INVALID_ARGUMENT;
        }
        if (cfg.n_components < 1) {
            ctx.set_error("PLS-logistic requires at least one PLS component");
            return N4M_ERR_INVALID_ARGUMENT;
        }

        std::vector<std::int32_t> y_labels;
        n4m_status_t status = validate_labels(ctx, labels, X.rows, n_classes, y_labels);
        if (status != N4M_OK) {
            return status;
        }
        const auto n = static_cast<std::size_t>(X.rows);
        const auto k = static_cast<std::size_t>(cfg.n_components);
        const auto c = static_cast<std::size_t>(n_classes);
        std::size_t n_times_c = 0;
        std::size_t n_times_k = 0;
        if (!checked_mul_size(n, c, n_times_c) ||
            !checked_mul_size(n, k, n_times_k)) {
            ctx.set_error("PLS-logistic working size overflows size_t");
            return N4M_ERR_INVALID_ARGUMENT;
        }

        std::vector<std::int64_t> class_counts(c, 0);
        std::vector<double> dummy_y(n_times_c, 0.0);
        for (std::size_t row = 0; row < n; ++row) {
            const auto cls = static_cast<std::size_t>(y_labels[row]);
            class_counts[cls] += 1;
            dummy_y[idx(row, c, cls)] = 1.0;
        }
        for (std::size_t cls = 0; cls < c; ++cls) {
            if (class_counts[cls] == 0) {
                ctx.set_errorf("class %llu has no labels",
                               static_cast<unsigned long long>(cls));
                return N4M_ERR_INVALID_ARGUMENT;
            }
        }

        // Match the sklearn convention by default: PLS-regression chassis
        // on the one-hot label matrix. The caller is responsible for
        // picking the solver (NIPALS to mirror scikit-learn, SIMPLS for
        // the legacy single-pass path). PLS-DA is left as an opt-in by
        // passing ``cfg.algorithm = N4M_ALGO_PLS_DA`` explicitly.
        Config pls_cfg = cfg;
        if (pls_cfg.algorithm != N4M_ALGO_PLS_DA) {
            pls_cfg.algorithm = N4M_ALGO_PLS_REGRESSION;
        }
        pls_cfg.deflation = N4M_DEFLATION_REGRESSION;
        n4m_matrix_view_t Y = rowmajor_f64_view(dummy_y, X.rows, n_classes);
        std::unique_ptr<Model> pls_model;
        status = fit_model(ctx, pls_cfg, X, Y, pls_model);
        if (status != N4M_OK) {
            return status;
        }

        std::vector<double> scores(n_times_k, 0.0);
        n4m_matrix_view_t score_view = rowmajor_f64_view(scores,
                                                         X.rows,
                                                         cfg.n_components);
        status = transform_into(ctx, *pls_model, X, score_view);
        if (status != N4M_OK) {
            return status;
        }

        out.n_samples = X.rows;
        out.n_classes = n_classes;
        out.n_components = cfg.n_components;
        status = fit_baseline_logistic(ctx,
                                       scores,
                                       y_labels,
                                       n,
                                       k,
                                       c,
                                       cfg.max_iter,
                                       out);
        if (status != N4M_OK) {
            out = PlsLogisticResult{};
            return status;
        }
        out.n_samples = X.rows;
        out.n_classes = n_classes;
        out.n_components = cfg.n_components;

        ctx.clear_error();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        ctx.set_error("out of memory while fitting PLS-logistic");
        out = PlsLogisticResult{};
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        ctx.set_error("unexpected exception while fitting PLS-logistic");
        out = PlsLogisticResult{};
        return N4M_ERR_INTERNAL;
    }
}

}  // namespace n4m::core
