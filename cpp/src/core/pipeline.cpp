// SPDX-License-Identifier: CeCILL-2.1

#include "core/pipeline.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include "core/matrix_view.hpp"
#include "core/status.hpp"

namespace {

[[nodiscard]] std::size_t idx(std::size_t row, std::size_t cols, std::size_t col) noexcept {
    return row * cols + col;
}

[[nodiscard]] unsigned long long ull(std::size_t value) noexcept {
    return static_cast<unsigned long long>(value);
}

[[nodiscard]] double read_value(const p4a_matrix_view_t& v,
                                std::size_t row,
                                std::size_t col) noexcept {
    const std::int64_t off =
        static_cast<std::int64_t>(row) * v.row_stride +
        static_cast<std::int64_t>(col) * v.col_stride;
    const std::size_t uoff = static_cast<std::size_t>(off);
    if (v.dtype == P4A_DTYPE_F64) {
        const auto* ptr = static_cast<const double*>(v.data);
        return ptr[uoff];
    }
    const auto* ptr = static_cast<const float*>(v.data);
    return static_cast<double>(ptr[uoff]);
}

void write_value(p4a_matrix_view_t& v,
                 std::size_t row,
                 std::size_t col,
                 double value) noexcept {
    const std::int64_t off =
        static_cast<std::int64_t>(row) * v.row_stride +
        static_cast<std::int64_t>(col) * v.col_stride;
    const std::size_t uoff = static_cast<std::size_t>(off);
    if (v.dtype == P4A_DTYPE_F64) {
        auto* ptr = static_cast<double*>(v.data);
        ptr[uoff] = value;
        return;
    }
    auto* ptr = static_cast<float*>(v.data);
    ptr[uoff] = static_cast<float>(value);
}

[[nodiscard]] p4a_status_t validate_float_view(::pls4all::core::Context& ctx,
                                               const p4a_matrix_view_t& v,
                                               const char* name) noexcept {
    const p4a_status_t status = ::pls4all::core::validate_nonnull_view(v);
    if (status != P4A_OK) {
        ctx.set_errorf("%s matrix view is invalid: %s",
                       name,
                       ::pls4all::core::status_to_string(status));
        return status;
    }
    if (v.dtype != P4A_DTYPE_F64 && v.dtype != P4A_DTYPE_F32) {
        ctx.set_errorf("%s dtype must be f64 or f32", name);
        return P4A_ERR_DTYPE_MISMATCH;
    }
    return P4A_OK;
}

[[nodiscard]] bool element_count(std::int64_t rows,
                                 std::int64_t cols,
                                 std::size_t& out) noexcept {
    if (rows < 0 || cols < 0) {
        return false;
    }
    const std::uint64_t urows = static_cast<std::uint64_t>(rows);
    const std::uint64_t ucols = static_cast<std::uint64_t>(cols);
    if (ucols != 0U &&
        urows > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) / ucols) {
        return false;
    }
    out = static_cast<std::size_t>(urows * ucols);
    return true;
}

[[nodiscard]] p4a_status_t copy_matrix_checked(::pls4all::core::Context& ctx,
                                               const p4a_matrix_view_t& src,
                                               const char* name,
                                               std::vector<double>& out) {
    std::size_t n_values = 0;
    if (!element_count(src.rows, src.cols, n_values)) {
        ctx.set_errorf("%s shape is invalid or too large", name);
        return P4A_ERR_INVALID_ARGUMENT;
    }
    out.assign(n_values, 0.0);
    const std::size_t rows = static_cast<std::size_t>(src.rows);
    const std::size_t cols = static_cast<std::size_t>(src.cols);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            const double value = read_value(src, row, col);
            if (!std::isfinite(value)) {
                ctx.set_errorf("%s contains NaN or Inf at row %llu col %llu",
                               name,
                               ull(row),
                               ull(col));
                return P4A_ERR_INVALID_ARGUMENT;
            }
            out[idx(row, cols, col)] = value;
        }
    }
    return P4A_OK;
}

void copy_to_output(const std::vector<double>& values,
                    std::size_t rows,
                    std::size_t cols,
                    p4a_matrix_view_t& out) noexcept {
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            write_value(out, row, col, values[idx(row, cols, col)]);
        }
    }
}

void fit_column_center(const std::vector<double>& values,
                       std::size_t rows,
                       std::size_t cols,
                       std::vector<double>& mean) {
    mean.assign(cols, 0.0);
    for (std::size_t col = 0; col < cols; ++col) {
        double sum = 0.0;
        for (std::size_t row = 0; row < rows; ++row) {
            sum += values[idx(row, cols, col)];
        }
        mean[col] = sum / static_cast<double>(rows);
    }
}

void fit_column_scale(const std::vector<double>& values,
                      const std::vector<double>& mean,
                      std::size_t rows,
                      std::size_t cols,
                      bool pareto,
                      std::vector<double>& scale) {
    scale.assign(cols, 1.0);
    const double denom = rows > 1U ? static_cast<double>(rows - 1U) : 1.0;
    for (std::size_t col = 0; col < cols; ++col) {
        double sumsq = 0.0;
        for (std::size_t row = 0; row < rows; ++row) {
            const double centered = values[idx(row, cols, col)] - mean[col];
            sumsq += centered * centered;
        }
        const double stddev = std::sqrt(sumsq / denom);
        const double raw_scale = pareto ? std::sqrt(stddev) : stddev;
        scale[col] = (raw_scale == 0.0 || !std::isfinite(raw_scale)) ? 1.0 : raw_scale;
    }
}

void apply_column_transform(const std::vector<double>& values,
                            const std::vector<double>& mean,
                            const std::vector<double>& scale,
                            std::size_t rows,
                            std::size_t cols,
                            std::vector<double>& out) {
    out.assign(rows * cols, 0.0);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            out[idx(row, cols, col)] =
                (values[idx(row, cols, col)] - mean[col]) / scale[col];
        }
    }
}

void apply_snv(const std::vector<double>& values,
               std::size_t rows,
               std::size_t cols,
               std::vector<double>& out) {
    out.assign(rows * cols, 0.0);
    const double denom = cols > 1U ? static_cast<double>(cols - 1U) : 1.0;
    for (std::size_t row = 0; row < rows; ++row) {
        double mean = 0.0;
        for (std::size_t col = 0; col < cols; ++col) {
            mean += values[idx(row, cols, col)];
        }
        mean /= static_cast<double>(cols);
        double sumsq = 0.0;
        for (std::size_t col = 0; col < cols; ++col) {
            const double centered = values[idx(row, cols, col)] - mean;
            sumsq += centered * centered;
        }
        const double stddev = std::sqrt(sumsq / denom);
        const double scale = (stddev == 0.0 || !std::isfinite(stddev)) ? 1.0 : stddev;
        for (std::size_t col = 0; col < cols; ++col) {
            out[idx(row, cols, col)] = (values[idx(row, cols, col)] - mean) / scale;
        }
    }
}

[[nodiscard]] bool solve_linear_system(std::vector<double> matrix,
                                       std::vector<double> rhs,
                                       std::size_t n,
                                       std::vector<double>& solution) {
    solution.assign(n, 0.0);
    for (std::size_t col = 0; col < n; ++col) {
        std::size_t pivot = col;
        double pivot_abs = std::fabs(matrix[idx(col, n, col)]);
        for (std::size_t row = col + 1U; row < n; ++row) {
            const double candidate = std::fabs(matrix[idx(row, n, col)]);
            if (candidate > pivot_abs) {
                pivot = row;
                pivot_abs = candidate;
            }
        }
        if (pivot_abs <= std::numeric_limits<double>::epsilon()) {
            return false;
        }
        if (pivot != col) {
            for (std::size_t j = 0; j < n; ++j) {
                std::swap(matrix[idx(col, n, j)], matrix[idx(pivot, n, j)]);
            }
            std::swap(rhs[col], rhs[pivot]);
        }
        const double diag = matrix[idx(col, n, col)];
        for (std::size_t j = col; j < n; ++j) {
            matrix[idx(col, n, j)] /= diag;
        }
        rhs[col] /= diag;
        for (std::size_t row = 0; row < n; ++row) {
            if (row == col) {
                continue;
            }
            const double factor = matrix[idx(row, n, col)];
            for (std::size_t j = col; j < n; ++j) {
                matrix[idx(row, n, j)] -= factor * matrix[idx(col, n, j)];
            }
            rhs[row] -= factor * rhs[col];
        }
    }
    solution = std::move(rhs);
    return true;
}

[[nodiscard]] p4a_status_t apply_detrend_poly(::pls4all::core::Context& ctx,
                                              const std::vector<double>& values,
                                              std::size_t rows,
                                              std::size_t cols,
                                              std::int32_t degree,
                                              std::vector<double>& out) {
    if (degree < 0) {
        ctx.set_error("detrend polynomial degree must be non-negative");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (static_cast<std::size_t>(degree) >= cols) {
        ctx.set_error("detrend polynomial degree must be smaller than the column count");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    const std::size_t terms = static_cast<std::size_t>(degree) + 1U;
    std::vector<double> x(cols, 0.0);
    for (std::size_t col = 0; col < cols; ++col) {
        x[col] = cols > 1U
            ? -1.0 + 2.0 * static_cast<double>(col) / static_cast<double>(cols - 1U)
            : 0.0;
    }
    std::vector<double> powers(cols * terms, 1.0);
    for (std::size_t col = 0; col < cols; ++col) {
        for (std::size_t term = 1; term < terms; ++term) {
            powers[idx(col, terms, term)] =
                powers[idx(col, terms, term - 1U)] * x[col];
        }
    }
    std::vector<double> gram(terms * terms, 0.0);
    for (std::size_t row_term = 0; row_term < terms; ++row_term) {
        for (std::size_t col_term = 0; col_term < terms; ++col_term) {
            double sum = 0.0;
            for (std::size_t col = 0; col < cols; ++col) {
                sum += powers[idx(col, terms, row_term)] *
                       powers[idx(col, terms, col_term)];
            }
            gram[idx(row_term, terms, col_term)] = sum;
        }
    }

    out.assign(rows * cols, 0.0);
    for (std::size_t row = 0; row < rows; ++row) {
        std::vector<double> rhs(terms, 0.0);
        for (std::size_t term = 0; term < terms; ++term) {
            double sum = 0.0;
            for (std::size_t col = 0; col < cols; ++col) {
                sum += values[idx(row, cols, col)] * powers[idx(col, terms, term)];
            }
            rhs[term] = sum;
        }
        std::vector<double> coeffs;
        if (!solve_linear_system(gram, rhs, terms, coeffs)) {
            ctx.set_error("failed to solve detrend polynomial normal equations");
            return P4A_ERR_NUMERICAL_FAILURE;
        }
        for (std::size_t col = 0; col < cols; ++col) {
            double trend = 0.0;
            for (std::size_t term = 0; term < terms; ++term) {
                trend += coeffs[term] * powers[idx(col, terms, term)];
            }
            out[idx(row, cols, col)] = values[idx(row, cols, col)] - trend;
        }
    }
    return P4A_OK;
}

[[nodiscard]] p4a_status_t apply_msc(::pls4all::core::Context& ctx,
                                     const std::vector<double>& values,
                                     const std::vector<double>& reference,
                                     std::size_t rows,
                                     std::size_t cols,
                                     std::vector<double>& out) {
    if (cols < 2U) {
        ctx.set_error("MSC requires at least 2 columns");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    double ref_mean = 0.0;
    for (double value : reference) {
        ref_mean += value;
    }
    ref_mean /= static_cast<double>(cols);
    double ref_ss = 0.0;
    for (double value : reference) {
        const double centered = value - ref_mean;
        ref_ss += centered * centered;
    }
    if (ref_ss <= std::numeric_limits<double>::epsilon() || !std::isfinite(ref_ss)) {
        ctx.set_error("MSC reference spectrum has zero variance");
        return P4A_ERR_NUMERICAL_FAILURE;
    }

    out.assign(rows * cols, 0.0);
    for (std::size_t row = 0; row < rows; ++row) {
        double row_mean = 0.0;
        for (std::size_t col = 0; col < cols; ++col) {
            row_mean += values[idx(row, cols, col)];
        }
        row_mean /= static_cast<double>(cols);

        double cov = 0.0;
        for (std::size_t col = 0; col < cols; ++col) {
            cov += (reference[col] - ref_mean) *
                   (values[idx(row, cols, col)] - row_mean);
        }
        const double slope = cov / ref_ss;
        if (std::fabs(slope) <= std::numeric_limits<double>::epsilon() ||
            !std::isfinite(slope)) {
            ctx.set_errorf("MSC slope vanished at row %llu", ull(row));
            return P4A_ERR_NUMERICAL_FAILURE;
        }
        const double intercept = row_mean - slope * ref_mean;
        for (std::size_t col = 0; col < cols; ++col) {
            out[idx(row, cols, col)] =
                (values[idx(row, cols, col)] - intercept) / slope;
        }
    }
    return P4A_OK;
}

[[nodiscard]] p4a_status_t require_no_params(::pls4all::core::Context& ctx,
                                             const ::pls4all::core::OperatorEntry& entry,
                                             const char* name) {
    if (!entry.params.empty()) {
        ctx.set_errorf("%s does not accept parameters in this release", name);
        return P4A_ERR_INVALID_ARGUMENT;
    }
    return P4A_OK;
}

[[nodiscard]] p4a_status_t parse_degree(::pls4all::core::Context& ctx,
                                        const ::pls4all::core::OperatorEntry& entry,
                                        std::int32_t& degree) {
    degree = 1;
    if (entry.params.empty()) {
        return P4A_OK;
    }
    if (entry.params.size() != 1U) {
        ctx.set_error("detrend polynomial expects zero params or one degree param");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    const double raw = entry.params[0];
    const double rounded = std::round(raw);
    if (!std::isfinite(raw) || std::fabs(raw - rounded) > 1e-12 ||
        rounded < 0.0 || rounded > 5.0) {
        ctx.set_error("detrend polynomial degree must be an integer in [0, 5]");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    degree = static_cast<std::int32_t>(rounded);
    return P4A_OK;
}

}  // namespace

namespace pls4all::core {

p4a_status_t Pipeline::fit(Context& ctx, const p4a_matrix_view_t& X) {
    fitted_ = false;
    states_.clear();

    p4a_status_t status = validate_float_view(ctx, X, "X");
    if (status != P4A_OK) {
        return status;
    }
    if (X.rows < 1 || X.cols < 1) {
        ctx.set_error("pipeline fit requires at least 1 row and 1 column");
        return P4A_ERR_INVALID_ARGUMENT;
    }

    std::vector<double> current;
    status = copy_matrix_checked(ctx, X, "X", current);
    if (status != P4A_OK) {
        return status;
    }

    const std::size_t rows = static_cast<std::size_t>(X.rows);
    const std::size_t cols = static_cast<std::size_t>(X.cols);
    states_.reserve(entries_.size());

    for (const OperatorEntry& entry : entries_) {
        OperatorState state;
        state.kind = entry.kind;
        state.n_features = X.cols;
        std::vector<double> next;

        switch (entry.kind) {
            case P4A_OP_IDENTITY:
                status = require_no_params(ctx, entry, "identity");
                if (status != P4A_OK) {
                    states_.clear();
                    return status;
                }
                next = current;
                break;
            case P4A_OP_CENTER:
                status = require_no_params(ctx, entry, "center");
                if (status != P4A_OK) {
                    states_.clear();
                    return status;
                }
                fit_column_center(current, rows, cols, state.location);
                state.scale.assign(cols, 1.0);
                apply_column_transform(current, state.location, state.scale, rows, cols, next);
                break;
            case P4A_OP_AUTOSCALE:
                status = require_no_params(ctx, entry, "autoscale");
                if (status != P4A_OK) {
                    states_.clear();
                    return status;
                }
                fit_column_center(current, rows, cols, state.location);
                fit_column_scale(current, state.location, rows, cols, false, state.scale);
                apply_column_transform(current, state.location, state.scale, rows, cols, next);
                break;
            case P4A_OP_PARETO_SCALE:
                status = require_no_params(ctx, entry, "pareto scale");
                if (status != P4A_OK) {
                    states_.clear();
                    return status;
                }
                fit_column_center(current, rows, cols, state.location);
                fit_column_scale(current, state.location, rows, cols, true, state.scale);
                apply_column_transform(current, state.location, state.scale, rows, cols, next);
                break;
            case P4A_OP_SNV:
                status = require_no_params(ctx, entry, "SNV");
                if (status != P4A_OK) {
                    states_.clear();
                    return status;
                }
                apply_snv(current, rows, cols, next);
                break;
            case P4A_OP_MSC:
                status = require_no_params(ctx, entry, "MSC");
                if (status != P4A_OK) {
                    states_.clear();
                    return status;
                }
                fit_column_center(current, rows, cols, state.location);
                status = apply_msc(ctx, current, state.location, rows, cols, next);
                if (status != P4A_OK) {
                    states_.clear();
                    return status;
                }
                break;
            case P4A_OP_DETREND_POLY: {
                std::int32_t degree = 1;
                status = parse_degree(ctx, entry, degree);
                if (status != P4A_OK) {
                    states_.clear();
                    return status;
                }
                state.scale.assign(1U, static_cast<double>(degree));
                status = apply_detrend_poly(ctx, current, rows, cols, degree, next);
                if (status != P4A_OK) {
                    states_.clear();
                    return status;
                }
                break;
            }
            default:
                ctx.set_errorf("pipeline operator %d is not implemented in this release",
                               static_cast<int>(entry.kind));
                states_.clear();
                return P4A_ERR_NOT_IMPLEMENTED;
        }

        states_.push_back(std::move(state));
        current.swap(next);
    }

    n_features_ = X.cols;
    fitted_ = true;
    ctx.clear_error();
    return P4A_OK;
}

p4a_status_t Pipeline::transform(Context& ctx,
                                 const p4a_matrix_view_t& X,
                                 p4a_matrix_view_t& out) const {
    if (!fitted_) {
        ctx.set_error("pipeline is not fitted");
        return P4A_ERR_NOT_FITTED;
    }

    p4a_status_t status = validate_float_view(ctx, X, "X");
    if (status != P4A_OK) {
        return status;
    }
    status = validate_float_view(ctx, out, "output");
    if (status != P4A_OK) {
        return status;
    }
    if (X.cols != n_features_) {
        ctx.set_errorf("X cols (%lld) must match fitted pipeline cols (%lld)",
                       static_cast<long long>(X.cols),
                       static_cast<long long>(n_features_));
        return P4A_ERR_SHAPE_MISMATCH;
    }
    if (out.rows != X.rows || out.cols != X.cols) {
        ctx.set_error("pipeline output shape must match X shape");
        return P4A_ERR_SHAPE_MISMATCH;
    }

    std::vector<double> current;
    status = copy_matrix_checked(ctx, X, "X", current);
    if (status != P4A_OK) {
        return status;
    }

    const std::size_t rows = static_cast<std::size_t>(X.rows);
    const std::size_t cols = static_cast<std::size_t>(X.cols);
    for (const OperatorState& state : states_) {
        std::vector<double> next;
        switch (state.kind) {
            case P4A_OP_IDENTITY:
                next = current;
                break;
            case P4A_OP_CENTER:
            case P4A_OP_AUTOSCALE:
            case P4A_OP_PARETO_SCALE:
                apply_column_transform(current, state.location, state.scale, rows, cols, next);
                break;
            case P4A_OP_SNV:
                apply_snv(current, rows, cols, next);
                break;
            case P4A_OP_MSC:
                status = apply_msc(ctx, current, state.location, rows, cols, next);
                if (status != P4A_OK) {
                    return status;
                }
                break;
            case P4A_OP_DETREND_POLY:
                if (state.scale.empty()) {
                    ctx.set_error("fitted detrend operator is missing its degree");
                    return P4A_ERR_INTERNAL;
                }
                status = apply_detrend_poly(ctx, current, rows, cols,
                                            static_cast<std::int32_t>(state.scale[0]), next);
                if (status != P4A_OK) {
                    return status;
                }
                break;
            default:
                ctx.set_error("fitted pipeline contains an unsupported operator");
                return P4A_ERR_INTERNAL;
        }
        current.swap(next);
    }

    copy_to_output(current, rows, cols, out);
    ctx.clear_error();
    return P4A_OK;
}

}  // namespace pls4all::core
