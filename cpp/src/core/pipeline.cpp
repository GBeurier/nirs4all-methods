// SPDX-License-Identifier: CeCILL-2.1

#include "core/pipeline.hpp"

#include <algorithm>
#include <cstddef>
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

struct OscParams {
    std::int32_t max_iter{100};
    double tol{1e-10};
};

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

void center_columns(const std::vector<double>& values,
                    std::size_t rows,
                    std::size_t cols,
                    std::vector<double>& mean,
                    std::vector<double>& centered) {
    fit_column_center(values, rows, cols, mean);
    centered.assign(rows * cols, 0.0);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            centered[idx(row, cols, col)] = values[idx(row, cols, col)] - mean[col];
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

[[nodiscard]] double vector_sumsq(const std::vector<double>& values) noexcept {
    double out = 0.0;
    for (double value : values) {
        out += value * value;
    }
    return out;
}

[[nodiscard]] double vector_dot(const std::vector<double>& left,
                                const std::vector<double>& right) noexcept {
    double out = 0.0;
    const std::size_t n = std::min(left.size(), right.size());
    for (std::size_t i = 0; i < n; ++i) {
        out += left[i] * right[i];
    }
    return out;
}

[[nodiscard]] p4a_status_t y_orthogonal_residual(::pls4all::core::Context& ctx,
                                                 const std::vector<double>& x_centered,
                                                 const std::vector<double>& y_centered,
                                                 std::size_t rows,
                                                 std::size_t x_cols,
                                                 std::size_t y_cols,
                                                 std::vector<double>& residual) {
    std::vector<double> gram(y_cols * y_cols, 0.0);
    for (std::size_t row = 0; row < y_cols; ++row) {
        for (std::size_t col = 0; col < y_cols; ++col) {
            double sum = 0.0;
            for (std::size_t sample = 0; sample < rows; ++sample) {
                sum += y_centered[idx(sample, y_cols, row)] *
                       y_centered[idx(sample, y_cols, col)];
            }
            gram[idx(row, y_cols, col)] = sum;
        }
    }

    residual.assign(rows * x_cols, 0.0);
    std::vector<double> rhs(y_cols, 0.0);
    std::vector<double> coeffs;
    for (std::size_t x_col = 0; x_col < x_cols; ++x_col) {
        std::fill(rhs.begin(), rhs.end(), 0.0);
        for (std::size_t y_col = 0; y_col < y_cols; ++y_col) {
            double sum = 0.0;
            for (std::size_t sample = 0; sample < rows; ++sample) {
                sum += y_centered[idx(sample, y_cols, y_col)] *
                       x_centered[idx(sample, x_cols, x_col)];
            }
            rhs[y_col] = sum;
        }
        if (!solve_linear_system(gram, rhs, y_cols, coeffs)) {
            ctx.set_error("OSC Y normal equations are singular");
            return P4A_ERR_NUMERICAL_FAILURE;
        }
        for (std::size_t sample = 0; sample < rows; ++sample) {
            double projection = 0.0;
            for (std::size_t y_col = 0; y_col < y_cols; ++y_col) {
                projection += y_centered[idx(sample, y_cols, y_col)] * coeffs[y_col];
            }
            residual[idx(sample, x_cols, x_col)] =
                x_centered[idx(sample, x_cols, x_col)] - projection;
        }
    }
    return P4A_OK;
}

[[nodiscard]] p4a_status_t dominant_weight(::pls4all::core::Context& ctx,
                                           const std::vector<double>& values,
                                           std::size_t rows,
                                           std::size_t cols,
                                           const OscParams& params,
                                           std::vector<double>& weights) {
    weights.assign(cols, 0.0);
    std::size_t best_col = 0;
    double best_ss = -1.0;
    for (std::size_t col = 0; col < cols; ++col) {
        double sumsq = 0.0;
        for (std::size_t row = 0; row < rows; ++row) {
            const double value = values[idx(row, cols, col)];
            sumsq += value * value;
        }
        if (sumsq > best_ss) {
            best_ss = sumsq;
            best_col = col;
        }
    }
    if (best_ss <= std::numeric_limits<double>::epsilon()) {
        ctx.set_error("OSC found no X variation orthogonal to Y");
        return P4A_ERR_NUMERICAL_FAILURE;
    }
    weights[best_col] = 1.0;

    std::vector<double> scores(rows, 0.0);
    std::vector<double> next(cols, 0.0);
    for (std::int32_t iter = 0; iter < params.max_iter; ++iter) {
        std::fill(scores.begin(), scores.end(), 0.0);
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t col = 0; col < cols; ++col) {
                scores[row] += values[idx(row, cols, col)] * weights[col];
            }
        }
        std::fill(next.begin(), next.end(), 0.0);
        for (std::size_t col = 0; col < cols; ++col) {
            for (std::size_t row = 0; row < rows; ++row) {
                next[col] += values[idx(row, cols, col)] * scores[row];
            }
        }
        const double norm = std::sqrt(vector_sumsq(next));
        if (norm <= std::numeric_limits<double>::epsilon()) {
            ctx.set_error("OSC dominant direction vanished");
            return P4A_ERR_NUMERICAL_FAILURE;
        }
        for (double& value : next) {
            value /= norm;
        }
        if (vector_dot(next, weights) < 0.0) {
            for (double& value : next) {
                value = -value;
            }
        }
        double diff = 0.0;
        for (std::size_t col = 0; col < cols; ++col) {
            const double delta = next[col] - weights[col];
            diff += delta * delta;
        }
        weights = next;
        if (diff <= params.tol) {
            return P4A_OK;
        }
    }
    return P4A_OK;
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

struct SavGolParams {
    std::int32_t window{5};
    std::int32_t poly_degree{2};
    std::int32_t derivative_order{0};
    double delta{1.0};
};

struct AslsParams {
    double lambda{100000.0};
    double asymmetry{0.001};
    std::int32_t iterations{10};
};

struct NorrisWilliamsParams {
    std::int32_t segment{5};
    std::int32_t gap{3};
    std::int32_t derivative_order{1};
};

struct WaveletParams {
    std::int32_t levels{1};
    double threshold{0.0};
};

[[nodiscard]] double factorial(std::int32_t n) noexcept {
    double out = 1.0;
    for (std::int32_t i = 2; i <= n; ++i) {
        out *= static_cast<double>(i);
    }
    return out;
}

[[nodiscard]] std::size_t next_power_of_two(std::size_t value) noexcept {
    std::size_t out = 1U;
    while (out < value) {
        out *= 2U;
    }
    return out;
}

[[nodiscard]] std::int32_t max_haar_levels(std::size_t size) noexcept {
    std::int32_t levels = 0;
    while (size >= 2U) {
        size /= 2U;
        ++levels;
    }
    return levels;
}

[[nodiscard]] double soft_threshold(double value, double threshold) noexcept {
    if (value > threshold) {
        return value - threshold;
    }
    if (value < -threshold) {
        return value + threshold;
    }
    return 0.0;
}

[[nodiscard]] bool operator_requires_y(p4a_operator_kind_t kind) noexcept {
    return kind == P4A_OP_OSC;
}

[[nodiscard]] bool pipeline_requires_y(
    const std::vector<::pls4all::core::OperatorEntry>& entries) noexcept {
    for (const auto& entry : entries) {
        if (operator_requires_y(entry.kind)) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] double binomial(std::int32_t n, std::int32_t k) noexcept {
    if (k < 0 || k > n) {
        return 0.0;
    }
    if (k > n - k) {
        k = n - k;
    }
    double out = 1.0;
    for (std::int32_t i = 1; i <= k; ++i) {
        out *= static_cast<double>(n - k + i);
        out /= static_cast<double>(i);
    }
    return out;
}

void norris_williams_filter(const NorrisWilliamsParams& params,
                            std::vector<double>& coeffs) {
    const std::size_t segment = static_cast<std::size_t>(params.segment);
    const std::size_t gap = static_cast<std::size_t>(params.gap);
    const std::size_t order = static_cast<std::size_t>(params.derivative_order);
    coeffs.clear();
    coeffs.reserve((order + 1U) * segment + order * gap);
    for (std::size_t block = 0; block <= order; ++block) {
        const double sign = ((params.derivative_order - static_cast<std::int32_t>(block)) % 2) == 0
            ? 1.0
            : -1.0;
        const double value =
            sign * binomial(params.derivative_order, static_cast<std::int32_t>(block)) /
            static_cast<double>(params.segment);
        for (std::size_t i = 0; i < segment; ++i) {
            coeffs.push_back(value);
        }
        if (block < order) {
            for (std::size_t i = 0; i < gap; ++i) {
                coeffs.push_back(0.0);
            }
        }
    }
}

[[nodiscard]] p4a_status_t apply_norris_williams(::pls4all::core::Context& ctx,
                                                 const std::vector<double>& values,
                                                 std::size_t rows,
                                                 std::size_t cols,
                                                 const NorrisWilliamsParams& params,
                                                 std::vector<double>& out) {
    (void)ctx;
    std::vector<double> coeffs;
    norris_williams_filter(params, coeffs);
    const std::int64_t half = static_cast<std::int64_t>(coeffs.size() / 2U);
    const std::int64_t max_col = static_cast<std::int64_t>(cols - 1U);
    out.assign(rows * cols, 0.0);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            double sum = 0.0;
            for (std::size_t k = 0; k < coeffs.size(); ++k) {
                std::int64_t source =
                    static_cast<std::int64_t>(col) +
                    static_cast<std::int64_t>(k) - half;
                source = std::max<std::int64_t>(0, std::min<std::int64_t>(source, max_col));
                sum += coeffs[k] *
                       values[idx(row, cols, static_cast<std::size_t>(source))];
            }
            out[idx(row, cols, col)] = sum;
        }
    }
    return P4A_OK;
}

[[nodiscard]] p4a_status_t compute_savgol_coeffs(::pls4all::core::Context& ctx,
                                                 const SavGolParams& params,
                                                 std::vector<double>& coeffs) {
    const std::size_t window = static_cast<std::size_t>(params.window);
    const std::size_t terms = static_cast<std::size_t>(params.poly_degree) + 1U;
    const std::int32_t half = params.window / 2;

    std::vector<double> powers(window * terms, 1.0);
    for (std::size_t sample = 0; sample < window; ++sample) {
        const double x = static_cast<double>(static_cast<std::int32_t>(sample) - half);
        for (std::size_t term = 1; term < terms; ++term) {
            powers[idx(sample, terms, term)] =
                powers[idx(sample, terms, term - 1U)] * x;
        }
    }

    std::vector<double> gram(terms * terms, 0.0);
    for (std::size_t row_term = 0; row_term < terms; ++row_term) {
        for (std::size_t col_term = 0; col_term < terms; ++col_term) {
            double sum = 0.0;
            for (std::size_t sample = 0; sample < window; ++sample) {
                sum += powers[idx(sample, terms, row_term)] *
                       powers[idx(sample, terms, col_term)];
            }
            gram[idx(row_term, terms, col_term)] = sum;
        }
    }

    std::vector<double> rhs(terms, 0.0);
    rhs[static_cast<std::size_t>(params.derivative_order)] =
        factorial(params.derivative_order) /
        std::pow(params.delta, static_cast<double>(params.derivative_order));

    std::vector<double> projection;
    if (!solve_linear_system(gram, rhs, terms, projection)) {
        ctx.set_error("failed to solve Savitzky-Golay normal equations");
        return P4A_ERR_NUMERICAL_FAILURE;
    }

    coeffs.assign(window, 0.0);
    for (std::size_t sample = 0; sample < window; ++sample) {
        double coeff = 0.0;
        for (std::size_t term = 0; term < terms; ++term) {
            coeff += projection[term] * powers[idx(sample, terms, term)];
        }
        coeffs[sample] = coeff;
    }
    return P4A_OK;
}

[[nodiscard]] p4a_status_t apply_savgol(::pls4all::core::Context& ctx,
                                        const std::vector<double>& values,
                                        std::size_t rows,
                                        std::size_t cols,
                                        const SavGolParams& params,
                                        std::vector<double>& out) {
    std::vector<double> coeffs;
    p4a_status_t status = compute_savgol_coeffs(ctx, params, coeffs);
    if (status != P4A_OK) {
        return status;
    }

    const std::int64_t half = static_cast<std::int64_t>(params.window / 2);
    const std::int64_t max_col = static_cast<std::int64_t>(cols - 1U);
    out.assign(rows * cols, 0.0);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            double sum = 0.0;
            for (std::size_t k = 0; k < coeffs.size(); ++k) {
                std::int64_t source =
                    static_cast<std::int64_t>(col) +
                    static_cast<std::int64_t>(k) - half;
                source = std::max<std::int64_t>(0, std::min<std::int64_t>(source, max_col));
                sum += coeffs[k] *
                       values[idx(row, cols, static_cast<std::size_t>(source))];
            }
            out[idx(row, cols, col)] = sum;
        }
    }
    return P4A_OK;
}

[[nodiscard]] p4a_status_t apply_asls(::pls4all::core::Context& ctx,
                                      const std::vector<double>& values,
                                      std::size_t rows,
                                      std::size_t cols,
                                      const AslsParams& params,
                                      std::vector<double>& out) {
    if (cols < 3U) {
        ctx.set_error("ASLS baseline requires at least 3 columns");
        return P4A_ERR_INVALID_ARGUMENT;
    }

    std::vector<double> penalty(cols * cols, 0.0);
    for (std::size_t diff = 0; diff + 2U < cols; ++diff) {
        const std::size_t i0 = diff;
        const std::size_t i1 = diff + 1U;
        const std::size_t i2 = diff + 2U;
        penalty[idx(i0, cols, i0)] += 1.0;
        penalty[idx(i0, cols, i1)] -= 2.0;
        penalty[idx(i0, cols, i2)] += 1.0;
        penalty[idx(i1, cols, i0)] -= 2.0;
        penalty[idx(i1, cols, i1)] += 4.0;
        penalty[idx(i1, cols, i2)] -= 2.0;
        penalty[idx(i2, cols, i0)] += 1.0;
        penalty[idx(i2, cols, i1)] -= 2.0;
        penalty[idx(i2, cols, i2)] += 1.0;
    }
    for (double& value : penalty) {
        value *= params.lambda;
    }

    out.assign(rows * cols, 0.0);
    std::vector<double> weights(cols, 1.0);
    std::vector<double> baseline(cols, 0.0);
    for (std::size_t row = 0; row < rows; ++row) {
        std::fill(weights.begin(), weights.end(), 1.0);
        for (std::int32_t iter = 0; iter < params.iterations; ++iter) {
            std::vector<double> matrix = penalty;
            std::vector<double> rhs(cols, 0.0);
            for (std::size_t col = 0; col < cols; ++col) {
                matrix[idx(col, cols, col)] += weights[col];
                rhs[col] = weights[col] * values[idx(row, cols, col)];
            }
            if (!solve_linear_system(matrix, rhs, cols, baseline)) {
                ctx.set_errorf("failed to solve ASLS baseline system at row %llu",
                               ull(row));
                return P4A_ERR_NUMERICAL_FAILURE;
            }
            for (std::size_t col = 0; col < cols; ++col) {
                weights[col] = values[idx(row, cols, col)] > baseline[col]
                    ? params.asymmetry
                    : 1.0 - params.asymmetry;
            }
        }
        for (std::size_t col = 0; col < cols; ++col) {
            out[idx(row, cols, col)] = values[idx(row, cols, col)] - baseline[col];
        }
    }
    return P4A_OK;
}

[[nodiscard]] p4a_status_t apply_wavelet_denoise(::pls4all::core::Context& ctx,
                                                 const std::vector<double>& values,
                                                 std::size_t rows,
                                                 std::size_t cols,
                                                 const WaveletParams& params,
                                                 std::vector<double>& out) {
    const std::size_t padded = next_power_of_two(cols);
    if (params.levels > max_haar_levels(padded)) {
        ctx.set_error("wavelet denoise levels exceed the padded signal length");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    constexpr double kInvSqrt2 = 0.707106781186547524400844362104849039;
    out.assign(rows * cols, 0.0);
    std::vector<double> work(padded, 0.0);
    std::vector<double> temp(padded, 0.0);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            work[col] = values[idx(row, cols, col)];
        }
        for (std::size_t col = cols; col < padded; ++col) {
            work[col] = work[cols - 1U];
        }

        std::size_t active = padded;
        for (std::int32_t level = 0; level < params.levels; ++level) {
            const std::size_t half = active / 2U;
            for (std::size_t i = 0; i < half; ++i) {
                const double left = work[2U * i];
                const double right = work[2U * i + 1U];
                temp[i] = (left + right) * kInvSqrt2;
                temp[half + i] =
                    soft_threshold((left - right) * kInvSqrt2, params.threshold);
            }
            std::copy(temp.begin(),
                      temp.begin() + static_cast<std::vector<double>::difference_type>(active),
                      work.begin());
            active = half;
        }

        active = padded >> static_cast<unsigned>(params.levels);
        for (std::int32_t level = params.levels - 1; level >= 0; --level) {
            const std::size_t half = active;
            const std::size_t size = half * 2U;
            for (std::size_t i = 0; i < half; ++i) {
                const double average = work[i];
                const double detail = work[half + i];
                temp[2U * i] = (average + detail) * kInvSqrt2;
                temp[2U * i + 1U] = (average - detail) * kInvSqrt2;
            }
            std::copy(temp.begin(),
                      temp.begin() + static_cast<std::vector<double>::difference_type>(size),
                      work.begin());
            active = size;
        }

        for (std::size_t col = 0; col < cols; ++col) {
            out[idx(row, cols, col)] = work[col];
        }
    }
    return P4A_OK;
}

[[nodiscard]] p4a_status_t apply_osc_filter(::pls4all::core::Context& ctx,
                                            const std::vector<double>& values,
                                            std::size_t rows,
                                            std::size_t cols,
                                            const std::vector<double>& mean,
                                            const std::vector<double>& weights,
                                            const std::vector<double>& loadings,
                                            std::vector<double>& out) {
    if (mean.size() != cols || weights.size() != cols || loadings.size() != cols) {
        ctx.set_error("OSC fitted state has inconsistent dimensions");
        return P4A_ERR_INTERNAL;
    }
    out.assign(rows * cols, 0.0);
    for (std::size_t row = 0; row < rows; ++row) {
        double score = 0.0;
        for (std::size_t col = 0; col < cols; ++col) {
            score += (values[idx(row, cols, col)] - mean[col]) * weights[col];
        }
        for (std::size_t col = 0; col < cols; ++col) {
            out[idx(row, cols, col)] = values[idx(row, cols, col)] - score * loadings[col];
        }
    }
    return P4A_OK;
}

[[nodiscard]] p4a_status_t fit_osc(::pls4all::core::Context& ctx,
                                   const std::vector<double>& values,
                                   const std::vector<double>& y_values,
                                   std::size_t rows,
                                   std::size_t cols,
                                   std::size_t y_cols,
                                   const OscParams& params,
                                   std::vector<double>& mean,
                                   std::vector<double>& weights,
                                   std::vector<double>& loadings,
                                   std::vector<double>& out) {
    std::vector<double> x_centered;
    center_columns(values, rows, cols, mean, x_centered);

    std::vector<double> y_mean;
    std::vector<double> y_centered;
    center_columns(y_values, rows, y_cols, y_mean, y_centered);

    std::vector<double> x_orthogonal;
    p4a_status_t status =
        y_orthogonal_residual(ctx, x_centered, y_centered, rows, cols, y_cols, x_orthogonal);
    if (status != P4A_OK) {
        return status;
    }

    status = dominant_weight(ctx, x_orthogonal, rows, cols, params, weights);
    if (status != P4A_OK) {
        return status;
    }

    std::vector<double> scores(rows, 0.0);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            scores[row] += x_centered[idx(row, cols, col)] * weights[col];
        }
    }
    const double score_ss = vector_sumsq(scores);
    if (score_ss <= std::numeric_limits<double>::epsilon()) {
        ctx.set_error("OSC training score vanished");
        return P4A_ERR_NUMERICAL_FAILURE;
    }

    loadings.assign(cols, 0.0);
    for (std::size_t col = 0; col < cols; ++col) {
        double sum = 0.0;
        for (std::size_t row = 0; row < rows; ++row) {
            sum += x_centered[idx(row, cols, col)] * scores[row];
        }
        loadings[col] = sum / score_ss;
    }
    return apply_osc_filter(ctx, values, rows, cols, mean, weights, loadings, out);
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

[[nodiscard]] p4a_status_t apply_emsc(::pls4all::core::Context& ctx,
                                      const std::vector<double>& values,
                                      const std::vector<double>& reference,
                                      std::size_t rows,
                                      std::size_t cols,
                                      std::int32_t degree,
                                      std::vector<double>& out) {
    if (degree < 0) {
        ctx.set_error("EMSC polynomial degree must be non-negative");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    const std::size_t poly_terms = static_cast<std::size_t>(degree) + 1U;
    const std::size_t terms = poly_terms + 1U;
    if (terms > cols) {
        ctx.set_error("EMSC design requires at least degree + 2 columns");
        return P4A_ERR_INVALID_ARGUMENT;
    }

    std::vector<double> x(cols, 0.0);
    for (std::size_t col = 0; col < cols; ++col) {
        x[col] = cols > 1U
            ? -1.0 + 2.0 * static_cast<double>(col) / static_cast<double>(cols - 1U)
            : 0.0;
    }

    std::vector<double> design(cols * terms, 1.0);
    for (std::size_t col = 0; col < cols; ++col) {
        design[idx(col, terms, 0)] = reference[col];
        design[idx(col, terms, 1)] = 1.0;
        for (std::size_t term = 2; term < terms; ++term) {
            design[idx(col, terms, term)] =
                design[idx(col, terms, term - 1U)] * x[col];
        }
    }

    std::vector<double> gram(terms * terms, 0.0);
    for (std::size_t row_term = 0; row_term < terms; ++row_term) {
        for (std::size_t col_term = 0; col_term < terms; ++col_term) {
            double sum = 0.0;
            for (std::size_t col = 0; col < cols; ++col) {
                sum += design[idx(col, terms, row_term)] *
                       design[idx(col, terms, col_term)];
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
                sum += values[idx(row, cols, col)] * design[idx(col, terms, term)];
            }
            rhs[term] = sum;
        }
        std::vector<double> coeffs;
        if (!solve_linear_system(gram, rhs, terms, coeffs)) {
            ctx.set_error("failed to solve EMSC normal equations");
            return P4A_ERR_NUMERICAL_FAILURE;
        }
        const double slope = coeffs[0];
        if (std::fabs(slope) <= std::numeric_limits<double>::epsilon() ||
            !std::isfinite(slope)) {
            ctx.set_errorf("EMSC multiplicative coefficient vanished at row %llu",
                           ull(row));
            return P4A_ERR_NUMERICAL_FAILURE;
        }
        for (std::size_t col = 0; col < cols; ++col) {
            double baseline = 0.0;
            for (std::size_t term = 1; term < terms; ++term) {
                baseline += coeffs[term] * design[idx(col, terms, term)];
            }
            out[idx(row, cols, col)] = (values[idx(row, cols, col)] - baseline) / slope;
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

[[nodiscard]] p4a_status_t parse_emsc_degree(::pls4all::core::Context& ctx,
                                             const ::pls4all::core::OperatorEntry& entry,
                                             std::int32_t& degree) {
    degree = 2;
    if (entry.params.empty()) {
        return P4A_OK;
    }
    if (entry.params.size() != 1U) {
        ctx.set_error("EMSC expects zero params or one polynomial degree param");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    const double raw = entry.params[0];
    const double rounded = std::round(raw);
    if (!std::isfinite(raw) || std::fabs(raw - rounded) > 1e-12 ||
        rounded < 0.0 || rounded > 5.0) {
        ctx.set_error("EMSC polynomial degree must be an integer in [0, 5]");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    degree = static_cast<std::int32_t>(rounded);
    return P4A_OK;
}

[[nodiscard]] p4a_status_t parse_integer_param(::pls4all::core::Context& ctx,
                                               double raw,
                                               std::int32_t min_value,
                                               std::int32_t max_value,
                                               const char* name,
                                               std::int32_t& out) {
    const double rounded = std::round(raw);
    if (!std::isfinite(raw) || std::fabs(raw - rounded) > 1e-12 ||
        rounded < static_cast<double>(min_value) ||
        rounded > static_cast<double>(max_value)) {
        ctx.set_errorf("%s must be an integer in [%d, %d]",
                       name,
                       static_cast<int>(min_value),
                       static_cast<int>(max_value));
        return P4A_ERR_INVALID_ARGUMENT;
    }
    out = static_cast<std::int32_t>(rounded);
    return P4A_OK;
}

[[nodiscard]] p4a_status_t validate_savgol_params(::pls4all::core::Context& ctx,
                                                  const SavGolParams& params) {
    if ((params.window % 2) == 0 || params.window < 3 || params.window > 501) {
        ctx.set_error("Savitzky-Golay window length must be an odd integer in [3, 501]");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (params.poly_degree < 0 || params.poly_degree >= params.window) {
        ctx.set_error("Savitzky-Golay polynomial degree must be non-negative and smaller than the window length");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (params.derivative_order < 0 || params.derivative_order > 2) {
        ctx.set_error("Savitzky-Golay derivative order must be 0, 1 or 2");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (params.derivative_order > params.poly_degree) {
        ctx.set_error("Savitzky-Golay derivative order must not exceed polynomial degree");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (!std::isfinite(params.delta) || params.delta <= 0.0) {
        ctx.set_error("Savitzky-Golay delta must be finite and positive");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    return P4A_OK;
}

[[nodiscard]] p4a_status_t parse_savgol_smooth(::pls4all::core::Context& ctx,
                                               const ::pls4all::core::OperatorEntry& entry,
                                               SavGolParams& params) {
    params = SavGolParams{};
    params.derivative_order = 0;
    if (entry.params.empty()) {
        return validate_savgol_params(ctx, params);
    }
    if (entry.params.size() != 2U) {
        ctx.set_error("Savitzky-Golay smooth expects zero params or window/poly_degree");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    p4a_status_t status = parse_integer_param(ctx, entry.params[0], 3, 501,
                                              "Savitzky-Golay window length",
                                              params.window);
    if (status != P4A_OK) {
        return status;
    }
    status = parse_integer_param(ctx, entry.params[1], 0, 500,
                                 "Savitzky-Golay polynomial degree",
                                 params.poly_degree);
    if (status != P4A_OK) {
        return status;
    }
    return validate_savgol_params(ctx, params);
}

[[nodiscard]] p4a_status_t parse_savgol_derivative(::pls4all::core::Context& ctx,
                                                   const ::pls4all::core::OperatorEntry& entry,
                                                   SavGolParams& params) {
    params = SavGolParams{};
    params.derivative_order = 1;
    if (entry.params.empty()) {
        return validate_savgol_params(ctx, params);
    }
    if (entry.params.size() != 3U && entry.params.size() != 4U) {
        ctx.set_error("Savitzky-Golay derivative expects zero params or window/poly_degree/derivative_order[/delta]");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    p4a_status_t status = parse_integer_param(ctx, entry.params[0], 3, 501,
                                              "Savitzky-Golay window length",
                                              params.window);
    if (status != P4A_OK) {
        return status;
    }
    status = parse_integer_param(ctx, entry.params[1], 0, 500,
                                 "Savitzky-Golay polynomial degree",
                                 params.poly_degree);
    if (status != P4A_OK) {
        return status;
    }
    status = parse_integer_param(ctx, entry.params[2], 1, 2,
                                 "Savitzky-Golay derivative order",
                                 params.derivative_order);
    if (status != P4A_OK) {
        return status;
    }
    if (entry.params.size() == 4U) {
        params.delta = entry.params[3];
    }
    return validate_savgol_params(ctx, params);
}

[[nodiscard]] p4a_status_t parse_asls(::pls4all::core::Context& ctx,
                                      const ::pls4all::core::OperatorEntry& entry,
                                      AslsParams& params) {
    params = AslsParams{};
    if (entry.params.empty()) {
        return P4A_OK;
    }
    if (entry.params.size() != 3U) {
        ctx.set_error("ASLS baseline expects zero params or lambda/asymmetry/iterations");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    params.lambda = entry.params[0];
    params.asymmetry = entry.params[1];
    if (!std::isfinite(params.lambda) || params.lambda <= 0.0) {
        ctx.set_error("ASLS lambda must be finite and positive");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (!std::isfinite(params.asymmetry) ||
        params.asymmetry <= 0.0 || params.asymmetry >= 1.0) {
        ctx.set_error("ASLS asymmetry must be finite and in (0, 1)");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    return parse_integer_param(ctx, entry.params[2], 1, 100,
                               "ASLS iteration count",
                               params.iterations);
}

[[nodiscard]] p4a_status_t validate_norris_williams(::pls4all::core::Context& ctx,
                                                    const NorrisWilliamsParams& params) {
    if ((params.segment % 2) == 0 || params.segment < 1 || params.segment > 101) {
        ctx.set_error("Norris-Williams segment size must be an odd integer in [1, 101]");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if ((params.gap % 2) == 0 || params.gap < 1 || params.gap > 101) {
        ctx.set_error("Norris-Williams gap size must be an odd integer in [1, 101]");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (params.derivative_order < 1 || params.derivative_order > 4) {
        ctx.set_error("Norris-Williams derivative order must be an integer in [1, 4]");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    const std::int32_t filter_length =
        (params.derivative_order + 1) * params.segment +
        params.derivative_order * params.gap;
    if (filter_length > 501) {
        ctx.set_error("Norris-Williams filter length must not exceed 501");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    return P4A_OK;
}

[[nodiscard]] p4a_status_t parse_norris_williams(::pls4all::core::Context& ctx,
                                                 const ::pls4all::core::OperatorEntry& entry,
                                                 NorrisWilliamsParams& params) {
    params = NorrisWilliamsParams{};
    if (entry.params.empty()) {
        return validate_norris_williams(ctx, params);
    }
    if (entry.params.size() != 3U) {
        ctx.set_error("Norris-Williams expects zero params or segment/gap/derivative_order");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    p4a_status_t status = parse_integer_param(ctx, entry.params[0], 1, 101,
                                              "Norris-Williams segment size",
                                              params.segment);
    if (status != P4A_OK) {
        return status;
    }
    status = parse_integer_param(ctx, entry.params[1], 1, 101,
                                 "Norris-Williams gap size",
                                 params.gap);
    if (status != P4A_OK) {
        return status;
    }
    status = parse_integer_param(ctx, entry.params[2], 1, 4,
                                 "Norris-Williams derivative order",
                                 params.derivative_order);
    if (status != P4A_OK) {
        return status;
    }
    return validate_norris_williams(ctx, params);
}

[[nodiscard]] p4a_status_t parse_wavelet(::pls4all::core::Context& ctx,
                                         const ::pls4all::core::OperatorEntry& entry,
                                         WaveletParams& params) {
    params = WaveletParams{};
    if (entry.params.empty()) {
        return P4A_OK;
    }
    if (entry.params.size() != 2U) {
        ctx.set_error("wavelet denoise expects zero params or levels/threshold");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    p4a_status_t status = parse_integer_param(ctx, entry.params[0], 0, 20,
                                              "wavelet denoise levels",
                                              params.levels);
    if (status != P4A_OK) {
        return status;
    }
    params.threshold = entry.params[1];
    if (!std::isfinite(params.threshold) || params.threshold < 0.0) {
        ctx.set_error("wavelet denoise threshold must be finite and non-negative");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    return P4A_OK;
}

[[nodiscard]] p4a_status_t parse_osc(::pls4all::core::Context& ctx,
                                     const ::pls4all::core::OperatorEntry& entry,
                                     OscParams& params) {
    params = OscParams{};
    if (entry.params.empty()) {
        return P4A_OK;
    }
    if (entry.params.size() != 2U) {
        ctx.set_error("OSC expects zero params or max_iter/tol");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    p4a_status_t status = parse_integer_param(ctx, entry.params[0], 1, 1000,
                                              "OSC max_iter",
                                              params.max_iter);
    if (status != P4A_OK) {
        return status;
    }
    params.tol = entry.params[1];
    if (!std::isfinite(params.tol) || params.tol <= 0.0) {
        ctx.set_error("OSC tol must be finite and positive");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    return P4A_OK;
}

}  // namespace

namespace pls4all::core {

p4a_status_t Pipeline::fit(Context& ctx,
                           const p4a_matrix_view_t& X,
                           const p4a_matrix_view_t* Y) {
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

    const bool needs_y = pipeline_requires_y(entries_);
    std::vector<double> y_values;
    std::size_t y_cols = 0;
    if (needs_y) {
        if (Y == nullptr) {
            ctx.set_error("pipeline operator OSC requires a non-null Y matrix at fit");
            return P4A_ERR_INVALID_ARGUMENT;
        }
        status = validate_float_view(ctx, *Y, "Y");
        if (status != P4A_OK) {
            return status;
        }
        if (Y->rows != X.rows) {
            ctx.set_errorf("Y rows (%lld) must match X rows (%lld) for supervised preprocessing",
                           static_cast<long long>(Y->rows),
                           static_cast<long long>(X.rows));
            return P4A_ERR_SHAPE_MISMATCH;
        }
        if (Y->cols < 1) {
            ctx.set_error("supervised preprocessing requires at least one Y column");
            return P4A_ERR_INVALID_ARGUMENT;
        }
        status = copy_matrix_checked(ctx, *Y, "Y", y_values);
        if (status != P4A_OK) {
            return status;
        }
        y_cols = static_cast<std::size_t>(Y->cols);
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
            case P4A_OP_EMSC: {
                std::int32_t degree = 2;
                status = parse_emsc_degree(ctx, entry, degree);
                if (status != P4A_OK) {
                    states_.clear();
                    return status;
                }
                fit_column_center(current, rows, cols, state.location);
                state.scale.assign(1U, static_cast<double>(degree));
                status = apply_emsc(ctx, current, state.location, rows, cols, degree, next);
                if (status != P4A_OK) {
                    states_.clear();
                    return status;
                }
                break;
            }
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
            case P4A_OP_SAVGOL_SMOOTH: {
                SavGolParams params;
                status = parse_savgol_smooth(ctx, entry, params);
                if (status != P4A_OK) {
                    states_.clear();
                    return status;
                }
                state.scale.assign({
                    static_cast<double>(params.window),
                    static_cast<double>(params.poly_degree),
                    static_cast<double>(params.derivative_order),
                    params.delta,
                });
                status = apply_savgol(ctx, current, rows, cols, params, next);
                if (status != P4A_OK) {
                    states_.clear();
                    return status;
                }
                break;
            }
            case P4A_OP_SAVGOL_DERIVATIVE: {
                SavGolParams params;
                status = parse_savgol_derivative(ctx, entry, params);
                if (status != P4A_OK) {
                    states_.clear();
                    return status;
                }
                state.scale.assign({
                    static_cast<double>(params.window),
                    static_cast<double>(params.poly_degree),
                    static_cast<double>(params.derivative_order),
                    params.delta,
                });
                status = apply_savgol(ctx, current, rows, cols, params, next);
                if (status != P4A_OK) {
                    states_.clear();
                    return status;
                }
                break;
            }
            case P4A_OP_ASLS_BASELINE: {
                AslsParams params;
                status = parse_asls(ctx, entry, params);
                if (status != P4A_OK) {
                    states_.clear();
                    return status;
                }
                state.scale.assign({
                    params.lambda,
                    params.asymmetry,
                    static_cast<double>(params.iterations),
                });
                status = apply_asls(ctx, current, rows, cols, params, next);
                if (status != P4A_OK) {
                    states_.clear();
                    return status;
                }
                break;
            }
            case P4A_OP_NORRIS_WILLIAMS: {
                NorrisWilliamsParams params;
                status = parse_norris_williams(ctx, entry, params);
                if (status != P4A_OK) {
                    states_.clear();
                    return status;
                }
                state.scale.assign({
                    static_cast<double>(params.segment),
                    static_cast<double>(params.gap),
                    static_cast<double>(params.derivative_order),
                });
                status = apply_norris_williams(ctx, current, rows, cols, params, next);
                if (status != P4A_OK) {
                    states_.clear();
                    return status;
                }
                break;
            }
            case P4A_OP_WAVELET_DENOISE: {
                WaveletParams params;
                status = parse_wavelet(ctx, entry, params);
                if (status != P4A_OK) {
                    states_.clear();
                    return status;
                }
                state.scale.assign({
                    static_cast<double>(params.levels),
                    params.threshold,
                });
                status = apply_wavelet_denoise(ctx, current, rows, cols, params, next);
                if (status != P4A_OK) {
                    states_.clear();
                    return status;
                }
                break;
            }
            case P4A_OP_OSC: {
                OscParams params;
                status = parse_osc(ctx, entry, params);
                if (status != P4A_OK) {
                    states_.clear();
                    return status;
                }
                state.scale.reserve(cols);
                state.extra.reserve(cols);
                status = fit_osc(ctx, current, y_values, rows, cols, y_cols, params,
                                 state.location, state.scale, state.extra, next);
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
            case P4A_OP_EMSC:
                if (state.scale.empty()) {
                    ctx.set_error("fitted EMSC operator is missing its degree");
                    return P4A_ERR_INTERNAL;
                }
                status = apply_emsc(ctx, current, state.location, rows, cols,
                                    static_cast<std::int32_t>(state.scale[0]), next);
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
            case P4A_OP_SAVGOL_SMOOTH:
            case P4A_OP_SAVGOL_DERIVATIVE: {
                if (state.scale.size() != 4U) {
                    ctx.set_error("fitted Savitzky-Golay operator is missing parameters");
                    return P4A_ERR_INTERNAL;
                }
                SavGolParams params;
                params.window = static_cast<std::int32_t>(state.scale[0]);
                params.poly_degree = static_cast<std::int32_t>(state.scale[1]);
                params.derivative_order = static_cast<std::int32_t>(state.scale[2]);
                params.delta = state.scale[3];
                status = apply_savgol(ctx, current, rows, cols, params, next);
                if (status != P4A_OK) {
                    return status;
                }
                break;
            }
            case P4A_OP_ASLS_BASELINE: {
                if (state.scale.size() != 3U) {
                    ctx.set_error("fitted ASLS operator is missing parameters");
                    return P4A_ERR_INTERNAL;
                }
                AslsParams params;
                params.lambda = state.scale[0];
                params.asymmetry = state.scale[1];
                params.iterations = static_cast<std::int32_t>(state.scale[2]);
                status = apply_asls(ctx, current, rows, cols, params, next);
                if (status != P4A_OK) {
                    return status;
                }
                break;
            }
            case P4A_OP_NORRIS_WILLIAMS: {
                if (state.scale.size() != 3U) {
                    ctx.set_error("fitted Norris-Williams operator is missing parameters");
                    return P4A_ERR_INTERNAL;
                }
                NorrisWilliamsParams params;
                params.segment = static_cast<std::int32_t>(state.scale[0]);
                params.gap = static_cast<std::int32_t>(state.scale[1]);
                params.derivative_order = static_cast<std::int32_t>(state.scale[2]);
                status = apply_norris_williams(ctx, current, rows, cols, params, next);
                if (status != P4A_OK) {
                    return status;
                }
                break;
            }
            case P4A_OP_WAVELET_DENOISE: {
                if (state.scale.size() != 2U) {
                    ctx.set_error("fitted wavelet denoise operator is missing parameters");
                    return P4A_ERR_INTERNAL;
                }
                WaveletParams params;
                params.levels = static_cast<std::int32_t>(state.scale[0]);
                params.threshold = state.scale[1];
                status = apply_wavelet_denoise(ctx, current, rows, cols, params, next);
                if (status != P4A_OK) {
                    return status;
                }
                break;
            }
            case P4A_OP_OSC: {
                status = apply_osc_filter(ctx, current, rows, cols,
                                          state.location, state.scale, state.extra, next);
                if (status != P4A_OK) {
                    return status;
                }
                break;
            }
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
