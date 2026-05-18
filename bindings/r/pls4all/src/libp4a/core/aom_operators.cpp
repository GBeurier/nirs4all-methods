// SPDX-License-Identifier: CECILL-2.1

#include "core/aom_operators.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "core/matrix_view.hpp"
#include "core/status.hpp"

namespace {

[[nodiscard]] std::size_t idx(std::size_t row, std::size_t cols, std::size_t col) noexcept {
    return row * cols + col;
}

[[nodiscard]] p4a_status_t element_count(::pls4all::core::Context& ctx,
                                         std::int64_t rows,
                                         std::int64_t cols,
                                         std::size_t& out) noexcept {
    if (rows < 0 || cols < 0) {
        ctx.set_error("AOM operator matrix shape is invalid");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    const auto urows = static_cast<std::uint64_t>(rows);
    const auto ucols = static_cast<std::uint64_t>(cols);
    if (ucols != 0U &&
        urows > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) / ucols) {
        ctx.set_error("AOM operator matrix shape is too large");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    out = static_cast<std::size_t>(urows * ucols);
    return P4A_OK;
}

[[nodiscard]] double read_value(const p4a_matrix_view_t& view,
                                std::size_t row,
                                std::size_t col) noexcept {
    const std::int64_t off =
        static_cast<std::int64_t>(row) * view.row_stride +
        static_cast<std::int64_t>(col) * view.col_stride;
    const auto uoff = static_cast<std::size_t>(off);
    if (view.dtype == P4A_DTYPE_F64) {
        const auto* ptr = static_cast<const double*>(view.data);
        return ptr[uoff];
    }
    const auto* ptr = static_cast<const float*>(view.data);
    return static_cast<double>(ptr[uoff]);
}

[[nodiscard]] p4a_status_t validate_float_view(::pls4all::core::Context& ctx,
                                               const p4a_matrix_view_t& view,
                                               const char* name) noexcept {
    const p4a_status_t status = ::pls4all::core::validate_nonnull_view(view);
    if (status != P4A_OK) {
        ctx.set_errorf("%s matrix view is invalid: %s",
                       name,
                       ::pls4all::core::status_to_string(status));
        return status;
    }
    if (view.dtype != P4A_DTYPE_F64 && view.dtype != P4A_DTYPE_F32) {
        ctx.set_errorf("%s dtype must be f64 or f32", name);
        return P4A_ERR_DTYPE_MISMATCH;
    }
    if (view.rows <= 0 || view.cols <= 0) {
        ctx.set_errorf("%s matrix must be non-empty", name);
        return P4A_ERR_INVALID_ARGUMENT;
    }
    return P4A_OK;
}

[[nodiscard]] p4a_status_t copy_matrix_checked(::pls4all::core::Context& ctx,
                                               const p4a_matrix_view_t& src,
                                               std::vector<double>& out) {
    std::size_t n_values = 0;
    p4a_status_t status = element_count(ctx, src.rows, src.cols, n_values);
    if (status != P4A_OK) {
        return status;
    }
    out.assign(n_values, 0.0);
    const auto rows = static_cast<std::size_t>(src.rows);
    const auto cols = static_cast<std::size_t>(src.cols);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            const double value = read_value(src, row, col);
            if (!std::isfinite(value)) {
                ctx.set_error("AOM operator input contains NaN or Inf");
                return P4A_ERR_INVALID_ARGUMENT;
            }
            out[idx(row, cols, col)] = value;
        }
    }
    return P4A_OK;
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

[[nodiscard]] p4a_status_t require_no_params(::pls4all::core::Context& ctx,
                                             const ::pls4all::core::OperatorEntry& entry,
                                             const char* name) {
    if (!entry.params.empty()) {
        ctx.set_errorf("%s does not accept parameters", name);
        return P4A_ERR_INVALID_ARGUMENT;
    }
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

[[nodiscard]] p4a_status_t parse_degree(::pls4all::core::Context& ctx,
                                        const ::pls4all::core::OperatorEntry& entry,
                                        std::int32_t& degree) {
    degree = 1;
    if (entry.params.empty()) {
        return P4A_OK;
    }
    if (entry.params.size() != 1U) {
        ctx.set_error("AOM detrend expects zero params or one degree param");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    return parse_integer_param(ctx, entry.params[0], 0, 5, "AOM detrend degree", degree);
}

struct SavGolParams {
    std::int32_t window{5};
    std::int32_t poly_degree{2};
    std::int32_t derivative_order{0};
};

[[nodiscard]] double factorial(std::int32_t n) noexcept {
    double out = 1.0;
    for (std::int32_t i = 2; i <= n; ++i) {
        out *= static_cast<double>(i);
    }
    return out;
}

[[nodiscard]] p4a_status_t validate_savgol_params(::pls4all::core::Context& ctx,
                                                  const SavGolParams& params) {
    if ((params.window % 2) == 0 || params.window < 3 || params.window > 501) {
        ctx.set_error("AOM Savitzky-Golay window length must be an odd integer in [3, 501]");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (params.poly_degree < 0 || params.poly_degree >= params.window) {
        ctx.set_error("AOM Savitzky-Golay polynomial degree must be non-negative and smaller than the window length");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (params.derivative_order < 0 || params.derivative_order > 2) {
        ctx.set_error("AOM Savitzky-Golay derivative order must be 0, 1 or 2");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (params.derivative_order > params.poly_degree) {
        ctx.set_error("AOM Savitzky-Golay derivative order must not exceed polynomial degree");
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
        ctx.set_error("AOM Savitzky-Golay smooth expects zero params or window/poly_degree");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    p4a_status_t status = parse_integer_param(ctx,
                                              entry.params[0],
                                              3,
                                              501,
                                              "AOM Savitzky-Golay window length",
                                              params.window);
    if (status != P4A_OK) {
        return status;
    }
    status = parse_integer_param(ctx,
                                 entry.params[1],
                                 0,
                                 500,
                                 "AOM Savitzky-Golay polynomial degree",
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
    if (entry.params.size() != 3U) {
        ctx.set_error("AOM Savitzky-Golay derivative expects zero params or window/poly_degree/derivative_order");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    p4a_status_t status = parse_integer_param(ctx,
                                              entry.params[0],
                                              3,
                                              501,
                                              "AOM Savitzky-Golay window length",
                                              params.window);
    if (status != P4A_OK) {
        return status;
    }
    status = parse_integer_param(ctx,
                                 entry.params[1],
                                 0,
                                 500,
                                 "AOM Savitzky-Golay polynomial degree",
                                 params.poly_degree);
    if (status != P4A_OK) {
        return status;
    }
    status = parse_integer_param(ctx,
                                 entry.params[2],
                                 1,
                                 2,
                                 "AOM Savitzky-Golay derivative order",
                                 params.derivative_order);
    if (status != P4A_OK) {
        return status;
    }
    return validate_savgol_params(ctx, params);
}

[[nodiscard]] p4a_status_t compute_savgol_coeffs(::pls4all::core::Context& ctx,
                                                 const SavGolParams& params,
                                                 std::vector<double>& coeffs) {
    const auto window = static_cast<std::size_t>(params.window);
    const auto terms = static_cast<std::size_t>(params.poly_degree) + 1U;
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
    rhs[static_cast<std::size_t>(params.derivative_order)] = factorial(params.derivative_order);

    std::vector<double> projection;
    if (!solve_linear_system(gram, rhs, terms, projection)) {
        ctx.set_error("failed to solve AOM Savitzky-Golay normal equations");
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

[[nodiscard]] p4a_status_t apply_xcorr_zero_pad(const std::vector<double>& values,
                                                std::size_t rows,
                                                std::size_t cols,
                                                const std::vector<double>& kernel,
                                                std::vector<double>& out) {
    if (kernel.empty()) {
        return P4A_ERR_INVALID_ARGUMENT;
    }
    const std::int64_t half = static_cast<std::int64_t>((kernel.size() - 1U) / 2U);
    out.assign(rows * cols, 0.0);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            double sum = 0.0;
            for (std::size_t k = 0; k < kernel.size(); ++k) {
                const std::int64_t source =
                    static_cast<std::int64_t>(col) +
                    static_cast<std::int64_t>(k) - half;
                if (source >= 0 && source < static_cast<std::int64_t>(cols)) {
                    sum += kernel[k] * values[idx(row, cols, static_cast<std::size_t>(source))];
                }
            }
            out[idx(row, cols, col)] = sum;
        }
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
    return apply_xcorr_zero_pad(values, rows, cols, coeffs, out);
}

[[nodiscard]] p4a_status_t apply_detrend_poly(::pls4all::core::Context& ctx,
                                              const std::vector<double>& values,
                                              std::size_t rows,
                                              std::size_t cols,
                                              std::int32_t degree,
                                              std::vector<double>& out) {
    if (degree < 0) {
        ctx.set_error("AOM detrend degree must be non-negative");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (static_cast<std::size_t>(degree) >= cols) {
        ctx.set_error("AOM detrend degree must be smaller than the column count");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    const auto terms = static_cast<std::size_t>(degree) + 1U;
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
            ctx.set_error("failed to solve AOM detrend normal equations");
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

[[nodiscard]] p4a_status_t apply_finite_difference(::pls4all::core::Context& ctx,
                                                   const ::pls4all::core::OperatorEntry& entry,
                                                   const std::vector<double>& values,
                                                   std::size_t rows,
                                                   std::size_t cols,
                                                   std::vector<double>& out) {
    std::int32_t order = 1;
    if (!entry.params.empty()) {
        if (entry.params.size() != 1U) {
            ctx.set_error("AOM finite difference expects zero params or one order param");
            return P4A_ERR_INVALID_ARGUMENT;
        }
        const p4a_status_t status = parse_integer_param(ctx,
                                                        entry.params[0],
                                                        1,
                                                        2,
                                                        "AOM finite difference order",
                                                        order);
        if (status != P4A_OK) {
            return status;
        }
    }
    std::vector<double> kernel;
    if (order == 1) {
        kernel = {-0.5, 0.0, 0.5};
    } else {
        kernel = {1.0, -2.0, 1.0};
    }
    return apply_xcorr_zero_pad(values, rows, cols, kernel, out);
}

struct NorrisWilliamsParams {
    std::int32_t smoothing{5};
    std::int32_t gap{5};
    std::int32_t derivative_order{1};
};

[[nodiscard]] p4a_status_t parse_norris_williams(::pls4all::core::Context& ctx,
                                                 const ::pls4all::core::OperatorEntry& entry,
                                                 NorrisWilliamsParams& params) {
    params = NorrisWilliamsParams{};
    if (entry.params.empty()) {
        return P4A_OK;
    }
    if (entry.params.size() != 3U) {
        ctx.set_error("AOM Norris-Williams expects zero params or smoothing/gap/derivative_order");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    p4a_status_t status = parse_integer_param(ctx,
                                              entry.params[0],
                                              1,
                                              101,
                                              "AOM Norris-Williams smoothing",
                                              params.smoothing);
    if (status != P4A_OK) {
        return status;
    }
    status = parse_integer_param(ctx,
                                 entry.params[1],
                                 1,
                                 101,
                                 "AOM Norris-Williams gap",
                                 params.gap);
    if (status != P4A_OK) {
        return status;
    }
    status = parse_integer_param(ctx,
                                 entry.params[2],
                                 1,
                                 2,
                                 "AOM Norris-Williams derivative order",
                                 params.derivative_order);
    if (status != P4A_OK) {
        return status;
    }
    if ((params.smoothing % 2) == 0) {
        ctx.set_error("AOM Norris-Williams smoothing must be odd");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    return P4A_OK;
}

[[nodiscard]] std::vector<double> convolve(const std::vector<double>& left,
                                           const std::vector<double>& right) {
    std::vector<double> out(left.size() + right.size() - 1U, 0.0);
    for (std::size_t i = 0; i < left.size(); ++i) {
        for (std::size_t j = 0; j < right.size(); ++j) {
            out[i + j] += left[i] * right[j];
        }
    }
    return out;
}

[[nodiscard]] p4a_status_t apply_norris_williams(::pls4all::core::Context& ctx,
                                                 const ::pls4all::core::OperatorEntry& entry,
                                                 const std::vector<double>& values,
                                                 std::size_t rows,
                                                 std::size_t cols,
                                                 std::vector<double>& out) {
    NorrisWilliamsParams params;
    p4a_status_t status = parse_norris_williams(ctx, entry, params);
    if (status != P4A_OK) {
        return status;
    }
    std::vector<double> smooth(static_cast<std::size_t>(params.smoothing),
                               1.0 / static_cast<double>(params.smoothing));
    std::vector<double> gap_kernel(static_cast<std::size_t>(2 * params.gap + 1), 0.0);
    gap_kernel.front() = -1.0 / (2.0 * static_cast<double>(params.gap));
    gap_kernel.back() = 1.0 / (2.0 * static_cast<double>(params.gap));
    std::vector<double> kernel = convolve(smooth, gap_kernel);
    if (params.derivative_order == 2) {
        kernel = convolve(kernel, gap_kernel);
    }
    return apply_xcorr_zero_pad(values, rows, cols, kernel, out);
}

[[nodiscard]] p4a_status_t apply_whittaker(::pls4all::core::Context& ctx,
                                           const ::pls4all::core::OperatorEntry& entry,
                                           const std::vector<double>& values,
                                           std::size_t rows,
                                           std::size_t cols,
                                           std::vector<double>& out) {
    double lambda = 1000.0;
    if (!entry.params.empty()) {
        if (entry.params.size() != 1U) {
            ctx.set_error("AOM Whittaker expects zero params or one lambda param");
            return P4A_ERR_INVALID_ARGUMENT;
        }
        lambda = entry.params[0];
    }
    if (!std::isfinite(lambda) || lambda <= 0.0) {
        ctx.set_error("AOM Whittaker lambda must be finite and positive");
        return P4A_ERR_INVALID_ARGUMENT;
    }

    std::vector<double> system(cols * cols, 0.0);
    for (std::size_t col = 0; col < cols; ++col) {
        system[idx(col, cols, col)] = 1.0;
    }
    if (cols >= 3U) {
        for (std::size_t diff = 0; diff + 2U < cols; ++diff) {
            const std::size_t i0 = diff;
            const std::size_t i1 = diff + 1U;
            const std::size_t i2 = diff + 2U;
            system[idx(i0, cols, i0)] += lambda;
            system[idx(i0, cols, i1)] -= 2.0 * lambda;
            system[idx(i0, cols, i2)] += lambda;
            system[idx(i1, cols, i0)] -= 2.0 * lambda;
            system[idx(i1, cols, i1)] += 4.0 * lambda;
            system[idx(i1, cols, i2)] -= 2.0 * lambda;
            system[idx(i2, cols, i0)] += lambda;
            system[idx(i2, cols, i1)] -= 2.0 * lambda;
            system[idx(i2, cols, i2)] += lambda;
        }
    }

    out.assign(rows * cols, 0.0);
    for (std::size_t row = 0; row < rows; ++row) {
        std::vector<double> rhs(cols, 0.0);
        for (std::size_t col = 0; col < cols; ++col) {
            rhs[col] = values[idx(row, cols, col)];
        }
        std::vector<double> solution;
        if (!solve_linear_system(system, rhs, cols, solution)) {
            ctx.set_error("failed to solve AOM Whittaker system");
            return P4A_ERR_NUMERICAL_FAILURE;
        }
        for (std::size_t col = 0; col < cols; ++col) {
            out[idx(row, cols, col)] = solution[col];
        }
    }
    return P4A_OK;
}

struct FckParams {
    double alpha{1.0};
    double scale{1.0};
    std::int32_t kernel_size{31};
    double sigma{3.0};
};

[[nodiscard]] p4a_status_t parse_fck(::pls4all::core::Context& ctx,
                                     const ::pls4all::core::OperatorEntry& entry,
                                     FckParams& params) {
    params = FckParams{};
    if (entry.params.empty()) {
        return P4A_OK;
    }
    if (entry.params.size() != 1U && entry.params.size() != 4U) {
        ctx.set_error("AOM FCK expects zero params, alpha, or alpha/scale/kernel_size/sigma");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    params.alpha = entry.params[0];
    if (entry.params.size() == 4U) {
        params.scale = entry.params[1];
        p4a_status_t status = parse_integer_param(ctx,
                                                  entry.params[2],
                                                  3,
                                                  501,
                                                  "AOM FCK kernel size",
                                                  params.kernel_size);
        if (status != P4A_OK) {
            return status;
        }
        params.sigma = entry.params[3];
    }
    if (!std::isfinite(params.alpha) || params.alpha < 0.0) {
        ctx.set_error("AOM FCK alpha must be finite and non-negative");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (!std::isfinite(params.scale) || params.scale <= 0.0) {
        ctx.set_error("AOM FCK scale must be finite and positive");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if ((params.kernel_size % 2) == 0) {
        ctx.set_error("AOM FCK kernel size must be odd");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (!std::isfinite(params.sigma) || params.sigma <= 0.0) {
        ctx.set_error("AOM FCK sigma must be finite and positive");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    return P4A_OK;
}

[[nodiscard]] p4a_status_t apply_fck(::pls4all::core::Context& ctx,
                                     const ::pls4all::core::OperatorEntry& entry,
                                     const std::vector<double>& values,
                                     std::size_t rows,
                                     std::size_t cols,
                                     std::vector<double>& out) {
    FckParams params;
    p4a_status_t status = parse_fck(ctx, entry, params);
    if (status != P4A_OK) {
        return status;
    }

    const std::int32_t half = params.kernel_size / 2;
    std::vector<double> kernel(static_cast<std::size_t>(params.kernel_size), 0.0);
    for (std::int32_t i = 0; i < params.kernel_size; ++i) {
        const double x = static_cast<double>(i - half) * params.scale;
        const double gauss = std::exp(-0.5 * (x / params.sigma) * (x / params.sigma));
        if (params.alpha < 0.1) {
            kernel[static_cast<std::size_t>(i)] = gauss;
        } else {
            constexpr double kEps = 1e-8;
            const double sign = (x > 0.0) ? 1.0 : ((x < 0.0) ? -1.0 : 0.0);
            const double frac = sign * std::pow(std::fabs(x) + kEps, params.alpha);
            kernel[static_cast<std::size_t>(i)] = gauss * frac;
        }
    }
    if (params.alpha >= 0.1) {
        double mean = 0.0;
        for (double value : kernel) {
            mean += value;
        }
        mean /= static_cast<double>(kernel.size());
        for (double& value : kernel) {
            value -= mean;
        }
    }
    double norm = 1e-8;
    for (double value : kernel) {
        norm += std::fabs(value);
    }
    for (double& value : kernel) {
        value /= norm;
    }
    return apply_xcorr_zero_pad(values, rows, cols, kernel, out);
}

}  // namespace

namespace pls4all::core {

p4a_status_t transform_aom_strict_operator(Context& ctx,
                                           const OperatorEntry& entry,
                                           const p4a_matrix_view_t& X,
                                           std::vector<double>& out) {
    p4a_status_t status = validate_float_view(ctx, X, "X");
    if (status != P4A_OK) {
        out.clear();
        return status;
    }

    std::vector<double> values;
    status = copy_matrix_checked(ctx, X, values);
    if (status != P4A_OK) {
        out.clear();
        return status;
    }

    const auto rows = static_cast<std::size_t>(X.rows);
    const auto cols = static_cast<std::size_t>(X.cols);
    switch (entry.kind) {
        case P4A_OP_IDENTITY:
            status = require_no_params(ctx, entry, "AOM identity");
            if (status == P4A_OK) {
                out = std::move(values);
            }
            break;
        case P4A_OP_DETREND_POLY: {
            std::int32_t degree = 1;
            status = parse_degree(ctx, entry, degree);
            if (status == P4A_OK) {
                status = apply_detrend_poly(ctx, values, rows, cols, degree, out);
            }
            break;
        }
        case P4A_OP_SAVGOL_SMOOTH: {
            SavGolParams params;
            status = parse_savgol_smooth(ctx, entry, params);
            if (status == P4A_OK) {
                status = apply_savgol(ctx, values, rows, cols, params, out);
            }
            break;
        }
        case P4A_OP_SAVGOL_DERIVATIVE: {
            SavGolParams params;
            status = parse_savgol_derivative(ctx, entry, params);
            if (status == P4A_OK) {
                status = apply_savgol(ctx, values, rows, cols, params, out);
            }
            break;
        }
        case P4A_OP_NORRIS_WILLIAMS:
            status = apply_norris_williams(ctx, entry, values, rows, cols, out);
            break;
        case P4A_OP_FINITE_DIFFERENCE:
            status = apply_finite_difference(ctx, entry, values, rows, cols, out);
            break;
        case P4A_OP_WHITTAKER:
            status = apply_whittaker(ctx, entry, values, rows, cols, out);
            break;
        case P4A_OP_FCK:
            status = apply_fck(ctx, entry, values, rows, cols, out);
            break;
        default:
            ctx.set_errorf("operator %d is not a supported strict-linear AOM operator",
                           static_cast<int>(entry.kind));
            status = P4A_ERR_UNSUPPORTED;
            break;
    }

    if (status != P4A_OK) {
        out.clear();
        return status;
    }
    ctx.clear_error();
    return P4A_OK;
}

}  // namespace pls4all::core
