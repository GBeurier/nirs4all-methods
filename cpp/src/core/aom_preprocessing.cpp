// SPDX-License-Identifier: CECILL-2.1

#include "core/aom_preprocessing.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <vector>

#include "core/matrix_view.hpp"
#include "core/operator_entry.hpp"
#include "core/pipeline.hpp"
#include "core/status.hpp"

namespace {

[[nodiscard]] std::size_t idx(std::size_t row, std::size_t cols, std::size_t col) noexcept {
    return row * cols + col;
}

[[nodiscard]] bool checked_matrix_size(std::int64_t rows,
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

[[nodiscard]] p4a_matrix_view_t rowmajor_f64_view(std::vector<double>& values,
                                                  std::int64_t rows,
                                                  std::int64_t cols) noexcept {
    p4a_matrix_view_t view{};
    view.data = values.data();
    view.rows = rows;
    view.cols = cols;
    view.row_stride = cols > 0 ? cols : 1;
    view.col_stride = 1;
    view.dtype = P4A_DTYPE_F64;
    return view;
}

[[nodiscard]] p4a_status_t validate_request(::pls4all::core::Context& ctx,
                                            const ::pls4all::core::OperatorBank& bank,
                                            const ::pls4all::core::GatingStrategy& gate,
                                            const p4a_matrix_view_t& X,
                                            const p4a_matrix_view_t* Y) {
    const p4a_status_t status = ::pls4all::core::validate_nonnull_view(X);
    if (status != P4A_OK) {
        ctx.set_errorf("X matrix view is invalid: %s",
                       ::pls4all::core::status_to_string(status));
        return status;
    }
    if (X.dtype != P4A_DTYPE_F64 && X.dtype != P4A_DTYPE_F32) {
        ctx.set_error("X dtype must be f64 or f32");
        return P4A_ERR_DTYPE_MISMATCH;
    }
    if (X.rows <= 0 || X.cols <= 0) {
        ctx.set_error("AOM preprocessing requires a non-empty X matrix");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (bank.entries().empty()) {
        ctx.set_error("AOM preprocessing requires at least one operator");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (gate.mode() != P4A_GATING_SOFT && gate.mode() != P4A_GATING_HARD) {
        ctx.set_error("AOM preprocessing currently supports soft and hard gating only");
        return P4A_ERR_UNSUPPORTED;
    }
    if (Y != nullptr) {
        const p4a_status_t y_status = ::pls4all::core::validate_nonnull_view(*Y);
        if (y_status != P4A_OK) {
            ctx.set_errorf("Y matrix view is invalid: %s",
                           ::pls4all::core::status_to_string(y_status));
            return y_status;
        }
        if (Y->dtype != P4A_DTYPE_F64 && Y->dtype != P4A_DTYPE_F32) {
            ctx.set_error("Y dtype must be f64 or f32");
            return P4A_ERR_DTYPE_MISMATCH;
        }
        if (Y->rows != X.rows) {
            ctx.set_errorf("Y rows (%lld) must match X rows (%lld)",
                           static_cast<long long>(Y->rows),
                           static_cast<long long>(X.rows));
            return P4A_ERR_SHAPE_MISMATCH;
        }
    }
    return P4A_OK;
}

[[nodiscard]] std::vector<double> gating_weights(p4a_gating_mode_t mode,
                                                 std::size_t n_operators) {
    if (n_operators == 0U) {
        return {};
    }
    if (mode == P4A_GATING_HARD) {
        std::vector<double> weights;
        weights.reserve(n_operators);
        weights.push_back(1.0);
        weights.resize(n_operators, 0.0);
        return weights;
    }
    const double weight = 1.0 / static_cast<double>(n_operators);
    std::vector<double> weights(n_operators, 0.0);
    std::fill(weights.begin(), weights.end(), weight);
    return weights;
}

}  // namespace

namespace pls4all::core {

p4a_status_t apply_aom_preprocessing(Context& ctx,
                                     const OperatorBank& bank,
                                     const GatingStrategy& gate,
                                     const p4a_matrix_view_t& X,
                                     const p4a_matrix_view_t* Y,
                                     AomPreprocessingResult& out) {
    try {
        out = AomPreprocessingResult{};
        p4a_status_t status = validate_request(ctx, bank, gate, X, Y);
        if (status != P4A_OK) {
            return status;
        }

        const auto rows = static_cast<std::size_t>(X.rows);
        const auto cols = static_cast<std::size_t>(X.cols);
        const std::size_t n_operators = bank.entries().size();
        if (n_operators > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
            ctx.set_error("AOM preprocessing operator count is too large");
            return P4A_ERR_INVALID_ARGUMENT;
        }
        std::size_t block_values = 0;
        if (!checked_matrix_size(X.rows, X.cols, block_values) ||
            n_operators > std::numeric_limits<std::size_t>::max() / block_values) {
            ctx.set_error("AOM preprocessing output shape is too large");
            return P4A_ERR_INVALID_ARGUMENT;
        }

        out.n_samples = X.rows;
        out.n_features = X.cols;
        out.n_operators = static_cast<std::int32_t>(n_operators);
        out.mode = gate.mode();
        out.weights = gating_weights(gate.mode(), n_operators);
        out.operator_kinds.reserve(n_operators);
        out.operator_outputs.assign(n_operators * block_values, 0.0);
        out.transformed.assign(block_values, 0.0);

        for (std::size_t op_index = 0; op_index < n_operators; ++op_index) {
            const OperatorEntry& entry = bank.entries()[op_index];
            Pipeline pipeline;
            const double* params = entry.params.empty() ? nullptr : entry.params.data();
            pipeline.add_operator(entry.kind,
                                  params,
                                  static_cast<std::int32_t>(entry.params.size()));
            status = pipeline.fit(ctx, X, Y);
            if (status != P4A_OK) {
                out = AomPreprocessingResult{};
                return status;
            }

            std::vector<double> operator_values(block_values, 0.0);
            p4a_matrix_view_t operator_view = rowmajor_f64_view(operator_values, X.rows, X.cols);
            status = pipeline.transform(ctx, X, operator_view);
            if (status != P4A_OK) {
                out = AomPreprocessingResult{};
                return status;
            }

            out.operator_kinds.push_back(static_cast<std::int64_t>(entry.kind));
            const double weight = out.weights[op_index];
            for (std::size_t row = 0; row < rows; ++row) {
                for (std::size_t col = 0; col < cols; ++col) {
                    const std::size_t value_index = idx(row, cols, col);
                    out.operator_outputs[op_index * block_values + value_index] =
                        operator_values[value_index];
                    out.transformed[value_index] += weight * operator_values[value_index];
                }
            }
        }

        ctx.clear_error();
        return P4A_OK;
    } catch (const std::bad_alloc&) {
        ctx.set_error("out of memory while applying AOM preprocessing");
        out = AomPreprocessingResult{};
        return P4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        ctx.set_error("unexpected exception while applying AOM preprocessing");
        out = AomPreprocessingResult{};
        return P4A_ERR_INTERNAL;
    }
}

}  // namespace pls4all::core
