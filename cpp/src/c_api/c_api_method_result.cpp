// SPDX-License-Identifier: CECILL-2.1
//
// extern "C" wrappers for n4m_method_result_t plus the universal fit
// functions that build one. Each fit function instantiates the internal
// result struct from core/extra_pls.hpp or core/model.hpp, then packs the
// relevant outputs into a n4m_method_result_t for transit across the C
// ABI boundary.

#include <stddef.h>
#include <stdint.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <vector>

#include "pls4all/p4a.h"

#include "core/aom_preprocessing.hpp"
#include "core/bipls_selection.hpp"
#include "core/bve_selection.hpp"
#include "core/cars_selection.hpp"
#include "core/ecr.hpp"
#include "core/config.hpp"
#include "core/context.hpp"
#include "core/emcuve_selection.hpp"
#include "core/extra_pls.hpp"
#include "core/ga_selection.hpp"
#include "core/gpr_pls.hpp"
#include "core/gating_strategy.hpp"
#include "core/interval_selection.hpp"
#include "core/ipw_selection.hpp"
#include "core/irf_selection.hpp"
#include "core/iriv_selection.hpp"
#include "core/kernel_pls.hpp"
#include "core/lw_pls.hpp"
#include "core/mb_pls.hpp"
#include "core/method_result.hpp"
#include "core/model.hpp"
#include "core/model_selection.hpp"
#include "core/multiblock_extensions.hpp"
#include "core/operator_bank.hpp"
#include "core/pls_diagnostics.hpp"
#include "core/pls_lda.hpp"
#include "core/pls_logistic.hpp"
#include "core/pls_monitoring.hpp"
#include "core/pso_selection.hpp"
#include "core/random_frog_selection.hpp"
#include "core/randomization_selection.hpp"
#include "core/recursive_pls.hpp"
#include "core/rep_selection.hpp"
#include "core/scars_selection.hpp"
#include "core/shaving_selection.hpp"
#include "core/sipls_selection.hpp"
#include "core/spa_selection.hpp"
#include "core/st_selection.hpp"
#include "core/stability_selection.hpp"
#include "core/t2_selection.hpp"
#include "core/tensor_pls.hpp"
#include "core/uve_selection.hpp"
#include "core/validation.hpp"
#include "core/vip_spa_selection.hpp"
#include "core/vissa_selection.hpp"
#include "core/variable_selection.hpp"
#include "core/wvc_selection.hpp"

namespace {

inline ::n4m::core::Context* as_core(n4m_context_t* ctx) noexcept {
    return static_cast<::n4m::core::Context*>(ctx);
}
inline const ::n4m::core::Config* as_core(const n4m_config_t* cfg) noexcept {
    return static_cast<const ::n4m::core::Config*>(cfg);
}
inline const ::n4m::core::Model* as_core(const n4m_model_t* model) noexcept {
    return static_cast<const ::n4m::core::Model*>(model);
}
inline const ::n4m::core::OperatorBank* as_core(
    const n4m_operator_bank_t* bank) noexcept {
    return static_cast<const ::n4m::core::OperatorBank*>(bank);
}
inline const ::n4m::core::GatingStrategy* as_core(
    const n4m_gating_strategy_t* gate) noexcept {
    return static_cast<const ::n4m::core::GatingStrategy*>(gate);
}
inline const ::n4m::core::ValidationPlan* as_core(
    const n4m_validation_plan_t* plan) noexcept {
    return static_cast<const ::n4m::core::ValidationPlan*>(plan);
}

// Convert a vector of int64 indices to a row-major double matrix (1 x n) for
// transit via the universal handle's int64_arrays bag. The handle owns the
// vector after std::move.
inline std::vector<double> i64_to_double_vector(
    const std::vector<std::int64_t>& src) {
    std::vector<double> out;
    out.reserve(src.size());
    for (const auto v : src) {
        out.push_back(static_cast<double>(v));
    }
    return out;
}

void set_error(n4m_context_t* ctx, const char* msg) noexcept {
    if (ctx == nullptr) return;
    try {
        as_core(ctx)->set_error(msg);
    } catch (...) {
    }
}

void copy_predictions(const ::n4m::core::Model& model,
                      const n4m_matrix_view_t& X,
                      ::n4m::core::Context& ctx,
                      std::vector<double>& out,
                      std::int64_t& out_rows,
                      std::int64_t& out_cols) {
    const std::size_t n = static_cast<std::size_t>(X.rows);
    const std::size_t q = static_cast<std::size_t>(model.n_targets);
    out.assign(n * q, 0.0);
    n4m_matrix_view_t pred_view{};
    pred_view.data = out.data();
    pred_view.rows = static_cast<std::int64_t>(n);
    pred_view.cols = static_cast<std::int64_t>(q);
    pred_view.row_stride = static_cast<std::int64_t>(q > 0 ? q : 1);
    pred_view.col_stride = 1;
    pred_view.dtype = N4M_DTYPE_F64;
    const n4m_status_t status =
        ::n4m::core::predict_into(ctx, model, X, pred_view);
    (void)status;  // predictions are best-effort; caller checks rmse / status
    out_rows = static_cast<std::int64_t>(n);
    out_cols = static_cast<std::int64_t>(q);
}

double in_sample_rmse(const std::vector<double>& predictions,
                       const n4m_matrix_view_t& Y) {
    const std::size_t n = static_cast<std::size_t>(Y.rows);
    const std::size_t q = static_cast<std::size_t>(Y.cols);
    if (n == 0 || q == 0) return 0.0;
    const auto* y_data = static_cast<const double*>(Y.data);
    const std::size_t y_rs = static_cast<std::size_t>(Y.row_stride);
    const std::size_t y_cs = static_cast<std::size_t>(Y.col_stride);
    double sumsq = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t t = 0; t < q; ++t) {
            const double truth = y_data[i * y_rs + t * y_cs];
            const double pred = predictions[i * q + t];
            const double d = truth - pred;
            sumsq += d * d;
        }
    }
    return std::sqrt(sumsq / static_cast<double>(n * q));
}

// Predict Yhat = (X - x_mean) @ coefficients + y_mean for any §26-style
// WeightedPlsResult / CpplsResult that stores its model in those three
// fields. coefficients is row-major (p x q).
void predict_from_coefficients(const n4m_matrix_view_t& X,
                                const std::vector<double>& coefficients,
                                const std::vector<double>& x_mean,
                                const std::vector<double>& y_mean,
                                std::int64_t n_features,
                                std::int64_t n_targets,
                                std::vector<double>& out,
                                std::int64_t& out_rows,
                                std::int64_t& out_cols) {
    const std::size_t n = static_cast<std::size_t>(X.rows);
    const std::size_t p = static_cast<std::size_t>(n_features);
    const std::size_t q = static_cast<std::size_t>(n_targets);
    const auto* x_data = static_cast<const double*>(X.data);
    const std::size_t x_rs = static_cast<std::size_t>(X.row_stride);
    const std::size_t x_cs = static_cast<std::size_t>(X.col_stride);
    out.assign(n * q, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t t = 0; t < q; ++t) {
            double acc = (t < y_mean.size()) ? y_mean[t] : 0.0;
            for (std::size_t f = 0; f < p; ++f) {
                const double xv = x_data[i * x_rs + f * x_cs];
                const double xm = (f < x_mean.size()) ? x_mean[f] : 0.0;
                acc += (xv - xm) * coefficients[f * q + t];
            }
            out[i * q + t] = acc;
        }
    }
    out_rows = static_cast<std::int64_t>(n);
    out_cols = static_cast<std::int64_t>(q);
}

}  // namespace

extern "C" {

/* ---- universal result accessors ---- */

N4M_API void n4m_method_result_destroy(n4m_method_result_t* result) {
    try {
        delete result;
    } catch (...) {
    }
}

N4M_API n4m_status_t n4m_method_result_get_double_matrix(
    const n4m_method_result_t* result,
    const char* name,
    const double** out_data, int64_t* out_rows, int64_t* out_cols) {
    if (result == nullptr || name == nullptr || out_data == nullptr ||
        out_rows == nullptr || out_cols == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    try {
        const auto array_iter = result->double_arrays.find(name);
        if (array_iter == result->double_arrays.end()) {
            return N4M_ERR_INVALID_ARGUMENT;
        }
        const auto shape_iter = result->double_shapes.find(name);
        const auto& values = array_iter->second;
        *out_data = values.empty() ? nullptr : values.data();
        if (shape_iter != result->double_shapes.end()) {
            *out_rows = shape_iter->second.first;
            *out_cols = shape_iter->second.second;
        } else {
            *out_rows = static_cast<int64_t>(values.size());
            *out_cols = 1;
        }
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_method_result_get_int_vector(
    const n4m_method_result_t* result,
    const char* name,
    const int32_t** out_data, int32_t* out_size) {
    if (result == nullptr || name == nullptr || out_data == nullptr ||
        out_size == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    try {
        const auto iter = result->int_arrays.find(name);
        if (iter == result->int_arrays.end()) {
            return N4M_ERR_INVALID_ARGUMENT;
        }
        const auto& values = iter->second;
        *out_data = values.empty() ? nullptr : values.data();
        *out_size = static_cast<int32_t>(values.size());
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_method_result_get_int64_vector(
    const n4m_method_result_t* result,
    const char* name,
    const int64_t** out_data, int64_t* out_size) {
    if (result == nullptr || name == nullptr || out_data == nullptr ||
        out_size == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    try {
        const auto iter = result->int64_arrays.find(name);
        if (iter == result->int64_arrays.end()) {
            return N4M_ERR_INVALID_ARGUMENT;
        }
        const auto& values = iter->second;
        *out_data = values.empty() ? nullptr : values.data();
        *out_size = static_cast<int64_t>(values.size());
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_method_result_get_scalar(
    const n4m_method_result_t* result,
    const char* name,
    double* out_value) {
    if (result == nullptr || name == nullptr || out_value == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    try {
        const auto iter = result->scalars.find(name);
        if (iter == result->scalars.end()) {
            return N4M_ERR_INVALID_ARGUMENT;
        }
        *out_value = iter->second;
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

/* ---- fit entry points ---- */

N4M_API n4m_status_t n4m_sparse_simpls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    double sparsity_lambda,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X == nullptr || Y == nullptr) {
        set_error(ctx, "null pointer in n4m_sparse_simpls_fit");
        return N4M_ERR_NULL_POINTER;
    }

    try {
        ::n4m::core::Config local = *as_core(cfg);
        local.algorithm = N4M_ALGO_SPARSE_PLS;
        local.solver = N4M_SOLVER_SIMPLS;
        local.deflation = N4M_DEFLATION_REGRESSION;
        local.sparsity_lambda = sparsity_lambda;
        std::unique_ptr<::n4m::core::Model> model;
        const n4m_status_t status = ::n4m::core::fit_pls_sparse_simpls(
            *as_core(ctx), local, *X, *Y, model);
        if (status != N4M_OK) return status;

        auto handle = std::make_unique<n4m_method_result_s>();
        const auto p = static_cast<std::int64_t>(model->n_features);
        const auto q = static_cast<std::int64_t>(model->n_targets);
        const auto k = static_cast<std::int64_t>(model->n_components);

        handle->set_double_matrix("coefficients", model->coefficients, p, q);
        handle->set_double_matrix("x_mean", model->x_mean, 1, p);
        handle->set_double_matrix("y_mean", model->y_mean, 1, q);
        if (!model->weights_w.empty()) {
            handle->set_double_matrix("weights_w", model->weights_w, p, k);
        }

        std::vector<double> predictions;
        std::int64_t pred_rows = 0;
        std::int64_t pred_cols = 0;
        copy_predictions(*model, *X, *as_core(ctx), predictions, pred_rows,
                         pred_cols);
        const double rmse = in_sample_rmse(predictions, *Y);
        handle->set_double_matrix("predictions", std::move(predictions),
                                   pred_rows, pred_cols);
        handle->set_scalar("rmse", rmse);

        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_sparse_simpls_fit");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_sparse_simpls_fit");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_di_pls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X_source,
    const n4m_matrix_view_t* Y_source,
    const n4m_matrix_view_t* X_target,
    double di_lambda,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X_source == nullptr ||
        Y_source == nullptr || X_target == nullptr) {
        set_error(ctx, "null pointer in n4m_di_pls_fit");
        return N4M_ERR_NULL_POINTER;
    }

    try {
        ::n4m::core::Config local = *as_core(cfg);
        local.algorithm = N4M_ALGO_PLS_REGRESSION;
        local.solver = N4M_SOLVER_SIMPLS;
        local.deflation = N4M_DEFLATION_REGRESSION;
        local.di_lambda = di_lambda;
        std::unique_ptr<::n4m::core::Model> model;
        const n4m_status_t status = ::n4m::core::fit_di_pls(
            *as_core(ctx), local, *X_source, *Y_source, *X_target, model);
        if (status != N4M_OK) return status;

        auto handle = std::make_unique<n4m_method_result_s>();
        const auto p = static_cast<std::int64_t>(model->n_features);
        const auto q = static_cast<std::int64_t>(model->n_targets);
        const auto k = static_cast<std::int64_t>(model->n_components);

        handle->set_double_matrix("coefficients", model->coefficients, p, q);
        handle->set_double_matrix("x_mean", model->x_mean, 1, p);
        handle->set_double_matrix("y_mean", model->y_mean, 1, q);
        if (!model->weights_w.empty()) {
            handle->set_double_matrix("weights_w", model->weights_w, p, k);
        }

        std::vector<double> predictions;
        std::int64_t pred_rows = 0;
        std::int64_t pred_cols = 0;
        copy_predictions(*model, *X_source, *as_core(ctx), predictions,
                         pred_rows, pred_cols);
        const double rmse = in_sample_rmse(predictions, *Y_source);
        handle->set_double_matrix("predictions", std::move(predictions),
                                   pred_rows, pred_cols);
        handle->set_scalar("rmse_source", rmse);

        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_di_pls_fit");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_di_pls_fit");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_recursive_pls_run(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    int32_t window_size,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X == nullptr || Y == nullptr) {
        set_error(ctx, "null pointer in n4m_recursive_pls_run");
        return N4M_ERR_NULL_POINTER;
    }

    try {
        ::n4m::core::RecursivePlsResult result;
        const n4m_status_t status = ::n4m::core::fit_predict_recursive_pls(
            *as_core(ctx), *as_core(cfg), *X, *Y, window_size, result);
        if (status != N4M_OK) return status;

        auto handle = std::make_unique<n4m_method_result_s>();
        const auto n = static_cast<std::int64_t>(result.n_samples);
        const auto q = static_cast<std::int64_t>(result.n_targets);
        handle->set_double_matrix("predictions",
                                   std::move(result.predictions), n, q);
        handle->set_int_vector("in_window", std::move(result.in_window));

        // Compute RMSE on the predictable subset.
        const auto* y_data = static_cast<const double*>(Y->data);
        const std::size_t y_rs = static_cast<std::size_t>(Y->row_stride);
        const std::size_t y_cs = static_cast<std::size_t>(Y->col_stride);
        const auto& predictions =
            handle->double_arrays.at("predictions");
        const auto& flags = handle->int_arrays.at("in_window");
        double sumsq = 0.0;
        std::size_t count = 0;
        for (std::size_t i = 0; i < flags.size(); ++i) {
            if (flags[i] == 0) continue;
            for (std::int64_t t = 0; t < q; ++t) {
                const double truth = y_data[
                    i * y_rs + static_cast<std::size_t>(t) * y_cs];
                const double pred = predictions[
                    i * static_cast<std::size_t>(q) + static_cast<std::size_t>(t)];
                const double d = truth - pred;
                sumsq += d * d;
                ++count;
            }
        }
        const double rmse = count > 0
            ? std::sqrt(sumsq / static_cast<double>(count)) : 0.0;
        handle->set_scalar("rmse_predictable", rmse);

        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_recursive_pls_run");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_recursive_pls_run");
        return N4M_ERR_INTERNAL;
    }
}

namespace {

void pack_weighted_result(n4m_method_result_s& handle,
                           const ::n4m::core::WeightedPlsResult& res,
                           const n4m_matrix_view_t& X,
                           const n4m_matrix_view_t& Y) {
    const auto p = static_cast<std::int64_t>(res.n_features);
    const auto q = static_cast<std::int64_t>(res.n_targets);
    handle.set_double_matrix("coefficients", res.coefficients, p, q);
    handle.set_double_matrix("x_mean", res.x_mean, 1, p);
    handle.set_double_matrix("y_mean", res.y_mean, 1, q);

    std::vector<double> predictions;
    std::int64_t pred_rows = 0;
    std::int64_t pred_cols = 0;
    predict_from_coefficients(X, res.coefficients, res.x_mean, res.y_mean,
                                p, q, predictions, pred_rows, pred_cols);
    const double rmse = in_sample_rmse(predictions, Y);
    handle.set_double_matrix("predictions", std::move(predictions),
                              pred_rows, pred_cols);
    handle.set_scalar("rmse", rmse);
}

}  // namespace

N4M_API n4m_status_t n4m_cppls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    double gamma,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X == nullptr || Y == nullptr) {
        set_error(ctx, "null pointer in n4m_cppls_fit");
        return N4M_ERR_NULL_POINTER;
    }
    try {
        ::n4m::core::CpplsResult res;
        const n4m_status_t status = ::n4m::core::fit_cppls(
            *as_core(ctx), *as_core(cfg), *X, *Y, gamma, res);
        if (status != N4M_OK) return status;

        auto handle = std::make_unique<n4m_method_result_s>();
        const auto p = static_cast<std::int64_t>(res.n_features);
        const auto q = static_cast<std::int64_t>(res.n_targets);
        handle->set_double_matrix("coefficients", res.coefficients, p, q);
        handle->set_double_matrix("x_mean", res.x_mean, 1, p);
        handle->set_double_matrix("y_mean", res.y_mean, 1, q);

        std::vector<double> predictions;
        std::int64_t pred_rows = 0;
        std::int64_t pred_cols = 0;
        predict_from_coefficients(*X, res.coefficients, res.x_mean,
                                    res.y_mean, p, q, predictions,
                                    pred_rows, pred_cols);
        const double rmse = in_sample_rmse(predictions, *Y);
        handle->set_double_matrix("predictions", std::move(predictions),
                                   pred_rows, pred_cols);
        handle->set_scalar("rmse", rmse);
        handle->set_scalar("gamma", res.gamma);

        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_cppls_fit");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_cppls_fit");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_weighted_pls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const double* sample_weights,
    int64_t sample_weights_size,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X == nullptr || Y == nullptr ||
        sample_weights == nullptr) {
        set_error(ctx, "null pointer in n4m_weighted_pls_fit");
        return N4M_ERR_NULL_POINTER;
    }
    if (sample_weights_size != X->rows) {
        set_error(ctx, "sample_weights_size must equal X.rows");
        return N4M_ERR_SHAPE_MISMATCH;
    }
    try {
        std::vector<double> weights(
            sample_weights,
            sample_weights + static_cast<std::size_t>(sample_weights_size));
        ::n4m::core::WeightedPlsResult res;
        const n4m_status_t status = ::n4m::core::fit_weighted_pls(
            *as_core(ctx), *as_core(cfg), *X, *Y, weights, res);
        if (status != N4M_OK) return status;

        auto handle = std::make_unique<n4m_method_result_s>();
        pack_weighted_result(*handle, res, *X, *Y);
        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_weighted_pls_fit");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_weighted_pls_fit");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_robust_pls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    double huber_k,
    int32_t max_irls_iter,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X == nullptr || Y == nullptr) {
        set_error(ctx, "null pointer in n4m_robust_pls_fit");
        return N4M_ERR_NULL_POINTER;
    }
    try {
        ::n4m::core::WeightedPlsResult res;
        const n4m_status_t status = ::n4m::core::fit_robust_pls(
            *as_core(ctx), *as_core(cfg), *X, *Y, huber_k, max_irls_iter,
            res);
        if (status != N4M_OK) return status;

        auto handle = std::make_unique<n4m_method_result_s>();
        pack_weighted_result(*handle, res, *X, *Y);
        handle->set_scalar("huber_k", huber_k);

        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_robust_pls_fit");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_robust_pls_fit");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_ridge_pls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    double ridge_lambda,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X == nullptr || Y == nullptr) {
        set_error(ctx, "null pointer in n4m_ridge_pls_fit");
        return N4M_ERR_NULL_POINTER;
    }
    try {
        ::n4m::core::WeightedPlsResult res;
        const n4m_status_t status = ::n4m::core::fit_ridge_pls(
            *as_core(ctx), *as_core(cfg), *X, *Y, ridge_lambda, res);
        if (status != N4M_OK) return status;

        auto handle = std::make_unique<n4m_method_result_s>();
        pack_weighted_result(*handle, res, *X, *Y);
        handle->set_scalar("ridge_lambda", ridge_lambda);

        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_ridge_pls_fit");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_ridge_pls_fit");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_continuum_regression_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    double tau,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X == nullptr || Y == nullptr) {
        set_error(ctx, "null pointer in n4m_continuum_regression_fit");
        return N4M_ERR_NULL_POINTER;
    }
    try {
        ::n4m::core::WeightedPlsResult res;
        const n4m_status_t status = ::n4m::core::fit_continuum_regression(
            *as_core(ctx), *as_core(cfg), *X, *Y, tau, res);
        if (status != N4M_OK) return status;

        auto handle = std::make_unique<n4m_method_result_s>();
        pack_weighted_result(*handle, res, *X, *Y);
        handle->set_scalar("tau", tau);

        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_continuum_regression_fit");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_continuum_regression_fit");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_n_pls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X_flat,
    int32_t mode_j,
    int32_t mode_k,
    const n4m_matrix_view_t* Y,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X_flat == nullptr ||
        Y == nullptr) {
        set_error(ctx, "null pointer in n4m_n_pls_fit");
        return N4M_ERR_NULL_POINTER;
    }
    try {
        ::n4m::core::NPlsResult res;
        const n4m_status_t status = ::n4m::core::fit_n_pls(
            *as_core(ctx), *as_core(cfg), *X_flat, mode_j, mode_k, *Y, res);
        if (status != N4M_OK) return status;

        auto handle = std::make_unique<n4m_method_result_s>();
        const auto J = static_cast<std::int64_t>(res.mode_j);
        const auto K = static_cast<std::int64_t>(res.mode_k);
        const auto jk = J * K;
        const auto q = static_cast<std::int64_t>(res.n_targets);
        const auto k = static_cast<std::int64_t>(res.n_components);
        const auto n = static_cast<std::int64_t>(res.n_samples);
        handle->set_double_matrix("coefficients", res.coefficients, jk, q);
        handle->set_double_matrix("w_j", res.w_j_per_component, J, k);
        handle->set_double_matrix("w_k", res.w_k_per_component, K, k);
        handle->set_double_matrix("scores_t", res.scores_t, n, k);
        handle->set_double_matrix("x_mean", res.x_mean, 1, jk);
        handle->set_double_matrix("y_mean", res.y_mean, 1, q);

        std::vector<double> predictions;
        const n4m_status_t pred_status = ::n4m::core::predict_n_pls(
            *as_core(ctx), res, *X_flat, predictions);
        (void)pred_status;
        const double rmse = in_sample_rmse(predictions, *Y);
        handle->set_double_matrix("predictions", std::move(predictions),
                                   n, q);
        handle->set_scalar("rmse", rmse);

        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_n_pls_fit");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_n_pls_fit");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_o2pls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    int32_t n_predictive,
    int32_t n_x_orthogonal,
    int32_t n_y_orthogonal,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X == nullptr || Y == nullptr) {
        set_error(ctx, "null pointer in n4m_o2pls_fit");
        return N4M_ERR_NULL_POINTER;
    }
    try {
        ::n4m::core::O2PlsResult res;
        const n4m_status_t status = ::n4m::core::fit_o2pls(
            *as_core(ctx), *as_core(cfg), *X, *Y,
            n_predictive, n_x_orthogonal, n_y_orthogonal, res);
        if (status != N4M_OK) return status;

        auto handle = std::make_unique<n4m_method_result_s>();
        const auto p = static_cast<std::int64_t>(res.n_features_x);
        const auto q = static_cast<std::int64_t>(res.n_features_y);
        const auto kp = static_cast<std::int64_t>(res.n_predictive);
        const auto kx = static_cast<std::int64_t>(res.n_x_orthogonal);
        const auto ky = static_cast<std::int64_t>(res.n_y_orthogonal);
        handle->set_double_matrix("coefficients", res.coefficients, p, q);
        handle->set_double_matrix("x_mean", res.x_mean, 1, p);
        handle->set_double_matrix("y_mean", res.y_mean, 1, q);
        handle->set_double_matrix("w_predictive", res.w_predictive, p, kp);
        handle->set_double_matrix("c_predictive", res.c_predictive, q, kp);
        handle->set_double_matrix("w_x_orthogonal", res.w_x_orthogonal,
                                   p, kx);
        handle->set_double_matrix("c_y_orthogonal", res.c_y_orthogonal,
                                   q, ky);
        handle->set_double_matrix("b_predictive", res.b_predictive, 1, kp);

        std::vector<double> predictions;
        std::int64_t pred_rows = 0;
        std::int64_t pred_cols = 0;
        predict_from_coefficients(*X, res.coefficients, res.x_mean,
                                    res.y_mean, p, q, predictions,
                                    pred_rows, pred_cols);
        const double rmse = in_sample_rmse(predictions, *Y);
        handle->set_double_matrix("predictions", std::move(predictions),
                                   pred_rows, pred_cols);
        handle->set_scalar("rmse", rmse);

        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_o2pls_fit");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_o2pls_fit");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_sparse_pls_da_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const int32_t* y_labels,
    int64_t y_labels_size,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X == nullptr ||
        y_labels == nullptr) {
        set_error(ctx, "null pointer in n4m_sparse_pls_da_fit");
        return N4M_ERR_NULL_POINTER;
    }
    if (y_labels_size != X->rows) {
        set_error(ctx, "y_labels_size must equal X.rows");
        return N4M_ERR_SHAPE_MISMATCH;
    }
    try {
        std::vector<std::int32_t> labels(
            y_labels, y_labels + static_cast<std::size_t>(y_labels_size));
        ::n4m::core::SparsePlsDaResult res;
        const n4m_status_t status = ::n4m::core::fit_sparse_pls_da(
            *as_core(ctx), *as_core(cfg), *X, labels, res);
        if (status != N4M_OK) return status;

        auto handle = std::make_unique<n4m_method_result_s>();
        const auto p = static_cast<std::int64_t>(X->cols);
        const auto q = static_cast<std::int64_t>(res.n_classes);
        handle->set_double_matrix("coefficients", res.coefficients, p, q);
        handle->set_double_matrix("x_mean", res.x_mean, 1, p);
        handle->set_double_matrix("y_mean", res.y_mean, 1, q);
        handle->set_double_matrix("class_priors", res.class_priors, 1, q);

        // Soft-assignment predictions: rebuild dummy-encoded Y for RMSE.
        std::vector<double> predictions;
        std::int64_t pred_rows = 0;
        std::int64_t pred_cols = 0;
        predict_from_coefficients(*X, res.coefficients, res.x_mean,
                                    res.y_mean, p, q, predictions,
                                    pred_rows, pred_cols);
        const auto n = static_cast<std::size_t>(X->rows);
        std::vector<double> y_dummy(n * static_cast<std::size_t>(q), 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t lbl =
                static_cast<std::size_t>(labels[i]);
            y_dummy[i * static_cast<std::size_t>(q) + lbl] = 1.0;
        }
        // Compute RMSE manually since `Y` is the dummy matrix.
        double sumsq = 0.0;
        for (std::size_t i = 0; i < n * static_cast<std::size_t>(q); ++i) {
            const double d = y_dummy[i] - predictions[i];
            sumsq += d * d;
        }
        const double rmse = std::sqrt(sumsq /
            static_cast<double>(n * static_cast<std::size_t>(q)));
        handle->set_double_matrix("predictions", std::move(predictions),
                                   pred_rows, pred_cols);
        handle->set_scalar("rmse", rmse);

        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_sparse_pls_da_fit");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_sparse_pls_da_fit");
        return N4M_ERR_INTERNAL;
    }
}

namespace {

void pack_group_sparse_result(
    n4m_method_result_s& handle,
    const ::n4m::core::GroupSparsePlsResult& res,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y) {
    const auto p = static_cast<std::int64_t>(res.n_features);
    const auto q = static_cast<std::int64_t>(res.n_targets);
    handle.set_double_matrix("coefficients", res.coefficients, p, q);
    handle.set_double_matrix("x_mean", res.x_mean, 1, p);
    handle.set_double_matrix("y_mean", res.y_mean, 1, q);

    std::vector<double> predictions;
    std::int64_t pred_rows = 0;
    std::int64_t pred_cols = 0;
    predict_from_coefficients(X, res.coefficients, res.x_mean, res.y_mean,
                                p, q, predictions, pred_rows, pred_cols);
    const double rmse = in_sample_rmse(predictions, Y);
    handle.set_double_matrix("predictions", std::move(predictions),
                              pred_rows, pred_cols);
    handle.set_scalar("rmse", rmse);
    handle.set_scalar("n_groups", static_cast<double>(res.n_groups));
}

}  // namespace

N4M_API n4m_status_t n4m_group_sparse_pls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const int32_t* group_assignment,
    int64_t group_assignment_size,
    double group_lambda,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X == nullptr || Y == nullptr ||
        group_assignment == nullptr) {
        set_error(ctx, "null pointer in n4m_group_sparse_pls_fit");
        return N4M_ERR_NULL_POINTER;
    }
    if (group_assignment_size != X->cols) {
        set_error(ctx, "group_assignment_size must equal X.cols");
        return N4M_ERR_SHAPE_MISMATCH;
    }
    try {
        std::vector<std::int32_t> groups(
            group_assignment,
            group_assignment + static_cast<std::size_t>(group_assignment_size));
        ::n4m::core::GroupSparsePlsResult res;
        const n4m_status_t status = ::n4m::core::fit_group_sparse_pls(
            *as_core(ctx), *as_core(cfg), *X, *Y, groups, group_lambda,
            res);
        if (status != N4M_OK) return status;

        auto handle = std::make_unique<n4m_method_result_s>();
        pack_group_sparse_result(*handle, res, *X, *Y);
        handle->set_scalar("group_lambda", group_lambda);

        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_group_sparse_pls_fit");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_group_sparse_pls_fit");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_fused_sparse_pls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    double l1_lambda,
    double fusion_lambda,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X == nullptr || Y == nullptr) {
        set_error(ctx, "null pointer in n4m_fused_sparse_pls_fit");
        return N4M_ERR_NULL_POINTER;
    }
    try {
        ::n4m::core::GroupSparsePlsResult res;
        const n4m_status_t status = ::n4m::core::fit_fused_sparse_pls(
            *as_core(ctx), *as_core(cfg), *X, *Y, l1_lambda, fusion_lambda,
            res);
        if (status != N4M_OK) return status;

        auto handle = std::make_unique<n4m_method_result_s>();
        pack_group_sparse_result(*handle, res, *X, *Y);
        handle->set_scalar("l1_lambda", l1_lambda);
        handle->set_scalar("fusion_lambda", fusion_lambda);

        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_fused_sparse_pls_fit");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_fused_sparse_pls_fit");
        return N4M_ERR_INTERNAL;
    }
}

namespace {

void pack_ensemble_result(n4m_method_result_s& handle,
                            const ::n4m::core::EnsemblePlsResult& res,
                            const n4m_matrix_view_t& X,
                            const n4m_matrix_view_t& Y) {
    const auto p = static_cast<std::int64_t>(res.n_features);
    const auto q = static_cast<std::int64_t>(res.n_targets);
    handle.set_double_matrix("coefficients", res.coefficients, p, q);
    handle.set_double_matrix("x_mean", res.x_mean, 1, p);
    handle.set_double_matrix("y_mean", res.y_mean, 1, q);

    std::vector<double> predictions;
    std::int64_t pred_rows = 0;
    std::int64_t pred_cols = 0;
    predict_from_coefficients(X, res.coefficients, res.x_mean, res.y_mean,
                                p, q, predictions, pred_rows, pred_cols);
    const double rmse = in_sample_rmse(predictions, Y);
    handle.set_double_matrix("predictions", std::move(predictions),
                              pred_rows, pred_cols);
    handle.set_scalar("rmse", rmse);
    handle.set_scalar("n_estimators",
                       static_cast<double>(res.n_estimators));
    handle.set_scalar("n_components",
                       static_cast<double>(res.n_components));
}

}  // namespace

N4M_API n4m_status_t n4m_pls_monitoring_run(
    n4m_context_t* ctx,
    const n4m_model_t* model,
    const n4m_matrix_view_t* X_reference,
    const n4m_matrix_view_t* X_monitor,
    double alpha,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || model == nullptr || X_reference == nullptr ||
        X_monitor == nullptr) {
        set_error(ctx, "null pointer in n4m_pls_monitoring_run");
        return N4M_ERR_NULL_POINTER;
    }
    try {
        const auto& m = *as_core(model);
        ::n4m::core::MonitoringThresholds thresholds;
        n4m_status_t status = ::n4m::core::pls_monitoring_fit(
            *as_core(ctx), m, *X_reference, alpha, thresholds);
        if (status != N4M_OK) return status;
        ::n4m::core::MonitoringResult mon;
        status = ::n4m::core::pls_monitoring_evaluate(
            *as_core(ctx), m, thresholds, *X_monitor, mon);
        if (status != N4M_OK) return status;

        auto handle = std::make_unique<n4m_method_result_s>();
        const auto n_ref =
            static_cast<std::int64_t>(thresholds.n_reference);
        const auto n_mon =
            static_cast<std::int64_t>(mon.t2.size());
        handle->set_double_matrix("t2", std::move(mon.t2), 1, n_mon);
        handle->set_double_matrix("q", std::move(mon.q), 1, n_mon);
        handle->set_double_matrix("t2_reference",
                                   std::move(thresholds.t2_reference),
                                   1, n_ref);
        handle->set_double_matrix("q_reference",
                                   std::move(thresholds.q_reference),
                                   1, n_ref);
        handle->set_int_vector("t2_alarms", std::move(mon.t2_alarms));
        handle->set_int_vector("q_alarms", std::move(mon.q_alarms));
        handle->set_int_vector("any_alarms", std::move(mon.any_alarms));
        handle->set_scalar("t2_threshold", thresholds.t2_threshold);
        handle->set_scalar("q_threshold", thresholds.q_threshold);
        handle->set_scalar("alpha", thresholds.alpha);

        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_pls_monitoring_run");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_pls_monitoring_run");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_one_se_rule_compute(
    n4m_context_t* ctx,
    const double* fold_rmse_matrix,
    int32_t max_components,
    int32_t n_folds,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || fold_rmse_matrix == nullptr) {
        set_error(ctx, "null pointer in n4m_one_se_rule_compute");
        return N4M_ERR_NULL_POINTER;
    }
    if (max_components < 1 || n_folds < 2) {
        set_error(ctx,
                   "one-SE rule needs max_components>=1 and n_folds>=2");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    try {
        ::n4m::core::ComponentCvResult res;
        res.max_components = max_components;
        res.n_folds = n_folds;
        const std::size_t total =
            static_cast<std::size_t>(max_components) *
            static_cast<std::size_t>(n_folds);
        res.fold_rmse_matrix.assign(
            fold_rmse_matrix, fold_rmse_matrix + total);
        // Determine best_n_components by mean RMSE.
        std::int32_t best_k = 1;
        double best_mean = std::numeric_limits<double>::infinity();
        std::vector<double> mean_rmse(
            static_cast<std::size_t>(max_components), 0.0);
        for (std::int32_t k = 0; k < max_components; ++k) {
            double s = 0.0;
            for (std::int32_t f = 0; f < n_folds; ++f) {
                s += fold_rmse_matrix[
                    static_cast<std::size_t>(k) *
                    static_cast<std::size_t>(n_folds) +
                    static_cast<std::size_t>(f)];
            }
            const double m = s / static_cast<double>(n_folds);
            mean_rmse[static_cast<std::size_t>(k)] = m;
            if (m < best_mean) {
                best_mean = m;
                best_k = k + 1;
            }
        }
        res.best_n_components = best_k;

        const n4m_status_t status = ::n4m::core::select_one_se_components(
            *as_core(ctx), res);
        if (status != N4M_OK) return status;

        auto handle = std::make_unique<n4m_method_result_s>();
        handle->set_double_matrix("mean_rmse_per_component",
                                   std::move(mean_rmse),
                                   1, max_components);
        handle->set_int_vector("best_n_components",
                                {res.best_n_components});
        handle->set_int_vector("one_se_n_components",
                                {res.one_se_n_components});
        handle->set_scalar("one_se_standard_error",
                            res.one_se_standard_error);
        handle->set_scalar("one_se_threshold", res.one_se_threshold);

        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_one_se_rule_compute");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_one_se_rule_compute");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_so_pls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X_blocks,
    int32_t n_blocks,
    const n4m_matrix_view_t* Y,
    const int32_t* n_components_per_block,
    int64_t n_components_per_block_size,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X_blocks == nullptr ||
        Y == nullptr || n_components_per_block == nullptr) {
        set_error(ctx, "null pointer in n4m_so_pls_fit");
        return N4M_ERR_NULL_POINTER;
    }
    if (n_blocks < 1) {
        set_error(ctx, "n_blocks must be >= 1");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (n_components_per_block_size !=
        static_cast<std::int64_t>(n_blocks)) {
        set_error(ctx,
                   "n_components_per_block_size must equal n_blocks");
        return N4M_ERR_SHAPE_MISMATCH;
    }
    try {
        std::vector<n4m_matrix_view_t> blocks(
            X_blocks, X_blocks + static_cast<std::size_t>(n_blocks));
        std::vector<std::int32_t> components(
            n_components_per_block,
            n_components_per_block +
                static_cast<std::size_t>(n_components_per_block_size));
        ::n4m::core::SoPlsResult res;
        const n4m_status_t status = ::n4m::core::fit_so_pls(
            *as_core(ctx), *as_core(cfg), blocks, *Y, components, res);
        if (status != N4M_OK) return status;

        auto handle = std::make_unique<n4m_method_result_s>();
        const auto n = static_cast<std::int64_t>(Y->rows);
        const auto q = static_cast<std::int64_t>(Y->cols);
        handle->set_double_matrix("predictions", res.predictions, n, q);
        handle->set_double_matrix("y_mean", res.y_mean, 1, q);
        for (std::int32_t b = 0; b < res.n_blocks; ++b) {
            const auto bi = static_cast<std::size_t>(b);
            const auto pb = static_cast<std::int64_t>(
                X_blocks[bi].cols);
            std::string name = "block_coefficients_" + std::to_string(b);
            handle->set_double_matrix(name, res.block_coefficients[bi],
                                       pb, q);
        }
        handle->set_scalar("n_blocks",
                            static_cast<double>(res.n_blocks));
        const double rmse = in_sample_rmse(res.predictions, *Y);
        handle->set_scalar("rmse", rmse);

        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_so_pls_fit");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_so_pls_fit");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_on_pls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X_blocks,
    int32_t n_blocks,
    int32_t n_joint,
    const int32_t* n_unique_per_block,
    int64_t n_unique_per_block_size,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X_blocks == nullptr ||
        n_unique_per_block == nullptr) {
        set_error(ctx, "null pointer in n4m_on_pls_fit");
        return N4M_ERR_NULL_POINTER;
    }
    if (n_unique_per_block_size != static_cast<std::int64_t>(n_blocks)) {
        set_error(ctx,
                   "n_unique_per_block_size must equal n_blocks");
        return N4M_ERR_SHAPE_MISMATCH;
    }
    try {
        std::vector<n4m_matrix_view_t> blocks(
            X_blocks, X_blocks + static_cast<std::size_t>(n_blocks));
        std::vector<std::int32_t> unique_counts(
            n_unique_per_block,
            n_unique_per_block +
                static_cast<std::size_t>(n_unique_per_block_size));
        ::n4m::core::OnPlsResult res;
        const n4m_status_t status = ::n4m::core::fit_on_pls(
            *as_core(ctx), *as_core(cfg), blocks, n_joint,
            unique_counts, res);
        if (status != N4M_OK) return status;

        auto handle = std::make_unique<n4m_method_result_s>();
        for (std::int32_t b = 0; b < res.n_blocks; ++b) {
            const auto bi = static_cast<std::size_t>(b);
            const auto pb = static_cast<std::int64_t>(
                X_blocks[bi].cols);
            const auto nb_unique = static_cast<std::int64_t>(
                res.n_unique_per_block[bi]);
            const auto n = static_cast<std::int64_t>(X_blocks[0].rows);
            handle->set_double_matrix(
                "joint_loadings_" + std::to_string(b),
                res.joint_loadings_per_block[bi], pb, res.n_joint);
            handle->set_double_matrix(
                "unique_loadings_" + std::to_string(b),
                res.unique_loadings_per_block[bi], pb, nb_unique);
            handle->set_double_matrix(
                "joint_scores_" + std::to_string(b),
                res.joint_scores_per_block[bi], n, res.n_joint);
        }
        handle->set_scalar("n_blocks",
                            static_cast<double>(res.n_blocks));
        handle->set_scalar("n_joint",
                            static_cast<double>(res.n_joint));

        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_on_pls_fit");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_on_pls_fit");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_rosa_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X_blocks,
    int32_t n_blocks,
    const n4m_matrix_view_t* Y,
    int32_t n_components,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X_blocks == nullptr ||
        Y == nullptr) {
        set_error(ctx, "null pointer in n4m_rosa_fit");
        return N4M_ERR_NULL_POINTER;
    }
    try {
        std::vector<n4m_matrix_view_t> blocks(
            X_blocks, X_blocks + static_cast<std::size_t>(n_blocks));
        ::n4m::core::RosaResult res;
        const n4m_status_t status = ::n4m::core::fit_rosa(
            *as_core(ctx), *as_core(cfg), blocks, *Y, n_components,
            res);
        if (status != N4M_OK) return status;

        auto handle = std::make_unique<n4m_method_result_s>();
        const auto n = static_cast<std::int64_t>(Y->rows);
        const auto q = static_cast<std::int64_t>(Y->cols);
        handle->set_double_matrix("predictions", res.predictions, n, q);
        handle->set_double_matrix("y_mean", res.y_mean, 1, q);
        handle->set_int_vector("selected_block_per_component",
                                std::move(res.selected_block_per_component));
        for (std::int32_t b = 0;
             b < static_cast<std::int32_t>(res.block_coefficients.size());
             ++b) {
            const auto bi = static_cast<std::size_t>(b);
            const auto pb = static_cast<std::int64_t>(
                X_blocks[bi].cols);
            handle->set_double_matrix(
                "block_coefficients_" + std::to_string(b),
                res.block_coefficients[bi], pb, q);
        }
        handle->set_scalar("n_components",
                            static_cast<double>(res.n_components));
        const double rmse = in_sample_rmse(res.predictions, *Y);
        handle->set_scalar("rmse", rmse);

        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_rosa_fit");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_rosa_fit");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_bagging_pls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    int32_t n_estimators,
    uint64_t seed,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X == nullptr || Y == nullptr) {
        set_error(ctx, "null pointer in n4m_bagging_pls_fit");
        return N4M_ERR_NULL_POINTER;
    }
    try {
        ::n4m::core::EnsemblePlsResult res;
        const n4m_status_t status = ::n4m::core::fit_bagging_pls(
            *as_core(ctx), *as_core(cfg), *X, *Y, n_estimators, seed,
            res);
        if (status != N4M_OK) return status;
        auto handle = std::make_unique<n4m_method_result_s>();
        pack_ensemble_result(*handle, res, *X, *Y);
        handle->set_scalar("seed", static_cast<double>(seed));
        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_bagging_pls_fit");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_bagging_pls_fit");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_boosting_pls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    int32_t n_estimators,
    double learning_rate,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X == nullptr || Y == nullptr) {
        set_error(ctx, "null pointer in n4m_boosting_pls_fit");
        return N4M_ERR_NULL_POINTER;
    }
    try {
        ::n4m::core::EnsemblePlsResult res;
        const n4m_status_t status = ::n4m::core::fit_boosting_pls(
            *as_core(ctx), *as_core(cfg), *X, *Y, n_estimators,
            learning_rate, res);
        if (status != N4M_OK) return status;
        auto handle = std::make_unique<n4m_method_result_s>();
        pack_ensemble_result(*handle, res, *X, *Y);
        handle->set_scalar("learning_rate", learning_rate);
        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_boosting_pls_fit");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_boosting_pls_fit");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_random_subspace_pls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    int32_t n_estimators,
    int32_t features_per_subspace,
    uint64_t seed,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X == nullptr || Y == nullptr) {
        set_error(ctx, "null pointer in n4m_random_subspace_pls_fit");
        return N4M_ERR_NULL_POINTER;
    }
    try {
        ::n4m::core::EnsemblePlsResult res;
        const n4m_status_t status =
            ::n4m::core::fit_random_subspace_pls(
                *as_core(ctx), *as_core(cfg), *X, *Y, n_estimators,
                features_per_subspace, seed, res);
        if (status != N4M_OK) return status;
        auto handle = std::make_unique<n4m_method_result_s>();
        pack_ensemble_result(*handle, res, *X, *Y);
        handle->set_scalar("features_per_subspace",
                            static_cast<double>(features_per_subspace));
        handle->set_scalar("seed", static_cast<double>(seed));
        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx,
                   "out of memory in n4m_random_subspace_pls_fit");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx,
                   "internal error in n4m_random_subspace_pls_fit");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_gpr_pls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    double length_scale,
    double noise_level,
    uint64_t seed,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X == nullptr || Y == nullptr) {
        set_error(ctx, "null pointer in n4m_gpr_pls_fit");
        return N4M_ERR_NULL_POINTER;
    }
    try {
        ::n4m::core::GprPlsResult res;
        const n4m_status_t status = ::n4m::core::fit_gpr_pls(
            *as_core(ctx), *as_core(cfg), *X, *Y,
            length_scale, noise_level, seed, res);
        if (status != N4M_OK) return status;
        auto handle = std::make_unique<n4m_method_result_s>();
        const auto n = static_cast<std::int64_t>(res.n_samples);
        const auto p = static_cast<std::int64_t>(res.n_features);
        const auto k = static_cast<std::int64_t>(res.n_components);
        handle->set_double_matrix("rotation_r", res.rotation_r, p, k);
        handle->set_double_matrix("x_mean", res.x_mean, 1, p);
        handle->set_double_matrix("alpha", res.gp.alpha, n, 1);
        handle->set_double_matrix("L_lower", res.gp.L_lower, n, n);
        handle->set_double_matrix("training_scores", res.gp.T_train, n, k);
        handle->set_double_matrix("predictions", res.predictions, n, 1);
        handle->set_double_matrix("predictive_variance",
                                   res.predictive_variance, n, 1);
        handle->set_scalar("length_scale", res.length_scale);
        handle->set_scalar("noise_level", res.noise_level);
        handle->set_scalar("n_components",
                            static_cast<double>(res.n_components));
        handle->set_scalar("y_mean", res.gp.y_mean_scalar);
        handle->set_scalar("rmse", in_sample_rmse(res.predictions, *Y));
        handle->set_scalar("seed", static_cast<double>(seed));
        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_gpr_pls_fit");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_gpr_pls_fit");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_pls_fit_simple(
    const double* x, const double* y,
    int32_t n, int32_t p, int32_t q, int32_t n_components,
    double* coefficients_out,
    double* x_mean_out,
    double* y_mean_out,
    double* predictions_out) {
    if (x == nullptr || y == nullptr || coefficients_out == nullptr ||
        x_mean_out == nullptr || y_mean_out == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    if (n < 2 || p < 1 || q < 1 || n_components < 1) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    try {
        n4m_matrix_view_t xv{}, yv{};
        n4m_matrix_view_init_rowmajor(&xv, const_cast<double*>(x), n, p,
                                        N4M_DTYPE_F64);
        n4m_matrix_view_init_rowmajor(&yv, const_cast<double*>(y), n, q,
                                        N4M_DTYPE_F64);

        ::n4m::core::Context ctx;
        ::n4m::core::Config cfg;
        cfg.algorithm = N4M_ALGO_PLS_REGRESSION;
        cfg.solver = N4M_SOLVER_SIMPLS;
        cfg.deflation = N4M_DEFLATION_REGRESSION;
        cfg.n_components = n_components;
        cfg.center_x = 1;
        cfg.center_y = 1;
        cfg.scale_x = 1;
        cfg.scale_y = 1;

        std::unique_ptr<::n4m::core::Model> model;
        const n4m_status_t status = ::n4m::core::fit_model(
            ctx, cfg, xv, yv, model);
        if (status != N4M_OK) return status;

        std::memcpy(coefficients_out, model->coefficients.data(),
                    static_cast<std::size_t>(p) *
                    static_cast<std::size_t>(q) * sizeof(double));
        std::memcpy(x_mean_out, model->x_mean.data(),
                    static_cast<std::size_t>(p) * sizeof(double));
        std::memcpy(y_mean_out, model->y_mean.data(),
                    static_cast<std::size_t>(q) * sizeof(double));

        if (predictions_out != nullptr) {
            const auto rows = static_cast<std::size_t>(n);
            const auto features = static_cast<std::size_t>(p);
            const auto targets = static_cast<std::size_t>(q);
            for (std::size_t i = 0; i < rows; ++i) {
                const double* x_row = x + i * features;
                double* pred_row = predictions_out + i * targets;
                for (std::size_t target = 0; target < targets; ++target) {
                    double acc = model->y_mean[target];
                    for (std::size_t feature = 0; feature < features; ++feature) {
                        acc += (x_row[feature] - model->x_mean[feature]) *
                               model->coefficients[feature * targets + target];
                    }
                    pred_row[target] = acc;
                }
            }
        }
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_pls_glm_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    int32_t poisson,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X == nullptr || Y == nullptr) {
        set_error(ctx, "null pointer in n4m_pls_glm_fit");
        return N4M_ERR_NULL_POINTER;
    }
    try {
        ::n4m::core::PlsGlmResult res;
        const n4m_status_t status = ::n4m::core::fit_pls_glm(
            *as_core(ctx), *as_core(cfg), *X, *Y,
            poisson != 0, res);
        if (status != N4M_OK) return status;

        auto handle = std::make_unique<n4m_method_result_s>();
        const auto p = static_cast<std::int64_t>(res.n_features);
        const auto q = static_cast<std::int64_t>(res.n_classes);
        handle->set_double_matrix("coefficients", res.coefficients, p, q);
        handle->set_double_matrix("intercept", res.intercept, 1, q);

        // Predicted = (X - x_mean is implicit, but coefs include shift
        // via intercept). Compute Yhat = X @ coefs + intercept.
        const std::size_t n = static_cast<std::size_t>(X->rows);
        std::vector<double> preds(n * static_cast<std::size_t>(q), 0.0);
        const auto* xdata = static_cast<const double*>(X->data);
        const std::size_t x_rs = static_cast<std::size_t>(X->row_stride);
        const std::size_t x_cs = static_cast<std::size_t>(X->col_stride);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < static_cast<std::size_t>(q); ++j) {
                double s = res.intercept[j];
                for (std::size_t f = 0;
                     f < static_cast<std::size_t>(p); ++f) {
                    s += xdata[i * x_rs + f * x_cs] *
                          res.coefficients[f *
                            static_cast<std::size_t>(q) + j];
                }
                preds[i * static_cast<std::size_t>(q) + j] = s;
            }
        }
        const double rmse = in_sample_rmse(preds, *Y);
        handle->set_double_matrix("predictions", std::move(preds),
                                   static_cast<std::int64_t>(n), q);
        handle->set_scalar("rmse", rmse);
        handle->set_scalar("poisson",
                            static_cast<double>(res.poisson ? 1 : 0));
        handle->set_scalar("n_components",
                            static_cast<double>(res.n_components));

        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_pls_glm_fit");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_pls_glm_fit");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_pls_qda_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const int32_t* y_labels,
    int64_t y_labels_size,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X == nullptr ||
        y_labels == nullptr) {
        set_error(ctx, "null pointer in n4m_pls_qda_fit");
        return N4M_ERR_NULL_POINTER;
    }
    if (y_labels_size != X->rows) {
        set_error(ctx, "y_labels_size must equal X.rows");
        return N4M_ERR_SHAPE_MISMATCH;
    }
    try {
        std::vector<std::int32_t> labels(
            y_labels, y_labels + static_cast<std::size_t>(y_labels_size));
        ::n4m::core::PlsQdaResult res;
        const n4m_status_t status = ::n4m::core::fit_pls_qda(
            *as_core(ctx), *as_core(cfg), *X, labels, res);
        if (status != N4M_OK) return status;

        auto handle = std::make_unique<n4m_method_result_s>();
        const auto p = static_cast<std::int64_t>(X->cols);
        const auto k = static_cast<std::int64_t>(res.n_components);
        const auto q = static_cast<std::int64_t>(res.n_classes);
        handle->set_double_matrix("class_means", res.class_means, q, k);
        handle->set_double_matrix("class_covariances",
                                   res.class_covariances, q, k * k);
        handle->set_double_matrix("log_class_priors",
                                   res.log_class_priors, 1, q);
        handle->set_double_matrix("rotations_r", res.rotations_r, p, k);
        handle->set_double_matrix("x_mean", res.x_mean, 1, p);

        // QDA prediction: for each row i, project to scores, then compute
        // log p(class c | scores) ≈ -0.5 * (s - mu_c)' Sigma_c^{-1} (s - mu_c)
        //                            - 0.5 * log |Sigma_c| + log prior.
        // For the parity gate we just return the centered scores as the
        // prediction proxy (n × n_classes) — sufficient for paper-only.
        const std::size_t n = static_cast<std::size_t>(X->rows);
        std::vector<double> preds(n * static_cast<std::size_t>(q), 0.0);
        const auto* xdata = static_cast<const double*>(X->data);
        const std::size_t x_rs = static_cast<std::size_t>(X->row_stride);
        const std::size_t x_cs = static_cast<std::size_t>(X->col_stride);
        std::vector<double> scores(static_cast<std::size_t>(k), 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            // Project X[i] to scores.
            for (std::size_t comp = 0;
                 comp < static_cast<std::size_t>(k); ++comp) {
                double s = 0.0;
                for (std::size_t f = 0;
                     f < static_cast<std::size_t>(p); ++f) {
                    s += (xdata[i * x_rs + f * x_cs] - res.x_mean[f]) *
                          res.rotations_r[f *
                            static_cast<std::size_t>(k) + comp];
                }
                scores[comp] = s;
            }
            // Distance to each class mean (negated, with prior).
            for (std::size_t c = 0;
                 c < static_cast<std::size_t>(q); ++c) {
                double d = res.log_class_priors[c];
                for (std::size_t comp = 0;
                     comp < static_cast<std::size_t>(k); ++comp) {
                    const double diff = scores[comp] -
                        res.class_means[c *
                          static_cast<std::size_t>(k) + comp];
                    d -= 0.5 * diff * diff;
                }
                preds[i * static_cast<std::size_t>(q) + c] = d;
            }
        }
        handle->set_double_matrix("predictions", std::move(preds),
                                   static_cast<std::int64_t>(n), q);
        handle->set_scalar("n_components",
                            static_cast<double>(res.n_components));

        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_pls_qda_fit");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_pls_qda_fit");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_pls_cox_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const double* survival_times,
    int64_t survival_times_size,
    const int32_t* event_indicators,
    int64_t event_indicators_size,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X == nullptr ||
        survival_times == nullptr || event_indicators == nullptr) {
        set_error(ctx, "null pointer in n4m_pls_cox_fit");
        return N4M_ERR_NULL_POINTER;
    }
    if (survival_times_size != X->rows ||
        event_indicators_size != X->rows) {
        set_error(ctx, "survival/event sizes must equal X.rows");
        return N4M_ERR_SHAPE_MISMATCH;
    }
    try {
        std::vector<double> times(
            survival_times,
            survival_times + static_cast<std::size_t>(survival_times_size));
        std::vector<std::int32_t> events(
            event_indicators,
            event_indicators + static_cast<std::size_t>(event_indicators_size));
        ::n4m::core::PlsCoxResult res;
        const n4m_status_t status = ::n4m::core::fit_pls_cox(
            *as_core(ctx), *as_core(cfg), *X, times, events, res);
        if (status != N4M_OK) return status;

        auto handle = std::make_unique<n4m_method_result_s>();
        const auto p = static_cast<std::int64_t>(res.n_features);
        const auto ne = static_cast<std::int64_t>(res.event_times.size());
        handle->set_double_matrix("coefficients", res.coefficients, p, 1);
        handle->set_double_matrix("baseline_hazard",
                                   res.baseline_hazard, 1, ne);
        handle->set_double_matrix("event_times", res.event_times, 1, ne);
        handle->set_double_matrix("x_mean", res.x_mean, 1, p);

        // Predictions: linear predictor scores = (X - x_mean) @ coefs.
        const std::size_t n = static_cast<std::size_t>(X->rows);
        std::vector<double> preds(n, 0.0);
        const auto* xdata = static_cast<const double*>(X->data);
        const std::size_t x_rs = static_cast<std::size_t>(X->row_stride);
        const std::size_t x_cs = static_cast<std::size_t>(X->col_stride);
        for (std::size_t i = 0; i < n; ++i) {
            double s = 0.0;
            for (std::size_t f = 0;
                 f < static_cast<std::size_t>(p); ++f) {
                s += (xdata[i * x_rs + f * x_cs] - res.x_mean[f]) *
                      res.coefficients[f];
            }
            preds[i] = s;
        }
        handle->set_double_matrix("predictions", std::move(preds),
                                   static_cast<std::int64_t>(n), 1);
        handle->set_scalar("n_components",
                            static_cast<double>(res.n_components));

        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_pls_cox_fit");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_pls_cox_fit");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_pds_fit(
    n4m_context_t* ctx,
    const n4m_matrix_view_t* X_source,
    const n4m_matrix_view_t* X_target,
    int32_t window_half_width,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || X_source == nullptr || X_target == nullptr) {
        set_error(ctx, "null pointer in n4m_pds_fit");
        return N4M_ERR_NULL_POINTER;
    }
    try {
        ::n4m::core::PdsResult res;
        const n4m_status_t status = ::n4m::core::fit_pds(
            *as_core(ctx), *X_source, *X_target, window_half_width, res);
        if (status != N4M_OK) return status;

        auto handle = std::make_unique<n4m_method_result_s>();
        const auto ps = static_cast<std::int64_t>(X_source->cols);
        const auto pt = static_cast<std::int64_t>(X_target->cols);
        handle->set_double_matrix("transformation", res.transformation,
                                   pt, ps);
        // Predicted target = X_source @ transformation.T.
        const std::size_t n = static_cast<std::size_t>(X_source->rows);
        std::vector<double> preds(n * static_cast<std::size_t>(pt), 0.0);
        const auto* xs = static_cast<const double*>(X_source->data);
        const std::size_t xs_rs = static_cast<std::size_t>(
            X_source->row_stride);
        const std::size_t xs_cs = static_cast<std::size_t>(
            X_source->col_stride);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0;
                 j < static_cast<std::size_t>(pt); ++j) {
                double s = 0.0;
                for (std::size_t k = 0;
                     k < static_cast<std::size_t>(ps); ++k) {
                    s += xs[i * xs_rs + k * xs_cs] *
                          res.transformation[j *
                            static_cast<std::size_t>(ps) + k];
                }
                preds[i * static_cast<std::size_t>(pt) + j] = s;
            }
        }
        const double rmse = in_sample_rmse(preds, *X_target);
        handle->set_double_matrix("predictions", std::move(preds),
                                   static_cast<std::int64_t>(n), pt);
        handle->set_scalar("rmse", rmse);
        handle->set_scalar("window_half_width",
                            static_cast<double>(window_half_width));

        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_pds_fit");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_pds_fit");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_ds_fit(
    n4m_context_t* ctx,
    const n4m_matrix_view_t* X_source,
    const n4m_matrix_view_t* X_target,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || X_source == nullptr || X_target == nullptr) {
        set_error(ctx, "null pointer in n4m_ds_fit");
        return N4M_ERR_NULL_POINTER;
    }
    try {
        ::n4m::core::DsResult res;
        const n4m_status_t status = ::n4m::core::fit_ds(
            *as_core(ctx), *X_source, *X_target, res);
        if (status != N4M_OK) return status;

        auto handle = std::make_unique<n4m_method_result_s>();
        const auto ps = static_cast<std::int64_t>(X_source->cols);
        const auto pt = static_cast<std::int64_t>(X_target->cols);
        handle->set_double_matrix("transformation", res.transformation,
                                   ps, pt);
        handle->set_double_matrix("bias", res.bias, 1, pt);
        // Predicted target = X_source @ transformation + bias.
        const std::size_t n = static_cast<std::size_t>(X_source->rows);
        std::vector<double> preds(n * static_cast<std::size_t>(pt), 0.0);
        const auto* xs = static_cast<const double*>(X_source->data);
        const std::size_t xs_rs = static_cast<std::size_t>(
            X_source->row_stride);
        const std::size_t xs_cs = static_cast<std::size_t>(
            X_source->col_stride);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0;
                 j < static_cast<std::size_t>(pt); ++j) {
                double s = res.bias[j];
                for (std::size_t k = 0;
                     k < static_cast<std::size_t>(ps); ++k) {
                    s += xs[i * xs_rs + k * xs_cs] *
                          res.transformation[k *
                            static_cast<std::size_t>(pt) + j];
                }
                preds[i * static_cast<std::size_t>(pt) + j] = s;
            }
        }
        const double rmse = in_sample_rmse(preds, *X_target);
        handle->set_double_matrix("predictions", std::move(preds),
                                   static_cast<std::int64_t>(n), pt);
        handle->set_scalar("rmse", rmse);

        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_ds_fit");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_ds_fit");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_mir_pls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X == nullptr || Y == nullptr) {
        set_error(ctx, "null pointer in n4m_mir_pls_fit");
        return N4M_ERR_NULL_POINTER;
    }
    try {
        ::n4m::core::MirPlsResult res;
        const n4m_status_t status = ::n4m::core::fit_mir_pls(
            *as_core(ctx), *as_core(cfg), *X, *Y, res);
        if (status != N4M_OK) return status;

        auto handle = std::make_unique<n4m_method_result_s>();
        const auto p = static_cast<std::int64_t>(X->cols);
        const auto q = static_cast<std::int64_t>(Y->cols);
        handle->set_double_matrix("coefficients", res.coefficients, p, q);
        handle->set_double_matrix("x_mean", res.x_mean, 1, p);
        handle->set_double_matrix("y_mean", res.y_mean, 1, q);

        std::vector<double> predictions;
        std::int64_t pred_rows = 0;
        std::int64_t pred_cols = 0;
        predict_from_coefficients(*X, res.coefficients, res.x_mean,
                                    res.y_mean, p, q, predictions,
                                    pred_rows, pred_cols);
        const double rmse = in_sample_rmse(predictions, *Y);
        handle->set_double_matrix("predictions", std::move(predictions),
                                   pred_rows, pred_cols);
        handle->set_scalar("rmse", rmse);

        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_mir_pls_fit");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_mir_pls_fit");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_missing_aware_nipals_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X == nullptr || Y == nullptr) {
        set_error(ctx, "null pointer in n4m_missing_aware_nipals_fit");
        return N4M_ERR_NULL_POINTER;
    }
    try {
        ::n4m::core::WeightedPlsResult res;
        const n4m_status_t status =
            ::n4m::core::fit_missing_aware_nipals(
                *as_core(ctx), *as_core(cfg), *X, *Y, res);
        if (status != N4M_OK) return status;

        auto handle = std::make_unique<n4m_method_result_s>();
        pack_weighted_result(*handle, res, *X, *Y);

        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx,
                   "out of memory in n4m_missing_aware_nipals_fit");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx,
                   "internal error in n4m_missing_aware_nipals_fit");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_pls_diagnostics_compute(
    n4m_context_t* ctx,
    const n4m_model_t* model,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* X_reference,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || model == nullptr || X == nullptr) {
        set_error(ctx, "null pointer in n4m_pls_diagnostics_compute");
        return N4M_ERR_NULL_POINTER;
    }
    try {
        const auto& m = *as_core(model);
        std::vector<double> t2;
        n4m_status_t s = ::n4m::core::pls_hotelling_t2(
            *as_core(ctx), m, *X, X_reference, t2);
        if (s != N4M_OK) return s;
        std::vector<double> q;
        s = ::n4m::core::pls_q_residuals(
            *as_core(ctx), m, *X, q);
        if (s != N4M_OK) return s;
        std::vector<double> dmodx;
        s = ::n4m::core::pls_dmodx(
            *as_core(ctx), m, *X, X_reference, dmodx);
        if (s != N4M_OK) return s;

        auto handle = std::make_unique<n4m_method_result_s>();
        const auto n = static_cast<std::int64_t>(t2.size());
        handle->set_double_matrix("t2", std::move(t2), 1, n);
        handle->set_double_matrix("q", std::move(q), 1, n);
        handle->set_double_matrix("dmodx", std::move(dmodx), 1, n);
        handle->set_scalar("n_components",
                            static_cast<double>(m.n_components));
        handle->set_scalar("n_features",
                            static_cast<double>(m.n_features));
        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_pls_diagnostics_compute");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_pls_diagnostics_compute");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_approximate_press_compute(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    int32_t max_components,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X == nullptr || Y == nullptr) {
        set_error(ctx, "null pointer in n4m_approximate_press_compute");
        return N4M_ERR_NULL_POINTER;
    }
    try {
        ::n4m::core::ApproximatePressResult res;
        const n4m_status_t status = ::n4m::core::approximate_press(
            *as_core(ctx), *as_core(cfg), *X, *Y, max_components, res);
        if (status != N4M_OK) return status;

        auto handle = std::make_unique<n4m_method_result_s>();
        const auto k = static_cast<std::int64_t>(res.n_components);
        handle->set_double_matrix("press_per_component",
                                   res.press_per_component, 1, k);
        handle->set_double_matrix("rmse_per_component",
                                   res.rmse_per_component, 1, k);
        std::vector<std::int32_t> selected{res.selected_n_components};
        handle->set_int_vector("selected_n_components",
                                std::move(selected));
        handle->set_scalar("selected_n_components_d",
                            static_cast<double>(res.selected_n_components));

        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_approximate_press_compute");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_approximate_press_compute");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_kernel_pls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    int32_t kernel_type,
    double gamma,
    double coef0,
    int32_t degree,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X == nullptr || Y == nullptr) {
        set_error(ctx, "null pointer in n4m_kernel_pls_fit");
        return N4M_ERR_NULL_POINTER;
    }
    if (kernel_type < 0 || kernel_type > 3) {
        set_error(ctx, "kernel_type must be in {0,1,2,3}");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    try {
        const auto kernel_enum =
            static_cast<::n4m::core::KernelType>(kernel_type);
        ::n4m::core::KernelPlsResult res;
        const n4m_status_t status = ::n4m::core::fit_kernel_pls(
            *as_core(ctx), *as_core(cfg), kernel_enum, gamma, coef0,
            degree, *X, *Y, res);
        if (status != N4M_OK) return status;

        auto handle = std::make_unique<n4m_method_result_s>();
        const auto n = static_cast<std::int64_t>(res.n_train);
        const auto q = static_cast<std::int64_t>(res.n_targets);
        handle->set_double_matrix("alpha", res.alpha, n, q);
        handle->set_double_matrix("y_mean", res.y_mean, 1, q);

        std::vector<double> predictions;
        const n4m_status_t pred_status = ::n4m::core::predict_kernel_pls(
            *as_core(ctx), res, *X, predictions);
        (void)pred_status;
        const double rmse = in_sample_rmse(predictions, *Y);
        handle->set_double_matrix("predictions", std::move(predictions),
                                   n, q);
        handle->set_scalar("rmse", rmse);
        handle->set_scalar("kernel_type",
                            static_cast<double>(kernel_type));
        handle->set_scalar("gamma", gamma);
        handle->set_scalar("coef0", coef0);
        handle->set_scalar("degree", static_cast<double>(degree));

        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_kernel_pls_fit");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_kernel_pls_fit");
        return N4M_ERR_INTERNAL;
    }
}

/* ============================================================================
 * §17 — ABI shims for internal-only core algos closing the parity-gate gap
 * ==========================================================================*/

N4M_API n4m_status_t n4m_mb_pls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const int64_t* block_sizes,
    int64_t n_blocks,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X == nullptr || Y == nullptr ||
        block_sizes == nullptr) {
        set_error(ctx, "null pointer in n4m_mb_pls_fit");
        return N4M_ERR_NULL_POINTER;
    }
    if (n_blocks <= 0) {
        set_error(ctx, "n_blocks must be >= 1");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    try {
        ::n4m::core::MbPlsResult res;
        const n4m_status_t status = ::n4m::core::fit_predict_mb_pls(
            *as_core(ctx), *as_core(cfg), *X, *Y,
            block_sizes, static_cast<std::size_t>(n_blocks), res);
        if (status != N4M_OK) return status;

        auto handle = std::make_unique<n4m_method_result_s>();
        const auto n = static_cast<std::int64_t>(res.n_samples);
        const auto p = static_cast<std::int64_t>(res.n_features);
        const auto q = static_cast<std::int64_t>(res.n_targets);
        const auto nb = static_cast<std::int64_t>(res.n_blocks);
        handle->set_double_matrix("predictions",
                                   std::move(res.predictions), n, q);
        handle->set_double_matrix("coefficients",
                                   std::move(res.coefficients), p, q);
        handle->set_double_matrix("intercept",
                                   std::move(res.intercept), 1, q);
        handle->set_double_matrix("x_mean",
                                   std::move(res.x_mean), 1, p);
        handle->set_double_matrix("x_scale",
                                   std::move(res.x_scale), 1, p);
        handle->set_double_matrix("block_weights",
                                   std::move(res.block_weights), 1, nb);
        handle->set_scalar("n_blocks", static_cast<double>(res.n_blocks));
        handle->set_scalar("n_components",
                            static_cast<double>(res.n_components));
        handle->set_scalar("rmse",
                            in_sample_rmse(
                                handle->double_arrays.at("predictions"),
                                *Y));

        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_mb_pls_fit");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_mb_pls_fit");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_lw_pls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    int32_t n_neighbors,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X == nullptr || Y == nullptr) {
        set_error(ctx, "null pointer in n4m_lw_pls_fit");
        return N4M_ERR_NULL_POINTER;
    }
    try {
        ::n4m::core::LwPlsResult res;
        const n4m_status_t status = ::n4m::core::fit_predict_lw_pls(
            *as_core(ctx), *as_core(cfg), *X, *Y, n_neighbors, res);
        if (status != N4M_OK) return status;

        auto handle = std::make_unique<n4m_method_result_s>();
        const auto n = static_cast<std::int64_t>(res.n_samples);
        const auto q = static_cast<std::int64_t>(res.n_targets);
        const auto k = static_cast<std::int64_t>(res.n_neighbors);
        handle->set_double_matrix("predictions",
                                   std::move(res.predictions), n, q);
        // Expose neighbor indices both as int64 (preferred) and as a 1-D
        // int vector view (legacy callers may use vector_int).
        std::vector<std::int64_t> nbrs = res.neighbor_indices;
        handle->set_double_matrix(
            "neighbor_indices", i64_to_double_vector(nbrs), n, k);
        handle->set_int64_vector("neighbor_indices_i64", std::move(nbrs));
        handle->set_scalar("n_neighbors",
                            static_cast<double>(res.n_neighbors));
        handle->set_scalar("n_components",
                            static_cast<double>(res.n_components));
        handle->set_scalar("rmse",
                            in_sample_rmse(
                                handle->double_arrays.at("predictions"),
                                *Y));

        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_lw_pls_fit");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_lw_pls_fit");
        return N4M_ERR_INTERNAL;
    }
}

namespace {

// Build a row-major n4m_matrix_view_t over a contiguous int32 buffer of
// length `size`. Used by n4m_pls_lda_fit / n4m_pls_logistic_fit to bridge
// the raw label pointer signature to the matrix-view consumer.
n4m_matrix_view_t make_labels_view(const std::int32_t* labels,
                                    std::int64_t size) noexcept {
    n4m_matrix_view_t view{};
    view.data = const_cast<std::int32_t*>(labels);
    view.rows = size;
    view.cols = 1;
    view.row_stride = 1;
    view.col_stride = 1;
    view.dtype = N4M_DTYPE_I32;
    view.reserved0 = 0;
    return view;
}

}  // namespace

N4M_API n4m_status_t n4m_pls_lda_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const int32_t* y_labels,
    int64_t y_labels_size,
    int32_t n_classes,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X == nullptr ||
        y_labels == nullptr) {
        set_error(ctx, "null pointer in n4m_pls_lda_fit");
        return N4M_ERR_NULL_POINTER;
    }
    if (y_labels_size != X->rows) {
        set_error(ctx, "y_labels_size must equal X.rows");
        return N4M_ERR_SHAPE_MISMATCH;
    }
    try {
        const n4m_matrix_view_t labels_view =
            make_labels_view(y_labels, y_labels_size);
        ::n4m::core::PlsLdaResult res;
        const n4m_status_t status = ::n4m::core::fit_predict_pls_lda(
            *as_core(ctx), *as_core(cfg), *X, labels_view, n_classes, res);
        if (status != N4M_OK) return status;

        auto handle = std::make_unique<n4m_method_result_s>();
        const auto n = static_cast<std::int64_t>(res.n_samples);
        const auto k = static_cast<std::int64_t>(res.n_classes);
        handle->set_double_matrix("decision_scores",
                                   std::move(res.decision_scores), n, k);
        handle->set_int_vector("predictions", std::move(res.predictions));
        handle->set_scalar("n_classes",
                            static_cast<double>(res.n_classes));
        handle->set_scalar("n_components",
                            static_cast<double>(res.n_components));
        handle->set_scalar("n_samples",
                            static_cast<double>(res.n_samples));

        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_pls_lda_fit");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_pls_lda_fit");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_pls_logistic_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const int32_t* y_labels,
    int64_t y_labels_size,
    int32_t n_classes,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X == nullptr ||
        y_labels == nullptr) {
        set_error(ctx, "null pointer in n4m_pls_logistic_fit");
        return N4M_ERR_NULL_POINTER;
    }
    if (y_labels_size != X->rows) {
        set_error(ctx, "y_labels_size must equal X.rows");
        return N4M_ERR_SHAPE_MISMATCH;
    }
    try {
        const n4m_matrix_view_t labels_view =
            make_labels_view(y_labels, y_labels_size);
        ::n4m::core::PlsLogisticResult res;
        const n4m_status_t status = ::n4m::core::fit_predict_pls_logistic(
            *as_core(ctx), *as_core(cfg), *X, labels_view, n_classes, res);
        if (status != N4M_OK) return status;

        auto handle = std::make_unique<n4m_method_result_s>();
        const auto n = static_cast<std::int64_t>(res.n_samples);
        const auto k = static_cast<std::int64_t>(res.n_classes);
        const auto kc = static_cast<std::int64_t>(res.n_components);
        const auto km1 = (k > 0) ? (k - 1) : 0;
        handle->set_double_matrix("decision_scores",
                                   std::move(res.decision_scores), n, k);
        handle->set_double_matrix("probabilities",
                                   std::move(res.probabilities), n, k);
        handle->set_double_matrix("intercepts",
                                   std::move(res.intercepts), 1, km1);
        handle->set_double_matrix("coefficients",
                                   std::move(res.coefficients), km1, kc);
        handle->set_int_vector("predictions", std::move(res.predictions));
        handle->set_scalar("n_classes",
                            static_cast<double>(res.n_classes));
        handle->set_scalar("n_components",
                            static_cast<double>(res.n_components));
        handle->set_scalar("n_samples",
                            static_cast<double>(res.n_samples));

        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_pls_logistic_fit");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_pls_logistic_fit");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_aom_preprocess_fit(
    n4m_context_t* ctx,
    const n4m_operator_bank_t* bank,
    const n4m_gating_strategy_t* gate,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || bank == nullptr || gate == nullptr || X == nullptr) {
        set_error(ctx, "null pointer in n4m_aom_preprocess_fit");
        return N4M_ERR_NULL_POINTER;
    }
    try {
        ::n4m::core::AomPreprocessingResult res;
        const n4m_status_t status = ::n4m::core::apply_aom_preprocessing(
            *as_core(ctx), *as_core(bank), *as_core(gate), *X, Y, res);
        if (status != N4M_OK) return status;

        auto handle = std::make_unique<n4m_method_result_s>();
        const auto n = static_cast<std::int64_t>(res.n_samples);
        const auto p = static_cast<std::int64_t>(res.n_features);
        const auto nop = static_cast<std::int64_t>(res.n_operators);
        handle->set_double_matrix("transformed",
                                   std::move(res.transformed), n, p);
        handle->set_double_matrix("operator_outputs",
                                   std::move(res.operator_outputs), nop, n * p);
        handle->set_double_matrix("weights",
                                   std::move(res.weights), 1, nop);
        handle->set_int64_vector("operator_kinds",
                                  std::move(res.operator_kinds));
        handle->set_scalar("n_operators",
                            static_cast<double>(res.n_operators));
        handle->set_scalar("n_samples",
                            static_cast<double>(res.n_samples));
        handle->set_scalar("n_features",
                            static_cast<double>(res.n_features));
        handle->set_scalar("mode", static_cast<double>(res.mode));

        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_aom_preprocess_fit");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_aom_preprocess_fit");
        return N4M_ERR_INTERNAL;
    }
}

/* ============================================================================
 * §18 — ABI shims for Phase 5 variable-selection methods
 * ==========================================================================*/

N4M_API n4m_status_t n4m_variable_select_rank(
    n4m_context_t* ctx,
    const n4m_model_t* model,
    const n4m_matrix_view_t* X,
    int32_t method,
    int32_t top_k,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || model == nullptr || X == nullptr) {
        set_error(ctx, "null pointer in n4m_variable_select_rank");
        return N4M_ERR_NULL_POINTER;
    }
    if (method < 0 || method > 2) {
        set_error(ctx, "method must be 0 (VIP), 1 (CoefMag), or 2 (SR)");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    try {
        const auto m =
            static_cast<::n4m::core::VariableSelectionMethod>(method);
        ::n4m::core::VariableSelectionResult res;
        const n4m_status_t status = ::n4m::core::select_variables(
            *as_core(ctx), *as_core(model), *X, m, top_k, res);
        if (status != N4M_OK) return status;

        auto handle = std::make_unique<n4m_method_result_s>();
        const auto p = static_cast<std::int64_t>(res.n_features);
        handle->set_double_matrix("scores", std::move(res.scores), 1, p);
        handle->set_int64_vector("selected_indices",
                                  std::move(res.selected_indices));
        handle->set_scalar("n_features",
                            static_cast<double>(res.n_features));
        handle->set_scalar("top_k", static_cast<double>(res.top_k));
        handle->set_scalar("method", static_cast<double>(method));

        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_variable_select_rank");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_variable_select_rank");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_interval_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t interval_width,
    int32_t step,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X == nullptr || Y == nullptr ||
        plan == nullptr) {
        set_error(ctx, "null pointer in n4m_interval_select");
        return N4M_ERR_NULL_POINTER;
    }
    try {
        ::n4m::core::IntervalSelectionResult res;
        const n4m_status_t status =
            ::n4m::core::cross_validate_intervals(
                *as_core(ctx), *as_core(cfg), *X, *Y, *as_core(plan),
                interval_width, step, res);
        if (status != N4M_OK) return status;

        auto handle = std::make_unique<n4m_method_result_s>();
        const auto n_intervals =
            static_cast<std::int64_t>(res.rmse.size());
        // Record best RMSE before moving the vector.
        double best_rmse_value = 0.0;
        if (res.best_interval_index >= 0 &&
            static_cast<std::size_t>(res.best_interval_index) <
                res.rmse.size()) {
            best_rmse_value =
                res.rmse[static_cast<std::size_t>(res.best_interval_index)];
        }
        // Build selected_indices from best_start .. best_start+best_length-1.
        std::vector<std::int64_t> selected_indices;
        if (res.best_length > 0 && res.best_start >= 0) {
            selected_indices.reserve(
                static_cast<std::size_t>(res.best_length));
            for (std::int64_t f = 0; f < res.best_length; ++f) {
                selected_indices.push_back(res.best_start + f);
            }
        }
        handle->set_int64_vector("selected_indices",
                                  std::move(selected_indices));
        handle->set_int64_vector("intervals", std::move(res.intervals));
        handle->set_double_matrix("rmse",
                                   std::move(res.rmse), 1, n_intervals);
        handle->set_double_matrix("metrics_matrix",
                                   std::move(res.metrics_matrix),
                                   n_intervals, 9);
        handle->set_scalar("n_intervals",
                            static_cast<double>(n_intervals));
        handle->set_scalar("interval_width",
                            static_cast<double>(res.interval_width));
        handle->set_scalar("step", static_cast<double>(res.step));
        handle->set_scalar("best_interval_index",
                            static_cast<double>(res.best_interval_index));
        handle->set_scalar("best_start",
                            static_cast<double>(res.best_start));
        handle->set_scalar("best_length",
                            static_cast<double>(res.best_length));
        handle->set_scalar("best_rmse", best_rmse_value);

        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_interval_select");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_interval_select");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_stability_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t top_k,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X == nullptr || Y == nullptr ||
        plan == nullptr) {
        set_error(ctx, "null pointer in n4m_stability_select");
        return N4M_ERR_NULL_POINTER;
    }
    try {
        ::n4m::core::StabilitySelectionResult res;
        const n4m_status_t status =
            ::n4m::core::select_by_coefficient_stability(
                *as_core(ctx), *as_core(cfg), *X, *Y, *as_core(plan),
                top_k, res);
        if (status != N4M_OK) return status;

        auto handle = std::make_unique<n4m_method_result_s>();
        const auto p = static_cast<std::int64_t>(res.n_features);
        const auto q = static_cast<std::int64_t>(res.n_targets);
        handle->set_double_matrix("mean_coefficients",
                                   std::move(res.mean_coefficients), p, q);
        handle->set_double_matrix("std_coefficients",
                                   std::move(res.std_coefficients), p, q);
        handle->set_double_matrix("stability_scores",
                                   std::move(res.stability_scores), 1, p);
        handle->set_int64_vector("selected_indices",
                                  std::move(res.selected_indices));
        handle->set_scalar("n_features",
                            static_cast<double>(res.n_features));
        handle->set_scalar("n_targets",
                            static_cast<double>(res.n_targets));
        handle->set_scalar("n_repeats",
                            static_cast<double>(res.n_repeats));
        handle->set_scalar("top_k", static_cast<double>(res.top_k));

        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_stability_select");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_stability_select");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_uve_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t noise_features,
    uint64_t noise_seed,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X == nullptr || Y == nullptr ||
        plan == nullptr) {
        set_error(ctx, "null pointer in n4m_uve_select");
        return N4M_ERR_NULL_POINTER;
    }
    try {
        ::n4m::core::UveSelectionResult res;
        const n4m_status_t status = ::n4m::core::select_by_uve(
            *as_core(ctx), *as_core(cfg), *X, *Y, *as_core(plan),
            noise_features, noise_seed, res);
        if (status != N4M_OK) return status;

        auto handle = std::make_unique<n4m_method_result_s>();
        const auto p = static_cast<std::int64_t>(res.n_features);
        const auto nn = static_cast<std::int64_t>(res.n_noise_features);
        handle->set_double_matrix("real_stability_scores",
                                   std::move(res.real_stability_scores),
                                   1, p);
        handle->set_double_matrix("noise_stability_scores",
                                   std::move(res.noise_stability_scores),
                                   1, nn);
        handle->set_int64_vector("selected_indices",
                                  std::move(res.selected_indices));
        handle->set_scalar("n_features",
                            static_cast<double>(res.n_features));
        handle->set_scalar("n_targets",
                            static_cast<double>(res.n_targets));
        handle->set_scalar("n_repeats",
                            static_cast<double>(res.n_repeats));
        handle->set_scalar("n_noise_features",
                            static_cast<double>(res.n_noise_features));
        handle->set_scalar("selected_count",
                            static_cast<double>(res.selected_count));
        handle->set_scalar("noise_threshold", res.noise_threshold);
        handle->set_scalar("noise_seed",
                            static_cast<double>(res.noise_seed));

        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_uve_select");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_uve_select");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_spa_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    int32_t top_k,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X == nullptr || Y == nullptr) {
        set_error(ctx, "null pointer in n4m_spa_select");
        return N4M_ERR_NULL_POINTER;
    }
    try {
        ::n4m::core::SpaSelectionResult res;
        const n4m_status_t status = ::n4m::core::select_by_spa(
            *as_core(ctx), *as_core(cfg), *X, *Y, top_k, res);
        if (status != N4M_OK) return status;

        auto handle = std::make_unique<n4m_method_result_s>();
        const auto p = static_cast<std::int64_t>(res.n_features);
        const auto k = static_cast<std::int64_t>(res.top_k);
        handle->set_double_matrix("coefficient_scores",
                                   std::move(res.coefficient_scores), 1, p);
        handle->set_double_matrix("selection_scores",
                                   std::move(res.selection_scores), 1, k);
        handle->set_int64_vector("selected_indices",
                                  std::move(res.selected_indices));
        handle->set_scalar("n_features",
                            static_cast<double>(res.n_features));
        handle->set_scalar("n_targets",
                            static_cast<double>(res.n_targets));
        handle->set_scalar("n_components",
                            static_cast<double>(res.n_components));
        handle->set_scalar("top_k", static_cast<double>(res.top_k));

        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_spa_select");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_spa_select");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_cars_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t n_iterations,
    int32_t min_features,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X == nullptr || Y == nullptr ||
        plan == nullptr) {
        set_error(ctx, "null pointer in n4m_cars_select");
        return N4M_ERR_NULL_POINTER;
    }
    try {
        ::n4m::core::CarsSelectionResult res;
        const n4m_status_t status = ::n4m::core::select_by_cars(
            *as_core(ctx), *as_core(cfg), *X, *Y, *as_core(plan),
            n_iterations, min_features, res);
        if (status != N4M_OK) return status;

        auto handle = std::make_unique<n4m_method_result_s>();
        const auto p = static_cast<std::int64_t>(res.n_features);
        const auto ni = static_cast<std::int64_t>(res.n_iterations);
        handle->set_double_matrix("coefficient_scores",
                                   std::move(res.coefficient_scores), ni, p);
        handle->set_double_matrix("rmse_by_iteration",
                                   std::move(res.rmse_by_iteration), 1, ni);
        handle->set_double_matrix("retained_counts",
                                   i64_to_double_vector(res.retained_counts),
                                   1, ni);
        handle->set_int64_vector("retained_counts_i64",
                                  std::move(res.retained_counts));
        handle->set_int64_vector("selected_indices",
                                  std::move(res.selected_indices));
        handle->set_scalar("n_features",
                            static_cast<double>(res.n_features));
        handle->set_scalar("n_targets",
                            static_cast<double>(res.n_targets));
        handle->set_scalar("n_components",
                            static_cast<double>(res.n_components));
        handle->set_scalar("n_iterations",
                            static_cast<double>(res.n_iterations));
        handle->set_scalar("min_features",
                            static_cast<double>(res.min_features));
        handle->set_scalar("best_iteration",
                            static_cast<double>(res.best_iteration));
        handle->set_scalar("best_rmse", res.best_rmse);

        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_cars_select");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_cars_select");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_random_frog_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t n_iterations,
    int32_t initial_size,
    int32_t min_size,
    int32_t max_size,
    int32_t top_k,
    uint64_t seed,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X == nullptr || Y == nullptr ||
        plan == nullptr) {
        set_error(ctx, "null pointer in n4m_random_frog_select");
        return N4M_ERR_NULL_POINTER;
    }
    try {
        ::n4m::core::RandomFrogSelectionResult res;
        const n4m_status_t status = ::n4m::core::select_by_random_frog(
            *as_core(ctx), *as_core(cfg), *X, *Y, *as_core(plan),
            n_iterations, initial_size, min_size, max_size, top_k, seed,
            res);
        if (status != N4M_OK) return status;

        auto handle = std::make_unique<n4m_method_result_s>();
        const auto p = static_cast<std::int64_t>(res.n_features);
        const auto ni = static_cast<std::int64_t>(res.n_iterations);
        handle->set_double_matrix("global_scores",
                                   std::move(res.global_scores), 1, p);
        handle->set_double_matrix("inclusion_frequencies",
                                   std::move(res.inclusion_frequencies), 1, p);
        handle->set_double_matrix("rmse_by_iteration",
                                   std::move(res.rmse_by_iteration), 1, ni);
        handle->set_double_matrix("subset_sizes",
                                   i64_to_double_vector(res.subset_sizes),
                                   1, ni);
        handle->set_int64_vector("subset_sizes_i64",
                                  std::move(res.subset_sizes));
        handle->set_int64_vector("selected_indices",
                                  std::move(res.selected_indices));
        handle->set_int64_vector("best_indices",
                                  std::move(res.best_indices));
        handle->set_scalar("n_features",
                            static_cast<double>(res.n_features));
        handle->set_scalar("n_targets",
                            static_cast<double>(res.n_targets));
        handle->set_scalar("n_components",
                            static_cast<double>(res.n_components));
        handle->set_scalar("n_iterations",
                            static_cast<double>(res.n_iterations));
        handle->set_scalar("initial_size",
                            static_cast<double>(res.initial_size));
        handle->set_scalar("min_size",
                            static_cast<double>(res.min_size));
        handle->set_scalar("max_size",
                            static_cast<double>(res.max_size));
        handle->set_scalar("top_k", static_cast<double>(res.top_k));
        handle->set_scalar("seed", static_cast<double>(res.seed));
        handle->set_scalar("best_rmse", res.best_rmse);

        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_random_frog_select");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_random_frog_select");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_scars_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t n_iterations,
    int32_t min_features,
    double sample_fraction,
    uint64_t seed,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X == nullptr || Y == nullptr ||
        plan == nullptr) {
        set_error(ctx, "null pointer in n4m_scars_select");
        return N4M_ERR_NULL_POINTER;
    }
    try {
        ::n4m::core::ScarsSelectionResult res;
        const n4m_status_t status = ::n4m::core::select_by_scars(
            *as_core(ctx), *as_core(cfg), *X, *Y, *as_core(plan),
            n_iterations, min_features, sample_fraction, seed, res);
        if (status != N4M_OK) return status;

        auto handle = std::make_unique<n4m_method_result_s>();
        const auto p = static_cast<std::int64_t>(res.n_features);
        const auto ni = static_cast<std::int64_t>(res.n_iterations);
        handle->set_double_matrix("coefficient_scores",
                                   std::move(res.coefficient_scores), ni, p);
        handle->set_double_matrix("stability_scores",
                                   std::move(res.stability_scores), 1, p);
        handle->set_double_matrix("rmse_by_iteration",
                                   std::move(res.rmse_by_iteration), 1, ni);
        handle->set_double_matrix("retained_counts",
                                   i64_to_double_vector(res.retained_counts),
                                   1, ni);
        handle->set_int64_vector("retained_counts_i64",
                                  std::move(res.retained_counts));
        handle->set_int64_vector("selected_indices",
                                  std::move(res.selected_indices));
        handle->set_scalar("n_features",
                            static_cast<double>(res.n_features));
        handle->set_scalar("n_targets",
                            static_cast<double>(res.n_targets));
        handle->set_scalar("n_components",
                            static_cast<double>(res.n_components));
        handle->set_scalar("n_iterations",
                            static_cast<double>(res.n_iterations));
        handle->set_scalar("min_features",
                            static_cast<double>(res.min_features));
        handle->set_scalar("sample_count",
                            static_cast<double>(res.sample_count));
        handle->set_scalar("sample_fraction", res.sample_fraction);
        handle->set_scalar("best_iteration",
                            static_cast<double>(res.best_iteration));
        handle->set_scalar("seed", static_cast<double>(res.seed));
        handle->set_scalar("best_rmse", res.best_rmse);

        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_scars_select");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_scars_select");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_ga_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t n_generations,
    int32_t population_size,
    int32_t min_features,
    int32_t max_features,
    double mutation_rate,
    uint64_t seed,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X == nullptr || Y == nullptr ||
        plan == nullptr) {
        set_error(ctx, "null pointer in n4m_ga_select");
        return N4M_ERR_NULL_POINTER;
    }
    try {
        ::n4m::core::GaSelectionResult res;
        const n4m_status_t status = ::n4m::core::select_by_ga(
            *as_core(ctx), *as_core(cfg), *X, *Y, *as_core(plan),
            n_generations, population_size, min_features, max_features,
            mutation_rate, seed, res);
        if (status != N4M_OK) return status;

        auto handle = std::make_unique<n4m_method_result_s>();
        const auto p = static_cast<std::int64_t>(res.n_features);
        const auto ng = static_cast<std::int64_t>(res.n_generations);
        handle->set_double_matrix("global_scores",
                                   std::move(res.global_scores), 1, p);
        handle->set_double_matrix("inclusion_frequencies",
                                   std::move(res.inclusion_frequencies), 1, p);
        handle->set_double_matrix("best_rmse_by_generation",
                                   std::move(res.best_rmse_by_generation),
                                   1, ng);
        handle->set_double_matrix("mean_rmse_by_generation",
                                   std::move(res.mean_rmse_by_generation),
                                   1, ng);
        handle->set_double_matrix("best_subset_sizes",
                                   i64_to_double_vector(res.best_subset_sizes),
                                   1, ng);
        handle->set_int64_vector("best_subset_sizes_i64",
                                  std::move(res.best_subset_sizes));
        handle->set_int64_vector("selected_indices",
                                  std::move(res.selected_indices));
        handle->set_scalar("n_features",
                            static_cast<double>(res.n_features));
        handle->set_scalar("n_targets",
                            static_cast<double>(res.n_targets));
        handle->set_scalar("n_components",
                            static_cast<double>(res.n_components));
        handle->set_scalar("n_generations",
                            static_cast<double>(res.n_generations));
        handle->set_scalar("population_size",
                            static_cast<double>(res.population_size));
        handle->set_scalar("min_features",
                            static_cast<double>(res.min_features));
        handle->set_scalar("max_features",
                            static_cast<double>(res.max_features));
        handle->set_scalar("mutation_rate", res.mutation_rate);
        handle->set_scalar("seed", static_cast<double>(res.seed));
        handle->set_scalar("best_rmse", res.best_rmse);

        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_ga_select");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_ga_select");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_pso_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t n_swarm,
    int32_t n_iterations,
    double w,
    double c1,
    double c2,
    double v_max,
    uint64_t seed,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X == nullptr || Y == nullptr ||
        plan == nullptr) {
        set_error(ctx, "null pointer in n4m_pso_select");
        return N4M_ERR_NULL_POINTER;
    }
    try {
        ::n4m::core::PsoSelectionResult res;
        const n4m_status_t status = ::n4m::core::select_by_pso(
            *as_core(ctx), *as_core(cfg), *X, *Y, *as_core(plan),
            n_swarm, n_iterations, w, c1, c2, v_max, seed, res);
        if (status != N4M_OK) return status;
        auto handle = std::make_unique<n4m_method_result_s>();
        const auto p = static_cast<std::int64_t>(res.n_features);
        const auto ni = static_cast<std::int64_t>(res.n_iterations);
        handle->set_double_matrix("inclusion_frequencies",
                                   std::move(res.inclusion_frequencies),
                                   1, p);
        handle->set_double_matrix("best_rmse_by_iteration",
                                   std::move(res.best_rmse_by_iteration),
                                   1, ni);
        handle->set_double_matrix("mean_rmse_by_iteration",
                                   std::move(res.mean_rmse_by_iteration),
                                   1, ni);
        handle->set_int64_vector("selected_indices",
                                  std::move(res.selected_indices));
        handle->set_scalar("n_features",
                            static_cast<double>(res.n_features));
        handle->set_scalar("n_targets",
                            static_cast<double>(res.n_targets));
        handle->set_scalar("n_components",
                            static_cast<double>(res.n_components));
        handle->set_scalar("n_swarm",
                            static_cast<double>(res.n_swarm));
        handle->set_scalar("n_iterations",
                            static_cast<double>(res.n_iterations));
        handle->set_scalar("w", res.w);
        handle->set_scalar("c1", res.c1);
        handle->set_scalar("c2", res.c2);
        handle->set_scalar("v_max", res.v_max);
        handle->set_scalar("seed", static_cast<double>(res.seed));
        handle->set_scalar("best_rmse", res.best_rmse);
        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_pso_select");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_pso_select");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_vissa_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t n_iterations,
    int32_t n_submodels,
    double ratio_kept,
    double threshold,
    double floor_probability,
    uint64_t seed,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X == nullptr || Y == nullptr ||
        plan == nullptr) {
        set_error(ctx, "null pointer in n4m_vissa_select");
        return N4M_ERR_NULL_POINTER;
    }
    try {
        ::n4m::core::VissaSelectionResult res;
        const n4m_status_t status = ::n4m::core::select_by_vissa(
            *as_core(ctx), *as_core(cfg), *X, *Y, *as_core(plan),
            n_iterations, n_submodels, ratio_kept, threshold,
            floor_probability, seed, res);
        if (status != N4M_OK) return status;
        auto handle = std::make_unique<n4m_method_result_s>();
        const auto p = static_cast<std::int64_t>(res.n_features);
        const auto ni = static_cast<std::int64_t>(res.n_iterations);
        handle->set_double_matrix("final_probabilities",
                                   std::move(res.final_probabilities), 1, p);
        handle->set_double_matrix("inclusion_frequencies",
                                   std::move(res.inclusion_frequencies),
                                   1, p);
        handle->set_double_matrix("best_rmse_by_iteration",
                                   std::move(res.best_rmse_by_iteration),
                                   1, ni);
        handle->set_double_matrix("mean_rmse_by_iteration",
                                   std::move(res.mean_rmse_by_iteration),
                                   1, ni);
        handle->set_double_matrix("top_k_per_iteration",
                                   std::move(res.top_k_per_iteration),
                                   ni, p);
        handle->set_int64_vector("selected_indices",
                                  std::move(res.selected_indices));
        handle->set_scalar("n_features",
                            static_cast<double>(res.n_features));
        handle->set_scalar("n_targets",
                            static_cast<double>(res.n_targets));
        handle->set_scalar("n_components",
                            static_cast<double>(res.n_components));
        handle->set_scalar("n_iterations",
                            static_cast<double>(res.n_iterations));
        handle->set_scalar("n_submodels",
                            static_cast<double>(res.n_submodels));
        handle->set_scalar("ratio_kept", res.ratio_kept);
        handle->set_scalar("threshold", res.threshold);
        handle->set_scalar("floor_probability", res.floor_probability);
        handle->set_scalar("seed", static_cast<double>(res.seed));
        handle->set_scalar("best_rmse", res.best_rmse);
        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_vissa_select");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_vissa_select");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_shaving_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t n_steps,
    int32_t min_features,
    double shave_fraction,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X == nullptr || Y == nullptr ||
        plan == nullptr) {
        set_error(ctx, "null pointer in n4m_shaving_select");
        return N4M_ERR_NULL_POINTER;
    }
    try {
        ::n4m::core::ShavingSelectionResult res;
        const n4m_status_t status = ::n4m::core::select_by_shaving(
            *as_core(ctx), *as_core(cfg), *X, *Y, *as_core(plan),
            n_steps, min_features, shave_fraction, res);
        if (status != N4M_OK) return status;

        auto handle = std::make_unique<n4m_method_result_s>();
        const auto p = static_cast<std::int64_t>(res.n_features);
        const auto ns = static_cast<std::int64_t>(res.n_steps);
        handle->set_double_matrix("coefficient_scores",
                                   std::move(res.coefficient_scores), ns, p);
        handle->set_double_matrix("rmse_by_step",
                                   std::move(res.rmse_by_step), 1, ns);
        handle->set_double_matrix("retained_counts",
                                   i64_to_double_vector(res.retained_counts),
                                   1, ns);
        handle->set_int64_vector("retained_counts_i64",
                                  std::move(res.retained_counts));
        handle->set_int64_vector("selected_indices",
                                  std::move(res.selected_indices));
        handle->set_scalar("n_features",
                            static_cast<double>(res.n_features));
        handle->set_scalar("n_targets",
                            static_cast<double>(res.n_targets));
        handle->set_scalar("n_components",
                            static_cast<double>(res.n_components));
        handle->set_scalar("n_steps", static_cast<double>(res.n_steps));
        handle->set_scalar("min_features",
                            static_cast<double>(res.min_features));
        handle->set_scalar("best_step",
                            static_cast<double>(res.best_step));
        handle->set_scalar("shave_fraction", res.shave_fraction);
        handle->set_scalar("best_rmse", res.best_rmse);

        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_shaving_select");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_shaving_select");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_bve_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t n_steps,
    int32_t min_features,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X == nullptr || Y == nullptr ||
        plan == nullptr) {
        set_error(ctx, "null pointer in n4m_bve_select");
        return N4M_ERR_NULL_POINTER;
    }
    try {
        ::n4m::core::BveSelectionResult res;
        const n4m_status_t status = ::n4m::core::select_by_bve(
            *as_core(ctx), *as_core(cfg), *X, *Y, *as_core(plan),
            n_steps, min_features, res);
        if (status != N4M_OK) return status;

        auto handle = std::make_unique<n4m_method_result_s>();
        const auto p = static_cast<std::int64_t>(res.n_features);
        const auto ns = static_cast<std::int64_t>(res.n_steps);
        handle->set_double_matrix("candidate_rmse",
                                   std::move(res.candidate_rmse), ns, p);
        handle->set_double_matrix("rmse_by_step",
                                   std::move(res.rmse_by_step), 1, ns);
        handle->set_double_matrix("retained_counts",
                                   i64_to_double_vector(res.retained_counts),
                                   1, ns);
        handle->set_int64_vector("retained_counts_i64",
                                  std::move(res.retained_counts));
        handle->set_int64_vector("removed_indices",
                                  std::move(res.removed_indices));
        handle->set_int64_vector("selected_indices",
                                  std::move(res.selected_indices));
        handle->set_scalar("n_features",
                            static_cast<double>(res.n_features));
        handle->set_scalar("n_targets",
                            static_cast<double>(res.n_targets));
        handle->set_scalar("n_components",
                            static_cast<double>(res.n_components));
        handle->set_scalar("n_steps", static_cast<double>(res.n_steps));
        handle->set_scalar("min_features",
                            static_cast<double>(res.min_features));
        handle->set_scalar("best_step",
                            static_cast<double>(res.best_step));
        handle->set_scalar("best_rmse", res.best_rmse);

        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_bve_select");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_bve_select");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_t2_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    const double* alpha_thresholds,
    int64_t n_alphas,
    int32_t min_selected,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X == nullptr || Y == nullptr ||
        plan == nullptr || alpha_thresholds == nullptr) {
        set_error(ctx, "null pointer in n4m_t2_select");
        return N4M_ERR_NULL_POINTER;
    }
    if (n_alphas <= 0) {
        set_error(ctx, "n_alphas must be >= 1");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    try {
        std::vector<double> alpha(
            alpha_thresholds,
            alpha_thresholds + static_cast<std::size_t>(n_alphas));
        ::n4m::core::T2SelectionResult res;
        const n4m_status_t status = ::n4m::core::select_by_t2(
            *as_core(ctx), *as_core(cfg), *X, *Y, *as_core(plan),
            alpha, min_selected, res);
        if (status != N4M_OK) return status;

        auto handle = std::make_unique<n4m_method_result_s>();
        const auto p = static_cast<std::int64_t>(res.n_features);
        const auto na = static_cast<std::int64_t>(res.n_alphas);
        handle->set_double_matrix("t2_scores",
                                   std::move(res.t2_scores), 1, p);
        handle->set_double_matrix("ucl_by_alpha",
                                   std::move(res.ucl_by_alpha), 1, na);
        handle->set_double_matrix("rmse_by_alpha",
                                   std::move(res.rmse_by_alpha), 1, na);
        handle->set_double_matrix("selected_counts",
                                   i64_to_double_vector(res.selected_counts),
                                   1, na);
        handle->set_int64_vector("selected_counts_i64",
                                  std::move(res.selected_counts));
        handle->set_int64_vector("selected_mask",
                                  std::move(res.selected_mask));
        handle->set_int64_vector("selected_indices_best_error",
                                  std::move(res.selected_indices_best_error));
        handle->set_int64_vector("selected_indices_min_set",
                                  std::move(res.selected_indices_min_set));
        // Convenience alias: parity-gate default key is selected_indices →
        // use the "best_error" subset (lowest RMSE), matching the README.
        handle->set_int64_vector("selected_indices",
                                  handle->int64_arrays.at(
                                      "selected_indices_best_error"));
        handle->set_scalar("n_features",
                            static_cast<double>(res.n_features));
        handle->set_scalar("n_targets",
                            static_cast<double>(res.n_targets));
        handle->set_scalar("n_components",
                            static_cast<double>(res.n_components));
        handle->set_scalar("n_alphas",
                            static_cast<double>(res.n_alphas));
        handle->set_scalar("min_selected",
                            static_cast<double>(res.min_selected));
        handle->set_scalar("best_error_index",
                            static_cast<double>(res.best_error_index));
        handle->set_scalar("min_set_index",
                            static_cast<double>(res.min_set_index));
        handle->set_scalar("best_rmse", res.best_rmse);
        handle->set_scalar("min_set_rmse", res.min_set_rmse);

        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_t2_select");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_t2_select");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_wvc_select(
    n4m_context_t* ctx,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    int32_t n_components,
    int32_t top_k,
    int32_t normalize,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || X == nullptr || Y == nullptr) {
        set_error(ctx, "null pointer in n4m_wvc_select");
        return N4M_ERR_NULL_POINTER;
    }
    try {
        ::n4m::core::WvcSelectionResult res;
        const n4m_status_t status = ::n4m::core::select_by_wvc(
            *as_core(ctx), *X, *Y, n_components, top_k,
            normalize != 0, res);
        if (status != N4M_OK) return status;

        auto handle = std::make_unique<n4m_method_result_s>();
        const auto p = static_cast<std::int64_t>(res.n_features);
        const auto q = static_cast<std::int64_t>(res.n_targets);
        const auto k = static_cast<std::int64_t>(res.n_components);
        handle->set_double_matrix("weights_w",
                                   std::move(res.weights_w), p, k);
        handle->set_double_matrix("loadings_p",
                                   std::move(res.loadings_p), p, k);
        handle->set_double_matrix("y_loadings_q",
                                   std::move(res.y_loadings_q), q, k);
        handle->set_double_matrix("wvc_scores",
                                   std::move(res.wvc_scores), 1, p);
        handle->set_double_matrix("final_scores",
                                   std::move(res.final_scores), 1, p);
        handle->set_int64_vector("selected_indices",
                                  std::move(res.selected_indices));
        handle->set_scalar("n_features",
                            static_cast<double>(res.n_features));
        handle->set_scalar("n_targets",
                            static_cast<double>(res.n_targets));
        handle->set_scalar("n_components",
                            static_cast<double>(res.n_components));
        handle->set_scalar("top_k", static_cast<double>(res.top_k));
        handle->set_scalar("normalize",
                            static_cast<double>(res.normalize));

        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_wvc_select");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_wvc_select");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_wvc_threshold_select(
    n4m_context_t* ctx,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    int32_t n_components,
    int32_t normalize,
    double score_threshold,
    double threshold_factor,
    int32_t min_selected,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || X == nullptr || Y == nullptr) {
        set_error(ctx, "null pointer in n4m_wvc_threshold_select");
        return N4M_ERR_NULL_POINTER;
    }
    try {
        ::n4m::core::WvcThresholdSelectionResult res;
        const n4m_status_t status =
            ::n4m::core::select_by_wvc_threshold(
                *as_core(ctx), *X, *Y, n_components, normalize != 0,
                score_threshold, threshold_factor, min_selected, res);
        if (status != N4M_OK) return status;

        auto handle = std::make_unique<n4m_method_result_s>();
        const auto p = static_cast<std::int64_t>(res.n_features);
        handle->set_double_matrix("final_scores",
                                   std::move(res.final_scores), 1, p);
        handle->set_int64_vector("ranked_indices",
                                  std::move(res.ranked_indices));
        handle->set_int64_vector("selected_indices",
                                  std::move(res.selected_indices));
        handle->set_scalar("n_features",
                            static_cast<double>(res.n_features));
        handle->set_scalar("n_targets",
                            static_cast<double>(res.n_targets));
        handle->set_scalar("n_components",
                            static_cast<double>(res.n_components));
        handle->set_scalar("min_selected",
                            static_cast<double>(res.min_selected));
        handle->set_scalar("normalize",
                            static_cast<double>(res.normalize));
        handle->set_scalar("score_threshold", res.score_threshold);
        handle->set_scalar("threshold_factor", res.threshold_factor);
        handle->set_scalar("mean_score", res.mean_score);
        handle->set_scalar("effective_threshold", res.effective_threshold);

        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_wvc_threshold_select");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_wvc_threshold_select");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_emcuve_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t noise_features,
    uint64_t noise_seed,
    int32_t n_ensembles,
    double vote_threshold,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X == nullptr || Y == nullptr ||
        plan == nullptr) {
        set_error(ctx, "null pointer in n4m_emcuve_select");
        return N4M_ERR_NULL_POINTER;
    }
    try {
        ::n4m::core::EmcuveSelectionResult res;
        const n4m_status_t status = ::n4m::core::select_by_emcuve(
            *as_core(ctx), *as_core(cfg), *X, *Y, *as_core(plan),
            noise_features, noise_seed, n_ensembles, vote_threshold, res);
        if (status != N4M_OK) return status;

        auto handle = std::make_unique<n4m_method_result_s>();
        const auto p = static_cast<std::int64_t>(res.n_features);
        const auto ne = static_cast<std::int64_t>(res.n_ensembles);
        handle->set_double_matrix("mean_real_stability_scores",
                                   std::move(res.mean_real_stability_scores),
                                   1, p);
        handle->set_double_matrix("vote_frequencies",
                                   std::move(res.vote_frequencies), 1, p);
        handle->set_double_matrix("noise_thresholds",
                                   std::move(res.noise_thresholds), 1, ne);
        handle->set_int64_vector("selected_indices",
                                  std::move(res.selected_indices));
        handle->set_scalar("n_features",
                            static_cast<double>(res.n_features));
        handle->set_scalar("n_targets",
                            static_cast<double>(res.n_targets));
        handle->set_scalar("n_repeats",
                            static_cast<double>(res.n_repeats));
        handle->set_scalar("n_noise_features",
                            static_cast<double>(res.n_noise_features));
        handle->set_scalar("n_ensembles",
                            static_cast<double>(res.n_ensembles));
        handle->set_scalar("selected_count",
                            static_cast<double>(res.selected_count));
        handle->set_scalar("noise_seed",
                            static_cast<double>(res.noise_seed));
        handle->set_scalar("vote_threshold", res.vote_threshold);

        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_emcuve_select");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_emcuve_select");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_randomization_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    int32_t n_permutations,
    uint64_t randomization_seed,
    double alpha,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X == nullptr || Y == nullptr) {
        set_error(ctx, "null pointer in n4m_randomization_select");
        return N4M_ERR_NULL_POINTER;
    }
    try {
        ::n4m::core::RandomizationSelectionResult res;
        const n4m_status_t status =
            ::n4m::core::select_by_randomization_test(
                *as_core(ctx), *as_core(cfg), *X, *Y,
                n_permutations, randomization_seed, alpha, res);
        if (status != N4M_OK) return status;

        auto handle = std::make_unique<n4m_method_result_s>();
        const auto p = static_cast<std::int64_t>(res.n_features);
        handle->set_double_matrix("observed_scores",
                                   std::move(res.observed_scores), 1, p);
        handle->set_double_matrix("p_values",
                                   std::move(res.p_values), 1, p);
        handle->set_double_matrix("exceedance_counts",
                                   i64_to_double_vector(res.exceedance_counts),
                                   1, p);
        handle->set_int64_vector("exceedance_counts_i64",
                                  std::move(res.exceedance_counts));
        handle->set_int64_vector("selected_indices",
                                  std::move(res.selected_indices));
        handle->set_scalar("n_features",
                            static_cast<double>(res.n_features));
        handle->set_scalar("n_targets",
                            static_cast<double>(res.n_targets));
        handle->set_scalar("n_permutations",
                            static_cast<double>(res.n_permutations));
        handle->set_scalar("selected_count",
                            static_cast<double>(res.selected_count));
        handle->set_scalar("randomization_seed",
                            static_cast<double>(res.randomization_seed));
        handle->set_scalar("alpha", res.alpha);

        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_randomization_select");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_randomization_select");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_bipls_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t interval_width,
    int32_t min_intervals,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X == nullptr || Y == nullptr ||
        plan == nullptr) {
        set_error(ctx, "null pointer in n4m_bipls_select");
        return N4M_ERR_NULL_POINTER;
    }
    try {
        ::n4m::core::BiplsSelectionResult res;
        const n4m_status_t status = ::n4m::core::select_by_bipls(
            *as_core(ctx), *as_core(cfg), *X, *Y, *as_core(plan),
            interval_width, min_intervals, res);
        if (status != N4M_OK) return status;

        auto handle = std::make_unique<n4m_method_result_s>();
        const auto path_len =
            static_cast<std::int64_t>(res.rmse_path.size());
        handle->set_int64_vector("intervals", std::move(res.intervals));
        handle->set_int64_vector("removed_intervals",
                                  std::move(res.removed_intervals));
        handle->set_double_matrix("active_counts",
                                   i64_to_double_vector(res.active_counts),
                                   1, path_len);
        handle->set_int64_vector("active_counts_i64",
                                  std::move(res.active_counts));
        handle->set_double_matrix("rmse_path",
                                   std::move(res.rmse_path), 1, path_len);
        handle->set_int64_vector("selected_interval_indices",
                                  std::move(res.selected_interval_indices));
        handle->set_int64_vector("selected_feature_indices",
                                  std::move(res.selected_feature_indices));
        // Alias for the parity-gate default key.
        handle->set_int64_vector("selected_indices",
                                  handle->int64_arrays.at(
                                      "selected_feature_indices"));
        handle->set_scalar("n_intervals",
                            static_cast<double>(res.n_intervals));
        handle->set_scalar("interval_width",
                            static_cast<double>(res.interval_width));
        handle->set_scalar("min_intervals",
                            static_cast<double>(res.min_intervals));
        handle->set_scalar("best_step",
                            static_cast<double>(res.best_step));

        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_bipls_select");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_bipls_select");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_sipls_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t interval_width,
    int32_t combination_size,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X == nullptr || Y == nullptr ||
        plan == nullptr) {
        set_error(ctx, "null pointer in n4m_sipls_select");
        return N4M_ERR_NULL_POINTER;
    }
    try {
        ::n4m::core::SiplsSelectionResult res;
        const n4m_status_t status = ::n4m::core::select_by_sipls(
            *as_core(ctx), *as_core(cfg), *X, *Y, *as_core(plan),
            interval_width, combination_size, res);
        if (status != N4M_OK) return status;

        auto handle = std::make_unique<n4m_method_result_s>();
        const auto nc =
            static_cast<std::int64_t>(res.rmse_by_combination.size());
        handle->set_int64_vector("intervals", std::move(res.intervals));
        handle->set_int64_vector("candidate_interval_indices",
                                  std::move(res.candidate_interval_indices));
        handle->set_double_matrix("rmse_by_combination",
                                   std::move(res.rmse_by_combination), 1, nc);
        handle->set_int64_vector("selected_interval_indices",
                                  std::move(res.selected_interval_indices));
        handle->set_int64_vector("selected_feature_indices",
                                  std::move(res.selected_feature_indices));
        handle->set_int64_vector("selected_indices",
                                  handle->int64_arrays.at(
                                      "selected_feature_indices"));
        handle->set_scalar("n_intervals",
                            static_cast<double>(res.n_intervals));
        handle->set_scalar("interval_width",
                            static_cast<double>(res.interval_width));
        handle->set_scalar("combination_size",
                            static_cast<double>(res.combination_size));
        handle->set_scalar("n_combinations",
                            static_cast<double>(res.n_combinations));
        handle->set_scalar("best_combination_index",
                            static_cast<double>(res.best_combination_index));

        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_sipls_select");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_sipls_select");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_rep_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t n_steps,
    int32_t min_features,
    int32_t remove_count,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X == nullptr || Y == nullptr ||
        plan == nullptr) {
        set_error(ctx, "null pointer in n4m_rep_select");
        return N4M_ERR_NULL_POINTER;
    }
    try {
        ::n4m::core::RepSelectionResult res;
        const n4m_status_t status = ::n4m::core::select_by_rep(
            *as_core(ctx), *as_core(cfg), *X, *Y, *as_core(plan),
            n_steps, min_features, remove_count, res);
        if (status != N4M_OK) return status;

        auto handle = std::make_unique<n4m_method_result_s>();
        const auto p = static_cast<std::int64_t>(res.n_features);
        const auto ns = static_cast<std::int64_t>(res.n_steps);
        handle->set_double_matrix("coefficient_scores",
                                   std::move(res.coefficient_scores), ns, p);
        handle->set_double_matrix("rmse_by_step",
                                   std::move(res.rmse_by_step), 1, ns);
        handle->set_double_matrix("retained_counts",
                                   i64_to_double_vector(res.retained_counts),
                                   1, ns);
        handle->set_double_matrix("removed_counts",
                                   i64_to_double_vector(res.removed_counts),
                                   1, ns);
        handle->set_int64_vector("retained_counts_i64",
                                  std::move(res.retained_counts));
        handle->set_int64_vector("removed_counts_i64",
                                  std::move(res.removed_counts));
        handle->set_int64_vector("removed_indices",
                                  std::move(res.removed_indices));
        handle->set_int64_vector("selected_indices",
                                  std::move(res.selected_indices));
        handle->set_scalar("n_features",
                            static_cast<double>(res.n_features));
        handle->set_scalar("n_targets",
                            static_cast<double>(res.n_targets));
        handle->set_scalar("n_components",
                            static_cast<double>(res.n_components));
        handle->set_scalar("n_steps", static_cast<double>(res.n_steps));
        handle->set_scalar("min_features",
                            static_cast<double>(res.min_features));
        handle->set_scalar("remove_count",
                            static_cast<double>(res.remove_count));
        handle->set_scalar("best_step",
                            static_cast<double>(res.best_step));
        handle->set_scalar("best_rmse", res.best_rmse);

        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_rep_select");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_rep_select");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_ipw_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t n_iterations,
    int32_t top_k,
    double damping,
    double weight_floor,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X == nullptr || Y == nullptr ||
        plan == nullptr) {
        set_error(ctx, "null pointer in n4m_ipw_select");
        return N4M_ERR_NULL_POINTER;
    }
    try {
        ::n4m::core::IpwSelectionResult res;
        const n4m_status_t status = ::n4m::core::select_by_ipw(
            *as_core(ctx), *as_core(cfg), *X, *Y, *as_core(plan),
            n_iterations, top_k, damping, weight_floor, res);
        if (status != N4M_OK) return status;

        auto handle = std::make_unique<n4m_method_result_s>();
        const auto p = static_cast<std::int64_t>(res.n_features);
        const auto ni = static_cast<std::int64_t>(res.n_iterations);
        const auto k = static_cast<std::int64_t>(res.top_k);
        handle->set_double_matrix("score_path",
                                   std::move(res.score_path), ni, p);
        handle->set_double_matrix("weight_path",
                                   std::move(res.weight_path), ni, p);
        handle->set_double_matrix("rmse_by_iteration",
                                   std::move(res.rmse_by_iteration), 1, ni);
        handle->set_int64_vector("selected_by_iteration",
                                  std::move(res.selected_by_iteration));
        handle->set_double_matrix(
            "selected_by_iteration_d",
            i64_to_double_vector(handle->int64_arrays.at(
                "selected_by_iteration")),
            ni, k);
        handle->set_int64_vector("ranking_indices",
                                  std::move(res.ranking_indices));
        handle->set_int64_vector("selected_indices",
                                  std::move(res.selected_indices));
        handle->set_scalar("n_features",
                            static_cast<double>(res.n_features));
        handle->set_scalar("n_targets",
                            static_cast<double>(res.n_targets));
        handle->set_scalar("n_components",
                            static_cast<double>(res.n_components));
        handle->set_scalar("n_iterations",
                            static_cast<double>(res.n_iterations));
        handle->set_scalar("top_k", static_cast<double>(res.top_k));
        handle->set_scalar("best_iteration",
                            static_cast<double>(res.best_iteration));
        handle->set_scalar("damping", res.damping);
        handle->set_scalar("weight_floor", res.weight_floor);
        handle->set_scalar("best_rmse", res.best_rmse);

        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_ipw_select");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_ipw_select");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_st_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    const double* thresholds,
    int64_t n_thresholds,
    int32_t min_selected,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X == nullptr || Y == nullptr ||
        plan == nullptr || thresholds == nullptr) {
        set_error(ctx, "null pointer in n4m_st_select");
        return N4M_ERR_NULL_POINTER;
    }
    if (n_thresholds <= 0) {
        set_error(ctx, "n_thresholds must be >= 1");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    try {
        std::vector<double> thr(
            thresholds,
            thresholds + static_cast<std::size_t>(n_thresholds));
        ::n4m::core::StSelectionResult res;
        const n4m_status_t status = ::n4m::core::select_by_st(
            *as_core(ctx), *as_core(cfg), *X, *Y, *as_core(plan),
            thr, min_selected, res);
        if (status != N4M_OK) return status;

        auto handle = std::make_unique<n4m_method_result_s>();
        const auto p = static_cast<std::int64_t>(res.n_features);
        const auto nt = static_cast<std::int64_t>(res.n_thresholds);
        handle->set_double_matrix("coefficient_scores",
                                   std::move(res.coefficient_scores), 1, p);
        handle->set_double_matrix("thresholds",
                                   std::move(res.thresholds), 1, nt);
        handle->set_double_matrix("rmse_by_threshold",
                                   std::move(res.rmse_by_threshold), 1, nt);
        handle->set_double_matrix("selected_counts",
                                   i64_to_double_vector(res.selected_counts),
                                   1, nt);
        handle->set_int64_vector("selected_counts_i64",
                                  std::move(res.selected_counts));
        handle->set_int64_vector("selected_masks",
                                  std::move(res.selected_masks));
        handle->set_int64_vector("ranking_indices",
                                  std::move(res.ranking_indices));
        handle->set_int64_vector("selected_indices",
                                  std::move(res.selected_indices));
        handle->set_scalar("n_features",
                            static_cast<double>(res.n_features));
        handle->set_scalar("n_targets",
                            static_cast<double>(res.n_targets));
        handle->set_scalar("n_components",
                            static_cast<double>(res.n_components));
        handle->set_scalar("n_thresholds",
                            static_cast<double>(res.n_thresholds));
        handle->set_scalar("min_selected",
                            static_cast<double>(res.min_selected));
        handle->set_scalar("best_threshold_index",
                            static_cast<double>(res.best_threshold_index));
        handle->set_scalar("best_threshold", res.best_threshold);
        handle->set_scalar("best_rmse", res.best_rmse);

        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_st_select");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_st_select");
        return N4M_ERR_INTERNAL;
    }
}

/* ---------------- §19 Phase 50+ numerical methods (ECR / IRIV / IRF) ----- */

N4M_API n4m_status_t n4m_ecr_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    double alpha,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X == nullptr || Y == nullptr) {
        set_error(ctx, "null pointer in n4m_ecr_fit");
        return N4M_ERR_NULL_POINTER;
    }
    try {
        ::n4m::core::EcrResult res;
        const n4m_status_t status = ::n4m::core::fit_ecr(
            *as_core(ctx), *as_core(cfg), *X, *Y, alpha, res);
        if (status != N4M_OK) return status;

        auto handle = std::make_unique<n4m_method_result_s>();
        const auto n = static_cast<std::int64_t>(res.n_samples);
        const auto p = static_cast<std::int64_t>(res.n_features);
        const auto q = static_cast<std::int64_t>(res.n_targets);
        const auto a = static_cast<std::int64_t>(res.n_components);
        handle->set_double_matrix("coefficients",
                                   std::move(res.coefficients), p, q);
        handle->set_double_matrix("predictions",
                                   std::move(res.predictions), n, q);
        handle->set_double_matrix("x_mean", std::move(res.x_mean), 1, p);
        handle->set_double_matrix("x_scale", std::move(res.x_scale), 1, p);
        handle->set_double_matrix("y_mean", std::move(res.y_mean), 1, q);
        handle->set_double_matrix("y_scale", std::move(res.y_scale), 1, q);
        handle->set_double_matrix("weights_w",
                                   std::move(res.weights_w), p, a);
        handle->set_double_matrix("loadings_p",
                                   std::move(res.loadings_p), p, a);
        handle->set_double_matrix("y_loadings",
                                   std::move(res.y_loadings), q, a);
        handle->set_double_matrix("wstar", std::move(res.wstar), p, a);
        handle->set_double_matrix("r2x", std::move(res.r2x), 1, a);
        handle->set_double_matrix("r2y", std::move(res.r2y), 1, a);
        handle->set_scalar("n_samples", static_cast<double>(res.n_samples));
        handle->set_scalar("n_features", static_cast<double>(res.n_features));
        handle->set_scalar("n_targets", static_cast<double>(res.n_targets));
        handle->set_scalar("n_components",
                            static_cast<double>(res.n_components));
        handle->set_scalar("alpha", res.alpha);
        handle->set_scalar("rmse", res.rmse);

        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_ecr_fit");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_ecr_fit");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_iriv_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t max_rounds,
    uint64_t seed,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X == nullptr || Y == nullptr ||
        plan == nullptr) {
        set_error(ctx, "null pointer in n4m_iriv_select");
        return N4M_ERR_NULL_POINTER;
    }
    try {
        ::n4m::core::IrivSelectionResult res;
        const n4m_status_t status = ::n4m::core::select_by_iriv(
            *as_core(ctx), *as_core(cfg), *X, *Y, *as_core(plan),
            max_rounds, seed, res);
        if (status != N4M_OK) return status;

        auto handle = std::make_unique<n4m_method_result_s>();
        const std::int64_t n_rem =
            static_cast<std::int64_t>(res.remaining_per_round.size());
        const std::int64_t n_rmv =
            static_cast<std::int64_t>(res.removed_per_round.size());
        handle->set_double_matrix(
            "remaining_per_round",
            i64_to_double_vector(res.remaining_per_round), 1, n_rem);
        handle->set_int64_vector("remaining_per_round_i64",
                                  std::move(res.remaining_per_round));
        handle->set_double_matrix(
            "removed_per_round",
            i64_to_double_vector(res.removed_per_round), 1, n_rmv);
        handle->set_int64_vector("removed_per_round_i64",
                                  std::move(res.removed_per_round));
        handle->set_int64_vector("selected_indices",
                                  std::move(res.selected_indices));
        handle->set_scalar("n_features",
                            static_cast<double>(res.n_features));
        handle->set_scalar("n_targets",
                            static_cast<double>(res.n_targets));
        handle->set_scalar("n_components",
                            static_cast<double>(res.n_components));
        handle->set_scalar("n_rounds",
                            static_cast<double>(res.n_rounds));
        handle->set_scalar("binary_matrix_rows",
                            static_cast<double>(res.binary_matrix_rows));
        handle->set_scalar("seed", static_cast<double>(res.seed));

        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_iriv_select");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_iriv_select");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_irf_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t n_iterations,
    int32_t window_size,
    int32_t initial_intervals,
    int32_t top_k,
    uint64_t seed,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X == nullptr || Y == nullptr ||
        plan == nullptr) {
        set_error(ctx, "null pointer in n4m_irf_select");
        return N4M_ERR_NULL_POINTER;
    }
    try {
        ::n4m::core::IrfSelectionResult res;
        const n4m_status_t status = ::n4m::core::select_by_irf(
            *as_core(ctx), *as_core(cfg), *X, *Y, *as_core(plan),
            n_iterations, window_size, initial_intervals, top_k, seed, res);
        if (status != N4M_OK) return status;

        auto handle = std::make_unique<n4m_method_result_s>();
        const std::int64_t n_int =
            static_cast<std::int64_t>(res.n_intervals);
        const std::int64_t n_it =
            static_cast<std::int64_t>(res.n_iterations);
        handle->set_double_matrix("probability",
                                   std::move(res.probability), 1, n_int);
        handle->set_double_matrix("rmse_by_iteration",
                                   std::move(res.rmse_by_iteration), 1, n_it);
        handle->set_double_matrix(
            "subset_sizes",
            i64_to_double_vector(res.subset_sizes), 1, n_it);
        handle->set_int64_vector("subset_sizes_i64",
                                  std::move(res.subset_sizes));
        handle->set_int64_vector("top_k_intervals",
                                  std::move(res.top_k_intervals));
        handle->set_int64_vector("selected_indices",
                                  std::move(res.selected_indices));
        handle->set_scalar("n_features",
                            static_cast<double>(res.n_features));
        handle->set_scalar("n_targets",
                            static_cast<double>(res.n_targets));
        handle->set_scalar("n_components",
                            static_cast<double>(res.n_components));
        handle->set_scalar("n_iterations",
                            static_cast<double>(res.n_iterations));
        handle->set_scalar("window_size",
                            static_cast<double>(res.window_size));
        handle->set_scalar("initial_intervals",
                            static_cast<double>(res.initial_intervals));
        handle->set_scalar("n_intervals",
                            static_cast<double>(res.n_intervals));
        handle->set_scalar("top_k", static_cast<double>(res.top_k));
        handle->set_scalar("seed", static_cast<double>(res.seed));
        handle->set_scalar("best_rmse", res.best_rmse);

        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_irf_select");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_irf_select");
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_vip_spa_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    double vip_threshold,
    int32_t top_k,
    n4m_method_result_t** out_result) {
    if (out_result == nullptr) return N4M_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X == nullptr || Y == nullptr) {
        set_error(ctx, "null pointer in n4m_vip_spa_select");
        return N4M_ERR_NULL_POINTER;
    }
    try {
        ::n4m::core::VipSpaSelectionResult res;
        const n4m_status_t status = ::n4m::core::select_by_vip_spa(
            *as_core(ctx), *as_core(cfg), *X, *Y, vip_threshold, top_k, res);
        if (status != N4M_OK) return status;

        auto handle = std::make_unique<n4m_method_result_s>();
        const auto p = static_cast<std::int64_t>(res.n_features);
        const auto k = static_cast<std::int64_t>(res.selected_indices.size());
        std::vector<double> vip_mask_doubles;
        vip_mask_doubles.reserve(res.vip_mask.size());
        for (const auto bit : res.vip_mask) {
            vip_mask_doubles.push_back(static_cast<double>(bit));
        }
        handle->set_double_matrix("vip_scores",
                                   std::move(res.vip_scores), 1, p);
        handle->set_double_matrix("vip_mask",
                                   std::move(vip_mask_doubles), 1, p);
        handle->set_double_matrix("selection_scores",
                                   std::move(res.selection_scores), 1, k);
        handle->set_int64_vector("selected_indices",
                                  std::move(res.selected_indices));
        handle->set_scalar("n_features",
                            static_cast<double>(res.n_features));
        handle->set_scalar("n_targets",
                            static_cast<double>(res.n_targets));
        handle->set_scalar("n_components",
                            static_cast<double>(res.n_components));
        handle->set_scalar("top_k", static_cast<double>(res.top_k));
        handle->set_scalar("n_selected",
                            static_cast<double>(res.n_selected));
        handle->set_scalar("n_candidates",
                            static_cast<double>(res.n_candidates));
        handle->set_scalar("vip_threshold", res.vip_threshold);

        *out_result = handle.release();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in n4m_vip_spa_select");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in n4m_vip_spa_select");
        return N4M_ERR_INTERNAL;
    }
}

}  // extern "C"
