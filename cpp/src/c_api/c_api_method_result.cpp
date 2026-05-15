// SPDX-License-Identifier: CeCILL-2.1
//
// extern "C" wrappers for p4a_method_result_t plus the universal fit
// functions that build one. Each fit function instantiates the internal
// result struct from core/extra_pls.hpp or core/model.hpp, then packs the
// relevant outputs into a p4a_method_result_t for transit across the C
// ABI boundary.

#include <stddef.h>
#include <stdint.h>

#include <cmath>
#include <memory>
#include <new>
#include <string>
#include <vector>

#include "pls4all/p4a.h"

#include "core/config.hpp"
#include "core/context.hpp"
#include "core/extra_pls.hpp"
#include "core/kernel_pls.hpp"
#include "core/method_result.hpp"
#include "core/model.hpp"
#include "core/recursive_pls.hpp"
#include "core/tensor_pls.hpp"

namespace {

inline ::pls4all::core::Context* as_core(p4a_context_t* ctx) noexcept {
    return static_cast<::pls4all::core::Context*>(ctx);
}
inline const ::pls4all::core::Config* as_core(const p4a_config_t* cfg) noexcept {
    return static_cast<const ::pls4all::core::Config*>(cfg);
}

void set_error(p4a_context_t* ctx, const char* msg) noexcept {
    if (ctx == nullptr) return;
    try {
        as_core(ctx)->set_error(msg);
    } catch (...) {
    }
}

void copy_predictions(const ::pls4all::core::Model& model,
                      const p4a_matrix_view_t& X,
                      ::pls4all::core::Context& ctx,
                      std::vector<double>& out,
                      std::int64_t& out_rows,
                      std::int64_t& out_cols) {
    const std::size_t n = static_cast<std::size_t>(X.rows);
    const std::size_t q = static_cast<std::size_t>(model.n_targets);
    out.assign(n * q, 0.0);
    p4a_matrix_view_t pred_view{};
    pred_view.data = out.data();
    pred_view.rows = static_cast<std::int64_t>(n);
    pred_view.cols = static_cast<std::int64_t>(q);
    pred_view.row_stride = static_cast<std::int64_t>(q > 0 ? q : 1);
    pred_view.col_stride = 1;
    pred_view.dtype = P4A_DTYPE_F64;
    const p4a_status_t status =
        ::pls4all::core::predict_into(ctx, model, X, pred_view);
    (void)status;  // predictions are best-effort; caller checks rmse / status
    out_rows = static_cast<std::int64_t>(n);
    out_cols = static_cast<std::int64_t>(q);
}

double in_sample_rmse(const std::vector<double>& predictions,
                       const p4a_matrix_view_t& Y) {
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
void predict_from_coefficients(const p4a_matrix_view_t& X,
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

P4A_API void p4a_method_result_destroy(p4a_method_result_t* result) {
    try {
        delete result;
    } catch (...) {
    }
}

P4A_API p4a_status_t p4a_method_result_get_double_matrix(
    const p4a_method_result_t* result,
    const char* name,
    const double** out_data, int64_t* out_rows, int64_t* out_cols) {
    if (result == nullptr || name == nullptr || out_data == nullptr ||
        out_rows == nullptr || out_cols == nullptr) {
        return P4A_ERR_NULL_POINTER;
    }
    try {
        const auto array_iter = result->double_arrays.find(name);
        if (array_iter == result->double_arrays.end()) {
            return P4A_ERR_INVALID_ARGUMENT;
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
        return P4A_OK;
    } catch (...) {
        return P4A_ERR_INTERNAL;
    }
}

P4A_API p4a_status_t p4a_method_result_get_int_vector(
    const p4a_method_result_t* result,
    const char* name,
    const int32_t** out_data, int32_t* out_size) {
    if (result == nullptr || name == nullptr || out_data == nullptr ||
        out_size == nullptr) {
        return P4A_ERR_NULL_POINTER;
    }
    try {
        const auto iter = result->int_arrays.find(name);
        if (iter == result->int_arrays.end()) {
            return P4A_ERR_INVALID_ARGUMENT;
        }
        const auto& values = iter->second;
        *out_data = values.empty() ? nullptr : values.data();
        *out_size = static_cast<int32_t>(values.size());
        return P4A_OK;
    } catch (...) {
        return P4A_ERR_INTERNAL;
    }
}

P4A_API p4a_status_t p4a_method_result_get_scalar(
    const p4a_method_result_t* result,
    const char* name,
    double* out_value) {
    if (result == nullptr || name == nullptr || out_value == nullptr) {
        return P4A_ERR_NULL_POINTER;
    }
    try {
        const auto iter = result->scalars.find(name);
        if (iter == result->scalars.end()) {
            return P4A_ERR_INVALID_ARGUMENT;
        }
        *out_value = iter->second;
        return P4A_OK;
    } catch (...) {
        return P4A_ERR_INTERNAL;
    }
}

/* ---- fit entry points ---- */

P4A_API p4a_status_t p4a_sparse_simpls_fit(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    double sparsity_lambda,
    p4a_method_result_t** out_result) {
    if (out_result == nullptr) return P4A_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X == nullptr || Y == nullptr) {
        set_error(ctx, "null pointer in p4a_sparse_simpls_fit");
        return P4A_ERR_NULL_POINTER;
    }

    try {
        ::pls4all::core::Config local = *as_core(cfg);
        local.algorithm = P4A_ALGO_SPARSE_PLS;
        local.solver = P4A_SOLVER_SIMPLS;
        local.deflation = P4A_DEFLATION_REGRESSION;
        local.sparsity_lambda = sparsity_lambda;
        std::unique_ptr<::pls4all::core::Model> model;
        const p4a_status_t status = ::pls4all::core::fit_pls_sparse_simpls(
            *as_core(ctx), local, *X, *Y, model);
        if (status != P4A_OK) return status;

        auto handle = std::make_unique<p4a_method_result_s>();
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
        return P4A_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in p4a_sparse_simpls_fit");
        return P4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in p4a_sparse_simpls_fit");
        return P4A_ERR_INTERNAL;
    }
}

P4A_API p4a_status_t p4a_di_pls_fit(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X_source,
    const p4a_matrix_view_t* Y_source,
    const p4a_matrix_view_t* X_target,
    double di_lambda,
    p4a_method_result_t** out_result) {
    if (out_result == nullptr) return P4A_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X_source == nullptr ||
        Y_source == nullptr || X_target == nullptr) {
        set_error(ctx, "null pointer in p4a_di_pls_fit");
        return P4A_ERR_NULL_POINTER;
    }

    try {
        ::pls4all::core::Config local = *as_core(cfg);
        local.algorithm = P4A_ALGO_PLS_REGRESSION;
        local.solver = P4A_SOLVER_SIMPLS;
        local.deflation = P4A_DEFLATION_REGRESSION;
        local.di_lambda = di_lambda;
        std::unique_ptr<::pls4all::core::Model> model;
        const p4a_status_t status = ::pls4all::core::fit_di_pls(
            *as_core(ctx), local, *X_source, *Y_source, *X_target, model);
        if (status != P4A_OK) return status;

        auto handle = std::make_unique<p4a_method_result_s>();
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
        return P4A_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in p4a_di_pls_fit");
        return P4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in p4a_di_pls_fit");
        return P4A_ERR_INTERNAL;
    }
}

P4A_API p4a_status_t p4a_recursive_pls_run(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    int32_t window_size,
    p4a_method_result_t** out_result) {
    if (out_result == nullptr) return P4A_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X == nullptr || Y == nullptr) {
        set_error(ctx, "null pointer in p4a_recursive_pls_run");
        return P4A_ERR_NULL_POINTER;
    }

    try {
        ::pls4all::core::RecursivePlsResult result;
        const p4a_status_t status = ::pls4all::core::fit_predict_recursive_pls(
            *as_core(ctx), *as_core(cfg), *X, *Y, window_size, result);
        if (status != P4A_OK) return status;

        auto handle = std::make_unique<p4a_method_result_s>();
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
        return P4A_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in p4a_recursive_pls_run");
        return P4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in p4a_recursive_pls_run");
        return P4A_ERR_INTERNAL;
    }
}

namespace {

void pack_weighted_result(p4a_method_result_s& handle,
                           const ::pls4all::core::WeightedPlsResult& res,
                           const p4a_matrix_view_t& X,
                           const p4a_matrix_view_t& Y) {
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

P4A_API p4a_status_t p4a_cppls_fit(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    double gamma,
    p4a_method_result_t** out_result) {
    if (out_result == nullptr) return P4A_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X == nullptr || Y == nullptr) {
        set_error(ctx, "null pointer in p4a_cppls_fit");
        return P4A_ERR_NULL_POINTER;
    }
    try {
        ::pls4all::core::CpplsResult res;
        const p4a_status_t status = ::pls4all::core::fit_cppls(
            *as_core(ctx), *as_core(cfg), *X, *Y, gamma, res);
        if (status != P4A_OK) return status;

        auto handle = std::make_unique<p4a_method_result_s>();
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
        return P4A_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in p4a_cppls_fit");
        return P4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in p4a_cppls_fit");
        return P4A_ERR_INTERNAL;
    }
}

P4A_API p4a_status_t p4a_weighted_pls_fit(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    const double* sample_weights,
    int64_t sample_weights_size,
    p4a_method_result_t** out_result) {
    if (out_result == nullptr) return P4A_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X == nullptr || Y == nullptr ||
        sample_weights == nullptr) {
        set_error(ctx, "null pointer in p4a_weighted_pls_fit");
        return P4A_ERR_NULL_POINTER;
    }
    if (sample_weights_size != X->rows) {
        set_error(ctx, "sample_weights_size must equal X.rows");
        return P4A_ERR_SHAPE_MISMATCH;
    }
    try {
        std::vector<double> weights(
            sample_weights,
            sample_weights + static_cast<std::size_t>(sample_weights_size));
        ::pls4all::core::WeightedPlsResult res;
        const p4a_status_t status = ::pls4all::core::fit_weighted_pls(
            *as_core(ctx), *as_core(cfg), *X, *Y, weights, res);
        if (status != P4A_OK) return status;

        auto handle = std::make_unique<p4a_method_result_s>();
        pack_weighted_result(*handle, res, *X, *Y);
        *out_result = handle.release();
        return P4A_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in p4a_weighted_pls_fit");
        return P4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in p4a_weighted_pls_fit");
        return P4A_ERR_INTERNAL;
    }
}

P4A_API p4a_status_t p4a_robust_pls_fit(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    double huber_k,
    int32_t max_irls_iter,
    p4a_method_result_t** out_result) {
    if (out_result == nullptr) return P4A_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X == nullptr || Y == nullptr) {
        set_error(ctx, "null pointer in p4a_robust_pls_fit");
        return P4A_ERR_NULL_POINTER;
    }
    try {
        ::pls4all::core::WeightedPlsResult res;
        const p4a_status_t status = ::pls4all::core::fit_robust_pls(
            *as_core(ctx), *as_core(cfg), *X, *Y, huber_k, max_irls_iter,
            res);
        if (status != P4A_OK) return status;

        auto handle = std::make_unique<p4a_method_result_s>();
        pack_weighted_result(*handle, res, *X, *Y);
        handle->set_scalar("huber_k", huber_k);

        *out_result = handle.release();
        return P4A_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in p4a_robust_pls_fit");
        return P4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in p4a_robust_pls_fit");
        return P4A_ERR_INTERNAL;
    }
}

P4A_API p4a_status_t p4a_ridge_pls_fit(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    double ridge_lambda,
    p4a_method_result_t** out_result) {
    if (out_result == nullptr) return P4A_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X == nullptr || Y == nullptr) {
        set_error(ctx, "null pointer in p4a_ridge_pls_fit");
        return P4A_ERR_NULL_POINTER;
    }
    try {
        ::pls4all::core::WeightedPlsResult res;
        const p4a_status_t status = ::pls4all::core::fit_ridge_pls(
            *as_core(ctx), *as_core(cfg), *X, *Y, ridge_lambda, res);
        if (status != P4A_OK) return status;

        auto handle = std::make_unique<p4a_method_result_s>();
        pack_weighted_result(*handle, res, *X, *Y);
        handle->set_scalar("ridge_lambda", ridge_lambda);

        *out_result = handle.release();
        return P4A_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in p4a_ridge_pls_fit");
        return P4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in p4a_ridge_pls_fit");
        return P4A_ERR_INTERNAL;
    }
}

P4A_API p4a_status_t p4a_continuum_regression_fit(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    double tau,
    p4a_method_result_t** out_result) {
    if (out_result == nullptr) return P4A_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X == nullptr || Y == nullptr) {
        set_error(ctx, "null pointer in p4a_continuum_regression_fit");
        return P4A_ERR_NULL_POINTER;
    }
    try {
        ::pls4all::core::WeightedPlsResult res;
        const p4a_status_t status = ::pls4all::core::fit_continuum_regression(
            *as_core(ctx), *as_core(cfg), *X, *Y, tau, res);
        if (status != P4A_OK) return status;

        auto handle = std::make_unique<p4a_method_result_s>();
        pack_weighted_result(*handle, res, *X, *Y);
        handle->set_scalar("tau", tau);

        *out_result = handle.release();
        return P4A_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in p4a_continuum_regression_fit");
        return P4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in p4a_continuum_regression_fit");
        return P4A_ERR_INTERNAL;
    }
}

P4A_API p4a_status_t p4a_n_pls_fit(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X_flat,
    int32_t mode_j,
    int32_t mode_k,
    const p4a_matrix_view_t* Y,
    p4a_method_result_t** out_result) {
    if (out_result == nullptr) return P4A_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X_flat == nullptr ||
        Y == nullptr) {
        set_error(ctx, "null pointer in p4a_n_pls_fit");
        return P4A_ERR_NULL_POINTER;
    }
    try {
        ::pls4all::core::NPlsResult res;
        const p4a_status_t status = ::pls4all::core::fit_n_pls(
            *as_core(ctx), *as_core(cfg), *X_flat, mode_j, mode_k, *Y, res);
        if (status != P4A_OK) return status;

        auto handle = std::make_unique<p4a_method_result_s>();
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
        const p4a_status_t pred_status = ::pls4all::core::predict_n_pls(
            *as_core(ctx), res, *X_flat, predictions);
        (void)pred_status;
        const double rmse = in_sample_rmse(predictions, *Y);
        handle->set_double_matrix("predictions", std::move(predictions),
                                   n, q);
        handle->set_scalar("rmse", rmse);

        *out_result = handle.release();
        return P4A_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in p4a_n_pls_fit");
        return P4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in p4a_n_pls_fit");
        return P4A_ERR_INTERNAL;
    }
}

P4A_API p4a_status_t p4a_kernel_pls_fit(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    int32_t kernel_type,
    double gamma,
    double coef0,
    int32_t degree,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    p4a_method_result_t** out_result) {
    if (out_result == nullptr) return P4A_ERR_NULL_POINTER;
    *out_result = nullptr;
    if (ctx == nullptr || cfg == nullptr || X == nullptr || Y == nullptr) {
        set_error(ctx, "null pointer in p4a_kernel_pls_fit");
        return P4A_ERR_NULL_POINTER;
    }
    if (kernel_type < 0 || kernel_type > 3) {
        set_error(ctx, "kernel_type must be in {0,1,2,3}");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    try {
        const auto kernel_enum =
            static_cast<::pls4all::core::KernelType>(kernel_type);
        ::pls4all::core::KernelPlsResult res;
        const p4a_status_t status = ::pls4all::core::fit_kernel_pls(
            *as_core(ctx), *as_core(cfg), kernel_enum, gamma, coef0,
            degree, *X, *Y, res);
        if (status != P4A_OK) return status;

        auto handle = std::make_unique<p4a_method_result_s>();
        const auto n = static_cast<std::int64_t>(res.n_train);
        const auto q = static_cast<std::int64_t>(res.n_targets);
        handle->set_double_matrix("alpha", res.alpha, n, q);
        handle->set_double_matrix("y_mean", res.y_mean, 1, q);

        std::vector<double> predictions;
        const p4a_status_t pred_status = ::pls4all::core::predict_kernel_pls(
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
        return P4A_OK;
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory in p4a_kernel_pls_fit");
        return P4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        set_error(ctx, "internal error in p4a_kernel_pls_fit");
        return P4A_ERR_INTERNAL;
    }
}

}  // extern "C"
