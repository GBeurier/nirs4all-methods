// SPDX-License-Identifier: CECILL-2.1
//
// C ABI of the role pipeline (n4m/estimator.h, ABI 2.14).

#include <algorithm>
#include <cstring>
#include <exception>
#include <memory>
#include <new>
#include <vector>

#include "core/common/context.hpp"
#include "core/estimator/role_pipeline.hpp"
#include "n4m/estimator.h"

namespace rp = n4m::estimator;

namespace {

template <typename Fn>
n4m_status_t guarded(n4m_context_t* ctx, Fn&& fn) noexcept {
    try {
        return fn();
    } catch (const std::bad_alloc&) {
        rp::set_error(ctx, "out of memory");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        rp::set_error(ctx, e.what());
        return N4M_ERR_INTERNAL;
    } catch (...) {
        rp::set_error(ctx, "internal error");
        return N4M_ERR_INTERNAL;
    }
}

n4m_status_t check_args(n4m_context_t* ctx, const void* pipeline) {
    if (ctx == nullptr) return N4M_ERR_NULL_POINTER;
    if (pipeline == nullptr) {
        ctx->set_error("role pipeline is NULL");
        return N4M_ERR_NULL_POINTER;
    }
    return N4M_OK;
}

using TerminalOp = n4m_status_t (*)(n4m_context_t*, const n4m_estimator_t*,
                                    const n4m_matrix_view_t*, n4m_matrix_view_t*);

// Runs X through the transformers, then `op` on the terminal step.
n4m_status_t terminal_op(n4m_context_t* ctx, const n4m_role_pipeline_t* p,
                         const n4m_matrix_view_t* X, n4m_matrix_view_t* out, TerminalOp op) {
    n4m_status_t st = check_args(ctx, p);
    if (st != N4M_OK) return st;
    return guarded(ctx, [&]() {
        std::vector<double> store;
        n4m_matrix_view_t features{};
        n4m_status_t s = rp::pipeline_features(ctx, *p, X, store, features);
        if (s != N4M_OK) return s;
        s = rp::pipeline_check_output(ctx, out, features.rows, p->terminal().adapter->n_outputs());
        if (s != N4M_OK) return s;
        ctx->clear_error();
        s = op(ctx, &p->terminal(), &features, out);
        if (s != N4M_OK) {
            return rp::pipeline_error(ctx, s, rp::step_label(*p, p->steps.size() - 1).c_str());
        }
        return N4M_OK;
    });
}

// The fitted estimator of stateful step `state`.
n4m_status_t state_at(n4m_context_t* ctx, const n4m_role_pipeline_t* p, int32_t state,
                      const n4m_estimator_t** out) {
    n4m_status_t st = check_args(ctx, p);
    if (st != N4M_OK) return st;
    if (!p->fitted) {
        ctx->set_error("role pipeline is not fitted");
        return N4M_ERR_NOT_FITTED;
    }
    if (state < 0 || static_cast<std::size_t>(state) >= p->states.size()) {
        ctx->set_errorf("state %d is out of range (the pipeline has %zu states)", state,
                        p->states.size());
        return N4M_ERR_INVALID_ARGUMENT;
    }
    *out = p->states[static_cast<std::size_t>(state)].get();
    return N4M_OK;
}

n4m_status_t state_error(n4m_context_t* ctx, const n4m_role_pipeline_t* p, n4m_status_t st,
                         int32_t state) {
    return rp::pipeline_error(ctx, st,
                              rp::state_label(*p, static_cast<std::size_t>(state)).c_str());
}

}  // namespace

N4M_API n4m_status_t n4m_role_pipeline_create(n4m_context_t* ctx, int32_t n_steps,
                                              const char* const* method_ids,
                                              const n4m_params_t* const* params,
                                              n4m_role_pipeline_t** out) {
    if (ctx == nullptr || out == nullptr) return N4M_ERR_NULL_POINTER;
    *out = nullptr;
    if (method_ids == nullptr && n_steps > 0) {
        ctx->set_error("method_ids is NULL");
        return N4M_ERR_NULL_POINTER;
    }
    return guarded(ctx, [&]() {
        std::unique_ptr<n4m_role_pipeline_s> created;
        const n4m_status_t st = rp::pipeline_create(ctx, n_steps, method_ids, params, created);
        if (st == N4M_OK) *out = created.release();
        return st;
    });
}

N4M_API void n4m_role_pipeline_destroy(n4m_role_pipeline_t* pipeline) { delete pipeline; }

N4M_API n4m_status_t n4m_role_pipeline_set_feature_names(n4m_context_t* ctx,
                                                         n4m_role_pipeline_t* pipeline,
                                                         const char* const* names, int64_t n) {
    const n4m_status_t st = check_args(ctx, pipeline);
    if (st != N4M_OK) return st;
    return guarded(ctx,
                   [&]() { return rp::pipeline_set_feature_names(ctx, *pipeline, names, n); });
}

N4M_API n4m_status_t n4m_role_pipeline_fit(n4m_context_t* ctx, n4m_role_pipeline_t* pipeline,
                                           const n4m_fit_inputs_v1_t* inputs) {
    const n4m_status_t st = check_args(ctx, pipeline);
    if (st != N4M_OK) return st;
    return guarded(ctx, [&]() { return rp::pipeline_fit(ctx, *pipeline, inputs); });
}

N4M_API n4m_status_t n4m_role_pipeline_import_states(n4m_context_t* ctx,
                                                     n4m_role_pipeline_t* pipeline,
                                                     int32_t n_states,
                                                     const void* const* states,
                                                     const size_t* state_sizes) {
    const n4m_status_t st = check_args(ctx, pipeline);
    if (st != N4M_OK) return st;
    return guarded(ctx, [&]() {
        return rp::pipeline_import(ctx, *pipeline, n_states, states, state_sizes);
    });
}

N4M_API n4m_status_t n4m_role_pipeline_is_fitted(const n4m_role_pipeline_t* pipeline,
                                                 int32_t* out) {
    if (pipeline == nullptr || out == nullptr) return N4M_ERR_NULL_POINTER;
    *out = pipeline->fitted ? 1 : 0;
    return N4M_OK;
}

N4M_API n4m_status_t n4m_role_pipeline_n_steps(const n4m_role_pipeline_t* pipeline,
                                               int32_t* out) {
    if (pipeline == nullptr || out == nullptr) return N4M_ERR_NULL_POINTER;
    *out = static_cast<int32_t>(pipeline->steps.size());
    return N4M_OK;
}

N4M_API n4m_status_t n4m_role_pipeline_n_states(const n4m_role_pipeline_t* pipeline,
                                                int32_t* out) {
    if (pipeline == nullptr || out == nullptr) return N4M_ERR_NULL_POINTER;
    *out = static_cast<int32_t>(
        std::count_if(pipeline->steps.begin(), pipeline->steps.end(),
                      [](const auto& step) { return step.state_index >= 0; }));
    return N4M_OK;
}

N4M_API n4m_status_t n4m_role_pipeline_step_info_v1(const n4m_role_pipeline_t* pipeline,
                                                    int32_t step,
                                                    n4m_role_pipeline_step_info_v1_t* out) {
    if (pipeline == nullptr || out == nullptr) return N4M_ERR_NULL_POINTER;
    const std::size_t caller = out->struct_size;
    const std::size_t n = std::min(caller, sizeof(*out));
    if (caller < sizeof(n4m_role_pipeline_step_info_v1_t) || step < 0 ||
        static_cast<std::size_t>(step) >= pipeline->steps.size()) {
        std::memset(out, 0, n);
        return N4M_ERR_INVALID_ARGUMENT;
    }
    const auto& s = pipeline->steps[static_cast<std::size_t>(step)];
    n4m_role_pipeline_step_info_v1_t info{};
    info.struct_size = static_cast<uint32_t>(caller);
    info.method_index = s.method_index;
    info.method_id = rp::method_at(s.method_index)->method_id;
    info.role = s.role;
    info.state_index = s.state_index;
    if (pipeline->fitted) {
        if (s.state_index < 0) {
            info.n_features_in = info.n_features_out = pipeline->n_features;
        } else {
            const auto& adapter = *pipeline->states[static_cast<std::size_t>(s.state_index)]->adapter;
            info.contains_training_rows =
                (adapter.capabilities() & N4M_CAP_RETAINS_TRAINING_ROWS) != 0 ? 1 : 0;
            info.n_features_in = adapter.n_features_in();
            info.n_features_out = (s.role & (N4M_ROLE_REGRESSOR | N4M_ROLE_CLASSIFIER)) != 0
                                      ? adapter.n_outputs()
                                      : adapter.transform_cols();
        }
    }
    std::memcpy(out, &info, n);
    return N4M_OK;
}

N4M_API n4m_status_t n4m_role_pipeline_n_features_in(const n4m_role_pipeline_t* pipeline,
                                                     int64_t* out) {
    if (pipeline == nullptr || out == nullptr) return N4M_ERR_NULL_POINTER;
    if (!pipeline->fitted) return N4M_ERR_NOT_FITTED;
    *out = pipeline->n_features;
    return N4M_OK;
}

N4M_API n4m_status_t n4m_role_pipeline_n_feature_names(const n4m_role_pipeline_t* pipeline,
                                                       int64_t* out) {
    if (pipeline == nullptr || out == nullptr) return N4M_ERR_NULL_POINTER;
    *out = static_cast<int64_t>(pipeline->feature_names.size());
    return N4M_OK;
}

N4M_API n4m_status_t n4m_role_pipeline_feature_name(const n4m_role_pipeline_t* pipeline,
                                                    int64_t index, const char** out_borrowed) {
    if (pipeline == nullptr || out_borrowed == nullptr) return N4M_ERR_NULL_POINTER;
    *out_borrowed = nullptr;
    if (index < 0 || static_cast<std::size_t>(index) >= pipeline->feature_names.size()) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    *out_borrowed = pipeline->feature_names[static_cast<std::size_t>(index)].c_str();
    return N4M_OK;
}

N4M_API n4m_status_t n4m_role_pipeline_check_features(n4m_context_t* ctx,
                                                      const n4m_role_pipeline_t* pipeline,
                                                      int64_t n_columns,
                                                      const char* const* names) {
    const n4m_status_t st = check_args(ctx, pipeline);
    if (st != N4M_OK) return st;
    return guarded(ctx, [&]() {
        return rp::pipeline_check_features(ctx, *pipeline, n_columns, names);
    });
}

N4M_API n4m_status_t n4m_role_pipeline_transform_cols(const n4m_role_pipeline_t* pipeline,
                                                      int64_t* out) {
    if (pipeline == nullptr || out == nullptr) return N4M_ERR_NULL_POINTER;
    if (!pipeline->fitted) return N4M_ERR_NOT_FITTED;
    *out = rp::pipeline_transform_cols(*pipeline);
    return N4M_OK;
}

N4M_API n4m_status_t n4m_role_pipeline_n_outputs(const n4m_role_pipeline_t* pipeline,
                                                 int64_t* out) {
    if (pipeline == nullptr || out == nullptr) return N4M_ERR_NULL_POINTER;
    if (!pipeline->fitted) return N4M_ERR_NOT_FITTED;
    return n4m_estimator_n_outputs(&pipeline->terminal(), out);
}

N4M_API n4m_status_t n4m_role_pipeline_transform(n4m_context_t* ctx,
                                                 const n4m_role_pipeline_t* pipeline,
                                                 const n4m_matrix_view_t* X,
                                                 n4m_matrix_view_t* out) {
    const n4m_status_t st = check_args(ctx, pipeline);
    if (st != N4M_OK) return st;
    return guarded(ctx, [&]() { return rp::pipeline_transform(ctx, *pipeline, X, out); });
}

N4M_API n4m_status_t n4m_role_pipeline_predict(n4m_context_t* ctx,
                                               const n4m_role_pipeline_t* pipeline,
                                               const n4m_matrix_view_t* X,
                                               n4m_matrix_view_t* out) {
    return terminal_op(ctx, pipeline, X, out, n4m_estimator_predict);
}

N4M_API n4m_status_t n4m_role_pipeline_decision_function(n4m_context_t* ctx,
                                                         const n4m_role_pipeline_t* pipeline,
                                                         const n4m_matrix_view_t* X,
                                                         n4m_matrix_view_t* out) {
    return terminal_op(ctx, pipeline, X, out, n4m_estimator_decision_function);
}

N4M_API n4m_status_t n4m_role_pipeline_predict_proba(n4m_context_t* ctx,
                                                     const n4m_role_pipeline_t* pipeline,
                                                     const n4m_matrix_view_t* X,
                                                     n4m_matrix_view_t* out) {
    return terminal_op(ctx, pipeline, X, out, n4m_estimator_predict_proba);
}

N4M_API n4m_status_t n4m_role_pipeline_predict_labels(n4m_context_t* ctx,
                                                      const n4m_role_pipeline_t* pipeline,
                                                      const n4m_matrix_view_t* X, int64_t* out,
                                                      int64_t n) {
    n4m_status_t st = check_args(ctx, pipeline);
    if (st != N4M_OK) return st;
    return guarded(ctx, [&]() {
        std::vector<double> store;
        n4m_matrix_view_t features{};
        n4m_status_t s = rp::pipeline_features(ctx, *pipeline, X, store, features);
        if (s != N4M_OK) return s;
        ctx->clear_error();
        s = n4m_estimator_predict_labels(ctx, &pipeline->terminal(), &features, out, n);
        if (s != N4M_OK) {
            return rp::pipeline_error(
                ctx, s, rp::step_label(*pipeline, pipeline->steps.size() - 1).c_str());
        }
        return N4M_OK;
    });
}

N4M_API n4m_status_t n4m_role_pipeline_classes(const n4m_role_pipeline_t* pipeline,
                                               int64_t* out, int64_t capacity,
                                               int64_t* out_count) {
    if (pipeline == nullptr) return N4M_ERR_NULL_POINTER;
    if (!pipeline->fitted) return N4M_ERR_NOT_FITTED;
    return n4m_estimator_classes(&pipeline->terminal(), out, capacity, out_count);
}

N4M_API n4m_status_t n4m_role_pipeline_export_state_size(n4m_context_t* ctx,
                                                         const n4m_role_pipeline_t* pipeline,
                                                         int32_t state, uint32_t flags,
                                                         size_t* out_size) {
    if (out_size == nullptr) return N4M_ERR_NULL_POINTER;
    *out_size = 0;
    const n4m_estimator_t* est = nullptr;
    n4m_status_t st = state_at(ctx, pipeline, state, &est);
    if (st != N4M_OK) return st;
    ctx->clear_error();
    st = n4m_estimator_export_size(ctx, est, flags, out_size);
    return st == N4M_OK ? st : state_error(ctx, pipeline, st, state);
}

N4M_API n4m_status_t n4m_role_pipeline_export_state_to_buffer(
    n4m_context_t* ctx, const n4m_role_pipeline_t* pipeline, int32_t state, uint32_t flags,
    void* buffer, size_t buffer_size, size_t* out_written) {
    if (out_written == nullptr) return N4M_ERR_NULL_POINTER;
    *out_written = 0;
    const n4m_estimator_t* est = nullptr;
    n4m_status_t st = state_at(ctx, pipeline, state, &est);
    if (st != N4M_OK) return st;
    ctx->clear_error();
    st = n4m_estimator_export_to_buffer(ctx, est, flags, buffer, buffer_size, out_written);
    return st == N4M_OK ? st : state_error(ctx, pipeline, st, state);
}
