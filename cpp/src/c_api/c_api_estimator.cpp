// SPDX-License-Identifier: CECILL-2.1
//
// C ABI of the generic estimator surface (n4m/estimator.h).

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <exception>
#include <limits>
#include <new>

#include "core/common/context.hpp"
#include "core/estimator/spec.hpp"
#include "n4m/estimator.h"

using n4m::estimator::FitInputs;
using n4m::estimator::MethodSpec;
using n4m::estimator::Params;
using n4m::estimator::set_error;
using n4m::estimator::set_error_named;

namespace {

constexpr const char* kInputNames[N4M_FIT_INPUT_COUNT] = {
    "y", "labels", "sample_weight", "groups", "feature_groups", "blocks", "axis", "target_domain"};

template <typename Fn>
n4m_status_t guarded(n4m_context_t* ctx, Fn&& fn) noexcept {
    try {
        return fn();
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        set_error(ctx, e.what());
        return N4M_ERR_INTERNAL;
    } catch (...) {
        set_error(ctx, "internal error");
        return N4M_ERR_INTERNAL;
    }
}

// Descriptor output rules: caller sets struct_size; the library writes the
// common prefix and zeroes it on error.
template <typename T>
n4m_status_t write_descriptor(T* out, std::size_t min_size, const T& value) noexcept {
    if (out == nullptr) return N4M_ERR_NULL_POINTER;
    const std::size_t caller = out->struct_size;
    const std::size_t n = std::min(caller, sizeof(T));
    if (caller < min_size) {
        std::memset(out, 0, std::min(caller, sizeof(T)));
        return N4M_ERR_INVALID_ARGUMENT;
    }
    std::memcpy(out, &value, n);
    out->struct_size = static_cast<std::uint32_t>(caller);
    return N4M_OK;
}

n4m_status_t copy_array(const std::int64_t* src, std::int64_t n, std::int64_t* out,
                        std::int64_t capacity, std::int64_t* out_count) noexcept {
    if (out_count == nullptr) return N4M_ERR_NULL_POINTER;
    *out_count = n;
    if (out == nullptr && capacity == 0) return N4M_OK;
    if (out == nullptr) return N4M_ERR_NULL_POINTER;
    if (capacity < n) return N4M_ERR_INVALID_ARGUMENT;
    std::copy(src, src + n, out);
    return N4M_OK;
}

n4m_status_t copy_array(const double* src, std::int64_t n, double* out, std::int64_t capacity,
                        std::int64_t* out_count) noexcept {
    if (out_count == nullptr) return N4M_ERR_NULL_POINTER;
    *out_count = n;
    if (out == nullptr && capacity == 0) return N4M_OK;
    if (out == nullptr) return N4M_ERR_NULL_POINTER;
    if (capacity < n) return N4M_ERR_INVALID_ARGUMENT;
    std::copy(src, src + n, out);
    return N4M_OK;
}

// Normalizes the caller's inputs and checks them against the manifest.
n4m_status_t normalize_inputs(n4m_context_t* ctx, const MethodSpec& spec,
                              const n4m_fit_inputs_v1_t* raw, FitInputs& in) {
    if (raw == nullptr) {
        set_error(ctx, "fit inputs are NULL");
        return N4M_ERR_NULL_POINTER;
    }
    constexpr std::size_t kMin = offsetof(n4m_fit_inputs_v1_t, X) + sizeof(void*);
    if (raw->struct_size < kMin) {
        set_error(ctx, "n4m_fit_inputs_v1_t.struct_size is too small");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    n4m_fit_inputs_v1_t v{};
    std::memcpy(&v, raw, std::min<std::size_t>(raw->struct_size, sizeof(v)));
    if (v.X == nullptr) {
        set_error(ctx, "fit input 'X' is required");
        return N4M_ERR_NULL_POINTER;
    }
    in = FitInputs{};
    in.X = v.X;
    in.Y = v.Y;
    in.labels = v.labels;
    in.n_labels = v.labels != nullptr ? v.n_labels : 0;
    in.sample_weight = v.sample_weight;
    in.n_sample_weight = v.sample_weight != nullptr ? v.n_sample_weight : 0;
    in.groups = v.groups;
    in.n_groups = v.groups != nullptr ? v.n_groups : 0;
    in.feature_groups = v.feature_groups;
    in.n_feature_groups = v.feature_groups != nullptr ? v.n_feature_groups : 0;
    in.block_sizes = v.block_sizes;
    in.n_blocks = v.block_sizes != nullptr ? v.n_blocks : 0;
    in.axis = v.axis;
    in.n_axis = v.axis != nullptr ? v.n_axis : 0;
    in.X_target = v.X_target;
    in.seed = v.seed;

    const bool present[N4M_FIT_INPUT_COUNT] = {
        in.Y != nullptr,          in.labels != nullptr,         in.sample_weight != nullptr,
        in.groups != nullptr,     in.feature_groups != nullptr, in.block_sizes != nullptr,
        in.axis != nullptr,       in.X_target != nullptr};
    for (int k = 0; k < N4M_FIT_INPUT_COUNT; ++k) {
        if (spec.inputs[k] == N4M_INPUT_REQUIRED && !present[k]) {
            set_error_named(ctx, "missing required fit input", kInputNames[k]);
            return N4M_ERR_INVALID_ARGUMENT;
        }
        if (spec.inputs[k] == N4M_INPUT_NONE && present[k]) {
            set_error_named(ctx, "fit input is not used by this method", kInputNames[k]);
            return N4M_ERR_INVALID_ARGUMENT;
        }
    }
    const std::int64_t rows = in.X->rows;
    const std::int64_t cols = in.X->cols;
    auto fail = [&](const char* name) {
        set_error_named(ctx, "fit input has an incompatible length", name);
        return N4M_ERR_SHAPE_MISMATCH;
    };
    if (in.Y != nullptr && in.Y->rows != rows) return fail("y");
    if (in.labels != nullptr && in.n_labels != rows) return fail("labels");
    if (in.sample_weight != nullptr && in.n_sample_weight != rows) return fail("sample_weight");
    if (in.groups != nullptr && in.n_groups != rows) return fail("groups");
    if (in.feature_groups != nullptr && in.n_feature_groups != cols) return fail("feature_groups");
    if (in.axis != nullptr && in.n_axis != cols) return fail("axis");
    if (in.X_target != nullptr && in.X_target->cols != cols) return fail("target_domain");
    if (in.block_sizes != nullptr) {
        std::int64_t total = 0;
        for (std::int64_t b = 0; b < in.n_blocks; ++b) {
            if (in.block_sizes[b] <= 0) return fail("blocks");
            total += in.block_sizes[b];
        }
        if (in.n_blocks <= 0 || total != cols) return fail("blocks");
    }
    return N4M_OK;
}

n4m_status_t check_fitted(n4m_context_t* ctx, const n4m_estimator_t* est) {
    if (est == nullptr) {
        set_error(ctx, "estimator is NULL");
        return N4M_ERR_NULL_POINTER;
    }
    if (!est->fitted) {
        set_error(ctx, "estimator is not fitted");
        return N4M_ERR_NOT_FITTED;
    }
    return N4M_OK;
}

n4m_status_t check_rows(n4m_context_t* ctx, const n4m_estimator_t* est,
                        const n4m_matrix_view_t* X, std::int64_t out_rows) {
    if (X == nullptr) {
        set_error(ctx, "X is NULL");
        return N4M_ERR_NULL_POINTER;
    }
    if (X->cols != est->adapter->n_features_in() || out_rows != X->rows) {
        set_error(ctx, "X or output shape does not match the fitted estimator");
        return N4M_ERR_SHAPE_MISMATCH;
    }
    return N4M_OK;
}

using MatrixOp = n4m_status_t (n4m::estimator::Adapter::*)(n4m_context_t*,
                                                           const n4m_matrix_view_t&,
                                                           n4m_matrix_view_t&) const;

n4m_status_t matrix_op(n4m_context_t* ctx, const n4m_estimator_t* est,
                       const n4m_matrix_view_t* X, n4m_matrix_view_t* out, std::uint64_t cap,
                       MatrixOp op) {
    return guarded(ctx, [&]() {
        n4m_status_t st = check_fitted(ctx, est);
        if (st != N4M_OK) return st;
        if (out == nullptr) {
            set_error(ctx, "output view is NULL");
            return N4M_ERR_NULL_POINTER;
        }
        if ((est->adapter->capabilities() & cap) == 0) {
            set_error(ctx, "operation is not supported by this estimator");
            return N4M_ERR_UNSUPPORTED;
        }
        st = check_rows(ctx, est, X, out->rows);
        if (st != N4M_OK) return st;
        return ((*est->adapter).*op)(ctx, *X, *out);
    });
}

}  // namespace

/* ---- Introspection -------------------------------------------------- */

N4M_API n4m_status_t n4m_method_count(int32_t* out_count) {
    if (out_count == nullptr) return N4M_ERR_NULL_POINTER;
    *out_count = n4m::estimator::method_count();
    return N4M_OK;
}

N4M_API n4m_status_t n4m_method_find(const char* method_id, int32_t* out_index) {
    if (out_index == nullptr || method_id == nullptr) return N4M_ERR_NULL_POINTER;
    *out_index = n4m::estimator::method_index(method_id);
    return *out_index < 0 ? N4M_ERR_INVALID_ARGUMENT : N4M_OK;
}

N4M_API n4m_status_t n4m_method_info_v1(int32_t index, n4m_method_info_v1_t* out) {
    const MethodSpec* spec = n4m::estimator::method_at(index);
    if (out == nullptr) return N4M_ERR_NULL_POINTER;
    if (spec == nullptr) {
        std::memset(out, 0, std::min<std::size_t>(out->struct_size, sizeof(*out)));
        return N4M_ERR_INVALID_ARGUMENT;
    }
    return guarded(nullptr, [&]() {
        n4m_method_info_v1_t info{};
        info.struct_size = sizeof(info);
        info.kind = spec->kind;
        info.method_id = spec->method_id;
        info.fq_name = spec->fq_name;
        info.roles = spec->roles;
        info.n_params = spec->n_params;
        info.capabilities = spec->factory(*spec)->capabilities();
        info.state_format = spec->state_format;
        for (int k = 0; k < N4M_FIT_INPUT_COUNT; ++k) info.inputs[k] = spec->inputs[k];
        return write_descriptor(out, sizeof(n4m_method_info_v1_t), info);
    });
}

N4M_API n4m_status_t n4m_method_param_info_v1(int32_t index, int32_t param,
                                              n4m_param_info_v1_t* out) {
    const MethodSpec* spec = n4m::estimator::method_at(index);
    if (out == nullptr) return N4M_ERR_NULL_POINTER;
    if (spec == nullptr || param < 0 || param >= spec->n_params) {
        std::memset(out, 0, std::min<std::size_t>(out->struct_size, sizeof(*out)));
        return N4M_ERR_INVALID_ARGUMENT;
    }
    const auto& p = spec->params[param];
    n4m_param_info_v1_t info{};
    info.struct_size = sizeof(info);
    info.type = p.type;
    info.name = p.name;
    info.has_default = p.has_default ? 1 : 0;
    info.n_choices = p.n_choices;
    info.default_length = p.has_default ? p.default_length : 0;
    info.min_value = p.min_value;
    info.max_value = p.max_value;
    info.choices = p.choices;
    return write_descriptor(out, sizeof(n4m_param_info_v1_t), info);
}

N4M_API n4m_status_t n4m_method_param_default_int(int32_t index, int32_t param, int64_t* out,
                                                  int64_t capacity, int64_t* out_count) {
    const MethodSpec* spec = n4m::estimator::method_at(index);
    if (spec == nullptr || param < 0 || param >= spec->n_params) return N4M_ERR_INVALID_ARGUMENT;
    const auto& p = spec->params[param];
    if (!p.has_default || p.default_int == nullptr) return N4M_ERR_INVALID_ARGUMENT;
    return copy_array(p.default_int, p.default_length, out, capacity, out_count);
}

N4M_API n4m_status_t n4m_method_param_default_double(int32_t index, int32_t param, double* out,
                                                     int64_t capacity, int64_t* out_count) {
    const MethodSpec* spec = n4m::estimator::method_at(index);
    if (spec == nullptr || param < 0 || param >= spec->n_params) return N4M_ERR_INVALID_ARGUMENT;
    const auto& p = spec->params[param];
    if (!p.has_default || p.default_double == nullptr) return N4M_ERR_INVALID_ARGUMENT;
    return copy_array(p.default_double, p.default_length, out, capacity, out_count);
}

/* ---- Params --------------------------------------------------------- */

N4M_API n4m_status_t n4m_params_create(n4m_context_t* ctx, int32_t method_index,
                                       n4m_params_t** out) {
    if (out == nullptr) return N4M_ERR_NULL_POINTER;
    *out = nullptr;
    const MethodSpec* spec = n4m::estimator::method_at(method_index);
    if (spec == nullptr) {
        set_error(ctx, "unknown method index");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    return guarded(ctx, [&]() {
        *out = new n4m_params_s(*spec);
        return N4M_OK;
    });
}

N4M_API void n4m_params_destroy(n4m_params_t* params) { delete params; }

N4M_API n4m_status_t n4m_params_set_int(n4m_params_t* p, const char* name, int64_t value) {
    if (p == nullptr) return N4M_ERR_NULL_POINTER;
    return guarded(nullptr, [&]() { return p->params.set_ints(name, N4M_METHOD_PARAM_INT, &value, 1); });
}

N4M_API n4m_status_t n4m_params_set_double(n4m_params_t* p, const char* name, double value) {
    if (p == nullptr) return N4M_ERR_NULL_POINTER;
    return guarded(nullptr,
                   [&]() { return p->params.set_doubles(name, N4M_METHOD_PARAM_DOUBLE, &value, 1); });
}

N4M_API n4m_status_t n4m_params_set_bool(n4m_params_t* p, const char* name, int32_t value) {
    if (p == nullptr) return N4M_ERR_NULL_POINTER;
    const std::int64_t v = value;
    return guarded(nullptr, [&]() { return p->params.set_ints(name, N4M_METHOD_PARAM_BOOL, &v, 1); });
}

N4M_API n4m_status_t n4m_params_set_enum(n4m_params_t* p, const char* name, const char* choice) {
    if (p == nullptr) return N4M_ERR_NULL_POINTER;
    return guarded(nullptr, [&]() { return p->params.set_enum(name, choice); });
}

N4M_API n4m_status_t n4m_params_set_int_array(n4m_params_t* p, const char* name,
                                              const int64_t* values, int64_t n) {
    if (p == nullptr) return N4M_ERR_NULL_POINTER;
    return guarded(nullptr,
                   [&]() { return p->params.set_ints(name, N4M_METHOD_PARAM_INT_ARRAY, values, n); });
}

N4M_API n4m_status_t n4m_params_set_double_array(n4m_params_t* p, const char* name,
                                                 const double* values, int64_t n) {
    if (p == nullptr) return N4M_ERR_NULL_POINTER;
    return guarded(nullptr, [&]() {
        return p->params.set_doubles(name, N4M_METHOD_PARAM_DOUBLE_ARRAY, values, n);
    });
}

N4M_API n4m_status_t n4m_params_validate(n4m_context_t* ctx, const n4m_params_t* params) {
    if (params == nullptr) {
        set_error(ctx, "params is NULL");
        return N4M_ERR_NULL_POINTER;
    }
    if (const char* missing = params->params.missing_required()) {
        set_error_named(ctx, "missing required parameter", missing);
        return N4M_ERR_INVALID_ARGUMENT;
    }
    return N4M_OK;
}

N4M_API n4m_status_t n4m_params_get_int(const n4m_params_t* p, const char* name, int64_t* out,
                                        int64_t capacity, int64_t* out_count) {
    if (p == nullptr || name == nullptr) return N4M_ERR_NULL_POINTER;
    const std::int32_t i = n4m::estimator::param_index(p->params.spec(), name);
    if (i < 0) return N4M_ERR_INVALID_ARGUMENT;
    return guarded(nullptr, [&]() {
        std::vector<std::int64_t> ints;
        std::vector<double> doubles;
        if (!p->params.resolved(i, &ints, &doubles) || !doubles.empty()) {
            return N4M_ERR_INVALID_ARGUMENT;
        }
        return copy_array(ints.data(), static_cast<std::int64_t>(ints.size()), out, capacity,
                          out_count);
    });
}

N4M_API n4m_status_t n4m_params_get_double(const n4m_params_t* p, const char* name, double* out,
                                           int64_t capacity, int64_t* out_count) {
    if (p == nullptr || name == nullptr) return N4M_ERR_NULL_POINTER;
    const std::int32_t i = n4m::estimator::param_index(p->params.spec(), name);
    if (i < 0) return N4M_ERR_INVALID_ARGUMENT;
    return guarded(nullptr, [&]() {
        std::vector<std::int64_t> ints;
        std::vector<double> doubles;
        if (!p->params.resolved(i, &ints, &doubles) || !ints.empty() ||
            (p->params.spec().params[i].type != N4M_METHOD_PARAM_DOUBLE &&
             p->params.spec().params[i].type != N4M_METHOD_PARAM_DOUBLE_ARRAY)) {
            return N4M_ERR_INVALID_ARGUMENT;
        }
        return copy_array(doubles.data(), static_cast<std::int64_t>(doubles.size()), out,
                          capacity, out_count);
    });
}

/* ---- Estimator life cycle ------------------------------------------ */

N4M_API n4m_status_t n4m_estimator_create(n4m_context_t* ctx, const char* method_id,
                                          const n4m_params_t* params, n4m_estimator_t** out) {
    if (out == nullptr || method_id == nullptr) {
        set_error(ctx, "null pointer in n4m_estimator_create");
        return N4M_ERR_NULL_POINTER;
    }
    *out = nullptr;
    const std::int32_t index = n4m::estimator::method_index(method_id);
    if (index < 0) {
        set_error_named(ctx, "unknown method", method_id);
        return N4M_ERR_INVALID_ARGUMENT;
    }
    const MethodSpec& spec = *n4m::estimator::method_at(index);
    if (spec.kind != N4M_METHOD_ESTIMATOR) {
        set_error_named(ctx, "method is a procedure, not an estimator", method_id);
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (params != nullptr && &params->params.spec() != &spec) {
        set_error(ctx, "params were created for another method");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    return guarded(ctx, [&]() {
        Params resolved = params != nullptr ? params->params : Params(spec);
        if (const char* missing = resolved.missing_required()) {
            set_error_named(ctx, "missing required parameter", missing);
            return N4M_ERR_INVALID_ARGUMENT;
        }
        *out = new n4m_estimator_s(index, resolved, spec.factory(spec));
        return N4M_OK;
    });
}

N4M_API void n4m_estimator_destroy(n4m_estimator_t* est) { delete est; }

N4M_API n4m_status_t n4m_estimator_fit(n4m_context_t* ctx, n4m_estimator_t* est,
                                       const n4m_fit_inputs_v1_t* inputs) {
    if (est == nullptr) {
        set_error(ctx, "estimator is NULL");
        return N4M_ERR_NULL_POINTER;
    }
    return guarded(ctx, [&]() {
        est->fitted = false;
        FitInputs in;
        n4m_status_t st = normalize_inputs(ctx, est->params.spec(), inputs, in);
        if (st != N4M_OK) return st;
        st = est->adapter->fit(ctx, est->params, in);
        est->fitted = st == N4M_OK;
        return st;
    });
}

N4M_API n4m_status_t n4m_estimator_is_fitted(const n4m_estimator_t* est, int32_t* out) {
    if (est == nullptr || out == nullptr) return N4M_ERR_NULL_POINTER;
    *out = est->fitted ? 1 : 0;
    return N4M_OK;
}

N4M_API n4m_status_t n4m_estimator_info(const n4m_estimator_t* est, int32_t* out_method_index,
                                        uint64_t* out_capabilities) {
    if (est == nullptr || out_method_index == nullptr || out_capabilities == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    *out_method_index = est->method_index;
    *out_capabilities = est->fitted ? est->adapter->capabilities() : 0;
    return N4M_OK;
}

N4M_API n4m_status_t n4m_estimator_get_params(n4m_context_t* ctx, const n4m_estimator_t* est,
                                              n4m_params_t** out_copy) {
    if (est == nullptr || out_copy == nullptr) return N4M_ERR_NULL_POINTER;
    *out_copy = nullptr;
    return guarded(ctx, [&]() {
        auto* copy = new n4m_params_s(est->params.spec());
        copy->params = est->params;
        *out_copy = copy;
        return N4M_OK;
    });
}

N4M_API n4m_status_t n4m_estimator_n_features_in(const n4m_estimator_t* est, int64_t* out) {
    if (est == nullptr || out == nullptr) return N4M_ERR_NULL_POINTER;
    if (!est->fitted) return N4M_ERR_NOT_FITTED;
    *out = est->adapter->n_features_in();
    return N4M_OK;
}

N4M_API n4m_status_t n4m_estimator_transform_cols(const n4m_estimator_t* est, int64_t* out) {
    if (est == nullptr || out == nullptr) return N4M_ERR_NULL_POINTER;
    if (!est->fitted) return N4M_ERR_NOT_FITTED;
    if ((est->adapter->capabilities() & N4M_CAP_TRANSFORM) == 0) return N4M_ERR_UNSUPPORTED;
    *out = est->adapter->transform_cols();
    return N4M_OK;
}

N4M_API n4m_status_t n4m_estimator_n_outputs(const n4m_estimator_t* est, int64_t* out) {
    if (est == nullptr || out == nullptr) return N4M_ERR_NULL_POINTER;
    if (!est->fitted) return N4M_ERR_NOT_FITTED;
    *out = est->adapter->n_outputs();
    return N4M_OK;
}

N4M_API n4m_status_t n4m_estimator_transform(n4m_context_t* ctx, const n4m_estimator_t* est,
                                             const n4m_matrix_view_t* X,
                                             n4m_matrix_view_t* out) {
    return matrix_op(ctx, est, X, out, N4M_CAP_TRANSFORM, &n4m::estimator::Adapter::transform);
}

N4M_API n4m_status_t n4m_estimator_predict(n4m_context_t* ctx, const n4m_estimator_t* est,
                                           const n4m_matrix_view_t* X, n4m_matrix_view_t* out) {
    return matrix_op(ctx, est, X, out, N4M_CAP_PREDICT, &n4m::estimator::Adapter::predict);
}

N4M_API n4m_status_t n4m_estimator_decision_function(n4m_context_t* ctx,
                                                     const n4m_estimator_t* est,
                                                     const n4m_matrix_view_t* X,
                                                     n4m_matrix_view_t* out) {
    return matrix_op(ctx, est, X, out, N4M_CAP_DECISION_FUNCTION,
                     &n4m::estimator::Adapter::decision_function);
}

N4M_API n4m_status_t n4m_estimator_predict_proba(n4m_context_t* ctx, const n4m_estimator_t* est,
                                                 const n4m_matrix_view_t* X,
                                                 n4m_matrix_view_t* out) {
    return matrix_op(ctx, est, X, out, N4M_CAP_PREDICT_PROBA,
                     &n4m::estimator::Adapter::predict_proba);
}

N4M_API n4m_status_t n4m_estimator_predict_labels(n4m_context_t* ctx,
                                                  const n4m_estimator_t* est,
                                                  const n4m_matrix_view_t* X, int64_t* out,
                                                  int64_t n) {
    return guarded(ctx, [&]() {
        n4m_status_t st = check_fitted(ctx, est);
        if (st != N4M_OK) return st;
        if (out == nullptr) return N4M_ERR_NULL_POINTER;
        if ((est->adapter->capabilities() & N4M_CAP_PREDICT_LABELS) == 0) {
            set_error(ctx, "operation is not supported by this estimator");
            return N4M_ERR_UNSUPPORTED;
        }
        st = check_rows(ctx, est, X, n);
        return st != N4M_OK ? st : est->adapter->predict_labels(ctx, *X, out);
    });
}

N4M_API n4m_status_t n4m_estimator_classes(const n4m_estimator_t* est, int64_t* out,
                                           int64_t capacity, int64_t* out_count) {
    if (est == nullptr) return N4M_ERR_NULL_POINTER;
    if (!est->fitted) return N4M_ERR_NOT_FITTED;
    const auto* classes = est->adapter->classes();
    if (classes == nullptr) return N4M_ERR_UNSUPPORTED;
    return copy_array(classes->data(), static_cast<std::int64_t>(classes->size()), out, capacity,
                      out_count);
}

N4M_API n4m_status_t n4m_estimator_selected_indices(const n4m_estimator_t* est, int64_t* out,
                                                    int64_t capacity, int64_t* out_count) {
    if (est == nullptr) return N4M_ERR_NULL_POINTER;
    if (!est->fitted) return N4M_ERR_NOT_FITTED;
    const auto* selected = est->adapter->selected_indices();
    if (selected == nullptr) return N4M_ERR_UNSUPPORTED;
    return copy_array(selected->data(), static_cast<std::int64_t>(selected->size()), out,
                      capacity, out_count);
}

N4M_API n4m_status_t n4m_estimator_apply_mask(n4m_context_t* ctx, const n4m_estimator_t* est,
                                              const n4m_matrix_view_t* X,
                                              const n4m_matrix_view_t* Y, uint8_t* mask,
                                              int64_t n) {
    return guarded(ctx, [&]() {
        n4m_status_t st = check_fitted(ctx, est);
        if (st != N4M_OK) return st;
        if (mask == nullptr) return N4M_ERR_NULL_POINTER;
        if ((est->adapter->capabilities() & N4M_CAP_APPLY_MASK) == 0) {
            set_error(ctx, "operation is not supported by this estimator");
            return N4M_ERR_UNSUPPORTED;
        }
        st = check_rows(ctx, est, X, n);
        return st != N4M_OK ? st : est->adapter->apply_mask(ctx, *X, Y, mask);
    });
}

N4M_API n4m_status_t n4m_estimator_fit_result(const n4m_estimator_t* est,
                                              const n4m_method_result_t** out_borrowed) {
    if (est == nullptr || out_borrowed == nullptr) return N4M_ERR_NULL_POINTER;
    *out_borrowed = nullptr;
    if (!est->fitted) return N4M_ERR_NOT_FITTED;
    *out_borrowed = est->adapter->fit_result();
    return *out_borrowed != nullptr ? N4M_OK : N4M_ERR_UNSUPPORTED;
}

/* ---- N4ME ------------------------------------------------------------ */

N4M_API n4m_status_t n4m_context_set_max_state_bytes(n4m_context_t* ctx, uint64_t max_bytes) {
    if (ctx == nullptr) return N4M_ERR_NULL_POINTER;
    if (max_bytes == 0) return N4M_ERR_INVALID_ARGUMENT;
    ctx->set_max_state_bytes(max_bytes);
    return N4M_OK;
}

namespace {
n4m_status_t encode_checked(n4m_context_t* ctx, const n4m_estimator_t* est, uint32_t flags,
                            std::vector<unsigned char>& bytes) {
    n4m_status_t st = check_fitted(ctx, est);
    if (st != N4M_OK) return st;
    if ((flags & ~N4M_EXPORT_ALLOW_TRAINING_ROWS) != 0) return N4M_ERR_INVALID_ARGUMENT;
    const std::uint64_t caps = est->adapter->capabilities();
    if ((caps & N4M_CAP_SERIALIZABLE) == 0) {
        set_error(ctx, "estimator state is not serializable");
        return N4M_ERR_UNSUPPORTED;
    }
    if ((caps & N4M_CAP_RETAINS_TRAINING_ROWS) != 0 &&
        (flags & N4M_EXPORT_ALLOW_TRAINING_ROWS) == 0) {
        set_error(ctx, "state retains training rows; pass N4M_EXPORT_ALLOW_TRAINING_ROWS");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    return n4m::estimator::encode_state(ctx, *est, bytes);
}
}  // namespace

N4M_API n4m_status_t n4m_estimator_export_size(n4m_context_t* ctx, const n4m_estimator_t* est,
                                               uint32_t flags, size_t* out_size) {
    if (out_size == nullptr) return N4M_ERR_NULL_POINTER;
    *out_size = 0;
    return guarded(ctx, [&]() {
        std::vector<unsigned char> bytes;
        n4m_status_t st = encode_checked(ctx, est, flags, bytes);
        if (st == N4M_OK) *out_size = bytes.size();
        return st;
    });
}

N4M_API n4m_status_t n4m_estimator_export_to_buffer(n4m_context_t* ctx,
                                                    const n4m_estimator_t* est, uint32_t flags,
                                                    void* buffer, size_t buffer_size,
                                                    size_t* out_written) {
    if (out_written == nullptr || buffer == nullptr) return N4M_ERR_NULL_POINTER;
    *out_written = 0;
    return guarded(ctx, [&]() {
        std::vector<unsigned char> bytes;
        n4m_status_t st = encode_checked(ctx, est, flags, bytes);
        if (st != N4M_OK) return st;
        if (buffer_size < bytes.size()) {
            set_error(ctx, "export buffer is too small");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        std::memcpy(buffer, bytes.data(), bytes.size());
        *out_written = bytes.size();
        return N4M_OK;
    });
}

N4M_API n4m_status_t n4m_estimator_import_from_buffer(n4m_context_t* ctx, const void* buffer,
                                                      size_t buffer_size, n4m_estimator_t** out) {
    if (out == nullptr || buffer == nullptr || ctx == nullptr) return N4M_ERR_NULL_POINTER;
    *out = nullptr;
    return guarded(ctx, [&]() {
        std::unique_ptr<n4m_estimator_s> est;
        n4m_status_t st = n4m::estimator::decode_state(
            ctx, static_cast<const unsigned char*>(buffer), buffer_size, ctx->max_state_bytes(),
            est);
        if (st == N4M_OK) *out = est.release();
        return st;
    });
}
