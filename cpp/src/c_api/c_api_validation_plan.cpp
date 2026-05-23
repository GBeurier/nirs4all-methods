// SPDX-License-Identifier: CECILL-2.1
//
// extern "C" wrappers for n4m_validation_plan_t.

#include <stddef.h>
#include <stdint.h>

#include <new>
#include <utility>
#include <vector>

#include "n4m/n4m.h"

#include "core/validation.hpp"

namespace {

inline ::n4m::core::ValidationPlan* as_core(n4m_validation_plan_t* p) noexcept {
    return static_cast<::n4m::core::ValidationPlan*>(p);
}
inline const ::n4m::core::ValidationPlan* as_core(const n4m_validation_plan_t* p) noexcept {
    return static_cast<const ::n4m::core::ValidationPlan*>(p);
}

}  // namespace

extern "C" {

N4M_API n4m_status_t n4m_validation_plan_create(n4m_validation_plan_t** out) {
    if (out == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    *out = nullptr;
    try {
        *out = new n4m_validation_plan_s{};
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API void n4m_validation_plan_destroy(n4m_validation_plan_t* plan) {
    try {
        delete plan;
    } catch (...) {
        // Best-effort cleanup; never propagate.
    }
}

N4M_API n4m_status_t n4m_validation_plan_set_n_samples(
    n4m_validation_plan_t* plan, int64_t n_samples) {
    if (plan == nullptr) return N4M_ERR_NULL_POINTER;
    if (n_samples < 0) return N4M_ERR_INVALID_ARGUMENT;
    try {
        as_core(plan)->n_samples = n_samples;
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_validation_plan_add_fold(
    n4m_validation_plan_t* plan,
    const int64_t* train_indices, int64_t n_train,
    const int64_t* test_indices, int64_t n_test) {
    if (plan == nullptr) return N4M_ERR_NULL_POINTER;
    if (n_train <= 0 || n_test <= 0) return N4M_ERR_INVALID_ARGUMENT;
    if (train_indices == nullptr || test_indices == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    try {
        ::n4m::core::ValidationFold fold;
        fold.train_indices.assign(train_indices, train_indices + n_train);
        fold.test_indices.assign(test_indices, test_indices + n_test);
        as_core(plan)->folds.push_back(std::move(fold));
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_validation_plan_get_n_samples(
    const n4m_validation_plan_t* plan, int64_t* out_n_samples) {
    if (plan == nullptr || out_n_samples == nullptr) return N4M_ERR_NULL_POINTER;
    try {
        *out_n_samples = as_core(plan)->n_samples;
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_validation_plan_get_n_folds(
    const n4m_validation_plan_t* plan, int32_t* out_n_folds) {
    if (plan == nullptr || out_n_folds == nullptr) return N4M_ERR_NULL_POINTER;
    try {
        *out_n_folds = static_cast<int32_t>(as_core(plan)->folds.size());
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

}  // extern "C"
