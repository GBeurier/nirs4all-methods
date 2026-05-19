// SPDX-License-Identifier: CECILL-2.1
//
// extern "C" wrappers for p4a_validation_plan_t.

#include <stddef.h>
#include <stdint.h>

#include <new>
#include <utility>
#include <vector>

#include "pls4all/p4a.h"

#include "core/validation.hpp"

namespace {

inline ::pls4all::core::ValidationPlan* as_core(p4a_validation_plan_t* p) noexcept {
    return static_cast<::pls4all::core::ValidationPlan*>(p);
}
inline const ::pls4all::core::ValidationPlan* as_core(const p4a_validation_plan_t* p) noexcept {
    return static_cast<const ::pls4all::core::ValidationPlan*>(p);
}

}  // namespace

extern "C" {

P4A_API p4a_status_t p4a_validation_plan_create(p4a_validation_plan_t** out) {
    if (out == nullptr) {
        return P4A_ERR_NULL_POINTER;
    }
    *out = nullptr;
    try {
        *out = new p4a_validation_plan_s{};
        return P4A_OK;
    } catch (const std::bad_alloc&) {
        return P4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        return P4A_ERR_INTERNAL;
    }
}

P4A_API void p4a_validation_plan_destroy(p4a_validation_plan_t* plan) {
    try {
        delete plan;
    } catch (...) {
        // Best-effort cleanup; never propagate.
    }
}

P4A_API p4a_status_t p4a_validation_plan_set_n_samples(
    p4a_validation_plan_t* plan, int64_t n_samples) {
    if (plan == nullptr) return P4A_ERR_NULL_POINTER;
    if (n_samples < 0) return P4A_ERR_INVALID_ARGUMENT;
    try {
        as_core(plan)->n_samples = n_samples;
        return P4A_OK;
    } catch (...) {
        return P4A_ERR_INTERNAL;
    }
}

P4A_API p4a_status_t p4a_validation_plan_add_fold(
    p4a_validation_plan_t* plan,
    const int64_t* train_indices, int64_t n_train,
    const int64_t* test_indices, int64_t n_test) {
    if (plan == nullptr) return P4A_ERR_NULL_POINTER;
    if (n_train <= 0 || n_test <= 0) return P4A_ERR_INVALID_ARGUMENT;
    if (train_indices == nullptr || test_indices == nullptr) {
        return P4A_ERR_NULL_POINTER;
    }
    try {
        ::pls4all::core::ValidationFold fold;
        fold.train_indices.assign(train_indices, train_indices + n_train);
        fold.test_indices.assign(test_indices, test_indices + n_test);
        as_core(plan)->folds.push_back(std::move(fold));
        return P4A_OK;
    } catch (const std::bad_alloc&) {
        return P4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        return P4A_ERR_INTERNAL;
    }
}

P4A_API p4a_status_t p4a_validation_plan_get_n_samples(
    const p4a_validation_plan_t* plan, int64_t* out_n_samples) {
    if (plan == nullptr || out_n_samples == nullptr) return P4A_ERR_NULL_POINTER;
    try {
        *out_n_samples = as_core(plan)->n_samples;
        return P4A_OK;
    } catch (...) {
        return P4A_ERR_INTERNAL;
    }
}

P4A_API p4a_status_t p4a_validation_plan_get_n_folds(
    const p4a_validation_plan_t* plan, int32_t* out_n_folds) {
    if (plan == nullptr || out_n_folds == nullptr) return P4A_ERR_NULL_POINTER;
    try {
        *out_n_folds = static_cast<int32_t>(as_core(plan)->folds.size());
        return P4A_OK;
    } catch (...) {
        return P4A_ERR_INTERNAL;
    }
}

}  // extern "C"
