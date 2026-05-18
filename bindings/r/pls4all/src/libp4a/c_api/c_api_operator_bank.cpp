// SPDX-License-Identifier: CECILL-2.1
//
// extern "C" wrappers for p4a_operator_bank_t.

#include <stdint.h>

#include <new>

#include "pls4all/p4a.h"

#include "core/operator_bank.hpp"

namespace {

inline ::pls4all::core::OperatorBank* as_core(p4a_operator_bank_t* b) noexcept {
    return static_cast<::pls4all::core::OperatorBank*>(b);
}
inline const ::pls4all::core::OperatorBank* as_core(const p4a_operator_bank_t* b) noexcept {
    return static_cast<const ::pls4all::core::OperatorBank*>(b);
}

}  // namespace

extern "C" {

P4A_API p4a_status_t p4a_operator_bank_create(p4a_operator_bank_t** out) {
    if (out == nullptr) return P4A_ERR_NULL_POINTER;
    try {
        *out = new p4a_operator_bank_s{};
        return P4A_OK;
    } catch (const std::bad_alloc&) {
        *out = nullptr;
        return P4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        *out = nullptr;
        return P4A_ERR_INTERNAL;
    }
}

P4A_API void p4a_operator_bank_destroy(p4a_operator_bank_t* bank) {
    try { delete bank; } catch (...) {}
}

P4A_API p4a_status_t p4a_operator_bank_add(p4a_operator_bank_t* bank,
                                            p4a_operator_kind_t kind,
                                            const double* params,
                                            int32_t n_params) {
    if (bank == nullptr) return P4A_ERR_NULL_POINTER;
    if (!::pls4all::core::operator_kind_is_valid(kind)) return P4A_ERR_INVALID_ARGUMENT;
    if (n_params < 0) return P4A_ERR_INVALID_ARGUMENT;
    if (n_params > 0 && params == nullptr) return P4A_ERR_NULL_POINTER;
    try {
        as_core(bank)->add(kind, params, n_params);
        return P4A_OK;
    } catch (const std::bad_alloc&) {
        return P4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        return P4A_ERR_INTERNAL;
    }
}

P4A_API p4a_status_t p4a_operator_bank_size(const p4a_operator_bank_t* bank,
                                             int32_t* out_size) {
    if (bank == nullptr || out_size == nullptr) return P4A_ERR_NULL_POINTER;
    try {
        *out_size = as_core(bank)->size();
        return P4A_OK;
    } catch (...) {
        return P4A_ERR_INTERNAL;
    }
}

}  // extern "C"
