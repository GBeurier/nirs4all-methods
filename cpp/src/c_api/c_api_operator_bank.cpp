// SPDX-License-Identifier: CECILL-2.1
//
// extern "C" wrappers for n4m_operator_bank_t.

#include <stdint.h>

#include <new>

#include "pls4all/p4a.h"

#include "core/operator_bank.hpp"

namespace {

inline ::n4m::core::OperatorBank* as_core(n4m_operator_bank_t* b) noexcept {
    return static_cast<::n4m::core::OperatorBank*>(b);
}
inline const ::n4m::core::OperatorBank* as_core(const n4m_operator_bank_t* b) noexcept {
    return static_cast<const ::n4m::core::OperatorBank*>(b);
}

}  // namespace

extern "C" {

N4M_API n4m_status_t n4m_operator_bank_create(n4m_operator_bank_t** out) {
    if (out == nullptr) return N4M_ERR_NULL_POINTER;
    try {
        *out = new n4m_operator_bank_s{};
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        *out = nullptr;
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        *out = nullptr;
        return N4M_ERR_INTERNAL;
    }
}

N4M_API void n4m_operator_bank_destroy(n4m_operator_bank_t* bank) {
    try { delete bank; } catch (...) {}
}

N4M_API n4m_status_t n4m_operator_bank_add(n4m_operator_bank_t* bank,
                                            n4m_operator_kind_t kind,
                                            const double* params,
                                            int32_t n_params) {
    if (bank == nullptr) return N4M_ERR_NULL_POINTER;
    if (!::n4m::core::operator_kind_is_valid(kind)) return N4M_ERR_INVALID_ARGUMENT;
    if (n_params < 0) return N4M_ERR_INVALID_ARGUMENT;
    if (n_params > 0 && params == nullptr) return N4M_ERR_NULL_POINTER;
    try {
        as_core(bank)->add(kind, params, n_params);
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_operator_bank_size(const n4m_operator_bank_t* bank,
                                             int32_t* out_size) {
    if (bank == nullptr || out_size == nullptr) return N4M_ERR_NULL_POINTER;
    try {
        *out_size = as_core(bank)->size();
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

}  // extern "C"
