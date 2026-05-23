// SPDX-License-Identifier: CECILL-2.1

#include <new>

#include "n4m/n4m.h"

#include "core/gating_strategy.hpp"

extern "C" {

N4M_API n4m_status_t n4m_gating_strategy_create(n4m_gating_strategy_t** out,
                                                 n4m_gating_mode_t mode) {
    if (out == nullptr) return N4M_ERR_NULL_POINTER;
    if (!::n4m::core::gating_mode_is_valid(mode)) {
        *out = nullptr;
        return N4M_ERR_INVALID_ARGUMENT;
    }
    try {
        *out = new n4m_gating_strategy_s(mode);
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        *out = nullptr;
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        *out = nullptr;
        return N4M_ERR_INTERNAL;
    }
}

N4M_API void n4m_gating_strategy_destroy(n4m_gating_strategy_t* gs) {
    try { delete gs; } catch (...) {}
}

}  // extern "C"
