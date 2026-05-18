// SPDX-License-Identifier: CECILL-2.1

#include <new>

#include "pls4all/p4a.h"

#include "core/gating_strategy.hpp"

extern "C" {

P4A_API p4a_status_t p4a_gating_strategy_create(p4a_gating_strategy_t** out,
                                                 p4a_gating_mode_t mode) {
    if (out == nullptr) return P4A_ERR_NULL_POINTER;
    if (!::pls4all::core::gating_mode_is_valid(mode)) {
        *out = nullptr;
        return P4A_ERR_INVALID_ARGUMENT;
    }
    try {
        *out = new p4a_gating_strategy_s(mode);
        return P4A_OK;
    } catch (const std::bad_alloc&) {
        *out = nullptr;
        return P4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        *out = nullptr;
        return P4A_ERR_INTERNAL;
    }
}

P4A_API void p4a_gating_strategy_destroy(p4a_gating_strategy_t* gs) {
    try { delete gs; } catch (...) {}
}

}  // extern "C"
