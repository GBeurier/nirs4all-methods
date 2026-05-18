// SPDX-License-Identifier: CECILL-2.1
//
// p4a_status_t -> string description.

#include "core/status.hpp"

#if defined(__GNUC__) || defined(__clang__)
#  define P4A_NO_UBSAN __attribute__((no_sanitize("undefined")))
#else
#  define P4A_NO_UBSAN
#endif

namespace pls4all::core {

P4A_NO_UBSAN const char* status_to_string(p4a_status_t s) noexcept {
    switch (s) {
        case P4A_OK:                       return "ok";
        case P4A_ERR_INVALID_ARGUMENT:     return "invalid argument";
        case P4A_ERR_NULL_POINTER:         return "null pointer";
        case P4A_ERR_SHAPE_MISMATCH:       return "shape mismatch";
        case P4A_ERR_DTYPE_MISMATCH:       return "dtype mismatch";
        case P4A_ERR_STRIDE_INVALID:       return "stride invalid";
        case P4A_ERR_NOT_FITTED:           return "model not fitted";
        case P4A_ERR_NUMERICAL_FAILURE:    return "numerical failure";
        case P4A_ERR_CONVERGENCE_FAILED:   return "convergence failed";
        case P4A_ERR_OUT_OF_MEMORY:        return "out of memory";
        case P4A_ERR_UNSUPPORTED:          return "unsupported operation";
        case P4A_ERR_NOT_IMPLEMENTED:      return "not implemented";
        case P4A_ERR_ABI_MISMATCH:         return "abi mismatch";
        case P4A_ERR_IO:                   return "io error";
        case P4A_ERR_CORRUPT_BUFFER:       return "corrupt buffer";
        case P4A_ERR_VERSION_INCOMPATIBLE: return "version incompatible";
        case P4A_ERR_BACKEND_UNAVAILABLE:  return "backend unavailable";
        case P4A_ERR_CANCELLED:            return "cancelled";
        case P4A_ERR_INTERNAL:             return "internal error";
    }
    return "unknown status";
}

}  // namespace pls4all::core

#undef P4A_NO_UBSAN
