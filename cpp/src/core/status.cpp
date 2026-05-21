// SPDX-License-Identifier: CECILL-2.1
//
// n4m_status_t -> string description.

#include "core/status.hpp"

#if defined(__GNUC__) || defined(__clang__)
#  define N4M_NO_UBSAN __attribute__((no_sanitize("undefined")))
#else
#  define N4M_NO_UBSAN
#endif

namespace n4m::core {

N4M_NO_UBSAN const char* status_to_string(n4m_status_t s) noexcept {
    switch (s) {
        case N4M_OK:                       return "ok";
        case N4M_ERR_INVALID_ARGUMENT:     return "invalid argument";
        case N4M_ERR_NULL_POINTER:         return "null pointer";
        case N4M_ERR_SHAPE_MISMATCH:       return "shape mismatch";
        case N4M_ERR_DTYPE_MISMATCH:       return "dtype mismatch";
        case N4M_ERR_STRIDE_INVALID:       return "stride invalid";
        case N4M_ERR_NOT_FITTED:           return "model not fitted";
        case N4M_ERR_NUMERICAL_FAILURE:    return "numerical failure";
        case N4M_ERR_CONVERGENCE_FAILED:   return "convergence failed";
        case N4M_ERR_OUT_OF_MEMORY:        return "out of memory";
        case N4M_ERR_UNSUPPORTED:          return "unsupported operation";
        case N4M_ERR_NOT_IMPLEMENTED:      return "not implemented";
        case N4M_ERR_ABI_MISMATCH:         return "abi mismatch";
        case N4M_ERR_IO:                   return "io error";
        case N4M_ERR_CORRUPT_BUFFER:       return "corrupt buffer";
        case N4M_ERR_VERSION_INCOMPATIBLE: return "version incompatible";
        case N4M_ERR_BACKEND_UNAVAILABLE:  return "backend unavailable";
        case N4M_ERR_CANCELLED:            return "cancelled";
        case N4M_ERR_INTERNAL:             return "internal error";
    }
    return "unknown status";
}

}  // namespace n4m::core

#undef N4M_NO_UBSAN
