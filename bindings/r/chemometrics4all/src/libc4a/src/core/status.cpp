// SPDX-License-Identifier: CECILL-2.1
//
// c4a_status_t -> string description.

#include "core/status.hpp"

#if defined(__GNUC__) || defined(__clang__)
#  define C4A_NO_UBSAN __attribute__((no_sanitize("undefined")))
#else
#  define C4A_NO_UBSAN
#endif

namespace chemometrics4all::core {

C4A_NO_UBSAN const char* status_to_string(c4a_status_t s) noexcept {
    switch (s) {
        case C4A_OK:                       return "ok";
        case C4A_ERR_INVALID_ARGUMENT:     return "invalid argument";
        case C4A_ERR_NULL_POINTER:         return "null pointer";
        case C4A_ERR_SHAPE_MISMATCH:       return "shape mismatch";
        case C4A_ERR_DTYPE_MISMATCH:       return "dtype mismatch";
        case C4A_ERR_STRIDE_INVALID:       return "stride invalid";
        case C4A_ERR_NOT_FITTED:           return "model not fitted";
        case C4A_ERR_NUMERICAL_FAILURE:    return "numerical failure";
        case C4A_ERR_CONVERGENCE_FAILED:   return "convergence failed";
        case C4A_ERR_OUT_OF_MEMORY:        return "out of memory";
        case C4A_ERR_UNSUPPORTED:          return "unsupported operation";
        case C4A_ERR_NOT_IMPLEMENTED:      return "not implemented";
        case C4A_ERR_ABI_MISMATCH:         return "abi mismatch";
        case C4A_ERR_IO:                   return "io error";
        case C4A_ERR_CORRUPT_BUFFER:       return "corrupt buffer";
        case C4A_ERR_VERSION_INCOMPATIBLE: return "version incompatible";
        case C4A_ERR_BACKEND_UNAVAILABLE:  return "backend unavailable";
        case C4A_ERR_CANCELLED:            return "cancelled";
        case C4A_ERR_INTERNAL:             return "internal error";
    }
    return "unknown status";
}

}  // namespace chemometrics4all::core

#undef C4A_NO_UBSAN
