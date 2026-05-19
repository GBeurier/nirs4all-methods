/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Composite filter reference implementation.
 *
 * Aggregation only — no algorithm replay.  See header for the contract.
 */

#include "composite.h"

#include <stddef.h>

c4a_status_t c4a_filter_composite_combine(
    int mode,
    const uint8_t* const* submasks,
    int n_sub,
    int64_t rows,
    uint8_t* mask_out,
    int64_t* out_n_samples,
    int64_t* out_n_kept,
    int64_t* out_n_excluded) {
    if (mask_out == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (rows < 0) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (mode != C4A_FILTER_COMPOSITE_ANY && mode != C4A_FILTER_COMPOSITE_ALL) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (n_sub < 0) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (n_sub > 0) {
        if (submasks == NULL) {
            return C4A_ERR_NULL_POINTER;
        }
        for (int k = 0; k < n_sub; ++k) {
            if (submasks[k] == NULL) {
                return C4A_ERR_NULL_POINTER;
            }
        }
    }

    int64_t kept = 0;

    if (n_sub == 0) {
        /* Empty composite keeps everything (matches the Python reference). */
        for (int64_t i = 0; i < rows; ++i) {
            mask_out[i] = 1;
        }
        kept = rows;
    } else if (mode == C4A_FILTER_COMPOSITE_ANY) {
        /* ANY = exclude if ANY subfilter excludes = keep iff ALL keep. */
        for (int64_t i = 0; i < rows; ++i) {
            uint8_t keep = 1;
            for (int k = 0; k < n_sub; ++k) {
                if (!submasks[k][i]) { keep = 0; break; }
            }
            mask_out[i] = keep;
            kept += keep;
        }
    } else {
        /* ALL = exclude only if ALL subfilters exclude = keep iff ANY keeps. */
        for (int64_t i = 0; i < rows; ++i) {
            uint8_t keep = 0;
            for (int k = 0; k < n_sub; ++k) {
                if (submasks[k][i]) { keep = 1; break; }
            }
            mask_out[i] = keep;
            kept += keep;
        }
    }
    if (out_n_samples)  *out_n_samples  = rows;
    if (out_n_kept)     *out_n_kept     = kept;
    if (out_n_excluded) *out_n_excluded = rows - kept;
    return C4A_OK;
}
