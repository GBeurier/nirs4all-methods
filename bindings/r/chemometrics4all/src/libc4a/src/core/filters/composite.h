/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Composite sample filter — internal engine.
 *
 * Combines an arbitrary number of pre-computed keep-masks with either AND
 * (mode == ALL: keep iff every submask keeps) or OR
 * (mode == ANY: keep iff any submask keeps) semantics.
 *
 * Mode encoding mirrors the public ABI surface that Phase 12 introduces:
 *   - C4A_COMPOSITE_ANY = 0  (default)
 *   - C4A_COMPOSITE_ALL = 1
 *
 * The c4a engine intentionally does NOT replay sub-filter algorithms — the
 * public CompositeFilter handle keeps references to its sub-filter handles
 * and asks each one to compute its keep mask, then aggregates. The engine
 * here is responsible only for the aggregation step (boolean combination of
 * pre-computed masks).
 *
 * Pure C, no dependencies. INTERNAL.
 */
#ifndef CHEMOMETRICS4ALL_CORE_FILTERS_COMPOSITE_H
#define CHEMOMETRICS4ALL_CORE_FILTERS_COMPOSITE_H

#include <stdint.h>

#include "chemometrics4all/c4a.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Internal mode constants — match the public ABI enum from Phase 12. */
#define C4A_FILTER_COMPOSITE_ANY 0
#define C4A_FILTER_COMPOSITE_ALL 1

/* Combine sub-filter keep-masks into the final keep-mask.
 *
 * Parameters:
 *   mode      — 0 (ANY) or 1 (ALL)
 *   submasks  — array of pointers to per-filter (rows,) keep-masks
 *   n_sub     — number of sub-filters (may be 0)
 *   rows      — common length of every submask
 *   mask_out  — (rows,) output buffer
 *   out_stats — optional (n_samples, n_kept, n_excluded) triple
 *
 * Returns:
 *   - C4A_OK on success.
 *   - C4A_ERR_NULL_POINTER if mask_out is NULL, or submasks is NULL while
 *     n_sub > 0, or any submasks[k] is NULL.
 *   - C4A_ERR_INVALID_ARGUMENT for mode not in {0, 1} or rows < 0.
 *
 * Edge case: n_sub == 0 returns an all-1 mask of length rows (keep all). */
c4a_status_t c4a_filter_composite_combine(
    int mode,
    const uint8_t* const* submasks,
    int n_sub,
    int64_t rows,
    uint8_t* mask_out,
    int64_t* out_n_samples,
    int64_t* out_n_kept,
    int64_t* out_n_excluded);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CHEMOMETRICS4ALL_CORE_FILTERS_COMPOSITE_H */
