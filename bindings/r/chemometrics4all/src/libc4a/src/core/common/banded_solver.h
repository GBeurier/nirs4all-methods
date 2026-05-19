/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Pentadiagonal (5-diagonal) symmetric LDLT solver + helpers shared by the
 * Whittaker-baseline operators (AsLS / AirPLS / ArPLS / IAsLS / BEADS).
 *
 * Storage layout for the input pentadiagonal symmetric matrix:
 *   main_diag[k] = A[k, k]              for k = 0 .. n-1
 *   super1[k]    = A[k, k+1]            for k = 0 .. n-2   (super1[n-1] unused)
 *   super2[k]    = A[k, k+2]            for k = 0 .. n-3   (super2[n-2..n-1] unused)
 *
 * After factorisation A = L D L^T where L is unit lower-triangular with
 * lower bandwidth 2 and D is diagonal. We store:
 *   D[k]    : diagonal of D
 *   L[k][0] : L[k+1, k]                 (super-of-super in stored form)
 *   L[k][1] : L[k+2, k]
 *
 * Indefinite W (e.g. ArPLS' negative-weight intermediates) is rare in
 * practice — the LDLT handles it gracefully by allowing negative D[k].
 * A *zero* diagonal pivot returns C4A_ERR_NUMERICAL_FAILURE: the system
 * is singular and we can't recover.
 *
 * Pure C, INTERNAL. Lives under cpp/src/core/common/.
 */
#ifndef CHEMOMETRICS4ALL_CORE_COMMON_BANDED_SOLVER_H
#define CHEMOMETRICS4ALL_CORE_COMMON_BANDED_SOLVER_H

#include <stdint.h>

#include "chemometrics4all/c4a.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_banded5_t {
    int64_t n;
    double* L;   /* (n, 2) row-major. L[k * 2 + 0] = L[k+1, k]; L[k * 2 + 1] = L[k+2, k] */
    double* D;   /* (n,) */
} c4a_banded5_t;

/* Allocate the factorisation. The pentadiagonal upper-triangular factor is
 * passed in via three arrays of length n (with the last 1-2 entries of
 * super1 / super2 unused). On success the L and D fields are populated.
 *
 * Returns:
 *   - C4A_OK on success.
 *   - C4A_ERR_NULL_POINTER if any input pointer is NULL.
 *   - C4A_ERR_INVALID_ARGUMENT for n < 1.
 *   - C4A_ERR_OUT_OF_MEMORY on alloc failure.
 *   - C4A_ERR_NUMERICAL_FAILURE on a zero pivot.
 */
c4a_status_t c4a_banded5_factor(const double* main_diag,
                                const double* super1, const double* super2,
                                int64_t n, c4a_banded5_t* out);

/* Same algorithm as c4a_banded5_factor but writes into caller-owned scratch
 * buffers. Used by AsLS / AirPLS / ArPLS / IAsLS / BEADS to hoist the L / D
 * allocations out of the per-iteration hot path.
 *
 * Layout of L_buf is identical to c4a_banded5_t::L (row-major (n, 2)).
 *
 * Returns the same status codes as c4a_banded5_factor minus C4A_ERR_OUT_OF_MEMORY
 * (no allocation happens here). */
c4a_status_t c4a_banded5_factor_into(double* L_buf, double* D_buf,
                                      const double* main_diag,
                                      const double* super1, const double* super2,
                                      int64_t n);

/* Solve A x = b using the precomputed factorisation. Both b and x are
 * length n; they may alias (b is read-only by convention but we copy
 * before forward-solving). */
c4a_status_t c4a_banded5_solve(const c4a_banded5_t* fact,
                                const double* b, double* x);

/* Equivalent of c4a_banded5_solve operating directly on caller-owned L / D
 * buffers produced by c4a_banded5_factor_into. */
c4a_status_t c4a_banded5_solve_into(const double* L_buf, const double* D_buf,
                                     int64_t n,
                                     const double* b, double* x);

/* Free the L and D buffers; safe to call with NULL. Zero out the struct. */
void c4a_banded5_free(c4a_banded5_t* fact);

/* Build the diagonals of `lam * D_2^T D_2` for n >= 3 — the canonical
 * Whittaker 2nd-order difference penalty shared across AsLS / AirPLS /
 * ArPLS / IAsLS / BEADS. The three arrays must be length n; trailing
 * entries that are not part of the band (super1[n-1], super2[n-2..n-1])
 * are zero-filled for completeness. */
void c4a_second_diff_penalty_pent5(int64_t n, double lam,
                                    double* main_diag,
                                    double* super1, double* super2);

/* Convergence helper: relative L2 difference
 *   ||w_new - w||_2 / max(||w||_2, DBL_MIN).
 * Matches `pybaselines.utils.relative_difference` exactly. */
double c4a_relative_l2_diff(const double* w, const double* w_new, int64_t n);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CHEMOMETRICS4ALL_CORE_COMMON_BANDED_SOLVER_H */
