/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * One-sided Jacobi SVD on a dense row-major matrix.
 *
 *   A (m x n) -> U (m x k) @ diag(S) (k) @ Vt (k x n),  k = min(m, n)
 *
 * Algorithm (de Rijk, Demmel & Veselic — one-sided variant):
 *
 *   1. Rotate pairs of columns of A by 2x2 orthogonal Givens rotations
 *      until any two columns are orthogonal up to a tolerance proportional
 *      to eps * ||A||_F.
 *   2. After convergence the columns of A are scaled right singular
 *      vectors:  A_final = U' @ diag(S), with U' having orthonormal
 *      columns. We extract S as the column norms and U' by dividing
 *      each column by its singular value (skipping zero columns).
 *   3. V is accumulated from the right by applying the same Givens
 *      rotations to an identity-initialised V matrix; V then holds the
 *      right singular vectors as columns. We finally return Vt = V^T.
 *   4. Sort by descending S, applying the permutation to U and Vt.
 *   5. Canonicalise signs: for each component i, flip U[:, i] and
 *      Vt[i, :] together so that the largest-absolute-value element of
 *      U[:, i] is positive (ties broken by smallest index).
 *
 * Complexity: O((m + n) * n^2) per sweep. The sweep limit is set high
 * enough (30 sweeps) that convergence to machine precision is reached
 * for the parity-fixture matrix sizes (rows up to ~50, cols up to ~200).
 *
 * Storage convention: row-major. A[i, j] = A[i * n + j]. The algorithm
 * operates on columns of A — accesses are inherently strided, but the
 * matrix sizes here keep the working set well within L1/L2.
 *
 * Numerical tolerance: convergence is declared when every off-diagonal
 * entry |a_p^T a_q| satisfies
 *
 *     |a_p^T a_q| <= eps * sqrt(||a_p||^2 * ||a_q||^2)
 *
 * with eps = 4 * DBL_EPSILON (paranoid; tighter than the standard 1
 * ULP test). Zero-norm columns are skipped to keep the rotation finite.
 *
 * There is no early stop on norm reduction: a small but genuine direction
 * keeps rotating until it converges. Termination on exactly
 * rank-deficient inputs comes from the resolution floor
 * (N4M_SVD_FLOOR_EXP): a null column that the rotations keep shrinking
 * without orthogonalising falls below it and is reported as an exact zero
 * singular value with a zero U column.
 */

#include "svd.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Jacobi sweep limit. The classical bound is O(log n) sweeps for
 * well-conditioned matrices; we use 30 to comfortably cover all parity
 * fixtures while still bailing out on pathological inputs. */
#define N4M_SVD_MAX_SWEEPS 60

/* Convergence tolerance scale. eps = 4 * DBL_EPSILON keeps the
 * orthogonalisation criterion within ~2 bits of the machine precision. */
static const double N4M_SVD_EPS = 4.0 * DBL_EPSILON;

/* Resolution floor of the Jacobi iteration, relative to the scaled input
 * (max|a| in [0.5, 1)): a column of norm below 2^N4M_SVD_FLOOR_EXP has a
 * squared norm below 2^-1000, where the rotation algebra no longer resolves
 * it. It takes no further rotation and its singular value is reported as
 * exactly 0. Inputs are refused when their own nonzero entries reach that
 * level (N4M_SVD_MAX_SPREAD_EXP), so a column can only fall below the floor
 * through cancellation, i.e. sigma < 2^-500 * max|A| from in-range entries:
 * exact rank deficiency in floating point (e.g. +-1 designs where one row
 * is the opposite of another, whose null column the rotations shrink by
 * ~eps per sweep without ever orthogonalising it). */
#define N4M_SVD_FLOOR_EXP (-500)

static const double N4M_SVD_ZERO_TOL = 1e-300;

/* Magnitude domain (see svd.h).
 *
 * The factorisations run on A * 2^-e with max|A| * 2^-e in [0.5, 1)
 * (power-of-two scaling is exact, and every algorithm here is homogeneous,
 * so results are bit-identical to an unscaled run whenever the unscaled run
 * neither overflows nor underflows). The read-only helpers skip the copy
 * when the input already lies in [2^-N4M_SVD_SAFE_EXP, 2^N4M_SVD_SAFE_EXP)
 * with its smallest nonzero magnitude above 2^-N4M_SVD_SAFE_MIN_EXP, where
 * squares and Gram sums cannot underflow.
 *
 * Resolution: nonzero entries smaller than 2^-N4M_SVD_MAX_SPREAD_EXP times
 * the largest one would have squares below the normal range once scaled.
 * Such inputs are refused explicitly (N4M_ERR_NUMERICAL_FAILURE) instead
 * of silently losing those entries. */
#define N4M_SVD_SAFE_EXP 128
#define N4M_SVD_SAFE_MIN_EXP 400
#define N4M_SVD_MAX_SPREAD_EXP 480

/* Scans A: non-finite entries or a dynamic range (max|a| / min nonzero
 * |a|) above 2^N4M_SVD_MAX_SPREAD_EXP are an explicit error. Sets *e to the
 * frexp exponent of max|A| when the input must be scaled (always, or
 * outside the safe band), else to 0. */
static n4m_status_t pow2_scale_exponent(const double* A, size_t count,
                                        int always, int* e) {
    double maxabs = 0.0;
    double minabs = HUGE_VAL;
    for (size_t i = 0; i < count; ++i) {
        if (!isfinite(A[i])) {
            return N4M_ERR_NUMERICAL_FAILURE;
        }
        const double a = fabs(A[i]);
        if (a > maxabs) {
            maxabs = a;
        }
        if (a > 0.0 && a < minabs) {
            minabs = a;
        }
    }
    *e = 0;
    if (maxabs == 0.0) {
        return N4M_OK;
    }
    int ex_max = 0;
    int ex_min = 0;
    (void)frexp(maxabs, &ex_max);
    (void)frexp(minabs, &ex_min);
    if (ex_max - ex_min > N4M_SVD_MAX_SPREAD_EXP) {
        return N4M_ERR_NUMERICAL_FAILURE;
    }
    if (always || ex_max > N4M_SVD_SAFE_EXP || ex_max < -N4M_SVD_SAFE_EXP ||
        ex_min < -N4M_SVD_SAFE_MIN_EXP) {
        *e = ex_max;
    }
    return N4M_OK;
}

static void scale_pow2(double* A, size_t count, int e) {
    for (size_t i = 0; i < count; ++i) {
        A[i] = ldexp(A[i], -e);
    }
}

/* Smallest magnitude published for a significant value: below 2^-1042 a
 * subnormal keeps fewer than 32 significant bits. */
#define N4M_SVD_MIN_PUBLISHED_EXP (-1042)

/* Multiplies values[0..count) by 2^e. A value that is significant in the
 * scaled space (|v| > tol * max|v|) must stay finite and at least
 * 2^N4M_SVD_MIN_PUBLISHED_EXP in magnitude after the multiplication;
 * otherwise it is not representable and the call fails instead of
 * publishing Inf or a value flushed towards zero. */
static n4m_status_t unscale_pow2(double* values, int64_t count, int e,
                                 double tol) {
    if (e == 0) {
        return N4M_OK;
    }
    double vmax = 0.0;
    for (int64_t i = 0; i < count; ++i) {
        if (fabs(values[i]) > vmax) {
            vmax = fabs(values[i]);
        }
    }
    for (int64_t i = 0; i < count; ++i) {
        const int significant = fabs(values[i]) > tol * vmax;
        values[i] = ldexp(values[i], e);
        if (!isfinite(values[i]) ||
            (significant &&
             fabs(values[i]) < ldexp(1.0, N4M_SVD_MIN_PUBLISHED_EXP))) {
            return N4M_ERR_NUMERICAL_FAILURE;
        }
    }
    return N4M_OK;
}

/* Input scaled into [0.5, 1) when outside the safe band: a copy for the
 * read-only helpers, NULL (use A) otherwise. */
static n4m_status_t scaled_input(const double* A, size_t count, int* e,
                                 double** copy) {
    *copy = NULL;
    const n4m_status_t st = pow2_scale_exponent(A, count, 0, e);
    if (st != N4M_OK || *e == 0) {
        return st;
    }
    *copy = (double*)malloc(count * sizeof(double));
    if (*copy == NULL) {
        return N4M_ERR_OUT_OF_MEMORY;
    }
    memcpy(*copy, A, count * sizeof(double));
    scale_pow2(*copy, count, *e);
    return N4M_OK;
}

static double rank_tol(int64_t m, int64_t n) {
    return (double)((m > n) ? m : n) * DBL_EPSILON;
}

/* Read / write helpers (kept inline-friendly via the compiler — no
 * function call overhead at -O2). */
static inline double A_at(const double* A, int64_t n, int64_t i, int64_t j) {
    return A[(size_t)i * (size_t)n + (size_t)j];
}
static inline void A_set(double* A, int64_t n, int64_t i, int64_t j,
                         double v) {
    A[(size_t)i * (size_t)n + (size_t)j] = v;
}

/* Forward declaration: tall-matrix (m >= n) one-sided Jacobi. The public
 * entry point dispatches to this directly for m >= n and via transpose
 * for m < n. */
static n4m_status_t svd_compact_tall(double* A, int64_t m, int64_t n,
                                      double* U, double* S, double* Vt);
static void householder_tridiagonal(double* V, int64_t n, double* d,
                                    double* e) {
    for (int64_t i = n - 1; i > 0; --i) {
        const int64_t l = i - 1;
        double h = 0.0;
        double scale = 0.0;

        if (l > 0) {
            for (int64_t k = 0; k <= l; ++k) {
                scale += fabs(V[(size_t)i * (size_t)n + (size_t)k]);
            }
            if (scale > 0.0) {
                for (int64_t k = 0; k <= l; ++k) {
                    V[(size_t)i * (size_t)n + (size_t)k] /= scale;
                    const double v = V[(size_t)i * (size_t)n + (size_t)k];
                    h += v * v;
                }

                const double f = V[(size_t)i * (size_t)n + (size_t)l];
                const double g = (f >= 0.0) ? -sqrt(h) : sqrt(h);
                e[i] = scale * g;
                h -= f * g;
                V[(size_t)i * (size_t)n + (size_t)l] = f - g;

                double fsum = 0.0;
                for (int64_t j = 0; j <= l; ++j) {
                    V[(size_t)j * (size_t)n + (size_t)i] =
                        V[(size_t)i * (size_t)n + (size_t)j] / h;
                    double gsum = 0.0;
                    for (int64_t k = 0; k <= j; ++k) {
                        gsum += V[(size_t)j * (size_t)n + (size_t)k] *
                                V[(size_t)i * (size_t)n + (size_t)k];
                    }
                    for (int64_t k = j + 1; k <= l; ++k) {
                        gsum += V[(size_t)k * (size_t)n + (size_t)j] *
                                V[(size_t)i * (size_t)n + (size_t)k];
                    }
                    e[j] = gsum / h;
                    fsum += e[j] * V[(size_t)i * (size_t)n + (size_t)j];
                }

                const double hh = fsum / (h + h);
                for (int64_t j = 0; j <= l; ++j) {
                    const double f2 =
                        V[(size_t)i * (size_t)n + (size_t)j];
                    const double g2 = e[j] - hh * f2;
                    e[j] = g2;
                    for (int64_t k = 0; k <= j; ++k) {
                        V[(size_t)j * (size_t)n + (size_t)k] -=
                            f2 * e[k] +
                            g2 * V[(size_t)i * (size_t)n + (size_t)k];
                    }
                }
            } else {
                e[i] = V[(size_t)i * (size_t)n + (size_t)l];
            }
        } else {
            e[i] = V[(size_t)i * (size_t)n + (size_t)l];
        }
        d[i] = h;
    }

    d[0] = 0.0;
    e[0] = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const int64_t l = i - 1;
        if (d[i] != 0.0) {
            for (int64_t j = 0; j <= l; ++j) {
                double g = 0.0;
                for (int64_t k = 0; k <= l; ++k) {
                    g += V[(size_t)i * (size_t)n + (size_t)k] *
                         V[(size_t)k * (size_t)n + (size_t)j];
                }
                for (int64_t k = 0; k <= l; ++k) {
                    V[(size_t)k * (size_t)n + (size_t)j] -=
                        g * V[(size_t)k * (size_t)n + (size_t)i];
                }
            }
        }
        d[i] = V[(size_t)i * (size_t)n + (size_t)i];
        V[(size_t)i * (size_t)n + (size_t)i] = 1.0;
        for (int64_t j = 0; j <= l; ++j) {
            V[(size_t)j * (size_t)n + (size_t)i] = 0.0;
            V[(size_t)i * (size_t)n + (size_t)j] = 0.0;
        }
    }
}

/* Implicit-shift QL (EISPACK tql2 / Numerical Recipes tqli) on the unreduced
 * block [lo, hi] of the tridiagonal (d, e), accumulating rotations in V.
 * Deflation uses the local test |e[m]| <= eps * (|d[m]| + |d[m+1]|), which
 * only involves the block's own entries. *iter counts QL sweeps against the
 * whole-matrix budget max_iter. */
static n4m_status_t ql_block(double* d, double* e, int64_t lo, int64_t hi,
                             int64_t n, double* V, int64_t* iter,
                             int64_t max_iter) {
    for (int64_t l = lo; l <= hi; ++l) {
        for (;;) {
            int64_t m = l;
            for (; m < hi; ++m) {
                const double dd = fabs(d[m]) + fabs(d[m + 1]);
                if (fabs(e[m]) <= DBL_EPSILON * dd) {
                    break;
                }
            }
            if (m == l) {
                break;
            }
            if (++*iter > max_iter) {
                return N4M_ERR_CONVERGENCE_FAILED;
            }

            double g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            double r = hypot(g, 1.0);
            g = d[m] - d[l] +
                e[l] / (g + ((g >= 0.0) ? fabs(r) : -fabs(r)));

            double s = 1.0;
            double c = 1.0;
            double p = 0.0;
            int early = 0;
            for (int64_t i = m - 1; i >= l; --i) {
                const double f = s * e[i];
                const double b = c * e[i];
                r = hypot(f, g);
                e[i + 1] = r;
                if (r == 0.0) {
                    d[i + 1] -= p;
                    e[m] = 0.0;
                    early = 1;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                r = (d[i] - g) * s + 2.0 * c * b;
                p = s * r;
                d[i + 1] = g + p;
                g = c * r - b;

                for (int64_t k = 0; k < n; ++k) {
                    const double vki =
                        V[(size_t)k * (size_t)n + (size_t)i];
                    const double vkip1 =
                        V[(size_t)k * (size_t)n + (size_t)(i + 1)];
                    V[(size_t)k * (size_t)n + (size_t)(i + 1)] =
                        s * vki + c * vkip1;
                    V[(size_t)k * (size_t)n + (size_t)i] =
                        c * vki - s * vkip1;
                }
            }
            if (early) {
                continue;
            }
            d[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        }
    }
    return N4M_OK;
}

/* Symmetric tridiagonal eigensolver organised like LAPACK dsteqr:
 *
 *  - The matrix is split once into unreduced blocks at every e[m] passing
 *    the local test |e[m]| <= eps * (|d[m]| + |d[m+1]|) (exact zeros
 *    included). Each block is iterated on its own, so a small block
 *    decoupled from a large one keeps its relative accuracy.
 *  - A block whose largest entry lies outside [2^-N4M_SVD_SAFE_EXP,
 *    2^N4M_SVD_SAFE_EXP) is multiplied by the exact power of two mapping it
 *    into [0.5, 1) before iterating (subnormal or huge blocks) and its
 *    eigenvalues are scaled back; an eigenvalue that overflows on the way
 *    back is an explicit N4M_ERR_NUMERICAL_FAILURE. Blocks inside the band
 *    run unscaled, bit-identical to the plain tql2 iteration.
 *  - No absolute threshold: a subnormal spectrum is resolved like any other.
 *  - Iteration budget: dsteqr's 30 QL sweeps per eigenvalue, shared by the
 *    whole matrix (30 * n). */
static n4m_status_t tridiagonal_ql_eigenvectors(double* d, double* e,
                                                int64_t n, double* V) {
    for (int64_t i = 1; i < n; ++i) {
        e[i - 1] = e[i];
    }
    e[n - 1] = 0.0;

    const int64_t max_iter = 30 * n;
    int64_t iter = 0;
    for (int64_t lo = 0; lo < n;) {
        int64_t hi = lo;
        while (hi < n - 1 &&
               fabs(e[hi]) > DBL_EPSILON * (fabs(d[hi]) + fabs(d[hi + 1]))) {
            ++hi;
        }
        if (hi > lo) {
            double bmax = 0.0;
            for (int64_t i = lo; i <= hi; ++i) {
                bmax = fmax(bmax, fabs(d[i]));
                if (i < hi) {
                    bmax = fmax(bmax, fabs(e[i]));
                }
            }
            int ex = 0;
            (void)frexp(bmax, &ex);
            const int scale =
                (ex > N4M_SVD_SAFE_EXP || ex < -N4M_SVD_SAFE_EXP) ? ex : 0;
            if (scale != 0) {
                for (int64_t i = lo; i <= hi; ++i) {
                    d[i] = ldexp(d[i], -scale);
                    e[i] = ldexp(e[i], -scale);
                }
            }
            const n4m_status_t st =
                ql_block(d, e, lo, hi, n, V, &iter, max_iter);
            if (st != N4M_OK) {
                return st;
            }
            if (scale != 0) {
                for (int64_t i = lo; i <= hi; ++i) {
                    d[i] = ldexp(d[i], scale);
                    if (!isfinite(d[i])) {
                        return N4M_ERR_NUMERICAL_FAILURE;
                    }
                }
            }
        }
        lo = hi + 1;
    }
    return N4M_OK;
}

static n4m_status_t symmetric_eigh(double* A, int64_t n, double* eigvals,
                                   double* V) {
    if (A == NULL || eigvals == NULL || V == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    if (n < 1) {
        return N4M_ERR_INVALID_ARGUMENT;
    }

    double* offdiag = (double*)malloc((size_t)n * sizeof(double));
    if (offdiag == NULL) {
        return N4M_ERR_OUT_OF_MEMORY;
    }

    for (int64_t i = 0; i < n; ++i) {
        memcpy(V + (size_t)i * (size_t)n,
               A + (size_t)i * (size_t)n,
               (size_t)n * sizeof(double));
    }

    householder_tridiagonal(V, n, eigvals, offdiag);
    const n4m_status_t st =
        tridiagonal_ql_eigenvectors(eigvals, offdiag, n, V);
    free(offdiag);
    return st;
}

n4m_status_t n4m_symmetric_eigh(const double* A, int64_t n, double* eigvals,
                                double* V) {
    if (A == NULL || eigvals == NULL || V == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    if (n < 1) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    const size_t n_size = (size_t)n;
    if (n_size > SIZE_MAX / n_size ||
        n_size * n_size > SIZE_MAX / sizeof(double)) {
        return N4M_ERR_OUT_OF_MEMORY;
    }
    const size_t nn = n_size * n_size;
    double* work = (double*)malloc(nn * sizeof(double));
    if (work == NULL) {
        return N4M_ERR_OUT_OF_MEMORY;
    }
    int e = 0;
    n4m_status_t st = pow2_scale_exponent(A, nn, 0, &e);
    if (st == N4M_OK) {
        memcpy(work, A, nn * sizeof(double));
        scale_pow2(work, nn, e);
        st = symmetric_eigh(work, n, eigvals, V);
    }
    free(work);
    return (st == N4M_OK)
               ? unscale_pow2(eigvals, n, e, (double)n * DBL_EPSILON)
               : st;
}

static void sort_eigpairs_desc(double* V, double* eigvals, int64_t n) {
    for (int64_t i = 0; i < n - 1; ++i) {
        int64_t best = i;
        for (int64_t j = i + 1; j < n; ++j) {
            if (eigvals[j] > eigvals[best]) best = j;
        }
        if (best == i) continue;
        const double ev = eigvals[i];
        eigvals[i] = eigvals[best];
        eigvals[best] = ev;
        for (int64_t r = 0; r < n; ++r) {
            const double tmp = V[(size_t)r * (size_t)n + (size_t)i];
            V[(size_t)r * (size_t)n + (size_t)i] =
                V[(size_t)r * (size_t)n + (size_t)best];
            V[(size_t)r * (size_t)n + (size_t)best] = tmp;
        }
    }
}

n4m_status_t n4m_svd_compact(double* A, int64_t m, int64_t n,
                              double* U, double* S, double* Vt) {
    if (m < 1 || n < 1) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (A == NULL || U == NULL || S == NULL || Vt == NULL) {
        return N4M_ERR_NULL_POINTER;
    }

    int e = 0;
    const n4m_status_t scan =
        pow2_scale_exponent(A, (size_t)m * (size_t)n, 1, &e);
    if (scan != N4M_OK) {
        return scan;
    }
    scale_pow2(A, (size_t)m * (size_t)n, e);
    if (m >= n) {
        const n4m_status_t tall = svd_compact_tall(A, m, n, U, S, Vt);
        return (tall == N4M_OK) ? unscale_pow2(S, n, e, rank_tol(m, n)) : tall;
    }

    /* Wide case (m < n): the standard one-sided Jacobi on the columns
     * of A doesn't converge cleanly when there are more columns than
     * rows (rank(A) <= m < n, so columns cannot be made fully
     * mutually orthogonal). Run it on A^T instead and swap U <-> V.
     *
     * A^T is (n x m), giving svd_compact_tall n_rows = n, n_cols = m
     * with n >= m. The resulting decomposition is
     *     A^T = U' @ diag(S') @ Vt'
     * and the SVD of A is
     *     A = (U' @ diag(S') @ Vt')^T = Vt'^T @ diag(S') @ U'^T
     * so we want
     *     U_out  = Vt'^T  (m x k)
     *     S_out  = S'     (k)
     *     Vt_out = U'^T   (k x n)
     * with k = m = min(m, n). */
    const int64_t k = m;
    double* At = (double*)malloc((size_t)n * (size_t)m * sizeof(double));
    double* Up = (double*)malloc((size_t)n * (size_t)k * sizeof(double));
    double* Sp = (double*)malloc((size_t)k * sizeof(double));
    double* Vtp = (double*)malloc((size_t)k * (size_t)m * sizeof(double));
    if (At == NULL || Up == NULL || Sp == NULL || Vtp == NULL) {
        free(At); free(Up); free(Sp); free(Vtp);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    /* Build A^T (n x m). */
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            At[(size_t)j * (size_t)m + (size_t)i] =
                A[(size_t)i * (size_t)n + (size_t)j];
        }
    }
    const n4m_status_t st = svd_compact_tall(At, n, m, Up, Sp, Vtp);
    free(At);
    if (st != N4M_OK) {
        free(Up); free(Sp); free(Vtp);
        return st;
    }
    /* U_out (m x k) = Vtp^T. Vtp is (k x m), so U_out[i, j] = Vtp[j, i]. */
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < k; ++j) {
            U[(size_t)i * (size_t)k + (size_t)j] =
                Vtp[(size_t)j * (size_t)m + (size_t)i];
        }
    }
    /* S_out = Sp. */
    memcpy(S, Sp, (size_t)k * sizeof(double));
    /* Vt_out (k x n) = Up^T. Up is (n x k), so Vt_out[j, i] = Up[i, j]. */
    for (int64_t j = 0; j < k; ++j) {
        for (int64_t i = 0; i < n; ++i) {
            Vt[(size_t)j * (size_t)n + (size_t)i] =
                Up[(size_t)i * (size_t)k + (size_t)j];
        }
    }
    free(Up); free(Sp); free(Vtp);

    /* The svd_compact_tall sign-canonicalisation acted on A^T's U
     * (== A's V) columns. The n4m contract is that we canonicalise on
     * A's U columns. Re-apply the flip here, propagating to Vt. */
    for (int64_t j = 0; j < k; ++j) {
        int64_t argmax = 0;
        double  maxabs = 0.0;
        for (int64_t i = 0; i < m; ++i) {
            const double a = fabs(U[(size_t)i * (size_t)k + (size_t)j]);
            if (a > maxabs) {
                maxabs = a;
                argmax = i;
            }
        }
        const double val = U[(size_t)argmax * (size_t)k + (size_t)j];
        if (val < 0.0) {
            for (int64_t i = 0; i < m; ++i) {
                U[(size_t)i * (size_t)k + (size_t)j] = -U[(size_t)i * (size_t)k + (size_t)j];
            }
            for (int64_t i = 0; i < n; ++i) {
                Vt[(size_t)j * (size_t)n + (size_t)i] = -Vt[(size_t)j * (size_t)n + (size_t)i];
            }
        }
    }
    return unscale_pow2(S, k, e, rank_tol(m, n));
}

n4m_status_t n4m_svd_truncated_dual_wide(const double* A, int64_t m,
                                         int64_t n, int64_t k, double* U,
                                         double* S, double* Vt) {
    if (m < 1 || n < 1 || m > n || k < 1 || k > m) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (A == NULL || U == NULL || S == NULL || Vt == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    int e = 0;
    double* scaled = NULL;
    const n4m_status_t scan =
        scaled_input(A, (size_t)m * (size_t)n, &e, &scaled);
    if (scan != N4M_OK) {
        return scan;
    }
    const double* As = (scaled != NULL) ? scaled : A;

    double* G = (double*)calloc((size_t)m * (size_t)m, sizeof(double));
    double* Ug = (double*)malloc((size_t)m * (size_t)m * sizeof(double));
    double* L = (double*)malloc((size_t)m * sizeof(double));
    double* colbuf = (double*)malloc((size_t)m * sizeof(double));
    if (G == NULL || Ug == NULL || L == NULL || colbuf == NULL) {
        free(G); free(Ug); free(L); free(colbuf); free(scaled);
        return N4M_ERR_OUT_OF_MEMORY;
    }

    for (int64_t c = 0; c < n; ++c) {
        for (int64_t i = 0; i < m; ++i) {
            colbuf[i] = As[(size_t)i * (size_t)n + (size_t)c];
        }
        for (int64_t i = 0; i < m; ++i) {
            const double ai = colbuf[i];
            double* grow = G + (size_t)i * (size_t)m;
            for (int64_t j = i; j < m; ++j) {
                grow[j] += ai * colbuf[j];
            }
        }
    }
    free(colbuf);
    for (int64_t i = 1; i < m; ++i) {
        for (int64_t j = 0; j < i; ++j) {
            G[(size_t)i * (size_t)m + (size_t)j] =
                G[(size_t)j * (size_t)m + (size_t)i];
        }
    }

    const n4m_status_t st = symmetric_eigh(G, m, L, Ug);
    free(G);
    if (st != N4M_OK) {
        free(Ug); free(L); free(scaled);
        return st;
    }
    sort_eigpairs_desc(Ug, L, m);

    for (int64_t comp = 0; comp < k; ++comp) {
        const double lambda = (L[comp] > 0.0) ? L[comp] : 0.0;
        const double sigma = sqrt(lambda);
        S[comp] = sigma;
        for (int64_t i = 0; i < m; ++i) {
            U[(size_t)i * (size_t)k + (size_t)comp] =
                Ug[(size_t)i * (size_t)m + (size_t)comp];
        }
    }

    if (k == 5) {
        const double inv0 = (S[0] > 0.0) ? (1.0 / S[0]) : 0.0;
        const double inv1 = (S[1] > 0.0) ? (1.0 / S[1]) : 0.0;
        const double inv2 = (S[2] > 0.0) ? (1.0 / S[2]) : 0.0;
        const double inv3 = (S[3] > 0.0) ? (1.0 / S[3]) : 0.0;
        const double inv4 = (S[4] > 0.0) ? (1.0 / S[4]) : 0.0;
        double* v0 = Vt;
        double* v1 = Vt + (size_t)n;
        double* v2 = Vt + (size_t)2 * (size_t)n;
        double* v3 = Vt + (size_t)3 * (size_t)n;
        double* v4 = Vt + (size_t)4 * (size_t)n;
        memset(Vt, 0, (size_t)5 * (size_t)n * sizeof(double));
        for (int64_t i = 0; i < m; ++i) {
            const double* row = As + (size_t)i * (size_t)n;
            const double* urow = U + (size_t)i * (size_t)k;
            const double u0 = urow[0];
            const double u1 = urow[1];
            const double u2 = urow[2];
            const double u3 = urow[3];
            const double u4 = urow[4];
            for (int64_t j = 0; j < n; ++j) {
                const double aij = row[j];
                v0[j] += aij * u0;
                v1[j] += aij * u1;
                v2[j] += aij * u2;
                v3[j] += aij * u3;
                v4[j] += aij * u4;
            }
        }
        for (int64_t j = 0; j < n; ++j) {
            v0[j] *= inv0;
            v1[j] *= inv1;
            v2[j] *= inv2;
            v3[j] *= inv3;
            v4[j] *= inv4;
        }
    } else if (k <= 8) {
        double inv_s[8];
        for (int64_t comp = 0; comp < k; ++comp) {
            inv_s[comp] = (S[comp] > 0.0) ? (1.0 / S[comp]) : 0.0;
        }
        for (int64_t j = 0; j < n; ++j) {
            double acc[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            for (int64_t i = 0; i < m; ++i) {
                const double aij = As[(size_t)i * (size_t)n + (size_t)j];
                const double* urow = U + (size_t)i * (size_t)k;
                for (int64_t comp = 0; comp < k; ++comp) {
                    acc[comp] += aij * urow[comp];
                }
            }
            for (int64_t comp = 0; comp < k; ++comp) {
                Vt[(size_t)comp * (size_t)n + (size_t)j] =
                    acc[comp] * inv_s[comp];
            }
        }
    } else {
        for (int64_t comp = 0; comp < k; ++comp) {
            double* vrow = Vt + (size_t)comp * (size_t)n;
            if (S[comp] > 0.0) {
                const double inv_sigma = 1.0 / S[comp];
                for (int64_t j = 0; j < n; ++j) {
                    double acc = 0.0;
                    for (int64_t i = 0; i < m; ++i) {
                        acc += As[(size_t)i * (size_t)n + (size_t)j] *
                               U[(size_t)i * (size_t)k + (size_t)comp];
                    }
                    vrow[j] = acc * inv_sigma;
                }
            } else {
                for (int64_t j = 0; j < n; ++j) {
                    vrow[j] = 0.0;
                }
            }
        }
    }

    for (int64_t comp = 0; comp < k; ++comp) {
        double* vrow = Vt + (size_t)comp * (size_t)n;

        int64_t argmax = 0;
        double maxabs = 0.0;
        for (int64_t i = 0; i < m; ++i) {
            const double a = fabs(U[(size_t)i * (size_t)k + (size_t)comp]);
            if (a > maxabs) {
                maxabs = a;
                argmax = i;
            }
        }
        if (U[(size_t)argmax * (size_t)k + (size_t)comp] < 0.0) {
            for (int64_t i = 0; i < m; ++i) {
                U[(size_t)i * (size_t)k + (size_t)comp] =
                    -U[(size_t)i * (size_t)k + (size_t)comp];
            }
            for (int64_t j = 0; j < n; ++j) {
                vrow[j] = -vrow[j];
            }
        }
    }

    free(Ug);
    free(L);
    free(scaled);
    return unscale_pow2(S, k, e, rank_tol(m, n));
}

n4m_status_t n4m_svd_truncated_tall_gram(const double* A, int64_t m,
                                         int64_t n, int64_t k, double* U,
                                         double* S, double* Vt) {
    if (m < 1 || n < 1 || m < n || k < 1 || k > n) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (A == NULL || U == NULL || S == NULL || Vt == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    int e = 0;
    double* scaled = NULL;
    const n4m_status_t scan =
        scaled_input(A, (size_t)m * (size_t)n, &e, &scaled);
    if (scan != N4M_OK) {
        return scan;
    }
    const double* As = (scaled != NULL) ? scaled : A;

    double* G = (double*)calloc((size_t)n * (size_t)n, sizeof(double));
    double* V = (double*)malloc((size_t)n * (size_t)n * sizeof(double));
    double* L = (double*)malloc((size_t)n * sizeof(double));
    if (G == NULL || V == NULL || L == NULL) {
        free(G); free(V); free(L); free(scaled);
        return N4M_ERR_OUT_OF_MEMORY;
    }

    for (int64_t r = 0; r < m; ++r) {
        const double* row = As + (size_t)r * (size_t)n;
        for (int64_t i = 0; i < n; ++i) {
            const double ai = row[i];
            double* grow = G + (size_t)i * (size_t)n;
            for (int64_t j = i; j < n; ++j) {
                grow[j] += ai * row[j];
            }
        }
    }
    for (int64_t i = 1; i < n; ++i) {
        for (int64_t j = 0; j < i; ++j) {
            G[(size_t)i * (size_t)n + (size_t)j] =
                G[(size_t)j * (size_t)n + (size_t)i];
        }
    }

    const n4m_status_t st = symmetric_eigh(G, n, L, V);
    free(G);
    if (st != N4M_OK) {
        free(V); free(L); free(scaled);
        return st;
    }
    sort_eigpairs_desc(V, L, n);

    for (int64_t comp = 0; comp < k; ++comp) {
        const double lambda = (L[comp] > 0.0) ? L[comp] : 0.0;
        const double sigma = sqrt(lambda);
        const double inv_sigma = (sigma > 0.0) ? (1.0 / sigma) : 0.0;
        S[comp] = sigma;

        double* vrow = Vt + (size_t)comp * (size_t)n;
        for (int64_t j = 0; j < n; ++j) {
            vrow[j] = V[(size_t)j * (size_t)n + (size_t)comp];
        }

        for (int64_t r = 0; r < m; ++r) {
            const double* row = As + (size_t)r * (size_t)n;
            double acc = 0.0;
            for (int64_t j = 0; j < n; ++j) {
                acc += row[j] * vrow[j];
            }
            U[(size_t)r * (size_t)k + (size_t)comp] = acc * inv_sigma;
        }

        int64_t argmax = 0;
        double maxabs = 0.0;
        for (int64_t r = 0; r < m; ++r) {
            const double a = fabs(U[(size_t)r * (size_t)k + (size_t)comp]);
            if (a > maxabs) {
                maxabs = a;
                argmax = r;
            }
        }
        if (U[(size_t)argmax * (size_t)k + (size_t)comp] < 0.0) {
            for (int64_t r = 0; r < m; ++r) {
                U[(size_t)r * (size_t)k + (size_t)comp] =
                    -U[(size_t)r * (size_t)k + (size_t)comp];
            }
            for (int64_t j = 0; j < n; ++j) {
                vrow[j] = -vrow[j];
            }
        }
    }

    free(V);
    free(L);
    free(scaled);
    return unscale_pow2(S, k, e, rank_tol(m, n));
}

static uint64_t splitmix64_next(uint64_t x) {
    x += UINT64_C(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

static double deterministic_omega(int64_t row, int64_t col) {
    uint64_t x = UINT64_C(0x6a09e667f3bcc909);
    x ^= (uint64_t)(row + 1) * UINT64_C(0xbf58476d1ce4e5b9);
    x ^= (uint64_t)(col + 1) * UINT64_C(0x94d049bb133111eb);
    x = splitmix64_next(x);
    const double unit = (double)(x >> 11) * (1.0 / 9007199254740992.0);
    return 2.0 * unit - 1.0;
}

static double column_norm2(const double* col, int64_t rows, int64_t cols) {
    double norm2 = 0.0;
    for (int64_t i = 0; i < rows; ++i) {
        const double v = col[(size_t)i * (size_t)cols];
        norm2 += v * v;
    }
    return norm2;
}

/* Gram-Schmidt on the columns of the row-major (rows x cols) matrix A. A
 * column that loses more than half its squared norm to the projections is
 * projected a second time (Kahan's "twice is enough"); a column whose
 * residual is at rounding level relative to its original norm is linearly
 * dependent on the previous ones (rank-deficient A, sketch wider than the
 * rank) and is zeroed instead of normalising rounding noise. */
static void orthonormalize_columns(double* A, int64_t rows, int64_t cols) {
    const double dependent_rtol2 =
        (double)rows * N4M_SVD_EPS * N4M_SVD_EPS;
    for (int64_t c = 0; c < cols; ++c) {
        double* col_c = A + (size_t)c;
        const double norm2_in = column_norm2(col_c, rows, cols);
        double norm2 = norm2_in;
        for (int pass = 0; pass < 2; ++pass) {
            for (int64_t prev = 0; prev < c; ++prev) {
                const double* col_prev = A + (size_t)prev;
                double dot = 0.0;
                for (int64_t i = 0; i < rows; ++i) {
                    dot += col_c[(size_t)i * (size_t)cols] *
                           col_prev[(size_t)i * (size_t)cols];
                }
                for (int64_t i = 0; i < rows; ++i) {
                    col_c[(size_t)i * (size_t)cols] -=
                        dot * col_prev[(size_t)i * (size_t)cols];
                }
            }
            const double before = norm2;
            norm2 = column_norm2(col_c, rows, cols);
            if (c == 0 || norm2 >= 0.5 * before) {
                break;
            }
        }
        if (!(norm2 > N4M_SVD_ZERO_TOL) || !isfinite(norm2) ||
            norm2 <= dependent_rtol2 * norm2_in) {
            for (int64_t i = 0; i < rows; ++i) {
                col_c[(size_t)i * (size_t)cols] = 0.0;
            }
            continue;
        }
        const double inv = 1.0 / sqrt(norm2);
        for (int64_t i = 0; i < rows; ++i) {
            col_c[(size_t)i * (size_t)cols] *= inv;
        }
    }
}

n4m_status_t n4m_svd_truncated_randomized(const double* A, int64_t m,
                                          int64_t n, int64_t k,
                                          double* U, double* S, double* Vt) {
    const int64_t k_full = (m < n) ? m : n;
    if (m < 1 || n < 1 || k < 1 || k > k_full) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (A == NULL || U == NULL || S == NULL || Vt == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    int e = 0;
    double* scaled = NULL;
    const n4m_status_t scan =
        scaled_input(A, (size_t)m * (size_t)n, &e, &scaled);
    if (scan != N4M_OK) {
        return scan;
    }
    const double* As = (scaled != NULL) ? scaled : A;

    int64_t l = k + 8;
    if (l > k_full) l = k_full;
    if (l < k) l = k;
    const int power_iter = 2;

    double* Omega = (double*)malloc((size_t)n * (size_t)l * sizeof(double));
    double* Q = (double*)malloc((size_t)m * (size_t)l * sizeof(double));
    double* Z = (double*)malloc((size_t)n * (size_t)l * sizeof(double));
    double* B = (double*)malloc((size_t)l * (size_t)n * sizeof(double));
    double* Ub = (double*)malloc((size_t)l * (size_t)l * sizeof(double));
    double* Sb = (double*)malloc((size_t)l * sizeof(double));
    double* Vtb = (double*)malloc((size_t)l * (size_t)n * sizeof(double));
    if (Omega == NULL || Q == NULL || Z == NULL || B == NULL ||
        Ub == NULL || Sb == NULL || Vtb == NULL) {
        free(Omega); free(Q); free(Z); free(B); free(Ub); free(Sb); free(Vtb);
        free(scaled);
        return N4M_ERR_OUT_OF_MEMORY;
    }

    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < l; ++j) {
            Omega[(size_t)i * (size_t)l + (size_t)j] =
                deterministic_omega(i, j);
        }
    }

    memset(Q, 0, (size_t)m * (size_t)l * sizeof(double));
    for (int64_t i = 0; i < m; ++i) {
        const double* row = As + (size_t)i * (size_t)n;
        double* qrow = Q + (size_t)i * (size_t)l;
        for (int64_t j = 0; j < n; ++j) {
            const double aij = row[j];
            const double* omegaj = Omega + (size_t)j * (size_t)l;
            for (int64_t c = 0; c < l; ++c) {
                qrow[c] += aij * omegaj[c];
            }
        }
    }
    orthonormalize_columns(Q, m, l);

    for (int iter = 0; iter < power_iter; ++iter) {
        memset(Z, 0, (size_t)n * (size_t)l * sizeof(double));
        for (int64_t i = 0; i < m; ++i) {
            const double* row = As + (size_t)i * (size_t)n;
            const double* qrow = Q + (size_t)i * (size_t)l;
            for (int64_t j = 0; j < n; ++j) {
                double* zrow = Z + (size_t)j * (size_t)l;
                const double aij = row[j];
                for (int64_t c = 0; c < l; ++c) {
                    zrow[c] += aij * qrow[c];
                }
            }
        }
        orthonormalize_columns(Z, n, l);

        memset(Q, 0, (size_t)m * (size_t)l * sizeof(double));
        for (int64_t i = 0; i < m; ++i) {
            const double* row = As + (size_t)i * (size_t)n;
            double* qrow = Q + (size_t)i * (size_t)l;
            for (int64_t j = 0; j < n; ++j) {
                const double aij = row[j];
                const double* zrow = Z + (size_t)j * (size_t)l;
                for (int64_t c = 0; c < l; ++c) {
                    qrow[c] += aij * zrow[c];
                }
            }
        }
        orthonormalize_columns(Q, m, l);
    }

    memset(B, 0, (size_t)l * (size_t)n * sizeof(double));
    for (int64_t i = 0; i < m; ++i) {
        const double* row = As + (size_t)i * (size_t)n;
        const double* qrow = Q + (size_t)i * (size_t)l;
        for (int64_t c = 0; c < l; ++c) {
            double* brow = B + (size_t)c * (size_t)n;
            const double qic = qrow[c];
            for (int64_t j = 0; j < n; ++j) {
                brow[j] += qic * row[j];
            }
        }
    }

    const n4m_status_t st = n4m_svd_compact(B, l, n, Ub, Sb, Vtb);
    if (st != N4M_OK) {
        free(Omega); free(Q); free(Z); free(B); free(Ub); free(Sb); free(Vtb);
        free(scaled);
        return st;
    }

    for (int64_t comp = 0; comp < k; ++comp) {
        S[comp] = Sb[comp];
        for (int64_t i = 0; i < m; ++i) {
            const double* qrow = Q + (size_t)i * (size_t)l;
            double acc = 0.0;
            for (int64_t c = 0; c < l; ++c) {
                acc += qrow[c] * Ub[(size_t)c * (size_t)l + (size_t)comp];
            }
            U[(size_t)i * (size_t)k + (size_t)comp] = acc;
        }
        memcpy(Vt + (size_t)comp * (size_t)n,
               Vtb + (size_t)comp * (size_t)n,
               (size_t)n * sizeof(double));

        int64_t argmax = 0;
        double maxabs = 0.0;
        for (int64_t i = 0; i < m; ++i) {
            const double a = fabs(U[(size_t)i * (size_t)k + (size_t)comp]);
            if (a > maxabs) {
                maxabs = a;
                argmax = i;
            }
        }
        if (U[(size_t)argmax * (size_t)k + (size_t)comp] < 0.0) {
            for (int64_t i = 0; i < m; ++i) {
                U[(size_t)i * (size_t)k + (size_t)comp] =
                    -U[(size_t)i * (size_t)k + (size_t)comp];
            }
            double* vrow = Vt + (size_t)comp * (size_t)n;
            for (int64_t j = 0; j < n; ++j) {
                vrow[j] = -vrow[j];
            }
        }
    }

    free(Omega); free(Q); free(Z); free(B); free(Ub); free(Sb); free(Vtb);
        free(scaled);
    return unscale_pow2(S, k, e, rank_tol(m, n));
}

static n4m_status_t svd_compact_tall(double* A, int64_t m, int64_t n,
                                      double* U, double* S, double* Vt) {
    /* Precondition: m >= n >= 1. */
    const int64_t k = n;

    /* V is accumulated as an n x n matrix; we later return the first
     * k rows of V^T. Allocate scratch for V row-major. */
    double* V = (double*)malloc((size_t)n * (size_t)n * sizeof(double));
    if (V == NULL) {
        return N4M_ERR_OUT_OF_MEMORY;
    }
    /* Initialise V to the identity. */
    memset(V, 0, (size_t)n * (size_t)n * sizeof(double));
    for (int64_t i = 0; i < n; ++i) {
        V[(size_t)i * (size_t)n + (size_t)i] = 1.0;
    }

    /* Jacobi sweeps. */
    const double floor2 = ldexp(1.0, 2 * N4M_SVD_FLOOR_EXP);
    int converged = 0;
    for (int sweep = 0; sweep < N4M_SVD_MAX_SWEEPS && !converged; ++sweep) {
        int rotated = 0;
        for (int64_t p = 0; p < n - 1; ++p) {
            for (int64_t q = p + 1; q < n; ++q) {
                /* Compute three inner products: alpha = a_p^T a_p,
                 *                                beta  = a_q^T a_q,
                 *                                gamma = a_p^T a_q. */
                double alpha = 0.0;
                double beta  = 0.0;
                double gamma = 0.0;
                for (int64_t i = 0; i < m; ++i) {
                    const double ap = A_at(A, n, i, p);
                    const double aq = A_at(A, n, i, q);
                    alpha += ap * ap;
                    beta  += aq * aq;
                    gamma += ap * aq;
                }

                /* Convergence test for this pair. A column below the
                 * resolution floor takes no rotation (see
                 * N4M_SVD_FLOOR_EXP). */
                /* sqrt(alpha * beta) unless that product underflows
                 * (weak columns), where it would zero the threshold. */
                const double ab = alpha * beta;
                const double tol =
                    N4M_SVD_EPS *
                    ((ab >= DBL_MIN) ? sqrt(ab) : sqrt(alpha) * sqrt(beta));
                if (fabs(gamma) <= tol || alpha < floor2 || beta < floor2) {
                    continue;
                }
                rotated = 1;

                /* Compute the Givens rotation diagonalising the 2x2
                 * Gram block [[alpha, gamma], [gamma, beta]].
                 *
                 * Applying R = [[c, -s], [s, c]] to the two columns
                 *   a_p' =  c*a_p + s*a_q
                 *   a_q' = -s*a_p + c*a_q
                 * leaves new_gamma = (beta-alpha)*c*s + (c^2-s^2)*gamma.
                 * Setting new_gamma = 0 with t = s/c gives the quadratic
                 *   t^2 + 2*tau*t - 1 = 0,  tau = (alpha - beta) / (2*gamma)
                 * whose smaller-magnitude root is
                 *   t = sign(tau) / (|tau| + sqrt(1 + tau^2)).
                 * Reference: Golub & Van Loan, "Matrix Computations"
                 * 4e, Algorithm 8.4.3. */
                const double tau = (alpha - beta) / (2.0 * gamma);
                double t;
                if (fabs(tau) > 0x1p500) {
                    t = 0.5 / tau; /* tau^2 would overflow; t = 1/(2 tau) */
                } else if (tau >= 0.0) {
                    t = 1.0 / (tau + sqrt(1.0 + tau * tau));
                } else {
                    t = 1.0 / (tau - sqrt(1.0 + tau * tau));
                }
                const double c = 1.0 / sqrt(1.0 + t * t);
                const double s = t * c;

                /* Apply rotation to columns p and q of A:
                 *   [a_p', a_q'] = [a_p, a_q] @ [[c, -s], [s, c]]
                 * i.e. a_p_new[i] =  c*a_p[i] + s*a_q[i]
                 *      a_q_new[i] = -s*a_p[i] + c*a_q[i] */
                for (int64_t i = 0; i < m; ++i) {
                    const double ap = A_at(A, n, i, p);
                    const double aq = A_at(A, n, i, q);
                    A_set(A, n, i, p,  c * ap + s * aq);
                    A_set(A, n, i, q, -s * ap + c * aq);
                }
                /* Apply the same rotation to columns p and q of V to
                 * accumulate the right factor. */
                for (int64_t i = 0; i < n; ++i) {
                    const double vp = V[(size_t)i * (size_t)n + (size_t)p];
                    const double vq = V[(size_t)i * (size_t)n + (size_t)q];
                    V[(size_t)i * (size_t)n + (size_t)p] =  c * vp + s * vq;
                    V[(size_t)i * (size_t)n + (size_t)q] = -s * vp + c * vq;
                }
            }
        }
        if (!rotated) {
            converged = 1;
            break;
        }
    }

    if (!converged) {
        free(V);
        return N4M_ERR_CONVERGENCE_FAILED;
    }

    /* After convergence, each column j of A is sigma_j * u_j where
     * sigma_j is the singular value and u_j the unit-norm left
     * singular vector. The corresponding right singular vector is
     * column j of V. Extract sigma_j as the column norm and u_j by
     * dividing the column by sigma_j (or zero-fill if sigma_j == 0). */
    double* sigmas = (double*)malloc((size_t)n * sizeof(double));
    if (sigmas == NULL) {
        free(V);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    for (int64_t j = 0; j < n; ++j) {
        double s2 = 0.0;
        for (int64_t i = 0; i < m; ++i) {
            const double v = A_at(A, n, i, j);
            s2 += v * v;
        }
        sigmas[j] = (s2 < floor2) ? 0.0 : sqrt(s2);
    }

    /* Build the permutation that sorts sigmas in descending order. */
    int64_t* perm = (int64_t*)malloc((size_t)n * sizeof(int64_t));
    if (perm == NULL) {
        free(V);
        free(sigmas);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    for (int64_t i = 0; i < n; ++i) perm[i] = i;
    /* Simple insertion sort by descending sigma. n is small (<= a few
     * hundred for parity fixtures); O(n^2) is fine. */
    for (int64_t i = 1; i < n; ++i) {
        const int64_t key = perm[i];
        const double  ks  = sigmas[key];
        int64_t j = i - 1;
        while (j >= 0 && sigmas[perm[j]] < ks) {
            perm[j + 1] = perm[j];
            --j;
        }
        perm[j + 1] = key;
    }

    /* Emit U (m x k) and S (k). The j-th output column comes from the
     * perm[j]-th column of A scaled by 1/sigma. */
    for (int64_t j = 0; j < k; ++j) {
        const int64_t pj = perm[j];
        const double  sj = sigmas[pj];
        S[j] = sj;
        if (sj > 0.0) {
            const double inv = 1.0 / sj;
            for (int64_t i = 0; i < m; ++i) {
                U[(size_t)i * (size_t)k + (size_t)j] = A_at(A, n, i, pj) * inv;
            }
        } else {
            for (int64_t i = 0; i < m; ++i) {
                U[(size_t)i * (size_t)k + (size_t)j] = 0.0;
            }
        }
    }

    /* Emit Vt (k x n) by transposing the relevant columns of V. */
    for (int64_t j = 0; j < k; ++j) {
        const int64_t pj = perm[j];
        for (int64_t i = 0; i < n; ++i) {
            Vt[(size_t)j * (size_t)n + (size_t)i] =
                V[(size_t)i * (size_t)n + (size_t)pj];
        }
    }

    /* Sign canonicalisation: for each component, flip sign so the
     * largest-absolute-value element of U[:, j] is positive. Flip
     * Vt[j, :] in lockstep so the product U @ diag(S) @ Vt is
     * preserved. Ties are broken by smallest index (argmax returns
     * the first occurrence). */
    for (int64_t j = 0; j < k; ++j) {
        int64_t argmax = 0;
        double  maxabs = 0.0;
        for (int64_t i = 0; i < m; ++i) {
            const double a = fabs(U[(size_t)i * (size_t)k + (size_t)j]);
            if (a > maxabs) {
                maxabs = a;
                argmax = i;
            }
        }
        const double val = U[(size_t)argmax * (size_t)k + (size_t)j];
        if (val < 0.0) {
            for (int64_t i = 0; i < m; ++i) {
                U[(size_t)i * (size_t)k + (size_t)j] = -U[(size_t)i * (size_t)k + (size_t)j];
            }
            for (int64_t i = 0; i < n; ++i) {
                Vt[(size_t)j * (size_t)n + (size_t)i] = -Vt[(size_t)j * (size_t)n + (size_t)i];
            }
        }
    }

    free(V);
    free(sigmas);
    free(perm);
    return N4M_OK;
}
