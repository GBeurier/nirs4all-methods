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
 */

#include "svd.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Jacobi sweep limit. The classical bound is O(log n) sweeps for
 * well-conditioned matrices; we use 30 to comfortably cover all parity
 * fixtures while still bailing out on pathological inputs. */
#define C4A_SVD_MAX_SWEEPS 60

/* Convergence tolerance scale. eps = 4 * DBL_EPSILON keeps the
 * orthogonalisation criterion within ~2 bits of the machine precision. */
static const double C4A_SVD_EPS = 4.0 * DBL_EPSILON;

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
static c4a_status_t svd_compact_tall(double* A, int64_t m, int64_t n,
                                      double* U, double* S, double* Vt);

c4a_status_t c4a_svd_compact(double* A, int64_t m, int64_t n,
                              double* U, double* S, double* Vt) {
    if (m < 1 || n < 1) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (A == NULL || U == NULL || S == NULL || Vt == NULL) {
        return C4A_ERR_NULL_POINTER;
    }

    if (m >= n) {
        return svd_compact_tall(A, m, n, U, S, Vt);
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
        return C4A_ERR_OUT_OF_MEMORY;
    }
    /* Build A^T (n x m). */
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            At[(size_t)j * (size_t)m + (size_t)i] =
                A[(size_t)i * (size_t)n + (size_t)j];
        }
    }
    const c4a_status_t st = svd_compact_tall(At, n, m, Up, Sp, Vtp);
    free(At);
    if (st != C4A_OK) {
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
     * (== A's V) columns. The c4a contract is that we canonicalise on
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
    return C4A_OK;
}

static c4a_status_t svd_compact_tall(double* A, int64_t m, int64_t n,
                                      double* U, double* S, double* Vt) {
    /* Precondition: m >= n >= 1. */
    const int64_t k = n;

    /* V is accumulated as an n x n matrix; we later return the first
     * k rows of V^T. Allocate scratch for V row-major. */
    double* V = (double*)malloc((size_t)n * (size_t)n * sizeof(double));
    if (V == NULL) {
        return C4A_ERR_OUT_OF_MEMORY;
    }
    /* Initialise V to the identity. */
    memset(V, 0, (size_t)n * (size_t)n * sizeof(double));
    for (int64_t i = 0; i < n; ++i) {
        V[(size_t)i * (size_t)n + (size_t)i] = 1.0;
    }

    /* Jacobi sweeps. */
    int converged = 0;
    for (int sweep = 0; sweep < C4A_SVD_MAX_SWEEPS && !converged; ++sweep) {
        converged = 1;
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

                /* Convergence test for this pair. */
                const double tol = C4A_SVD_EPS * sqrt(alpha * beta);
                if (fabs(gamma) <= tol) {
                    continue;
                }
                converged = 0;

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
                if (tau >= 0.0) {
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
    }

    if (!converged) {
        free(V);
        return C4A_ERR_CONVERGENCE_FAILED;
    }

    /* After convergence, each column j of A is sigma_j * u_j where
     * sigma_j is the singular value and u_j the unit-norm left
     * singular vector. The corresponding right singular vector is
     * column j of V. Extract sigma_j as the column norm and u_j by
     * dividing the column by sigma_j (or zero-fill if sigma_j == 0). */
    double* sigmas = (double*)malloc((size_t)n * sizeof(double));
    if (sigmas == NULL) {
        free(V);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    for (int64_t j = 0; j < n; ++j) {
        double s2 = 0.0;
        for (int64_t i = 0; i < m; ++i) {
            const double v = A_at(A, n, i, j);
            s2 += v * v;
        }
        sigmas[j] = sqrt(s2);
    }

    /* Build the permutation that sorts sigmas in descending order. */
    int64_t* perm = (int64_t*)malloc((size_t)n * sizeof(int64_t));
    if (perm == NULL) {
        free(V);
        free(sigmas);
        return C4A_ERR_OUT_OF_MEMORY;
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
    return C4A_OK;
}
