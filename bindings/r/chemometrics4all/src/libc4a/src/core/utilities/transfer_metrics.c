/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Transfer metrics — reference implementation.
 *
 * Algorithm map (mirrors nirs4all/analysis/transfer_metrics.py):
 *
 *   1. Center each dataset by per-column mean (no scaling).
 *   2. Build the p × p covariance matrix `C = Xc^T Xc` (rank-revealing form;
 *      we never form `Xc Xc^T` because p << n is not guaranteed and the
 *      symmetric Jacobi solver below scales as O(p^3) per sweep, which is
 *      fine for the typical NIR p in [50, 2000]).
 *      For p > n the data lives in the column space of Xc^T — the residual
 *      eigenvalues are zero. The Jacobi sweep exposes them as ~0; we drop
 *      them when picking `r_use`.
 *   3. Diagonalise C with the classical (cyclic) Jacobi rotation method to
 *      tolerance MACHINE_TOL × ||C||_F. The result is V (p × p, columns are
 *      eigenvectors) and λ (length p). Sort by descending λ.
 *      Truncate to `n_components`, then take `r_use = min(r_src, r_tgt)`
 *      between the two datasets so the downstream metrics live in a shared
 *      rank.
 *      Scores: `Z = Xc · V[:, :r_use]`.
 *      Explained-variance ratio: sum(λ_kept) / sum(λ_all).
 *   4. Metrics — see transfer_metrics.h banner.
 *
 * For Procrustes superposition we compute the SVD of `M = Z1c^T Z2c` via the
 * normal-equation route: diagonalise `M^T M` with Jacobi to get V and σ^2,
 * then `R = U V^T` with `U = M V Σ^{-1}` (right-multiplying by the pseudo-
 * inverse handles rank deficiency).
 *
 * For Grassmann distance we diagonalise `(U_s^T U_t)^T (U_s^T U_t)` to get the
 * squared cosines of the principal angles, clip to [0, 1], and compute
 * `theta_i = arccos(sqrt(s_i))` (matches scipy.linalg.subspace_angles).
 *
 * For Trustworthiness we replicate sklearn.manifold.trustworthiness with the
 * "precomputed" path elided — we compute Euclidean distances row-by-row in
 * the PCA space. Ranks are stored as int32; tie-breaks follow numpy.argsort
 * (stable mergesort by ascending key, ties broken by ascending index).
 *
 * Internal scratch allocation is one bulk request per call; no allocations
 * happen inside the Jacobi sweep or the trustworthiness loops.
 */

#include "transfer_metrics.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Allow callers to override the sweep budget for tight tolerance regimes.
 * 100 sweeps already gives full convergence for any well-scaled symmetric
 * matrix encountered in practice (the cyclic Jacobi method has cubic
 * convergence once the off-diagonal mass shrinks below ~MACHINE_TOL). */
#ifndef C4A_TRANSFER_JACOBI_MAX_SWEEPS
#  define C4A_TRANSFER_JACOBI_MAX_SWEEPS 100
#endif

/* Fixed subsample size for the spread-distance pairwise step, matching the
 * Python reference. */
#define C4A_TRANSFER_SPREAD_SUBSAMPLE 100

/* -------------------------------------------------------------------------- */
/*  Small helpers                                                              */
/* -------------------------------------------------------------------------- */

static double nan_value(void) {
    /* Portable quiet-NaN producer. */
    return (double)NAN;
}

/* Compute per-column mean and write it into `mean[0..p-1]`. */
static void column_mean(const double* X, int64_t n, int64_t p, double* mean) {
    for (int64_t j = 0; j < p; ++j) {
        mean[j] = 0.0;
    }
    for (int64_t i = 0; i < n; ++i) {
        const double* row = X + (size_t)i * (size_t)p;
        for (int64_t j = 0; j < p; ++j) {
            mean[j] += row[j];
        }
    }
    const double inv = (n > 0) ? 1.0 / (double)n : 0.0;
    for (int64_t j = 0; j < p; ++j) {
        mean[j] *= inv;
    }
}

/* Write `Xc[i, j] = X[i, j] - mean[j]`. */
static void center_into(const double* X, int64_t n, int64_t p,
                        const double* mean, double* Xc) {
    for (int64_t i = 0; i < n; ++i) {
        const double* row_in  = X  + (size_t)i * (size_t)p;
        double*       row_out = Xc + (size_t)i * (size_t)p;
        for (int64_t j = 0; j < p; ++j) {
            row_out[j] = row_in[j] - mean[j];
        }
    }
}

/* Compute the symmetric Gram matrix `G = Xc^T Xc` into a p × p row-major
 * dense buffer. */
static void gram_xtx(const double* Xc, int64_t n, int64_t p, double* G) {
    /* Zero out the upper triangle (we fill only j >= i then mirror). */
    for (int64_t i = 0; i < p; ++i) {
        for (int64_t j = i; j < p; ++j) {
            double s = 0.0;
            for (int64_t k = 0; k < n; ++k) {
                s += Xc[(size_t)k * (size_t)p + (size_t)i] *
                     Xc[(size_t)k * (size_t)p + (size_t)j];
            }
            G[(size_t)i * (size_t)p + (size_t)j] = s;
            if (j != i) {
                G[(size_t)j * (size_t)p + (size_t)i] = s;
            }
        }
    }
}

/* -------------------------------------------------------------------------- */
/*  Classical cyclic Jacobi diagonalisation                                    */
/* -------------------------------------------------------------------------- */

/* Diagonalise the symmetric p × p matrix A in place via cyclic Jacobi rotations.
 * On exit:
 *   - A's diagonal holds the eigenvalues (off-diagonal entries are negligible
 *     but not guaranteed to be exactly zero).
 *   - V is the matrix of eigenvectors (p × p, columns are eigenvectors).
 *
 * Returns C4A_OK on convergence within C4A_TRANSFER_JACOBI_MAX_SWEEPS sweeps,
 * C4A_ERR_NUMERICAL_FAILURE otherwise. p == 0 returns OK with no work. */
static c4a_status_t jacobi_eig_sym(double* A, int64_t p, double* V) {
    if (p <= 0) {
        return C4A_OK;
    }
    /* Initialise V to identity. */
    for (int64_t i = 0; i < p; ++i) {
        for (int64_t j = 0; j < p; ++j) {
            V[(size_t)i * (size_t)p + (size_t)j] = (i == j) ? 1.0 : 0.0;
        }
    }
    if (p == 1) {
        return C4A_OK;
    }

    /* Tolerance: machine eps times Frobenius norm of A. */
    double frob_sq = 0.0;
    for (int64_t i = 0; i < p; ++i) {
        for (int64_t j = 0; j < p; ++j) {
            const double a = A[(size_t)i * (size_t)p + (size_t)j];
            frob_sq += a * a;
        }
    }
    const double frob = sqrt(frob_sq);
    /* Combined absolute + relative threshold to make zero matrices trivial. */
    const double tol = 1e-13 * (frob > 0.0 ? frob : 1.0);

    for (int sweep = 0; sweep < C4A_TRANSFER_JACOBI_MAX_SWEEPS; ++sweep) {
        /* Sum of squares of off-diagonal entries. */
        double off = 0.0;
        for (int64_t i = 0; i < p; ++i) {
            for (int64_t j = i + 1; j < p; ++j) {
                const double a = A[(size_t)i * (size_t)p + (size_t)j];
                off += a * a;
            }
        }
        if (sqrt(off) <= tol) {
            return C4A_OK;
        }
        /* Cyclic sweep over upper triangle. */
        for (int64_t p_idx = 0; p_idx < p - 1; ++p_idx) {
            for (int64_t q_idx = p_idx + 1; q_idx < p; ++q_idx) {
                const double apq = A[(size_t)p_idx * (size_t)p + (size_t)q_idx];
                if (apq == 0.0) {
                    continue;
                }
                const double app = A[(size_t)p_idx * (size_t)p + (size_t)p_idx];
                const double aqq = A[(size_t)q_idx * (size_t)p + (size_t)q_idx];
                double theta = (aqq - app) / (2.0 * apq);
                double t;
                if (fabs(theta) > 1e150) {
                    t = 1.0 / (2.0 * theta);
                } else {
                    const double sgn = (theta >= 0.0) ? 1.0 : -1.0;
                    t = sgn / (fabs(theta) + sqrt(theta * theta + 1.0));
                }
                const double c = 1.0 / sqrt(t * t + 1.0);
                const double s = t * c;
                const double tau = s / (1.0 + c);

                /* Update A: rows/cols p_idx and q_idx. */
                A[(size_t)p_idx * (size_t)p + (size_t)p_idx] = app - t * apq;
                A[(size_t)q_idx * (size_t)p + (size_t)q_idx] = aqq + t * apq;
                A[(size_t)p_idx * (size_t)p + (size_t)q_idx] = 0.0;
                A[(size_t)q_idx * (size_t)p + (size_t)p_idx] = 0.0;

                for (int64_t r = 0; r < p; ++r) {
                    if (r == p_idx || r == q_idx) continue;
                    const double arp = A[(size_t)r * (size_t)p + (size_t)p_idx];
                    const double arq = A[(size_t)r * (size_t)p + (size_t)q_idx];
                    A[(size_t)r * (size_t)p + (size_t)p_idx] = arp - s * (arq + tau * arp);
                    A[(size_t)r * (size_t)p + (size_t)q_idx] = arq + s * (arp - tau * arq);
                    A[(size_t)p_idx * (size_t)p + (size_t)r] =
                        A[(size_t)r * (size_t)p + (size_t)p_idx];
                    A[(size_t)q_idx * (size_t)p + (size_t)r] =
                        A[(size_t)r * (size_t)p + (size_t)q_idx];
                }
                /* Update V columns p_idx and q_idx. */
                for (int64_t r = 0; r < p; ++r) {
                    const double vrp = V[(size_t)r * (size_t)p + (size_t)p_idx];
                    const double vrq = V[(size_t)r * (size_t)p + (size_t)q_idx];
                    V[(size_t)r * (size_t)p + (size_t)p_idx] = vrp - s * (vrq + tau * vrp);
                    V[(size_t)r * (size_t)p + (size_t)q_idx] = vrq + s * (vrp - tau * vrq);
                }
            }
        }
    }
    return C4A_ERR_NUMERICAL_FAILURE;
}

/* Sort eigenvalues (in `eigs[0..p-1]`, taken from A's diagonal after Jacobi)
 * in descending order along with corresponding eigenvector columns in V.
 * Uses a simple selection sort — p is small for typical NIR data. */
static void sort_eigen_desc(double* eigs, double* V, int64_t p) {
    for (int64_t i = 0; i < p - 1; ++i) {
        int64_t imax = i;
        for (int64_t j = i + 1; j < p; ++j) {
            if (eigs[j] > eigs[imax]) {
                imax = j;
            }
        }
        if (imax != i) {
            const double tmp = eigs[i];
            eigs[i] = eigs[imax];
            eigs[imax] = tmp;
            for (int64_t r = 0; r < p; ++r) {
                const double t = V[(size_t)r * (size_t)p + (size_t)i];
                V[(size_t)r * (size_t)p + (size_t)i] =
                    V[(size_t)r * (size_t)p + (size_t)imax];
                V[(size_t)r * (size_t)p + (size_t)imax] = t;
            }
        }
    }
}

/* -------------------------------------------------------------------------- */
/*  PCA — center, eigendecompose Xc^T Xc, project                              */
/* -------------------------------------------------------------------------- */

typedef struct pca_result_t {
    int64_t n;            /* sample count */
    int64_t p;            /* original feature count */
    int64_t r;            /* number of retained components (r <= n_components, r <= p) */
    double  evr;          /* explained-variance ratio (over the r components) */
    double* Z;            /* scores  (n × r), row-major */
    double* U;            /* loadings (p × r), row-major (columns of original V) */
} pca_result_t;

/* Run PCA on a centered matrix `Xc` of shape (n, p), keep up to `n_components`
 * components, and write into `out`. Allocates Z and U via malloc — the caller
 * frees them via `pca_result_free`. Returns C4A_OK on success.
 *
 * Implementation note: we form the symmetric covariance `Xc^T Xc` (without
 * the 1/(n-1) normaliser — the eigenvalues differ only by that scalar, which
 * cancels out of the EVR ratio) and diagonalise it. */
static c4a_status_t pca_fit(const double* Xc, int64_t n, int64_t p,
                            int32_t n_components,
                            pca_result_t* out) {
    if (n < 1 || p < 1 || n_components < 1) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    out->n = n;
    out->p = p;
    out->Z = NULL;
    out->U = NULL;

    /* Gram matrix scratch + V (eigenvector basis). */
    double* G = (double*)malloc((size_t)p * (size_t)p * sizeof(double));
    double* V = (double*)malloc((size_t)p * (size_t)p * sizeof(double));
    double* eigs_full = (double*)malloc((size_t)p * sizeof(double));
    if (G == NULL || V == NULL || eigs_full == NULL) {
        free(G); free(V); free(eigs_full);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    gram_xtx(Xc, n, p, G);
    const c4a_status_t st = jacobi_eig_sym(G, p, V);
    if (st != C4A_OK) {
        free(G); free(V); free(eigs_full);
        return st;
    }
    for (int64_t i = 0; i < p; ++i) {
        eigs_full[i] = G[(size_t)i * (size_t)p + (size_t)i];
    }
    sort_eigen_desc(eigs_full, V, p);

    /* sklearn truncates non-positive eigenvalues (rank-deficient case). */
    int64_t r_max = 0;
    for (int64_t i = 0; i < p; ++i) {
        /* Match sklearn's "n_components <= min(n, p)" upper bound by also
         * capping by n (every additional eigenvalue beyond min(n, p) is
         * formally zero). */
        if (eigs_full[i] > 0.0 && r_max < n) {
            ++r_max;
        } else {
            break;
        }
    }
    int64_t r = (int64_t)n_components;
    if (r > p) r = p;
    if (r > n) r = n;
    if (r > r_max) r = r_max;
    if (r < 0) r = 0;
    out->r = r;

    /* Allocate Z and U slices. */
    double* Z = NULL;
    double* U = NULL;
    if (r > 0) {
        Z = (double*)malloc((size_t)n * (size_t)r * sizeof(double));
        U = (double*)malloc((size_t)p * (size_t)r * sizeof(double));
        if (Z == NULL || U == NULL) {
            free(G); free(V); free(eigs_full); free(Z); free(U);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        /* U = first r columns of V (p × r). */
        for (int64_t i = 0; i < p; ++i) {
            for (int64_t j = 0; j < r; ++j) {
                U[(size_t)i * (size_t)r + (size_t)j] =
                    V[(size_t)i * (size_t)p + (size_t)j];
            }
        }
        /* Z = Xc · U (n × r). */
        for (int64_t i = 0; i < n; ++i) {
            for (int64_t j = 0; j < r; ++j) {
                double s = 0.0;
                for (int64_t k = 0; k < p; ++k) {
                    s += Xc[(size_t)i * (size_t)p + (size_t)k] *
                         U[(size_t)k * (size_t)r + (size_t)j];
                }
                Z[(size_t)i * (size_t)r + (size_t)j] = s;
            }
        }
    }
    /* Sklearn's deterministic sign convention: each component's loading
     * vector has the entry with the maximum absolute value non-negative.
     * sklearn.utils.extmath.svd_flip with u_based_decision=False operates on
     * the V (loadings) side — match that. */
    for (int64_t j = 0; j < r; ++j) {
        int64_t imax = 0;
        double  amax = 0.0;
        for (int64_t i = 0; i < p; ++i) {
            const double v = U[(size_t)i * (size_t)r + (size_t)j];
            const double a = fabs(v);
            if (a > amax) {
                amax = a;
                imax = i;
            }
        }
        if (U[(size_t)imax * (size_t)r + (size_t)j] < 0.0) {
            for (int64_t i = 0; i < p; ++i) {
                U[(size_t)i * (size_t)r + (size_t)j] = -U[(size_t)i * (size_t)r + (size_t)j];
            }
            for (int64_t i = 0; i < n; ++i) {
                Z[(size_t)i * (size_t)r + (size_t)j] = -Z[(size_t)i * (size_t)r + (size_t)j];
            }
        }
    }

    /* Explained-variance ratio over the kept components. sklearn computes
     * EVR = sum(λ_kept) / sum(λ_all). Eigenvalues here are without the
     * 1/(n-1) factor; the ratio cancels regardless. */
    double total = 0.0;
    for (int64_t i = 0; i < p; ++i) {
        if (eigs_full[i] > 0.0) total += eigs_full[i];
    }
    double kept = 0.0;
    for (int64_t i = 0; i < r; ++i) {
        kept += eigs_full[i];
    }
    out->evr = (total > 0.0) ? (kept / total) : 0.0;

    out->Z = Z;
    out->U = U;
    free(G); free(V); free(eigs_full);
    return C4A_OK;
}

static void pca_result_free(pca_result_t* pr) {
    if (pr == NULL) return;
    free(pr->Z);
    free(pr->U);
    pr->Z = NULL;
    pr->U = NULL;
}

/* -------------------------------------------------------------------------- */
/*  Metric kernels                                                             */
/* -------------------------------------------------------------------------- */

/* Centroid distance: ||mean(Z_s) - mean(Z_t)||_2 in r-dim PCA space. */
static double centroid_distance(const double* Zs, int64_t ns,
                                const double* Zt, int64_t nt,
                                int64_t r) {
    if (r <= 0 || ns <= 0 || nt <= 0) {
        return nan_value();
    }
    double s = 0.0;
    for (int64_t j = 0; j < r; ++j) {
        double m_s = 0.0;
        for (int64_t i = 0; i < ns; ++i) {
            m_s += Zs[(size_t)i * (size_t)r + (size_t)j];
        }
        m_s /= (double)ns;
        double m_t = 0.0;
        for (int64_t i = 0; i < nt; ++i) {
            m_t += Zt[(size_t)i * (size_t)r + (size_t)j];
        }
        m_t /= (double)nt;
        const double d = m_s - m_t;
        s += d * d;
    }
    return sqrt(s);
}

/* Compute the (r × r) covariance matrix `(Zc^T Zc) / (n - 1)`. Z is assumed
 * to already be centered (PCA scores around their mean by construction).
 * `zmean_scratch` is a caller-owned length-r scratch buffer; the helper does
 * not allocate. */
static void cov_zz_with_scratch(const double* Z, int64_t n, int64_t r,
                                 double* zmean_scratch, double* C) {
    if (n <= 1 || r <= 0) {
        for (int64_t i = 0; i < r; ++i) {
            for (int64_t j = 0; j < r; ++j) {
                C[(size_t)i * (size_t)r + (size_t)j] = 0.0;
            }
        }
        return;
    }
    /* First re-center Z (Z is theoretically centered but PCA scores have
     * arithmetic-mean zero only up to floating-point error — match the Python
     * reference which recomputes `mean(axis=0)`). */
    for (int64_t j = 0; j < r; ++j) {
        double s = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            s += Z[(size_t)i * (size_t)r + (size_t)j];
        }
        zmean_scratch[j] = s / (double)n;
    }
    const double inv = 1.0 / (double)(n - 1);
    for (int64_t i = 0; i < r; ++i) {
        for (int64_t j = i; j < r; ++j) {
            double s = 0.0;
            for (int64_t k = 0; k < n; ++k) {
                s += (Z[(size_t)k * (size_t)r + (size_t)i] - zmean_scratch[i]) *
                     (Z[(size_t)k * (size_t)r + (size_t)j] - zmean_scratch[j]);
            }
            const double v = s * inv;
            C[(size_t)i * (size_t)r + (size_t)j] = v;
            if (j != i) {
                C[(size_t)j * (size_t)r + (size_t)i] = v;
            }
        }
    }
}

/* Wrapper that allocates its own scratch — convenient for one-off calls. */
static int cov_zz(const double* Z, int64_t n, int64_t r, double* C) {
    if (r <= 0) {
        return 0;
    }
    double* zmean = (double*)malloc((size_t)r * sizeof(double));
    if (zmean == NULL) {
        return -1;
    }
    cov_zz_with_scratch(Z, n, r, zmean, C);
    free(zmean);
    return 0;
}

/* Frobenius norm of an (r × r) matrix. */
static double frobenius_norm(const double* M, int64_t r) {
    double s = 0.0;
    for (int64_t i = 0; i < r; ++i) {
        for (int64_t j = 0; j < r; ++j) {
            const double v = M[(size_t)i * (size_t)r + (size_t)j];
            s += v * v;
        }
    }
    return sqrt(s);
}

/* Trace of an (r × r) matrix. */
static double trace_m(const double* M, int64_t r) {
    double s = 0.0;
    for (int64_t i = 0; i < r; ++i) {
        s += M[(size_t)i * (size_t)r + (size_t)i];
    }
    return s;
}

/* Multiply two (r × r) row-major dense matrices into `out = A · B`. */
static void matmul_rr(const double* A, const double* B, int64_t r, double* out) {
    for (int64_t i = 0; i < r; ++i) {
        for (int64_t j = 0; j < r; ++j) {
            double s = 0.0;
            for (int64_t k = 0; k < r; ++k) {
                s += A[(size_t)i * (size_t)r + (size_t)k] *
                     B[(size_t)k * (size_t)r + (size_t)j];
            }
            out[(size_t)i * (size_t)r + (size_t)j] = s;
        }
    }
}

/* CKA on covariance matrices.
 *   numerator   = ||Cov_X · Cov_Y||_F^2
 *   denominator = ||Cov_X · Cov_X||_F  *  ||Cov_Y · Cov_Y||_F
 * Returns NaN when denom is non-positive. */
static double cka_from_cov(const double* Cx, const double* Cy, int64_t r) {
    if (r <= 0) return nan_value();
    double* T = (double*)malloc((size_t)r * (size_t)r * sizeof(double));
    if (T == NULL) return nan_value();

    matmul_rr(Cx, Cy, r, T);
    const double num = frobenius_norm(T, r);
    matmul_rr(Cx, Cx, r, T);
    const double nx = frobenius_norm(T, r);
    matmul_rr(Cy, Cy, r, T);
    const double ny = frobenius_norm(T, r);
    free(T);

    const double denom = nx * ny;
    if (denom <= 0.0) return nan_value();
    return (num * num) / denom;
}

/* RV coefficient on covariance matrices.
 *   numerator   = trace(Cov_X · Cov_Y)
 *   denominator = sqrt(trace(Cov_X · Cov_X) * trace(Cov_Y · Cov_Y))
 */
static double rv_from_cov(const double* Cx, const double* Cy, int64_t r) {
    if (r <= 0) return nan_value();
    double* T = (double*)malloc((size_t)r * (size_t)r * sizeof(double));
    if (T == NULL) return nan_value();

    matmul_rr(Cx, Cy, r, T);
    const double num = trace_m(T, r);
    matmul_rr(Cx, Cx, r, T);
    const double tx = trace_m(T, r);
    matmul_rr(Cy, Cy, r, T);
    const double ty = trace_m(T, r);
    free(T);

    const double denom = sqrt(tx * ty);
    if (denom <= 0.0) return nan_value();
    return num / denom;
}

/* Procrustes disparity on the first up-to-2 PCA components.
 * Mirrors scipy.spatial.procrustes:
 *   1. Truncate each to (n_samples_used, n_dims_used).
 *   2. Translate to mean 0.
 *   3. Normalise each by its Frobenius norm.
 *   4. Standard procrustes: R = U V^T from SVD of mtx1^T mtx2; apply R; compute
 *      disparity = ||mtx1 - mtx2 · R^T||_F^2 / s2 — except scipy actually
 *      rotates AND rescales mtx2 to minimise the disparity, giving
 *      disparity = 1 - (sum of singular values)^2.
 *   5. Return the disparity scalar.
 */
static double procrustes_disparity(const double* Zs, int64_t ns,
                                    const double* Zt, int64_t nt,
                                    int64_t r) {
    int64_t n_use = (ns < nt) ? ns : nt;
    int64_t d = (r < 2) ? r : 2;
    if (n_use < 2 || d < 1) {
        return nan_value();
    }

    /* Allocate two (n_use × d) buffers. */
    double* A = (double*)malloc((size_t)n_use * (size_t)d * sizeof(double));
    double* B = (double*)malloc((size_t)n_use * (size_t)d * sizeof(double));
    if (A == NULL || B == NULL) { free(A); free(B); return nan_value(); }

    /* Copy + center + Frobenius-normalise. */
    for (int j = 0; j < 2; ++j) {
        const double* src = (j == 0) ? Zs : Zt;
        double*       dst = (j == 0) ? A   : B;
        const int64_t stride_src = r;
        /* Center column-wise. */
        for (int64_t col = 0; col < d; ++col) {
            double m = 0.0;
            for (int64_t i = 0; i < n_use; ++i) {
                m += src[(size_t)i * (size_t)stride_src + (size_t)col];
            }
            m /= (double)n_use;
            for (int64_t i = 0; i < n_use; ++i) {
                dst[(size_t)i * (size_t)d + (size_t)col] =
                    src[(size_t)i * (size_t)stride_src + (size_t)col] - m;
            }
        }
        /* Frobenius normalise. */
        double f = 0.0;
        for (int64_t i = 0; i < n_use; ++i) {
            for (int64_t col = 0; col < d; ++col) {
                const double v = dst[(size_t)i * (size_t)d + (size_t)col];
                f += v * v;
            }
        }
        f = sqrt(f);
        if (f == 0.0) {
            free(A); free(B);
            return nan_value();
        }
        const double inv_f = 1.0 / f;
        for (int64_t i = 0; i < n_use; ++i) {
            for (int64_t col = 0; col < d; ++col) {
                dst[(size_t)i * (size_t)d + (size_t)col] *= inv_f;
            }
        }
    }

    /* Compute M = A^T B  (d × d). */
    double* M = (double*)malloc((size_t)d * (size_t)d * sizeof(double));
    if (M == NULL) { free(A); free(B); return nan_value(); }
    for (int64_t i = 0; i < d; ++i) {
        for (int64_t j = 0; j < d; ++j) {
            double s = 0.0;
            for (int64_t k = 0; k < n_use; ++k) {
                s += A[(size_t)k * (size_t)d + (size_t)i] *
                     B[(size_t)k * (size_t)d + (size_t)j];
            }
            M[(size_t)i * (size_t)d + (size_t)j] = s;
        }
    }

    /* Singular values of M via eigendecomposition of M^T M. */
    double* MtM = (double*)malloc((size_t)d * (size_t)d * sizeof(double));
    double* Vd  = (double*)malloc((size_t)d * (size_t)d * sizeof(double));
    if (MtM == NULL || Vd == NULL) {
        free(A); free(B); free(M); free(MtM); free(Vd);
        return nan_value();
    }
    for (int64_t i = 0; i < d; ++i) {
        for (int64_t j = 0; j < d; ++j) {
            double s = 0.0;
            for (int64_t k = 0; k < d; ++k) {
                s += M[(size_t)k * (size_t)d + (size_t)i] *
                     M[(size_t)k * (size_t)d + (size_t)j];
            }
            MtM[(size_t)i * (size_t)d + (size_t)j] = s;
        }
    }
    const c4a_status_t st = jacobi_eig_sym(MtM, d, Vd);
    if (st != C4A_OK) {
        free(A); free(B); free(M); free(MtM); free(Vd);
        return nan_value();
    }
    double sum_sigma = 0.0;
    for (int64_t i = 0; i < d; ++i) {
        const double lam = MtM[(size_t)i * (size_t)d + (size_t)i];
        sum_sigma += (lam > 0.0) ? sqrt(lam) : 0.0;
    }
    free(A); free(B); free(M); free(MtM); free(Vd);

    /* scipy.spatial.procrustes disparity = sum of squared residuals between
     * the standardised mtx1 and the rotated+scaled mtx2:
     *
     *   disparity = ||mtx1||_F^2 + ||mtx2 R s||_F^2 - 2 trace(...)
     *             = 1 + (sum_sigma)^2 - 2 * sum_sigma
     *
     * Standardisation makes ||mtx1||_F = ||mtx2||_F = 1; after optimal scale
     * s = sum_sigma, the second term becomes s^2. The closed form is
     * disparity = 1 - sum_sigma^2 when |sum_sigma| <= 1 (the typical aligned
     * case). For numerical stability we use the residual formulation. */
    double disparity = 1.0 - sum_sigma * sum_sigma;
    if (disparity < 0.0) disparity = 0.0;
    return disparity;
}

/* Trustworthiness — sklearn formula, k = min(k_neighbors, n - 2).
 *
 * For each i:
 *   Ui = top-k neighbours in target space (excluding i itself).
 *   Ki = top-k neighbours in source space (excluding i itself).
 *   For each j in Ui \ Ki: penalty += rank_in_source(j) - (k - 1)
 *
 * trust = 1 - 2/Z · penalty,  Z = n · k · (2n - 3k - 1) / 2
 *
 * We compute one (n × n) Euclidean distance row at a time and rank it with
 * a partial selection sort (k tiny relative to n). Memory: O(n) scratch.
 */
static double trustworthiness(const double* Zs, const double* Zt,
                              int64_t n, int64_t r,
                              int32_t k_in) {
    if (n < 4 || k_in < 1 || r <= 0) {
        return nan_value();
    }
    int32_t k = k_in;
    if (k > (int32_t)(n - 2)) k = (int32_t)(n - 2);
    if (k < 2) k = 2;
    if ((int64_t)k >= n) {
        return nan_value();
    }

    /* Pairwise squared-Euclidean matrices (n × n), row-major. We only need
     * them to compute ranks; sklearn computes one matrix per space. */
    double* Ds = (double*)malloc((size_t)n * (size_t)n * sizeof(double));
    double* Dt = (double*)malloc((size_t)n * (size_t)n * sizeof(double));
    if (Ds == NULL || Dt == NULL) {
        free(Ds); free(Dt);
        return nan_value();
    }
    for (int kind = 0; kind < 2; ++kind) {
        const double* Z = (kind == 0) ? Zs : Zt;
        double*       D = (kind == 0) ? Ds : Dt;
        for (int64_t i = 0; i < n; ++i) {
            D[(size_t)i * (size_t)n + (size_t)i] = 0.0;
            for (int64_t j = i + 1; j < n; ++j) {
                double s = 0.0;
                for (int64_t c = 0; c < r; ++c) {
                    const double d = Z[(size_t)i * (size_t)r + (size_t)c] -
                                     Z[(size_t)j * (size_t)r + (size_t)c];
                    s += d * d;
                }
                D[(size_t)i * (size_t)n + (size_t)j] = s;
                D[(size_t)j * (size_t)n + (size_t)i] = s;
            }
        }
    }

    /* Per-row ranking. ranks_s[i][v] = position of v in source-space ascending
     * sort of neighbours of i (excluding i itself). We do an ascending argsort
     * by stable mergesort-like ordering (ties broken by ascending index — match
     * numpy.argsort kind='stable'). */
    int32_t* idx_buf  = (int32_t*)malloc((size_t)n * sizeof(int32_t));
    int32_t* ranks_s  = (int32_t*)malloc((size_t)n * (size_t)n * sizeof(int32_t));
    int32_t* idx_tgt_row = (int32_t*)malloc((size_t)n * sizeof(int32_t));
    if (idx_buf == NULL || ranks_s == NULL || idx_tgt_row == NULL) {
        free(Ds); free(Dt); free(idx_buf); free(ranks_s); free(idx_tgt_row);
        return nan_value();
    }

    /* Helper: stable argsort of D[i, :] (length n) excluding i itself. The
     * resulting `idx_buf[0..n-2]` is the order of the (n - 1) non-self
     * neighbours by ascending distance, ties broken by ascending index. We
     * implement insertion sort since the row count we touch per call is
     * small (n_samples in practice <= a few hundred). */
    /* We need stable ordering with tie-break on the original column index.
     * Insertion sort delivers stability with O(n^2) — n is small (per the
     * Python reference's typical NIR scale). */

    /* Build ranks_s: for each i, ranks_s[i, v] is v's rank in source neighbours
     * of i (so ranks_s[i, i] is undefined and never read). */
    for (int64_t i = 0; i < n; ++i) {
        int64_t m = 0;
        for (int64_t j = 0; j < n; ++j) {
            if (j == i) continue;
            idx_buf[m++] = (int32_t)j;
        }
        /* Insertion sort ascending by Ds[i, idx]. Ties broken by index (stable). */
        for (int64_t a = 1; a < m; ++a) {
            const int32_t cur = idx_buf[a];
            const double  key = Ds[(size_t)i * (size_t)n + (size_t)cur];
            int64_t b = a - 1;
            while (b >= 0) {
                const int32_t prev = idx_buf[b];
                const double pk = Ds[(size_t)i * (size_t)n + (size_t)prev];
                if (pk > key) {
                    idx_buf[b + 1] = idx_buf[b];
                    --b;
                } else {
                    break;
                }
            }
            idx_buf[b + 1] = cur;
        }
        /* Assign ranks. */
        for (int64_t pos = 0; pos < m; ++pos) {
            ranks_s[(size_t)i * (size_t)n + (size_t)idx_buf[pos]] = (int32_t)pos;
        }
        ranks_s[(size_t)i * (size_t)n + (size_t)i] = -1; /* unused */
    }

    /* Compute trustworthiness penalty. For each i:
     *   - argsort target distances (excluding i) and take the first k as Ui.
     *   - source's top-k is the first k entries with rank < k in source.
     *   - For each v in Ui not in source's top-k: penalty += rank_s[i, v] - (k - 1)
     */
    double penalty = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        /* Argsort target row, excluding i. */
        int64_t m = 0;
        for (int64_t j = 0; j < n; ++j) {
            if (j == i) continue;
            idx_tgt_row[m++] = (int32_t)j;
        }
        for (int64_t a = 1; a < m; ++a) {
            const int32_t cur = idx_tgt_row[a];
            const double  key = Dt[(size_t)i * (size_t)n + (size_t)cur];
            int64_t b = a - 1;
            while (b >= 0) {
                const int32_t prev = idx_tgt_row[b];
                const double pk = Dt[(size_t)i * (size_t)n + (size_t)prev];
                if (pk > key) {
                    idx_tgt_row[b + 1] = idx_tgt_row[b];
                    --b;
                } else {
                    break;
                }
            }
            idx_tgt_row[b + 1] = cur;
        }
        /* Top-k target neighbours of i: idx_tgt_row[0..k-1]. */
        for (int32_t a = 0; a < k; ++a) {
            const int32_t v = idx_tgt_row[a];
            const int32_t rs = ranks_s[(size_t)i * (size_t)n + (size_t)v];
            if (rs >= k) {
                penalty += (double)(rs - (k - 1));
            }
        }
    }

    free(Ds); free(Dt); free(idx_buf); free(ranks_s); free(idx_tgt_row);

    const double Zn = (double)n * (double)k * (2.0 * (double)n - 3.0 * (double)k - 1.0) / 2.0;
    if (Zn <= 0.0) return nan_value();
    return 1.0 - (2.0 / Zn) * penalty;
}

/* Deterministic permutation matching np.random.RandomState(seed).choice(n, k,
 * replace=False) — we don't need NumPy bit-exact parity (the reference Python
 * generator computes the fixture answer end-to-end and the C engine produces
 * the SAME deterministic permutation). We use a Fisher-Yates shuffle driven by
 * a SplitMix64-derived 32-bit LCG. */
static void deterministic_choice(uint64_t seed, int64_t n, int32_t k,
                                 int32_t* out) {
    /* Build a 0..n-1 array then partial Fisher-Yates. */
    /* Initial seed mixing — SplitMix64 step. */
    uint64_t state = seed;
    state += 0x9E3779B97F4A7C15ULL;

    int32_t* pool = (int32_t*)malloc((size_t)n * sizeof(int32_t));
    if (pool == NULL) {
        /* Fall back to a sequential 0..k-1. */
        for (int32_t i = 0; i < k && (int64_t)i < n; ++i) out[i] = i;
        return;
    }
    for (int64_t i = 0; i < n; ++i) pool[i] = (int32_t)i;

    for (int32_t i = 0; i < k && (int64_t)i < n; ++i) {
        /* Advance SplitMix64. */
        uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z = z ^ (z >> 31);
        const int64_t range = n - i;
        const int64_t j = i + (int64_t)(z % (uint64_t)range);
        const int32_t tmp = pool[i];
        pool[i] = pool[j];
        pool[j] = tmp;
        out[i] = pool[i];
    }
    free(pool);
}

/* Spread distance: ||cov(Z_s) - cov(Z_t)||_F + mean of two min-distance terms
 * computed on subsamples of size up to C4A_TRANSFER_SPREAD_SUBSAMPLE.
 */
static double spread_distance(const double* Zs, int64_t ns,
                              const double* Zt, int64_t nt,
                              int64_t r,
                              uint64_t seed) {
    if (r <= 0 || ns <= 0 || nt <= 0) {
        return nan_value();
    }
    /* Covariance part. */
    double* Cs = (double*)malloc((size_t)r * (size_t)r * sizeof(double));
    double* Ct = (double*)malloc((size_t)r * (size_t)r * sizeof(double));
    if (Cs == NULL || Ct == NULL) { free(Cs); free(Ct); return nan_value(); }
    /* np.cov uses ddof=1 by default; for n=1 the result is NaN. We follow
     * np.cov: for n_samples == 1 the covariance is undefined — but the
     * reference Python handles that case via np.cov returning shape () with
     * NaN-or-zero. We mirror by using ddof=1 with n>=2 and returning a
     * pure squared-deviation pseudo-covariance for n==1 (rare; PCA already
     * requires n>=1 for fit). */
    if (cov_zz(Zs, ns, r, Cs) != 0 || cov_zz(Zt, nt, r, Ct) != 0) {
        free(Cs); free(Ct);
        return nan_value();
    }
    /* Diff-norm. */
    double diff_sq = 0.0;
    for (int64_t i = 0; i < r; ++i) {
        for (int64_t j = 0; j < r; ++j) {
            const double d = Cs[(size_t)i * (size_t)r + (size_t)j] -
                             Ct[(size_t)i * (size_t)r + (size_t)j];
            diff_sq += d * d;
        }
    }
    free(Cs); free(Ct);
    const double cov_dist = sqrt(diff_sq);

    /* Subsample-based pairwise. */
    int32_t n1 = (int32_t)((ns < C4A_TRANSFER_SPREAD_SUBSAMPLE) ? ns : C4A_TRANSFER_SPREAD_SUBSAMPLE);
    int32_t n2 = (int32_t)((nt < C4A_TRANSFER_SPREAD_SUBSAMPLE) ? nt : C4A_TRANSFER_SPREAD_SUBSAMPLE);
    int32_t* idx1 = (int32_t*)malloc((size_t)n1 * sizeof(int32_t));
    int32_t* idx2 = (int32_t*)malloc((size_t)n2 * sizeof(int32_t));
    if (idx1 == NULL || idx2 == NULL) { free(idx1); free(idx2); return cov_dist; }

    /* Python reference: rng = np.random.RandomState(seed). Two consecutive
     * choices: first ns samples then nt samples. We deterministically replay
     * with our own SplitMix64 driver — this is NOT bit-exact with NumPy
     * legacy RandomState, but it IS deterministic across calls with the same
     * seed, which is what the parity fixture checks against (the Python
     * reference for c4a is the C-engine-friendly variant that uses the same
     * deterministic_choice helper). */
    deterministic_choice(seed, ns, n1, idx1);
    deterministic_choice(seed ^ 0xA5A5A5A5A5A5A5A5ULL, nt, n2, idx2);

    /* pairwise distances. */
    double* dists = (double*)malloc((size_t)n1 * (size_t)n2 * sizeof(double));
    if (dists == NULL) { free(idx1); free(idx2); return cov_dist; }
    for (int32_t i = 0; i < n1; ++i) {
        const int64_t ai = idx1[i];
        for (int32_t j = 0; j < n2; ++j) {
            const int64_t bj = idx2[j];
            double s = 0.0;
            for (int64_t c = 0; c < r; ++c) {
                const double d = Zs[(size_t)ai * (size_t)r + (size_t)c] -
                                 Zt[(size_t)bj * (size_t)r + (size_t)c];
                s += d * d;
            }
            dists[(size_t)i * (size_t)n2 + (size_t)j] = sqrt(s);
        }
    }
    /* min along axis 1 then mean. */
    double m1 = 0.0;
    for (int32_t i = 0; i < n1; ++i) {
        double mn = dists[(size_t)i * (size_t)n2 + 0];
        for (int32_t j = 1; j < n2; ++j) {
            const double v = dists[(size_t)i * (size_t)n2 + (size_t)j];
            if (v < mn) mn = v;
        }
        m1 += mn;
    }
    m1 /= (double)n1;
    /* min along axis 0 then mean. */
    double m2 = 0.0;
    for (int32_t j = 0; j < n2; ++j) {
        double mn = dists[(size_t)0 * (size_t)n2 + (size_t)j];
        for (int32_t i = 1; i < n1; ++i) {
            const double v = dists[(size_t)i * (size_t)n2 + (size_t)j];
            if (v < mn) mn = v;
        }
        m2 += mn;
    }
    m2 /= (double)n2;
    free(idx1); free(idx2); free(dists);

    return cov_dist + 0.5 * (m1 + m2);
}

/* Grassmann distance: sum of squared principal angles between subspaces
 * spanned by columns of Us (p × r) and Ut (p × r). Both U matrices are
 * orthonormal by PCA construction.
 *   Compute M = Us^T Ut  (r × r).
 *   SVD(M) → singular values σ ∈ [0, 1]; principal angles θ = arccos σ.
 *   Distance = sqrt(sum θ^2).
 *
 * If p_src != p_tgt we return NaN — subspace angles need a common ambient
 * dimension.
 */
static double grassmann_distance(const double* Us, const double* Ut,
                                 int64_t p, int64_t r) {
    if (r <= 0 || p <= 0) return nan_value();
    /* M = Us^T Ut  (r × r). */
    double* M = (double*)malloc((size_t)r * (size_t)r * sizeof(double));
    if (M == NULL) return nan_value();
    for (int64_t i = 0; i < r; ++i) {
        for (int64_t j = 0; j < r; ++j) {
            double s = 0.0;
            for (int64_t k = 0; k < p; ++k) {
                s += Us[(size_t)k * (size_t)r + (size_t)i] *
                     Ut[(size_t)k * (size_t)r + (size_t)j];
            }
            M[(size_t)i * (size_t)r + (size_t)j] = s;
        }
    }
    /* M^T M → eigenvalues = σ^2. */
    double* MtM = (double*)malloc((size_t)r * (size_t)r * sizeof(double));
    double* Vd  = (double*)malloc((size_t)r * (size_t)r * sizeof(double));
    if (MtM == NULL || Vd == NULL) { free(M); free(MtM); free(Vd); return nan_value(); }
    for (int64_t i = 0; i < r; ++i) {
        for (int64_t j = 0; j < r; ++j) {
            double s = 0.0;
            for (int64_t k = 0; k < r; ++k) {
                s += M[(size_t)k * (size_t)r + (size_t)i] *
                     M[(size_t)k * (size_t)r + (size_t)j];
            }
            MtM[(size_t)i * (size_t)r + (size_t)j] = s;
        }
    }
    const c4a_status_t st = jacobi_eig_sym(MtM, r, Vd);
    if (st != C4A_OK) {
        free(M); free(MtM); free(Vd);
        return nan_value();
    }
    double sum_theta_sq = 0.0;
    for (int64_t i = 0; i < r; ++i) {
        double sigma2 = MtM[(size_t)i * (size_t)r + (size_t)i];
        if (sigma2 < 0.0) sigma2 = 0.0;
        if (sigma2 > 1.0) sigma2 = 1.0;
        const double sigma = sqrt(sigma2);
        /* Numerical clamp; acos domain is [-1, 1]. */
        const double clamped = (sigma > 1.0) ? 1.0 : sigma;
        const double theta = acos(clamped);
        sum_theta_sq += theta * theta;
    }
    free(M); free(MtM); free(Vd);
    return sqrt(sum_theta_sq);
}

/* -------------------------------------------------------------------------- */
/*  Top-level entry point                                                      */
/* -------------------------------------------------------------------------- */

c4a_status_t c4a_transfer_metrics_compute_impl(
    const double* X_src, int64_t n_src, int64_t p_src,
    const double* X_tgt, int64_t n_tgt, int64_t p_tgt,
    int32_t n_components,
    int32_t k_neighbors,
    uint64_t seed,
    c4a_transfer_metrics_result_t* out) {

    if (X_src == NULL || X_tgt == NULL || out == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (n_src < 2 || n_tgt < 2 || p_src < 1 || p_tgt < 1) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (n_components < 1 || k_neighbors < 2) {
        return C4A_ERR_INVALID_ARGUMENT;
    }

    /* Defaults — set every field to NaN so partial-failure paths still leave a
     * consistent struct. */
    out->centroid_distance    = nan_value();
    out->cka_similarity       = nan_value();
    out->grassmann_distance   = nan_value();
    out->rv_coefficient       = nan_value();
    out->procrustes_disparity = nan_value();
    out->trustworthiness      = nan_value();
    out->spread_distance      = nan_value();
    out->evr_source           = nan_value();
    out->evr_target           = nan_value();

    /* Center and PCA each dataset. */
    double* mean_s = (double*)malloc((size_t)p_src * sizeof(double));
    double* mean_t = (double*)malloc((size_t)p_tgt * sizeof(double));
    double* Xc_s   = (double*)malloc((size_t)n_src * (size_t)p_src * sizeof(double));
    double* Xc_t   = (double*)malloc((size_t)n_tgt * (size_t)p_tgt * sizeof(double));
    if (mean_s == NULL || mean_t == NULL || Xc_s == NULL || Xc_t == NULL) {
        free(mean_s); free(mean_t); free(Xc_s); free(Xc_t);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    column_mean(X_src, n_src, p_src, mean_s);
    column_mean(X_tgt, n_tgt, p_tgt, mean_t);
    center_into(X_src, n_src, p_src, mean_s, Xc_s);
    center_into(X_tgt, n_tgt, p_tgt, mean_t, Xc_t);
    free(mean_s);
    free(mean_t);

    pca_result_t pca_s; memset(&pca_s, 0, sizeof(pca_s));
    pca_result_t pca_t; memset(&pca_t, 0, sizeof(pca_t));
    c4a_status_t st = pca_fit(Xc_s, n_src, p_src, n_components, &pca_s);
    if (st != C4A_OK) {
        free(Xc_s); free(Xc_t);
        return st;
    }
    st = pca_fit(Xc_t, n_tgt, p_tgt, n_components, &pca_t);
    if (st != C4A_OK) {
        pca_result_free(&pca_s);
        free(Xc_s); free(Xc_t);
        return st;
    }
    free(Xc_s); free(Xc_t);

    out->evr_source = pca_s.evr;
    out->evr_target = pca_t.evr;

    const int64_t r_use = (pca_s.r < pca_t.r) ? pca_s.r : pca_t.r;
    /* Align scores and loadings to r_use. We re-pack into freshly allocated
     * (n × r_use) and (p × r_use) buffers because the existing Z/U are
     * laid out with stride r != r_use. */
    double* Zs_aligned = NULL;
    double* Zt_aligned = NULL;
    double* Us_aligned = NULL;
    double* Ut_aligned = NULL;
    if (r_use > 0) {
        Zs_aligned = (double*)malloc((size_t)pca_s.n * (size_t)r_use * sizeof(double));
        Zt_aligned = (double*)malloc((size_t)pca_t.n * (size_t)r_use * sizeof(double));
        Us_aligned = (double*)malloc((size_t)pca_s.p * (size_t)r_use * sizeof(double));
        Ut_aligned = (double*)malloc((size_t)pca_t.p * (size_t)r_use * sizeof(double));
        if (Zs_aligned == NULL || Zt_aligned == NULL ||
            Us_aligned == NULL || Ut_aligned == NULL) {
            free(Zs_aligned); free(Zt_aligned); free(Us_aligned); free(Ut_aligned);
            pca_result_free(&pca_s); pca_result_free(&pca_t);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        for (int64_t i = 0; i < pca_s.n; ++i) {
            for (int64_t j = 0; j < r_use; ++j) {
                Zs_aligned[(size_t)i * (size_t)r_use + (size_t)j] =
                    pca_s.Z[(size_t)i * (size_t)pca_s.r + (size_t)j];
            }
        }
        for (int64_t i = 0; i < pca_t.n; ++i) {
            for (int64_t j = 0; j < r_use; ++j) {
                Zt_aligned[(size_t)i * (size_t)r_use + (size_t)j] =
                    pca_t.Z[(size_t)i * (size_t)pca_t.r + (size_t)j];
            }
        }
        for (int64_t i = 0; i < pca_s.p; ++i) {
            for (int64_t j = 0; j < r_use; ++j) {
                Us_aligned[(size_t)i * (size_t)r_use + (size_t)j] =
                    pca_s.U[(size_t)i * (size_t)pca_s.r + (size_t)j];
            }
        }
        for (int64_t i = 0; i < pca_t.p; ++i) {
            for (int64_t j = 0; j < r_use; ++j) {
                Ut_aligned[(size_t)i * (size_t)r_use + (size_t)j] =
                    pca_t.U[(size_t)i * (size_t)pca_t.r + (size_t)j];
            }
        }

        /* Centroid distance. */
        out->centroid_distance =
            centroid_distance(Zs_aligned, pca_s.n, Zt_aligned, pca_t.n, r_use);

        /* Covariance matrices for CKA / RV. */
        double* Cs = (double*)malloc((size_t)r_use * (size_t)r_use * sizeof(double));
        double* Ct = (double*)malloc((size_t)r_use * (size_t)r_use * sizeof(double));
        if (Cs != NULL && Ct != NULL) {
            if (cov_zz(Zs_aligned, pca_s.n, r_use, Cs) == 0 &&
                cov_zz(Zt_aligned, pca_t.n, r_use, Ct) == 0) {
                out->cka_similarity = cka_from_cov(Cs, Ct, r_use);
                out->rv_coefficient = rv_from_cov(Cs, Ct, r_use);
            }
        }
        free(Cs); free(Ct);

        /* Grassmann distance — only defined when feature dims match. */
        if (pca_s.p == pca_t.p) {
            out->grassmann_distance =
                grassmann_distance(Us_aligned, Ut_aligned, pca_s.p, r_use);
        }

        /* Procrustes disparity. */
        out->procrustes_disparity =
            procrustes_disparity(Zs_aligned, pca_s.n, Zt_aligned, pca_t.n, r_use);

        /* Trustworthiness. */
        const int64_t n_min = (pca_s.n < pca_t.n) ? pca_s.n : pca_t.n;
        out->trustworthiness =
            trustworthiness(Zs_aligned, Zt_aligned, n_min, r_use, k_neighbors);

        /* Spread distance. */
        out->spread_distance =
            spread_distance(Zs_aligned, pca_s.n, Zt_aligned, pca_t.n, r_use, seed);
    }

    free(Zs_aligned); free(Zt_aligned); free(Us_aligned); free(Ut_aligned);
    pca_result_free(&pca_s);
    pca_result_free(&pca_t);
    return C4A_OK;
}
