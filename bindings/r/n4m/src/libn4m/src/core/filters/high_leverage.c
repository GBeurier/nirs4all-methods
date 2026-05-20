/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * HighLeverage sample filter — reference implementation.
 *
 * For the documented algorithm see ``high_leverage.h``. Two numerical paths:
 *
 *   - HAT  : direct hat-matrix diagonal. Build ``M = X_c^T X_c + reg * I``
 *            (where reg = 1e-10 * trace(X_c^T X_c) / cols), factor with the
 *            shared Householder QR helper, then per row x_i solve
 *                M * v = x_i        (R v = Q^T x_i; back-substitute)
 *                h_i  = x_i^T v
 *
 *   - PCA  : symmetric Jacobi eigendecomposition of ``X_c^T X_c`` gives
 *            ``X_c^T X_c = V diag(lambda) V^T`` with V orthonormal. The
 *            principal directions are the eigenvectors sorted by descending
 *            eigenvalue. We keep the top k = (n_components or auto). In the
 *            PCA score space the precision matrix is exactly diagonal,
 *            ``precision_jj = 1 / (lambda_j + reg)``, so the leverage per
 *            row reduces to
 *                scores_ij = (x_c_i) . V[:, j]
 *                h_i      = sum_j scores_ij^2 / (lambda_j + reg)
 *
 * Centring: when ``center`` is non-zero the per-column training mean is
 * stored on the state and subtracted from both fit and transform inputs.
 *
 * Tikhonov regularisation matches the nirs4all 0.8.x reference to within
 * the contracted 1e-9 abs / 1e-10 rel tolerance.
 *
 * Pure C, no dependencies. INTERNAL.
 */

#include "high_leverage.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "core/common/linalg.h"

struct n4m_filter_leverage_state_t {
    /* Constructor parameters. */
    int32_t method;
    double  threshold_multiplier;
    int     use_absolute;
    double  absolute_threshold;
    int32_t n_components;
    int     center;

    /* Fitted state. */
    int     fitted;
    int     effective_method;  /* method actually used (hat or pca after auto-fallback) */
    int64_t fit_cols;
    int32_t k;                 /* number of effective components (pca) or cols (hat) */

    /* center == 0 implies all-zero mean buffer. Always allocated. */
    double* mean;              /* (cols,) */

    /* HAT path. */
    double* M_qr;              /* (cols, cols) row-major, in-place Householder factor */
    double* tau;               /* (cols,) Householder scalars */

    /* PCA path. */
    double* components;        /* (k, cols) row-major */
    double* inv_eigvals;       /* (k,) = 1 / (lambda_j + reg) */

    double  threshold;
};

/* --------------------------------------------------------------------------
 * Jacobi symmetric eigendecomposition (cyclic, row-by-row sweep).
 *
 * Input  : A in-place (n x n) row-major symmetric matrix.
 * Output : on success A's columns store the eigenvectors V (so A becomes V),
 *          and eigvals[i] stores the eigenvalue corresponding to V[:, i].
 *
 * The algorithm uses the classical Jacobi rotation with the standard tau
 * shortcut for stability (see Golub & Van Loan, Matrix Computations, ch. 8).
 * Convergence: O(n^3) per sweep, typically <= 20 sweeps for n <= 64.
 * -------------------------------------------------------------------------- */
static n4m_status_t jacobi_eigh(double* A, int64_t n, double* eigvals,
                                  double* V) {
    if (n < 1) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    /* Initialise V = I. */
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            V[(size_t)i * (size_t)n + (size_t)j] = (i == j) ? 1.0 : 0.0;
        }
    }
    const int max_sweeps = 100;
    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        /* off-diagonal Frobenius norm */
        double off = 0.0;
        for (int64_t p = 0; p < n - 1; ++p) {
            for (int64_t q = p + 1; q < n; ++q) {
                const double v = A[(size_t)p * (size_t)n + (size_t)q];
                off += v * v;
            }
        }
        if (off <= 1e-30) {
            break;
        }
        for (int64_t p = 0; p < n - 1; ++p) {
            for (int64_t q = p + 1; q < n; ++q) {
                const double app = A[(size_t)p * (size_t)n + (size_t)p];
                const double aqq = A[(size_t)q * (size_t)n + (size_t)q];
                const double apq = A[(size_t)p * (size_t)n + (size_t)q];
                if (fabs(apq) <= 1e-18 * (fabs(app) + fabs(aqq))) {
                    A[(size_t)p * (size_t)n + (size_t)q] = 0.0;
                    A[(size_t)q * (size_t)n + (size_t)p] = 0.0;
                    continue;
                }
                /* Compute rotation angle. */
                double t;
                if (fabs(app - aqq) <= 1e-300) {
                    t = (apq > 0) ? 1.0 : -1.0;
                } else {
                    const double theta = (aqq - app) / (2.0 * apq);
                    const double s = (theta >= 0.0) ? 1.0 : -1.0;
                    t = s / (fabs(theta) + sqrt(theta * theta + 1.0));
                }
                const double c = 1.0 / sqrt(t * t + 1.0);
                const double sn = t * c;
                const double tau = sn / (1.0 + c);
                /* Update A. */
                A[(size_t)p * (size_t)n + (size_t)p] = app - t * apq;
                A[(size_t)q * (size_t)n + (size_t)q] = aqq + t * apq;
                A[(size_t)p * (size_t)n + (size_t)q] = 0.0;
                A[(size_t)q * (size_t)n + (size_t)p] = 0.0;
                for (int64_t i = 0; i < n; ++i) {
                    if (i == p || i == q) continue;
                    const double aip = A[(size_t)i * (size_t)n + (size_t)p];
                    const double aiq = A[(size_t)i * (size_t)n + (size_t)q];
                    A[(size_t)i * (size_t)n + (size_t)p] = aip - sn * (aiq + tau * aip);
                    A[(size_t)p * (size_t)n + (size_t)i] = A[(size_t)i * (size_t)n + (size_t)p];
                    A[(size_t)i * (size_t)n + (size_t)q] = aiq + sn * (aip - tau * aiq);
                    A[(size_t)q * (size_t)n + (size_t)i] = A[(size_t)i * (size_t)n + (size_t)q];
                }
                /* Update V (eigenvectors stored as columns). */
                for (int64_t i = 0; i < n; ++i) {
                    const double vip = V[(size_t)i * (size_t)n + (size_t)p];
                    const double viq = V[(size_t)i * (size_t)n + (size_t)q];
                    V[(size_t)i * (size_t)n + (size_t)p] = vip - sn * (viq + tau * vip);
                    V[(size_t)i * (size_t)n + (size_t)q] = viq + sn * (vip - tau * viq);
                }
            }
        }
    }
    for (int64_t i = 0; i < n; ++i) {
        eigvals[i] = A[(size_t)i * (size_t)n + (size_t)i];
    }
    return N4M_OK;
}

/* Sort the eigenpairs by descending eigenvalue. V columns are permuted to
 * follow the sort. */
static void sort_eigpairs_desc(double* V, double* eigvals, int64_t n) {
    /* Insertion sort — small n. */
    for (int64_t i = 1; i < n; ++i) {
        const double key = eigvals[i];
        int64_t j = i - 1;
        while (j >= 0 && eigvals[j] < key) {
            eigvals[j + 1] = eigvals[j];
            /* swap column j and j+1 of V */
            for (int64_t r = 0; r < n; ++r) {
                const double t = V[(size_t)r * (size_t)n + (size_t)j];
                V[(size_t)r * (size_t)n + (size_t)j] =
                    V[(size_t)r * (size_t)n + (size_t)j + 1];
                V[(size_t)r * (size_t)n + (size_t)j + 1] = t;
            }
            --j;
        }
        eigvals[j + 1] = key;
    }
}

/* Match sklearn / nirs4all sign convention on PCA components: each row's
 * maximum-absolute element is non-negative. The 'components' here is the
 * (k, cols) matrix where row k is the k-th component vector. */
static void flip_signs_max_abs(double* components, int64_t k, int64_t cols) {
    for (int64_t i = 0; i < k; ++i) {
        double* row = components + (size_t)i * (size_t)cols;
        int64_t arg = 0;
        double best = fabs(row[0]);
        for (int64_t j = 1; j < cols; ++j) {
            const double v = fabs(row[j]);
            if (v > best) {
                best = v;
                arg = j;
            }
        }
        if (row[arg] < 0.0) {
            for (int64_t j = 0; j < cols; ++j) {
                row[j] = -row[j];
            }
        }
    }
}

n4m_filter_leverage_state_t* n4m_filter_leverage_state_new(
    int32_t method,
    double  threshold_multiplier,
    int     use_absolute,
    double  absolute_threshold,
    int32_t n_components,
    int     center) {
    if (method != N4M_FILTER_LEVERAGE_METHOD_HAT &&
        method != N4M_FILTER_LEVERAGE_METHOD_PCA) {
        return NULL;
    }
    if (use_absolute) {
        if (!(absolute_threshold > 0.0 && absolute_threshold < 1.0)) {
            return NULL;
        }
    } else {
        if (!(threshold_multiplier > 0.0)) {
            return NULL;
        }
    }
    n4m_filter_leverage_state_t* s =
        (n4m_filter_leverage_state_t*)malloc(sizeof(*s));
    if (s == NULL) {
        return NULL;
    }
    memset(s, 0, sizeof(*s));
    s->method               = method;
    s->threshold_multiplier = threshold_multiplier;
    s->use_absolute         = use_absolute ? 1 : 0;
    s->absolute_threshold   = absolute_threshold;
    s->n_components         = n_components;
    s->center               = center ? 1 : 0;
    s->threshold            = (double)NAN;
    return s;
}

void n4m_filter_leverage_state_free(n4m_filter_leverage_state_t* state) {
    if (state == NULL) return;
    free(state->mean);
    free(state->M_qr);
    free(state->tau);
    free(state->components);
    free(state->inv_eigvals);
    free(state);
}

int n4m_filter_leverage_state_is_fitted(const n4m_filter_leverage_state_t* state) {
    return (state != NULL && state->fitted) ? 1 : 0;
}

double n4m_filter_leverage_state_threshold(
    const n4m_filter_leverage_state_t* state) {
    if (state == NULL || !state->fitted) return (double)NAN;
    return state->threshold;
}

/* Compute (centred) X^T X (cols, cols) row-major. */
static void compute_xtx(const double* Xc, int64_t rows, int64_t cols,
                         double* XtX) {
    for (int64_t a = 0; a < cols; ++a) {
        for (int64_t b = 0; b < cols; ++b) {
            double s = 0.0;
            for (int64_t r = 0; r < rows; ++r) {
                s += Xc[(size_t)r * (size_t)cols + (size_t)a]
                   * Xc[(size_t)r * (size_t)cols + (size_t)b];
            }
            XtX[(size_t)a * (size_t)cols + (size_t)b] = s;
        }
    }
}

/* Pre-condition: state has its mean buffer allocated and populated.
 * Allocates and fills state->M_qr (cols x cols Householder factor) and
 * state->tau. */
static n4m_status_t fit_hat(n4m_filter_leverage_state_t* state,
                              const double* Xc, int64_t rows, int64_t cols) {
    state->effective_method = N4M_FILTER_LEVERAGE_METHOD_HAT;
    state->k = (int32_t)cols;
    /* M_qr = X_c^T X_c + reg * I */
    double* M = (double*)malloc((size_t)cols * (size_t)cols * sizeof(double));
    if (M == NULL) {
        return N4M_ERR_OUT_OF_MEMORY;
    }
    compute_xtx(Xc, rows, cols, M);
    double trace = 0.0;
    for (int64_t i = 0; i < cols; ++i) {
        trace += M[(size_t)i * (size_t)cols + (size_t)i];
    }
    const double reg = 1e-10 * trace / (double)cols;
    for (int64_t i = 0; i < cols; ++i) {
        M[(size_t)i * (size_t)cols + (size_t)i] += reg;
    }
    /* QR factor in place. */
    double* tau = (double*)malloc((size_t)cols * sizeof(double));
    if (tau == NULL) {
        free(M);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    const n4m_status_t qrst = n4m_householder_qr(M, cols, cols, tau);
    if (qrst != N4M_OK) {
        free(M); free(tau);
        return qrst;
    }
    state->M_qr = M;
    state->tau  = tau;
    return N4M_OK;
}

/* PCA path: compute the top-k eigenpairs of X_c^T X_c, store the components
 * (k, cols) and precomputed 1/(lambda_j + reg). */
static n4m_status_t fit_pca(n4m_filter_leverage_state_t* state,
                             const double* Xc, int64_t rows, int64_t cols) {
    state->effective_method = N4M_FILTER_LEVERAGE_METHOD_PCA;
    int64_t k_max = rows - 1;
    if (cols < k_max) k_max = cols;
    if (50 < k_max) k_max = 50;
    if (k_max < 1) k_max = 1;
    int64_t k;
    if (state->n_components > 0) {
        k = state->n_components;
        if (k > rows - 1) k = rows - 1;
        if (k > cols)     k = cols;
        if (k < 1)        k = 1;
    } else {
        k = k_max;
    }
    /* Build X_c^T X_c (cols x cols). */
    double* M = (double*)malloc((size_t)cols * (size_t)cols * sizeof(double));
    double* V = (double*)malloc((size_t)cols * (size_t)cols * sizeof(double));
    double* eigvals = (double*)malloc((size_t)cols * sizeof(double));
    if (M == NULL || V == NULL || eigvals == NULL) {
        free(M); free(V); free(eigvals);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    compute_xtx(Xc, rows, cols, M);
    const n4m_status_t jst = jacobi_eigh(M, cols, eigvals, V);
    if (jst != N4M_OK) {
        free(M); free(V); free(eigvals);
        return jst;
    }
    sort_eigpairs_desc(V, eigvals, cols);
    /* Extract top-k components (rows of V^T = columns of V), apply sign
     * convention, and compute 1/(lambda_j + reg). */
    double trace = 0.0;
    for (int64_t j = 0; j < cols; ++j) {
        trace += eigvals[j];
    }
    const double reg = 1e-10 * trace / (double)cols;
    double* comps = (double*)malloc((size_t)k * (size_t)cols * sizeof(double));
    double* inv_eig = (double*)malloc((size_t)k * sizeof(double));
    if (comps == NULL || inv_eig == NULL) {
        free(comps); free(inv_eig);
        free(M); free(V); free(eigvals);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    for (int64_t i = 0; i < k; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            comps[(size_t)i * (size_t)cols + (size_t)j] =
                V[(size_t)j * (size_t)cols + (size_t)i];
        }
        const double denom = eigvals[i] + reg;
        if (denom <= 0.0) {
            free(comps); free(inv_eig);
            free(M); free(V); free(eigvals);
            return N4M_ERR_NUMERICAL_FAILURE;
        }
        inv_eig[i] = 1.0 / denom;
    }
    flip_signs_max_abs(comps, k, cols);
    state->components  = comps;
    state->inv_eigvals = inv_eig;
    state->k           = (int32_t)k;
    free(M); free(V); free(eigvals);
    return N4M_OK;
}

/* Compute per-sample leverages on a (rows, cols) row-major matrix that has
 * already been centred (Xc). Writes to ``out`` of length rows.
 *
 * Allocates a scratch buffer of length max(cols, k); caller-free not
 * needed because the buffer is internal to this call. */
static n4m_status_t compute_leverages(const n4m_filter_leverage_state_t* state,
                                       const double* Xc, int64_t rows,
                                       int64_t cols, double* out) {
    if (state->effective_method == N4M_FILTER_LEVERAGE_METHOD_HAT) {
        double* tmp = (double*)malloc((size_t)cols * sizeof(double));
        double* sol = (double*)malloc((size_t)cols * sizeof(double));
        if (tmp == NULL || sol == NULL) {
            free(tmp); free(sol);
            return N4M_ERR_OUT_OF_MEMORY;
        }
        for (int64_t r = 0; r < rows; ++r) {
            const double* xrow = Xc + (size_t)r * (size_t)cols;
            memcpy(tmp, xrow, (size_t)cols * sizeof(double));
            const n4m_status_t qst = n4m_apply_qt(state->M_qr, cols, cols,
                                                     state->tau, tmp);
            if (qst != N4M_OK) { free(tmp); free(sol); return qst; }
            const n4m_status_t bst = n4m_back_solve_R(state->M_qr, cols, cols,
                                                         tmp, sol);
            if (bst != N4M_OK) { free(tmp); free(sol); return bst; }
            double h = 0.0;
            for (int64_t j = 0; j < cols; ++j) {
                h += xrow[j] * sol[j];
            }
            out[r] = h;
        }
        free(tmp); free(sol);
    } else {
        /* PCA: scores = Xc @ components^T, then leverage = sum scores_ij^2 / (lambda_j + reg). */
        const int64_t k = (int64_t)state->k;
        double* scores = (double*)malloc((size_t)k * sizeof(double));
        if (scores == NULL) {
            return N4M_ERR_OUT_OF_MEMORY;
        }
        for (int64_t r = 0; r < rows; ++r) {
            const double* xrow = Xc + (size_t)r * (size_t)cols;
            for (int64_t i = 0; i < k; ++i) {
                double s = 0.0;
                const double* comp = state->components + (size_t)i * (size_t)cols;
                for (int64_t j = 0; j < cols; ++j) {
                    s += comp[j] * xrow[j];
                }
                scores[i] = s;
            }
            double h = 0.0;
            for (int64_t i = 0; i < k; ++i) {
                h += scores[i] * scores[i] * state->inv_eigvals[i];
            }
            out[r] = h;
        }
        free(scores);
    }
    return N4M_OK;
}

n4m_status_t n4m_filter_leverage_state_fit(n4m_filter_leverage_state_t* state,
                                            const double* X,
                                            int64_t rows, int64_t cols) {
    if (state == NULL || X == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    if (rows < 2 || cols < 1) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    /* Reset previous fit state. */
    free(state->mean);        state->mean = NULL;
    free(state->M_qr);        state->M_qr = NULL;
    free(state->tau);         state->tau  = NULL;
    free(state->components);  state->components  = NULL;
    free(state->inv_eigvals); state->inv_eigvals = NULL;
    state->fitted   = 0;
    state->fit_cols = cols;

    /* Compute the centring vector. */
    state->mean = (double*)malloc((size_t)cols * sizeof(double));
    if (state->mean == NULL) {
        return N4M_ERR_OUT_OF_MEMORY;
    }
    if (state->center) {
        for (int64_t j = 0; j < cols; ++j) {
            double s = 0.0;
            for (int64_t i = 0; i < rows; ++i) {
                s += X[(size_t)i * (size_t)cols + (size_t)j];
            }
            state->mean[j] = s / (double)rows;
        }
    } else {
        for (int64_t j = 0; j < cols; ++j) state->mean[j] = 0.0;
    }
    /* Build the centred matrix. */
    double* Xc = (double*)malloc((size_t)rows * (size_t)cols * sizeof(double));
    if (Xc == NULL) {
        return N4M_ERR_OUT_OF_MEMORY;
    }
    if (state->center) {
        for (int64_t i = 0; i < rows; ++i) {
            for (int64_t j = 0; j < cols; ++j) {
                Xc[(size_t)i * (size_t)cols + (size_t)j] =
                    X[(size_t)i * (size_t)cols + (size_t)j] - state->mean[j];
            }
        }
    } else {
        memcpy(Xc, X, (size_t)rows * (size_t)cols * sizeof(double));
    }

    /* Decide effective method: PCA if requested or if rows <= cols. */
    n4m_status_t st;
    if (state->method == N4M_FILTER_LEVERAGE_METHOD_PCA || cols >= rows) {
        st = fit_pca(state, Xc, rows, cols);
    } else {
        st = fit_hat(state, Xc, rows, cols);
    }
    if (st != N4M_OK) {
        free(Xc);
        return st;
    }

    /* Compute training-time leverages to derive the threshold. */
    double* lev = (double*)malloc((size_t)rows * sizeof(double));
    if (lev == NULL) {
        free(Xc);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    st = compute_leverages(state, Xc, rows, cols, lev);
    if (st != N4M_OK) {
        free(lev); free(Xc);
        return st;
    }
    if (state->use_absolute) {
        state->threshold = state->absolute_threshold;
    } else {
        double sum = 0.0;
        for (int64_t i = 0; i < rows; ++i) sum += lev[i];
        const double mean_lev = sum / (double)rows;
        state->threshold = state->threshold_multiplier * mean_lev;
    }
    free(lev);
    free(Xc);
    state->fitted = 1;
    return N4M_OK;
}

n4m_status_t n4m_filter_leverage_state_apply(
    const n4m_filter_leverage_state_t* state,
    const double* X, int64_t rows, int64_t cols,
    uint8_t* mask_out,
    int64_t* out_n_samples,
    int64_t* out_n_kept,
    int64_t* out_n_excluded) {
    if (state == NULL || X == NULL || mask_out == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    if (!state->fitted) {
        return N4M_ERR_NOT_FITTED;
    }
    if (cols != state->fit_cols) {
        return N4M_ERR_SHAPE_MISMATCH;
    }
    if (rows < 0 || cols < 0) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (rows == 0) {
        if (out_n_samples)  *out_n_samples  = 0;
        if (out_n_kept)     *out_n_kept     = 0;
        if (out_n_excluded) *out_n_excluded = 0;
        return N4M_OK;
    }
    double* Xc = (double*)malloc((size_t)rows * (size_t)cols * sizeof(double));
    double* lev = (double*)malloc((size_t)rows * sizeof(double));
    if (Xc == NULL || lev == NULL) {
        free(Xc); free(lev);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    if (state->center) {
        for (int64_t i = 0; i < rows; ++i) {
            for (int64_t j = 0; j < cols; ++j) {
                Xc[(size_t)i * (size_t)cols + (size_t)j] =
                    X[(size_t)i * (size_t)cols + (size_t)j] - state->mean[j];
            }
        }
    } else {
        memcpy(Xc, X, (size_t)rows * (size_t)cols * sizeof(double));
    }
    const n4m_status_t st = compute_leverages(state, Xc, rows, cols, lev);
    if (st != N4M_OK) {
        free(Xc); free(lev);
        return st;
    }
    int64_t kept = 0;
    for (int64_t i = 0; i < rows; ++i) {
        const uint8_t keep = (lev[i] <= state->threshold) ? 1 : 0;
        mask_out[i] = keep;
        kept += keep;
    }
    if (out_n_samples)  *out_n_samples  = rows;
    if (out_n_kept)     *out_n_kept     = kept;
    if (out_n_excluded) *out_n_excluded = rows - kept;
    free(Xc); free(lev);
    return N4M_OK;
}
