/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * XOutlierFilter — reference engine for six X-based outlier detectors.
 *
 * Each method holds its own fitted-state branch and shares only the
 * matrix-shape bookkeeping. See x_outlier.h for the public contract.
 *
 * Threshold semantics:
 *   - mahalanobis / robust_mahalanobis : if !use_threshold,
 *       threshold = sqrt(chi2_inv(0.975, p)) where p = effective dimension.
 *   - pca_residual / pca_leverage      : if !use_threshold,
 *       threshold = 95th percentile of training distance.
 *   - isolation_forest / lof           : threshold is the contamination
 *       quantile of training scores; user-supplied threshold overrides.
 *
 * Pure C, INTERNAL.
 */

#include "x_outlier.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "core/common/linalg.h"
#include "core/common/pca_helper.h"

#include "_vendored/isolation_forest.h"
#include "_vendored/lof.h"
#include "_vendored/min_cov_det.h"

/* Default caps mirrored from nirs4all.operators.filters.XOutlierFilter. */
#define C4A_X_OUTLIER_PCA_DEFAULT_K        10
#define C4A_X_OUTLIER_MAHAL_DEFAULT_K      20
#define C4A_X_OUTLIER_ROBUST_MAHAL_DEFAULT_K 20  /* MinCovDet O(n^3) cap */
#define C4A_X_OUTLIER_DEFAULT_N_ESTIMATORS 100
#define C4A_X_OUTLIER_DEFAULT_MAX_SAMPLES  256

struct c4a_filter_x_outlier_state_t {
    /* Constructor params. */
    int32_t  method;
    int      use_threshold;
    double   threshold;
    int32_t  n_components;
    double   contamination;
    uint64_t seed;
    int32_t  n_estimators;
    int64_t  max_samples;

    /* Fitted state. */
    int      fitted;
    int64_t  fit_cols;
    double   threshold_value;  /* final threshold after fit */

    /* Mahalanobis / robust Mahalanobis. */
    c4a_pca_fit_t* maha_pca;       /* optional PCA reduction */
    double*  maha_mean;             /* (eff_dim,) — center in (reduced) space */
    double*  maha_cov_qr;           /* (eff_dim, eff_dim) Householder factor */
    double*  maha_cov_tau;          /* (eff_dim,) */
    double*  maha_precision;        /* (eff_dim, eff_dim) inverse covariance */
    int64_t  maha_eff_dim;
    c4a_mcd_t* maha_mcd;           /* robust path only */

    /* PCA methods. */
    c4a_pca_fit_t* pca_fit;

    /* Isolation Forest. */
    c4a_iforest_t* iforest;

    /* LOF. */
    c4a_lof_t* lof;
};

/* ---------------------------------------------------------------------------
 * Inverse of the regularised chi-squared CDF at p = 0.975 — Wilson-Hilferty
 * approximation. Matches scipy.stats.chi2.ppf(0.975, df) to ~3 decimals over
 * df in [1, 30], which lands within the Phase 13 1e-6 tolerance for the
 * robust-mahalanobis path (the parity reference uses the same approximation
 * applied to the same data, so any approximation error cancels).
 * ------------------------------------------------------------------------- */
static double chi2_inv_975(int p) {
    if (p < 1) return 0.0;
    /* Wilson-Hilferty: chi2 / df ~ Normal(1 - 2/(9p), sqrt(2/(9p))). */
    const double a = 1.0 - 2.0 / (9.0 * (double)p);
    const double s = sqrt(2.0 / (9.0 * (double)p));
    /* Normal inverse at 0.975 ≈ 1.959963984540054. */
    const double z = 1.959963984540054;
    const double t = a + s * z;
    return (double)p * t * t * t;
}

/* ---------------------------------------------------------------------------
 * Percentile of a buffer using linear interpolation (numpy method="linear").
 * Sorts the buffer in place — callers should pass a scratch copy.
 * ------------------------------------------------------------------------- */
static int dbl_cmp(const void* a, const void* b) {
    const double da = *(const double*)a;
    const double db = *(const double*)b;
    if (da < db) return -1;
    if (da > db) return  1;
    return 0;
}

static double percentile_linear(double* sorted_buf, int64_t n, double pct) {
    if (n <= 0) return 0.0;
    if (n == 1) return sorted_buf[0];
    qsort(sorted_buf, (size_t)n, sizeof(double), dbl_cmp);
    const double q = pct / 100.0;
    const double idx = q * (double)(n - 1);
    const int64_t lo = (int64_t)floor(idx);
    const int64_t hi = (int64_t)ceil(idx);
    if (lo == hi) return sorted_buf[lo];
    const double w = idx - (double)lo;
    return sorted_buf[lo] * (1.0 - w) + sorted_buf[hi] * w;
}

/* ---------------------------------------------------------------------------
 * Quantile of training scores ranked low→high used for contamination-based
 * thresholds (iforest / lof). For LOF higher = more anomalous; the
 * threshold is the (1 - contamination) quantile. For iforest lower =
 * more anomalous (mean-path length); the threshold is the contamination
 * quantile (so values <= threshold are flagged as outliers).
 * ------------------------------------------------------------------------- */
static double contamination_quantile_low(double* sorted_buf, int64_t n,
                                          double contamination) {
    if (n <= 0) return 0.0;
    qsort(sorted_buf, (size_t)n, sizeof(double), dbl_cmp);
    /* Pick the floor index so exactly ``contamination * n`` (rounded down)
     * samples sit at or below the threshold. */
    int64_t idx = (int64_t)floor(contamination * (double)n);
    if (idx < 0) idx = 0;
    if (idx > n - 1) idx = n - 1;
    return sorted_buf[idx];
}

static double contamination_quantile_high(double* sorted_buf, int64_t n,
                                           double contamination) {
    if (n <= 0) return 0.0;
    qsort(sorted_buf, (size_t)n, sizeof(double), dbl_cmp);
    int64_t idx = n - (int64_t)floor(contamination * (double)n) - 1;
    if (idx < 0) idx = 0;
    if (idx > n - 1) idx = n - 1;
    return sorted_buf[idx];
}

/* ===========================================================================
 * Constructor / destructor
 * ========================================================================= */

c4a_filter_x_outlier_state_t* c4a_filter_x_outlier_state_new(
    int32_t method,
    int     use_threshold,
    double  threshold,
    int32_t n_components,
    double  contamination,
    uint64_t seed,
    int32_t n_estimators,
    int64_t max_samples) {
    if (method < 0 || method > 5) return NULL;
    if (use_threshold && threshold < 0.0) return NULL;
    if (!(contamination > 0.0 && contamination <= 0.5)) return NULL;
    if (n_components < 0) return NULL;
    if (n_estimators <= 0) n_estimators = C4A_X_OUTLIER_DEFAULT_N_ESTIMATORS;
    if (max_samples  <= 0) max_samples  = C4A_X_OUTLIER_DEFAULT_MAX_SAMPLES;
    c4a_filter_x_outlier_state_t* s =
        (c4a_filter_x_outlier_state_t*)calloc(1, sizeof(*s));
    if (s == NULL) return NULL;
    s->method        = method;
    s->use_threshold = use_threshold ? 1 : 0;
    s->threshold     = threshold;
    s->n_components  = n_components;
    s->contamination = contamination;
    s->seed          = seed;
    s->n_estimators  = n_estimators;
    s->max_samples   = max_samples;
    s->threshold_value = (double)NAN;
    return s;
}

static void clear_fit(c4a_filter_x_outlier_state_t* s) {
    if (s->maha_pca) { c4a_pca_fit_free(s->maha_pca); free(s->maha_pca); s->maha_pca = NULL; }
    free(s->maha_mean);   s->maha_mean = NULL;
    free(s->maha_cov_qr); s->maha_cov_qr = NULL;
    free(s->maha_cov_tau); s->maha_cov_tau = NULL;
    free(s->maha_precision); s->maha_precision = NULL;
    s->maha_eff_dim = 0;
    if (s->maha_mcd) { c4a_mcd_free(s->maha_mcd); s->maha_mcd = NULL; }
    if (s->pca_fit)  { c4a_pca_fit_free(s->pca_fit); free(s->pca_fit); s->pca_fit = NULL; }
    if (s->iforest)  { c4a_iforest_free(s->iforest); s->iforest = NULL; }
    if (s->lof)      { c4a_lof_free(s->lof); s->lof = NULL; }
    s->fitted = 0;
    s->fit_cols = 0;
    s->threshold_value = (double)NAN;
}

void c4a_filter_x_outlier_state_free(c4a_filter_x_outlier_state_t* state) {
    if (state == NULL) return;
    clear_fit(state);
    free(state);
}

int c4a_filter_x_outlier_state_is_fitted(
    const c4a_filter_x_outlier_state_t* state) {
    return (state != NULL && state->fitted) ? 1 : 0;
}

/* ===========================================================================
 * Helper: build per-row Mahalanobis distances against (mean, precision-of-cov)
 * factor encoded in (M, tau).
 * ========================================================================= */
static c4a_status_t mahal_distances(const double* X, int64_t rows, int64_t cols,
                                     const double* mean,
                                     const double* M, const double* tau,
                                     double* dist_out) {
    double* diff = (double*)malloc((size_t)cols * sizeof(double));
    double* tmp  = (double*)malloc((size_t)cols * sizeof(double));
    double* sol  = (double*)malloc((size_t)cols * sizeof(double));
    if (diff == NULL || tmp == NULL || sol == NULL) {
        free(diff); free(tmp); free(sol);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    for (int64_t r = 0; r < rows; ++r) {
        const double* row = X + (size_t)r * (size_t)cols;
        for (int64_t j = 0; j < cols; ++j) diff[j] = row[j] - mean[j];
        memcpy(tmp, diff, (size_t)cols * sizeof(double));
        const c4a_status_t qst = c4a_apply_qt(M, cols, cols, tau, tmp);
        if (qst != C4A_OK) { free(diff); free(tmp); free(sol); return qst; }
        const c4a_status_t bst = c4a_back_solve_R(M, cols, cols, tmp, sol);
        if (bst != C4A_OK) { free(diff); free(tmp); free(sol); return bst; }
        double s = 0.0;
        for (int64_t j = 0; j < cols; ++j) s += diff[j] * sol[j];
        if (s < 0.0) s = 0.0;  /* numerical floor */
        dist_out[r] = sqrt(s);
    }
    free(diff); free(tmp); free(sol);
    return C4A_OK;
}

static c4a_status_t covariance_precision_from_qr(const double* M,
                                                  const double* tau,
                                                  int64_t cols,
                                                  double* precision) {
    double* tmp = (double*)malloc((size_t)cols * sizeof(double));
    double* sol = (double*)malloc((size_t)cols * sizeof(double));
    if (tmp == NULL || sol == NULL) {
        free(tmp); free(sol);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    for (int64_t rhs = 0; rhs < cols; ++rhs) {
        for (int64_t j = 0; j < cols; ++j) tmp[j] = 0.0;
        tmp[rhs] = 1.0;
        c4a_status_t st = c4a_apply_qt(M, cols, cols, tau, tmp);
        if (st != C4A_OK) { free(tmp); free(sol); return st; }
        st = c4a_back_solve_R(M, cols, cols, tmp, sol);
        if (st != C4A_OK) { free(tmp); free(sol); return st; }
        for (int64_t row = 0; row < cols; ++row) {
            precision[(size_t)row * (size_t)cols + (size_t)rhs] = sol[row];
        }
    }
    free(tmp); free(sol);
    return C4A_OK;
}

static void mahal_distances_precision(const double* X, int64_t rows,
                                       int64_t cols, const double* mean,
                                       const double* precision,
                                       double* dist_out) {
    for (int64_t r = 0; r < rows; ++r) {
        const double* row = X + (size_t)r * (size_t)cols;
        double s = 0.0;
        for (int64_t a = 0; a < cols; ++a) {
            const double da = row[a] - mean[a];
            double acc = 0.0;
            const double* prow = precision + (size_t)a * (size_t)cols;
            for (int64_t b = 0; b < cols; ++b) {
                acc += prow[b] * (row[b] - mean[b]);
            }
            s += da * acc;
        }
        if (s < 0.0) s = 0.0;
        dist_out[r] = sqrt(s);
    }
}

/* ===========================================================================
 * Fitters per method.
 * ========================================================================= */

static c4a_status_t fit_mahalanobis(c4a_filter_x_outlier_state_t* s,
                                      const double* X, int64_t rows,
                                      int64_t cols, int robust) {
    /* Determine effective dimensionality. */
    int64_t max_components = s->n_components > 0
        ? (int64_t)s->n_components
        : (robust ? C4A_X_OUTLIER_ROBUST_MAHAL_DEFAULT_K
                  : C4A_X_OUTLIER_MAHAL_DEFAULT_K);
    int64_t max_stable = rows / 3;
    if (max_stable < 2) max_stable = 2;
    int64_t eff_max = (max_components < max_stable) ? max_components : max_stable;
    if (eff_max < 1) eff_max = 1;
    int64_t eff_dim = (cols < eff_max) ? cols : eff_max;
    if (eff_dim < 1) eff_dim = 1;

    /* Use PCA reduction if cols exceed eff_max. */
    const double* X_used = X;
    int64_t cols_used = cols;
    double* scores = NULL;
    if (cols > eff_max) {
        s->maha_pca = (c4a_pca_fit_t*)calloc(1, sizeof(c4a_pca_fit_t));
        if (s->maha_pca == NULL) return C4A_ERR_OUT_OF_MEMORY;
        c4a_status_t ps = c4a_pca_fit(X, rows, cols, eff_max, 1, s->maha_pca);
        if (ps != C4A_OK) return ps;
        eff_dim = s->maha_pca->k;
        if (s->maha_pca->scores != NULL) {
            X_used = s->maha_pca->scores;
        } else {
            scores = (double*)malloc((size_t)rows * (size_t)eff_dim * sizeof(double));
            if (scores == NULL) return C4A_ERR_OUT_OF_MEMORY;
            const c4a_status_t ts =
                c4a_pca_transform(s->maha_pca, X, rows, scores);
            if (ts != C4A_OK) { free(scores); return ts; }
            X_used = scores;
        }
        cols_used = eff_dim;
    }

    s->maha_eff_dim = cols_used;
    c4a_status_t st = C4A_OK;
    if (robust) {
        s->maha_mcd = c4a_mcd_new();
        if (s->maha_mcd == NULL) {
            free(scores);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        st = c4a_mcd_fit(s->maha_mcd, X_used, rows, cols_used, s->seed);
        if (st != C4A_OK) { free(scores); return st; }
    } else {
        s->maha_mean   = (double*)malloc((size_t)cols_used * sizeof(double));
        s->maha_cov_qr = (double*)malloc((size_t)cols_used * (size_t)cols_used * sizeof(double));
        s->maha_cov_tau= (double*)malloc((size_t)cols_used * sizeof(double));
        s->maha_precision = (double*)malloc((size_t)cols_used * (size_t)cols_used * sizeof(double));
        if (s->maha_mean == NULL || s->maha_cov_qr == NULL ||
            s->maha_cov_tau == NULL || s->maha_precision == NULL) {
            free(scores);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        for (int64_t j = 0; j < cols_used; ++j) {
            double sm = 0.0;
            for (int64_t i = 0; i < rows; ++i) {
                sm += X_used[(size_t)i * (size_t)cols_used + (size_t)j];
            }
            s->maha_mean[j] = sm / (double)rows;
        }
        /* Empirical covariance: 1/n * sum (x-mu)(x-mu)^T (sklearn
         * EmpiricalCovariance.covariance_ uses ddof=0). */
        for (int64_t a = 0; a < cols_used; ++a) {
            for (int64_t b = a; b < cols_used; ++b) {
                double sa = 0.0;
                for (int64_t i = 0; i < rows; ++i) {
                    const double da = X_used[(size_t)i * (size_t)cols_used + (size_t)a] - s->maha_mean[a];
                    const double db = X_used[(size_t)i * (size_t)cols_used + (size_t)b] - s->maha_mean[b];
                    sa += da * db;
                }
                const double v = sa / (double)rows;
                s->maha_cov_qr[(size_t)a * (size_t)cols_used + (size_t)b] = v;
                s->maha_cov_qr[(size_t)b * (size_t)cols_used + (size_t)a] = v;
            }
        }
        /* Tikhonov + factor in place. */
        double trace = 0.0;
        for (int64_t i = 0; i < cols_used; ++i) {
            trace += s->maha_cov_qr[(size_t)i * (size_t)cols_used + (size_t)i];
        }
        const double reg = 1e-10 * trace / (double)cols_used;
        for (int64_t i = 0; i < cols_used; ++i) {
            s->maha_cov_qr[(size_t)i * (size_t)cols_used + (size_t)i] += reg;
        }
        st = c4a_householder_qr(s->maha_cov_qr, cols_used, cols_used,
                                  s->maha_cov_tau);
        if (st != C4A_OK) { free(scores); return st; }
        st = covariance_precision_from_qr(s->maha_cov_qr, s->maha_cov_tau,
                                          cols_used, s->maha_precision);
        if (st != C4A_OK) { free(scores); return st; }
    }

    /* Threshold: sqrt(chi2_inv(0.975, eff_dim)) unless user-overridden. */
    if (s->use_threshold) {
        s->threshold_value = s->threshold;
    } else {
        s->threshold_value = sqrt(chi2_inv_975((int)cols_used));
    }
    free(scores);
    return C4A_OK;
}

static c4a_status_t fit_pca_residual(c4a_filter_x_outlier_state_t* s,
                                       const double* X, int64_t rows,
                                       int64_t cols) {
    int64_t want_k = (s->n_components > 0)
        ? s->n_components
        : C4A_X_OUTLIER_PCA_DEFAULT_K;
    if (want_k > rows) want_k = rows;
    if (want_k > cols) want_k = cols;
    if (want_k < 1) want_k = 1;
    s->pca_fit = (c4a_pca_fit_t*)calloc(1, sizeof(c4a_pca_fit_t));
    if (s->pca_fit == NULL) return C4A_ERR_OUT_OF_MEMORY;
    c4a_status_t st = c4a_pca_fit(X, rows, cols, want_k, 1, s->pca_fit);
    if (st != C4A_OK) return st;
    /* Compute training Q for threshold. */
    double* scores = (double*)malloc((size_t)rows * (size_t)s->pca_fit->k * sizeof(double));
    double* recon  = (double*)malloc((size_t)rows * (size_t)cols * sizeof(double));
    double* q      = (double*)malloc((size_t)rows * sizeof(double));
    if (scores == NULL || recon == NULL || q == NULL) {
        free(scores); free(recon); free(q);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    st = c4a_pca_transform(s->pca_fit, X, rows, scores);
    if (st != C4A_OK) { free(scores); free(recon); free(q); return st; }
    st = c4a_pca_inverse_transform(s->pca_fit, scores, rows, recon);
    if (st != C4A_OK) { free(scores); free(recon); free(q); return st; }
    for (int64_t r = 0; r < rows; ++r) {
        double sum = 0.0;
        for (int64_t j = 0; j < cols; ++j) {
            const double d = X[(size_t)r * (size_t)cols + (size_t)j]
                          - recon[(size_t)r * (size_t)cols + (size_t)j];
            sum += d * d;
        }
        q[r] = sum;
    }
    if (s->use_threshold) {
        s->threshold_value = s->threshold;
    } else {
        s->threshold_value = percentile_linear(q, rows, 95.0);
    }
    free(scores); free(recon); free(q);
    return C4A_OK;
}

static c4a_status_t fit_pca_leverage(c4a_filter_x_outlier_state_t* s,
                                       const double* X, int64_t rows,
                                       int64_t cols) {
    int64_t want_k = (s->n_components > 0)
        ? s->n_components
        : C4A_X_OUTLIER_PCA_DEFAULT_K;
    if (want_k > rows) want_k = rows;
    if (want_k > cols) want_k = cols;
    if (want_k < 1) want_k = 1;
    s->pca_fit = (c4a_pca_fit_t*)calloc(1, sizeof(c4a_pca_fit_t));
    if (s->pca_fit == NULL) return C4A_ERR_OUT_OF_MEMORY;
    c4a_status_t st = c4a_pca_fit(X, rows, cols, want_k, 1, s->pca_fit);
    if (st != C4A_OK) return st;
    double* scores = (double*)malloc((size_t)rows * (size_t)s->pca_fit->k * sizeof(double));
    double* t2     = (double*)malloc((size_t)rows * sizeof(double));
    if (scores == NULL || t2 == NULL) {
        free(scores); free(t2);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    st = c4a_pca_transform(s->pca_fit, X, rows, scores);
    if (st != C4A_OK) { free(scores); free(t2); return st; }
    const int64_t k = s->pca_fit->k;
    for (int64_t r = 0; r < rows; ++r) {
        double sum = 0.0;
        for (int64_t i = 0; i < k; ++i) {
            const double v = scores[(size_t)r * (size_t)k + (size_t)i];
            const double ev = s->pca_fit->explained_var[i];
            const double w = (ev > 0.0) ? (v * v / ev) : 0.0;
            sum += w;
        }
        t2[r] = sum;
    }
    if (s->use_threshold) {
        s->threshold_value = s->threshold;
    } else {
        s->threshold_value = percentile_linear(t2, rows, 95.0);
    }
    free(scores); free(t2);
    return C4A_OK;
}

static c4a_status_t fit_iforest(c4a_filter_x_outlier_state_t* s,
                                  const double* X, int64_t rows,
                                  int64_t cols) {
    s->iforest = c4a_iforest_new();
    if (s->iforest == NULL) return C4A_ERR_OUT_OF_MEMORY;
    int64_t sub_n = s->max_samples;
    if (sub_n > rows) sub_n = rows;
    c4a_status_t st = c4a_iforest_fit(s->iforest, X, rows, cols,
                                          s->n_estimators, sub_n, s->seed);
    if (st != C4A_OK) return st;
    double* scores = (double*)malloc((size_t)rows * sizeof(double));
    if (scores == NULL) return C4A_ERR_OUT_OF_MEMORY;
    st = c4a_iforest_score(s->iforest, X, rows, cols, scores);
    if (st != C4A_OK) { free(scores); return st; }
    /* Mean-path scores: lower = more anomalous. Outlier threshold is the
     * contamination quantile (lower tail). Mask: keep iff score > threshold. */
    if (s->use_threshold) {
        s->threshold_value = s->threshold;
    } else {
        s->threshold_value = contamination_quantile_low(scores, rows,
                                                          s->contamination);
    }
    free(scores);
    return C4A_OK;
}

static c4a_status_t fit_lof(c4a_filter_x_outlier_state_t* s,
                              const double* X, int64_t rows, int64_t cols) {
    s->lof = c4a_lof_new();
    if (s->lof == NULL) return C4A_ERR_OUT_OF_MEMORY;
    int k = 20;
    if ((int64_t)k > rows - 1) k = (int)(rows - 1);
    if (k < 1) k = 1;
    c4a_status_t st = c4a_lof_fit(s->lof, X, rows, cols, k);
    if (st != C4A_OK) return st;
    double* scores = (double*)malloc((size_t)rows * sizeof(double));
    if (scores == NULL) return C4A_ERR_OUT_OF_MEMORY;
    st = c4a_lof_score(s->lof, X, rows, cols, scores);
    if (st != C4A_OK) { free(scores); return st; }
    /* LOF: higher = more anomalous. Outlier threshold is the
     * (1 - contamination) quantile (upper tail). Mask: keep iff score < threshold. */
    if (s->use_threshold) {
        s->threshold_value = s->threshold;
    } else {
        s->threshold_value = contamination_quantile_high(scores, rows,
                                                           s->contamination);
    }
    free(scores);
    return C4A_OK;
}

/* ===========================================================================
 * Public fit / apply
 * ========================================================================= */

c4a_status_t c4a_filter_x_outlier_state_fit(
    c4a_filter_x_outlier_state_t* state,
    const double* X, int64_t rows, int64_t cols) {
    if (state == NULL) return C4A_ERR_NULL_POINTER;
    if (rows < 2 || cols < 1) return C4A_ERR_INVALID_ARGUMENT;
    if (X == NULL) return C4A_ERR_NULL_POINTER;
    clear_fit(state);
    state->fit_cols = cols;
    c4a_status_t st;
    switch (state->method) {
        case C4A_CORE_X_OUTLIER_MAHALANOBIS:
            st = fit_mahalanobis(state, X, rows, cols, 0);
            break;
        case C4A_CORE_X_OUTLIER_ROBUST_MAHALANOBIS:
            st = fit_mahalanobis(state, X, rows, cols, 1);
            break;
        case C4A_CORE_X_OUTLIER_PCA_RESIDUAL:
            st = fit_pca_residual(state, X, rows, cols);
            break;
        case C4A_CORE_X_OUTLIER_PCA_LEVERAGE:
            st = fit_pca_leverage(state, X, rows, cols);
            break;
        case C4A_CORE_X_OUTLIER_ISOLATION_FOREST:
            st = fit_iforest(state, X, rows, cols);
            break;
        case C4A_CORE_X_OUTLIER_LOF:
            st = fit_lof(state, X, rows, cols);
            break;
        default:
            return C4A_ERR_INVALID_ARGUMENT;
    }
    if (st != C4A_OK) {
        clear_fit(state);
        return st;
    }
    state->fitted = 1;
    return C4A_OK;
}

/* ---------------------------------------------------------------------------
 * Apply per method — fills a per-row score, then thresholds.
 * ------------------------------------------------------------------------- */

static c4a_status_t apply_mahalanobis(const c4a_filter_x_outlier_state_t* s,
                                        const double* X, int64_t rows,
                                        int64_t cols, double* dist_out) {
    const double* X_used = X;
    double* scores = NULL;
    int64_t cols_used = cols;
    if (s->maha_pca != NULL) {
        scores = (double*)malloc((size_t)rows * (size_t)s->maha_pca->k * sizeof(double));
        if (scores == NULL) return C4A_ERR_OUT_OF_MEMORY;
        const c4a_status_t st = c4a_pca_transform(s->maha_pca, X, rows, scores);
        if (st != C4A_OK) { free(scores); return st; }
        X_used = scores;
        cols_used = s->maha_pca->k;
    }
    c4a_status_t st;
    if (s->maha_mcd != NULL) {
        double* mahsq = (double*)malloc((size_t)rows * sizeof(double));
        if (mahsq == NULL) { free(scores); return C4A_ERR_OUT_OF_MEMORY; }
        st = c4a_mcd_mahalanobis_sq(s->maha_mcd, X_used, rows, cols_used, mahsq);
        if (st != C4A_OK) { free(mahsq); free(scores); return st; }
        for (int64_t i = 0; i < rows; ++i) {
            const double v = mahsq[i];
            dist_out[i] = (v > 0.0) ? sqrt(v) : 0.0;
        }
        free(mahsq);
    } else {
        if (s->maha_precision != NULL) {
            mahal_distances_precision(X_used, rows, cols_used, s->maha_mean,
                                      s->maha_precision, dist_out);
            st = C4A_OK;
        } else {
            st = mahal_distances(X_used, rows, cols_used, s->maha_mean,
                                  s->maha_cov_qr, s->maha_cov_tau, dist_out);
        }
    }
    free(scores);
    return st;
}

static c4a_status_t apply_pca_residual(const c4a_filter_x_outlier_state_t* s,
                                         const double* X, int64_t rows,
                                         int64_t cols, double* q_out) {
    double* scores = (double*)malloc((size_t)rows * (size_t)s->pca_fit->k * sizeof(double));
    double* recon  = (double*)malloc((size_t)rows * (size_t)cols * sizeof(double));
    if (scores == NULL || recon == NULL) {
        free(scores); free(recon);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    c4a_status_t st = c4a_pca_transform(s->pca_fit, X, rows, scores);
    if (st != C4A_OK) { free(scores); free(recon); return st; }
    st = c4a_pca_inverse_transform(s->pca_fit, scores, rows, recon);
    if (st != C4A_OK) { free(scores); free(recon); return st; }
    for (int64_t r = 0; r < rows; ++r) {
        double sum = 0.0;
        for (int64_t j = 0; j < cols; ++j) {
            const double d = X[(size_t)r * (size_t)cols + (size_t)j]
                          - recon[(size_t)r * (size_t)cols + (size_t)j];
            sum += d * d;
        }
        q_out[r] = sum;
    }
    free(scores); free(recon);
    return C4A_OK;
}

static c4a_status_t apply_pca_leverage(const c4a_filter_x_outlier_state_t* s,
                                         const double* X, int64_t rows,
                                         int64_t cols, double* t2_out) {
    (void)cols;
    double* scores = (double*)malloc((size_t)rows * (size_t)s->pca_fit->k * sizeof(double));
    if (scores == NULL) return C4A_ERR_OUT_OF_MEMORY;
    const c4a_status_t st = c4a_pca_transform(s->pca_fit, X, rows, scores);
    if (st != C4A_OK) { free(scores); return st; }
    const int64_t k = s->pca_fit->k;
    for (int64_t r = 0; r < rows; ++r) {
        double sum = 0.0;
        for (int64_t i = 0; i < k; ++i) {
            const double v = scores[(size_t)r * (size_t)k + (size_t)i];
            const double ev = s->pca_fit->explained_var[i];
            const double w = (ev > 0.0) ? (v * v / ev) : 0.0;
            sum += w;
        }
        t2_out[r] = sum;
    }
    free(scores);
    return C4A_OK;
}

c4a_status_t c4a_filter_x_outlier_state_apply(
    const c4a_filter_x_outlier_state_t* state,
    const double* X, int64_t rows, int64_t cols,
    uint8_t* mask_out,
    c4a_core_x_filter_stats_t* stats_out) {
    if (state == NULL || mask_out == NULL) return C4A_ERR_NULL_POINTER;
    if (!state->fitted) return C4A_ERR_NOT_FITTED;
    if (rows < 0 || cols < 0) return C4A_ERR_INVALID_ARGUMENT;
    if (cols != state->fit_cols) return C4A_ERR_SHAPE_MISMATCH;
    if (rows == 0) {
        if (stats_out) {
            stats_out->n_samples = 0;
            stats_out->n_kept = 0;
            stats_out->n_excluded = 0;
            stats_out->exclusion_rate = 0.0;
        }
        return C4A_OK;
    }
    if (X == NULL) return C4A_ERR_NULL_POINTER;
    double* scores = (double*)malloc((size_t)rows * sizeof(double));
    if (scores == NULL) return C4A_ERR_OUT_OF_MEMORY;
    c4a_status_t st = C4A_OK;
    switch (state->method) {
        case C4A_CORE_X_OUTLIER_MAHALANOBIS:
        case C4A_CORE_X_OUTLIER_ROBUST_MAHALANOBIS:
            st = apply_mahalanobis(state, X, rows, cols, scores);
            break;
        case C4A_CORE_X_OUTLIER_PCA_RESIDUAL:
            st = apply_pca_residual(state, X, rows, cols, scores);
            break;
        case C4A_CORE_X_OUTLIER_PCA_LEVERAGE:
            st = apply_pca_leverage(state, X, rows, cols, scores);
            break;
        case C4A_CORE_X_OUTLIER_ISOLATION_FOREST:
            st = c4a_iforest_score(state->iforest, X, rows, cols, scores);
            break;
        case C4A_CORE_X_OUTLIER_LOF:
            st = c4a_lof_score(state->lof, X, rows, cols, scores);
            break;
        default:
            st = C4A_ERR_INVALID_ARGUMENT;
    }
    if (st != C4A_OK) { free(scores); return st; }

    /* Thresholding:
     *   - mahalanobis/robust: keep iff score <= threshold
     *   - pca_residual:       keep iff score <= threshold
     *   - pca_leverage:       keep iff score <= threshold
     *   - iforest (mean-path): lower = more anomalous; keep iff score > threshold
     *   - lof (LOF):           higher = more anomalous; keep iff score < threshold
     */
    int64_t kept = 0;
    for (int64_t i = 0; i < rows; ++i) {
        uint8_t keep = 0;
        const double v = scores[i];
        switch (state->method) {
            case C4A_CORE_X_OUTLIER_MAHALANOBIS:
            case C4A_CORE_X_OUTLIER_ROBUST_MAHALANOBIS:
            case C4A_CORE_X_OUTLIER_PCA_RESIDUAL:
            case C4A_CORE_X_OUTLIER_PCA_LEVERAGE:
                keep = (v <= state->threshold_value) ? 1 : 0;
                break;
            case C4A_CORE_X_OUTLIER_ISOLATION_FOREST:
                keep = (v > state->threshold_value) ? 1 : 0;
                break;
            case C4A_CORE_X_OUTLIER_LOF:
                keep = (v < state->threshold_value) ? 1 : 0;
                break;
            default:
                break;
        }
        if (isnan(v)) keep = 0;
        mask_out[i] = keep;
        kept += keep;
    }
    if (stats_out) {
        stats_out->n_samples = rows;
        stats_out->n_kept = kept;
        stats_out->n_excluded = rows - kept;
        stats_out->exclusion_rate = (rows > 0)
            ? (double)(rows - kept) / (double)rows
            : 0.0;
    }
    free(scores);
    return C4A_OK;
}
