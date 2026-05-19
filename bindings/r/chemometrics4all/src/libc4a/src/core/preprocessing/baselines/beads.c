/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * BEADS — Ning & Selesnick 2014.
 *
 * Canonical external reference: pybaselines.Baseline().beads with
 * fit_parabola=True, filter_type=1, cost_function=2, asymmetry=6,
 * freq_cutoff=0.005, eps_0=eps_1=1e-6, and no derivative smoothing.
 *
 * The implementation keeps pybaselines' banded formulation but specializes it
 * to filter_type=1, so A and B are tridiagonal and the iterative system has
 * four lower and four upper diagonals.
 */

#include "beads.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define C4A_BEADS_FREQ_CUTOFF 0.005
#define C4A_BEADS_ASYMMETRY 6.0
#define C4A_BEADS_EPS0 1.0e-6
#define C4A_BEADS_EPS1 1.0e-6
#define C4A_BEADS_PI 3.141592653589793238462643383279502884

struct c4a_pp_beads_state_t {
    double  lam_0;
    double  lam_1;
    double  lam_2;
    int32_t max_iter;
    double  tol;
};

c4a_pp_beads_state_t* c4a_pp_beads_state_new(double lam_0, double lam_1,
                                              double lam_2,
                                              int32_t max_iter, double tol) {
    if (!(lam_0 > 0.0) || !(lam_1 > 0.0) || !(lam_2 > 0.0) ||
        max_iter < 0 || !(tol >= 0.0)) {
        return NULL;
    }
    c4a_pp_beads_state_t* s =
        (c4a_pp_beads_state_t*)malloc(sizeof(*s));
    if (s == NULL) return NULL;
    s->lam_0    = lam_0;
    s->lam_1    = lam_1;
    s->lam_2    = lam_2;
    s->max_iter = max_iter;
    s->tol      = tol;
    return s;
}

void c4a_pp_beads_state_free(c4a_pp_beads_state_t* state) {
    free(state);
}

static inline double band_get(const double* ab, int32_t lower, int32_t upper,
                              int64_t n, int64_t i, int64_t j) {
    const int64_t row = (int64_t)upper + i - j;
    if (row < 0 || row >= (int64_t)lower + (int64_t)upper + 1) {
        return 0.0;
    }
    return ab[(size_t)row * (size_t)n + (size_t)j];
}

static inline double* band_ptr(double* ab, int32_t lower, int32_t upper,
                               int64_t n, int64_t i, int64_t j) {
    const int64_t row = (int64_t)upper + i - j;
    if (row < 0 || row >= (int64_t)lower + (int64_t)upper + 1) {
        return NULL;
    }
    return ab + (size_t)row * (size_t)n + (size_t)j;
}

static void band_dot_vector(const double* ab, int32_t lower, int32_t upper,
                            const double* x, int64_t n, double* out) {
    for (int64_t i = 0; i < n; ++i) {
        const int64_t lo = i > lower ? i - lower : 0;
        const int64_t hi = (i + upper < n) ? i + upper : n - 1;
        double sum = 0.0;
        for (int64_t j = lo; j <= hi; ++j) {
            sum += band_get(ab, lower, upper, n, i, j) * x[j];
        }
        out[i] = sum;
    }
}

static void band_dot_banded(const double* a, int32_t a_lower, int32_t a_upper,
                            const double* b, int32_t b_lower, int32_t b_upper,
                            int64_t n, double* out) {
    const int32_t c_lower = a_lower + b_lower;
    const int32_t c_upper = a_upper + b_upper;
    memset(out, 0, (size_t)(c_lower + c_upper + 1) * (size_t)n * sizeof(double));
    for (int64_t i = 0; i < n; ++i) {
        const int64_t k_lo = i > a_lower ? i - a_lower : 0;
        const int64_t k_hi = (i + a_upper < n) ? i + a_upper : n - 1;
        for (int64_t k = k_lo; k <= k_hi; ++k) {
            const double aik = band_get(a, a_lower, a_upper, n, i, k);
            if (aik == 0.0) continue;
            const int64_t j_lo = k > b_lower ? k - b_lower : 0;
            const int64_t j_hi = (k + b_upper < n) ? k + b_upper : n - 1;
            for (int64_t j = j_lo; j <= j_hi; ++j) {
                double* cij = band_ptr(out, c_lower, c_upper, n, i, j);
                if (cij != NULL) {
                    *cij += aik * band_get(b, b_lower, b_upper, n, k, j);
                }
            }
        }
    }
}

static c4a_status_t solve_banded_no_pivot(double* ab, int32_t lower,
                                           int32_t upper, int64_t n,
                                           double* rhs, double* x) {
    for (int64_t k = 0; k < n - 1; ++k) {
        const double pivot = band_get(ab, lower, upper, n, k, k);
        if (fabs(pivot) <= DBL_EPSILON) {
            return C4A_ERR_INVALID_ARGUMENT;
        }
        const int64_t i_hi = (k + lower < n) ? k + lower : n - 1;
        const int64_t j_hi = (k + upper < n) ? k + upper : n - 1;
        for (int64_t i = k + 1; i <= i_hi; ++i) {
            double* aik_ptr = band_ptr(ab, lower, upper, n, i, k);
            if (aik_ptr == NULL || *aik_ptr == 0.0) continue;
            const double factor = *aik_ptr / pivot;
            *aik_ptr = 0.0;
            for (int64_t j = k + 1; j <= j_hi; ++j) {
                double* aij = band_ptr(ab, lower, upper, n, i, j);
                if (aij != NULL) {
                    *aij -= factor * band_get(ab, lower, upper, n, k, j);
                }
            }
            rhs[i] -= factor * rhs[k];
        }
    }

    for (int64_t i = n - 1; i >= 0; --i) {
        double sum = rhs[i];
        const int64_t hi = (i + upper < n) ? i + upper : n - 1;
        for (int64_t j = i + 1; j <= hi; ++j) {
            sum -= band_get(ab, lower, upper, n, i, j) * x[j];
        }
        const double diag = band_get(ab, lower, upper, n, i, i);
        if (fabs(diag) <= DBL_EPSILON) {
            return C4A_ERR_INVALID_ARGUMENT;
        }
        x[i] = sum / diag;
    }
    return C4A_OK;
}

static c4a_status_t solve_tridiagonal_constant(double off, double diag,
                                                const double* rhs, int64_t n,
                                                double* cprime,
                                                double* dprime,
                                                double* out) {
    if (n <= 0 || fabs(diag) <= DBL_EPSILON) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    cprime[0] = (n > 1) ? off / diag : 0.0;
    dprime[0] = rhs[0] / diag;
    for (int64_t i = 1; i < n; ++i) {
        const double denom = diag - off * cprime[i - 1];
        if (fabs(denom) <= DBL_EPSILON) {
            return C4A_ERR_INVALID_ARGUMENT;
        }
        cprime[i] = (i < n - 1) ? off / denom : 0.0;
        dprime[i] = (rhs[i] - off * dprime[i - 1]) / denom;
    }
    out[n - 1] = dprime[n - 1];
    for (int64_t i = n - 2; i >= 0; --i) {
        out[i] = dprime[i] - cprime[i] * out[i + 1];
    }
    return C4A_OK;
}

static void build_high_pass(int64_t n, double* A, double* B,
                            double* out_off, double* out_diag) {
    const double cos_freq = cos(2.0 * C4A_BEADS_PI * C4A_BEADS_FREQ_CUTOFF);
    const double denom = (1.0 + cos_freq) > DBL_EPSILON
        ? (1.0 + cos_freq) : DBL_EPSILON;
    const double t = (1.0 - cos_freq) / denom;
    const double a_off = -1.0 + t;
    const double a_diag = 2.0 + 2.0 * t;
    memset(A, 0, (size_t)3 * (size_t)n * sizeof(double));
    memset(B, 0, (size_t)3 * (size_t)n * sizeof(double));
    for (int64_t j = 0; j < n; ++j) {
        A[(size_t)n + (size_t)j] = a_diag;
        B[(size_t)n + (size_t)j] = 2.0;
        if (j > 0) {
            A[j] = a_off;
            B[j] = -1.0;
        }
        if (j < n - 1) {
            A[(size_t)2 * (size_t)n + (size_t)j] = a_off;
            B[(size_t)2 * (size_t)n + (size_t)j] = -1.0;
        }
    }
    *out_off = a_off;
    *out_diag = a_diag;
}

static void subtract_endpoint_parabola(const double* y, int64_t n,
                                       double* parabola, double* centered) {
    double ymin = y[0];
    for (int64_t i = 1; i < n; ++i) {
        if (y[i] < ymin) ymin = y[i];
    }
    const double y1 = y[0] - ymin;
    const double y2 = y[n - 1] - ymin;
    const double c = 0.5 * (y2 + y1);
    const double b = c - y1;
    const double scale = 2.0 / (double)(n - 1);
    for (int64_t i = 0; i < n; ++i) {
        const double x = -1.0 + scale * (double)i;
        parabola[i] = ymin + b * x + c * x * x;
        centered[i] = y[i] - parabola[i];
    }
}

static void abs_diffs(const double* x, int64_t n, double* d1, double* d2) {
    for (int64_t i = 0; i < n - 1; ++i) {
        d1[i] = fabs(x[i + 1] - x[i]);
    }
    for (int64_t i = 0; i < n - 2; ++i) {
        d2[i] = fabs(x[i + 2] - 2.0 * x[i + 1] + x[i]);
    }
}

static void build_difference_diags(const double* d1, const double* d2,
                                   int64_t n, double lam_1, double lam_2,
                                   double lam_0, const double* x,
                                   double* scratch, double* out) {
    const size_t total = (size_t)5 * (size_t)n;
    double* d1_diag = scratch;
    double* d2_diag = scratch + total;
    memset(d1_diag, 0, total * sizeof(double));
    memset(d2_diag, 0, total * sizeof(double));
    memset(out, 0, total * sizeof(double));

    for (int64_t j = 1; j < n; ++j) {
        const double w = -1.0 / (d1[j - 1] + C4A_BEADS_EPS1);
        d1_diag[(size_t)1 * (size_t)n + (size_t)j] = w;
        d1_diag[(size_t)3 * (size_t)n + (size_t)(j - 1)] = w;
    }
    for (int64_t j = 0; j < n; ++j) {
        d1_diag[(size_t)2 * (size_t)n + (size_t)j] =
            -(d1_diag[(size_t)1 * (size_t)n + (size_t)j] +
              d1_diag[(size_t)3 * (size_t)n + (size_t)j]);
    }

    for (int64_t j = 2; j < n; ++j) {
        d2_diag[j] = 1.0 / (d2[j - 2] + C4A_BEADS_EPS1);
    }
    for (int64_t j = 0; j < n - 2; ++j) {
        d2_diag[(size_t)4 * (size_t)n + (size_t)j] =
            1.0 / (d2[j] + C4A_BEADS_EPS1);
    }
    for (int64_t j = 0; j < n; ++j) {
        const double row0 = d2_diag[j];
        const double rolled = (j + 1 < n) ? d2_diag[j + 1] : d2_diag[0];
        d2_diag[(size_t)n + (size_t)j] = 2.0 * (row0 - rolled) - 4.0 * row0;
    }
    for (int64_t j = 0; j < n - 1; ++j) {
        d2_diag[(size_t)3 * (size_t)n + (size_t)j] =
            d2_diag[(size_t)n + (size_t)(j + 1)];
    }
    for (int64_t j = 0; j < n; ++j) {
        d2_diag[(size_t)2 * (size_t)n + (size_t)j] =
            -(d2_diag[j] +
              d2_diag[(size_t)n + (size_t)j] +
              d2_diag[(size_t)3 * (size_t)n + (size_t)j] +
              d2_diag[(size_t)4 * (size_t)n + (size_t)j]);
    }

    for (size_t i = 0; i < total; ++i) {
        out[i] = lam_1 * d1_diag[i] + lam_2 * d2_diag[i];
    }
    const double gamma_factor = lam_0 * (1.0 + C4A_BEADS_ASYMMETRY) / 2.0;
    for (int64_t j = 0; j < n; ++j) {
        const double ax = fabs(x[j]);
        out[(size_t)2 * (size_t)n + (size_t)j] +=
            (ax > C4A_BEADS_EPS0)
                ? gamma_factor / ax
                : gamma_factor / C4A_BEADS_EPS0;
    }
}

static double beads_theta(const double* x, int64_t n) {
    const double asym = C4A_BEADS_ASYMMETRY;
    double theta = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double xi = x[i];
        if (xi > C4A_BEADS_EPS0) {
            theta += xi;
        } else if (xi < -C4A_BEADS_EPS0) {
            theta -= asym * xi;
        } else {
            theta += ((1.0 + asym) / (4.0 * C4A_BEADS_EPS0)) * xi * xi
                + ((1.0 - asym) / 2.0) * xi
                + C4A_BEADS_EPS0 * (1.0 + asym) / 4.0;
        }
    }
    return theta;
}

static double beads_loss_sum(const double* values, int64_t n) {
    double total = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double x = values[i];
        total += x - C4A_BEADS_EPS1 * log(x + C4A_BEADS_EPS1);
    }
    return total;
}

static double dot_self(const double* x, int64_t n) {
    double total = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        total += x[i] * x[i];
    }
    return total;
}
c4a_status_t c4a_pp_beads_state_apply(const c4a_pp_beads_state_t* state,
                                       const double* X,
                                       int64_t rows, int64_t cols,
                                       double* out) {
    if (state == NULL || X == NULL || out == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (rows < 0 || cols < 0) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (rows == 0 || cols == 0) {
        return C4A_OK;
    }
    if (cols < 3) {
        return C4A_ERR_INVALID_ARGUMENT;
    }

    const double  lam_0    = state->lam_0;
    const double  lam_1    = state->lam_1;
    const double  lam_2    = state->lam_2;
    const int32_t max_iter = state->max_iter;
    const double  tol      = state->tol;
    const int64_t n        = cols;

    double* A = (double*)malloc((size_t)3 * (size_t)n * sizeof(double));
    double* B = (double*)malloc((size_t)3 * (size_t)n * sizeof(double));
    double* BTB = (double*)malloc((size_t)5 * (size_t)n * sizeof(double));
    double* temp7 = (double*)malloc((size_t)7 * (size_t)n * sizeof(double));
    double* temp9 = (double*)malloc((size_t)9 * (size_t)n * sizeof(double));
    double* diag_scratch = (double*)malloc((size_t)10 * (size_t)n * sizeof(double));
    double* d_diag = (double*)malloc((size_t)5 * (size_t)n * sizeof(double));
    double* parabola = (double*)malloc((size_t)n * sizeof(double));
    double* y_work = (double*)malloc((size_t)n * sizeof(double));
    double* x = (double*)malloc((size_t)n * sizeof(double));
    double* diff = (double*)malloc((size_t)n * sizeof(double));
    double* d = (double*)malloc((size_t)n * sizeof(double));
    double* rhs = (double*)malloc((size_t)n * sizeof(double));
    double* sol = (double*)malloc((size_t)n * sizeof(double));
    double* tmp = (double*)malloc((size_t)n * sizeof(double));
    double* tmp2 = (double*)malloc((size_t)n * sizeof(double));
    double* d1_abs = (double*)malloc((size_t)n * sizeof(double));
    double* d2_abs = (double*)malloc((size_t)n * sizeof(double));
    double* cprime = (double*)malloc((size_t)n * sizeof(double));
    double* dprime = (double*)malloc((size_t)n * sizeof(double));
    if (!A || !B || !BTB || !temp7 || !temp9 || !diag_scratch || !d_diag ||
        !parabola || !y_work || !x || !diff || !d || !rhs || !sol ||
        !tmp || !tmp2 || !d1_abs || !d2_abs || !cprime || !dprime) {
        free(A); free(B); free(BTB); free(temp7); free(temp9);
        free(diag_scratch); free(d_diag); free(parabola); free(y_work);
        free(x); free(diff); free(d); free(rhs); free(sol); free(tmp);
        free(tmp2); free(d1_abs); free(d2_abs); free(cprime); free(dprime);
        return C4A_ERR_OUT_OF_MEMORY;
    }

    double a_off = 0.0;
    double a_diag = 0.0;
    build_high_pass(n, A, B, &a_off, &a_diag);
    band_dot_banded(B, 1, 1, B, 1, 1, n, BTB);

    for (int64_t r = 0; r < rows; ++r) {
        const double* y = X + (size_t)r * (size_t)cols;
        subtract_endpoint_parabola(y, n, parabola, y_work);

        c4a_status_t st = solve_tridiagonal_constant(
            a_off, a_diag, y_work, n, cprime, dprime, sol);
        if (st != C4A_OK) {
            free(A); free(B); free(BTB); free(temp7); free(temp9);
            free(diag_scratch); free(d_diag); free(parabola); free(y_work);
            free(x); free(diff); free(d); free(rhs); free(sol); free(tmp);
            free(tmp2); free(d1_abs); free(d2_abs); free(cprime); free(dprime);
            return st;
        }
        band_dot_vector(BTB, 2, 2, sol, n, d);
        for (int64_t i = 0; i < n; ++i) {
            tmp[i] = lam_0 * (1.0 - C4A_BEADS_ASYMMETRY) / 2.0;
        }
        band_dot_vector(A, 1, 1, tmp, n, tmp2);
        for (int64_t i = 0; i < n; ++i) {
            d[i] -= tmp2[i];
            x[i] = y_work[i];
        }

        double cost_old = 0.0;
        for (int32_t it = 0; it <= max_iter; ++it) {
            abs_diffs(x, n, d1_abs, d2_abs);
            build_difference_diags(d1_abs, d2_abs, n, lam_1, lam_2, lam_0,
                                   x, diag_scratch, d_diag);
            band_dot_banded(A, 1, 1, d_diag, 2, 2, n, temp7);
            band_dot_banded(temp7, 3, 3, A, 1, 1, n, temp9);
            for (int32_t row = 0; row < 5; ++row) {
                double* dst = temp9 + (size_t)(row + 2) * (size_t)n;
                const double* src = BTB + (size_t)row * (size_t)n;
                for (int64_t j = 0; j < n; ++j) {
                    dst[j] += src[j];
                }
            }
            memcpy(rhs, d, (size_t)n * sizeof(double));
            st = solve_banded_no_pivot(temp9, 4, 4, n, rhs, sol);
            if (st != C4A_OK) {
                free(A); free(B); free(BTB); free(temp7); free(temp9);
                free(diag_scratch); free(d_diag); free(parabola); free(y_work);
                free(x); free(diff); free(d); free(rhs); free(sol); free(tmp);
                free(tmp2); free(d1_abs); free(d2_abs); free(cprime); free(dprime);
                return st;
            }
            band_dot_vector(A, 1, 1, sol, n, x);

            abs_diffs(x, n, d1_abs, d2_abs);
            for (int64_t i = 0; i < n; ++i) {
                diff[i] = y_work[i] - x[i];
            }
            st = solve_tridiagonal_constant(a_off, a_diag, diff, n,
                                            cprime, dprime, sol);
            if (st != C4A_OK) {
                free(A); free(B); free(BTB); free(temp7); free(temp9);
                free(diag_scratch); free(d_diag); free(parabola); free(y_work);
                free(x); free(diff); free(d); free(rhs); free(sol); free(tmp);
                free(tmp2); free(d1_abs); free(d2_abs); free(cprime); free(dprime);
                return st;
            }
            band_dot_vector(B, 1, 1, sol, n, tmp);
            const double cost =
                0.5 * dot_self(tmp, n)
                + lam_0 * beads_theta(x, n)
                + lam_1 * beads_loss_sum(d1_abs, n - 1)
                + lam_2 * beads_loss_sum(d2_abs, n - 2);
            const double denom = fabs(cost_old) > DBL_EPSILON
                ? fabs(cost_old) : DBL_EPSILON;
            const double rdiff = fabs(cost - cost_old) / denom;
            if (rdiff < tol) {
                break;
            }
            cost_old = cost;
        }

        for (int64_t i = 0; i < n; ++i) {
            diff[i] = y_work[i] - x[i];
        }
        st = solve_tridiagonal_constant(a_off, a_diag, diff, n,
                                        cprime, dprime, sol);
        if (st != C4A_OK) {
            free(A); free(B); free(BTB); free(temp7); free(temp9);
            free(diag_scratch); free(d_diag); free(parabola); free(y_work);
            free(x); free(diff); free(d); free(rhs); free(sol); free(tmp);
            free(tmp2); free(d1_abs); free(d2_abs); free(cprime); free(dprime);
            return st;
        }
        band_dot_vector(B, 1, 1, sol, n, tmp);
        double* orow = out + (size_t)r * (size_t)cols;
        for (int64_t i = 0; i < n; ++i) {
            const double baseline = diff[i] - tmp[i] + parabola[i];
            orow[i] = y[i] - baseline;
        }
    }

    free(A); free(B); free(BTB); free(temp7); free(temp9);
    free(diag_scratch); free(d_diag); free(parabola); free(y_work);
    free(x); free(diff); free(d); free(rhs); free(sol); free(tmp);
    free(tmp2); free(d1_abs); free(d2_abs); free(cprime); free(dprime);
    return C4A_OK;
}
