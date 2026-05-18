/* SPDX-License-Identifier: CECILL-2.1
 *
 * R/C dispatcher for the chemometrics4all R binding (Phase 23).
 *
 * Each `r_*` entry point is registered with R via R_CallMethodDef and
 * exposed as `C_*` in package.R / ffi.R (see R_init_chemometrics4all
 * at the bottom of this file).
 *
 * Layout / stride convention:
 *
 *   R matrices are *column-major* with row_stride = 1, col_stride = rows.
 *   libc4a's c4a_matrix_view_t supports column-major natively via
 *   c4a_matrix_view_init_colmajor — so we pass R buffers in without any
 *   transpose. However, the libc4a public C ABI documents that operator
 *   inputs must be row-major contiguous F64 (see the §5 prelude of
 *   c4a.h, last paragraph). Empirically the implementations of every
 *   operator under test in this binding accept column-major inputs
 *   correctly (because the underlying core uses the stride fields), but
 *   to stay strictly on the documented contract we materialise a
 *   row-major working buffer on the way in and copy back into a
 *   column-major R matrix on the way out. The extra allocation is two
 *   matrix-sized double buffers per call — acceptable for binding work.
 *
 * Error handling: any C ABI status != C4A_OK is reported via Rf_error
 * with the canonical c4a_status_to_string text. We do NOT use a
 * c4a_context_t for these wrappers — the operators we expose do not
 * consume one, and the per-status string is sufficient at the binding
 * boundary.
 */

#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <R_ext/Visibility.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "chemometrics4all/c4a.h"

/* ------------------------------------------------------------------ */
/* Small helpers                                                       */
/* ------------------------------------------------------------------ */

static void copy_R_colmajor_to_rowmajor(const double *src, double *dst,
                                        int64_t rows, int64_t cols) {
    /* R: src[i + j*rows]  ->  rowmajor: dst[i*cols + j] */
    for (int64_t j = 0; j < cols; ++j) {
        for (int64_t i = 0; i < rows; ++i) {
            dst[i * cols + j] = src[i + j * rows];
        }
    }
}

static void copy_rowmajor_to_R_colmajor(const double *src, double *dst,
                                        int64_t rows, int64_t cols) {
    for (int64_t j = 0; j < cols; ++j) {
        for (int64_t i = 0; i < rows; ++i) {
            dst[i + j * rows] = src[i * cols + j];
        }
    }
}

static void c4a_throw(c4a_status_t status, const char *op) {
    const char *msg = c4a_status_to_string(status);
    Rf_error("chemometrics4all: %s failed (status=%d, %s)", op, (int)status,
             msg ? msg : "<no message>");
}

static c4a_status_t init_view_rowmajor(c4a_matrix_view_t *out, void *data,
                                       int64_t rows, int64_t cols) {
    return c4a_matrix_view_init_rowmajor(out, data, rows, cols, C4A_DTYPE_F64);
}

/* Allocate a row-major working copy from an R column-major matrix `x`.
 * Returns the pointer to a newly-allocated buffer; the caller MUST free()
 * it. Stores the (rows, cols) into *out_rows, *out_cols. */
static double *r_matrix_to_rowmajor(SEXP x, int64_t *out_rows,
                                    int64_t *out_cols) {
    SEXP dim = Rf_getAttrib(x, R_DimSymbol);
    if (dim == R_NilValue || Rf_length(dim) != 2) {
        Rf_error("chemometrics4all: input must be a 2-D matrix");
    }
    int64_t rows = INTEGER(dim)[0];
    int64_t cols = INTEGER(dim)[1];
    double *buf = (double *)R_alloc((size_t)(rows * cols), sizeof(double));
    copy_R_colmajor_to_rowmajor(REAL(x), buf, rows, cols);
    *out_rows = rows;
    *out_cols = cols;
    return buf;
}

/* Allocate a row-major output buffer of size (rows*cols), produce an R
 * column-major matrix from it. */
static SEXP rowmajor_to_R_matrix(const double *src, int64_t rows,
                                 int64_t cols) {
    SEXP out = PROTECT(Rf_allocMatrix(REALSXP, (int)rows, (int)cols));
    copy_rowmajor_to_R_colmajor(src, REAL(out), rows, cols);
    UNPROTECT(1);
    return out;
}

/* ------------------------------------------------------------------ */
/* Library / ABI helpers                                               */
/* ------------------------------------------------------------------ */

static SEXP r_c4a_abi_version(void) {
    SEXP v = PROTECT(Rf_allocVector(INTSXP, 3));
    INTEGER(v)[0] = (int)c4a_get_abi_version_major();
    INTEGER(v)[1] = (int)c4a_get_abi_version_minor();
    INTEGER(v)[2] = (int)c4a_get_abi_version_patch();
    UNPROTECT(1);
    return v;
}

static SEXP r_c4a_version_string(void) {
    const char *s = c4a_get_version_string();
    return Rf_mkString(s ? s : "");
}

/* ------------------------------------------------------------------ */
/* Preprocessing wrappers                                              */
/* ------------------------------------------------------------------ */

static SEXP r_c4a_pp_snv_transform(SEXP x, SEXP with_mean, SEXP with_std,
                                   SEXP ddof) {
    int64_t rows, cols;
    double *in = r_matrix_to_rowmajor(x, &rows, &cols);
    double *out_buf = (double *)R_alloc((size_t)(rows * cols), sizeof(double));

    c4a_pp_snv_handle_t *h = NULL;
    c4a_status_t s = c4a_pp_snv_create(&h,
                                       LOGICAL(with_mean)[0],
                                       LOGICAL(with_std)[0],
                                       INTEGER(ddof)[0]);
    if (s != C4A_OK) c4a_throw(s, "c4a_pp_snv_create");

    c4a_matrix_view_t Xv, Yv;
    if ((s = init_view_rowmajor(&Xv, in, rows, cols)) != C4A_OK) {
        c4a_pp_snv_destroy(h); c4a_throw(s, "matrix_view (in)");
    }
    if ((s = init_view_rowmajor(&Yv, out_buf, rows, cols)) != C4A_OK) {
        c4a_pp_snv_destroy(h); c4a_throw(s, "matrix_view (out)");
    }
    s = c4a_pp_snv_transform(h, Xv, Yv);
    c4a_pp_snv_destroy(h);
    if (s != C4A_OK) c4a_throw(s, "c4a_pp_snv_transform");

    return rowmajor_to_R_matrix(out_buf, rows, cols);
}

static SEXP r_c4a_pp_msc_fit_transform(SEXP x_fit, SEXP x) {
    int64_t fit_rows, fit_cols, rows, cols;
    double *fit_in = r_matrix_to_rowmajor(x_fit, &fit_rows, &fit_cols);
    double *in     = r_matrix_to_rowmajor(x, &rows, &cols);
    double *out_buf = (double *)R_alloc((size_t)(rows * cols), sizeof(double));

    c4a_pp_msc_handle_t *h = NULL;
    c4a_status_t s = c4a_pp_msc_create(&h);
    if (s != C4A_OK) c4a_throw(s, "c4a_pp_msc_create");

    c4a_matrix_view_t Xfit_v, Xv, Yv;
    init_view_rowmajor(&Xfit_v, fit_in, fit_rows, fit_cols);
    init_view_rowmajor(&Xv,     in,     rows,     cols);
    init_view_rowmajor(&Yv,     out_buf, rows,    cols);

    s = c4a_pp_msc_fit(h, Xfit_v);
    if (s != C4A_OK) { c4a_pp_msc_destroy(h); c4a_throw(s, "c4a_pp_msc_fit"); }
    s = c4a_pp_msc_transform(h, Xv, Yv);
    c4a_pp_msc_destroy(h);
    if (s != C4A_OK) c4a_throw(s, "c4a_pp_msc_transform");

    return rowmajor_to_R_matrix(out_buf, rows, cols);
}

static SEXP r_c4a_pp_emsc_fit_transform(SEXP x_fit, SEXP x, SEXP degree) {
    int64_t fit_rows, fit_cols, rows, cols;
    double *fit_in = r_matrix_to_rowmajor(x_fit, &fit_rows, &fit_cols);
    double *in     = r_matrix_to_rowmajor(x, &rows, &cols);
    double *out_buf = (double *)R_alloc((size_t)(rows * cols), sizeof(double));

    c4a_pp_emsc_handle_t *h = NULL;
    c4a_status_t s = c4a_pp_emsc_create(&h, INTEGER(degree)[0]);
    if (s != C4A_OK) c4a_throw(s, "c4a_pp_emsc_create");

    c4a_matrix_view_t Xfit_v, Xv, Yv;
    init_view_rowmajor(&Xfit_v, fit_in, fit_rows, fit_cols);
    init_view_rowmajor(&Xv,     in,     rows,     cols);
    init_view_rowmajor(&Yv,     out_buf, rows,    cols);

    s = c4a_pp_emsc_fit(h, Xfit_v);
    if (s != C4A_OK) { c4a_pp_emsc_destroy(h); c4a_throw(s, "c4a_pp_emsc_fit"); }
    s = c4a_pp_emsc_transform(h, Xv, Yv);
    c4a_pp_emsc_destroy(h);
    if (s != C4A_OK) c4a_throw(s, "c4a_pp_emsc_transform");

    return rowmajor_to_R_matrix(out_buf, rows, cols);
}

static SEXP r_c4a_pp_savgol_transform(SEXP x, SEXP window_length,
                                      SEXP polyorder, SEXP deriv, SEXP delta,
                                      SEXP mode, SEXP cval) {
    int64_t rows, cols;
    double *in = r_matrix_to_rowmajor(x, &rows, &cols);
    double *out_buf = (double *)R_alloc((size_t)(rows * cols), sizeof(double));

    c4a_pp_savgol_handle_t *h = NULL;
    c4a_status_t s = c4a_pp_savgol_create(&h,
                                          INTEGER(window_length)[0],
                                          INTEGER(polyorder)[0],
                                          INTEGER(deriv)[0],
                                          REAL(delta)[0],
                                          (c4a_pp_savgol_mode_t)INTEGER(mode)[0],
                                          REAL(cval)[0]);
    if (s != C4A_OK) c4a_throw(s, "c4a_pp_savgol_create");

    c4a_matrix_view_t Xv, Yv;
    init_view_rowmajor(&Xv, in, rows, cols);
    init_view_rowmajor(&Yv, out_buf, rows, cols);

    s = c4a_pp_savgol_transform(h, Xv, Yv);
    c4a_pp_savgol_destroy(h);
    if (s != C4A_OK) c4a_throw(s, "c4a_pp_savgol_transform");

    return rowmajor_to_R_matrix(out_buf, rows, cols);
}

static SEXP r_c4a_pp_first_derivative_transform(SEXP x, SEXP delta,
                                                SEXP edge_order) {
    int64_t rows, cols;
    double *in = r_matrix_to_rowmajor(x, &rows, &cols);
    double *out_buf = (double *)R_alloc((size_t)(rows * cols), sizeof(double));

    c4a_pp_first_derivative_handle_t *h = NULL;
    c4a_status_t s = c4a_pp_first_derivative_create(&h, REAL(delta)[0],
                                                    INTEGER(edge_order)[0]);
    if (s != C4A_OK) c4a_throw(s, "c4a_pp_first_derivative_create");

    c4a_matrix_view_t Xv, Yv;
    init_view_rowmajor(&Xv, in, rows, cols);
    init_view_rowmajor(&Yv, out_buf, rows, cols);

    s = c4a_pp_first_derivative_transform(h, Xv, Yv);
    c4a_pp_first_derivative_destroy(h);
    if (s != C4A_OK) c4a_throw(s, "c4a_pp_first_derivative_transform");

    return rowmajor_to_R_matrix(out_buf, rows, cols);
}

static SEXP r_c4a_pp_second_derivative_transform(SEXP x, SEXP delta,
                                                 SEXP edge_order) {
    int64_t rows, cols;
    double *in = r_matrix_to_rowmajor(x, &rows, &cols);
    double *out_buf = (double *)R_alloc((size_t)(rows * cols), sizeof(double));

    c4a_pp_second_derivative_handle_t *h = NULL;
    c4a_status_t s = c4a_pp_second_derivative_create(&h, REAL(delta)[0],
                                                     INTEGER(edge_order)[0]);
    if (s != C4A_OK) c4a_throw(s, "c4a_pp_second_derivative_create");

    c4a_matrix_view_t Xv, Yv;
    init_view_rowmajor(&Xv, in, rows, cols);
    init_view_rowmajor(&Yv, out_buf, rows, cols);

    s = c4a_pp_second_derivative_transform(h, Xv, Yv);
    c4a_pp_second_derivative_destroy(h);
    if (s != C4A_OK) c4a_throw(s, "c4a_pp_second_derivative_transform");

    return rowmajor_to_R_matrix(out_buf, rows, cols);
}

static SEXP r_c4a_pp_to_absorbance_transform(SEXP x, SEXP is_percent,
                                             SEXP epsilon, SEXP clip_negative) {
    int64_t rows, cols;
    double *in = r_matrix_to_rowmajor(x, &rows, &cols);
    double *out_buf = (double *)R_alloc((size_t)(rows * cols), sizeof(double));

    c4a_pp_to_absorbance_handle_t *h = NULL;
    c4a_status_t s = c4a_pp_to_absorbance_create(&h,
                                                 LOGICAL(is_percent)[0],
                                                 REAL(epsilon)[0],
                                                 LOGICAL(clip_negative)[0]);
    if (s != C4A_OK) c4a_throw(s, "c4a_pp_to_absorbance_create");

    c4a_matrix_view_t Xv, Yv;
    init_view_rowmajor(&Xv, in, rows, cols);
    init_view_rowmajor(&Yv, out_buf, rows, cols);

    s = c4a_pp_to_absorbance_transform(h, Xv, Yv);
    c4a_pp_to_absorbance_destroy(h);
    if (s != C4A_OK) c4a_throw(s, "c4a_pp_to_absorbance_transform");

    return rowmajor_to_R_matrix(out_buf, rows, cols);
}

static SEXP r_c4a_pp_kubelka_munk_transform(SEXP x, SEXP is_percent,
                                            SEXP epsilon) {
    int64_t rows, cols;
    double *in = r_matrix_to_rowmajor(x, &rows, &cols);
    double *out_buf = (double *)R_alloc((size_t)(rows * cols), sizeof(double));

    c4a_pp_kubelka_munk_handle_t *h = NULL;
    c4a_status_t s = c4a_pp_kubelka_munk_create(&h, LOGICAL(is_percent)[0],
                                                REAL(epsilon)[0]);
    if (s != C4A_OK) c4a_throw(s, "c4a_pp_kubelka_munk_create");

    c4a_matrix_view_t Xv, Yv;
    init_view_rowmajor(&Xv, in, rows, cols);
    init_view_rowmajor(&Yv, out_buf, rows, cols);

    s = c4a_pp_kubelka_munk_transform(h, Xv, Yv);
    c4a_pp_kubelka_munk_destroy(h);
    if (s != C4A_OK) c4a_throw(s, "c4a_pp_kubelka_munk_transform");

    return rowmajor_to_R_matrix(out_buf, rows, cols);
}

static SEXP r_c4a_pp_detrend_transform(SEXP x, SEXP polyorder) {
    int64_t rows, cols;
    double *in = r_matrix_to_rowmajor(x, &rows, &cols);
    double *out_buf = (double *)R_alloc((size_t)(rows * cols), sizeof(double));

    c4a_pp_detrend_handle_t *h = NULL;
    c4a_status_t s = c4a_pp_detrend_create(&h, INTEGER(polyorder)[0]);
    if (s != C4A_OK) c4a_throw(s, "c4a_pp_detrend_create");

    c4a_matrix_view_t Xv, Yv;
    init_view_rowmajor(&Xv, in, rows, cols);
    init_view_rowmajor(&Yv, out_buf, rows, cols);

    s = c4a_pp_detrend_transform(h, Xv, Yv);
    c4a_pp_detrend_destroy(h);
    if (s != C4A_OK) c4a_throw(s, "c4a_pp_detrend_transform");

    return rowmajor_to_R_matrix(out_buf, rows, cols);
}

static SEXP r_c4a_pp_asls_transform(SEXP x, SEXP lam, SEXP p, SEXP max_iter,
                                    SEXP tol) {
    int64_t rows, cols;
    double *in = r_matrix_to_rowmajor(x, &rows, &cols);
    double *out_buf = (double *)R_alloc((size_t)(rows * cols), sizeof(double));

    c4a_pp_asls_handle_t *h = NULL;
    c4a_status_t s = c4a_pp_asls_create(&h, REAL(lam)[0], REAL(p)[0],
                                        INTEGER(max_iter)[0], REAL(tol)[0]);
    if (s != C4A_OK) c4a_throw(s, "c4a_pp_asls_create");

    c4a_matrix_view_t Xv, Yv;
    init_view_rowmajor(&Xv, in, rows, cols);
    init_view_rowmajor(&Yv, out_buf, rows, cols);

    s = c4a_pp_asls_transform(h, Xv, Yv);
    c4a_pp_asls_destroy(h);
    if (s != C4A_OK) c4a_throw(s, "c4a_pp_asls_transform");

    return rowmajor_to_R_matrix(out_buf, rows, cols);
}

static SEXP r_c4a_pp_airpls_transform(SEXP x, SEXP lam, SEXP max_iter,
                                      SEXP tol) {
    int64_t rows, cols;
    double *in = r_matrix_to_rowmajor(x, &rows, &cols);
    double *out_buf = (double *)R_alloc((size_t)(rows * cols), sizeof(double));

    c4a_pp_airpls_handle_t *h = NULL;
    c4a_status_t s = c4a_pp_airpls_create(&h, REAL(lam)[0],
                                          INTEGER(max_iter)[0], REAL(tol)[0]);
    if (s != C4A_OK) c4a_throw(s, "c4a_pp_airpls_create");

    c4a_matrix_view_t Xv, Yv;
    init_view_rowmajor(&Xv, in, rows, cols);
    init_view_rowmajor(&Yv, out_buf, rows, cols);

    s = c4a_pp_airpls_transform(h, Xv, Yv);
    c4a_pp_airpls_destroy(h);
    if (s != C4A_OK) c4a_throw(s, "c4a_pp_airpls_transform");

    return rowmajor_to_R_matrix(out_buf, rows, cols);
}

static SEXP r_c4a_pp_wavelet_denoise_transform(SEXP x, SEXP family,
                                               SEXP boundary, SEXP level,
                                               SEXP threshold_mode,
                                               SEXP noise_estimator) {
    int64_t rows, cols;
    double *in = r_matrix_to_rowmajor(x, &rows, &cols);
    double *out_buf = (double *)R_alloc((size_t)(rows * cols), sizeof(double));

    c4a_pp_wavelet_denoise_handle_t *h = NULL;
    c4a_status_t s = c4a_pp_wavelet_denoise_create(
        &h,
        (c4a_pp_wavelet_family_t)INTEGER(family)[0],
        (c4a_pp_wavelet_boundary_t)INTEGER(boundary)[0],
        INTEGER(level)[0],
        (c4a_pp_wavelet_threshold_t)INTEGER(threshold_mode)[0],
        (c4a_pp_wavelet_noise_t)INTEGER(noise_estimator)[0]);
    if (s != C4A_OK) c4a_throw(s, "c4a_pp_wavelet_denoise_create");

    c4a_matrix_view_t Xv, Yv;
    init_view_rowmajor(&Xv, in, rows, cols);
    init_view_rowmajor(&Yv, out_buf, rows, cols);

    s = c4a_pp_wavelet_denoise_transform(h, Xv, Yv);
    c4a_pp_wavelet_denoise_destroy(h);
    if (s != C4A_OK) c4a_throw(s, "c4a_pp_wavelet_denoise_transform");

    return rowmajor_to_R_matrix(out_buf, rows, cols);
}

/* ------------------------------------------------------------------ */
/* Splitter wrappers                                                   */
/* ------------------------------------------------------------------ */

static SEXP build_split_result(const c4a_split_result_t *r) {
    SEXP train = PROTECT(Rf_allocVector(INTSXP, (R_xlen_t)r->n_train));
    SEXP test  = PROTECT(Rf_allocVector(INTSXP, (R_xlen_t)r->n_test));
    for (int64_t i = 0; i < r->n_train; ++i) INTEGER(train)[i] = (int)(r->train_idx[i] + 1);
    for (int64_t i = 0; i < r->n_test;  ++i) INTEGER(test)[i]  = (int)(r->test_idx[i]  + 1);
    SEXP out = PROTECT(Rf_allocVector(VECSXP, 2));
    SET_VECTOR_ELT(out, 0, train);
    SET_VECTOR_ELT(out, 1, test);
    SEXP names = PROTECT(Rf_allocVector(STRSXP, 2));
    SET_STRING_ELT(names, 0, Rf_mkChar("train_idx"));
    SET_STRING_ELT(names, 1, Rf_mkChar("test_idx"));
    Rf_setAttrib(out, R_NamesSymbol, names);
    UNPROTECT(4);
    return out;
}

static SEXP r_c4a_split_kennard_stone_split(SEXP x, SEXP test_size) {
    int64_t rows, cols;
    double *in = r_matrix_to_rowmajor(x, &rows, &cols);

    c4a_split_kennard_stone_handle_t *h = NULL;
    c4a_status_t s = c4a_split_kennard_stone_create(&h, REAL(test_size)[0]);
    if (s != C4A_OK) c4a_throw(s, "c4a_split_kennard_stone_create");

    c4a_matrix_view_t Xv;
    init_view_rowmajor(&Xv, in, rows, cols);

    c4a_split_result_t r = {0};
    s = c4a_split_kennard_stone_split(h, Xv, &r);
    c4a_split_kennard_stone_destroy(h);
    if (s != C4A_OK) {
        c4a_split_result_destroy(&r);
        c4a_throw(s, "c4a_split_kennard_stone_split");
    }
    SEXP out = build_split_result(&r);
    c4a_split_result_destroy(&r);
    return out;
}

static SEXP r_c4a_split_spxy_split(SEXP x, SEXP y, SEXP test_size) {
    int64_t rows, cols, y_rows, y_cols;
    double *in_x = r_matrix_to_rowmajor(x, &rows, &cols);
    double *in_y = r_matrix_to_rowmajor(y, &y_rows, &y_cols);

    c4a_split_spxy_handle_t *h = NULL;
    c4a_status_t s = c4a_split_spxy_create(&h, REAL(test_size)[0]);
    if (s != C4A_OK) c4a_throw(s, "c4a_split_spxy_create");

    c4a_matrix_view_t Xv, Yv;
    init_view_rowmajor(&Xv, in_x, rows, cols);
    init_view_rowmajor(&Yv, in_y, y_rows, y_cols);

    c4a_split_result_t r = {0};
    s = c4a_split_spxy_split(h, Xv, Yv, &r);
    c4a_split_spxy_destroy(h);
    if (s != C4A_OK) {
        c4a_split_result_destroy(&r);
        c4a_throw(s, "c4a_split_spxy_split");
    }
    SEXP out = build_split_result(&r);
    c4a_split_result_destroy(&r);
    return out;
}

/* ------------------------------------------------------------------ */
/* Filter wrappers                                                     */
/* ------------------------------------------------------------------ */

static SEXP r_c4a_filter_y_outlier_fit_apply(SEXP y, SEXP method,
                                             SEXP threshold,
                                             SEXP lower_pct,
                                             SEXP upper_pct) {
    R_xlen_t n = Rf_length(y);
    double *yv = REAL(y);

    c4a_filter_y_outlier_handle_t *h = NULL;
    c4a_status_t s = c4a_filter_y_outlier_create(
        &h, (c4a_y_outlier_method_t)INTEGER(method)[0],
        REAL(threshold)[0], REAL(lower_pct)[0], REAL(upper_pct)[0]);
    if (s != C4A_OK) c4a_throw(s, "c4a_filter_y_outlier_create");

    s = c4a_filter_y_outlier_fit(h, yv, (int64_t)n);
    if (s != C4A_OK) {
        c4a_filter_y_outlier_destroy(h);
        c4a_throw(s, "c4a_filter_y_outlier_fit");
    }

    uint8_t *mask = (uint8_t *)R_alloc((size_t)n, sizeof(uint8_t));
    c4a_filter_stats_t stats = {0};
    s = c4a_filter_y_outlier_apply(h, yv, (int64_t)n, mask, &stats);
    c4a_filter_y_outlier_destroy(h);
    if (s != C4A_OK) c4a_throw(s, "c4a_filter_y_outlier_apply");

    SEXP mask_R = PROTECT(Rf_allocVector(LGLSXP, n));
    for (R_xlen_t i = 0; i < n; ++i) LOGICAL(mask_R)[i] = mask[i] ? TRUE : FALSE;

    const char *names[] = {"mask", "n_samples", "n_kept", "n_excluded",
                           "exclusion_rate", ""};
    SEXP out = PROTECT(Rf_mkNamed(VECSXP, names));
    SET_VECTOR_ELT(out, 0, mask_R);
    SET_VECTOR_ELT(out, 1, Rf_ScalarReal((double)stats.n_samples));
    SET_VECTOR_ELT(out, 2, Rf_ScalarReal((double)stats.n_kept));
    SET_VECTOR_ELT(out, 3, Rf_ScalarReal((double)stats.n_excluded));
    SET_VECTOR_ELT(out, 4, Rf_ScalarReal(stats.exclusion_rate));
    UNPROTECT(2);
    return out;
}

/* ------------------------------------------------------------------ */
/* Method registration                                                 */
/* ------------------------------------------------------------------ */

/* The R-side names use a `C_` prefix to match the autogenerated symbol
 * objects produced by `useDynLib(chemometrics4all, .registration = TRUE)`
 * (each entry below becomes a `NativeSymbolInfo` named exactly as the
 * first column of the table — we choose the `C_*` convention so that
 * the corresponding R-side `.Call(C_<name>, ...)` reads naturally). */
#define CDEF(name, nargs) {"C_" #name, (DL_FUNC)&r_ ## name, nargs}

static const R_CallMethodDef R_CallEntries[] = {
    CDEF(c4a_abi_version, 0),
    CDEF(c4a_version_string, 0),

    CDEF(c4a_pp_snv_transform, 4),                /* X, with_mean, with_std, ddof */
    CDEF(c4a_pp_msc_fit_transform, 2),            /* X_fit, X */
    CDEF(c4a_pp_emsc_fit_transform, 3),           /* X_fit, X, degree */
    CDEF(c4a_pp_savgol_transform, 7),             /* X, w, p, deriv, delta, mode, cval */
    CDEF(c4a_pp_first_derivative_transform, 3),   /* X, delta, edge_order */
    CDEF(c4a_pp_second_derivative_transform, 3),  /* X, delta, edge_order */
    CDEF(c4a_pp_to_absorbance_transform, 4),      /* X, is_percent, epsilon, clip_neg */
    CDEF(c4a_pp_kubelka_munk_transform, 3),       /* X, is_percent, epsilon */
    CDEF(c4a_pp_detrend_transform, 2),            /* X, polyorder */
    CDEF(c4a_pp_asls_transform, 5),               /* X, lam, p, max_iter, tol */
    CDEF(c4a_pp_airpls_transform, 4),             /* X, lam, max_iter, tol */
    CDEF(c4a_pp_wavelet_denoise_transform, 6),    /* X, family, boundary, level, thr, noise */

    CDEF(c4a_split_kennard_stone_split, 2),       /* X, test_size */
    CDEF(c4a_split_spxy_split, 3),                /* X, Y, test_size */

    CDEF(c4a_filter_y_outlier_fit_apply, 5),      /* y, method, threshold, lo, hi */

    {NULL, NULL, 0}
};

/* The R load entrypoint registers C symbols. We use the
 * R_CallMethodDef table and let R generate the C_<name> R-side
 * accessors automatically (because we passed .registration = TRUE in
 * useDynLib). */
attribute_visible void R_init_chemometrics4all(DllInfo *dll) {
    R_registerRoutines(dll, NULL, R_CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
    R_forceSymbols(dll, TRUE);
}
