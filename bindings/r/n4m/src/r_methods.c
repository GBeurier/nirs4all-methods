/* SPDX-License-Identifier: CECILL-2.1
 *
 * R/C dispatcher for the nirs4all-methods R binding (Phase 23).
 *
 * Each `r_*` entry point is registered with R via R_CallMethodDef and
 * exposed as `C_*` in package.R / ffi.R (see R_init_n4m
 * at the bottom of this file).
 *
 * Layout / stride convention:
 *
 *   R matrices are *column-major* with row_stride = 1, col_stride = rows.
 *   libn4m's n4m_matrix_view_t supports column-major natively via
 *   n4m_matrix_view_init_colmajor — so we pass R buffers in without any
 *   transpose. However, the libn4m public C ABI documents that operator
 *   inputs must be row-major contiguous F64 (see the §5 prelude of
 *   n4m.h, last paragraph). Empirically the implementations of every
 *   operator under test in this binding accept column-major inputs
 *   correctly (because the underlying core uses the stride fields), but
 *   to stay strictly on the documented contract we materialise a
 *   row-major working buffer on the way in and copy back into a
 *   column-major R matrix on the way out. The extra allocation is two
 *   matrix-sized double buffers per call — acceptable for binding work.
 *
 * Error handling: any C ABI status != N4M_OK is reported via Rf_error
 * with the canonical n4m_status_to_string text. We do NOT use a
 * n4m_context_t for these wrappers — the operators we expose do not
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

#include "n4m/n4m.h"

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

static void n4m_throw(n4m_status_t status, const char *op) {
    const char *msg = n4m_status_to_string(status);
    Rf_error("nirs4all-methods: %s failed (status=%d, %s)", op, (int)status,
             msg ? msg : "<no message>");
}

static n4m_status_t init_view_rowmajor(n4m_matrix_view_t *out, void *data,
                                       int64_t rows, int64_t cols) {
    return n4m_matrix_view_init_rowmajor(out, data, rows, cols, N4M_DTYPE_F64);
}

/* Allocate a row-major working copy from an R column-major matrix `x`.
 * Returns the pointer to a newly-allocated buffer; the caller MUST free()
 * it. Stores the (rows, cols) into *out_rows, *out_cols. */
static double *r_matrix_to_rowmajor(SEXP x, int64_t *out_rows,
                                    int64_t *out_cols) {
    SEXP dim = Rf_getAttrib(x, R_DimSymbol);
    if (dim == R_NilValue || Rf_length(dim) != 2) {
        Rf_error("nirs4all-methods: input must be a 2-D matrix");
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

static SEXP rowmajor_i32_to_R_matrix(const int32_t *src, int64_t rows,
                                     int64_t cols) {
    SEXP out = PROTECT(Rf_allocMatrix(INTSXP, (int)rows, (int)cols));
    for (int64_t j = 0; j < cols; ++j) {
        for (int64_t i = 0; i < rows; ++i) {
            INTEGER(out)[i + j * rows] = (int)src[i * cols + j];
        }
    }
    UNPROTECT(1);
    return out;
}

static void prepare_same_shape_matrix(SEXP x,
                                      int64_t *rows, int64_t *cols,
                                      double **in, double **out_buf,
                                      n4m_matrix_view_t *Xv,
                                      n4m_matrix_view_t *Yv,
                                      const char *op) {
    *in = r_matrix_to_rowmajor(x, rows, cols);
    *out_buf = (double *)R_alloc((size_t)(*rows * *cols), sizeof(double));
    n4m_status_t s = init_view_rowmajor(Xv, *in, *rows, *cols);
    if (s != N4M_OK) n4m_throw(s, op);
    s = init_view_rowmajor(Yv, *out_buf, *rows, *cols);
    if (s != N4M_OK) n4m_throw(s, op);
}

static double r_aug_param(SEXP params, int idx, const char *op) {
    if (TYPEOF(params) != REALSXP || Rf_length(params) <= idx) {
        Rf_error("nirs4all-methods: %s missing numeric parameter %d", op, idx);
    }
    return REAL(params)[idx];
}

static int32_t r_aug_param_i32(SEXP params, int idx, const char *op) {
    return (int32_t)r_aug_param(params, idx, op);
}

static int r_aug_param_int(SEXP params, int idx, const char *op) {
    return (int)r_aug_param(params, idx, op);
}

static n4m_rng_pcg64_state_t *r_create_rng(SEXP seed, const char *op) {
    n4m_rng_pcg64_state_t *rng = NULL;
    n4m_status_t s = n4m_rng_pcg64_create((uint64_t)REAL(seed)[0], &rng);
    if (s != N4M_OK) n4m_throw(s, op);
    return rng;
}

static double *r_wavelengths_to_rowmajor(SEXP wavelengths, int64_t cols,
                                         const char *op) {
    if (wavelengths == R_NilValue || TYPEOF(wavelengths) != REALSXP ||
        Rf_length(wavelengths) != cols) {
        Rf_error("nirs4all-methods: %s requires numeric wavelengths with length ncol(X)",
                 op);
    }
    double *wl = (double *)R_alloc((size_t)cols, sizeof(double));
    memcpy(wl, REAL(wavelengths), (size_t)cols * sizeof(double));
    return wl;
}

/* ------------------------------------------------------------------ */
/* Library / ABI helpers                                               */
/* ------------------------------------------------------------------ */

static SEXP r_n4m_abi_version(void) {
    SEXP v = PROTECT(Rf_allocVector(INTSXP, 3));
    INTEGER(v)[0] = (int)n4m_get_abi_version_major();
    INTEGER(v)[1] = (int)n4m_get_abi_version_minor();
    INTEGER(v)[2] = (int)n4m_get_abi_version_patch();
    UNPROTECT(1);
    return v;
}

static SEXP r_n4m_version_string(void) {
    const char *s = n4m_get_version_string();
    return Rf_mkString(s ? s : "");
}

/* ------------------------------------------------------------------ */
/* Preprocessing wrappers                                              */
/* ------------------------------------------------------------------ */

static SEXP r_n4m_pp_snv_transform(SEXP x, SEXP with_mean, SEXP with_std,
                                   SEXP ddof) {
    int64_t rows, cols;
    double *in = r_matrix_to_rowmajor(x, &rows, &cols);
    double *out_buf = (double *)R_alloc((size_t)(rows * cols), sizeof(double));

    n4m_pp_snv_handle_t *h = NULL;
    n4m_status_t s = n4m_pp_snv_create(&h,
                                       LOGICAL(with_mean)[0],
                                       LOGICAL(with_std)[0],
                                       INTEGER(ddof)[0]);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_snv_create");

    n4m_matrix_view_t Xv, Yv;
    if ((s = init_view_rowmajor(&Xv, in, rows, cols)) != N4M_OK) {
        n4m_pp_snv_destroy(h); n4m_throw(s, "matrix_view (in)");
    }
    if ((s = init_view_rowmajor(&Yv, out_buf, rows, cols)) != N4M_OK) {
        n4m_pp_snv_destroy(h); n4m_throw(s, "matrix_view (out)");
    }
    s = n4m_pp_snv_transform(h, Xv, Yv);
    n4m_pp_snv_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_snv_transform");

    return rowmajor_to_R_matrix(out_buf, rows, cols);
}

static SEXP r_n4m_pp_lsnv_transform(SEXP x, SEXP window, SEXP pad_mode,
                                    SEXP constant_value) {
    int64_t rows, cols;
    double *in, *out_buf;
    n4m_matrix_view_t Xv, Yv;
    prepare_same_shape_matrix(x, &rows, &cols, &in, &out_buf, &Xv, &Yv,
                              "n4m_pp_lsnv_matrix");

    n4m_pp_lsnv_handle_t *h = NULL;
    n4m_status_t s = n4m_pp_lsnv_create(&h, INTEGER(window)[0],
                                        INTEGER(pad_mode)[0],
                                        REAL(constant_value)[0]);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_lsnv_create");

    s = n4m_pp_lsnv_transform(h, Xv, Yv);
    n4m_pp_lsnv_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_lsnv_transform");
    return rowmajor_to_R_matrix(out_buf, rows, cols);
}

static SEXP r_n4m_pp_rnv_transform(SEXP x, SEXP with_center,
                                   SEXP with_scale, SEXP k) {
    int64_t rows, cols;
    double *in, *out_buf;
    n4m_matrix_view_t Xv, Yv;
    prepare_same_shape_matrix(x, &rows, &cols, &in, &out_buf, &Xv, &Yv,
                              "n4m_pp_rnv_matrix");

    n4m_pp_rnv_handle_t *h = NULL;
    n4m_status_t s = n4m_pp_rnv_create(&h, LOGICAL(with_center)[0],
                                       LOGICAL(with_scale)[0], REAL(k)[0]);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_rnv_create");

    s = n4m_pp_rnv_transform(h, Xv, Yv);
    n4m_pp_rnv_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_rnv_transform");
    return rowmajor_to_R_matrix(out_buf, rows, cols);
}

static SEXP r_n4m_pp_area_transform(SEXP x, SEXP method) {
    int64_t rows, cols;
    double *in, *out_buf;
    n4m_matrix_view_t Xv, Yv;
    prepare_same_shape_matrix(x, &rows, &cols, &in, &out_buf, &Xv, &Yv,
                              "n4m_pp_area_matrix");

    n4m_pp_area_handle_t *h = NULL;
    n4m_status_t s = n4m_pp_area_create(&h, INTEGER(method)[0]);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_area_create");

    s = n4m_pp_area_transform(h, Xv, Yv);
    n4m_pp_area_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_area_transform");
    return rowmajor_to_R_matrix(out_buf, rows, cols);
}

static SEXP r_n4m_pp_normalize_transform(SEXP x, SEXP feature_min,
                                         SEXP feature_max) {
    int64_t rows, cols;
    double *in, *out_buf;
    n4m_matrix_view_t Xv, Yv;
    prepare_same_shape_matrix(x, &rows, &cols, &in, &out_buf, &Xv, &Yv,
                              "n4m_pp_normalize_matrix");

    n4m_pp_normalize_handle_t *h = NULL;
    n4m_status_t s = n4m_pp_normalize_create(&h, REAL(feature_min)[0],
                                             REAL(feature_max)[0]);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_normalize_create");

    s = n4m_pp_normalize_transform(h, Xv, Yv);
    n4m_pp_normalize_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_normalize_transform");
    return rowmajor_to_R_matrix(out_buf, rows, cols);
}

static SEXP r_n4m_pp_simple_scale_transform(SEXP x) {
    int64_t rows, cols;
    double *in, *out_buf;
    n4m_matrix_view_t Xv, Yv;
    prepare_same_shape_matrix(x, &rows, &cols, &in, &out_buf, &Xv, &Yv,
                              "n4m_pp_simple_scale_matrix");

    n4m_pp_simple_scale_handle_t *h = NULL;
    n4m_status_t s = n4m_pp_simple_scale_create(&h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_simple_scale_create");

    s = n4m_pp_simple_scale_transform(h, Xv, Yv);
    n4m_pp_simple_scale_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_simple_scale_transform");
    return rowmajor_to_R_matrix(out_buf, rows, cols);
}

static SEXP r_n4m_pp_msc_fit_transform(SEXP x_fit, SEXP x) {
    int64_t fit_rows, fit_cols, rows, cols;
    double *fit_in = r_matrix_to_rowmajor(x_fit, &fit_rows, &fit_cols);
    double *in     = r_matrix_to_rowmajor(x, &rows, &cols);
    double *out_buf = (double *)R_alloc((size_t)(rows * cols), sizeof(double));

    n4m_pp_msc_handle_t *h = NULL;
    n4m_status_t s = n4m_pp_msc_create(&h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_msc_create");

    n4m_matrix_view_t Xfit_v, Xv, Yv;
    init_view_rowmajor(&Xfit_v, fit_in, fit_rows, fit_cols);
    init_view_rowmajor(&Xv,     in,     rows,     cols);
    init_view_rowmajor(&Yv,     out_buf, rows,    cols);

    s = n4m_pp_msc_fit(h, Xfit_v);
    if (s != N4M_OK) { n4m_pp_msc_destroy(h); n4m_throw(s, "n4m_pp_msc_fit"); }
    s = n4m_pp_msc_transform(h, Xv, Yv);
    n4m_pp_msc_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_msc_transform");

    return rowmajor_to_R_matrix(out_buf, rows, cols);
}

static SEXP r_n4m_pp_msc_fit_inverse_transform(SEXP x_fit, SEXP x) {
    int64_t fit_rows, fit_cols, rows, cols;
    double *fit_in = r_matrix_to_rowmajor(x_fit, &fit_rows, &fit_cols);
    double *in     = r_matrix_to_rowmajor(x, &rows, &cols);
    double *out_buf = (double *)R_alloc((size_t)(rows * cols), sizeof(double));

    n4m_pp_msc_handle_t *h = NULL;
    n4m_status_t s = n4m_pp_msc_create(&h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_msc_create");

    n4m_matrix_view_t Xfit_v, Xv, Yv;
    init_view_rowmajor(&Xfit_v, fit_in, fit_rows, fit_cols);
    init_view_rowmajor(&Xv,     in,     rows,     cols);
    init_view_rowmajor(&Yv,     out_buf, rows,    cols);

    s = n4m_pp_msc_fit(h, Xfit_v);
    if (s != N4M_OK) { n4m_pp_msc_destroy(h); n4m_throw(s, "n4m_pp_msc_fit"); }
    s = n4m_pp_msc_inverse_transform(h, Xv, Yv);
    n4m_pp_msc_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_msc_inverse_transform");

    return rowmajor_to_R_matrix(out_buf, rows, cols);
}

static SEXP r_n4m_pp_emsc_fit_transform(SEXP x_fit, SEXP x, SEXP degree) {
    int64_t fit_rows, fit_cols, rows, cols;
    double *fit_in = r_matrix_to_rowmajor(x_fit, &fit_rows, &fit_cols);
    double *in     = r_matrix_to_rowmajor(x, &rows, &cols);
    double *out_buf = (double *)R_alloc((size_t)(rows * cols), sizeof(double));

    n4m_pp_emsc_handle_t *h = NULL;
    n4m_status_t s = n4m_pp_emsc_create(&h, INTEGER(degree)[0]);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_emsc_create");

    n4m_matrix_view_t Xfit_v, Xv, Yv;
    init_view_rowmajor(&Xfit_v, fit_in, fit_rows, fit_cols);
    init_view_rowmajor(&Xv,     in,     rows,     cols);
    init_view_rowmajor(&Yv,     out_buf, rows,    cols);

    s = n4m_pp_emsc_fit(h, Xfit_v);
    if (s != N4M_OK) { n4m_pp_emsc_destroy(h); n4m_throw(s, "n4m_pp_emsc_fit"); }
    s = n4m_pp_emsc_transform(h, Xv, Yv);
    n4m_pp_emsc_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_emsc_transform");

    return rowmajor_to_R_matrix(out_buf, rows, cols);
}

static SEXP r_n4m_pp_baseline_fit_transform(SEXP x_fit, SEXP x) {
    int64_t fit_rows, fit_cols, rows, cols;
    double *fit_in = r_matrix_to_rowmajor(x_fit, &fit_rows, &fit_cols);
    double *in     = r_matrix_to_rowmajor(x, &rows, &cols);
    double *out_buf = (double *)R_alloc((size_t)(rows * cols), sizeof(double));

    n4m_pp_baseline_handle_t *h = NULL;
    n4m_status_t s = n4m_pp_baseline_create(&h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_baseline_create");

    n4m_matrix_view_t Xfit_v, Xv, Yv;
    init_view_rowmajor(&Xfit_v, fit_in, fit_rows, fit_cols);
    init_view_rowmajor(&Xv,     in,     rows,     cols);
    init_view_rowmajor(&Yv,     out_buf, rows,    cols);

    s = n4m_pp_baseline_fit(h, Xfit_v);
    if (s != N4M_OK) {
        n4m_pp_baseline_destroy(h);
        n4m_throw(s, "n4m_pp_baseline_fit");
    }
    s = n4m_pp_baseline_transform(h, Xv, Yv);
    n4m_pp_baseline_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_baseline_transform");

    return rowmajor_to_R_matrix(out_buf, rows, cols);
}

static SEXP r_n4m_pp_baseline_fit_inverse_transform(SEXP x_fit, SEXP x) {
    int64_t fit_rows, fit_cols, rows, cols;
    double *fit_in = r_matrix_to_rowmajor(x_fit, &fit_rows, &fit_cols);
    double *in     = r_matrix_to_rowmajor(x, &rows, &cols);
    double *out_buf = (double *)R_alloc((size_t)(rows * cols), sizeof(double));

    n4m_pp_baseline_handle_t *h = NULL;
    n4m_status_t s = n4m_pp_baseline_create(&h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_baseline_create");

    n4m_matrix_view_t Xfit_v, Xv, Yv;
    init_view_rowmajor(&Xfit_v, fit_in, fit_rows, fit_cols);
    init_view_rowmajor(&Xv,     in,     rows,     cols);
    init_view_rowmajor(&Yv,     out_buf, rows,    cols);

    s = n4m_pp_baseline_fit(h, Xfit_v);
    if (s != N4M_OK) {
        n4m_pp_baseline_destroy(h);
        n4m_throw(s, "n4m_pp_baseline_fit");
    }
    s = n4m_pp_baseline_inverse_transform(h, Xv, Yv);
    n4m_pp_baseline_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_baseline_inverse_transform");

    return rowmajor_to_R_matrix(out_buf, rows, cols);
}

static SEXP r_n4m_pp_savgol_transform(SEXP x, SEXP window_length,
                                      SEXP polyorder, SEXP deriv, SEXP delta,
                                      SEXP mode, SEXP cval) {
    int64_t rows, cols;
    double *in = r_matrix_to_rowmajor(x, &rows, &cols);
    double *out_buf = (double *)R_alloc((size_t)(rows * cols), sizeof(double));

    n4m_pp_savgol_handle_t *h = NULL;
    n4m_status_t s = n4m_pp_savgol_create(&h,
                                          INTEGER(window_length)[0],
                                          INTEGER(polyorder)[0],
                                          INTEGER(deriv)[0],
                                          REAL(delta)[0],
                                          (n4m_pp_savgol_mode_t)INTEGER(mode)[0],
                                          REAL(cval)[0]);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_savgol_create");

    n4m_matrix_view_t Xv, Yv;
    init_view_rowmajor(&Xv, in, rows, cols);
    init_view_rowmajor(&Yv, out_buf, rows, cols);

    s = n4m_pp_savgol_transform(h, Xv, Yv);
    n4m_pp_savgol_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_savgol_transform");

    return rowmajor_to_R_matrix(out_buf, rows, cols);
}

static SEXP r_n4m_pp_first_derivative_transform(SEXP x, SEXP delta,
                                                SEXP edge_order) {
    int64_t rows, cols;
    double *in = r_matrix_to_rowmajor(x, &rows, &cols);
    double *out_buf = (double *)R_alloc((size_t)(rows * cols), sizeof(double));

    n4m_pp_first_derivative_handle_t *h = NULL;
    n4m_status_t s = n4m_pp_first_derivative_create(&h, REAL(delta)[0],
                                                    INTEGER(edge_order)[0]);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_first_derivative_create");

    n4m_matrix_view_t Xv, Yv;
    init_view_rowmajor(&Xv, in, rows, cols);
    init_view_rowmajor(&Yv, out_buf, rows, cols);

    s = n4m_pp_first_derivative_transform(h, Xv, Yv);
    n4m_pp_first_derivative_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_first_derivative_transform");

    return rowmajor_to_R_matrix(out_buf, rows, cols);
}

static SEXP r_n4m_pp_second_derivative_transform(SEXP x, SEXP delta,
                                                 SEXP edge_order) {
    int64_t rows, cols;
    double *in = r_matrix_to_rowmajor(x, &rows, &cols);
    double *out_buf = (double *)R_alloc((size_t)(rows * cols), sizeof(double));

    n4m_pp_second_derivative_handle_t *h = NULL;
    n4m_status_t s = n4m_pp_second_derivative_create(&h, REAL(delta)[0],
                                                     INTEGER(edge_order)[0]);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_second_derivative_create");

    n4m_matrix_view_t Xv, Yv;
    init_view_rowmajor(&Xv, in, rows, cols);
    init_view_rowmajor(&Yv, out_buf, rows, cols);

    s = n4m_pp_second_derivative_transform(h, Xv, Yv);
    n4m_pp_second_derivative_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_second_derivative_transform");

    return rowmajor_to_R_matrix(out_buf, rows, cols);
}

static SEXP r_n4m_pp_norris_williams_transform(SEXP x, SEXP segment,
                                               SEXP gap,
                                               SEXP derivative_order,
                                               SEXP delta) {
    int64_t rows, cols;
    double *in, *out_buf;
    n4m_matrix_view_t Xv, Yv;
    prepare_same_shape_matrix(x, &rows, &cols, &in, &out_buf, &Xv, &Yv,
                              "n4m_pp_norris_williams_matrix");

    n4m_pp_norris_williams_handle_t *h = NULL;
    n4m_status_t s = n4m_pp_norris_williams_create(
        &h, INTEGER(segment)[0], INTEGER(gap)[0],
        INTEGER(derivative_order)[0], REAL(delta)[0]);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_norris_williams_create");

    s = n4m_pp_norris_williams_transform(h, Xv, Yv);
    n4m_pp_norris_williams_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_norris_williams_transform");
    return rowmajor_to_R_matrix(out_buf, rows, cols);
}

static SEXP r_n4m_pp_gaussian_transform(SEXP x, SEXP sigma, SEXP order,
                                        SEXP mode, SEXP cval,
                                        SEXP truncate) {
    int64_t rows, cols;
    double *in, *out_buf;
    n4m_matrix_view_t Xv, Yv;
    prepare_same_shape_matrix(x, &rows, &cols, &in, &out_buf, &Xv, &Yv,
                              "n4m_pp_gaussian_matrix");

    n4m_pp_gaussian_handle_t *h = NULL;
    n4m_status_t s = n4m_pp_gaussian_create(
        &h, REAL(sigma)[0], INTEGER(order)[0],
        (n4m_pp_gaussian_mode_t)INTEGER(mode)[0],
        REAL(cval)[0], REAL(truncate)[0]);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_gaussian_create");

    s = n4m_pp_gaussian_transform(h, Xv, Yv);
    n4m_pp_gaussian_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_gaussian_transform");
    return rowmajor_to_R_matrix(out_buf, rows, cols);
}

static SEXP r_n4m_pp_to_absorbance_transform(SEXP x, SEXP is_percent,
                                             SEXP epsilon, SEXP clip_negative) {
    int64_t rows, cols;
    double *in = r_matrix_to_rowmajor(x, &rows, &cols);
    double *out_buf = (double *)R_alloc((size_t)(rows * cols), sizeof(double));

    n4m_pp_to_absorbance_handle_t *h = NULL;
    n4m_status_t s = n4m_pp_to_absorbance_create(&h,
                                                 LOGICAL(is_percent)[0],
                                                 REAL(epsilon)[0],
                                                 LOGICAL(clip_negative)[0]);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_to_absorbance_create");

    n4m_matrix_view_t Xv, Yv;
    init_view_rowmajor(&Xv, in, rows, cols);
    init_view_rowmajor(&Yv, out_buf, rows, cols);

    s = n4m_pp_to_absorbance_transform(h, Xv, Yv);
    n4m_pp_to_absorbance_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_to_absorbance_transform");

    return rowmajor_to_R_matrix(out_buf, rows, cols);
}

static SEXP r_n4m_pp_from_absorbance_transform(SEXP x, SEXP is_percent) {
    int64_t rows, cols;
    double *in, *out_buf;
    n4m_matrix_view_t Xv, Yv;
    prepare_same_shape_matrix(x, &rows, &cols, &in, &out_buf, &Xv, &Yv,
                              "n4m_pp_from_absorbance_matrix");

    n4m_pp_from_absorbance_handle_t *h = NULL;
    n4m_status_t s = n4m_pp_from_absorbance_create(&h,
                                                   LOGICAL(is_percent)[0]);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_from_absorbance_create");

    s = n4m_pp_from_absorbance_transform(h, Xv, Yv);
    n4m_pp_from_absorbance_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_from_absorbance_transform");
    return rowmajor_to_R_matrix(out_buf, rows, cols);
}

static SEXP r_n4m_pp_pct_to_frac_transform(SEXP x) {
    int64_t rows, cols;
    double *in, *out_buf;
    n4m_matrix_view_t Xv, Yv;
    prepare_same_shape_matrix(x, &rows, &cols, &in, &out_buf, &Xv, &Yv,
                              "n4m_pp_pct_to_frac_matrix");

    n4m_pp_pct_to_frac_handle_t *h = NULL;
    n4m_status_t s = n4m_pp_pct_to_frac_create(&h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_pct_to_frac_create");

    s = n4m_pp_pct_to_frac_transform(h, Xv, Yv);
    n4m_pp_pct_to_frac_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_pct_to_frac_transform");
    return rowmajor_to_R_matrix(out_buf, rows, cols);
}

static SEXP r_n4m_pp_frac_to_pct_transform(SEXP x) {
    int64_t rows, cols;
    double *in, *out_buf;
    n4m_matrix_view_t Xv, Yv;
    prepare_same_shape_matrix(x, &rows, &cols, &in, &out_buf, &Xv, &Yv,
                              "n4m_pp_frac_to_pct_matrix");

    n4m_pp_frac_to_pct_handle_t *h = NULL;
    n4m_status_t s = n4m_pp_frac_to_pct_create(&h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_frac_to_pct_create");

    s = n4m_pp_frac_to_pct_transform(h, Xv, Yv);
    n4m_pp_frac_to_pct_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_frac_to_pct_transform");
    return rowmajor_to_R_matrix(out_buf, rows, cols);
}

static SEXP r_n4m_pp_kubelka_munk_transform(SEXP x, SEXP is_percent,
                                            SEXP epsilon) {
    int64_t rows, cols;
    double *in = r_matrix_to_rowmajor(x, &rows, &cols);
    double *out_buf = (double *)R_alloc((size_t)(rows * cols), sizeof(double));

    n4m_pp_kubelka_munk_handle_t *h = NULL;
    n4m_status_t s = n4m_pp_kubelka_munk_create(&h, LOGICAL(is_percent)[0],
                                                REAL(epsilon)[0]);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_kubelka_munk_create");

    n4m_matrix_view_t Xv, Yv;
    init_view_rowmajor(&Xv, in, rows, cols);
    init_view_rowmajor(&Yv, out_buf, rows, cols);

    s = n4m_pp_kubelka_munk_transform(h, Xv, Yv);
    n4m_pp_kubelka_munk_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_kubelka_munk_transform");

    return rowmajor_to_R_matrix(out_buf, rows, cols);
}

static SEXP r_n4m_pp_detrend_transform(SEXP x, SEXP polyorder) {
    int64_t rows, cols;
    double *in = r_matrix_to_rowmajor(x, &rows, &cols);
    double *out_buf = (double *)R_alloc((size_t)(rows * cols), sizeof(double));

    n4m_pp_detrend_handle_t *h = NULL;
    n4m_status_t s = n4m_pp_detrend_create(&h, INTEGER(polyorder)[0]);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_detrend_create");

    n4m_matrix_view_t Xv, Yv;
    init_view_rowmajor(&Xv, in, rows, cols);
    init_view_rowmajor(&Yv, out_buf, rows, cols);

    s = n4m_pp_detrend_transform(h, Xv, Yv);
    n4m_pp_detrend_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_detrend_transform");

    return rowmajor_to_R_matrix(out_buf, rows, cols);
}

static SEXP r_n4m_pp_asls_transform(SEXP x, SEXP lam, SEXP p, SEXP max_iter,
                                    SEXP tol) {
    int64_t rows, cols;
    double *in = r_matrix_to_rowmajor(x, &rows, &cols);
    double *out_buf = (double *)R_alloc((size_t)(rows * cols), sizeof(double));

    n4m_pp_asls_handle_t *h = NULL;
    n4m_status_t s = n4m_pp_asls_create(&h, REAL(lam)[0], REAL(p)[0],
                                        INTEGER(max_iter)[0], REAL(tol)[0]);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_asls_create");

    n4m_matrix_view_t Xv, Yv;
    init_view_rowmajor(&Xv, in, rows, cols);
    init_view_rowmajor(&Yv, out_buf, rows, cols);

    s = n4m_pp_asls_transform(h, Xv, Yv);
    n4m_pp_asls_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_asls_transform");

    return rowmajor_to_R_matrix(out_buf, rows, cols);
}

static SEXP r_n4m_pp_airpls_transform(SEXP x, SEXP lam, SEXP max_iter,
                                      SEXP tol) {
    int64_t rows, cols;
    double *in = r_matrix_to_rowmajor(x, &rows, &cols);
    double *out_buf = (double *)R_alloc((size_t)(rows * cols), sizeof(double));

    n4m_pp_airpls_handle_t *h = NULL;
    n4m_status_t s = n4m_pp_airpls_create(&h, REAL(lam)[0],
                                          INTEGER(max_iter)[0], REAL(tol)[0]);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_airpls_create");

    n4m_matrix_view_t Xv, Yv;
    init_view_rowmajor(&Xv, in, rows, cols);
    init_view_rowmajor(&Yv, out_buf, rows, cols);

    s = n4m_pp_airpls_transform(h, Xv, Yv);
    n4m_pp_airpls_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_airpls_transform");

    return rowmajor_to_R_matrix(out_buf, rows, cols);
}

static SEXP r_n4m_pp_arpls_transform(SEXP x, SEXP lam, SEXP max_iter,
                                     SEXP tol) {
    int64_t rows, cols;
    double *in, *out_buf;
    n4m_matrix_view_t Xv, Yv;
    prepare_same_shape_matrix(x, &rows, &cols, &in, &out_buf, &Xv, &Yv,
                              "n4m_pp_arpls_matrix");

    n4m_pp_arpls_handle_t *h = NULL;
    n4m_status_t s = n4m_pp_arpls_create(&h, REAL(lam)[0],
                                         INTEGER(max_iter)[0], REAL(tol)[0]);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_arpls_create");

    s = n4m_pp_arpls_transform(h, Xv, Yv);
    n4m_pp_arpls_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_arpls_transform");
    return rowmajor_to_R_matrix(out_buf, rows, cols);
}

static SEXP r_n4m_pp_modpoly_transform(SEXP x, SEXP polyorder,
                                       SEXP max_iter, SEXP tol) {
    int64_t rows, cols;
    double *in, *out_buf;
    n4m_matrix_view_t Xv, Yv;
    prepare_same_shape_matrix(x, &rows, &cols, &in, &out_buf, &Xv, &Yv,
                              "n4m_pp_modpoly_matrix");

    n4m_pp_modpoly_handle_t *h = NULL;
    n4m_status_t s = n4m_pp_modpoly_create(&h, INTEGER(polyorder)[0],
                                           INTEGER(max_iter)[0], REAL(tol)[0]);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_modpoly_create");

    s = n4m_pp_modpoly_transform(h, Xv, Yv);
    n4m_pp_modpoly_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_modpoly_transform");
    return rowmajor_to_R_matrix(out_buf, rows, cols);
}

static SEXP r_n4m_pp_imodpoly_transform(SEXP x, SEXP polyorder,
                                        SEXP max_iter, SEXP tol) {
    int64_t rows, cols;
    double *in, *out_buf;
    n4m_matrix_view_t Xv, Yv;
    prepare_same_shape_matrix(x, &rows, &cols, &in, &out_buf, &Xv, &Yv,
                              "n4m_pp_imodpoly_matrix");

    n4m_pp_imodpoly_handle_t *h = NULL;
    n4m_status_t s = n4m_pp_imodpoly_create(&h, INTEGER(polyorder)[0],
                                            INTEGER(max_iter)[0], REAL(tol)[0]);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_imodpoly_create");

    s = n4m_pp_imodpoly_transform(h, Xv, Yv);
    n4m_pp_imodpoly_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_imodpoly_transform");
    return rowmajor_to_R_matrix(out_buf, rows, cols);
}

static SEXP r_n4m_pp_snip_transform(SEXP x, SEXP max_half_window) {
    int64_t rows, cols;
    double *in, *out_buf;
    n4m_matrix_view_t Xv, Yv;
    prepare_same_shape_matrix(x, &rows, &cols, &in, &out_buf, &Xv, &Yv,
                              "n4m_pp_snip_matrix");

    n4m_pp_snip_handle_t *h = NULL;
    n4m_status_t s = n4m_pp_snip_create(&h, INTEGER(max_half_window)[0]);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_snip_create");

    s = n4m_pp_snip_transform(h, Xv, Yv);
    n4m_pp_snip_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_snip_transform");
    return rowmajor_to_R_matrix(out_buf, rows, cols);
}

static SEXP r_n4m_pp_rolling_ball_transform(SEXP x, SEXP half_window,
                                            SEXP smooth_half_window) {
    int64_t rows, cols;
    double *in, *out_buf;
    n4m_matrix_view_t Xv, Yv;
    prepare_same_shape_matrix(x, &rows, &cols, &in, &out_buf, &Xv, &Yv,
                              "n4m_pp_rolling_ball_matrix");

    n4m_pp_rolling_ball_handle_t *h = NULL;
    n4m_status_t s = n4m_pp_rolling_ball_create(
        &h, INTEGER(half_window)[0], INTEGER(smooth_half_window)[0]);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_rolling_ball_create");

    s = n4m_pp_rolling_ball_transform(h, Xv, Yv);
    n4m_pp_rolling_ball_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_rolling_ball_transform");
    return rowmajor_to_R_matrix(out_buf, rows, cols);
}

static SEXP r_n4m_pp_iasls_transform(SEXP x, SEXP lam, SEXP p,
                                     SEXP lam_1, SEXP polyorder,
                                     SEXP diff_order, SEXP max_iter,
                                     SEXP tol) {
    int64_t rows, cols;
    double *in, *out_buf;
    n4m_matrix_view_t Xv, Yv;
    prepare_same_shape_matrix(x, &rows, &cols, &in, &out_buf, &Xv, &Yv,
                              "n4m_pp_iasls_matrix");

    n4m_pp_iasls_handle_t *h = NULL;
    n4m_status_t s = n4m_pp_iasls_create_ex(
        &h, REAL(lam)[0], REAL(p)[0], REAL(lam_1)[0],
        INTEGER(polyorder)[0], INTEGER(diff_order)[0],
        INTEGER(max_iter)[0], REAL(tol)[0]);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_iasls_create_ex");

    s = n4m_pp_iasls_transform(h, Xv, Yv);
    n4m_pp_iasls_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_iasls_transform");
    return rowmajor_to_R_matrix(out_buf, rows, cols);
}

static SEXP r_n4m_pp_beads_transform(SEXP x, SEXP lam_0, SEXP lam_1,
                                     SEXP lam_2, SEXP max_iter, SEXP tol) {
    int64_t rows, cols;
    double *in, *out_buf;
    n4m_matrix_view_t Xv, Yv;
    prepare_same_shape_matrix(x, &rows, &cols, &in, &out_buf, &Xv, &Yv,
                              "n4m_pp_beads_matrix");

    n4m_pp_beads_handle_t *h = NULL;
    n4m_status_t s = n4m_pp_beads_create(
        &h, REAL(lam_0)[0], REAL(lam_1)[0], REAL(lam_2)[0],
        INTEGER(max_iter)[0], REAL(tol)[0]);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_beads_create");

    s = n4m_pp_beads_transform(h, Xv, Yv);
    n4m_pp_beads_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_beads_transform");
    return rowmajor_to_R_matrix(out_buf, rows, cols);
}

static SEXP r_n4m_pp_wavelet_denoise_transform(SEXP x, SEXP family,
                                               SEXP boundary, SEXP level,
                                               SEXP threshold_mode,
                                               SEXP noise_estimator) {
    int64_t rows, cols;
    double *in = r_matrix_to_rowmajor(x, &rows, &cols);
    double *out_buf = (double *)R_alloc((size_t)(rows * cols), sizeof(double));

    n4m_pp_wavelet_denoise_handle_t *h = NULL;
    n4m_status_t s = n4m_pp_wavelet_denoise_create(
        &h,
        (n4m_pp_wavelet_family_t)INTEGER(family)[0],
        (n4m_pp_wavelet_boundary_t)INTEGER(boundary)[0],
        INTEGER(level)[0],
        (n4m_pp_wavelet_threshold_t)INTEGER(threshold_mode)[0],
        (n4m_pp_wavelet_noise_t)INTEGER(noise_estimator)[0]);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_wavelet_denoise_create");

    n4m_matrix_view_t Xv, Yv;
    init_view_rowmajor(&Xv, in, rows, cols);
    init_view_rowmajor(&Yv, out_buf, rows, cols);

    s = n4m_pp_wavelet_denoise_transform(h, Xv, Yv);
    n4m_pp_wavelet_denoise_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_wavelet_denoise_transform");

    return rowmajor_to_R_matrix(out_buf, rows, cols);
}

static SEXP r_n4m_pp_wavelet_transform(SEXP x, SEXP family, SEXP boundary) {
    int64_t rows, cols;
    double *in = r_matrix_to_rowmajor(x, &rows, &cols);

    n4m_pp_wavelet_handle_t *h = NULL;
    n4m_status_t s = n4m_pp_wavelet_create(
        &h,
        (n4m_pp_wavelet_family_t)INTEGER(family)[0],
        (n4m_pp_wavelet_boundary_t)INTEGER(boundary)[0]);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_wavelet_create");

    int64_t out_cols = 0;
    s = n4m_pp_wavelet_output_cols(h, cols, &out_cols);
    if (s != N4M_OK) {
        n4m_pp_wavelet_destroy(h);
        n4m_throw(s, "n4m_pp_wavelet_output_cols");
    }

    double *out_buf = (double *)R_alloc((size_t)(rows * out_cols), sizeof(double));
    n4m_matrix_view_t Xv, Yv;
    if ((s = init_view_rowmajor(&Xv, in, rows, cols)) != N4M_OK) {
        n4m_pp_wavelet_destroy(h); n4m_throw(s, "matrix_view (in)");
    }
    if ((s = init_view_rowmajor(&Yv, out_buf, rows, out_cols)) != N4M_OK) {
        n4m_pp_wavelet_destroy(h); n4m_throw(s, "matrix_view (out)");
    }

    s = n4m_pp_wavelet_transform(h, Xv, Yv);
    n4m_pp_wavelet_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_wavelet_transform");
    return rowmajor_to_R_matrix(out_buf, rows, out_cols);
}

static SEXP r_n4m_pp_haar_transform(SEXP x) {
    int64_t rows, cols;
    double *in = r_matrix_to_rowmajor(x, &rows, &cols);

    n4m_pp_haar_handle_t *h = NULL;
    n4m_status_t s = n4m_pp_haar_create(&h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_haar_create");

    int64_t out_cols = 0;
    s = n4m_pp_haar_output_cols(cols, &out_cols);
    if (s != N4M_OK) {
        n4m_pp_haar_destroy(h);
        n4m_throw(s, "n4m_pp_haar_output_cols");
    }

    double *out_buf = (double *)R_alloc((size_t)(rows * out_cols), sizeof(double));
    n4m_matrix_view_t Xv, Yv;
    if ((s = init_view_rowmajor(&Xv, in, rows, cols)) != N4M_OK) {
        n4m_pp_haar_destroy(h); n4m_throw(s, "matrix_view (in)");
    }
    if ((s = init_view_rowmajor(&Yv, out_buf, rows, out_cols)) != N4M_OK) {
        n4m_pp_haar_destroy(h); n4m_throw(s, "matrix_view (out)");
    }

    s = n4m_pp_haar_transform(h, Xv, Yv);
    n4m_pp_haar_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_haar_transform");
    return rowmajor_to_R_matrix(out_buf, rows, out_cols);
}

static SEXP r_n4m_pp_wavelet_features_transform(SEXP x, SEXP family,
                                                SEXP boundary,
                                                SEXP max_level,
                                                SEXP entropy) {
    int64_t rows, cols;
    double *in = r_matrix_to_rowmajor(x, &rows, &cols);

    n4m_pp_wavelet_features_handle_t *h = NULL;
    n4m_status_t s = n4m_pp_wavelet_features_create_ex(
        &h,
        (n4m_pp_wavelet_family_t)INTEGER(family)[0],
        (n4m_pp_wavelet_boundary_t)INTEGER(boundary)[0],
        INTEGER(max_level)[0],
        (n4m_pp_wavelet_features_entropy_t)INTEGER(entropy)[0]);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_wavelet_features_create_ex");

    int64_t out_cols = 0;
    s = n4m_pp_wavelet_features_output_cols(h, cols, &out_cols);
    if (s != N4M_OK) {
        n4m_pp_wavelet_features_destroy(h);
        n4m_throw(s, "n4m_pp_wavelet_features_output_cols");
    }

    double *out_buf = (double *)R_alloc((size_t)(rows * out_cols), sizeof(double));
    n4m_matrix_view_t Xv, Yv;
    if ((s = init_view_rowmajor(&Xv, in, rows, cols)) != N4M_OK) {
        n4m_pp_wavelet_features_destroy(h); n4m_throw(s, "matrix_view (in)");
    }
    if ((s = init_view_rowmajor(&Yv, out_buf, rows, out_cols)) != N4M_OK) {
        n4m_pp_wavelet_features_destroy(h); n4m_throw(s, "matrix_view (out)");
    }

    s = n4m_pp_wavelet_features_transform(h, Xv, Yv);
    n4m_pp_wavelet_features_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_wavelet_features_transform");
    return rowmajor_to_R_matrix(out_buf, rows, out_cols);
}

static SEXP r_n4m_pp_osc_fit_transform(SEXP x, SEXP y, SEXP n_components,
                                       SEXP scale) {
    int64_t rows, cols;
    double *in = r_matrix_to_rowmajor(x, &rows, &cols);
    double *out_buf = (double *)R_alloc((size_t)(rows * cols), sizeof(double));

    n4m_pp_osc_handle_t *h = NULL;
    n4m_status_t s = n4m_pp_osc_create(
        &h, INTEGER(n_components)[0], LOGICAL(scale)[0]);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_osc_create");

    n4m_matrix_view_t Xv, Yv;
    if ((s = init_view_rowmajor(&Xv, in, rows, cols)) != N4M_OK) {
        n4m_pp_osc_destroy(h); n4m_throw(s, "matrix_view (in)");
    }
    if ((s = init_view_rowmajor(&Yv, out_buf, rows, cols)) != N4M_OK) {
        n4m_pp_osc_destroy(h); n4m_throw(s, "matrix_view (out)");
    }
    s = n4m_pp_osc_fit(h, Xv, REAL(y), (int64_t)Rf_length(y));
    if (s != N4M_OK) {
        n4m_pp_osc_destroy(h);
        n4m_throw(s, "n4m_pp_osc_fit");
    }
    s = n4m_pp_osc_transform(h, Xv, Yv);
    n4m_pp_osc_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_osc_transform");
    return rowmajor_to_R_matrix(out_buf, rows, cols);
}

static SEXP r_n4m_pp_epo_fit_transform(SEXP x, SEXP d, SEXP scale) {
    int64_t rows, cols;
    double *in = r_matrix_to_rowmajor(x, &rows, &cols);
    double *out_buf = (double *)R_alloc((size_t)(rows * cols), sizeof(double));

    n4m_pp_epo_handle_t *h = NULL;
    n4m_status_t s = n4m_pp_epo_create(&h, LOGICAL(scale)[0]);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_epo_create");

    n4m_matrix_view_t Xv, Yv;
    if ((s = init_view_rowmajor(&Xv, in, rows, cols)) != N4M_OK) {
        n4m_pp_epo_destroy(h); n4m_throw(s, "matrix_view (in)");
    }
    if ((s = init_view_rowmajor(&Yv, out_buf, rows, cols)) != N4M_OK) {
        n4m_pp_epo_destroy(h); n4m_throw(s, "matrix_view (out)");
    }
    s = n4m_pp_epo_fit(h, Xv, REAL(d), (int64_t)Rf_length(d));
    if (s != N4M_OK) {
        n4m_pp_epo_destroy(h);
        n4m_throw(s, "n4m_pp_epo_fit");
    }
    s = n4m_pp_epo_transform_with_d(h, Xv, REAL(d), (int64_t)Rf_length(d), Yv);
    n4m_pp_epo_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_epo_transform_with_d");
    return rowmajor_to_R_matrix(out_buf, rows, cols);
}

static SEXP r_n4m_pp_epo_fit_inverse_transform(SEXP x, SEXP d, SEXP scale) {
    int64_t rows, cols;
    double *in = r_matrix_to_rowmajor(x, &rows, &cols);
    double *out_buf = (double *)R_alloc((size_t)(rows * cols), sizeof(double));

    n4m_pp_epo_handle_t *h = NULL;
    n4m_status_t s = n4m_pp_epo_create(&h, LOGICAL(scale)[0]);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_epo_create");

    n4m_matrix_view_t Xv, Yv;
    if ((s = init_view_rowmajor(&Xv, in, rows, cols)) != N4M_OK) {
        n4m_pp_epo_destroy(h); n4m_throw(s, "matrix_view (in)");
    }
    if ((s = init_view_rowmajor(&Yv, out_buf, rows, cols)) != N4M_OK) {
        n4m_pp_epo_destroy(h); n4m_throw(s, "matrix_view (out)");
    }
    s = n4m_pp_epo_fit(h, Xv, REAL(d), (int64_t)Rf_length(d));
    if (s != N4M_OK) {
        n4m_pp_epo_destroy(h);
        n4m_throw(s, "n4m_pp_epo_fit");
    }
    s = n4m_pp_epo_inverse_transform(h, Xv, Yv);
    n4m_pp_epo_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_epo_inverse_transform");
    return rowmajor_to_R_matrix(out_buf, rows, cols);
}

static SEXP r_n4m_pp_flex_pca_fit_transform(SEXP x, SEXP n_components) {
    int64_t rows, cols;
    double *in = r_matrix_to_rowmajor(x, &rows, &cols);
    n4m_matrix_view_t Xv;
    n4m_status_t s = init_view_rowmajor(&Xv, in, rows, cols);
    if (s != N4M_OK) n4m_throw(s, "matrix_view (in)");

    n4m_pp_flex_pca_handle_t *h = NULL;
    s = n4m_pp_flex_pca_create(&h, REAL(n_components)[0]);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_flex_pca_create");
    s = n4m_pp_flex_pca_fit(h, Xv);
    if (s != N4M_OK) {
        n4m_pp_flex_pca_destroy(h);
        n4m_throw(s, "n4m_pp_flex_pca_fit");
    }
    int64_t out_cols = 0;
    s = n4m_pp_flex_pca_output_cols(h, &out_cols);
    if (s != N4M_OK) {
        n4m_pp_flex_pca_destroy(h);
        n4m_throw(s, "n4m_pp_flex_pca_output_cols");
    }
    double *out_buf = (double *)R_alloc((size_t)(rows * out_cols), sizeof(double));
    n4m_matrix_view_t Yv;
    if ((s = init_view_rowmajor(&Yv, out_buf, rows, out_cols)) != N4M_OK) {
        n4m_pp_flex_pca_destroy(h); n4m_throw(s, "matrix_view (out)");
    }
    s = n4m_pp_flex_pca_transform(h, Xv, Yv);
    n4m_pp_flex_pca_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_flex_pca_transform");
    return rowmajor_to_R_matrix(out_buf, rows, out_cols);
}

static SEXP r_n4m_pp_flex_svd_fit_transform(SEXP x, SEXP n_components) {
    int64_t rows, cols;
    double *in = r_matrix_to_rowmajor(x, &rows, &cols);
    n4m_matrix_view_t Xv;
    n4m_status_t s = init_view_rowmajor(&Xv, in, rows, cols);
    if (s != N4M_OK) n4m_throw(s, "matrix_view (in)");

    n4m_pp_flex_svd_handle_t *h = NULL;
    s = n4m_pp_flex_svd_create(&h, REAL(n_components)[0]);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_flex_svd_create");
    s = n4m_pp_flex_svd_fit(h, Xv);
    if (s != N4M_OK) {
        n4m_pp_flex_svd_destroy(h);
        n4m_throw(s, "n4m_pp_flex_svd_fit");
    }
    int64_t out_cols = 0;
    s = n4m_pp_flex_svd_output_cols(h, &out_cols);
    if (s != N4M_OK) {
        n4m_pp_flex_svd_destroy(h);
        n4m_throw(s, "n4m_pp_flex_svd_output_cols");
    }
    double *out_buf = (double *)R_alloc((size_t)(rows * out_cols), sizeof(double));
    n4m_matrix_view_t Yv;
    if ((s = init_view_rowmajor(&Yv, out_buf, rows, out_cols)) != N4M_OK) {
        n4m_pp_flex_svd_destroy(h); n4m_throw(s, "matrix_view (out)");
    }
    s = n4m_pp_flex_svd_transform(h, Xv, Yv);
    n4m_pp_flex_svd_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_flex_svd_transform");
    return rowmajor_to_R_matrix(out_buf, rows, out_cols);
}

static SEXP r_n4m_pp_fck_static_transform(SEXP x, SEXP kernel_size,
                                          SEXP filter_orders,
                                          SEXP filter_scales) {
    int64_t rows, cols;
    double *in = r_matrix_to_rowmajor(x, &rows, &cols);
    int32_t n_orders = (int32_t)Rf_length(filter_orders);
    int32_t n_scales = (int32_t)Rf_length(filter_scales);

    n4m_pp_fck_static_handle_t *h = NULL;
    n4m_status_t s = n4m_pp_fck_static_create(
        &h, INTEGER(kernel_size)[0], REAL(filter_orders), n_orders,
        REAL(filter_scales), n_scales);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_fck_static_create");

    int32_t out_cols32 = 0;
    s = n4m_pp_fck_static_output_cols(n_orders * n_scales,
                                      (int32_t)cols, &out_cols32);
    if (s != N4M_OK) {
        n4m_pp_fck_static_destroy(h);
        n4m_throw(s, "n4m_pp_fck_static_output_cols");
    }
    int64_t out_cols = (int64_t)out_cols32;
    double *out_buf = (double *)R_alloc((size_t)(rows * out_cols), sizeof(double));
    n4m_matrix_view_t Xv, Yv;
    if ((s = init_view_rowmajor(&Xv, in, rows, cols)) != N4M_OK) {
        n4m_pp_fck_static_destroy(h); n4m_throw(s, "matrix_view (in)");
    }
    if ((s = init_view_rowmajor(&Yv, out_buf, rows, out_cols)) != N4M_OK) {
        n4m_pp_fck_static_destroy(h); n4m_throw(s, "matrix_view (out)");
    }
    s = n4m_pp_fck_static_transform(h, Xv, Yv);
    n4m_pp_fck_static_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_fck_static_transform");
    return rowmajor_to_R_matrix(out_buf, rows, out_cols);
}

static SEXP r_n4m_pp_crop_transform(SEXP x, SEXP start, SEXP end) {
    int64_t rows, cols;
    double *in = r_matrix_to_rowmajor(x, &rows, &cols);

    n4m_pp_crop_handle_t *h = NULL;
    n4m_status_t s = n4m_pp_crop_create(
        &h, (int64_t)INTEGER(start)[0], (int64_t)INTEGER(end)[0]);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_crop_create");
    int64_t out_cols = n4m_pp_crop_output_cols(h, cols);
    if (out_cols <= 0) {
        n4m_pp_crop_destroy(h);
        Rf_error("nirs4all-methods: n4m_pp_crop_output_cols returned %lld",
                 (long long)out_cols);
    }
    double *out_buf = (double *)R_alloc((size_t)(rows * out_cols), sizeof(double));
    n4m_matrix_view_t Xv, Yv;
    if ((s = init_view_rowmajor(&Xv, in, rows, cols)) != N4M_OK) {
        n4m_pp_crop_destroy(h); n4m_throw(s, "matrix_view (in)");
    }
    if ((s = init_view_rowmajor(&Yv, out_buf, rows, out_cols)) != N4M_OK) {
        n4m_pp_crop_destroy(h); n4m_throw(s, "matrix_view (out)");
    }
    s = n4m_pp_crop_transform(h, Xv, Yv);
    n4m_pp_crop_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_crop_transform");
    return rowmajor_to_R_matrix(out_buf, rows, out_cols);
}

static SEXP r_n4m_pp_resample_transform(SEXP x, SEXP num_samples) {
    int64_t rows, cols;
    double *in = r_matrix_to_rowmajor(x, &rows, &cols);

    n4m_pp_resample_handle_t *h = NULL;
    n4m_status_t s = n4m_pp_resample_create(&h, (int64_t)INTEGER(num_samples)[0]);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_resample_create");
    int64_t out_cols = n4m_pp_resample_output_cols(h, cols);
    if (out_cols <= 0) {
        n4m_pp_resample_destroy(h);
        Rf_error("nirs4all-methods: n4m_pp_resample_output_cols returned %lld",
                 (long long)out_cols);
    }
    double *out_buf = (double *)R_alloc((size_t)(rows * out_cols), sizeof(double));
    n4m_matrix_view_t Xv, Yv;
    if ((s = init_view_rowmajor(&Xv, in, rows, cols)) != N4M_OK) {
        n4m_pp_resample_destroy(h); n4m_throw(s, "matrix_view (in)");
    }
    if ((s = init_view_rowmajor(&Yv, out_buf, rows, out_cols)) != N4M_OK) {
        n4m_pp_resample_destroy(h); n4m_throw(s, "matrix_view (out)");
    }
    s = n4m_pp_resample_transform(h, Xv, Yv);
    n4m_pp_resample_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_resample_transform");
    return rowmajor_to_R_matrix(out_buf, rows, out_cols);
}

static SEXP r_n4m_pp_resampler_fit_transform(SEXP x, SEXP source_wl,
                                             SEXP target_wl, SEXP method,
                                             SEXP crop_min, SEXP crop_max,
                                             SEXP use_crop, SEXP fill_value,
                                             SEXP bounds_error,
                                             SEXP extrapolate) {
    int64_t rows, cols;
    double *in = r_matrix_to_rowmajor(x, &rows, &cols);

    n4m_pp_resampler_handle_t *h = NULL;
    n4m_status_t s = n4m_pp_resampler_create(
        &h, REAL(target_wl), (int64_t)Rf_length(target_wl),
        INTEGER(method)[0], REAL(crop_min)[0], REAL(crop_max)[0],
        LOGICAL(use_crop)[0], REAL(fill_value)[0],
        LOGICAL(bounds_error)[0], LOGICAL(extrapolate)[0]);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_resampler_create");
    s = n4m_pp_resampler_fit(h, REAL(source_wl), (int64_t)Rf_length(source_wl));
    if (s != N4M_OK) {
        n4m_pp_resampler_destroy(h);
        n4m_throw(s, "n4m_pp_resampler_fit");
    }
    int64_t out_cols = n4m_pp_resampler_output_cols(h);
    if (out_cols <= 0) {
        n4m_pp_resampler_destroy(h);
        Rf_error("nirs4all-methods: n4m_pp_resampler_output_cols returned %lld",
                 (long long)out_cols);
    }
    double *out_buf = (double *)R_alloc((size_t)(rows * out_cols), sizeof(double));
    n4m_matrix_view_t Xv, Yv;
    if ((s = init_view_rowmajor(&Xv, in, rows, cols)) != N4M_OK) {
        n4m_pp_resampler_destroy(h); n4m_throw(s, "matrix_view (in)");
    }
    if ((s = init_view_rowmajor(&Yv, out_buf, rows, out_cols)) != N4M_OK) {
        n4m_pp_resampler_destroy(h); n4m_throw(s, "matrix_view (out)");
    }
    s = n4m_pp_resampler_transform(h, Xv, Yv);
    n4m_pp_resampler_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_resampler_transform");
    return rowmajor_to_R_matrix(out_buf, rows, out_cols);
}

static SEXP r_n4m_pp_kbins_disc_fit_transform(SEXP x, SEXP n_bins,
                                              SEXP strategy) {
    int64_t rows, cols;
    double *in = r_matrix_to_rowmajor(x, &rows, &cols);
    int32_t *out_buf = (int32_t *)R_alloc((size_t)(rows * cols), sizeof(int32_t));

    n4m_pp_kbins_disc_handle_t *h = NULL;
    n4m_status_t s = n4m_pp_kbins_disc_create(
        &h, INTEGER(n_bins)[0], INTEGER(strategy)[0]);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_kbins_disc_create");

    n4m_matrix_view_t Xv, Yv;
    if ((s = init_view_rowmajor(&Xv, in, rows, cols)) != N4M_OK) {
        n4m_pp_kbins_disc_destroy(h); n4m_throw(s, "matrix_view (in)");
    }
    s = n4m_matrix_view_init_rowmajor(&Yv, out_buf, rows, cols, N4M_DTYPE_I32);
    if (s != N4M_OK) {
        n4m_pp_kbins_disc_destroy(h); n4m_throw(s, "matrix_view (out)");
    }
    s = n4m_pp_kbins_disc_fit(h, Xv);
    if (s != N4M_OK) {
        n4m_pp_kbins_disc_destroy(h);
        n4m_throw(s, "n4m_pp_kbins_disc_fit");
    }
    s = n4m_pp_kbins_disc_transform(h, Xv, Yv);
    n4m_pp_kbins_disc_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_kbins_disc_transform");
    return rowmajor_i32_to_R_matrix(out_buf, rows, cols);
}

static SEXP r_n4m_pp_range_disc_transform(SEXP x, SEXP edges) {
    int64_t rows, cols;
    double *in = r_matrix_to_rowmajor(x, &rows, &cols);
    int32_t *out_buf = (int32_t *)R_alloc((size_t)(rows * cols), sizeof(int32_t));

    n4m_pp_range_disc_handle_t *h = NULL;
    n4m_status_t s = n4m_pp_range_disc_create(
        &h, REAL(edges), (int64_t)Rf_length(edges));
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_range_disc_create");

    n4m_matrix_view_t Xv, Yv;
    if ((s = init_view_rowmajor(&Xv, in, rows, cols)) != N4M_OK) {
        n4m_pp_range_disc_destroy(h); n4m_throw(s, "matrix_view (in)");
    }
    s = n4m_matrix_view_init_rowmajor(&Yv, out_buf, rows, cols, N4M_DTYPE_I32);
    if (s != N4M_OK) {
        n4m_pp_range_disc_destroy(h); n4m_throw(s, "matrix_view (out)");
    }
    s = n4m_pp_range_disc_transform(h, Xv, Yv);
    n4m_pp_range_disc_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_range_disc_transform");
    return rowmajor_i32_to_R_matrix(out_buf, rows, cols);
}

/* ------------------------------------------------------------------ */
/* Splitter wrappers                                                   */
/* ------------------------------------------------------------------ */

static SEXP build_split_result(const n4m_split_result_t *r) {
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

static SEXP r_n4m_split_kennard_stone_split(SEXP x, SEXP test_size) {
    int64_t rows, cols;
    double *in = r_matrix_to_rowmajor(x, &rows, &cols);

    n4m_split_kennard_stone_handle_t *h = NULL;
    n4m_status_t s = n4m_split_kennard_stone_create(&h, REAL(test_size)[0]);
    if (s != N4M_OK) n4m_throw(s, "n4m_split_kennard_stone_create");

    n4m_matrix_view_t Xv;
    init_view_rowmajor(&Xv, in, rows, cols);

    n4m_split_result_t r = {0};
    s = n4m_split_kennard_stone_split(h, Xv, &r);
    n4m_split_kennard_stone_destroy(h);
    if (s != N4M_OK) {
        n4m_split_result_destroy(&r);
        n4m_throw(s, "n4m_split_kennard_stone_split");
    }
    SEXP out = build_split_result(&r);
    n4m_split_result_destroy(&r);
    return out;
}

static SEXP r_n4m_split_spxy_split(SEXP x, SEXP y, SEXP test_size) {
    int64_t rows, cols, y_rows, y_cols;
    double *in_x = r_matrix_to_rowmajor(x, &rows, &cols);
    double *in_y = r_matrix_to_rowmajor(y, &y_rows, &y_cols);

    n4m_split_spxy_handle_t *h = NULL;
    n4m_status_t s = n4m_split_spxy_create(&h, REAL(test_size)[0]);
    if (s != N4M_OK) n4m_throw(s, "n4m_split_spxy_create");

    n4m_matrix_view_t Xv, Yv;
    init_view_rowmajor(&Xv, in_x, rows, cols);
    init_view_rowmajor(&Yv, in_y, y_rows, y_cols);

    n4m_split_result_t r = {0};
    s = n4m_split_spxy_split(h, Xv, Yv, &r);
    n4m_split_spxy_destroy(h);
    if (s != N4M_OK) {
        n4m_split_result_destroy(&r);
        n4m_throw(s, "n4m_split_spxy_split");
    }
    SEXP out = build_split_result(&r);
    n4m_split_result_destroy(&r);
    return out;
}

static SEXP r_n4m_split_kbins_stratified_split(SEXP y, SEXP test_size,
                                               SEXP seed, SEXP n_bins,
                                               SEXP strategy) {
    int64_t rows, cols;
    double *in_y = r_matrix_to_rowmajor(y, &rows, &cols);

    n4m_split_kbins_stratified_handle_t *h = NULL;
    n4m_status_t s = n4m_split_kbins_stratified_create(
        &h, REAL(test_size)[0], (uint64_t)REAL(seed)[0],
        INTEGER(n_bins)[0], INTEGER(strategy)[0]);
    if (s != N4M_OK) n4m_throw(s, "n4m_split_kbins_stratified_create");

    n4m_matrix_view_t Yv;
    s = init_view_rowmajor(&Yv, in_y, rows, cols);
    if (s != N4M_OK) {
        n4m_split_kbins_stratified_destroy(h);
        n4m_throw(s, "matrix_view (Y)");
    }

    n4m_split_result_t r = {0};
    s = n4m_split_kbins_stratified_split(h, Yv, &r);
    n4m_split_kbins_stratified_destroy(h);
    if (s != N4M_OK) {
        n4m_split_result_destroy(&r);
        n4m_throw(s, "n4m_split_kbins_stratified_split");
    }
    SEXP out = build_split_result(&r);
    n4m_split_result_destroy(&r);
    return out;
}

/* ------------------------------------------------------------------ */
/* Filter wrappers                                                     */
/* ------------------------------------------------------------------ */

static SEXP r_n4m_filter_y_outlier_fit_apply(SEXP y, SEXP method,
                                             SEXP threshold,
                                             SEXP lower_pct,
                                             SEXP upper_pct) {
    R_xlen_t n = Rf_length(y);
    double *yv = REAL(y);

    n4m_filter_y_outlier_handle_t *h = NULL;
    n4m_status_t s = n4m_filter_y_outlier_create(
        &h, (int32_t)INTEGER(method)[0],
        REAL(threshold)[0], REAL(lower_pct)[0], REAL(upper_pct)[0]);
    if (s != N4M_OK) n4m_throw(s, "n4m_filter_y_outlier_create");

    s = n4m_filter_y_outlier_fit(h, yv, (int64_t)n);
    if (s != N4M_OK) {
        n4m_filter_y_outlier_destroy(h);
        n4m_throw(s, "n4m_filter_y_outlier_fit");
    }

    uint8_t *mask = (uint8_t *)R_alloc((size_t)n, sizeof(uint8_t));
    n4m_filter_stats_t stats = {0};
    s = n4m_filter_y_outlier_apply(h, yv, (int64_t)n, mask, &stats);
    n4m_filter_y_outlier_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_filter_y_outlier_apply");

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

static SEXP r_n4m_filter_x_outlier_fit_apply(SEXP x, SEXP method,
                                             SEXP use_threshold,
                                             SEXP threshold,
                                             SEXP n_components,
                                             SEXP contamination,
                                             SEXP seed,
                                             SEXP n_estimators,
                                             SEXP max_samples) {
    int64_t rows, cols;
    double *in = r_matrix_to_rowmajor(x, &rows, &cols);

    n4m_filter_x_outlier_handle_t *h = NULL;
    n4m_status_t s = n4m_filter_x_outlier_create(
        &h, INTEGER(method)[0], LOGICAL(use_threshold)[0],
        REAL(threshold)[0], INTEGER(n_components)[0],
        REAL(contamination)[0], (uint64_t)REAL(seed)[0],
        INTEGER(n_estimators)[0], (int64_t)REAL(max_samples)[0]);
    if (s != N4M_OK) n4m_throw(s, "n4m_filter_x_outlier_create");

    n4m_matrix_view_t Xv;
    if ((s = init_view_rowmajor(&Xv, in, rows, cols)) != N4M_OK) {
        n4m_filter_x_outlier_destroy(h); n4m_throw(s, "matrix_view (in)");
    }
    s = n4m_filter_x_outlier_fit(h, Xv);
    if (s != N4M_OK) {
        n4m_filter_x_outlier_destroy(h);
        n4m_throw(s, "n4m_filter_x_outlier_fit");
    }

    uint8_t *mask = (uint8_t *)R_alloc((size_t)rows, sizeof(uint8_t));
    n4m_filter_stats_t stats = {0};
    s = n4m_filter_x_outlier_apply(h, Xv, mask, &stats);
    n4m_filter_x_outlier_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_filter_x_outlier_apply");

    SEXP mask_R = PROTECT(Rf_allocVector(LGLSXP, (R_xlen_t)rows));
    for (int64_t i = 0; i < rows; ++i) LOGICAL(mask_R)[i] = mask[i] ? TRUE : FALSE;

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
/* Advanced chemometric wrappers                                       */
/* ------------------------------------------------------------------ */

static void ensure_same_cols(int64_t a, int64_t b, const char *lhs,
                             const char *rhs) {
    if (a != b) {
        Rf_error("nirs4all-methods: %s and %s must have the same number of columns",
                 lhs, rhs);
    }
}

static const double *optional_real_vector(SEXP x, int64_t *n,
                                          const char *name) {
    if (x == R_NilValue || Rf_length(x) == 0) {
        *n = 0;
        return NULL;
    }
    if (!Rf_isReal(x)) {
        Rf_error("nirs4all-methods: %s must be a numeric vector", name);
    }
    *n = (int64_t)Rf_length(x);
    return REAL(x);
}

static SEXP r_n4m_pp_direct_standardization_fit_transform(
    SEXP source, SEXP target, SEXP x, SEXP fit_intercept, SEXP ridge) {
    int64_t sr, sc, tr, tc, rows, cols;
    double *source_in = r_matrix_to_rowmajor(source, &sr, &sc);
    double *target_in = r_matrix_to_rowmajor(target, &tr, &tc);
    double *in = r_matrix_to_rowmajor(x, &rows, &cols);
    ensure_same_cols(sc, tc, "source", "target");
    ensure_same_cols(sc, cols, "source", "X");
    double *out_buf = (double *)R_alloc((size_t)(rows * cols), sizeof(double));

    n4m_pp_direct_standardization_handle_t *h = NULL;
    n4m_status_t s = n4m_pp_direct_standardization_create(
        &h, LOGICAL(fit_intercept)[0], REAL(ridge)[0]);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_direct_standardization_create");

    n4m_matrix_view_t Sv, Tv, Xv, Yv;
    init_view_rowmajor(&Sv, source_in, sr, sc);
    init_view_rowmajor(&Tv, target_in, tr, tc);
    init_view_rowmajor(&Xv, in, rows, cols);
    init_view_rowmajor(&Yv, out_buf, rows, cols);
    s = n4m_pp_direct_standardization_fit(h, Sv, Tv);
    if (s != N4M_OK) {
        n4m_pp_direct_standardization_destroy(h);
        n4m_throw(s, "n4m_pp_direct_standardization_fit");
    }
    s = n4m_pp_direct_standardization_transform(h, Xv, Yv);
    n4m_pp_direct_standardization_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_direct_standardization_transform");
    return rowmajor_to_R_matrix(out_buf, rows, cols);
}

static SEXP r_n4m_pp_robust_direct_standardization_fit_transform(
    SEXP source, SEXP target, SEXP x, SEXP fit_intercept, SEXP ridge,
    SEXP trim_quantile, SEXP max_iter) {
    int64_t sr, sc, tr, tc, rows, cols;
    double *source_in = r_matrix_to_rowmajor(source, &sr, &sc);
    double *target_in = r_matrix_to_rowmajor(target, &tr, &tc);
    double *in = r_matrix_to_rowmajor(x, &rows, &cols);
    ensure_same_cols(sc, tc, "source", "target");
    ensure_same_cols(sc, cols, "source", "X");
    double *out_buf = (double *)R_alloc((size_t)(rows * cols), sizeof(double));

    n4m_pp_robust_direct_standardization_handle_t *h = NULL;
    n4m_status_t s = n4m_pp_robust_direct_standardization_create(
        &h, LOGICAL(fit_intercept)[0], REAL(ridge)[0],
        REAL(trim_quantile)[0], INTEGER(max_iter)[0]);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_robust_direct_standardization_create");

    n4m_matrix_view_t Sv, Tv, Xv, Yv;
    init_view_rowmajor(&Sv, source_in, sr, sc);
    init_view_rowmajor(&Tv, target_in, tr, tc);
    init_view_rowmajor(&Xv, in, rows, cols);
    init_view_rowmajor(&Yv, out_buf, rows, cols);
    s = n4m_pp_robust_direct_standardization_fit(h, Sv, Tv);
    if (s != N4M_OK) {
        n4m_pp_robust_direct_standardization_destroy(h);
        n4m_throw(s, "n4m_pp_robust_direct_standardization_fit");
    }
    s = n4m_pp_robust_direct_standardization_transform(h, Xv, Yv);
    n4m_pp_robust_direct_standardization_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_robust_direct_standardization_transform");
    return rowmajor_to_R_matrix(out_buf, rows, cols);
}

static SEXP r_n4m_pp_piecewise_direct_standardization_fit_transform(
    SEXP source, SEXP target, SEXP x, SEXP window_size,
    SEXP fit_intercept, SEXP ridge) {
    int64_t sr, sc, tr, tc, rows, cols;
    double *source_in = r_matrix_to_rowmajor(source, &sr, &sc);
    double *target_in = r_matrix_to_rowmajor(target, &tr, &tc);
    double *in = r_matrix_to_rowmajor(x, &rows, &cols);
    ensure_same_cols(sc, tc, "source", "target");
    ensure_same_cols(sc, cols, "source", "X");
    double *out_buf = (double *)R_alloc((size_t)(rows * cols), sizeof(double));

    n4m_pp_piecewise_direct_standardization_handle_t *h = NULL;
    n4m_status_t s = n4m_pp_piecewise_direct_standardization_create(
        &h, INTEGER(window_size)[0], LOGICAL(fit_intercept)[0], REAL(ridge)[0]);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_piecewise_direct_standardization_create");

    n4m_matrix_view_t Sv, Tv, Xv, Yv;
    init_view_rowmajor(&Sv, source_in, sr, sc);
    init_view_rowmajor(&Tv, target_in, tr, tc);
    init_view_rowmajor(&Xv, in, rows, cols);
    init_view_rowmajor(&Yv, out_buf, rows, cols);
    s = n4m_pp_piecewise_direct_standardization_fit(h, Sv, Tv);
    if (s != N4M_OK) {
        n4m_pp_piecewise_direct_standardization_destroy(h);
        n4m_throw(s, "n4m_pp_piecewise_direct_standardization_fit");
    }
    s = n4m_pp_piecewise_direct_standardization_transform(h, Xv, Yv);
    n4m_pp_piecewise_direct_standardization_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_piecewise_direct_standardization_transform");
    return rowmajor_to_R_matrix(out_buf, rows, cols);
}

static SEXP r_n4m_pp_saps_fit_transform(
    SEXP source, SEXP target, SEXP x, SEXP n_components, SEXP score_weight,
    SEXP fit_intercept, SEXP ridge) {
    int64_t sr, sc, tr, tc, rows, cols;
    double *source_in = r_matrix_to_rowmajor(source, &sr, &sc);
    double *target_in = r_matrix_to_rowmajor(target, &tr, &tc);
    double *in = r_matrix_to_rowmajor(x, &rows, &cols);
    ensure_same_cols(sc, tc, "source", "target");
    ensure_same_cols(sc, cols, "source", "X");
    double *out_buf = (double *)R_alloc((size_t)(rows * cols), sizeof(double));

    n4m_pp_saps_handle_t *h = NULL;
    n4m_status_t s = n4m_pp_saps_create(
        &h, INTEGER(n_components)[0], REAL(score_weight)[0],
        LOGICAL(fit_intercept)[0], REAL(ridge)[0]);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_saps_create");

    n4m_matrix_view_t Sv, Tv, Xv, Yv;
    init_view_rowmajor(&Sv, source_in, sr, sc);
    init_view_rowmajor(&Tv, target_in, tr, tc);
    init_view_rowmajor(&Xv, in, rows, cols);
    init_view_rowmajor(&Yv, out_buf, rows, cols);
    s = n4m_pp_saps_fit(h, Sv, Tv);
    if (s != N4M_OK) {
        n4m_pp_saps_destroy(h);
        n4m_throw(s, "n4m_pp_saps_fit");
    }
    s = n4m_pp_saps_transform(h, Xv, Yv);
    n4m_pp_saps_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_saps_transform");
    return rowmajor_to_R_matrix(out_buf, rows, cols);
}

static SEXP r_n4m_pp_slope_bias_fit_transform(SEXP source, SEXP target, SEXP y) {
    R_xlen_t n_source = Rf_length(source);
    R_xlen_t n_target = Rf_length(target);
    R_xlen_t n_y = Rf_length(y);
    if (n_source != n_target) {
        Rf_error("nirs4all-methods: source and target must have the same length");
    }
    SEXP out = PROTECT(Rf_allocVector(REALSXP, n_y));
    n4m_pp_slope_bias_handle_t *h = NULL;
    n4m_status_t s = n4m_pp_slope_bias_create(&h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_slope_bias_create");
    s = n4m_pp_slope_bias_fit(h, REAL(source), REAL(target), (int64_t)n_source);
    if (s != N4M_OK) {
        n4m_pp_slope_bias_destroy(h);
        n4m_throw(s, "n4m_pp_slope_bias_fit");
    }
    s = n4m_pp_slope_bias_transform(h, REAL(y), (int64_t)n_y, REAL(out));
    n4m_pp_slope_bias_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_slope_bias_transform");
    UNPROTECT(1);
    return out;
}

static SEXP r_n4m_pp_local_centering_fit_transform(SEXP source, SEXP target, SEXP x) {
    int64_t sr, sc, tr, tc, rows, cols;
    double *source_in = r_matrix_to_rowmajor(source, &sr, &sc);
    double *target_in = r_matrix_to_rowmajor(target, &tr, &tc);
    double *in = r_matrix_to_rowmajor(x, &rows, &cols);
    ensure_same_cols(sc, tc, "source", "target");
    ensure_same_cols(sc, cols, "source", "X");
    double *out_buf = (double *)R_alloc((size_t)(rows * cols), sizeof(double));

    n4m_pp_local_centering_handle_t *h = NULL;
    n4m_status_t s = n4m_pp_local_centering_create(&h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_local_centering_create");
    n4m_matrix_view_t Sv, Tv, Xv, Yv;
    init_view_rowmajor(&Sv, source_in, sr, sc);
    init_view_rowmajor(&Tv, target_in, tr, tc);
    init_view_rowmajor(&Xv, in, rows, cols);
    init_view_rowmajor(&Yv, out_buf, rows, cols);
    s = n4m_pp_local_centering_fit(h, Sv, Tv);
    if (s != N4M_OK) {
        n4m_pp_local_centering_destroy(h);
        n4m_throw(s, "n4m_pp_local_centering_fit");
    }
    s = n4m_pp_local_centering_transform(h, Xv, Yv);
    n4m_pp_local_centering_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_local_centering_transform");
    return rowmajor_to_R_matrix(out_buf, rows, cols);
}

static SEXP r_n4m_pp_weighted_snv_fit_transform(SEXP x, SEXP weights,
                                                SEXP ddof, SEXP eps) {
    int64_t rows, cols, n_weights;
    double *in, *out_buf;
    n4m_matrix_view_t Xv, Yv;
    const double *w = optional_real_vector(weights, &n_weights, "weights");
    prepare_same_shape_matrix(x, &rows, &cols, &in, &out_buf, &Xv, &Yv,
                              "n4m_pp_weighted_snv_matrix");
    n4m_pp_weighted_snv_handle_t *h = NULL;
    n4m_status_t s = n4m_pp_weighted_snv_create(
        &h, w, n_weights, INTEGER(ddof)[0], REAL(eps)[0]);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_weighted_snv_create");
    s = n4m_pp_weighted_snv_fit(h, Xv);
    if (s != N4M_OK) {
        n4m_pp_weighted_snv_destroy(h);
        n4m_throw(s, "n4m_pp_weighted_snv_fit");
    }
    s = n4m_pp_weighted_snv_transform(h, Xv, Yv);
    n4m_pp_weighted_snv_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_weighted_snv_transform");
    return rowmajor_to_R_matrix(out_buf, rows, cols);
}

static SEXP r_n4m_pp_vsn_fit_transform(SEXP x, SEXP eps) {
    int64_t rows, cols;
    double *in, *out_buf;
    n4m_matrix_view_t Xv, Yv;
    prepare_same_shape_matrix(x, &rows, &cols, &in, &out_buf, &Xv, &Yv,
                              "n4m_pp_vsn_matrix");
    n4m_pp_vsn_handle_t *h = NULL;
    n4m_status_t s = n4m_pp_vsn_create(&h, REAL(eps)[0]);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_vsn_create");
    s = n4m_pp_vsn_fit(h, Xv);
    if (s != N4M_OK) {
        n4m_pp_vsn_destroy(h);
        n4m_throw(s, "n4m_pp_vsn_fit");
    }
    s = n4m_pp_vsn_transform(h, Xv, Yv);
    n4m_pp_vsn_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_vsn_transform");
    return rowmajor_to_R_matrix(out_buf, rows, cols);
}

static SEXP r_n4m_pp_piecewise_snv_fit_transform(SEXP x, SEXP window,
                                                 SEXP ddof, SEXP eps) {
    int64_t rows, cols;
    double *in, *out_buf;
    n4m_matrix_view_t Xv, Yv;
    prepare_same_shape_matrix(x, &rows, &cols, &in, &out_buf, &Xv, &Yv,
                              "n4m_pp_piecewise_snv_matrix");
    n4m_pp_piecewise_snv_handle_t *h = NULL;
    n4m_status_t s = n4m_pp_piecewise_snv_create(
        &h, INTEGER(window)[0], INTEGER(ddof)[0], REAL(eps)[0]);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_piecewise_snv_create");
    s = n4m_pp_piecewise_snv_fit(h, Xv);
    if (s != N4M_OK) {
        n4m_pp_piecewise_snv_destroy(h);
        n4m_throw(s, "n4m_pp_piecewise_snv_fit");
    }
    s = n4m_pp_piecewise_snv_transform(h, Xv, Yv);
    n4m_pp_piecewise_snv_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_piecewise_snv_transform");
    return rowmajor_to_R_matrix(out_buf, rows, cols);
}

static SEXP r_n4m_pp_piecewise_msc_fit_transform(SEXP x, SEXP reference,
                                                 SEXP window, SEXP eps) {
    int64_t rows, cols, n_ref;
    double *in, *out_buf;
    n4m_matrix_view_t Xv, Yv;
    const double *ref = optional_real_vector(reference, &n_ref, "reference");
    prepare_same_shape_matrix(x, &rows, &cols, &in, &out_buf, &Xv, &Yv,
                              "n4m_pp_piecewise_msc_matrix");
    n4m_pp_piecewise_msc_handle_t *h = NULL;
    n4m_status_t s = n4m_pp_piecewise_msc_create(
        &h, ref, n_ref, INTEGER(window)[0], REAL(eps)[0]);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_piecewise_msc_create");
    s = n4m_pp_piecewise_msc_fit(h, Xv);
    if (s != N4M_OK) {
        n4m_pp_piecewise_msc_destroy(h);
        n4m_throw(s, "n4m_pp_piecewise_msc_fit");
    }
    s = n4m_pp_piecewise_msc_transform(h, Xv, Yv);
    n4m_pp_piecewise_msc_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_piecewise_msc_transform");
    return rowmajor_to_R_matrix(out_buf, rows, cols);
}

static SEXP r_n4m_pp_localized_msc_fit_transform(SEXP x, SEXP reference,
                                                 SEXP window, SEXP eps) {
    int64_t rows, cols, n_ref;
    double *in, *out_buf;
    n4m_matrix_view_t Xv, Yv;
    const double *ref = optional_real_vector(reference, &n_ref, "reference");
    prepare_same_shape_matrix(x, &rows, &cols, &in, &out_buf, &Xv, &Yv,
                              "n4m_pp_localized_msc_matrix");
    n4m_pp_localized_msc_handle_t *h = NULL;
    n4m_status_t s = n4m_pp_localized_msc_create(
        &h, ref, n_ref, INTEGER(window)[0], REAL(eps)[0]);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_localized_msc_create");
    s = n4m_pp_localized_msc_fit(h, Xv);
    if (s != N4M_OK) {
        n4m_pp_localized_msc_destroy(h);
        n4m_throw(s, "n4m_pp_localized_msc_fit");
    }
    s = n4m_pp_localized_msc_transform(h, Xv, Yv);
    n4m_pp_localized_msc_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_pp_localized_msc_transform");
    return rowmajor_to_R_matrix(out_buf, rows, cols);
}

#define DEFINE_ALIGN_WRAPPER(RNAME, HANDLE_T, CREATE_F, FIT_F, TRANSFORM_F, DESTROY_F) \
static SEXP r_ ## RNAME(SEXP x, SEXP reference, SEXP interval_size, SEXP max_shift) { \
    int64_t rows, cols, n_ref; \
    double *in, *out_buf; \
    n4m_matrix_view_t Xv, Yv; \
    const double *ref = optional_real_vector(reference, &n_ref, "reference"); \
    prepare_same_shape_matrix(x, &rows, &cols, &in, &out_buf, &Xv, &Yv, #RNAME "_matrix"); \
    HANDLE_T *h = NULL; \
    n4m_status_t s = CREATE_F(&h, ref, n_ref, INTEGER(interval_size)[0], INTEGER(max_shift)[0]); \
    if (s != N4M_OK) n4m_throw(s, #CREATE_F); \
    s = FIT_F(h, Xv); \
    if (s != N4M_OK) { DESTROY_F(h); n4m_throw(s, #FIT_F); } \
    s = TRANSFORM_F(h, Xv, Yv); \
    DESTROY_F(h); \
    if (s != N4M_OK) n4m_throw(s, #TRANSFORM_F); \
    return rowmajor_to_R_matrix(out_buf, rows, cols); \
}

DEFINE_ALIGN_WRAPPER(n4m_pp_xcorr_align_fit_transform,
                     n4m_pp_xcorr_align_handle_t,
                     n4m_pp_xcorr_align_create,
                     n4m_pp_xcorr_align_fit,
                     n4m_pp_xcorr_align_transform,
                     n4m_pp_xcorr_align_destroy)
DEFINE_ALIGN_WRAPPER(n4m_pp_icoshift_align_fit_transform,
                     n4m_pp_icoshift_align_handle_t,
                     n4m_pp_icoshift_align_create,
                     n4m_pp_icoshift_align_fit,
                     n4m_pp_icoshift_align_transform,
                     n4m_pp_icoshift_align_destroy)
DEFINE_ALIGN_WRAPPER(n4m_pp_dtw_align_fit_transform,
                     n4m_pp_dtw_align_handle_t,
                     n4m_pp_dtw_align_create,
                     n4m_pp_dtw_align_fit,
                     n4m_pp_dtw_align_transform,
                     n4m_pp_dtw_align_destroy)
DEFINE_ALIGN_WRAPPER(n4m_pp_cow_align_fit_transform,
                     n4m_pp_cow_align_handle_t,
                     n4m_pp_cow_align_create,
                     n4m_pp_cow_align_fit,
                     n4m_pp_cow_align_transform,
                     n4m_pp_cow_align_destroy)

static SEXP r_n4m_filter_variance_fit_transform(SEXP x, SEXP threshold,
                                                SEXP top_k) {
    int64_t rows, cols, out_cols = 0;
    double *in = r_matrix_to_rowmajor(x, &rows, &cols);
    n4m_matrix_view_t Xv;
    init_view_rowmajor(&Xv, in, rows, cols);
    n4m_filter_variance_handle_t *h = NULL;
    n4m_status_t s = n4m_filter_variance_create(&h, REAL(threshold)[0],
                                                INTEGER(top_k)[0]);
    if (s != N4M_OK) n4m_throw(s, "n4m_filter_variance_create");
    s = n4m_filter_variance_fit(h, Xv);
    if (s != N4M_OK) {
        n4m_filter_variance_destroy(h);
        n4m_throw(s, "n4m_filter_variance_fit");
    }
    s = n4m_filter_variance_output_cols(h, &out_cols);
    if (s != N4M_OK) {
        n4m_filter_variance_destroy(h);
        n4m_throw(s, "n4m_filter_variance_output_cols");
    }
    double *out_buf = (double *)R_alloc((size_t)(rows * out_cols), sizeof(double));
    n4m_matrix_view_t Yv;
    init_view_rowmajor(&Yv, out_buf, rows, out_cols);
    s = n4m_filter_variance_transform(h, Xv, Yv);
    n4m_filter_variance_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_filter_variance_transform");
    return rowmajor_to_R_matrix(out_buf, rows, out_cols);
}

static SEXP r_n4m_filter_correlation_fit_transform(SEXP x, SEXP y,
                                                   SEXP threshold,
                                                   SEXP top_k) {
    int64_t rows, cols, out_cols = 0;
    double *in = r_matrix_to_rowmajor(x, &rows, &cols);
    if (Rf_length(y) != rows) {
        Rf_error("nirs4all-methods: y length must match nrow(X)");
    }
    n4m_matrix_view_t Xv;
    init_view_rowmajor(&Xv, in, rows, cols);
    n4m_filter_correlation_handle_t *h = NULL;
    n4m_status_t s = n4m_filter_correlation_create(&h, REAL(threshold)[0],
                                                   INTEGER(top_k)[0]);
    if (s != N4M_OK) n4m_throw(s, "n4m_filter_correlation_create");
    s = n4m_filter_correlation_fit(h, Xv, REAL(y), rows);
    if (s != N4M_OK) {
        n4m_filter_correlation_destroy(h);
        n4m_throw(s, "n4m_filter_correlation_fit");
    }
    s = n4m_filter_correlation_output_cols(h, &out_cols);
    if (s != N4M_OK) {
        n4m_filter_correlation_destroy(h);
        n4m_throw(s, "n4m_filter_correlation_output_cols");
    }
    double *out_buf = (double *)R_alloc((size_t)(rows * out_cols), sizeof(double));
    n4m_matrix_view_t Yv;
    init_view_rowmajor(&Yv, out_buf, rows, out_cols);
    s = n4m_filter_correlation_transform(h, Xv, Yv);
    n4m_filter_correlation_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_filter_correlation_transform");
    return rowmajor_to_R_matrix(out_buf, rows, out_cols);
}

static SEXP r_n4m_interval_generator_fit_transform(SEXP x, SEXP interval_size,
                                                   SEXP step) {
    int64_t rows, cols, out_cols = 0;
    double *in = r_matrix_to_rowmajor(x, &rows, &cols);
    n4m_matrix_view_t Xv;
    init_view_rowmajor(&Xv, in, rows, cols);
    n4m_interval_generator_handle_t *h = NULL;
    n4m_status_t s = n4m_interval_generator_create(
        &h, INTEGER(interval_size)[0], INTEGER(step)[0]);
    if (s != N4M_OK) n4m_throw(s, "n4m_interval_generator_create");
    s = n4m_interval_generator_fit(h, Xv);
    if (s != N4M_OK) {
        n4m_interval_generator_destroy(h);
        n4m_throw(s, "n4m_interval_generator_fit");
    }
    s = n4m_interval_generator_output_cols(h, &out_cols);
    if (s != N4M_OK) {
        n4m_interval_generator_destroy(h);
        n4m_throw(s, "n4m_interval_generator_output_cols");
    }
    double *out_buf = (double *)R_alloc((size_t)(rows * out_cols), sizeof(double));
    n4m_matrix_view_t Yv;
    init_view_rowmajor(&Yv, out_buf, rows, out_cols);
    s = n4m_interval_generator_transform(h, Xv, Yv);
    n4m_interval_generator_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_interval_generator_transform");
    return rowmajor_to_R_matrix(out_buf, rows, out_cols);
}

/* ------------------------------------------------------------------ */
/* Augmentation and spectral-quality wrappers                          */
/* ------------------------------------------------------------------ */

#define R_AUG_DONE() do {                                                   \
    n4m_rng_pcg64_destroy(rng);                                             \
    if (s != N4M_OK) n4m_throw(s, op);                                      \
    return rowmajor_to_R_matrix(out_buf, rows, cols);                       \
} while (0)

#define R_AUG_APPLY0(TYPE, CREATE, APPLY, DESTROY) do {                     \
    TYPE *h = NULL;                                                         \
    s = CREATE(&h, rng);                                                    \
    if (s == N4M_OK) s = APPLY(h, Xv, Yv);                                  \
    DESTROY(h);                                                             \
    R_AUG_DONE();                                                           \
} while (0)

#define R_AUG_APPLY(TYPE, CREATE, APPLY, DESTROY, ...) do {                 \
    TYPE *h = NULL;                                                         \
    s = CREATE(&h, rng, __VA_ARGS__);                                       \
    if (s == N4M_OK) s = APPLY(h, Xv, Yv);                                  \
    DESTROY(h);                                                             \
    R_AUG_DONE();                                                           \
} while (0)

#define R_AUG_APPLY_WITH_WL(TYPE, CREATE, APPLY, DESTROY, ...) do {         \
    double *wl = r_wavelengths_to_rowmajor(wavelengths, cols, op);          \
    n4m_matrix_view_t Wv;                                                   \
    s = init_view_rowmajor(&Wv, wl, 1, cols);                               \
    if (s != N4M_OK) R_AUG_DONE();                                          \
    TYPE *h = NULL;                                                         \
    s = CREATE(&h, rng, __VA_ARGS__);                                       \
    if (s == N4M_OK) s = APPLY(h, Xv, Wv, Yv);                              \
    DESTROY(h);                                                             \
    R_AUG_DONE();                                                           \
} while (0)

#define R_AUG_CREATE_WL(TYPE, CREATE, APPLY, DESTROY, ...) do {             \
    double *wl = r_wavelengths_to_rowmajor(wavelengths, cols, op);          \
    TYPE *h = NULL;                                                         \
    s = CREATE(&h, rng, __VA_ARGS__, wl, cols);                             \
    if (s == N4M_OK) s = APPLY(h, Xv, Yv);                                  \
    DESTROY(h);                                                             \
    R_AUG_DONE();                                                           \
} while (0)

static SEXP r_n4m_aug_apply(SEXP name, SEXP x, SEXP wavelengths, SEXP seed,
                            SEXP params) {
    if (TYPEOF(name) != STRSXP || Rf_length(name) < 1) {
        Rf_error("nirs4all-methods: augmenter name must be a character scalar");
    }
    const char *op = CHAR(STRING_ELT(name, 0));
    int64_t rows, cols;
    double *in, *out_buf;
    n4m_matrix_view_t Xv, Yv;
    prepare_same_shape_matrix(x, &rows, &cols, &in, &out_buf, &Xv, &Yv, op);
    n4m_rng_pcg64_state_t *rng = r_create_rng(seed, "n4m_rng_pcg64_create");
    n4m_status_t s = N4M_OK;

    if (strcmp(op, "gaussian_noise") == 0) {
        R_AUG_APPLY(n4m_aug_gaussian_noise_handle_t,
                    n4m_aug_gaussian_noise_create,
                    n4m_aug_gaussian_noise_apply,
                    n4m_aug_gaussian_noise_destroy,
                    r_aug_param(params, 0, op));
    }
    if (strcmp(op, "multiplicative_noise") == 0) {
        R_AUG_APPLY(n4m_aug_multiplicative_noise_handle_t,
                    n4m_aug_multiplicative_noise_create,
                    n4m_aug_multiplicative_noise_apply,
                    n4m_aug_multiplicative_noise_destroy,
                    r_aug_param(params, 0, op));
    }
    if (strcmp(op, "spike_noise") == 0) {
        R_AUG_APPLY(n4m_aug_spike_noise_handle_t,
                    n4m_aug_spike_noise_create,
                    n4m_aug_spike_noise_apply,
                    n4m_aug_spike_noise_destroy,
                    r_aug_param_i32(params, 0, op),
                    r_aug_param_i32(params, 1, op),
                    r_aug_param(params, 2, op),
                    r_aug_param(params, 3, op));
    }
    if (strcmp(op, "hetero_noise") == 0) {
        R_AUG_APPLY(n4m_aug_hetero_noise_handle_t,
                    n4m_aug_hetero_noise_create,
                    n4m_aug_hetero_noise_apply,
                    n4m_aug_hetero_noise_destroy,
                    r_aug_param(params, 0, op),
                    r_aug_param(params, 1, op));
    }
    if (strcmp(op, "linear_drift") == 0) {
        R_AUG_APPLY(n4m_aug_linear_drift_handle_t,
                    n4m_aug_linear_drift_create,
                    n4m_aug_linear_drift_apply,
                    n4m_aug_linear_drift_destroy,
                    r_aug_param(params, 0, op),
                    r_aug_param(params, 1, op),
                    r_aug_param(params, 2, op),
                    r_aug_param(params, 3, op));
    }
    if (strcmp(op, "poly_drift") == 0) {
        const int32_t degree = r_aug_param_i32(params, 0, op);
        const int64_t n_coeffs = (int64_t)degree + 1;
        double *lo = (double *)R_alloc((size_t)n_coeffs, sizeof(double));
        double *hi = (double *)R_alloc((size_t)n_coeffs, sizeof(double));
        for (int64_t i = 0; i < n_coeffs; ++i) {
            lo[i] = r_aug_param(params, 1, op);
            hi[i] = r_aug_param(params, 2, op);
        }
        R_AUG_APPLY(n4m_aug_poly_drift_handle_t,
                    n4m_aug_poly_drift_create,
                    n4m_aug_poly_drift_apply,
                    n4m_aug_poly_drift_destroy,
                    degree, lo, hi);
    }
    if (strcmp(op, "path_length") == 0) {
        R_AUG_APPLY(n4m_aug_path_length_handle_t,
                    n4m_aug_path_length_create,
                    n4m_aug_path_length_apply,
                    n4m_aug_path_length_destroy,
                    r_aug_param(params, 0, op),
                    r_aug_param(params, 1, op));
    }
    if (strcmp(op, "wavelength_shift") == 0) {
        R_AUG_CREATE_WL(n4m_aug_wavelength_shift_handle_t,
                        n4m_aug_wavelength_shift_create,
                        n4m_aug_wavelength_shift_apply,
                        n4m_aug_wavelength_shift_destroy,
                        r_aug_param(params, 0, op),
                        r_aug_param(params, 1, op));
    }
    if (strcmp(op, "wavelength_stretch") == 0) {
        R_AUG_CREATE_WL(n4m_aug_wavelength_stretch_handle_t,
                        n4m_aug_wavelength_stretch_create,
                        n4m_aug_wavelength_stretch_apply,
                        n4m_aug_wavelength_stretch_destroy,
                        r_aug_param(params, 0, op),
                        r_aug_param(params, 1, op));
    }
    if (strcmp(op, "local_warp") == 0) {
        R_AUG_CREATE_WL(n4m_aug_local_warp_handle_t,
                        n4m_aug_local_warp_create,
                        n4m_aug_local_warp_apply,
                        n4m_aug_local_warp_destroy,
                        r_aug_param_i32(params, 0, op),
                        r_aug_param(params, 1, op));
    }
    if (strcmp(op, "band_perturb") == 0) {
        R_AUG_APPLY(n4m_aug_band_perturb_handle_t,
                    n4m_aug_band_perturb_create,
                    n4m_aug_band_perturb_apply,
                    n4m_aug_band_perturb_destroy,
                    r_aug_param_i32(params, 0, op),
                    r_aug_param_i32(params, 1, op),
                    r_aug_param_i32(params, 2, op),
                    r_aug_param(params, 3, op),
                    r_aug_param(params, 4, op),
                    r_aug_param(params, 5, op),
                    r_aug_param(params, 6, op));
    }
    if (strcmp(op, "band_mask") == 0) {
        R_AUG_APPLY(n4m_aug_band_mask_handle_t,
                    n4m_aug_band_mask_create,
                    n4m_aug_band_mask_apply,
                    n4m_aug_band_mask_destroy,
                    r_aug_param_i32(params, 0, op),
                    r_aug_param_i32(params, 1, op),
                    r_aug_param_i32(params, 2, op),
                    r_aug_param_i32(params, 3, op),
                    r_aug_param_i32(params, 4, op));
    }
    if (strcmp(op, "channel_dropout") == 0) {
        R_AUG_APPLY(n4m_aug_channel_dropout_handle_t,
                    n4m_aug_channel_dropout_create,
                    n4m_aug_channel_dropout_apply,
                    n4m_aug_channel_dropout_destroy,
                    r_aug_param(params, 0, op),
                    r_aug_param_i32(params, 1, op));
    }
    if (strcmp(op, "gauss_jitter") == 0) {
        R_AUG_APPLY(n4m_aug_gauss_jitter_handle_t,
                    n4m_aug_gauss_jitter_create,
                    n4m_aug_gauss_jitter_apply,
                    n4m_aug_gauss_jitter_destroy,
                    r_aug_param(params, 0, op),
                    r_aug_param(params, 1, op),
                    r_aug_param_i32(params, 2, op));
    }
    if (strcmp(op, "unsharp_mask") == 0) {
        R_AUG_APPLY(n4m_aug_unsharp_mask_handle_t,
                    n4m_aug_unsharp_mask_create,
                    n4m_aug_unsharp_mask_apply,
                    n4m_aug_unsharp_mask_destroy,
                    r_aug_param(params, 0, op),
                    r_aug_param(params, 1, op),
                    r_aug_param(params, 2, op),
                    r_aug_param_i32(params, 3, op));
    }
    if (strcmp(op, "magnitude_warp") == 0) {
        R_AUG_CREATE_WL(n4m_aug_magnitude_warp_handle_t,
                        n4m_aug_magnitude_warp_create,
                        n4m_aug_magnitude_warp_apply,
                        n4m_aug_magnitude_warp_destroy,
                        r_aug_param_i32(params, 0, op),
                        r_aug_param(params, 1, op),
                        r_aug_param(params, 2, op));
    }
    if (strcmp(op, "local_clip") == 0) {
        R_AUG_APPLY(n4m_aug_local_clip_handle_t,
                    n4m_aug_local_clip_create,
                    n4m_aug_local_clip_apply,
                    n4m_aug_local_clip_destroy,
                    r_aug_param_i32(params, 0, op),
                    r_aug_param_i32(params, 1, op),
                    r_aug_param_i32(params, 2, op));
    }
    if (strcmp(op, "mixup") == 0) {
        R_AUG_APPLY(n4m_aug_mixup_handle_t,
                    n4m_aug_mixup_create,
                    n4m_aug_mixup_apply,
                    n4m_aug_mixup_destroy,
                    r_aug_param(params, 0, op));
    }
    if (strcmp(op, "local_mixup") == 0) {
        R_AUG_APPLY(n4m_aug_local_mixup_handle_t,
                    n4m_aug_local_mixup_create,
                    n4m_aug_local_mixup_apply,
                    n4m_aug_local_mixup_destroy,
                    r_aug_param(params, 0, op),
                    r_aug_param_i32(params, 1, op));
    }
    if (strcmp(op, "scatter_sim") == 0) {
        R_AUG_APPLY(n4m_aug_scatter_sim_handle_t,
                    n4m_aug_scatter_sim_create,
                    n4m_aug_scatter_sim_apply,
                    n4m_aug_scatter_sim_destroy,
                    r_aug_param(params, 0, op),
                    r_aug_param(params, 1, op),
                    r_aug_param(params, 2, op),
                    r_aug_param(params, 3, op));
    }
    if (strcmp(op, "particle_size") == 0) {
        R_AUG_CREATE_WL(n4m_aug_particle_size_handle_t,
                        n4m_aug_particle_size_create,
                        n4m_aug_particle_size_apply,
                        n4m_aug_particle_size_destroy,
                        r_aug_param(params, 0, op),
                        r_aug_param(params, 1, op),
                        r_aug_param_int(params, 2, op),
                        r_aug_param(params, 3, op),
                        r_aug_param(params, 4, op),
                        r_aug_param(params, 5, op),
                        r_aug_param(params, 6, op),
                        r_aug_param(params, 7, op),
                        r_aug_param_int(params, 8, op),
                        r_aug_param(params, 9, op));
    }
    if (strcmp(op, "emsc_distort") == 0) {
        R_AUG_CREATE_WL(n4m_aug_emsc_distort_handle_t,
                        n4m_aug_emsc_distort_create,
                        n4m_aug_emsc_distort_apply,
                        n4m_aug_emsc_distort_destroy,
                        r_aug_param(params, 0, op),
                        r_aug_param(params, 1, op),
                        r_aug_param(params, 2, op),
                        r_aug_param(params, 3, op),
                        r_aug_param_i32(params, 4, op),
                        r_aug_param(params, 5, op),
                        r_aug_param(params, 6, op));
    }
    if (strcmp(op, "batch_effect") == 0) {
        R_AUG_CREATE_WL(n4m_aug_batch_effect_handle_t,
                        n4m_aug_batch_effect_create,
                        n4m_aug_batch_effect_apply,
                        n4m_aug_batch_effect_destroy,
                        r_aug_param(params, 0, op),
                        r_aug_param(params, 1, op),
                        r_aug_param(params, 2, op),
                        r_aug_param_i32(params, 3, op));
    }
    if (strcmp(op, "instrument_broaden") == 0) {
        R_AUG_CREATE_WL(n4m_aug_instrument_broaden_handle_t,
                        n4m_aug_instrument_broaden_create,
                        n4m_aug_instrument_broaden_apply,
                        n4m_aug_instrument_broaden_destroy,
                        r_aug_param(params, 0, op),
                        r_aug_param_int(params, 1, op),
                        r_aug_param(params, 2, op),
                        r_aug_param(params, 3, op),
                        r_aug_param_i32(params, 4, op));
    }
    if (strcmp(op, "dead_band") == 0) {
        R_AUG_APPLY(n4m_aug_dead_band_handle_t,
                    n4m_aug_dead_band_create,
                    n4m_aug_dead_band_apply,
                    n4m_aug_dead_band_destroy,
                    r_aug_param_i32(params, 0, op),
                    r_aug_param_i32(params, 1, op),
                    r_aug_param_i32(params, 2, op),
                    r_aug_param(params, 3, op),
                    r_aug_param(params, 4, op),
                    r_aug_param_i32(params, 5, op));
    }
    if (strcmp(op, "temperature") == 0) {
        R_AUG_CREATE_WL(n4m_aug_temperature_handle_t,
                        n4m_aug_temperature_create,
                        n4m_aug_temperature_apply,
                        n4m_aug_temperature_destroy,
                        r_aug_param(params, 0, op),
                        r_aug_param_int(params, 1, op),
                        r_aug_param(params, 2, op),
                        r_aug_param(params, 3, op),
                        r_aug_param_int(params, 4, op),
                        r_aug_param_int(params, 5, op),
                        r_aug_param_int(params, 6, op),
                        r_aug_param_int(params, 7, op));
    }
    if (strcmp(op, "moisture") == 0) {
        R_AUG_CREATE_WL(n4m_aug_moisture_handle_t,
                        n4m_aug_moisture_create,
                        n4m_aug_moisture_apply,
                        n4m_aug_moisture_destroy,
                        r_aug_param(params, 0, op),
                        r_aug_param_int(params, 1, op),
                        r_aug_param(params, 2, op),
                        r_aug_param(params, 3, op),
                        r_aug_param(params, 4, op),
                        r_aug_param(params, 5, op),
                        r_aug_param(params, 6, op),
                        r_aug_param(params, 7, op),
                        r_aug_param_int(params, 8, op),
                        r_aug_param_int(params, 9, op));
    }
    if (strcmp(op, "detector_rolloff") == 0) {
        R_AUG_APPLY_WITH_WL(n4m_aug_detector_rolloff_handle_t,
                            n4m_aug_detector_rolloff_create,
                            n4m_aug_detector_rolloff_apply,
                            n4m_aug_detector_rolloff_destroy,
                            r_aug_param_i32(params, 0, op),
                            r_aug_param(params, 1, op),
                            r_aug_param(params, 2, op),
                            r_aug_param_i32(params, 3, op));
    }
    if (strcmp(op, "stray_light") == 0) {
        R_AUG_APPLY_WITH_WL(n4m_aug_stray_light_handle_t,
                            n4m_aug_stray_light_create,
                            n4m_aug_stray_light_apply,
                            n4m_aug_stray_light_destroy,
                            r_aug_param(params, 0, op),
                            r_aug_param(params, 1, op),
                            r_aug_param(params, 2, op),
                            r_aug_param_i32(params, 3, op));
    }
    if (strcmp(op, "edge_curve") == 0) {
        R_AUG_APPLY_WITH_WL(n4m_aug_edge_curve_handle_t,
                            n4m_aug_edge_curve_create,
                            n4m_aug_edge_curve_apply,
                            n4m_aug_edge_curve_destroy,
                            r_aug_param(params, 0, op),
                            r_aug_param_i32(params, 1, op),
                            r_aug_param(params, 2, op),
                            r_aug_param(params, 3, op));
    }
    if (strcmp(op, "truncated_peak") == 0) {
        R_AUG_APPLY_WITH_WL(n4m_aug_truncated_peak_handle_t,
                            n4m_aug_truncated_peak_create,
                            n4m_aug_truncated_peak_apply,
                            n4m_aug_truncated_peak_destroy,
                            r_aug_param(params, 0, op),
                            r_aug_param(params, 1, op),
                            r_aug_param(params, 2, op),
                            r_aug_param(params, 3, op),
                            r_aug_param(params, 4, op),
                            r_aug_param_i32(params, 5, op),
                            r_aug_param_i32(params, 6, op));
    }
    if (strcmp(op, "edge_artifacts") == 0) {
        R_AUG_APPLY_WITH_WL(n4m_aug_edge_artifacts_handle_t,
                            n4m_aug_edge_artifacts_create,
                            n4m_aug_edge_artifacts_apply,
                            n4m_aug_edge_artifacts_destroy,
                            r_aug_param_i32(params, 0, op),
                            r_aug_param(params, 1, op),
                            r_aug_param_i32(params, 2, op));
    }
    if (strcmp(op, "spline_smooth") == 0) {
        R_AUG_APPLY0(n4m_aug_spline_smooth_handle_t,
                     n4m_aug_spline_smooth_create,
                     n4m_aug_spline_smooth_apply,
                     n4m_aug_spline_smooth_destroy);
    }
    if (strcmp(op, "spline_x_perturb") == 0) {
        R_AUG_APPLY(n4m_aug_spline_x_perturb_handle_t,
                    n4m_aug_spline_x_perturb_create,
                    n4m_aug_spline_x_perturb_apply,
                    n4m_aug_spline_x_perturb_destroy,
                    r_aug_param_i32(params, 0, op),
                    r_aug_param(params, 1, op),
                    r_aug_param(params, 2, op),
                    r_aug_param(params, 3, op));
    }
    if (strcmp(op, "spline_y_perturb") == 0) {
        R_AUG_APPLY(n4m_aug_spline_y_perturb_handle_t,
                    n4m_aug_spline_y_perturb_create,
                    n4m_aug_spline_y_perturb_apply,
                    n4m_aug_spline_y_perturb_destroy,
                    r_aug_param_i32(params, 0, op),
                    r_aug_param(params, 1, op));
    }
    if (strcmp(op, "rotate_translate") == 0) {
        R_AUG_APPLY(n4m_aug_rotate_translate_handle_t,
                    n4m_aug_rotate_translate_create,
                    n4m_aug_rotate_translate_apply,
                    n4m_aug_rotate_translate_destroy,
                    r_aug_param(params, 0, op),
                    r_aug_param(params, 1, op));
    }
    if (strcmp(op, "random_x_op") == 0) {
        R_AUG_APPLY(n4m_aug_random_x_op_handle_t,
                    n4m_aug_random_x_op_create,
                    n4m_aug_random_x_op_apply,
                    n4m_aug_random_x_op_destroy,
                    r_aug_param_i32(params, 0, op),
                    r_aug_param(params, 1, op),
                    r_aug_param(params, 2, op));
    }

    n4m_rng_pcg64_destroy(rng);
    Rf_error("nirs4all-methods: unknown augmenter '%s'", op);
}

#undef R_AUG_CREATE_WL
#undef R_AUG_APPLY_WITH_WL
#undef R_AUG_APPLY
#undef R_AUG_APPLY0
#undef R_AUG_DONE

static SEXP r_n4m_filter_quality_apply(SEXP x, SEXP max_nan_ratio,
                                       SEXP max_zero_ratio,
                                       SEXP min_variance,
                                       SEXP use_max,
                                       SEXP max_value,
                                       SEXP use_min,
                                       SEXP min_value,
                                       SEXP check_inf) {
    int64_t rows, cols;
    double *in = r_matrix_to_rowmajor(x, &rows, &cols);
    n4m_matrix_view_t Xv;
    n4m_status_t s = init_view_rowmajor(&Xv, in, rows, cols);
    if (s != N4M_OK) n4m_throw(s, "n4m_filter_quality_matrix");

    n4m_filter_quality_handle_t *h = NULL;
    s = n4m_filter_quality_create(&h,
                                  REAL(max_nan_ratio)[0],
                                  REAL(max_zero_ratio)[0],
                                  REAL(min_variance)[0],
                                  LOGICAL(use_max)[0],
                                  REAL(max_value)[0],
                                  LOGICAL(use_min)[0],
                                  REAL(min_value)[0],
                                  LOGICAL(check_inf)[0]);
    if (s != N4M_OK) n4m_throw(s, "n4m_filter_quality_create");

    uint8_t *mask = (uint8_t *)R_alloc((size_t)rows, sizeof(uint8_t));
    n4m_filter_stats_t stats = {0};
    s = n4m_filter_quality_apply(h, Xv, mask, &stats);
    n4m_filter_quality_destroy(h);
    if (s != N4M_OK) n4m_throw(s, "n4m_filter_quality_apply");

    SEXP mask_R = PROTECT(Rf_allocVector(LGLSXP, (R_xlen_t)rows));
    for (int64_t i = 0; i < rows; ++i) {
        LOGICAL(mask_R)[i] = mask[i] ? TRUE : FALSE;
    }
    UNPROTECT(1);
    return mask_R;
}

/* ------------------------------------------------------------------ */
/* Method registration                                                 */
/* ------------------------------------------------------------------ */

/* The R-side names use a `C_` prefix to match the autogenerated symbol
 * objects produced by `useDynLib(n4m, .registration = TRUE)`
 * (each entry below becomes a `NativeSymbolInfo` named exactly as the
 * first column of the table — we choose the `C_*` convention so that
 * the corresponding R-side `.Call(C_<name>, ...)` reads naturally). */
#define CDEF(name, nargs) {"C_" #name, (DL_FUNC)&r_ ## name, nargs}

static const R_CallMethodDef R_CallEntries[] = {
    CDEF(n4m_abi_version, 0),
    CDEF(n4m_version_string, 0),

    CDEF(n4m_pp_snv_transform, 4),                /* X, with_mean, with_std, ddof */
    CDEF(n4m_pp_lsnv_transform, 4),               /* X, window, pad_mode, constant */
    CDEF(n4m_pp_rnv_transform, 4),                /* X, center, scale, k */
    CDEF(n4m_pp_area_transform, 2),               /* X, method */
    CDEF(n4m_pp_normalize_transform, 3),          /* X, feature_min, feature_max */
    CDEF(n4m_pp_simple_scale_transform, 1),       /* X */
    CDEF(n4m_pp_msc_fit_transform, 2),            /* X_fit, X */
    CDEF(n4m_pp_msc_fit_inverse_transform, 2),    /* X_fit, X */
    CDEF(n4m_pp_emsc_fit_transform, 3),           /* X_fit, X, degree */
    CDEF(n4m_pp_baseline_fit_transform, 2),       /* X_fit, X */
    CDEF(n4m_pp_baseline_fit_inverse_transform, 2), /* X_fit, X */
    CDEF(n4m_pp_savgol_transform, 7),             /* X, w, p, deriv, delta, mode, cval */
    CDEF(n4m_pp_first_derivative_transform, 3),   /* X, delta, edge_order */
    CDEF(n4m_pp_second_derivative_transform, 3),  /* X, delta, edge_order */
    CDEF(n4m_pp_norris_williams_transform, 5),    /* X, segment, gap, order, delta */
    CDEF(n4m_pp_gaussian_transform, 6),           /* X, sigma, order, mode, cval, truncate */
    CDEF(n4m_pp_to_absorbance_transform, 4),      /* X, is_percent, epsilon, clip_neg */
    CDEF(n4m_pp_from_absorbance_transform, 2),    /* X, is_percent */
    CDEF(n4m_pp_pct_to_frac_transform, 1),        /* X */
    CDEF(n4m_pp_frac_to_pct_transform, 1),        /* X */
    CDEF(n4m_pp_kubelka_munk_transform, 3),       /* X, is_percent, epsilon */
    CDEF(n4m_pp_detrend_transform, 2),            /* X, polyorder */
    CDEF(n4m_pp_asls_transform, 5),               /* X, lam, p, max_iter, tol */
    CDEF(n4m_pp_airpls_transform, 4),             /* X, lam, max_iter, tol */
    CDEF(n4m_pp_arpls_transform, 4),              /* X, lam, max_iter, tol */
    CDEF(n4m_pp_modpoly_transform, 4),            /* X, polyorder, max_iter, tol */
    CDEF(n4m_pp_imodpoly_transform, 4),           /* X, polyorder, max_iter, tol */
    CDEF(n4m_pp_snip_transform, 2),               /* X, max_half_window */
    CDEF(n4m_pp_rolling_ball_transform, 3),       /* X, half_window, smooth_half_window */
    CDEF(n4m_pp_iasls_transform, 8),              /* X, lam, p, lam_1, polyorder, diff_order, max_iter, tol */
    CDEF(n4m_pp_beads_transform, 6),              /* X, lam0, lam1, lam2, max_iter, tol */
    CDEF(n4m_pp_wavelet_denoise_transform, 6),    /* X, family, boundary, level, thr, noise */
    CDEF(n4m_pp_wavelet_transform, 3),            /* X, family, boundary */
    CDEF(n4m_pp_haar_transform, 1),               /* X */
    CDEF(n4m_pp_wavelet_features_transform, 5),   /* X, family, boundary, max_level, entropy */
    CDEF(n4m_pp_osc_fit_transform, 4),            /* X, y, n_components, scale */
    CDEF(n4m_pp_epo_fit_transform, 3),            /* X, d, scale */
    CDEF(n4m_pp_epo_fit_inverse_transform, 3),    /* X, d, scale */
    CDEF(n4m_pp_flex_pca_fit_transform, 2),       /* X, n_components */
    CDEF(n4m_pp_flex_svd_fit_transform, 2),       /* X, n_components */
    CDEF(n4m_pp_fck_static_transform, 4),         /* X, kernel_size, orders, scales */
    CDEF(n4m_pp_crop_transform, 3),               /* X, start, end */
    CDEF(n4m_pp_resample_transform, 2),           /* X, num_samples */
    CDEF(n4m_pp_resampler_fit_transform, 10),     /* X, source, target, method, crop, fill, bounds */
    CDEF(n4m_pp_kbins_disc_fit_transform, 3),     /* X, n_bins, strategy */
    CDEF(n4m_pp_range_disc_transform, 2),         /* X, edges */

    CDEF(n4m_split_kennard_stone_split, 2),       /* X, test_size */
    CDEF(n4m_split_spxy_split, 3),                /* X, Y, test_size */
    CDEF(n4m_split_kbins_stratified_split, 5),    /* Y, test_size, seed, n_bins, strategy */

    CDEF(n4m_filter_y_outlier_fit_apply, 5),      /* y, method, threshold, lo, hi */
    CDEF(n4m_filter_x_outlier_fit_apply, 9),      /* X, method, threshold mode, params */

    CDEF(n4m_pp_direct_standardization_fit_transform, 5),
    CDEF(n4m_pp_robust_direct_standardization_fit_transform, 7),
    CDEF(n4m_pp_piecewise_direct_standardization_fit_transform, 6),
    CDEF(n4m_pp_saps_fit_transform, 7),
    CDEF(n4m_pp_slope_bias_fit_transform, 3),
    CDEF(n4m_pp_local_centering_fit_transform, 3),
    CDEF(n4m_pp_weighted_snv_fit_transform, 4),
    CDEF(n4m_pp_vsn_fit_transform, 2),
    CDEF(n4m_pp_piecewise_snv_fit_transform, 4),
    CDEF(n4m_pp_piecewise_msc_fit_transform, 4),
    CDEF(n4m_pp_localized_msc_fit_transform, 4),
    CDEF(n4m_pp_xcorr_align_fit_transform, 4),
    CDEF(n4m_pp_icoshift_align_fit_transform, 4),
    CDEF(n4m_pp_dtw_align_fit_transform, 4),
    CDEF(n4m_pp_cow_align_fit_transform, 4),
    CDEF(n4m_filter_variance_fit_transform, 3),
    CDEF(n4m_filter_correlation_fit_transform, 4),
    CDEF(n4m_interval_generator_fit_transform, 3),
    CDEF(n4m_aug_apply, 5),                       /* name, X, wavelengths, seed, params */
    CDEF(n4m_filter_quality_apply, 9),            /* X, quality thresholds/options */

    {NULL, NULL, 0}
};

/* The R load entrypoint registers C symbols. We use the
 * R_CallMethodDef table and let R generate the C_<name> R-side
 * accessors automatically (because we passed .registration = TRUE in
 * useDynLib). */
attribute_visible void R_init_n4m(DllInfo *dll) {
    R_registerRoutines(dll, NULL, R_CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
    R_forceSymbols(dll, TRUE);
}
