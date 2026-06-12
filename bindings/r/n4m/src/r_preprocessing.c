/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Thin R wrappers around the libn4m preprocessing and splitter C ABI.
 * These functions do only R <-> C shape/layout translation; the numerical
 * kernels remain in libn4m.
 */
#define R_NO_REMAP
#include <R.h>
#include <R_ext/Arith.h>
#include <Rinternals.h>

#include <limits.h>
#include <stdint.h>
#include <string.h>

#include "n4m/n4m.h"

static void r_pp_throw_status(const char* fn, n4m_status_t status) {
    Rf_error("%s failed: %s", fn, n4m_status_to_string(status));
}

static int r_pp_flag(SEXP value, int dflt) {
    if (Rf_length(value) != 1)
        Rf_error("expected a length-1 logical or integer scalar");
    if (TYPEOF(value) == LGLSXP) {
        const int v = LOGICAL(value)[0];
        return v == NA_LOGICAL ? dflt : (v ? 1 : 0);
    }
    if (TYPEOF(value) == INTSXP) {
        const int v = INTEGER(value)[0];
        return v == NA_INTEGER ? dflt : (v ? 1 : 0);
    }
    Rf_error("expected logical or integer scalar");
}

static int r_pp_int_scalar(SEXP value, const char* name) {
    if (Rf_length(value) != 1)
        Rf_error("%s must be a length-1 integer", name);
    if (TYPEOF(value) == INTSXP) {
        const int v = INTEGER(value)[0];
        if (v == NA_INTEGER) Rf_error("%s must not be NA", name);
        return v;
    }
    if (TYPEOF(value) == REALSXP) {
        const double v = REAL(value)[0];
        if (!R_finite(v) || v != (int)v) Rf_error("%s must be an integer", name);
        return (int)v;
    }
    Rf_error("%s must be an integer", name);
}

static double r_pp_double_scalar(SEXP value, const char* name) {
    if (Rf_length(value) != 1)
        Rf_error("%s must be a length-1 numeric scalar", name);
    if (TYPEOF(value) != REALSXP && TYPEOF(value) != INTSXP)
        Rf_error("%s must be numeric", name);
    const double v = Rf_asReal(value);
    if (!R_finite(v)) Rf_error("%s must be finite", name);
    return v;
}

static void r_pp_matrix_shape(SEXP X, int64_t* rows, int64_t* cols) {
    if (TYPEOF(X) != REALSXP) Rf_error("X must be a numeric matrix");
    SEXP dim = Rf_getAttrib(X, R_DimSymbol);
    if (Rf_length(dim) != 2) Rf_error("X must be a 2D matrix");
    *rows = (int64_t)INTEGER(dim)[0];
    *cols = (int64_t)INTEGER(dim)[1];
    if (*rows < 1 || *cols < 1) Rf_error("X must have positive dimensions");
}

static void r_pp_copy_r_to_rowmajor(SEXP X, int64_t rows, int64_t cols, double* out) {
    const double* x = REAL(X);
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            out[i * cols + j] = x[i + j * rows];
        }
    }
}

static SEXP r_pp_rowmajor_to_matrix(const double* data, int64_t rows, int64_t cols) {
    if (rows > INT_MAX || cols > INT_MAX)
        Rf_error("matrix dimensions exceed R's integer matrix limits");
    SEXP out = PROTECT(Rf_allocMatrix(REALSXP, (int)rows, (int)cols));
    double* dst = REAL(out);
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            dst[i + j * rows] = data[i * cols + j];
        }
    }
    UNPROTECT(1);
    return out;
}

static int r_pp_savgol_mode(SEXP value) {
    if (Rf_length(value) != 1)
        Rf_error("mode must be a length-1 character or integer scalar");
    if (TYPEOF(value) == STRSXP) {
        const char* s = CHAR(STRING_ELT(value, 0));
        if (strcmp(s, "mirror") == 0) return N4M_PP_SAVGOL_MIRROR;
        if (strcmp(s, "constant") == 0) return N4M_PP_SAVGOL_CONSTANT;
        if (strcmp(s, "nearest") == 0) return N4M_PP_SAVGOL_NEAREST;
        if (strcmp(s, "wrap") == 0) return N4M_PP_SAVGOL_WRAP;
        if (strcmp(s, "interp") == 0) return N4M_PP_SAVGOL_INTERP;
        Rf_error("unknown Savitzky-Golay mode: %s", s);
    }
    const int mode = r_pp_int_scalar(value, "mode");
    if (mode < N4M_PP_SAVGOL_MIRROR || mode > N4M_PP_SAVGOL_INTERP)
        Rf_error("mode integer must be in 0..4");
    return mode;
}

SEXP r_n4m_snv_transform(SEXP X, SEXP with_mean, SEXP with_std, SEXP ddof) {
    int64_t rows = 0;
    int64_t cols = 0;
    r_pp_matrix_shape(X, &rows, &cols);

    SEXP X_rm = PROTECT(Rf_allocVector(REALSXP, (R_xlen_t)(rows * cols)));
    SEXP out_rm = PROTECT(Rf_allocVector(REALSXP, (R_xlen_t)(rows * cols)));
    r_pp_copy_r_to_rowmajor(X, rows, cols, REAL(X_rm));

    n4m_pp_snv_handle_t* handle = NULL;
    n4m_status_t status = n4m_pp_snv_create(
        &handle,
        r_pp_flag(with_mean, 1),
        r_pp_flag(with_std, 1),
        r_pp_int_scalar(ddof, "ddof"));
    if (status != N4M_OK) {
        UNPROTECT(2);
        r_pp_throw_status("n4m_pp_snv_create", status);
    }

    n4m_matrix_view_t X_view;
    n4m_matrix_view_t out_view;
    n4m_matrix_view_init_rowmajor(&X_view, REAL(X_rm), rows, cols, N4M_DTYPE_F64);
    n4m_matrix_view_init_rowmajor(&out_view, REAL(out_rm), rows, cols, N4M_DTYPE_F64);
    status = n4m_pp_snv_transform(handle, X_view, out_view);
    n4m_pp_snv_destroy(handle);
    if (status != N4M_OK) {
        UNPROTECT(2);
        r_pp_throw_status("n4m_pp_snv_transform", status);
    }

    SEXP out = PROTECT(r_pp_rowmajor_to_matrix(REAL(out_rm), rows, cols));
    UNPROTECT(3);
    return out;
}

SEXP r_n4m_savgol_transform(SEXP X, SEXP window_length, SEXP polyorder,
                            SEXP deriv, SEXP delta, SEXP mode, SEXP cval) {
    int64_t rows = 0;
    int64_t cols = 0;
    r_pp_matrix_shape(X, &rows, &cols);

    SEXP X_rm = PROTECT(Rf_allocVector(REALSXP, (R_xlen_t)(rows * cols)));
    SEXP out_rm = PROTECT(Rf_allocVector(REALSXP, (R_xlen_t)(rows * cols)));
    r_pp_copy_r_to_rowmajor(X, rows, cols, REAL(X_rm));

    n4m_pp_savgol_handle_t* handle = NULL;
    n4m_status_t status = n4m_pp_savgol_create(
        &handle,
        (int32_t)r_pp_int_scalar(window_length, "window_length"),
        (int32_t)r_pp_int_scalar(polyorder, "polyorder"),
        (int32_t)r_pp_int_scalar(deriv, "deriv"),
        r_pp_double_scalar(delta, "delta"),
        (n4m_pp_savgol_mode_t)r_pp_savgol_mode(mode),
        r_pp_double_scalar(cval, "cval"));
    if (status != N4M_OK) {
        UNPROTECT(2);
        r_pp_throw_status("n4m_pp_savgol_create", status);
    }

    n4m_matrix_view_t X_view;
    n4m_matrix_view_t out_view;
    n4m_matrix_view_init_rowmajor(&X_view, REAL(X_rm), rows, cols, N4M_DTYPE_F64);
    n4m_matrix_view_init_rowmajor(&out_view, REAL(out_rm), rows, cols, N4M_DTYPE_F64);
    status = n4m_pp_savgol_transform(handle, X_view, out_view);
    n4m_pp_savgol_destroy(handle);
    if (status != N4M_OK) {
        UNPROTECT(2);
        r_pp_throw_status("n4m_pp_savgol_transform", status);
    }

    SEXP out = PROTECT(r_pp_rowmajor_to_matrix(REAL(out_rm), rows, cols));
    UNPROTECT(3);
    return out;
}

SEXP r_n4m_kennard_stone_split(SEXP X, SEXP test_size, SEXP zero_based) {
    int64_t rows = 0;
    int64_t cols = 0;
    r_pp_matrix_shape(X, &rows, &cols);
    const int offset = r_pp_flag(zero_based, 0) ? 0 : 1;

    SEXP X_rm = PROTECT(Rf_allocVector(REALSXP, (R_xlen_t)(rows * cols)));
    r_pp_copy_r_to_rowmajor(X, rows, cols, REAL(X_rm));

    n4m_split_kennard_stone_handle_t* handle = NULL;
    n4m_status_t status = n4m_split_kennard_stone_create(
        &handle, r_pp_double_scalar(test_size, "test_size"));
    if (status != N4M_OK) {
        UNPROTECT(1);
        r_pp_throw_status("n4m_split_kennard_stone_create", status);
    }

    n4m_matrix_view_t X_view;
    n4m_matrix_view_init_rowmajor(&X_view, REAL(X_rm), rows, cols, N4M_DTYPE_F64);
    n4m_split_result_t result = {0};
    status = n4m_split_kennard_stone_split(handle, X_view, &result);
    n4m_split_kennard_stone_destroy(handle);
    if (status != N4M_OK) {
        n4m_split_result_destroy(&result);
        UNPROTECT(1);
        r_pp_throw_status("n4m_split_kennard_stone_split", status);
    }
    if (result.n_train > INT_MAX || result.n_test > INT_MAX) {
        n4m_split_result_destroy(&result);
        UNPROTECT(1);
        Rf_error("split size exceeds R vector length limits");
    }

    for (int64_t i = 0; i < result.n_train; ++i) {
        if (result.train_idx[i] > INT_MAX - offset) {
            n4m_split_result_destroy(&result);
            UNPROTECT(1);
            Rf_error("train index exceeds R integer limits");
        }
    }
    for (int64_t i = 0; i < result.n_test; ++i) {
        if (result.test_idx[i] > INT_MAX - offset) {
            n4m_split_result_destroy(&result);
            UNPROTECT(1);
            Rf_error("test index exceeds R integer limits");
        }
    }

    SEXP train = PROTECT(Rf_allocVector(INTSXP, (R_xlen_t)result.n_train));
    SEXP test = PROTECT(Rf_allocVector(INTSXP, (R_xlen_t)result.n_test));
    for (int64_t i = 0; i < result.n_train; ++i) {
        INTEGER(train)[i] = (int)result.train_idx[i] + offset;
    }
    for (int64_t i = 0; i < result.n_test; ++i) {
        INTEGER(test)[i] = (int)result.test_idx[i] + offset;
    }
    n4m_split_result_destroy(&result);

    SEXP out = PROTECT(Rf_allocVector(VECSXP, 2));
    SEXP names = PROTECT(Rf_allocVector(STRSXP, 2));
    SET_VECTOR_ELT(out, 0, train);
    SET_VECTOR_ELT(out, 1, test);
    SET_STRING_ELT(names, 0, Rf_mkChar("train"));
    SET_STRING_ELT(names, 1, Rf_mkChar("test"));
    Rf_setAttrib(out, R_NamesSymbol, names);

    UNPROTECT(5);
    return out;
}
