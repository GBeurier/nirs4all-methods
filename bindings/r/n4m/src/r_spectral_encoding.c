/* SPDX-License-Identifier: CECILL-2.1 */
#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>

#include "n4m/n4m.h"

static void spectral_check(n4m_status_t status) {
    if (status != N4M_OK)
        Rf_error("spectral encoder: %s", n4m_status_to_string(status));
}
static n4m_matrix_view_t spectral_view(SEXP x) {
    if (!Rf_isMatrix(x) || TYPEOF(x) != REALSXP)
        Rf_error("X must be a double matrix");
    SEXP dims = Rf_getAttrib(x, R_DimSymbol);
    n4m_matrix_view_t v = {0};
    v.data = REAL(x);
    v.rows = INTEGER(dims)[0];
    v.cols = INTEGER(dims)[1];
    v.row_stride = 1;
    v.col_stride = v.rows;
    v.dtype = N4M_DTYPE_F64;
    return v;
}
static void spectral_finalizer(SEXP ptr) {
    n4m_decomposition_spectral_destroy((n4m_spectral_encoder_t*)R_ExternalPtrAddr(ptr));
    R_ClearExternalPtr(ptr);
}
static n4m_spectral_encoder_t* spectral_handle(SEXP ptr) {
    if (TYPEOF(ptr) != EXTPTRSXP || R_ExternalPtrTag(ptr) != Rf_install("n4m_spectral"))
        Rf_error("expected a fitted n4m spectral encoder");
    n4m_spectral_encoder_t* h = (n4m_spectral_encoder_t*)R_ExternalPtrAddr(ptr);
    if (!h)
        Rf_error("spectral encoder handle is closed");
    return h;
}
SEXP r_n4m_spectral_fit(SEXP x, SEXP parameters) {
    n4m_matrix_view_t xv = spectral_view(x);
    if (TYPEOF(parameters) != REALSXP || Rf_length(parameters) != 8)
        Rf_error("expected eight numeric encoder parameters");
    const double* p = REAL(parameters);
    for (int i = 0; i < 8; ++i)
        if (!R_finite(p[i]))
            Rf_error("encoder parameters must be finite");
    const int integer_positions[] = {0, 1, 2, 4, 5, 6};
    for (int i = 0; i < 6; ++i) {
        double v = p[integer_positions[i]];
        if (v < 0 || v > 1000000 || v != (int)v)
            Rf_error("invalid integer encoder parameter");
    }
    SEXP ptr = PROTECT(R_MakeExternalPtr(NULL, Rf_install("n4m_spectral"), R_NilValue));
    n4m_spectral_encoder_t* h = NULL;
    spectral_check(n4m_decomposition_spectral_create(
        (int)p[0], (int)p[1], (int)p[2], p[3], (int)p[4], (int)p[5], (int)p[6], p[7], &h));
    R_SetExternalPtrAddr(ptr, h);
    R_RegisterCFinalizerEx(ptr, spectral_finalizer, TRUE);
    spectral_check(n4m_decomposition_spectral_fit(h, xv));
    SEXP features = PROTECT(Rf_ScalarInteger((int)xv.cols));
    Rf_setAttrib(ptr, Rf_install("n_features_in"), features);
    UNPROTECT(2);
    return ptr;
}
SEXP r_n4m_spectral_transform(SEXP ptr, SEXP x) {
    n4m_spectral_encoder_t* h = spectral_handle(ptr);
    n4m_matrix_view_t xv = spectral_view(x);
    int64_t q = 0;
    spectral_check(n4m_decomposition_spectral_output_cols(h, &q));
    if (q > 2147483647)
        Rf_error("encoded width exceeds R matrix dimensions");
    SEXP out = PROTECT(Rf_allocMatrix(REALSXP, (int)xv.rows, (int)q));
    spectral_check(n4m_decomposition_spectral_transform(h, xv, spectral_view(out)));
    UNPROTECT(1);
    return out;
}
SEXP r_n4m_spectral_affine(SEXP ptr) {
    n4m_spectral_encoder_t* h = spectral_handle(ptr);
    int64_t q = 0;
    spectral_check(n4m_decomposition_spectral_output_cols(h, &q));
    int p = Rf_asInteger(Rf_getAttrib(ptr, Rf_install("n_features_in")));
    if (q > 2147483647)
        Rf_error("encoded width exceeds R matrix dimensions");
    SEXP op = PROTECT(Rf_allocMatrix(REALSXP, (int)q, p));
    SEXP offset = PROTECT(Rf_allocMatrix(REALSXP, 1, (int)q));
    spectral_check(
        n4m_decomposition_spectral_export_affine(h, spectral_view(op), spectral_view(offset)));
    SEXP out = PROTECT(Rf_allocVector(VECSXP, 2));
    SET_VECTOR_ELT(out, 0, op);
    SET_VECTOR_ELT(out, 1, offset);
    SEXP names = PROTECT(Rf_allocVector(STRSXP, 2));
    SET_STRING_ELT(names, 0, Rf_mkChar("operator"));
    SET_STRING_ELT(names, 1, Rf_mkChar("offset"));
    Rf_setAttrib(out, R_NamesSymbol, names);
    UNPROTECT(4);
    return out;
}
