/* SPDX-License-Identifier: CECILL-2.1 */
/* Generic R-to-C ABI bridge for fitted preprocessing pipelines. */
#define R_NO_REMAP
#include <R.h>
#include <R_ext/Arith.h>
#include <Rinternals.h>

#include <stdint.h>
#include <stdio.h>

#include "n4m/n4m.h"

static void r_pipeline_finalize(SEXP ptr) {
    if (TYPEOF(ptr) != EXTPTRSXP) return;
    n4m_pipeline_t* pipeline = (n4m_pipeline_t*)R_ExternalPtrAddr(ptr);
    if (pipeline != NULL) n4m_pipeline_destroy(pipeline);
    R_ClearExternalPtr(ptr);
}

static void r_pipeline_matrix(SEXP value, const char* name,
                              int64_t* rows, int64_t* cols) {
    if (TYPEOF(value) != REALSXP) Rf_error("%s must be a double matrix", name);
    SEXP dim = Rf_getAttrib(value, R_DimSymbol);
    if (TYPEOF(dim) != INTSXP || Rf_length(dim) != 2)
        Rf_error("%s must be a matrix", name);
    *rows = INTEGER(dim)[0];
    *cols = INTEGER(dim)[1];
    if (*rows < 1 || *cols < 1) Rf_error("%s must be nonempty", name);
}

SEXP r_n4m_preprocess_fit(SEXP X, SEXP Y, SEXP kinds, SEXP params) {
    int64_t rows = 0, cols = 0;
    r_pipeline_matrix(X, "X", &rows, &cols);
    n4m_matrix_view_t x_view;
    n4m_status_t status = n4m_matrix_view_init_colmajor(
        &x_view, REAL(X), rows, cols, N4M_DTYPE_F64);
    if (status != N4M_OK) Rf_error("X view: %s", n4m_status_to_string(status));
    n4m_matrix_view_t y_view;
    const n4m_matrix_view_t* y_ptr = NULL;
    if (Y != R_NilValue) {
        int64_t y_rows = 0, y_cols = 0;
        r_pipeline_matrix(Y, "Y", &y_rows, &y_cols);
        if (y_rows != rows) Rf_error("Y must have one row per X row");
        status = n4m_matrix_view_init_colmajor(
            &y_view, REAL(Y), y_rows, y_cols, N4M_DTYPE_F64);
        if (status != N4M_OK) Rf_error("Y view: %s", n4m_status_to_string(status));
        y_ptr = &y_view;
    }
    if (TYPEOF(kinds) != INTSXP || TYPEOF(params) != VECSXP ||
        Rf_length(kinds) < 1 || Rf_length(kinds) != Rf_length(params))
        Rf_error("kinds and params must be aligned nonempty vectors");
    for (R_xlen_t i = 0; i < XLENGTH(kinds); ++i) {
        const int kind = INTEGER(kinds)[i];
        SEXP values = VECTOR_ELT(params, i);
        if (kind < N4M_OP_IDENTITY || kind > N4M_OP_WAVELET_DENOISE ||
            TYPEOF(values) != REALSXP || XLENGTH(values) > INT32_MAX)
            Rf_error("unsupported preprocessing kind or invalid params");
        for (R_xlen_t j = 0; j < XLENGTH(values); ++j) {
            if (!R_finite(REAL(values)[j])) Rf_error("params must be finite");
        }
    }
    n4m_pipeline_t* pipeline = NULL;
    status = n4m_pipeline_create(&pipeline);
    if (status != N4M_OK) Rf_error("pipeline create: %s", n4m_status_to_string(status));
    for (R_xlen_t i = 0; i < XLENGTH(kinds); ++i) {
        SEXP values = VECTOR_ELT(params, i);
        status = n4m_pipeline_add_operator(
            pipeline, (n4m_operator_kind_t)INTEGER(kinds)[i],
            XLENGTH(values) == 0 ? NULL : REAL(values), (int32_t)XLENGTH(values));
        if (status != N4M_OK) {
            n4m_pipeline_destroy(pipeline);
            Rf_error("pipeline add operator: %s", n4m_status_to_string(status));
        }
    }
    n4m_context_t* context = NULL;
    status = n4m_context_create(&context);
    if (status != N4M_OK) {
        n4m_pipeline_destroy(pipeline);
        Rf_error("context create: %s", n4m_status_to_string(status));
    }
    status = n4m_pipeline_fit(context, pipeline, &x_view, y_ptr);
    char fit_error[512] = {0};
    if (status != N4M_OK) {
        const char* detail = n4m_context_last_error(context);
        snprintf(fit_error, sizeof(fit_error), "%s", detail == NULL ? "" : detail);
    }
    n4m_context_destroy(context);
    if (status != N4M_OK) {
        n4m_pipeline_destroy(pipeline);
        Rf_error("pipeline fit: %s: %s", n4m_status_to_string(status), fit_error);
    }
    SEXP ptr = PROTECT(R_MakeExternalPtr(pipeline, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ptr, r_pipeline_finalize, TRUE);
    UNPROTECT(1);
    return ptr;
}

SEXP r_n4m_preprocess_transform(SEXP ptr, SEXP X) {
    if (TYPEOF(ptr) != EXTPTRSXP || R_ExternalPtrAddr(ptr) == NULL)
        Rf_error("pipeline state must be a live native external pointer");
    int64_t rows = 0, cols = 0;
    r_pipeline_matrix(X, "X", &rows, &cols);
    n4m_matrix_view_t x_view;
    n4m_status_t status = n4m_matrix_view_init_colmajor(
        &x_view, REAL(X), rows, cols, N4M_DTYPE_F64);
    if (status != N4M_OK) Rf_error("X view: %s", n4m_status_to_string(status));
    SEXP output = PROTECT(Rf_allocMatrix(REALSXP, (int)rows, (int)cols));
    n4m_matrix_view_t output_view;
    status = n4m_matrix_view_init_colmajor(
        &output_view, REAL(output), rows, cols, N4M_DTYPE_F64);
    if (status != N4M_OK) {
        UNPROTECT(1);
        Rf_error("output view: %s", n4m_status_to_string(status));
    }
    n4m_context_t* context = NULL;
    status = n4m_context_create(&context);
    if (status != N4M_OK) {
        UNPROTECT(1);
        Rf_error("context create: %s", n4m_status_to_string(status));
    }
    status = n4m_pipeline_transform(
        context, (const n4m_pipeline_t*)R_ExternalPtrAddr(ptr), &x_view, &output_view);
    char transform_error[512] = {0};
    if (status != N4M_OK) {
        const char* detail = n4m_context_last_error(context);
        snprintf(transform_error, sizeof(transform_error), "%s", detail == NULL ? "" : detail);
    }
    n4m_context_destroy(context);
    if (status != N4M_OK) {
        UNPROTECT(1);
        Rf_error("pipeline transform: %s: %s", n4m_status_to_string(status), transform_error);
    }
    Rf_setAttrib(output, R_DimNamesSymbol, Rf_getAttrib(X, R_DimNamesSymbol));
    UNPROTECT(1);
    return output;
}
