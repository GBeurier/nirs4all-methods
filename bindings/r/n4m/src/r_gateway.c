/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * R .Call gateway over the n4m libn4m C ABI.
 *
 * The native shared library libn4m must be locatable at load time. R loads
 * n4m.so first; its dependencies (libn4m) are then resolved by the
 * runtime linker. Set the LD_LIBRARY_PATH / DYLD_LIBRARY_PATH / PATH
 * appropriately, or run `R CMD INSTALL --configure-args=...` to embed a
 * specific rpath in the Makevars.
 */

#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "n4m/n4m.h"

/* ---- helpers ---------------------------------------------------------- */

static SEXP r_make_string(const char* s) {
    if (s == NULL) s = "";
    return Rf_mkString(s);
}

/* Copy the per-context error message, destroy ctx, then longjmp via
 * Rf_error. Doing the destroy BEFORE Rf_error prevents leaking the ctx
 * on the longjmp (longjmp skips any cleanup below the throw site).
 * Callers must have already freed any other C resources they owned. */
static void r_throw_status(const char* fn, n4m_status_t status, n4m_context_t* ctx) {
    const char* status_str = n4m_status_to_string(status);
    char buf[4096];
    buf[0] = '\0';
    if (ctx != NULL) {
        const char* msg = n4m_context_last_error(ctx);
        if (msg != NULL && msg[0] != '\0') {
            size_t n = strlen(msg);
            if (n >= sizeof(buf)) n = sizeof(buf) - 1;
            memcpy(buf, msg, n);
            buf[n] = '\0';
        }
        n4m_context_destroy(ctx);
    }
    if (buf[0] != '\0') {
        Rf_error("%s failed: %s (%s)", fn, status_str, buf);
    } else {
        Rf_error("%s failed: %s", fn, status_str);
    }
}

static int resolve_algo(const char* name,
                        n4m_algorithm_t* out_algo,
                        n4m_solver_t* out_solver,
                        n4m_deflation_t* out_deflation) {
    *out_deflation = N4M_DEFLATION_REGRESSION;
    if (strcmp(name, "pls_nipals") == 0) {
        *out_algo = N4M_ALGO_PLS_REGRESSION;
        *out_solver = N4M_SOLVER_NIPALS;
        return 1;
    }
    if (strcmp(name, "pls_orthogonal_scores") == 0) {
        *out_algo = N4M_ALGO_PLS_REGRESSION;
        *out_solver = N4M_SOLVER_ORTHOGONAL_SCORES;
        return 1;
    }
    if (strcmp(name, "pls_simpls") == 0) {
        *out_algo = N4M_ALGO_PLS_REGRESSION;
        *out_solver = N4M_SOLVER_SIMPLS;
        return 1;
    }
    if (strcmp(name, "pls_kernel_algorithm") == 0) {
        *out_algo = N4M_ALGO_PLS_REGRESSION;
        *out_solver = N4M_SOLVER_KERNEL_ALGORITHM;
        return 1;
    }
    if (strcmp(name, "pls_wide_kernel") == 0) {
        *out_algo = N4M_ALGO_PLS_REGRESSION;
        *out_solver = N4M_SOLVER_WIDE_KERNEL;
        return 1;
    }
    if (strcmp(name, "pls_svd") == 0) {
        *out_algo = N4M_ALGO_PLS_REGRESSION;
        *out_solver = N4M_SOLVER_SVD;
        return 1;
    }
    if (strcmp(name, "pls_power") == 0) {
        *out_algo = N4M_ALGO_PLS_REGRESSION;
        *out_solver = N4M_SOLVER_POWER;
        return 1;
    }
    if (strcmp(name, "pls_randomized_svd") == 0) {
        *out_algo = N4M_ALGO_PLS_REGRESSION;
        *out_solver = N4M_SOLVER_RANDOMIZED_SVD;
        return 1;
    }
    if (strcmp(name, "pcr_svd") == 0 || strcmp(name, "pcr") == 0) {
        *out_algo = N4M_ALGO_PCR;
        *out_solver = N4M_SOLVER_SVD;
        return 1;
    }
    if (strcmp(name, "opls") == 0 || strcmp(name, "opls_nipals") == 0) {
        *out_algo = N4M_ALGO_OPLS;
        *out_solver = N4M_SOLVER_NIPALS;
        *out_deflation = N4M_DEFLATION_ORTHOGONAL;
        return 1;
    }
    return 0;
}

/* ---- external pointer finalizers ------------------------------------- */

static void r_model_finalize(SEXP ptr) {
    if (TYPEOF(ptr) != EXTPTRSXP) return;
    n4m_model_t* model = (n4m_model_t*)R_ExternalPtrAddr(ptr);
    if (model != NULL) {
        n4m_model_destroy(model);
        R_ClearExternalPtr(ptr);
    }
}

/* ---- version queries -------------------------------------------------- */

SEXP r_n4m_version(void) {
    return r_make_string(n4m_get_version_string());
}

SEXP r_n4m_abi_version(void) {
    SEXP out = PROTECT(Rf_allocVector(INTSXP, 3));
    INTEGER(out)[0] = (int)n4m_get_abi_version_major();
    INTEGER(out)[1] = (int)n4m_get_abi_version_minor();
    INTEGER(out)[2] = (int)n4m_get_abi_version_patch();
    UNPROTECT(1);
    return out;
}

/* ---- fit -------------------------------------------------------------- */

/* Convert a length-1 LGLSXP/INTSXP into a {0,1} flag, treating NA as
 * the provided default. Errors on length != 1 (NULL / empty vectors)
 * so a typo in the R wrapper surfaces at the boundary, not as a wild
 * read of LOGICAL(NULL)[0]. */
static int sexp_flag(SEXP s, int dflt) {
    if (Rf_length(s) != 1)
        Rf_error("expected a length-1 logical or integer scalar");
    if (TYPEOF(s) == LGLSXP) {
        int v = LOGICAL(s)[0];
        return (v == NA_LOGICAL) ? dflt : (v ? 1 : 0);
    }
    if (TYPEOF(s) == INTSXP) {
        int v = INTEGER(s)[0];
        return (v == NA_INTEGER) ? dflt : (v ? 1 : 0);
    }
    Rf_error("expected logical or integer scalar");
}

SEXP r_n4m_fit(SEXP X, SEXP Y, SEXP algo_sexp, SEXP n_components_sexp,
                    SEXP store_scores_sexp,
                    SEXP center_x_sexp, SEXP scale_x_sexp,
                    SEXP center_y_sexp, SEXP scale_y_sexp,
                    SEXP embedded_snv_savgol_sexp) {
    if (TYPEOF(X) != REALSXP) Rf_error("X must be a numeric matrix");
    if (TYPEOF(Y) != REALSXP) Rf_error("Y must be a numeric matrix");
    if (TYPEOF(algo_sexp) != STRSXP) Rf_error("algo must be character");
    if (TYPEOF(n_components_sexp) != INTSXP) Rf_error("n_components must be integer");
    if (embedded_snv_savgol_sexp != R_NilValue &&
        (TYPEOF(embedded_snv_savgol_sexp) != REALSXP ||
         XLENGTH(embedded_snv_savgol_sexp) != 2))
        Rf_error("embedded_snv_savgol must be NULL or two double values");
    const int store_scores = sexp_flag(store_scores_sexp, 0);
    const int center_x = sexp_flag(center_x_sexp, 1);
    const int scale_x  = sexp_flag(scale_x_sexp,  1);
    const int center_y = sexp_flag(center_y_sexp, 1);
    const int scale_y  = sexp_flag(scale_y_sexp,  1);

    SEXP X_dim = Rf_getAttrib(X, R_DimSymbol);
    SEXP Y_dim = Rf_getAttrib(Y, R_DimSymbol);
    if (Rf_length(X_dim) != 2) Rf_error("X must be a 2D matrix");
    if (Rf_length(Y_dim) != 2) Rf_error("Y must be a 2D matrix");

    const int64_t n_rows = INTEGER(X_dim)[0];
    const int64_t n_cols = INTEGER(X_dim)[1];
    const int64_t y_rows = INTEGER(Y_dim)[0];
    const int64_t y_cols = INTEGER(Y_dim)[1];
    if (n_rows != y_rows) Rf_error("nrow(X) must equal nrow(Y)");

    const char* algo_name = CHAR(STRING_ELT(algo_sexp, 0));
    n4m_algorithm_t algo;
    n4m_solver_t solver;
    n4m_deflation_t deflation;
    if (!resolve_algo(algo_name, &algo, &solver, &deflation)) {
        Rf_error("unknown algo: %s", algo_name);
    }
    const int n_components = INTEGER(n_components_sexp)[0];
    if (n_components < 1) Rf_error("n_components must be >= 1");

    /* R matrices are column-major; we materialize a row-major copy so the
     * matrix view we hand the C ABI is straightforward. For benchmark
     * volumes this is a single copy on entry. */
    SEXP X_rm = PROTECT(Rf_allocMatrix(REALSXP, (int)n_rows, (int)n_cols));
    SEXP Y_rm = PROTECT(Rf_allocMatrix(REALSXP, (int)n_rows, (int)y_cols));
    double* xrm = REAL(X_rm);
    const double* x = REAL(X);
    for (int64_t i = 0; i < n_rows; ++i) {
        for (int64_t j = 0; j < n_cols; ++j) {
            xrm[i * n_cols + j] = x[i + j * n_rows];
        }
    }
    double* yrm = REAL(Y_rm);
    const double* y = REAL(Y);
    for (int64_t i = 0; i < n_rows; ++i) {
        for (int64_t j = 0; j < y_cols; ++j) {
            yrm[i * y_cols + j] = y[i + j * n_rows];
        }
    }

    n4m_context_t* ctx = NULL;
    n4m_status_t status = n4m_context_create(&ctx);
    if (status != N4M_OK) r_throw_status("n4m_context_create", status, NULL);

    n4m_config_t* cfg = NULL;
    status = n4m_config_create(&cfg);
    if (status != N4M_OK) {
        n4m_context_destroy(ctx);
        r_throw_status("n4m_config_create", status, NULL);
    }
    n4m_config_set_algorithm(cfg, algo);
    n4m_config_set_solver(cfg, solver);
    n4m_config_set_deflation(cfg, deflation);
    n4m_config_set_n_components(cfg, n_components);
    /* Match the public C ABI Config defaults (scale_x = scale_y = 1)
     * so the R binding produces byte-equivalent results to the Python
     * binding under default settings. */
    n4m_config_set_center_x(cfg, center_x);
    n4m_config_set_scale_x(cfg, scale_x);
    n4m_config_set_center_y(cfg, center_y);
    n4m_config_set_scale_y(cfg, scale_y);
    n4m_config_set_store_scores(cfg, store_scores ? 1 : 0);

    /* The native model owns a snapshot of this fitted preprocessing chain.
     * Keep the non-owning config hook alive until n4m_model_fit returns. */
    n4m_pipeline_t* embedded = NULL;
    if (embedded_snv_savgol_sexp != R_NilValue) {
        status = n4m_pipeline_create(&embedded);
        if (status == N4M_OK)
            status = n4m_pipeline_add_operator(embedded, N4M_OP_SNV, NULL, 0);
        if (status == N4M_OK)
            status = n4m_pipeline_add_operator(embedded, N4M_OP_SAVGOL_SMOOTH,
                                               REAL(embedded_snv_savgol_sexp), 2);
        if (status == N4M_OK)
            status = n4m_config_set_pipeline(cfg, embedded);
        if (status != N4M_OK) {
            if (embedded != NULL) n4m_pipeline_destroy(embedded);
            n4m_config_destroy(cfg);
            UNPROTECT(2);
            r_throw_status("n4m embedded pipeline setup", status, ctx);
        }
    }

    n4m_matrix_view_t X_view;
    n4m_matrix_view_t Y_view;
    n4m_matrix_view_init_rowmajor(&X_view, xrm, n_rows, n_cols, N4M_DTYPE_F64);
    n4m_matrix_view_init_rowmajor(&Y_view, yrm, n_rows, y_cols, N4M_DTYPE_F64);

    n4m_model_t* model = NULL;
    status = n4m_model_fit(ctx, cfg, &X_view, &Y_view, &model);
    if (embedded != NULL) n4m_pipeline_destroy(embedded);
    n4m_config_destroy(cfg);
    if (status != N4M_OK) {
        UNPROTECT(2);
        /* r_throw_status copies the last_error message then destroys ctx. */
        r_throw_status("n4m_model_fit", status, ctx);
    }
    n4m_context_destroy(ctx);
    UNPROTECT(2);

    SEXP ptr = PROTECT(R_MakeExternalPtr(model, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ptr, r_model_finalize, TRUE);
    /* attach feature/target counts as attributes for downstream predict. */
    SEXP n_features_attr = PROTECT(Rf_allocVector(INTSXP, 1));
    SEXP n_targets_attr = PROTECT(Rf_allocVector(INTSXP, 1));
    int32_t nf = 0;
    int32_t nt = 0;
    n4m_model_get_n_features(model, &nf);
    n4m_model_get_n_targets(model, &nt);
    INTEGER(n_features_attr)[0] = (int)nf;
    INTEGER(n_targets_attr)[0] = (int)nt;
    Rf_setAttrib(ptr, Rf_install("n_features"), n_features_attr);
    Rf_setAttrib(ptr, Rf_install("n_targets"), n_targets_attr);
    UNPROTECT(3);
    return ptr;
}

/* ---- predict ---------------------------------------------------------- */

SEXP r_n4m_predict(SEXP model_ptr, SEXP X) {
    if (TYPEOF(model_ptr) != EXTPTRSXP) Rf_error("model must be an external pointer");
    if (TYPEOF(X) != REALSXP) Rf_error("X must be a numeric matrix");
    n4m_model_t* model = (n4m_model_t*)R_ExternalPtrAddr(model_ptr);
    if (model == NULL) Rf_error("model handle is NULL (already freed?)");

    SEXP X_dim = Rf_getAttrib(X, R_DimSymbol);
    if (Rf_length(X_dim) != 2) Rf_error("X must be a 2D matrix");
    const int64_t n_rows = INTEGER(X_dim)[0];
    const int64_t n_cols = INTEGER(X_dim)[1];

    SEXP X_rm = PROTECT(Rf_allocMatrix(REALSXP, (int)n_rows, (int)n_cols));
    double* xrm = REAL(X_rm);
    const double* x = REAL(X);
    for (int64_t i = 0; i < n_rows; ++i) {
        for (int64_t j = 0; j < n_cols; ++j) {
            xrm[i * n_cols + j] = x[i + j * n_rows];
        }
    }

    int32_t n_targets = 0;
    n4m_model_get_n_targets(model, &n_targets);
    SEXP out_rm = PROTECT(Rf_allocVector(REALSXP, (R_xlen_t)(n_rows * n_targets)));
    double* outrm = REAL(out_rm);

    n4m_context_t* ctx = NULL;
    n4m_status_t status = n4m_context_create(&ctx);
    if (status != N4M_OK) {
        UNPROTECT(2);
        r_throw_status("n4m_context_create", status, NULL);
    }
    n4m_matrix_view_t X_view;
    n4m_matrix_view_init_rowmajor(&X_view, xrm, n_rows, n_cols, N4M_DTYPE_F64);
    n4m_matrix_view_t out_view;
    n4m_matrix_view_init_rowmajor(&out_view, outrm, n_rows, n_targets, N4M_DTYPE_F64);
    status = n4m_model_predict(ctx, model, &X_view, &out_view);
    if (status != N4M_OK) {
        UNPROTECT(2);
        /* r_throw_status owns ctx destruction. */
        r_throw_status("n4m_model_predict", status, ctx);
    }
    n4m_context_destroy(ctx);

    /* Convert row-major predictions into a column-major R matrix. */
    SEXP out = PROTECT(Rf_allocMatrix(REALSXP, (int)n_rows, (int)n_targets));
    double* outdat = REAL(out);
    for (int64_t i = 0; i < n_rows; ++i) {
        for (int64_t j = 0; j < n_targets; ++j) {
            outdat[i + j * n_rows] = outrm[i * n_targets + j];
        }
    }
    UNPROTECT(3);
    return out;
}

/* N4MM is the portable fitted-model format shared by all libn4m bindings.
 * R's external pointers cannot be persisted with saveRDS(), so expose the
 * bytes explicitly and leave file I/O to the R host. */
SEXP r_n4m_model_export(SEXP model_ptr) {
    if (TYPEOF(model_ptr) != EXTPTRSXP) Rf_error("model must be an external pointer");
    const n4m_model_t* model = (const n4m_model_t*)R_ExternalPtrAddr(model_ptr);
    if (model == NULL) Rf_error("model handle is NULL (already freed?)");
    size_t size = 0;
    n4m_status_t status = n4m_model_export_size(model, &size);
    if (status != N4M_OK) r_throw_status("n4m_model_export_size", status, NULL);
    if (size == 0 || size > (size_t)R_XLEN_T_MAX) Rf_error("N4MM model size is invalid for R");
    SEXP bytes = PROTECT(Rf_allocVector(RAWSXP, (R_xlen_t)size));
    size_t written = 0;
    status = n4m_model_export_to_buffer(model, RAW(bytes), size, &written);
    if (status != N4M_OK || written != size) {
        UNPROTECT(1);
        if (status != N4M_OK) r_throw_status("n4m_model_export_to_buffer", status, NULL);
        Rf_error("n4m_model_export_to_buffer wrote an unexpected byte count");
    }
    UNPROTECT(1);
    return bytes;
}

SEXP r_n4m_model_import(SEXP bytes) {
    if (TYPEOF(bytes) != RAWSXP || XLENGTH(bytes) == 0)
        Rf_error("bytes must be a non-empty raw N4MM vector");
    n4m_context_t* ctx = NULL;
    n4m_status_t status = n4m_context_create(&ctx);
    if (status != N4M_OK) r_throw_status("n4m_context_create", status, NULL);
    n4m_model_t* model = NULL;
    status = n4m_model_import_from_buffer(ctx, RAW(bytes), (size_t)XLENGTH(bytes), &model);
    if (status != N4M_OK) r_throw_status("n4m_model_import_from_buffer", status, ctx);
    n4m_context_destroy(ctx);
    SEXP ptr = PROTECT(R_MakeExternalPtr(model, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ptr, r_model_finalize, TRUE);
    int32_t nf = 0, nt = 0;
    n4m_model_get_n_features(model, &nf);
    n4m_model_get_n_targets(model, &nt);
    SEXP n_features_attr = PROTECT(Rf_ScalarInteger((int)nf));
    SEXP n_targets_attr = PROTECT(Rf_ScalarInteger((int)nt));
    Rf_setAttrib(ptr, Rf_install("n_features"), n_features_attr);
    Rf_setAttrib(ptr, Rf_install("n_targets"), n_targets_attr);
    UNPROTECT(3);
    return ptr;
}

SEXP r_n4m_model_inspect(SEXP bytes) {
    if (TYPEOF(bytes) != RAWSXP || XLENGTH(bytes) == 0)
        Rf_error("bytes must be a non-empty raw N4MM vector");
    uint32_t format = 0, major = 0, minor = 0, patch = 0;
    n4m_status_t status = n4m_serialization_inspect(
        RAW(bytes), (size_t)XLENGTH(bytes), &format, &major, &minor, &patch);
    if (status != N4M_OK) r_throw_status("n4m_serialization_inspect", status, NULL);
    SEXP result = PROTECT(Rf_allocVector(VECSXP, 2));
    SEXP abi = PROTECT(Rf_allocVector(REALSXP, 3));
    REAL(abi)[0] = (double)major;
    REAL(abi)[1] = (double)minor;
    REAL(abi)[2] = (double)patch;
    SET_VECTOR_ELT(result, 0, Rf_ScalarReal((double)format));
    SET_VECTOR_ELT(result, 1, abi);
    SEXP names = PROTECT(Rf_allocVector(STRSXP, 2));
    SET_STRING_ELT(names, 0, Rf_mkChar("format_version"));
    SET_STRING_ELT(names, 1, Rf_mkChar("writer_abi"));
    Rf_setAttrib(result, R_NamesSymbol, names);
    UNPROTECT(3);
    return result;
}

SEXP r_n4m_model_pipeline_info(SEXP bytes) {
    if (TYPEOF(bytes) != RAWSXP || XLENGTH(bytes) == 0)
        Rf_error("bytes must be a non-empty raw N4MM vector");
    n4m_serialized_pipeline_info_v1_t info;
    memset(&info, 0, sizeof(info));
    n4m_status_t status = n4m_serialization_inspect_pipeline_v1(
        RAW(bytes), (size_t)XLENGTH(bytes), &info, sizeof(info));
    if (status != N4M_OK)
        r_throw_status("n4m_serialization_inspect_pipeline_v1", status, NULL);
    SEXP out = PROTECT(Rf_allocVector(VECSXP, 7));
    SEXP names = PROTECT(Rf_allocVector(STRSXP, 7));
    const char* labels[] = {"present", "semantic_profile", "window_length",
                            "polyorder", "raw_n_features", "model_n_features",
                            "fingerprint"};
    SET_VECTOR_ELT(out, 0, Rf_ScalarLogical(info.present != 0));
    SET_VECTOR_ELT(out, 1, Rf_ScalarInteger((int)info.semantic_profile));
    SET_VECTOR_ELT(out, 2, Rf_ScalarInteger((int)info.savgol_window));
    SET_VECTOR_ELT(out, 3, Rf_ScalarInteger((int)info.savgol_poly_degree));
    SET_VECTOR_ELT(out, 4, Rf_ScalarInteger((int)info.raw_n_features));
    SET_VECTOR_ELT(out, 5, Rf_ScalarInteger((int)info.model_n_features));
    char fingerprint[17];
    snprintf(fingerprint, sizeof(fingerprint), "%016llx",
             (unsigned long long)info.fingerprint);
    SET_VECTOR_ELT(out, 6, Rf_mkString(fingerprint));
    for (int i = 0; i < 7; ++i) SET_STRING_ELT(names, i, Rf_mkChar(labels[i]));
    Rf_setAttrib(out, R_NamesSymbol, names);
    UNPROTECT(2);
    return out;
}

SEXP r_n4m_model_descriptor(SEXP bytes) {
    if (TYPEOF(bytes) != RAWSXP || XLENGTH(bytes) == 0)
        Rf_error("bytes must be a non-empty raw N4MM vector");
    n4m_serialized_model_info_v1_t info;
    memset(&info, 0, sizeof(info));
    n4m_status_t status = n4m_serialization_inspect_model_v1(
        RAW(bytes), (size_t)XLENGTH(bytes), &info);
    if (status != N4M_OK)
        r_throw_status("n4m_serialization_inspect_model_v1", status, NULL);
    SEXP out = PROTECT(Rf_allocVector(VECSXP, 8));
    SEXP names = PROTECT(Rf_allocVector(STRSXP, 8));
    const char* labels[] = {"format_version", "algorithm", "solver",
                            "deflation", "n_features", "n_targets",
                            "n_components", "capabilities"};
    SET_VECTOR_ELT(out, 0, Rf_ScalarInteger((int)info.format_version));
    SET_VECTOR_ELT(out, 1, Rf_ScalarInteger((int)info.algorithm));
    SET_VECTOR_ELT(out, 2, Rf_ScalarInteger((int)info.solver));
    SET_VECTOR_ELT(out, 3, Rf_ScalarInteger((int)info.deflation));
    SET_VECTOR_ELT(out, 4, Rf_ScalarInteger((int)info.n_features));
    SET_VECTOR_ELT(out, 5, Rf_ScalarInteger((int)info.n_targets));
    SET_VECTOR_ELT(out, 6, Rf_ScalarInteger((int)info.n_components));
    SET_VECTOR_ELT(out, 7, Rf_ScalarReal((double)info.capabilities));
    for (int i = 0; i < 8; ++i) SET_STRING_ELT(names, i, Rf_mkChar(labels[i]));
    Rf_setAttrib(out, R_NamesSymbol, names);
    UNPROTECT(2);
    return out;
}

/* ---- tagged model-array accessor ------------------------------------- */

/* Return a fitted-model array (e.g. coefficients) as a column-major R
 * matrix. `which_sexp` is the integer n4m_model_array_t tag. The core
 * allocates the array; we copy it into an R matrix (reading stride-aware
 * so any layout the core returns is handled) and release the core buffer
 * with n4m_array_free before returning — we never free across the
 * boundary except via that call. */
SEXP r_n4m_model_get_array(SEXP model_ptr, SEXP which_sexp) {
    if (TYPEOF(model_ptr) != EXTPTRSXP) Rf_error("model must be an external pointer");
    if (Rf_length(which_sexp) != 1 || TYPEOF(which_sexp) != INTSXP)
        Rf_error("which must be a length-1 integer");
    n4m_model_t* model = (n4m_model_t*)R_ExternalPtrAddr(model_ptr);
    if (model == NULL) Rf_error("model handle is NULL (already freed?)");
    const int which = INTEGER(which_sexp)[0];

    n4m_context_t* ctx = NULL;
    n4m_status_t status = n4m_context_create(&ctx);
    if (status != N4M_OK) r_throw_status("n4m_context_create", status, NULL);

    n4m_array_t* arr = NULL;
    status = n4m_model_get_array(ctx, model, (n4m_model_array_t)which, &arr);
    if (status != N4M_OK) {
        /* r_throw_status copies the last_error message then destroys ctx. */
        r_throw_status("n4m_model_get_array", status, ctx);
    }

    int64_t rows = 0;
    int64_t cols = 0;
    status = n4m_array_shape(arr, &rows, &cols);
    if (status != N4M_OK) {
        n4m_array_free(arr);
        r_throw_status("n4m_array_shape", status, ctx);
    }
    n4m_matrix_view_t view;
    status = n4m_array_view(arr, &view);
    if (status != N4M_OK) {
        n4m_array_free(arr);
        r_throw_status("n4m_array_view", status, ctx);
    }
    if (view.dtype != N4M_DTYPE_F64) {
        n4m_array_free(arr);
        n4m_context_destroy(ctx);
        Rf_error("n4m_model_get_array returned a non-f64 array");
    }

    /* Stride-aware copy into a column-major R matrix (rows x cols). */
    SEXP out = PROTECT(Rf_allocMatrix(REALSXP, (int)rows, (int)cols));
    double* outdat = REAL(out);
    const double* src = (const double*)view.data;
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            outdat[i + j * rows] =
                src[i * view.row_stride + j * view.col_stride];
        }
    }
    n4m_array_free(arr);
    n4m_context_destroy(ctx);
    UNPROTECT(1);
    return out;
}
