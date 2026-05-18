/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * R .Call gateway over the pls4all libp4a C ABI.
 *
 * The native shared library libp4a must be locatable at load time. R loads
 * pls4all.so first; its dependencies (libp4a) are then resolved by the
 * runtime linker. Set the LD_LIBRARY_PATH / DYLD_LIBRARY_PATH / PATH
 * appropriately, or run `R CMD INSTALL --configure-args=...` to embed a
 * specific rpath in the Makevars.
 */

#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "pls4all/p4a.h"

/* ---- helpers ---------------------------------------------------------- */

static SEXP r_make_string(const char* s) {
    if (s == NULL) s = "";
    return Rf_mkString(s);
}

/* Copy the per-context error message, destroy ctx, then longjmp via
 * Rf_error. Doing the destroy BEFORE Rf_error prevents leaking the ctx
 * on the longjmp (longjmp skips any cleanup below the throw site).
 * Callers must have already freed any other C resources they owned. */
static void r_throw_status(const char* fn, p4a_status_t status, p4a_context_t* ctx) {
    const char* status_str = p4a_status_to_string(status);
    char buf[4096];
    buf[0] = '\0';
    if (ctx != NULL) {
        const char* msg = p4a_context_last_error(ctx);
        if (msg != NULL && msg[0] != '\0') {
            size_t n = strlen(msg);
            if (n >= sizeof(buf)) n = sizeof(buf) - 1;
            memcpy(buf, msg, n);
            buf[n] = '\0';
        }
        p4a_context_destroy(ctx);
    }
    if (buf[0] != '\0') {
        Rf_error("%s failed: %s (%s)", fn, status_str, buf);
    } else {
        Rf_error("%s failed: %s", fn, status_str);
    }
}

static int resolve_algo(const char* name,
                        p4a_algorithm_t* out_algo,
                        p4a_solver_t* out_solver,
                        p4a_deflation_t* out_deflation) {
    *out_deflation = P4A_DEFLATION_REGRESSION;
    if (strcmp(name, "pls_nipals") == 0) {
        *out_algo = P4A_ALGO_PLS_REGRESSION;
        *out_solver = P4A_SOLVER_NIPALS;
        return 1;
    }
    if (strcmp(name, "pls_orthogonal_scores") == 0) {
        *out_algo = P4A_ALGO_PLS_REGRESSION;
        *out_solver = P4A_SOLVER_ORTHOGONAL_SCORES;
        return 1;
    }
    if (strcmp(name, "pls_simpls") == 0) {
        *out_algo = P4A_ALGO_PLS_REGRESSION;
        *out_solver = P4A_SOLVER_SIMPLS;
        return 1;
    }
    if (strcmp(name, "pls_kernel_algorithm") == 0) {
        *out_algo = P4A_ALGO_PLS_REGRESSION;
        *out_solver = P4A_SOLVER_KERNEL_ALGORITHM;
        return 1;
    }
    if (strcmp(name, "pls_wide_kernel") == 0) {
        *out_algo = P4A_ALGO_PLS_REGRESSION;
        *out_solver = P4A_SOLVER_WIDE_KERNEL;
        return 1;
    }
    if (strcmp(name, "pls_svd") == 0) {
        *out_algo = P4A_ALGO_PLS_REGRESSION;
        *out_solver = P4A_SOLVER_SVD;
        return 1;
    }
    if (strcmp(name, "pls_power") == 0) {
        *out_algo = P4A_ALGO_PLS_REGRESSION;
        *out_solver = P4A_SOLVER_POWER;
        return 1;
    }
    if (strcmp(name, "pls_randomized_svd") == 0) {
        *out_algo = P4A_ALGO_PLS_REGRESSION;
        *out_solver = P4A_SOLVER_RANDOMIZED_SVD;
        return 1;
    }
    if (strcmp(name, "pcr_svd") == 0 || strcmp(name, "pcr") == 0) {
        *out_algo = P4A_ALGO_PCR;
        *out_solver = P4A_SOLVER_SVD;
        return 1;
    }
    if (strcmp(name, "opls") == 0 || strcmp(name, "opls_nipals") == 0) {
        *out_algo = P4A_ALGO_OPLS;
        *out_solver = P4A_SOLVER_NIPALS;
        *out_deflation = P4A_DEFLATION_ORTHOGONAL;
        return 1;
    }
    return 0;
}

/* ---- external pointer finalizers ------------------------------------- */

static void r_model_finalize(SEXP ptr) {
    if (TYPEOF(ptr) != EXTPTRSXP) return;
    p4a_model_t* model = (p4a_model_t*)R_ExternalPtrAddr(ptr);
    if (model != NULL) {
        p4a_model_destroy(model);
        R_ClearExternalPtr(ptr);
    }
}

/* ---- version queries -------------------------------------------------- */

SEXP r_pls4all_version(void) {
    return r_make_string(p4a_get_version_string());
}

SEXP r_pls4all_abi_version(void) {
    SEXP out = PROTECT(Rf_allocVector(INTSXP, 3));
    INTEGER(out)[0] = (int)p4a_get_abi_version_major();
    INTEGER(out)[1] = (int)p4a_get_abi_version_minor();
    INTEGER(out)[2] = (int)p4a_get_abi_version_patch();
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

SEXP r_pls4all_fit(SEXP X, SEXP Y, SEXP algo_sexp, SEXP n_components_sexp,
                    SEXP store_scores_sexp,
                    SEXP center_x_sexp, SEXP scale_x_sexp,
                    SEXP center_y_sexp, SEXP scale_y_sexp) {
    if (TYPEOF(X) != REALSXP) Rf_error("X must be a numeric matrix");
    if (TYPEOF(Y) != REALSXP) Rf_error("Y must be a numeric matrix");
    if (TYPEOF(algo_sexp) != STRSXP) Rf_error("algo must be character");
    if (TYPEOF(n_components_sexp) != INTSXP) Rf_error("n_components must be integer");
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
    p4a_algorithm_t algo;
    p4a_solver_t solver;
    p4a_deflation_t deflation;
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

    p4a_context_t* ctx = NULL;
    p4a_status_t status = p4a_context_create(&ctx);
    if (status != P4A_OK) r_throw_status("p4a_context_create", status, NULL);

    p4a_config_t* cfg = NULL;
    status = p4a_config_create(&cfg);
    if (status != P4A_OK) {
        p4a_context_destroy(ctx);
        r_throw_status("p4a_config_create", status, NULL);
    }
    p4a_config_set_algorithm(cfg, algo);
    p4a_config_set_solver(cfg, solver);
    p4a_config_set_deflation(cfg, deflation);
    p4a_config_set_n_components(cfg, n_components);
    /* Match the public C ABI Config defaults (scale_x = scale_y = 1)
     * so the R binding produces byte-equivalent results to the Python
     * binding under default settings. */
    p4a_config_set_center_x(cfg, center_x);
    p4a_config_set_scale_x(cfg, scale_x);
    p4a_config_set_center_y(cfg, center_y);
    p4a_config_set_scale_y(cfg, scale_y);
    p4a_config_set_store_scores(cfg, store_scores ? 1 : 0);

    p4a_matrix_view_t X_view;
    p4a_matrix_view_t Y_view;
    p4a_matrix_view_init_rowmajor(&X_view, xrm, n_rows, n_cols, P4A_DTYPE_F64);
    p4a_matrix_view_init_rowmajor(&Y_view, yrm, n_rows, y_cols, P4A_DTYPE_F64);

    p4a_model_t* model = NULL;
    status = p4a_model_fit(ctx, cfg, &X_view, &Y_view, &model);
    p4a_config_destroy(cfg);
    if (status != P4A_OK) {
        UNPROTECT(2);
        /* r_throw_status copies the last_error message then destroys ctx. */
        r_throw_status("p4a_model_fit", status, ctx);
    }
    p4a_context_destroy(ctx);
    UNPROTECT(2);

    SEXP ptr = PROTECT(R_MakeExternalPtr(model, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ptr, r_model_finalize, TRUE);
    /* attach feature/target counts as attributes for downstream predict. */
    SEXP n_features_attr = PROTECT(Rf_allocVector(INTSXP, 1));
    SEXP n_targets_attr = PROTECT(Rf_allocVector(INTSXP, 1));
    int32_t nf = 0;
    int32_t nt = 0;
    p4a_model_get_n_features(model, &nf);
    p4a_model_get_n_targets(model, &nt);
    INTEGER(n_features_attr)[0] = (int)nf;
    INTEGER(n_targets_attr)[0] = (int)nt;
    Rf_setAttrib(ptr, Rf_install("n_features"), n_features_attr);
    Rf_setAttrib(ptr, Rf_install("n_targets"), n_targets_attr);
    UNPROTECT(3);
    return ptr;
}

/* ---- predict ---------------------------------------------------------- */

SEXP r_pls4all_predict(SEXP model_ptr, SEXP X) {
    if (TYPEOF(model_ptr) != EXTPTRSXP) Rf_error("model must be an external pointer");
    if (TYPEOF(X) != REALSXP) Rf_error("X must be a numeric matrix");
    p4a_model_t* model = (p4a_model_t*)R_ExternalPtrAddr(model_ptr);
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
    p4a_model_get_n_targets(model, &n_targets);
    SEXP out_rm = PROTECT(Rf_allocVector(REALSXP, (R_xlen_t)(n_rows * n_targets)));
    double* outrm = REAL(out_rm);

    p4a_context_t* ctx = NULL;
    p4a_status_t status = p4a_context_create(&ctx);
    if (status != P4A_OK) {
        UNPROTECT(2);
        r_throw_status("p4a_context_create", status, NULL);
    }
    p4a_matrix_view_t X_view;
    p4a_matrix_view_init_rowmajor(&X_view, xrm, n_rows, n_cols, P4A_DTYPE_F64);
    p4a_matrix_view_t out_view;
    p4a_matrix_view_init_rowmajor(&out_view, outrm, n_rows, n_targets, P4A_DTYPE_F64);
    status = p4a_model_predict(ctx, model, &X_view, &out_view);
    if (status != P4A_OK) {
        UNPROTECT(2);
        /* r_throw_status owns ctx destruction. */
        r_throw_status("p4a_model_predict", status, ctx);
    }
    p4a_context_destroy(ctx);

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
