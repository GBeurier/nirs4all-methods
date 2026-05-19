/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * R .Call gateway for MethodResult-returning entry points
 * (sparse SIMPLS / CPPLS / weighted / MB-PLS / PLS-GLM / MIR-PLS),
 * selectors (variable_select_rank, spa_select, cars_select), and
 * diagnostics (pls_diagnostics_compute, approximate_press_compute,
 * pls_monitoring_run).
 *
 * Each function follows the same shape:
 *   - validate SEXP types / dims
 *   - rowmajor-copy X (and Y) so the C ABI gets contiguous row-major
 *     views (R matrices are column-major)
 *   - build a Config with sane defaults (PLS_REGRESSION + SIMPLS)
 *   - call the p4a_*_fit / p4a_*_select function
 *   - marshall the p4a_method_result_t into an R named list
 *   - destroy ctx, cfg, mr in all paths.
 */

#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "pls4all/p4a.h"

/* ---- shared helpers ------------------------------------------------- */

/* `r_throw` copies the per-context error message, destroys the ctx,
 * then longjmps via Rf_error. Marked noreturn so the compiler knows
 * downstream paths are unreachable. Ctx destruction must happen
 * BEFORE Rf_error: longjmp skips any cleanup below the call site. */
#if defined(__GNUC__) || defined(__clang__)
#  define P4A_R_NORETURN __attribute__((noreturn))
#else
#  define P4A_R_NORETURN
#endif

static void r_throw(const char* fn, p4a_status_t status, p4a_context_t* ctx)
    P4A_R_NORETURN;

static void r_throw(const char* fn, p4a_status_t status, p4a_context_t* ctx) {
    const char* status_str = p4a_status_to_string(status);
    char buf[4096];
    buf[0] = '\0';
    if (ctx != NULL) {
        const char* msg = p4a_context_last_error(ctx);
        if (msg != NULL && msg[0] != '\0') {
            /* Truncating copy: per-context error buffer is bounded at
             * P4A_ERROR_BUFFER_BYTES = 4096 anyway. */
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

static void colmajor_to_rowmajor(const double* src, double* dst,
                                  int64_t rows, int64_t cols) {
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            dst[i * cols + j] = src[i + j * rows];
        }
    }
}

static void rowmajor_to_colmajor(const double* src, double* dst,
                                  int64_t rows, int64_t cols) {
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            dst[i + j * rows] = src[i * cols + j];
        }
    }
}

static p4a_config_t* basic_cfg(int n_components) {
    p4a_config_t* cfg = NULL;
    p4a_status_t status = p4a_config_create(&cfg);
    if (status != P4A_OK) Rf_error("p4a_config_create failed");
    p4a_config_set_algorithm(cfg, P4A_ALGO_PLS_REGRESSION);
    p4a_config_set_solver(cfg, P4A_SOLVER_SIMPLS);
    p4a_config_set_deflation(cfg, P4A_DEFLATION_REGRESSION);
    p4a_config_set_n_components(cfg, n_components);
    p4a_config_set_center_x(cfg, 1);
    p4a_config_set_scale_x(cfg, 0);
    p4a_config_set_center_y(cfg, 1);
    p4a_config_set_scale_y(cfg, 0);
    p4a_config_set_tol(cfg, 1e-6);
    p4a_config_set_max_iter(cfg, 500);
    return cfg;
}

/* Extract a named double matrix as a column-major R matrix.
 * Returns R_NilValue when the name is absent. */
static SEXP mr_matrix(const p4a_method_result_t* mr, const char* name) {
    const double* data = NULL;
    int64_t rows = 0, cols = 0;
    p4a_status_t st = p4a_method_result_get_double_matrix(mr, name,
                                                          &data, &rows, &cols);
    if (st != P4A_OK || data == NULL) return R_NilValue;
    SEXP out = PROTECT(Rf_allocMatrix(REALSXP, (int)rows, (int)cols));
    rowmajor_to_colmajor(data, REAL(out), rows, cols);
    UNPROTECT(1);
    return out;
}

static SEXP mr_int_vector(const p4a_method_result_t* mr, const char* name) {
    const int32_t* data = NULL;
    int32_t size = 0;
    p4a_status_t st = p4a_method_result_get_int_vector(mr, name, &data, &size);
    if (st != P4A_OK || data == NULL) return R_NilValue;
    SEXP out = PROTECT(Rf_allocVector(INTSXP, size));
    for (int32_t i = 0; i < size; ++i) INTEGER(out)[i] = data[i];
    UNPROTECT(1);
    return out;
}

/* R has no int64, so widen to double to preserve magnitude. */
static SEXP mr_int64_as_double(const p4a_method_result_t* mr, const char* name) {
    const int64_t* data = NULL;
    int64_t size = 0;
    p4a_status_t st = p4a_method_result_get_int64_vector(mr, name, &data, &size);
    if (st != P4A_OK || data == NULL) return R_NilValue;
    SEXP out = PROTECT(Rf_allocVector(REALSXP, (R_xlen_t)size));
    for (int64_t i = 0; i < size; ++i) REAL(out)[i] = (double)data[i];
    UNPROTECT(1);
    return out;
}

static SEXP mr_scalar(const p4a_method_result_t* mr, const char* name) {
    double v = 0.0;
    p4a_status_t st = p4a_method_result_get_scalar(mr, name, &v);
    if (st != P4A_OK) return R_NilValue;
    return Rf_ScalarReal(v);
}

/* Pack a list of (name, SEXP) pairs into a named R list.
 * NULL SEXPs are silently dropped so absent keys don't produce NA. */
static SEXP make_named_list(const char** names, SEXP* values, int n) {
    int present = 0;
    for (int i = 0; i < n; ++i) if (values[i] != R_NilValue) present++;
    SEXP out = PROTECT(Rf_allocVector(VECSXP, present));
    SEXP nm = PROTECT(Rf_allocVector(STRSXP, present));
    int idx = 0;
    for (int i = 0; i < n; ++i) {
        if (values[i] == R_NilValue) continue;
        SET_VECTOR_ELT(out, idx, values[i]);
        SET_STRING_ELT(nm, idx, Rf_mkChar(names[i]));
        idx++;
    }
    Rf_setAttrib(out, R_NamesSymbol, nm);
    UNPROTECT(2);
    return out;
}

/* Allocate, validate, and rowmajor-copy X, Y. Both X_rm and Y_rm are
 * PROTECTed; caller must UNPROTECT(2) (or n+2 if it adds more).
 * On error, throws — no need to clean up because nothing is protected
 * before the throw. */
static void grab_xy(SEXP X, SEXP Y,
                     int64_t* n_rows, int64_t* n_cols, int64_t* y_cols,
                     SEXP* X_rm_out, SEXP* Y_rm_out) {
    if (TYPEOF(X) != REALSXP) Rf_error("X must be numeric");
    if (TYPEOF(Y) != REALSXP) Rf_error("Y must be numeric");
    SEXP X_dim = Rf_getAttrib(X, R_DimSymbol);
    SEXP Y_dim = Rf_getAttrib(Y, R_DimSymbol);
    if (Rf_length(X_dim) != 2) Rf_error("X must be a 2D matrix");
    if (Rf_length(Y_dim) != 2) Rf_error("Y must be a 2D matrix");
    *n_rows = INTEGER(X_dim)[0];
    *n_cols = INTEGER(X_dim)[1];
    const int64_t yr = INTEGER(Y_dim)[0];
    *y_cols = INTEGER(Y_dim)[1];
    if (yr != *n_rows) Rf_error("nrow(X) (%lld) must equal nrow(Y) (%lld)",
                                (long long)*n_rows, (long long)yr);
    SEXP X_rm = PROTECT(Rf_allocMatrix(REALSXP, (int)*n_rows, (int)*n_cols));
    SEXP Y_rm = PROTECT(Rf_allocMatrix(REALSXP, (int)*n_rows, (int)*y_cols));
    colmajor_to_rowmajor(REAL(X), REAL(X_rm), *n_rows, *n_cols);
    colmajor_to_rowmajor(REAL(Y), REAL(Y_rm), *n_rows, *y_cols);
    *X_rm_out = X_rm;
    *Y_rm_out = Y_rm;
}

/* Pack result keys into a named list, then destroy the MethodResult. */
static SEXP pack_and_destroy(p4a_method_result_t* mr,
                              const char** keys, int n_keys,
                              const char* extra_int_key /* may be NULL */,
                              const char* extra_int64_key /* may be NULL */) {
    SEXP* slots = (SEXP*)R_alloc((size_t)(n_keys + 2), sizeof(SEXP));
    int slot_count = 0;
    for (int i = 0; i < n_keys; ++i) {
        SEXP v = mr_matrix(mr, keys[i]);
        if (v == R_NilValue) v = mr_scalar(mr, keys[i]);
        slots[slot_count++] = PROTECT(v);
    }
    int extra_count = 0;
    const char** all_keys = (const char**)R_alloc((size_t)(n_keys + 2), sizeof(char*));
    for (int i = 0; i < n_keys; ++i) all_keys[i] = keys[i];
    if (extra_int_key) {
        SEXP v = mr_int_vector(mr, extra_int_key);
        slots[slot_count++] = PROTECT(v);
        all_keys[n_keys + extra_count] = extra_int_key;
        extra_count++;
    }
    if (extra_int64_key) {
        SEXP v = mr_int64_as_double(mr, extra_int64_key);
        slots[slot_count++] = PROTECT(v);
        all_keys[n_keys + extra_count] = extra_int64_key;
        extra_count++;
    }
    SEXP out = make_named_list(all_keys, slots, slot_count);
    UNPROTECT(slot_count);
    p4a_method_result_destroy(mr);
    return out;
}

/* ====================================================================
 * Method-result fits
 * ==================================================================== */

SEXP r_p4a_sparse_simpls_fit(SEXP X, SEXP Y, SEXP n_components_sexp,
                              SEXP sparsity_lambda_sexp) {
    const int nc = INTEGER(n_components_sexp)[0];
    const double lambda = REAL(sparsity_lambda_sexp)[0];
    int64_t n_rows = 0, n_cols = 0, y_cols = 0;
    SEXP X_rm, Y_rm;
    grab_xy(X, Y, &n_rows, &n_cols, &y_cols, &X_rm, &Y_rm);
    p4a_context_t* ctx = NULL;
    p4a_status_t st = p4a_context_create(&ctx);
    if (st != P4A_OK) { UNPROTECT(2); Rf_error("p4a_context_create failed"); }
    p4a_config_t* cfg = basic_cfg(nc);
    p4a_matrix_view_t Xv, Yv;
    p4a_matrix_view_init_rowmajor(&Xv, REAL(X_rm), n_rows, n_cols, P4A_DTYPE_F64);
    p4a_matrix_view_init_rowmajor(&Yv, REAL(Y_rm), n_rows, y_cols, P4A_DTYPE_F64);
    p4a_method_result_t* mr = NULL;
    st = p4a_sparse_simpls_fit(ctx, cfg, &Xv, &Yv, lambda, &mr);
    p4a_config_destroy(cfg);
    if (st != P4A_OK) {
        UNPROTECT(2);
        r_throw("p4a_sparse_simpls_fit", st, ctx);
    }
    p4a_context_destroy(ctx);
    UNPROTECT(2);
    const char* keys[] = {"coefficients", "predictions", "x_mean", "y_mean",
                            "weights_w", "rmse"};
    return pack_and_destroy(mr, keys, 6, NULL, NULL);
}

SEXP r_p4a_cppls_fit(SEXP X, SEXP Y, SEXP n_components_sexp, SEXP gamma_sexp) {
    const int nc = INTEGER(n_components_sexp)[0];
    const double gamma = REAL(gamma_sexp)[0];
    int64_t n_rows = 0, n_cols = 0, y_cols = 0;
    SEXP X_rm, Y_rm;
    grab_xy(X, Y, &n_rows, &n_cols, &y_cols, &X_rm, &Y_rm);
    p4a_context_t* ctx = NULL;
    p4a_status_t st = p4a_context_create(&ctx);
    if (st != P4A_OK) { UNPROTECT(2); Rf_error("p4a_context_create failed"); }
    p4a_config_t* cfg = basic_cfg(nc);
    p4a_matrix_view_t Xv, Yv;
    p4a_matrix_view_init_rowmajor(&Xv, REAL(X_rm), n_rows, n_cols, P4A_DTYPE_F64);
    p4a_matrix_view_init_rowmajor(&Yv, REAL(Y_rm), n_rows, y_cols, P4A_DTYPE_F64);
    p4a_method_result_t* mr = NULL;
    st = p4a_cppls_fit(ctx, cfg, &Xv, &Yv, gamma, &mr);
    p4a_config_destroy(cfg);
    if (st != P4A_OK) {
        UNPROTECT(2);
        r_throw("p4a_cppls_fit", st, ctx);
    }
    p4a_context_destroy(ctx);
    UNPROTECT(2);
    const char* keys[] = {"coefficients", "predictions", "x_mean", "y_mean", "rmse"};
    return pack_and_destroy(mr, keys, 5, NULL, NULL);
}

SEXP r_p4a_weighted_pls_fit(SEXP X, SEXP Y, SEXP n_components_sexp,
                             SEXP sample_weights_sexp) {
    if (TYPEOF(sample_weights_sexp) != REALSXP)
        Rf_error("sample_weights must be numeric");
    const int nc = INTEGER(n_components_sexp)[0];
    int64_t n_rows = 0, n_cols = 0, y_cols = 0;
    SEXP X_rm, Y_rm;
    grab_xy(X, Y, &n_rows, &n_cols, &y_cols, &X_rm, &Y_rm);
    if (Rf_length(sample_weights_sexp) != n_rows) {
        UNPROTECT(2);
        Rf_error("length(sample_weights) (%d) must equal nrow(X) (%lld)",
                  Rf_length(sample_weights_sexp), (long long)n_rows);
    }
    p4a_context_t* ctx = NULL;
    p4a_status_t st = p4a_context_create(&ctx);
    if (st != P4A_OK) { UNPROTECT(2); Rf_error("p4a_context_create failed"); }
    p4a_config_t* cfg = basic_cfg(nc);
    p4a_matrix_view_t Xv, Yv;
    p4a_matrix_view_init_rowmajor(&Xv, REAL(X_rm), n_rows, n_cols, P4A_DTYPE_F64);
    p4a_matrix_view_init_rowmajor(&Yv, REAL(Y_rm), n_rows, y_cols, P4A_DTYPE_F64);
    p4a_method_result_t* mr = NULL;
    st = p4a_weighted_pls_fit(ctx, cfg, &Xv, &Yv,
                               REAL(sample_weights_sexp),
                               (int64_t)Rf_length(sample_weights_sexp), &mr);
    p4a_config_destroy(cfg);
    if (st != P4A_OK) {
        UNPROTECT(2);
        r_throw("p4a_weighted_pls_fit", st, ctx);
    }
    p4a_context_destroy(ctx);
    UNPROTECT(2);
    const char* keys[] = {"coefficients", "predictions", "x_mean", "y_mean", "rmse"};
    return pack_and_destroy(mr, keys, 5, NULL, NULL);
}

SEXP r_p4a_mb_pls_fit(SEXP X, SEXP Y, SEXP n_components_sexp,
                       SEXP block_sizes_sexp) {
    if (TYPEOF(block_sizes_sexp) != INTSXP && TYPEOF(block_sizes_sexp) != REALSXP)
        Rf_error("block_sizes must be integer or numeric");
    const int nc = INTEGER(n_components_sexp)[0];
    int64_t n_rows = 0, n_cols = 0, y_cols = 0;
    SEXP X_rm, Y_rm;
    grab_xy(X, Y, &n_rows, &n_cols, &y_cols, &X_rm, &Y_rm);
    const int n_blocks = Rf_length(block_sizes_sexp);
    int64_t* bs = (int64_t*)R_alloc((size_t)n_blocks, sizeof(int64_t));
    int64_t bs_sum = 0;
    for (int i = 0; i < n_blocks; ++i) {
        int64_t v = (TYPEOF(block_sizes_sexp) == INTSXP)
            ? (int64_t)INTEGER(block_sizes_sexp)[i]
            : (int64_t)REAL(block_sizes_sexp)[i];
        bs[i] = v;
        bs_sum += v;
    }
    if (bs_sum != n_cols) {
        UNPROTECT(2);
        Rf_error("sum(block_sizes)=%lld must equal ncol(X)=%lld",
                  (long long)bs_sum, (long long)n_cols);
    }
    p4a_context_t* ctx = NULL;
    p4a_status_t st = p4a_context_create(&ctx);
    if (st != P4A_OK) { UNPROTECT(2); Rf_error("p4a_context_create failed"); }
    p4a_config_t* cfg = basic_cfg(nc);
    p4a_config_set_solver(cfg, P4A_SOLVER_NIPALS); /* MB-PLS requires NIPALS */
    p4a_matrix_view_t Xv, Yv;
    p4a_matrix_view_init_rowmajor(&Xv, REAL(X_rm), n_rows, n_cols, P4A_DTYPE_F64);
    p4a_matrix_view_init_rowmajor(&Yv, REAL(Y_rm), n_rows, y_cols, P4A_DTYPE_F64);
    p4a_method_result_t* mr = NULL;
    st = p4a_mb_pls_fit(ctx, cfg, &Xv, &Yv, bs, n_blocks, &mr);
    p4a_config_destroy(cfg);
    if (st != P4A_OK) {
        UNPROTECT(2);
        r_throw("p4a_mb_pls_fit", st, ctx);
    }
    p4a_context_destroy(ctx);
    UNPROTECT(2);
    const char* keys[] = {"coefficients", "predictions", "x_mean", "x_scale",
                            "intercept", "block_weights", "rmse"};
    return pack_and_destroy(mr, keys, 7, NULL, NULL);
}

SEXP r_p4a_pls_glm_fit(SEXP X, SEXP Y, SEXP n_components_sexp, SEXP poisson_sexp) {
    const int nc = INTEGER(n_components_sexp)[0];
    const int poisson = INTEGER(poisson_sexp)[0];
    int64_t n_rows = 0, n_cols = 0, y_cols = 0;
    SEXP X_rm, Y_rm;
    grab_xy(X, Y, &n_rows, &n_cols, &y_cols, &X_rm, &Y_rm);
    p4a_context_t* ctx = NULL;
    p4a_status_t st = p4a_context_create(&ctx);
    if (st != P4A_OK) { UNPROTECT(2); Rf_error("p4a_context_create failed"); }
    p4a_config_t* cfg = basic_cfg(nc);
    p4a_matrix_view_t Xv, Yv;
    p4a_matrix_view_init_rowmajor(&Xv, REAL(X_rm), n_rows, n_cols, P4A_DTYPE_F64);
    p4a_matrix_view_init_rowmajor(&Yv, REAL(Y_rm), n_rows, y_cols, P4A_DTYPE_F64);
    p4a_method_result_t* mr = NULL;
    st = p4a_pls_glm_fit(ctx, cfg, &Xv, &Yv, poisson, &mr);
    p4a_config_destroy(cfg);
    if (st != P4A_OK) {
        UNPROTECT(2);
        r_throw("p4a_pls_glm_fit", st, ctx);
    }
    p4a_context_destroy(ctx);
    UNPROTECT(2);
    const char* keys[] = {"coefficients", "intercept", "predictions",
                            "x_mean", "rmse"};
    return pack_and_destroy(mr, keys, 5, NULL, NULL);
}

SEXP r_p4a_mir_pls_fit(SEXP X, SEXP Y, SEXP n_components_sexp) {
    const int nc = INTEGER(n_components_sexp)[0];
    int64_t n_rows = 0, n_cols = 0, y_cols = 0;
    SEXP X_rm, Y_rm;
    grab_xy(X, Y, &n_rows, &n_cols, &y_cols, &X_rm, &Y_rm);
    p4a_context_t* ctx = NULL;
    p4a_status_t st = p4a_context_create(&ctx);
    if (st != P4A_OK) { UNPROTECT(2); Rf_error("p4a_context_create failed"); }
    p4a_config_t* cfg = basic_cfg(nc);
    p4a_matrix_view_t Xv, Yv;
    p4a_matrix_view_init_rowmajor(&Xv, REAL(X_rm), n_rows, n_cols, P4A_DTYPE_F64);
    p4a_matrix_view_init_rowmajor(&Yv, REAL(Y_rm), n_rows, y_cols, P4A_DTYPE_F64);
    p4a_method_result_t* mr = NULL;
    st = p4a_mir_pls_fit(ctx, cfg, &Xv, &Yv, &mr);
    p4a_config_destroy(cfg);
    if (st != P4A_OK) {
        UNPROTECT(2);
        r_throw("p4a_mir_pls_fit", st, ctx);
    }
    p4a_context_destroy(ctx);
    UNPROTECT(2);
    const char* keys[] = {"coefficients", "predictions", "x_mean", "y_mean", "rmse"};
    return pack_and_destroy(mr, keys, 5, NULL, NULL);
}

/* ====================================================================
 * Selectors
 * ==================================================================== */

/* VIP (0) / coefficient (1) / selectivity ratio (2) variants of
 * variable_select_rank. Operates on a fitted model + the X used for
 * fitting (or another X if the model has been re-applied). */
SEXP r_p4a_variable_select_rank(SEXP model_ptr, SEXP X, SEXP method_sexp,
                                 SEXP top_k_sexp) {
    if (TYPEOF(model_ptr) != EXTPTRSXP) Rf_error("model must be an external pointer");
    if (TYPEOF(X) != REALSXP) Rf_error("X must be numeric");
    p4a_model_t* model = (p4a_model_t*)R_ExternalPtrAddr(model_ptr);
    if (model == NULL) Rf_error("model handle is NULL");
    const int method = INTEGER(method_sexp)[0];
    const int top_k = INTEGER(top_k_sexp)[0];
    SEXP X_dim = Rf_getAttrib(X, R_DimSymbol);
    const int64_t n_rows = INTEGER(X_dim)[0];
    const int64_t n_cols = INTEGER(X_dim)[1];
    SEXP X_rm = PROTECT(Rf_allocMatrix(REALSXP, (int)n_rows, (int)n_cols));
    colmajor_to_rowmajor(REAL(X), REAL(X_rm), n_rows, n_cols);
    p4a_context_t* ctx = NULL;
    p4a_status_t st = p4a_context_create(&ctx);
    if (st != P4A_OK) { UNPROTECT(1); Rf_error("p4a_context_create failed"); }
    p4a_matrix_view_t Xv;
    p4a_matrix_view_init_rowmajor(&Xv, REAL(X_rm), n_rows, n_cols, P4A_DTYPE_F64);
    p4a_method_result_t* mr = NULL;
    st = p4a_variable_select_rank(ctx, model, &Xv, method, top_k, &mr);
    if (st != P4A_OK) {
        UNPROTECT(1);
        r_throw("p4a_variable_select_rank", st, ctx);
    }
    p4a_context_destroy(ctx);
    UNPROTECT(1);
    const char* keys[] = {"scores"};
    return pack_and_destroy(mr, keys, 1, NULL, "selected_indices");
}

SEXP r_p4a_spa_select(SEXP X, SEXP Y, SEXP n_components_sexp, SEXP top_k_sexp) {
    const int nc = INTEGER(n_components_sexp)[0];
    const int top_k = INTEGER(top_k_sexp)[0];
    int64_t n_rows = 0, n_cols = 0, y_cols = 0;
    SEXP X_rm, Y_rm;
    grab_xy(X, Y, &n_rows, &n_cols, &y_cols, &X_rm, &Y_rm);
    p4a_context_t* ctx = NULL;
    p4a_status_t st = p4a_context_create(&ctx);
    if (st != P4A_OK) { UNPROTECT(2); Rf_error("p4a_context_create failed"); }
    p4a_config_t* cfg = basic_cfg(nc);
    p4a_matrix_view_t Xv, Yv;
    p4a_matrix_view_init_rowmajor(&Xv, REAL(X_rm), n_rows, n_cols, P4A_DTYPE_F64);
    p4a_matrix_view_init_rowmajor(&Yv, REAL(Y_rm), n_rows, y_cols, P4A_DTYPE_F64);
    p4a_method_result_t* mr = NULL;
    st = p4a_spa_select(ctx, cfg, &Xv, &Yv, top_k, &mr);
    p4a_config_destroy(cfg);
    if (st != P4A_OK) {
        UNPROTECT(2);
        r_throw("p4a_spa_select", st, ctx);
    }
    p4a_context_destroy(ctx);
    UNPROTECT(2);
    const char* keys[] = {"best_rmse"};
    return pack_and_destroy(mr, keys, 1, NULL, "selected_indices");
}

/* CARS — passes NULL ValidationPlan (C side falls back to a default
 * 5-fold scheme). */
SEXP r_p4a_cars_select(SEXP X, SEXP Y, SEXP n_components_sexp,
                        SEXP n_iterations_sexp, SEXP min_features_sexp) {
    const int nc = INTEGER(n_components_sexp)[0];
    const int n_iter = INTEGER(n_iterations_sexp)[0];
    const int min_f = INTEGER(min_features_sexp)[0];
    int64_t n_rows = 0, n_cols = 0, y_cols = 0;
    SEXP X_rm, Y_rm;
    grab_xy(X, Y, &n_rows, &n_cols, &y_cols, &X_rm, &Y_rm);
    p4a_context_t* ctx = NULL;
    p4a_status_t st = p4a_context_create(&ctx);
    if (st != P4A_OK) { UNPROTECT(2); Rf_error("p4a_context_create failed"); }
    p4a_config_t* cfg = basic_cfg(nc);
    p4a_matrix_view_t Xv, Yv;
    p4a_matrix_view_init_rowmajor(&Xv, REAL(X_rm), n_rows, n_cols, P4A_DTYPE_F64);
    p4a_matrix_view_init_rowmajor(&Yv, REAL(Y_rm), n_rows, y_cols, P4A_DTYPE_F64);
    p4a_method_result_t* mr = NULL;
    st = p4a_cars_select(ctx, cfg, &Xv, &Yv, NULL, n_iter, min_f, &mr);
    p4a_config_destroy(cfg);
    if (st != P4A_OK) {
        UNPROTECT(2);
        r_throw("p4a_cars_select", st, ctx);
    }
    p4a_context_destroy(ctx);
    UNPROTECT(2);
    const char* keys[] = {"best_rmse"};
    return pack_and_destroy(mr, keys, 1, NULL, "selected_indices");
}

/* ====================================================================
 * Diagnostics
 * ==================================================================== */

/* Compute T², Q, DModX from a fitted model + design matrix X.
 * A NULL X_reference mirrors the Python/MATLAB default: the core falls back
 * to the training score distribution stored on the fitted model. */
SEXP r_p4a_pls_diagnostics_compute(SEXP model_ptr, SEXP X) {
    if (TYPEOF(model_ptr) != EXTPTRSXP) Rf_error("model must be an external pointer");
    if (TYPEOF(X) != REALSXP) Rf_error("X must be numeric");
    p4a_model_t* model = (p4a_model_t*)R_ExternalPtrAddr(model_ptr);
    if (model == NULL) Rf_error("model handle is NULL");
    SEXP X_dim = Rf_getAttrib(X, R_DimSymbol);
    const int64_t n_rows = INTEGER(X_dim)[0];
    const int64_t n_cols = INTEGER(X_dim)[1];
    SEXP X_rm = PROTECT(Rf_allocMatrix(REALSXP, (int)n_rows, (int)n_cols));
    colmajor_to_rowmajor(REAL(X), REAL(X_rm), n_rows, n_cols);
    p4a_context_t* ctx = NULL;
    p4a_status_t st = p4a_context_create(&ctx);
    if (st != P4A_OK) { UNPROTECT(1); Rf_error("p4a_context_create failed"); }
    p4a_matrix_view_t Xv;
    p4a_matrix_view_init_rowmajor(&Xv, REAL(X_rm), n_rows, n_cols, P4A_DTYPE_F64);
    p4a_method_result_t* mr = NULL;
    st = p4a_pls_diagnostics_compute(ctx, model, &Xv, NULL, &mr);
    if (st != P4A_OK) {
        UNPROTECT(1);
        r_throw("p4a_pls_diagnostics_compute", st, ctx);
    }
    p4a_context_destroy(ctx);
    UNPROTECT(1);
    const char* keys[] = {"t2", "q", "dmodx"};
    return pack_and_destroy(mr, keys, 3, NULL, NULL);
}

SEXP r_p4a_approximate_press_compute(SEXP X, SEXP Y, SEXP max_components_sexp) {
    const int max_k = INTEGER(max_components_sexp)[0];
    int64_t n_rows = 0, n_cols = 0, y_cols = 0;
    SEXP X_rm, Y_rm;
    grab_xy(X, Y, &n_rows, &n_cols, &y_cols, &X_rm, &Y_rm);
    p4a_context_t* ctx = NULL;
    p4a_status_t st = p4a_context_create(&ctx);
    if (st != P4A_OK) { UNPROTECT(2); Rf_error("p4a_context_create failed"); }
    p4a_config_t* cfg = basic_cfg(max_k);
    p4a_matrix_view_t Xv, Yv;
    p4a_matrix_view_init_rowmajor(&Xv, REAL(X_rm), n_rows, n_cols, P4A_DTYPE_F64);
    p4a_matrix_view_init_rowmajor(&Yv, REAL(Y_rm), n_rows, y_cols, P4A_DTYPE_F64);
    p4a_method_result_t* mr = NULL;
    st = p4a_approximate_press_compute(ctx, cfg, &Xv, &Yv, max_k, &mr);
    p4a_config_destroy(cfg);
    if (st != P4A_OK) {
        UNPROTECT(2);
        r_throw("p4a_approximate_press_compute", st, ctx);
    }
    p4a_context_destroy(ctx);
    UNPROTECT(2);
    const char* keys[] = {"press_per_component", "rmse_per_component"};
    return pack_and_destroy(mr, keys, 2, "selected_n_components", NULL);
}

SEXP r_p4a_pls_monitoring_run(SEXP model_ptr, SEXP X_ref, SEXP X_mon,
                               SEXP alpha_sexp) {
    if (TYPEOF(model_ptr) != EXTPTRSXP) Rf_error("model must be an external pointer");
    if (TYPEOF(X_ref) != REALSXP || TYPEOF(X_mon) != REALSXP)
        Rf_error("X_reference and X_monitor must be numeric matrices");
    p4a_model_t* model = (p4a_model_t*)R_ExternalPtrAddr(model_ptr);
    if (model == NULL) Rf_error("model handle is NULL");
    const double alpha = REAL(alpha_sexp)[0];
    SEXP rd = Rf_getAttrib(X_ref, R_DimSymbol);
    SEXP md = Rf_getAttrib(X_mon, R_DimSymbol);
    if (Rf_length(rd) != 2 || Rf_length(md) != 2)
        Rf_error("X_reference and X_monitor must be 2D");
    const int64_t rr = INTEGER(rd)[0], rc = INTEGER(rd)[1];
    const int64_t mrr = INTEGER(md)[0], mc = INTEGER(md)[1];
    if (rc != mc) Rf_error("X_reference and X_monitor must have the same ncol");
    SEXP R_rm = PROTECT(Rf_allocMatrix(REALSXP, (int)rr, (int)rc));
    SEXP M_rm = PROTECT(Rf_allocMatrix(REALSXP, (int)mrr, (int)mc));
    colmajor_to_rowmajor(REAL(X_ref), REAL(R_rm), rr, rc);
    colmajor_to_rowmajor(REAL(X_mon), REAL(M_rm), mrr, mc);
    p4a_context_t* ctx = NULL;
    p4a_status_t st = p4a_context_create(&ctx);
    if (st != P4A_OK) { UNPROTECT(2); Rf_error("p4a_context_create failed"); }
    p4a_matrix_view_t Rv, Mv;
    p4a_matrix_view_init_rowmajor(&Rv, REAL(R_rm), rr, rc, P4A_DTYPE_F64);
    p4a_matrix_view_init_rowmajor(&Mv, REAL(M_rm), mrr, mc, P4A_DTYPE_F64);
    p4a_method_result_t* mr = NULL;
    st = p4a_pls_monitoring_run(ctx, model, &Rv, &Mv, alpha, &mr);
    if (st != P4A_OK) {
        UNPROTECT(2);
        r_throw("p4a_pls_monitoring_run", st, ctx);
    }
    p4a_context_destroy(ctx);
    UNPROTECT(2);

    /* monitoring exposes a mix of double matrices, int vectors, and
     * scalars — we don't fit pack_and_destroy's generic mold so do it
     * manually. */
    SEXP t2 = PROTECT(mr_matrix(mr, "t2"));
    SEXP q  = PROTECT(mr_matrix(mr, "q"));
    SEXP t2a = PROTECT(mr_int_vector(mr, "t2_alarms"));
    SEXP qa  = PROTECT(mr_int_vector(mr, "q_alarms"));
    SEXP aa  = PROTECT(mr_int_vector(mr, "any_alarms"));
    SEXP t2t = PROTECT(mr_scalar(mr, "t2_threshold"));
    SEXP qt  = PROTECT(mr_scalar(mr, "q_threshold"));
    const char* nm[] = {"t2", "q", "t2_alarms", "q_alarms", "any_alarms",
                          "t2_threshold", "q_threshold"};
    SEXP vals[] = {t2, q, t2a, qa, aa, t2t, qt};
    SEXP out = make_named_list(nm, vals, 7);
    UNPROTECT(7);
    p4a_method_result_destroy(mr);
    return out;
}
