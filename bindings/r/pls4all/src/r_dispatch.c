/* SPDX-License-Identifier: CECILL-2.1
 *
 * R .Call dispatcher covering ALL MethodResult-returning entry points
 * of libn4m (33 method-result fits + 24 selectors + 4 diagnostics).
 *
 *   res <- .Call("r_p4a_dispatch_fit",
 *                  algo, X, Y, n_components, params,
 *                  center_x, scale_x, center_y, scale_y)
 *
 * `params` is an R named list whose elements supply algorithm-specific
 * parameters (sparsity_lambda, sample_weights, block_sizes, X_target,
 * y_labels, …). Each algorithm documents the required field names.
 *
 * Returns a named R list with all MethodResult arrays converted to
 * column-major R matrices, vectors as numeric or integer, and indices
 * shifted to 1-based.
 */

#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "n4m/n4m.h"

#include <stdarg.h>
#include <stdio.h>

/* ---- shared helpers (file-local) ----------------------------------- */

#if defined(__GNUC__) || defined(__clang__)
#  define N4M_R_NORETURN __attribute__((noreturn))
#  define N4M_R_PRINTF(a, b) __attribute__((format(printf, a, b)))
#else
#  define N4M_R_NORETURN
#  define N4M_R_PRINTF(a, b)
#endif

/* Destroy ctx + cfg and longjmp via Rf_error. Used by validation
 * branches that need to bail out AFTER ctx/cfg have been allocated.
 * SEXP PROTECTs are unwound by R itself on the longjmp. */
static void cleanup_err(n4m_context_t* ctx, n4m_config_t* cfg,
                         const char* fmt, ...) N4M_R_NORETURN
                                                  N4M_R_PRINTF(3, 4);

static void cleanup_err(n4m_context_t* ctx, n4m_config_t* cfg,
                         const char* fmt, ...) {
    if (cfg) n4m_config_destroy(cfg);
    if (ctx) n4m_context_destroy(ctx);
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    Rf_error("%s", buf);
}

/* Validate that a SEXP is a 2D REAL matrix; cleanup_err on failure.
 * Returns the dim vector. */
static SEXP need_real_matrix(SEXP v, n4m_context_t* ctx, n4m_config_t* cfg,
                              const char* what) {
    if (TYPEOF(v) != REALSXP)
        cleanup_err(ctx, cfg, "%s must be a numeric matrix", what);
    SEXP dim = Rf_getAttrib(v, R_DimSymbol);
    if (Rf_length(dim) != 2)
        cleanup_err(ctx, cfg, "%s must be a 2D matrix (got length(dim)=%d)",
                     what, (int)Rf_length(dim));
    return dim;
}

static void r_throw(const char* fn, n4m_status_t status, n4m_context_t* ctx)
    N4M_R_NORETURN;

static void r_throw(const char* fn, n4m_status_t status, n4m_context_t* ctx) {
    const char* status_str = n4m_status_to_string(status);
    char buf[4096];
    buf[0] = '\0';
    if (ctx) {
        const char* msg = n4m_context_last_error(ctx);
        if (msg && msg[0]) {
            size_t n = strlen(msg);
            if (n >= sizeof(buf)) n = sizeof(buf) - 1;
            memcpy(buf, msg, n);
            buf[n] = '\0';
        }
        n4m_context_destroy(ctx);
    }
    if (buf[0])
        Rf_error("%s failed: %s (%s)", fn, status_str, buf);
    else
        Rf_error("%s failed: %s", fn, status_str);
}

static void colmajor_to_rowmajor(const double* src, double* dst,
                                  int64_t rows, int64_t cols) {
    for (int64_t i = 0; i < rows; ++i)
        for (int64_t j = 0; j < cols; ++j)
            dst[i * cols + j] = src[i + j * rows];
}

static void rowmajor_to_colmajor(const double* src, double* dst,
                                  int64_t rows, int64_t cols) {
    for (int64_t i = 0; i < rows; ++i)
        for (int64_t j = 0; j < cols; ++j)
            dst[i + j * rows] = src[i * cols + j];
}

/* Extract a named element from an R named list. Returns R_NilValue
 * when the name is absent (caller treats it as default). */
static SEXP get_list_element(SEXP list, const char* name) {
    if (list == R_NilValue || TYPEOF(list) != VECSXP) return R_NilValue;
    SEXP names = Rf_getAttrib(list, R_NamesSymbol);
    if (names == R_NilValue) return R_NilValue;
    R_xlen_t n = Rf_length(list);
    for (R_xlen_t i = 0; i < n; ++i) {
        if (strcmp(CHAR(STRING_ELT(names, i)), name) == 0)
            return VECTOR_ELT(list, i);
    }
    return R_NilValue;
}

static double get_double(SEXP list, const char* name, double dflt) {
    SEXP v = get_list_element(list, name);
    if (v == R_NilValue || Rf_length(v) == 0) return dflt;
    if (TYPEOF(v) == REALSXP) return REAL(v)[0];
    if (TYPEOF(v) == INTSXP)  return (double)INTEGER(v)[0];
    if (TYPEOF(v) == LGLSXP)  return (double)LOGICAL(v)[0];
    return dflt;
}

static int get_int(SEXP list, const char* name, int dflt) {
    return (int)get_double(list, name, (double)dflt);
}

static uint64_t get_u64(SEXP list, const char* name, uint64_t dflt) {
    return (uint64_t)get_double(list, name, (double)dflt);
}

/* ---- MethodResult → R list ----------------------------------------- */

static SEXP mr_matrix(const n4m_method_result_t* mr, const char* name) {
    const double* data = NULL;
    int64_t rows = 0, cols = 0;
    if (n4m_method_result_get_double_matrix(mr, name,
                                             &data, &rows, &cols) != N4M_OK || !data)
        return R_NilValue;
    SEXP out = PROTECT(Rf_allocMatrix(REALSXP, (int)rows, (int)cols));
    rowmajor_to_colmajor(data, REAL(out), rows, cols);
    UNPROTECT(1);
    return out;
}

static SEXP mr_int_vector(const n4m_method_result_t* mr, const char* name) {
    const int32_t* data = NULL;
    int32_t size = 0;
    if (n4m_method_result_get_int_vector(mr, name, &data, &size) != N4M_OK || !data)
        return R_NilValue;
    SEXP out = PROTECT(Rf_allocVector(INTSXP, size));
    for (int32_t i = 0; i < size; ++i) INTEGER(out)[i] = data[i];
    UNPROTECT(1);
    return out;
}

/* Allow-list of int64 fields whose contents are 0-based row / feature /
 * interval indices and need a +1 shift to expose 1-based R indices. */
static int is_index_int64_name(const char* name) {
    static const char* nm[] = {
        "selected_indices", "neighbor_indices_i64",
        "top_k_intervals",  "removed_indices",
        NULL
    };
    for (int i = 0; nm[i]; ++i)
        if (strcmp(name, nm[i]) == 0) return 1;
    return 0;
}

static SEXP mr_int64_vector(const n4m_method_result_t* mr, const char* name) {
    const int64_t* data = NULL;
    int64_t size = 0;
    if (n4m_method_result_get_int64_vector(mr, name, &data, &size) != N4M_OK || !data)
        return R_NilValue;
    SEXP out = PROTECT(Rf_allocVector(REALSXP, (R_xlen_t)size));
    int shift = is_index_int64_name(name);
    for (int64_t i = 0; i < size; ++i)
        REAL(out)[i] = (double)(shift ? (data[i] + 1) : data[i]);
    UNPROTECT(1);
    return out;
}

static SEXP mr_scalar(const n4m_method_result_t* mr, const char* name) {
    double v = 0.0;
    if (n4m_method_result_get_scalar(mr, name, &v) != N4M_OK) return R_NilValue;
    return Rf_ScalarReal(v);
}

/* Build an R named list dropping absent keys (NULL entries). */
static SEXP make_list(const char** names, SEXP* vals, int n) {
    int present = 0;
    for (int i = 0; i < n; ++i) if (vals[i] != R_NilValue) present++;
    SEXP out = PROTECT(Rf_allocVector(VECSXP, present));
    SEXP nm = PROTECT(Rf_allocVector(STRSXP, present));
    int idx = 0;
    for (int i = 0; i < n; ++i) {
        if (vals[i] == R_NilValue) continue;
        SET_VECTOR_ELT(out, idx, vals[i]);
        SET_STRING_ELT(nm, idx, Rf_mkChar(names[i]));
        idx++;
    }
    Rf_setAttrib(out, R_NamesSymbol, nm);
    UNPROTECT(2);
    return out;
}

/* Pack a MethodResult by category. NULL-terminated key arrays. */
static SEXP pack_result(n4m_method_result_t* mr,
                          const char** dmat, const char** iv,
                          const char** i64, const char** sc) {
    int n_dmat = 0, n_iv = 0, n_i64 = 0, n_sc = 0;
    while (dmat && dmat[n_dmat]) n_dmat++;
    while (iv && iv[n_iv]) n_iv++;
    while (i64 && i64[n_i64]) n_i64++;
    while (sc && sc[n_sc]) n_sc++;
    int total = n_dmat + n_iv + n_i64 + n_sc;
    const char** names = (const char**)R_alloc((size_t)total, sizeof(char*));
    SEXP* vals = (SEXP*)R_alloc((size_t)total, sizeof(SEXP));
    int idx = 0;
    for (int i = 0; i < n_dmat; ++i) {
        names[idx] = dmat[i];
        vals[idx++] = PROTECT(mr_matrix(mr, dmat[i]));
    }
    for (int i = 0; i < n_iv; ++i) {
        names[idx] = iv[i];
        vals[idx++] = PROTECT(mr_int_vector(mr, iv[i]));
    }
    for (int i = 0; i < n_i64; ++i) {
        names[idx] = i64[i];
        vals[idx++] = PROTECT(mr_int64_vector(mr, i64[i]));
    }
    for (int i = 0; i < n_sc; ++i) {
        names[idx] = sc[i];
        vals[idx++] = PROTECT(mr_scalar(mr, sc[i]));
    }
    SEXP out = make_list(names, vals, total);
    UNPROTECT(total);
    n4m_method_result_destroy(mr);
    return out;
}

/* ---- (X, Y) rowmajor copy helper ----------------------------------- */

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
    if (INTEGER(Y_dim)[0] != *n_rows)
        Rf_error("nrow(X) (%lld) must equal nrow(Y) (%lld)",
                 (long long)*n_rows, (long long)INTEGER(Y_dim)[0]);
    *y_cols = INTEGER(Y_dim)[1];
    SEXP X_rm = PROTECT(Rf_allocMatrix(REALSXP, (int)*n_rows, (int)*n_cols));
    SEXP Y_rm = PROTECT(Rf_allocMatrix(REALSXP, (int)*n_rows, (int)*y_cols));
    colmajor_to_rowmajor(REAL(X), REAL(X_rm), *n_rows, *n_cols);
    colmajor_to_rowmajor(REAL(Y), REAL(Y_rm), *n_rows, *y_cols);
    *X_rm_out = X_rm;
    *Y_rm_out = Y_rm;
}

/* ---- Config builder with caller-controlled center/scale flags ------- */

static n4m_config_t* build_cfg(int n_components,
                                int center_x, int scale_x,
                                int center_y, int scale_y) {
    n4m_config_t* cfg = NULL;
    if (n4m_config_create(&cfg) != N4M_OK)
        Rf_error("n4m_config_create failed");
    n4m_config_set_algorithm(cfg, N4M_ALGO_PLS_REGRESSION);
    n4m_config_set_solver(cfg, N4M_SOLVER_SIMPLS);
    n4m_config_set_deflation(cfg, N4M_DEFLATION_REGRESSION);
    n4m_config_set_n_components(cfg, n_components);
    n4m_config_set_center_x(cfg, center_x);
    n4m_config_set_scale_x(cfg,  scale_x);
    n4m_config_set_center_y(cfg, center_y);
    n4m_config_set_scale_y(cfg,  scale_y);
    n4m_config_set_tol(cfg, 1e-6);
    n4m_config_set_max_iter(cfg, 500);
    n4m_config_set_store_scores(cfg, 0);
    return cfg;
}

/* ---- Coerce R int/double-vector list field to int32 buffer --------- */

static int32_t* coerce_int32_vec(SEXP v, int* n_out) {
    *n_out = (int)Rf_length(v);
    int32_t* out = (int32_t*)R_alloc((size_t)*n_out, sizeof(int32_t));
    if (TYPEOF(v) == INTSXP) {
        for (int i = 0; i < *n_out; ++i) out[i] = INTEGER(v)[i];
    } else if (TYPEOF(v) == REALSXP) {
        for (int i = 0; i < *n_out; ++i) out[i] = (int32_t)REAL(v)[i];
    } else {
        Rf_error("expected numeric or integer vector");
    }
    return out;
}

static int64_t* coerce_int64_vec(SEXP v, int* n_out) {
    *n_out = (int)Rf_length(v);
    int64_t* out = (int64_t*)R_alloc((size_t)*n_out, sizeof(int64_t));
    if (TYPEOF(v) == INTSXP) {
        for (int i = 0; i < *n_out; ++i) out[i] = (int64_t)INTEGER(v)[i];
    } else if (TYPEOF(v) == REALSXP) {
        for (int i = 0; i < *n_out; ++i) out[i] = (int64_t)REAL(v)[i];
    } else {
        Rf_error("expected numeric or integer vector");
    }
    return out;
}

/* ---- Default ValidationPlan helper --------------------------------- */

static n4m_validation_plan_t* make_default_plan(int n, int requested_folds) {
    if (n < 4) return NULL;
    int k = requested_folds;
    if (k > n / 2) k = n / 2;
    if (k < 2)     k = 2;
    n4m_validation_plan_t* plan = NULL;
    if (n4m_validation_plan_create(&plan) != N4M_OK) return NULL;
    if (n4m_validation_plan_set_n_samples(plan, (int64_t)n) != N4M_OK) {
        n4m_validation_plan_destroy(plan);
        return NULL;
    }
    /* Match the canonical Python/registry plan: equal floor-sized
       contiguous folds, with the remainder assigned to the final fold. */
    int fold_size = n / k;
    int64_t* train = (int64_t*)R_alloc((size_t)n, sizeof(int64_t));
    int64_t* test  = (int64_t*)R_alloc((size_t)n, sizeof(int64_t));
    for (int f = 0; f < k; ++f) {
        int t0 = f * fold_size;
        int t1 = (f < k - 1) ? (t0 + fold_size) : n;
        int n_test = t1 - t0;
        int ti = 0, te = 0;
        for (int i = 0; i < n; ++i) {
            if (i >= t0 && i < t1) test[te++] = (int64_t)i;
            else                    train[ti++] = (int64_t)i;
        }
        if (n4m_validation_plan_add_fold(plan, train, ti, test, te) != N4M_OK) {
            n4m_validation_plan_destroy(plan);
            return NULL;
        }
    }
    return plan;
}

/* ---- AOM / POP selector helpers ------------------------------------ */

static n4m_status_t add_bank_op(n4m_operator_bank_t* bank,
                                 n4m_operator_kind_t kind,
                                 const double* params,
                                 int32_t n_params) {
    return n4m_operator_bank_add(bank, kind, params, n_params);
}

static n4m_status_t make_compact_aom_bank(int n_ops,
                                           n4m_operator_bank_t** out_bank) {
    if (!out_bank) return N4M_ERR_INVALID_ARGUMENT;
    *out_bank = NULL;
    if (n_ops < 1) n_ops = 1;
    if (n_ops > 9) n_ops = 9;
    n4m_operator_bank_t* bank = NULL;
    n4m_status_t st = n4m_operator_bank_create(&bank);
    if (st != N4M_OK) return st;

    const double sg11_p2[] = {11.0, 2.0};
    const double sg21_p3[] = {21.0, 3.0};
    const double sg11_p2_d1[] = {11.0, 2.0, 1.0};
    const double sg21_p3_d1[] = {21.0, 3.0, 1.0};
    const double sg11_p2_d2[] = {11.0, 2.0, 2.0};
    const double detrend_1[] = {1.0};
    const double detrend_2[] = {2.0};
    const double fd_1[] = {1.0};

    for (int i = 0; st == N4M_OK && i < n_ops; ++i) {
        switch (i) {
        case 0:
            st = add_bank_op(bank, N4M_OP_IDENTITY, NULL, 0);
            break;
        case 1:
            st = add_bank_op(bank, N4M_OP_SAVGOL_SMOOTH, sg11_p2, 2);
            break;
        case 2:
            st = add_bank_op(bank, N4M_OP_SAVGOL_SMOOTH, sg21_p3, 2);
            break;
        case 3:
            st = add_bank_op(bank, N4M_OP_SAVGOL_DERIVATIVE, sg11_p2_d1, 3);
            break;
        case 4:
            st = add_bank_op(bank, N4M_OP_SAVGOL_DERIVATIVE, sg21_p3_d1, 3);
            break;
        case 5:
            st = add_bank_op(bank, N4M_OP_SAVGOL_DERIVATIVE, sg11_p2_d2, 3);
            break;
        case 6:
            st = add_bank_op(bank, N4M_OP_DETREND_POLY, detrend_1, 1);
            break;
        case 7:
            st = add_bank_op(bank, N4M_OP_DETREND_POLY, detrend_2, 1);
            break;
        default:
            st = add_bank_op(bank, N4M_OP_FINITE_DIFFERENCE, fd_1, 1);
            break;
        }
    }
    if (st != N4M_OK) {
        n4m_operator_bank_destroy(bank);
        return st;
    }
    *out_bank = bank;
    return N4M_OK;
}

static SEXP double_matrix_from_rowmajor(const double* data,
                                         int64_t rows, int64_t cols) {
    SEXP out = PROTECT(Rf_allocMatrix(REALSXP, (int)rows, (int)cols));
    rowmajor_to_colmajor(data, REAL(out), rows, cols);
    UNPROTECT(1);
    return out;
}

static SEXP double_vector_from_i32(const int32_t* data, int32_t n,
                                    int shift) {
    SEXP out = PROTECT(Rf_allocVector(REALSXP, n));
    for (int32_t i = 0; i < n; ++i)
        REAL(out)[i] = (double)(shift ? (data[i] + 1) : data[i]);
    UNPROTECT(1);
    return out;
}

static SEXP double_vector_from_op_kinds(const n4m_operator_kind_t* data,
                                         int32_t n) {
    SEXP out = PROTECT(Rf_allocVector(REALSXP, n));
    for (int32_t i = 0; i < n; ++i)
        REAL(out)[i] = (double)data[i];
    UNPROTECT(1);
    return out;
}

static SEXP double_vector_from_f64(const double* data, int32_t n) {
    SEXP out = PROTECT(Rf_allocVector(REALSXP, n));
    for (int32_t i = 0; i < n; ++i) REAL(out)[i] = data[i];
    UNPROTECT(1);
    return out;
}

static SEXP pack_aom_global_result(n4m_aom_global_result_t* res) {
    const double* preds = NULL;
    int64_t pred_rows = 0, pred_cols = 0;
    const n4m_operator_kind_t* kinds = NULL;
    int32_t n_kinds = 0;
    const double* op_scores = NULL;
    int32_t n_scores = 0;
    const double* curves = NULL;
    int32_t curve_rows = 0, curve_cols = 0;
    int32_t selected_op = 0, selected_k = 0, n_ops = 0, max_k = 0;
    double best_score = 0.0;

    n4m_aom_global_result_get_predictions(res, &preds, &pred_rows, &pred_cols);
    n4m_aom_global_result_get_operator_kinds(res, &kinds, &n_kinds);
    n4m_aom_global_result_get_operator_scores(res, &op_scores, &n_scores);
    n4m_aom_global_result_get_rmse_curves(res, &curves, &curve_rows, &curve_cols);
    n4m_aom_global_result_get_selected_operator_index(res, &selected_op);
    n4m_aom_global_result_get_selected_n_components(res, &selected_k);
    n4m_aom_global_result_get_n_operators(res, &n_ops);
    n4m_aom_global_result_get_max_components(res, &max_k);
    n4m_aom_global_result_get_best_score(res, &best_score);

    const char* names[] = {
        "predictions", "operator_kinds", "operator_scores",
        "rmse_curves", "selected_operator_index",
        "selected_n_components", "n_operators", "max_components",
        "best_score"
    };
    SEXP vals[9];
    vals[0] = PROTECT(double_matrix_from_rowmajor(preds, pred_rows, pred_cols));
    vals[1] = PROTECT(double_vector_from_op_kinds(kinds, n_kinds));
    vals[2] = PROTECT(double_vector_from_f64(op_scores, n_scores));
    vals[3] = PROTECT(double_matrix_from_rowmajor(curves, curve_rows, curve_cols));
    vals[4] = PROTECT(Rf_ScalarInteger(selected_op + 1));
    vals[5] = PROTECT(Rf_ScalarInteger(selected_k));
    vals[6] = PROTECT(Rf_ScalarInteger(n_ops));
    vals[7] = PROTECT(Rf_ScalarInteger(max_k));
    vals[8] = PROTECT(Rf_ScalarReal(best_score));
    SEXP out = make_list(names, vals, 9);
    UNPROTECT(9);
    n4m_aom_global_result_destroy(res);
    return out;
}

static SEXP pack_aom_per_component_result(n4m_aom_per_component_result_t* res) {
    const double* preds = NULL;
    int64_t pred_rows = 0, pred_cols = 0;
    const n4m_operator_kind_t* kinds = NULL;
    int32_t n_kinds = 0;
    const int32_t* selected_ops = NULL;
    int32_t n_selected = 0;
    const double* component_scores = NULL;
    int32_t comp_rows = 0, comp_cols = 0;
    const double* prefix_scores = NULL;
    int32_t n_prefix = 0;
    int32_t selected_k = 0, n_ops = 0, max_k = 0;
    double best_score = 0.0;

    n4m_aom_per_component_result_get_predictions(res, &preds, &pred_rows, &pred_cols);
    n4m_aom_per_component_result_get_operator_kinds(res, &kinds, &n_kinds);
    n4m_aom_per_component_result_get_selected_operator_indices(
        res, &selected_ops, &n_selected);
    n4m_aom_per_component_result_get_component_scores(
        res, &component_scores, &comp_rows, &comp_cols);
    n4m_aom_per_component_result_get_prefix_scores(res, &prefix_scores, &n_prefix);
    n4m_aom_per_component_result_get_selected_n_components(res, &selected_k);
    n4m_aom_per_component_result_get_n_operators(res, &n_ops);
    n4m_aom_per_component_result_get_max_components(res, &max_k);
    n4m_aom_per_component_result_get_best_score(res, &best_score);

    const char* names[] = {
        "predictions", "operator_kinds", "selected_operator_indices",
        "component_scores", "prefix_scores", "selected_n_components",
        "n_operators", "max_components", "best_score"
    };
    SEXP vals[9];
    vals[0] = PROTECT(double_matrix_from_rowmajor(preds, pred_rows, pred_cols));
    vals[1] = PROTECT(double_vector_from_op_kinds(kinds, n_kinds));
    vals[2] = PROTECT(double_vector_from_i32(selected_ops, n_selected, 1));
    vals[3] = PROTECT(double_matrix_from_rowmajor(component_scores,
                                                   comp_rows, comp_cols));
    vals[4] = PROTECT(double_vector_from_f64(prefix_scores, n_prefix));
    vals[5] = PROTECT(Rf_ScalarInteger(selected_k));
    vals[6] = PROTECT(Rf_ScalarInteger(n_ops));
    vals[7] = PROTECT(Rf_ScalarInteger(max_k));
    vals[8] = PROTECT(Rf_ScalarReal(best_score));
    SEXP out = make_list(names, vals, 9);
    UNPROTECT(9);
    n4m_aom_per_component_result_destroy(res);
    return out;
}

/* =====================================================================
 *  Main dispatcher
 * ===================================================================== */

SEXP r_p4a_dispatch_fit(SEXP algo_sexp, SEXP X, SEXP Y,
                         SEXP n_components_sexp, SEXP params,
                         SEXP center_x_sexp, SEXP scale_x_sexp,
                         SEXP center_y_sexp, SEXP scale_y_sexp) {
    /* Early validation BEFORE ctx/cfg allocation: pass NULL for both. */
    if (TYPEOF(algo_sexp) != STRSXP) cleanup_err(NULL, NULL, "algo must be character");
    const char* algo = CHAR(STRING_ELT(algo_sexp, 0));
    int n_components = INTEGER(n_components_sexp)[0];
    if (n_components < 1) cleanup_err(NULL, NULL, "n_components must be >= 1");

    int center_x = INTEGER(center_x_sexp)[0];
    int scale_x  = INTEGER(scale_x_sexp)[0];
    int center_y = INTEGER(center_y_sexp)[0];
    int scale_y  = INTEGER(scale_y_sexp)[0];

    int64_t n_rows = 0, n_cols = 0, y_cols = 0;
    SEXP X_rm, Y_rm;
    grab_xy(X, Y, &n_rows, &n_cols, &y_cols, &X_rm, &Y_rm);
    int n = (int)n_rows, p = (int)n_cols, q = (int)y_cols;

    n4m_context_t* ctx = NULL;
    n4m_config_t* cfg = NULL;
    if (n4m_context_create(&ctx) != N4M_OK) {
        UNPROTECT(2);
        cleanup_err(NULL, NULL, "n4m_context_create failed");
    }
    cfg = build_cfg(n_components, center_x, scale_x, center_y, scale_y);
    /* Item #21: honour per-method parity conventions injected by the
     * orchestrator via the params list (BENCH_R_PARAMS_JSON). When absent
     * these keys default to build_cfg's hardcoded values, so non-bench
     * callers are unaffected. solver enum: NIPALS=0, SIMPLS=1, SVD=5. */
    {
        int solver_v = get_int(params, "solver", -1);
        if (solver_v >= 0)
            n4m_config_set_solver(cfg, (n4m_solver_t)solver_v);
        int sx = get_int(params, "scale_x", -1);
        if (sx >= 0) n4m_config_set_scale_x(cfg, sx);
        int sy = get_int(params, "scale_y", -1);
        if (sy >= 0) n4m_config_set_scale_y(cfg, sy);
    }

    n4m_matrix_view_t Xv, Yv;
    n4m_matrix_view_init_rowmajor(&Xv, REAL(X_rm), n, p, N4M_DTYPE_F64);
    n4m_matrix_view_init_rowmajor(&Yv, REAL(Y_rm), n, q, N4M_DTYPE_F64);

    n4m_method_result_t* mr = NULL;
    n4m_status_t st = N4M_OK;
    SEXP out = R_NilValue;


    /* Predeclared key arrays so the DISPATCH_PLAN_SELECT macro doesn't
     * need compound literals in its args (preprocessor would split on
     * the internal commas). */
    static const char* SEL_IDX[]    = {"selected_indices", NULL};
    static const char* SEL_IDX_TKI[] = {"selected_indices", "top_k_intervals", NULL};
    static const char* SEL_RMSE[]   = {"best_rmse", NULL};
    static const char* PSO_DMAT[]   = {"inclusion_frequencies",
                                         "best_rmse_by_iteration",
                                         "mean_rmse_by_iteration", NULL};
    static const char* VISSA_DMAT[] = {"final_probabilities",
                                         "inclusion_frequencies",
                                         "best_rmse_by_iteration",
                                         "mean_rmse_by_iteration",
                                         "top_k_per_iteration", NULL};
    static const char* IRIV_DMAT[]  = {"remaining_per_round",
                                         "removed_per_round", NULL};
    static const char* IRIV_SC[]    = {"n_rounds", NULL};
    static const char* IRF_DMAT[]   = {"probability", "rmse_by_iteration",
                                         "subset_sizes", NULL};

    /* Standard regressor key sets used across many fits. */
    static const char* REG_DMAT[] = {"coefficients", "predictions",
                                       "x_mean", "y_mean", NULL};
    static const char* REG_SCALAR[] = {"rmse", NULL};

    /* ---------------- MethodResult fits ---------------- */

    if (strcmp(algo, "sparse_simpls") == 0) {
        double l = get_double(params, "sparsity_lambda", 0.05);
        st = n4m_sparse_simpls_fit(ctx, cfg, &Xv, &Yv, l, &mr);
        if (st == N4M_OK) {
            static const char* dm[] = {"coefficients", "predictions", "x_mean",
                                          "y_mean", "weights_w", NULL};
            out = pack_result(mr, dm, NULL, NULL, REG_SCALAR);
        }
    } else if (strcmp(algo, "cppls") == 0) {
        double g = get_double(params, "gamma", 0.5);
        st = n4m_cppls_fit(ctx, cfg, &Xv, &Yv, g, &mr);
        if (st == N4M_OK) out = pack_result(mr, REG_DMAT, NULL, NULL, REG_SCALAR);
    } else if (strcmp(algo, "ecr") == 0) {
        double a = get_double(params, "alpha", 0.5);
        st = n4m_ecr_fit(ctx, cfg, &Xv, &Yv, a, &mr);
        if (st == N4M_OK) {
            static const char* dm[] = {"coefficients", "predictions",
                                          "x_mean", "y_mean", "x_scale", "y_scale",
                                          "weights_w", "loadings_p", "y_loadings",
                                          "wstar", "r2x", "r2y", NULL};
            static const char* sc[] = {"alpha", "rmse", NULL};
            out = pack_result(mr, dm, NULL, NULL, sc);
        }
    } else if (strcmp(algo, "di_pls") == 0) {
        SEXP xt = get_list_element(params, "X_target");
        if (xt == R_NilValue || TYPEOF(xt) != REALSXP)
            cleanup_err(ctx, cfg, "di_pls requires params$X_target (numeric matrix)");
        SEXP xt_dim = Rf_getAttrib(xt, R_DimSymbol);
        int xt_n = INTEGER(xt_dim)[0], xt_p = INTEGER(xt_dim)[1];
        if (xt_p != p) cleanup_err(ctx, cfg, "X_target ncol must equal X ncol");
        SEXP XT_rm = PROTECT(Rf_allocMatrix(REALSXP, xt_n, xt_p));
        colmajor_to_rowmajor(REAL(xt), REAL(XT_rm), xt_n, xt_p);
        double dl = get_double(params, "di_lambda", 1.0);
        n4m_matrix_view_t XTv;
        n4m_matrix_view_init_rowmajor(&XTv, REAL(XT_rm), xt_n, xt_p, N4M_DTYPE_F64);
        st = n4m_di_pls_fit(ctx, cfg, &Xv, &Yv, &XTv, dl, &mr);
        UNPROTECT(1);
        if (st == N4M_OK) out = pack_result(mr, REG_DMAT, NULL, NULL, REG_SCALAR);
    } else if (strcmp(algo, "weighted_pls") == 0) {
        SEXP w = get_list_element(params, "sample_weights");
        if (w == R_NilValue || TYPEOF(w) != REALSXP || Rf_length(w) != n)
            cleanup_err(ctx, cfg, "weighted_pls requires numeric params$sample_weights of length n");
        st = n4m_weighted_pls_fit(ctx, cfg, &Xv, &Yv,
                                   REAL(w), (int64_t)Rf_length(w), &mr);
        if (st == N4M_OK) out = pack_result(mr, REG_DMAT, NULL, NULL, REG_SCALAR);
    } else if (strcmp(algo, "robust_pls") == 0) {
        double k = get_double(params, "huber_k", 1.345);
        int it = get_int(params, "max_irls_iter", 20);
        st = n4m_robust_pls_fit(ctx, cfg, &Xv, &Yv, k, it, &mr);
        if (st == N4M_OK) out = pack_result(mr, REG_DMAT, NULL, NULL, REG_SCALAR);
    } else if (strcmp(algo, "ridge_pls") == 0) {
        double l = get_double(params, "ridge_lambda", 1.0);
        st = n4m_ridge_pls_fit(ctx, cfg, &Xv, &Yv, l, &mr);
        if (st == N4M_OK) out = pack_result(mr, REG_DMAT, NULL, NULL, REG_SCALAR);
    } else if (strcmp(algo, "continuum_regression") == 0) {
        double t = get_double(params, "tau", 0.5);
        st = n4m_continuum_regression_fit(ctx, cfg, &Xv, &Yv, t, &mr);
        if (st == N4M_OK) out = pack_result(mr, REG_DMAT, NULL, NULL, REG_SCALAR);
    } else if (strcmp(algo, "recursive_pls") == 0) {
        int w = get_int(params, "window_size", 50);
        st = n4m_recursive_pls_run(ctx, cfg, &Xv, &Yv, w, &mr);
        if (st == N4M_OK) {
            static const char* dm[] = {"predictions", NULL};
            static const char* iv[] = {"in_window", NULL};
            static const char* sc[] = {"rmse_predictable", NULL};
            out = pack_result(mr, dm, iv, NULL, sc);
        }
    } else if (strcmp(algo, "n_pls") == 0) {
        int mj = get_int(params, "mode_j", 0);
        int mk = get_int(params, "mode_k", 0);
        if (mj <= 0 || mk <= 0 || mj * mk != p)
            cleanup_err(ctx, cfg, "n_pls requires mode_j * mode_k == ncol(X)");
        st = n4m_n_pls_fit(ctx, cfg, &Xv, mj, mk, &Yv, &mr);
        if (st == N4M_OK) out = pack_result(mr, REG_DMAT, NULL, NULL, REG_SCALAR);
    } else if (strcmp(algo, "kernel_pls") == 0) {
        int kt = get_int(params, "kernel_type", 1);
        double gamma = get_double(params, "gamma", 0.0);
        double coef0 = get_double(params, "coef0", 1.0);
        int deg = get_int(params, "degree", 3);
        st = n4m_kernel_pls_fit(ctx, cfg, kt, gamma, coef0, deg, &Xv, &Yv, &mr);
        if (st == N4M_OK) {
            static const char* dm[] = {"predictions", "alpha", "y_mean", NULL};
            static const char* sc[] = {"rmse", "kernel_type", NULL};
            out = pack_result(mr, dm, NULL, NULL, sc);
        }
    } else if (strcmp(algo, "o2pls") == 0) {
        int np = get_int(params, "n_predictive", 2);
        int nx = get_int(params, "n_x_orthogonal", 1);
        int ny2 = get_int(params, "n_y_orthogonal", 1);
        st = n4m_o2pls_fit(ctx, cfg, &Xv, &Yv, np, nx, ny2, &mr);
        if (st == N4M_OK) {
            static const char* dm[] = {"coefficients", "predictions",
                                          "x_mean", "y_mean",
                                          "w_predictive", "c_predictive",
                                          "w_x_orthogonal", "c_y_orthogonal",
                                          "b_predictive", NULL};
            out = pack_result(mr, dm, NULL, NULL, REG_SCALAR);
        }
    } else if (strcmp(algo, "sparse_pls_da") == 0) {
        SEXP yl = get_list_element(params, "y_labels");
        if (yl == R_NilValue) cleanup_err(ctx, cfg, "sparse_pls_da requires params$y_labels");
        int nl = 0;
        int32_t* labels = coerce_int32_vec(yl, &nl);
        if (nl != n) cleanup_err(ctx, cfg, "y_labels must have length n");
        st = n4m_sparse_pls_da_fit(ctx, cfg, &Xv, labels, nl, &mr);
        if (st == N4M_OK) {
            static const char* dm[] = {"coefficients", "predictions",
                                          "x_mean", "y_mean",
                                          "class_priors", NULL};
            out = pack_result(mr, dm, NULL, NULL, REG_SCALAR);
        }
    } else if (strcmp(algo, "group_sparse_pls") == 0) {
        SEXP g = get_list_element(params, "group_assignment");
        if (g == R_NilValue) cleanup_err(ctx, cfg, "group_sparse_pls requires params$group_assignment");
        int gn = 0;
        int32_t* groups = coerce_int32_vec(g, &gn);
        if (gn != p) cleanup_err(ctx, cfg, "group_assignment must have length ncol(X)");
        double gl = get_double(params, "group_lambda", 0.05);
        st = n4m_group_sparse_pls_fit(ctx, cfg, &Xv, &Yv, groups, gn, gl, &mr);
        if (st == N4M_OK) {
            static const char* sc[] = {"rmse", "n_groups", NULL};
            out = pack_result(mr, REG_DMAT, NULL, NULL, sc);
        }
    } else if (strcmp(algo, "fused_sparse_pls") == 0) {
        double l1 = get_double(params, "l1_lambda", 0.05);
        double fl = get_double(params, "fusion_lambda", 0.05);
        st = n4m_fused_sparse_pls_fit(ctx, cfg, &Xv, &Yv, l1, fl, &mr);
        if (st == N4M_OK) {
            static const char* sc[] = {"rmse", "l1_lambda", "fusion_lambda", NULL};
            out = pack_result(mr, REG_DMAT, NULL, NULL, sc);
        }
    } else if (strcmp(algo, "so_pls") == 0 ||
                strcmp(algo, "rosa") == 0 ||
                strcmp(algo, "on_pls") == 0) {
        SEXP bs = get_list_element(params, "block_sizes");
        if (bs == R_NilValue) cleanup_err(ctx, cfg, "%s requires params$block_sizes", algo);
        int bsn = 0;
        int64_t* bsv = coerce_int64_vec(bs, &bsn);
        int64_t bsum = 0;
        for (int i = 0; i < bsn; ++i) {
            if (bsv[i] <= 0)
                cleanup_err(ctx, cfg,
                             "block_sizes[%d] must be positive (got %lld)",
                             i + 1, (long long)bsv[i]);
            bsum += bsv[i];
        }
        if (bsum != (int64_t)p)
            cleanup_err(ctx, cfg,
                         "sum(block_sizes)=%lld must equal ncol(X)=%d",
                         (long long)bsum, p);
        /* Build per-block row-major copies. */
        n4m_matrix_view_t* blocks = (n4m_matrix_view_t*)
            R_alloc((size_t)bsn, sizeof(n4m_matrix_view_t));
        int64_t col_off = 0;
        for (int b = 0; b < bsn; ++b) {
            int64_t bcols = bsv[b];
            double* buf = (double*)R_alloc((size_t)n * (size_t)bcols, sizeof(double));
            for (int i = 0; i < n; ++i)
                for (int64_t j = 0; j < bcols; ++j)
                    buf[i * bcols + j] = REAL(X_rm)[i * p + col_off + j];
            n4m_matrix_view_init_rowmajor(&blocks[b], buf, n, bcols, N4M_DTYPE_F64);
            col_off += bcols;
        }
        if (strcmp(algo, "so_pls") == 0) {
            SEXP ncpb_sexp = get_list_element(params, "n_components_per_block");
            if (ncpb_sexp == R_NilValue)
                cleanup_err(ctx, cfg, "so_pls requires params$n_components_per_block");
            int ncn = 0;
            int32_t* ncpb = coerce_int32_vec(ncpb_sexp, &ncn);
            if (ncn != bsn) cleanup_err(ctx, cfg, "n_components_per_block must have length n_blocks");
            st = n4m_so_pls_fit(ctx, cfg, blocks, bsn, &Yv, ncpb, ncn, &mr);
        } else if (strcmp(algo, "rosa") == 0) {
            st = n4m_rosa_fit(ctx, cfg, blocks, bsn, &Yv, n_components, &mr);
        } else {
            int njoint = get_int(params, "n_joint", 1);
            SEXP upb_sexp = get_list_element(params, "n_unique_per_block");
            if (upb_sexp == R_NilValue)
                cleanup_err(ctx, cfg, "on_pls requires params$n_unique_per_block");
            int upbn = 0;
            int32_t* upb = coerce_int32_vec(upb_sexp, &upbn);
            if (upbn != bsn) cleanup_err(ctx, cfg, "n_unique_per_block must have length n_blocks");
            st = n4m_on_pls_fit(ctx, cfg, blocks, bsn, njoint, upb, upbn, &mr);
        }
        if (st == N4M_OK) {
            static const char* dm[] = {"predictions", "y_mean",
                                          "joint_loadings_0",
                                          "unique_loadings_0",
                                          "joint_loadings_1",
                                          "unique_loadings_1",
                                          "joint_loadings_2",
                                          "unique_loadings_2", NULL};
            static const char* sc[] = {"n_blocks", "n_components", NULL};
            out = pack_result(mr, dm, NULL, NULL, sc);
        }
    } else if (strcmp(algo, "bagging_pls") == 0) {
        int ne = get_int(params, "n_estimators", 50);
        uint64_t seed = get_u64(params, "seed", 0);
        st = n4m_bagging_pls_fit(ctx, cfg, &Xv, &Yv, ne, seed, &mr);
        if (st == N4M_OK) {
            static const char* sc[] = {"rmse", "n_estimators", NULL};
            out = pack_result(mr, REG_DMAT, NULL, NULL, sc);
        }
    } else if (strcmp(algo, "boosting_pls") == 0) {
        int ne = get_int(params, "n_estimators", 50);
        double lr = get_double(params, "learning_rate", 0.1);
        st = n4m_boosting_pls_fit(ctx, cfg, &Xv, &Yv, ne, lr, &mr);
        if (st == N4M_OK) {
            static const char* sc[] = {"rmse", "n_estimators", NULL};
            out = pack_result(mr, REG_DMAT, NULL, NULL, sc);
        }
    } else if (strcmp(algo, "random_subspace_pls") == 0) {
        int ne = get_int(params, "n_estimators", 50);
        int fps = get_int(params, "features_per_subspace", 10);
        uint64_t seed = get_u64(params, "seed", 0);
        st = n4m_random_subspace_pls_fit(ctx, cfg, &Xv, &Yv, ne, fps, seed, &mr);
        if (st == N4M_OK) {
            static const char* sc[] = {"rmse", "n_estimators", "features_per_subspace", NULL};
            out = pack_result(mr, REG_DMAT, NULL, NULL, sc);
        }
    } else if (strcmp(algo, "gpr_pls") == 0) {
        double ls = get_double(params, "length_scale", 1.0);
        double nl = get_double(params, "noise_level", 1e-4);
        uint64_t seed = get_u64(params, "seed", 0);
        st = n4m_gpr_pls_fit(ctx, cfg, &Xv, &Yv, ls, nl, seed, &mr);
        if (st == N4M_OK) {
            static const char* dm[] = {"rotation_r", "x_mean", "alpha", "L_lower",
                                          "training_scores", "predictions",
                                          "predictive_variance", NULL};
            static const char* sc[] = {"length_scale", "noise_level",
                                          "n_components", "y_mean", "rmse", NULL};
            out = pack_result(mr, dm, NULL, NULL, sc);
        }
    } else if (strcmp(algo, "pls_glm") == 0) {
        int poisson = get_int(params, "poisson", 0);
        st = n4m_pls_glm_fit(ctx, cfg, &Xv, &Yv, poisson, &mr);
        if (st == N4M_OK) {
            static const char* dm[] = {"coefficients", "intercept",
                                          "predictions", "x_mean", NULL};
            static const char* sc[] = {"rmse", "poisson", NULL};
            out = pack_result(mr, dm, NULL, NULL, sc);
        }
    } else if (strcmp(algo, "pls_qda") == 0) {
        SEXP yl = get_list_element(params, "y_labels");
        if (yl == R_NilValue) cleanup_err(ctx, cfg, "pls_qda requires params$y_labels");
        int nl = 0;
        int32_t* labels = coerce_int32_vec(yl, &nl);
        if (nl != n) cleanup_err(ctx, cfg, "y_labels must have length n");
        st = n4m_pls_qda_fit(ctx, cfg, &Xv, labels, nl, &mr);
        if (st == N4M_OK) {
            static const char* dm[] = {"class_means", "class_covariances",
                                          "log_class_priors", "rotations_r",
                                          "x_mean", "predictions", NULL};
            out = pack_result(mr, dm, NULL, NULL, REG_SCALAR);
        }
    } else if (strcmp(algo, "pls_cox") == 0) {
        SEXP sts_sexp = get_list_element(params, "survival_times");
        SEXP ev_sexp  = get_list_element(params, "event_indicators");
        if (sts_sexp == R_NilValue || TYPEOF(sts_sexp) != REALSXP ||
            ev_sexp == R_NilValue)
            cleanup_err(ctx, cfg, "pls_cox requires params$survival_times (numeric) "
                      "and params$event_indicators (integer)");
        if (Rf_length(sts_sexp) != n)
            cleanup_err(ctx, cfg, "survival_times must have length n");
        int en = 0;
        int32_t* ev = coerce_int32_vec(ev_sexp, &en);
        if (en != n) cleanup_err(ctx, cfg, "event_indicators must have length n");
        st = n4m_pls_cox_fit(ctx, cfg, &Xv, REAL(sts_sexp), Rf_length(sts_sexp),
                              ev, en, &mr);
        if (st == N4M_OK) {
            static const char* dm[] = {"coefficients", "baseline_hazard",
                                          "event_times", "x_mean",
                                          "predictions", NULL};
            out = pack_result(mr, dm, NULL, NULL, NULL);
        }
    } else if (strcmp(algo, "pds") == 0) {
        SEXP xt = get_list_element(params, "X_target");
        if (xt == R_NilValue || TYPEOF(xt) != REALSXP)
            cleanup_err(ctx, cfg, "pds requires params$X_target (numeric matrix)");
        SEXP xtd = Rf_getAttrib(xt, R_DimSymbol);
        int xtn = INTEGER(xtd)[0], xtp = INTEGER(xtd)[1];
        SEXP XT_rm = PROTECT(Rf_allocMatrix(REALSXP, xtn, xtp));
        colmajor_to_rowmajor(REAL(xt), REAL(XT_rm), xtn, xtp);
        int hw = get_int(params, "window_half_width", 2);
        n4m_matrix_view_t XTv;
        n4m_matrix_view_init_rowmajor(&XTv, REAL(XT_rm), xtn, xtp, N4M_DTYPE_F64);
        st = n4m_pds_fit(ctx, &Xv, &XTv, hw, &mr);
        UNPROTECT(1);
        if (st == N4M_OK) {
            static const char* dm[] = {"transformation", "predictions", NULL};
            static const char* sc[] = {"rmse", "window_half_width", NULL};
            out = pack_result(mr, dm, NULL, NULL, sc);
        }
    } else if (strcmp(algo, "ds") == 0) {
        SEXP xt = get_list_element(params, "X_target");
        if (xt == R_NilValue || TYPEOF(xt) != REALSXP)
            cleanup_err(ctx, cfg, "ds requires params$X_target (numeric matrix)");
        SEXP xtd = Rf_getAttrib(xt, R_DimSymbol);
        int xtn = INTEGER(xtd)[0], xtp = INTEGER(xtd)[1];
        SEXP XT_rm = PROTECT(Rf_allocMatrix(REALSXP, xtn, xtp));
        colmajor_to_rowmajor(REAL(xt), REAL(XT_rm), xtn, xtp);
        n4m_matrix_view_t XTv;
        n4m_matrix_view_init_rowmajor(&XTv, REAL(XT_rm), xtn, xtp, N4M_DTYPE_F64);
        st = n4m_ds_fit(ctx, &Xv, &XTv, &mr);
        UNPROTECT(1);
        if (st == N4M_OK) {
            static const char* dm[] = {"transformation", "bias",
                                          "predictions", NULL};
            out = pack_result(mr, dm, NULL, NULL, REG_SCALAR);
        }
    } else if (strcmp(algo, "mir_pls") == 0) {
        st = n4m_mir_pls_fit(ctx, cfg, &Xv, &Yv, &mr);
        if (st == N4M_OK) out = pack_result(mr, REG_DMAT, NULL, NULL, REG_SCALAR);
    } else if (strcmp(algo, "missing_aware_nipals") == 0) {
        st = n4m_missing_aware_nipals_fit(ctx, cfg, &Xv, &Yv, &mr);
        if (st == N4M_OK) out = pack_result(mr, REG_DMAT, NULL, NULL, REG_SCALAR);
    } else if (strcmp(algo, "mb_pls") == 0) {
        SEXP bs = get_list_element(params, "block_sizes");
        if (bs == R_NilValue) cleanup_err(ctx, cfg, "mb_pls requires params$block_sizes");
        int bsn = 0;
        int64_t* bsv = coerce_int64_vec(bs, &bsn);
        int64_t bsum = 0;
        for (int i = 0; i < bsn; ++i) {
            if (bsv[i] <= 0)
                cleanup_err(ctx, cfg,
                             "block_sizes[%d] must be positive (got %lld)",
                             i + 1, (long long)bsv[i]);
            bsum += bsv[i];
        }
        if (bsum != (int64_t)p)
            cleanup_err(ctx, cfg,
                         "sum(block_sizes)=%lld must equal ncol(X)=%d",
                         (long long)bsum, p);
        n4m_config_set_solver(cfg, N4M_SOLVER_NIPALS);
        st = n4m_mb_pls_fit(ctx, cfg, &Xv, &Yv, bsv, bsn, &mr);
        if (st == N4M_OK) {
            static const char* dm[] = {"coefficients", "predictions",
                                          "x_mean", "x_scale", "intercept",
                                          "block_weights", NULL};
            static const char* sc[] = {"rmse", "n_blocks", NULL};
            out = pack_result(mr, dm, NULL, NULL, sc);
        }
    } else if (strcmp(algo, "lw_pls") == 0) {
        int nn = get_int(params, "n_neighbors", 30);
        st = n4m_lw_pls_fit(ctx, cfg, &Xv, &Yv, nn, &mr);
        if (st == N4M_OK) {
            static const char* dm[] = {"predictions", "neighbor_indices", NULL};
            static const char* i64[] = {"neighbor_indices_i64", NULL};
            static const char* sc[] = {"n_neighbors", "n_components", "rmse", NULL};
            out = pack_result(mr, dm, NULL, i64, sc);
        }
    } else if (strcmp(algo, "pls_lda") == 0) {
        SEXP yl = get_list_element(params, "y_labels");
        if (yl == R_NilValue) cleanup_err(ctx, cfg, "pls_lda requires params$y_labels");
        int nl = 0;
        int32_t* labels = coerce_int32_vec(yl, &nl);
        int nc = get_int(params, "n_classes", 0);
        if (nc <= 0) {
            for (int i = 0; i < nl; ++i) if (labels[i] + 1 > nc) nc = labels[i] + 1;
        }
        st = n4m_pls_lda_fit(ctx, cfg, &Xv, labels, nl, nc, &mr);
        if (st == N4M_OK) {
            static const char* dm[] = {"decision_scores", NULL};
            static const char* iv[] = {"predictions", NULL};
            static const char* sc[] = {"n_classes", "n_components", NULL};
            out = pack_result(mr, dm, iv, NULL, sc);
        }
    } else if (strcmp(algo, "pls_logistic") == 0) {
        SEXP yl = get_list_element(params, "y_labels");
        if (yl == R_NilValue) cleanup_err(ctx, cfg, "pls_logistic requires params$y_labels");
        int nl = 0;
        int32_t* labels = coerce_int32_vec(yl, &nl);
        int nc = get_int(params, "n_classes", 0);
        if (nc <= 0) {
            for (int i = 0; i < nl; ++i) if (labels[i] + 1 > nc) nc = labels[i] + 1;
        }
        st = n4m_pls_logistic_fit(ctx, cfg, &Xv, labels, nl, nc, &mr);
        if (st == N4M_OK) {
            static const char* dm[] = {"decision_scores", "probabilities",
                                          "intercepts", "coefficients", NULL};
            static const char* iv[] = {"predictions", NULL};
            static const char* sc[] = {"n_classes", "n_components", NULL};
            out = pack_result(mr, dm, iv, NULL, sc);
        }
    } else if (strcmp(algo, "aom_preprocess") == 0) {
        int n_ops = get_int(params, "n_operators", 3);
        int mode = get_int(params, "gating_mode", 0);
        if (n_ops < 1) n_ops = 1;
        if (n_ops > 5) n_ops = 5;
        n4m_operator_kind_t kinds[5] = {
            N4M_OP_IDENTITY,
            N4M_OP_CENTER,
            N4M_OP_PARETO_SCALE,
            N4M_OP_AUTOSCALE,
            N4M_OP_SNV
        };
        n4m_operator_bank_t* bank = NULL;
        n4m_gating_strategy_t* gate = NULL;
        st = n4m_operator_bank_create(&bank);
        for (int i = 0; st == N4M_OK && i < n_ops; ++i) {
            st = n4m_operator_bank_add(bank, kinds[i], NULL, 0);
        }
        if (st == N4M_OK) {
            st = n4m_gating_strategy_create(
                &gate, (n4m_gating_mode_t)mode);
        }
        if (st == N4M_OK) {
            st = n4m_aom_preprocess_fit(ctx, bank, gate, &Xv, &Yv, &mr);
        }
        if (gate) n4m_gating_strategy_destroy(gate);
        if (bank) n4m_operator_bank_destroy(bank);
        if (st == N4M_OK) {
            static const char* dm[] = {"transformed", "operator_outputs",
                                          "weights", NULL};
            static const char* i64[] = {"operator_kinds", NULL};
            static const char* sc[] = {"n_operators", "mode", NULL};
            out = pack_result(mr, dm, NULL, i64, sc);
        }
    } else if (strcmp(algo, "aom_pls") == 0 ||
               strcmp(algo, "pop_pls") == 0) {
        int n_ops = get_int(params, "n_operators", 9);
        int cv = get_int(params, "cv", 3);
        int max_components = get_int(params, "max_components", n_components);
        if (max_components < 1) max_components = 1;
        if (max_components > p) max_components = p;
        if (max_components >= n) max_components = n - 1;
        if (cv < 2) cv = 2;

        n4m_operator_bank_t* bank = NULL;
        n4m_validation_plan_t* plan = NULL;
        st = make_compact_aom_bank(n_ops, &bank);
        if (st == N4M_OK) {
            plan = make_default_plan(n, cv);
            if (!plan) st = N4M_ERR_INVALID_ARGUMENT;
        }
        if (st == N4M_OK && strcmp(algo, "aom_pls") == 0) {
            n4m_aom_global_result_t* aom = NULL;
            st = n4m_aom_global_select(ctx, cfg, bank, &Xv, &Yv, plan,
                                        max_components, &aom);
            if (st == N4M_OK) out = pack_aom_global_result(aom);
        } else if (st == N4M_OK) {
            n4m_aom_per_component_result_t* pop = NULL;
            st = n4m_aom_per_component_select(ctx, cfg, bank, &Xv, &Yv, plan,
                                               max_components, &pop);
            if (st == N4M_OK) out = pack_aom_per_component_result(pop);
        }
        if (plan) n4m_validation_plan_destroy(plan);
        if (bank) n4m_operator_bank_destroy(bank);
    }

    /* ---------------- Selectors ---------------- */
    else if (strcmp(algo, "spa_select") == 0) {
        int tk = get_int(params, "top_k", 10);
        st = n4m_spa_select(ctx, cfg, &Xv, &Yv, tk, &mr);
        if (st == N4M_OK) {
            static const char* i64[] = {"selected_indices", NULL};
            static const char* sc[] = {"best_rmse", NULL};
            out = pack_result(mr, NULL, NULL, i64, sc);
        }
    } else if (strcmp(algo, "wvc_select") == 0) {
        int tk = get_int(params, "top_k", 10);
        int norm = get_int(params, "normalize", 1);
        st = n4m_wvc_select(ctx, &Xv, &Yv, n_components, tk, norm, &mr);
        if (st == N4M_OK) {
            static const char* dm[] = {"scores", NULL};
            static const char* i64[] = {"selected_indices", NULL};
            out = pack_result(mr, dm, NULL, i64, NULL);
        }
    } else if (strcmp(algo, "wvc_threshold_select") == 0) {
        int norm = get_int(params, "normalize", 1);
        double thr = get_double(params, "threshold", 0.0);
        double tf = get_double(params, "threshold_factor", 1.0);
        int ms = get_int(params, "min_selected", 1);
        st = n4m_wvc_threshold_select(ctx, &Xv, &Yv, n_components, norm, thr, tf, ms, &mr);
        if (st == N4M_OK) {
            static const char* dm[] = {"scores", NULL};
            static const char* i64[] = {"selected_indices", NULL};
            out = pack_result(mr, dm, NULL, i64, NULL);
        }
    } else if (strcmp(algo, "randomization_select") == 0) {
        int np = get_int(params, "n_permutations", 100);
        uint64_t seed = get_u64(params, "randomization_seed", 0);
        double a = get_double(params, "alpha", 0.05);
        st = n4m_randomization_select(ctx, cfg, &Xv, &Yv, np, seed, a, &mr);
        if (st == N4M_OK) {
            static const char* i64[] = {"selected_indices", NULL};
            static const char* sc[] = {"best_rmse", NULL};
            out = pack_result(mr, NULL, NULL, i64, sc);
        }
    } else if (strcmp(algo, "vip_spa_select") == 0) {
        double vt = get_double(params, "vip_threshold", 0.3);
        int tk = get_int(params, "top_k", 10);
        st = n4m_vip_spa_select(ctx, cfg, &Xv, &Yv, vt, tk, &mr);
        if (st == N4M_OK) {
            static const char* dm[] = {"vip_scores", "vip_mask", "selection_scores", NULL};
            static const char* i64[] = {"selected_indices", NULL};
            static const char* sc[] = {"n_selected", "vip_threshold", NULL};
            out = pack_result(mr, dm, NULL, i64, sc);
        }
    }
    /* Selectors that need a ValidationPlan share boilerplate via a macro. */
#define DISPATCH_PLAN_SELECT(name, call_args, dmat_keys, iv_keys, i64_keys, sc_keys) \
    else if (strcmp(algo, name) == 0) { \
        n4m_validation_plan_t* _plan = make_default_plan(n, 3); \
        if (!_plan) cleanup_err(ctx, cfg, name " could not build a default validation plan"); \
        st = call_args; \
        n4m_validation_plan_destroy(_plan); \
        if (st == N4M_OK) \
            out = pack_result(mr, dmat_keys, iv_keys, i64_keys, sc_keys); \
    }
    DISPATCH_PLAN_SELECT("cars_select",
        n4m_cars_select(ctx, cfg, &Xv, &Yv, _plan,
                         get_int(params, "n_iterations", 50),
                         get_int(params, "min_features", 5), &mr),
        NULL, NULL, SEL_IDX,
        SEL_RMSE)
    DISPATCH_PLAN_SELECT("interval_select",
        n4m_interval_select(ctx, cfg, &Xv, &Yv, _plan,
                             get_int(params, "interval_width", 10),
                             get_int(params, "step", 1), &mr),
        NULL, NULL, SEL_IDX,
        SEL_RMSE)
    DISPATCH_PLAN_SELECT("stability_select",
        n4m_stability_select(ctx, cfg, &Xv, &Yv, _plan,
                              get_int(params, "top_k", 10), &mr),
        NULL, NULL, SEL_IDX,
        SEL_RMSE)
    DISPATCH_PLAN_SELECT("uve_select",
        n4m_uve_select(ctx, cfg, &Xv, &Yv, _plan,
                        get_int(params, "noise_features", p),
                        get_u64(params, "noise_seed", 0), &mr),
        NULL, NULL, SEL_IDX,
        SEL_RMSE)
    DISPATCH_PLAN_SELECT("random_frog_select",
        n4m_random_frog_select(ctx, cfg, &Xv, &Yv, _plan,
                                get_int(params, "n_iterations", 100),
                                get_int(params, "initial_size", 30),
                                get_int(params, "min_size", n_components),
                                get_int(params, "max_size", p),
                                get_int(params, "top_k", 10),
                                get_u64(params, "seed", 0), &mr),
        NULL, NULL, SEL_IDX,
        SEL_RMSE)
    DISPATCH_PLAN_SELECT("scars_select",
        n4m_scars_select(ctx, cfg, &Xv, &Yv, _plan,
                          get_int(params, "n_iterations", 50),
                          get_int(params, "min_features", 5),
                          get_double(params, "sample_fraction", 0.8),
                          get_u64(params, "seed", 0), &mr),
        NULL, NULL, SEL_IDX,
        SEL_RMSE)
    DISPATCH_PLAN_SELECT("ga_select",
        n4m_ga_select(ctx, cfg, &Xv, &Yv, _plan,
                       get_int(params, "n_generations", 50),
                       get_int(params, "population_size", 50),
                       get_int(params, "min_features", n_components),
                       get_int(params, "max_features", p),
                       get_double(params, "mutation_rate", 0.01),
                       get_u64(params, "seed", 0), &mr),
        NULL, NULL, SEL_IDX,
        SEL_RMSE)
    DISPATCH_PLAN_SELECT("pso_select",
        n4m_pso_select(ctx, cfg, &Xv, &Yv, _plan,
                        get_int(params, "n_swarm", 30),
                        get_int(params, "n_iterations", 50),
                        get_double(params, "w", 0.729),
                        get_double(params, "c1", 1.494),
                        get_double(params, "c2", 1.494),
                        get_double(params, "v_max", 4.0),
                        get_u64(params, "seed", 0), &mr),
        PSO_DMAT,
        NULL, SEL_IDX,
        SEL_RMSE)
    DISPATCH_PLAN_SELECT("vissa_select",
        n4m_vissa_select(ctx, cfg, &Xv, &Yv, _plan,
                          get_int(params, "n_iterations", 20),
                          get_int(params, "n_submodels", 100),
                          get_double(params, "ratio_kept", 0.1),
                          get_double(params, "threshold", 0.5),
                          get_double(params, "floor_probability", 0.01),
                          get_u64(params, "seed", 0), &mr),
        VISSA_DMAT,
        NULL, SEL_IDX,
        SEL_RMSE)
    DISPATCH_PLAN_SELECT("shaving_select",
        n4m_shaving_select(ctx, cfg, &Xv, &Yv, _plan,
                            get_int(params, "n_steps", 10),
                            get_int(params, "min_features", 5),
                            get_double(params, "shave_fraction", 0.1), &mr),
        NULL, NULL, SEL_IDX,
        SEL_RMSE)
    DISPATCH_PLAN_SELECT("bve_select",
        n4m_bve_select(ctx, cfg, &Xv, &Yv, _plan,
                        get_int(params, "n_steps", 10),
                        get_int(params, "min_features", 5), &mr),
        NULL, NULL, SEL_IDX,
        SEL_RMSE)
    DISPATCH_PLAN_SELECT("emcuve_select",
        n4m_emcuve_select(ctx, cfg, &Xv, &Yv, _plan,
                           get_int(params, "noise_features", p),
                           get_u64(params, "noise_seed", 0),
                           get_int(params, "n_ensembles", 5),
                           get_double(params, "vote_threshold", 0.5), &mr),
        NULL, NULL, SEL_IDX,
        SEL_RMSE)
    DISPATCH_PLAN_SELECT("bipls_select",
        n4m_bipls_select(ctx, cfg, &Xv, &Yv, _plan,
                          get_int(params, "interval_width", 10),
                          get_int(params, "min_intervals", 1), &mr),
        NULL, NULL, SEL_IDX,
        SEL_RMSE)
    DISPATCH_PLAN_SELECT("sipls_select",
        n4m_sipls_select(ctx, cfg, &Xv, &Yv, _plan,
                          get_int(params, "interval_width", 10),
                          get_int(params, "combination_size", 2), &mr),
        NULL, NULL, SEL_IDX,
        SEL_RMSE)
    DISPATCH_PLAN_SELECT("rep_select",
        n4m_rep_select(ctx, cfg, &Xv, &Yv, _plan,
                        get_int(params, "n_steps", 10),
                        get_int(params, "min_features", 5),
                        get_int(params, "remove_count", 1), &mr),
        NULL, NULL, SEL_IDX,
        SEL_RMSE)
    DISPATCH_PLAN_SELECT("ipw_select",
        n4m_ipw_select(ctx, cfg, &Xv, &Yv, _plan,
                        get_int(params, "n_iterations", 10),
                        get_int(params, "top_k", 10),
                        get_double(params, "damping", 0.5),
                        get_double(params, "weight_floor", 1e-6), &mr),
        NULL, NULL, SEL_IDX,
        SEL_RMSE)
    DISPATCH_PLAN_SELECT("iriv_select",
        n4m_iriv_select(ctx, cfg, &Xv, &Yv, _plan,
                         get_int(params, "max_rounds", 20),
                         get_u64(params, "seed", 0), &mr),
        IRIV_DMAT,
        NULL, SEL_IDX,
        IRIV_SC)
    DISPATCH_PLAN_SELECT("irf_select",
        n4m_irf_select(ctx, cfg, &Xv, &Yv, _plan,
                        get_int(params, "n_iterations", 100),
                        get_int(params, "window_size", 10),
                        get_int(params, "initial_intervals", 10),
                        get_int(params, "top_k", 5),
                        get_u64(params, "seed", 0), &mr),
        IRF_DMAT,
        NULL, SEL_IDX_TKI,
        SEL_RMSE)

    /* Selectors with vector params (thresholds): plan + array. */
    else if (strcmp(algo, "t2_select") == 0) {
        SEXP thr = get_list_element(params, "alpha_thresholds");
        if (thr == R_NilValue || TYPEOF(thr) != REALSXP)
            cleanup_err(ctx, cfg, "t2_select requires numeric params$alpha_thresholds");
        int ms = get_int(params, "min_selected", n_components);
        n4m_validation_plan_t* _plan = make_default_plan(n, 3);
        if (!_plan) cleanup_err(ctx, cfg, "t2_select plan creation failed");
        st = n4m_t2_select(ctx, cfg, &Xv, &Yv, _plan,
                            REAL(thr), Rf_length(thr), ms, &mr);
        n4m_validation_plan_destroy(_plan);
        if (st == N4M_OK) {
            static const char* i64[] = {"selected_indices", NULL};
            static const char* sc[] = {"best_rmse", NULL};
            out = pack_result(mr, NULL, NULL, i64, sc);
        }
    } else if (strcmp(algo, "st_select") == 0) {
        SEXP thr = get_list_element(params, "thresholds");
        if (thr == R_NilValue || TYPEOF(thr) != REALSXP)
            cleanup_err(ctx, cfg, "st_select requires numeric params$thresholds");
        int ms = get_int(params, "min_selected", n_components);
        n4m_validation_plan_t* _plan = make_default_plan(n, 3);
        if (!_plan) cleanup_err(ctx, cfg, "st_select plan creation failed");
        st = n4m_st_select(ctx, cfg, &Xv, &Yv, _plan,
                            REAL(thr), Rf_length(thr), ms, &mr);
        n4m_validation_plan_destroy(_plan);
        if (st == N4M_OK) {
            static const char* i64[] = {"selected_indices", NULL};
            static const char* sc[] = {"best_rmse", NULL};
            out = pack_result(mr, NULL, NULL, i64, sc);
        }
    }

    /* ---------------- Diagnostics ---------------- */
    else if (strcmp(algo, "approximate_press_compute") == 0) {
        st = n4m_approximate_press_compute(ctx, cfg, &Xv, &Yv, n_components, &mr);
        if (st == N4M_OK) {
            static const char* dm[] = {"press_per_component",
                                          "rmse_per_component", NULL};
            static const char* iv[] = {"selected_n_components", NULL};
            out = pack_result(mr, dm, iv, NULL, NULL);
        }
    } else if (strcmp(algo, "one_se_rule_compute") == 0) {
        SEXP fr = get_list_element(params, "fold_rmse_matrix");
        if (fr == R_NilValue || TYPEOF(fr) != REALSXP)
            cleanup_err(ctx, cfg, "one_se_rule_compute requires numeric matrix params$fold_rmse_matrix");
        SEXP fr_dim = Rf_getAttrib(fr, R_DimSymbol);
        int max_k = INTEGER(fr_dim)[0], n_folds = INTEGER(fr_dim)[1];
        double* fr_rm = (double*)R_alloc((size_t)max_k * n_folds, sizeof(double));
        colmajor_to_rowmajor(REAL(fr), fr_rm, max_k, n_folds);
        st = n4m_one_se_rule_compute(ctx, fr_rm, max_k, n_folds, &mr);
        if (st == N4M_OK) {
            static const char* dm[] = {"mean_rmse_per_component", NULL};
            static const char* iv[] = {"best_n_components", "one_se_n_components", NULL};
            static const char* sc[] = {"one_se_standard_error", "one_se_threshold", NULL};
            out = pack_result(mr, dm, iv, NULL, sc);
        }
    } else {
        n4m_config_destroy(cfg);
        UNPROTECT(2);
        r_throw("unknown algo", N4M_ERR_INVALID_ARGUMENT, ctx);
    }

    n4m_config_destroy(cfg);
    UNPROTECT(2);  /* X_rm, Y_rm */
    if (st != N4M_OK) r_throw(algo, st, ctx);
    n4m_context_destroy(ctx);
    return out;
}
