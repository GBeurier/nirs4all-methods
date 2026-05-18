/* SPDX-License-Identifier: CECILL-2.1
 *
 * MATLAB / Octave MEX dispatcher covering ALL MethodResult-returning
 * entry points of libp4a v1.1.0 + selectors + diagnostics.
 *
 *   out_struct = p4a_method_fit_mex(algo, X, Y, n_components, params)
 *
 * `algo` is one of (see ALGO_TABLE below):
 *
 *   MethodResult fits:
 *     "sparse_simpls" "cppls" "ecr" "di_pls" "weighted_pls" "robust_pls"
 *     "ridge_pls" "continuum_regression" "recursive_pls" "n_pls"
 *     "kernel_pls" "o2pls" "sparse_pls_da" "group_sparse_pls"
 *     "fused_sparse_pls" "so_pls" "on_pls" "rosa" "bagging_pls"
 *     "boosting_pls" "random_subspace_pls" "gpr_pls" "pls_glm"
 *     "pls_qda" "pls_cox" "pds" "ds" "mir_pls" "missing_aware_nipals"
 *     "mb_pls" "lw_pls" "pls_lda" "pls_logistic"
 *
 *   Selectors:
 *     "variable_select_rank" "interval_select" "stability_select"
 *     "uve_select" "spa_select" "cars_select" "random_frog_select"
 *     "scars_select" "ga_select" "pso_select" "vissa_select"
 *     "shaving_select" "bve_select" "t2_select" "wvc_select"
 *     "wvc_threshold_select" "emcuve_select" "randomization_select"
 *     "bipls_select" "sipls_select" "rep_select" "ipw_select"
 *     "st_select" "iriv_select" "irf_select" "vip_spa_select"
 *
 *   Diagnostics:
 *     "pls_diagnostics_compute" "approximate_press_compute"
 *     "pls_monitoring_run" "one_se_rule_compute"
 *
 * `params` is a struct whose fields supply algorithm-specific
 * parameters and required side inputs (sample_weights, block_sizes,
 * X_target, y_labels, …). Each algorithm documents its expected
 * fields below.
 *
 * Returns a struct with the MethodResult arrays + scalars converted
 * to MATLAB-friendly types (column-major matrices, row-vectors, scalar
 * doubles, MATLAB 1-based indices for *_indices fields).
 *
 * The dispatcher trades a single string compare per fit for the
 * convenience of not accruing 60+ separate MEX entry points.
 */

#include "mex.h"
#include "pls4all/p4a.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- buffer helpers (column-major mxArray <-> row-major C) ---------- */

static double* colmajor_to_rowmajor_alloc(const mxArray *mx, int *rows_out,
                                            int *cols_out) {
    int rows = (int)mxGetM(mx);
    int cols = (int)mxGetN(mx);
    *rows_out = rows;
    *cols_out = cols;
    if (rows * cols == 0) return NULL;
    const double *src = mxGetPr(mx);
    double *dst = (double *)malloc((size_t)rows * (size_t)cols * sizeof(double));
    if (!dst) return NULL;
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            dst[i * cols + j] = src[i + j * rows];
    return dst;
}

static mxArray* rowmajor_to_colmajor_mx(const double *src, int rows, int cols) {
    mxArray *out = mxCreateDoubleMatrix(rows, cols, mxREAL);
    double *dst = mxGetPr(out);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            dst[i + j * rows] = src[i * cols + j];
    return out;
}

/* ---- MethodResult → struct field conversion ------------------------- */

static mxArray* mr_matrix(const p4a_method_result_t *mr, const char *name) {
    const double *data = NULL;
    int64_t rows = 0, cols = 0;
    p4a_status_t st = p4a_method_result_get_double_matrix(mr, name,
                                                          &data, &rows, &cols);
    if (st != P4A_OK || !data) return mxCreateDoubleMatrix(0, 0, mxREAL);
    return rowmajor_to_colmajor_mx(data, (int)rows, (int)cols);
}

static mxArray* mr_int_vector(const p4a_method_result_t *mr, const char *name) {
    const int32_t *data = NULL;
    int32_t size = 0;
    p4a_status_t st = p4a_method_result_get_int_vector(mr, name, &data, &size);
    if (st != P4A_OK || !data) return mxCreateDoubleMatrix(0, 0, mxREAL);
    mxArray *mx = mxCreateDoubleMatrix(1, size, mxREAL);
    double *dst = mxGetPr(mx);
    for (int32_t i = 0; i < size; ++i) dst[i] = (double)data[i];
    return mx;
}

/* Explicit allow-list of int64 fields that hold 0-based row / feature /
 * interval indices and therefore need a +1 shift to expose 1-based
 * MATLAB indices. Anything not on this list is returned verbatim. */
static int is_index_int64_name(const char *name) {
    static const char *idx_names[] = {
        "selected_indices",
        "neighbor_indices_i64",
        "top_k_intervals",
        "removed_indices",
        NULL,
    };
    for (int i = 0; idx_names[i]; ++i)
        if (strcmp(name, idx_names[i]) == 0) return 1;
    return 0;
}

static mxArray* mr_int64_vector(const p4a_method_result_t *mr, const char *name) {
    const int64_t *data = NULL;
    int64_t size = 0;
    p4a_status_t st = p4a_method_result_get_int64_vector(mr, name, &data, &size);
    if (st != P4A_OK || !data) return mxCreateDoubleMatrix(0, 0, mxREAL);
    mxArray *mx = mxCreateDoubleMatrix(1, (int)size, mxREAL);
    double *dst = mxGetPr(mx);
    int is_index_field = is_index_int64_name(name);
    for (int64_t i = 0; i < size; ++i)
        dst[i] = (double)(is_index_field ? (data[i] + 1) : data[i]);
    return mx;
}

static mxArray* mr_scalar(const p4a_method_result_t *mr, const char *name) {
    double v = 0.0;
    p4a_status_t st = p4a_method_result_get_scalar(mr, name, &v);
    if (st != P4A_OK) return mxCreateDoubleMatrix(0, 0, mxREAL);
    return mxCreateDoubleScalar(v);
}

/* ---- Common Config builder ------------------------------------------ */

static p4a_config_t* basic_cfg(int n_components) {
    p4a_config_t *cfg = NULL;
    if (p4a_config_create(&cfg) != P4A_OK)
        mexErrMsgIdAndTxt("pls4all:cfg", "p4a_config_create failed");
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
    p4a_config_set_store_scores(cfg, 0);
    return cfg;
}

/* ---- struct-field extraction (with defaults) ------------------------ */

static double get_scalar_field(const mxArray *params, const char *name,
                                double dflt) {
    if (!params || !mxIsStruct(params)) return dflt;
    mxArray *f = mxGetField(params, 0, name);
    if (!f || mxGetNumberOfElements(f) == 0) return dflt;
    return mxGetScalar(f);
}

static int get_int_field(const mxArray *params, const char *name, int dflt) {
    return (int)get_scalar_field(params, name, (double)dflt);
}

static uint64_t get_u64_field(const mxArray *params, const char *name,
                                uint64_t dflt) {
    if (!params || !mxIsStruct(params)) return dflt;
    mxArray *f = mxGetField(params, 0, name);
    if (!f || mxGetNumberOfElements(f) == 0) return dflt;
    return (uint64_t)mxGetScalar(f);
}

static const mxArray* get_array_field(const mxArray *params, const char *name) {
    if (!params || !mxIsStruct(params)) return NULL;
    return mxGetField(params, 0, name);
}

/* Get an int32 vector field, allocated via mxMalloc so it survives
 * normal MEX cleanup. Sets *n_out to the length. Returns NULL when the
 * field is absent or wrong type. */
static int32_t* get_int32_vec_field(const mxArray *params, const char *name,
                                      int *n_out) {
    *n_out = 0;
    const mxArray *f = get_array_field(params, name);
    if (!f) return NULL;
    int n = (int)mxGetNumberOfElements(f);
    int32_t *out = (int32_t *)mxMalloc((size_t)n * sizeof(int32_t));
    if (mxIsDouble(f)) {
        const double *src = mxGetPr(f);
        for (int i = 0; i < n; ++i) out[i] = (int32_t)src[i];
    } else if (mxIsInt32(f)) {
        const int32_t *src = (const int32_t *)mxGetData(f);
        memcpy(out, src, (size_t)n * sizeof(int32_t));
    } else {
        mxFree(out);
        return NULL;
    }
    *n_out = n;
    return out;
}

/* Same as int32 but emits int64. Accepts double / int32 / int64 inputs. */
static int64_t* get_int64_vec_field(const mxArray *params, const char *name,
                                      int *n_out) {
    *n_out = 0;
    const mxArray *f = get_array_field(params, name);
    if (!f) return NULL;
    int n = (int)mxGetNumberOfElements(f);
    int64_t *out = (int64_t *)mxMalloc((size_t)n * sizeof(int64_t));
    if (mxIsDouble(f)) {
        const double *src = mxGetPr(f);
        for (int i = 0; i < n; ++i) out[i] = (int64_t)src[i];
    } else if (mxIsInt32(f)) {
        const int32_t *src = (const int32_t *)mxGetData(f);
        for (int i = 0; i < n; ++i) out[i] = (int64_t)src[i];
    } else if (mxIsInt64(f)) {
        const int64_t *src = (const int64_t *)mxGetData(f);
        memcpy(out, src, (size_t)n * sizeof(int64_t));
    } else {
        mxFree(out);
        return NULL;
    }
    *n_out = n;
    return out;
}

/* Build a deterministic k-fold contiguous-blocks ValidationPlan over
 * n samples. Auto-clamps requested k to min(requested, n/2) so very small
 * datasets still get at least 2 folds. Returns NULL on alloc / API
 * failure or when n < 4 (not enough rows for a meaningful CV).
 * Caller owns the result and must release it via
 * p4a_validation_plan_destroy. */
static p4a_validation_plan_t* make_default_plan(int n, int requested_folds) {
    if (n < 4) return NULL;
    int n_folds = requested_folds;
    if (n_folds > n / 2) n_folds = n / 2;
    if (n_folds < 2) n_folds = 2;
    p4a_validation_plan_t *plan = NULL;
    if (p4a_validation_plan_create(&plan) != P4A_OK) return NULL;
    if (p4a_validation_plan_set_n_samples(plan, (int64_t)n) != P4A_OK) {
        p4a_validation_plan_destroy(plan);
        return NULL;
    }
    int fold_size = (n + n_folds - 1) / n_folds;
    int64_t *train = (int64_t *)mxMalloc((size_t)n * sizeof(int64_t));
    int64_t *test  = (int64_t *)mxMalloc((size_t)n * sizeof(int64_t));
    for (int f = 0; f < n_folds; ++f) {
        int t_start = f * fold_size;
        int t_end = t_start + fold_size;
        if (t_end > n) t_end = n;
        int n_test = t_end - t_start;
        int n_train = n - n_test;
        int ti = 0, te = 0;
        for (int i = 0; i < n; ++i) {
            if (i >= t_start && i < t_end) test[te++] = (int64_t)i;
            else                            train[ti++] = (int64_t)i;
        }
        if (p4a_validation_plan_add_fold(plan, train, ti, test, te) != P4A_OK) {
            mxFree(train); mxFree(test);
            p4a_validation_plan_destroy(plan);
            return NULL;
        }
    }
    mxFree(train); mxFree(test);
    return plan;
}

/* Get a double vector field (e.g. thresholds, sample_weights). */
static const double* get_double_vec_field(const mxArray *params,
                                            const char *name, int *n_out) {
    *n_out = 0;
    const mxArray *f = get_array_field(params, name);
    if (!f || !mxIsDouble(f)) return NULL;
    *n_out = (int)mxGetNumberOfElements(f);
    return mxGetPr(f);
}

/* ---- Result packagers ------------------------------------------------ */

/* Pack a MethodResult into a struct using key lists. `dmat_keys`,
 * `iv_keys`, `i64_keys`, `scalar_keys` are NULL-terminated arrays of
 * field names. The result destroys `mr` on the way out. */
static mxArray* pack_result(p4a_method_result_t *mr,
                              const char **dmat, const char **iv,
                              const char **i64, const char **scalar) {
    int n_dmat = 0, n_iv = 0, n_i64 = 0, n_sc = 0;
    while (dmat && dmat[n_dmat]) n_dmat++;
    while (iv && iv[n_iv]) n_iv++;
    while (i64 && i64[n_i64]) n_i64++;
    while (scalar && scalar[n_sc]) n_sc++;
    int total = n_dmat + n_iv + n_i64 + n_sc;
    const char **names = (const char **)mxMalloc(sizeof(char*) * (size_t)total);
    int idx = 0;
    for (int i = 0; i < n_dmat; ++i)   names[idx++] = dmat[i];
    for (int i = 0; i < n_iv; ++i)     names[idx++] = iv[i];
    for (int i = 0; i < n_i64; ++i)    names[idx++] = i64[i];
    for (int i = 0; i < n_sc; ++i)     names[idx++] = scalar[i];
    mxArray *out = mxCreateStructMatrix(1, 1, total, names);
    mxFree(names);
    int field = 0;
    for (int i = 0; i < n_dmat; ++i)
        mxSetFieldByNumber(out, 0, field++, mr_matrix(mr, dmat[i]));
    for (int i = 0; i < n_iv; ++i)
        mxSetFieldByNumber(out, 0, field++, mr_int_vector(mr, iv[i]));
    for (int i = 0; i < n_i64; ++i)
        mxSetFieldByNumber(out, 0, field++, mr_int64_vector(mr, i64[i]));
    for (int i = 0; i < n_sc; ++i)
        mxSetFieldByNumber(out, 0, field++, mr_scalar(mr, scalar[i]));
    p4a_method_result_destroy(mr);
    return out;
}

/* Throw with the C ABI's per-context error if present. Always destroys
 * ctx + cfg + X/Y buffers before longjmp via mexErrMsgIdAndTxt. */
static void throw_status(p4a_status_t st, const char *algo,
                          p4a_context_t *ctx, p4a_config_t *cfg,
                          double *X, double *Y) {
    char buf[512];
    const char *msg = ctx ? p4a_context_last_error(ctx) : NULL;
    if (msg && msg[0]) {
        snprintf(buf, sizeof(buf), "%s failed: %s (%s)", algo,
                  p4a_status_to_string(st), msg);
    } else {
        snprintf(buf, sizeof(buf), "%s failed: %s", algo,
                  p4a_status_to_string(st));
    }
    if (cfg) p4a_config_destroy(cfg);
    if (ctx) p4a_context_destroy(ctx);
    if (X) free(X);
    if (Y) free(Y);
    mexErrMsgIdAndTxt("pls4all:fit", "%s", buf);
}

/* ---- mexFunction ----------------------------------------------------- */

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    (void)nlhs;
    if (nrhs < 4) {
        mexErrMsgIdAndTxt("pls4all:nargin",
                          "Usage: p4a_method_fit_mex(algo, X, Y, n_components, params)");
    }
    if (!mxIsChar(prhs[0]))
        mexErrMsgIdAndTxt("pls4all:type", "algo must be a string");
    if (!mxIsDouble(prhs[1]) || !mxIsDouble(prhs[2]))
        mexErrMsgIdAndTxt("pls4all:type", "X and Y must be real double matrices");
    if (mxIsComplex(prhs[1]) || mxIsComplex(prhs[2]))
        mexErrMsgIdAndTxt("pls4all:type", "X and Y must be real (no complex part)");

    char algo[80];
    if (mxGetString(prhs[0], algo, sizeof(algo)) != 0)
        mexErrMsgIdAndTxt("pls4all:str", "algo name too long");
    int n_components = (int)mxGetScalar(prhs[3]);
    if (n_components < 1)
        mexErrMsgIdAndTxt("pls4all:bad_arg", "n_components must be >= 1");
    const mxArray *params = (nrhs >= 5) ? prhs[4] : NULL;

    int n, p, ny, q;
    double *X = colmajor_to_rowmajor_alloc(prhs[1], &n, &p);
    double *Y = colmajor_to_rowmajor_alloc(prhs[2], &ny, &q);
    if (!X || !Y) {
        free(X); free(Y);
        mexErrMsgIdAndTxt("pls4all:oom", "out of memory copying X / Y");
    }
    if (n != ny) {
        free(X); free(Y);
        mexErrMsgIdAndTxt("pls4all:shape", "size(X,1) must equal size(Y,1)");
    }

    p4a_context_t *ctx = NULL;
    if (p4a_context_create(&ctx) != P4A_OK) {
        free(X); free(Y);
        mexErrMsgIdAndTxt("pls4all:ctx", "p4a_context_create failed");
    }
    p4a_config_t *cfg = basic_cfg(n_components);

    p4a_matrix_view_t Xv, Yv;
    p4a_matrix_view_init_rowmajor(&Xv, X, n, p, P4A_DTYPE_F64);
    p4a_matrix_view_init_rowmajor(&Yv, Y, n, q, P4A_DTYPE_F64);

    p4a_method_result_t *mr = NULL;
    p4a_status_t st = P4A_OK;
    mxArray *out = NULL;

    /* Standard regressor output key sets. */
    static const char *REG_DMAT[] = {"coefficients", "predictions",
                                       "x_mean", "y_mean", NULL};
    static const char *REG_SCALAR[] = {"rmse", NULL};

    /* ============================================================
     *  Model-result fits — each block: call C, then pack.
     * ============================================================ */

    if (strcmp(algo, "sparse_simpls") == 0) {
        double lambda = get_scalar_field(params, "sparsity_lambda", 0.05);
        st = p4a_sparse_simpls_fit(ctx, cfg, &Xv, &Yv, lambda, &mr);
        if (st == P4A_OK) {
            static const char *dm[] = {"coefficients", "predictions", "x_mean",
                                          "y_mean", "weights_w", NULL};
            out = pack_result(mr, dm, NULL, NULL, REG_SCALAR);
        }
    } else if (strcmp(algo, "cppls") == 0) {
        double gamma = get_scalar_field(params, "gamma", 0.5);
        st = p4a_cppls_fit(ctx, cfg, &Xv, &Yv, gamma, &mr);
        if (st == P4A_OK) out = pack_result(mr, REG_DMAT, NULL, NULL, REG_SCALAR);
    } else if (strcmp(algo, "ecr") == 0) {
        double alpha = get_scalar_field(params, "alpha", 0.5);
        st = p4a_ecr_fit(ctx, cfg, &Xv, &Yv, alpha, &mr);
        if (st == P4A_OK) {
            static const char *dm[] = {"coefficients", "predictions",
                                          "x_mean", "y_mean", "x_scale", "y_scale",
                                          "weights_w", "loadings_p", "y_loadings",
                                          "wstar", "r2x", "r2y", NULL};
            static const char *sc[] = {"alpha", "rmse", NULL};
            out = pack_result(mr, dm, NULL, NULL, sc);
        }
    } else if (strcmp(algo, "di_pls") == 0) {
        const mxArray *xt = get_array_field(params, "X_target");
        if (!xt || !mxIsDouble(xt))
            throw_status(P4A_ERR_INVALID_ARGUMENT, "di_pls (params.X_target)",
                          ctx, cfg, X, Y);
        int xt_rows = (int)mxGetM(xt), xt_cols = (int)mxGetN(xt);
        if (xt_cols != p)
            throw_status(P4A_ERR_SHAPE_MISMATCH, "di_pls (X_target ncol)",
                          ctx, cfg, X, Y);
        double *XT = colmajor_to_rowmajor_alloc(xt, &xt_rows, &xt_cols);
        double di_lambda = get_scalar_field(params, "di_lambda", 1.0);
        p4a_matrix_view_t XTv;
        p4a_matrix_view_init_rowmajor(&XTv, XT, xt_rows, xt_cols, P4A_DTYPE_F64);
        st = p4a_di_pls_fit(ctx, cfg, &Xv, &Yv, &XTv, di_lambda, &mr);
        free(XT);
        if (st == P4A_OK) out = pack_result(mr, REG_DMAT, NULL, NULL, REG_SCALAR);
    } else if (strcmp(algo, "weighted_pls") == 0) {
        int wn = 0;
        const double *w = get_double_vec_field(params, "sample_weights", &wn);
        if (!w || wn != n)
            throw_status(P4A_ERR_SHAPE_MISMATCH, "weighted_pls (sample_weights)",
                          ctx, cfg, X, Y);
        st = p4a_weighted_pls_fit(ctx, cfg, &Xv, &Yv, w, wn, &mr);
        if (st == P4A_OK) out = pack_result(mr, REG_DMAT, NULL, NULL, REG_SCALAR);
    } else if (strcmp(algo, "robust_pls") == 0) {
        double k = get_scalar_field(params, "huber_k", 1.345);
        int it = get_int_field(params, "max_irls_iter", 20);
        st = p4a_robust_pls_fit(ctx, cfg, &Xv, &Yv, k, it, &mr);
        if (st == P4A_OK) out = pack_result(mr, REG_DMAT, NULL, NULL, REG_SCALAR);
    } else if (strcmp(algo, "ridge_pls") == 0) {
        double l = get_scalar_field(params, "ridge_lambda", 1.0);
        st = p4a_ridge_pls_fit(ctx, cfg, &Xv, &Yv, l, &mr);
        if (st == P4A_OK) out = pack_result(mr, REG_DMAT, NULL, NULL, REG_SCALAR);
    } else if (strcmp(algo, "continuum_regression") == 0) {
        double t = get_scalar_field(params, "tau", 0.5);
        st = p4a_continuum_regression_fit(ctx, cfg, &Xv, &Yv, t, &mr);
        if (st == P4A_OK) out = pack_result(mr, REG_DMAT, NULL, NULL, REG_SCALAR);
    } else if (strcmp(algo, "recursive_pls") == 0) {
        int w = get_int_field(params, "window_size", 50);
        st = p4a_recursive_pls_run(ctx, cfg, &Xv, &Yv, w, &mr);
        if (st == P4A_OK) {
            static const char *dm[] = {"predictions", NULL};
            static const char *iv[] = {"in_window", NULL};
            static const char *sc[] = {"rmse_predictable", NULL};
            out = pack_result(mr, dm, iv, NULL, sc);
        }
    } else if (strcmp(algo, "n_pls") == 0) {
        int mj = get_int_field(params, "mode_j", 0);
        int mk = get_int_field(params, "mode_k", 0);
        if (mj <= 0 || mk <= 0 || mj * mk != p)
            throw_status(P4A_ERR_SHAPE_MISMATCH,
                          "n_pls (mode_j*mode_k must equal size(X,2))",
                          ctx, cfg, X, Y);
        st = p4a_n_pls_fit(ctx, cfg, &Xv, mj, mk, &Yv, &mr);
        if (st == P4A_OK) out = pack_result(mr, REG_DMAT, NULL, NULL, REG_SCALAR);
    } else if (strcmp(algo, "kernel_pls") == 0) {
        int kt = get_int_field(params, "kernel_type", 1);
        double gamma = get_scalar_field(params, "gamma", 0.0);
        double coef0 = get_scalar_field(params, "coef0", 1.0);
        int degree = get_int_field(params, "degree", 3);
        st = p4a_kernel_pls_fit(ctx, cfg, kt, gamma, coef0, degree,
                                 &Xv, &Yv, &mr);
        if (st == P4A_OK) {
            static const char *dm[] = {"predictions", "alpha", "y_mean", NULL};
            static const char *sc[] = {"rmse", "kernel_type", NULL};
            out = pack_result(mr, dm, NULL, NULL, sc);
        }
    } else if (strcmp(algo, "o2pls") == 0) {
        int np = get_int_field(params, "n_predictive", 2);
        int nx = get_int_field(params, "n_x_orthogonal", 1);
        int ny2 = get_int_field(params, "n_y_orthogonal", 1);
        st = p4a_o2pls_fit(ctx, cfg, &Xv, &Yv, np, nx, ny2, &mr);
        if (st == P4A_OK) {
            static const char *dm[] = {"coefficients", "predictions",
                                          "x_mean", "y_mean",
                                          "w_predictive", "c_predictive",
                                          "w_x_orthogonal", "c_y_orthogonal",
                                          "b_predictive", NULL};
            out = pack_result(mr, dm, NULL, NULL, REG_SCALAR);
        }
    } else if (strcmp(algo, "sparse_pls_da") == 0) {
        int nl = 0;
        int32_t *labels = get_int32_vec_field(params, "y_labels", &nl);
        if (!labels || nl != n)
            throw_status(P4A_ERR_SHAPE_MISMATCH,
                          "sparse_pls_da (y_labels length)",
                          ctx, cfg, X, Y);
        st = p4a_sparse_pls_da_fit(ctx, cfg, &Xv, labels, nl, &mr);
        mxFree(labels);
        if (st == P4A_OK) {
            static const char *dm[] = {"coefficients", "predictions",
                                          "x_mean", "y_mean",
                                          "class_priors", NULL};
            out = pack_result(mr, dm, NULL, NULL, REG_SCALAR);
        }
    } else if (strcmp(algo, "group_sparse_pls") == 0) {
        int gn = 0;
        int32_t *groups = get_int32_vec_field(params, "group_assignment", &gn);
        if (!groups || gn != p)
            throw_status(P4A_ERR_SHAPE_MISMATCH,
                          "group_sparse_pls (group_assignment length)",
                          ctx, cfg, X, Y);
        double gl = get_scalar_field(params, "group_lambda", 0.05);
        st = p4a_group_sparse_pls_fit(ctx, cfg, &Xv, &Yv, groups, gn, gl, &mr);
        mxFree(groups);
        if (st == P4A_OK) {
            static const char *sc[] = {"rmse", "n_groups", NULL};
            out = pack_result(mr, REG_DMAT, NULL, NULL, sc);
        }
    } else if (strcmp(algo, "fused_sparse_pls") == 0) {
        double l1 = get_scalar_field(params, "l1_lambda", 0.05);
        double fl = get_scalar_field(params, "fusion_lambda", 0.05);
        st = p4a_fused_sparse_pls_fit(ctx, cfg, &Xv, &Yv, l1, fl, &mr);
        if (st == P4A_OK) {
            static const char *sc[] = {"rmse", "l1_lambda", "fusion_lambda", NULL};
            out = pack_result(mr, REG_DMAT, NULL, NULL, sc);
        }
    } else if (strcmp(algo, "so_pls") == 0 ||
                strcmp(algo, "rosa") == 0 ||
                strcmp(algo, "on_pls") == 0) {
        /* Block fits — concatenated X with explicit block_sizes. */
        int bsn = 0;
        int64_t *bs = get_int64_vec_field(params, "block_sizes", &bsn);
        if (!bs || bsn < 1)
            throw_status(P4A_ERR_INVALID_ARGUMENT,
                          "block fits require params.block_sizes",
                          ctx, cfg, X, Y);
        int64_t bsum = 0;
        for (int i = 0; i < bsn; ++i) bsum += bs[i];
        if (bsum != (int64_t)p) {
            mxFree(bs);
            throw_status(P4A_ERR_SHAPE_MISMATCH,
                          "sum(block_sizes) must equal size(X, 2)",
                          ctx, cfg, X, Y);
        }
        /* Build the per-block matrix views from the row-major X buffer. */
        p4a_matrix_view_t *blocks = (p4a_matrix_view_t *)
            mxMalloc(sizeof(p4a_matrix_view_t) * (size_t)bsn);
        double **block_bufs = (double **)
            mxMalloc(sizeof(double *) * (size_t)bsn);
        int64_t col_off = 0;
        for (int b = 0; b < bsn; ++b) {
            int64_t bcols = bs[b];
            block_bufs[b] = (double *)mxMalloc((size_t)n * (size_t)bcols * sizeof(double));
            for (int i = 0; i < n; ++i)
                for (int64_t j = 0; j < bcols; ++j)
                    block_bufs[b][i * bcols + j] = X[i * p + col_off + j];
            p4a_matrix_view_init_rowmajor(&blocks[b], block_bufs[b],
                                            n, bcols, P4A_DTYPE_F64);
            col_off += bcols;
        }
        if (strcmp(algo, "so_pls") == 0) {
            int ncn = 0;
            int32_t *ncpb = get_int32_vec_field(params, "n_components_per_block", &ncn);
            if (!ncpb || ncn != bsn) {
                if (ncpb) mxFree(ncpb);
                for (int b = 0; b < bsn; ++b) mxFree(block_bufs[b]);
                mxFree(blocks); mxFree(block_bufs); mxFree(bs);
                throw_status(P4A_ERR_INVALID_ARGUMENT,
                              "so_pls requires n_components_per_block of length n_blocks",
                              ctx, cfg, X, Y);
            }
            st = p4a_so_pls_fit(ctx, cfg, blocks, bsn, &Yv, ncpb, ncn, &mr);
            mxFree(ncpb);
        } else if (strcmp(algo, "rosa") == 0) {
            st = p4a_rosa_fit(ctx, cfg, blocks, bsn, &Yv, n_components, &mr);
        } else {
            int njoint = get_int_field(params, "n_joint", 1);
            int upbn = 0;
            int32_t *upb = get_int32_vec_field(params, "n_unique_per_block", &upbn);
            if (!upb || upbn != bsn) {
                if (upb) mxFree(upb);
                for (int b = 0; b < bsn; ++b) mxFree(block_bufs[b]);
                mxFree(blocks); mxFree(block_bufs); mxFree(bs);
                throw_status(P4A_ERR_INVALID_ARGUMENT,
                              "on_pls requires n_unique_per_block of length n_blocks",
                              ctx, cfg, X, Y);
            }
            st = p4a_on_pls_fit(ctx, cfg, blocks, bsn, njoint, upb, upbn, &mr);
            mxFree(upb);
        }
        for (int b = 0; b < bsn; ++b) mxFree(block_bufs[b]);
        mxFree(blocks); mxFree(block_bufs); mxFree(bs);
        if (st == P4A_OK) {
            static const char *dm[] = {"predictions", "y_mean", NULL};
            static const char *sc[] = {"n_blocks", "n_components", NULL};
            out = pack_result(mr, dm, NULL, NULL, sc);
        }
    } else if (strcmp(algo, "bagging_pls") == 0) {
        int ne = get_int_field(params, "n_estimators", 50);
        uint64_t seed = get_u64_field(params, "seed", 0);
        st = p4a_bagging_pls_fit(ctx, cfg, &Xv, &Yv, ne, seed, &mr);
        if (st == P4A_OK) {
            static const char *sc[] = {"rmse", "n_estimators", NULL};
            out = pack_result(mr, REG_DMAT, NULL, NULL, sc);
        }
    } else if (strcmp(algo, "boosting_pls") == 0) {
        int ne = get_int_field(params, "n_estimators", 50);
        double lr = get_scalar_field(params, "learning_rate", 0.1);
        st = p4a_boosting_pls_fit(ctx, cfg, &Xv, &Yv, ne, lr, &mr);
        if (st == P4A_OK) {
            static const char *sc[] = {"rmse", "n_estimators", NULL};
            out = pack_result(mr, REG_DMAT, NULL, NULL, sc);
        }
    } else if (strcmp(algo, "random_subspace_pls") == 0) {
        int ne = get_int_field(params, "n_estimators", 50);
        int fps = get_int_field(params, "features_per_subspace", 10);
        uint64_t seed = get_u64_field(params, "seed", 0);
        st = p4a_random_subspace_pls_fit(ctx, cfg, &Xv, &Yv, ne, fps, seed, &mr);
        if (st == P4A_OK) {
            static const char *sc[] = {"rmse", "n_estimators", "features_per_subspace", NULL};
            out = pack_result(mr, REG_DMAT, NULL, NULL, sc);
        }
    } else if (strcmp(algo, "gpr_pls") == 0) {
        double ls = get_scalar_field(params, "length_scale", 1.0);
        double nl = get_scalar_field(params, "noise_level", 1e-4);
        uint64_t seed = get_u64_field(params, "seed", 0);
        st = p4a_gpr_pls_fit(ctx, cfg, &Xv, &Yv, ls, nl, seed, &mr);
        if (st == P4A_OK) {
            static const char *dm[] = {"rotation_r", "x_mean", "alpha", "L_lower",
                                          "training_scores", "predictions",
                                          "predictive_variance", NULL};
            static const char *sc[] = {"length_scale", "noise_level",
                                          "n_components", "y_mean", "rmse", NULL};
            out = pack_result(mr, dm, NULL, NULL, sc);
        }
    } else if (strcmp(algo, "pls_glm") == 0) {
        int poisson = get_int_field(params, "poisson", 0);
        st = p4a_pls_glm_fit(ctx, cfg, &Xv, &Yv, poisson, &mr);
        if (st == P4A_OK) {
            static const char *dm[] = {"coefficients", "intercept",
                                          "predictions", "x_mean", NULL};
            static const char *sc[] = {"rmse", "poisson", NULL};
            out = pack_result(mr, dm, NULL, NULL, sc);
        }
    } else if (strcmp(algo, "pls_qda") == 0) {
        int nl = 0;
        int32_t *labels = get_int32_vec_field(params, "y_labels", &nl);
        if (!labels || nl != n)
            throw_status(P4A_ERR_SHAPE_MISMATCH,
                          "pls_qda (y_labels length)", ctx, cfg, X, Y);
        st = p4a_pls_qda_fit(ctx, cfg, &Xv, labels, nl, &mr);
        mxFree(labels);
        if (st == P4A_OK) {
            static const char *dm[] = {"class_means", "class_covariances",
                                          "log_class_priors", "rotations_r",
                                          "x_mean", "predictions", NULL};
            out = pack_result(mr, dm, NULL, NULL, REG_SCALAR);
        }
    } else if (strcmp(algo, "pls_cox") == 0) {
        int sn = 0, en = 0;
        const double *sts = get_double_vec_field(params, "survival_times", &sn);
        int32_t *ev = get_int32_vec_field(params, "event_indicators", &en);
        if (!sts || sn != n || !ev || en != n) {
            if (ev) mxFree(ev);
            throw_status(P4A_ERR_SHAPE_MISMATCH,
                          "pls_cox (survival_times / event_indicators length)",
                          ctx, cfg, X, Y);
        }
        st = p4a_pls_cox_fit(ctx, cfg, &Xv, sts, sn, ev, en, &mr);
        mxFree(ev);
        if (st == P4A_OK) {
            static const char *dm[] = {"coefficients", "baseline_hazard",
                                          "event_times", "x_mean",
                                          "predictions", NULL};
            out = pack_result(mr, dm, NULL, NULL, NULL);
        }
    } else if (strcmp(algo, "pds") == 0) {
        const mxArray *xt = get_array_field(params, "X_target");
        if (!xt || !mxIsDouble(xt))
            throw_status(P4A_ERR_INVALID_ARGUMENT,
                          "pds (params.X_target)", ctx, cfg, X, Y);
        int xtr, xtc;
        double *XT = colmajor_to_rowmajor_alloc(xt, &xtr, &xtc);
        int hw = get_int_field(params, "window_half_width", 2);
        p4a_matrix_view_t XTv;
        p4a_matrix_view_init_rowmajor(&XTv, XT, xtr, xtc, P4A_DTYPE_F64);
        /* PDS: X = source, XTv = target */
        st = p4a_pds_fit(ctx, &Xv, &XTv, hw, &mr);
        free(XT);
        if (st == P4A_OK) {
            static const char *dm[] = {"transformation", "predictions", NULL};
            static const char *sc[] = {"rmse", "window_half_width", NULL};
            out = pack_result(mr, dm, NULL, NULL, sc);
        }
    } else if (strcmp(algo, "ds") == 0) {
        const mxArray *xt = get_array_field(params, "X_target");
        if (!xt || !mxIsDouble(xt))
            throw_status(P4A_ERR_INVALID_ARGUMENT,
                          "ds (params.X_target)", ctx, cfg, X, Y);
        int xtr, xtc;
        double *XT = colmajor_to_rowmajor_alloc(xt, &xtr, &xtc);
        p4a_matrix_view_t XTv;
        p4a_matrix_view_init_rowmajor(&XTv, XT, xtr, xtc, P4A_DTYPE_F64);
        st = p4a_ds_fit(ctx, &Xv, &XTv, &mr);
        free(XT);
        if (st == P4A_OK) {
            static const char *dm[] = {"transformation", "bias",
                                          "predictions", NULL};
            out = pack_result(mr, dm, NULL, NULL, REG_SCALAR);
        }
    } else if (strcmp(algo, "mir_pls") == 0) {
        st = p4a_mir_pls_fit(ctx, cfg, &Xv, &Yv, &mr);
        if (st == P4A_OK) out = pack_result(mr, REG_DMAT, NULL, NULL, REG_SCALAR);
    } else if (strcmp(algo, "missing_aware_nipals") == 0) {
        st = p4a_missing_aware_nipals_fit(ctx, cfg, &Xv, &Yv, &mr);
        if (st == P4A_OK) out = pack_result(mr, REG_DMAT, NULL, NULL, REG_SCALAR);
    } else if (strcmp(algo, "mb_pls") == 0) {
        int bsn = 0;
        int64_t *bs = get_int64_vec_field(params, "block_sizes", &bsn);
        if (!bs || bsn < 1)
            throw_status(P4A_ERR_INVALID_ARGUMENT,
                          "mb_pls requires params.block_sizes",
                          ctx, cfg, X, Y);
        int64_t bsum = 0;
        for (int i = 0; i < bsn; ++i) bsum += bs[i];
        if (bsum != (int64_t)p) {
            mxFree(bs);
            throw_status(P4A_ERR_SHAPE_MISMATCH,
                          "sum(block_sizes) must equal size(X,2)",
                          ctx, cfg, X, Y);
        }
        p4a_config_set_solver(cfg, P4A_SOLVER_NIPALS);
        st = p4a_mb_pls_fit(ctx, cfg, &Xv, &Yv, bs, bsn, &mr);
        mxFree(bs);
        if (st == P4A_OK) {
            static const char *dm[] = {"coefficients", "predictions",
                                          "x_mean", "x_scale", "intercept",
                                          "block_weights", NULL};
            static const char *sc[] = {"rmse", "n_blocks", NULL};
            out = pack_result(mr, dm, NULL, NULL, sc);
        }
    } else if (strcmp(algo, "lw_pls") == 0) {
        int nn = get_int_field(params, "n_neighbors", 30);
        st = p4a_lw_pls_fit(ctx, cfg, &Xv, &Yv, nn, &mr);
        if (st == P4A_OK) {
            static const char *dm[] = {"predictions", "neighbor_indices", NULL};
            static const char *i64[] = {"neighbor_indices_i64", NULL};
            static const char *sc[] = {"n_neighbors", "n_components", "rmse", NULL};
            out = pack_result(mr, dm, NULL, i64, sc);
        }
    } else if (strcmp(algo, "pls_lda") == 0) {
        int nl = 0, nc = 0;
        int32_t *labels = get_int32_vec_field(params, "y_labels", &nl);
        nc = get_int_field(params, "n_classes", 0);
        if (!labels || nl != n)
            throw_status(P4A_ERR_SHAPE_MISMATCH,
                          "pls_lda (y_labels)", ctx, cfg, X, Y);
        if (nc <= 0) {
            for (int i = 0; i < nl; ++i) if (labels[i] + 1 > nc) nc = labels[i] + 1;
        }
        st = p4a_pls_lda_fit(ctx, cfg, &Xv, labels, nl, nc, &mr);
        mxFree(labels);
        if (st == P4A_OK) {
            static const char *dm[] = {"decision_scores", NULL};
            static const char *iv[] = {"predictions", NULL};
            static const char *sc[] = {"n_classes", "n_components", NULL};
            out = pack_result(mr, dm, iv, NULL, sc);
        }
    } else if (strcmp(algo, "pls_logistic") == 0) {
        int nl = 0, nc = 0;
        int32_t *labels = get_int32_vec_field(params, "y_labels", &nl);
        nc = get_int_field(params, "n_classes", 0);
        if (!labels || nl != n)
            throw_status(P4A_ERR_SHAPE_MISMATCH,
                          "pls_logistic (y_labels)", ctx, cfg, X, Y);
        if (nc <= 0) {
            for (int i = 0; i < nl; ++i) if (labels[i] + 1 > nc) nc = labels[i] + 1;
        }
        st = p4a_pls_logistic_fit(ctx, cfg, &Xv, labels, nl, nc, &mr);
        mxFree(labels);
        if (st == P4A_OK) {
            static const char *dm[] = {"decision_scores", "probabilities",
                                          "intercepts", "coefficients", NULL};
            static const char *iv[] = {"predictions", NULL};
            static const char *sc[] = {"n_classes", "n_components", NULL};
            out = pack_result(mr, dm, iv, NULL, sc);
        }
    } else if (strcmp(algo, "aom_preprocess") == 0) {
        int n_ops = get_int_field(params, "n_operators", 3);
        int mode = get_int_field(params, "gating_mode", 0);
        if (n_ops < 1) n_ops = 1;
        if (n_ops > 5) n_ops = 5;
        p4a_operator_kind_t kinds[5] = {
            P4A_OP_IDENTITY,
            P4A_OP_CENTER,
            P4A_OP_PARETO_SCALE,
            P4A_OP_AUTOSCALE,
            P4A_OP_SNV
        };
        p4a_operator_bank_t *bank = NULL;
        p4a_gating_strategy_t *gate = NULL;
        st = p4a_operator_bank_create(&bank);
        for (int i = 0; st == P4A_OK && i < n_ops; ++i) {
            st = p4a_operator_bank_add(bank, kinds[i], NULL, 0);
        }
        if (st == P4A_OK) {
            st = p4a_gating_strategy_create(
                &gate, (p4a_gating_mode_t)mode);
        }
        if (st == P4A_OK) {
            st = p4a_aom_preprocess_fit(ctx, bank, gate, &Xv, &Yv, &mr);
        }
        if (gate) p4a_gating_strategy_destroy(gate);
        if (bank) p4a_operator_bank_destroy(bank);
        if (st == P4A_OK) {
            static const char *dm[] = {"transformed", "operator_outputs",
                                          "weights", NULL};
            static const char *i64[] = {"operator_kinds", NULL};
            static const char *sc[] = {"n_operators", "mode", NULL};
            out = pack_result(mr, dm, NULL, i64, sc);
        }
    }

    /* ============================================================
     *  Selectors
     * ============================================================ */
    else if (strcmp(algo, "variable_select_rank") == 0) {
        /* Convenience branch: fit an internal SIMPLS model on the given
         * X/Y with store_scores=1, then call p4a_variable_select_rank
         * with the requested method (0=VIP, 1=coef magnitude, 2=SR) and
         * top_k. The model is destroyed before returning. */
        int method = get_int_field(params, "method", 0);
        int top_k = get_int_field(params, "top_k", 10);
        p4a_config_set_store_scores(cfg, 1);
        p4a_model_t *model = NULL;
        st = p4a_model_fit(ctx, cfg, &Xv, &Yv, &model);
        if (st == P4A_OK) {
            st = p4a_variable_select_rank(ctx, model, &Xv,
                                            (int32_t)method,
                                            (int32_t)top_k, &mr);
        }
        if (model) p4a_model_destroy(model);
        if (st == P4A_OK) {
            static const char *dm[] = {"scores", NULL};
            static const char *i64[] = {"selected_indices", NULL};
            static const char *sc[] = {"n_features", "top_k", "method", NULL};
            out = pack_result(mr, dm, NULL, i64, sc);
        }
    } else if (strcmp(algo, "spa_select") == 0) {
        int top_k = get_int_field(params, "top_k", 10);
        st = p4a_spa_select(ctx, cfg, &Xv, &Yv, top_k, &mr);
        if (st == P4A_OK) {
            static const char *i64[] = {"selected_indices", NULL};
            static const char *sc[] = {"best_rmse", NULL};
            out = pack_result(mr, NULL, NULL, i64, sc);
        }
    } else if (strcmp(algo, "cars_select") == 0) {
        int it = get_int_field(params, "n_iterations", 50);
        int mf = get_int_field(params, "min_features", 5);
        { p4a_validation_plan_t *_plan = make_default_plan(n, 3);
            if (!_plan) throw_status(P4A_ERR_OUT_OF_MEMORY, "cars_select", ctx, cfg, X, Y);
            st = p4a_cars_select(ctx, cfg, &Xv, &Yv, _plan, it, mf, &mr);
            p4a_validation_plan_destroy(_plan);
        }
        if (st == P4A_OK) {
            static const char *i64[] = {"selected_indices", NULL};
            static const char *sc[] = {"best_rmse", NULL};
            out = pack_result(mr, NULL, NULL, i64, sc);
        }
    } else if (strcmp(algo, "interval_select") == 0) {
        int iw = get_int_field(params, "interval_width", 10);
        int step = get_int_field(params, "step", 1);
        { p4a_validation_plan_t *_plan = make_default_plan(n, 3);
            if (!_plan) throw_status(P4A_ERR_OUT_OF_MEMORY, "interval_select", ctx, cfg, X, Y);
            st = p4a_interval_select(ctx, cfg, &Xv, &Yv, _plan, iw, step, &mr);
            p4a_validation_plan_destroy(_plan);
        }
        if (st == P4A_OK) {
            static const char *i64[] = {"selected_indices", NULL};
            static const char *sc[] = {"best_rmse", NULL};
            out = pack_result(mr, NULL, NULL, i64, sc);
        }
    } else if (strcmp(algo, "stability_select") == 0) {
        int k = get_int_field(params, "top_k", 10);
        { p4a_validation_plan_t *_plan = make_default_plan(n, 3);
            if (!_plan) throw_status(P4A_ERR_OUT_OF_MEMORY, "stability_select", ctx, cfg, X, Y);
            st = p4a_stability_select(ctx, cfg, &Xv, &Yv, _plan, k, &mr);
            p4a_validation_plan_destroy(_plan);
        }
        if (st == P4A_OK) {
            static const char *i64[] = {"selected_indices", NULL};
            static const char *sc[] = {"best_rmse", NULL};
            out = pack_result(mr, NULL, NULL, i64, sc);
        }
    } else if (strcmp(algo, "uve_select") == 0) {
        int nf = get_int_field(params, "noise_features", 0);
        uint64_t seed = get_u64_field(params, "noise_seed", 0);
        { p4a_validation_plan_t *_plan = make_default_plan(n, 3);
            if (!_plan) throw_status(P4A_ERR_OUT_OF_MEMORY, "uve_select", ctx, cfg, X, Y);
            st = p4a_uve_select(ctx, cfg, &Xv, &Yv, _plan, nf, seed, &mr);
            p4a_validation_plan_destroy(_plan);
        }
        if (st == P4A_OK) {
            static const char *i64[] = {"selected_indices", NULL};
            static const char *sc[] = {"best_rmse", NULL};
            out = pack_result(mr, NULL, NULL, i64, sc);
        }
    } else if (strcmp(algo, "random_frog_select") == 0) {
        int it = get_int_field(params, "n_iterations", 100);
        int isz = get_int_field(params, "initial_size", 30);
        int minz = get_int_field(params, "min_size", 1);
        int maxz = get_int_field(params, "max_size", p);
        int tk = get_int_field(params, "top_k", 10);
        uint64_t seed = get_u64_field(params, "seed", 0);
        { p4a_validation_plan_t *_plan = make_default_plan(n, 3);
            if (!_plan) throw_status(P4A_ERR_OUT_OF_MEMORY, "random_frog_select", ctx, cfg, X, Y);
            st = p4a_random_frog_select(ctx, cfg, &Xv, &Yv, _plan, it, isz, minz, maxz, tk, seed, &mr);
            p4a_validation_plan_destroy(_plan);
        }
        if (st == P4A_OK) {
            static const char *i64[] = {"selected_indices", NULL};
            static const char *sc[] = {"best_rmse", NULL};
            out = pack_result(mr, NULL, NULL, i64, sc);
        }
    } else if (strcmp(algo, "scars_select") == 0) {
        int it = get_int_field(params, "n_iterations", 50);
        int mf = get_int_field(params, "min_features", 5);
        double sf = get_scalar_field(params, "sample_fraction", 0.8);
        uint64_t seed = get_u64_field(params, "seed", 0);
        { p4a_validation_plan_t *_plan = make_default_plan(n, 3);
            if (!_plan) throw_status(P4A_ERR_OUT_OF_MEMORY, "scars_select", ctx, cfg, X, Y);
            st = p4a_scars_select(ctx, cfg, &Xv, &Yv, _plan, it, mf, sf, seed, &mr);
            p4a_validation_plan_destroy(_plan);
        }
        if (st == P4A_OK) {
            static const char *i64[] = {"selected_indices", NULL};
            static const char *sc[] = {"best_rmse", NULL};
            out = pack_result(mr, NULL, NULL, i64, sc);
        }
    } else if (strcmp(algo, "ga_select") == 0) {
        int g = get_int_field(params, "n_generations", 50);
        int psz = get_int_field(params, "population_size", 50);
        int minf = get_int_field(params, "min_features", 1);
        int maxf = get_int_field(params, "max_features", p);
        double mr_rate = get_scalar_field(params, "mutation_rate", 0.01);
        uint64_t seed = get_u64_field(params, "seed", 0);
        { p4a_validation_plan_t *_plan = make_default_plan(n, 3);
            if (!_plan) throw_status(P4A_ERR_OUT_OF_MEMORY, "ga_select", ctx, cfg, X, Y);
            st = p4a_ga_select(ctx, cfg, &Xv, &Yv, _plan, g, psz, minf, maxf,
                            mr_rate, seed, &mr);
            p4a_validation_plan_destroy(_plan);
        }
        if (st == P4A_OK) {
            static const char *i64[] = {"selected_indices", NULL};
            static const char *sc[] = {"best_rmse", NULL};
            out = pack_result(mr, NULL, NULL, i64, sc);
        }
    } else if (strcmp(algo, "pso_select") == 0) {
        int sw = get_int_field(params, "n_swarm", 30);
        int it = get_int_field(params, "n_iterations", 50);
        double w = get_scalar_field(params, "w", 0.729);
        double c1 = get_scalar_field(params, "c1", 1.494);
        double c2 = get_scalar_field(params, "c2", 1.494);
        double vmax = get_scalar_field(params, "v_max", 4.0);
        uint64_t seed = get_u64_field(params, "seed", 0);
        { p4a_validation_plan_t *_plan = make_default_plan(n, 3);
            if (!_plan) throw_status(P4A_ERR_OUT_OF_MEMORY, "pso_select", ctx, cfg, X, Y);
            st = p4a_pso_select(ctx, cfg, &Xv, &Yv, _plan, sw, it, w, c1, c2, vmax, seed, &mr);
            p4a_validation_plan_destroy(_plan);
        }
        if (st == P4A_OK) {
            static const char *dm[] = {"inclusion_frequencies",
                                          "best_rmse_by_iteration",
                                          "mean_rmse_by_iteration", NULL};
            static const char *i64[] = {"selected_indices", NULL};
            static const char *sc[] = {"best_rmse", NULL};
            out = pack_result(mr, dm, NULL, i64, sc);
        }
    } else if (strcmp(algo, "vissa_select") == 0) {
        int it = get_int_field(params, "n_iterations", 20);
        int sub = get_int_field(params, "n_submodels", 100);
        double rk = get_scalar_field(params, "ratio_kept", 0.1);
        double th = get_scalar_field(params, "threshold", 0.5);
        double fp = get_scalar_field(params, "floor_probability", 0.01);
        uint64_t seed = get_u64_field(params, "seed", 0);
        { p4a_validation_plan_t *_plan = make_default_plan(n, 3);
            if (!_plan) throw_status(P4A_ERR_OUT_OF_MEMORY, "vissa_select", ctx, cfg, X, Y);
            st = p4a_vissa_select(ctx, cfg, &Xv, &Yv, _plan, it, sub, rk, th, fp, seed, &mr);
            p4a_validation_plan_destroy(_plan);
        }
        if (st == P4A_OK) {
            static const char *dm[] = {"final_probabilities",
                                          "inclusion_frequencies",
                                          "best_rmse_by_iteration",
                                          "mean_rmse_by_iteration",
                                          "top_k_per_iteration", NULL};
            static const char *i64[] = {"selected_indices", NULL};
            static const char *sc[] = {"best_rmse", NULL};
            out = pack_result(mr, dm, NULL, i64, sc);
        }
    } else if (strcmp(algo, "shaving_select") == 0) {
        int ns = get_int_field(params, "n_steps", 10);
        int mf = get_int_field(params, "min_features", 5);
        double sf = get_scalar_field(params, "shave_fraction", 0.1);
        { p4a_validation_plan_t *_plan = make_default_plan(n, 3);
            if (!_plan) throw_status(P4A_ERR_OUT_OF_MEMORY, "shaving_select", ctx, cfg, X, Y);
            st = p4a_shaving_select(ctx, cfg, &Xv, &Yv, _plan, ns, mf, sf, &mr);
            p4a_validation_plan_destroy(_plan);
        }
        if (st == P4A_OK) {
            static const char *i64[] = {"selected_indices", NULL};
            static const char *sc[] = {"best_rmse", NULL};
            out = pack_result(mr, NULL, NULL, i64, sc);
        }
    } else if (strcmp(algo, "bve_select") == 0) {
        int ns = get_int_field(params, "n_steps", 10);
        int mf = get_int_field(params, "min_features", 5);
        { p4a_validation_plan_t *_plan = make_default_plan(n, 3);
            if (!_plan) throw_status(P4A_ERR_OUT_OF_MEMORY, "bve_select", ctx, cfg, X, Y);
            st = p4a_bve_select(ctx, cfg, &Xv, &Yv, _plan, ns, mf, &mr);
            p4a_validation_plan_destroy(_plan);
        }
        if (st == P4A_OK) {
            static const char *i64[] = {"selected_indices", NULL};
            static const char *sc[] = {"best_rmse", NULL};
            out = pack_result(mr, NULL, NULL, i64, sc);
        }
    } else if (strcmp(algo, "t2_select") == 0) {
        int nt = 0;
        const double *thr = get_double_vec_field(params, "alpha_thresholds", &nt);
        if (!thr || nt < 1)
            throw_status(P4A_ERR_INVALID_ARGUMENT,
                          "t2_select requires params.alpha_thresholds",
                          ctx, cfg, X, Y);
        int ms = get_int_field(params, "min_selected", 1);
        { p4a_validation_plan_t *_plan = make_default_plan(n, 3);
            if (!_plan) throw_status(P4A_ERR_OUT_OF_MEMORY, "t2_select", ctx, cfg, X, Y);
            st = p4a_t2_select(ctx, cfg, &Xv, &Yv, _plan, thr, nt, ms, &mr);
            p4a_validation_plan_destroy(_plan);
        }
        if (st == P4A_OK) {
            static const char *i64[] = {"selected_indices", NULL};
            static const char *sc[] = {"best_rmse", NULL};
            out = pack_result(mr, NULL, NULL, i64, sc);
        }
    } else if (strcmp(algo, "wvc_select") == 0) {
        int tk = get_int_field(params, "top_k", 10);
        int norm = get_int_field(params, "normalize", 1);
        st = p4a_wvc_select(ctx, &Xv, &Yv, n_components, tk, norm, &mr);
        if (st == P4A_OK) {
            static const char *dm[] = {"scores", NULL};
            static const char *i64[] = {"selected_indices", NULL};
            out = pack_result(mr, dm, NULL, i64, NULL);
        }
    } else if (strcmp(algo, "wvc_threshold_select") == 0) {
        int norm = get_int_field(params, "normalize", 1);
        double thr = get_scalar_field(params, "threshold", 0.0);
        double tf = get_scalar_field(params, "threshold_factor", 1.0);
        int ms = get_int_field(params, "min_selected", 1);
        st = p4a_wvc_threshold_select(ctx, &Xv, &Yv, n_components, norm,
                                        thr, tf, ms, &mr);
        if (st == P4A_OK) {
            static const char *dm[] = {"scores", NULL};
            static const char *i64[] = {"selected_indices", NULL};
            out = pack_result(mr, dm, NULL, i64, NULL);
        }
    } else if (strcmp(algo, "emcuve_select") == 0) {
        int nf = get_int_field(params, "noise_features", 0);
        uint64_t seed = get_u64_field(params, "noise_seed", 0);
        int ne = get_int_field(params, "n_ensembles", 5);
        double vt = get_scalar_field(params, "vote_threshold", 0.5);
        { p4a_validation_plan_t *_plan = make_default_plan(n, 3);
            if (!_plan) throw_status(P4A_ERR_OUT_OF_MEMORY, "emcuve_select", ctx, cfg, X, Y);
            st = p4a_emcuve_select(ctx, cfg, &Xv, &Yv, _plan, nf, seed, ne, vt, &mr);
            p4a_validation_plan_destroy(_plan);
        }
        if (st == P4A_OK) {
            static const char *i64[] = {"selected_indices", NULL};
            static const char *sc[] = {"best_rmse", NULL};
            out = pack_result(mr, NULL, NULL, i64, sc);
        }
    } else if (strcmp(algo, "randomization_select") == 0) {
        int np = get_int_field(params, "n_permutations", 100);
        uint64_t seed = get_u64_field(params, "randomization_seed", 0);
        double a = get_scalar_field(params, "alpha", 0.05);
        st = p4a_randomization_select(ctx, cfg, &Xv, &Yv, np, seed, a, &mr);
        if (st == P4A_OK) {
            static const char *i64[] = {"selected_indices", NULL};
            static const char *sc[] = {"best_rmse", NULL};
            out = pack_result(mr, NULL, NULL, i64, sc);
        }
    } else if (strcmp(algo, "bipls_select") == 0) {
        int iw = get_int_field(params, "interval_width", 10);
        int mi = get_int_field(params, "min_intervals", 1);
        { p4a_validation_plan_t *_plan = make_default_plan(n, 3);
            if (!_plan) throw_status(P4A_ERR_OUT_OF_MEMORY, "bipls_select", ctx, cfg, X, Y);
            st = p4a_bipls_select(ctx, cfg, &Xv, &Yv, _plan, iw, mi, &mr);
            p4a_validation_plan_destroy(_plan);
        }
        if (st == P4A_OK) {
            static const char *i64[] = {"selected_indices", NULL};
            static const char *sc[] = {"best_rmse", NULL};
            out = pack_result(mr, NULL, NULL, i64, sc);
        }
    } else if (strcmp(algo, "sipls_select") == 0) {
        int iw = get_int_field(params, "interval_width", 10);
        int cs = get_int_field(params, "combination_size", 2);
        { p4a_validation_plan_t *_plan = make_default_plan(n, 3);
            if (!_plan) throw_status(P4A_ERR_OUT_OF_MEMORY, "sipls_select", ctx, cfg, X, Y);
            st = p4a_sipls_select(ctx, cfg, &Xv, &Yv, _plan, iw, cs, &mr);
            p4a_validation_plan_destroy(_plan);
        }
        if (st == P4A_OK) {
            static const char *i64[] = {"selected_indices", NULL};
            static const char *sc[] = {"best_rmse", NULL};
            out = pack_result(mr, NULL, NULL, i64, sc);
        }
    } else if (strcmp(algo, "rep_select") == 0) {
        int ns = get_int_field(params, "n_steps", 10);
        int mf = get_int_field(params, "min_features", 5);
        int rc = get_int_field(params, "remove_count", 1);
        { p4a_validation_plan_t *_plan = make_default_plan(n, 3);
            if (!_plan) throw_status(P4A_ERR_OUT_OF_MEMORY, "rep_select", ctx, cfg, X, Y);
            st = p4a_rep_select(ctx, cfg, &Xv, &Yv, _plan, ns, mf, rc, &mr);
            p4a_validation_plan_destroy(_plan);
        }
        if (st == P4A_OK) {
            static const char *i64[] = {"selected_indices", NULL};
            static const char *sc[] = {"best_rmse", NULL};
            out = pack_result(mr, NULL, NULL, i64, sc);
        }
    } else if (strcmp(algo, "ipw_select") == 0) {
        int it = get_int_field(params, "n_iterations", 10);
        int tk = get_int_field(params, "top_k", 10);
        double damp = get_scalar_field(params, "damping", 0.5);
        double wf = get_scalar_field(params, "weight_floor", 1e-6);
        { p4a_validation_plan_t *_plan = make_default_plan(n, 3);
            if (!_plan) throw_status(P4A_ERR_OUT_OF_MEMORY, "ipw_select", ctx, cfg, X, Y);
            st = p4a_ipw_select(ctx, cfg, &Xv, &Yv, _plan, it, tk, damp, wf, &mr);
            p4a_validation_plan_destroy(_plan);
        }
        if (st == P4A_OK) {
            static const char *i64[] = {"selected_indices", NULL};
            static const char *sc[] = {"best_rmse", NULL};
            out = pack_result(mr, NULL, NULL, i64, sc);
        }
    } else if (strcmp(algo, "st_select") == 0) {
        int nt = 0;
        const double *thr = get_double_vec_field(params, "thresholds", &nt);
        if (!thr || nt < 1)
            throw_status(P4A_ERR_INVALID_ARGUMENT,
                          "st_select requires params.thresholds",
                          ctx, cfg, X, Y);
        int ms = get_int_field(params, "min_selected", 1);
        { p4a_validation_plan_t *_plan = make_default_plan(n, 3);
            if (!_plan) throw_status(P4A_ERR_OUT_OF_MEMORY, "st_select", ctx, cfg, X, Y);
            st = p4a_st_select(ctx, cfg, &Xv, &Yv, _plan, thr, nt, ms, &mr);
            p4a_validation_plan_destroy(_plan);
        }
        if (st == P4A_OK) {
            static const char *i64[] = {"selected_indices", NULL};
            static const char *sc[] = {"best_rmse", NULL};
            out = pack_result(mr, NULL, NULL, i64, sc);
        }
    } else if (strcmp(algo, "iriv_select") == 0) {
        int mr_rounds = get_int_field(params, "max_rounds", 20);
        uint64_t seed = get_u64_field(params, "seed", 0);
        { p4a_validation_plan_t *_plan = make_default_plan(n, 3);
            if (!_plan) throw_status(P4A_ERR_OUT_OF_MEMORY, "iriv_select", ctx, cfg, X, Y);
            st = p4a_iriv_select(ctx, cfg, &Xv, &Yv, _plan, mr_rounds, seed, &mr);
            p4a_validation_plan_destroy(_plan);
        }
        if (st == P4A_OK) {
            static const char *dm[] = {"remaining_per_round",
                                          "removed_per_round", NULL};
            static const char *i64[] = {"selected_indices", NULL};
            static const char *sc[] = {"n_rounds", NULL};
            out = pack_result(mr, dm, NULL, i64, sc);
        }
    } else if (strcmp(algo, "irf_select") == 0) {
        int it = get_int_field(params, "n_iterations", 100);
        int ws = get_int_field(params, "window_size", 10);
        int ii = get_int_field(params, "initial_intervals", 10);
        int tk = get_int_field(params, "top_k", 5);
        uint64_t seed = get_u64_field(params, "seed", 0);
        { p4a_validation_plan_t *_plan = make_default_plan(n, 3);
            if (!_plan) throw_status(P4A_ERR_OUT_OF_MEMORY, "irf_select", ctx, cfg, X, Y);
            st = p4a_irf_select(ctx, cfg, &Xv, &Yv, _plan, it, ws, ii, tk, seed, &mr);
            p4a_validation_plan_destroy(_plan);
        }
        if (st == P4A_OK) {
            static const char *dm[] = {"probability", "rmse_by_iteration",
                                          "subset_sizes", NULL};
            static const char *i64[] = {"selected_indices", "top_k_intervals", NULL};
            static const char *sc[] = {"best_rmse", NULL};
            out = pack_result(mr, dm, NULL, i64, sc);
        }
    } else if (strcmp(algo, "vip_spa_select") == 0) {
        double vt = get_scalar_field(params, "vip_threshold", 0.3);
        int tk = get_int_field(params, "top_k", 10);
        st = p4a_vip_spa_select(ctx, cfg, &Xv, &Yv, vt, tk, &mr);
        if (st == P4A_OK) {
            static const char *dm[] = {"vip_scores", "vip_mask",
                                          "selection_scores", NULL};
            static const char *i64[] = {"selected_indices", NULL};
            static const char *sc[] = {"n_selected", "vip_threshold", NULL};
            out = pack_result(mr, dm, NULL, i64, sc);
        }
    }

    /* ============================================================
     *  Diagnostics
     * ============================================================ */
    else if (strcmp(algo, "pls_diagnostics_compute") == 0) {
        /* Fit SIMPLS model on X/Y (store_scores=1) and call
         * p4a_pls_diagnostics_compute. params.X_reference (optional)
         * overrides the reference distribution; when absent we pass
         * NULL and the core falls back to the stored training scores. */
        const mxArray *xr = get_array_field(params, "X_reference");
        double *XR = NULL;
        p4a_matrix_view_t XRv;
        const p4a_matrix_view_t *xr_ptr = NULL;
        if (xr) {
            if (!mxIsDouble(xr) || mxIsComplex(xr))
                throw_status(P4A_ERR_INVALID_ARGUMENT,
                              "pls_diagnostics_compute (params.X_reference must be real double)",
                              ctx, cfg, X, Y);
            int xrr, xrc;
            XR = colmajor_to_rowmajor_alloc(xr, &xrr, &xrc);
            if (xrc != p) {
                free(XR);
                throw_status(P4A_ERR_SHAPE_MISMATCH,
                              "pls_diagnostics_compute (X_reference ncol)",
                              ctx, cfg, X, Y);
            }
            p4a_matrix_view_init_rowmajor(&XRv, XR, xrr, xrc, P4A_DTYPE_F64);
            xr_ptr = &XRv;
        }
        p4a_config_set_store_scores(cfg, 1);
        p4a_model_t *model = NULL;
        st = p4a_model_fit(ctx, cfg, &Xv, &Yv, &model);
        if (st == P4A_OK) {
            st = p4a_pls_diagnostics_compute(ctx, model, &Xv, xr_ptr, &mr);
        }
        if (model) p4a_model_destroy(model);
        free(XR);
        if (st == P4A_OK) {
            static const char *dm[] = {"t2", "q", "dmodx", NULL};
            static const char *sc[] = {"n_components", "n_features", NULL};
            out = pack_result(mr, dm, NULL, NULL, sc);
        }
    } else if (strcmp(algo, "pls_monitoring_run") == 0) {
        /* X/Y positional inputs are the reference dataset
         * (X_reference, Y_reference). params.X_monitor must be a real
         * double matrix with the same number of columns as X. The
         * internal SIMPLS fit uses store_scores=1 so the C core can
         * derive reference thresholds. */
        const mxArray *xm = get_array_field(params, "X_monitor");
        if (!xm || !mxIsDouble(xm) || mxIsComplex(xm))
            throw_status(P4A_ERR_INVALID_ARGUMENT,
                          "pls_monitoring_run (params.X_monitor must be a real double matrix)",
                          ctx, cfg, X, Y);
        int xmr, xmc;
        double *XM = colmajor_to_rowmajor_alloc(xm, &xmr, &xmc);
        if (xmc != p) {
            free(XM);
            throw_status(P4A_ERR_SHAPE_MISMATCH,
                          "pls_monitoring_run (X_monitor must have same number of columns as X_reference)",
                          ctx, cfg, X, Y);
        }
        double alpha = get_scalar_field(params, "alpha", 0.05);
        p4a_matrix_view_t XMv;
        p4a_matrix_view_init_rowmajor(&XMv, XM, xmr, xmc, P4A_DTYPE_F64);
        p4a_config_set_store_scores(cfg, 1);
        p4a_model_t *model = NULL;
        st = p4a_model_fit(ctx, cfg, &Xv, &Yv, &model);
        if (st == P4A_OK) {
            st = p4a_pls_monitoring_run(ctx, model, &Xv, &XMv, alpha, &mr);
        }
        if (model) p4a_model_destroy(model);
        free(XM);
        if (st == P4A_OK) {
            static const char *dm[] = {"t2", "q", "t2_reference",
                                          "q_reference", NULL};
            static const char *iv[] = {"t2_alarms", "q_alarms",
                                          "any_alarms", NULL};
            static const char *sc[] = {"t2_threshold", "q_threshold",
                                          "alpha", NULL};
            out = pack_result(mr, dm, iv, NULL, sc);
        }
    } else if (strcmp(algo, "approximate_press_compute") == 0) {
        st = p4a_approximate_press_compute(ctx, cfg, &Xv, &Yv, n_components, &mr);
        if (st == P4A_OK) {
            static const char *dm[] = {"press_per_component",
                                          "rmse_per_component", NULL};
            static const char *iv[] = {"selected_n_components", NULL};
            out = pack_result(mr, dm, iv, NULL, NULL);
        }
    } else if (strcmp(algo, "one_se_rule_compute") == 0) {
        const mxArray *fold_rmse = get_array_field(params, "fold_rmse_matrix");
        if (!fold_rmse || !mxIsDouble(fold_rmse))
            throw_status(P4A_ERR_INVALID_ARGUMENT,
                          "one_se_rule_compute requires params.fold_rmse_matrix",
                          ctx, cfg, X, Y);
        int max_k = (int)mxGetM(fold_rmse);
        int n_folds = (int)mxGetN(fold_rmse);
        /* Convert to row-major. */
        int rmr, rmc;
        double *fr_rm = colmajor_to_rowmajor_alloc(fold_rmse, &rmr, &rmc);
        st = p4a_one_se_rule_compute(ctx, fr_rm, max_k, n_folds, &mr);
        free(fr_rm);
        if (st == P4A_OK) {
            static const char *dm[] = {"mean_rmse_per_component", NULL};
            static const char *iv[] = {"best_n_components",
                                          "one_se_n_components", NULL};
            static const char *sc[] = {"one_se_standard_error",
                                          "one_se_threshold", NULL};
            out = pack_result(mr, dm, iv, NULL, sc);
        }
    } else {
        throw_status(P4A_ERR_INVALID_ARGUMENT, algo, ctx, cfg, X, Y);
    }

    p4a_config_destroy(cfg);
    free(X); free(Y);
    if (st != P4A_OK) {
        /* throw_status doesn't return; do it now with NULLed-out cfg / X / Y. */
        throw_status(st, algo, ctx, NULL, NULL, NULL);
    }
    p4a_context_destroy(ctx);
    plhs[0] = out;
}
