/* SPDX-License-Identifier: CeCILL-2.1
 *
 * MATLAB / Octave MEX dispatcher for MethodResult-returning fits.
 *
 *   result = p4a_method_fit_mex(algo, X, Y, n_components, params)
 *
 * where `algo` is one of:
 *   "sparse_simpls" "cppls" "weighted_pls" "mb_pls" "pls_glm"
 *   "mir_pls" "spa_select" "cars_select" "approximate_press"
 *
 * `params` is a struct whose fields supply algorithm-specific
 * parameters (and any required side inputs like `sample_weights`,
 * `block_sizes`).
 *
 * Returns a struct with the MethodResult arrays + scalars converted
 * to MATLAB-friendly types (column-major matrices, row-vectors, scalar
 * doubles).
 *
 * One MEX dispatches all algorithms so we don't accrue a wall of
 * one-off MEX entry points — the cost of a single string compare per
 * fit is negligible compared to the fit itself.
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

static void add_double_matrix(mxArray *out, int field_idx,
                                const p4a_method_result_t *mr,
                                const char *name) {
    const double *data = NULL;
    int64_t rows = 0, cols = 0;
    p4a_status_t st = p4a_method_result_get_double_matrix(mr, name,
                                                          &data, &rows, &cols);
    if (st != P4A_OK || !data) {
        mxSetFieldByNumber(out, 0, field_idx, mxCreateDoubleMatrix(0, 0, mxREAL));
        return;
    }
    mxSetFieldByNumber(out, 0, field_idx,
                       rowmajor_to_colmajor_mx(data, (int)rows, (int)cols));
}

static void add_int_vector(mxArray *out, int field_idx,
                            const p4a_method_result_t *mr,
                            const char *name) {
    const int32_t *data = NULL;
    int32_t size = 0;
    p4a_status_t st = p4a_method_result_get_int_vector(mr, name, &data, &size);
    if (st != P4A_OK || !data) {
        mxSetFieldByNumber(out, 0, field_idx, mxCreateDoubleMatrix(0, 0, mxREAL));
        return;
    }
    mxArray *mx = mxCreateDoubleMatrix(1, size, mxREAL);
    double *dst = mxGetPr(mx);
    for (int32_t i = 0; i < size; ++i) dst[i] = (double)data[i];
    mxSetFieldByNumber(out, 0, field_idx, mx);
}

static void add_int64_vector(mxArray *out, int field_idx,
                              const p4a_method_result_t *mr,
                              const char *name) {
    const int64_t *data = NULL;
    int64_t size = 0;
    p4a_status_t st = p4a_method_result_get_int64_vector(mr, name, &data, &size);
    if (st != P4A_OK || !data) {
        mxSetFieldByNumber(out, 0, field_idx, mxCreateDoubleMatrix(0, 0, mxREAL));
        return;
    }
    mxArray *mx = mxCreateDoubleMatrix(1, (int)size, mxREAL);
    double *dst = mxGetPr(mx);
    for (int64_t i = 0; i < size; ++i) dst[i] = (double)data[i];
    mxSetFieldByNumber(out, 0, field_idx, mx);
}

static void add_scalar(mxArray *out, int field_idx,
                        const p4a_method_result_t *mr,
                        const char *name) {
    double v = 0.0;
    p4a_status_t st = p4a_method_result_get_scalar(mr, name, &v);
    if (st != P4A_OK) {
        mxSetFieldByNumber(out, 0, field_idx, mxCreateDoubleMatrix(0, 0, mxREAL));
        return;
    }
    mxSetFieldByNumber(out, 0, field_idx, mxCreateDoubleScalar(v));
}

/* ---- Common Config builder ------------------------------------------ */

static p4a_config_t* basic_cfg(int n_components) {
    p4a_config_t *cfg = NULL;
    if (p4a_config_create(&cfg) != P4A_OK) {
        mexErrMsgIdAndTxt("pls4all:cfg", "p4a_config_create failed");
    }
    p4a_config_set_algorithm(cfg, P4A_ALGO_PLS_REGRESSION);
    p4a_config_set_solver(cfg, P4A_SOLVER_SIMPLS);
    p4a_config_set_deflation(cfg, P4A_DEFLATION_REGRESSION);
    p4a_config_set_n_components(cfg, n_components);
    p4a_config_set_center_x(cfg, 1);
    p4a_config_set_scale_x(cfg, 0);  /* match the R/Python defaults */
    p4a_config_set_center_y(cfg, 1);
    p4a_config_set_scale_y(cfg, 0);
    p4a_config_set_tol(cfg, 1e-6);
    p4a_config_set_max_iter(cfg, 500);
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

static const mxArray* get_array_field(const mxArray *params, const char *name) {
    if (!params || !mxIsStruct(params)) return NULL;
    return mxGetField(params, 0, name);
}

/* ---- Result packagers ------------------------------------------------ */

static mxArray* pack_result(p4a_method_result_t *mr,
                              const char **dmat_keys, int n_dmat,
                              const char **iv_keys, int n_iv,
                              const char **i64_keys, int n_i64,
                              const char **scalar_keys, int n_scalar) {
    int total = n_dmat + n_iv + n_i64 + n_scalar;
    const char **names = (const char **)mxMalloc(sizeof(char*) * (size_t)total);
    int idx = 0;
    for (int i = 0; i < n_dmat; ++i)   names[idx++] = dmat_keys[i];
    for (int i = 0; i < n_iv; ++i)     names[idx++] = iv_keys[i];
    for (int i = 0; i < n_i64; ++i)    names[idx++] = i64_keys[i];
    for (int i = 0; i < n_scalar; ++i) names[idx++] = scalar_keys[i];
    mxArray *out = mxCreateStructMatrix(1, 1, total, names);
    mxFree(names);
    int field = 0;
    for (int i = 0; i < n_dmat; ++i)   add_double_matrix(out, field++, mr, dmat_keys[i]);
    for (int i = 0; i < n_iv; ++i)     add_int_vector(out, field++, mr, iv_keys[i]);
    for (int i = 0; i < n_i64; ++i)    add_int64_vector(out, field++, mr, i64_keys[i]);
    for (int i = 0; i < n_scalar; ++i) add_scalar(out, field++, mr, scalar_keys[i]);
    p4a_method_result_destroy(mr);
    return out;
}

/* ---- mexFunction ----------------------------------------------------- */

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs < 4) {
        mexErrMsgIdAndTxt("pls4all:nargin",
                          "Usage: p4a_method_fit_mex(algo, X, Y, n_components, params)");
    }
    if (!mxIsChar(prhs[0]))
        mexErrMsgIdAndTxt("pls4all:type", "algo must be a string");
    if (!mxIsDouble(prhs[1]) || !mxIsDouble(prhs[2]))
        mexErrMsgIdAndTxt("pls4all:type", "X and Y must be real double matrices");
    char algo_buf[64];
    if (mxGetString(prhs[0], algo_buf, sizeof(algo_buf)) != 0)
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

    if (strcmp(algo_buf, "sparse_simpls") == 0) {
        double lambda = get_scalar_field(params, "sparsity_lambda", 0.05);
        st = p4a_sparse_simpls_fit(ctx, cfg, &Xv, &Yv, lambda, &mr);
        if (st == P4A_OK) {
            const char *dm[] = {"coefficients", "predictions", "x_mean", "y_mean", "weights_w"};
            const char *sc[] = {"rmse"};
            out = pack_result(mr, dm, 5, NULL, 0, NULL, 0, sc, 1);
        }
    } else if (strcmp(algo_buf, "cppls") == 0) {
        double gamma = get_scalar_field(params, "gamma", 0.5);
        st = p4a_cppls_fit(ctx, cfg, &Xv, &Yv, gamma, &mr);
        if (st == P4A_OK) {
            const char *dm[] = {"coefficients", "predictions", "x_mean", "y_mean"};
            const char *sc[] = {"rmse"};
            out = pack_result(mr, dm, 4, NULL, 0, NULL, 0, sc, 1);
        }
    } else if (strcmp(algo_buf, "weighted_pls") == 0) {
        const mxArray *w = get_array_field(params, "sample_weights");
        if (!w || !mxIsDouble(w) || (int)mxGetNumberOfElements(w) != n) {
            p4a_config_destroy(cfg); p4a_context_destroy(ctx); free(X); free(Y);
            mexErrMsgIdAndTxt("pls4all:bad_arg",
                "weighted_pls requires params.sample_weights (length n)");
        }
        st = p4a_weighted_pls_fit(ctx, cfg, &Xv, &Yv,
                                    mxGetPr(w), (int64_t)mxGetNumberOfElements(w), &mr);
        if (st == P4A_OK) {
            const char *dm[] = {"coefficients", "predictions", "x_mean", "y_mean"};
            const char *sc[] = {"rmse"};
            out = pack_result(mr, dm, 4, NULL, 0, NULL, 0, sc, 1);
        }
    } else if (strcmp(algo_buf, "mb_pls") == 0) {
        const mxArray *bs = get_array_field(params, "block_sizes");
        if (!bs || (!mxIsDouble(bs) && !mxIsInt32(bs))) {
            p4a_config_destroy(cfg); p4a_context_destroy(ctx); free(X); free(Y);
            mexErrMsgIdAndTxt("pls4all:bad_arg",
                "mb_pls requires params.block_sizes");
        }
        int n_blocks = (int)mxGetNumberOfElements(bs);
        int64_t *bs_buf = (int64_t *)malloc((size_t)n_blocks * sizeof(int64_t));
        int64_t sum = 0;
        if (mxIsDouble(bs)) {
            const double *src = mxGetPr(bs);
            for (int i = 0; i < n_blocks; ++i) { bs_buf[i] = (int64_t)src[i]; sum += bs_buf[i]; }
        } else {
            const int32_t *src = (const int32_t *)mxGetData(bs);
            for (int i = 0; i < n_blocks; ++i) { bs_buf[i] = (int64_t)src[i]; sum += bs_buf[i]; }
        }
        if (sum != (int64_t)p) {
            free(bs_buf); p4a_config_destroy(cfg); p4a_context_destroy(ctx); free(X); free(Y);
            mexErrMsgIdAndTxt("pls4all:shape",
                "sum(block_sizes) must equal size(X, 2)");
        }
        p4a_config_set_solver(cfg, P4A_SOLVER_NIPALS);
        st = p4a_mb_pls_fit(ctx, cfg, &Xv, &Yv, bs_buf, n_blocks, &mr);
        free(bs_buf);
        if (st == P4A_OK) {
            const char *dm[] = {"coefficients", "predictions", "x_mean",
                                  "x_scale", "intercept", "block_weights"};
            const char *sc[] = {"rmse"};
            out = pack_result(mr, dm, 6, NULL, 0, NULL, 0, sc, 1);
        }
    } else if (strcmp(algo_buf, "pls_glm") == 0) {
        int poisson = get_int_field(params, "poisson", 0);
        st = p4a_pls_glm_fit(ctx, cfg, &Xv, &Yv, poisson, &mr);
        if (st == P4A_OK) {
            const char *dm[] = {"coefficients", "intercept", "predictions", "x_mean"};
            const char *sc[] = {"rmse"};
            out = pack_result(mr, dm, 4, NULL, 0, NULL, 0, sc, 1);
        }
    } else if (strcmp(algo_buf, "mir_pls") == 0) {
        st = p4a_mir_pls_fit(ctx, cfg, &Xv, &Yv, &mr);
        if (st == P4A_OK) {
            const char *dm[] = {"coefficients", "predictions", "x_mean", "y_mean"};
            const char *sc[] = {"rmse"};
            out = pack_result(mr, dm, 4, NULL, 0, NULL, 0, sc, 1);
        }
    } else if (strcmp(algo_buf, "spa_select") == 0) {
        int top_k = get_int_field(params, "top_k", 10);
        st = p4a_spa_select(ctx, cfg, &Xv, &Yv, top_k, &mr);
        if (st == P4A_OK) {
            const char *i64[] = {"selected_indices"};
            const char *sc[] = {"best_rmse"};
            out = pack_result(mr, NULL, 0, NULL, 0, i64, 1, sc, 1);
        }
    } else if (strcmp(algo_buf, "cars_select") == 0) {
        int n_iter = get_int_field(params, "n_iterations", 50);
        int min_f  = get_int_field(params, "min_features", 5);
        st = p4a_cars_select(ctx, cfg, &Xv, &Yv, NULL, n_iter, min_f, &mr);
        if (st == P4A_OK) {
            const char *i64[] = {"selected_indices"};
            const char *sc[] = {"best_rmse"};
            out = pack_result(mr, NULL, 0, NULL, 0, i64, 1, sc, 1);
        }
    } else if (strcmp(algo_buf, "approximate_press") == 0) {
        int max_k = n_components;  /* convention: n_components is max */
        st = p4a_approximate_press_compute(ctx, cfg, &Xv, &Yv, max_k, &mr);
        if (st == P4A_OK) {
            const char *dm[] = {"press_per_component", "rmse_per_component"};
            const char *iv[] = {"selected_n_components"};
            out = pack_result(mr, dm, 2, iv, 1, NULL, 0, NULL, 0);
        }
    } else {
        p4a_config_destroy(cfg); p4a_context_destroy(ctx); free(X); free(Y);
        mexErrMsgIdAndTxt("pls4all:bad_arg", "unknown algo: %s", algo_buf);
    }

    p4a_config_destroy(cfg);
    free(X); free(Y);
    if (st != P4A_OK) {
        const char *msg = p4a_context_last_error(ctx);
        char buf[512];
        snprintf(buf, sizeof(buf), "%s failed: %s (%s)", algo_buf,
                  p4a_status_to_string(st), msg ? msg : "");
        p4a_context_destroy(ctx);
        mexErrMsgIdAndTxt("pls4all:fit", "%s", buf);
    }
    p4a_context_destroy(ctx);
    plhs[0] = out;
}
