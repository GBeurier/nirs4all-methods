/* SPDX-License-Identifier: CECILL-2.1
 *
 * MATLAB / Octave MEX dispatcher for Model-based fits (PCR, OPLS).
 *
 *   [coefs, x_mean, y_mean, predictions] = p4a_model_fit_mex(algo, X, Y, n_components)
 *
 * `algo` is one of:
 *   "pcr"   -> P4A_ALGO_PCR + P4A_SOLVER_SVD + P4A_DEFLATION_REGRESSION
 *   "opls"  -> P4A_ALGO_OPLS + P4A_SOLVER_NIPALS + P4A_DEFLATION_ORTHOGONAL
 *
 * Inputs:
 *   algo         — string ("pcr" or "opls")
 *   X            — double matrix (n × p), column-major (MATLAB default).
 *   Y            — double matrix (n × q), column-major.
 *   n_components — positive integer (scalar).
 *
 * Outputs:
 *   coefficients — (p × q) column-major.
 *   x_mean       — (1 × p) row vector.
 *   y_mean       — (1 × q) row vector.
 *   predictions  — (n × q) in-sample predictions.
 *
 * Uses the full p4a_model_fit() API with configurable algorithm/solver/deflation.
 */

#include "mex.h"
#include "pls4all/p4a.h"

#include <stdlib.h>
#include <string.h>

/* Transpose a column-major mxArray into a freshly malloc'd row-major
 * buffer the C ABI expects. Returns NULL on alloc failure. */
static double* colmajor_to_rowmajor(const mxArray *mx, int *rows_out,
                                     int *cols_out) {
    int rows = (int)mxGetM(mx);
    int cols = (int)mxGetN(mx);
    *rows_out = rows;
    *cols_out = cols;
    if (rows * cols == 0) {
        return NULL;
    }
    const double *src = mxGetPr(mx);
    double *dst = (double *)malloc((size_t)rows * (size_t)cols *
                                     sizeof(double));
    if (dst == NULL) {
        return NULL;
    }
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            dst[i * cols + j] = src[i + j * rows];   /* col-major → row */
        }
    }
    return dst;
}

/* Copy a row-major buffer (rows × cols) into a freshly created MATLAB
 * column-major double matrix and return the mxArray. */
static mxArray* rowmajor_to_colmajor(const double *src, int rows, int cols) {
    mxArray *out = mxCreateDoubleMatrix(rows, cols, mxREAL);
    double *dst = mxGetPr(out);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            dst[i + j * rows] = src[i * cols + j];
        }
    }
    return out;
}

static mxArray* view_to_colmajor_mx(const p4a_matrix_view_t *view) {
    int rows = (int)view->rows;
    int cols = (int)view->cols;
    mxArray *out = mxCreateDoubleMatrix(rows, cols, mxREAL);
    double *dst = mxGetPr(out);
    const double *src = (const double *)view->data;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            dst[i + j * rows] =
                src[i * view->row_stride + j * view->col_stride];
        }
    }
    return out;
}

static mxArray* model_array_to_mx(p4a_context_t *ctx,
                                   const p4a_model_t *model,
                                   p4a_model_array_t which,
                                   const char *name) {
    p4a_array_t *arr = NULL;
    p4a_status_t status = p4a_model_get_array(ctx, model, which, &arr);
    if (status != P4A_OK || arr == NULL) {
        mexErrMsgIdAndTxt("pls4all:model_array",
                          "p4a_model_get_array(%s) failed with status %d",
                          name, (int)status);
    }
    p4a_dtype_t dtype;
    status = p4a_array_dtype(arr, &dtype);
    if (status != P4A_OK || dtype != P4A_DTYPE_F64) {
        p4a_array_free(arr);
        mexErrMsgIdAndTxt("pls4all:model_array",
                          "p4a_model_get_array(%s) returned non-f64 data",
                          name);
    }
    p4a_matrix_view_t view;
    status = p4a_array_view(arr, &view);
    if (status != P4A_OK) {
        p4a_array_free(arr);
        mexErrMsgIdAndTxt("pls4all:model_array",
                          "p4a_array_view(%s) failed with status %d",
                          name, (int)status);
    }
    mxArray *out = view_to_colmajor_mx(&view);
    p4a_array_free(arr);
    return out;
}

static mxArray* predict_to_mx(p4a_context_t *ctx,
                               const p4a_model_t *model,
                               const p4a_matrix_view_t *X_view) {
    p4a_array_t *arr = NULL;
    p4a_status_t status = p4a_model_predict_alloc(ctx, model, X_view, &arr);
    if (status != P4A_OK || arr == NULL) {
        mexErrMsgIdAndTxt("pls4all:predict",
                          "p4a_model_predict_alloc failed with status %d",
                          (int)status);
    }
    p4a_matrix_view_t view;
    status = p4a_array_view(arr, &view);
    if (status != P4A_OK) {
        p4a_array_free(arr);
        mexErrMsgIdAndTxt("pls4all:predict",
                          "p4a_array_view(predictions) failed with status %d",
                          (int)status);
    }
    mxArray *out = view_to_colmajor_mx(&view);
    p4a_array_free(arr);
    return out;
}

static int resolve_algo(const char *name,
                        p4a_algorithm_t *out_algo,
                        p4a_solver_t *out_solver,
                        p4a_deflation_t *out_deflation) {
    if (strcmp(name, "pcr") == 0 || strcmp(name, "pcr_svd") == 0) {
        *out_algo = P4A_ALGO_PCR;
        *out_solver = P4A_SOLVER_SVD;
        *out_deflation = P4A_DEFLATION_REGRESSION;
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

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]) {
    if (nrhs != 4) {
        mexErrMsgIdAndTxt("pls4all:nargin",
                          "Usage: p4a_model_fit_mex(algo, X, Y, n_components)");
    }
    if (!mxIsChar(prhs[0])) {
        mexErrMsgIdAndTxt("pls4all:type",
                          "algo must be a string");
    }
    if (!mxIsDouble(prhs[1]) || !mxIsDouble(prhs[2])) {
        mexErrMsgIdAndTxt("pls4all:type",
                          "X and Y must be real double matrices");
    }
    if (!mxIsDouble(prhs[3]) && !mxIsClass(prhs[3], "int32") &&
        !mxIsClass(prhs[3], "int64")) {
        mexErrMsgIdAndTxt("pls4all:type",
                          "n_components must be numeric");
    }

    char algo_name[80];
    if (mxGetString(prhs[0], algo_name, sizeof(algo_name)) != 0) {
        mexErrMsgIdAndTxt("pls4all:str", "algo name too long");
    }

    p4a_algorithm_t algo;
    p4a_solver_t solver;
    p4a_deflation_t deflation;
    if (!resolve_algo(algo_name, &algo, &solver, &deflation)) {
        mexErrMsgIdAndTxt("pls4all:bad_arg",
                          "unknown algo: %s (supported: pcr, opls)", algo_name);
    }

    int n_components = (int)mxGetScalar(prhs[3]);
    if (n_components < 1) {
        mexErrMsgIdAndTxt("pls4all:bad_arg",
                          "n_components must be >= 1");
    }

    int n_x, p, n_y, q;
    double *X = colmajor_to_rowmajor(prhs[1], &n_x, &p);
    double *Y = colmajor_to_rowmajor(prhs[2], &n_y, &q);
    if (X == NULL || Y == NULL) {
        if (X) free(X); if (Y) free(Y);
        mexErrMsgIdAndTxt("pls4all:oom",
                          "out of memory copying X / Y");
    }
    if (n_x != n_y) {
        free(X); free(Y);
        mexErrMsgIdAndTxt("pls4all:shape",
                          "size(X, 1) must equal size(Y, 1)");
    }
    int n = n_x;

    /* Create context */
    p4a_context_t *ctx = NULL;
    p4a_status_t status = p4a_context_create(&ctx);
    if (status != P4A_OK) {
        free(X); free(Y);
        mexErrMsgIdAndTxt("pls4all:ctx",
                          "p4a_context_create failed with status %d",
                          (int)status);
    }

    /* Create and configure config */
    p4a_config_t *cfg = NULL;
    status = p4a_config_create(&cfg);
    if (status != P4A_OK) {
        p4a_context_destroy(ctx);
        free(X); free(Y);
        mexErrMsgIdAndTxt("pls4all:cfg",
                          "p4a_config_create failed with status %d",
                          (int)status);
    }
    p4a_config_set_algorithm(cfg, algo);
    p4a_config_set_solver(cfg, solver);
    p4a_config_set_deflation(cfg, deflation);
    p4a_config_set_n_components(cfg, n_components);
    p4a_config_set_center_x(cfg, 1);
    p4a_config_set_scale_x(cfg, 0);
    p4a_config_set_center_y(cfg, 1);
    p4a_config_set_scale_y(cfg, 0);
    p4a_config_set_tol(cfg, 1e-6);
    p4a_config_set_max_iter(cfg, 500);
    p4a_config_set_store_scores(cfg, 0);

    /* Create matrix views */
    p4a_matrix_view_t X_view, Y_view;
    p4a_matrix_view_init_rowmajor(&X_view, X, n, p, P4A_DTYPE_F64);
    p4a_matrix_view_init_rowmajor(&Y_view, Y, n, q, P4A_DTYPE_F64);

    /* Fit model */
    p4a_model_t *model = NULL;
    status = p4a_model_fit(ctx, cfg, &X_view, &Y_view, &model);
    if (status != P4A_OK) {
        const char *msg = p4a_context_last_error(ctx);
        p4a_config_destroy(cfg);
        p4a_context_destroy(ctx);
        free(X); free(Y);
        if (msg && msg[0]) {
            mexErrMsgIdAndTxt("pls4all:fit",
                              "p4a_model_fit failed: %s (%s)",
                              p4a_status_to_string(status), msg);
        } else {
            mexErrMsgIdAndTxt("pls4all:fit",
                              "p4a_model_fit failed with status %d",
                              (int)status);
        }
    }

    plhs[0] = model_array_to_mx(ctx, model, P4A_MODEL_COEFFICIENTS,
                                "coefficients");
    if (nlhs >= 2)
        plhs[1] = model_array_to_mx(ctx, model, P4A_MODEL_X_MEAN, "x_mean");
    if (nlhs >= 3)
        plhs[2] = model_array_to_mx(ctx, model, P4A_MODEL_Y_MEAN, "y_mean");
    if (nlhs >= 4)
        plhs[3] = predict_to_mx(ctx, model, &X_view);

    /* Cleanup */
    p4a_model_destroy(model);
    p4a_config_destroy(cfg);
    p4a_context_destroy(ctx);
    free(X); free(Y);
}
