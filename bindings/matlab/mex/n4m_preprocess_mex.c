/* SPDX-License-Identifier: CECILL-2.1
 *
 * MATLAB / Octave MEX dispatcher for stateless preprocessing operators.
 *
 *   Xout = n4m_preprocess_mex("snv", X, with_mean, with_std, ddof)
 *   Xout = n4m_preprocess_mex("savgol", X, window, polyorder, deriv, delta, mode, cval)
 */

#include "mex.h"
#include "n4m/n4m.h"

#include <stdlib.h>
#include <string.h>

static double* colmajor_to_rowmajor(const mxArray* mx, int* rows_out, int* cols_out) {
    int rows = (int)mxGetM(mx);
    int cols = (int)mxGetN(mx);
    const double* src = mxGetPr(mx);
    double* dst = (double*)malloc((size_t)rows * (size_t)cols * sizeof(double));
    *rows_out = rows;
    *cols_out = cols;
    if (dst == NULL && rows * cols > 0) {
        return NULL;
    }
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            dst[i * cols + j] = src[i + j * rows];
        }
    }
    return dst;
}

static mxArray* rowmajor_to_colmajor(const double* src, int rows, int cols) {
    mxArray* out = mxCreateDoubleMatrix(rows, cols, mxREAL);
    double* dst = mxGetPr(out);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            dst[i + j * rows] = src[i * cols + j];
        }
    }
    return out;
}

static n4m_matrix_view_t rowmajor_view(double* data, int rows, int cols) {
    n4m_matrix_view_t view;
    n4m_status_t status = n4m_matrix_view_init_rowmajor(
        &view, data, (int64_t)rows, (int64_t)cols, N4M_DTYPE_F64);
    if (status != N4M_OK) {
        mexErrMsgIdAndTxt("pls4all:view", "failed to create matrix view: status %d", (int)status);
    }
    return view;
}

static int mode_from_arg(const mxArray* arg) {
    if (mxIsChar(arg)) {
        char mode[32];
        if (mxGetString(arg, mode, sizeof(mode)) != 0) {
            mexErrMsgIdAndTxt("pls4all:savgol", "mode string is too long");
        }
        if (strcmp(mode, "mirror") == 0) return N4M_PP_SAVGOL_MIRROR;
        if (strcmp(mode, "constant") == 0) return N4M_PP_SAVGOL_CONSTANT;
        if (strcmp(mode, "nearest") == 0) return N4M_PP_SAVGOL_NEAREST;
        if (strcmp(mode, "wrap") == 0) return N4M_PP_SAVGOL_WRAP;
        if (strcmp(mode, "interp") == 0) return N4M_PP_SAVGOL_INTERP;
        mexErrMsgIdAndTxt("pls4all:savgol", "unsupported Savitzky-Golay mode: %s", mode);
    }
    return (int)mxGetScalar(arg);
}

static void run_snv(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    if (nrhs != 5) {
        mexErrMsgIdAndTxt("pls4all:nargin", "Usage: n4m_preprocess_mex('snv', X, with_mean, with_std, ddof)");
    }
    (void)nlhs;
    int rows = 0, cols = 0;
    double* X = colmajor_to_rowmajor(prhs[1], &rows, &cols);
    double* out = (double*)malloc((size_t)rows * (size_t)cols * sizeof(double));
    if ((X == NULL || out == NULL) && rows * cols > 0) {
        free(X);
        free(out);
        mexErrMsgIdAndTxt("pls4all:oom", "out of memory in snv preprocessing");
    }

    n4m_pp_snv_handle_t* handle = NULL;
    n4m_status_t status = n4m_pp_snv_create(
        &handle,
        mxGetScalar(prhs[2]) != 0.0 ? 1 : 0,
        mxGetScalar(prhs[3]) != 0.0 ? 1 : 0,
        (int)mxGetScalar(prhs[4]));
    if (status == N4M_OK) {
        n4m_matrix_view_t xv = rowmajor_view(X, rows, cols);
        n4m_matrix_view_t ov = rowmajor_view(out, rows, cols);
        status = n4m_pp_snv_transform(handle, xv, ov);
    }
    n4m_pp_snv_destroy(handle);
    if (status != N4M_OK) {
        free(X);
        free(out);
        mexErrMsgIdAndTxt("pls4all:snv", "SNV transform failed with status %d", (int)status);
    }
    plhs[0] = rowmajor_to_colmajor(out, rows, cols);
    free(X);
    free(out);
}

static void run_savgol(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    if (nrhs != 8) {
        mexErrMsgIdAndTxt("pls4all:nargin", "Usage: n4m_preprocess_mex('savgol', X, window, polyorder, deriv, delta, mode, cval)");
    }
    (void)nlhs;
    int rows = 0, cols = 0;
    double* X = colmajor_to_rowmajor(prhs[1], &rows, &cols);
    double* out = (double*)malloc((size_t)rows * (size_t)cols * sizeof(double));
    if ((X == NULL || out == NULL) && rows * cols > 0) {
        free(X);
        free(out);
        mexErrMsgIdAndTxt("pls4all:oom", "out of memory in Savitzky-Golay preprocessing");
    }

    n4m_pp_savgol_handle_t* handle = NULL;
    n4m_status_t status = n4m_pp_savgol_create(
        &handle,
        (int32_t)mxGetScalar(prhs[2]),
        (int32_t)mxGetScalar(prhs[3]),
        (int32_t)mxGetScalar(prhs[4]),
        mxGetScalar(prhs[5]),
        (n4m_pp_savgol_mode_t)mode_from_arg(prhs[6]),
        mxGetScalar(prhs[7]));
    if (status == N4M_OK) {
        n4m_matrix_view_t xv = rowmajor_view(X, rows, cols);
        n4m_matrix_view_t ov = rowmajor_view(out, rows, cols);
        status = n4m_pp_savgol_transform(handle, xv, ov);
    }
    n4m_pp_savgol_destroy(handle);
    if (status != N4M_OK) {
        free(X);
        free(out);
        mexErrMsgIdAndTxt("pls4all:savgol", "Savitzky-Golay transform failed with status %d", (int)status);
    }
    plhs[0] = rowmajor_to_colmajor(out, rows, cols);
    free(X);
    free(out);
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    if (nrhs < 2 || nlhs > 1 || !mxIsChar(prhs[0]) || !mxIsDouble(prhs[1])) {
        mexErrMsgIdAndTxt("pls4all:nargin", "Usage: n4m_preprocess_mex(command, X, ...)");
    }
    char cmd[32];
    if (mxGetString(prhs[0], cmd, sizeof(cmd)) != 0) {
        mexErrMsgIdAndTxt("pls4all:cmd", "preprocessing command is too long");
    }
    if (strcmp(cmd, "snv") == 0) {
        run_snv(nlhs, plhs, nrhs, prhs);
        return;
    }
    if (strcmp(cmd, "savgol") == 0) {
        run_savgol(nlhs, plhs, nrhs, prhs);
        return;
    }
    mexErrMsgIdAndTxt("pls4all:cmd", "unknown preprocessing command: %s", cmd);
}
