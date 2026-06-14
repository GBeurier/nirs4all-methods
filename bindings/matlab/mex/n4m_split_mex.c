/* SPDX-License-Identifier: CECILL-2.1
 *
 * MATLAB / Octave MEX dispatcher for sample splitters.
 *
 *   split = n4m_split_mex("kennard_stone", X, test_size, zero_based)
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

static n4m_matrix_view_t rowmajor_view(double* data, int rows, int cols) {
    n4m_matrix_view_t view;
    n4m_status_t status = n4m_matrix_view_init_rowmajor(
        &view, data, (int64_t)rows, (int64_t)cols, N4M_DTYPE_F64);
    if (status != N4M_OK) {
        mexErrMsgIdAndTxt("pls4all:view", "failed to create matrix view: status %d", (int)status);
    }
    return view;
}

static mxArray* indices_to_column(const int64_t* values, int64_t count, int zero_based) {
    mxArray* out = mxCreateDoubleMatrix((mwSize)count, 1, mxREAL);
    double* dst = mxGetPr(out);
    for (int64_t i = 0; i < count; ++i) {
        dst[i] = (double)(values[i] + (zero_based ? 0 : 1));
    }
    return out;
}

static void run_kennard_stone(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    if (nrhs < 3 || nrhs > 4 || nlhs > 1) {
        mexErrMsgIdAndTxt("pls4all:nargin", "Usage: n4m_split_mex('kennard_stone', X, test_size, zero_based)");
    }
    int zero_based = 0;
    if (nrhs >= 4) {
        zero_based = mxGetScalar(prhs[3]) != 0.0;
    }

    int rows = 0, cols = 0;
    double* X = colmajor_to_rowmajor(prhs[1], &rows, &cols);
    if (X == NULL && rows * cols > 0) {
        mexErrMsgIdAndTxt("pls4all:oom", "out of memory copying X");
    }

    n4m_split_kennard_stone_handle_t* handle = NULL;
    n4m_status_t status = n4m_model_selection_kennard_stone_create(&handle, mxGetScalar(prhs[2]));
    n4m_split_result_t result;
    result.train_idx = NULL;
    result.n_train = 0;
    result.test_idx = NULL;
    result.n_test = 0;
    result._owner = NULL;
    if (status == N4M_OK) {
        n4m_matrix_view_t xv = rowmajor_view(X, rows, cols);
        status = n4m_model_selection_kennard_stone_split(handle, xv, &result);
    }
    n4m_model_selection_kennard_stone_destroy(handle);
    free(X);
    if (status != N4M_OK) {
        n4m_split_result_destroy(&result);
        mexErrMsgIdAndTxt("pls4all:split", "Kennard-Stone split failed with status %d", (int)status);
    }

    const char* fields[] = {"train", "test"};
    mxArray* out = mxCreateStructMatrix(1, 1, 2, fields);
    mxSetField(out, 0, "train", indices_to_column(result.train_idx, result.n_train, zero_based));
    mxSetField(out, 0, "test", indices_to_column(result.test_idx, result.n_test, zero_based));
    n4m_split_result_destroy(&result);
    plhs[0] = out;
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    if (nrhs < 2 || !mxIsChar(prhs[0]) || !mxIsDouble(prhs[1])) {
        mexErrMsgIdAndTxt("pls4all:nargin", "Usage: n4m_split_mex(command, X, ...)");
    }
    char cmd[32];
    if (mxGetString(prhs[0], cmd, sizeof(cmd)) != 0) {
        mexErrMsgIdAndTxt("pls4all:cmd", "split command is too long");
    }
    if (strcmp(cmd, "kennard_stone") == 0) {
        run_kennard_stone(nlhs, plhs, nrhs, prhs);
        return;
    }
    mexErrMsgIdAndTxt("pls4all:cmd", "unknown split command: %s", cmd);
}
