/* SPDX-License-Identifier: CECILL-2.1
 *
 * MATLAB / Octave MEX entry: `pls4all_pls_fit_mex(X, Y, n_components)`.
 *
 * Inputs:
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
 * Calls the public C ABI helper `n4m_pls_fit_simple` (ABI 1.13+).
 * Build via build_mex.m or the bundled CMake target. The MEX uses the
 * mxGetData / mxCreateDoubleMatrix C interface so it works on both
 * MATLAB (>= R2018a) and Octave (>= 6.0).
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

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]) {
    if (nrhs != 3) {
        mexErrMsgIdAndTxt("pls4all:nargin",
                          "Usage: n4m_pls_fit_mex(X, Y, n_components)");
    }
    if (!mxIsDouble(prhs[0]) || !mxIsDouble(prhs[1])) {
        mexErrMsgIdAndTxt("pls4all:type",
                          "X and Y must be real double matrices");
    }
    if (!mxIsDouble(prhs[2]) && !mxIsClass(prhs[2], "int32") &&
        !mxIsClass(prhs[2], "int64")) {
        mexErrMsgIdAndTxt("pls4all:type",
                          "n_components must be numeric");
    }
    int n_components = (int)mxGetScalar(prhs[2]);
    if (n_components < 1) {
        mexErrMsgIdAndTxt("pls4all:bad_arg",
                          "n_components must be >= 1");
    }

    int n_x, p, n_y, q;
    double *X = colmajor_to_rowmajor(prhs[0], &n_x, &p);
    double *Y = colmajor_to_rowmajor(prhs[1], &n_y, &q);
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

    double *coefs = (double *)malloc((size_t)p * (size_t)q *
                                      sizeof(double));
    double *x_mean = (double *)malloc((size_t)p * sizeof(double));
    double *y_mean = (double *)malloc((size_t)q * sizeof(double));
    double *preds = (double *)malloc((size_t)n * (size_t)q *
                                      sizeof(double));
    if (!coefs || !x_mean || !y_mean || !preds) {
        free(X); free(Y);
        free(coefs); free(x_mean); free(y_mean); free(preds);
        mexErrMsgIdAndTxt("pls4all:oom",
                          "out of memory allocating outputs");
    }

    n4m_status_t status = n4m_pls_fit_simple(
        X, Y, n, p, q, n_components,
        coefs, x_mean, y_mean, preds);
    free(X); free(Y);
    if (status != N4M_OK) {
        free(coefs); free(x_mean); free(y_mean); free(preds);
        mexErrMsgIdAndTxt("pls4all:fit",
                          "n4m_pls_fit_simple failed with status %d",
                          (int)status);
    }

    plhs[0] = rowmajor_to_colmajor(coefs, p, q);
    if (nlhs >= 2) plhs[1] = rowmajor_to_colmajor(x_mean, 1, p);
    if (nlhs >= 3) plhs[2] = rowmajor_to_colmajor(y_mean, 1, q);
    if (nlhs >= 4) plhs[3] = rowmajor_to_colmajor(preds, n, q);

    free(coefs); free(x_mean); free(y_mean); free(preds);
}
