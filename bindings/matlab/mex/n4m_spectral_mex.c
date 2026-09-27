/* SPDX-License-Identifier: CECILL-2.1 */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "mex.h"
#include "n4m/n4m.h"

typedef struct Entry {
    uint64_t id;
    int64_t p;
    n4m_spectral_encoder_t* encoder;
    struct Entry* next;
} Entry;
static Entry* entries = NULL;
static uint64_t next_id = 1;
static int registered = 0;
static void cleanup(void) {
    while (entries) {
        Entry* next = entries->next;
        n4m_decomposition_spectral_destroy(entries->encoder);
        free(entries);
        entries = next;
    }
}
static void check(n4m_status_t s) {
    if (s != N4M_OK)
        mexErrMsgIdAndTxt("n4m:spectral", "%s", n4m_status_to_string(s));
}
static n4m_matrix_view_t view(const mxArray* x) {
    if (!mxIsDouble(x) || mxIsComplex(x) || mxIsSparse(x) || mxGetNumberOfDimensions(x) != 2)
        mexErrMsgIdAndTxt("n4m:spectral", "Expected a real dense double matrix");
    n4m_matrix_view_t v = {0};
    v.data = mxGetPr(x);
    v.rows = (int64_t)mxGetM(x);
    v.cols = (int64_t)mxGetN(x);
    v.row_stride = 1;
    v.col_stride = v.rows;
    v.dtype = N4M_DTYPE_F64;
    return v;
}
static Entry** lookup(const mxArray* value) {
    if (!mxIsUint64(value) || mxGetNumberOfElements(value) != 1)
        mexErrMsgIdAndTxt("n4m:spectral", "Invalid encoder handle");
    uint64_t id = *(uint64_t*)mxGetData(value);
    Entry** link = &entries;
    while (*link && (*link)->id != id)
        link = &(*link)->next;
    if (!*link)
        mexErrMsgIdAndTxt("n4m:spectral", "Closed or unknown encoder handle");
    return link;
}
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    char command[24];
    if (nrhs < 2 || !mxIsChar(prhs[0]) || mxGetString(prhs[0], command, sizeof(command)))
        mexErrMsgIdAndTxt("n4m:spectral", "Expected an encoder command");
    if (!strcmp(command, "fit")) {
        if (nrhs != 3 || nlhs != 1)
            mexErrMsgIdAndTxt("n4m:spectral", "fit requires X and parameters");
        n4m_matrix_view_t x = view(prhs[1]);
        if (!mxIsDouble(prhs[2]) || mxIsComplex(prhs[2]) || mxIsSparse(prhs[2])
            || mxGetNumberOfElements(prhs[2]) != 8)
            mexErrMsgIdAndTxt("n4m:spectral", "Expected eight encoder parameters");
        const double* p = mxGetPr(prhs[2]);
        for (int i = 0; i < 8; ++i)
            if (!isfinite(p[i]))
                mexErrMsgIdAndTxt("n4m:spectral", "Nonfinite encoder parameter");
        const int positions[] = {0, 1, 2, 4, 5, 6};
        for (int i = 0; i < 6; ++i) {
            double value = p[positions[i]];
            if (value < 0 || value > 1000000 || value != floor(value))
                mexErrMsgIdAndTxt("n4m:spectral", "Invalid integer encoder parameter");
        }
        plhs[0] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
        Entry* e = (Entry*)calloc(1, sizeof(Entry));
        if (!e)
            mexErrMsgIdAndTxt("n4m:spectral", "Allocation failed");
        n4m_status_t status = n4m_decomposition_spectral_create((int)p[0],
                                                                (int)p[1],
                                                                (int)p[2],
                                                                p[3],
                                                                (int)p[4],
                                                                (int)p[5],
                                                                (int)p[6],
                                                                p[7],
                                                                &e->encoder);
        if (status == N4M_OK)
            status = n4m_decomposition_spectral_fit(e->encoder, x);
        if (status != N4M_OK) {
            n4m_decomposition_spectral_destroy(e->encoder);
            free(e);
            check(status);
        }
        e->id = next_id++;
        e->p = x.cols;
        e->next = entries;
        entries = e;
        if (!registered) {
            mexAtExit(cleanup);
            registered = 1;
        }
        mexLock();
        *(uint64_t*)mxGetData(plhs[0]) = e->id;
        return;
    }
    Entry** link = lookup(prhs[1]);
    Entry* e = *link;
    if (!strcmp(command, "destroy")) {
        if (nrhs != 2 || nlhs != 0)
            mexErrMsgIdAndTxt("n4m:spectral", "destroy requires one handle");
        *link = e->next;
        n4m_decomposition_spectral_destroy(e->encoder);
        free(e);
        mexUnlock();
        return;
    }
    int64_t q = 0;
    check(n4m_decomposition_spectral_output_cols(e->encoder, &q));
    if (!strcmp(command, "transform")) {
        if (nrhs != 3 || nlhs != 1)
            mexErrMsgIdAndTxt("n4m:spectral", "transform requires handle and X");
        n4m_matrix_view_t x = view(prhs[2]);
        plhs[0] = mxCreateDoubleMatrix((mwSize)x.rows, (mwSize)q, mxREAL);
        check(n4m_decomposition_spectral_transform(e->encoder, x, view(plhs[0])));
    } else if (!strcmp(command, "affine")) {
        if (nrhs != 2 || nlhs != 1)
            mexErrMsgIdAndTxt("n4m:spectral", "affine requires one handle");
        const char* fields[] = {"operator", "offset"};
        plhs[0] = mxCreateStructMatrix(1, 1, 2, fields);
        mxArray* op = mxCreateDoubleMatrix((mwSize)q, (mwSize)e->p, mxREAL);
        mxSetField(plhs[0], 0, "operator", op);
        mxArray* offset = mxCreateDoubleMatrix(1, (mwSize)q, mxREAL);
        mxSetField(plhs[0], 0, "offset", offset);
        check(n4m_decomposition_spectral_export_affine(e->encoder, view(op), view(offset)));
    } else
        mexErrMsgIdAndTxt("n4m:spectral", "Unknown encoder command");
}
