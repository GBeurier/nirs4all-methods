/* SPDX-License-Identifier: CECILL-2.1 */
/* Raw-pointer adapters for by-value matrix views; no numerical code. */
#include "n4m/n4m.h"

static n4m_matrix_view_t spectral_view(double* data, int n, int p) {
    n4m_matrix_view_t v = {0};
    v.data = data;
    v.rows = n;
    v.cols = p;
    v.row_stride = p;
    v.col_stride = 1;
    v.dtype = N4M_DTYPE_F64;
    return v;
}
int n4m_wasm_spectral_fit(n4m_spectral_encoder_t* h, double* x, int n, int p) {
    return n4m_decomposition_spectral_fit(h, spectral_view(x, n, p));
}
int n4m_wasm_spectral_transform(
    n4m_spectral_encoder_t* h, double* x, int n, int p, double* out, int q) {
    return n4m_decomposition_spectral_transform(
        h, spectral_view(x, n, p), spectral_view(out, n, q));
}
int n4m_wasm_spectral_affine(n4m_spectral_encoder_t* h, double* op, double* offset, int p, int q) {
    return n4m_decomposition_spectral_export_affine(
        h, spectral_view(op, q, p), spectral_view(offset, 1, q));
}
