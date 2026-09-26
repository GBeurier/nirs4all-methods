/* SPDX-License-Identifier: CECILL-2.1
 * Native C-ABI N-PLS oracle: compare the WASM raw-pointer shim with the
 * canonical SIMPLS/no-scale direct fit and held-out affine predictions.
 */
#include "n4m/n4m.h"
#include "n4m/estimators/regression.h"
#include <math.h>
#include <stdio.h>

int n4m_wasm_model_fit(const char*, const double*, int, const double*,
                       const double*, int, int, int, int, double*, double*,
                       double*, double*, int*, double*);
int n4m_wasm_model_predict_from_coeffs(const double*, const double*,
                                       const double*, const double*,
                                       const double*, int, int, int, double*);

static int compare(const double* lhs, const double* rhs, int count,
                   const char* label, double* max_diff) {
    for (int i = 0; i < count; ++i) {
        const double diff = fabs(lhs[i] - rhs[i]);
        if (diff > *max_diff) *max_diff = diff;
        if (!isfinite(lhs[i]) || diff > 1e-10) {
            fprintf(stderr, "%s[%d]: %.17g != %.17g\n", label, i, lhs[i], rhs[i]);
            return 0;
        }
    }
    return 1;
}

static int run_case(int q) {
    enum { N = 24, J = 3, K = 4, P = J * K, HELDOUT = 3 };
    double x[N * P], y[N * 2], new_x[HELDOUT * P];
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < P; ++col) {
            x[row * P + col] = sin((row + 1.0) * (col + 1.0) / 8.0)
                + cos((row + 1.0) / 5.0 - col / 7.0) + (row + 1.0) * col / 100.0;
        }
        y[row * q] = 1.1 + 0.8 * x[row * P + 1] - 0.4 * x[row * P + 7];
        if (q == 2) y[row * q + 1] = -0.5 + 0.3 * x[row * P + 3] + 0.7 * x[row * P + 10];
    }
    const int heldout_rows[] = {2, 10, 19};
    for (int row = 0; row < HELDOUT; ++row)
        for (int col = 0; col < P; ++col)
            new_x[row * P + col] = x[heldout_rows[row] * P + col] + 0.023 * (col + 1);
    n4m_matrix_view_t xv, yv;
    n4m_matrix_view_init_rowmajor(&xv, x, N, P, N4M_DTYPE_F64);
    n4m_matrix_view_init_rowmajor(&yv, y, N, q, N4M_DTYPE_F64);
    n4m_context_t* ctx = NULL;
    n4m_config_t* cfg = NULL;
    n4m_method_result_t* result = NULL;
    n4m_status_t status = n4m_context_create(&ctx);
    if (status == N4M_OK) status = n4m_config_create(&cfg);
    if (status == N4M_OK) status = n4m_config_set_n_components(cfg, 2);
    if (status == N4M_OK) status = n4m_config_set_center_x(cfg, 1);
    if (status == N4M_OK) status = n4m_config_set_center_y(cfg, 1);
    if (status == N4M_OK) status = n4m_config_set_scale_x(cfg, 0);
    if (status == N4M_OK) status = n4m_config_set_scale_y(cfg, 0);
    if (status == N4M_OK) status = n4m_config_set_solver(cfg, N4M_SOLVER_SIMPLS);
    if (status == N4M_OK) status = n4m_estimators_n_pls_fit(ctx, cfg, &xv, J, K, &yv, &result);
    if (status != N4M_OK || result == NULL) {
        fprintf(stderr, "direct NPLS q=%d failed: %d %s\n", q, status,
                ctx == NULL ? "" : n4m_context_last_error(ctx));
        return 1;
    }
    double coef[P * 2], xm[P], ym[2], intercept[2], train[N * 2];
    int has_intercept = -1;
    const double dims[] = {J, K};
    status = n4m_wasm_model_fit("NPLS", dims, 2, x, y, N, P, q, 2,
                                coef, xm, ym, intercept, &has_intercept, train);
    if (status != N4M_OK || has_intercept != 0) {
        fprintf(stderr, "shim NPLS q=%d failed: %d / intercept=%d\n", q, status, has_intercept);
        return 1;
    }
    const char* keys[] = {"coefficients", "x_mean", "y_mean", "predictions"};
    const double* got[] = {coef, xm, ym, train};
    const int sizes[] = {P * q, P, q, N * q};
    double max_diff = 0.0;
    for (int field = 0; field < 4; ++field) {
        const double* expected = NULL;
        int64_t rows = 0, cols = 0;
        status = n4m_method_result_get_double_matrix(result, keys[field],
                                                      &expected, &rows, &cols);
        if (status != N4M_OK || expected == NULL || rows * cols != sizes[field] ||
            !compare(got[field], expected, sizes[field], keys[field], &max_diff)) return 1;
    }
    double replay[HELDOUT * 2], expected[HELDOUT * 2];
    status = n4m_wasm_model_predict_from_coeffs(coef, xm, ym, NULL,
                                                new_x, HELDOUT, P, q, replay);
    if (status != N4M_OK) return 1;
    const double *direct_coef = NULL, *direct_xm = NULL, *direct_ym = NULL;
    int64_t rows = 0, cols = 0;
    if (n4m_method_result_get_double_matrix(result, "coefficients", &direct_coef, &rows, &cols) != N4M_OK ||
        n4m_method_result_get_double_matrix(result, "x_mean", &direct_xm, &rows, &cols) != N4M_OK ||
        n4m_method_result_get_double_matrix(result, "y_mean", &direct_ym, &rows, &cols) != N4M_OK) return 1;
    for (int row = 0; row < HELDOUT; ++row) {
        for (int target = 0; target < q; ++target) {
            double value = direct_ym[target];
            for (int col = 0; col < P; ++col)
                value += (new_x[row * P + col] - direct_xm[col]) * direct_coef[col * q + target];
            expected[row * q + target] = value;
        }
    }
    if (!compare(replay, expected, HELDOUT * q, "held-out", &max_diff)) return 1;
    const double bad_dims[][2] = {{0, K}, {J, 3}, {1.5, K}, {2147483647, 2147483647}};
    for (int index = 0; index < 4; ++index) {
        status = n4m_wasm_model_fit("NPLS", bad_dims[index], 2, x, y, N, P, q, 2,
                                    coef, xm, ym, intercept, &has_intercept, NULL);
        if (status != N4M_ERR_INVALID_ARGUMENT && status != N4M_ERR_SHAPE_MISMATCH) {
            fprintf(stderr, "bad dimensions accepted q=%d case=%d status=%d\n", q, index, status);
            return 1;
        }
    }
    printf("NPLS %dx%d -> %d target(s), %d held-out: max_abs_diff=%.17g\n",
           N, P, q, HELDOUT, max_diff);
    n4m_method_result_destroy(result);
    n4m_config_destroy(cfg);
    n4m_context_destroy(ctx);
    return 0;
}

int main(void) { return run_case(1) || run_case(2); }
