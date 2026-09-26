/* SPDX-License-Identifier: CECILL-2.1
 * Native held-out oracle for the canonical SIMPLS/OmicsPLS O2PLS shim.
 * Compile with wasm_entry.c and a matching native libn4m, as with
 * native_portable_ensemble_parity.c. No WASM artifact is exercised here.
 */
#include "n4m/n4m.h"
#include "n4m/estimators/multiblock.h"
#include <math.h>
#include <stdio.h>

int n4m_wasm_model_fit(const char*, const double*, int, const double*,
                       const double*, int, int, int, int, double*, double*,
                       double*, double*, int*, double*);
int n4m_wasm_model_predict_from_coeffs(const double*, const double*,
                                       const double*, const double*,
                                       const double*, int, int, int, double*);

static int check(const double* actual, const double* expected, int size,
                 const char* label, double* max_diff) {
    for (int i = 0; i < size; ++i) {
        const double delta = fabs(actual[i] - expected[i]);
        if (delta > *max_diff) *max_diff = delta;
        if (!isfinite(actual[i]) || delta > 1e-10) {
            fprintf(stderr, "%s[%d]: %.17g != %.17g\n", label, i, actual[i], expected[i]);
            return 0;
        }
    }
    return 1;
}

int main(void) {
    enum { N = 24, P = 8, Q = 3, HELDOUT = 3 };
    double x[N * P], y[N * Q], new_x[HELDOUT * P];
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < P; ++col) {
            x[row * P + col] = sin((row + 1.0) * (col + 1.0) / 9.0)
                + cos((row + 1.0) / 5.0 - col / 7.0) + (row + 1.0) * col / 100.0;
        }
        y[row * Q] = 1.2 + 0.7 * x[row * P + 1] - 0.3 * x[row * P + 5];
        y[row * Q + 1] = -0.4 + 0.2 * x[row * P + 2] + 0.5 * x[row * P + 6];
        y[row * Q + 2] = 0.8 - 0.6 * x[row * P + 0] + 0.4 * x[row * P + 7];
    }
    const int selected[] = {2, 9, 18};
    for (int row = 0; row < HELDOUT; ++row)
        for (int col = 0; col < P; ++col)
            new_x[row * P + col] = x[selected[row] * P + col] + 0.031 * (col + 1);

    n4m_matrix_view_t xv, yv;
    n4m_matrix_view_init_rowmajor(&xv, x, N, P, N4M_DTYPE_F64);
    n4m_matrix_view_init_rowmajor(&yv, y, N, Q, N4M_DTYPE_F64);
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
    if (status == N4M_OK)
        status = n4m_estimators_o2pls_fit(ctx, cfg, &xv, &yv, 2, 1, 1, &result);
    if (status != N4M_OK || result == NULL) {
        fprintf(stderr, "direct O2PLS fit failed: %d %s\n", status,
                ctx == NULL ? "" : n4m_context_last_error(ctx));
        return 1;
    }

    double coef[P * Q], x_mean[P], y_mean[Q], intercept[Q], train_pred[N * Q];
    int has_intercept = -1;
    const double params[] = {2, 1, 1};
    status = n4m_wasm_model_fit("O2PLS", params, 3, x, y, N, P, Q, 2,
                                coef, x_mean, y_mean, intercept, &has_intercept,
                                train_pred);
    if (status != N4M_OK || has_intercept != 0) {
        fprintf(stderr, "shim O2PLS fit failed: %d / intercept=%d\n", status, has_intercept);
        return 1;
    }
    const char* keys[] = {"coefficients", "x_mean", "y_mean", "predictions"};
    const double* actual[] = {coef, x_mean, y_mean, train_pred};
    const int sizes[] = {P * Q, P, Q, N * Q};
    double max_diff = 0.0;
    for (int field = 0; field < 4; ++field) {
        const double* expected = NULL;
        int64_t rows = 0, cols = 0;
        status = n4m_method_result_get_double_matrix(result, keys[field],
                                                      &expected, &rows, &cols);
        if (status != N4M_OK || expected == NULL || rows * cols != sizes[field] ||
            !check(actual[field], expected, sizes[field], keys[field], &max_diff)) return 1;
    }
    if (fabs(x_mean[0]) > 1e-12 || fabs(y_mean[0]) > 1e-12) {
        fprintf(stderr, "O2PLS selected legacy centred branch\n");
        return 1;
    }
    double shim_pred[HELDOUT * Q], direct_pred[HELDOUT * Q];
    status = n4m_wasm_model_predict_from_coeffs(coef, x_mean, y_mean, NULL,
                                                new_x, HELDOUT, P, Q, shim_pred);
    if (status != N4M_OK) return 1;
    for (int row = 0; row < HELDOUT; ++row) {
        for (int target = 0; target < Q; ++target) {
            double value = 0.0;
            const double* direct_coef = NULL;
            int64_t rows = 0, cols = 0;
            status = n4m_method_result_get_double_matrix(result, "coefficients",
                                                          &direct_coef, &rows, &cols);
            if (status != N4M_OK) return 1;
            for (int col = 0; col < P; ++col)
                value += new_x[row * P + col] * direct_coef[col * Q + target];
            direct_pred[row * Q + target] = value;
        }
    }
    if (!check(shim_pred, direct_pred, HELDOUT * Q, "held-out", &max_diff)) return 1;
    status = n4m_wasm_model_fit("O2PLS", NULL, 0, x, y, N, P, Q, 2,
                                coef, x_mean, y_mean, intercept, &has_intercept, NULL);
    if (status != N4M_OK) {
        fprintf(stderr, "canonical O2PLS defaults failed: %d\n", status);
        return 1;
    }
    const double* direct_default_coef = NULL;
    int64_t default_rows = 0, default_cols = 0;
    status = n4m_method_result_get_double_matrix(result, "coefficients",
                                                  &direct_default_coef,
                                                  &default_rows, &default_cols);
    if (status != N4M_OK || default_rows != P || default_cols != Q ||
        !check(coef, direct_default_coef, P * Q, "default coefficients", &max_diff)) return 1;
    const double invalid[][3] = {{0, 1, 1}, {2, -1, 1}, {2, 1.5, 1}};
    for (int index = 0; index < 3; ++index) {
        status = n4m_wasm_model_fit("O2PLS", invalid[index], 3, x, y, N, P, Q, 2,
                                    coef, x_mean, y_mean, intercept, &has_intercept, NULL);
        if (status != N4M_ERR_INVALID_ARGUMENT) {
            fprintf(stderr, "invalid O2PLS components accepted (%d): %d\n", index, status);
            return 1;
        }
    }
    printf("O2PLS %dx%d -> %d targets, %d held-out rows: max_abs_diff=%.17g\n",
           N, P, Q, HELDOUT, max_diff);
    n4m_method_result_destroy(result);
    n4m_config_destroy(cfg);
    n4m_context_destroy(ctx);
    return 0;
}
