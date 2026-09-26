/* SPDX-License-Identifier: CECILL-2.1
 * Native C-ABI oracle for the four portable Methods model tokens. Compare the
 * raw WASM entry shim with direct C fits under the R/Python default Config.
 */
#include "n4m/n4m.h"
#include "n4m/estimators/regression.h"
#include "n4m/ensemble.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

static double max_abs_diff = 0.0;

int n4m_wasm_model_fit(const char*, const double*, int, const double*,
                       const double*, int, int, int, int, double*, double*,
                       double*, double*, int*, double*);

static int same(const double* got, const double* want, int count, const char* label) {
    for (int i = 0; i < count; ++i) {
        const double diff = fabs(got[i] - want[i]);
        if (diff > max_abs_diff) max_abs_diff = diff;
        if (!isfinite(got[i]) || diff > 1e-10) {
            fprintf(stderr, "%s[%d]: got %.17g want %.17g\n", label, i, got[i], want[i]);
            return 0;
        }
    }
    return 1;
}

int main(void) {
    enum { N = 21, P = 12 };
    double x[N * P], y[N];
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < P; ++col) {
            x[row * P + col] = sin((double)((row + 1) * (col + 1)) / 9.0)
                + cos((double)(row + 1) + (double)(col + 1) / 7.0)
                + (double)((row + 1) * (col + 1)) / 100.0;
        }
        y[row] = 1.3 + 0.7 * x[row * P + 1] - 0.4 * x[row * P + 5];
    }
    n4m_matrix_view_t xv, yv;
    n4m_matrix_view_init_rowmajor(&xv, x, N, P, N4M_DTYPE_F64);
    n4m_matrix_view_init_rowmajor(&yv, y, N, 1, N4M_DTYPE_F64);
    const char* names[] = {"FusedSparsePLS", "BaggingPLS", "BoostingPLS", "RandomSubspacePLS"};
    for (int k = 0; k < 4; ++k) {
        max_abs_diff = 0.0;
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
        if (status != N4M_OK) return 1;
        switch (k) {
            case 0: status = n4m_estimators_fused_sparse_pls_fit(ctx, cfg, &xv, &yv, 0.05, 0.05, &result); break;
            case 1: status = n4m_ensemble_bagging_pls_fit(ctx, cfg, &xv, &yv, 50, 0, &result); break;
            case 2: status = n4m_ensemble_boosting_pls_fit(ctx, cfg, &xv, &yv, 50, 0.1, &result); break;
            case 3: status = n4m_ensemble_random_subspace_pls_fit(ctx, cfg, &xv, &yv, 50, 10, 0, &result); break;
        }
        if (status != N4M_OK || result == NULL) {
            fprintf(stderr, "%s direct fit failed: %d\n", names[k], status);
            return 1;
        }
        if (k == 2) {
            n4m_method_result_t* rejected = NULL;
            status = n4m_ensemble_boosting_pls_fit(ctx, cfg, &xv, &yv,
                                                   50, 1.2, &rejected);
            if (status != N4M_ERR_INVALID_ARGUMENT || rejected != NULL) {
                fprintf(stderr, "BoostingPLS accepted learning_rate=1.2: %d\n", status);
                return 1;
            }
        }
        double coefficients[P], x_mean[P], y_mean[1], intercept[1], predictions[N];
        int has_intercept = -1;
        status = n4m_wasm_model_fit(names[k], NULL, 0, x, y, N, P, 1, 2,
                                    coefficients, x_mean, y_mean, intercept,
                                    &has_intercept, predictions);
        if (status != N4M_OK || has_intercept != 0) {
            fprintf(stderr, "%s shim failed: %d, intercept=%d\n", names[k], status, has_intercept);
            return 1;
        }
        const char* keys[] = {"coefficients", "x_mean", "y_mean", "predictions"};
        const double* got[] = {coefficients, x_mean, y_mean, predictions};
        const int sizes[] = {P, P, 1, N};
        for (int field = 0; field < 4; ++field) {
            const double* expected = NULL;
            int64_t rows = 0, cols = 0;
            status = n4m_method_result_get_double_matrix(result, keys[field],
                                                          &expected, &rows, &cols);
            if (status != N4M_OK || expected == NULL || rows * cols != sizes[field] ||
                !same(got[field], expected, sizes[field], keys[field])) return 1;
        }
        const double *direct_coef = NULL, *direct_x_mean = NULL, *direct_y_mean = NULL;
        int64_t nr = 0, nc = 0;
        if (n4m_method_result_get_double_matrix(result, "coefficients",
                                                 &direct_coef, &nr, &nc) != N4M_OK ||
            n4m_method_result_get_double_matrix(result, "x_mean",
                                                 &direct_x_mean, &nr, &nc) != N4M_OK ||
            n4m_method_result_get_double_matrix(result, "y_mean",
                                                 &direct_y_mean, &nr, &nc) != N4M_OK) return 1;
        const int heldout_rows[] = {1, 7, 16};
        for (int t = 0; t < 3; ++t) {
            double direct_pred = direct_y_mean[0], shim_pred = y_mean[0];
            for (int col = 0; col < P; ++col) {
                const double shifted = x[heldout_rows[t] * P + col] + 0.031;
                direct_pred += (shifted - direct_x_mean[col]) * direct_coef[col];
                shim_pred += (shifted - x_mean[col]) * coefficients[col];
            }
            if (!same(&shim_pred, &direct_pred, 1, "held-out prediction")) return 1;
        }
        n4m_method_result_destroy(result);
        n4m_config_destroy(cfg);
        n4m_context_destroy(ctx);
        printf("%s: native C-ABI oracle OK (max_abs_diff=%.17g)\n", names[k], max_abs_diff);
    }
    return 0;
}
