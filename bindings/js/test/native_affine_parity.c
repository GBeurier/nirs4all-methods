/* SPDX-License-Identifier: CECILL-2.1
 * Native check for the raw-pointer shim used by fitModel. The held-out
 * predictions are frozen in nirs4all's portable-affine-methods.R recipe test.
 * Build after cmake --preset dev-release && cmake --build --preset dev-release:
 * gcc -std=c11 -Wall -Wextra -Werror -ffunction-sections -fdata-sections \
 *   -Icpp/include -Ibuild/dev-release/generated \
 *   bindings/js/src/wasm_entry.c bindings/js/test/native_affine_parity.c \
 *   -Wl,--gc-sections -Lbuild/dev-release/cpp/src \
 *   -Wl,-rpath,build/dev-release/cpp/src -ln4m -lm -o /tmp/n4m-affine-parity
 * /tmp/n4m-affine-parity
 * (The unused MSC shim requires --gc-sections with a shared native libn4m.)
 */
#include <math.h>
#include <stdio.h>

int n4m_wasm_model_fit(const char* model, const double* params, int n_params,
                       const double* x, const double* y, int n, int p, int q,
                       int n_components, double* coefficients_out,
                       double* x_mean_out, double* y_mean_out,
                       double* intercept_out, int* has_intercept_out,
                       double* predictions_out);

typedef struct {
    const char* model;
    double held_out[3];
    int has_intercept;
} affine_case;

int main(void) {
    const affine_case cases[] = {
        {"Ridge", {1.119585507454503, 2.2247502591042467, 0.8880639379404299}, 1},
        {"RidgePLS", {1.0183856539647425, 2.164229001896515, 0.9228238984475714}, 0},
        {"RobustPLS", {1.2000108169670543, 1.5460522175157725, 0.7548630520848121}, 0},
        {"CPPLS", {0.9908421322688001, 2.2185030364862115, 0.9364520576202219}, 0},
        {"SparseSIMPLS", {0.9908421322687999, 2.2185030364862115, 0.9364520576202217}, 0},
        {"ECR", {1.1953068770625324, 2.0055006340124004, 0.4704252881332379}, 0},
        {"ContinuumRegression", {1.0601251478760563, 2.285195247718287, 0.919738050799327}, 0},
        {"MIRPLS", {1.184521178890891, 2.2497913086895096, 0.26128963080778034}, 0},
    };
    double x[21 * 8], y[21];
    for (int i = 0; i < 21; ++i) {
        for (int j = 0; j < 8; ++j) {
            const int ri = i + 1, cj = j + 1;
            x[i * 8 + j] = sin((double)(ri * cj) / 9.0) +
                           cos((double)ri + (double)cj / 7.0) +
                           (double)(ri * cj) / 100.0;
        }
        y[i] = 1.3 + 0.7 * x[i * 8 + 1] - 0.4 * x[i * 8 + 5];
    }
    const int test_rows[] = {1, 7, 16};
    for (unsigned k = 0; k < sizeof(cases) / sizeof(cases[0]); ++k) {
        double coef[8] = {0}, x_mean[8] = {0}, y_mean[1] = {0};
        double intercept[1] = {0};
        int has_intercept = -1;
        int status = n4m_wasm_model_fit(cases[k].model, NULL, 0, x, y, 21, 8, 1, 2,
                                        coef, x_mean, y_mean, intercept, &has_intercept,
                                        NULL);
        if (status != 0 || has_intercept != cases[k].has_intercept) {
            fprintf(stderr, "%s: status=%d intercept_flag=%d\n",
                    cases[k].model, status, has_intercept);
            return 1;
        }
        for (int t = 0; t < 3; ++t) {
            double pred = has_intercept ? intercept[0] : y_mean[0];
            for (int j = 0; j < 8; ++j) {
                const double value = x[test_rows[t] * 8 + j] + 0.031;
                pred += (has_intercept ? value : value - x_mean[j]) * coef[j];
            }
            if (!isfinite(pred) || fabs(pred - cases[k].held_out[t]) >= 1e-10) {
                fprintf(stderr, "%s[%d]: got %.17g expected %.17g\n",
                        cases[k].model, t, pred, cases[k].held_out[t]);
                return 1;
            }
        }
        printf("%s: held-out parity OK\n", cases[k].model);
    }
    return 0;
}
