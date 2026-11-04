/* SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later */
#include <math.h>
#include <stdio.h>
#include "n4m/fanova.h"
#include "n4m/mixed_effects.h"

#define REQUIRE(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "installed consumer check failed at line %d: %s\n", \
                __LINE__, #condition); \
        return 1; \
    } \
} while (0)

static int mixed_effects_fit(void) {
    double x[20], y[20];
    int64_t groups[20];
    const double residuals[4] = {-0.3, -0.1, 0.1, 0.3};
    for (int g = 0; g < 5; ++g) {
        for (int j = 0; j < 4; ++j) {
            const int i = g * 4 + j;
            x[i] = 1;
            y[i] = 2 + (g - 2) + residuals[j];
            groups[i] = g;
        }
    }
    n4m_matrix_view_t X, Y, G;
    REQUIRE(n4m_matrix_view_init_rowmajor(&X, x, 20, 1, N4M_DTYPE_F64) == N4M_OK);
    REQUIRE(n4m_matrix_view_init_rowmajor(&Y, y, 20, 1, N4M_DTYPE_F64) == N4M_OK);
    REQUIRE(n4m_matrix_view_init_rowmajor(&G, groups, 20, 1, N4M_DTYPE_I64) == N4M_OK);
    n4m_mixed_effects_options_v1_t options;
    REQUIRE(n4m_mixed_effects_options_init_v1(&options) == N4M_OK);
    n4m_mixed_effects_summary_v1_t summary = {0};
    summary.struct_size = sizeof(summary);
    double beta[1], covariance[1], standard_error[1], variances[2];
    REQUIRE(n4m_mixed_effects_fit_v1(&X, &Y, &G, &options,
        beta, 1, covariance, 1, standard_error, 1, variances, 2, &summary) == N4M_OK);
    REQUIRE(summary.fit_state == N4M_MIXED_EFFECTS_FITTED);
    REQUIRE(fabs(beta[0] - 2) < 1e-8);
    REQUIRE(fabs(variances[0] - 1.0 / 15) < 1e-5);
    REQUIRE(fabs(variances[1] - 149.0 / 60) < 1e-4);
    REQUIRE(fabs(covariance[0] - 0.5) < 1e-4);
    REQUIRE(fabs(standard_error[0] - sqrt(0.5)) < 1e-4);
    REQUIRE(summary.converged == 1 && summary.evaluations > 0);
    return 0;
}

static int fanova_fit(void) {
    double x[512], y[256], bounds[4] = {0, 1, 0, 1};
    const uint32_t types[2] = {2, 2};
    for (int i = 0; i < 256; ++i) {
        const int a = (i / 2) % 2, b = i % 2;
        x[2 * i] = a; x[2 * i + 1] = b;
        y[i] = a == b ? -1 : 1;
    }
    n4m_matrix_view_t X, Y, B;
    REQUIRE(n4m_matrix_view_init_rowmajor(&X, x, 256, 2, N4M_DTYPE_F64) == N4M_OK);
    REQUIRE(n4m_matrix_view_init_rowmajor(&Y, y, 256, 1, N4M_DTYPE_F64) == N4M_OK);
    REQUIRE(n4m_matrix_view_init_rowmajor(&B, bounds, 2, 2, N4M_DTYPE_F64) == N4M_OK);
    n4m_fanova_options_v1_t options;
    REQUIRE(n4m_fanova_options_init_v1(&options) == N4M_OK);
    options.num_trees = 8;
    options.max_depth = 4;
    options.min_samples_leaf = 1;
    n4m_fanova_summary_v1_t summary = {0};
    summary.struct_size = sizeof(summary);
    double mains[4], pairs[8];
    REQUIRE(n4m_fanova_fit_v1(&X, &Y, &B, types, NULL, NULL, &options,
        mains, 4, pairs, 8, &summary) == N4M_OK);
    REQUIRE(summary.fit_state == N4M_FANOVA_FITTED);
    REQUIRE(summary.trees == 8 && summary.informative_trees == 8);
    REQUIRE(fabs(mains[0]) < 1e-12 && fabs(mains[2]) < 1e-12);
    REQUIRE(fabs(pairs[2] - 1) < 1e-12 && pairs[2] == pairs[4]);
    REQUIRE(isnan(pairs[0]) && summary.train_rmse < 1e-12);
    for (int i = 0; i < 256; ++i) y[i] = 1;
    REQUIRE(n4m_fanova_fit_v1(&X, &Y, &B, types, NULL, NULL, &options,
        mains, 4, pairs, 8, &summary) == N4M_OK);
    REQUIRE(summary.fit_state == N4M_FANOVA_CONSTANT_RESPONSE);
    REQUIRE(isnan(mains[0]) && isnan(pairs[2]));
    return 0;
}

int main(void) {
    REQUIRE(n4m_fanova_backend_available_v1() == 1);
    REQUIRE(n4m_mixed_effects_backend_available_v1() == 1);
    REQUIRE(mixed_effects_fit() == 0);
    REQUIRE(fanova_fit() == 0);
    puts("Installed static C consumer: analytic mixed effects and fANOVA passed");
    return 0;
}
