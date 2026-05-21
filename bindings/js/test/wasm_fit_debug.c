/* SPDX-License-Identifier: CECILL-2.1
 *
 * Pure-C reference test for the Emscripten libp4a_static.a build.
 *
 * This file exists to demonstrate that the WASM library itself fits
 * models correctly when the calls are made from a C translation unit
 * compiled into the same WASM module. The Node smoke test at
 * `run_smoke.mjs` currently exercises only Context/Config lifecycle
 * because the same fit invoked from a JS host returns garbage — see
 * the README "Status" section.
 *
 * Build (manual, not part of the CMake project — kept for diagnostics):
 *
 *   source $EMSDK/emsdk_env.sh
 *   emcc bindings/js/test/wasm_fit_debug.c \
 *     -I cpp/include \
 *     -I build/emscripten/generated \
 *     build/emscripten/cpp/src/libp4a_static.a \
 *     -O2 -s WASM_BIGINT=1 -s ALLOW_MEMORY_GROWTH=1 \
 *     -s INITIAL_MEMORY=64MB -lstdc++ -lm \
 *     -o /tmp/wasm_fit_debug.js
 *   node /tmp/wasm_fit_debug.js
 *
 * Expected output (passes when the WASM library is correct):
 *   x_mean: 0.12 0.017 0.018 0.025 -0.003
 *   coefs:  1.0 0.5 -0.3 ~0 ~0
 */

#include "n4m/n4m.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    int n = 50, p = 5, q = 1;
    double *X = malloc((size_t)n * p * sizeof(double));
    double *Y = malloc((size_t)n * q * sizeof(double));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < p; ++j) {
            X[i * p + j] = sin((double)(i + 1) * (j + 1) * 0.3);
        }
        Y[i] = X[i * p] + 0.5 * X[i * p + 1] - 0.3 * X[i * p + 2];
    }

    n4m_context_t *ctx;
    n4m_context_create(&ctx);
    n4m_config_t *cfg;
    n4m_config_create(&cfg);
    n4m_config_set_algorithm(cfg, N4M_ALGO_PLS_REGRESSION);
    n4m_config_set_solver(cfg, N4M_SOLVER_SIMPLS);
    n4m_config_set_deflation(cfg, N4M_DEFLATION_REGRESSION);
    n4m_config_set_n_components(cfg, 3);
    n4m_config_set_center_x(cfg, 1);
    n4m_config_set_center_y(cfg, 1);

    n4m_matrix_view_t xv, yv;
    n4m_matrix_view_init_rowmajor(&xv, X, n, p, N4M_DTYPE_F64);
    n4m_matrix_view_init_rowmajor(&yv, Y, n, q, N4M_DTYPE_F64);

    n4m_model_t *m;
    n4m_status_t s = n4m_model_fit(ctx, cfg, &xv, &yv, &m);
    if (s != 0) {
        fprintf(stderr, "fit failed: %s\n", n4m_context_last_error(ctx));
        return 1;
    }

    n4m_array_t *arr;
    n4m_matrix_view_t v;
    if (n4m_model_get_array(ctx, m, N4M_MODEL_X_MEAN, &arr) == 0) {
        n4m_array_view(arr, &v);
        const double *d = (const double *)v.data;
        printf("x_mean: %g %g %g %g %g\n", d[0], d[1], d[2], d[3], d[4]);
        n4m_array_free(arr);
    }
    if (n4m_model_get_array(ctx, m, N4M_MODEL_COEFFICIENTS, &arr) == 0) {
        n4m_array_view(arr, &v);
        const double *d = (const double *)v.data;
        printf("coefs: %g %g %g %g %g\n", d[0], d[1], d[2], d[3], d[4]);
        n4m_array_free(arr);
    }

    n4m_model_destroy(m);
    n4m_config_destroy(cfg);
    n4m_context_destroy(ctx);
    free(X);
    free(Y);
    return 0;
}
