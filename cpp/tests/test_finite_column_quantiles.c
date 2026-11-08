/* SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later */
/* Focused public-C procedure/result contract; no fit or private headers. */
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "n4m/n4m.h"
#include "n4m/estimator.h"

#define CHECK(expr) do { \
    if (!(expr)) { fprintf(stderr, "line %d: %s\n", __LINE__, #expr); return 1; } \
} while (0)

int main(void) {
    n4m_context_t* ctx = NULL;
    n4m_params_t* params = NULL;
    n4m_method_result_t* result = NULL;
    int32_t index = -1, entries = 0;
    const double probabilities[] = {0.1, 0.5, 0.9, 0.0, 1.0, 0.5};
    double X[] = {3, 7, NAN, INFINITY, 1, INFINITY, NAN, 2,
                  4, -INFINITY, INFINITY, NAN, 2, NAN, -INFINITY, 6};
    const double expected[] = {1.3, 7, NAN, 2.4, 2.5, 7, NAN, 4,
                               3.7, 7, NAN, 5.6, 1, 7, NAN, 2,
                               4, 7, NAN, 6, 2.5, 7, NAN, 4};
    n4m_matrix_view_t view;
    n4m_fit_inputs_v1_t inputs = {0};
    const double* quantiles = NULL;
    const int64_t* counts = NULL;
    int64_t rows = 0, cols = 0, count_size = 0;

    CHECK(n4m_context_create(&ctx) == N4M_OK);
    CHECK(n4m_method_find("utilities.finite_column_quantiles", &index) == N4M_OK);
    CHECK(n4m_params_create(ctx, index, &params) == N4M_OK);
    CHECK(n4m_matrix_view_init_rowmajor(&view, X, 4, 4, N4M_DTYPE_F64) == N4M_OK);
    inputs.struct_size = sizeof(inputs);
    inputs.X = &view;
    CHECK(n4m_procedure_run(ctx, index, params, &inputs, &result) == N4M_ERR_INVALID_ARGUMENT);
    CHECK(result == NULL);
    CHECK(n4m_params_set_double_array(params, "probabilities", probabilities, 6) == N4M_OK);

    /* A valid empty matrix view must be refused before output-size division. */
    view.cols = 0;
    CHECK(n4m_procedure_run(ctx, index, params, &inputs, &result) == N4M_ERR_SHAPE_MISMATCH);
    CHECK(result == NULL);
    view.cols = 4;

    /* Row-major, column-major and interleaved views must agree. */
    for (int layout = 0; layout < 3; ++layout) {
        double storage[32] = {0};
        view.data = layout == 0 ? X : storage;
        view.row_stride = layout == 1 ? 1 : (layout == 2 ? 8 : 4);
        view.col_stride = layout == 1 ? 4 : (layout == 2 ? 2 : 1);
        if (layout != 0) {
            for (int r = 0; r < 4; ++r)
                for (int c = 0; c < 4; ++c)
                    storage[r * view.row_stride + c * view.col_stride] = X[r * 4 + c];
        }
        CHECK(n4m_procedure_run(ctx, index, params, &inputs, &result) == N4M_OK);
        CHECK(n4m_method_result_entry_count(result, &entries) == N4M_OK && entries == 2);
        CHECK(n4m_method_result_get_double_matrix(result, "quantiles", &quantiles, &rows, &cols) == N4M_OK);
        CHECK(rows == 6 && cols == 4);
        CHECK(n4m_method_result_get_int64_vector(result, "finite_counts", &counts, &count_size) == N4M_OK);
        CHECK(count_size == 4 && counts[0] == 4 && counts[1] == 1 && counts[2] == 0 && counts[3] == 2);
        for (int i = 0; i < 24; ++i) {
            if (isnan(expected[i])) CHECK(isnan(quantiles[i]));
            else CHECK(fabs(quantiles[i] - expected[i]) < 1e-12);
        }
        n4m_method_result_destroy(result);
        result = NULL;
    }

    double singleton[] = {9, NAN};
    CHECK(n4m_matrix_view_init_rowmajor(&view, singleton, 1, 2, N4M_DTYPE_F64) == N4M_OK);
    CHECK(n4m_procedure_run(ctx, index, params, &inputs, &result) == N4M_OK);
    CHECK(n4m_method_result_get_double_matrix(result, "quantiles", &quantiles, &rows, &cols) == N4M_OK);
    CHECK(rows == 6 && cols == 2);
    CHECK(n4m_method_result_get_int64_vector(result, "finite_counts", &counts, &count_size) == N4M_OK);
    CHECK(count_size == 2 && counts[0] == 1 && counts[1] == 0);
    for (int i = 0; i < 6; ++i) CHECK(quantiles[2 * i] == 9 && isnan(quantiles[2 * i + 1]));
    n4m_method_result_destroy(result);
    result = NULL;

    /* Endpoints must not subtract opposite extreme finite values. */
    double extremes[] = {-DBL_MAX, DBL_MAX};
    const double endpoints[] = {1, 0};
    CHECK(n4m_matrix_view_init_rowmajor(&view, extremes, 2, 1, N4M_DTYPE_F64) == N4M_OK);
    CHECK(n4m_params_set_double_array(params, "probabilities", endpoints, 2) == N4M_OK);
    CHECK(n4m_procedure_run(ctx, index, params, &inputs, &result) == N4M_OK);
    CHECK(n4m_method_result_get_double_matrix(result, "quantiles", &quantiles, &rows, &cols) == N4M_OK);
    CHECK(rows == 2 && cols == 1 && quantiles[0] == DBL_MAX && quantiles[1] == -DBL_MAX);
    n4m_method_result_destroy(result);
    result = NULL;

    const double invalid[] = {-0.1, 1.1, NAN, INFINITY, -INFINITY};
    for (int i = 0; i < 5; ++i) {
        CHECK(n4m_params_set_double_array(params, "probabilities", &invalid[i], 1) == N4M_ERR_INVALID_ARGUMENT);
    }
    /* Empty arrays are marshalable; this procedure refuses them at run. */
    CHECK(n4m_params_set_double_array(params, "probabilities", probabilities, 0) == N4M_OK);
    CHECK(n4m_procedure_run(ctx, index, params, &inputs, &result) == N4M_ERR_INVALID_ARGUMENT);
    CHECK(result == NULL);
    n4m_params_destroy(params);
    n4m_context_destroy(ctx);
    puts("finite-column quantile C contract PASS");
    return 0;
}
