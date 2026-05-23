// SPDX-License-Identifier: CECILL-2.1

#include "n4m/n4m.h"

#include <cmath>
#include <cstdint>
#include <vector>

#include "fixtures/pipeline_fixtures.hpp"
#include "harness.hpp"

namespace {

constexpr double kAbsTol = 1e-12;

std::vector<double> copy_values(const n4m_array_t* arr) {
    std::int64_t rows = 0;
    std::int64_t cols = 0;
    n4m_array_shape(arr, &rows, &cols);
    n4m_matrix_view_t view{};
    n4m_array_view(arr, &view);
    const auto* ptr = static_cast<const double*>(view.data);
    return std::vector<double>(ptr, ptr + rows * cols);
}

void check_close_vector(int& failures,
                        const std::vector<double>& actual,
                        const std::vector<double>& expected) {
    CHECK_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const double diff = std::fabs(actual[i] - expected[i]);
        if (diff > kAbsTol) {
            CHECK_EQ(actual[i], expected[i]);
            return;
        }
    }
}

}  // namespace

TEST(pipeline_phase3a, transform_before_fit_returns_not_fitted) {
    n4m_context_t* ctx = nullptr;
    n4m_pipeline_t* pipe = nullptr;
    CHECK_EQ(n4m_context_create(&ctx), N4M_OK);
    CHECK_EQ(n4m_pipeline_create(&pipe), N4M_OK);

    double x[] = {1.0, 2.0, 3.0, 4.0};
    n4m_matrix_view_t X{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X, x, 2, 2, N4M_DTYPE_F64), N4M_OK);
    n4m_array_t* out = nullptr;
    CHECK_EQ(n4m_pipeline_transform_alloc(ctx, pipe, &X, &out), N4M_ERR_NOT_FITTED);
    CHECK_EQ(out, nullptr);
    CHECK_STR_CONTAINS(n4m_context_last_error(ctx), "not fitted");

    n4m_pipeline_destroy(pipe);
    n4m_context_destroy(ctx);
}

TEST(pipeline_phase3a, center_transform_alloc_matches_numpy_reference) {
    n4m_context_t* ctx = nullptr;
    n4m_pipeline_t* pipe = nullptr;
    CHECK_EQ(n4m_context_create(&ctx), N4M_OK);
    CHECK_EQ(n4m_pipeline_create(&pipe), N4M_OK);
    CHECK_EQ(n4m_pipeline_add_operator(pipe, N4M_OP_CENTER, nullptr, 0), N4M_OK);

    double x[] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    };
    n4m_matrix_view_t X{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X, x, 3, 3, N4M_DTYPE_F64), N4M_OK);
    CHECK_EQ(n4m_pipeline_fit(ctx, pipe, &X, nullptr), N4M_OK);

    n4m_array_t* out = nullptr;
    CHECK_EQ(n4m_pipeline_transform_alloc(ctx, pipe, &X, &out), N4M_OK);
    std::int64_t rows = 0;
    std::int64_t cols = 0;
    CHECK_EQ(n4m_array_shape(out, &rows, &cols), N4M_OK);
    CHECK_EQ(rows, 3);
    CHECK_EQ(cols, 3);
    check_close_vector(failures, copy_values(out), {
        -3.0, -3.0, -3.0,
         0.0,  0.0,  0.0,
         3.0,  3.0,  3.0,
    });

    n4m_array_free(out);
    n4m_pipeline_destroy(pipe);
    n4m_context_destroy(ctx);
}

TEST(pipeline_phase3a, generated_fixtures_match_python_reference) {
    for (const auto& fixture : ::n4m::test::fixtures::kPipelineFixtures) {
        n4m_context_t* ctx = nullptr;
        n4m_pipeline_t* pipe = nullptr;
        CHECK_EQ(n4m_context_create(&ctx), N4M_OK);
        CHECK_EQ(n4m_pipeline_create(&pipe), N4M_OK);
        std::size_t param_offset = 0;
        for (std::size_t op = 0; op < fixture.n_operators; ++op) {
            const std::int32_t n_params = fixture.n_params[op];
            const double* params = n_params > 0 ? fixture.params + param_offset : nullptr;
            CHECK_EQ(n4m_pipeline_add_operator(pipe, fixture.operators[op], params, n_params), N4M_OK);
            param_offset += static_cast<std::size_t>(n_params);
        }

        n4m_matrix_view_t X{};
        CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                               const_cast<double*>(fixture.X.values),
                                               fixture.X.rows,
                                               fixture.X.cols,
                                               N4M_DTYPE_F64),
                 N4M_OK);
        n4m_matrix_view_t Y{};
        CHECK_EQ(n4m_matrix_view_init_rowmajor(&Y,
                                               const_cast<double*>(fixture.Y.values),
                                               fixture.Y.rows,
                                               fixture.Y.cols,
                                               N4M_DTYPE_F64),
                 N4M_OK);
        CHECK_EQ(n4m_pipeline_fit(ctx, pipe, &X, &Y), N4M_OK);
        n4m_array_t* out = nullptr;
        CHECK_EQ(n4m_pipeline_transform_alloc(ctx, pipe, &X, &out), N4M_OK);
        check_close_vector(failures, copy_values(out),
                           std::vector<double>(
                               fixture.transform_train.values,
                               fixture.transform_train.values + fixture.transform_train.size));
        n4m_array_free(out);
        n4m_pipeline_destroy(pipe);
        n4m_context_destroy(ctx);
    }
}

TEST(pipeline_phase3a, autoscale_and_pareto_use_training_statistics) {
    n4m_context_t* ctx = nullptr;
    CHECK_EQ(n4m_context_create(&ctx), N4M_OK);

    double x[] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    };
    n4m_matrix_view_t X{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X, x, 3, 3, N4M_DTYPE_F64), N4M_OK);

    n4m_pipeline_t* autoscale = nullptr;
    CHECK_EQ(n4m_pipeline_create(&autoscale), N4M_OK);
    CHECK_EQ(n4m_pipeline_add_operator(autoscale, N4M_OP_AUTOSCALE, nullptr, 0), N4M_OK);
    CHECK_EQ(n4m_pipeline_fit(ctx, autoscale, &X, nullptr), N4M_OK);
    double autoscaled[9] = {};
    n4m_matrix_view_t A{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&A, autoscaled, 3, 3, N4M_DTYPE_F64), N4M_OK);
    CHECK_EQ(n4m_pipeline_transform(ctx, autoscale, &X, &A), N4M_OK);
    check_close_vector(failures, std::vector<double>(autoscaled, autoscaled + 9), {
        -1.0, -1.0, -1.0,
         0.0,  0.0,  0.0,
         1.0,  1.0,  1.0,
    });

    n4m_pipeline_t* pareto = nullptr;
    CHECK_EQ(n4m_pipeline_create(&pareto), N4M_OK);
    CHECK_EQ(n4m_pipeline_add_operator(pareto, N4M_OP_PARETO_SCALE, nullptr, 0), N4M_OK);
    CHECK_EQ(n4m_pipeline_fit(ctx, pareto, &X, nullptr), N4M_OK);
    n4m_array_t* out = nullptr;
    CHECK_EQ(n4m_pipeline_transform_alloc(ctx, pareto, &X, &out), N4M_OK);
    const double s = std::sqrt(3.0);
    check_close_vector(failures, copy_values(out), {
        -s, -s, -s,
         0.0, 0.0, 0.0,
         s, s, s,
    });

    n4m_array_free(out);
    n4m_pipeline_destroy(pareto);
    n4m_pipeline_destroy(autoscale);
    n4m_context_destroy(ctx);
}

TEST(pipeline_phase3a, snv_is_rowwise_and_handles_flat_rows) {
    n4m_context_t* ctx = nullptr;
    n4m_pipeline_t* pipe = nullptr;
    CHECK_EQ(n4m_context_create(&ctx), N4M_OK);
    CHECK_EQ(n4m_pipeline_create(&pipe), N4M_OK);
    CHECK_EQ(n4m_pipeline_add_operator(pipe, N4M_OP_SNV, nullptr, 0), N4M_OK);

    double x[] = {
        1.0, 2.0, 3.0,
        2.0, 2.0, 2.0,
        3.0, 5.0, 7.0,
    };
    n4m_matrix_view_t X{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X, x, 3, 3, N4M_DTYPE_F64), N4M_OK);
    CHECK_EQ(n4m_pipeline_fit(ctx, pipe, &X, nullptr), N4M_OK);
    n4m_array_t* out = nullptr;
    CHECK_EQ(n4m_pipeline_transform_alloc(ctx, pipe, &X, &out), N4M_OK);
    check_close_vector(failures, copy_values(out), {
        -1.0, 0.0, 1.0,
         0.0, 0.0, 0.0,
        -1.0, 0.0, 1.0,
    });

    n4m_array_free(out);
    n4m_pipeline_destroy(pipe);
    n4m_context_destroy(ctx);
}

TEST(pipeline_phase3c, detrend_poly_default_degree_is_linear) {
    n4m_context_t* ctx = nullptr;
    n4m_pipeline_t* pipe = nullptr;
    CHECK_EQ(n4m_context_create(&ctx), N4M_OK);
    CHECK_EQ(n4m_pipeline_create(&pipe), N4M_OK);
    CHECK_EQ(n4m_pipeline_add_operator(pipe, N4M_OP_DETREND_POLY, nullptr, 0), N4M_OK);

    double x[] = {
        1.0, 2.0, 4.0,
        3.0, 2.0, 1.0,
    };
    n4m_matrix_view_t X{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X, x, 2, 3, N4M_DTYPE_F64), N4M_OK);
    CHECK_EQ(n4m_pipeline_fit(ctx, pipe, &X, nullptr), N4M_OK);
    n4m_array_t* out = nullptr;
    CHECK_EQ(n4m_pipeline_transform_alloc(ctx, pipe, &X, &out), N4M_OK);
    check_close_vector(failures, copy_values(out), {
         1.0 / 6.0, -1.0 / 3.0,  1.0 / 6.0,
         0.0,        0.0,         0.0,
    });

    n4m_array_free(out);
    n4m_pipeline_destroy(pipe);
    n4m_context_destroy(ctx);
}

TEST(pipeline_phase3c, detrend_poly_rejects_invalid_degrees) {
    n4m_context_t* ctx = nullptr;
    CHECK_EQ(n4m_context_create(&ctx), N4M_OK);

    double x[] = {1.0, 2.0, 3.0, 4.0};
    n4m_matrix_view_t X{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X, x, 2, 2, N4M_DTYPE_F64), N4M_OK);

    n4m_pipeline_t* non_integer = nullptr;
    const double non_integer_degree[] = {1.5};
    CHECK_EQ(n4m_pipeline_create(&non_integer), N4M_OK);
    CHECK_EQ(n4m_pipeline_add_operator(non_integer, N4M_OP_DETREND_POLY,
                                       non_integer_degree, 1),
             N4M_OK);
    CHECK_EQ(n4m_pipeline_fit(ctx, non_integer, &X, nullptr), N4M_ERR_INVALID_ARGUMENT);
    CHECK_STR_CONTAINS(n4m_context_last_error(ctx), "integer");
    n4m_pipeline_destroy(non_integer);

    n4m_pipeline_t* too_many = nullptr;
    const double too_many_params[] = {1.0, 2.0};
    CHECK_EQ(n4m_pipeline_create(&too_many), N4M_OK);
    CHECK_EQ(n4m_pipeline_add_operator(too_many, N4M_OP_DETREND_POLY,
                                       too_many_params, 2),
             N4M_OK);
    CHECK_EQ(n4m_pipeline_fit(ctx, too_many, &X, nullptr), N4M_ERR_INVALID_ARGUMENT);
    CHECK_STR_CONTAINS(n4m_context_last_error(ctx), "zero params or one");
    n4m_pipeline_destroy(too_many);

    n4m_pipeline_t* degree_equals_cols = nullptr;
    const double degree[] = {2.0};
    CHECK_EQ(n4m_pipeline_create(&degree_equals_cols), N4M_OK);
    CHECK_EQ(n4m_pipeline_add_operator(degree_equals_cols, N4M_OP_DETREND_POLY,
                                       degree, 1),
             N4M_OK);
    CHECK_EQ(n4m_pipeline_fit(ctx, degree_equals_cols, &X, nullptr),
             N4M_ERR_INVALID_ARGUMENT);
    CHECK_STR_CONTAINS(n4m_context_last_error(ctx), "smaller than the column count");
    n4m_pipeline_destroy(degree_equals_cols);

    n4m_context_destroy(ctx);
}

TEST(pipeline_phase3c, no_param_operators_reject_params_at_fit) {
    n4m_context_t* ctx = nullptr;
    n4m_pipeline_t* pipe = nullptr;
    CHECK_EQ(n4m_context_create(&ctx), N4M_OK);
    CHECK_EQ(n4m_pipeline_create(&pipe), N4M_OK);
    const double param[] = {1.0};
    CHECK_EQ(n4m_pipeline_add_operator(pipe, N4M_OP_CENTER, param, 1), N4M_OK);

    double x[] = {1.0, 2.0, 3.0, 4.0};
    n4m_matrix_view_t X{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X, x, 2, 2, N4M_DTYPE_F64), N4M_OK);
    CHECK_EQ(n4m_pipeline_fit(ctx, pipe, &X, nullptr), N4M_ERR_INVALID_ARGUMENT);
    CHECK_STR_CONTAINS(n4m_context_last_error(ctx), "center");

    n4m_pipeline_destroy(pipe);
    n4m_context_destroy(ctx);
}

TEST(pipeline_phase3e, emsc_corrects_affine_scatter_to_fitted_reference) {
    n4m_context_t* ctx = nullptr;
    n4m_pipeline_t* pipe = nullptr;
    CHECK_EQ(n4m_context_create(&ctx), N4M_OK);
    CHECK_EQ(n4m_pipeline_create(&pipe), N4M_OK);
    const double degree[] = {0.0};
    CHECK_EQ(n4m_pipeline_add_operator(pipe, N4M_OP_EMSC, degree, 1), N4M_OK);

    double x[] = {
        1.2, 2.4, 4.2, 6.6,
        0.3, 1.5, 3.3, 5.7,
        2.1, 3.3, 5.1, 7.5,
    };
    n4m_matrix_view_t X{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X, x, 3, 4, N4M_DTYPE_F64), N4M_OK);
    CHECK_EQ(n4m_pipeline_fit(ctx, pipe, &X, nullptr), N4M_OK);
    n4m_array_t* out = nullptr;
    CHECK_EQ(n4m_pipeline_transform_alloc(ctx, pipe, &X, &out), N4M_OK);
    check_close_vector(failures, copy_values(out), {
        1.2, 2.4, 4.2, 6.6,
        1.2, 2.4, 4.2, 6.6,
        1.2, 2.4, 4.2, 6.6,
    });

    n4m_array_free(out);
    n4m_pipeline_destroy(pipe);
    n4m_context_destroy(ctx);
}

TEST(pipeline_phase3e, emsc_rejects_invalid_degree_and_undersized_design) {
    n4m_context_t* ctx = nullptr;
    CHECK_EQ(n4m_context_create(&ctx), N4M_OK);

    double x[] = {
        1.0, 2.0, 4.0,
        1.5, 2.5, 4.5,
    };
    n4m_matrix_view_t X{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X, x, 2, 3, N4M_DTYPE_F64), N4M_OK);

    n4m_pipeline_t* non_integer = nullptr;
    const double non_integer_degree[] = {1.5};
    CHECK_EQ(n4m_pipeline_create(&non_integer), N4M_OK);
    CHECK_EQ(n4m_pipeline_add_operator(non_integer, N4M_OP_EMSC,
                                       non_integer_degree, 1),
             N4M_OK);
    CHECK_EQ(n4m_pipeline_fit(ctx, non_integer, &X, nullptr), N4M_ERR_INVALID_ARGUMENT);
    CHECK_STR_CONTAINS(n4m_context_last_error(ctx), "integer");
    n4m_pipeline_destroy(non_integer);

    n4m_pipeline_t* too_wide = nullptr;
    const double degree[] = {2.0};
    CHECK_EQ(n4m_pipeline_create(&too_wide), N4M_OK);
    CHECK_EQ(n4m_pipeline_add_operator(too_wide, N4M_OP_EMSC, degree, 1), N4M_OK);
    CHECK_EQ(n4m_pipeline_fit(ctx, too_wide, &X, nullptr), N4M_ERR_INVALID_ARGUMENT);
    CHECK_STR_CONTAINS(n4m_context_last_error(ctx), "degree + 2 columns");
    n4m_pipeline_destroy(too_wide);

    n4m_context_destroy(ctx);
}

TEST(pipeline_phase3d, savgol_smooth_matches_quadratic_interior) {
    n4m_context_t* ctx = nullptr;
    n4m_pipeline_t* pipe = nullptr;
    CHECK_EQ(n4m_context_create(&ctx), N4M_OK);
    CHECK_EQ(n4m_pipeline_create(&pipe), N4M_OK);
    const double params[] = {5.0, 2.0};
    CHECK_EQ(n4m_pipeline_add_operator(pipe, N4M_OP_SAVGOL_SMOOTH, params, 2), N4M_OK);

    double x[] = {
        4.0, 1.0, 0.0, 1.0, 4.0,
        5.0, 2.0, 1.0, 2.0, 5.0,
    };
    n4m_matrix_view_t X{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X, x, 2, 5, N4M_DTYPE_F64), N4M_OK);
    CHECK_EQ(n4m_pipeline_fit(ctx, pipe, &X, nullptr), N4M_OK);
    n4m_array_t* out = nullptr;
    CHECK_EQ(n4m_pipeline_transform_alloc(ctx, pipe, &X, &out), N4M_OK);
    const std::vector<double> values = copy_values(out);
    CHECK_EQ(values.size(), 10U);
    CHECK_EQ(std::fabs(values[2] - 0.0) <= kAbsTol, true);
    CHECK_EQ(std::fabs(values[7] - 1.0) <= kAbsTol, true);

    n4m_array_free(out);
    n4m_pipeline_destroy(pipe);
    n4m_context_destroy(ctx);
}

TEST(pipeline_phase3d, savgol_derivative_default_is_first_derivative) {
    n4m_context_t* ctx = nullptr;
    n4m_pipeline_t* pipe = nullptr;
    CHECK_EQ(n4m_context_create(&ctx), N4M_OK);
    CHECK_EQ(n4m_pipeline_create(&pipe), N4M_OK);
    CHECK_EQ(n4m_pipeline_add_operator(pipe, N4M_OP_SAVGOL_DERIVATIVE, nullptr, 0), N4M_OK);

    double x[] = {
        0.0, 1.0, 2.0, 3.0, 4.0,
    };
    n4m_matrix_view_t X{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X, x, 1, 5, N4M_DTYPE_F64), N4M_OK);
    CHECK_EQ(n4m_pipeline_fit(ctx, pipe, &X, nullptr), N4M_OK);
    n4m_array_t* out = nullptr;
    CHECK_EQ(n4m_pipeline_transform_alloc(ctx, pipe, &X, &out), N4M_OK);
    check_close_vector(failures, copy_values(out), {
        0.5, 0.8, 1.0, 0.8, 0.5,
    });

    n4m_array_free(out);
    n4m_pipeline_destroy(pipe);
    n4m_context_destroy(ctx);
}

TEST(pipeline_phase3d, savgol_rejects_invalid_params_at_fit) {
    n4m_context_t* ctx = nullptr;
    CHECK_EQ(n4m_context_create(&ctx), N4M_OK);

    double x[] = {
        0.0, 1.0, 2.0, 3.0, 4.0,
        1.0, 2.0, 3.0, 4.0, 5.0,
    };
    n4m_matrix_view_t X{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X, x, 2, 5, N4M_DTYPE_F64), N4M_OK);

    n4m_pipeline_t* even_window = nullptr;
    const double even_params[] = {4.0, 2.0};
    CHECK_EQ(n4m_pipeline_create(&even_window), N4M_OK);
    CHECK_EQ(n4m_pipeline_add_operator(even_window, N4M_OP_SAVGOL_SMOOTH,
                                       even_params, 2),
             N4M_OK);
    CHECK_EQ(n4m_pipeline_fit(ctx, even_window, &X, nullptr), N4M_ERR_INVALID_ARGUMENT);
    CHECK_STR_CONTAINS(n4m_context_last_error(ctx), "odd integer");
    n4m_pipeline_destroy(even_window);

    n4m_pipeline_t* derivative_too_high = nullptr;
    const double derivative_params[] = {5.0, 1.0, 2.0};
    CHECK_EQ(n4m_pipeline_create(&derivative_too_high), N4M_OK);
    CHECK_EQ(n4m_pipeline_add_operator(derivative_too_high, N4M_OP_SAVGOL_DERIVATIVE,
                                       derivative_params, 3),
             N4M_OK);
    CHECK_EQ(n4m_pipeline_fit(ctx, derivative_too_high, &X, nullptr),
             N4M_ERR_INVALID_ARGUMENT);
    CHECK_STR_CONTAINS(n4m_context_last_error(ctx), "must not exceed");
    n4m_pipeline_destroy(derivative_too_high);

    n4m_context_destroy(ctx);
}

TEST(pipeline_phase3f, asls_removes_constant_baseline) {
    n4m_context_t* ctx = nullptr;
    n4m_pipeline_t* pipe = nullptr;
    CHECK_EQ(n4m_context_create(&ctx), N4M_OK);
    CHECK_EQ(n4m_pipeline_create(&pipe), N4M_OK);
    const double params[] = {1000.0, 0.01, 8.0};
    CHECK_EQ(n4m_pipeline_add_operator(pipe, N4M_OP_ASLS_BASELINE, params, 3), N4M_OK);

    double x[] = {
        3.0, 3.0, 3.0, 3.0, 3.0,
        -1.5, -1.5, -1.5, -1.5, -1.5,
    };
    n4m_matrix_view_t X{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X, x, 2, 5, N4M_DTYPE_F64), N4M_OK);
    CHECK_EQ(n4m_pipeline_fit(ctx, pipe, &X, nullptr), N4M_OK);
    n4m_array_t* out = nullptr;
    CHECK_EQ(n4m_pipeline_transform_alloc(ctx, pipe, &X, &out), N4M_OK);
    for (double value : copy_values(out)) {
        CHECK_EQ(std::fabs(value) <= 1e-8, true);
    }

    n4m_array_free(out);
    n4m_pipeline_destroy(pipe);
    n4m_context_destroy(ctx);
}

TEST(pipeline_phase3f, asls_rejects_invalid_params_at_fit) {
    n4m_context_t* ctx = nullptr;
    CHECK_EQ(n4m_context_create(&ctx), N4M_OK);

    double x[] = {
        1.0, 2.0, 3.0,
        2.0, 3.0, 4.0,
    };
    n4m_matrix_view_t X{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X, x, 2, 3, N4M_DTYPE_F64), N4M_OK);

    n4m_pipeline_t* bad_lambda = nullptr;
    const double lambda_params[] = {0.0, 0.01, 5.0};
    CHECK_EQ(n4m_pipeline_create(&bad_lambda), N4M_OK);
    CHECK_EQ(n4m_pipeline_add_operator(bad_lambda, N4M_OP_ASLS_BASELINE,
                                       lambda_params, 3),
             N4M_OK);
    CHECK_EQ(n4m_pipeline_fit(ctx, bad_lambda, &X, nullptr), N4M_ERR_INVALID_ARGUMENT);
    CHECK_STR_CONTAINS(n4m_context_last_error(ctx), "lambda");
    n4m_pipeline_destroy(bad_lambda);

    n4m_pipeline_t* bad_iterations = nullptr;
    const double iteration_params[] = {1000.0, 0.01, 1.5};
    CHECK_EQ(n4m_pipeline_create(&bad_iterations), N4M_OK);
    CHECK_EQ(n4m_pipeline_add_operator(bad_iterations, N4M_OP_ASLS_BASELINE,
                                       iteration_params, 3),
             N4M_OK);
    CHECK_EQ(n4m_pipeline_fit(ctx, bad_iterations, &X, nullptr),
             N4M_ERR_INVALID_ARGUMENT);
    CHECK_STR_CONTAINS(n4m_context_last_error(ctx), "iteration");
    n4m_pipeline_destroy(bad_iterations);

    n4m_context_destroy(ctx);
}

TEST(pipeline_phase3g, norris_williams_gap_segment_derivative) {
    n4m_context_t* ctx = nullptr;
    n4m_pipeline_t* pipe = nullptr;
    CHECK_EQ(n4m_context_create(&ctx), N4M_OK);
    CHECK_EQ(n4m_pipeline_create(&pipe), N4M_OK);
    const double params[] = {1.0, 1.0, 1.0};
    CHECK_EQ(n4m_pipeline_add_operator(pipe, N4M_OP_NORRIS_WILLIAMS,
                                       params, 3),
             N4M_OK);

    double x[] = {
        0.0, 1.0, 2.0, 3.0, 4.0,
    };
    n4m_matrix_view_t X{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X, x, 1, 5, N4M_DTYPE_F64), N4M_OK);
    CHECK_EQ(n4m_pipeline_fit(ctx, pipe, &X, nullptr), N4M_OK);
    n4m_array_t* out = nullptr;
    CHECK_EQ(n4m_pipeline_transform_alloc(ctx, pipe, &X, &out), N4M_OK);
    check_close_vector(failures, copy_values(out), {
        1.0, 2.0, 2.0, 2.0, 1.0,
    });

    n4m_array_free(out);
    n4m_pipeline_destroy(pipe);
    n4m_context_destroy(ctx);
}

TEST(pipeline_phase3g, norris_williams_rejects_invalid_params_at_fit) {
    n4m_context_t* ctx = nullptr;
    CHECK_EQ(n4m_context_create(&ctx), N4M_OK);

    double x[] = {
        0.0, 1.0, 2.0, 3.0, 4.0,
        1.0, 2.0, 3.0, 4.0, 5.0,
    };
    n4m_matrix_view_t X{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X, x, 2, 5, N4M_DTYPE_F64), N4M_OK);

    n4m_pipeline_t* even_segment = nullptr;
    const double even_params[] = {2.0, 1.0, 1.0};
    CHECK_EQ(n4m_pipeline_create(&even_segment), N4M_OK);
    CHECK_EQ(n4m_pipeline_add_operator(even_segment, N4M_OP_NORRIS_WILLIAMS,
                                       even_params, 3),
             N4M_OK);
    CHECK_EQ(n4m_pipeline_fit(ctx, even_segment, &X, nullptr), N4M_ERR_INVALID_ARGUMENT);
    CHECK_STR_CONTAINS(n4m_context_last_error(ctx), "segment");
    n4m_pipeline_destroy(even_segment);

    n4m_pipeline_t* bad_order = nullptr;
    const double order_params[] = {1.0, 1.0, 5.0};
    CHECK_EQ(n4m_pipeline_create(&bad_order), N4M_OK);
    CHECK_EQ(n4m_pipeline_add_operator(bad_order, N4M_OP_NORRIS_WILLIAMS,
                                       order_params, 3),
             N4M_OK);
    CHECK_EQ(n4m_pipeline_fit(ctx, bad_order, &X, nullptr), N4M_ERR_INVALID_ARGUMENT);
    CHECK_STR_CONTAINS(n4m_context_last_error(ctx), "derivative");
    n4m_pipeline_destroy(bad_order);

    n4m_context_destroy(ctx);
}

TEST(pipeline_phase3h, wavelet_denoise_zero_threshold_is_identity) {
    n4m_context_t* ctx = nullptr;
    n4m_pipeline_t* pipe = nullptr;
    CHECK_EQ(n4m_context_create(&ctx), N4M_OK);
    CHECK_EQ(n4m_pipeline_create(&pipe), N4M_OK);
    const double params[] = {2.0, 0.0};
    CHECK_EQ(n4m_pipeline_add_operator(pipe, N4M_OP_WAVELET_DENOISE, params, 2), N4M_OK);

    double x[] = {
        1.0, 2.0, 4.0, 8.0,
        -1.0, 0.5, 0.25, 3.0,
    };
    n4m_matrix_view_t X{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X, x, 2, 4, N4M_DTYPE_F64), N4M_OK);
    CHECK_EQ(n4m_pipeline_fit(ctx, pipe, &X, nullptr), N4M_OK);
    n4m_array_t* out = nullptr;
    CHECK_EQ(n4m_pipeline_transform_alloc(ctx, pipe, &X, &out), N4M_OK);
    check_close_vector(failures, copy_values(out), {
        1.0, 2.0, 4.0, 8.0,
        -1.0, 0.5, 0.25, 3.0,
    });

    n4m_array_free(out);
    n4m_pipeline_destroy(pipe);
    n4m_context_destroy(ctx);
}

TEST(pipeline_phase3h, wavelet_denoise_rejects_invalid_params_at_fit) {
    n4m_context_t* ctx = nullptr;
    CHECK_EQ(n4m_context_create(&ctx), N4M_OK);

    double x[] = {
        1.0, 2.0, 3.0,
        2.0, 3.0, 4.0,
    };
    n4m_matrix_view_t X{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X, x, 2, 3, N4M_DTYPE_F64), N4M_OK);

    n4m_pipeline_t* negative_threshold = nullptr;
    const double threshold_params[] = {1.0, -0.1};
    CHECK_EQ(n4m_pipeline_create(&negative_threshold), N4M_OK);
    CHECK_EQ(n4m_pipeline_add_operator(negative_threshold, N4M_OP_WAVELET_DENOISE,
                                       threshold_params, 2),
             N4M_OK);
    CHECK_EQ(n4m_pipeline_fit(ctx, negative_threshold, &X, nullptr),
             N4M_ERR_INVALID_ARGUMENT);
    CHECK_STR_CONTAINS(n4m_context_last_error(ctx), "threshold");
    n4m_pipeline_destroy(negative_threshold);

    n4m_pipeline_t* too_many_levels = nullptr;
    const double levels_params[] = {4.0, 0.1};
    CHECK_EQ(n4m_pipeline_create(&too_many_levels), N4M_OK);
    CHECK_EQ(n4m_pipeline_add_operator(too_many_levels, N4M_OP_WAVELET_DENOISE,
                                       levels_params, 2),
             N4M_OK);
    CHECK_EQ(n4m_pipeline_fit(ctx, too_many_levels, &X, nullptr), N4M_ERR_INVALID_ARGUMENT);
    CHECK_STR_CONTAINS(n4m_context_last_error(ctx), "levels");
    n4m_pipeline_destroy(too_many_levels);

    n4m_context_destroy(ctx);
}

TEST(pipeline_phase3i, osc_requires_y_and_valid_params_at_fit) {
    n4m_context_t* ctx = nullptr;
    n4m_pipeline_t* pipe = nullptr;
    CHECK_EQ(n4m_context_create(&ctx), N4M_OK);
    CHECK_EQ(n4m_pipeline_create(&pipe), N4M_OK);
    const double params[] = {100.0, 1e-12};
    CHECK_EQ(n4m_pipeline_add_operator(pipe, N4M_OP_OSC, params, 2), N4M_OK);

    double x[] = {
        1.0, 2.0, 1.5,
        2.0, 2.5, 2.0,
        3.0, 3.0, 3.5,
        4.0, 3.5, 4.0,
    };
    double y[] = {0.0, 1.0, 1.5, 3.0};
    n4m_matrix_view_t X{};
    n4m_matrix_view_t Y{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X, x, 4, 3, N4M_DTYPE_F64), N4M_OK);
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&Y, y, 4, 1, N4M_DTYPE_F64), N4M_OK);

    CHECK_EQ(n4m_pipeline_fit(ctx, pipe, &X, nullptr), N4M_ERR_INVALID_ARGUMENT);
    CHECK_STR_CONTAINS(n4m_context_last_error(ctx), "Y");
    CHECK_EQ(n4m_pipeline_fit(ctx, pipe, &X, &Y), N4M_OK);

    n4m_array_t* out = nullptr;
    CHECK_EQ(n4m_pipeline_transform_alloc(ctx, pipe, &X, &out), N4M_OK);
    std::int64_t rows = 0;
    std::int64_t cols = 0;
    CHECK_EQ(n4m_array_shape(out, &rows, &cols), N4M_OK);
    CHECK_EQ(rows, 4);
    CHECK_EQ(cols, 3);

    n4m_array_free(out);
    n4m_pipeline_destroy(pipe);
    n4m_context_destroy(ctx);
}

TEST(pipeline_phase3i, osc_rejects_invalid_params_and_singular_y) {
    n4m_context_t* ctx = nullptr;
    CHECK_EQ(n4m_context_create(&ctx), N4M_OK);

    double x[] = {
        1.0, 2.0, 1.5,
        2.0, 2.5, 2.0,
        3.0, 3.0, 3.5,
        4.0, 3.5, 4.0,
    };
    double y_constant[] = {1.0, 1.0, 1.0, 1.0};
    n4m_matrix_view_t X{};
    n4m_matrix_view_t Y{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X, x, 4, 3, N4M_DTYPE_F64), N4M_OK);
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&Y, y_constant, 4, 1, N4M_DTYPE_F64), N4M_OK);

    n4m_pipeline_t* bad_tol = nullptr;
    const double bad_params[] = {100.0, 0.0};
    CHECK_EQ(n4m_pipeline_create(&bad_tol), N4M_OK);
    CHECK_EQ(n4m_pipeline_add_operator(bad_tol, N4M_OP_OSC, bad_params, 2), N4M_OK);
    CHECK_EQ(n4m_pipeline_fit(ctx, bad_tol, &X, &Y), N4M_ERR_INVALID_ARGUMENT);
    CHECK_STR_CONTAINS(n4m_context_last_error(ctx), "tol");
    n4m_pipeline_destroy(bad_tol);

    n4m_pipeline_t* singular = nullptr;
    CHECK_EQ(n4m_pipeline_create(&singular), N4M_OK);
    CHECK_EQ(n4m_pipeline_add_operator(singular, N4M_OP_OSC, nullptr, 0), N4M_OK);
    CHECK_EQ(n4m_pipeline_fit(ctx, singular, &X, &Y), N4M_ERR_NUMERICAL_FAILURE);
    CHECK_STR_CONTAINS(n4m_context_last_error(ctx), "singular");
    n4m_pipeline_destroy(singular);

    n4m_context_destroy(ctx);
}

TEST(pipeline_phase3j, epo_requires_y_and_valid_params_at_fit) {
    n4m_context_t* ctx = nullptr;
    n4m_pipeline_t* pipe = nullptr;
    CHECK_EQ(n4m_context_create(&ctx), N4M_OK);
    CHECK_EQ(n4m_pipeline_create(&pipe), N4M_OK);
    const double params[] = {100.0, 1e-12};
    CHECK_EQ(n4m_pipeline_add_operator(pipe, N4M_OP_EPO, params, 2), N4M_OK);

    double x[] = {
        1.0, 2.0, 1.5,
        2.0, 2.5, 2.0,
        3.0, 3.0, 3.5,
        4.0, 3.5, 4.0,
    };
    double y[] = {0.0, 1.0, 1.5, 3.0};
    n4m_matrix_view_t X{};
    n4m_matrix_view_t Y{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X, x, 4, 3, N4M_DTYPE_F64), N4M_OK);
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&Y, y, 4, 1, N4M_DTYPE_F64), N4M_OK);

    CHECK_EQ(n4m_pipeline_fit(ctx, pipe, &X, nullptr), N4M_ERR_INVALID_ARGUMENT);
    CHECK_STR_CONTAINS(n4m_context_last_error(ctx), "Y");
    CHECK_EQ(n4m_pipeline_fit(ctx, pipe, &X, &Y), N4M_OK);

    n4m_array_t* out = nullptr;
    CHECK_EQ(n4m_pipeline_transform_alloc(ctx, pipe, &X, &out), N4M_OK);
    std::int64_t rows = 0;
    std::int64_t cols = 0;
    CHECK_EQ(n4m_array_shape(out, &rows, &cols), N4M_OK);
    CHECK_EQ(rows, 4);
    CHECK_EQ(cols, 3);

    n4m_array_free(out);
    n4m_pipeline_destroy(pipe);
    n4m_context_destroy(ctx);
}

TEST(pipeline_phase3j, epo_rejects_invalid_params_and_zero_external_covariance) {
    n4m_context_t* ctx = nullptr;
    CHECK_EQ(n4m_context_create(&ctx), N4M_OK);

    double x[] = {
        1.0, 2.0, 3.0,
        1.0, 2.0, 3.0,
        1.0, 2.0, 3.0,
        1.0, 2.0, 3.0,
    };
    double y[] = {0.0, 1.0, 2.0, 3.0};
    n4m_matrix_view_t X{};
    n4m_matrix_view_t Y{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X, x, 4, 3, N4M_DTYPE_F64), N4M_OK);
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&Y, y, 4, 1, N4M_DTYPE_F64), N4M_OK);

    n4m_pipeline_t* bad_tol = nullptr;
    const double bad_params[] = {100.0, 0.0};
    CHECK_EQ(n4m_pipeline_create(&bad_tol), N4M_OK);
    CHECK_EQ(n4m_pipeline_add_operator(bad_tol, N4M_OP_EPO, bad_params, 2), N4M_OK);
    CHECK_EQ(n4m_pipeline_fit(ctx, bad_tol, &X, &Y), N4M_ERR_INVALID_ARGUMENT);
    CHECK_STR_CONTAINS(n4m_context_last_error(ctx), "tol");
    n4m_pipeline_destroy(bad_tol);

    n4m_pipeline_t* zero_covariance = nullptr;
    CHECK_EQ(n4m_pipeline_create(&zero_covariance), N4M_OK);
    CHECK_EQ(n4m_pipeline_add_operator(zero_covariance, N4M_OP_EPO, nullptr, 0), N4M_OK);
    CHECK_EQ(n4m_pipeline_fit(ctx, zero_covariance, &X, &Y), N4M_ERR_NUMERICAL_FAILURE);
    CHECK_STR_CONTAINS(n4m_context_last_error(ctx), "covariance");
    n4m_pipeline_destroy(zero_covariance);

    n4m_context_destroy(ctx);
}
