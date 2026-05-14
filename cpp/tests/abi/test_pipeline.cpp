// SPDX-License-Identifier: CeCILL-2.1

#include "pls4all/p4a.h"

#include <cmath>
#include <cstdint>
#include <vector>

#include "fixtures/pipeline_fixtures.hpp"
#include "harness.hpp"

namespace {

constexpr double kAbsTol = 1e-12;

std::vector<double> copy_values(const p4a_array_t* arr) {
    std::int64_t rows = 0;
    std::int64_t cols = 0;
    p4a_array_shape(arr, &rows, &cols);
    p4a_matrix_view_t view{};
    p4a_array_view(arr, &view);
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
    p4a_context_t* ctx = nullptr;
    p4a_pipeline_t* pipe = nullptr;
    CHECK_EQ(p4a_context_create(&ctx), P4A_OK);
    CHECK_EQ(p4a_pipeline_create(&pipe), P4A_OK);

    double x[] = {1.0, 2.0, 3.0, 4.0};
    p4a_matrix_view_t X{};
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&X, x, 2, 2, P4A_DTYPE_F64), P4A_OK);
    p4a_array_t* out = nullptr;
    CHECK_EQ(p4a_pipeline_transform_alloc(ctx, pipe, &X, &out), P4A_ERR_NOT_FITTED);
    CHECK_EQ(out, nullptr);
    CHECK_STR_CONTAINS(p4a_context_last_error(ctx), "not fitted");

    p4a_pipeline_destroy(pipe);
    p4a_context_destroy(ctx);
}

TEST(pipeline_phase3a, center_transform_alloc_matches_numpy_reference) {
    p4a_context_t* ctx = nullptr;
    p4a_pipeline_t* pipe = nullptr;
    CHECK_EQ(p4a_context_create(&ctx), P4A_OK);
    CHECK_EQ(p4a_pipeline_create(&pipe), P4A_OK);
    CHECK_EQ(p4a_pipeline_add_operator(pipe, P4A_OP_CENTER, nullptr, 0), P4A_OK);

    double x[] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    };
    p4a_matrix_view_t X{};
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&X, x, 3, 3, P4A_DTYPE_F64), P4A_OK);
    CHECK_EQ(p4a_pipeline_fit(ctx, pipe, &X, nullptr), P4A_OK);

    p4a_array_t* out = nullptr;
    CHECK_EQ(p4a_pipeline_transform_alloc(ctx, pipe, &X, &out), P4A_OK);
    std::int64_t rows = 0;
    std::int64_t cols = 0;
    CHECK_EQ(p4a_array_shape(out, &rows, &cols), P4A_OK);
    CHECK_EQ(rows, 3);
    CHECK_EQ(cols, 3);
    check_close_vector(failures, copy_values(out), {
        -3.0, -3.0, -3.0,
         0.0,  0.0,  0.0,
         3.0,  3.0,  3.0,
    });

    p4a_array_free(out);
    p4a_pipeline_destroy(pipe);
    p4a_context_destroy(ctx);
}

TEST(pipeline_phase3a, generated_fixtures_match_python_reference) {
    for (const auto& fixture : ::pls4all::test::fixtures::kPipelineFixtures) {
        p4a_context_t* ctx = nullptr;
        p4a_pipeline_t* pipe = nullptr;
        CHECK_EQ(p4a_context_create(&ctx), P4A_OK);
        CHECK_EQ(p4a_pipeline_create(&pipe), P4A_OK);
        for (std::size_t op = 0; op < fixture.n_operators; ++op) {
            CHECK_EQ(p4a_pipeline_add_operator(pipe, fixture.operators[op], nullptr, 0), P4A_OK);
        }

        p4a_matrix_view_t X{};
        CHECK_EQ(p4a_matrix_view_init_rowmajor(&X,
                                               const_cast<double*>(fixture.X.values),
                                               fixture.X.rows,
                                               fixture.X.cols,
                                               P4A_DTYPE_F64),
                 P4A_OK);
        CHECK_EQ(p4a_pipeline_fit(ctx, pipe, &X, nullptr), P4A_OK);
        p4a_array_t* out = nullptr;
        CHECK_EQ(p4a_pipeline_transform_alloc(ctx, pipe, &X, &out), P4A_OK);
        check_close_vector(failures, copy_values(out),
                           std::vector<double>(
                               fixture.transform_train.values,
                               fixture.transform_train.values + fixture.transform_train.size));
        p4a_array_free(out);
        p4a_pipeline_destroy(pipe);
        p4a_context_destroy(ctx);
    }
}

TEST(pipeline_phase3a, autoscale_and_pareto_use_training_statistics) {
    p4a_context_t* ctx = nullptr;
    CHECK_EQ(p4a_context_create(&ctx), P4A_OK);

    double x[] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    };
    p4a_matrix_view_t X{};
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&X, x, 3, 3, P4A_DTYPE_F64), P4A_OK);

    p4a_pipeline_t* autoscale = nullptr;
    CHECK_EQ(p4a_pipeline_create(&autoscale), P4A_OK);
    CHECK_EQ(p4a_pipeline_add_operator(autoscale, P4A_OP_AUTOSCALE, nullptr, 0), P4A_OK);
    CHECK_EQ(p4a_pipeline_fit(ctx, autoscale, &X, nullptr), P4A_OK);
    double autoscaled[9] = {};
    p4a_matrix_view_t A{};
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&A, autoscaled, 3, 3, P4A_DTYPE_F64), P4A_OK);
    CHECK_EQ(p4a_pipeline_transform(ctx, autoscale, &X, &A), P4A_OK);
    check_close_vector(failures, std::vector<double>(autoscaled, autoscaled + 9), {
        -1.0, -1.0, -1.0,
         0.0,  0.0,  0.0,
         1.0,  1.0,  1.0,
    });

    p4a_pipeline_t* pareto = nullptr;
    CHECK_EQ(p4a_pipeline_create(&pareto), P4A_OK);
    CHECK_EQ(p4a_pipeline_add_operator(pareto, P4A_OP_PARETO_SCALE, nullptr, 0), P4A_OK);
    CHECK_EQ(p4a_pipeline_fit(ctx, pareto, &X, nullptr), P4A_OK);
    p4a_array_t* out = nullptr;
    CHECK_EQ(p4a_pipeline_transform_alloc(ctx, pareto, &X, &out), P4A_OK);
    const double s = std::sqrt(3.0);
    check_close_vector(failures, copy_values(out), {
        -s, -s, -s,
         0.0, 0.0, 0.0,
         s, s, s,
    });

    p4a_array_free(out);
    p4a_pipeline_destroy(pareto);
    p4a_pipeline_destroy(autoscale);
    p4a_context_destroy(ctx);
}

TEST(pipeline_phase3a, snv_is_rowwise_and_handles_flat_rows) {
    p4a_context_t* ctx = nullptr;
    p4a_pipeline_t* pipe = nullptr;
    CHECK_EQ(p4a_context_create(&ctx), P4A_OK);
    CHECK_EQ(p4a_pipeline_create(&pipe), P4A_OK);
    CHECK_EQ(p4a_pipeline_add_operator(pipe, P4A_OP_SNV, nullptr, 0), P4A_OK);

    double x[] = {
        1.0, 2.0, 3.0,
        2.0, 2.0, 2.0,
        3.0, 5.0, 7.0,
    };
    p4a_matrix_view_t X{};
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&X, x, 3, 3, P4A_DTYPE_F64), P4A_OK);
    CHECK_EQ(p4a_pipeline_fit(ctx, pipe, &X, nullptr), P4A_OK);
    p4a_array_t* out = nullptr;
    CHECK_EQ(p4a_pipeline_transform_alloc(ctx, pipe, &X, &out), P4A_OK);
    check_close_vector(failures, copy_values(out), {
        -1.0, 0.0, 1.0,
         0.0, 0.0, 0.0,
        -1.0, 0.0, 1.0,
    });

    p4a_array_free(out);
    p4a_pipeline_destroy(pipe);
    p4a_context_destroy(ctx);
}

TEST(pipeline_phase3a, unsupported_operators_fail_at_fit_with_context_error) {
    p4a_context_t* ctx = nullptr;
    p4a_pipeline_t* pipe = nullptr;
    CHECK_EQ(p4a_context_create(&ctx), P4A_OK);
    CHECK_EQ(p4a_pipeline_create(&pipe), P4A_OK);
    CHECK_EQ(p4a_pipeline_add_operator(pipe, P4A_OP_MSC, nullptr, 0), P4A_OK);

    double x[] = {1.0, 2.0, 3.0, 4.0};
    p4a_matrix_view_t X{};
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&X, x, 2, 2, P4A_DTYPE_F64), P4A_OK);
    CHECK_EQ(p4a_pipeline_fit(ctx, pipe, &X, nullptr), P4A_ERR_NOT_IMPLEMENTED);
    CHECK_STR_CONTAINS(p4a_context_last_error(ctx), "not implemented");

    p4a_pipeline_destroy(pipe);
    p4a_context_destroy(ctx);
}
