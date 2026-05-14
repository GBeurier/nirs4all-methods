// SPDX-License-Identifier: CeCILL-2.1

#include "core/validation.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "fixtures/advanced_validation_fixtures.hpp"
#include "harness.hpp"

namespace {

p4a_matrix_view_t matrix_view(const ::pls4all::test::fixtures::MatrixRef& ref) {
    p4a_matrix_view_t view{};
    view.data = const_cast<double*>(ref.values);
    view.rows = ref.rows;
    view.cols = ref.cols;
    view.row_stride = ref.cols > 0 ? ref.cols : 1;
    view.col_stride = 1;
    view.dtype = P4A_DTYPE_F64;
    return view;
}

std::vector<std::int64_t> offsets_for_train(const ::pls4all::core::ValidationPlan& plan) {
    std::vector<std::int64_t> offsets;
    offsets.reserve(plan.folds.size() + 1U);
    std::int64_t offset = 0;
    offsets.push_back(offset);
    for (const auto& fold : plan.folds) {
        offset += static_cast<std::int64_t>(fold.train_indices.size());
        offsets.push_back(offset);
    }
    return offsets;
}

std::vector<std::int64_t> offsets_for_test(const ::pls4all::core::ValidationPlan& plan) {
    std::vector<std::int64_t> offsets;
    offsets.reserve(plan.folds.size() + 1U);
    std::int64_t offset = 0;
    offsets.push_back(offset);
    for (const auto& fold : plan.folds) {
        offset += static_cast<std::int64_t>(fold.test_indices.size());
        offsets.push_back(offset);
    }
    return offsets;
}

std::vector<std::int64_t> flatten_train(const ::pls4all::core::ValidationPlan& plan) {
    std::vector<std::int64_t> out;
    for (const auto& fold : plan.folds) {
        out.insert(out.end(), fold.train_indices.begin(), fold.train_indices.end());
    }
    return out;
}

std::vector<std::int64_t> flatten_test(const ::pls4all::core::ValidationPlan& plan) {
    std::vector<std::int64_t> out;
    for (const auto& fold : plan.folds) {
        out.insert(out.end(), fold.test_indices.begin(), fold.test_indices.end());
    }
    return out;
}

void check_index_ref(int& failures,
                     const char* label,
                     const std::vector<std::int64_t>& actual,
                     const ::pls4all::test::fixtures::IndexRef& expected) {
    if (actual.size() != expected.size) {
        ++failures;
        std::fprintf(stderr,
                     "  FAIL %s size: actual %lu expected %lu\n",
                     label,
                     static_cast<unsigned long>(actual.size()),
                     static_cast<unsigned long>(expected.size));
        return;
    }
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected.values[i]) {
            ++failures;
            std::fprintf(stderr,
                         "  FAIL %s[%lu]: actual %lld expected %lld\n",
                         label,
                         static_cast<unsigned long>(i),
                         static_cast<long long>(actual[i]),
                         static_cast<long long>(expected.values[i]));
            return;
        }
    }
}

void check_fixture(int& failures,
                   const ::pls4all::test::fixtures::AdvancedValidationFixture& fixture) {
    ::pls4all::core::Context ctx;
    ::pls4all::core::ValidationPlan plan;
    if (std::strcmp(fixture.kind, "external") == 0) {
        CHECK_EQ(::pls4all::core::make_external_folds_validation_plan(ctx,
                                                                      fixture.n_samples,
                                                                      fixture.fold_ids.values,
                                                                      fixture.n_splits,
                                                                      plan),
                 P4A_OK);
    } else if (std::strcmp(fixture.kind, "repeated_kfold") == 0) {
        CHECK_EQ(::pls4all::core::make_repeated_kfold_validation_plan(ctx,
                                                                      fixture.n_samples,
                                                                      fixture.n_splits,
                                                                      fixture.n_repeats,
                                                                      fixture.seed,
                                                                      plan),
                 P4A_OK);
    } else if (std::strcmp(fixture.kind, "monte_carlo") == 0) {
        CHECK_EQ(::pls4all::core::make_monte_carlo_validation_plan(ctx,
                                                                   fixture.n_samples,
                                                                   fixture.test_count,
                                                                   fixture.n_repeats,
                                                                   fixture.seed,
                                                                   plan),
                 P4A_OK);
    } else if (std::strcmp(fixture.kind, "kennard_stone") == 0) {
        p4a_matrix_view_t X = matrix_view(fixture.X);
        CHECK_EQ(::pls4all::core::make_kennard_stone_validation_plan(ctx,
                                                                     X,
                                                                     fixture.train_count,
                                                                     plan),
                 P4A_OK);
    } else if (std::strcmp(fixture.kind, "spxy") == 0) {
        p4a_matrix_view_t X = matrix_view(fixture.X);
        p4a_matrix_view_t Y = matrix_view(fixture.Y);
        CHECK_EQ(::pls4all::core::make_spxy_validation_plan(ctx,
                                                            X,
                                                            Y,
                                                            fixture.train_count,
                                                            plan),
                 P4A_OK);
    } else {
        CHECK(false);
        return;
    }

    CHECK_EQ(plan.n_samples, fixture.n_samples);
    CHECK_EQ(plan.folds.size() + 1U, fixture.train_offsets.size);
    check_index_ref(failures, "train_offsets", offsets_for_train(plan), fixture.train_offsets);
    check_index_ref(failures, "train_indices", flatten_train(plan), fixture.train_indices);
    check_index_ref(failures, "test_offsets", offsets_for_test(plan), fixture.test_offsets);
    check_index_ref(failures, "test_indices", flatten_test(plan), fixture.test_indices);
}

}  // namespace

TEST(validation_phase3q, generated_advanced_split_fixtures_match_python_reference) {
    for (const auto& fixture : ::pls4all::test::fixtures::kAdvancedValidationFixtures) {
        check_fixture(failures, fixture);
    }
}

TEST(validation_phase3q, rejects_invalid_advanced_split_requests) {
    ::pls4all::core::Context ctx;
    ::pls4all::core::ValidationPlan plan;
    const std::int64_t bad_fold_ids[] = {0, 1, 3, 0};
    CHECK_EQ(::pls4all::core::make_external_folds_validation_plan(ctx, 4, nullptr, 2, plan),
             P4A_ERR_NULL_POINTER);
    CHECK_EQ(::pls4all::core::make_external_folds_validation_plan(ctx, 4, bad_fold_ids, 2, plan),
             P4A_ERR_INVALID_ARGUMENT);
    CHECK_EQ(::pls4all::core::make_repeated_kfold_validation_plan(ctx, 4, 2, 0, 7, plan),
             P4A_ERR_INVALID_ARGUMENT);
    CHECK_EQ(::pls4all::core::make_monte_carlo_validation_plan(ctx, 4, 4, 1, 7, plan),
             P4A_ERR_INVALID_ARGUMENT);

    const double x_values[] = {
        0.0, 0.1,
        1.0, 0.4,
        0.2, 1.5,
    };
    p4a_matrix_view_t X{};
    X.data = const_cast<double*>(x_values);
    X.rows = 3;
    X.cols = 2;
    X.row_stride = 2;
    X.col_stride = 1;
    X.dtype = P4A_DTYPE_F64;
    CHECK_EQ(::pls4all::core::make_kennard_stone_validation_plan(ctx, X, 1, plan),
             P4A_ERR_INVALID_ARGUMENT);

    const double y_values[] = {0.0, 1.0};
    p4a_matrix_view_t Y{};
    Y.data = const_cast<double*>(y_values);
    Y.rows = 2;
    Y.cols = 1;
    Y.row_stride = 1;
    Y.col_stride = 1;
    Y.dtype = P4A_DTYPE_F64;
    CHECK_EQ(::pls4all::core::make_spxy_validation_plan(ctx, X, Y, 2, plan),
             P4A_ERR_INVALID_ARGUMENT);
}
