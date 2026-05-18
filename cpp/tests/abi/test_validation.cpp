// SPDX-License-Identifier: CECILL-2.1

#include "core/validation.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "fixtures/validation_fixtures.hpp"
#include "harness.hpp"

namespace {

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
                   const ::pls4all::test::fixtures::ValidationFixture& fixture) {
    ::pls4all::core::Context ctx;
    ::pls4all::core::ValidationPlan plan;
    if (std::strcmp(fixture.kind, "kfold") == 0) {
        CHECK_EQ(::pls4all::core::make_kfold_validation_plan(ctx,
                                                             fixture.n_samples,
                                                             fixture.n_splits,
                                                             plan),
                 P4A_OK);
    } else if (std::strcmp(fixture.kind, "leave_one_out") == 0) {
        CHECK_EQ(::pls4all::core::make_leave_one_out_validation_plan(ctx,
                                                                     fixture.n_samples,
                                                                     plan),
                 P4A_OK);
    } else if (std::strcmp(fixture.kind, "holdout") == 0) {
        CHECK_EQ(::pls4all::core::make_holdout_validation_plan(ctx,
                                                               fixture.n_samples,
                                                               fixture.holdout_start,
                                                               fixture.holdout_count,
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

TEST(validation_phase3l, generated_split_fixtures_match_numpy_reference) {
    for (const auto& fixture : ::pls4all::test::fixtures::kValidationFixtures) {
        check_fixture(failures, fixture);
    }
}

TEST(validation_phase3l, rejects_invalid_split_requests) {
    ::pls4all::core::Context ctx;
    ::pls4all::core::ValidationPlan plan;
    CHECK_EQ(::pls4all::core::make_kfold_validation_plan(ctx, 1, 2, plan),
             P4A_ERR_INVALID_ARGUMENT);
    CHECK_EQ(::pls4all::core::make_kfold_validation_plan(ctx, 4, 1, plan),
             P4A_ERR_INVALID_ARGUMENT);
    CHECK_EQ(::pls4all::core::make_kfold_validation_plan(ctx, 4, 5, plan),
             P4A_ERR_INVALID_ARGUMENT);
    CHECK_EQ(::pls4all::core::make_leave_one_out_validation_plan(ctx, 1, plan),
             P4A_ERR_INVALID_ARGUMENT);
    CHECK_EQ(::pls4all::core::make_holdout_validation_plan(ctx, 5, -1, 2, plan),
             P4A_ERR_INVALID_ARGUMENT);
    CHECK_EQ(::pls4all::core::make_holdout_validation_plan(ctx, 5, 1, 0, plan),
             P4A_ERR_INVALID_ARGUMENT);
    CHECK_EQ(::pls4all::core::make_holdout_validation_plan(ctx, 5, 4, 2, plan),
             P4A_ERR_INVALID_ARGUMENT);
    CHECK_EQ(::pls4all::core::make_holdout_validation_plan(ctx, 5, 0, 5, plan),
             P4A_ERR_INVALID_ARGUMENT);
}
