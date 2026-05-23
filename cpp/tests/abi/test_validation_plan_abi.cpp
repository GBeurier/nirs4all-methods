// SPDX-License-Identifier: CECILL-2.1
//
// ABI surface tests for n4m_validation_plan_t.

#include <cstdint>

#include "n4m/n4m.h"

#include "harness.hpp"

TEST(validation_plan_abi_phase6f, create_destroy_basic_round_trip) {
    n4m_validation_plan_t* plan = nullptr;
    CHECK_EQ(n4m_validation_plan_create(&plan), N4M_OK);
    CHECK(plan != nullptr);

    int64_t n_samples = -1;
    CHECK_EQ(n4m_validation_plan_get_n_samples(plan, &n_samples), N4M_OK);
    CHECK_EQ(n_samples, static_cast<int64_t>(0));

    int32_t n_folds = -1;
    CHECK_EQ(n4m_validation_plan_get_n_folds(plan, &n_folds), N4M_OK);
    CHECK_EQ(n_folds, 0);

    CHECK_EQ(n4m_validation_plan_set_n_samples(plan, 10), N4M_OK);
    CHECK_EQ(n4m_validation_plan_get_n_samples(plan, &n_samples), N4M_OK);
    CHECK_EQ(n_samples, static_cast<int64_t>(10));

    const int64_t train[] = {0, 1, 2, 3, 4, 5, 6};
    const int64_t test[]  = {7, 8, 9};
    CHECK_EQ(n4m_validation_plan_add_fold(plan, train, 7, test, 3), N4M_OK);
    CHECK_EQ(n4m_validation_plan_get_n_folds(plan, &n_folds), N4M_OK);
    CHECK_EQ(n_folds, 1);

    CHECK_EQ(n4m_validation_plan_add_fold(plan, test, 3, train, 7), N4M_OK);
    CHECK_EQ(n4m_validation_plan_get_n_folds(plan, &n_folds), N4M_OK);
    CHECK_EQ(n_folds, 2);

    n4m_validation_plan_destroy(plan);
    n4m_validation_plan_destroy(nullptr);  // must be a no-op
}

TEST(validation_plan_abi_phase6f, rejects_invalid_arguments) {
    CHECK_EQ(n4m_validation_plan_create(nullptr), N4M_ERR_NULL_POINTER);

    n4m_validation_plan_t* plan = nullptr;
    CHECK_EQ(n4m_validation_plan_create(&plan), N4M_OK);

    CHECK_EQ(n4m_validation_plan_set_n_samples(nullptr, 5),
             N4M_ERR_NULL_POINTER);
    CHECK_EQ(n4m_validation_plan_set_n_samples(plan, -1),
             N4M_ERR_INVALID_ARGUMENT);

    const int64_t a[] = {0};
    const int64_t b[] = {1};

    CHECK_EQ(n4m_validation_plan_add_fold(nullptr, a, 1, b, 1),
             N4M_ERR_NULL_POINTER);
    CHECK_EQ(n4m_validation_plan_add_fold(plan, a, 0, b, 1),
             N4M_ERR_INVALID_ARGUMENT);
    CHECK_EQ(n4m_validation_plan_add_fold(plan, a, 1, b, 0),
             N4M_ERR_INVALID_ARGUMENT);
    CHECK_EQ(n4m_validation_plan_add_fold(plan, nullptr, 1, b, 1),
             N4M_ERR_NULL_POINTER);
    CHECK_EQ(n4m_validation_plan_add_fold(plan, a, 1, nullptr, 1),
             N4M_ERR_NULL_POINTER);

    int32_t n_folds = 99;
    CHECK_EQ(n4m_validation_plan_get_n_folds(plan, &n_folds), N4M_OK);
    CHECK_EQ(n_folds, 0);  // failed add_fold left the plan unchanged

    int64_t n_samples = 99;
    CHECK_EQ(n4m_validation_plan_get_n_samples(nullptr, &n_samples),
             N4M_ERR_NULL_POINTER);
    CHECK_EQ(n4m_validation_plan_get_n_samples(plan, nullptr),
             N4M_ERR_NULL_POINTER);
    CHECK_EQ(n4m_validation_plan_get_n_folds(nullptr, &n_folds),
             N4M_ERR_NULL_POINTER);
    CHECK_EQ(n4m_validation_plan_get_n_folds(plan, nullptr),
             N4M_ERR_NULL_POINTER);

    n4m_validation_plan_destroy(plan);
}
