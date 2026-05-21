// SPDX-License-Identifier: CECILL-2.1
// Generated mechanically from parity/fixtures/*aom-pop-selection*.json.
#pragma once

#include <cstddef>
#include <cstdint>

#include "phase1_fixtures.hpp"

namespace n4m::test::fixtures {

struct AomPopSelectionIndexRef {
    const std::int64_t* values;
    std::size_t size;
};

struct AomPopSelectionFixture {
    const char* id;
    std::int32_t max_components;
    std::int32_t n_operators;
    std::int32_t n_folds;
    std::int32_t selected_n_components;
    double best_score;
    MatrixRef X;
    MatrixRef Y;
    MatrixRef operator_params;
    MatrixRef component_scores;
    MatrixRef prefix_scores;
    MatrixRef predictions;
    AomPopSelectionIndexRef operator_kinds;
    AomPopSelectionIndexRef selected_operator_indices;
    AomPopSelectionIndexRef operator_param_offsets;
    AomPopSelectionIndexRef fold_train_offsets;
    AomPopSelectionIndexRef fold_train_indices;
    AomPopSelectionIndexRef fold_test_offsets;
    AomPopSelectionIndexRef fold_test_indices;
};

inline const double synthetic_aom_pop_simpls_covariance_cv_v1_X[] = {
    0.10000000000000001, 0.32000000000000001, 0.69999999999999996, 1.1799999999999999,
    1.72, 2.1800000000000002, 2.52, 2.8199999999999998,
    3.0800000000000001, 0.22, 0.47999999999999998, 0.85999999999999999,
    1.3400000000000001, 1.9199999999999999, 2.4199999999999999, 2.8199999999999998,
    3.1400000000000001, 3.4500000000000002, 0.17999999999999999, 0.38,
    0.76000000000000001, 1.22, 1.78, 2.2599999999999998,
    2.5800000000000001, 2.8799999999999999, 3.1600000000000001, 0.38,
    0.71999999999999997, 1.1799999999999999, 1.72, 2.3399999999999999,
    2.8999999999999999, 3.3199999999999998, 3.7000000000000002, 4.0800000000000001,
    0.52000000000000002, 0.93999999999999995, 1.48, 2.0600000000000001,
    2.7200000000000002, 3.2799999999999998, 3.7799999999999998, 4.1799999999999997,
    4.6200000000000001, 0.69999999999999996, 1.1799999999999999, 1.78,
    2.4199999999999999, 3.1000000000000001, 3.7400000000000002, 4.2800000000000002,
    4.7599999999999998, 5.2000000000000002, 0.88, 1.4399999999999999,
    2.1000000000000001, 2.8199999999999998, 3.5600000000000001, 4.2400000000000002,
    4.8399999999999999, 5.3600000000000003, 5.8600000000000003, 1.0800000000000001,
    1.72, 2.4399999999999999, 3.2200000000000002, 4.0199999999999996,
    4.7800000000000002, 5.46, 6.04, 6.5999999999999996,
    1.3, 2.02, 2.8199999999999998, 3.6600000000000001,
    4.54, 5.3399999999999999, 6.0999999999999996, 6.7599999999999998,
    7.3799999999999999, 1.54, 2.3399999999999999, 3.2200000000000002,
    4.1399999999999997, 5.0999999999999996, 6, 6.8200000000000003,
    7.5599999999999996, 8.2200000000000006, 1.8200000000000001, 2.7000000000000002,
    3.6600000000000001, 4.6600000000000001, 5.7000000000000002, 6.6799999999999997,
    7.5800000000000001, 8.4000000000000004, 9.1600000000000001, 2.0800000000000001,
    3.04, 4.0800000000000001, 5.1600000000000001, 6.2800000000000002,
    7.3399999999999999, 8.3200000000000003, 9.2200000000000006, 10.050000000000001,
};

inline const double synthetic_aom_pop_simpls_covariance_cv_v1_Y[] = {
    0.29999999999999999, 0.47999999999999998, 0.35999999999999999, 0.71999999999999997,
    0.93999999999999995, 1.22, 1.52, 1.8400000000000001,
    2.2000000000000002, 2.5800000000000001, 3.02, 3.4399999999999999,
};

inline const double synthetic_aom_pop_simpls_covariance_cv_v1_operator_params[] = {
    1, 5, 2, 1,
    10, 1, 1, 7,
    3,
};

inline const double synthetic_aom_pop_simpls_covariance_cv_v1_component_scores[] = {
    1.3847272978822802, 1.3426873041494416, 1.3847460579442241, 1.3849466307911069,
    1.3847314351720581, 1.38494533030675, 1.4018865148526301, 1.4903189190348269,
    1.380373807379091, 1.3828863942795075, 1.4015321999831543, 1.3710607849434868,
    1.3768500991250929, 1.3909724906529632, 1.3751022615785675, 1.3864739254301413,
    1.4018764037430531, 1.3876081609829434,
};

inline const double synthetic_aom_pop_simpls_covariance_cv_v1_prefix_scores[] = {
    1.3426873041494416, 1.3710607849434868, 1.3751022615785675,
};

inline const double synthetic_aom_pop_simpls_covariance_cv_v1_predictions[] = {
    0.57789430801087172, 0.075976494340784351, 0.14273234593901174, 0.58890123362335711,
    1.3409813078387034, 1.6655686236394378, 2.1167845621931676, 2.1486849492486195,
    2.3723745405849388, 2.5866299471090475, 2.4734378092783591, 2.5300338781936995,
};

inline const std::int64_t synthetic_aom_pop_simpls_covariance_cv_v1_operator_kinds[] = {
    0, 7, 8, 15, 16, 17,
};

inline const std::int64_t synthetic_aom_pop_simpls_covariance_cv_v1_selected_operator_indices[] = {
    1,
};

inline const std::int64_t synthetic_aom_pop_simpls_covariance_cv_v1_operator_param_offsets[] = {
    0, 0, 1, 3, 4, 5, 9,
};

inline const std::int64_t synthetic_aom_pop_simpls_covariance_cv_v1_fold_train_offsets[] = {
    0, 8, 16, 24,
};

inline const std::int64_t synthetic_aom_pop_simpls_covariance_cv_v1_fold_train_indices[] = {
    0, 1, 4, 5, 6, 7, 9, 10,
    0, 1, 2, 3, 6, 8, 10, 11,
    2, 3, 4, 5, 7, 8, 9, 11,
};

inline const std::int64_t synthetic_aom_pop_simpls_covariance_cv_v1_fold_test_offsets[] = {
    0, 4, 8, 12,
};

inline const std::int64_t synthetic_aom_pop_simpls_covariance_cv_v1_fold_test_indices[] = {
    2, 3, 8, 11, 4, 5, 7, 9,
    0, 1, 6, 10,
};

inline const AomPopSelectionFixture kAomPopSelectionFixtures[] = {
    {
        "synthetic_aom_pop_simpls_covariance_cv_v1",
        3,
        6,
        3,
        1,
        1.3426873041494416,
        MatrixRef{12, 9, synthetic_aom_pop_simpls_covariance_cv_v1_X, sizeof(synthetic_aom_pop_simpls_covariance_cv_v1_X) / sizeof(double), false},
        MatrixRef{12, 1, synthetic_aom_pop_simpls_covariance_cv_v1_Y, sizeof(synthetic_aom_pop_simpls_covariance_cv_v1_Y) / sizeof(double), false},
        MatrixRef{1, 9, synthetic_aom_pop_simpls_covariance_cv_v1_operator_params, sizeof(synthetic_aom_pop_simpls_covariance_cv_v1_operator_params) / sizeof(double), false},
        MatrixRef{3, 6, synthetic_aom_pop_simpls_covariance_cv_v1_component_scores, sizeof(synthetic_aom_pop_simpls_covariance_cv_v1_component_scores) / sizeof(double), false},
        MatrixRef{1, 3, synthetic_aom_pop_simpls_covariance_cv_v1_prefix_scores, sizeof(synthetic_aom_pop_simpls_covariance_cv_v1_prefix_scores) / sizeof(double), false},
        MatrixRef{12, 1, synthetic_aom_pop_simpls_covariance_cv_v1_predictions, sizeof(synthetic_aom_pop_simpls_covariance_cv_v1_predictions) / sizeof(double), false},
        AomPopSelectionIndexRef{synthetic_aom_pop_simpls_covariance_cv_v1_operator_kinds, sizeof(synthetic_aom_pop_simpls_covariance_cv_v1_operator_kinds) / sizeof(std::int64_t)},
        AomPopSelectionIndexRef{synthetic_aom_pop_simpls_covariance_cv_v1_selected_operator_indices, sizeof(synthetic_aom_pop_simpls_covariance_cv_v1_selected_operator_indices) / sizeof(std::int64_t)},
        AomPopSelectionIndexRef{synthetic_aom_pop_simpls_covariance_cv_v1_operator_param_offsets, sizeof(synthetic_aom_pop_simpls_covariance_cv_v1_operator_param_offsets) / sizeof(std::int64_t)},
        AomPopSelectionIndexRef{synthetic_aom_pop_simpls_covariance_cv_v1_fold_train_offsets, sizeof(synthetic_aom_pop_simpls_covariance_cv_v1_fold_train_offsets) / sizeof(std::int64_t)},
        AomPopSelectionIndexRef{synthetic_aom_pop_simpls_covariance_cv_v1_fold_train_indices, sizeof(synthetic_aom_pop_simpls_covariance_cv_v1_fold_train_indices) / sizeof(std::int64_t)},
        AomPopSelectionIndexRef{synthetic_aom_pop_simpls_covariance_cv_v1_fold_test_offsets, sizeof(synthetic_aom_pop_simpls_covariance_cv_v1_fold_test_offsets) / sizeof(std::int64_t)},
        AomPopSelectionIndexRef{synthetic_aom_pop_simpls_covariance_cv_v1_fold_test_indices, sizeof(synthetic_aom_pop_simpls_covariance_cv_v1_fold_test_indices) / sizeof(std::int64_t)}
    }
};

}  // namespace n4m::test::fixtures
