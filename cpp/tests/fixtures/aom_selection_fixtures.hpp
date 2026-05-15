// SPDX-License-Identifier: CeCILL-2.1
// Generated mechanically from parity/fixtures/*aom-selection*.json.
#pragma once

#include <cstddef>
#include <cstdint>

#include "phase1_fixtures.hpp"

namespace pls4all::test::fixtures {

struct AomSelectionIndexRef {
    const std::int64_t* values;
    std::size_t size;
};

struct AomSelectionFixture {
    const char* id;
    std::int32_t max_components;
    std::int32_t n_operators;
    std::int32_t n_folds;
    std::int32_t selected_operator_index;
    std::int32_t selected_n_components;
    double best_score;
    MatrixRef X;
    MatrixRef Y;
    MatrixRef operator_params;
    MatrixRef operator_scores;
    MatrixRef rmse_curves;
    MatrixRef predictions;
    AomSelectionIndexRef operator_kinds;
    AomSelectionIndexRef operator_param_offsets;
    AomSelectionIndexRef fold_train_offsets;
    AomSelectionIndexRef fold_train_indices;
    AomSelectionIndexRef fold_test_offsets;
    AomSelectionIndexRef fold_test_indices;
};

inline const double synthetic_aom_global_simpls_cv_v1_X[] = {
    1, 2, 3, 4.5,
    5.5, 7, 1.2, 2.1000000000000001,
    3.3999999999999999, 4.7999999999999998, 5.7999999999999998, 7.5,
    0.90000000000000002, 1.8, 2.7999999999999998, 4.2000000000000002,
    5.2000000000000002, 6.5999999999999996, 1.5, 2.5,
    3.7999999999999998, 5, 6.0999999999999996, 7.7999999999999998,
    1.8, 2.7999999999999998, 4.0999999999999996, 5.4000000000000004,
    6.4000000000000004, 8.0999999999999996, 2.1000000000000001, 3,
    4.5, 5.9000000000000004, 6.9000000000000004, 8.5999999999999996,
    2.2999999999999998, 3.3999999999999999, 4.9000000000000004, 6.2000000000000002,
    7.2999999999999998, 9, 2.6000000000000001, 3.7000000000000002,
    5.2000000000000002, 6.5999999999999996, 7.7999999999999998, 9.4000000000000004,
    2.8999999999999999, 4, 5.5999999999999996, 7.0999999999999996,
    8.1999999999999993, 9.9000000000000004,
};

inline const double synthetic_aom_global_simpls_cv_v1_Y[] = {
    1.1000000000000001, 1.3999999999999999, 0.90000000000000002, 1.8,
    2.1000000000000001, 2.5, 2.7999999999999998, 3.2000000000000002,
    3.6000000000000001,
};

inline const double synthetic_aom_global_simpls_cv_v1_operator_params[] = {
    1,
};

inline const double synthetic_aom_global_simpls_cv_v1_operator_scores[] = {
    0.03241764499634122, 0.31066557971380465,
};

inline const double synthetic_aom_global_simpls_cv_v1_rmse_curves[] = {
    0.034545572906423995, 0.036344939672721538, 0.03241764499634122, 0.48358707123294709,
    0.31066557971380465, 0.39570483321527833,
};

inline const double synthetic_aom_global_simpls_cv_v1_predictions[] = {
    1.0977576532980311, 1.4128118639894953, 0.8937608443110916, 1.7780106960608222,
    2.1087970018757831, 2.4981554602375518, 2.8365307095373504, 3.1897748821870908,
    3.5844008885027874,
};

inline const std::int64_t synthetic_aom_global_simpls_cv_v1_operator_kinds[] = {
    0, 7,
};

inline const std::int64_t synthetic_aom_global_simpls_cv_v1_operator_param_offsets[] = {
    0, 0, 1,
};

inline const std::int64_t synthetic_aom_global_simpls_cv_v1_fold_train_offsets[] = {
    0, 6, 12, 18,
};

inline const std::int64_t synthetic_aom_global_simpls_cv_v1_fold_train_indices[] = {
    1, 3, 4, 5, 6, 8, 0, 1,
    2, 4, 6, 7, 0, 2, 3, 5,
    7, 8,
};

inline const std::int64_t synthetic_aom_global_simpls_cv_v1_fold_test_offsets[] = {
    0, 3, 6, 9,
};

inline const std::int64_t synthetic_aom_global_simpls_cv_v1_fold_test_indices[] = {
    0, 2, 7, 3, 5, 8, 1, 4,
    6,
};

inline const AomSelectionFixture kAomSelectionFixtures[] = {
    {
        "synthetic_aom_global_simpls_cv_v1",
        3,
        2,
        3,
        0,
        3,
        0.03241764499634122,
        MatrixRef{9, 6, synthetic_aom_global_simpls_cv_v1_X, sizeof(synthetic_aom_global_simpls_cv_v1_X) / sizeof(double), false},
        MatrixRef{9, 1, synthetic_aom_global_simpls_cv_v1_Y, sizeof(synthetic_aom_global_simpls_cv_v1_Y) / sizeof(double), false},
        MatrixRef{1, 1, synthetic_aom_global_simpls_cv_v1_operator_params, sizeof(synthetic_aom_global_simpls_cv_v1_operator_params) / sizeof(double), false},
        MatrixRef{1, 2, synthetic_aom_global_simpls_cv_v1_operator_scores, sizeof(synthetic_aom_global_simpls_cv_v1_operator_scores) / sizeof(double), false},
        MatrixRef{2, 3, synthetic_aom_global_simpls_cv_v1_rmse_curves, sizeof(synthetic_aom_global_simpls_cv_v1_rmse_curves) / sizeof(double), false},
        MatrixRef{9, 1, synthetic_aom_global_simpls_cv_v1_predictions, sizeof(synthetic_aom_global_simpls_cv_v1_predictions) / sizeof(double), false},
        AomSelectionIndexRef{synthetic_aom_global_simpls_cv_v1_operator_kinds, sizeof(synthetic_aom_global_simpls_cv_v1_operator_kinds) / sizeof(std::int64_t)},
        AomSelectionIndexRef{synthetic_aom_global_simpls_cv_v1_operator_param_offsets, sizeof(synthetic_aom_global_simpls_cv_v1_operator_param_offsets) / sizeof(std::int64_t)},
        AomSelectionIndexRef{synthetic_aom_global_simpls_cv_v1_fold_train_offsets, sizeof(synthetic_aom_global_simpls_cv_v1_fold_train_offsets) / sizeof(std::int64_t)},
        AomSelectionIndexRef{synthetic_aom_global_simpls_cv_v1_fold_train_indices, sizeof(synthetic_aom_global_simpls_cv_v1_fold_train_indices) / sizeof(std::int64_t)},
        AomSelectionIndexRef{synthetic_aom_global_simpls_cv_v1_fold_test_offsets, sizeof(synthetic_aom_global_simpls_cv_v1_fold_test_offsets) / sizeof(std::int64_t)},
        AomSelectionIndexRef{synthetic_aom_global_simpls_cv_v1_fold_test_indices, sizeof(synthetic_aom_global_simpls_cv_v1_fold_test_indices) / sizeof(std::int64_t)}
    }
};

}  // namespace pls4all::test::fixtures
