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

inline const double synthetic_aom_global_simpls_sg_cv_v1_X[] = {
    0.20000000000000001, 0.55000000000000004, 1.1000000000000001, 1.75,
    2.3500000000000001, 2.7000000000000002, 3.1000000000000001, 0.29999999999999999,
    0.69999999999999996, 1.2, 1.8500000000000001, 2.5499999999999998,
    2.9500000000000002, 3.4500000000000002, 0.17999999999999999, 0.47999999999999998,
    0.94999999999999996, 1.55, 2.1000000000000001, 2.52,
    2.9500000000000002, 0.45000000000000001, 0.88, 1.4199999999999999,
    2.0499999999999998, 2.7200000000000002, 3.1800000000000002, 3.7000000000000002,
    0.62, 1.05, 1.6499999999999999, 2.3199999999999998,
    3, 3.4500000000000002, 4.0199999999999996, 0.78000000000000003,
    1.24, 1.8799999999999999, 2.5800000000000001, 3.3199999999999998,
    3.8199999999999998, 4.3799999999999999, 0.92000000000000004, 1.45,
    2.1200000000000001, 2.8599999999999999, 3.6200000000000001, 4.1799999999999997,
    4.7800000000000002, 1.0800000000000001, 1.6799999999999999, 2.3799999999999999,
    3.1800000000000002, 3.98, 4.5800000000000001, 5.2199999999999998,
    1.22, 1.8600000000000001, 2.6600000000000001, 3.52,
    4.3399999999999999, 5.0199999999999996, 5.7000000000000002, 1.3799999999999999,
    2.0800000000000001, 2.9500000000000002, 3.8799999999999999, 4.7599999999999998,
    5.4800000000000004, 6.2000000000000002,
};

inline const double synthetic_aom_global_simpls_sg_cv_v1_Y[] = {
    0.45000000000000001, 0.62, 0.38, 0.78000000000000003,
    0.95999999999999996, 1.1799999999999999, 1.4199999999999999, 1.7,
    1.98, 2.2999999999999998,
};

inline const double synthetic_aom_global_simpls_sg_cv_v1_operator_params[] = {
    5, 2, 5, 2,
    1, 1, 1, 3,
    1, 1,
};

inline const double synthetic_aom_global_simpls_sg_cv_v1_operator_scores[] = {
    0.021847902496073289, 0.025611280520301961, 0.027507269093284614, 0.018059541712523237,
    0.023486840590688753, 0.50663613263883467,
};

inline const double synthetic_aom_global_simpls_sg_cv_v1_rmse_curves[] = {
    0.024528164527235802, 0.021847902496073289, 0.025182965969335218, 0.025611280520301961,
    0.025715409293449544, 0.02948869472794937, 0.027507269093284614, 0.031950097978757912,
    0.033533221172191111, 0.018059541712523237, 0.022201588308907831, 0.020355484127993154,
    0.023486840590688753, 0.024382894130144189, 0.029427451030387807, 0.50663613263883467,
    0.57407048568003938, 0.55031795072202683,
};

inline const double synthetic_aom_global_simpls_sg_cv_v1_predictions[] = {
    0.45950995413342555, 0.63026308595504499, 0.34380578274632034, 0.78391160632326007,
    0.96075180311863317, 1.1956943033702951, 1.4345528421352123, 1.6983264904137996,
    1.9820899245994084, 2.2810942072045957,
};

inline const std::int64_t synthetic_aom_global_simpls_sg_cv_v1_operator_kinds[] = {
    0, 8, 9, 15, 10, 7,
};

inline const std::int64_t synthetic_aom_global_simpls_sg_cv_v1_operator_param_offsets[] = {
    0, 0, 2, 5, 6, 9, 10,
};

inline const std::int64_t synthetic_aom_global_simpls_sg_cv_v1_fold_train_offsets[] = {
    0, 8, 16, 24, 32, 40,
};

inline const std::int64_t synthetic_aom_global_simpls_sg_cv_v1_fold_train_indices[] = {
    0, 1, 2, 3, 4, 5, 6, 9,
    0, 1, 3, 4, 5, 7, 8, 9,
    0, 1, 2, 3, 6, 7, 8, 9,
    0, 2, 4, 5, 6, 7, 8, 9,
    1, 2, 3, 4, 5, 6, 7, 8,
};

inline const std::int64_t synthetic_aom_global_simpls_sg_cv_v1_fold_test_offsets[] = {
    0, 2, 4, 6, 8, 10,
};

inline const std::int64_t synthetic_aom_global_simpls_sg_cv_v1_fold_test_indices[] = {
    7, 8, 2, 6, 4, 5, 1, 3,
    0, 9,
};

inline const double synthetic_aom_global_simpls_fck_whittaker_cv_v1_X[] = {
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
    7.5800000000000001, 8.4000000000000004, 9.1600000000000001,
};

inline const double synthetic_aom_global_simpls_fck_whittaker_cv_v1_Y[] = {
    0.29999999999999999, 0.47999999999999998, 0.35999999999999999, 0.71999999999999997,
    0.93999999999999995, 1.22, 1.52, 1.8400000000000001,
    2.2000000000000002, 2.5800000000000001, 3.02,
};

inline const double synthetic_aom_global_simpls_fck_whittaker_cv_v1_operator_params[] = {
    10, 1, 1, 7,
    3, 1,
};

inline const double synthetic_aom_global_simpls_fck_whittaker_cv_v1_operator_scores[] = {
    0.014704524576819429, 0.020702022574690223, 0.015418624816780737, 0.33999211407448687,
};

inline const double synthetic_aom_global_simpls_fck_whittaker_cv_v1_rmse_curves[] = {
    0.023955749079044519, 0.016339263884604265, 0.014704524576819429, 0.023897497755225278,
    0.028746857475013366, 0.020702022574690223, 0.022918171154587792, 0.019887156783053372,
    0.015418624816780737, 0.33999211407448687, 0.36652599028691019, 0.40200213366173038,
};

inline const double synthetic_aom_global_simpls_fck_whittaker_cv_v1_predictions[] = {
    0.29667987551082775, 0.47057449883229663, 0.36388397405242867, 0.7298986925824329,
    0.94214423469316944, 1.2214997327879136, 1.5127067399929068, 1.8451482803273778,
    2.1918809803879871, 2.5907837928005266, 3.0147991980321365,
};

inline const std::int64_t synthetic_aom_global_simpls_fck_whittaker_cv_v1_operator_kinds[] = {
    0, 16, 17, 7,
};

inline const std::int64_t synthetic_aom_global_simpls_fck_whittaker_cv_v1_operator_param_offsets[] = {
    0, 0, 1, 5, 6,
};

inline const std::int64_t synthetic_aom_global_simpls_fck_whittaker_cv_v1_fold_train_offsets[] = {
    0, 7, 14, 22,
};

inline const std::int64_t synthetic_aom_global_simpls_fck_whittaker_cv_v1_fold_train_indices[] = {
    0, 1, 2, 4, 6, 9, 10, 0,
    2, 3, 5, 7, 8, 9, 1, 3,
    4, 5, 6, 7, 8, 10,
};

inline const std::int64_t synthetic_aom_global_simpls_fck_whittaker_cv_v1_fold_test_offsets[] = {
    0, 4, 8, 11,
};

inline const std::int64_t synthetic_aom_global_simpls_fck_whittaker_cv_v1_fold_test_indices[] = {
    3, 5, 7, 8, 1, 4, 6, 10,
    0, 2, 9,
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
    },
    {
        "synthetic_aom_global_simpls_sg_cv_v1",
        3,
        6,
        5,
        3,
        1,
        0.018059541712523237,
        MatrixRef{10, 7, synthetic_aom_global_simpls_sg_cv_v1_X, sizeof(synthetic_aom_global_simpls_sg_cv_v1_X) / sizeof(double), false},
        MatrixRef{10, 1, synthetic_aom_global_simpls_sg_cv_v1_Y, sizeof(synthetic_aom_global_simpls_sg_cv_v1_Y) / sizeof(double), false},
        MatrixRef{1, 10, synthetic_aom_global_simpls_sg_cv_v1_operator_params, sizeof(synthetic_aom_global_simpls_sg_cv_v1_operator_params) / sizeof(double), false},
        MatrixRef{1, 6, synthetic_aom_global_simpls_sg_cv_v1_operator_scores, sizeof(synthetic_aom_global_simpls_sg_cv_v1_operator_scores) / sizeof(double), false},
        MatrixRef{6, 3, synthetic_aom_global_simpls_sg_cv_v1_rmse_curves, sizeof(synthetic_aom_global_simpls_sg_cv_v1_rmse_curves) / sizeof(double), false},
        MatrixRef{10, 1, synthetic_aom_global_simpls_sg_cv_v1_predictions, sizeof(synthetic_aom_global_simpls_sg_cv_v1_predictions) / sizeof(double), false},
        AomSelectionIndexRef{synthetic_aom_global_simpls_sg_cv_v1_operator_kinds, sizeof(synthetic_aom_global_simpls_sg_cv_v1_operator_kinds) / sizeof(std::int64_t)},
        AomSelectionIndexRef{synthetic_aom_global_simpls_sg_cv_v1_operator_param_offsets, sizeof(synthetic_aom_global_simpls_sg_cv_v1_operator_param_offsets) / sizeof(std::int64_t)},
        AomSelectionIndexRef{synthetic_aom_global_simpls_sg_cv_v1_fold_train_offsets, sizeof(synthetic_aom_global_simpls_sg_cv_v1_fold_train_offsets) / sizeof(std::int64_t)},
        AomSelectionIndexRef{synthetic_aom_global_simpls_sg_cv_v1_fold_train_indices, sizeof(synthetic_aom_global_simpls_sg_cv_v1_fold_train_indices) / sizeof(std::int64_t)},
        AomSelectionIndexRef{synthetic_aom_global_simpls_sg_cv_v1_fold_test_offsets, sizeof(synthetic_aom_global_simpls_sg_cv_v1_fold_test_offsets) / sizeof(std::int64_t)},
        AomSelectionIndexRef{synthetic_aom_global_simpls_sg_cv_v1_fold_test_indices, sizeof(synthetic_aom_global_simpls_sg_cv_v1_fold_test_indices) / sizeof(std::int64_t)}
    },
    {
        "synthetic_aom_global_simpls_fck_whittaker_cv_v1",
        3,
        4,
        3,
        0,
        3,
        0.014704524576819429,
        MatrixRef{11, 9, synthetic_aom_global_simpls_fck_whittaker_cv_v1_X, sizeof(synthetic_aom_global_simpls_fck_whittaker_cv_v1_X) / sizeof(double), false},
        MatrixRef{11, 1, synthetic_aom_global_simpls_fck_whittaker_cv_v1_Y, sizeof(synthetic_aom_global_simpls_fck_whittaker_cv_v1_Y) / sizeof(double), false},
        MatrixRef{1, 6, synthetic_aom_global_simpls_fck_whittaker_cv_v1_operator_params, sizeof(synthetic_aom_global_simpls_fck_whittaker_cv_v1_operator_params) / sizeof(double), false},
        MatrixRef{1, 4, synthetic_aom_global_simpls_fck_whittaker_cv_v1_operator_scores, sizeof(synthetic_aom_global_simpls_fck_whittaker_cv_v1_operator_scores) / sizeof(double), false},
        MatrixRef{4, 3, synthetic_aom_global_simpls_fck_whittaker_cv_v1_rmse_curves, sizeof(synthetic_aom_global_simpls_fck_whittaker_cv_v1_rmse_curves) / sizeof(double), false},
        MatrixRef{11, 1, synthetic_aom_global_simpls_fck_whittaker_cv_v1_predictions, sizeof(synthetic_aom_global_simpls_fck_whittaker_cv_v1_predictions) / sizeof(double), false},
        AomSelectionIndexRef{synthetic_aom_global_simpls_fck_whittaker_cv_v1_operator_kinds, sizeof(synthetic_aom_global_simpls_fck_whittaker_cv_v1_operator_kinds) / sizeof(std::int64_t)},
        AomSelectionIndexRef{synthetic_aom_global_simpls_fck_whittaker_cv_v1_operator_param_offsets, sizeof(synthetic_aom_global_simpls_fck_whittaker_cv_v1_operator_param_offsets) / sizeof(std::int64_t)},
        AomSelectionIndexRef{synthetic_aom_global_simpls_fck_whittaker_cv_v1_fold_train_offsets, sizeof(synthetic_aom_global_simpls_fck_whittaker_cv_v1_fold_train_offsets) / sizeof(std::int64_t)},
        AomSelectionIndexRef{synthetic_aom_global_simpls_fck_whittaker_cv_v1_fold_train_indices, sizeof(synthetic_aom_global_simpls_fck_whittaker_cv_v1_fold_train_indices) / sizeof(std::int64_t)},
        AomSelectionIndexRef{synthetic_aom_global_simpls_fck_whittaker_cv_v1_fold_test_offsets, sizeof(synthetic_aom_global_simpls_fck_whittaker_cv_v1_fold_test_offsets) / sizeof(std::int64_t)},
        AomSelectionIndexRef{synthetic_aom_global_simpls_fck_whittaker_cv_v1_fold_test_indices, sizeof(synthetic_aom_global_simpls_fck_whittaker_cv_v1_fold_test_indices) / sizeof(std::int64_t)}
    }
};

}  // namespace pls4all::test::fixtures
