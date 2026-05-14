// SPDX-License-Identifier: CeCILL-2.1
// Generated mechanically from parity/fixtures/*pipeline*.json.
#pragma once

#include <cstddef>
#include <cstdint>

#include "pls4all/p4a.h"
#include "phase1_fixtures.hpp"

namespace pls4all::test::fixtures {

struct PipelineFixture {
    const char* id;
    const p4a_operator_kind_t* operators;
    std::size_t n_operators;
    const double* params;
    const std::int32_t* n_params;
    MatrixRef X;
    MatrixRef transform_train;
};

inline const double synthetic_pipeline_identity_v1_x[] = {
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 2, 4, 8,
};

inline const double synthetic_pipeline_identity_v1_transform_train[] = {
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 2, 4, 8,
};

inline const p4a_operator_kind_t synthetic_pipeline_identity_v1_operators[] = {
    P4A_OP_IDENTITY,
};

inline const double synthetic_pipeline_identity_v1_params[] = {
    0.0,
};

inline const std::int32_t synthetic_pipeline_identity_v1_n_params[] = {
    0,
};

inline const double synthetic_pipeline_center_v1_x[] = {
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 2, 4, 8,
};

inline const double synthetic_pipeline_center_v1_transform_train[] = {
    -2.5, -2.75, -3.5, 0.5,
    0.25, -0.5, 3.5, 3.25,
    2.5, -1.5, -0.75, 1.5,
};

inline const p4a_operator_kind_t synthetic_pipeline_center_v1_operators[] = {
    P4A_OP_CENTER,
};

inline const double synthetic_pipeline_center_v1_params[] = {
    0.0,
};

inline const std::int32_t synthetic_pipeline_center_v1_n_params[] = {
    0,
};

inline const double synthetic_pipeline_autoscale_v1_x[] = {
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 2, 4, 8,
};

inline const double synthetic_pipeline_autoscale_v1_transform_train[] = {
    -0.94491118252306805, -1.1000000000000001, -1.3228756555322951, 0.1889822365046136,
    0.10000000000000001, -0.1889822365046136, 1.3228756555322951, 1.3,
    0.94491118252306805, -0.56694670951384085, -0.29999999999999999, 0.56694670951384085,
};

inline const p4a_operator_kind_t synthetic_pipeline_autoscale_v1_operators[] = {
    P4A_OP_AUTOSCALE,
};

inline const double synthetic_pipeline_autoscale_v1_params[] = {
    0.0,
};

inline const std::int32_t synthetic_pipeline_autoscale_v1_n_params[] = {
    0,
};

inline const double synthetic_pipeline_pareto_v1_x[] = {
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 2, 4, 8,
};

inline const double synthetic_pipeline_pareto_v1_transform_train[] = {
    -1.5369703823781609, -1.7392527130926085, -2.1517585353294253, 0.30739407647563216,
    0.15811388300841897, -0.30739407647563216, 2.1517585353294253, 2.0554804791094465,
    1.5369703823781609, -0.92218222942689643, -0.47434164902525688, 0.92218222942689643,
};

inline const p4a_operator_kind_t synthetic_pipeline_pareto_v1_operators[] = {
    P4A_OP_PARETO_SCALE,
};

inline const double synthetic_pipeline_pareto_v1_params[] = {
    0.0,
};

inline const std::int32_t synthetic_pipeline_pareto_v1_n_params[] = {
    0,
};

inline const double synthetic_pipeline_snv_v1_x[] = {
    1, 2, 3, 2,
    2, 2, 3, 5,
    7,
};

inline const double synthetic_pipeline_snv_v1_transform_train[] = {
    -1, 0, 1, 0,
    0, 0, -1, 0,
    1,
};

inline const p4a_operator_kind_t synthetic_pipeline_snv_v1_operators[] = {
    P4A_OP_SNV,
};

inline const double synthetic_pipeline_snv_v1_params[] = {
    0.0,
};

inline const std::int32_t synthetic_pipeline_snv_v1_n_params[] = {
    0,
};

inline const double synthetic_pipeline_msc_v1_x[] = {
    1.3500000000000001, 2.4500000000000002, 4.6500000000000004, 7.9500000000000011,
    12.350000000000001, 0.46999999999999997, 1.29, 2.9299999999999997,
    5.3899999999999997, 8.6699999999999999, 2.25, 3.6000000000000001,
    6.3000000000000007, 10.350000000000001, 15.750000000000002, 1.0700000000000001,
    1.6899999999999999, 2.9300000000000002, 4.79, 7.2700000000000005,
};

inline const double synthetic_pipeline_msc_v1_transform_train[] = {
    1.2849999999999995, 2.2574999999999998, 4.2024999999999997, 7.1200000000000019,
    11.010000000000002, 1.2850000000000006, 2.2575000000000007, 4.2025000000000006,
    7.120000000000001, 11.010000000000002, 1.2849999999999995, 2.2574999999999998,
    4.2025000000000006, 7.1200000000000019, 11.010000000000003, 1.285000000000001,
    2.2575000000000007, 4.2025000000000015, 7.120000000000001, 11.010000000000002,
};

inline const p4a_operator_kind_t synthetic_pipeline_msc_v1_operators[] = {
    P4A_OP_MSC,
};

inline const double synthetic_pipeline_msc_v1_params[] = {
    0.0,
};

inline const std::int32_t synthetic_pipeline_msc_v1_n_params[] = {
    0,
};

inline const double synthetic_pipeline_detrend_poly_v1_x[] = {
    1.52, 1.0680000000000001, 1.002, 1.0720000000000001,
    1.478, 2.1000000000000001, -1.3100000000000001, -0.86199999999999999,
    -0.56800000000000006, -0.2279999999999999, -0.082000000000000003, 0.10999999999999992,
    1.1000000000000001, 0.66400000000000003, 0.28600000000000003, 0.12599999999999995,
    0.034000000000000009, 0.12000000000000004,
};

inline const double synthetic_pipeline_detrend_poly_v1_transform_train[] = {
    0.014642857142856736, -0.038928571428571646, 0.034285714285714253, -0.01571428571428557,
    0.011071428571428621, -0.0053571428571426161, -0.010000000000000231, 0.024857142857142689,
    -0.027428571428571802, 0.033142857142856946, -0.033428571428571606, 0.012857142857142775,
    -0.0082142857142857295, 0.022499999999999964, -0.022857142857142798, 0.015714285714285722,
    -0.011785714285714274, 0.0046428571428571014,
};

inline const p4a_operator_kind_t synthetic_pipeline_detrend_poly_v1_operators[] = {
    P4A_OP_DETREND_POLY,
};

inline const double synthetic_pipeline_detrend_poly_v1_params[] = {
    2,
};

inline const std::int32_t synthetic_pipeline_detrend_poly_v1_n_params[] = {
    1,
};

inline const PipelineFixture kPipelineFixtures[] = {
    {
        "synthetic_pipeline_identity_v1",
        synthetic_pipeline_identity_v1_operators,
        sizeof(synthetic_pipeline_identity_v1_operators) / sizeof(p4a_operator_kind_t),
        synthetic_pipeline_identity_v1_params,
        synthetic_pipeline_identity_v1_n_params,
        MatrixRef{4, 3, synthetic_pipeline_identity_v1_x, sizeof(synthetic_pipeline_identity_v1_x) / sizeof(double), false},
        MatrixRef{4, 3, synthetic_pipeline_identity_v1_transform_train, sizeof(synthetic_pipeline_identity_v1_transform_train) / sizeof(double), false}
    },
    {
        "synthetic_pipeline_center_v1",
        synthetic_pipeline_center_v1_operators,
        sizeof(synthetic_pipeline_center_v1_operators) / sizeof(p4a_operator_kind_t),
        synthetic_pipeline_center_v1_params,
        synthetic_pipeline_center_v1_n_params,
        MatrixRef{4, 3, synthetic_pipeline_center_v1_x, sizeof(synthetic_pipeline_center_v1_x) / sizeof(double), false},
        MatrixRef{4, 3, synthetic_pipeline_center_v1_transform_train, sizeof(synthetic_pipeline_center_v1_transform_train) / sizeof(double), false}
    },
    {
        "synthetic_pipeline_autoscale_v1",
        synthetic_pipeline_autoscale_v1_operators,
        sizeof(synthetic_pipeline_autoscale_v1_operators) / sizeof(p4a_operator_kind_t),
        synthetic_pipeline_autoscale_v1_params,
        synthetic_pipeline_autoscale_v1_n_params,
        MatrixRef{4, 3, synthetic_pipeline_autoscale_v1_x, sizeof(synthetic_pipeline_autoscale_v1_x) / sizeof(double), false},
        MatrixRef{4, 3, synthetic_pipeline_autoscale_v1_transform_train, sizeof(synthetic_pipeline_autoscale_v1_transform_train) / sizeof(double), false}
    },
    {
        "synthetic_pipeline_pareto_v1",
        synthetic_pipeline_pareto_v1_operators,
        sizeof(synthetic_pipeline_pareto_v1_operators) / sizeof(p4a_operator_kind_t),
        synthetic_pipeline_pareto_v1_params,
        synthetic_pipeline_pareto_v1_n_params,
        MatrixRef{4, 3, synthetic_pipeline_pareto_v1_x, sizeof(synthetic_pipeline_pareto_v1_x) / sizeof(double), false},
        MatrixRef{4, 3, synthetic_pipeline_pareto_v1_transform_train, sizeof(synthetic_pipeline_pareto_v1_transform_train) / sizeof(double), false}
    },
    {
        "synthetic_pipeline_snv_v1",
        synthetic_pipeline_snv_v1_operators,
        sizeof(synthetic_pipeline_snv_v1_operators) / sizeof(p4a_operator_kind_t),
        synthetic_pipeline_snv_v1_params,
        synthetic_pipeline_snv_v1_n_params,
        MatrixRef{3, 3, synthetic_pipeline_snv_v1_x, sizeof(synthetic_pipeline_snv_v1_x) / sizeof(double), false},
        MatrixRef{3, 3, synthetic_pipeline_snv_v1_transform_train, sizeof(synthetic_pipeline_snv_v1_transform_train) / sizeof(double), false}
    },
    {
        "synthetic_pipeline_msc_v1",
        synthetic_pipeline_msc_v1_operators,
        sizeof(synthetic_pipeline_msc_v1_operators) / sizeof(p4a_operator_kind_t),
        synthetic_pipeline_msc_v1_params,
        synthetic_pipeline_msc_v1_n_params,
        MatrixRef{4, 5, synthetic_pipeline_msc_v1_x, sizeof(synthetic_pipeline_msc_v1_x) / sizeof(double), false},
        MatrixRef{4, 5, synthetic_pipeline_msc_v1_transform_train, sizeof(synthetic_pipeline_msc_v1_transform_train) / sizeof(double), false}
    },
    {
        "synthetic_pipeline_detrend_poly_v1",
        synthetic_pipeline_detrend_poly_v1_operators,
        sizeof(synthetic_pipeline_detrend_poly_v1_operators) / sizeof(p4a_operator_kind_t),
        synthetic_pipeline_detrend_poly_v1_params,
        synthetic_pipeline_detrend_poly_v1_n_params,
        MatrixRef{3, 6, synthetic_pipeline_detrend_poly_v1_x, sizeof(synthetic_pipeline_detrend_poly_v1_x) / sizeof(double), false},
        MatrixRef{3, 6, synthetic_pipeline_detrend_poly_v1_transform_train, sizeof(synthetic_pipeline_detrend_poly_v1_transform_train) / sizeof(double), false}
    }
};

}  // namespace pls4all::test::fixtures
