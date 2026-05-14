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

inline const PipelineFixture kPipelineFixtures[] = {
    {
        "synthetic_pipeline_identity_v1",
        synthetic_pipeline_identity_v1_operators,
        sizeof(synthetic_pipeline_identity_v1_operators) / sizeof(p4a_operator_kind_t),
        MatrixRef{4, 3, synthetic_pipeline_identity_v1_x, sizeof(synthetic_pipeline_identity_v1_x) / sizeof(double), false},
        MatrixRef{4, 3, synthetic_pipeline_identity_v1_transform_train, sizeof(synthetic_pipeline_identity_v1_transform_train) / sizeof(double), false}
    },
    {
        "synthetic_pipeline_center_v1",
        synthetic_pipeline_center_v1_operators,
        sizeof(synthetic_pipeline_center_v1_operators) / sizeof(p4a_operator_kind_t),
        MatrixRef{4, 3, synthetic_pipeline_center_v1_x, sizeof(synthetic_pipeline_center_v1_x) / sizeof(double), false},
        MatrixRef{4, 3, synthetic_pipeline_center_v1_transform_train, sizeof(synthetic_pipeline_center_v1_transform_train) / sizeof(double), false}
    },
    {
        "synthetic_pipeline_autoscale_v1",
        synthetic_pipeline_autoscale_v1_operators,
        sizeof(synthetic_pipeline_autoscale_v1_operators) / sizeof(p4a_operator_kind_t),
        MatrixRef{4, 3, synthetic_pipeline_autoscale_v1_x, sizeof(synthetic_pipeline_autoscale_v1_x) / sizeof(double), false},
        MatrixRef{4, 3, synthetic_pipeline_autoscale_v1_transform_train, sizeof(synthetic_pipeline_autoscale_v1_transform_train) / sizeof(double), false}
    },
    {
        "synthetic_pipeline_pareto_v1",
        synthetic_pipeline_pareto_v1_operators,
        sizeof(synthetic_pipeline_pareto_v1_operators) / sizeof(p4a_operator_kind_t),
        MatrixRef{4, 3, synthetic_pipeline_pareto_v1_x, sizeof(synthetic_pipeline_pareto_v1_x) / sizeof(double), false},
        MatrixRef{4, 3, synthetic_pipeline_pareto_v1_transform_train, sizeof(synthetic_pipeline_pareto_v1_transform_train) / sizeof(double), false}
    },
    {
        "synthetic_pipeline_snv_v1",
        synthetic_pipeline_snv_v1_operators,
        sizeof(synthetic_pipeline_snv_v1_operators) / sizeof(p4a_operator_kind_t),
        MatrixRef{3, 3, synthetic_pipeline_snv_v1_x, sizeof(synthetic_pipeline_snv_v1_x) / sizeof(double), false},
        MatrixRef{3, 3, synthetic_pipeline_snv_v1_transform_train, sizeof(synthetic_pipeline_snv_v1_transform_train) / sizeof(double), false}
    }
};

}  // namespace pls4all::test::fixtures
