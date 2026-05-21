// SPDX-License-Identifier: CECILL-2.1
// Generated mechanically from parity/fixtures/*aom-preprocessing*.json.
#pragma once

#include <cstddef>
#include <cstdint>

#include "phase1_fixtures.hpp"

namespace n4m::test::fixtures {

struct AomPreprocessingIndexRef {
    const std::int64_t* values;
    std::size_t size;
};

struct AomPreprocessingFixture {
    const char* id;
    std::int32_t gating_mode;
    std::int32_t n_operators;
    MatrixRef X;
    MatrixRef weights;
    MatrixRef operator_outputs;
    MatrixRef transformed;
    AomPreprocessingIndexRef operator_kinds;
};

inline const double synthetic_aom_soft_preprocessing_v1_X[] = {
    1, 2, 3, 5,
    2, 3.5, 5.5, 8,
    3, 5, 7, 10,
    4, 6.5, 8.5, 11.5,
    5, 8, 11, 14,
};

inline const double synthetic_aom_soft_preprocessing_v1_weights[] = {
    0.33333333333333331, 0.33333333333333331, 0.33333333333333331,
};

inline const double synthetic_aom_soft_preprocessing_v1_operator_outputs[] = {
    1, 2, 3, 5,
    2, 3.5, 5.5, 8,
    3, 5, 7, 10,
    4, 6.5, 8.5, 11.5,
    5, 8, 11, 14,
    -1.2649110640673518, -1.2649110640673518, -1.3241694217637887, -1.3740575635810122,
    -0.63245553203367588, -0.63245553203367588, -0.49656353316142077, -0.49699954427398302,
    0, 0, 0, 0.087705801930703126,
    0.63245553203367588, 0.63245553203367588, 0.49656353316142077, 0.5262348115842177,
    1.2649110640673518, 1.2649110640673518, 1.3241694217637887, 1.2571164943400754,
    -1.0246950765959599, -0.43915503282683993, 0.14638501094227999, 1.3174650984805198,
    -1.0584754935143139, -0.48112522432468813, 0.28867513459481287, 1.2509255832441892,
    -1.0883838657625979, -0.41860917913946072, 0.25116550748367644, 1.255827537418382,
    -1.142760089046696, -0.35464968280759529, 0.2758386421836852, 1.221571129670606,
    -1.1618950038622251, -0.3872983346207417, 0.3872983346207417, 1.1618950038622251,
};

inline const double synthetic_aom_soft_preprocessing_v1_transformed[] = {
    -0.42986871355443723, 0.098644634368602768, 0.60740519639283042, 1.6478025116331692,
    0.1030229914840034, 0.79547308121387861, 1.7640372004777971, 2.9179753463234022,
    0.63720537807913402, 1.5271302736201795, 2.417055169161225, 3.7811777797830279,
    1.1632318143289933, 2.2592686164086935, 3.0908007251150349, 4.4159353137516071,
    1.7010053534017087, 2.9592042431488697, 4.2371559187948433, 5.4730038327340989,
};

inline const std::int64_t synthetic_aom_soft_preprocessing_v1_operator_kinds[] = {
    0, 2, 4,
};

inline const double synthetic_aom_hard_preprocessing_v1_X[] = {
    1, 2, 3, 5,
    2, 3.5, 5.5, 8,
    3, 5, 7, 10,
    4, 6.5, 8.5, 11.5,
    5, 8, 11, 14,
};

inline const double synthetic_aom_hard_preprocessing_v1_weights[] = {
    1, 0, 0,
};

inline const double synthetic_aom_hard_preprocessing_v1_operator_outputs[] = {
    -2, -3, -4, -4.6999999999999993,
    -1, -1.5, -1.5, -1.6999999999999993,
    0, 0, 0, 0.30000000000000071,
    1, 1.5, 1.5, 1.8000000000000007,
    2, 3, 4, 4.3000000000000007,
    1, 2, 3, 5,
    2, 3.5, 5.5, 8,
    3, 5, 7, 10,
    4, 6.5, 8.5, 11.5,
    5, 8, 11, 14,
    -1.0246950765959599, -0.43915503282683993, 0.14638501094227999, 1.3174650984805198,
    -1.0584754935143139, -0.48112522432468813, 0.28867513459481287, 1.2509255832441892,
    -1.0883838657625979, -0.41860917913946072, 0.25116550748367644, 1.255827537418382,
    -1.142760089046696, -0.35464968280759529, 0.2758386421836852, 1.221571129670606,
    -1.1618950038622251, -0.3872983346207417, 0.3872983346207417, 1.1618950038622251,
};

inline const double synthetic_aom_hard_preprocessing_v1_transformed[] = {
    -2, -3, -4, -4.6999999999999993,
    -1, -1.5, -1.5, -1.6999999999999993,
    0, 0, 0, 0.30000000000000071,
    1, 1.5, 1.5, 1.8000000000000007,
    2, 3, 4, 4.3000000000000007,
};

inline const std::int64_t synthetic_aom_hard_preprocessing_v1_operator_kinds[] = {
    1, 0, 4,
};

inline const AomPreprocessingFixture kAomPreprocessingFixtures[] = {
    {
        "synthetic_aom_soft_preprocessing_v1",
        1,
        3,
        MatrixRef{5, 4, synthetic_aom_soft_preprocessing_v1_X, sizeof(synthetic_aom_soft_preprocessing_v1_X) / sizeof(double), false},
        MatrixRef{1, 3, synthetic_aom_soft_preprocessing_v1_weights, sizeof(synthetic_aom_soft_preprocessing_v1_weights) / sizeof(double), false},
        MatrixRef{3, 20, synthetic_aom_soft_preprocessing_v1_operator_outputs, sizeof(synthetic_aom_soft_preprocessing_v1_operator_outputs) / sizeof(double), false},
        MatrixRef{5, 4, synthetic_aom_soft_preprocessing_v1_transformed, sizeof(synthetic_aom_soft_preprocessing_v1_transformed) / sizeof(double), false},
        AomPreprocessingIndexRef{synthetic_aom_soft_preprocessing_v1_operator_kinds, sizeof(synthetic_aom_soft_preprocessing_v1_operator_kinds) / sizeof(std::int64_t)}
    },
    {
        "synthetic_aom_hard_preprocessing_v1",
        0,
        3,
        MatrixRef{5, 4, synthetic_aom_hard_preprocessing_v1_X, sizeof(synthetic_aom_hard_preprocessing_v1_X) / sizeof(double), false},
        MatrixRef{1, 3, synthetic_aom_hard_preprocessing_v1_weights, sizeof(synthetic_aom_hard_preprocessing_v1_weights) / sizeof(double), false},
        MatrixRef{3, 20, synthetic_aom_hard_preprocessing_v1_operator_outputs, sizeof(synthetic_aom_hard_preprocessing_v1_operator_outputs) / sizeof(double), false},
        MatrixRef{5, 4, synthetic_aom_hard_preprocessing_v1_transformed, sizeof(synthetic_aom_hard_preprocessing_v1_transformed) / sizeof(double), false},
        AomPreprocessingIndexRef{synthetic_aom_hard_preprocessing_v1_operator_kinds, sizeof(synthetic_aom_hard_preprocessing_v1_operator_kinds) / sizeof(std::int64_t)}
    }
};

}  // namespace n4m::test::fixtures
