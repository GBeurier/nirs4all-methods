// SPDX-License-Identifier: CECILL-2.1
// Generated mechanically from parity/fixtures/*advanced-validation*.json.
#pragma once

#include <cstddef>
#include <cstdint>

namespace n4m::test::fixtures {

struct IndexRef {
    const std::int64_t* values;
    std::size_t size;
};

struct MatrixRef {
    std::int64_t rows;
    std::int64_t cols;
    const double* values;
    std::size_t size;
    bool sign_invariant;
};

struct AdvancedValidationFixture {
    const char* id;
    const char* kind;
    std::int64_t n_samples;
    std::int32_t n_splits;
    std::int32_t n_repeats;
    std::int64_t test_count;
    std::int64_t train_count;
    std::uint64_t seed;
    IndexRef fold_ids;
    MatrixRef X;
    MatrixRef Y;
    IndexRef train_offsets;
    IndexRef train_indices;
    IndexRef test_offsets;
    IndexRef test_indices;
};

inline const std::int64_t synthetic_validation_external_folds_v1_fold_ids[] = {
    0, 1, 2, 0, 1, 2, 0, 1,
    2,
};

inline const std::int64_t synthetic_validation_external_folds_v1_train_offsets[] = {
    0, 6, 12, 18,
};

inline const std::int64_t synthetic_validation_external_folds_v1_train_indices[] = {
    1, 2, 4, 5, 7, 8, 0, 2,
    3, 5, 6, 8, 0, 1, 3, 4,
    6, 7,
};

inline const std::int64_t synthetic_validation_external_folds_v1_test_offsets[] = {
    0, 3, 6, 9,
};

inline const std::int64_t synthetic_validation_external_folds_v1_test_indices[] = {
    0, 3, 6, 1, 4, 7, 2, 5,
    8,
};

inline const std::int64_t synthetic_validation_repeated_kfold_v1_train_offsets[] = {
    0, 8, 16, 24, 33, 41, 49, 57,
    66,
};

inline const std::int64_t synthetic_validation_repeated_kfold_v1_train_indices[] = {
    0, 1, 2, 4, 5, 6, 8, 10,
    1, 2, 3, 5, 6, 7, 8, 9,
    0, 3, 4, 5, 7, 8, 9, 10,
    0, 1, 2, 3, 4, 6, 7, 9,
    10, 0, 1, 2, 3, 5, 7, 8,
    9, 2, 4, 5, 6, 7, 8, 9,
    10, 0, 1, 3, 4, 6, 8, 9,
    10, 0, 1, 2, 3, 4, 5, 6,
    7, 10,
};

inline const std::int64_t synthetic_validation_repeated_kfold_v1_test_offsets[] = {
    0, 3, 6, 9, 11, 14, 17, 20,
    22,
};

inline const std::int64_t synthetic_validation_repeated_kfold_v1_test_indices[] = {
    9, 3, 7, 4, 0, 10, 6, 1,
    2, 8, 5, 4, 6, 10, 3, 0,
    1, 7, 5, 2, 8, 9,
};

inline const std::int64_t synthetic_validation_monte_carlo_v1_train_offsets[] = {
    0, 9, 18, 27, 36,
};

inline const std::int64_t synthetic_validation_monte_carlo_v1_train_indices[] = {
    0, 1, 2, 5, 6, 7, 8, 10,
    11, 0, 1, 2, 5, 6, 7, 8,
    9, 11, 0, 2, 3, 4, 5, 6,
    8, 9, 11, 0, 4, 5, 6, 7,
    8, 9, 10, 11,
};

inline const std::int64_t synthetic_validation_monte_carlo_v1_test_offsets[] = {
    0, 3, 6, 9, 12,
};

inline const std::int64_t synthetic_validation_monte_carlo_v1_test_indices[] = {
    3, 4, 9, 3, 4, 10, 7, 1,
    10, 2, 3, 1,
};

inline const double synthetic_validation_kennard_stone_v1_X[] = {
    0.75725659234499976, -0.87938579173569931, -1, 0.0053353506256878583,
    1.1652779879630932, -0.17283653939443672, -0.77777777777777779, -0.021851187265602479,
    0.32188382858030362, -0.61589896993975746, -0.55555555555555558, 0.071780410722898605,
    -0.80082720964307752, 0.91796050290479703, -0.33333333333333337, -0.089346104557816095,
    0.43209563405215201, 0.43176856986818435, -0.11111111111111116, 0.030021478689198296,
    1.6622959419905048, -0.36369500397678717, 0.11111111111111116, -0.014478874789290533,
    -0.96569637979966794, 0.13341579102966028, 0.33333333333333326, -0.0089140599203340089,
    1.2821688253349417, 0.23109846042492643, 0.55555555555555536, -0.026934503027655578,
    1.0986112869159821, -0.31804163912492178, 0.77777777777777768, 0.072367510583831993,
    -0.88150098336954219, -0.079052259227060218, 1, -0.032834646669964898,
};

inline const std::int64_t synthetic_validation_kennard_stone_v1_train_offsets[] = {
    0, 6,
};

inline const std::int64_t synthetic_validation_kennard_stone_v1_train_indices[] = {
    3, 5, 9, 0, 4, 8,
};

inline const std::int64_t synthetic_validation_kennard_stone_v1_test_offsets[] = {
    0, 4,
};

inline const std::int64_t synthetic_validation_kennard_stone_v1_test_indices[] = {
    1, 2, 6, 7,
};

inline const double synthetic_validation_spxy_v1_X[] = {
    0.88287156551382373, -0.80597398098363893, 0.43163402263945028, 1,
    0.20300151018303497, -0.14937126841641579, 0.15409717050812269, 0.95949297361449737,
    0.330956059782127, -0.53274501409180608, 0.088539419713128759, 0.84125353283118121,
    -1.3414629564844529, 0.67089120446321715, -0.93371766787839861, 0.6548607339452851,
    -0.97376842751281711, 0.69814352434294791, -0.21244704827610544, 0.41541501300188644,
    -0.39675388717846855, 0.016360668078737373, 0.033881620272372265, 0.14231483827328512,
    -0.99719766220943229, 1.5363320662020112, -0.92682878188567819, -0.142314838273285,
    -0.016564896064257414, -0.0029167698029344061, -0.17944803216431424, -0.41541501300188632,
    -0.55764270508352498, -0.30322110379353356, -0.56507936151294591, -0.65486073394528499,
    0.35694506511907087, -0.45460459342628678, 0.45561922655786424, -0.84125353283118109,
    0.60580740581100678, 0.074842030886373451, 0.39337244752610168, -0.95949297361449737,
    0.89054756314930517, -0.76588494001940721, 0.15040590420462455, -1,
};

inline const double synthetic_validation_spxy_v1_Y[] = {
    1.1502061406388746, -0.61115965577230735, 0.20668065039321537, -0.20971168435052512,
    0.65401197184967241, -0.15642598240162645, -1.1584690231000081, 1.2389530617651245,
    -1.2563594031393281, 0.30225338357396159, -0.36197759025020348, -0.066280396640831185,
    -1.5652941766998651, 1.3432034403252437, 0.083761729416266362, 0.23778936446852014,
    0.076671956811009551, 0.68782209805729, 0.41851177251735233, -0.63550368898767307,
    0.23681602156270976, -0.48097815316205633, 1.2739093592465833, -0.23173010096963859,
};

inline const std::int64_t synthetic_validation_spxy_v1_train_offsets[] = {
    0, 7,
};

inline const std::int64_t synthetic_validation_spxy_v1_train_indices[] = {
    0, 6, 8, 11, 4, 1, 10,
};

inline const std::int64_t synthetic_validation_spxy_v1_test_offsets[] = {
    0, 5,
};

inline const std::int64_t synthetic_validation_spxy_v1_test_indices[] = {
    2, 3, 5, 7, 9,
};

inline const AdvancedValidationFixture kAdvancedValidationFixtures[] = {
    {
        "synthetic_validation_external_folds_v1",
        "external",
        9,
        3,
        0,
        0,
        0,
        0ULL,
        IndexRef{synthetic_validation_external_folds_v1_fold_ids, sizeof(synthetic_validation_external_folds_v1_fold_ids) / sizeof(std::int64_t)},
        MatrixRef{0, 0, nullptr, 0, false},
        MatrixRef{0, 0, nullptr, 0, false},
        IndexRef{synthetic_validation_external_folds_v1_train_offsets, sizeof(synthetic_validation_external_folds_v1_train_offsets) / sizeof(std::int64_t)},
        IndexRef{synthetic_validation_external_folds_v1_train_indices, sizeof(synthetic_validation_external_folds_v1_train_indices) / sizeof(std::int64_t)},
        IndexRef{synthetic_validation_external_folds_v1_test_offsets, sizeof(synthetic_validation_external_folds_v1_test_offsets) / sizeof(std::int64_t)},
        IndexRef{synthetic_validation_external_folds_v1_test_indices, sizeof(synthetic_validation_external_folds_v1_test_indices) / sizeof(std::int64_t)}
    },
    {
        "synthetic_validation_repeated_kfold_v1",
        "repeated_kfold",
        11,
        4,
        2,
        0,
        0,
        91ULL,
        IndexRef{nullptr, 0},
        MatrixRef{0, 0, nullptr, 0, false},
        MatrixRef{0, 0, nullptr, 0, false},
        IndexRef{synthetic_validation_repeated_kfold_v1_train_offsets, sizeof(synthetic_validation_repeated_kfold_v1_train_offsets) / sizeof(std::int64_t)},
        IndexRef{synthetic_validation_repeated_kfold_v1_train_indices, sizeof(synthetic_validation_repeated_kfold_v1_train_indices) / sizeof(std::int64_t)},
        IndexRef{synthetic_validation_repeated_kfold_v1_test_offsets, sizeof(synthetic_validation_repeated_kfold_v1_test_offsets) / sizeof(std::int64_t)},
        IndexRef{synthetic_validation_repeated_kfold_v1_test_indices, sizeof(synthetic_validation_repeated_kfold_v1_test_indices) / sizeof(std::int64_t)}
    },
    {
        "synthetic_validation_monte_carlo_v1",
        "monte_carlo",
        12,
        0,
        4,
        3,
        0,
        92ULL,
        IndexRef{nullptr, 0},
        MatrixRef{0, 0, nullptr, 0, false},
        MatrixRef{0, 0, nullptr, 0, false},
        IndexRef{synthetic_validation_monte_carlo_v1_train_offsets, sizeof(synthetic_validation_monte_carlo_v1_train_offsets) / sizeof(std::int64_t)},
        IndexRef{synthetic_validation_monte_carlo_v1_train_indices, sizeof(synthetic_validation_monte_carlo_v1_train_indices) / sizeof(std::int64_t)},
        IndexRef{synthetic_validation_monte_carlo_v1_test_offsets, sizeof(synthetic_validation_monte_carlo_v1_test_offsets) / sizeof(std::int64_t)},
        IndexRef{synthetic_validation_monte_carlo_v1_test_indices, sizeof(synthetic_validation_monte_carlo_v1_test_indices) / sizeof(std::int64_t)}
    },
    {
        "synthetic_validation_kennard_stone_v1",
        "kennard_stone",
        10,
        0,
        0,
        0,
        6,
        93ULL,
        IndexRef{nullptr, 0},
        MatrixRef{10, 4, synthetic_validation_kennard_stone_v1_X, sizeof(synthetic_validation_kennard_stone_v1_X) / sizeof(double), false},
        MatrixRef{0, 0, nullptr, 0, false},
        IndexRef{synthetic_validation_kennard_stone_v1_train_offsets, sizeof(synthetic_validation_kennard_stone_v1_train_offsets) / sizeof(std::int64_t)},
        IndexRef{synthetic_validation_kennard_stone_v1_train_indices, sizeof(synthetic_validation_kennard_stone_v1_train_indices) / sizeof(std::int64_t)},
        IndexRef{synthetic_validation_kennard_stone_v1_test_offsets, sizeof(synthetic_validation_kennard_stone_v1_test_offsets) / sizeof(std::int64_t)},
        IndexRef{synthetic_validation_kennard_stone_v1_test_indices, sizeof(synthetic_validation_kennard_stone_v1_test_indices) / sizeof(std::int64_t)}
    },
    {
        "synthetic_validation_spxy_v1",
        "spxy",
        12,
        0,
        0,
        0,
        7,
        94ULL,
        IndexRef{nullptr, 0},
        MatrixRef{12, 4, synthetic_validation_spxy_v1_X, sizeof(synthetic_validation_spxy_v1_X) / sizeof(double), false},
        MatrixRef{12, 2, synthetic_validation_spxy_v1_Y, sizeof(synthetic_validation_spxy_v1_Y) / sizeof(double), false},
        IndexRef{synthetic_validation_spxy_v1_train_offsets, sizeof(synthetic_validation_spxy_v1_train_offsets) / sizeof(std::int64_t)},
        IndexRef{synthetic_validation_spxy_v1_train_indices, sizeof(synthetic_validation_spxy_v1_train_indices) / sizeof(std::int64_t)},
        IndexRef{synthetic_validation_spxy_v1_test_offsets, sizeof(synthetic_validation_spxy_v1_test_offsets) / sizeof(std::int64_t)},
        IndexRef{synthetic_validation_spxy_v1_test_indices, sizeof(synthetic_validation_spxy_v1_test_indices) / sizeof(std::int64_t)}
    }
};

}  // namespace n4m::test::fixtures
