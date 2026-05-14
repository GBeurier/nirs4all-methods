// SPDX-License-Identifier: CeCILL-2.1
// Generated mechanically from parity/fixtures/*validation*.json.
#pragma once

#include <cstddef>
#include <cstdint>

namespace pls4all::test::fixtures {

struct IndexRef {
    const std::int64_t* values;
    std::size_t size;
};

struct ValidationFixture {
    const char* id;
    const char* kind;
    std::int64_t n_samples;
    std::int32_t n_splits;
    std::int64_t holdout_start;
    std::int64_t holdout_count;
    IndexRef train_offsets;
    IndexRef train_indices;
    IndexRef test_offsets;
    IndexRef test_indices;
};

inline const std::int64_t synthetic_validation_kfold_balanced_v1_train_offsets[] = {
    0, 6, 13, 20,
};

inline const std::int64_t synthetic_validation_kfold_balanced_v1_train_indices[] = {
    4, 5, 6, 7, 8, 9, 0, 1,
    2, 3, 7, 8, 9, 0, 1, 2,
    3, 4, 5, 6,
};

inline const std::int64_t synthetic_validation_kfold_balanced_v1_test_offsets[] = {
    0, 4, 7, 10,
};

inline const std::int64_t synthetic_validation_kfold_balanced_v1_test_indices[] = {
    0, 1, 2, 3, 4, 5, 6, 7,
    8, 9,
};

inline const std::int64_t synthetic_validation_leave_one_out_v1_train_offsets[] = {
    0, 4, 8, 12, 16, 20,
};

inline const std::int64_t synthetic_validation_leave_one_out_v1_train_indices[] = {
    1, 2, 3, 4, 0, 2, 3, 4,
    0, 1, 3, 4, 0, 1, 2, 4,
    0, 1, 2, 3,
};

inline const std::int64_t synthetic_validation_leave_one_out_v1_test_offsets[] = {
    0, 1, 2, 3, 4, 5,
};

inline const std::int64_t synthetic_validation_leave_one_out_v1_test_indices[] = {
    0, 1, 2, 3, 4,
};

inline const std::int64_t synthetic_validation_holdout_v1_train_offsets[] = {
    0, 5,
};

inline const std::int64_t synthetic_validation_holdout_v1_train_indices[] = {
    0, 1, 5, 6, 7,
};

inline const std::int64_t synthetic_validation_holdout_v1_test_offsets[] = {
    0, 3,
};

inline const std::int64_t synthetic_validation_holdout_v1_test_indices[] = {
    2, 3, 4,
};

inline const ValidationFixture kValidationFixtures[] = {
    {
        "synthetic_validation_kfold_balanced_v1",
        "kfold",
        10,
        3,
        0,
        0,
        IndexRef{synthetic_validation_kfold_balanced_v1_train_offsets, sizeof(synthetic_validation_kfold_balanced_v1_train_offsets) / sizeof(std::int64_t)},
        IndexRef{synthetic_validation_kfold_balanced_v1_train_indices, sizeof(synthetic_validation_kfold_balanced_v1_train_indices) / sizeof(std::int64_t)},
        IndexRef{synthetic_validation_kfold_balanced_v1_test_offsets, sizeof(synthetic_validation_kfold_balanced_v1_test_offsets) / sizeof(std::int64_t)},
        IndexRef{synthetic_validation_kfold_balanced_v1_test_indices, sizeof(synthetic_validation_kfold_balanced_v1_test_indices) / sizeof(std::int64_t)}
    },
    {
        "synthetic_validation_leave_one_out_v1",
        "leave_one_out",
        5,
        0,
        0,
        0,
        IndexRef{synthetic_validation_leave_one_out_v1_train_offsets, sizeof(synthetic_validation_leave_one_out_v1_train_offsets) / sizeof(std::int64_t)},
        IndexRef{synthetic_validation_leave_one_out_v1_train_indices, sizeof(synthetic_validation_leave_one_out_v1_train_indices) / sizeof(std::int64_t)},
        IndexRef{synthetic_validation_leave_one_out_v1_test_offsets, sizeof(synthetic_validation_leave_one_out_v1_test_offsets) / sizeof(std::int64_t)},
        IndexRef{synthetic_validation_leave_one_out_v1_test_indices, sizeof(synthetic_validation_leave_one_out_v1_test_indices) / sizeof(std::int64_t)}
    },
    {
        "synthetic_validation_holdout_v1",
        "holdout",
        8,
        0,
        2,
        3,
        IndexRef{synthetic_validation_holdout_v1_train_offsets, sizeof(synthetic_validation_holdout_v1_train_offsets) / sizeof(std::int64_t)},
        IndexRef{synthetic_validation_holdout_v1_train_indices, sizeof(synthetic_validation_holdout_v1_train_indices) / sizeof(std::int64_t)},
        IndexRef{synthetic_validation_holdout_v1_test_offsets, sizeof(synthetic_validation_holdout_v1_test_offsets) / sizeof(std::int64_t)},
        IndexRef{synthetic_validation_holdout_v1_test_indices, sizeof(synthetic_validation_holdout_v1_test_indices) / sizeof(std::int64_t)}
    }
};

}  // namespace pls4all::test::fixtures
