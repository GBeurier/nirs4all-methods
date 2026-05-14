// SPDX-License-Identifier: CeCILL-2.1

#include "pls4all/p4a.h"
#include "harness.hpp"

TEST(matrix_view, rowmajor_init_sets_strides_correctly) {
    double buf[20];
    p4a_matrix_view_t v;
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&v, buf, 4, 5, P4A_DTYPE_F64), P4A_OK);
    CHECK_EQ(v.rows, 4);
    CHECK_EQ(v.cols, 5);
    CHECK_EQ(v.row_stride, 5);
    CHECK_EQ(v.col_stride, 1);
    CHECK_EQ(v.dtype, P4A_DTYPE_F64);
    CHECK_EQ(p4a_matrix_view_validate(&v), P4A_OK);
}

TEST(matrix_view, colmajor_init_sets_strides_correctly) {
    double buf[20];
    p4a_matrix_view_t v;
    CHECK_EQ(p4a_matrix_view_init_colmajor(&v, buf, 4, 5, P4A_DTYPE_F64), P4A_OK);
    CHECK_EQ(v.row_stride, 1);
    CHECK_EQ(v.col_stride, 4);
    CHECK_EQ(p4a_matrix_view_validate(&v), P4A_OK);
}

TEST(matrix_view, strided_init_accepts_transposed) {
    double buf[20];
    p4a_matrix_view_t v;
    // Transposed view of a 4x5 row-major buffer = 5x4 with strides (1, 5).
    CHECK_EQ(p4a_matrix_view_init_strided(&v, buf, 5, 4, 1, 5, P4A_DTYPE_F64),
              P4A_OK);
    CHECK_EQ(p4a_matrix_view_validate(&v), P4A_OK);
}

TEST(matrix_view, strided_init_accepts_slice) {
    double buf[40];
    p4a_matrix_view_t v;
    // Slice every other row in a 4x10 row-major buffer = 2x10 with row_stride=20.
    CHECK_EQ(p4a_matrix_view_init_strided(&v, buf, 2, 10, 20, 1, P4A_DTYPE_F64),
              P4A_OK);
    CHECK_EQ(p4a_matrix_view_validate(&v), P4A_OK);
}

TEST(matrix_view, empty_view_validates) {
    p4a_matrix_view_t v;
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&v, nullptr, 0, 0, P4A_DTYPE_F64),
              P4A_OK);
    CHECK_EQ(p4a_matrix_view_validate(&v), P4A_OK);
}

TEST(matrix_view, negative_dimensions_rejected) {
    double buf[20];
    p4a_matrix_view_t v;
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&v, buf, -1, 5, P4A_DTYPE_F64),
              P4A_ERR_INVALID_ARGUMENT);
    CHECK_EQ(p4a_matrix_view_init_colmajor(&v, buf, 4, -1, P4A_DTYPE_F64),
              P4A_ERR_INVALID_ARGUMENT);
}

TEST(matrix_view, negative_stride_rejected) {
    double buf[20];
    p4a_matrix_view_t v;
    CHECK_EQ(p4a_matrix_view_init_strided(&v, buf, 4, 5, -1, 1, P4A_DTYPE_F64),
              P4A_ERR_STRIDE_INVALID);
    CHECK_EQ(p4a_matrix_view_init_strided(&v, buf, 4, 5, 5, -1, P4A_DTYPE_F64),
              P4A_ERR_STRIDE_INVALID);
}

TEST(matrix_view, zero_stride_rejected_when_dim_gt_one) {
    double buf[20];
    p4a_matrix_view_t v;
    CHECK_EQ(p4a_matrix_view_init_strided(&v, buf, 4, 5, 0, 1, P4A_DTYPE_F64),
              P4A_ERR_STRIDE_INVALID);
    CHECK_EQ(p4a_matrix_view_init_strided(&v, buf, 4, 5, 5, 0, P4A_DTYPE_F64),
              P4A_ERR_STRIDE_INVALID);
}

TEST(matrix_view, zero_stride_ok_when_dim_le_one) {
    double buf[5];
    p4a_matrix_view_t v;
    CHECK_EQ(p4a_matrix_view_init_strided(&v, buf, 1, 5, 0, 1, P4A_DTYPE_F64),
              P4A_OK);
    CHECK_EQ(p4a_matrix_view_init_strided(&v, buf, 5, 1, 1, 0, P4A_DTYPE_F64),
              P4A_OK);
}

TEST(matrix_view, unknown_dtype_rejected) {
    double buf[20];
    p4a_matrix_view_t v;
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&v, buf, 4, 5, P4A_DTYPE_UNKNOWN),
              P4A_ERR_INVALID_ARGUMENT);
}

TEST(matrix_view, non_empty_with_null_data_rejected) {
    p4a_matrix_view_t v;
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&v, nullptr, 4, 5, P4A_DTYPE_F64),
              P4A_ERR_NULL_POINTER);
}

TEST(matrix_view, validate_null_rejected) {
    CHECK_EQ(p4a_matrix_view_validate(nullptr), P4A_ERR_NULL_POINTER);
}

TEST(matrix_view, init_null_out_rejected) {
    CHECK_EQ(p4a_matrix_view_init_rowmajor(nullptr, nullptr, 0, 0, P4A_DTYPE_F64),
              P4A_ERR_NULL_POINTER);
    CHECK_EQ(p4a_matrix_view_init_colmajor(nullptr, nullptr, 0, 0, P4A_DTYPE_F64),
              P4A_ERR_NULL_POINTER);
    CHECK_EQ(p4a_matrix_view_init_strided(nullptr, nullptr, 0, 0, 1, 1, P4A_DTYPE_F64),
              P4A_ERR_NULL_POINTER);
}
