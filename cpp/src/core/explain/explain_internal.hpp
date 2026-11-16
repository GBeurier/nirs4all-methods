// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
//
// Internal helpers and kernels behind the n4m/explain.h C ABI. Never installed.
// Every function validates its inputs before writing any caller buffer and
// reports failures as a status plus (when a context is given) a message.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "n4m/explain.h"

namespace n4m::explain {

// Writes `message` into the context error buffer (when ctx is non-null) and
// returns `status`.
n4m_status_t fail(n4m_context_t* ctx, n4m_status_t status, const char* message) noexcept;

// Overflow-checked int64 arithmetic on nonnegative operands.
bool checked_mul(int64_t a, int64_t b, int64_t& out) noexcept;
bool checked_add(int64_t a, int64_t b, int64_t& out) noexcept;

// Validates a non-null F64 view with rows >= min_rows and cols >= 1 whose
// strided extent fits in the address space.
n4m_status_t validate_f64_view(n4m_context_t* ctx, const n4m_matrix_view_t* view,
                               const char* name, int64_t min_rows);

inline double view_at(const n4m_matrix_view_t& view, int64_t row, int64_t col) noexcept {
    return static_cast<const double*>(view.data)[row * view.row_stride + col * view.col_stride];
}

// Copies a validated view into a dense row-major buffer; returns false when a
// value is not finite.
bool copy_finite_rowmajor(const n4m_matrix_view_t& view, std::vector<double>& out);

// Validates a feature -> group partition (NULL means one group per feature,
// which requires n_groups == n_features) and writes the dense mapping.
n4m_status_t validate_groups(n4m_context_t* ctx, const int64_t* feature_group,
                             int64_t n_features, int64_t n_groups,
                             std::vector<std::size_t>& group_of_feature);

// Solves the symmetric positive semi-definite system M x = B (n x n, n x k,
// row-major) with the library Householder QR. When the factor is numerically
// rank deficient (|R_ii| <= 1e-12 max |R_jj|) the solve is repeated once with
// a ridge of 1e-8 times the mean diagonal of M; `ridged` reports it.
n4m_status_t solve_spd(std::vector<double> matrix, std::size_t n, const std::vector<double>& rhs,
                       std::size_t k, std::vector<double>& solution, bool& ridged);

n4m_status_t linear_shap(n4m_context_t* ctx, const n4m_matrix_view_t* X,
                         const n4m_matrix_view_t* coefficients, const double* intercept,
                         const double* reference, const int64_t* feature_group, int64_t n_groups,
                         double* attributions, int64_t attributions_size, double* expected,
                         int64_t expected_size, double* predictions, int64_t predictions_size);

struct TreeEnsembleView {
    int64_t n_trees;
    const int64_t* tree_offsets;
    const int64_t* feature;
    const double* threshold;
    const int64_t* left;
    const int64_t* right;
    const uint8_t* default_left;
    const double* cover;
    const double* value;
    int64_t n_outputs;
    int32_t decision_type;
    const double* base_offset;
    double tree_scale;
};

n4m_status_t tree_shap_path_dependent(n4m_context_t* ctx, const n4m_matrix_view_t* X,
                                      const TreeEnsembleView& trees,
                                      const int64_t* feature_group, int64_t n_groups,
                                      double* attributions, int64_t attributions_size,
                                      double* expected, int64_t expected_size,
                                      double* predictions, int64_t predictions_size);

n4m_status_t groups_fixed_width(int64_t n_features, int64_t width, int64_t* feature_group,
                                int64_t feature_group_size, int64_t* out_n_groups);

n4m_status_t groups_from_ranges(int64_t n_features, const int64_t* ranges, int64_t n_ranges,
                                int64_t* feature_group, int64_t feature_group_size,
                                int64_t* range_of_group, int64_t range_of_group_size,
                                int64_t* out_n_groups);

n4m_status_t summarize(const double* attributions, int64_t n_rows, int64_t n_groups,
                       int64_t n_outputs, int64_t output_index, double* mean_abs,
                       double* mean_signed, int64_t summary_size);

}  // namespace n4m::explain
