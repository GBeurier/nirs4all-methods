// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
//
// Ask/tell explanation session behind n4m_explain_session_t.
//
// The plan is a deterministic sequence of model evaluations walked by a
// cursor (row, coalition, base). Row -1 is the optional global phase that
// evaluates the empty coalition once for the BACKGROUND and MEAN baselines
// (identical for every explained row). Per explained row, coalition 0 is the
// full coalition (x itself, one evaluation), coalition 1 is the per-row empty
// coalition for INTERPOLATE, and the method coalitions follow. Every non-full
// coalition costs n_base evaluations (background rows for BACKGROUND, 1
// otherwise); their predictions are averaged into v(S).
//
// Row plans (permutations, sampled masks, per-coalition sums) are created
// lazily, in row order, when the emitter enters a row, and are released as
// soon as the row is finalized, so memory is bounded by the rows spanned by
// one batch, never by the full plan.

#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <vector>

#include "core/common/rng_pcg64.h"
#include "n4m/explain.h"

namespace n4m::explain {

class Session {
  public:
    static n4m_status_t create(n4m_context_t* ctx, const n4m_matrix_view_t* X,
                               const n4m_matrix_view_t* background, const int64_t* feature_group,
                               int64_t n_groups, const n4m_explain_options_v1_t* options,
                               n4m_explain_session_t*& out);

    Session() = default;
    ~Session() = default;
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

    void info(int64_t* n_rows, int64_t* n_features, int64_t* n_groups, int64_t* n_outputs,
              int64_t* total_evaluations) const noexcept;
    n4m_status_t next_batch(n4m_context_t* ctx, double* out_rows, int64_t capacity_rows,
                            int64_t* out_rows_written);
    n4m_status_t submit(n4m_context_t* ctx, const double* predictions, int64_t rows);
    n4m_status_t result(n4m_context_t* ctx, double* attributions, int64_t attributions_size,
                        double* expected, int64_t expected_size, double* predictions,
                        int64_t predictions_size, double* std_error,
                        int64_t std_error_size) const;
    // GRADIENT sessions (ABI 2.24).
    n4m_status_t set_gradient_step(n4m_context_t* ctx, double relative_step);
    n4m_status_t gradients(n4m_context_t* ctx, double* gradients, int64_t gradients_size,
                           double* mean_gradient, int64_t mean_gradient_size, double* steps,
                           int64_t steps_size) const;

    // n4m_explain_perturb_rows_v1: the rows the session's baseline filling
    // writes for one coalition (absent groups), for every row of X.
    static n4m_status_t perturb_rows(n4m_context_t* ctx, const n4m_matrix_view_t* X,
                                     const n4m_matrix_view_t* background,
                                     const int64_t* feature_group, int64_t n_groups,
                                     int32_t baseline, const int64_t* absent_groups,
                                     int64_t n_absent, double* out_rows,
                                     int64_t out_rows_capacity, int64_t* out_n_rows);

  private:
    enum class State { kReady, kPending, kComplete, kPoisoned };

    struct Cursor {
        int64_t row = 0;  // -1: global empty phase; n_rows: end of plan
        std::size_t coalition = 0;
        std::size_t base = 0;
    };

    struct RowPlan {
        std::vector<double> values;          // coalition x output prediction sums
        std::vector<std::size_t> perms;      // PERMUTATION: n_permutations x n_groups
        std::vector<unsigned char> masks;    // sampled KERNEL / LIME: coalition x group
    };

    Cursor start_cursor() const noexcept;
    bool at_end(const Cursor& cursor) const noexcept;
    void advance(Cursor& cursor) const noexcept;
    RowPlan make_plan();
    void coalition_mask(const RowPlan& plan, std::size_t coalition, unsigned char* mask) const;
    void fill_row(const Cursor& cursor, const RowPlan* plan, double* out);
    void interpolate(const double* x, const unsigned char* present, double* out) const;
    // Fills absent features (present_ == 0) of x from the baseline (background
    // row `base` for BACKGROUND); the one filling rule of every coalition.
    void fill_present(const double* x, std::size_t base, double* out) const;
    double value(const RowPlan& plan, std::size_t coalition, std::size_t output) const noexcept;
    double empty_value(const RowPlan& plan, std::size_t output) const noexcept;
    n4m_status_t finalize_row(n4m_context_t* ctx, std::size_t row, const RowPlan& plan);
    void finalize_exact(std::size_t row, const RowPlan& plan);
    void finalize_permutation(std::size_t row, const RowPlan& plan);
    n4m_status_t finalize_kernel(n4m_context_t* ctx, std::size_t row, const RowPlan& plan);
    n4m_status_t finalize_lime(n4m_context_t* ctx, std::size_t row, const RowPlan& plan);
    void finalize_occlusion(std::size_t row, const RowPlan& plan);
    void finalize_gradient(std::size_t row, const RowPlan& plan);
    void update_gradient_steps(double relative_step);
    void sample_subset(std::size_t size, unsigned char* mask, unsigned char present_value);

    // Configuration.
    int32_t method_ = N4M_EXPLAIN_PERMUTATION_SHAP;
    int32_t baseline_ = N4M_EXPLAIN_BASELINE_BACKGROUND;
    std::size_t n_rows_ = 0;
    std::size_t n_features_ = 0;
    std::size_t n_groups_ = 0;
    std::size_t n_outputs_ = 0;
    std::size_t n_base_ = 1;
    std::size_t n_permutations_ = 0;
    std::size_t n_method_coalitions_ = 0;
    std::size_t first_method_coalition_ = 1;
    std::size_t coalitions_per_row_ = 1;
    bool kernel_enumerate_ = false;
    bool global_empty_ = false;
    bool row_empty_ = false;
    double lime_width_ = 1.0;
    double lime_ridge_ = 1.0;
    std::size_t max_batch_rows_ = 4096;
    int64_t total_evaluations_ = 0;
    std::vector<double> x_;
    std::vector<double> background_;
    std::vector<double> reference_;
    std::vector<std::size_t> group_of_feature_;
    std::vector<double> kernel_size_cdf_;  // sampled KERNEL: cumulative size weights
    // GRADIENT: per-feature scale s_j, absolute steps h_j = step * s_j.
    double gradient_step_ = 1e-3;
    std::vector<double> gradient_scale_;
    std::vector<double> gradient_steps_;
    std::vector<double> gradients_;  // [row][feature][output]
    n4m_rng_pcg64 rng_{};

    // Ask/tell state.
    State state_ = State::kReady;
    Cursor emit_;
    Cursor batch_start_;
    std::size_t pending_rows_ = 0;
    std::deque<RowPlan> plans_;
    std::vector<double> global_empty_value_;
    std::vector<double> attributions_;
    std::vector<double> expected_;
    std::vector<double> predictions_;
    std::vector<double> std_error_;
    std::vector<std::size_t> subset_scratch_;

    // fill_row cache: feature-level presence of the last coalition.
    int64_t cached_row_ = -2;
    std::size_t cached_coalition_ = 0;
    std::vector<unsigned char> group_mask_;
    std::vector<unsigned char> present_;
};

}  // namespace n4m::explain

struct n4m_explain_session_s : public n4m::explain::Session {};
