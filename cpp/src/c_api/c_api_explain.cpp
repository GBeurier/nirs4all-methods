// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
//
// extern "C" wrappers for n4m/explain.h. Every wrapper catches all C++
// exceptions and translates them to a status; context-taking functions also
// write a message into the context error buffer.

#include <cstdint>
#include <exception>
#include <new>

#include "core/common/context.hpp"
#include "core/explain/explain_internal.hpp"
#include "core/explain/explain_session.hpp"
#include "n4m/explain.h"

static_assert(sizeof(n4m_explain_options_v1_t) == 72, "n4m_explain_options_v1_t must be 72 bytes");

namespace {

template <typename Fn>
n4m_status_t guarded(n4m_context_t* ctx, Fn&& function) noexcept {
    if (ctx == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    try {
        ctx->clear_error();
        return function();
    } catch (const std::bad_alloc&) {
        ctx->set_error("out of memory");
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        ctx->set_error(e.what());
        return N4M_ERR_INTERNAL;
    } catch (...) {
        ctx->set_error("internal error");
        return N4M_ERR_INTERNAL;
    }
}

template <typename Fn>
n4m_status_t guarded_noctx(Fn&& function) noexcept {
    try {
        return function();
    } catch (const std::bad_alloc&) {
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

}  // namespace

extern "C" {

N4M_API n4m_status_t n4m_explain_options_init_v1(n4m_explain_options_v1_t* options) {
    if (options == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    options->struct_size = sizeof(*options);
    options->method = N4M_EXPLAIN_PERMUTATION_SHAP;
    options->baseline = N4M_EXPLAIN_BASELINE_BACKGROUND;
    options->n_outputs = 1;
    options->n_permutations = 10;
    options->n_coalitions = 0;
    options->lime_kernel_width = 0.0;
    options->lime_ridge = 1.0;
    options->max_batch_rows = 4096;
    options->seed = 0;
    return N4M_OK;
}

N4M_API n4m_status_t n4m_explain_session_create_v1(n4m_context_t* ctx,
                                                   const n4m_matrix_view_t* X,
                                                   const n4m_matrix_view_t* background,
                                                   const int64_t* feature_group,
                                                   int64_t n_groups,
                                                   const n4m_explain_options_v1_t* options,
                                                   n4m_explain_session_t** out_session) {
    if (out_session == nullptr) {
        if (ctx != nullptr) {
            ctx->set_error("out_session is NULL");
        }
        return N4M_ERR_NULL_POINTER;
    }
    *out_session = nullptr;
    return guarded(ctx, [&]() {
        return n4m::explain::Session::create(ctx, X, background, feature_group, n_groups,
                                             options, *out_session);
    });
}

N4M_API void n4m_explain_session_destroy(n4m_explain_session_t* session) {
    delete session;
}

N4M_API n4m_status_t n4m_explain_session_info_v1(const n4m_explain_session_t* session,
                                                 int64_t* n_rows,
                                                 int64_t* n_features,
                                                 int64_t* n_groups,
                                                 int64_t* n_outputs,
                                                 int64_t* total_evaluations) {
    if (session == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    session->info(n_rows, n_features, n_groups, n_outputs, total_evaluations);
    return N4M_OK;
}

N4M_API n4m_status_t n4m_explain_session_next_batch_v1(n4m_context_t* ctx,
                                                       n4m_explain_session_t* session,
                                                       double* out_rows,
                                                       int64_t capacity_rows,
                                                       int64_t* out_rows_written) {
    return guarded(ctx, [&]() {
        if (session == nullptr) {
            return n4m::explain::fail(ctx, N4M_ERR_NULL_POINTER, "session is NULL");
        }
        return session->next_batch(ctx, out_rows, capacity_rows, out_rows_written);
    });
}

N4M_API n4m_status_t n4m_explain_session_submit_v1(n4m_context_t* ctx,
                                                   n4m_explain_session_t* session,
                                                   const double* predictions,
                                                   int64_t rows) {
    return guarded(ctx, [&]() {
        if (session == nullptr) {
            return n4m::explain::fail(ctx, N4M_ERR_NULL_POINTER, "session is NULL");
        }
        return session->submit(ctx, predictions, rows);
    });
}

N4M_API n4m_status_t n4m_explain_session_result_v1(n4m_context_t* ctx,
                                                   const n4m_explain_session_t* session,
                                                   double* attributions,
                                                   int64_t attributions_size,
                                                   double* expected,
                                                   int64_t expected_size,
                                                   double* predictions,
                                                   int64_t predictions_size,
                                                   double* std_error,
                                                   int64_t std_error_size) {
    return guarded(ctx, [&]() {
        if (session == nullptr) {
            return n4m::explain::fail(ctx, N4M_ERR_NULL_POINTER, "session is NULL");
        }
        return session->result(ctx, attributions, attributions_size, expected, expected_size,
                               predictions, predictions_size, std_error, std_error_size);
    });
}

N4M_API n4m_status_t n4m_explain_linear_v1(n4m_context_t* ctx,
                                           const n4m_matrix_view_t* X,
                                           const n4m_matrix_view_t* coefficients,
                                           const double* intercept,
                                           const double* reference,
                                           const int64_t* feature_group,
                                           int64_t n_groups,
                                           double* attributions,
                                           int64_t attributions_size,
                                           double* expected,
                                           int64_t expected_size,
                                           double* predictions,
                                           int64_t predictions_size) {
    return guarded(ctx, [&]() {
        return n4m::explain::linear_shap(ctx, X, coefficients, intercept, reference,
                                         feature_group, n_groups, attributions,
                                         attributions_size, expected, expected_size, predictions,
                                         predictions_size);
    });
}

N4M_API n4m_status_t n4m_explain_tree_path_dependent_v1(n4m_context_t* ctx,
                                                        const n4m_matrix_view_t* X,
                                                        int64_t n_trees,
                                                        const int64_t* tree_offsets,
                                                        const int64_t* feature,
                                                        const double* threshold,
                                                        const int64_t* left,
                                                        const int64_t* right,
                                                        const uint8_t* default_left,
                                                        const double* cover,
                                                        const double* value,
                                                        int64_t n_outputs,
                                                        int32_t decision_type,
                                                        const double* base_offset,
                                                        double tree_scale,
                                                        const int64_t* feature_group,
                                                        int64_t n_groups,
                                                        double* attributions,
                                                        int64_t attributions_size,
                                                        double* expected,
                                                        int64_t expected_size,
                                                        double* predictions,
                                                        int64_t predictions_size) {
    return guarded(ctx, [&]() {
        const n4m::explain::TreeEnsembleView trees{n_trees,      tree_offsets, feature,
                                                   threshold,    left,         right,
                                                   default_left, cover,        value,
                                                   n_outputs,    decision_type, base_offset,
                                                   tree_scale};
        return n4m::explain::tree_shap_path_dependent(ctx, X, trees, feature_group, n_groups,
                                                      attributions, attributions_size, expected,
                                                      expected_size, predictions,
                                                      predictions_size);
    });
}

N4M_API n4m_status_t n4m_explain_groups_fixed_width_v1(int64_t n_features,
                                                       int64_t width,
                                                       int64_t* feature_group,
                                                       int64_t feature_group_size,
                                                       int64_t* out_n_groups) {
    return guarded_noctx([&]() {
        return n4m::explain::groups_fixed_width(n_features, width, feature_group,
                                                feature_group_size, out_n_groups);
    });
}

N4M_API n4m_status_t n4m_explain_groups_from_ranges_v1(int64_t n_features,
                                                       const int64_t* ranges,
                                                       int64_t n_ranges,
                                                       int64_t* feature_group,
                                                       int64_t feature_group_size,
                                                       int64_t* range_of_group,
                                                       int64_t range_of_group_size,
                                                       int64_t* out_n_groups) {
    return guarded_noctx([&]() {
        return n4m::explain::groups_from_ranges(n_features, ranges, n_ranges, feature_group,
                                                feature_group_size, range_of_group,
                                                range_of_group_size, out_n_groups);
    });
}

N4M_API n4m_status_t n4m_explain_summarize_v1(const double* attributions,
                                              int64_t n_rows,
                                              int64_t n_groups,
                                              int64_t n_outputs,
                                              int64_t output_index,
                                              double* mean_abs,
                                              double* mean_signed,
                                              int64_t summary_size) {
    return guarded_noctx([&]() {
        return n4m::explain::summarize(attributions, n_rows, n_groups, n_outputs, output_index,
                                       mean_abs, mean_signed, summary_size);
    });
}

N4M_API n4m_status_t n4m_explain_session_set_gradient_step_v1(n4m_context_t* ctx,
                                                              n4m_explain_session_t* session,
                                                              double relative_step) {
    return guarded(ctx, [&]() {
        if (session == nullptr) {
            return n4m::explain::fail(ctx, N4M_ERR_NULL_POINTER, "session is NULL");
        }
        return session->set_gradient_step(ctx, relative_step);
    });
}

N4M_API n4m_status_t n4m_explain_session_gradients_v1(n4m_context_t* ctx,
                                                      const n4m_explain_session_t* session,
                                                      double* gradients,
                                                      int64_t gradients_size,
                                                      double* mean_gradient,
                                                      int64_t mean_gradient_size,
                                                      double* steps,
                                                      int64_t steps_size) {
    return guarded(ctx, [&]() {
        if (session == nullptr) {
            return n4m::explain::fail(ctx, N4M_ERR_NULL_POINTER, "session is NULL");
        }
        return session->gradients(ctx, gradients, gradients_size, mean_gradient,
                                  mean_gradient_size, steps, steps_size);
    });
}

N4M_API n4m_status_t n4m_explain_perturb_rows_v1(n4m_context_t* ctx,
                                                 const n4m_matrix_view_t* X,
                                                 const n4m_matrix_view_t* background,
                                                 const int64_t* feature_group,
                                                 int64_t n_groups,
                                                 int32_t baseline,
                                                 const int64_t* absent_groups,
                                                 int64_t n_absent,
                                                 double* out_rows,
                                                 int64_t out_rows_capacity,
                                                 int64_t* out_n_rows) {
    return guarded(ctx, [&]() {
        return n4m::explain::Session::perturb_rows(ctx, X, background, feature_group, n_groups,
                                                   baseline, absent_groups, n_absent, out_rows,
                                                   out_rows_capacity, out_n_rows);
    });
}

N4M_API n4m_status_t n4m_explain_summarize_std_error_v1(const double* std_error,
                                                        int64_t n_rows,
                                                        int64_t n_groups,
                                                        int64_t n_outputs,
                                                        int64_t output_index,
                                                        double* mean_signed_std_error,
                                                        int64_t summary_size) {
    return guarded_noctx([&]() {
        return n4m::explain::summarize_std_error(std_error, n_rows, n_groups, n_outputs,
                                                 output_index, mean_signed_std_error,
                                                 summary_size);
    });
}

N4M_API n4m_status_t n4m_explain_aggregate_profile_v1(n4m_context_t* ctx,
                                                      const double* values,
                                                      int64_t n_features,
                                                      const int64_t* feature_group,
                                                      int64_t n_groups,
                                                      int32_t rule,
                                                      double* out,
                                                      int64_t out_size) {
    return guarded(ctx, [&]() {
        return n4m::explain::aggregate_profile(ctx, values, n_features, feature_group, n_groups,
                                               rule, out, out_size);
    });
}

N4M_API n4m_status_t n4m_explain_compare_profiles_v1(n4m_context_t* ctx,
                                                     const n4m_matrix_view_t* profiles,
                                                     int64_t top_k,
                                                     double* spearman,
                                                     int64_t spearman_size,
                                                     double* top_k_jaccard,
                                                     int64_t jaccard_size) {
    return guarded(ctx, [&]() {
        return n4m::explain::compare_profiles(ctx, profiles, top_k, spearman, spearman_size,
                                              top_k_jaccard, jaccard_size);
    });
}

}  // extern "C"
