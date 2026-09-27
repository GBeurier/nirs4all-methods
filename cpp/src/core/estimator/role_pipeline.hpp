// SPDX-License-Identifier: CECILL-2.1
//
// Role pipeline (n4m/estimator.h, ABI 2.14): a trained linear recipe of
// estimator steps (sample filters, transformers / selectors, one regressor
// or classifier). Steps are fitted, applied and serialized through the
// generic estimator life cycle; this layer only validates the recipe,
// routes the fit inputs and moves rows between steps.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/estimator/spec.hpp"

struct n4m_role_pipeline_s {
    struct Step {
        std::int32_t method_index;
        ::n4m::estimator::Params params;
        std::uint32_t role;         // the N4M_ROLE_* the step plays
        std::int32_t state_index;   // -1 for sample filters
    };
    std::vector<Step> steps;
    std::vector<std::string> feature_names;
    // One fitted estimator per stateful step, in step order, once fitted.
    std::vector<std::unique_ptr<n4m_estimator_s>> states;
    std::int64_t n_features = 0;
    bool fitted = false;

    const n4m_estimator_s& terminal() const { return *states.back(); }
};

namespace n4m::estimator {

n4m_status_t pipeline_create(n4m_context_t* ctx, std::int32_t n_steps,
                             const char* const* method_ids, const n4m_params_t* const* params,
                             std::unique_ptr<n4m_role_pipeline_s>& out);
n4m_status_t pipeline_set_feature_names(n4m_context_t* ctx, n4m_role_pipeline_s& pipeline,
                                        const char* const* names, std::int64_t n);
n4m_status_t pipeline_fit(n4m_context_t* ctx, n4m_role_pipeline_s& pipeline,
                          const n4m_fit_inputs_v1_t* inputs);
n4m_status_t pipeline_import(n4m_context_t* ctx, n4m_role_pipeline_s& pipeline,
                             std::int32_t n_states, const void* const* states,
                             const std::size_t* sizes);
n4m_status_t pipeline_check_features(n4m_context_t* ctx, const n4m_role_pipeline_s& pipeline,
                                     std::int64_t n_columns, const char* const* names);
// Input of the terminal step for new rows X: X itself, or its image through
// the transformers and selectors (stored in `store`).
n4m_status_t pipeline_features(n4m_context_t* ctx, const n4m_role_pipeline_s& pipeline,
                               const n4m_matrix_view_t* X, std::vector<double>& store,
                               n4m_matrix_view_t& features);
// Refuses an output view that is not a valid rows x cols float64 matrix.
n4m_status_t pipeline_check_output(n4m_context_t* ctx, const n4m_matrix_view_t* out,
                                   std::int64_t rows, std::int64_t cols);
n4m_status_t pipeline_transform(n4m_context_t* ctx, const n4m_role_pipeline_s& pipeline,
                                const n4m_matrix_view_t* X, n4m_matrix_view_t* out);
std::int64_t pipeline_transform_cols(const n4m_role_pipeline_s& pipeline) noexcept;
// Rewrites the context message as "<where>: <message>" and returns st.
n4m_status_t pipeline_error(n4m_context_t* ctx, n4m_status_t st, const char* where);
// "step <i> (<method id>)" and "state <k> (step <i> (<method id>))".
std::string step_label(const n4m_role_pipeline_s& pipeline, std::size_t step);
std::string state_label(const n4m_role_pipeline_s& pipeline, std::size_t state);

}  // namespace n4m::estimator
