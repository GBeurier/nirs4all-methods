// SPDX-License-Identifier: CECILL-2.1
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "n4m/estimator.h"
#include "n4m/multimodal.h"

struct n4m_multimodal_pipeline_s {
    struct DeleteEstimator {
        void operator()(n4m_estimator_t* p) const { n4m_estimator_destroy(p); }
    };
    using Estimator = std::unique_ptr<n4m_estimator_t, DeleteEstimator>;
    struct Source {
        std::string name, representation, dtype, identity;
        std::vector<std::int64_t> shape;
        std::uint32_t encoder;
        double weight;
        std::int64_t n_components, random_state, numeric_column, categorical_column;
        std::int32_t with_mean, with_std, whiten, ignore_unknown;
        Estimator state;
        std::vector<std::string> categories;
    };
    std::vector<Source> sources;
    double alpha;
    Estimator model;
    std::int64_t encoded_cols = 0;
    bool fitted = false;
};

namespace n4m::multimodal {
n4m_status_t create(n4m_context_t*, const n4m_multimodal_recipe_v1_t*,
                    std::unique_ptr<n4m_multimodal_pipeline_s>&);
n4m_status_t fit(n4m_context_t*, n4m_multimodal_pipeline_s&, int32_t,
                 const n4m_multimodal_source_view_v1_t*, const n4m_matrix_view_t*);
n4m_status_t features(n4m_context_t*, const n4m_multimodal_pipeline_s&, int32_t,
                      const n4m_multimodal_source_view_v1_t*, std::vector<double>&,
                      std::int64_t& rows, std::int64_t& cols);
n4m_status_t save(n4m_context_t*, const n4m_multimodal_pipeline_s&,
                  std::vector<unsigned char>&);
n4m_status_t load(n4m_context_t*, const n4m_multimodal_recipe_v1_t*,
                  const void*, std::size_t, std::unique_ptr<n4m_multimodal_pipeline_s>&);
}  // namespace n4m::multimodal
