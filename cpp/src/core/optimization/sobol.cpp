// SPDX-License-Identifier: CECILL-2.1
//
// SobolSampler (F1) — Sobol low-discrepancy sequence over the search space
// using the embedded Joe-Kuo direction numbers (see sobol_direction.hpp). One
// Sobol dimension is assigned per parameter (in space order); the unscrambled
// sequence is bit-identical to scipy.stats.qmc.Sobol(scramble=False). Plugs into
// the base per-parameter hooks so constraints/conditions/forced/sorted-tuple are
// handled by the base sampler.

#include "core/optimization/optimizer.hpp"
#include "core/optimization/sobol_direction.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace n4m::core::opt {

SobolSampler::SobolSampler(const SearchSpace& space, const n4m_optimizer_options_t& opts)
    : Optimizer(space, opts) {
    sobol_dims_ = std::min<int>(static_cast<int>(space.params.size()), kSobolMaxDim);
    x_.assign(static_cast<std::size_t>(sobol_dims_), 0u);
    cached_point_.assign(static_cast<std::size_t>(sobol_dims_), 0.0);
}

int SobolSampler::dim_of(const ParamSpec& p) const {
    for (std::size_t j = 0; j < space_.params.size(); ++j) {
        if (space_.params[j].name == p.name) {
            return static_cast<int>(j) < sobol_dims_ ? static_cast<int>(j) : -1;
        }
    }
    return -1;
}

void SobolSampler::ensure_point() {
    const std::int64_t i = static_cast<std::int64_t>(trials().size());
    if (exhausted_ || i == cached_ask_index_) return;
    // Advance the Gray-code state from next_index_ up to i.
    while (next_index_ < i) {
        std::uint64_t v = static_cast<std::uint64_t>(next_index_);
        int c = 0;
        while ((v >> c) & 1u) ++c;  // rightmost zero bit of next_index_
        if (c >= kSobolBits) {  // 2^30 points exhausted → fall back to base random
            exhausted_ = true;
            return;
        }
        for (int d = 0; d < sobol_dims_; ++d) x_[static_cast<std::size_t>(d)] ^= kSobolSv[d][c];
        ++next_index_;
    }
    const double scale = static_cast<double>(std::uint64_t{1} << kSobolBits);  // 2^30
    for (int d = 0; d < sobol_dims_; ++d) {
        cached_point_[static_cast<std::size_t>(d)] =
            static_cast<double>(x_[static_cast<std::size_t>(d)]) / scale;
    }
    cached_ask_index_ = i;
}

bool SobolSampler::override_numeric(const ParamSpec& p, double* out) {
    const int d = dim_of(p);
    if (d < 0) return false;  // beyond the direction table → base random
    ensure_point();
    if (exhausted_) return false;  // sequence resolution exhausted → base random
    *out = numeric_from_unit(p, cached_point_[static_cast<std::size_t>(d)]);
    return true;
}

bool SobolSampler::override_categorical(const ParamSpec& p, std::int32_t* out_index) {
    const int d = dim_of(p);
    if (d < 0) return false;
    const int n = static_cast<int>(std::max(p.labels.size(), p.num_values.size()));
    if (n <= 0) return false;
    ensure_point();
    if (exhausted_) return false;
    int idx = static_cast<int>(cached_point_[static_cast<std::size_t>(d)] * n);
    if (idx >= n) idx = n - 1;
    if (idx < 0) idx = 0;
    *out_index = idx;
    return true;
}

}  // namespace n4m::core::opt
