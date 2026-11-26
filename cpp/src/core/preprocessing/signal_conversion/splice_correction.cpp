// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
#include "core/preprocessing/signal_conversion/splice_correction.hpp"

#include <cmath>
#include <cstddef>
#include <limits>

namespace n4m::core {

namespace {

template<typename V>
auto& at(V& values, std::int64_t index) {
    return values[static_cast<std::size_t>(index)];
}

double mean_of(const double* row, std::int64_t stride, const std::vector<std::int64_t>& cols) {
    double sum = 0.0;
    for (std::int64_t c : cols)
        sum += row[c * stride];
    return sum / static_cast<double>(cols.size());
}

}  // namespace

n4m_status_t plan_splice_correction(const double* axis,
                                    std::int64_t n,
                                    const double* splices,
                                    std::int64_t n_splices,
                                    std::int64_t reference,
                                    bool multiplicative,
                                    double window,
                                    bool taper,
                                    SplicePlan& plan) {
    if (axis == nullptr || n <= 0 || splices == nullptr || n_splices <= 0 || !(window >= 0.0)
        || !std::isfinite(window) || reference < 0 || reference > n_splices) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    for (std::int64_t i = 0; i < n; ++i) {
        if (!std::isfinite(axis[i]) || (i > 0 && !(axis[i] > axis[i - 1]))) {
            return N4M_ERR_INVALID_ARGUMENT;
        }
    }
    SplicePlan p;
    p.axis.assign(axis, axis + n);
    p.reference = reference;
    p.multiplicative = multiplicative;
    p.taper = taper;
    p.begin.push_back(0);
    std::int64_t col = 0;
    for (std::int64_t k = 0; k < n_splices; ++k) {
        const double splice = splices[k];
        if (!std::isfinite(splice))
            return N4M_ERR_INVALID_ARGUMENT;
        while (col < n && axis[col] < splice)
            ++col;
        // Every segment holds at least one band.
        if (col == p.begin.back() || col == n)
            return N4M_ERR_INVALID_ARGUMENT;
        p.begin.push_back(col);
    }
    p.begin.push_back(n);
    for (std::int64_t k = 0; k < n_splices; ++k) {
        const double splice = splices[k];
        const std::int64_t split = at(p.begin, k + 1);
        std::vector<std::int64_t> left, right;
        for (std::int64_t c = at(p.begin, k); c < split; ++c) {
            if (axis[c] >= splice - window)
                left.push_back(c);
        }
        for (std::int64_t c = split; c < at(p.begin, k + 2); ++c) {
            if (axis[c] <= splice + window)
                right.push_back(c);
        }
        if (left.empty())
            left.push_back(split - 1);
        if (right.empty())
            right.push_back(split);
        p.left.push_back(std::move(left));
        p.right.push_back(std::move(right));
    }
    plan = std::move(p);
    return N4M_OK;
}

n4m_status_t apply_splice_correction(const SplicePlan& plan,
                                     const n4m_matrix_view_t& X,
                                     n4m_matrix_view_t& out) noexcept {
    const auto* values = static_cast<const double*>(X.data);
    auto* result = static_cast<double*>(out.data);
    const auto segments = static_cast<std::int64_t>(plan.begin.size()) - 1;
    const std::int64_t width = plan.begin.back();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (std::int64_t row = 0; row < X.rows; ++row) {
        const double* in = values + row * X.row_stride;
        double* dst = result + row * out.row_stride;
        for (std::int64_t c = 0; c < width; ++c)
            dst[c * out.col_stride] = in[c * X.col_stride];
        // Corrects segment `k` against its corrected neighbour across splice
        // `s`; `fixed_left` tells whether the neighbour is on the left.
        auto correct = [&](std::int64_t k, std::int64_t s, bool fixed_left) {
            const auto& fixed = fixed_left ? at(plan.left, s) : at(plan.right, s);
            const auto& moved = fixed_left ? at(plan.right, s) : at(plan.left, s);
            const double level = mean_of(dst, out.col_stride, fixed);
            const double own = mean_of(dst, out.col_stride, moved);
            const double delta = plan.multiplicative ? level / own : level - own;
            const std::int64_t lo = at(plan.begin, k), hi = at(plan.begin, k + 1) - 1;
            const bool ok = std::isfinite(delta);
            // Linear in wavelength: full at the splice end, none at the far end.
            const double near = at(plan.axis, fixed_left ? lo : hi);
            const double far = at(plan.axis, fixed_left ? hi : lo);
            for (std::int64_t c = lo; c <= hi; ++c) {
                double& v = dst[c * out.col_stride];
                if (!ok) {
                    v = nan;
                    continue;
                }
                const double w =
                    !plan.taper || hi == lo ? 1.0 : (at(plan.axis, c) - far) / (near - far);
                v = plan.multiplicative ? v * (1.0 + (delta - 1.0) * w) : v + delta * w;
            }
        };
        for (std::int64_t k = plan.reference + 1; k < segments; ++k)
            correct(k, k - 1, true);
        for (std::int64_t k = plan.reference - 1; k >= 0; --k)
            correct(k, k, false);
    }
    return N4M_OK;
}

}  // namespace n4m::core
