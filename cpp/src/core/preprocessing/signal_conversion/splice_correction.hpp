// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
//
// Detector splice (jump) correction of multi-detector spectra (ASD, Spectral
// Evolution, SVC field spectroradiometers), reached through the generic
// estimator roles only.
#pragma once

#include <cstdint>
#include <vector>

#include "n4m/n4m.h"

namespace n4m::core {

// Column layout of a splice correction over a fitted axis. Segment k holds
// columns [begin[k], begin[k + 1]); a band at or after splices[k - 1] starts
// segment k. For each splice, `left` / `right` are the columns whose mean
// estimates the level on each side.
struct SplicePlan {
    std::vector<std::int64_t> begin;  // n_segments + 1 entries
    std::vector<std::vector<std::int64_t>> left, right;
    std::vector<double> axis;
    std::int64_t reference = 0;
    bool multiplicative = true;
    bool taper = false;
};

// Builds the plan. The axis must be finite and strictly increasing, every
// segment non-empty, `window` >= 0 (axis units on each side of a splice; 0 uses
// the two bands next to the splice) and `reference` a segment index.
n4m_status_t plan_splice_correction(const double* axis,
                                    std::int64_t n,
                                    const double* splices,
                                    std::int64_t n_splices,
                                    std::int64_t reference,
                                    bool multiplicative,
                                    double window,
                                    bool taper,
                                    SplicePlan& plan);

// Segments are corrected outward from the reference, each one matched to its
// already corrected neighbour: multiplied by mean(neighbour side) / mean(own
// side), or shifted by their difference. With `taper`, the correction is full
// at the splice and decreases linearly to none at the far end of the segment.
// A segment whose factor is not finite, and those beyond it, are NaN.
n4m_status_t apply_splice_correction(const SplicePlan& plan,
                                     const n4m_matrix_view_t& X,
                                     n4m_matrix_view_t& out) noexcept;

}  // namespace n4m::core
