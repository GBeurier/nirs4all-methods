// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
#include <algorithm>
#include <limits>
#include <new>

#include "core/common/matrix_view.hpp"
#include "core/ranking/ranking.hpp"
#include "n4m/ranking.h"

extern "C" {
N4M_API n4m_status_t n4m_ranking_options_init_v1(n4m_ranking_options_v1_t* o) {
    if (!o)
        return N4M_ERR_NULL_POINTER;
    *o = {sizeof(*o),
          N4M_RANKING_AVERAGE_RANK,
          0,
          0,
          25,
          32,
          400,
          25.0 / 3,
          25.0 / 6,
          0,
          0.740466587452148,
          1,
          1e-8,
          10000};
    return N4M_OK;
}
N4M_API n4m_status_t n4m_ranking_compute_v1(const n4m_matrix_view_t* scores,
                                            const uint8_t* present,
                                            const n4m_ranking_options_v1_t* options,
                                            double* candidates,
                                            int64_t candidates_size,
                                            double* summary,
                                            int64_t summary_size) {
    if (!scores || !present || !options || !candidates || !summary)
        return N4M_ERR_NULL_POINTER;
    const n4m_status_t valid = n4m::core::validate_nonnull_view(*scores);
    if (valid != N4M_OK)
        return valid;
    if (scores->dtype != N4M_DTYPE_F64)
        return N4M_ERR_DTYPE_MISMATCH;
    if (scores->rows <= 0 || scores->cols <= 0)
        return N4M_ERR_INVALID_ARGUMENT;
    // Validate element and byte offsets before any pointer arithmetic.
    constexpr int64_t max_offset = std::numeric_limits<std::ptrdiff_t>::max() / sizeof(double);
    if ((scores->rows > 1 && scores->row_stride > max_offset / (scores->rows - 1))
        || (scores->cols > 1 && scores->col_stride > max_offset / (scores->cols - 1)))
        return N4M_ERR_STRIDE_INVALID;
    const int64_t row_offset = (scores->rows - 1) * scores->row_stride;
    const int64_t col_offset = (scores->cols - 1) * scores->col_stride;
    if (row_offset > max_offset - col_offset)
        return N4M_ERR_STRIDE_INVALID;
    // Explicit resource limits: O(blocks*k^2) paired comparisons; batch BT
    // additionally uses O(k^2) storage. Fail before overflow or allocation.
    constexpr int64_t max_cells = 100000000;
    constexpr int64_t max_candidates = 10000;
    if (scores->cols > max_candidates || scores->rows > max_cells / scores->cols)
        return N4M_ERR_UNSUPPORTED;
    if (options->method == N4M_RANKING_BRADLEY_TERRY && scores->cols > 2000)
        return N4M_ERR_UNSUPPORTED;
    if (candidates_size != scores->cols * 10 || summary_size != 9)
        return N4M_ERR_SHAPE_MISMATCH;
    try {
        n4m::ranking::Result result;
        const n4m_status_t status = n4m::ranking::compute(*scores, present, *options, result);
        if (status != N4M_OK)
            return status;
        std::copy(result.candidates.begin(), result.candidates.end(), candidates);
        std::copy(result.summary.begin(), result.summary.end(), summary);
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}
N4M_API n4m_status_t n4m_ranking_relative_gain_v1(const double* baseline,
                                                  const double* candidate,
                                                  const uint8_t* present,
                                                  const double* variances,
                                                  int64_t n,
                                                  double* output,
                                                  int64_t output_size) {
    if (!baseline || !candidate || !present || !output)
        return N4M_ERR_NULL_POINTER;
    if (n <= 0)
        return N4M_ERR_INVALID_ARGUMENT;
    if (n > 100000000)
        return N4M_ERR_UNSUPPORTED;
    if (output_size != 8)
        return N4M_ERR_SHAPE_MISMATCH;
    try {
        std::array<double, 8> result;
        const n4m_status_t status =
            n4m::ranking::relative_gain(baseline, candidate, present, variances, n, result);
        if (status != N4M_OK)
            return status;
        std::copy(result.begin(), result.end(), output);
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}
}
