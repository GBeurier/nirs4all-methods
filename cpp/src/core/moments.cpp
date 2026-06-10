// SPDX-License-Identifier: CECILL-2.1
//
// Moment substrate for exact row-additive sufficient statistics.

#include "core/moments.hpp"

#include <cmath>
#include <cstddef>
#if defined(N4M_USE_CUDA)
#  include <cstdlib>
#endif
#include <limits>
#if defined(N4M_USE_CUDA)
#  include <string>
#endif
#include <vector>

#include "core/common/linalg.hpp"
#include "core/common/matrix_view.hpp"
#if defined(N4M_USE_CUDA)
#  include "core/cuda_dispatch.hpp"
#endif

namespace n4m::core {

namespace {

[[nodiscard]] bool product_overflows(std::size_t a, std::size_t b) noexcept {
    return b != 0 &&
           a > (std::numeric_limits<std::size_t>::max() / b);
}

#if defined(N4M_USE_CUDA)
// GPU moment/gram build only pays off at scale (consumer-GPU fp64 is weak;
// below the crossover the host BLAS build + CPU/GPU overlap wins).
// n*p*p product; env-tunable. 0 disables the gate (always try GPU).
[[nodiscard]] static std::size_t moment_cuda_min_product() {
    if (const char* e = std::getenv("N4M_CUDA_MOMENT_MIN_PRODUCT")) {
        char* end = nullptr;
        unsigned long long v = std::strtoull(e, &end, 10);
        if (end != e) return static_cast<std::size_t>(v);
    }
    return 15000000000ULL;
}

[[nodiscard]] bool moment_cuda_product_is_large(std::size_t n,
                                                std::size_t p) noexcept {
    const std::size_t min_product = moment_cuda_min_product();
    if (min_product == 0U) return true;
    if (product_overflows(n, p)) return true;
    const std::size_t np = n * p;
    if (product_overflows(np, p)) return true;
    return np * p >= min_product;
}
#endif

[[nodiscard]] n4m_status_t validate_xy(Context& ctx,
                                       const n4m_matrix_view_t& X,
                                       const n4m_matrix_view_t& Y) {
    n4m_status_t st = validate_nonnull_view(X);
    if (st != N4M_OK) {
        ctx.set_error("X matrix view is invalid");
        return st;
    }
    st = validate_nonnull_view(Y);
    if (st != N4M_OK) {
        ctx.set_error("Y matrix view is invalid");
        return st;
    }
    if ((X.dtype != N4M_DTYPE_F64 && X.dtype != N4M_DTYPE_F32) ||
        (Y.dtype != N4M_DTYPE_F64 && Y.dtype != N4M_DTYPE_F32)) {
        ctx.set_error("moments require F64 or F32 X and Y matrices");
        return N4M_ERR_DTYPE_MISMATCH;
    }
    if (X.rows <= 0 || X.cols <= 0 || Y.rows <= 0 || Y.cols <= 0) {
        ctx.set_error("moments require non-empty X and Y");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (X.rows != Y.rows) {
        ctx.set_error("moments: X and Y must have the same number of rows");
        return N4M_ERR_SHAPE_MISMATCH;
    }
    return N4M_OK;
}

template <typename T>
[[nodiscard]] n4m_status_t copy_full_typed(Context& ctx,
                                          const n4m_matrix_view_t& V,
                                          const char* name,
                                          std::vector<double>& out) {
    const auto* data = static_cast<const T*>(V.data);
    const auto rows = static_cast<std::size_t>(V.rows);
    const auto cols = static_cast<std::size_t>(V.cols);
    const auto rs = static_cast<std::size_t>(V.row_stride);
    const auto cs = static_cast<std::size_t>(V.col_stride);
    if (product_overflows(rows, cols)) {
        ctx.set_errorf("%s dimensions overflow addressable memory", name);
        return N4M_ERR_INVALID_ARGUMENT;
    }
    out.assign(rows * cols, 0.0);
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            const double v = static_cast<double>(data[r * rs + c * cs]);
            if (!std::isfinite(v)) {
                ctx.set_errorf("%s contains NaN or Inf", name);
                return N4M_ERR_INVALID_ARGUMENT;
            }
            out[r * cols + c] = v;
        }
    }
    return N4M_OK;
}

[[nodiscard]] n4m_status_t copy_full(Context& ctx,
                                     const n4m_matrix_view_t& V,
                                     const char* name,
                                     std::vector<double>& out) {
    if (V.dtype == N4M_DTYPE_F64) {
        return copy_full_typed<double>(ctx, V, name, out);
    }
    return copy_full_typed<float>(ctx, V, name, out);
}

template <typename T>
[[nodiscard]] n4m_status_t copy_subset_typed(Context& ctx,
                                            const n4m_matrix_view_t& V,
                                            const char* name,
                                            const std::int64_t* row_indices,
                                            std::int64_t n_indices,
                                            std::vector<double>& out) {
    const auto* data = static_cast<const T*>(V.data);
    const auto src_rows = V.rows;
    const auto cols = static_cast<std::size_t>(V.cols);
    const auto rs = static_cast<std::size_t>(V.row_stride);
    const auto cs = static_cast<std::size_t>(V.col_stride);
    const auto rows = static_cast<std::size_t>(n_indices);
    if (product_overflows(rows, cols)) {
        ctx.set_errorf("%s subset dimensions overflow addressable memory", name);
        return N4M_ERR_INVALID_ARGUMENT;
    }
    out.assign(rows * cols, 0.0);
    for (std::size_t rr = 0; rr < rows; ++rr) {
        const std::int64_t src = row_indices[rr];
        if (src < 0 || src >= src_rows) {
            ctx.set_errorf("row_indices[%zu]=%lld out of range [0, %lld)",
                           rr,
                           static_cast<long long>(src),
                           static_cast<long long>(src_rows));
            return N4M_ERR_INVALID_ARGUMENT;
        }
        const auto src_u = static_cast<std::size_t>(src);
        for (std::size_t c = 0; c < cols; ++c) {
            const double v = static_cast<double>(data[src_u * rs + c * cs]);
            if (!std::isfinite(v)) {
                ctx.set_errorf("%s subset contains NaN or Inf", name);
                return N4M_ERR_INVALID_ARGUMENT;
            }
            out[rr * cols + c] = v;
        }
    }
    return N4M_OK;
}

[[nodiscard]] n4m_status_t copy_subset(Context& ctx,
                                       const n4m_matrix_view_t& V,
                                       const char* name,
                                       const std::int64_t* row_indices,
                                       std::int64_t n_indices,
                                       std::vector<double>& out) {
    if (V.dtype == N4M_DTYPE_F64) {
        return copy_subset_typed<double>(
            ctx, V, name, row_indices, n_indices, out);
    }
    return copy_subset_typed<float>(
        ctx, V, name, row_indices, n_indices, out);
}

void column_sums(const std::vector<double>& data,
                 std::size_t rows,
                 std::size_t cols,
                 std::vector<double>& sums) {
    sums.assign(cols, 0.0);
    for (std::size_t r = 0; r < rows; ++r) {
        const double* row = data.data() + r * cols;
        for (std::size_t c = 0; c < cols; ++c) {
            sums[c] += row[c];
        }
    }
}

[[nodiscard]] n4m_status_t compute_from_contiguous(Context& ctx,
                                                   const std::vector<double>& X,
                                                   const std::vector<double>& Y,
                                                   std::int64_t n,
                                                   std::int64_t p,
                                                   std::int64_t q,
                                                   MomentStats& out) {
    if (n <= 0 || p <= 0 || q <= 0) {
        ctx.set_error("moments require positive n, p and q");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    const auto nn = static_cast<std::size_t>(n);
    const auto pp = static_cast<std::size_t>(p);
    const auto qq = static_cast<std::size_t>(q);
    if (product_overflows(pp, pp) ||
        product_overflows(pp, qq) ||
        product_overflows(qq, qq)) {
        ctx.set_error("moment matrix dimensions overflow addressable memory");
        return N4M_ERR_INVALID_ARGUMENT;
    }

    out = MomentStats{};
    out.n_samples = n;
    out.n_features = p;
    out.n_targets = q;
    column_sums(X, nn, pp, out.x_sum);
    column_sums(Y, nn, qq, out.y_sum);

    out.xtx.assign(pp * pp, 0.0);
    out.xty.assign(pp * qq, 0.0);
    out.yty.assign(qq * qq, 0.0);

    bool built_on_device = false;
#if defined(N4M_USE_CUDA)
    if (moment_cuda_product_is_large(nn, pp) &&
        ::n4m::cuda_dispatch::cuda_runtime_available()) {
        std::string err;
        if (::n4m::cuda_dispatch::build_moments_device(
                nn, pp, qq, X.data(), Y.data(),
                out.xtx.data(), out.xty.data(), out.yty.data(), &err) == 0) {
            built_on_device = true;
        }
    }
#endif
    if (!built_on_device) {
        linalg::gemm(linalg::Trans_Yes, linalg::Trans_No,
                     pp, pp, nn, 1.0,
                     X.data(), pp, X.data(), pp,
                     0.0, out.xtx.data(), pp);
        linalg::gemm(linalg::Trans_Yes, linalg::Trans_No,
                     pp, qq, nn, 1.0,
                     X.data(), pp, Y.data(), qq,
                     0.0, out.xty.data(), qq);
        linalg::gemm(linalg::Trans_Yes, linalg::Trans_No,
                     qq, qq, nn, 1.0,
                     Y.data(), qq, Y.data(), qq,
                     0.0, out.yty.data(), qq);
    }
    return recompute_centered_moments(ctx, out);
}

}  // namespace

n4m_status_t recompute_centered_moments(Context& ctx, MomentStats& stats) {
    if (stats.n_samples <= 0 || stats.n_features <= 0 || stats.n_targets <= 0) {
        ctx.set_error("cannot center moments with non-positive dimensions");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    const auto p = static_cast<std::size_t>(stats.n_features);
    const auto q = static_cast<std::size_t>(stats.n_targets);
    const double inv_n = 1.0 / static_cast<double>(stats.n_samples);
    if (stats.x_sum.size() != p || stats.y_sum.size() != q ||
        stats.xtx.size() != p * p || stats.xty.size() != p * q ||
        stats.yty.size() != q * q) {
        ctx.set_error("raw moment buffers have inconsistent shapes");
        return N4M_ERR_INVALID_ARGUMENT;
    }

    stats.x_mean.assign(p, 0.0);
    stats.y_mean.assign(q, 0.0);
    for (std::size_t i = 0; i < p; ++i) {
        stats.x_mean[i] = stats.x_sum[i] * inv_n;
    }
    for (std::size_t j = 0; j < q; ++j) {
        stats.y_mean[j] = stats.y_sum[j] * inv_n;
    }

    stats.cxx.assign(p * p, 0.0);
    for (std::size_t i = 0; i < p; ++i) {
        for (std::size_t j = 0; j < p; ++j) {
            stats.cxx[i * p + j] =
                stats.xtx[i * p + j] - stats.x_sum[i] * stats.x_sum[j] * inv_n;
        }
    }

    stats.cxy.assign(p * q, 0.0);
    for (std::size_t i = 0; i < p; ++i) {
        for (std::size_t j = 0; j < q; ++j) {
            stats.cxy[i * q + j] =
                stats.xty[i * q + j] - stats.x_sum[i] * stats.y_sum[j] * inv_n;
        }
    }

    stats.cyy.assign(q * q, 0.0);
    for (std::size_t i = 0; i < q; ++i) {
        for (std::size_t j = 0; j < q; ++j) {
            stats.cyy[i * q + j] =
                stats.yty[i * q + j] - stats.y_sum[i] * stats.y_sum[j] * inv_n;
        }
    }
    return N4M_OK;
}

n4m_status_t compute_moments(Context& ctx,
                             const n4m_matrix_view_t& X,
                             const n4m_matrix_view_t& Y,
                             MomentStats& out) {
    n4m_status_t st = validate_xy(ctx, X, Y);
    if (st != N4M_OK) return st;

    std::vector<double> Xc;
    std::vector<double> Yc;
    st = copy_full(ctx, X, "X", Xc);
    if (st != N4M_OK) return st;
    st = copy_full(ctx, Y, "Y", Yc);
    if (st != N4M_OK) return st;
    return compute_from_contiguous(ctx, Xc, Yc, X.rows, X.cols, Y.cols, out);
}

n4m_status_t compute_moments_subset(Context& ctx,
                                    const n4m_matrix_view_t& X,
                                    const n4m_matrix_view_t& Y,
                                    const std::int64_t* row_indices,
                                    std::int64_t n_indices,
                                    MomentStats& out) {
    n4m_status_t st = validate_xy(ctx, X, Y);
    if (st != N4M_OK) return st;
    if (n_indices <= 0) {
        ctx.set_error("moments subset requires at least one row index");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (row_indices == nullptr) {
        ctx.set_error("moments subset row_indices must not be NULL");
        return N4M_ERR_NULL_POINTER;
    }

    std::vector<double> Xc;
    std::vector<double> Yc;
    st = copy_subset(ctx, X, "X", row_indices, n_indices, Xc);
    if (st != N4M_OK) return st;
    st = copy_subset(ctx, Y, "Y", row_indices, n_indices, Yc);
    if (st != N4M_OK) return st;
    return compute_from_contiguous(ctx, Xc, Yc, n_indices, X.cols, Y.cols, out);
}

n4m_status_t subtract_moments(Context& ctx,
                              const MomentStats& lhs,
                              const MomentStats& rhs,
                              MomentStats& out) {
    if (lhs.n_features <= 0 || lhs.n_targets <= 0 ||
        lhs.n_features != rhs.n_features ||
        lhs.n_targets != rhs.n_targets) {
        ctx.set_error("moment subtraction requires matching positive p and q");
        return N4M_ERR_SHAPE_MISMATCH;
    }
    if (rhs.n_samples < 0 || lhs.n_samples <= rhs.n_samples) {
        ctx.set_error("moment subtraction would leave an empty/negative row set");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    const auto p = static_cast<std::size_t>(lhs.n_features);
    const auto q = static_cast<std::size_t>(lhs.n_targets);
    if (lhs.x_sum.size() != p || rhs.x_sum.size() != p ||
        lhs.y_sum.size() != q || rhs.y_sum.size() != q ||
        lhs.xtx.size() != p * p || rhs.xtx.size() != p * p ||
        lhs.xty.size() != p * q || rhs.xty.size() != p * q ||
        lhs.yty.size() != q * q || rhs.yty.size() != q * q) {
        ctx.set_error("moment subtraction received inconsistent raw buffers");
        return N4M_ERR_INVALID_ARGUMENT;
    }

    out = MomentStats{};
    out.n_samples = lhs.n_samples - rhs.n_samples;
    out.n_features = lhs.n_features;
    out.n_targets = lhs.n_targets;
    out.x_sum.assign(p, 0.0);
    out.y_sum.assign(q, 0.0);
    out.xtx.assign(p * p, 0.0);
    out.xty.assign(p * q, 0.0);
    out.yty.assign(q * q, 0.0);

    for (std::size_t i = 0; i < p; ++i) {
        out.x_sum[i] = lhs.x_sum[i] - rhs.x_sum[i];
    }
    for (std::size_t i = 0; i < q; ++i) {
        out.y_sum[i] = lhs.y_sum[i] - rhs.y_sum[i];
    }
    for (std::size_t i = 0; i < p * p; ++i) {
        out.xtx[i] = lhs.xtx[i] - rhs.xtx[i];
    }
    for (std::size_t i = 0; i < p * q; ++i) {
        out.xty[i] = lhs.xty[i] - rhs.xty[i];
    }
    for (std::size_t i = 0; i < q * q; ++i) {
        out.yty[i] = lhs.yty[i] - rhs.yty[i];
    }
    return recompute_centered_moments(ctx, out);
}

}  // namespace n4m::core
