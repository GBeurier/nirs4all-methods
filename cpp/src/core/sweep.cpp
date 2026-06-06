// SPDX-License-Identifier: CECILL-2.1
//
// Exact Ridge CV sweep over row-additive moments.

#include "core/sweep.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <vector>

#include "core/common/linalg.h"
#include "core/common/linalg.hpp"
#include "core/common/matrix_view.hpp"
#include "core/common/parallel.hpp"
#include "core/common/svd.h"
#if defined(N4M_USE_CUDA)
#  include "core/cuda_dispatch.hpp"
#endif
#include "core/model.hpp"
#include "core/moments.hpp"

namespace n4m::core {

namespace {

struct RidgeMomentFit {
    std::int64_t n_features{0};
    std::int64_t n_targets{0};
    std::vector<double> coefficients;  // p x q, original X scale
    std::vector<double> intercept;     // 1 x q
    std::vector<double> x_mean;        // 1 x p
    std::vector<double> x_scale;       // 1 x p
    std::vector<double> y_mean;        // 1 x q
};

struct Pls1MomentDesign {
    std::vector<double> C;
    std::vector<double> s;
    double yy{0.0};
    std::vector<double> x_mean;
    std::vector<double> x_scale;
    std::vector<double> y_mean;
    std::vector<double> y_scale;
};

struct FoldMaterializedDesign {
    std::vector<std::int64_t> train_rows;
    std::vector<double> X_train;
    std::vector<double> Y_train;
};

struct RidgeDualDesign {
    std::size_t n_samples{0};
    std::size_t n_features{0};
    std::size_t n_targets{0};
    std::size_t n_heldout{0};
    std::vector<double> X_work;  // centered/scaled train X, n x p
    std::vector<double> Y_work;  // centered train Y, n x q
    std::vector<double> K;       // X_work X_work', n x n
    std::vector<double> K_cross; // held-out X_work train X_work', h x n
    std::vector<double> x_mean;
    std::vector<double> y_mean;
    std::vector<double> x_scale;
};

struct RidgeMomentEigenPath {
    std::int64_t n_features{0};
    std::int64_t n_targets{0};
    std::vector<double> eigenvalues;    // p
    std::vector<double> eigenvectors;   // p x p, eigenvectors as columns
    std::vector<double> projected_xy;   // p x q = V.T @ xy_scaled
    std::vector<double> x_offset;       // p
    std::vector<double> y_offset;       // q
    std::vector<double> scale;          // p
};

[[nodiscard]] bool score_only_requested(const Config& cfg) noexcept {
    return cfg.aom_score_only != 0;
}

[[nodiscard]] bool product_overflows(std::size_t a, std::size_t b) noexcept {
    return b != 0 &&
           a > (std::numeric_limits<std::size_t>::max() / b);
}

constexpr std::int32_t kCudaPls1MomentDefaultMinDeviceFeatures = 1024;

[[nodiscard, maybe_unused]] std::size_t cuda_pls_min_device_features(
    const Config& cfg) noexcept {
    const auto threshold = cfg.cuda_pls_min_device_features > 0
        ? cfg.cuda_pls_min_device_features
        : kCudaPls1MomentDefaultMinDeviceFeatures;
    return static_cast<std::size_t>(threshold);
}

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
        ctx.set_error("sweep requires F64 or F32 X and Y matrices");
        return N4M_ERR_DTYPE_MISMATCH;
    }
    if (X.rows <= 1 || X.cols <= 0 || Y.rows <= 1 || Y.cols <= 0) {
        ctx.set_error("sweep requires at least two samples and non-empty X/Y");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (X.rows != Y.rows) {
        ctx.set_error("sweep: X and Y must have the same number of rows");
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

[[nodiscard]] n4m_status_t build_fold_ids(Context& ctx,
                                          std::int64_t n,
                                          std::int32_t requested_cv,
                                          const std::int32_t* given,
                                          std::int64_t n_given,
                                          std::vector<std::int32_t>& ids,
                                          std::int32_t& out_cv) {
    if (given == nullptr || n_given == 0) {
        if (requested_cv < 2 || requested_cv > n) {
            ctx.set_error("sweep cv must be in [2, n_samples] when fold_ids are not provided");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        ids.assign(static_cast<std::size_t>(n), 0);
        for (std::int64_t i = 0; i < n; ++i) {
            ids[static_cast<std::size_t>(i)] =
                static_cast<std::int32_t>((i * requested_cv) / n);
        }
        out_cv = requested_cv;
    } else {
        if (n_given != n) {
            ctx.set_error("sweep fold_ids length must equal n_samples");
            return N4M_ERR_SHAPE_MISMATCH;
        }
        ids.assign(given, given + static_cast<std::size_t>(n_given));
        std::int32_t max_id = -1;
        for (const auto id : ids) {
            if (id < 0) {
                ctx.set_error("sweep fold_ids must be non-negative");
                return N4M_ERR_INVALID_ARGUMENT;
            }
            max_id = std::max(max_id, id);
        }
        out_cv = (requested_cv > 0) ? requested_cv : (max_id + 1);
        if (out_cv < 2 || out_cv > n) {
            ctx.set_error("sweep inferred/provided cv must be in [2, n_samples]");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        for (const auto id : ids) {
            if (id >= out_cv) {
                ctx.set_error("sweep fold_ids contain id >= cv");
                return N4M_ERR_INVALID_ARGUMENT;
            }
        }
    }

    std::vector<std::int64_t> counts(static_cast<std::size_t>(out_cv), 0);
    for (const auto id : ids) {
        ++counts[static_cast<std::size_t>(id)];
    }
    for (std::int32_t f = 0; f < out_cv; ++f) {
        if (counts[static_cast<std::size_t>(f)] == 0) {
            ctx.set_error("sweep requires every fold to contain at least one row");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        if (counts[static_cast<std::size_t>(f)] == n) {
            ctx.set_error("sweep fold would leave an empty train set");
            return N4M_ERR_INVALID_ARGUMENT;
        }
    }
    return N4M_OK;
}

std::vector<std::vector<std::int64_t>> fold_rows_from_ids(
    const std::vector<std::int32_t>& ids,
    std::int32_t cv) {
    std::vector<std::vector<std::int64_t>> rows(static_cast<std::size_t>(cv));
    for (std::size_t i = 0; i < ids.size(); ++i) {
        rows[static_cast<std::size_t>(ids[i])].push_back(
            static_cast<std::int64_t>(i));
    }
    return rows;
}

std::vector<std::vector<std::int64_t>> train_rows_from_fold_rows(
    std::size_t n,
    const std::vector<std::vector<std::int64_t>>& fold_rows) {
    std::vector<std::vector<std::int64_t>> train_rows(fold_rows.size());
    std::vector<unsigned char> heldout(n, 0);
    for (std::size_t fold = 0; fold < fold_rows.size(); ++fold) {
        std::fill(heldout.begin(), heldout.end(), static_cast<unsigned char>(0));
        for (const auto row : fold_rows[fold]) {
            heldout[static_cast<std::size_t>(row)] = 1;
        }
        auto& rows = train_rows[fold];
        rows.reserve(n - fold_rows[fold].size());
        for (std::size_t i = 0; i < n; ++i) {
            if (heldout[i] == 0) rows.push_back(static_cast<std::int64_t>(i));
        }
    }
    return train_rows;
}

std::vector<FoldMaterializedDesign> build_fold_designs(
    const std::vector<double>& X,
    const std::vector<double>& Y,
    std::size_t p,
    std::size_t q,
    const std::vector<std::vector<std::int64_t>>& train_rows) {
    std::vector<FoldMaterializedDesign> out(train_rows.size());
    for (std::size_t fold = 0; fold < train_rows.size(); ++fold) {
        auto& d = out[fold];
        d.train_rows = train_rows[fold];
        const auto rows = d.train_rows.size();
        d.X_train.assign(rows * p, 0.0);
        d.Y_train.assign(rows * q, 0.0);
        for (std::size_t rr = 0; rr < rows; ++rr) {
            const auto src = static_cast<std::size_t>(d.train_rows[rr]);
            std::copy_n(X.data() + src * p, p, d.X_train.data() + rr * p);
            std::copy_n(Y.data() + src * q, q, d.Y_train.data() + rr * q);
        }
    }
    return out;
}

void column_means(const std::vector<double>& mat,
                  std::size_t rows,
                  std::size_t cols,
                  std::vector<double>& means) {
    means.assign(cols, 0.0);
    if (rows == 0) return;
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            means[c] += mat[r * cols + c];
        }
    }
    const double inv = 1.0 / static_cast<double>(rows);
    for (double& v : means) v *= inv;
}

[[nodiscard]] n4m_status_t prepare_ridge_dual_design(
    Context& ctx,
    const Config& cfg,
    const std::vector<double>& X_train,
    const std::vector<double>& Y_train,
    std::size_t rows,
    std::size_t p,
    std::size_t q,
    RidgeDualDesign& out) {
    if (rows == 0 || p == 0 || q == 0) {
        ctx.set_error("sweep dual Ridge requires non-empty train matrices");
        return N4M_ERR_INVALID_ARGUMENT;
    }

    out = RidgeDualDesign{};
    out.n_samples = rows;
    out.n_features = p;
    out.n_targets = q;
    out.X_work = X_train;
    out.Y_work = Y_train;
    out.x_mean.assign(p, 0.0);
    out.y_mean.assign(q, 0.0);
    out.x_scale.assign(p, 1.0);

    const bool fit_intercept = cfg.ridge_fit_intercept != 0;
    if (fit_intercept && cfg.center_x != 0) {
        column_means(out.X_work, rows, p, out.x_mean);
        for (std::size_t r = 0; r < rows; ++r) {
            for (std::size_t c = 0; c < p; ++c) {
                out.X_work[r * p + c] -= out.x_mean[c];
            }
        }
    }
    if (fit_intercept && cfg.center_y != 0) {
        column_means(out.Y_work, rows, q, out.y_mean);
        for (std::size_t r = 0; r < rows; ++r) {
            for (std::size_t c = 0; c < q; ++c) {
                out.Y_work[r * q + c] -= out.y_mean[c];
            }
        }
    }
    if (cfg.scale_x != 0) {
        for (std::size_t c = 0; c < p; ++c) {
            double ss = 0.0;
            for (std::size_t r = 0; r < rows; ++r) {
                const double v = out.X_work[r * p + c];
                ss += v * v;
            }
            const double scale = std::sqrt(ss / static_cast<double>(rows));
            out.x_scale[c] = (scale > 0.0) ? scale : 1.0;
        }
        for (std::size_t r = 0; r < rows; ++r) {
            for (std::size_t c = 0; c < p; ++c) {
                out.X_work[r * p + c] /= out.x_scale[c];
            }
        }
    }

    out.K.assign(rows * rows, 0.0);
    linalg::gemm(linalg::Trans_No, linalg::Trans_Yes,
                 rows, rows, p, 1.0,
                 out.X_work.data(), p,
                 out.X_work.data(), p,
                 0.0, out.K.data(), rows);
    return N4M_OK;
}

[[nodiscard]] n4m_status_t prepare_ridge_dual_cross(
    Context& ctx,
    const Config& cfg,
    const RidgeDualDesign& design,
    const std::vector<double>& X_all,
    const std::vector<std::int64_t>& heldout_rows,
    RidgeDualDesign& out) {
    const auto n = design.n_samples;
    const auto p = design.n_features;
    const auto h = heldout_rows.size();
    if (n == 0 || p == 0 || design.X_work.size() != n * p ||
        design.x_mean.size() != p || design.x_scale.size() != p) {
        ctx.set_error("sweep dual Ridge cross-kernel received inconsistent design");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (product_overflows(h, n)) {
        ctx.set_error("sweep dual Ridge cross-kernel dimensions overflow");
        return N4M_ERR_INVALID_ARGUMENT;
    }

    out = design;
    out.n_heldout = h;
    out.K_cross.assign(h * n, 0.0);
    std::vector<double> X_heldout_work(h * p, 0.0);
    const bool fit_intercept = cfg.ridge_fit_intercept != 0;
    const bool center_x = fit_intercept && cfg.center_x != 0;
    for (std::size_t rr = 0; rr < h; ++rr) {
        const auto src = static_cast<std::size_t>(heldout_rows[rr]);
        if ((src + 1U) * p > X_all.size()) {
            ctx.set_error("sweep dual Ridge cross-kernel row index out of range");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        for (std::size_t col = 0; col < p; ++col) {
            double value = X_all[src * p + col];
            if (center_x) value -= design.x_mean[col];
            X_heldout_work[rr * p + col] = value / design.x_scale[col];
        }
    }
    linalg::gemm(linalg::Trans_No, linalg::Trans_Yes,
                 h, n, p, 1.0,
                 X_heldout_work.data(), p,
                 design.X_work.data(), p,
                 0.0, out.K_cross.data(), n);
    return N4M_OK;
}

[[nodiscard]] bool should_use_ridge_dual_cross(std::size_t n_train,
                                               std::size_t n_heldout,
                                               std::size_t n_lambdas,
                                               std::size_t n_targets) noexcept {
    if (n_train == 0 || n_heldout == 0 || n_lambdas == 0 || n_targets == 0) {
        return false;
    }
    // Cross-kernel scoring precomputes O(h*n*p) work and saves
    // O(lambda*q*p*(n+h)) coefficient reconstruction/prediction work.
    return n_heldout * n_train <=
           n_lambdas * n_targets * (n_train + n_heldout);
}

[[nodiscard]] n4m_status_t solve_square_qr(const std::vector<double>& A,
                                           std::size_t m,
                                           const std::vector<double>& B,
                                           std::size_t q,
                                           std::vector<double>& X) {
    std::vector<double> Af = A;
    std::vector<double> tau(m, 0.0);
    n4m_status_t st = n4m_householder_qr(
        Af.data(), static_cast<std::int64_t>(m),
        static_cast<std::int64_t>(m), tau.data());
    if (st != N4M_OK) return st;

    X.assign(m * q, 0.0);
    std::vector<double> rhs(m, 0.0);
    std::vector<double> sol(m, 0.0);
    for (std::size_t target = 0; target < q; ++target) {
        for (std::size_t i = 0; i < m; ++i) {
            rhs[i] = B[i * q + target];
        }
        st = n4m_apply_qt(Af.data(), static_cast<std::int64_t>(m),
                          static_cast<std::int64_t>(m), tau.data(),
                          rhs.data());
        if (st != N4M_OK) return st;
        st = n4m_back_solve_R(Af.data(), static_cast<std::int64_t>(m),
                              static_cast<std::int64_t>(m), rhs.data(),
                              sol.data());
        if (st != N4M_OK) return st;
        for (std::size_t i = 0; i < m; ++i) {
            X[i * q + target] = sol[i];
        }
    }
    return N4M_OK;
}

[[nodiscard]] n4m_status_t predict_ridge_from_dual_design(
    Context& ctx,
    const RidgeDualDesign& design,
    double lambda,
    std::vector<double>& pred) {
    if (!std::isfinite(lambda) || lambda < 0.0) {
        ctx.set_error("sweep ridge lambda must be finite and >= 0");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    const std::size_t n = design.n_samples;
    const std::size_t h = design.n_heldout;
    const std::size_t q = design.n_targets;
    if (h == 0 || design.K_cross.size() != h * n ||
        design.K.size() != n * n ||
        design.Y_work.size() != n * q ||
        design.y_mean.size() != q) {
        ctx.set_error("sweep dual Ridge prediction received inconsistent design");
        return N4M_ERR_INVALID_ARGUMENT;
    }

    std::vector<double> K = design.K;
    for (std::size_t i = 0; i < n; ++i) K[i * n + i] += lambda;

    std::vector<double> alpha;
    n4m_status_t st = solve_square_qr(K, n, design.Y_work, q, alpha);
    if (st != N4M_OK) {
        ctx.set_error("sweep dual Ridge solve failed");
        return st;
    }

    pred.assign(h * q, 0.0);
    linalg::gemm(linalg::Trans_No, linalg::Trans_No,
                 h, q, n, 1.0,
                 design.K_cross.data(), n,
                 alpha.data(), q,
                 0.0, pred.data(), q);
    for (std::size_t row = 0; row < h; ++row) {
        for (std::size_t target = 0; target < q; ++target) {
            pred[row * q + target] += design.y_mean[target];
        }
    }
    return N4M_OK;
}

[[nodiscard]] n4m_status_t ridge_dual_cross_heldout_sse(
    Context& ctx,
    const RidgeDualDesign& design,
    const std::vector<double>& Y,
    std::size_t q,
    const std::vector<std::int64_t>& heldout_rows,
    double lambda,
    double& sse) {
    if (!std::isfinite(lambda) || lambda < 0.0) {
        ctx.set_error("sweep ridge lambda must be finite and >= 0");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    const std::size_t n = design.n_samples;
    const std::size_t h = design.n_heldout;
    if (q == 0 || q != design.n_targets ||
        h == 0 || h != heldout_rows.size() ||
        design.K_cross.size() != h * n ||
        design.K.size() != n * n ||
        design.Y_work.size() != n * q ||
        design.y_mean.size() != q) {
        ctx.set_error("sweep dual Ridge SSE received inconsistent design");
        return N4M_ERR_INVALID_ARGUMENT;
    }

    std::vector<double> K = design.K;
    for (std::size_t i = 0; i < n; ++i) K[i * n + i] += lambda;

    std::vector<double> alpha;
    n4m_status_t st = solve_square_qr(K, n, design.Y_work, q, alpha);
    if (st != N4M_OK) {
        ctx.set_error("sweep dual Ridge solve failed");
        return st;
    }

    std::vector<double> pred(h * q, 0.0);
    linalg::gemm(linalg::Trans_No, linalg::Trans_No,
                 h, q, n, 1.0,
                 design.K_cross.data(), n,
                 alpha.data(), q,
                 0.0, pred.data(), q);
    double local_sse = 0.0;
    for (std::size_t row = 0; row < h; ++row) {
        const auto src = static_cast<std::size_t>(heldout_rows[row]);
        if ((src + 1U) * q > Y.size()) {
            ctx.set_error("sweep dual Ridge SSE row index out of range");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        for (std::size_t target = 0; target < q; ++target) {
            const double value =
                design.y_mean[target] + pred[row * q + target];
            const double d = Y[src * q + target] - value;
            local_sse += d * d;
        }
    }
    sse = local_sse;
    return N4M_OK;
}

[[nodiscard]] n4m_status_t fit_ridge_from_dual_design(Context& ctx,
                                                      const RidgeDualDesign& design,
                                                      double lambda,
                                                      RidgeMomentFit& out) {
    if (!std::isfinite(lambda) || lambda < 0.0) {
        ctx.set_error("sweep ridge lambda must be finite and >= 0");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    const std::size_t n = design.n_samples;
    const std::size_t p = design.n_features;
    const std::size_t q = design.n_targets;

    std::vector<double> K = design.K;
    for (std::size_t i = 0; i < n; ++i) K[i * n + i] += lambda;

    std::vector<double> alpha;
    n4m_status_t st = solve_square_qr(K, n, design.Y_work, q, alpha);
    if (st != N4M_OK) {
        ctx.set_error("sweep dual Ridge solve failed");
        return st;
    }

    std::vector<double> beta_scaled(p * q, 0.0);
    linalg::gemm(linalg::Trans_Yes, linalg::Trans_No,
                 p, q, n, 1.0,
                 design.X_work.data(), p,
                 alpha.data(), q,
                 0.0, beta_scaled.data(), q);

    out = RidgeMomentFit{};
    out.n_features = static_cast<std::int64_t>(p);
    out.n_targets = static_cast<std::int64_t>(q);
    out.x_mean = design.x_mean;
    out.y_mean = design.y_mean;
    out.x_scale = design.x_scale;
    out.coefficients.assign(p * q, 0.0);
    for (std::size_t c = 0; c < p; ++c) {
        for (std::size_t target = 0; target < q; ++target) {
            out.coefficients[c * q + target] =
                beta_scaled[c * q + target] / out.x_scale[c];
        }
    }

    out.intercept.assign(q, 0.0);
    for (std::size_t target = 0; target < q; ++target) {
        double v = out.y_mean[target];
        for (std::size_t c = 0; c < p; ++c) {
            v -= out.x_mean[c] * out.coefficients[c * q + target];
        }
        out.intercept[target] = v;
    }
    return N4M_OK;
}

void copy_model_as_linear_fit(const Model& model, RidgeMomentFit& out) {
    const auto p = static_cast<std::size_t>(model.n_features);
    const auto q = static_cast<std::size_t>(model.n_targets);
    out = RidgeMomentFit{};
    out.n_features = model.n_features;
    out.n_targets = model.n_targets;
    out.coefficients = model.coefficients;
    out.x_mean = model.x_mean;
    out.x_scale = model.x_scale;
    out.y_mean = model.y_mean;
    out.intercept.assign(q, 0.0);
    for (std::size_t target = 0; target < q; ++target) {
        double v = out.y_mean[target];
        for (std::size_t feature = 0; feature < p; ++feature) {
            v -= out.x_mean[feature] * out.coefficients[feature * q + target];
        }
        out.intercept[target] = v;
    }
}

[[nodiscard]] n4m_status_t copy_model_prefix_as_linear_fit(Context& ctx,
                                                           const Model& model,
                                                           std::int32_t n_components,
                                                           RidgeMomentFit& out) {
    const auto p = static_cast<std::size_t>(model.n_features);
    const auto q = static_cast<std::size_t>(model.n_targets);
    const auto max_components = static_cast<std::size_t>(model.n_components);
    const auto k = static_cast<std::size_t>(n_components);
    if (k == 0 || k > max_components) {
        ctx.set_error("sweep PLS prefix component count is invalid");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (model.weights_w.size() != p * max_components ||
        model.loadings_p.size() != p * max_components ||
        model.y_loadings_q.size() != q * max_components ||
        model.x_mean.size() != p ||
        model.x_scale.size() != p ||
        model.y_mean.size() != q ||
        model.y_scale.size() != q) {
        ctx.set_error("sweep PLS model is missing prefix arrays");
        return N4M_ERR_INTERNAL;
    }

    std::vector<double> ptw(k * k, 0.0);
    for (std::size_t row_comp = 0; row_comp < k; ++row_comp) {
        for (std::size_t col_comp = 0; col_comp < k; ++col_comp) {
            double sum = 0.0;
            for (std::size_t feature = 0; feature < p; ++feature) {
                sum += model.loadings_p[feature * max_components + row_comp] *
                       model.weights_w[feature * max_components + col_comp];
            }
            ptw[row_comp * k + col_comp] = sum;
        }
    }

    std::vector<double> identity(k * k, 0.0);
    for (std::size_t i = 0; i < k; ++i) {
        identity[i * k + i] = 1.0;
    }
    std::vector<double> ptw_inv;
    n4m_status_t st = solve_square_qr(ptw, k, identity, k, ptw_inv);
    if (st != N4M_OK) {
        ctx.set_error("sweep PLS prefix failed to invert P.T @ W");
        return st;
    }

    out = RidgeMomentFit{};
    out.n_features = model.n_features;
    out.n_targets = model.n_targets;
    out.x_mean = model.x_mean;
    out.x_scale = model.x_scale;
    out.y_mean = model.y_mean;
    out.coefficients.assign(p * q, 0.0);
    for (std::size_t feature = 0; feature < p; ++feature) {
        for (std::size_t target = 0; target < q; ++target) {
            double sum = 0.0;
            for (std::size_t comp = 0; comp < k; ++comp) {
                double rotation = 0.0;
                for (std::size_t inner = 0; inner < k; ++inner) {
                    rotation += model.weights_w[feature * max_components + inner] *
                                ptw_inv[inner * k + comp];
                }
                sum += rotation * model.y_loadings_q[target * max_components + comp];
            }
            out.coefficients[feature * q + target] =
                sum * model.y_scale[target] / model.x_scale[feature];
        }
    }

    out.intercept.assign(q, 0.0);
    for (std::size_t target = 0; target < q; ++target) {
        double v = out.y_mean[target];
        for (std::size_t feature = 0; feature < p; ++feature) {
            v -= out.x_mean[feature] * out.coefficients[feature * q + target];
        }
        out.intercept[target] = v;
    }
    return N4M_OK;
}

[[nodiscard]] n4m_status_t fit_pls_materialized(Context& ctx,
                                                const Config& cfg,
                                                const std::vector<double>& X_train,
                                                const std::vector<double>& Y_train,
                                                std::size_t rows,
                                                std::size_t p,
                                                std::size_t q,
                                                std::int32_t n_components,
                                                RidgeMomentFit& out) {
    Config local = cfg;
    local.algorithm = N4M_ALGO_PLS_REGRESSION;
    local.deflation = N4M_DEFLATION_REGRESSION;
    local.n_components = n_components;
    n4m_matrix_view_t Xv{const_cast<double*>(X_train.data()),
                         static_cast<std::int64_t>(rows),
                         static_cast<std::int64_t>(p),
                         static_cast<std::int64_t>(p),
                         1,
                         N4M_DTYPE_F64,
                         0};
    n4m_matrix_view_t Yv{const_cast<double*>(Y_train.data()),
                         static_cast<std::int64_t>(rows),
                         static_cast<std::int64_t>(q),
                         static_cast<std::int64_t>(q),
                         1,
                         N4M_DTYPE_F64,
                         0};
    std::unique_ptr<Model> model;
    const n4m_status_t st = fit_model(ctx, local, Xv, Yv, model);
    if (st != N4M_OK) return st;
    if (!model || model->coefficients.size() != p * q) {
        ctx.set_error("sweep PLS fit returned inconsistent coefficients");
        return N4M_ERR_INTERNAL;
    }
    copy_model_as_linear_fit(*model, out);
    return N4M_OK;
}

[[nodiscard]] n4m_status_t fit_pls_model_materialized(
    Context& ctx,
    const Config& cfg,
    const std::vector<double>& X_train,
    const std::vector<double>& Y_train,
    std::size_t rows,
    std::size_t p,
    std::size_t q,
    std::int32_t n_components,
    std::unique_ptr<Model>& model) {
    Config local = cfg;
    local.algorithm = N4M_ALGO_PLS_REGRESSION;
    local.deflation = N4M_DEFLATION_REGRESSION;
    local.n_components = n_components;
    n4m_matrix_view_t Xv{const_cast<double*>(X_train.data()),
                         static_cast<std::int64_t>(rows),
                         static_cast<std::int64_t>(p),
                         static_cast<std::int64_t>(p),
                         1,
                         N4M_DTYPE_F64,
                         0};
    n4m_matrix_view_t Yv{const_cast<double*>(Y_train.data()),
                         static_cast<std::int64_t>(rows),
                         static_cast<std::int64_t>(q),
                         static_cast<std::int64_t>(q),
                         1,
                         N4M_DTYPE_F64,
                         0};
    const n4m_status_t st = fit_model(ctx, local, Xv, Yv, model);
    if (st != N4M_OK) return st;
    if (!model || model->coefficients.size() != p * q) {
        ctx.set_error("sweep PLS fit returned inconsistent coefficients");
        return N4M_ERR_INTERNAL;
    }
    return N4M_OK;
}

void design_cross_moments(const Config& cfg,
                          const MomentStats& stats,
                          std::vector<double>& xx,
                          std::vector<double>& xy,
                          std::vector<double>& x_offset,
                          std::vector<double>& y_offset) {
    const auto p = static_cast<std::size_t>(stats.n_features);
    const auto q = static_cast<std::size_t>(stats.n_targets);
    const bool fit_intercept = cfg.ridge_fit_intercept != 0;
    const bool center_x = fit_intercept && cfg.center_x != 0;
    const bool center_y = fit_intercept && cfg.center_y != 0;

    x_offset.assign(p, 0.0);
    y_offset.assign(q, 0.0);
    if (center_x) x_offset = stats.x_mean;
    if (center_y) y_offset = stats.y_mean;

    if (center_x) {
        xx = stats.cxx;
    } else {
        xx = stats.xtx;
    }

    xy.assign(p * q, 0.0);
    for (std::size_t i = 0; i < p; ++i) {
        for (std::size_t j = 0; j < q; ++j) {
            double v = stats.xty[i * q + j];
            if (center_y) v -= stats.x_sum[i] * y_offset[j];
            if (center_x) v -= x_offset[i] * stats.y_sum[j];
            if (center_x && center_y) {
                v += static_cast<double>(stats.n_samples) * x_offset[i] * y_offset[j];
            }
            xy[i * q + j] = v;
        }
    }
}

[[nodiscard]] n4m_status_t fit_ridge_from_moments(Context& ctx,
                                                  const Config& cfg,
                                                  const MomentStats& stats,
                                                  double lambda,
                                                  RidgeMomentFit& out) {
    if (!std::isfinite(lambda) || lambda < 0.0) {
        ctx.set_error("sweep ridge lambda must be finite and >= 0");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    const auto p = static_cast<std::size_t>(stats.n_features);
    const auto q = static_cast<std::size_t>(stats.n_targets);

    std::vector<double> xx;
    std::vector<double> xy;
    std::vector<double> x_offset;
    std::vector<double> y_offset;
    design_cross_moments(cfg, stats, xx, xy, x_offset, y_offset);

    std::vector<double> scale(p, 1.0);
    if (cfg.scale_x != 0) {
        const double inv_n = 1.0 / static_cast<double>(stats.n_samples);
        for (std::size_t i = 0; i < p; ++i) {
            const double var = std::max(0.0, xx[i * p + i] * inv_n);
            const double s = std::sqrt(var);
            scale[i] = (s > 0.0) ? s : 1.0;
        }
        for (std::size_t i = 0; i < p; ++i) {
            for (std::size_t j = 0; j < p; ++j) {
                xx[i * p + j] /= (scale[i] * scale[j]);
            }
            for (std::size_t j = 0; j < q; ++j) {
                xy[i * q + j] /= scale[i];
            }
        }
    }
    for (std::size_t i = 0; i < p; ++i) {
        xx[i * p + i] += lambda;
    }

    std::vector<double> beta_scaled;
    n4m_status_t st = solve_square_qr(xx, p, xy, q, beta_scaled);
    if (st != N4M_OK) {
        ctx.set_error("sweep ridge solve failed");
        return st;
    }

    out = RidgeMomentFit{};
    out.n_features = stats.n_features;
    out.n_targets = stats.n_targets;
    out.x_mean = std::move(x_offset);
    out.y_mean = std::move(y_offset);
    out.x_scale = std::move(scale);
    out.coefficients.assign(p * q, 0.0);
    for (std::size_t i = 0; i < p; ++i) {
        for (std::size_t j = 0; j < q; ++j) {
            out.coefficients[i * q + j] = beta_scaled[i * q + j] / out.x_scale[i];
        }
    }

    out.intercept.assign(q, 0.0);
    for (std::size_t j = 0; j < q; ++j) {
        double v = out.y_mean[j];
        for (std::size_t i = 0; i < p; ++i) {
            v -= out.x_mean[i] * out.coefficients[i * q + j];
        }
        out.intercept[j] = v;
    }
    return N4M_OK;
}

[[nodiscard]] n4m_status_t prepare_ridge_moment_eigen_path(
    Context& ctx,
    const Config& cfg,
    const MomentStats& stats,
    RidgeMomentEigenPath& out) {
    const auto p = static_cast<std::size_t>(stats.n_features);
    const auto q = static_cast<std::size_t>(stats.n_targets);
    if (p == 0 || q == 0) {
        ctx.set_error("ridge moment eigen path requires p>0 and q>0");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (product_overflows(p, p) || product_overflows(p, q)) {
        ctx.set_error("ridge moment eigen path dimensions are too large");
        return N4M_ERR_OUT_OF_MEMORY;
    }

    std::vector<double> xx;
    std::vector<double> xy;
    std::vector<double> x_offset;
    std::vector<double> y_offset;
    design_cross_moments(cfg, stats, xx, xy, x_offset, y_offset);
    if (xx.size() != p * p || xy.size() != p * q ||
        x_offset.size() != p || y_offset.size() != q) {
        ctx.set_error("ridge moment eigen path received inconsistent moments");
        return N4M_ERR_INVALID_ARGUMENT;
    }

    std::vector<double> scale(p, 1.0);
    if (cfg.scale_x != 0) {
        const double inv_n = 1.0 / static_cast<double>(stats.n_samples);
        for (std::size_t i = 0; i < p; ++i) {
            const double var = std::max(0.0, xx[i * p + i] * inv_n);
            const double s = std::sqrt(var);
            scale[i] = (s > 0.0) ? s : 1.0;
        }
        for (std::size_t i = 0; i < p; ++i) {
            for (std::size_t j = 0; j < p; ++j) {
                xx[i * p + j] /= (scale[i] * scale[j]);
            }
            for (std::size_t j = 0; j < q; ++j) {
                xy[i * q + j] /= scale[i];
            }
        }
    }

    for (std::size_t i = 0; i < p; ++i) {
        for (std::size_t j = i + 1; j < p; ++j) {
            const double sym = 0.5 * (xx[i * p + j] + xx[j * p + i]);
            xx[i * p + j] = sym;
            xx[j * p + i] = sym;
        }
    }

    RidgeMomentEigenPath path{};
    path.n_features = stats.n_features;
    path.n_targets = stats.n_targets;
    path.eigenvalues.assign(p, 0.0);
    path.eigenvectors.assign(p * p, 0.0);
    n4m_status_t st = n4m_symmetric_eigh(
        xx.data(), static_cast<std::int64_t>(p), path.eigenvalues.data(),
        path.eigenvectors.data());
    if (st != N4M_OK) {
        ctx.set_error("ridge moment eigen path decomposition failed");
        return st;
    }

    path.projected_xy.assign(p * q, 0.0);
    for (std::size_t eig = 0; eig < p; ++eig) {
        for (std::size_t target = 0; target < q; ++target) {
            double acc = 0.0;
            for (std::size_t row = 0; row < p; ++row) {
                acc += path.eigenvectors[row * p + eig] *
                       xy[row * q + target];
            }
            path.projected_xy[eig * q + target] = acc;
        }
    }
    path.x_offset = std::move(x_offset);
    path.y_offset = std::move(y_offset);
    path.scale = std::move(scale);
    out = std::move(path);
    return N4M_OK;
}

[[nodiscard]] n4m_status_t fit_ridge_from_eigen_path(
    Context& ctx,
    const RidgeMomentEigenPath& path,
    double lambda,
    RidgeMomentFit& out) {
    if (!std::isfinite(lambda) || lambda < 0.0) {
        ctx.set_error("sweep ridge lambda must be finite and >= 0");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    const auto p = static_cast<std::size_t>(path.n_features);
    const auto q = static_cast<std::size_t>(path.n_targets);
    if (p == 0 || q == 0 ||
        path.eigenvalues.size() != p ||
        path.eigenvectors.size() != p * p ||
        path.projected_xy.size() != p * q ||
        path.x_offset.size() != p ||
        path.y_offset.size() != q ||
        path.scale.size() != p) {
        ctx.set_error("ridge moment eigen path is inconsistent");
        return N4M_ERR_INVALID_ARGUMENT;
    }

    double max_abs_eig = 0.0;
    for (const double eig : path.eigenvalues) {
        if (!std::isfinite(eig)) {
            ctx.set_error("ridge moment eigen path has non-finite eigenvalues");
            return N4M_ERR_NUMERICAL_FAILURE;
        }
        max_abs_eig = std::max(max_abs_eig, std::abs(eig));
    }
    const double tol = 64.0 * std::numeric_limits<double>::epsilon() *
                       std::max(1.0, max_abs_eig + std::abs(lambda));

    std::vector<double> beta_scaled(p * q, 0.0);
    for (std::size_t eig_ix = 0; eig_ix < p; ++eig_ix) {
        double eig = path.eigenvalues[eig_ix];
        if (eig < 0.0) {
            if (std::abs(eig) <= tol) {
                eig = 0.0;
            } else {
                ctx.set_error("ridge moment eigen path is not positive semidefinite");
                return N4M_ERR_NUMERICAL_FAILURE;
            }
        }
        const double denom = eig + lambda;
        if (!std::isfinite(denom) || std::abs(denom) <= tol) {
            ctx.set_error("ridge moment eigen path solve is singular");
            return N4M_ERR_NUMERICAL_FAILURE;
        }
        for (std::size_t target = 0; target < q; ++target) {
            const double factor =
                path.projected_xy[eig_ix * q + target] / denom;
            for (std::size_t row = 0; row < p; ++row) {
                beta_scaled[row * q + target] +=
                    path.eigenvectors[row * p + eig_ix] * factor;
            }
        }
    }

    out = RidgeMomentFit{};
    out.n_features = path.n_features;
    out.n_targets = path.n_targets;
    out.x_mean = path.x_offset;
    out.y_mean = path.y_offset;
    out.x_scale = path.scale;
    out.coefficients.assign(p * q, 0.0);
    for (std::size_t i = 0; i < p; ++i) {
        for (std::size_t j = 0; j < q; ++j) {
            out.coefficients[i * q + j] = beta_scaled[i * q + j] /
                                          out.x_scale[i];
        }
    }

    out.intercept.assign(q, 0.0);
    for (std::size_t j = 0; j < q; ++j) {
        double v = out.y_mean[j];
        for (std::size_t i = 0; i < p; ++i) {
            v -= out.x_mean[i] * out.coefficients[i * q + j];
        }
        out.intercept[j] = v;
    }
    return N4M_OK;
}

[[nodiscard]] n4m_status_t fit_ridge_from_moment_path_or_direct(
    Context& ctx,
    const Config& cfg,
    const MomentStats& stats,
    const RidgeMomentEigenPath* path,
    double lambda,
    RidgeMomentFit& out,
    bool* used_eigen_path = nullptr) {
    if (used_eigen_path != nullptr) {
        *used_eigen_path = false;
    }
    if (path != nullptr) {
        Context path_ctx;
        const n4m_status_t path_st =
            fit_ridge_from_eigen_path(path_ctx, *path, lambda, out);
        if (path_st == N4M_OK) {
            if (used_eigen_path != nullptr) {
                *used_eigen_path = true;
            }
            return N4M_OK;
        }
    }
    return fit_ridge_from_moments(ctx, cfg, stats, lambda, out);
}

void predict_rows(const std::vector<double>& X,
                  std::size_t p,
                  const std::vector<std::int64_t>& rows,
                  const RidgeMomentFit& fit,
                  std::vector<double>& out,
                  std::size_t out_row_offset,
                  std::size_t out_row_stride) {
    const auto q = static_cast<std::size_t>(fit.n_targets);
    for (std::size_t rr = 0; rr < rows.size(); ++rr) {
        const auto src = static_cast<std::size_t>(rows[rr]);
        const double* x = X.data() + src * p;
        double* yhat = out.data() + (out_row_offset + rr) * out_row_stride;
        for (std::size_t target = 0; target < q; ++target) {
            double acc = fit.intercept[target];
            for (std::size_t col = 0; col < p; ++col) {
                acc += x[col] * fit.coefficients[col * q + target];
            }
            yhat[target] = acc;
        }
    }
}

void predict_all(const std::vector<double>& X,
                 std::size_t n,
                 std::size_t p,
                 const RidgeMomentFit& fit,
                 std::vector<double>& out) {
    std::vector<std::int64_t> rows(n, 0);
    for (std::size_t i = 0; i < n; ++i) rows[i] = static_cast<std::int64_t>(i);
    predict_rows(X, p, rows, fit, out, 0, static_cast<std::size_t>(fit.n_targets));
}

double heldout_sse(const std::vector<double>& Y,
                   std::size_t q,
                   const std::vector<std::int64_t>& rows,
                   const std::vector<double>& pred) {
    double sse = 0.0;
    for (std::size_t rr = 0; rr < rows.size(); ++rr) {
        const auto src = static_cast<std::size_t>(rows[rr]);
        for (std::size_t target = 0; target < q; ++target) {
            const double d = Y[src * q + target] - pred[rr * q + target];
            sse += d * d;
        }
    }
    return sse;
}

double heldout_sse_from_fit(const std::vector<double>& X,
                            const std::vector<double>& Y,
                            std::size_t p,
                            std::size_t q,
                            const std::vector<std::int64_t>& rows,
                            const RidgeMomentFit& fit) {
    double sse = 0.0;
    for (const auto row_id : rows) {
        const auto src = static_cast<std::size_t>(row_id);
        const double* x = X.data() + src * p;
        const double* y = Y.data() + src * q;
        for (std::size_t target = 0; target < q; ++target) {
            double acc = fit.intercept[target];
            for (std::size_t col = 0; col < p; ++col) {
                acc += x[col] * fit.coefficients[col * q + target];
            }
            const double d = y[target] - acc;
            sse += d * d;
        }
    }
    return sse;
}

double ridge_heldout_sse_from_moments(const MomentStats& heldout,
                                      const RidgeMomentFit& fit) {
    const auto p = static_cast<std::size_t>(heldout.n_features);
    const auto q = static_cast<std::size_t>(heldout.n_targets);
#if defined(N4M_USE_BLAS) && !defined(N4M_USE_CUDA)
    constexpr std::size_t kMomentSseBlasMinFeatures = 64;
    std::vector<double> xtx_beta;
    const bool use_blas_quadratic =
        q == 1U && p >= kMomentSseBlasMinFeatures &&
        heldout.xtx.size() == p * p &&
        fit.coefficients.size() == p;
    if (use_blas_quadratic) {
        xtx_beta.assign(p, 0.0);
    }
#endif
    double sse = 0.0;
    for (std::size_t target = 0; target < q; ++target) {
        const double intercept = fit.intercept[target];
        double target_sse = heldout.yty[target * q + target];
        target_sse -= 2.0 * intercept * heldout.y_sum[target];
        target_sse += static_cast<double>(heldout.n_samples) *
                      intercept * intercept;

        double xb_sum = 0.0;
        double yxb = 0.0;
        double xbx = 0.0;
#if defined(N4M_USE_BLAS) && !defined(N4M_USE_CUDA)
        if (use_blas_quadratic && target == 0U) {
            const double* beta = fit.coefficients.data();
            linalg::gemv(linalg::Trans_No, p, p, 1.0,
                         heldout.xtx.data(), beta, 0.0, xtx_beta.data());
            for (std::size_t i = 0; i < p; ++i) {
                const double bi = beta[i];
                xb_sum += heldout.x_sum[i] * bi;
                yxb += heldout.xty[i] * bi;
                xbx += bi * xtx_beta[i];
            }
        } else
#endif
        {
            for (std::size_t i = 0; i < p; ++i) {
                const double bi = fit.coefficients[i * q + target];
                xb_sum += heldout.x_sum[i] * bi;
                yxb += heldout.xty[i * q + target] * bi;
                for (std::size_t j = 0; j < p; ++j) {
                    const double bj = fit.coefficients[j * q + target];
                    xbx += bi * heldout.xtx[i * p + j] * bj;
                }
            }
        }
        target_sse -= 2.0 * yxb;
        target_sse += 2.0 * intercept * xb_sum;
        target_sse += xbx;
        sse += target_sse;
    }
    if (sse < 0.0 && sse > -1e-9) return 0.0;
    return sse;
}

[[nodiscard]] n4m_status_t pls1_design_moments(
    Context& ctx,
    const Config& cfg,
    const MomentStats& stats,
    std::vector<double>& xx,
    std::vector<double>& xy,
    double& yy,
    std::vector<double>& x_mean,
    std::vector<double>& x_scale,
    std::vector<double>& y_mean,
    std::vector<double>& y_scale) {
    if (stats.n_samples <= 1 || stats.n_features <= 0 ||
        stats.n_targets != 1) {
        ctx.set_error("PLS1 moment fit requires n>1, p>0 and q=1");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    const auto p = static_cast<std::size_t>(stats.n_features);
    if (stats.x_sum.size() != p || stats.y_sum.size() != 1U ||
        stats.xtx.size() != p * p || stats.xty.size() != p ||
        stats.yty.size() != 1U || stats.cxx.size() != p * p ||
        stats.cxy.size() != p || stats.cyy.size() != 1U) {
        ctx.set_error("PLS1 moment fit received inconsistent moment buffers");
        return N4M_ERR_INVALID_ARGUMENT;
    }

    x_mean.assign(p, 0.0);
    y_mean.assign(1, 0.0);
    if (cfg.center_x != 0) x_mean = stats.x_mean;
    if (cfg.center_y != 0) y_mean[0] = stats.y_mean[0];

    xx.assign(p * p, 0.0);
    if (cfg.center_x != 0) {
        xx = stats.cxx;
    } else {
        xx = stats.xtx;
    }

    xy.assign(p, 0.0);
    for (std::size_t i = 0; i < p; ++i) {
        double v = stats.xty[i];
        if (cfg.center_y != 0) v -= stats.x_sum[i] * y_mean[0];
        if (cfg.center_x != 0) v -= x_mean[i] * stats.y_sum[0];
        if (cfg.center_x != 0 && cfg.center_y != 0) {
            v += static_cast<double>(stats.n_samples) * x_mean[i] * y_mean[0];
        }
        xy[i] = v;
    }

    yy = stats.yty[0];
    if (cfg.center_y != 0) {
        yy -= 2.0 * y_mean[0] * stats.y_sum[0];
        yy += static_cast<double>(stats.n_samples) * y_mean[0] * y_mean[0];
    }
    if (yy < 0.0 && yy > -1e-9) yy = 0.0;

    x_scale.assign(p, 1.0);
    y_scale.assign(1, 1.0);
    if (cfg.scale_x != 0) {
        const double denom =
            static_cast<double>(std::max<std::int64_t>(stats.n_samples - 1, 1));
        for (std::size_t i = 0; i < p; ++i) {
            const double var = std::max(0.0, xx[i * p + i] / denom);
            const double s = std::sqrt(var);
            x_scale[i] = (s > 0.0 && std::isfinite(s)) ? s : 1.0;
        }
    }
    if (cfg.scale_y != 0) {
        const double denom =
            static_cast<double>(std::max<std::int64_t>(stats.n_samples - 1, 1));
        const double var = std::max(0.0, yy / denom);
        const double s = std::sqrt(var);
        y_scale[0] = (s > 0.0 && std::isfinite(s)) ? s : 1.0;
    }

    for (std::size_t i = 0; i < p; ++i) {
        for (std::size_t j = 0; j < p; ++j) {
            xx[i * p + j] /= (x_scale[i] * x_scale[j]);
        }
        xy[i] /= (x_scale[i] * y_scale[0]);
    }
    yy /= (y_scale[0] * y_scale[0]);
    if (yy < 0.0 && yy > -1e-9) yy = 0.0;
    return N4M_OK;
}

[[nodiscard]] n4m_status_t pls1_prefix_fit_from_components(
    Context& ctx,
    const std::vector<double>& W,
    const std::vector<double>& P,
    const std::vector<double>& Q,
    std::size_t p,
    std::size_t max_components,
    std::size_t n_components,
    const std::vector<double>& x_mean,
    const std::vector<double>& x_scale,
    const std::vector<double>& y_mean,
    const std::vector<double>& y_scale,
    RidgeMomentFit& out) {
    if (n_components == 0 || n_components > max_components) {
        ctx.set_error("PLS1 moment prefix component count is invalid");
        return N4M_ERR_INVALID_ARGUMENT;
    }

    std::vector<double> ptw(n_components * n_components, 0.0);
#if defined(N4M_USE_CUDA)
    for (std::size_t row_comp = 0; row_comp < n_components; ++row_comp) {
        for (std::size_t col_comp = 0; col_comp < n_components; ++col_comp) {
            double sum = 0.0;
            for (std::size_t feature = 0; feature < p; ++feature) {
                sum += P[feature * max_components + row_comp] *
                       W[feature * max_components + col_comp];
            }
            ptw[row_comp * n_components + col_comp] = sum;
        }
    }
#else
    std::vector<double> W_prefix(p * n_components, 0.0);
    std::vector<double> P_prefix(p * n_components, 0.0);
    for (std::size_t feature = 0; feature < p; ++feature) {
        for (std::size_t comp = 0; comp < n_components; ++comp) {
            W_prefix[feature * n_components + comp] =
                W[feature * max_components + comp];
            P_prefix[feature * n_components + comp] =
                P[feature * max_components + comp];
        }
    }
    linalg::gemm(linalg::Trans_Yes, linalg::Trans_No,
                 n_components, n_components, p, 1.0,
                 P_prefix.data(), n_components,
                 W_prefix.data(), n_components,
                 0.0, ptw.data(), n_components);
#endif

    std::vector<double> identity(n_components * n_components, 0.0);
    for (std::size_t i = 0; i < n_components; ++i) {
        identity[i * n_components + i] = 1.0;
    }
    std::vector<double> ptw_inv;
    n4m_status_t st = solve_square_qr(
        ptw, n_components, identity, n_components, ptw_inv);
    if (st != N4M_OK) {
        ctx.set_error("PLS1 moment prefix failed to invert P.T @ W");
        return st;
    }

    out = RidgeMomentFit{};
    out.n_features = static_cast<std::int64_t>(p);
    out.n_targets = 1;
    out.x_mean = x_mean;
    out.x_scale = x_scale;
    out.y_mean = y_mean;
    out.coefficients.assign(p, 0.0);
#if defined(N4M_USE_CUDA)
    for (std::size_t feature = 0; feature < p; ++feature) {
        double coef_std = 0.0;
        for (std::size_t comp = 0; comp < n_components; ++comp) {
            double rotation = 0.0;
            for (std::size_t inner = 0; inner < n_components; ++inner) {
                rotation += W[feature * max_components + inner] *
                            ptw_inv[inner * n_components + comp];
            }
            coef_std += rotation * Q[comp];
        }
        out.coefficients[feature] =
            coef_std * y_scale[0] / out.x_scale[feature];
    }
#else
    std::vector<double> rotations(p * n_components, 0.0);
    linalg::gemm(linalg::Trans_No, linalg::Trans_No,
                 p, n_components, n_components, 1.0,
                 W_prefix.data(), n_components,
                 ptw_inv.data(), n_components,
                 0.0, rotations.data(), n_components);
    std::vector<double> coef_std(p, 0.0);
    linalg::gemv(linalg::Trans_No, p, n_components, 1.0,
                 rotations.data(), Q.data(), 0.0, coef_std.data());
    for (std::size_t feature = 0; feature < p; ++feature) {
        out.coefficients[feature] =
            coef_std[feature] * y_scale[0] / out.x_scale[feature];
    }
#endif

    out.intercept.assign(1, y_mean[0]);
    for (std::size_t feature = 0; feature < p; ++feature) {
        out.intercept[0] -= x_mean[feature] * out.coefficients[feature];
    }
    return N4M_OK;
}

[[nodiscard]] n4m_status_t fit_pls1_moment_prefixes(
    Context& ctx,
    const Config& cfg,
    const MomentStats& stats,
    std::int32_t max_components,
    std::vector<RidgeMomentFit>& prefix_fits,
    bool* used_cuda_device_components = nullptr) {
    if (used_cuda_device_components != nullptr) {
        *used_cuda_device_components = false;
    }
    if (max_components < 1) {
        ctx.set_error("PLS1 moment fit requires at least one component");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    const auto p = static_cast<std::size_t>(stats.n_features);
    const auto a = static_cast<std::size_t>(max_components);

    std::vector<double> C;
    std::vector<double> s;
    double yy = 0.0;
    std::vector<double> x_mean;
    std::vector<double> x_scale;
    std::vector<double> y_mean;
    std::vector<double> y_scale;
    n4m_status_t st = pls1_design_moments(
        ctx, cfg, stats, C, s, yy, x_mean, x_scale, y_mean, y_scale);
    if (st != N4M_OK) return st;

    std::vector<double> W(p * a, 0.0);
    std::vector<double> P(p * a, 0.0);
    std::vector<double> Q(a, 0.0);
    constexpr double eps = std::numeric_limits<double>::epsilon();

    bool pls1_components_ready = false;
#if defined(N4M_USE_CUDA)
    // The cuBLAS component loop amortizes host/device copies only on very wide
    // moment screens; medium-width PLS1 is faster on the scalar host loop.
    if (p >= cuda_pls_min_device_features(cfg) &&
        ::n4m::cuda_dispatch::cuda_runtime_available()) {
        std::string cuda_error;
        double yy_after = yy;
        const int cuda_status = ::n4m::cuda_dispatch::pls1_moment_components(
            p, a, C.data(), s.data(), yy, eps, W.data(), P.data(), Q.data(),
            &yy_after, &cuda_error);
        if (cuda_status != 0) {
            if (!cuda_error.empty()) {
                ctx.set_error(cuda_error.c_str());
            } else {
                ctx.set_error("CUDA PLS1 moment component loop failed");
            }
            return (cuda_status == 1)
                ? N4M_ERR_NUMERICAL_FAILURE
                : N4M_ERR_BACKEND_UNAVAILABLE;
        }
        yy = yy_after;
        pls1_components_ready = true;
        if (used_cuda_device_components != nullptr) {
            *used_cuda_device_components = true;
        }
    }
#endif
    if (!pls1_components_ready) {
        for (std::size_t comp = 0; comp < a; ++comp) {
            if (yy <= eps) {
                ctx.set_error("PLS1 moment Y residual vanished");
                return N4M_ERR_NUMERICAL_FAILURE;
            }

            std::vector<double> w(p, 0.0);
            for (std::size_t feature = 0; feature < p; ++feature) {
                w[feature] = s[feature] / yy;
            }
            double w_norm_sq = 0.0;
            for (const double v : w) w_norm_sq += v * v;
            const double w_norm = std::sqrt(w_norm_sq);
            if (w_norm <= eps || !std::isfinite(w_norm)) {
                ctx.set_error("PLS1 moment X weights vanished");
                return N4M_ERR_NUMERICAL_FAILURE;
            }
            const double w_denom = w_norm + eps;
            for (double& v : w) v /= w_denom;

            std::size_t sign_idx = 0;
            double sign_abs = std::fabs(w[0]);
            for (std::size_t feature = 1; feature < p; ++feature) {
                const double candidate = std::fabs(w[feature]);
                if (candidate > sign_abs) {
                    sign_abs = candidate;
                    sign_idx = feature;
                }
            }
            if (w[sign_idx] < 0.0) {
                for (double& v : w) v = -v;
            }

            std::vector<double> c_w(p, 0.0);
#if defined(N4M_USE_CUDA)
            for (std::size_t i = 0; i < p; ++i) {
                double sum = 0.0;
                for (std::size_t j = 0; j < p; ++j) {
                    sum += C[i * p + j] * w[j];
                }
                c_w[i] = sum;
            }
#else
            linalg::gemv(linalg::Trans_No, p, p, 1.0,
                         C.data(), w.data(), 0.0, c_w.data());
#endif
            double tt = 0.0;
            for (std::size_t i = 0; i < p; ++i) tt += w[i] * c_w[i];
            if (tt <= eps || !std::isfinite(tt)) {
                ctx.set_error("PLS1 moment X score vanished");
                return N4M_ERR_NUMERICAL_FAILURE;
            }

            double q_load = 0.0;
            for (std::size_t i = 0; i < p; ++i) q_load += w[i] * s[i];
            q_load /= tt;

            std::vector<double> p_load(p, 0.0);
            for (std::size_t i = 0; i < p; ++i) p_load[i] = c_w[i] / tt;

            for (std::size_t i = 0; i < p; ++i) {
                W[i * a + comp] = w[i];
                P[i * a + comp] = p_load[i];
            }
            Q[comp] = q_load;

#if defined(N4M_USE_CUDA)
            for (std::size_t i = 0; i < p; ++i) {
                for (std::size_t j = 0; j < p; ++j) {
                    C[i * p + j] -= tt * p_load[i] * p_load[j];
                }
            }
#else
            linalg::ger(p, p, -tt, p_load.data(), p_load.data(), C.data(), p);
#endif
            for (std::size_t i = 0; i < p; ++i) {
                s[i] -= tt * p_load[i] * q_load;
            }
            yy -= tt * q_load * q_load;
            if (yy < 0.0 && yy > -1e-9) yy = 0.0;
        }
    }

    prefix_fits.assign(a, RidgeMomentFit{});
    for (std::size_t comp = 1; comp <= a; ++comp) {
        st = pls1_prefix_fit_from_components(
            ctx, W, P, Q, p, a, comp, x_mean, x_scale, y_mean, y_scale,
            prefix_fits[comp - 1]);
        if (st != N4M_OK) return st;
    }
    return N4M_OK;
}

[[nodiscard]] n4m_status_t fit_pls1_moment_prefixes_for_folds(
    Context& ctx,
    const Config& cfg,
    const std::vector<MomentStats>& stats_by_fold,
    std::int32_t max_components,
    std::vector<std::vector<RidgeMomentFit>>& prefix_fits_by_fold,
    bool* used_cuda_device_components = nullptr,
    bool* used_cuda_parallel_folds = nullptr,
    bool* used_cuda_many_batched = nullptr) {
    if (used_cuda_device_components != nullptr) {
        *used_cuda_device_components = false;
    }
    if (used_cuda_parallel_folds != nullptr) {
        *used_cuda_parallel_folds = false;
    }
    if (used_cuda_many_batched != nullptr) {
        *used_cuda_many_batched = false;
    }
    if (stats_by_fold.empty()) {
        ctx.set_error("PLS1 moment fold fit requires at least one fold");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (max_components < 1) {
        ctx.set_error("PLS1 moment fold fit requires at least one component");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    prefix_fits_by_fold.assign(stats_by_fold.size(), {});

#if defined(N4M_USE_CUDA)
    const auto p = static_cast<std::size_t>(stats_by_fold.front().n_features);
    const auto a = static_cast<std::size_t>(max_components);
    if (stats_by_fold.size() > 1 &&
        p >= cuda_pls_min_device_features(cfg) &&
        ::n4m::cuda_dispatch::cuda_runtime_available()) {
        std::vector<Pls1MomentDesign> designs(stats_by_fold.size());
        std::vector<std::vector<double>> W_by_fold(stats_by_fold.size());
        std::vector<std::vector<double>> P_by_fold(stats_by_fold.size());
        std::vector<std::vector<double>> Q_by_fold(stats_by_fold.size());
        std::vector<const double*> C_ptrs(stats_by_fold.size(), nullptr);
        std::vector<const double*> s_ptrs(stats_by_fold.size(), nullptr);
        std::vector<double*> W_ptrs(stats_by_fold.size(), nullptr);
        std::vector<double*> P_ptrs(stats_by_fold.size(), nullptr);
        std::vector<double*> Q_ptrs(stats_by_fold.size(), nullptr);
        std::vector<double> yy(stats_by_fold.size(), 0.0);
        std::vector<double> yy_out(stats_by_fold.size(), 0.0);

        for (std::size_t fold = 0; fold < stats_by_fold.size(); ++fold) {
            if (stats_by_fold[fold].n_features !=
                static_cast<std::int64_t>(p)) {
                ctx.set_error("PLS1 moment fold fit received inconsistent feature counts");
                return N4M_ERR_INVALID_ARGUMENT;
            }
            auto& design = designs[fold];
            n4m_status_t st = pls1_design_moments(
                ctx, cfg, stats_by_fold[fold], design.C, design.s, design.yy,
                design.x_mean, design.x_scale, design.y_mean, design.y_scale);
            if (st != N4M_OK) return st;
            W_by_fold[fold].assign(p * a, 0.0);
            P_by_fold[fold].assign(p * a, 0.0);
            Q_by_fold[fold].assign(a, 0.0);
            C_ptrs[fold] = design.C.data();
            s_ptrs[fold] = design.s.data();
            yy[fold] = design.yy;
            W_ptrs[fold] = W_by_fold[fold].data();
            P_ptrs[fold] = P_by_fold[fold].data();
            Q_ptrs[fold] = Q_by_fold[fold].data();
        }

        std::string cuda_error;
        bool used_parallel_folds = false;
        bool used_many_batched = false;
        const int cuda_status =
            ::n4m::cuda_dispatch::pls1_moment_components_many(
                stats_by_fold.size(), p, a, C_ptrs.data(), s_ptrs.data(),
                yy.data(), std::numeric_limits<double>::epsilon(),
                W_ptrs.data(), P_ptrs.data(), Q_ptrs.data(), yy_out.data(),
                cfg.cuda_pls_parallel_folds != 0,
                cfg.cuda_pls_many_batched != 0, &used_parallel_folds,
                &used_many_batched, &cuda_error);
        if (cuda_status != 0) {
            if (!cuda_error.empty()) {
                ctx.set_error(cuda_error.c_str());
            } else {
                ctx.set_error("CUDA PLS1 moment fold workspace failed");
            }
            return (cuda_status == 1)
                ? N4M_ERR_NUMERICAL_FAILURE
                : N4M_ERR_BACKEND_UNAVAILABLE;
        }

        for (std::size_t fold = 0; fold < stats_by_fold.size(); ++fold) {
            prefix_fits_by_fold[fold].assign(a, RidgeMomentFit{});
            for (std::size_t comp = 1; comp <= a; ++comp) {
                n4m_status_t st = pls1_prefix_fit_from_components(
                    ctx, W_by_fold[fold], P_by_fold[fold], Q_by_fold[fold],
                    p, a, comp,
                    designs[fold].x_mean, designs[fold].x_scale,
                    designs[fold].y_mean, designs[fold].y_scale,
                    prefix_fits_by_fold[fold][comp - 1]);
                if (st != N4M_OK) return st;
            }
        }
        if (used_cuda_device_components != nullptr) {
            *used_cuda_device_components = true;
        }
        if (used_cuda_parallel_folds != nullptr) {
            *used_cuda_parallel_folds = used_parallel_folds;
        }
        if (used_cuda_many_batched != nullptr) {
            *used_cuda_many_batched = used_many_batched;
        }
        return N4M_OK;
    }
#endif

    const auto n_folds = static_cast<std::int64_t>(stats_by_fold.size());
    std::vector<n4m_status_t> fold_status(stats_by_fold.size(), N4M_OK);
    std::vector<std::string> fold_errors(stats_by_fold.size());
    std::vector<std::uint8_t> fold_used_cuda(stats_by_fold.size(), 0);

    N4M_PARALLEL_FOR_STATIC
    for (std::int64_t fold_i = 0; fold_i < n_folds; ++fold_i) {
        const auto fold = static_cast<std::size_t>(fold_i);
        bool used_cuda_device = false;
        Context fold_ctx;
        try {
            fold_status[fold] = fit_pls1_moment_prefixes(
                fold_ctx, cfg, stats_by_fold[fold], max_components,
                prefix_fits_by_fold[fold], &used_cuda_device);
            fold_used_cuda[fold] = used_cuda_device ? 1U : 0U;
            if (fold_status[fold] != N4M_OK &&
                fold_ctx.last_error()[0] != '\0') {
                fold_errors[fold] = fold_ctx.last_error();
            }
        } catch (const std::bad_alloc&) {
            fold_status[fold] = N4M_ERR_OUT_OF_MEMORY;
            fold_errors[fold] =
                "out of memory while fitting PLS1 moment fold prefixes";
        } catch (const std::exception& exc) {
            fold_status[fold] = N4M_ERR_INTERNAL;
            fold_errors[fold] = exc.what();
        } catch (...) {
            fold_status[fold] = N4M_ERR_INTERNAL;
            fold_errors[fold] =
                "unexpected exception while fitting PLS1 moment fold prefixes";
        }
    }

    bool any_cuda_device_components = false;
    for (std::size_t fold = 0; fold < stats_by_fold.size(); ++fold) {
        if (fold_status[fold] != N4M_OK) {
            if (!fold_errors[fold].empty()) {
                ctx.set_error(fold_errors[fold].c_str());
            } else {
                ctx.set_error("PLS1 moment fold fit failed");
            }
            return fold_status[fold];
        }
        any_cuda_device_components = any_cuda_device_components ||
                                     (fold_used_cuda[fold] != 0U);
    }
    if (used_cuda_device_components != nullptr) {
        *used_cuda_device_components = any_cuda_device_components;
    }
    return N4M_OK;
}

[[nodiscard]] std::vector<std::int32_t> unique_components_desc(
    const std::vector<std::int32_t>& components) {
    std::vector<std::int32_t> out = components;
    std::sort(out.begin(), out.end(), std::greater<std::int32_t>{});
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

[[nodiscard]] bool component_prefix_still_needed(
    const std::vector<std::int32_t>& components,
    const std::vector<unsigned char>& ok,
    std::int32_t component) noexcept {
    for (std::size_t comp_ix = 0; comp_ix < components.size(); ++comp_ix) {
        if (ok[comp_ix] != 0 && components[comp_ix] == component) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] n4m_status_t recover_pls1_moment_prefix_for_job(
    const Config& cfg,
    const MomentStats& stats,
    const std::vector<std::int32_t>& components,
    const std::vector<std::int32_t>& components_desc,
    const std::vector<unsigned char>& ok,
    std::vector<RidgeMomentFit>& prefix_fits,
    std::int32_t& recovered_component,
    std::int64_t& n_attempts,
    std::int64_t& n_host_attempts,
    std::int64_t& n_cuda_device_attempts) {
    prefix_fits.clear();
    recovered_component = 0;
    Context fit_ctx;
    for (const auto attempt_comp : components_desc) {
        if (!component_prefix_still_needed(components, ok, attempt_comp)) {
            continue;
        }
        bool used_cuda_device = false;
        ++n_attempts;
        const n4m_status_t st = fit_pls1_moment_prefixes(
            fit_ctx, cfg, stats, attempt_comp, prefix_fits,
            &used_cuda_device);
        if (used_cuda_device) {
            ++n_cuda_device_attempts;
        } else {
            ++n_host_attempts;
        }
        if (st == N4M_OK &&
            prefix_fits.size() >= static_cast<std::size_t>(attempt_comp)) {
            recovered_component = attempt_comp;
            return N4M_OK;
        }
        fit_ctx.clear_error();
        prefix_fits.clear();
    }
    return N4M_OK;
}

[[nodiscard]] n4m_status_t recover_pls1_moment_prefixes_for_folds(
    Context& ctx,
    const Config& cfg,
    const std::vector<MomentStats>& train_stats,
    const std::vector<std::int32_t>& components_desc,
    std::int32_t failed_component,
    std::vector<std::vector<RidgeMomentFit>>& prefix_fits_by_fold,
    std::int32_t& recovered_component,
    bool& used_cuda_device,
    bool& used_cuda_parallel_folds,
    bool& used_cuda_many_batched) {
    prefix_fits_by_fold.clear();
    recovered_component = 0;
    used_cuda_device = false;
    used_cuda_parallel_folds = false;
    used_cuda_many_batched = false;
    for (const auto attempt_comp : components_desc) {
        if (attempt_comp >= failed_component) {
            continue;
        }
        bool attempt_used_cuda_device = false;
        bool attempt_used_cuda_parallel_folds = false;
        bool attempt_used_cuda_many_batched = false;
        const n4m_status_t st = fit_pls1_moment_prefixes_for_folds(
            ctx, cfg, train_stats, attempt_comp, prefix_fits_by_fold,
            &attempt_used_cuda_device, &attempt_used_cuda_parallel_folds,
            &attempt_used_cuda_many_batched);
        bool complete = (st == N4M_OK &&
                         prefix_fits_by_fold.size() == train_stats.size());
        if (complete) {
            for (const auto& prefix_fits : prefix_fits_by_fold) {
                if (prefix_fits.size() <
                    static_cast<std::size_t>(attempt_comp)) {
                    complete = false;
                    break;
                }
            }
        }
        if (complete) {
            recovered_component = attempt_comp;
            used_cuda_device = attempt_used_cuda_device;
            used_cuda_parallel_folds = attempt_used_cuda_parallel_folds;
            used_cuda_many_batched = attempt_used_cuda_many_batched;
            return N4M_OK;
        }
        ctx.clear_error();
        prefix_fits_by_fold.clear();
    }
    return N4M_OK;
}

}  // namespace

n4m_status_t run_moment_sweep(Context& ctx,
                              const Config& cfg,
                              const n4m_matrix_view_t& X,
                              const n4m_matrix_view_t& Y,
                              std::int32_t cv,
                              const std::int32_t* fold_ids,
                              std::int64_t n_fold_ids,
                              const double* ridge_lambdas,
                              std::int64_t n_ridge_lambdas,
                              const std::int32_t* pls_components,
                              std::int64_t n_pls_components,
                              std::int32_t heads_mask,
                              SweepResult& out) {
    n4m_status_t st = validate_xy(ctx, X, Y);
    if (st != N4M_OK) return st;
    if ((heads_mask & ~(kSweepHeadRidge | kSweepHeadPls)) != 0) {
        ctx.set_error("sweep heads_mask contains unsupported bits");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    const bool do_ridge = (heads_mask & kSweepHeadRidge) != 0;
    const bool do_pls = (heads_mask & kSweepHeadPls) != 0;
    if (!do_ridge && !do_pls) {
        ctx.set_error("sweep requires at least one head");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (n_pls_components < 0) {
        ctx.set_error("sweep n_pls_components must be >= 0");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (n_pls_components > 0 && pls_components == nullptr) {
        ctx.set_error("sweep invalid pls_components/n_pls_components");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (n_ridge_lambdas < 0 ||
        (n_ridge_lambdas > 0 && ridge_lambdas == nullptr)) {
        ctx.set_error("sweep invalid ridge_lambdas/n_ridge_lambdas");
        return N4M_ERR_INVALID_ARGUMENT;
    }

    std::vector<double> lambdas;
    if (do_ridge) {
        if (n_ridge_lambdas == 0) {
            lambdas.push_back(cfg.ridge_lambda);
        } else {
            lambdas.assign(ridge_lambdas,
                           ridge_lambdas + static_cast<std::size_t>(n_ridge_lambdas));
        }
        for (const double lambda : lambdas) {
            if (!std::isfinite(lambda) || lambda < 0.0) {
                ctx.set_error("sweep ridge lambdas must be finite and >= 0");
                return N4M_ERR_INVALID_ARGUMENT;
            }
        }
    }

    const auto n = static_cast<std::size_t>(X.rows);
    const auto p = static_cast<std::size_t>(X.cols);
    const auto q = static_cast<std::size_t>(Y.cols);
    std::vector<std::int32_t> ids;
    std::int32_t actual_cv = 0;
    st = build_fold_ids(ctx, X.rows, cv, fold_ids, n_fold_ids, ids, actual_cv);
    if (st != N4M_OK) return st;
    const auto fold_rows = fold_rows_from_ids(ids, actual_cv);
    std::size_t min_train_rows = n;
    for (const auto& rows : fold_rows) {
        min_train_rows = std::min(min_train_rows, n - rows.size());
    }

    std::vector<std::int32_t> components;
    if (do_pls) {
        if (n_pls_components == 0) {
            components.push_back(cfg.n_components);
        } else {
            components.assign(
                pls_components,
                pls_components + static_cast<std::size_t>(n_pls_components));
        }
        const auto max_components =
            static_cast<std::int32_t>(std::min<std::size_t>(p, min_train_rows));
        for (const auto comp : components) {
            if (comp < 1 || comp > max_components) {
                ctx.set_error("sweep PLS components must be in [1, min(p, n_train)]");
                return N4M_ERR_INVALID_ARGUMENT;
            }
        }
    }

    const auto train_rows = train_rows_from_fold_rows(n, fold_rows);
    const bool use_pls_moment_route =
        do_pls && q == 1U &&
        cfg.solver == N4M_SOLVER_NIPALS &&
        cfg.deflation == N4M_DEFLATION_REGRESSION;

    std::vector<unsigned char> use_materialized_fold(
        static_cast<std::size_t>(actual_cv), 0);
    bool need_all_stats = use_pls_moment_route;
    if (do_ridge) {
        need_all_stats = need_all_stats || (p <= n);
        for (std::int32_t fold = 0; fold < actual_cv; ++fold) {
            const auto fold_ix = static_cast<std::size_t>(fold);
            if (p > train_rows[fold_ix].size()) {
                use_materialized_fold[fold_ix] = 1;
            } else {
                need_all_stats = true;
            }
        }
    }
    bool need_fold_designs = do_pls && !use_pls_moment_route;
    for (const auto v : use_materialized_fold) {
        if (v != 0) {
            need_fold_designs = true;
            break;
        }
    }
    std::vector<double> Xc;
    std::vector<double> Yc;
    bool have_full_copies = false;
    auto ensure_full_copies = [&]() -> n4m_status_t {
        if (have_full_copies) return N4M_OK;
        n4m_status_t local_status = copy_full(ctx, X, "X", Xc);
        if (local_status != N4M_OK) return local_status;
        local_status = copy_full(ctx, Y, "Y", Yc);
        if (local_status != N4M_OK) return local_status;
        have_full_copies = true;
        return N4M_OK;
    };

    std::vector<FoldMaterializedDesign> fold_designs;
    auto ensure_fold_designs = [&]() -> n4m_status_t {
        if (!fold_designs.empty()) return N4M_OK;
        st = ensure_full_copies();
        if (st != N4M_OK) return st;
        fold_designs = build_fold_designs(Xc, Yc, p, q, train_rows);
        return N4M_OK;
    };
    if (need_fold_designs) {
        st = ensure_fold_designs();
        if (st != N4M_OK) return st;
    }

    std::vector<RidgeDualDesign> fold_dual_designs(
        static_cast<std::size_t>(actual_cv));
    std::vector<unsigned char> use_dual_cross_fold(
        static_cast<std::size_t>(actual_cv), 0);
    if (do_ridge) {
        for (std::int32_t fold = 0; fold < actual_cv; ++fold) {
            const auto fold_ix = static_cast<std::size_t>(fold);
            if (use_materialized_fold[fold_ix] == 0) continue;
            st = prepare_ridge_dual_design(
                ctx, cfg, fold_designs[fold_ix].X_train,
                fold_designs[fold_ix].Y_train,
                fold_designs[fold_ix].train_rows.size(), p, q,
                fold_dual_designs[fold_ix]);
            if (st != N4M_OK) return st;
            if (should_use_ridge_dual_cross(
                    fold_designs[fold_ix].train_rows.size(),
                    fold_rows[fold_ix].size(), lambdas.size(), q)) {
                st = ensure_full_copies();
                if (st != N4M_OK) return st;
                st = prepare_ridge_dual_cross(
                    ctx, cfg, fold_dual_designs[fold_ix], Xc,
                    fold_rows[fold_ix], fold_dual_designs[fold_ix]);
                if (st != N4M_OK) return st;
                use_dual_cross_fold[fold_ix] = 1;
            }
        }
    }

    RidgeDualDesign all_dual_design;
    if (do_ridge && p > n) {
        st = ensure_full_copies();
        if (st != N4M_OK) return st;
        st = prepare_ridge_dual_design(ctx, cfg, Xc, Yc, n, p, q,
                                       all_dual_design);
        if (st != N4M_OK) return st;
    }

    MomentStats all_stats;
    if (need_all_stats) {
        st = compute_moments(ctx, X, Y, all_stats);
        if (st != N4M_OK) return st;
    }

    std::vector<MomentStats> heldout_stats(static_cast<std::size_t>(actual_cv));
    std::vector<MomentStats> train_stats(static_cast<std::size_t>(actual_cv));
    if (do_ridge || use_pls_moment_route) {
        for (std::int32_t fold = 0; fold < actual_cv; ++fold) {
            const auto fold_ix = static_cast<std::size_t>(fold);
            const bool ridge_needs_moments =
                do_ridge && use_materialized_fold[fold_ix] == 0;
            if (!use_pls_moment_route && !ridge_needs_moments) continue;
            const auto& rows = fold_rows[static_cast<std::size_t>(fold)];
            st = compute_moments_subset(
                ctx, X, Y, rows.data(), static_cast<std::int64_t>(rows.size()),
                heldout_stats[fold_ix]);
            if (st != N4M_OK) return st;
            st = subtract_moments(ctx, all_stats, heldout_stats[fold_ix],
                                  train_stats[fold_ix]);
            if (st != N4M_OK) return st;
        }
    }

    out = SweepResult{};
    out.n_samples = X.rows;
    out.n_features = X.cols;
    out.n_targets = Y.cols;
    out.cv = actual_cv;
    out.n_candidates = static_cast<std::int64_t>(lambdas.size() + components.size());
    out.fold_ids = std::move(ids);
    out.candidate_scores.assign(static_cast<std::size_t>(out.n_candidates) * 4, 0.0);
    if (do_ridge && !lambdas.empty()) {
        bool has_moment_ridge_fold = false;
        bool has_dual_ridge_fold = false;
        for (const auto use_materialized : use_materialized_fold) {
            if (use_materialized != 0) {
                has_dual_ridge_fold = true;
            } else {
                has_moment_ridge_fold = true;
            }
        }
        if (has_moment_ridge_fold) {
            out.n_ridge_moment_candidates =
                static_cast<std::int64_t>(lambdas.size());
        }
        if (has_dual_ridge_fold) {
            out.n_ridge_dual_materialized_candidates =
                static_cast<std::int64_t>(lambdas.size());
        }
    }

    double best = std::numeric_limits<double>::infinity();
    std::size_t best_id = 0;
    std::int32_t best_head = -1;
    double best_param = 0.0;
    std::vector<double> best_oof;
    std::size_t candidate_id = 0;
    const bool score_only = score_only_requested(cfg);
    out.score_only = score_only ? 1 : 0;
    if (!score_only) {
        st = ensure_full_copies();
        if (st != N4M_OK) return st;
    }

    auto record_candidate = [&](std::size_t id,
                                std::int32_t head,
                                double param,
                                double rmse,
                                std::vector<double>& candidate_oof) {
        out.candidate_scores[id * 4 + 0] = static_cast<double>(id);
        out.candidate_scores[id * 4 + 1] = static_cast<double>(head);
        out.candidate_scores[id * 4 + 2] = param;
        out.candidate_scores[id * 4 + 3] = rmse;
        if (rmse < best) {
            best = rmse;
            best_id = id;
            best_head = head;
            best_param = param;
            if (!score_only) best_oof = std::move(candidate_oof);
        }
    };

    std::vector<RidgeMomentEigenPath> ridge_moment_eigen_paths;
    std::vector<unsigned char> ridge_moment_eigen_path_ready;
    if (do_ridge && lambdas.size() > 1U) {
        ridge_moment_eigen_paths.resize(static_cast<std::size_t>(actual_cv));
        ridge_moment_eigen_path_ready.assign(
            static_cast<std::size_t>(actual_cv), 0);
        for (std::int32_t fold = 0; fold < actual_cv; ++fold) {
            const auto fold_ix = static_cast<std::size_t>(fold);
            if (use_materialized_fold[fold_ix] != 0) continue;
            Context path_ctx;
            const n4m_status_t path_st = prepare_ridge_moment_eigen_path(
                path_ctx, cfg, train_stats[fold_ix],
                ridge_moment_eigen_paths[fold_ix]);
            if (path_st == N4M_OK) {
                ridge_moment_eigen_path_ready[fold_ix] = 1;
                ++out.n_ridge_moment_eigen_path_preparations;
            }
        }
    }

    for (std::size_t candidate = 0; candidate < lambdas.size(); ++candidate) {
        const double lambda = lambdas[candidate];
        bool candidate_ok = true;
        double sse = 0.0;
        std::int64_t count = 0;
        std::vector<double> candidate_oof;
        if (!score_only) candidate_oof.assign(n * q, 0.0);
        for (std::int32_t fold = 0; fold < actual_cv; ++fold) {
            const auto fold_ix = static_cast<std::size_t>(fold);
            RidgeMomentFit fit;
            const auto& rows = fold_rows[fold_ix];
            std::vector<double> pred;
            if (!score_only) pred.assign(rows.size() * q, 0.0);
            bool have_pred = false;
            if (use_materialized_fold[fold_ix] != 0) {
                if (use_dual_cross_fold[fold_ix] != 0) {
                    ++out.n_ridge_dual_cross_cv_fits;
                    if (score_only) {
                        double fold_sse = 0.0;
                        st = ridge_dual_cross_heldout_sse(
                            ctx, fold_dual_designs[fold_ix], Yc, q, rows,
                            lambda, fold_sse);
                        if (st == N4M_OK) sse += fold_sse;
                    } else {
                        st = predict_ridge_from_dual_design(
                            ctx, fold_dual_designs[fold_ix], lambda, pred);
                        have_pred = (st == N4M_OK);
                    }
                } else {
                    ++out.n_ridge_dual_materialized_cv_fits;
                    st = fit_ridge_from_dual_design(ctx,
                                                    fold_dual_designs[fold_ix],
                                                    lambda, fit);
                    if (st == N4M_OK) {
                        if (score_only) {
                            sse += heldout_sse_from_fit(Xc, Yc, p, q, rows, fit);
                        } else {
                            predict_rows(Xc, p, rows, fit, pred, 0, q);
                            have_pred = true;
                        }
                    }
                }
            } else {
                ++out.n_ridge_moment_cv_fits;
                const RidgeMomentEigenPath* path =
                    (!ridge_moment_eigen_path_ready.empty() &&
                     ridge_moment_eigen_path_ready[fold_ix] != 0)
                        ? &ridge_moment_eigen_paths[fold_ix]
                        : nullptr;
                bool used_eigen_path = false;
                st = fit_ridge_from_moment_path_or_direct(
                    ctx, cfg, train_stats[fold_ix], path, lambda, fit,
                    &used_eigen_path);
                if (st == N4M_OK) {
                    if (used_eigen_path) {
                        ++out.n_ridge_moment_eigen_path_cv_fits;
                    } else {
                        ++out.n_ridge_moment_direct_cv_fits;
                    }
                    if (score_only) {
                        sse += ridge_heldout_sse_from_moments(
                            heldout_stats[fold_ix], fit);
                    } else {
                        predict_rows(Xc, p, rows, fit, pred, 0, q);
                        have_pred = true;
                    }
                }
            }
            if (st != N4M_OK) {
                candidate_ok = false;
                ctx.clear_error();
                break;
            }
            if (have_pred) {
                sse += heldout_sse(Yc, q, rows, pred);
            }
            if (!score_only) {
                for (std::size_t rr = 0; rr < rows.size(); ++rr) {
                    const auto row = static_cast<std::size_t>(rows[rr]);
                    for (std::size_t target = 0; target < q; ++target) {
                        candidate_oof[row * q + target] = pred[rr * q + target];
                    }
                }
            }
            count += static_cast<std::int64_t>(rows.size() * q);
        }
        const double rmse = (candidate_ok && count > 0)
            ? std::sqrt(sse / static_cast<double>(count))
            : std::numeric_limits<double>::infinity();
        record_candidate(candidate_id, 0, lambda, rmse, candidate_oof);
        ++candidate_id;
    }

    std::vector<double> pls_sse(components.size(), 0.0);
    std::vector<std::int64_t> pls_count(components.size(), 0);
    std::vector<unsigned char> pls_ok(components.size(), 1);
    std::vector<std::vector<double>> pls_oof;
    std::int64_t pls_moment_cv_fits = 0;
    std::int64_t pls_moment_host_cv_fits = 0;
    std::int64_t pls_moment_cuda_device_cv_fits = 0;
    std::int64_t pls_moment_cuda_parallel_fold_batches = 0;
    std::int64_t pls_moment_cuda_parallel_fold_jobs = 0;
    std::int64_t pls_moment_cuda_many_batched_batches = 0;
    std::int64_t pls_moment_cuda_many_batched_jobs = 0;
    std::int64_t pls_materialized_cv_fits = 0;
    std::int64_t pls_moment_final_fits = 0;
    std::int64_t pls_moment_host_final_fits = 0;
    std::int64_t pls_moment_cuda_device_final_fits = 0;
    std::int64_t pls_materialized_final_fits = 0;
    if (!score_only) {
        pls_oof.assign(components.size(), std::vector<double>(n * q, 0.0));
    }
    bool pls_used_moment_route = false;
    if (!components.empty()) {
        const auto max_comp = *std::max_element(components.begin(), components.end());
        const auto fallback_components_desc = unique_components_desc(components);
        bool moment_attempt_ok = false;
        if (use_pls_moment_route) {
            std::vector<std::vector<RidgeMomentFit>> prefix_fits_by_fold;
            bool used_cuda_device = false;
            bool used_cuda_parallel_folds = false;
            bool used_cuda_many_batched = false;
            st = fit_pls1_moment_prefixes_for_folds(
                ctx, cfg, train_stats, max_comp, prefix_fits_by_fold,
                &used_cuda_device, &used_cuda_parallel_folds,
                &used_cuda_many_batched);
            if (st != N4M_OK) {
                ctx.clear_error();
            } else {
                moment_attempt_ok = true;
                pls_moment_cv_fits += actual_cv;
                if (used_cuda_device) {
                    pls_moment_cuda_device_cv_fits += actual_cv;
                    if (used_cuda_parallel_folds) {
                        ++pls_moment_cuda_parallel_fold_batches;
                        pls_moment_cuda_parallel_fold_jobs += actual_cv;
                    }
                    if (used_cuda_many_batched) {
                        ++pls_moment_cuda_many_batched_batches;
                        pls_moment_cuda_many_batched_jobs += actual_cv;
                    }
                } else {
                    pls_moment_host_cv_fits += actual_cv;
                }
            }
            for (std::int32_t fold = 0; moment_attempt_ok && fold < actual_cv; ++fold) {
                const auto fold_ix = static_cast<std::size_t>(fold);
                const auto& rows = fold_rows[fold_ix];
                for (std::size_t comp_ix = 0; comp_ix < components.size(); ++comp_ix) {
                    const auto comp = static_cast<std::size_t>(components[comp_ix]);
                    const auto& fit = prefix_fits_by_fold[fold_ix][comp - 1U];
                    pls_sse[comp_ix] +=
                        ridge_heldout_sse_from_moments(heldout_stats[fold_ix], fit);
                    if (!score_only) {
                        std::vector<double> pred(rows.size() * q, 0.0);
                        predict_rows(Xc, p, rows, fit, pred, 0, q);
                        for (std::size_t rr = 0; rr < rows.size(); ++rr) {
                            const auto row = static_cast<std::size_t>(rows[rr]);
                            for (std::size_t target = 0; target < q; ++target) {
                                pls_oof[comp_ix][row * q + target] =
                                    pred[rr * q + target];
                            }
                        }
                    }
                    pls_count[comp_ix] +=
                        static_cast<std::int64_t>(rows.size() * q);
                }
            }
            if (!moment_attempt_ok) {
                for (std::int32_t fold = 0; fold < actual_cv; ++fold) {
                    const auto fold_ix = static_cast<std::size_t>(fold);
                    std::vector<RidgeMomentFit> prefix_fits;
                    std::int32_t recovered_comp = 0;
                    st = recover_pls1_moment_prefix_for_job(
                        cfg, train_stats[fold_ix], components,
                        fallback_components_desc, pls_ok, prefix_fits,
                        recovered_comp, pls_moment_cv_fits,
                        pls_moment_host_cv_fits,
                        pls_moment_cuda_device_cv_fits);
                    if (st != N4M_OK) return st;
                    const auto& rows = fold_rows[fold_ix];
                    for (std::size_t comp_ix = 0;
                         comp_ix < components.size(); ++comp_ix) {
                        if (pls_ok[comp_ix] == 0) continue;
                        const auto comp =
                            static_cast<std::size_t>(components[comp_ix]);
                        if (components[comp_ix] > recovered_comp ||
                            prefix_fits.size() < comp) {
                            pls_ok[comp_ix] = 0;
                            continue;
                        }
                        const auto& fit = prefix_fits[comp - 1U];
                        pls_sse[comp_ix] +=
                            ridge_heldout_sse_from_moments(
                                heldout_stats[fold_ix], fit);
                        if (!score_only) {
                            std::vector<double> pred(rows.size() * q, 0.0);
                            predict_rows(Xc, p, rows, fit, pred, 0, q);
                            for (std::size_t rr = 0; rr < rows.size(); ++rr) {
                                const auto row =
                                    static_cast<std::size_t>(rows[rr]);
                                for (std::size_t target = 0; target < q;
                                     ++target) {
                                    pls_oof[comp_ix][row * q + target] =
                                        pred[rr * q + target];
                                }
                            }
                        }
                        pls_count[comp_ix] +=
                            static_cast<std::int64_t>(rows.size() * q);
                    }
                    bool any_needed = false;
                    for (std::size_t comp_ix = 0;
                         comp_ix < components.size(); ++comp_ix) {
                        any_needed = any_needed || (pls_ok[comp_ix] != 0);
                    }
                    if (!any_needed) break;
                }
                for (std::size_t comp_ix = 0; comp_ix < components.size();
                     ++comp_ix) {
                    if (pls_ok[comp_ix] != 0 && pls_count[comp_ix] > 0) {
                        moment_attempt_ok = true;
                        break;
                    }
                }
            }
            if (moment_attempt_ok) {
                pls_used_moment_route = true;
                out.n_pls_moment_candidates =
                    static_cast<std::int64_t>(components.size());
            } else {
                std::fill(pls_sse.begin(), pls_sse.end(), 0.0);
                std::fill(pls_count.begin(), pls_count.end(), 0);
                std::fill(pls_ok.begin(), pls_ok.end(), 1);
                if (!score_only) {
                    for (auto& oof : pls_oof) std::fill(oof.begin(), oof.end(), 0.0);
                }
            }
        }

        if (!pls_used_moment_route) {
            st = ensure_fold_designs();
            if (st != N4M_OK) return st;
            for (std::int32_t fold = 0; fold < actual_cv; ++fold) {
                const auto fold_ix = static_cast<std::size_t>(fold);
                const auto& design = fold_designs[fold_ix];
                std::unique_ptr<Model> max_model;
                ++pls_materialized_cv_fits;
                st = fit_pls_model_materialized(ctx, cfg, design.X_train,
                                                design.Y_train,
                                                design.train_rows.size(), p, q,
                                                max_comp, max_model);
                const bool use_prefix_model = (st == N4M_OK);
                if (!use_prefix_model) {
                    ctx.clear_error();
                }
                const auto& rows = fold_rows[fold_ix];
                for (std::size_t comp_ix = 0; comp_ix < components.size(); ++comp_ix) {
                    if (pls_ok[comp_ix] == 0) continue;
                    const std::int32_t comp = components[comp_ix];
                    RidgeMomentFit fit;
                    if (use_prefix_model) {
                        st = copy_model_prefix_as_linear_fit(ctx, *max_model, comp, fit);
                    } else {
                        ++pls_materialized_cv_fits;
                        st = fit_pls_materialized(ctx, cfg, design.X_train,
                                                  design.Y_train,
                                                  design.train_rows.size(), p, q,
                                                  comp, fit);
                    }
                    if (st != N4M_OK) {
                        pls_ok[comp_ix] = 0;
                        ctx.clear_error();
                        continue;
                    }
                    if (score_only) {
                        pls_sse[comp_ix] +=
                            heldout_sse_from_fit(Xc, Yc, p, q, rows, fit);
                    } else {
                        std::vector<double> pred(rows.size() * q, 0.0);
                        predict_rows(Xc, p, rows, fit, pred, 0, q);
                        pls_sse[comp_ix] += heldout_sse(Yc, q, rows, pred);
                        for (std::size_t rr = 0; rr < rows.size(); ++rr) {
                            const auto row = static_cast<std::size_t>(rows[rr]);
                            for (std::size_t target = 0; target < q; ++target) {
                                pls_oof[comp_ix][row * q + target] =
                                    pred[rr * q + target];
                            }
                        }
                    }
                    pls_count[comp_ix] +=
                        static_cast<std::int64_t>(rows.size() * q);
                }
            }
        }
        for (std::size_t comp_ix = 0; comp_ix < components.size(); ++comp_ix) {
            const double rmse = (pls_ok[comp_ix] != 0 && pls_count[comp_ix] > 0)
                ? std::sqrt(pls_sse[comp_ix] / static_cast<double>(pls_count[comp_ix]))
                : std::numeric_limits<double>::infinity();
            std::vector<double> empty_oof;
            record_candidate(candidate_id, 1, static_cast<double>(components[comp_ix]),
                             rmse,
                             score_only ? empty_oof : pls_oof[comp_ix]);
            ++candidate_id;
        }
    }
    if (!std::isfinite(best)) {
        ctx.set_error("sweep: all candidates failed");
        return N4M_ERR_NUMERICAL_FAILURE;
    }

    out.selected_candidate_id = static_cast<std::int64_t>(best_id);
    out.selected_head_id = best_head;
    out.selected_param = best_param;
    out.selected_cv_rmse = best;
    out.score_only = score_only ? 1 : 0;
    out.n_pls_moment_cv_fits = pls_moment_cv_fits;
    out.n_pls_moment_host_cv_fits = pls_moment_host_cv_fits;
    out.n_pls_moment_cuda_device_cv_fits = pls_moment_cuda_device_cv_fits;
    out.n_pls_moment_cuda_parallel_fold_batches =
        pls_moment_cuda_parallel_fold_batches;
    out.n_pls_moment_cuda_parallel_fold_jobs =
        pls_moment_cuda_parallel_fold_jobs;
    out.n_pls_moment_cuda_many_batched_batches =
        pls_moment_cuda_many_batched_batches;
    out.n_pls_moment_cuda_many_batched_jobs =
        pls_moment_cuda_many_batched_jobs;
    out.n_pls_materialized_cv_fits = pls_materialized_cv_fits;

    if (score_only) {
        ctx.clear_error();
        return N4M_OK;
    }

    RidgeMomentFit final_fit;
    if (out.selected_head_id == 0) {
        if (p > n) {
            ++out.n_ridge_dual_materialized_final_fits;
            st = fit_ridge_from_dual_design(ctx, all_dual_design,
                                            out.selected_param, final_fit);
        } else {
            ++out.n_ridge_moment_final_fits;
            st = fit_ridge_from_moments(ctx, cfg, all_stats, out.selected_param,
                                        final_fit);
        }
    } else {
        if (pls_used_moment_route) {
            const auto selected_comp =
                static_cast<std::int32_t>(out.selected_param);
            std::vector<RidgeMomentFit> final_prefix_fits;
            bool used_cuda_device = false;
            ++pls_moment_final_fits;
            st = fit_pls1_moment_prefixes(ctx, cfg, all_stats, selected_comp,
                                          final_prefix_fits, &used_cuda_device);
            if (st == N4M_OK) {
                if (used_cuda_device) {
                    ++pls_moment_cuda_device_final_fits;
                } else {
                    ++pls_moment_host_final_fits;
                }
                final_fit =
                    final_prefix_fits[static_cast<std::size_t>(
                        selected_comp - 1)];
            }
        } else {
            ++pls_materialized_final_fits;
            st = fit_pls_materialized(ctx, cfg, Xc, Yc, n, p, q,
                                      static_cast<std::int32_t>(out.selected_param),
                                      final_fit);
        }
    }
    if (st != N4M_OK) return st;
    out.n_pls_moment_final_fits = pls_moment_final_fits;
    out.n_pls_moment_host_final_fits = pls_moment_host_final_fits;
    out.n_pls_moment_cuda_device_final_fits =
        pls_moment_cuda_device_final_fits;
    out.n_pls_materialized_final_fits = pls_materialized_final_fits;
    out.coefficients = final_fit.coefficients;
    out.intercept = final_fit.intercept;
    out.x_mean = final_fit.x_mean;
    out.x_scale = final_fit.x_scale;
    out.y_mean = final_fit.y_mean;
    out.predictions.assign(n * q, 0.0);
    predict_all(Xc, n, p, final_fit, out.predictions);

    out.oof_predictions = std::move(best_oof);
    return N4M_OK;
}

n4m_status_t run_moment_final_fit(Context& ctx,
                                  const Config& cfg,
                                  const n4m_matrix_view_t& X,
                                  const n4m_matrix_view_t& Y,
                                  std::int32_t head_id,
                                  double param,
                                  SweepResult& out) {
    n4m_status_t st = validate_xy(ctx, X, Y);
    if (st != N4M_OK) return st;
    if (head_id != 0 && head_id != 1) {
        ctx.set_error("sweep final fit head_id must be 0=Ridge or 1=PLS");
        return N4M_ERR_INVALID_ARGUMENT;
    }

    const auto n = static_cast<std::size_t>(X.rows);
    const auto p = static_cast<std::size_t>(X.cols);
    const auto q = static_cast<std::size_t>(Y.cols);

    std::vector<double> Xc;
    std::vector<double> Yc;
    st = copy_full(ctx, X, "X", Xc);
    if (st != N4M_OK) return st;
    st = copy_full(ctx, Y, "Y", Yc);
    if (st != N4M_OK) return st;

    out = SweepResult{};
    out.n_samples = X.rows;
    out.n_features = X.cols;
    out.n_targets = Y.cols;
    out.cv = 0;
    out.n_candidates = 1;
    out.selected_candidate_id = 0;
    out.selected_head_id = head_id;
    out.selected_param = param;
    out.selected_cv_rmse = std::numeric_limits<double>::quiet_NaN();
    out.score_only = 0;
    out.candidate_scores.assign(4, 0.0);
    out.candidate_scores[0] = 0.0;
    out.candidate_scores[1] = static_cast<double>(head_id);
    out.candidate_scores[2] = param;
    out.candidate_scores[3] = out.selected_cv_rmse;

    RidgeMomentFit final_fit;
    if (head_id == 0) {
        if (!std::isfinite(param) || param < 0.0) {
            ctx.set_error("sweep final Ridge lambda must be finite and >= 0");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        if (p > n) {
            RidgeDualDesign all_dual_design;
            st = prepare_ridge_dual_design(ctx, cfg, Xc, Yc, n, p, q,
                                           all_dual_design);
            if (st != N4M_OK) return st;
            st = fit_ridge_from_dual_design(ctx, all_dual_design, param,
                                            final_fit);
            out.n_ridge_dual_materialized_candidates = 1;
            out.n_ridge_dual_materialized_final_fits = 1;
        } else {
            MomentStats all_stats;
            st = compute_moments(ctx, X, Y, all_stats);
            if (st != N4M_OK) return st;
            st = fit_ridge_from_moments(ctx, cfg, all_stats, param, final_fit);
            out.n_ridge_moment_candidates = 1;
            out.n_ridge_moment_final_fits = 1;
        }
    } else {
        const auto component = static_cast<std::int32_t>(std::llround(param));
        const auto max_components =
            static_cast<std::int32_t>(std::min<std::size_t>(p, n));
        if (component < 1 ||
            component > max_components ||
            std::fabs(static_cast<double>(component) - param) > 1e-9) {
            ctx.set_error("sweep final PLS components must be an integer in [1, min(p, n)]");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        const bool use_pls_moment_route =
            q == 1U &&
            cfg.solver == N4M_SOLVER_NIPALS &&
            cfg.deflation == N4M_DEFLATION_REGRESSION;
        if (use_pls_moment_route) {
            MomentStats all_stats;
            st = compute_moments(ctx, X, Y, all_stats);
            if (st != N4M_OK) return st;
            std::vector<RidgeMomentFit> prefix_fits;
            bool used_cuda_device = false;
            st = fit_pls1_moment_prefixes(ctx, cfg, all_stats, component,
                                          prefix_fits, &used_cuda_device);
            if (st != N4M_OK) return st;
            final_fit = prefix_fits[static_cast<std::size_t>(component - 1)];
            out.n_pls_moment_candidates = 1;
            out.n_pls_moment_final_fits = 1;
            if (used_cuda_device) {
                out.n_pls_moment_cuda_device_final_fits = 1;
            } else {
                out.n_pls_moment_host_final_fits = 1;
            }
        } else {
            st = fit_pls_materialized(ctx, cfg, Xc, Yc, n, p, q, component,
                                      final_fit);
            if (st != N4M_OK) return st;
            out.n_pls_materialized_final_fits = 1;
        }
    }
    if (st != N4M_OK) return st;

    out.coefficients = final_fit.coefficients;
    out.intercept = final_fit.intercept;
    out.x_mean = final_fit.x_mean;
    out.x_scale = final_fit.x_scale;
    out.y_mean = final_fit.y_mean;
    out.predictions.assign(n * q, 0.0);
    predict_all(Xc, n, p, final_fit, out.predictions);
    ctx.clear_error();
    return N4M_OK;
}

n4m_status_t score_ridge_moment_sweep(Context& ctx,
                                      const Config& cfg,
                                      const MomentStats& all_stats,
                                      const std::vector<MomentStats>& heldout_stats,
                                      const double* ridge_lambdas,
                                      std::int64_t n_ridge_lambdas,
                                      SweepResult& out) {
    if (all_stats.n_samples <= 1 || all_stats.n_features <= 0 ||
        all_stats.n_targets <= 0) {
        ctx.set_error("ridge moment score requires non-empty all-sample moments");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (heldout_stats.empty()) {
        ctx.set_error("ridge moment score requires at least one held-out fold");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (n_ridge_lambdas < 0 ||
        (n_ridge_lambdas > 0 && ridge_lambdas == nullptr)) {
        ctx.set_error("ridge moment score invalid ridge_lambdas/n_ridge_lambdas");
        return N4M_ERR_INVALID_ARGUMENT;
    }

    const auto p = static_cast<std::size_t>(all_stats.n_features);
    const auto q = static_cast<std::size_t>(all_stats.n_targets);
    if (all_stats.x_sum.size() != p || all_stats.y_sum.size() != q ||
        all_stats.xtx.size() != p * p || all_stats.xty.size() != p * q ||
        all_stats.yty.size() != q * q) {
        ctx.set_error("ridge moment score received inconsistent all-sample moments");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    for (const auto& heldout : heldout_stats) {
        if (heldout.n_samples <= 0 ||
            heldout.n_samples >= all_stats.n_samples ||
            heldout.n_features != all_stats.n_features ||
            heldout.n_targets != all_stats.n_targets ||
            heldout.x_sum.size() != p ||
            heldout.y_sum.size() != q ||
            heldout.xtx.size() != p * p ||
            heldout.xty.size() != p * q ||
            heldout.yty.size() != q * q) {
            ctx.set_error("ridge moment score received inconsistent held-out moments");
            return N4M_ERR_INVALID_ARGUMENT;
        }
    }

    std::vector<double> lambdas;
    if (n_ridge_lambdas == 0) {
        lambdas.push_back(cfg.ridge_lambda);
    } else {
        lambdas.assign(ridge_lambdas,
                       ridge_lambdas + static_cast<std::size_t>(n_ridge_lambdas));
    }
    for (const double lambda : lambdas) {
        if (!std::isfinite(lambda) || lambda < 0.0) {
            ctx.set_error("ridge moment score lambdas must be finite and >= 0");
            return N4M_ERR_INVALID_ARGUMENT;
        }
    }

    std::vector<MomentStats> train_stats(heldout_stats.size());
    for (std::size_t fold = 0; fold < heldout_stats.size(); ++fold) {
        const n4m_status_t st =
            subtract_moments(ctx, all_stats, heldout_stats[fold],
                             train_stats[fold]);
        if (st != N4M_OK) return st;
    }
    std::vector<RidgeMomentEigenPath> eigen_paths;
    std::vector<unsigned char> eigen_path_ready;
    std::int64_t eigen_path_preparations = 0;
    if (lambdas.size() > 1U) {
        eigen_paths.resize(heldout_stats.size());
        eigen_path_ready.assign(heldout_stats.size(), 0);
        for (std::size_t fold = 0; fold < train_stats.size(); ++fold) {
            Context path_ctx;
            const n4m_status_t st = prepare_ridge_moment_eigen_path(
                path_ctx, cfg, train_stats[fold], eigen_paths[fold]);
            if (st == N4M_OK) {
                eigen_path_ready[fold] = 1;
                ++eigen_path_preparations;
            }
        }
    }

    out = SweepResult{};
    out.n_samples = all_stats.n_samples;
    out.n_features = all_stats.n_features;
    out.n_targets = all_stats.n_targets;
    out.cv = static_cast<std::int32_t>(heldout_stats.size());
    out.n_candidates = static_cast<std::int64_t>(lambdas.size());
    out.n_ridge_moment_candidates = out.n_candidates;
    out.n_ridge_moment_eigen_path_preparations = eigen_path_preparations;
    out.candidate_scores.assign(lambdas.size() * 4U, 0.0);
    const bool score_only = score_only_requested(cfg);
    out.score_only = score_only ? 1 : 0;

    double best = std::numeric_limits<double>::infinity();
    std::size_t best_id = 0;
    for (std::size_t candidate = 0; candidate < lambdas.size(); ++candidate) {
        const double lambda = lambdas[candidate];
        bool candidate_ok = true;
        double sse = 0.0;
        std::int64_t count = 0;
        for (std::size_t fold = 0; fold < heldout_stats.size(); ++fold) {
            RidgeMomentFit fit;
            ++out.n_ridge_moment_cv_fits;
            const RidgeMomentEigenPath* path =
                (!eigen_path_ready.empty() && eigen_path_ready[fold] != 0)
                    ? &eigen_paths[fold]
                    : nullptr;
            bool used_eigen_path = false;
            n4m_status_t st = fit_ridge_from_moment_path_or_direct(
                ctx, cfg, train_stats[fold], path, lambda, fit,
                &used_eigen_path);
            if (st != N4M_OK) {
                candidate_ok = false;
                ctx.clear_error();
                break;
            }
            if (used_eigen_path) {
                ++out.n_ridge_moment_eigen_path_cv_fits;
            } else {
                ++out.n_ridge_moment_direct_cv_fits;
            }
            sse += ridge_heldout_sse_from_moments(heldout_stats[fold], fit);
            count += heldout_stats[fold].n_samples *
                     static_cast<std::int64_t>(q);
        }
        const double rmse = (candidate_ok && count > 0)
            ? std::sqrt(std::max(0.0, sse) / static_cast<double>(count))
            : std::numeric_limits<double>::infinity();
        out.candidate_scores[candidate * 4U + 0U] =
            static_cast<double>(candidate);
        out.candidate_scores[candidate * 4U + 1U] = 0.0;
        out.candidate_scores[candidate * 4U + 2U] = lambda;
        out.candidate_scores[candidate * 4U + 3U] = rmse;
        if (rmse < best) {
            best = rmse;
            best_id = candidate;
        }
    }
    if (!std::isfinite(best)) {
        ctx.set_error("ridge moment score: all candidates failed");
        return N4M_ERR_NUMERICAL_FAILURE;
    }

    out.selected_candidate_id = static_cast<std::int64_t>(best_id);
    out.selected_head_id = 0;
    out.selected_param = lambdas[best_id];
    out.selected_cv_rmse = best;
    out.score_only = score_only ? 1 : 0;

    if (score_only) {
        ctx.clear_error();
        return N4M_OK;
    }

    RidgeMomentFit final_fit;
    n4m_status_t st =
        fit_ridge_from_moments(ctx, cfg, all_stats, lambdas[best_id],
                               final_fit);
    if (st != N4M_OK) return st;
    out.n_ridge_moment_final_fits = 1;

    out.coefficients = std::move(final_fit.coefficients);
    out.intercept = std::move(final_fit.intercept);
    out.x_mean = std::move(final_fit.x_mean);
    out.x_scale = std::move(final_fit.x_scale);
    out.y_mean = std::move(final_fit.y_mean);
    ctx.clear_error();
    return N4M_OK;
}

n4m_status_t score_ridge_moment_sweeps_score_only(
    Context& ctx,
    const Config& cfg,
    const std::vector<MomentStats>& all_stats_by_chain,
    const std::vector<std::vector<MomentStats>>& heldout_stats_by_chain,
    const double* ridge_lambdas,
    std::int64_t n_ridge_lambdas,
    std::vector<SweepResult>& out) {
    out.clear();
    if (!score_only_requested(cfg)) {
        ctx.set_error("ridge moment batch score requires score_only");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (all_stats_by_chain.empty()) {
        ctx.set_error("ridge moment batch score requires at least one chain");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (all_stats_by_chain.size() != heldout_stats_by_chain.size()) {
        ctx.set_error("ridge moment batch score all/held-out sizes differ");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (n_ridge_lambdas < 0 ||
        (n_ridge_lambdas > 0 && ridge_lambdas == nullptr)) {
        ctx.set_error("ridge moment batch score invalid ridge_lambdas/n_ridge_lambdas");
        return N4M_ERR_INVALID_ARGUMENT;
    }

    const auto n_chains = all_stats_by_chain.size();
    const auto p =
        static_cast<std::size_t>(all_stats_by_chain.front().n_features);
    const auto q =
        static_cast<std::size_t>(all_stats_by_chain.front().n_targets);
    if (p == 0 || q == 0) {
        ctx.set_error("ridge moment batch score requires p>0 and q>0");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    const auto n_folds = heldout_stats_by_chain.front().size();
    if (n_folds == 0) {
        ctx.set_error("ridge moment batch score requires at least one held-out fold");
        return N4M_ERR_INVALID_ARGUMENT;
    }

    for (std::size_t chain = 0; chain < n_chains; ++chain) {
        const auto& all_stats = all_stats_by_chain[chain];
        if (all_stats.n_samples <= 1 || all_stats.n_features <= 0 ||
            all_stats.n_targets <= 0 ||
            static_cast<std::size_t>(all_stats.n_features) != p ||
            static_cast<std::size_t>(all_stats.n_targets) != q ||
            all_stats.x_sum.size() != p || all_stats.y_sum.size() != q ||
            all_stats.xtx.size() != p * p ||
            all_stats.xty.size() != p * q ||
            all_stats.yty.size() != q * q) {
            ctx.set_error("ridge moment batch score received inconsistent all-sample moments");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        if (heldout_stats_by_chain[chain].size() != n_folds) {
            ctx.set_error("ridge moment batch score fold counts differ");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        for (const auto& heldout : heldout_stats_by_chain[chain]) {
            if (heldout.n_samples <= 0 ||
                heldout.n_samples >= all_stats.n_samples ||
                heldout.n_features != all_stats.n_features ||
                heldout.n_targets != all_stats.n_targets ||
                heldout.x_sum.size() != p ||
                heldout.y_sum.size() != q ||
                heldout.xtx.size() != p * p ||
                heldout.xty.size() != p * q ||
                heldout.yty.size() != q * q) {
                ctx.set_error("ridge moment batch score received inconsistent held-out moments");
                return N4M_ERR_INVALID_ARGUMENT;
            }
        }
    }

    std::vector<double> lambdas;
    if (n_ridge_lambdas == 0) {
        lambdas.push_back(cfg.ridge_lambda);
    } else {
        lambdas.assign(
            ridge_lambdas,
            ridge_lambdas + static_cast<std::size_t>(n_ridge_lambdas));
    }
    for (const double lambda : lambdas) {
        if (!std::isfinite(lambda) || lambda < 0.0) {
            ctx.set_error("ridge moment batch score lambdas must be finite and >= 0");
            return N4M_ERR_INVALID_ARGUMENT;
        }
    }

    std::vector<MomentStats> train_stats(n_chains * n_folds);
    for (std::size_t chain = 0; chain < n_chains; ++chain) {
        for (std::size_t fold = 0; fold < n_folds; ++fold) {
            const n4m_status_t st = subtract_moments(
                ctx,
                all_stats_by_chain[chain],
                heldout_stats_by_chain[chain][fold],
                train_stats[chain * n_folds + fold]);
            if (st != N4M_OK) return st;
        }
    }

    const auto n_lambdas = lambdas.size();
    const auto jobs_per_chain = n_lambdas * n_folds;
    const auto n_jobs = n_chains * jobs_per_chain;
    const auto n_train_jobs = n_chains * n_folds;
    const auto n_train_jobs_i = static_cast<std::int64_t>(n_train_jobs);
    std::vector<double> sse_by_job(n_jobs, 0.0);
    std::vector<std::int64_t> count_by_job(n_jobs, 0);
    std::vector<n4m_status_t> status_by_job(n_jobs, N4M_OK);
    std::vector<std::string> error_by_job(n_jobs);
    std::vector<std::uint8_t> eigen_path_prepared_by_train_job(
        n_train_jobs, 0);
    std::vector<std::uint8_t> eigen_path_used_by_job(n_jobs, 0);
    std::vector<std::uint8_t> direct_fit_used_by_job(n_jobs, 0);
    const bool try_eigen_path = n_lambdas > 1U;

    N4M_PARALLEL_FOR_STATIC
    for (std::int64_t train_job_i = 0; train_job_i < n_train_jobs_i;
         ++train_job_i) {
        const auto train_job = static_cast<std::size_t>(train_job_i);
        const auto chain = train_job / n_folds;
        const auto fold = train_job % n_folds;

        try {
            RidgeMomentEigenPath eigen_path;
            const RidgeMomentEigenPath* path = nullptr;
            if (try_eigen_path) {
                Context path_ctx;
                const n4m_status_t path_st = prepare_ridge_moment_eigen_path(
                    path_ctx, cfg, train_stats[train_job], eigen_path);
                if (path_st == N4M_OK) {
                    path = &eigen_path;
                    eigen_path_prepared_by_train_job[train_job] = 1;
                }
            }

            for (std::size_t lambda_ix = 0; lambda_ix < n_lambdas;
                 ++lambda_ix) {
                const auto job = chain * jobs_per_chain +
                                 lambda_ix * n_folds + fold;
                Context job_ctx;
                RidgeMomentFit fit;
                bool used_eigen_path = false;
                const n4m_status_t st = fit_ridge_from_moment_path_or_direct(
                    job_ctx, cfg, train_stats[train_job], path,
                    lambdas[lambda_ix], fit, &used_eigen_path);
                status_by_job[job] = st;
                if (st == N4M_OK) {
                    if (used_eigen_path) {
                        eigen_path_used_by_job[job] = 1;
                    } else {
                        direct_fit_used_by_job[job] = 1;
                    }
                    sse_by_job[job] = ridge_heldout_sse_from_moments(
                        heldout_stats_by_chain[chain][fold], fit);
                    count_by_job[job] =
                        heldout_stats_by_chain[chain][fold].n_samples *
                        static_cast<std::int64_t>(q);
                } else if (job_ctx.last_error()[0] != '\0') {
                    error_by_job[job] = job_ctx.last_error();
                }
            }
        } catch (const std::bad_alloc&) {
            for (std::size_t lambda_ix = 0; lambda_ix < n_lambdas;
                 ++lambda_ix) {
                const auto job = chain * jobs_per_chain +
                                 lambda_ix * n_folds + fold;
                status_by_job[job] = N4M_ERR_OUT_OF_MEMORY;
                error_by_job[job] =
                    "out of memory while fitting ridge moment batch job";
            }
        } catch (const std::exception& exc) {
            for (std::size_t lambda_ix = 0; lambda_ix < n_lambdas;
                 ++lambda_ix) {
                const auto job = chain * jobs_per_chain +
                                 lambda_ix * n_folds + fold;
                status_by_job[job] = N4M_ERR_INTERNAL;
                error_by_job[job] = exc.what();
            }
        } catch (...) {
            for (std::size_t lambda_ix = 0; lambda_ix < n_lambdas;
                 ++lambda_ix) {
                const auto job = chain * jobs_per_chain +
                                 lambda_ix * n_folds + fold;
                status_by_job[job] = N4M_ERR_INTERNAL;
                error_by_job[job] =
                    "unexpected exception while fitting ridge moment batch job";
            }
        }
    }

    for (std::size_t job = 0; job < n_jobs; ++job) {
        if (status_by_job[job] == N4M_ERR_OUT_OF_MEMORY ||
            status_by_job[job] == N4M_ERR_INTERNAL) {
            if (!error_by_job[job].empty()) {
                ctx.set_error(error_by_job[job].c_str());
            } else {
                ctx.set_error("ridge moment batch score failed");
            }
            return status_by_job[job];
        }
    }

    out.assign(n_chains, SweepResult{});
    for (std::size_t chain = 0; chain < n_chains; ++chain) {
        SweepResult result{};
        result.n_samples = all_stats_by_chain[chain].n_samples;
        result.n_features = all_stats_by_chain[chain].n_features;
        result.n_targets = all_stats_by_chain[chain].n_targets;
        result.cv = static_cast<std::int32_t>(n_folds);
        result.n_candidates = static_cast<std::int64_t>(n_lambdas);
        result.n_ridge_moment_candidates = result.n_candidates;
        result.n_ridge_moment_cv_fits =
            static_cast<std::int64_t>(n_lambdas * n_folds);
        for (std::size_t fold = 0; fold < n_folds; ++fold) {
            const auto train_job = chain * n_folds + fold;
            result.n_ridge_moment_eigen_path_preparations +=
                static_cast<std::int64_t>(
                    eigen_path_prepared_by_train_job[train_job]);
        }
        for (std::size_t lambda_ix = 0; lambda_ix < n_lambdas; ++lambda_ix) {
            for (std::size_t fold = 0; fold < n_folds; ++fold) {
                const auto job = chain * jobs_per_chain +
                                 lambda_ix * n_folds + fold;
                result.n_ridge_moment_eigen_path_cv_fits +=
                    static_cast<std::int64_t>(eigen_path_used_by_job[job]);
                result.n_ridge_moment_direct_cv_fits +=
                    static_cast<std::int64_t>(direct_fit_used_by_job[job]);
            }
        }
        if (chain == 0) {
            result.n_ridge_moment_score_batch_calls = 1;
            result.n_ridge_moment_score_batch_jobs =
                static_cast<std::int64_t>(n_jobs);
        }
        result.score_only = 1;
        result.candidate_scores.assign(n_lambdas * 4U, 0.0);

        double best = std::numeric_limits<double>::infinity();
        std::size_t best_id = 0;
        for (std::size_t lambda_ix = 0; lambda_ix < n_lambdas; ++lambda_ix) {
            bool candidate_ok = true;
            double sse = 0.0;
            std::int64_t count = 0;
            for (std::size_t fold = 0; fold < n_folds; ++fold) {
                const auto job = chain * jobs_per_chain +
                                 lambda_ix * n_folds + fold;
                if (status_by_job[job] != N4M_OK) {
                    candidate_ok = false;
                    break;
                }
                sse += sse_by_job[job];
                count += count_by_job[job];
            }
            const double rmse = (candidate_ok && count > 0)
                ? std::sqrt(std::max(0.0, sse) /
                            static_cast<double>(count))
                : std::numeric_limits<double>::infinity();
            result.candidate_scores[lambda_ix * 4U + 0U] =
                static_cast<double>(lambda_ix);
            result.candidate_scores[lambda_ix * 4U + 1U] = 0.0;
            result.candidate_scores[lambda_ix * 4U + 2U] =
                lambdas[lambda_ix];
            result.candidate_scores[lambda_ix * 4U + 3U] = rmse;
            if (rmse < best) {
                best = rmse;
                best_id = lambda_ix;
            }
        }
        if (!std::isfinite(best)) {
            ctx.set_error("ridge moment batch score: all candidates failed");
            return N4M_ERR_NUMERICAL_FAILURE;
        }

        result.selected_candidate_id = static_cast<std::int64_t>(best_id);
        result.selected_head_id = 0;
        result.selected_param = lambdas[best_id];
        result.selected_cv_rmse = best;
        out[chain] = std::move(result);
    }

    ctx.clear_error();
    return N4M_OK;
}

n4m_status_t score_pls1_moment_sweep(Context& ctx,
                                     const Config& cfg,
                                     const MomentStats& all_stats,
                                     const std::vector<MomentStats>& heldout_stats,
                                     const std::int32_t* pls_components,
                                     std::int64_t n_pls_components,
                                     SweepResult& out) {
    if (all_stats.n_samples <= 1 || all_stats.n_features <= 0 ||
        all_stats.n_targets != 1) {
        ctx.set_error("PLS1 moment score requires n>1, p>0 and q=1");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (cfg.solver != N4M_SOLVER_NIPALS ||
        cfg.deflation != N4M_DEFLATION_REGRESSION) {
        ctx.set_error("PLS1 moment score supports NIPALS regression deflation only");
        return N4M_ERR_UNSUPPORTED;
    }
    if (heldout_stats.empty()) {
        ctx.set_error("PLS1 moment score requires at least one held-out fold");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (n_pls_components < 0 ||
        (n_pls_components > 0 && pls_components == nullptr)) {
        ctx.set_error("PLS1 moment score invalid components payload");
        return N4M_ERR_INVALID_ARGUMENT;
    }

    const auto p = static_cast<std::size_t>(all_stats.n_features);
    if (all_stats.x_sum.size() != p || all_stats.y_sum.size() != 1U ||
        all_stats.xtx.size() != p * p || all_stats.xty.size() != p ||
        all_stats.yty.size() != 1U) {
        ctx.set_error("PLS1 moment score received inconsistent all-sample moments");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    std::size_t min_train = static_cast<std::size_t>(all_stats.n_samples);
    for (const auto& heldout : heldout_stats) {
        if (heldout.n_samples <= 0 ||
            heldout.n_samples >= all_stats.n_samples ||
            heldout.n_features != all_stats.n_features ||
            heldout.n_targets != 1 ||
            heldout.x_sum.size() != p ||
            heldout.y_sum.size() != 1U ||
            heldout.xtx.size() != p * p ||
            heldout.xty.size() != p ||
            heldout.yty.size() != 1U) {
            ctx.set_error("PLS1 moment score received inconsistent held-out moments");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        min_train = std::min(
            min_train,
            static_cast<std::size_t>(all_stats.n_samples - heldout.n_samples));
    }

    std::vector<std::int32_t> components;
    if (n_pls_components == 0) {
        components.push_back(cfg.n_components);
    } else {
        components.assign(
            pls_components,
            pls_components + static_cast<std::size_t>(n_pls_components));
    }
    const auto max_valid =
        static_cast<std::int32_t>(std::min<std::size_t>(p, min_train));
    for (const auto comp : components) {
        if (comp < 1 || comp > max_valid) {
            ctx.set_error("PLS1 moment score components must be in [1, min(p, n_train)]");
            return N4M_ERR_INVALID_ARGUMENT;
        }
    }
    const auto max_comp = *std::max_element(components.begin(), components.end());
    const auto fallback_components_desc = unique_components_desc(components);

    std::vector<MomentStats> train_stats(heldout_stats.size());
    for (std::size_t fold = 0; fold < heldout_stats.size(); ++fold) {
        const n4m_status_t st =
            subtract_moments(ctx, all_stats, heldout_stats[fold],
                             train_stats[fold]);
        if (st != N4M_OK) return st;
    }

    std::vector<double> sse(components.size(), 0.0);
    std::vector<std::int64_t> count(components.size(), 0);
    std::vector<unsigned char> ok(components.size(), 1);
    std::int64_t pls_moment_cv_fits = 0;
    std::int64_t pls_moment_host_cv_fits = 0;
    std::int64_t pls_moment_cuda_device_cv_fits = 0;
    std::int64_t pls_moment_cuda_parallel_fold_batches = 0;
    std::int64_t pls_moment_cuda_parallel_fold_jobs = 0;
    std::int64_t pls_moment_cuda_many_batched_batches = 0;
    std::int64_t pls_moment_cuda_many_batched_jobs = 0;
    std::vector<std::vector<RidgeMomentFit>> prefix_fits_by_fold;
    bool used_cuda_device_cv = false;
    bool used_cuda_parallel_folds = false;
    bool used_cuda_many_batched = false;
    n4m_status_t fold_fit_status =
        fit_pls1_moment_prefixes_for_folds(ctx, cfg, train_stats,
                                           max_comp, prefix_fits_by_fold,
                                           &used_cuda_device_cv,
                                           &used_cuda_parallel_folds,
                                           &used_cuda_many_batched);
    if (fold_fit_status != N4M_OK) {
        ctx.clear_error();
    } else {
        pls_moment_cv_fits = static_cast<std::int64_t>(heldout_stats.size());
        if (used_cuda_device_cv) {
            pls_moment_cuda_device_cv_fits = pls_moment_cv_fits;
            if (used_cuda_parallel_folds) {
                pls_moment_cuda_parallel_fold_batches = 1;
                pls_moment_cuda_parallel_fold_jobs = pls_moment_cv_fits;
            }
            if (used_cuda_many_batched) {
                pls_moment_cuda_many_batched_batches = 1;
                pls_moment_cuda_many_batched_jobs = pls_moment_cv_fits;
            }
        } else {
            pls_moment_host_cv_fits = pls_moment_cv_fits;
        }
    }
    for (std::size_t fold = 0;
         fold_fit_status == N4M_OK && fold < heldout_stats.size();
         ++fold) {
        for (std::size_t comp_ix = 0; comp_ix < components.size(); ++comp_ix) {
            if (ok[comp_ix] == 0) continue;
            const auto comp = static_cast<std::size_t>(components[comp_ix]);
            sse[comp_ix] += ridge_heldout_sse_from_moments(
                heldout_stats[fold], prefix_fits_by_fold[fold][comp - 1]);
            count[comp_ix] += heldout_stats[fold].n_samples;
        }
    }
    if (fold_fit_status != N4M_OK) {
        for (std::size_t fold = 0; fold < heldout_stats.size(); ++fold) {
            std::vector<RidgeMomentFit> prefix_fits;
            std::int32_t recovered_comp = 0;
            n4m_status_t st = recover_pls1_moment_prefix_for_job(
                cfg, train_stats[fold], components, fallback_components_desc,
                ok, prefix_fits, recovered_comp, pls_moment_cv_fits,
                pls_moment_host_cv_fits, pls_moment_cuda_device_cv_fits);
            if (st != N4M_OK) return st;
            for (std::size_t comp_ix = 0; comp_ix < components.size();
                 ++comp_ix) {
                if (ok[comp_ix] == 0) continue;
                const auto comp = static_cast<std::size_t>(components[comp_ix]);
                if (components[comp_ix] > recovered_comp ||
                    prefix_fits.size() < comp) {
                    ok[comp_ix] = 0;
                    continue;
                }
                sse[comp_ix] += ridge_heldout_sse_from_moments(
                    heldout_stats[fold], prefix_fits[comp - 1]);
                count[comp_ix] += heldout_stats[fold].n_samples;
            }
            bool any_needed = false;
            for (std::size_t comp_ix = 0; comp_ix < components.size();
                 ++comp_ix) {
                any_needed = any_needed || (ok[comp_ix] != 0);
            }
            if (!any_needed) break;
        }
    }

    out = SweepResult{};
    out.n_samples = all_stats.n_samples;
    out.n_features = all_stats.n_features;
    out.n_targets = all_stats.n_targets;
    out.cv = static_cast<std::int32_t>(heldout_stats.size());
    out.n_candidates = static_cast<std::int64_t>(components.size());
    out.n_pls_moment_candidates = out.n_candidates;
    out.n_pls_moment_cv_fits = pls_moment_cv_fits;
    out.n_pls_moment_host_cv_fits = pls_moment_host_cv_fits;
    out.n_pls_moment_cuda_device_cv_fits = pls_moment_cuda_device_cv_fits;
    out.n_pls_moment_cuda_parallel_fold_batches =
        pls_moment_cuda_parallel_fold_batches;
    out.n_pls_moment_cuda_parallel_fold_jobs =
        pls_moment_cuda_parallel_fold_jobs;
    out.n_pls_moment_cuda_many_batched_batches =
        pls_moment_cuda_many_batched_batches;
    out.n_pls_moment_cuda_many_batched_jobs =
        pls_moment_cuda_many_batched_jobs;
    out.candidate_scores.assign(components.size() * 4U, 0.0);
    const bool score_only = score_only_requested(cfg);
    out.score_only = score_only ? 1 : 0;

    double best = std::numeric_limits<double>::infinity();
    std::size_t best_id = 0;
    for (std::size_t comp_ix = 0; comp_ix < components.size(); ++comp_ix) {
        const double rmse = (ok[comp_ix] != 0 && count[comp_ix] > 0)
            ? std::sqrt(std::max(0.0, sse[comp_ix]) /
                        static_cast<double>(count[comp_ix]))
            : std::numeric_limits<double>::infinity();
        out.candidate_scores[comp_ix * 4U + 0U] =
            static_cast<double>(comp_ix);
        out.candidate_scores[comp_ix * 4U + 1U] = 1.0;
        out.candidate_scores[comp_ix * 4U + 2U] =
            static_cast<double>(components[comp_ix]);
        out.candidate_scores[comp_ix * 4U + 3U] = rmse;
        if (rmse < best) {
            best = rmse;
            best_id = comp_ix;
        }
    }
    if (!std::isfinite(best)) {
        ctx.set_error("PLS1 moment score: all candidates failed");
        return N4M_ERR_NUMERICAL_FAILURE;
    }

    out.selected_candidate_id = static_cast<std::int64_t>(best_id);
    out.selected_head_id = 1;
    out.selected_param = static_cast<double>(components[best_id]);
    out.selected_cv_rmse = best;
    out.score_only = score_only ? 1 : 0;

    if (score_only) {
        ctx.clear_error();
        return N4M_OK;
    }

    std::vector<RidgeMomentFit> final_prefix_fits;
    bool used_cuda_device = false;
    out.n_pls_moment_final_fits = 1;
    const auto selected_comp = static_cast<std::size_t>(components[best_id]);
    n4m_status_t st =
        fit_pls1_moment_prefixes(ctx, cfg, all_stats,
                                 static_cast<std::int32_t>(selected_comp),
                                 final_prefix_fits, &used_cuda_device);
    if (st != N4M_OK) return st;
    if (used_cuda_device) {
        out.n_pls_moment_cuda_device_final_fits = 1;
    } else {
        out.n_pls_moment_host_final_fits = 1;
    }
    const auto& final_fit = final_prefix_fits[selected_comp - 1];

    out.coefficients = final_fit.coefficients;
    out.intercept = final_fit.intercept;
    out.x_mean = final_fit.x_mean;
    out.x_scale = final_fit.x_scale;
    out.y_mean = final_fit.y_mean;
    ctx.clear_error();
    return N4M_OK;
}

n4m_status_t score_pls1_moment_sweeps_score_only(
    Context& ctx,
    const Config& cfg,
    const std::vector<MomentStats>& all_stats_by_chain,
    const std::vector<std::vector<MomentStats>>& heldout_stats_by_chain,
    const std::int32_t* pls_components,
    std::int64_t n_pls_components,
    std::vector<SweepResult>& out) {
    out.clear();
    if (!score_only_requested(cfg)) {
        ctx.set_error("PLS1 moment batch score requires score_only");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (all_stats_by_chain.empty()) {
        ctx.set_error("PLS1 moment batch score requires at least one chain");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (all_stats_by_chain.size() != heldout_stats_by_chain.size()) {
        ctx.set_error("PLS1 moment batch score all/held-out sizes differ");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (cfg.solver != N4M_SOLVER_NIPALS ||
        cfg.deflation != N4M_DEFLATION_REGRESSION) {
        ctx.set_error("PLS1 moment batch score supports NIPALS regression deflation only");
        return N4M_ERR_UNSUPPORTED;
    }
    if (n_pls_components < 0 ||
        (n_pls_components > 0 && pls_components == nullptr)) {
        ctx.set_error("PLS1 moment batch score invalid components payload");
        return N4M_ERR_INVALID_ARGUMENT;
    }

    const auto n_chains = all_stats_by_chain.size();
    const auto p = static_cast<std::size_t>(all_stats_by_chain.front().n_features);
    if (p == 0) {
        ctx.set_error("PLS1 moment batch score requires p>0");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    const auto n_folds = heldout_stats_by_chain.front().size();
    if (n_folds == 0) {
        ctx.set_error("PLS1 moment batch score requires at least one held-out fold");
        return N4M_ERR_INVALID_ARGUMENT;
    }

    std::size_t min_train =
        static_cast<std::size_t>(all_stats_by_chain.front().n_samples);
    for (std::size_t chain = 0; chain < n_chains; ++chain) {
        const auto& all_stats = all_stats_by_chain[chain];
        if (all_stats.n_samples <= 1 || all_stats.n_features <= 0 ||
            all_stats.n_targets != 1 ||
            static_cast<std::size_t>(all_stats.n_features) != p ||
            all_stats.x_sum.size() != p || all_stats.y_sum.size() != 1U ||
            all_stats.xtx.size() != p * p || all_stats.xty.size() != p ||
            all_stats.yty.size() != 1U) {
            ctx.set_error("PLS1 moment batch score received inconsistent all-sample moments");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        if (heldout_stats_by_chain[chain].size() != n_folds) {
            ctx.set_error("PLS1 moment batch score fold counts differ");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        for (const auto& heldout : heldout_stats_by_chain[chain]) {
            if (heldout.n_samples <= 0 ||
                heldout.n_samples >= all_stats.n_samples ||
                heldout.n_features != all_stats.n_features ||
                heldout.n_targets != 1 ||
                heldout.x_sum.size() != p ||
                heldout.y_sum.size() != 1U ||
                heldout.xtx.size() != p * p ||
                heldout.xty.size() != p ||
                heldout.yty.size() != 1U) {
                ctx.set_error("PLS1 moment batch score received inconsistent held-out moments");
                return N4M_ERR_INVALID_ARGUMENT;
            }
            min_train = std::min(
                min_train,
                static_cast<std::size_t>(all_stats.n_samples - heldout.n_samples));
        }
    }

    std::vector<std::int32_t> components;
    if (n_pls_components == 0) {
        components.push_back(cfg.n_components);
    } else {
        components.assign(
            pls_components,
            pls_components + static_cast<std::size_t>(n_pls_components));
    }
    const auto max_valid =
        static_cast<std::int32_t>(std::min<std::size_t>(p, min_train));
    for (const auto comp : components) {
        if (comp < 1 || comp > max_valid) {
            ctx.set_error("PLS1 moment batch score components must be in [1, min(p, n_train)]");
            return N4M_ERR_INVALID_ARGUMENT;
        }
    }
    const auto max_comp = *std::max_element(components.begin(), components.end());
    const auto fallback_components_desc = unique_components_desc(components);

    std::vector<MomentStats> train_stats(n_chains * n_folds);
    for (std::size_t chain = 0; chain < n_chains; ++chain) {
        for (std::size_t fold = 0; fold < n_folds; ++fold) {
            const n4m_status_t st = subtract_moments(
                ctx,
                all_stats_by_chain[chain],
                heldout_stats_by_chain[chain][fold],
                train_stats[chain * n_folds + fold]);
            if (st != N4M_OK) return st;
        }
    }

    std::vector<std::vector<RidgeMomentFit>> prefix_fits_by_job;
    bool used_cuda_device_cv = false;
    bool used_cuda_parallel_folds = false;
    bool used_cuda_many_batched = false;
    bool have_batched_prefix_fits = false;
    std::int32_t batched_recovered_comp = 0;
    n4m_status_t fit_status = fit_pls1_moment_prefixes_for_folds(
        ctx, cfg, train_stats, max_comp, prefix_fits_by_job,
        &used_cuda_device_cv, &used_cuda_parallel_folds,
        &used_cuda_many_batched);
    if (fit_status == N4M_OK) {
        have_batched_prefix_fits = true;
        batched_recovered_comp = max_comp;
    } else {
        // A rank-deficient late component should not fail a broad screen when
        // smaller component prefixes are still scoreable. First try to keep a
        // shared batch/GPU route for a lower requested prefix. Any higher
        // components are then handled by the exact per-job fallback below.
        ctx.clear_error();
        fit_status = recover_pls1_moment_prefixes_for_folds(
            ctx, cfg, train_stats, fallback_components_desc, max_comp,
            prefix_fits_by_job, batched_recovered_comp,
            used_cuda_device_cv, used_cuda_parallel_folds,
            used_cuda_many_batched);
        if (fit_status != N4M_OK) return fit_status;
        have_batched_prefix_fits = (batched_recovered_comp > 0);
    }

    out.assign(n_chains, SweepResult{});
    std::vector<n4m_status_t> status_by_chain(n_chains, N4M_OK);
    std::vector<std::string> error_by_chain(n_chains);
    const auto n_chains_i = static_cast<std::int64_t>(n_chains);

    N4M_PARALLEL_FOR_STATIC
    for (std::int64_t chain_i = 0; chain_i < n_chains_i; ++chain_i) {
        const auto chain = static_cast<std::size_t>(chain_i);
        try {
            std::vector<double> sse(components.size(), 0.0);
            std::vector<std::int64_t> count(components.size(), 0);
            std::vector<unsigned char> ok(components.size(), 1);
            std::int64_t moment_cv_fits = 0;
            std::int64_t moment_host_cv_fits = 0;
            std::int64_t moment_cuda_device_cv_fits = 0;
            if (have_batched_prefix_fits) {
                for (std::size_t fold = 0; fold < n_folds; ++fold) {
                    const auto job = chain * n_folds + fold;
                    for (std::size_t comp_ix = 0; comp_ix < components.size();
                         ++comp_ix) {
                        if (components[comp_ix] > batched_recovered_comp) {
                            continue;
                        }
                        const auto comp =
                            static_cast<std::size_t>(components[comp_ix]);
                        sse[comp_ix] += ridge_heldout_sse_from_moments(
                            heldout_stats_by_chain[chain][fold],
                            prefix_fits_by_job[job][comp - 1]);
                        count[comp_ix] +=
                            heldout_stats_by_chain[chain][fold].n_samples;
                    }
                }
                moment_cv_fits = static_cast<std::int64_t>(n_folds);
                if (used_cuda_device_cv) {
                    moment_cuda_device_cv_fits = moment_cv_fits;
                } else {
                    moment_host_cv_fits = moment_cv_fits;
                }
            }
            if (!have_batched_prefix_fits ||
                batched_recovered_comp < max_comp) {
                Context fit_ctx;
                for (std::size_t fold = 0; fold < n_folds; ++fold) {
                    const auto job = chain * n_folds + fold;
                    std::vector<std::int32_t> fold_attempt_components_desc;
                    fold_attempt_components_desc.reserve(
                        fallback_components_desc.size());
                    for (const auto attempt_comp :
                         fallback_components_desc) {
                        if (have_batched_prefix_fits &&
                            attempt_comp <= batched_recovered_comp) {
                            continue;
                        }
                        bool needed = false;
                        for (std::size_t comp_ix = 0;
                             comp_ix < components.size(); ++comp_ix) {
                            if (ok[comp_ix] != 0 &&
                                (!have_batched_prefix_fits ||
                                 components[comp_ix] >
                                     batched_recovered_comp) &&
                                components[comp_ix] == attempt_comp) {
                                needed = true;
                                break;
                            }
                        }
                        if (needed) {
                            fold_attempt_components_desc.push_back(
                                attempt_comp);
                        }
                    }
                    if (fold_attempt_components_desc.empty()) {
                        break;
                    }

                    std::vector<RidgeMomentFit> prefix_fits;
                    std::int32_t recovered_comp = 0;
                    for (const auto attempt_comp :
                         fold_attempt_components_desc) {
                        bool used_cuda_device = false;
                        ++moment_cv_fits;
                        const n4m_status_t st = fit_pls1_moment_prefixes(
                            fit_ctx, cfg, train_stats[job], attempt_comp,
                            prefix_fits, &used_cuda_device);
                        if (used_cuda_device) {
                            ++moment_cuda_device_cv_fits;
                        } else {
                            ++moment_host_cv_fits;
                        }
                        if (st == N4M_OK &&
                            prefix_fits.size() >=
                                static_cast<std::size_t>(attempt_comp)) {
                            recovered_comp = attempt_comp;
                            break;
                        }
                        fit_ctx.clear_error();
                        prefix_fits.clear();
                    }
                    for (std::size_t comp_ix = 0; comp_ix < components.size();
                         ++comp_ix) {
                        if (ok[comp_ix] == 0) continue;
                        if (have_batched_prefix_fits &&
                            components[comp_ix] <= batched_recovered_comp) {
                            continue;
                        }
                        const auto comp = components[comp_ix];
                        if (comp > recovered_comp ||
                            prefix_fits.size() < static_cast<std::size_t>(comp)) {
                            ok[comp_ix] = 0;
                            continue;
                        }
                        sse[comp_ix] += ridge_heldout_sse_from_moments(
                            heldout_stats_by_chain[chain][fold],
                            prefix_fits[static_cast<std::size_t>(comp - 1)]);
                        count[comp_ix] +=
                            heldout_stats_by_chain[chain][fold].n_samples;
                    }
                }
            }

            SweepResult result{};
            result.n_samples = all_stats_by_chain[chain].n_samples;
            result.n_features = all_stats_by_chain[chain].n_features;
            result.n_targets = all_stats_by_chain[chain].n_targets;
            result.cv = static_cast<std::int32_t>(n_folds);
            result.n_candidates =
                static_cast<std::int64_t>(components.size());
            result.n_pls_moment_candidates = result.n_candidates;
            result.n_pls_moment_cv_fits = moment_cv_fits;
            result.n_pls_moment_host_cv_fits = moment_host_cv_fits;
            result.n_pls_moment_cuda_device_cv_fits =
                moment_cuda_device_cv_fits;
            if (have_batched_prefix_fits && chain == 0) {
                result.n_pls_moment_score_batch_calls = 1;
                result.n_pls_moment_score_batch_jobs =
                    static_cast<std::int64_t>(train_stats.size());
            }
            if (have_batched_prefix_fits && chain == 0 && used_cuda_device_cv &&
                used_cuda_parallel_folds) {
                result.n_pls_moment_cuda_parallel_fold_batches = 1;
                result.n_pls_moment_cuda_parallel_fold_jobs =
                    static_cast<std::int64_t>(train_stats.size());
            }
            if (have_batched_prefix_fits && chain == 0 && used_cuda_device_cv &&
                used_cuda_many_batched) {
                result.n_pls_moment_cuda_many_batched_batches = 1;
                result.n_pls_moment_cuda_many_batched_jobs =
                    static_cast<std::int64_t>(train_stats.size());
            }
            result.score_only = 1;
            result.candidate_scores.assign(components.size() * 4U, 0.0);

            double best = std::numeric_limits<double>::infinity();
            std::size_t best_id = 0;
            for (std::size_t comp_ix = 0; comp_ix < components.size();
                 ++comp_ix) {
                const double rmse = (ok[comp_ix] != 0 && count[comp_ix] > 0)
                    ? std::sqrt(std::max(0.0, sse[comp_ix]) /
                                static_cast<double>(count[comp_ix]))
                    : std::numeric_limits<double>::infinity();
                result.candidate_scores[comp_ix * 4U + 0U] =
                    static_cast<double>(comp_ix);
                result.candidate_scores[comp_ix * 4U + 1U] = 1.0;
                result.candidate_scores[comp_ix * 4U + 2U] =
                    static_cast<double>(components[comp_ix]);
                result.candidate_scores[comp_ix * 4U + 3U] = rmse;
                if (rmse < best) {
                    best = rmse;
                    best_id = comp_ix;
                }
            }
            result.selected_cv_rmse = best;
            result.selected_head_id = 1;
            if (std::isfinite(best)) {
                result.selected_candidate_id = static_cast<std::int64_t>(best_id);
                result.selected_param = static_cast<double>(components[best_id]);
            }

            out[chain] = std::move(result);
        } catch (const std::bad_alloc&) {
            status_by_chain[chain] = N4M_ERR_OUT_OF_MEMORY;
            error_by_chain[chain] =
                "out of memory while scoring PLS1 moment batch chain";
        } catch (const std::exception& exc) {
            status_by_chain[chain] = N4M_ERR_INTERNAL;
            error_by_chain[chain] = exc.what();
        } catch (...) {
            status_by_chain[chain] = N4M_ERR_INTERNAL;
            error_by_chain[chain] =
                "unexpected exception while scoring PLS1 moment batch chain";
        }
    }

    for (std::size_t chain = 0; chain < n_chains; ++chain) {
        if (status_by_chain[chain] != N4M_OK) {
            if (!error_by_chain[chain].empty()) {
                ctx.set_error(error_by_chain[chain].c_str());
            } else {
                ctx.set_error("PLS1 moment batch score failed");
            }
            return status_by_chain[chain];
        }
    }

    ctx.clear_error();
    return N4M_OK;
}

n4m_status_t score_pls1_gcv_moment_screen(Context& ctx,
                                          const Config& cfg,
                                          const MomentStats& all_stats,
                                          const std::int32_t* pls_components,
                                          std::int64_t n_pls_components,
                                          SweepResult& out) {
    if (!score_only_requested(cfg)) {
        ctx.set_error("PLS1 GCV moment screen requires score_only");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (all_stats.n_samples <= 1 || all_stats.n_features <= 0 ||
        all_stats.n_targets != 1) {
        ctx.set_error("PLS1 GCV moment screen requires n>1, p>0 and q=1");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (cfg.solver != N4M_SOLVER_NIPALS ||
        cfg.deflation != N4M_DEFLATION_REGRESSION) {
        ctx.set_error("PLS1 GCV moment screen supports NIPALS regression deflation only");
        return N4M_ERR_UNSUPPORTED;
    }
    if (n_pls_components < 0 ||
        (n_pls_components > 0 && pls_components == nullptr)) {
        ctx.set_error("PLS1 GCV moment screen invalid components payload");
        return N4M_ERR_INVALID_ARGUMENT;
    }

    const auto p = static_cast<std::size_t>(all_stats.n_features);
    if (all_stats.x_sum.size() != p || all_stats.y_sum.size() != 1U ||
        all_stats.xtx.size() != p * p || all_stats.xty.size() != p ||
        all_stats.yty.size() != 1U || all_stats.cxx.size() != p * p ||
        all_stats.cxy.size() != p || all_stats.cyy.size() != 1U) {
        ctx.set_error("PLS1 GCV moment screen received inconsistent all-sample moments");
        return N4M_ERR_INVALID_ARGUMENT;
    }

    std::vector<std::int32_t> components;
    if (n_pls_components == 0) {
        components.push_back(cfg.n_components);
    } else {
        components.assign(
            pls_components,
            pls_components + static_cast<std::size_t>(n_pls_components));
    }
    const auto max_valid = static_cast<std::int32_t>(
        std::min<std::size_t>(
            p,
            static_cast<std::size_t>(all_stats.n_samples - 1)));
    for (const auto comp : components) {
        if (comp < 1 || comp > max_valid) {
            ctx.set_error(
                "PLS1 GCV moment screen components must be in [1, min(p, n-1)]");
            return N4M_ERR_INVALID_ARGUMENT;
        }
    }
    const auto max_comp = *std::max_element(components.begin(), components.end());

    std::vector<RidgeMomentFit> prefix_fits;
    n4m_status_t st =
        fit_pls1_moment_prefixes(ctx, cfg, all_stats, max_comp, prefix_fits);
    if (st != N4M_OK) return st;

    out = SweepResult{};
    out.n_samples = all_stats.n_samples;
    out.n_features = all_stats.n_features;
    out.n_targets = all_stats.n_targets;
    out.cv = 0;
    out.n_candidates = static_cast<std::int64_t>(components.size());
    out.n_pls_moment_candidates = out.n_candidates;
    out.n_pls_gcv_proxy_candidates = out.n_candidates;
    out.n_pls_gcv_proxy_fits = 1;
    out.score_only = 1;
    out.candidate_scores.assign(components.size() * 4U, 0.0);

    const double n = static_cast<double>(all_stats.n_samples);
    double best = std::numeric_limits<double>::infinity();
    std::size_t best_id = 0;
    for (std::size_t comp_ix = 0; comp_ix < components.size(); ++comp_ix) {
        const auto comp = static_cast<std::size_t>(components[comp_ix]);
        const double rss = ridge_heldout_sse_from_moments(
            all_stats, prefix_fits[comp - 1]);
        const double df = static_cast<double>(comp);
        const double denom = 1.0 - (df / n);
        const double gcv_mse =
            (denom > 0.0)
                ? (std::max(0.0, rss) / n) / (denom * denom)
                : std::numeric_limits<double>::infinity();
        const double proxy_rmse =
            std::sqrt(std::max(0.0, gcv_mse));
        out.candidate_scores[comp_ix * 4U + 0U] =
            static_cast<double>(comp_ix);
        out.candidate_scores[comp_ix * 4U + 1U] = 1.0;
        out.candidate_scores[comp_ix * 4U + 2U] =
            static_cast<double>(components[comp_ix]);
        out.candidate_scores[comp_ix * 4U + 3U] = proxy_rmse;
        if (std::isfinite(proxy_rmse) && proxy_rmse < best) {
            best = proxy_rmse;
            best_id = comp_ix;
        }
    }
    if (!std::isfinite(best)) {
        ctx.set_error("PLS1 GCV moment screen: all candidates failed");
        return N4M_ERR_NUMERICAL_FAILURE;
    }

    out.selected_candidate_id = static_cast<std::int64_t>(best_id);
    out.selected_head_id = 1;
    out.selected_param = static_cast<double>(components[best_id]);
    out.selected_cv_rmse = best;
    ctx.clear_error();
    return N4M_OK;
}

n4m_status_t score_pls1_gcv_moment_screens_score_only(
    Context& ctx,
    const Config& cfg,
    const std::vector<MomentStats>& all_stats_by_chain,
    const std::int32_t* pls_components,
    std::int64_t n_pls_components,
    std::vector<SweepResult>& out) {
    out.clear();
    if (!score_only_requested(cfg)) {
        ctx.set_error("PLS1 GCV moment batch screen requires score_only");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (all_stats_by_chain.empty()) {
        ctx.set_error("PLS1 GCV moment batch screen requires at least one chain");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (cfg.solver != N4M_SOLVER_NIPALS ||
        cfg.deflation != N4M_DEFLATION_REGRESSION) {
        ctx.set_error("PLS1 GCV moment batch screen supports NIPALS regression deflation only");
        return N4M_ERR_UNSUPPORTED;
    }
    if (n_pls_components < 0 ||
        (n_pls_components > 0 && pls_components == nullptr)) {
        ctx.set_error("PLS1 GCV moment batch screen invalid components payload");
        return N4M_ERR_INVALID_ARGUMENT;
    }

    const auto n_chains = all_stats_by_chain.size();
    const auto p = static_cast<std::size_t>(all_stats_by_chain.front().n_features);
    if (p == 0) {
        ctx.set_error("PLS1 GCV moment batch screen requires p>0");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    std::size_t min_rows =
        static_cast<std::size_t>(all_stats_by_chain.front().n_samples);
    for (const auto& all_stats : all_stats_by_chain) {
        if (all_stats.n_samples <= 1 || all_stats.n_features <= 0 ||
            all_stats.n_targets != 1 ||
            static_cast<std::size_t>(all_stats.n_features) != p ||
            all_stats.x_sum.size() != p || all_stats.y_sum.size() != 1U ||
            all_stats.xtx.size() != p * p || all_stats.xty.size() != p ||
            all_stats.yty.size() != 1U || all_stats.cxx.size() != p * p ||
            all_stats.cxy.size() != p || all_stats.cyy.size() != 1U) {
            ctx.set_error("PLS1 GCV moment batch screen received inconsistent moments");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        min_rows = std::min(
            min_rows,
            static_cast<std::size_t>(all_stats.n_samples));
    }

    std::vector<std::int32_t> components;
    if (n_pls_components == 0) {
        components.push_back(cfg.n_components);
    } else {
        components.assign(
            pls_components,
            pls_components + static_cast<std::size_t>(n_pls_components));
    }
    const auto max_valid = static_cast<std::int32_t>(
        std::min<std::size_t>(p, min_rows - 1U));
    for (const auto comp : components) {
        if (comp < 1 || comp > max_valid) {
            ctx.set_error(
                "PLS1 GCV moment batch screen components must be in [1, min(p, n-1)]");
            return N4M_ERR_INVALID_ARGUMENT;
        }
    }
    const auto max_comp = *std::max_element(components.begin(), components.end());

    std::vector<std::vector<RidgeMomentFit>> prefix_fits_by_chain;
    bool used_cuda_device = false;
    n4m_status_t st = fit_pls1_moment_prefixes_for_folds(
        ctx, cfg, all_stats_by_chain, max_comp,
        prefix_fits_by_chain, &used_cuda_device);
    if (st != N4M_OK) return st;

    out.assign(n_chains, SweepResult{});
    for (std::size_t chain = 0; chain < n_chains; ++chain) {
        SweepResult result{};
        result.n_samples = all_stats_by_chain[chain].n_samples;
        result.n_features = all_stats_by_chain[chain].n_features;
        result.n_targets = all_stats_by_chain[chain].n_targets;
        result.cv = 0;
        result.n_candidates = static_cast<std::int64_t>(components.size());
        result.n_pls_moment_candidates = result.n_candidates;
        result.n_pls_gcv_proxy_candidates = result.n_candidates;
        result.n_pls_gcv_proxy_fits = 1;
        if (chain == 0) {
            result.n_pls_gcv_proxy_batch_calls = 1;
            result.n_pls_gcv_proxy_batch_jobs =
                static_cast<std::int64_t>(n_chains);
        }
        result.score_only = 1;
        result.candidate_scores.assign(components.size() * 4U, 0.0);

        const double n = static_cast<double>(all_stats_by_chain[chain].n_samples);
        double best = std::numeric_limits<double>::infinity();
        std::size_t best_id = 0;
        for (std::size_t comp_ix = 0; comp_ix < components.size(); ++comp_ix) {
            const auto comp = static_cast<std::size_t>(components[comp_ix]);
            const double rss = ridge_heldout_sse_from_moments(
                all_stats_by_chain[chain],
                prefix_fits_by_chain[chain][comp - 1]);
            const double df = static_cast<double>(comp);
            const double denom = 1.0 - (df / n);
            const double gcv_mse =
                (denom > 0.0)
                    ? (std::max(0.0, rss) / n) / (denom * denom)
                    : std::numeric_limits<double>::infinity();
            const double proxy_rmse = std::sqrt(std::max(0.0, gcv_mse));
            result.candidate_scores[comp_ix * 4U + 0U] =
                static_cast<double>(comp_ix);
            result.candidate_scores[comp_ix * 4U + 1U] = 1.0;
            result.candidate_scores[comp_ix * 4U + 2U] =
                static_cast<double>(components[comp_ix]);
            result.candidate_scores[comp_ix * 4U + 3U] = proxy_rmse;
            if (std::isfinite(proxy_rmse) && proxy_rmse < best) {
                best = proxy_rmse;
                best_id = comp_ix;
            }
        }
        if (!std::isfinite(best)) {
            ctx.set_error("PLS1 GCV moment batch screen: all candidates failed");
            return N4M_ERR_NUMERICAL_FAILURE;
        }

        result.selected_candidate_id = static_cast<std::int64_t>(best_id);
        result.selected_head_id = 1;
        result.selected_param = static_cast<double>(components[best_id]);
        result.selected_cv_rmse = best;
        out[chain] = std::move(result);
    }

    (void)used_cuda_device;
    ctx.clear_error();
    return N4M_OK;
}

}  // namespace n4m::core
