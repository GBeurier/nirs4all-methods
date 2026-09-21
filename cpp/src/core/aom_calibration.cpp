// SPDX-License-Identifier: CECILL-2.1
#include "core/aom_calibration.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>

#if defined(N4M_AOM_USE_LAPACKE)
#    include <lapacke.h>
#endif

#include "core/aom_operators.hpp"
#include "core/common/matrix_view.hpp"
#include "core/common/svd.h"
#include "core/component_coefficients.hpp"
#include "core/config.hpp"
#include "core/linalg.hpp"
#include "core/model.hpp"
#include "core/ridge.hpp"

namespace n4m::core {
namespace {
using Vec = std::vector<double>;
struct Failure {
    n4m_status_t status;
};
void require(n4m_status_t status) {
    if (status != N4M_OK)
        throw Failure{status};
}
n4m_matrix_view_t view(Vec& x, std::size_t n, std::size_t p) {
    n4m_matrix_view_t v{};
    v.data = x.data();
    v.rows = static_cast<std::int64_t>(n);
    v.cols = static_cast<std::int64_t>(p);
    v.row_stride = static_cast<std::int64_t>(p);
    v.col_stride = 1;
    v.dtype = N4M_DTYPE_F64;
    return v;
}
double dot(const Vec& a, const Vec& b) {
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}
Vec transpose(const Vec& x, std::size_t n, std::size_t p) {
    Vec result(x.size());
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j)
            result[j * n + i] = x[i * p + j];
    return result;
}
Vec product(const Vec& a, const Vec& b, std::size_t n, std::size_t p, std::size_t q) {
    Vec result(n * q);
    n4m::linalg::gemm(n4m::linalg::Trans_No,
                      n4m::linalg::Trans_No,
                      n,
                      q,
                      p,
                      1.0,
                      a.data(),
                      p,
                      b.data(),
                      q,
                      0.0,
                      result.data(),
                      q);
    return result;
}
struct Branch {
    Vec train, valid, mean, reference;
    int kind;
};
Vec branch_apply(const Vec& x, std::size_t n, std::size_t p, int kind, const Vec& ref) {
    Vec result = x;
    if (kind == 0)
        return result;
    const double width = static_cast<double>(p);
    double refmean = std::accumulate(ref.begin(), ref.end(), 0.0) / width;
    double refss = 0;
    if (kind == 2)
        for (double v : ref)
            refss += (v - refmean) * (v - refmean);
    for (std::size_t i = 0; i < n; ++i) {
        double mean = 0;
        for (std::size_t j = 0; j < p; ++j)
            mean += x[i * p + j];
        mean /= width;
        double denom = 0;
        for (std::size_t j = 0; j < p; ++j)
            denom += (x[i * p + j] - mean) * (kind == 1 ? x[i * p + j] - mean : ref[j] - refmean);
        if (kind == 1)
            denom = std::sqrt(denom / width);
        else
            denom = refss > 1e-12 ? denom / refss : 0.0;
        const double intercept = kind == 2 ? mean - denom * refmean : mean;
        if (std::abs(denom) < 1e-12)
            denom = 1.0;
        for (std::size_t j = 0; j < p; ++j)
            result[i * p + j] = (x[i * p + j] - intercept) / denom;
    }
    return result;
}
Branch prepare(const Vec& x, const Vec& v, std::size_t n, std::size_t m, std::size_t p, int kind) {
    Branch b;
    b.kind = kind;
    b.reference.assign(p, 0);
    b.mean.assign(p, 0);
    const double sample_count = static_cast<double>(n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j)
            b.reference[j] += x[i * p + j] / sample_count;
    b.train = branch_apply(x, n, p, kind, b.reference);
    b.valid = branch_apply(v, m, p, kind, b.reference);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j)
            b.mean[j] += b.train[i * p + j] / sample_count;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j)
            b.train[i * p + j] -= b.mean[j];
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < p; ++j)
            b.valid[i * p + j] -= b.mean[j];
    return b;
}
Vec transform(Context& ctx,
              Vec x,
              std::size_t n,
              std::size_t p,
              const OperatorBank& bank,
              const std::int32_t* offsets,
              std::size_t chain) {
    for (std::int32_t i = offsets[chain]; i < offsets[chain + 1]; ++i) {
        Vec result;
        auto v = view(x, n, p);
        require(transform_aom_strict_operator(
            ctx, bank.entries()[static_cast<std::size_t>(i)], v, result));
        x.swap(result);
    }
    return x;
}
Vec adjoint(Context& ctx,
            const Vec& beta,
            std::size_t p,
            const OperatorBank& bank,
            const std::int32_t* offsets,
            std::size_t chain) {
    // Only the final selected coefficient is folded; avoid a dense chain matrix.
    Vec current = beta;
    for (std::int32_t i = offsets[chain + 1] - 1; i >= offsets[chain]; --i) {
        const auto& entry = bank.entries()[static_cast<std::size_t>(i)];
        if (entry.kind == N4M_OP_DETREND_POLY || entry.kind == N4M_OP_IDENTITY) {
            Vec result;
            auto v = view(current, 1, p);
            require(transform_aom_strict_operator(ctx, entry, v, result));
            current.swap(result);
        } else {
            BandedLinearOperator op;
            require(build_aom_banded_operator(ctx, entry, static_cast<std::int64_t>(p), op));
            Vec result(p);
            for (std::size_t j = 0; j < p; ++j)
                for (int k = op.offsets[j]; k < op.offsets[j + 1]; ++k) {
                    const auto coefficient_index = static_cast<std::size_t>(k);
                    const auto source_index =
                        static_cast<std::size_t>(op.src_indices[coefficient_index]);
                    result[source_index] += op.coeffs[coefficient_index] * current[j];
                }
            current.swap(result);
        }
    }
    return current;
}
struct Path {
    Vec coefficients;
    int count;
};
Path pls_path(Context& ctx, Vec z, Vec y, std::size_t n, std::size_t p, int count) {
    Config cfg;
    cfg.solver = N4M_SOLVER_SIMPLS;
    cfg.center_x = cfg.center_y = 0;
    cfg.scale_x = cfg.scale_y = 0;
    cfg.n_components =
        static_cast<int>(std::min(static_cast<std::size_t>(count), std::min(n - 1, p)));
    cfg.tol = 1e-12;
    std::unique_ptr<Model> model;
    auto xv = view(z, n, p);
    auto yv = view(y, n, 1);
    require(fit_model(ctx, cfg, xv, yv, model));
    Path result;
    result.count = model->n_components;
    require(compute_regression_coefficients_by_component(ctx, *model, result.coefficients));
    return result;
}
Path ridge_path(
    const Vec& z, const Vec& y, std::size_t n, std::size_t p, const double* alphas, int count) {
    bool primal = n > p;
    std::size_t d = primal ? p : n;
    Vec zt = transpose(z, n, p);
    Vec gram = primal ? product(zt, z, p, n, p) : product(z, zt, n, p, n);
    Vec values(d), vectors(d * d);
#if defined(N4M_AOM_USE_LAPACKE)
    if (d > static_cast<std::size_t>(std::numeric_limits<lapack_int>::max()))
        throw Failure{N4M_ERR_INVALID_ARGUMENT};
    vectors = gram;
    const auto eigen_status = LAPACKE_dsyevd(LAPACK_ROW_MAJOR,
                                             'V',
                                             'U',
                                             static_cast<lapack_int>(d),
                                             vectors.data(),
                                             static_cast<lapack_int>(d),
                                             values.data())
                                      == 0
                                  ? N4M_OK
                                  : N4M_ERR_CONVERGENCE_FAILED;
#else
    const auto eigen_status = n4m_symmetric_eigh(
        gram.data(), static_cast<std::int64_t>(d), values.data(), vectors.data());
#endif
    if (eigen_status == N4M_ERR_CONVERGENCE_FAILED || eigen_status == N4M_ERR_NUMERICAL_FAILURE) {
        // Preserve the declared alpha grid and CV objective if the portable
        // eigensolver does not converge on a degenerate operator view.
        // Reuse the existing direct Ridge kernel; never discard this view.
        Context fallback_context;
        Config config;
        config.center_x = config.center_y = config.scale_x = config.scale_y = 0;
        Vec design(z), target(y);
        auto xv = view(design, n, p);
        auto yv = view(target, n, 1);
        Path result{{}, count};
        result.coefficients.reserve(static_cast<std::size_t>(count) * p);
        for (int a = 0; a < count; ++a) {
            RidgeResult fit;
            require(fit_ridge(
                fallback_context, config, xv, yv, alphas[a], RidgeSolver::kAuto, false, fit));
            result.coefficients.insert(
                result.coefficients.end(), fit.coefficients.begin(), fit.coefficients.end());
        }
        return result;
    }
    require(eigen_status);
    Vec rhs = primal ? product(zt, y, p, n, 1) : y;
    Vec projected = product(transpose(vectors, d, d), rhs, d, d, 1);
    Vec map = primal ? vectors : product(zt, vectors, p, n, n);
    Path result;
    result.count = count;
    result.coefficients.resize(static_cast<std::size_t>(count) * p);
    Vec weights(d);
    for (int a = 0; a < count; ++a) {
        for (std::size_t j = 0; j < d; ++j)
            weights[j] = projected[j] / (std::max(0.0, values[j]) + alphas[a]);
        Vec beta = product(map, weights, p, d, 1);
        const auto offset = static_cast<Vec::difference_type>(static_cast<std::size_t>(a) * p);
        std::copy(beta.begin(), beta.end(), result.coefficients.begin() + offset);
    }
    return result;
}
std::pair<std::size_t, std::size_t> screen(Context& ctx,
                                           const std::vector<Branch>& branches,
                                           const Vec& y,
                                           std::size_t n,
                                           std::size_t p,
                                           const OperatorBank& bank,
                                           const std::int32_t* offsets,
                                           std::size_t chains,
                                           std::size_t rank) {
    double best = -1;
    std::pair<std::size_t, std::size_t> selected{0, 0};
    double yy = dot(y, y);
    const auto r = std::min(rank, std::min(n, p));
    for (std::size_t b = 0; b < branches.size(); ++b) {
        const auto& x = branches[b].train;
        Vec u(n * r), s(r), vt(r * p);
        const auto rows = static_cast<std::int64_t>(n);
        const auto cols = static_cast<std::int64_t>(p);
        const auto rank_count = static_cast<std::int64_t>(r);
        require(n <= p ? n4m_svd_truncated_dual_wide(
                             x.data(), rows, cols, rank_count, u.data(), s.data(), vt.data())
                       : n4m_svd_truncated_tall_gram(
                             x.data(), rows, cols, rank_count, u.data(), s.data(), vt.data()));
        for (std::size_t k = 0; k < r; ++k)
            for (std::size_t j = 0; j < p; ++j)
                vt[k * p + j] *= s[k];
        Vec g = product(transpose(x, n, p), y, p, n, 1);
        for (std::size_t c = 0; c < chains; ++c) {
            Vec ag = transform(ctx, g, 1, p, bank, offsets, c);
            Vec av = transform(ctx, vt, r, p, bank, offsets, c);
            double score = dot(ag, ag) / (dot(av, av) * yy + 1e-12);
            if (score > best) {
                best = score;
                selected = {b, c};
            }
        }
    }
    return selected;
}
}  // namespace
n4m_status_t fit_aom_calibration(Context& ctx,
                                 const n4m_matrix_view_t& X,
                                 const n4m_matrix_view_t& Y,
                                 const OperatorBank& bank,
                                 const std::int32_t* offsets,
                                 std::int32_t chains,
                                 const std::int32_t* kinds,
                                 std::int32_t nb,
                                 const std::int32_t* folds,
                                 std::int32_t nf,
                                 std::int32_t head,
                                 bool fast,
                                 std::int32_t rank,
                                 const double* alphas,
                                 std::int32_t np,
                                 AomCalibrationResult& out) {
    try {
        require(validate_nonnull_view(X));
        require(validate_nonnull_view(Y));
        if (X.dtype != N4M_DTYPE_F64 || Y.dtype != N4M_DTYPE_F64)
            return N4M_ERR_DTYPE_MISMATCH;
        if (X.rows < 3 || X.cols < 1 || Y.rows != X.rows || Y.cols != 1 || chains < 1 || nb < 1
            || nf < 2 || np < 1 || rank < 1 || head < 0 || head > 3)
            return N4M_ERR_INVALID_ARGUMENT;
        if (offsets[0] != 0 || offsets[chains] != bank.size())
            return N4M_ERR_INVALID_ARGUMENT;
        for (int c = 0; c < chains; ++c)
            if (offsets[c] < 0 || offsets[c] >= offsets[c + 1])
                return N4M_ERR_INVALID_ARGUMENT;
        for (int b = 0; b < nb; ++b)
            if (kinds[b] < 0 || kinds[b] > 2)
                return N4M_ERR_INVALID_ARGUMENT;
        if (head != 0)
            for (int a = 0; a < np; ++a)
                if (!std::isfinite(alphas[a]) || alphas[a] <= 0)
                    return N4M_ERR_INVALID_ARGUMENT;
        const bool pooled_rmse = head == 3;
        if (pooled_rmse)
            head = 1;
        const auto n = static_cast<std::size_t>(X.rows);
        const auto p = static_cast<std::size_t>(X.cols);
        const auto chain_count = static_cast<std::size_t>(chains);
        const auto branch_count = static_cast<std::size_t>(nb);
        const auto fold_count = static_cast<std::size_t>(nf);
        const auto parameter_count = static_cast<std::size_t>(np);
        if (p > std::numeric_limits<std::size_t>::max() / sizeof(double) / n)
            return N4M_ERR_INVALID_ARGUMENT;
        Vec x(n * p), y(n);
        const auto* y_data = static_cast<const double*>(Y.data);
        const auto* x_data = static_cast<const double*>(X.data);
        for (std::size_t i = 0; i < n; ++i) {
            if (folds[i] < 0 || folds[i] >= nf)
                return N4M_ERR_INVALID_ARGUMENT;
            const auto row = static_cast<std::int64_t>(i);
            y[i] = *(y_data + row * Y.row_stride);
            if (!std::isfinite(y[i]))
                return N4M_ERR_INVALID_ARGUMENT;
            for (std::size_t j = 0; j < p; ++j) {
                const auto column = static_cast<std::int64_t>(j);
                x[i * p + j] = *(x_data + row * X.row_stride + column * X.col_stride);
                if (!std::isfinite(x[i * p + j]))
                    return N4M_ERR_INVALID_ARGUMENT;
            }
        }
        double alpha_base = 1.0;
        Vec absolute_alphas;
        if (head == 2) {
            const double sample_count = static_cast<double>(n);
            Vec mean_x(p);
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < p; ++j)
                    mean_x[j] += x[i * p + j] / sample_count;
            alpha_base = 0;
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < p; ++j)
                    alpha_base +=
                        (x[i * p + j] - mean_x[j]) * (x[i * p + j] - mean_x[j]) / sample_count;
            alpha_base = std::max(alpha_base, 1e-12);
            absolute_alphas.assign(alphas, alphas + np);
            for (double& a : absolute_alphas)
                a *= alpha_base;
            alphas = absolute_alphas.data();
            head = 1;
        }
        const std::size_t cells =
            fast ? parameter_count : branch_count * chain_count * parameter_count;
        out.scores.assign(cells, 0.0);
        for (std::size_t f = 0; f < fold_count; ++f) {
            Vec tr, va, yt, yv;
            for (std::size_t i = 0; i < n; ++i) {
                const bool validation_row = folds[i] == static_cast<std::int32_t>(f);
                auto& dest = validation_row ? va : tr;
                auto& target = validation_row ? yv : yt;
                const auto begin_offset = static_cast<Vec::difference_type>(i * p);
                const auto end_offset = static_cast<Vec::difference_type>((i + 1) * p);
                dest.insert(dest.end(), x.begin() + begin_offset, x.begin() + end_offset);
                target.push_back(y[i]);
            }
            if (yt.size() < 2 || yv.empty())
                return N4M_ERR_INVALID_ARGUMENT;
            double mean =
                std::accumulate(yt.begin(), yt.end(), 0.0) / static_cast<double>(yt.size());
            for (double& v : yt)
                v -= mean;
            std::vector<Branch> branches;
            for (std::size_t b = 0; b < branch_count; ++b)
                branches.push_back(prepare(tr, va, yt.size(), yv.size(), p, kinds[b]));
            auto chosen = fast ? screen(ctx,
                                        branches,
                                        yt,
                                        yt.size(),
                                        p,
                                        bank,
                                        offsets,
                                        chain_count,
                                        static_cast<std::size_t>(rank))
                               : std::pair<std::size_t, std::size_t>{0, 0};
            for (std::size_t b = 0; b < branch_count; ++b)
                for (std::size_t c = 0; c < chain_count; ++c) {
                    if (fast && (b != chosen.first || c != chosen.second))
                        continue;
                    Vec z = transform(ctx, branches[b].train, yt.size(), p, bank, offsets, c);
                    Vec zv = transform(ctx, branches[b].valid, yv.size(), p, bank, offsets, c);
                    Path path{{}, 0};
                    try {
                        path = head == 0 ? pls_path(ctx, z, yt, yt.size(), p, np)
                                         : ridge_path(z, yt, yt.size(), p, alphas, np);
                    } catch (const Failure& failure) {
                        if (failure.status != N4M_ERR_NUMERICAL_FAILURE)
                            throw;
                        // A failed candidate is invalid in this fold; never average
                        // its score over fewer folds or abort other valid candidates.
                    }
                    for (std::size_t a = 0; a < parameter_count; ++a) {
                        const std::size_t cell =
                            fast ? a : (b * chain_count + c) * parameter_count + a;
                        if (a >= static_cast<std::size_t>(path.count)) {
                            out.scores[cell] = std::numeric_limits<double>::infinity();
                            continue;
                        }
                        double sse = 0;
                        for (std::size_t i = 0; i < yv.size(); ++i) {
                            double prediction = mean;
                            for (std::size_t j = 0; j < p; ++j)
                                prediction += zv[i * p + j] * path.coefficients[a * p + j];
                            sse += (prediction - yv[i]) * (prediction - yv[i]);
                        }
                        out.scores[cell] += pooled_rmse
                                                ? sse / static_cast<double>(n)
                                                : std::sqrt(sse / static_cast<double>(yv.size()))
                                                      / static_cast<double>(nf);
                    }
                }
        }
        if (pooled_rmse)
            for (double& score : out.scores)
                score = std::sqrt(score);
        auto winner = std::min_element(out.scores.begin(), out.scores.end());
        if (!std::isfinite(*winner))
            return N4M_ERR_NUMERICAL_FAILURE;
        const auto selected = static_cast<std::size_t>(std::distance(out.scores.begin(), winner));
        out.parameter = static_cast<std::int32_t>(selected % parameter_count);
        out.chain = static_cast<std::int32_t>((selected / parameter_count) % chain_count);
        out.branch = static_cast<std::int32_t>(selected / (parameter_count * chain_count));
        double mean = std::accumulate(y.begin(), y.end(), 0.0) / static_cast<double>(n);
        for (double& v : y)
            v -= mean;
        std::vector<Branch> branches;
        for (std::size_t b = 0; b < branch_count; ++b)
            branches.push_back(prepare(x, {}, n, 0, p, kinds[b]));
        if (fast) {
            auto selected_pair = screen(
                ctx, branches, y, n, p, bank, offsets, chain_count, static_cast<std::size_t>(rank));
            out.branch = static_cast<std::int32_t>(selected_pair.first);
            out.chain = static_cast<std::int32_t>(selected_pair.second);
        }
        auto& branch = branches[static_cast<std::size_t>(out.branch)];
        Vec z =
            transform(ctx, branch.train, n, p, bank, offsets, static_cast<std::size_t>(out.chain));
        Path path = head == 0 ? pls_path(ctx, z, y, n, p, out.parameter + 1)
                              : ridge_path(z, y, n, p, alphas + out.parameter, 1);
        const auto a = head == 0 ? static_cast<std::size_t>(out.parameter) : 0U;
        if (a >= static_cast<std::size_t>(path.count))
            return N4M_ERR_NUMERICAL_FAILURE;
        const auto begin_offset = static_cast<Vec::difference_type>(a * p);
        const auto end_offset = static_cast<Vec::difference_type>((a + 1) * p);
        Vec beta(path.coefficients.begin() + begin_offset, path.coefficients.begin() + end_offset);
        out.coefficients =
            adjoint(ctx, beta, p, bank, offsets, static_cast<std::size_t>(out.chain));
        out.state = branch.mean;
        out.state.insert(out.state.end(), branch.reference.begin(), branch.reference.end());
        out.state.push_back(mean - dot(branch.mean, out.coefficients));
        out.state.push_back(alpha_base);
        ctx.clear_error();
        return N4M_OK;
    } catch (const Failure& failure) {
        return failure.status;
    } catch (const std::bad_alloc&) {
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}
n4m_status_t predict_aom_calibration(const n4m_matrix_view_t& X,
                                     int branch,
                                     const double* coef,
                                     const double* state,
                                     n4m_matrix_view_t& out) {
    if (validate_nonnull_view(X) != N4M_OK || validate_nonnull_view(out) != N4M_OK)
        return N4M_ERR_INVALID_ARGUMENT;
    if (X.dtype != N4M_DTYPE_F64 || out.dtype != N4M_DTYPE_F64)
        return N4M_ERR_DTYPE_MISMATCH;
    if (X.cols < 1 || X.rows != out.rows || out.cols != 1 || branch < 0 || branch > 2)
        return N4M_ERR_SHAPE_MISMATCH;
    try {
        const auto n = static_cast<std::size_t>(X.rows);
        const auto p = static_cast<std::size_t>(X.cols);
        if (n > 0 && p > std::numeric_limits<std::size_t>::max() / sizeof(double) / n)
            return N4M_ERR_INVALID_ARGUMENT;
        Vec x(n * p), ref(state + p, state + 2 * p);
        const auto* x_data = static_cast<const double*>(X.data);
        auto* output_data = static_cast<double*>(out.data);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < p; ++j) {
                const auto row = static_cast<std::int64_t>(i);
                const auto column = static_cast<std::int64_t>(j);
                double v = *(x_data + row * X.row_stride + column * X.col_stride);
                if (!std::isfinite(v))
                    return N4M_ERR_INVALID_ARGUMENT;
                x[i * p + j] = v;
            }
        Vec z = branch_apply(x, n, p, branch, ref);
        for (std::size_t i = 0; i < n; ++i) {
            double value = state[2 * p];
            for (std::size_t j = 0; j < p; ++j)
                value += z[i * p + j] * coef[j];
            const auto row = static_cast<std::int64_t>(i);
            *(output_data + row * out.row_stride) = value;
        }
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

}  // namespace n4m::core
