// SPDX-License-Identifier: CECILL-2.1
#include "core/spectral_encoding.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#include "core/common/svd.h"
#include "core/linalg.hpp"

#if defined(N4M_SPECTRAL_USE_LAPACKE)
#    include <lapacke.h>
#endif

namespace n4m::core {
namespace {
using Vec = std::vector<double>;
Vec prepare(const Vec& x, std::size_t n, std::size_t p, bool snv) {
    Vec z = x;
    if (snv)
        for (std::size_t i = 0; i < n; ++i) {
            double mean = 0, variance = 0;
            for (std::size_t j = 0; j < p; ++j)
                mean += z[i * p + j] / static_cast<double>(p);
            for (std::size_t j = 0; j < p; ++j)
                variance += std::pow(z[i * p + j] - mean, 2) / static_cast<double>(p);
            double scale = std::max(std::sqrt(variance), 1e-8);
            for (std::size_t j = 0; j < p; ++j)
                z[i * p + j] = (z[i * p + j] - mean) / scale;
        }
    return z;
}
// Cyclic nonnegative coordinate descent, fixed right factor. Returns the
// projected-gradient violation used by the alternating NMF stopping criterion.
double
coordinate_update(const Vec& x, const Vec& h, Vec& w, std::size_t n, std::size_t p, std::size_t r) {
    Vec gram(r * r), cross(n * r);
    for (std::size_t a = 0; a < r; ++a) {
        for (std::size_t b = 0; b < r; ++b)
            for (std::size_t j = 0; j < p; ++j)
                gram[a * r + b] += h[a * p + j] * h[b * p + j];
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < p; ++j)
                cross[i * r + a] += x[i * p + j] * h[a * p + j];
    }
    double violation = 0;
    for (std::size_t a = 0; a < r; ++a)
        for (std::size_t i = 0; i < n; ++i) {
            double gradient = -cross[i * r + a];
            for (std::size_t b = 0; b < r; ++b)
                gradient += gram[a * r + b] * w[i * r + b];
            violation += std::abs(w[i * r + a] == 0 ? std::min(0.0, gradient) : gradient);
            if (gram[a * r + a] > 0)
                w[i * r + a] = std::max(0.0, w[i * r + a] - gradient / gram[a * r + a]);
        }
    return violation;
}
Vec transpose(const Vec& x, std::size_t n, std::size_t p) {
    Vec t(x.size());
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j)
            t[j * n + i] = x[i * p + j];
    return t;
}
n4m_status_t svd_with_fallback(
    const Vec& x, std::size_t n, std::size_t p, std::size_t r, Vec& u, Vec& s, Vec& vt) {
    const auto rows = static_cast<int64_t>(n);
    const auto cols = static_cast<int64_t>(p);
    const auto count = static_cast<int64_t>(r);
    auto status = n >= p ? n4m_svd_truncated_tall_gram(
                               x.data(), rows, cols, count, u.data(), s.data(), vt.data())
                         : n4m_svd_truncated_dual_wide(
                               x.data(), rows, cols, count, u.data(), s.data(), vt.data());
    if (status == N4M_OK
        || (status != N4M_ERR_CONVERGENCE_FAILED && status != N4M_ERR_NUMERICAL_FAILURE))
        return status;
    // Preserve the exact leading-r local subspace if the portable Gram QL
    // iteration fails. Other statuses (including allocation failure) propagate.
    std::fill(u.begin(), u.end(), 0.0);
    std::fill(s.begin(), s.end(), 0.0);
    std::fill(vt.begin(), vt.end(), 0.0);
#if defined(N4M_SPECTRAL_USE_LAPACKE)
    const auto d = std::min(n, p);
    if (d > static_cast<std::size_t>(std::numeric_limits<lapack_int>::max()))
        return N4M_ERR_INVALID_ARGUMENT;
    Vec xt = transpose(x, n, p), gram(d * d), eigenvalues(d);
    if (n >= p)
        n4m::linalg::gemm(n4m::linalg::Trans_No,
                          n4m::linalg::Trans_No,
                          p,
                          p,
                          n,
                          1.0,
                          xt.data(),
                          n,
                          x.data(),
                          p,
                          0.0,
                          gram.data(),
                          p);
    else
        n4m::linalg::gemm(n4m::linalg::Trans_No,
                          n4m::linalg::Trans_No,
                          n,
                          n,
                          p,
                          1.0,
                          x.data(),
                          p,
                          xt.data(),
                          n,
                          0.0,
                          gram.data(),
                          n);
    const auto info = LAPACKE_dsyevd(LAPACK_ROW_MAJOR,
                                     'V',
                                     'U',
                                     static_cast<lapack_int>(d),
                                     gram.data(),
                                     static_cast<lapack_int>(d),
                                     eigenvalues.data());
    if (info == LAPACK_WORK_MEMORY_ERROR || info == LAPACK_TRANSPOSE_MEMORY_ERROR)
        return N4M_ERR_OUT_OF_MEMORY;
    if (info < 0)
        return N4M_ERR_INTERNAL;
    if (info > 0)
        return N4M_ERR_CONVERGENCE_FAILED;
    for (std::size_t k = 0; k < r; ++k)
        s[k] = std::sqrt(std::max(0.0, eigenvalues[d - 1 - k]));
    const double cutoff = 4 * std::sqrt(std::numeric_limits<double>::epsilon()) * s[0];
    if (n >= p) {
        for (std::size_t k = 0; k < r; ++k) {
            if (s[k] <= cutoff)
                continue;
            for (std::size_t j = 0; j < p; ++j)
                vt[k * p + j] = gram[j * d + d - 1 - k];
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < p; ++j)
                    u[i * r + k] += x[i * p + j] * vt[k * p + j] / s[k];
        }
    } else {
        for (std::size_t k = 0; k < r; ++k) {
            if (s[k] <= cutoff)
                continue;
            for (std::size_t i = 0; i < n; ++i)
                u[i * r + k] = gram[i * d + d - 1 - k];
            for (std::size_t j = 0; j < p; ++j)
                for (std::size_t i = 0; i < n; ++i)
                    vt[k * p + j] += u[i * r + k] * x[i * p + j] / s[k];
        }
    }
    return N4M_OK;
#else
    // Portable builds use the independent one-sided Jacobi decomposition.
    const auto d = std::min(n, p);
    Vec copy(x), full_u(n * d), full_s(d), full_vt(d * p);
    status = n4m_svd_compact(copy.data(), rows, cols, full_u.data(), full_s.data(), full_vt.data());
    if (status != N4M_OK)
        return status;
    for (std::size_t k = 0; k < r; ++k) {
        s[k] = full_s[k];
        for (std::size_t i = 0; i < n; ++i)
            u[i * r + k] = full_u[i * d + k];
        for (std::size_t j = 0; j < p; ++j)
            vt[k * p + j] = full_vt[k * p + j];
    }
    return N4M_OK;
#endif
}
}  // namespace

n4m_status_t SpectralEncoding::fit(const Vec& x, std::size_t n, std::size_t p) {
    if (n < 2 || p < 1 || x.size() != n * p || kind < 0 || kind > 1 || width < 1 || rank < 1
        || iterations < 1 || !std::isfinite(overlap) || overlap < 0 || overlap >= 1
        || !std::isfinite(tolerance) || tolerance < 0 || (kind == 1 && snv))
        return N4M_ERR_INVALID_ARGUMENT;
    for (double v : x)
        if (!std::isfinite(v))
            return N4M_ERR_INVALID_ARGUMENT;
    Vec z = prepare(x, n, p, snv);
    features = p;
    outputs = 0;
    center.assign(p, 0);
    scale.assign(p, 1);
    basis.clear();
    if (kind == 0) {
        for (std::size_t j = 0; j < p; ++j) {
            for (std::size_t i = 0; i < n; ++i)
                center[j] += z[i * p + j] / static_cast<double>(n);
            double variance = 0;
            for (std::size_t i = 0; i < n; ++i)
                variance += std::pow(z[i * p + j] - center[j], 2) / static_cast<double>(n);
            if (standardize)
                scale[j] = std::max(std::sqrt(variance), 1e-8);
            for (std::size_t i = 0; i < n; ++i)
                z[i * p + j] = (z[i * p + j] - center[j]) / scale[j];
        }
        const auto w = std::min(p, static_cast<std::size_t>(width));
        std::vector<std::size_t> starts;
        if (overlap == 0) {
            for (std::size_t start = 0; start < p; start += w)
                starts.push_back(start);
        } else {
            const auto step = std::max(
                std::size_t{1},
                static_cast<std::size_t>(std::nearbyint(static_cast<double>(w) * (1 - overlap))));
            for (std::size_t start = 0; start + w <= p; start += step)
                starts.push_back(start);
            if (starts.back() != p - w)
                starts.push_back(p - w);
        }
        // Store one row per mode (q x p); also the orientation of the public
        // affine operator: transform(X) = (X - center) @ basis.T.
        for (auto start : starts) {
            const auto cols = std::min(w, p - start);
            const auto r = std::min(static_cast<std::size_t>(rank), std::min(n, cols) - 1);
            if (!r)
                continue;
            Vec block(n * cols), u(n * r), s(r), vt(r * cols);
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < cols; ++j)
                    block[i * cols + j] = z[i * p + start + j];
            auto status = svd_with_fallback(block, n, cols, r, u, s, vt);
            if (status != N4M_OK)
                return status;
            for (std::size_t a = 0; a < r; ++a) {
                basis.resize((outputs + 1) * p, 0);
                // Gram eigenvalues below roundoff cannot yield reliable
                // right vectors (the dual path divides by their square root).
                // Keep the output shape, but use a zero mode in both paths.
                const double cutoff = 4 * std::sqrt(std::numeric_limits<double>::epsilon()) * s[0];
                if (s[a] > cutoff)
                    for (std::size_t j = 0; j < cols; ++j)
                        basis[outputs * p + start + j] = vt[a * cols + j] / scale[start + j];
                ++outputs;
            }
        }
        return outputs ? N4M_OK : N4M_ERR_INVALID_ARGUMENT;
    }
    const auto r = std::min(static_cast<std::size_t>(rank), std::min(n, p) - 1);
    if (!r)
        return N4M_ERR_INVALID_ARGUMENT;
    double mean = std::accumulate(z.begin(), z.end(), 0.0) / static_cast<double>(z.size());
    double variance = 0;
    for (double v : z)
        variance += (v - mean) * (v - mean) / static_cast<double>(z.size());
    std::fill(scale.begin(), scale.end(), std::max(std::sqrt(variance), 1e-8));
    for (std::size_t j = 0; j < p; ++j) {
        center[j] = z[j];
        for (std::size_t i = 0; i < n; ++i)
            center[j] = std::min(center[j], z[i * p + j]);
        for (std::size_t i = 0; i < n; ++i)
            z[i * p + j] = std::max(0.0, (z[i * p + j] - center[j]) / scale[j]);
    }
    Vec u(n * r), s(r), vt(r * p);
    auto status = svd_with_fallback(z, n, p, r, u, s, vt);
    if (status != N4M_OK)
        return status;
    Vec w(n * r), h(r * p);
    for (std::size_t i = 0; i < n; ++i)
        w[i * r] = std::sqrt(s[0]) * std::abs(u[i * r]);
    for (std::size_t j = 0; j < p; ++j)
        h[j] = std::sqrt(s[0]) * std::abs(vt[j]);
    for (std::size_t a = 1; a < r; ++a) {
        double up = 0, un = 0, vp = 0, vn = 0;
        for (std::size_t i = 0; i < n; ++i) {
            double v = u[i * r + a];
            (v > 0 ? up : un) += v * v;
        }
        for (std::size_t j = 0; j < p; ++j) {
            double v = vt[a * p + j];
            (v > 0 ? vp : vn) += v * v;
        }
        bool positive = up * vp > un * vn;
        double nu = std::sqrt(positive ? up : un), nv = std::sqrt(positive ? vp : vn);
        double factor = std::sqrt(s[a] * nu * nv), sign = positive ? 1.0 : -1.0;
        if (nu > 0 && nv > 0) {
            for (std::size_t i = 0; i < n; ++i)
                w[i * r + a] = factor * std::max(0.0, sign * u[i * r + a]) / nu;
            for (std::size_t j = 0; j < p; ++j)
                h[a * p + j] = factor * std::max(0.0, sign * vt[a * p + j]) / nv;
        }
    }
    mean = std::accumulate(z.begin(), z.end(), 0.0) / static_cast<double>(z.size());
    for (double& v : w)
        if (v < 1e-6)
            v = mean;
    for (double& v : h)
        if (v < 1e-6)
            v = mean;
    Vec zt = transpose(z, n, p);
    double initial = 0;
    for (int it = 0; it < iterations; ++it) {
        double violation = coordinate_update(z, h, w, n, p, r);
        Vec ht = transpose(h, r, p), wt = transpose(w, n, r);
        violation += coordinate_update(zt, wt, ht, p, n, r);
        h = transpose(ht, p, r);
        if (it == 0)
            initial = violation;
        if (initial == 0 || violation <= tolerance * initial)
            break;
    }
    basis = std::move(h);
    outputs = r;
    return N4M_OK;
}

n4m_status_t SpectralEncoding::transform(const Vec& x, std::size_t n, Vec& out) const {
    if (!outputs || x.size() != n * features)
        return N4M_ERR_INVALID_ARGUMENT;
    for (double v : x)
        if (!std::isfinite(v))
            return N4M_ERR_INVALID_ARGUMENT;
    Vec z = prepare(x, n, features, snv);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < features; ++j) {
            z[i * features + j] -= center[j];
            if (kind == 1)
                z[i * features + j] = std::max(0.0, z[i * features + j] / scale[j]);
        }
    out.assign(n * outputs, 0);
    if (kind == 0) {
        for (std::size_t a = 0; a < outputs; ++a)
            for (std::size_t j = 0; j < features; ++j) {
                const double coefficient = basis[a * features + j];
                if (coefficient != 0)
                    for (std::size_t i = 0; i < n; ++i)
                        out[i * outputs + a] += z[i * features + j] * coefficient;
            }
    } else {
        double initial = 0;
        for (int it = 0; it < iterations; ++it) {
            double violation = coordinate_update(z, basis, out, n, features, outputs);
            if (it == 0)
                initial = violation;
            if (initial == 0 || violation <= tolerance * initial)
                break;
        }
    }
    return N4M_OK;
}
}  // namespace n4m::core
