// SPDX-License-Identifier: CECILL-2.1
//
// C ABI for advanced chemometric operators promoted from the Python Tier-2
// surface. The implementations live here for now to keep the promotion scoped:
// the public ABI remains opaque handles and row-major F64 matrices only.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <new>
#include <utility>
#include <vector>

#include "n4m/n4m.h"
#include "core/common/matrix_view.hpp"

namespace {

struct MatrixIn {
    const double* data = nullptr;
    std::int64_t rows = 0;
    std::int64_t cols = 0;
};

struct MatrixOut {
    double* data = nullptr;
    std::int64_t rows = 0;
    std::int64_t cols = 0;
};

n4m_status_t require_f64_rowmajor(const n4m_matrix_view_t& v,
                                  MatrixIn& out) noexcept {
    const n4m_status_t s = ::n4m::core::validate_nonnull_view(v);
    if (s != N4M_OK) return s;
    if (v.dtype != N4M_DTYPE_F64) return N4M_ERR_DTYPE_MISMATCH;
    if (v.col_stride != 1) return N4M_ERR_STRIDE_INVALID;
    if (v.rows > 0 && v.cols > 0 && v.row_stride != v.cols) {
        return N4M_ERR_STRIDE_INVALID;
    }
    out.data = static_cast<const double*>(v.data);
    out.rows = v.rows;
    out.cols = v.cols;
    return N4M_OK;
}

n4m_status_t require_f64_rowmajor_mut(n4m_matrix_view_t& v,
                                      MatrixOut& out) noexcept {
    const n4m_status_t s = ::n4m::core::validate_nonnull_view(v);
    if (s != N4M_OK) return s;
    if (v.dtype != N4M_DTYPE_F64) return N4M_ERR_DTYPE_MISMATCH;
    if (v.col_stride != 1) return N4M_ERR_STRIDE_INVALID;
    if (v.rows > 0 && v.cols > 0 && v.row_stride != v.cols) {
        return N4M_ERR_STRIDE_INVALID;
    }
    out.data = static_cast<double*>(v.data);
    out.rows = v.rows;
    out.cols = v.cols;
    return N4M_OK;
}

n4m_status_t require_same_shape(const MatrixIn& a, const MatrixIn& b) noexcept {
    return (a.rows == b.rows && a.cols == b.cols) ? N4M_OK : N4M_ERR_SHAPE_MISMATCH;
}

n4m_status_t require_out_shape(const MatrixIn& x, const MatrixOut& out) noexcept {
    return (x.rows == out.rows && x.cols == out.cols) ? N4M_OK : N4M_ERR_SHAPE_MISMATCH;
}

std::size_t to_index(std::int64_t value) noexcept {
    return static_cast<std::size_t>(value);
}

std::size_t row_major(std::size_t row, std::size_t cols, std::size_t col) noexcept {
    return row * cols + col;
}

template <typename Row, typename Col>
double at(const MatrixIn& m, Row i, Col j) noexcept {
    return m.data[row_major(static_cast<std::size_t>(i), to_index(m.cols),
                            static_cast<std::size_t>(j))];
}

bool solve_linear(std::vector<double> a, std::vector<double> b,
                  std::int64_t n, std::vector<double>& x) {
    const std::size_t n_sz = to_index(n);
    x.assign(n_sz, 0.0);
    for (std::size_t k = 0; k < n_sz; ++k) {
        std::size_t pivot = k;
        double best = std::fabs(a[row_major(k, n_sz, k)]);
        for (std::size_t i = k + 1; i < n_sz; ++i) {
            const double v = std::fabs(a[row_major(i, n_sz, k)]);
            if (v > best) {
                best = v;
                pivot = i;
            }
        }
        if (best <= std::numeric_limits<double>::epsilon()) return false;
        if (pivot != k) {
            for (std::size_t j = k; j < n_sz; ++j) {
                std::swap(a[row_major(k, n_sz, j)],
                          a[row_major(pivot, n_sz, j)]);
            }
            std::swap(b[k], b[pivot]);
        }
        const double diag = a[row_major(k, n_sz, k)];
        for (std::size_t i = k + 1; i < n_sz; ++i) {
            const double factor = a[row_major(i, n_sz, k)] / diag;
            a[row_major(i, n_sz, k)] = 0.0;
            for (std::size_t j = k + 1; j < n_sz; ++j) {
                a[row_major(i, n_sz, j)] -= factor * a[row_major(k, n_sz, j)];
            }
            b[i] -= factor * b[k];
        }
    }
    for (std::size_t i = n_sz; i-- > 0; ) {
        double s = b[i];
        for (std::size_t j = i + 1; j < n_sz; ++j) {
            s -= a[row_major(i, n_sz, j)] * x[j];
        }
        const double diag = a[row_major(i, n_sz, i)];
        if (diag == 0.0) return false;
        x[i] = s / diag;
    }
    return true;
}

std::vector<double> fit_ols(const MatrixIn& x, const MatrixIn& y,
                            bool intercept, double ridge,
                            const std::vector<std::int64_t>* rows = nullptr,
                            std::int64_t lo = 0, std::int64_t hi = -1) {
    const std::int64_t x_cols = (hi < 0) ? x.cols : (hi - lo);
    const std::int64_t params = x_cols + (intercept ? 1 : 0);
    const std::size_t params_sz = to_index(params);
    const std::size_t y_cols_sz = to_index(y.cols);
    const std::size_t n_rows_sz = rows ? rows->size() : to_index(x.rows);
    std::vector<double> xtx(params_sz * params_sz, 0.0);
    std::vector<double> coef(params_sz * y_cols_sz, 0.0);
    std::vector<double> xty(params_sz, 0.0);

    for (std::size_t rr = 0; rr < n_rows_sz; ++rr) {
        const std::int64_t i = rows ? (*rows)[rr] : static_cast<std::int64_t>(rr);
        for (std::size_t a = 0; a < params_sz; ++a) {
            const std::int64_t a_i = static_cast<std::int64_t>(a);
            const double va = (intercept && a_i == params - 1) ? 1.0 : at(x, i, lo + a_i);
            for (std::size_t b = 0; b < params_sz; ++b) {
                const std::int64_t b_i = static_cast<std::int64_t>(b);
                const double vb = (intercept && b_i == params - 1) ? 1.0 : at(x, i, lo + b_i);
                xtx[row_major(a, params_sz, b)] += va * vb;
            }
        }
    }
    if (ridge > 0.0) {
        for (std::size_t a = 0; a < params_sz; ++a) {
            xtx[row_major(a, params_sz, a)] += ridge;
        }
    }
    for (std::size_t out_col = 0; out_col < y_cols_sz; ++out_col) {
        const std::int64_t out_col_i = static_cast<std::int64_t>(out_col);
        std::fill(xty.begin(), xty.end(), 0.0);
        for (std::size_t rr = 0; rr < n_rows_sz; ++rr) {
            const std::int64_t i = rows ? (*rows)[rr] : static_cast<std::int64_t>(rr);
            const double yy = at(y, i, out_col_i);
            for (std::size_t a = 0; a < params_sz; ++a) {
                const std::int64_t a_i = static_cast<std::int64_t>(a);
                const double va = (intercept && a_i == params - 1) ? 1.0 : at(x, i, lo + a_i);
                xty[a] += va * yy;
            }
        }
        std::vector<double> sol;
        if (!solve_linear(xtx, xty, params, sol)) {
            std::vector<double> regularized = xtx;
            double diag_scale = 0.0;
            for (std::size_t a = 0; a < params_sz; ++a) {
                diag_scale += std::fabs(xtx[row_major(a, params_sz, a)]);
            }
            diag_scale = diag_scale > 0.0 ? diag_scale / static_cast<double>(params) : 1.0;
            const double jitter = (ridge > 0.0 ? ridge * 1e-8 : 1e-12) * diag_scale;
            for (std::size_t a = 0; a < params_sz; ++a) {
                regularized[row_major(a, params_sz, a)] += jitter;
            }
            if (!solve_linear(regularized, xty, params, sol)) {
                throw N4M_ERR_NUMERICAL_FAILURE;
            }
        }
        for (std::size_t a = 0; a < params_sz; ++a) {
            coef[row_major(a, y_cols_sz, out_col)] = sol[a];
        }
    }
    return coef;
}

double quantile_sorted(std::vector<double> values, double q) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    if (q <= 0.0) return values.front();
    if (q >= 1.0) return values.back();
    const double pos = q * static_cast<double>(values.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
    const std::size_t hi = static_cast<std::size_t>(std::ceil(pos));
    const double frac = pos - static_cast<double>(lo);
    return values[lo] * (1.0 - frac) + values[hi] * frac;
}

std::vector<std::pair<std::int64_t, std::int64_t>> intervals(std::int64_t cols,
                                                             std::int64_t width,
                                                             std::int64_t step) {
    width = std::max<std::int64_t>(1, width);
    step = std::max<std::int64_t>(1, step);
    std::vector<std::pair<std::int64_t, std::int64_t>> out;
    for (std::int64_t lo = 0; lo < cols; lo += step) {
        out.emplace_back(lo, std::min(cols, lo + width));
    }
    return out;
}

double shift_sample(const double* row, std::int64_t cols, std::int64_t j,
                    int shift) {
    const double pos = static_cast<double>(j) - static_cast<double>(shift);
    if (pos <= 0.0) return row[0];
    if (pos >= static_cast<double>(cols - 1)) return row[cols - 1];
    const std::int64_t lo = static_cast<std::int64_t>(std::floor(pos));
    const double frac = pos - static_cast<double>(lo);
    return row[lo] * (1.0 - frac) + row[lo + 1] * frac;
}

struct DSState {
    bool fit_intercept = true;
    double ridge = 0.0;
    bool fitted = false;
    std::int64_t features = 0;
    std::vector<double> coef;
};

n4m_status_t ds_fit(DSState& s, const n4m_matrix_view_t& source_v,
                    const n4m_matrix_view_t& target_v) {
    MatrixIn source, target;
    n4m_status_t st = require_f64_rowmajor(source_v, source);
    if (st != N4M_OK) return st;
    st = require_f64_rowmajor(target_v, target);
    if (st != N4M_OK) return st;
    st = require_same_shape(source, target);
    if (st != N4M_OK) return st;
    if (source.rows < 1 || source.cols < 1) return N4M_ERR_INVALID_ARGUMENT;
    s.coef = fit_ols(source, target, s.fit_intercept, s.ridge);
    s.features = source.cols;
    s.fitted = true;
    return N4M_OK;
}

n4m_status_t ds_transform(const DSState& s, const n4m_matrix_view_t& x_v,
                          n4m_matrix_view_t& out_v) {
    if (!s.fitted) return N4M_ERR_NOT_FITTED;
    MatrixIn x;
    MatrixOut out;
    n4m_status_t st = require_f64_rowmajor(x_v, x);
    if (st != N4M_OK) return st;
    st = require_f64_rowmajor_mut(out_v, out);
    if (st != N4M_OK) return st;
    if (x.cols != s.features || out.rows != x.rows || out.cols != s.features) {
        return N4M_ERR_SHAPE_MISMATCH;
    }
    const std::size_t x_cols = to_index(x.cols);
    const std::size_t out_cols = to_index(out.cols);
    const std::size_t features = to_index(s.features);
    const std::size_t params = features + (s.fit_intercept ? 1U : 0U);
    for (std::size_t i = 0; i < to_index(x.rows); ++i) {
        const double* row = x.data + row_major(i, x_cols, 0);
        double* dst = out.data + row_major(i, out_cols, 0);
        for (std::size_t j = 0; j < features; ++j) {
            double value = 0.0;
            for (std::size_t k = 0; k < features; ++k) {
                value += row[k] * s.coef[row_major(k, features, j)];
            }
            if (s.fit_intercept) {
                value += s.coef[row_major(params - 1, features, j)];
            }
            dst[j] = value;
        }
    }
    return N4M_OK;
}

struct PDSState {
    std::int32_t window = 5;
    bool fit_intercept = true;
    double ridge = 0.0;
    bool fitted = false;
    std::int64_t features = 0;
    std::vector<std::pair<std::int64_t, std::int64_t>> windows;
    std::vector<std::vector<double>> coefs;
};

n4m_status_t pds_fit(PDSState& s, const n4m_matrix_view_t& source_v,
                     const n4m_matrix_view_t& target_v) {
    MatrixIn source, target;
    n4m_status_t st = require_f64_rowmajor(source_v, source);
    if (st != N4M_OK) return st;
    st = require_f64_rowmajor(target_v, target);
    if (st != N4M_OK) return st;
    st = require_same_shape(source, target);
    if (st != N4M_OK) return st;
    if (source.rows < 1 || source.cols < 1 || s.window < 1) return N4M_ERR_INVALID_ARGUMENT;
    s.features = source.cols;
    s.windows.clear();
    s.coefs.clear();
    const std::int64_t half = s.window / 2;
    for (std::int64_t j = 0; j < source.cols; ++j) {
        const std::int64_t lo = std::max<std::int64_t>(0, j - half);
        const std::int64_t hi = std::min<std::int64_t>(source.cols, j + half + 1);
        std::vector<double> ycol(static_cast<std::size_t>(source.rows));
        MatrixIn y{ycol.data(), source.rows, 1};
        for (std::int64_t i = 0; i < source.rows; ++i) {
            ycol[static_cast<std::size_t>(i)] = at(target, i, j);
        }
        s.windows.emplace_back(lo, hi);
        s.coefs.push_back(fit_ols(source, y, s.fit_intercept, s.ridge, nullptr, lo, hi));
    }
    s.fitted = true;
    return N4M_OK;
}

n4m_status_t pds_transform(const PDSState& s, const n4m_matrix_view_t& x_v,
                           n4m_matrix_view_t& out_v) {
    if (!s.fitted) return N4M_ERR_NOT_FITTED;
    MatrixIn x;
    MatrixOut out;
    n4m_status_t st = require_f64_rowmajor(x_v, x);
    if (st != N4M_OK) return st;
    st = require_f64_rowmajor_mut(out_v, out);
    if (st != N4M_OK) return st;
    if (x.cols != s.features || out.rows != x.rows || out.cols != s.features) {
        return N4M_ERR_SHAPE_MISMATCH;
    }
    const std::size_t x_cols = to_index(x.cols);
    const std::size_t out_cols = to_index(out.cols);
    const std::size_t features = to_index(s.features);
    for (std::size_t i = 0; i < to_index(x.rows); ++i) {
        const double* row = x.data + row_major(i, x_cols, 0);
        double* dst = out.data + row_major(i, out_cols, 0);
        for (std::size_t j = 0; j < features; ++j) {
            const auto [lo, hi] = s.windows[j];
            const auto& coef = s.coefs[j];
            double value = 0.0;
            for (std::int64_t k = lo; k < hi; ++k) {
                value += row[k] * coef[static_cast<std::size_t>(k - lo)];
            }
            if (s.fit_intercept) value += coef[static_cast<std::size_t>(hi - lo)];
            dst[j] = value;
        }
    }
    return N4M_OK;
}

struct SAPSState {
    std::int32_t n_components = 5;
    double score_weight = 1.0;
    bool fit_intercept = true;
    double ridge = 0.0;
    bool fitted = false;
    std::int64_t features = 0;
    std::int64_t components = 0;
    std::vector<double> mean;
    std::vector<double> loadings;
    std::vector<double> coef;
};

void fit_power_components(SAPSState& s, const MatrixIn& x) {
    const std::size_t rows = to_index(x.rows);
    const std::size_t cols = to_index(x.cols);
    s.mean.assign(cols, 0.0);
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) s.mean[j] += at(x, i, j);
    }
    for (double& v : s.mean) v /= static_cast<double>(x.rows);
    s.components = std::min<std::int64_t>(std::max<std::int32_t>(1, s.n_components), x.cols);
    const std::size_t components = to_index(s.components);
    s.loadings.assign(components * cols, 0.0);
    std::vector<double> cov(cols * cols, 0.0);
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t a = 0; a < cols; ++a) {
            const double va = at(x, i, a) - s.mean[a];
            for (std::size_t b = 0; b < cols; ++b) {
                cov[row_major(a, cols, b)] += va * (at(x, i, b) - s.mean[b]);
            }
        }
    }
    for (std::size_t comp = 0; comp < components; ++comp) {
        std::vector<double> v(cols, 0.0);
        v[comp % cols] = 1.0;
        for (int iter = 0; iter < 30; ++iter) {
            std::vector<double> next(cols, 0.0);
            for (std::size_t a = 0; a < cols; ++a) {
                for (std::size_t b = 0; b < cols; ++b) {
                    next[a] += cov[row_major(a, cols, b)] * v[b];
                }
            }
            double norm = 0.0;
            for (double value : next) norm += value * value;
            norm = std::sqrt(norm);
            if (norm <= 0.0) break;
            for (double& value : next) value /= norm;
            v.swap(next);
        }
        double lambda = 0.0;
        for (std::size_t a = 0; a < cols; ++a) {
            for (std::size_t b = 0; b < cols; ++b) {
                lambda += v[a] * cov[row_major(a, cols, b)] * v[b];
            }
        }
        for (std::size_t a = 0; a < cols; ++a) {
            s.loadings[row_major(comp, cols, a)] = v[a];
        }
        for (std::size_t a = 0; a < cols; ++a) {
            for (std::size_t b = 0; b < cols; ++b) {
                cov[row_major(a, cols, b)] -= lambda * v[a] * v[b];
            }
        }
    }
}

std::vector<double> augmented_matrix(const SAPSState& s, const MatrixIn& x) {
    const std::int64_t cols = x.cols + s.components;
    const std::size_t rows_sz = to_index(x.rows);
    const std::size_t x_cols = to_index(x.cols);
    const std::size_t components = to_index(s.components);
    const std::size_t cols_sz = to_index(cols);
    std::vector<double> out(rows_sz * cols_sz, 0.0);
    for (std::size_t i = 0; i < rows_sz; ++i) {
        for (std::size_t j = 0; j < x_cols; ++j) {
            out[row_major(i, cols_sz, j)] = at(x, i, j);
        }
        for (std::size_t c = 0; c < components; ++c) {
            double score = 0.0;
            for (std::size_t j = 0; j < x_cols; ++j) {
                score += (at(x, i, j) - s.mean[j]) *
                         s.loadings[row_major(c, x_cols, j)];
            }
            out[row_major(i, cols_sz, x_cols + c)] = s.score_weight * score;
        }
    }
    return out;
}

n4m_status_t saps_fit(SAPSState& s, const n4m_matrix_view_t& source_v,
                      const n4m_matrix_view_t& target_v) {
    MatrixIn source, target;
    n4m_status_t st = require_f64_rowmajor(source_v, source);
    if (st != N4M_OK) return st;
    st = require_f64_rowmajor(target_v, target);
    if (st != N4M_OK) return st;
    st = require_same_shape(source, target);
    if (st != N4M_OK) return st;
    fit_power_components(s, source);
    std::vector<double> aug = augmented_matrix(s, source);
    MatrixIn a{aug.data(), source.rows, source.cols + s.components};
    s.coef = fit_ols(a, target, s.fit_intercept, s.ridge);
    s.features = source.cols;
    s.fitted = true;
    return N4M_OK;
}

n4m_status_t saps_transform(const SAPSState& s, const n4m_matrix_view_t& x_v,
                            n4m_matrix_view_t& out_v) {
    if (!s.fitted) return N4M_ERR_NOT_FITTED;
    MatrixIn x;
    MatrixOut out;
    n4m_status_t st = require_f64_rowmajor(x_v, x);
    if (st != N4M_OK) return st;
    st = require_f64_rowmajor_mut(out_v, out);
    if (st != N4M_OK) return st;
    if (x.cols != s.features || out.rows != x.rows || out.cols != s.features) {
        return N4M_ERR_SHAPE_MISMATCH;
    }
    std::vector<double> aug = augmented_matrix(s, x);
    const std::int64_t a_cols = x.cols + s.components;
    const std::size_t rows = to_index(x.rows);
    const std::size_t a_cols_sz = to_index(a_cols);
    const std::size_t features = to_index(s.features);
    const std::size_t out_cols = to_index(out.cols);
    const std::size_t params = a_cols_sz + (s.fit_intercept ? 1U : 0U);
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < features; ++j) {
            double value = 0.0;
            for (std::size_t k = 0; k < a_cols_sz; ++k) {
                value += aug[row_major(i, a_cols_sz, k)] *
                         s.coef[row_major(k, features, j)];
            }
            if (s.fit_intercept) {
                value += s.coef[row_major(params - 1, features, j)];
            }
            out.data[row_major(i, out_cols, j)] = value;
        }
    }
    return N4M_OK;
}

struct SlopeBiasState {
    bool fitted = false;
    double slope = 1.0;
    double bias = 0.0;
};

n4m_status_t slope_fit(SlopeBiasState& s, const double* source,
                       const double* target, std::int64_t n) {
    if (source == nullptr || target == nullptr) return N4M_ERR_NULL_POINTER;
    if (n < 1) return N4M_ERR_INVALID_ARGUMENT;
    double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
    for (std::int64_t i = 0; i < n; ++i) {
        sx += source[i];
        sy += target[i];
        sxx += source[i] * source[i];
        sxy += source[i] * target[i];
    }
    const double denom = static_cast<double>(n) * sxx - sx * sx;
    if (std::fabs(denom) <= std::numeric_limits<double>::epsilon()) {
        return N4M_ERR_NUMERICAL_FAILURE;
    }
    s.slope = (static_cast<double>(n) * sxy - sx * sy) / denom;
    s.bias = (sy - s.slope * sx) / static_cast<double>(n);
    s.fitted = true;
    return N4M_OK;
}

struct VectorWeightsState {
    bool fitted = false;
    bool has_initial = false;
    double eps = 1e-12;
    std::int32_t ddof = 0;
    std::vector<double> initial;
    std::vector<double> weights;
};

n4m_status_t normalize_weights(std::vector<double>& w) {
    double total = 0.0;
    for (double value : w) total += value;
    if (!(total > 0.0)) return N4M_ERR_INVALID_ARGUMENT;
    for (double& value : w) value /= total;
    return N4M_OK;
}

n4m_status_t weighted_snv_fit(VectorWeightsState& s, const n4m_matrix_view_t& x_v) {
    MatrixIn x;
    n4m_status_t st = require_f64_rowmajor(x_v, x);
    if (st != N4M_OK) return st;
    if (x.cols < 1) return N4M_ERR_INVALID_ARGUMENT;
    if (s.has_initial) {
        if (static_cast<std::int64_t>(s.initial.size()) != x.cols) return N4M_ERR_SHAPE_MISMATCH;
        s.weights = s.initial;
    } else {
        s.weights.assign(static_cast<std::size_t>(x.cols), 1.0);
    }
    st = normalize_weights(s.weights);
    if (st != N4M_OK) return st;
    s.fitted = true;
    return N4M_OK;
}

n4m_status_t vsn_fit(VectorWeightsState& s, const n4m_matrix_view_t& x_v) {
    MatrixIn x;
    n4m_status_t st = require_f64_rowmajor(x_v, x);
    if (st != N4M_OK) return st;
    const std::size_t rows = to_index(x.rows);
    const std::size_t cols = to_index(x.cols);
    std::vector<double> col_mean(cols, 0.0);
    std::vector<double> row_mean(rows, 0.0);
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            const double value = at(x, i, j);
            col_mean[j] += value;
            row_mean[i] += value;
        }
    }
    for (double& v : col_mean) v /= static_cast<double>(x.rows);
    for (double& v : row_mean) v /= static_cast<double>(x.cols);
    double row_grand = 0.0;
    for (double v : row_mean) row_grand += v;
    row_grand /= static_cast<double>(x.rows);
    double row_norm = 0.0;
    for (double v : row_mean) row_norm += (v - row_grand) * (v - row_grand);
    row_norm = std::sqrt(row_norm);
    s.weights.assign(cols, s.eps);
    for (std::size_t j = 0; j < cols; ++j) {
        double dot = 0.0;
        double col_norm = 0.0;
        for (std::size_t i = 0; i < rows; ++i) {
            const double xc = at(x, i, j) - col_mean[j];
            dot += xc * (row_mean[i] - row_grand);
            col_norm += xc * xc;
        }
        const double denom = std::sqrt(col_norm) * row_norm;
        if (denom > s.eps) s.weights[j] = std::fabs(dot / denom);
        if (s.weights[j] < s.eps) s.weights[j] = s.eps;
    }
    st = normalize_weights(s.weights);
    if (st != N4M_OK) return st;
    s.fitted = true;
    return N4M_OK;
}

n4m_status_t weighted_snv_transform(const VectorWeightsState& s,
                                    const n4m_matrix_view_t& x_v,
                                    n4m_matrix_view_t& out_v) {
    if (!s.fitted) return N4M_ERR_NOT_FITTED;
    MatrixIn x;
    MatrixOut out;
    n4m_status_t st = require_f64_rowmajor(x_v, x);
    if (st != N4M_OK) return st;
    st = require_f64_rowmajor_mut(out_v, out);
    if (st != N4M_OK) return st;
    st = require_out_shape(x, out);
    if (st != N4M_OK) return st;
    if (static_cast<std::int64_t>(s.weights.size()) != x.cols) return N4M_ERR_SHAPE_MISMATCH;
    const std::size_t rows = to_index(x.rows);
    const std::size_t cols = to_index(x.cols);
    const std::size_t out_cols = to_index(out.cols);
    for (std::size_t i = 0; i < rows; ++i) {
        double mean = 0.0;
        for (std::size_t j = 0; j < cols; ++j) mean += at(x, i, j) * s.weights[j];
        double var = 0.0;
        for (std::size_t j = 0; j < cols; ++j) {
            const double c = at(x, i, j) - mean;
            var += c * c * s.weights[j];
        }
        if (s.ddof != 0 && x.cols > s.ddof) {
            var *= static_cast<double>(x.cols) / static_cast<double>(x.cols - s.ddof);
        }
        const double scale = std::sqrt(std::max(var, s.eps));
        for (std::size_t j = 0; j < cols; ++j) {
            out.data[row_major(i, out_cols, j)] = (at(x, i, j) - mean) / scale;
        }
    }
    return N4M_OK;
}

struct PiecewiseSNVState {
    bool fitted = false;
    std::int32_t window = 32;
    std::int32_t ddof = 0;
    double eps = 1e-12;
    std::int64_t features = 0;
    std::vector<std::pair<std::int64_t, std::int64_t>> bands;
};

n4m_status_t piecewise_snv_fit(PiecewiseSNVState& s, const n4m_matrix_view_t& x_v) {
    MatrixIn x;
    n4m_status_t st = require_f64_rowmajor(x_v, x);
    if (st != N4M_OK) return st;
    s.features = x.cols;
    s.bands = intervals(x.cols, s.window, s.window);
    s.fitted = true;
    return N4M_OK;
}

n4m_status_t piecewise_snv_transform(const PiecewiseSNVState& s,
                                     const n4m_matrix_view_t& x_v,
                                     n4m_matrix_view_t& out_v) {
    if (!s.fitted) return N4M_ERR_NOT_FITTED;
    MatrixIn x;
    MatrixOut out;
    n4m_status_t st = require_f64_rowmajor(x_v, x);
    if (st != N4M_OK) return st;
    st = require_f64_rowmajor_mut(out_v, out);
    if (st != N4M_OK) return st;
    st = require_out_shape(x, out);
    if (st != N4M_OK || x.cols != s.features) return N4M_ERR_SHAPE_MISMATCH;
    const std::size_t rows = to_index(x.rows);
    const std::size_t out_cols = to_index(out.cols);
    for (const auto& band : s.bands) {
        const std::int64_t lo = band.first;
        const std::int64_t hi = band.second;
        const std::size_t lo_sz = to_index(lo);
        const std::size_t hi_sz = to_index(hi);
        const double width = static_cast<double>(hi - lo);
        for (std::size_t i = 0; i < rows; ++i) {
            double mean = 0.0;
            for (std::size_t j = lo_sz; j < hi_sz; ++j) mean += at(x, i, j);
            mean /= width;
            double var = 0.0;
            for (std::size_t j = lo_sz; j < hi_sz; ++j) {
                const double c = at(x, i, j) - mean;
                var += c * c;
            }
            const double denom = (hi - lo - s.ddof) > 0 ? static_cast<double>(hi - lo - s.ddof) : width;
            const double scale = std::sqrt(std::max(var / denom, s.eps));
            for (std::size_t j = lo_sz; j < hi_sz; ++j) {
                out.data[row_major(i, out_cols, j)] = (at(x, i, j) - mean) / scale;
            }
        }
    }
    return N4M_OK;
}

struct MSCState {
    bool fitted = false;
    bool localized = false;
    bool has_reference = false;
    std::int32_t window = 32;
    double eps = 1e-12;
    std::int64_t features = 0;
    std::vector<double> initial_reference;
    std::vector<double> reference;
    std::vector<std::pair<std::int64_t, std::int64_t>> bands;
};

n4m_status_t msc_fit(MSCState& s, const n4m_matrix_view_t& x_v) {
    MatrixIn x;
    n4m_status_t st = require_f64_rowmajor(x_v, x);
    if (st != N4M_OK) return st;
    s.features = x.cols;
    if (s.has_reference) {
        if (static_cast<std::int64_t>(s.initial_reference.size()) != x.cols) {
            return N4M_ERR_SHAPE_MISMATCH;
        }
        s.reference = s.initial_reference;
    } else {
        const std::size_t rows = to_index(x.rows);
        const std::size_t cols = to_index(x.cols);
        s.reference.assign(cols, 0.0);
        for (std::size_t i = 0; i < rows; ++i) {
            for (std::size_t j = 0; j < cols; ++j) s.reference[j] += at(x, i, j);
        }
        for (double& v : s.reference) v /= static_cast<double>(x.rows);
    }
    s.bands = intervals(x.cols, s.window, s.window);
    s.fitted = true;
    return N4M_OK;
}

void apply_msc_segment(const MatrixIn& x, MatrixOut& out, const MSCState& s,
                       std::int64_t row_idx, std::int64_t lo, std::int64_t hi,
                       std::int64_t write_lo, std::int64_t write_hi) {
    const double width = static_cast<double>(hi - lo);
    const std::size_t row_sz = to_index(row_idx);
    const std::size_t out_cols = to_index(out.cols);
    const std::size_t lo_sz = to_index(lo);
    const std::size_t hi_sz = to_index(hi);
    const std::size_t write_lo_sz = to_index(write_lo);
    const std::size_t write_hi_sz = to_index(write_hi);
    double ref_mean = 0.0;
    double row_mean = 0.0;
    for (std::size_t j = lo_sz; j < hi_sz; ++j) {
        ref_mean += s.reference[j];
        row_mean += at(x, row_idx, j);
    }
    ref_mean /= width;
    row_mean /= width;
    double denom = 0.0;
    double numer = 0.0;
    for (std::size_t j = lo_sz; j < hi_sz; ++j) {
        const double rc = s.reference[j] - ref_mean;
        denom += rc * rc;
        numer += (at(x, row_idx, j) - row_mean) * rc;
    }
    const double slope = numer / std::max(denom, s.eps);
    const double intercept = row_mean - slope * ref_mean;
    double scale = slope;
    if (!std::isfinite(scale) || std::fabs(scale) <= s.eps) {
        scale = scale < 0.0 ? -s.eps : s.eps;
    }
    for (std::size_t j = write_lo_sz; j < write_hi_sz; ++j) {
        out.data[row_major(row_sz, out_cols, j)] =
            (at(x, row_idx, j) - intercept) / scale;
    }
}

n4m_status_t msc_transform(const MSCState& s, const n4m_matrix_view_t& x_v,
                           n4m_matrix_view_t& out_v) {
    if (!s.fitted) return N4M_ERR_NOT_FITTED;
    MatrixIn x;
    MatrixOut out;
    n4m_status_t st = require_f64_rowmajor(x_v, x);
    if (st != N4M_OK) return st;
    st = require_f64_rowmajor_mut(out_v, out);
    if (st != N4M_OK) return st;
    st = require_out_shape(x, out);
    if (st != N4M_OK || x.cols != s.features) return N4M_ERR_SHAPE_MISMATCH;
    if (s.localized) {
        const std::int64_t half = std::max<std::int64_t>(0, s.window / 2);
        for (std::int64_t row = 0; row < x.rows; ++row) {
            for (std::int64_t j = 0; j < x.cols; ++j) {
                const std::int64_t lo = std::max<std::int64_t>(0, j - half);
                const std::int64_t hi = std::min<std::int64_t>(x.cols, j + half + 1);
                apply_msc_segment(x, out, s, row, lo, hi, j, j + 1);
            }
        }
    } else {
        for (const auto& band : s.bands) {
            for (std::int64_t row = 0; row < x.rows; ++row) {
                apply_msc_segment(x, out, s, row, band.first, band.second, band.first, band.second);
            }
        }
    }
    return N4M_OK;
}

struct AlignState {
    bool fitted = false;
    bool interval_mode = false;
    bool dtw = false;
    bool cow = false;
    std::int32_t interval_size = 32;
    std::int32_t max_shift = 5;
    std::int64_t features = 0;
    std::vector<double> reference;
};

n4m_status_t align_fit(AlignState& s, const n4m_matrix_view_t& x_v) {
    MatrixIn x;
    n4m_status_t st = require_f64_rowmajor(x_v, x);
    if (st != N4M_OK) return st;
    s.features = x.cols;
    if (s.reference.empty()) {
        const std::size_t rows = to_index(x.rows);
        const std::size_t cols = to_index(x.cols);
        s.reference.assign(cols, 0.0);
        for (std::size_t i = 0; i < rows; ++i) {
            for (std::size_t j = 0; j < cols; ++j) s.reference[j] += at(x, i, j);
        }
        for (double& v : s.reference) v /= static_cast<double>(x.rows);
    } else if (static_cast<std::int64_t>(s.reference.size()) != x.cols) {
        return N4M_ERR_SHAPE_MISMATCH;
    }
    s.fitted = true;
    return N4M_OK;
}

void align_segment(const MatrixIn& x, MatrixOut& out, const AlignState& s,
                   std::int64_t row, std::int64_t lo, std::int64_t hi) {
    const std::size_t row_sz = to_index(row);
    const std::size_t x_cols = to_index(x.cols);
    const std::size_t out_cols = to_index(out.cols);
    const std::size_t lo_sz = to_index(lo);
    double ref_mean = 0.0;
    for (std::int64_t j = lo; j < hi; ++j) ref_mean += s.reference[to_index(j)];
    ref_mean /= static_cast<double>(hi - lo);
    const double* row_ptr = x.data + row_major(row_sz, x_cols, lo_sz);
    int best_shift = 0;
    double best_score = -std::numeric_limits<double>::infinity();
    for (int shift = -s.max_shift; shift <= s.max_shift; ++shift) {
        double shifted_mean = 0.0;
        for (std::int64_t j = 0; j < hi - lo; ++j) shifted_mean += shift_sample(row_ptr, hi - lo, j, shift);
        shifted_mean /= static_cast<double>(hi - lo);
        double score = 0.0;
        for (std::int64_t j = 0; j < hi - lo; ++j) {
            score += (shift_sample(row_ptr, hi - lo, j, shift) - shifted_mean)
                   * (s.reference[static_cast<std::size_t>(lo + j)] - ref_mean);
        }
        if (score > best_score) {
            best_score = score;
            best_shift = shift;
        }
    }
    for (std::int64_t j = 0; j < hi - lo; ++j) {
        out.data[row_major(row_sz, out_cols, to_index(lo + j))] =
            shift_sample(row_ptr, hi - lo, j, best_shift);
    }
}

void dtw_row(const MatrixIn& x, MatrixOut& out, const AlignState& s,
             std::int64_t row) {
    const std::int64_t n = x.cols;
    const std::size_t n_sz = to_index(n);
    const std::size_t stride = n_sz + 1U;
    const std::size_t row_sz = to_index(row);
    const std::size_t out_cols = to_index(out.cols);
    std::vector<double> cost(stride * stride, std::numeric_limits<double>::infinity());
    cost.at(0) = 0.0;
    for (std::size_t i = 1; i <= n_sz; ++i) {
        for (std::size_t j = 1; j <= n_sz; ++j) {
            const double d = std::fabs(s.reference[static_cast<std::size_t>(i - 1)] - at(x, row, j - 1));
            const double prev = std::min({
                cost[row_major(i - 1, stride, j)],
                cost[row_major(i, stride, j - 1)],
                cost[row_major(i - 1, stride, j - 1)],
            });
            cost[row_major(i, stride, j)] = d + prev;
        }
    }
    std::vector<double> sums(n_sz, 0.0);
    std::vector<std::int64_t> counts(n_sz, 0);
    std::size_t i = n_sz;
    std::size_t j = n_sz;
    while (i > 0 && j > 0) {
        sums[i - 1] += at(x, row, j - 1);
        counts[i - 1] += 1;
        const double diag = cost[row_major(i - 1, stride, j - 1)];
        const double up = cost[row_major(i - 1, stride, j)];
        const double left = cost[row_major(i, stride, j - 1)];
        if (diag <= up && diag <= left) {
            --i; --j;
        } else if (up <= left) {
            --i;
        } else {
            --j;
        }
    }
    for (std::size_t col = 0; col < n_sz; ++col) {
        out.data[row_major(row_sz, out_cols, col)] =
            counts[col] > 0
                ? sums[col] / static_cast<double>(counts[col])
                : at(x, row, col);
    }
}

std::vector<double> resample_segment(const double* row, std::int64_t start,
                                     std::int64_t end, std::int64_t new_len) {
    std::vector<double> out(static_cast<std::size_t>(new_len), 0.0);
    const std::int64_t old_len = end - start;
    if (old_len <= 0 || new_len <= 0) return out;
    if (new_len == 1) {
        out[0] = row[start];
        return out;
    }
    if (old_len == new_len) {
        for (std::int64_t i = 0; i < new_len; ++i) out[static_cast<std::size_t>(i)] = row[start + i];
        return out;
    }
    for (std::int64_t i = 0; i < new_len; ++i) {
        const double pos = static_cast<double>(i) * static_cast<double>(old_len - 1)
                         / static_cast<double>(new_len - 1);
        std::int64_t lo = static_cast<std::int64_t>(std::floor(pos));
        if (lo > old_len - 2) lo = old_len - 2;
        const double frac = pos - static_cast<double>(lo);
        const double a = row[start + lo];
        const double b = row[start + lo + 1];
        out[static_cast<std::size_t>(i)] = a + frac * (b - a);
    }
    return out;
}

double centered_norm(std::vector<double>& values) {
    if (values.empty()) return 0.0;
    double mean = 0.0;
    for (double value : values) mean += value;
    mean /= static_cast<double>(values.size());
    double norm_sq = 0.0;
    for (double& value : values) {
        value -= mean;
        norm_sq += value * value;
    }
    return std::sqrt(norm_sq);
}

double centered_correlation(const std::vector<double>& ref_centered,
                            double ref_norm,
                            std::vector<double> sample) {
    const double sample_norm = centered_norm(sample);
    if (ref_norm == 0.0 && sample_norm == 0.0) return 1.0;
    if (ref_norm == 0.0 || sample_norm == 0.0) return 0.0;
    double dot = 0.0;
    for (std::size_t i = 0; i < ref_centered.size(); ++i) dot += ref_centered[i] * sample[i];
    return dot / (ref_norm * sample_norm);
}

void cow_row(const MatrixIn& x, MatrixOut& out, const AlignState& s,
             std::int64_t row) {
    const std::int64_t n = x.cols;
    const std::int64_t min_len = 3;
    std::int64_t segment_len = std::max<std::int64_t>(min_len, s.interval_size);
    if (segment_len > n / 2 || n < 2 * min_len + 1) {
        align_segment(x, out, s, row, 0, n);
        return;
    }
    std::int64_t num_intervals = n / segment_len;
    if (num_intervals < 2) {
        align_segment(x, out, s, row, 0, n);
        return;
    }
    const std::int64_t max_slack = std::max<std::int64_t>(1, segment_len - min_len);
    const std::int64_t slack = std::max<std::int64_t>(1, std::min<std::int64_t>(s.max_shift, max_slack));
    std::vector<std::int64_t> boundaries(static_cast<std::size_t>(num_intervals + 1), 0);
    for (std::int64_t i = 0; i < num_intervals; ++i) boundaries[static_cast<std::size_t>(i)] = i * segment_len;
    boundaries[static_cast<std::size_t>(num_intervals)] = n;

    std::vector<std::vector<double>> ref_segments(static_cast<std::size_t>(num_intervals));
    std::vector<double> ref_norms(static_cast<std::size_t>(num_intervals), 0.0);
    for (std::int64_t k = 0; k < num_intervals; ++k) {
        const std::int64_t lo = boundaries[static_cast<std::size_t>(k)];
        const std::int64_t hi = boundaries[static_cast<std::size_t>(k + 1)];
        auto& seg = ref_segments[static_cast<std::size_t>(k)];
        seg.assign(s.reference.begin() + lo, s.reference.begin() + hi);
        ref_norms[static_cast<std::size_t>(k)] = centered_norm(seg);
    }

    const std::int64_t states = slack * num_intervals + 1;
    auto idx = [states](std::int64_t r, std::int64_t c) -> std::size_t {
        return static_cast<std::size_t>(r * states + c);
    };
    std::vector<double> best(static_cast<std::size_t>(num_intervals * states), -1.0);
    std::vector<std::int64_t> possible(static_cast<std::size_t>(num_intervals * states), -1);
    std::vector<std::int64_t> best_end(static_cast<std::size_t>((num_intervals - 1) * states), -1);
    const std::size_t row_sz = to_index(row);
    const std::size_t x_cols = to_index(x.cols);
    const std::size_t out_cols = to_index(out.cols);
    const double* row_ptr = x.data + row_major(row_sz, x_cols, 0);

    std::int64_t start_idx = 0;
    std::int64_t interval_start = boundaries[static_cast<std::size_t>(num_intervals - 1)] - slack;
    std::int64_t interval_end = boundaries[static_cast<std::size_t>(num_intervals - 1)] + slack + 1;
    const std::int64_t last_len = n - boundaries[static_cast<std::size_t>(num_intervals - 1)];
    for (std::int64_t new_start = interval_start; new_start < interval_end && start_idx < states; ++new_start) {
        if (new_start < 0 || n - new_start < min_len) continue;
        std::vector<double> sample = resample_segment(row_ptr, new_start, n, last_len);
        best[idx(0, start_idx)] = centered_correlation(
            ref_segments[static_cast<std::size_t>(num_intervals - 1)],
            ref_norms[static_cast<std::size_t>(num_intervals - 1)],
            std::move(sample));
        possible[idx(0, start_idx)] = new_start;
        ++start_idx;
    }

    std::int64_t end_first = interval_start;
    std::int64_t end_last = interval_end;
    for (std::int64_t interval_idx = 1; interval_idx < num_intervals; ++interval_idx) {
        const std::int64_t ref_idx = num_intervals - interval_idx - 1;
        const auto& ref = ref_segments[static_cast<std::size_t>(ref_idx)];
        const double ref_norm = ref_norms[static_cast<std::size_t>(ref_idx)];
        const std::int64_t slack_mult = std::min(interval_idx + 1, num_intervals - interval_idx - 1);
        const std::int64_t max_border_shift = slack * slack_mult;
        start_idx = 0;
        interval_start = boundaries[static_cast<std::size_t>(ref_idx)] - max_border_shift;
        interval_end = boundaries[static_cast<std::size_t>(ref_idx)] + max_border_shift + 1;
        std::int64_t first_valid = interval_start;
        std::int64_t last_valid = interval_end;
        const std::int64_t target_len = boundaries[static_cast<std::size_t>(ref_idx + 1)]
                                      - boundaries[static_cast<std::size_t>(ref_idx)];
        for (std::int64_t new_start = interval_start; new_start < interval_end && start_idx < states; ++new_start) {
            bool any = false;
            double best_val = -std::numeric_limits<double>::infinity();
            std::int64_t chosen_end = -1;
            for (std::int64_t len = std::max(min_len, segment_len - slack);
                 len <= segment_len + slack; ++len) {
                const std::int64_t new_end = new_start + len;
                if (new_start < 0 || new_end > n || new_end < end_first || new_end >= end_last) continue;
                const std::int64_t prev_col = new_end - end_first;
                if (prev_col < 0 || prev_col >= states) continue;
                std::vector<double> sample = resample_segment(row_ptr, new_start, new_end, target_len);
                const double corr = centered_correlation(ref, ref_norm, std::move(sample));
                const double cumulative = corr + best[idx(interval_idx - 1, prev_col)];
                if (cumulative > best_val) {
                    best_val = cumulative;
                    chosen_end = new_end;
                }
                any = true;
            }
            if (!any) {
                if (new_start == interval_start) ++first_valid;
                continue;
            }
            last_valid = new_start;
            possible[idx(interval_idx, start_idx)] = new_start;
            best_end[idx(interval_idx - 1, start_idx)] = chosen_end;
            best[idx(interval_idx, start_idx)] = best_val;
            ++start_idx;
        }
        end_first = first_valid;
        end_last = last_valid + 1;
    }

    std::vector<std::int64_t> path(static_cast<std::size_t>(num_intervals + 1), 0);
    path.front() = 0;
    path.back() = n;
    std::int64_t best_col = 0;
    double best_value = -std::numeric_limits<double>::infinity();
    for (std::int64_t c = 0; c < states; ++c) {
        if (best[idx(num_intervals - 1, c)] > best_value) {
            best_value = best[idx(num_intervals - 1, c)];
            best_col = c;
        }
    }
    const std::int64_t back_base = num_intervals - 1;
    for (std::int64_t i = 1; i < num_intervals; ++i) {
        const std::int64_t r = back_base - i;
        const std::int64_t border = best_end[idx(r, best_col)];
        path[static_cast<std::size_t>(i)] = border;
        for (std::int64_t c = 0; c < states; ++c) {
            if (possible[idx(r, c)] == border) {
                best_col = c;
                break;
            }
        }
    }

    for (std::int64_t k = 0; k < num_intervals; ++k) {
        const std::int64_t src_lo = path[static_cast<std::size_t>(k)];
        const std::int64_t src_hi = path[static_cast<std::size_t>(k + 1)];
        const std::int64_t dst_lo = boundaries[static_cast<std::size_t>(k)];
        const std::int64_t dst_hi = boundaries[static_cast<std::size_t>(k + 1)];
        const std::int64_t dst_len = dst_hi - dst_lo;
        std::vector<double> segment = resample_segment(row_ptr, src_lo, src_hi, dst_len);
        for (std::int64_t j = 0; j < dst_len; ++j) {
            out.data[row_major(row_sz, out_cols, to_index(dst_lo + j))] =
                segment[static_cast<std::size_t>(j)];
        }
    }
}

n4m_status_t align_transform(const AlignState& s, const n4m_matrix_view_t& x_v,
                             n4m_matrix_view_t& out_v) {
    if (!s.fitted) return N4M_ERR_NOT_FITTED;
    MatrixIn x;
    MatrixOut out;
    n4m_status_t st = require_f64_rowmajor(x_v, x);
    if (st != N4M_OK) return st;
    st = require_f64_rowmajor_mut(out_v, out);
    if (st != N4M_OK) return st;
    st = require_out_shape(x, out);
    if (st != N4M_OK || x.cols != s.features) return N4M_ERR_SHAPE_MISMATCH;
    if (s.dtw) {
        for (std::int64_t row = 0; row < x.rows; ++row) dtw_row(x, out, s, row);
    } else if (s.cow) {
        for (std::int64_t row = 0; row < x.rows; ++row) cow_row(x, out, s, row);
    } else if (s.interval_mode) {
        for (const auto& band : intervals(x.cols, s.interval_size, s.interval_size)) {
            for (std::int64_t row = 0; row < x.rows; ++row) {
                align_segment(x, out, s, row, band.first, band.second);
            }
        }
    } else {
        for (std::int64_t row = 0; row < x.rows; ++row) {
            align_segment(x, out, s, row, 0, x.cols);
        }
    }
    return N4M_OK;
}

struct SelectorState {
    bool fitted = false;
    bool correlation = false;
    double threshold = 0.0;
    std::int32_t top_k = -1;
    std::int64_t features = 0;
    std::vector<std::int64_t> indices;
};

n4m_status_t variance_fit(SelectorState& s, const n4m_matrix_view_t& x_v,
                          const double* y, std::int64_t y_len) {
    MatrixIn x;
    n4m_status_t st = require_f64_rowmajor(x_v, x);
    if (st != N4M_OK) return st;
    if (s.correlation && (y == nullptr || y_len != x.rows)) return N4M_ERR_SHAPE_MISMATCH;
    const std::size_t rows = to_index(x.rows);
    const std::size_t cols = to_index(x.cols);
    std::vector<double> scores(cols, 0.0);
    if (s.correlation) {
        double y_mean = 0.0;
        for (std::size_t i = 0; i < rows; ++i) y_mean += y[i];
        y_mean /= static_cast<double>(x.rows);
        double y_norm = 0.0;
        for (std::size_t i = 0; i < rows; ++i) y_norm += (y[i] - y_mean) * (y[i] - y_mean);
        y_norm = std::sqrt(y_norm);
        for (std::size_t j = 0; j < cols; ++j) {
            double mean = 0.0;
            for (std::size_t i = 0; i < rows; ++i) mean += at(x, i, j);
            mean /= static_cast<double>(x.rows);
            double dot = 0.0;
            double norm = 0.0;
            for (std::size_t i = 0; i < rows; ++i) {
                const double xc = at(x, i, j) - mean;
                dot += xc * (y[i] - y_mean);
                norm += xc * xc;
            }
            const double denom = std::sqrt(norm) * y_norm;
            scores[j] = denom > 0.0 ? std::fabs(dot / denom) : 0.0;
        }
    } else {
        for (std::size_t j = 0; j < cols; ++j) {
            double mean = 0.0;
            for (std::size_t i = 0; i < rows; ++i) mean += at(x, i, j);
            mean /= static_cast<double>(x.rows);
            double var = 0.0;
            for (std::size_t i = 0; i < rows; ++i) {
                const double d = at(x, i, j) - mean;
                var += d * d;
            }
            scores[j] = var / static_cast<double>(x.rows);
        }
    }
    std::vector<std::int64_t> idx;
    if (s.top_k > 0) {
        idx.resize(cols);
        for (std::size_t j = 0; j < cols; ++j) idx[j] = static_cast<std::int64_t>(j);
        std::stable_sort(idx.begin(), idx.end(), [&](std::int64_t a, std::int64_t b) {
            return scores[to_index(a)] < scores[to_index(b)];
        });
        const std::int64_t keep = std::min<std::int64_t>(s.top_k, x.cols);
        idx.erase(idx.begin(), idx.end() - keep);
    } else {
        for (std::size_t j = 0; j < cols; ++j) {
            if (scores[j] > s.threshold) idx.push_back(static_cast<std::int64_t>(j));
        }
    }
    s.indices = idx;
    s.features = x.cols;
    s.fitted = true;
    return N4M_OK;
}

n4m_status_t selector_transform(const SelectorState& s,
                                const n4m_matrix_view_t& x_v,
                                n4m_matrix_view_t& out_v) {
    if (!s.fitted) return N4M_ERR_NOT_FITTED;
    MatrixIn x;
    MatrixOut out;
    n4m_status_t st = require_f64_rowmajor(x_v, x);
    if (st != N4M_OK) return st;
    st = require_f64_rowmajor_mut(out_v, out);
    if (st != N4M_OK) return st;
    if (x.cols != s.features || out.rows != x.rows ||
        out.cols != static_cast<std::int64_t>(s.indices.size())) {
        return N4M_ERR_SHAPE_MISMATCH;
    }
    const std::size_t rows = to_index(x.rows);
    const std::size_t out_cols = to_index(out.cols);
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t k = 0; k < out_cols; ++k) {
            out.data[row_major(i, out_cols, k)] = at(x, i, s.indices[k]);
        }
    }
    return N4M_OK;
}

struct IntervalState {
    bool fitted = false;
    std::int32_t width = 32;
    std::int32_t step = 0;
    std::int64_t features = 0;
    std::vector<std::pair<std::int64_t, std::int64_t>> bands;
};

n4m_status_t interval_fit(IntervalState& s, const n4m_matrix_view_t& x_v) {
    MatrixIn x;
    n4m_status_t st = require_f64_rowmajor(x_v, x);
    if (st != N4M_OK) return st;
    s.features = x.cols;
    const std::int64_t actual_step = s.step > 0 ? s.step : s.width;
    s.bands = intervals(x.cols, s.width, actual_step);
    s.fitted = true;
    return N4M_OK;
}

std::int64_t interval_output_cols(const IntervalState& s) {
    std::int64_t total = 0;
    for (const auto& band : s.bands) total += band.second - band.first;
    return total;
}

n4m_status_t interval_transform(const IntervalState& s,
                                const n4m_matrix_view_t& x_v,
                                n4m_matrix_view_t& out_v) {
    if (!s.fitted) return N4M_ERR_NOT_FITTED;
    MatrixIn x;
    MatrixOut out;
    n4m_status_t st = require_f64_rowmajor(x_v, x);
    if (st != N4M_OK) return st;
    st = require_f64_rowmajor_mut(out_v, out);
    if (st != N4M_OK) return st;
    if (x.cols != s.features || out.rows != x.rows || out.cols != interval_output_cols(s)) {
        return N4M_ERR_SHAPE_MISMATCH;
    }
    const std::size_t rows = to_index(x.rows);
    const std::size_t out_cols = to_index(out.cols);
    std::size_t dst_col = 0;
    for (const auto& band : s.bands) {
        for (std::int64_t j = band.first; j < band.second; ++j, ++dst_col) {
            for (std::size_t i = 0; i < rows; ++i) {
                out.data[row_major(i, out_cols, dst_col)] = at(x, i, j);
            }
        }
    }
    return N4M_OK;
}

template <typename Handle, typename... Args>
n4m_status_t make_handle(Handle** out, Args&&... args) noexcept {
    if (out == nullptr) return N4M_ERR_NULL_POINTER;
    *out = nullptr;
    try {
        *out = new Handle{std::forward<Args>(args)...};
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

template <typename Fn>
n4m_status_t call_status(Fn&& fn) noexcept {
    try {
        return fn();
    } catch (n4m_status_t s) {
        return s;
    } catch (const std::bad_alloc&) {
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

}  // namespace

struct n4m_pp_direct_standardization_handle_t { DSState s; };
struct n4m_pp_robust_direct_standardization_handle_t {
    DSState s;
    double trim_quantile;
    std::int32_t max_iter;
};
struct n4m_pp_piecewise_direct_standardization_handle_t { PDSState s; };
struct n4m_pp_saps_handle_t { SAPSState s; };
struct n4m_pp_slope_bias_handle_t { SlopeBiasState s; };
struct n4m_pp_local_centering_handle_t {
    bool fitted = false;
    std::int64_t features = 0;
    std::vector<double> delta;
};
struct n4m_pp_weighted_snv_handle_t { VectorWeightsState s; };
struct n4m_pp_vsn_handle_t { VectorWeightsState s; };
struct n4m_pp_piecewise_snv_handle_t { PiecewiseSNVState s; };
struct n4m_pp_piecewise_msc_handle_t { MSCState s; };
struct n4m_pp_localized_msc_handle_t { MSCState s; };
struct n4m_pp_xcorr_align_handle_t { AlignState s; };
struct n4m_pp_icoshift_align_handle_t { AlignState s; };
struct n4m_pp_dtw_align_handle_t { AlignState s; };
struct n4m_pp_cow_align_handle_t { AlignState s; };
struct n4m_filter_variance_handle_t { SelectorState s; };
struct n4m_filter_correlation_handle_t { SelectorState s; };
struct n4m_interval_generator_handle_t { IntervalState s; };

extern "C" n4m_status_t n4m_pp_direct_standardization_create(
    n4m_pp_direct_standardization_handle_t** out, int fit_intercept, double ridge) {
    n4m_pp_direct_standardization_handle_t init;
    init.s.fit_intercept = fit_intercept != 0;
    init.s.ridge = ridge;
    return make_handle(out, init);
}

extern "C" void n4m_pp_direct_standardization_destroy(
    n4m_pp_direct_standardization_handle_t* h) { delete h; }

extern "C" n4m_status_t n4m_pp_direct_standardization_fit(
    n4m_pp_direct_standardization_handle_t* h, n4m_matrix_view_t source,
    n4m_matrix_view_t target) {
    return call_status([&] { return h ? ds_fit(h->s, source, target) : N4M_ERR_NULL_POINTER; });
}

extern "C" n4m_status_t n4m_pp_direct_standardization_transform(
    const n4m_pp_direct_standardization_handle_t* h, n4m_matrix_view_t x,
    n4m_matrix_view_t out) {
    return call_status([&] { return h ? ds_transform(h->s, x, out) : N4M_ERR_NULL_POINTER; });
}

extern "C" n4m_status_t n4m_pp_direct_standardization_is_fitted(
    const n4m_pp_direct_standardization_handle_t* h, int* out) {
    if (!out) return N4M_ERR_NULL_POINTER;
    *out = (h && h->s.fitted) ? 1 : 0;
    return N4M_OK;
}

extern "C" n4m_status_t n4m_pp_robust_direct_standardization_create(
    n4m_pp_robust_direct_standardization_handle_t** out, int fit_intercept,
    double ridge, double trim_quantile, int32_t max_iter) {
    n4m_pp_robust_direct_standardization_handle_t init;
    init.s.fit_intercept = fit_intercept != 0;
    init.s.ridge = ridge;
    init.trim_quantile = trim_quantile;
    init.max_iter = max_iter;
    return make_handle(out, init);
}

extern "C" void n4m_pp_robust_direct_standardization_destroy(
    n4m_pp_robust_direct_standardization_handle_t* h) { delete h; }

extern "C" n4m_status_t n4m_pp_robust_direct_standardization_fit(
    n4m_pp_robust_direct_standardization_handle_t* h, n4m_matrix_view_t source_v,
    n4m_matrix_view_t target_v) {
    return call_status([&] {
        if (!h) return N4M_ERR_NULL_POINTER;
        MatrixIn source, target;
        n4m_status_t st = require_f64_rowmajor(source_v, source);
        if (st != N4M_OK) return st;
        st = require_f64_rowmajor(target_v, target);
        if (st != N4M_OK) return st;
        st = require_same_shape(source, target);
        if (st != N4M_OK) return st;
        const std::size_t source_rows = to_index(source.rows);
        const std::size_t source_cols = to_index(source.cols);
        std::vector<std::int64_t> keep(source_rows);
        for (std::size_t i = 0; i < source_rows; ++i) keep[i] = static_cast<std::int64_t>(i);
        std::vector<double> coef;
        for (std::int32_t iter = 0; iter < std::max<std::int32_t>(1, h->max_iter); ++iter) {
            coef = fit_ols(source, target, h->s.fit_intercept, h->s.ridge, &keep);
            DSState tmp = h->s;
            tmp.fitted = true;
            tmp.features = source.cols;
            tmp.coef = coef;
            std::vector<double> mapped(source_rows * source_cols);
            n4m_matrix_view_t out_v{mapped.data(), source.rows, source.cols, source.cols, 1, N4M_DTYPE_F64, 0};
            st = ds_transform(tmp, source_v, out_v);
            if (st != N4M_OK) return st;
            std::vector<double> residual(source_rows, 0.0);
            for (std::size_t i = 0; i < source_rows; ++i) {
                double sum = 0.0;
                for (std::size_t j = 0; j < source_cols; ++j) {
                    const double d = mapped[row_major(i, source_cols, j)] - at(target, i, j);
                    sum += d * d;
                }
                residual[i] = std::sqrt(sum);
            }
            const double cutoff = quantile_sorted(residual, h->trim_quantile);
            std::vector<std::int64_t> next;
            for (std::size_t i = 0; i < source_rows; ++i) {
                if (residual[i] <= cutoff) next.push_back(static_cast<std::int64_t>(i));
            }
            if (static_cast<std::int64_t>(next.size()) <= source.cols) {
                next.resize(source_rows);
                for (std::size_t i = 0; i < source_rows; ++i) next[i] = static_cast<std::int64_t>(i);
            }
            keep.swap(next);
        }
        h->s.coef = coef.empty() ? fit_ols(source, target, h->s.fit_intercept, h->s.ridge, &keep) : coef;
        h->s.features = source.cols;
        h->s.fitted = true;
        return N4M_OK;
    });
}

extern "C" n4m_status_t n4m_pp_robust_direct_standardization_transform(
    const n4m_pp_robust_direct_standardization_handle_t* h, n4m_matrix_view_t x,
    n4m_matrix_view_t out) {
    return call_status([&] { return h ? ds_transform(h->s, x, out) : N4M_ERR_NULL_POINTER; });
}

extern "C" n4m_status_t n4m_pp_robust_direct_standardization_is_fitted(
    const n4m_pp_robust_direct_standardization_handle_t* h, int* out) {
    if (!out) return N4M_ERR_NULL_POINTER;
    *out = (h && h->s.fitted) ? 1 : 0;
    return N4M_OK;
}

extern "C" n4m_status_t n4m_pp_piecewise_direct_standardization_create(
    n4m_pp_piecewise_direct_standardization_handle_t** out, int32_t window_size,
    int fit_intercept, double ridge) {
    n4m_pp_piecewise_direct_standardization_handle_t init;
    init.s.window = window_size;
    init.s.fit_intercept = fit_intercept != 0;
    init.s.ridge = ridge;
    return make_handle(out, init);
}

extern "C" void n4m_pp_piecewise_direct_standardization_destroy(
    n4m_pp_piecewise_direct_standardization_handle_t* h) { delete h; }

extern "C" n4m_status_t n4m_pp_piecewise_direct_standardization_fit(
    n4m_pp_piecewise_direct_standardization_handle_t* h, n4m_matrix_view_t source,
    n4m_matrix_view_t target) {
    return call_status([&] { return h ? pds_fit(h->s, source, target) : N4M_ERR_NULL_POINTER; });
}

extern "C" n4m_status_t n4m_pp_piecewise_direct_standardization_transform(
    const n4m_pp_piecewise_direct_standardization_handle_t* h, n4m_matrix_view_t x,
    n4m_matrix_view_t out) {
    return call_status([&] { return h ? pds_transform(h->s, x, out) : N4M_ERR_NULL_POINTER; });
}

extern "C" n4m_status_t n4m_pp_piecewise_direct_standardization_is_fitted(
    const n4m_pp_piecewise_direct_standardization_handle_t* h, int* out) {
    if (!out) return N4M_ERR_NULL_POINTER;
    *out = (h && h->s.fitted) ? 1 : 0;
    return N4M_OK;
}

extern "C" n4m_status_t n4m_pp_saps_create(
    n4m_pp_saps_handle_t** out, int32_t n_components, double score_weight,
    int fit_intercept, double ridge) {
    n4m_pp_saps_handle_t init;
    init.s.n_components = n_components;
    init.s.score_weight = score_weight;
    init.s.fit_intercept = fit_intercept != 0;
    init.s.ridge = ridge;
    return make_handle(out, init);
}

extern "C" void n4m_pp_saps_destroy(n4m_pp_saps_handle_t* h) { delete h; }
extern "C" n4m_status_t n4m_pp_saps_fit(n4m_pp_saps_handle_t* h,
    n4m_matrix_view_t source, n4m_matrix_view_t target) {
    return call_status([&] { return h ? saps_fit(h->s, source, target) : N4M_ERR_NULL_POINTER; });
}
extern "C" n4m_status_t n4m_pp_saps_transform(const n4m_pp_saps_handle_t* h,
    n4m_matrix_view_t x, n4m_matrix_view_t out) {
    return call_status([&] { return h ? saps_transform(h->s, x, out) : N4M_ERR_NULL_POINTER; });
}
extern "C" n4m_status_t n4m_pp_saps_is_fitted(const n4m_pp_saps_handle_t* h,
    int* out) {
    if (!out) return N4M_ERR_NULL_POINTER;
    *out = (h && h->s.fitted) ? 1 : 0;
    return N4M_OK;
}

extern "C" n4m_status_t n4m_pp_slope_bias_create(n4m_pp_slope_bias_handle_t** out) {
    return make_handle(out, n4m_pp_slope_bias_handle_t{});
}
extern "C" void n4m_pp_slope_bias_destroy(n4m_pp_slope_bias_handle_t* h) { delete h; }
extern "C" n4m_status_t n4m_pp_slope_bias_fit(n4m_pp_slope_bias_handle_t* h,
    const double* source, const double* target, int64_t n) {
    return call_status([&] { return h ? slope_fit(h->s, source, target, n) : N4M_ERR_NULL_POINTER; });
}
extern "C" n4m_status_t n4m_pp_slope_bias_transform(const n4m_pp_slope_bias_handle_t* h,
    const double* y, int64_t n, double* out) {
    if (!h || !y || !out) return N4M_ERR_NULL_POINTER;
    if (!h->s.fitted) return N4M_ERR_NOT_FITTED;
    for (int64_t i = 0; i < n; ++i) out[i] = h->s.slope * y[i] + h->s.bias;
    return N4M_OK;
}
extern "C" n4m_status_t n4m_pp_slope_bias_is_fitted(const n4m_pp_slope_bias_handle_t* h,
    int* out) {
    if (!out) return N4M_ERR_NULL_POINTER;
    *out = (h && h->s.fitted) ? 1 : 0;
    return N4M_OK;
}

extern "C" n4m_status_t n4m_pp_local_centering_create(n4m_pp_local_centering_handle_t** out) {
    return make_handle(out, n4m_pp_local_centering_handle_t{});
}
extern "C" void n4m_pp_local_centering_destroy(n4m_pp_local_centering_handle_t* h) { delete h; }
extern "C" n4m_status_t n4m_pp_local_centering_fit(n4m_pp_local_centering_handle_t* h,
    n4m_matrix_view_t source_v, n4m_matrix_view_t target_v) {
    return call_status([&] {
        if (!h) return N4M_ERR_NULL_POINTER;
        MatrixIn source, target;
        n4m_status_t st = require_f64_rowmajor(source_v, source);
        if (st != N4M_OK) return st;
        st = require_f64_rowmajor(target_v, target);
        if (st != N4M_OK) return st;
        st = require_same_shape(source, target);
        if (st != N4M_OK) return st;
        h->features = source.cols;
        const std::size_t source_rows = to_index(source.rows);
        const std::size_t source_cols = to_index(source.cols);
        h->delta.assign(source_cols, 0.0);
        for (std::size_t i = 0; i < source_rows; ++i) {
            for (std::size_t j = 0; j < source_cols; ++j) {
                h->delta[j] += at(target, i, j) - at(source, i, j);
            }
        }
        for (double& v : h->delta) v /= static_cast<double>(source.rows);
        h->fitted = true;
        return N4M_OK;
    });
}
extern "C" n4m_status_t n4m_pp_local_centering_transform(
    const n4m_pp_local_centering_handle_t* h, n4m_matrix_view_t x_v, n4m_matrix_view_t out_v) {
    return call_status([&] {
        if (!h) return N4M_ERR_NULL_POINTER;
        if (!h->fitted) return N4M_ERR_NOT_FITTED;
        MatrixIn x;
        MatrixOut out;
        n4m_status_t st = require_f64_rowmajor(x_v, x);
        if (st != N4M_OK) return st;
        st = require_f64_rowmajor_mut(out_v, out);
        if (st != N4M_OK) return st;
        st = require_out_shape(x, out);
        if (st != N4M_OK || x.cols != h->features) return N4M_ERR_SHAPE_MISMATCH;
        const std::size_t rows = to_index(x.rows);
        const std::size_t cols = to_index(x.cols);
        const std::size_t out_cols = to_index(out.cols);
        for (std::size_t i = 0; i < rows; ++i) {
            for (std::size_t j = 0; j < cols; ++j) {
                out.data[row_major(i, out_cols, j)] = at(x, i, j) + h->delta[j];
            }
        }
        return N4M_OK;
    });
}
extern "C" n4m_status_t n4m_pp_local_centering_is_fitted(
    const n4m_pp_local_centering_handle_t* h, int* out) {
    if (!out) return N4M_ERR_NULL_POINTER;
    *out = (h && h->fitted) ? 1 : 0;
    return N4M_OK;
}

extern "C" n4m_status_t n4m_pp_weighted_snv_create(
    n4m_pp_weighted_snv_handle_t** out, const double* weights, int64_t n_weights,
    int32_t ddof, double eps) {
    n4m_pp_weighted_snv_handle_t init;
    init.s.ddof = ddof;
    init.s.eps = eps;
    if (weights != nullptr && n_weights > 0) {
        init.s.has_initial = true;
        init.s.initial.assign(weights, weights + n_weights);
    }
    return make_handle(out, init);
}
extern "C" void n4m_pp_weighted_snv_destroy(n4m_pp_weighted_snv_handle_t* h) { delete h; }
extern "C" n4m_status_t n4m_pp_weighted_snv_fit(n4m_pp_weighted_snv_handle_t* h,
    n4m_matrix_view_t x) {
    return call_status([&] { return h ? weighted_snv_fit(h->s, x) : N4M_ERR_NULL_POINTER; });
}
extern "C" n4m_status_t n4m_pp_weighted_snv_transform(const n4m_pp_weighted_snv_handle_t* h,
    n4m_matrix_view_t x, n4m_matrix_view_t out) {
    return call_status([&] { return h ? weighted_snv_transform(h->s, x, out) : N4M_ERR_NULL_POINTER; });
}
extern "C" n4m_status_t n4m_pp_weighted_snv_is_fitted(const n4m_pp_weighted_snv_handle_t* h,
    int* out) {
    if (!out) return N4M_ERR_NULL_POINTER;
    *out = (h && h->s.fitted) ? 1 : 0;
    return N4M_OK;
}

extern "C" n4m_status_t n4m_pp_vsn_create(n4m_pp_vsn_handle_t** out, double eps) {
    n4m_pp_vsn_handle_t init;
    init.s.eps = eps;
    return make_handle(out, init);
}
extern "C" void n4m_pp_vsn_destroy(n4m_pp_vsn_handle_t* h) { delete h; }
extern "C" n4m_status_t n4m_pp_vsn_fit(n4m_pp_vsn_handle_t* h, n4m_matrix_view_t x) {
    return call_status([&] { return h ? vsn_fit(h->s, x) : N4M_ERR_NULL_POINTER; });
}
extern "C" n4m_status_t n4m_pp_vsn_transform(const n4m_pp_vsn_handle_t* h,
    n4m_matrix_view_t x, n4m_matrix_view_t out) {
    return call_status([&] { return h ? weighted_snv_transform(h->s, x, out) : N4M_ERR_NULL_POINTER; });
}
extern "C" n4m_status_t n4m_pp_vsn_is_fitted(const n4m_pp_vsn_handle_t* h, int* out) {
    if (!out) return N4M_ERR_NULL_POINTER;
    *out = (h && h->s.fitted) ? 1 : 0;
    return N4M_OK;
}

extern "C" n4m_status_t n4m_pp_piecewise_snv_create(
    n4m_pp_piecewise_snv_handle_t** out, int32_t window, int32_t ddof, double eps) {
    n4m_pp_piecewise_snv_handle_t init;
    init.s.window = window;
    init.s.ddof = ddof;
    init.s.eps = eps;
    return make_handle(out, init);
}
extern "C" void n4m_pp_piecewise_snv_destroy(n4m_pp_piecewise_snv_handle_t* h) { delete h; }
extern "C" n4m_status_t n4m_pp_piecewise_snv_fit(n4m_pp_piecewise_snv_handle_t* h,
    n4m_matrix_view_t x) {
    return call_status([&] { return h ? piecewise_snv_fit(h->s, x) : N4M_ERR_NULL_POINTER; });
}
extern "C" n4m_status_t n4m_pp_piecewise_snv_transform(const n4m_pp_piecewise_snv_handle_t* h,
    n4m_matrix_view_t x, n4m_matrix_view_t out) {
    return call_status([&] { return h ? piecewise_snv_transform(h->s, x, out) : N4M_ERR_NULL_POINTER; });
}
extern "C" n4m_status_t n4m_pp_piecewise_snv_is_fitted(const n4m_pp_piecewise_snv_handle_t* h,
    int* out) {
    if (!out) return N4M_ERR_NULL_POINTER;
    *out = (h && h->s.fitted) ? 1 : 0;
    return N4M_OK;
}

#define DEFINE_MSC_API(prefix, HandleType, localized_flag) \
extern "C" n4m_status_t prefix##_create(HandleType** out, const double* reference, int64_t n_reference, int32_t window, double eps) { \
    HandleType init; init.s.window = window; init.s.eps = eps; init.s.localized = localized_flag; \
    if (reference != nullptr && n_reference > 0) { init.s.has_reference = true; init.s.initial_reference.assign(reference, reference + n_reference); } \
    return make_handle(out, init); \
} \
extern "C" void prefix##_destroy(HandleType* h) { delete h; } \
extern "C" n4m_status_t prefix##_fit(HandleType* h, n4m_matrix_view_t x) { \
    return call_status([&] { return h ? msc_fit(h->s, x) : N4M_ERR_NULL_POINTER; }); \
} \
extern "C" n4m_status_t prefix##_transform(const HandleType* h, n4m_matrix_view_t x, n4m_matrix_view_t out) { \
    return call_status([&] { return h ? msc_transform(h->s, x, out) : N4M_ERR_NULL_POINTER; }); \
} \
extern "C" n4m_status_t prefix##_is_fitted(const HandleType* h, int* out) { \
    if (!out) return N4M_ERR_NULL_POINTER; \
    *out = (h && h->s.fitted) ? 1 : 0; \
    return N4M_OK; \
}

DEFINE_MSC_API(n4m_pp_piecewise_msc, n4m_pp_piecewise_msc_handle_t, false)
DEFINE_MSC_API(n4m_pp_localized_msc, n4m_pp_localized_msc_handle_t, true)

#define DEFINE_ALIGN_API(prefix, HandleType, interval_flag, dtw_flag, cow_flag) \
extern "C" n4m_status_t prefix##_create(HandleType** out, const double* reference, int64_t n_reference, int32_t interval_size, int32_t max_shift) { \
    HandleType init; init.s.interval_mode = interval_flag; init.s.dtw = dtw_flag; init.s.cow = cow_flag; init.s.interval_size = interval_size; init.s.max_shift = max_shift; \
    if (reference != nullptr && n_reference > 0) init.s.reference.assign(reference, reference + n_reference); \
    return make_handle(out, init); \
} \
extern "C" void prefix##_destroy(HandleType* h) { delete h; } \
extern "C" n4m_status_t prefix##_fit(HandleType* h, n4m_matrix_view_t x) { \
    return call_status([&] { return h ? align_fit(h->s, x) : N4M_ERR_NULL_POINTER; }); \
} \
extern "C" n4m_status_t prefix##_transform(const HandleType* h, n4m_matrix_view_t x, n4m_matrix_view_t out) { \
    return call_status([&] { return h ? align_transform(h->s, x, out) : N4M_ERR_NULL_POINTER; }); \
} \
extern "C" n4m_status_t prefix##_is_fitted(const HandleType* h, int* out) { \
    if (!out) return N4M_ERR_NULL_POINTER; \
    *out = (h && h->s.fitted) ? 1 : 0; \
    return N4M_OK; \
}

DEFINE_ALIGN_API(n4m_pp_xcorr_align, n4m_pp_xcorr_align_handle_t, false, false, false)
DEFINE_ALIGN_API(n4m_pp_icoshift_align, n4m_pp_icoshift_align_handle_t, true, false, false)
DEFINE_ALIGN_API(n4m_pp_cow_align, n4m_pp_cow_align_handle_t, false, false, true)
DEFINE_ALIGN_API(n4m_pp_dtw_align, n4m_pp_dtw_align_handle_t, false, true, false)

#define DEFINE_SELECTOR_CREATE(prefix, HandleType, corr_flag) \
extern "C" n4m_status_t prefix##_create(HandleType** out, double threshold, int32_t top_k) { \
    HandleType init; init.s.correlation = corr_flag; init.s.threshold = threshold; init.s.top_k = top_k; return make_handle(out, init); \
} \
extern "C" void prefix##_destroy(HandleType* h) { delete h; } \
extern "C" n4m_status_t prefix##_transform(const HandleType* h, n4m_matrix_view_t x, n4m_matrix_view_t out) { \
    return call_status([&] { return h ? selector_transform(h->s, x, out) : N4M_ERR_NULL_POINTER; }); \
} \
extern "C" n4m_status_t prefix##_output_cols(const HandleType* h, int64_t* out) { \
    if (!out) return N4M_ERR_NULL_POINTER; \
    if (!h || !h->s.fitted) return N4M_ERR_NOT_FITTED; \
    *out = static_cast<int64_t>(h->s.indices.size()); \
    return N4M_OK; \
} \
extern "C" n4m_status_t prefix##_is_fitted(const HandleType* h, int* out) { \
    if (!out) return N4M_ERR_NULL_POINTER; \
    *out = (h && h->s.fitted) ? 1 : 0; \
    return N4M_OK; \
}

DEFINE_SELECTOR_CREATE(n4m_filter_variance, n4m_filter_variance_handle_t, false)
extern "C" n4m_status_t n4m_filter_variance_fit(n4m_filter_variance_handle_t* h,
    n4m_matrix_view_t x) {
    return call_status([&] { return h ? variance_fit(h->s, x, nullptr, 0) : N4M_ERR_NULL_POINTER; });
}

DEFINE_SELECTOR_CREATE(n4m_filter_correlation, n4m_filter_correlation_handle_t, true)
extern "C" n4m_status_t n4m_filter_correlation_fit(n4m_filter_correlation_handle_t* h,
    n4m_matrix_view_t x, const double* y, int64_t n_y) {
    return call_status([&] { return h ? variance_fit(h->s, x, y, n_y) : N4M_ERR_NULL_POINTER; });
}

extern "C" n4m_status_t n4m_interval_generator_create(
    n4m_interval_generator_handle_t** out, int32_t interval_size, int32_t step) {
    n4m_interval_generator_handle_t init;
    init.s.width = interval_size;
    init.s.step = step;
    return make_handle(out, init);
}
extern "C" void n4m_interval_generator_destroy(n4m_interval_generator_handle_t* h) { delete h; }
extern "C" n4m_status_t n4m_interval_generator_fit(n4m_interval_generator_handle_t* h,
    n4m_matrix_view_t x) {
    return call_status([&] { return h ? interval_fit(h->s, x) : N4M_ERR_NULL_POINTER; });
}
extern "C" n4m_status_t n4m_interval_generator_transform(
    const n4m_interval_generator_handle_t* h, n4m_matrix_view_t x, n4m_matrix_view_t out) {
    return call_status([&] { return h ? interval_transform(h->s, x, out) : N4M_ERR_NULL_POINTER; });
}
extern "C" n4m_status_t n4m_interval_generator_output_cols(
    const n4m_interval_generator_handle_t* h, int64_t* out) {
    if (!out) return N4M_ERR_NULL_POINTER;
    if (!h || !h->s.fitted) return N4M_ERR_NOT_FITTED;
    *out = interval_output_cols(h->s);
    return N4M_OK;
}
extern "C" n4m_status_t n4m_interval_generator_is_fitted(
    const n4m_interval_generator_handle_t* h, int* out) {
    if (!out) return N4M_ERR_NULL_POINTER;
    *out = (h && h->s.fitted) ? 1 : 0;
    return N4M_OK;
}
