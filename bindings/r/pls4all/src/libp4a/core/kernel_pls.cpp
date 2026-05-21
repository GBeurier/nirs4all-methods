// SPDX-License-Identifier: CECILL-2.1

#include "core/kernel_pls.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <new>
#include <vector>

#include "core/matrix_view.hpp"

namespace n4m::core {

namespace {

constexpr double kEps = 1e-12;

[[nodiscard]] n4m_status_t copy_matrix(Context& ctx,
                                        const n4m_matrix_view_t& V,
                                        const char* name,
                                        std::vector<double>& out) {
    if (validate_nonnull_view(V) != N4M_OK || V.dtype != N4M_DTYPE_F64) {
        ctx.set_errorf("%s must be a finite F64 row-major matrix", name);
        return N4M_ERR_INVALID_ARGUMENT;
    }
    const std::size_t rows = static_cast<std::size_t>(V.rows);
    const std::size_t cols = static_cast<std::size_t>(V.cols);
    out.assign(rows * cols, 0.0);
    const auto* data = static_cast<const double*>(V.data);
    if (data == nullptr) {
        if (rows > 0 && cols > 0) {
            ctx.set_errorf("%s has null data", name);
            return N4M_ERR_NULL_POINTER;
        }
        return N4M_OK;
    }
    const std::size_t rs = static_cast<std::size_t>(V.row_stride);
    const std::size_t cs = static_cast<std::size_t>(V.col_stride);
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            const double v = data[r * rs + c * cs];
            if (!std::isfinite(v)) {
                ctx.set_errorf("%s contains NaN or Inf", name);
                return N4M_ERR_INVALID_ARGUMENT;
            }
            out[r * cols + c] = v;
        }
    }
    return N4M_OK;
}

double kernel_value(KernelType kernel, double gamma, double coef0,
                    std::int32_t degree,
                    const double* a, const double* b, std::size_t p) {
    if (kernel == KernelType::LINEAR) {
        double s = 0.0;
        for (std::size_t i = 0; i < p; ++i) s += a[i] * b[i];
        return s;
    }
    if (kernel == KernelType::RBF) {
        double sumsq = 0.0;
        for (std::size_t i = 0; i < p; ++i) {
            const double d = a[i] - b[i];
            sumsq += d * d;
        }
        return std::exp(-gamma * sumsq);
    }
    if (kernel == KernelType::POLYNOMIAL) {
        double dot_val = 0.0;
        for (std::size_t i = 0; i < p; ++i) dot_val += a[i] * b[i];
        return std::pow(gamma * dot_val + coef0,
                        static_cast<double>(degree));
    }
    // SIGMOID
    double dot_val = 0.0;
    for (std::size_t i = 0; i < p; ++i) dot_val += a[i] * b[i];
    return std::tanh(gamma * dot_val + coef0);
}

void compute_gram(KernelType kernel, double gamma, double coef0,
                  std::int32_t degree,
                  const std::vector<double>& X1, std::size_t n1,
                  const std::vector<double>& X2, std::size_t n2,
                  std::size_t p, std::vector<double>& K) {
    K.assign(n1 * n2, 0.0);
    for (std::size_t i = 0; i < n1; ++i) {
        for (std::size_t j = 0; j < n2; ++j) {
            K[i * n2 + j] = kernel_value(kernel, gamma, coef0, degree,
                                         X1.data() + i * p,
                                         X2.data() + j * p, p);
        }
    }
}

double squared_norm(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x * x;
    return s;
}

}  // namespace

n4m_status_t fit_kernel_pls(Context& ctx,
                             const Config& cfg,
                             KernelType kernel,
                             double gamma,
                             double coef0,
                             std::int32_t degree,
                             const n4m_matrix_view_t& X,
                             const n4m_matrix_view_t& Y,
                             KernelPlsResult& out) {
    out = KernelPlsResult{};
    if (X.rows != Y.rows) {
        ctx.set_error("X and Y must share row count");
        return N4M_ERR_SHAPE_MISMATCH;
    }
    if (cfg.n_components < 1) {
        ctx.set_error("n_components must be >= 1");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (degree < 1) {
        ctx.set_error("polynomial degree must be >= 1");
        return N4M_ERR_INVALID_ARGUMENT;
    }

    std::vector<double> X_buf, Y_buf;
    n4m_status_t status = copy_matrix(ctx, X, "X", X_buf);
    if (status != N4M_OK) return status;
    status = copy_matrix(ctx, Y, "Y", Y_buf);
    if (status != N4M_OK) return status;

    const std::size_t n = static_cast<std::size_t>(X.rows);
    const std::size_t p = static_cast<std::size_t>(X.cols);
    const std::size_t q = static_cast<std::size_t>(Y.cols);
    if (n < 2 || p < 1 || q < 1) {
        ctx.set_error("kernel PLS requires at least 2 rows and 1 feature");
        return N4M_ERR_INVALID_ARGUMENT;
    }

    if (gamma <= 0.0) {
        gamma = 1.0 / static_cast<double>(p);
    }

    out.n_train = static_cast<std::int64_t>(n);
    out.n_features = static_cast<std::int32_t>(p);
    out.n_targets = static_cast<std::int32_t>(q);
    out.n_components = cfg.n_components;
    out.kernel_type = kernel;
    out.gamma = gamma;
    out.coef0 = coef0;
    out.degree = degree;
    out.x_train = X_buf;

    // Center Y.
    out.y_mean.assign(q, 0.0);
    if (cfg.center_y != 0) {
        for (std::size_t r = 0; r < n; ++r) {
            for (std::size_t target = 0; target < q; ++target) {
                out.y_mean[target] += Y_buf[r * q + target];
            }
        }
        for (std::size_t target = 0; target < q; ++target)
            out.y_mean[target] /= static_cast<double>(n);
        for (std::size_t r = 0; r < n; ++r) {
            for (std::size_t target = 0; target < q; ++target) {
                Y_buf[r * q + target] -= out.y_mean[target];
            }
        }
    }

    // Build Gram K_train.
    std::vector<double> K;
    compute_gram(kernel, gamma, coef0, degree, X_buf, n, X_buf, n, p, K);
    // Centering of K: K_c = K - 1_n K - K 1_n + 1_n K 1_n.
    out.K_train_row_means.assign(n, 0.0);
    double K_global = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            out.K_train_row_means[i] += K[i * n + j];
        }
        K_global += out.K_train_row_means[i];
        out.K_train_row_means[i] /= static_cast<double>(n);
    }
    out.K_train_global_mean = K_global / static_cast<double>(n * n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            K[i * n + j] = K[i * n + j] - out.K_train_row_means[i] -
                            out.K_train_row_means[j] +
                            out.K_train_global_mean;
        }
    }

    // Kernel NIPALS for PLS regression (Rosipal & Trejo 2001).
    const std::size_t a = static_cast<std::size_t>(cfg.n_components);
    std::vector<double> Y_res = Y_buf;
    std::vector<double> alpha(n * q, 0.0);
    std::vector<double> Kr = K;  // residual Gram

    for (std::size_t comp = 0; comp < a; ++comp) {
        // Initialize u with first column of Y_res.
        std::vector<double> u(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) u[i] = Y_res[i * q];

        std::vector<double> t(n, 0.0), c(q, 0.0);
        for (int iter = 0; iter < 100; ++iter) {
            // t = Kr u / ||Kr u||
            std::fill(t.begin(), t.end(), 0.0);
            for (std::size_t i = 0; i < n; ++i) {
                double s = 0.0;
                for (std::size_t j = 0; j < n; ++j) {
                    s += Kr[i * n + j] * u[j];
                }
                t[i] = s;
            }
            double t_norm = std::sqrt(squared_norm(t));
            if (t_norm < kEps) break;
            for (double& x : t) x /= t_norm;
            // c = Y_res' t / (t.t)
            std::fill(c.begin(), c.end(), 0.0);
            for (std::size_t target = 0; target < q; ++target) {
                double s = 0.0;
                for (std::size_t i = 0; i < n; ++i) {
                    s += Y_res[i * q + target] * t[i];
                }
                c[target] = s;  // t.t = 1
            }
            // u_new = Y_res c / (c.c)
            const double cc = squared_norm(c);
            if (cc < kEps) break;
            std::vector<double> u_new(n, 0.0);
            for (std::size_t i = 0; i < n; ++i) {
                double s = 0.0;
                for (std::size_t target = 0; target < q; ++target) {
                    s += Y_res[i * q + target] * c[target];
                }
                u_new[i] = s / cc;
            }
            double change = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                const double d = u_new[i] - u[i];
                change += d * d;
            }
            u = std::move(u_new);
            if (change < 1e-14) break;
        }

        const double tt = squared_norm(t);
        if (tt < kEps) break;

        // alpha gets a u t' / (t' Kr t) contribution: dual coefficients.
        // For prediction: predicted Y = K_test * alpha, where alpha sums
        // u t' / (t' Kr t) over components.
        double denom = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            double tmp = 0.0;
            for (std::size_t j = 0; j < n; ++j) {
                tmp += Kr[i * n + j] * t[j];
            }
            denom += t[i] * tmp;
        }
        if (std::fabs(denom) < kEps) break;
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t target = 0; target < q; ++target) {
                alpha[i * q + target] += u[i] * c[target] / denom;
            }
        }

        // Deflate residual Y and residual Kr:
        //  Y_res -= t * c'
        //  Kr -= t t' Kr - Kr t t' + t t' Kr t t' (sandwich centering)
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t target = 0; target < q; ++target) {
                Y_res[i * q + target] -= t[i] * c[target];
            }
        }
        std::vector<double> Kt(n, 0.0);  // Kr t
        for (std::size_t i = 0; i < n; ++i) {
            double s = 0.0;
            for (std::size_t j = 0; j < n; ++j) {
                s += Kr[i * n + j] * t[j];
            }
            Kt[i] = s;
        }
        // Kr deflation: Kr = (I - t t') Kr (I - t t')
        // = Kr - t (t' Kr) - (Kr t) t' + t (t' Kr t) t'
        const double tKt = denom;
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                Kr[i * n + j] -= t[i] * Kt[j] + Kt[i] * t[j] -
                                  t[i] * tKt * t[j];
            }
        }
    }

    out.alpha = std::move(alpha);
    ctx.clear_error();
    return N4M_OK;
}

n4m_status_t predict_kernel_pls(Context& ctx,
                                 const KernelPlsResult& model,
                                 const n4m_matrix_view_t& X,
                                 std::vector<double>& out_predictions) {
    out_predictions.clear();
    if (model.x_train.empty()) {
        ctx.set_error("kernel PLS model is not fitted");
        return N4M_ERR_NOT_FITTED;
    }
    if (X.cols != model.n_features) {
        ctx.set_error("X cols must match training n_features");
        return N4M_ERR_SHAPE_MISMATCH;
    }
    std::vector<double> X_buf;
    n4m_status_t status = copy_matrix(ctx, X, "X", X_buf);
    if (status != N4M_OK) return status;
    const std::size_t n_test = static_cast<std::size_t>(X.rows);
    const std::size_t n_train = static_cast<std::size_t>(model.n_train);
    const std::size_t p = static_cast<std::size_t>(model.n_features);
    const std::size_t q = static_cast<std::size_t>(model.n_targets);

    std::vector<double> K_test;
    compute_gram(model.kernel_type, model.gamma, model.coef0, model.degree,
                 X_buf, n_test, model.x_train, n_train, p, K_test);
    // Center K_test using training means:
    //   K_c[i,j] = K_test[i,j] - row_mean_test[i] - col_mean_train[j] +
    //              global_mean_train
    // where row_mean_test[i] = mean_j K_test[i,j]
    std::vector<double> row_mean_test(n_test, 0.0);
    for (std::size_t i = 0; i < n_test; ++i) {
        double s = 0.0;
        for (std::size_t j = 0; j < n_train; ++j) s += K_test[i * n_train + j];
        row_mean_test[i] = s / static_cast<double>(n_train);
    }
    for (std::size_t i = 0; i < n_test; ++i) {
        for (std::size_t j = 0; j < n_train; ++j) {
            K_test[i * n_train + j] =
                K_test[i * n_train + j] - row_mean_test[i] -
                model.K_train_row_means[j] + model.K_train_global_mean;
        }
    }
    // Predictions = K_test @ alpha + y_mean.
    out_predictions.assign(n_test * q, 0.0);
    for (std::size_t i = 0; i < n_test; ++i) {
        for (std::size_t target = 0; target < q; ++target) {
            double s = model.y_mean[target];
            for (std::size_t j = 0; j < n_train; ++j) {
                s += K_test[i * n_train + j] * model.alpha[j * q + target];
            }
            out_predictions[i * q + target] = s;
        }
    }
    ctx.clear_error();
    return N4M_OK;
}

}  // namespace n4m::core
