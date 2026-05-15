// SPDX-License-Identifier: CeCILL-2.1
//
// Non-linear kernel PLS (§10.2). Verifies RBF and polynomial kernels
// produce finite predictions on a non-linear synthetic dataset.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "pls4all/p4a.h"

#include "core/config.hpp"
#include "core/context.hpp"
#include "core/kernel_pls.hpp"

#include "harness.hpp"

namespace {

p4a_matrix_view_t make_view(const std::vector<double>& data,
                            std::int64_t rows,
                            std::int64_t cols) {
    p4a_matrix_view_t view{};
    view.data = const_cast<double*>(data.data());
    view.rows = rows;
    view.cols = cols;
    view.row_stride = cols;
    view.col_stride = 1;
    view.dtype = P4A_DTYPE_F64;
    return view;
}

void make_dataset(std::vector<double>& Xv, std::vector<double>& Yv) {
    // 12 samples, 2 features, non-linear relationship
    // y = sin(x1) * cos(x2) + 0.1 * noise
    const std::vector<std::pair<double, double>> pts = {
        {0.1, 0.2}, {0.3, 0.4}, {0.5, 0.1}, {0.7, 0.9},
        {0.2, 0.6}, {0.4, 0.3}, {0.6, 0.8}, {0.8, 0.5},
        {0.9, 0.2}, {0.1, 0.7}, {0.3, 0.9}, {0.5, 0.4},
    };
    Xv.clear();
    Yv.clear();
    for (auto& pt : pts) {
        Xv.push_back(pt.first);
        Xv.push_back(pt.second);
        Yv.push_back(std::sin(2.0 * pt.first) * std::cos(2.0 * pt.second));
    }
}

}  // namespace

TEST(kernel_pls_phase24, rbf_kernel_fits_and_predicts) {
    std::vector<double> Xv;
    std::vector<double> Yv;
    make_dataset(Xv, Yv);
    p4a_matrix_view_t X = make_view(Xv, 12, 2);
    p4a_matrix_view_t Y = make_view(Yv, 12, 1);

    ::pls4all::core::Context ctx;
    ::pls4all::core::Config cfg;
    cfg.n_components = 3;
    cfg.center_x = 1;
    cfg.center_y = 1;
    ::pls4all::core::KernelPlsResult model;
    CHECK_EQ(::pls4all::core::fit_kernel_pls(
                 ctx, cfg, ::pls4all::core::KernelType::RBF, 1.0, 0.0, 3,
                 X, Y, model),
             P4A_OK);
    CHECK_EQ(model.n_train, 12);

    std::vector<double> preds;
    CHECK_EQ(::pls4all::core::predict_kernel_pls(ctx, model, X, preds),
             P4A_OK);
    CHECK_EQ(preds.size(), static_cast<std::size_t>(12));
    for (double v : preds) CHECK(std::isfinite(v));
}

TEST(kernel_pls_phase24, polynomial_kernel_fits_and_predicts) {
    std::vector<double> Xv;
    std::vector<double> Yv;
    make_dataset(Xv, Yv);
    p4a_matrix_view_t X = make_view(Xv, 12, 2);
    p4a_matrix_view_t Y = make_view(Yv, 12, 1);

    ::pls4all::core::Context ctx;
    ::pls4all::core::Config cfg;
    cfg.n_components = 3;
    cfg.center_y = 1;
    ::pls4all::core::KernelPlsResult model;
    CHECK_EQ(::pls4all::core::fit_kernel_pls(
                 ctx, cfg, ::pls4all::core::KernelType::POLYNOMIAL,
                 1.0, 1.0, 2, X, Y, model),
             P4A_OK);
    std::vector<double> preds;
    CHECK_EQ(::pls4all::core::predict_kernel_pls(ctx, model, X, preds),
             P4A_OK);
    for (double v : preds) CHECK(std::isfinite(v));
}

TEST(kernel_pls_phase24, rejects_invalid_inputs) {
    std::vector<double> Xv = {0.1, 0.2, 0.3, 0.4};
    std::vector<double> Yv = {1.0};
    p4a_matrix_view_t X = make_view(Xv, 2, 2);
    p4a_matrix_view_t Y = make_view(Yv, 1, 1);

    ::pls4all::core::Context ctx;
    ::pls4all::core::Config cfg;
    cfg.n_components = 1;
    ::pls4all::core::KernelPlsResult model;
    CHECK_EQ(::pls4all::core::fit_kernel_pls(
                 ctx, cfg, ::pls4all::core::KernelType::RBF, 1.0, 0.0, 3,
                 X, Y, model),
             P4A_ERR_SHAPE_MISMATCH);
}
