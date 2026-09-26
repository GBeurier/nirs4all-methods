// SPDX-License-Identifier: CECILL-2.1
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "n4m/n4m.h"

#define CHECK(expr) do { if (!(expr)) throw std::runtime_error(#expr); } while (0)

namespace {

n4m_matrix_view_t matrix(double* data, int64_t rows, int64_t cols) {
    n4m_matrix_view_t result{};
    CHECK(n4m_matrix_view_init_rowmajor(&result, data, rows, cols, N4M_DTYPE_F64) == N4M_OK);
    return result;
}

void test_sample_roles() {
    double x_data[] = {
        1,2,4, 2,5,3, 3,1,5, 4,7,2, 5,3,8,
        6,8,6, 7,4,7, 8,9,10, 9,6,9, 10,10,11
    };
    double y_data[] = {1,2,3,4,5,6,7,8,9,100};
    auto x = matrix(x_data, 10, 3);
    auto y = matrix(y_data, 10, 1);
    auto bad_y = matrix(y_data, 5, 2);
    uint8_t got[10]{};
    uint8_t want[10]{};
    n4m_filter_stats_t gs{}, ws{};
    const int64_t yi[] = {N4M_Y_OUTLIER_IQR};
    const double yd[] = {1.5, 5.0, 95.0};
    n4m_sample_filter_t* generic = nullptr;
    CHECK(n4m_sample_filter_create(N4M_SAMPLE_FILTER_Y_OUTLIER, yi, 1, yd, 3, 0, &generic) == N4M_OK);
    CHECK(n4m_sample_filter_apply(generic, &x, &y, got, &gs) == N4M_ERR_NOT_FITTED);
    CHECK(n4m_sample_filter_fit(generic, &x, &bad_y) == N4M_ERR_SHAPE_MISMATCH);
    CHECK(n4m_sample_filter_fit(generic, &x, &y) == N4M_OK);
    CHECK(n4m_sample_filter_apply(generic, &x, &y, got, &gs) == N4M_OK);
    n4m_filter_y_outlier_handle_t* direct_y = nullptr;
    CHECK(n4m_outlier_detection_y_outlier_create(&direct_y, 0, 1.5, 5.0, 95.0) == N4M_OK);
    CHECK(n4m_outlier_detection_y_outlier_fit(direct_y, y_data, 10) == N4M_OK);
    CHECK(n4m_outlier_detection_y_outlier_apply(direct_y, y_data, 10, want, &ws) == N4M_OK);
    for (int j = 0; j < 10; ++j) CHECK(got[j] == want[j]);
    CHECK(got[9] == 0 && gs.n_kept == ws.n_kept);
    n4m_outlier_detection_y_outlier_destroy(direct_y);
    n4m_sample_filter_destroy(generic);

    const int64_t xi[] = {N4M_X_OUTLIER_MAHALANOBIS, 0, 0, 100, 256};
    const double xd[] = {0.0, 0.2};
    CHECK(n4m_sample_filter_create(N4M_SAMPLE_FILTER_X_OUTLIER, xi, 5, xd, 2, 17, &generic) == N4M_OK);
    n4m_filter_x_outlier_handle_t* direct_x = nullptr;
    CHECK(n4m_outlier_detection_x_outlier_create(&direct_x, 0, 0, 0.0, 0, 0.2, 17, 100, 256) == N4M_OK);
    CHECK(n4m_sample_filter_fit(generic, &x, nullptr) == N4M_OK);
    CHECK(n4m_outlier_detection_x_outlier_fit(direct_x, x) == N4M_OK);
    CHECK(n4m_sample_filter_apply(generic, &x, &y, got, &gs) == N4M_OK);
    CHECK(n4m_outlier_detection_x_outlier_apply(direct_x, x, want, &ws) == N4M_OK);
    for (int j = 0; j < 10; ++j) CHECK(got[j] == want[j]);
    n4m_outlier_detection_x_outlier_destroy(direct_x);
    n4m_sample_filter_destroy(generic);

    const int64_t li[] = {0, 0, 0, 1};
    const double ld[] = {2.0, 0.0};
    CHECK(n4m_sample_filter_create(N4M_SAMPLE_FILTER_HIGH_LEVERAGE, li, 4, ld, 2, 0, &generic) == N4M_OK);
    n4m_filter_leverage_handle_t* direct_l = nullptr;
    CHECK(n4m_outlier_detection_high_leverage_create(&direct_l, 0, 2.0, 0, 0.0, 0, 1) == N4M_OK);
    CHECK(n4m_sample_filter_fit(generic, &x, nullptr) == N4M_OK);
    CHECK(n4m_outlier_detection_high_leverage_fit(direct_l, x) == N4M_OK);
    CHECK(n4m_sample_filter_apply(generic, &x, nullptr, got, &gs) == N4M_OK);
    CHECK(n4m_outlier_detection_high_leverage_apply(direct_l, x, want, &ws) == N4M_OK);
    for (int j = 0; j < 10; ++j) CHECK(got[j] == want[j]);
    n4m_outlier_detection_high_leverage_destroy(direct_l);
    n4m_sample_filter_destroy(generic);

    const int64_t qi[] = {0, 0, 1};
    const double qd[] = {0.1, 0.5, 1e-8, 0.0, 0.0};
    CHECK(n4m_sample_filter_create(N4M_SAMPLE_FILTER_SPECTRAL_QUALITY, qi, 3, qd, 5, 0, &generic) == N4M_OK);
    n4m_filter_quality_handle_t* direct_q = nullptr;
    CHECK(n4m_outlier_detection_spectral_quality_create(&direct_q, 0.1, 0.5, 1e-8, 0, 0.0, 0, 0.0, 1) == N4M_OK);
    CHECK(n4m_sample_filter_fit(generic, &x, nullptr) == N4M_OK);
    CHECK(n4m_sample_filter_apply(generic, &x, nullptr, got, &gs) == N4M_OK);
    CHECK(n4m_outlier_detection_spectral_quality_apply(direct_q, x, want, &ws) == N4M_OK);
    for (int j = 0; j < 10; ++j) CHECK(got[j] == want[j]);
    n4m_outlier_detection_spectral_quality_destroy(direct_q);
    n4m_sample_filter_destroy(generic);

    const int64_t ci[] = {N4M_COMPOSITE_ANY};
    CHECK(n4m_sample_filter_create(N4M_SAMPLE_FILTER_COMPOSITE, ci, 1, nullptr, 0, 0, &generic) == N4M_OK);
    CHECK(n4m_sample_filter_fit(generic, &x, nullptr) == N4M_ERR_INVALID_ARGUMENT);
    CHECK(n4m_sample_filter_add_child(generic, N4M_SAMPLE_FILTER_HIGH_LEVERAGE, li, 4, ld, 2, 0) == N4M_OK);
    CHECK(n4m_sample_filter_add_child(generic, N4M_SAMPLE_FILTER_SPECTRAL_QUALITY, qi, 3, qd, 5, 0) == N4M_OK);
    CHECK(n4m_sample_filter_fit(generic, &x, nullptr) == N4M_OK);
    CHECK(n4m_sample_filter_apply(generic, &x, nullptr, got, &gs) == N4M_OK);
    n4m_filter_leverage_handle_t* composite_l = nullptr;
    n4m_filter_quality_handle_t* composite_q = nullptr;
    n4m_filter_composite_handle_t* direct_c = nullptr;
    CHECK(n4m_outlier_detection_high_leverage_create(&composite_l, 0, 2.0, 0, 0.0, 0, 1) == N4M_OK);
    CHECK(n4m_outlier_detection_spectral_quality_create(&composite_q, 0.1, 0.5, 1e-8, 0, 0.0, 0, 0.0, 1) == N4M_OK);
    CHECK(n4m_outlier_detection_composite_create(&direct_c, 0) == N4M_OK);
    CHECK(n4m_outlier_detection_composite_add_leverage(direct_c, composite_l) == N4M_OK);
    CHECK(n4m_outlier_detection_composite_add_quality(direct_c, composite_q) == N4M_OK);
    CHECK(n4m_outlier_detection_high_leverage_fit(composite_l, x) == N4M_OK);
    CHECK(n4m_outlier_detection_composite_apply(direct_c, x, want, &ws) == N4M_OK);
    for (int j = 0; j < 10; ++j) CHECK(got[j] == want[j]);
    n4m_outlier_detection_composite_destroy(direct_c);
    n4m_outlier_detection_high_leverage_destroy(composite_l);
    n4m_outlier_detection_spectral_quality_destroy(composite_q);
    n4m_sample_filter_destroy(generic); // owns both children; no borrowed dangling handles
    CHECK(n4m_sample_filter_create(N4M_SAMPLE_FILTER_Y_OUTLIER, yi, 1, yd, 2, 0, &generic) == N4M_ERR_INVALID_ARGUMENT);
    CHECK(generic == nullptr);
}

void test_feature_roles() {
    double train[] = {1,4,2, 2,4,3, 3,4,4, 4,4,5, 5,4,6};
    double test[] = {6,9,7, 7,9,8};
    double y_data[] = {1,2,3,4,5};
    auto x = matrix(train, 5, 3);
    auto xt = matrix(test, 2, 3);
    auto y = matrix(y_data, 5, 1);
    auto bad_y = matrix(y_data, 1, 5);
    for (int kind = 0; kind < 2; ++kind) {
        n4m_feature_filter_t* h = nullptr;
        CHECK(n4m_feature_filter_create(kind, 0.01, -1, &h) == N4M_OK);
        int64_t count = 99;
        CHECK(n4m_feature_filter_selected_indices(h, nullptr, 0, &count) == N4M_ERR_NOT_FITTED);
        CHECK(n4m_feature_filter_fit(h, &x, &bad_y) != N4M_OK);
        CHECK(n4m_feature_filter_fit(h, &x, kind == 0 ? nullptr : &y) == N4M_OK);
        CHECK(n4m_feature_filter_selected_indices(h, nullptr, 0, &count) == N4M_OK);
        CHECK(count == 2);
        int64_t indices[] = {-1, -1};
        CHECK(n4m_feature_filter_selected_indices(h, indices, 1, &count) == N4M_ERR_INVALID_ARGUMENT);
        CHECK(indices[0] == -1);
        CHECK(n4m_feature_filter_selected_indices(h, indices, 2, &count) == N4M_OK);
        CHECK(indices[0] == 0 && indices[1] == 2);
        double out_data[4]{};
        auto out = matrix(out_data, 2, 2);
        CHECK(n4m_feature_filter_transform(h, &xt, &out) == N4M_OK);
        CHECK(out_data[0] == 6 && out_data[1] == 7 && out_data[2] == 7 && out_data[3] == 8);
        n4m_feature_filter_destroy(h);
    }
}

} // namespace

int main() {
    try { test_sample_roles(); test_feature_roles(); }
    catch (const std::exception& e) { std::fprintf(stderr, "filter role: %s\n", e.what()); return 1; }
    return 0;
}
