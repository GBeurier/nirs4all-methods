// SPDX-License-Identifier: CECILL-2.1
//
// Manifest-driven conformance suite for the generic estimator surface.
// Every estimator in the compiled manifest is fitted on a train split,
// predicts held-out rows, round-trips through N4ME with bitwise-identical
// outputs, and refuses missing or unused inputs.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "n4m/n4m.h"

#define CHECK(expr)                                                                   \
    do {                                                                              \
        if (!(expr)) throw std::runtime_error(std::string(#expr) + " @" + current_); \
    } while (0)

namespace {

std::string current_;

constexpr int64_t kTrain = 36;
constexpr int64_t kTest = 12;
constexpr int64_t kCols = 12;

n4m_matrix_view_t view(double* data, int64_t rows, int64_t cols) {
    n4m_matrix_view_t v{};
    CHECK(n4m_matrix_view_init_rowmajor(&v, data, rows, cols, N4M_DTYPE_F64) == N4M_OK);
    return v;
}

// Deterministic smooth spectra with a linear response.
struct Data {
    std::vector<double> x_train, y_train, x_test, x_target;
    Data() {
        auto spectrum = [](int64_t i, int64_t j) {
            const double a = std::sin(0.37 * static_cast<double>(i) + 0.11);
            const double b = std::cos(0.23 * static_cast<double>(i * i % 17));
            const double t = static_cast<double>(j) / kCols;
            return a * std::exp(-8.0 * (t - 0.3) * (t - 0.3)) +
                   b * std::exp(-10.0 * (t - 0.7) * (t - 0.7)) + 0.05 * std::sin(i * 1.7 + j);
        };
        for (int64_t i = 0; i < kTrain + kTest; ++i) {
            std::vector<double>& x = i < kTrain ? x_train : x_test;
            double y = 0.0;
            for (int64_t j = 0; j < kCols; ++j) {
                const double v = spectrum(i, j);
                x.push_back(v);
                y += (j % 3 == 0 ? 1.5 : -0.4) * v;
            }
            if (i < kTrain) y_train.push_back(y + 0.01 * std::sin(3.1 * i));
        }
        for (int64_t i = 0; i < kTrain; ++i) {
            for (int64_t j = 0; j < kCols; ++j) x_target.push_back(spectrum(i + 101, j) + 0.2);
        }
    }
};

struct Inputs {
    Data data;
    n4m_matrix_view_t X{}, Y{}, Xt{};
    std::vector<int64_t> feature_groups, blocks;
    Inputs() {
        X = view(data.x_train.data(), kTrain, kCols);
        Y = view(data.y_train.data(), kTrain, 1);
        Xt = view(data.x_target.data(), kTrain, kCols);
        for (int64_t j = 0; j < kCols; ++j) feature_groups.push_back(j / 4);
        blocks = {4, 4, 4};
    }
    // Inputs the method requires, plus Y.
    n4m_fit_inputs_v1_t for_method(const n4m_method_info_v1_t& info) {
        n4m_fit_inputs_v1_t in{};
        in.struct_size = sizeof(in);
        in.X = &X;
        in.seed = 7;
        auto wants = [&](n4m_fit_input_t k) { return info.inputs[k] != N4M_INPUT_NONE; };
        if (wants(N4M_FIT_INPUT_Y)) in.Y = &Y;
        if (wants(N4M_FIT_INPUT_FEATURE_GROUPS)) {
            in.feature_groups = feature_groups.data();
            in.n_feature_groups = kCols;
        }
        if (wants(N4M_FIT_INPUT_BLOCKS)) {
            in.block_sizes = blocks.data();
            in.n_blocks = static_cast<int64_t>(blocks.size());
        }
        if (wants(N4M_FIT_INPUT_TARGET_DOMAIN)) in.X_target = &Xt;
        return in;
    }
};

// Parameters the test must supply (required, no default).
void fill_required(int32_t index, n4m_params_t* params) {
    n4m_method_info_v1_t info{};
    info.struct_size = sizeof(info);
    CHECK(n4m_method_info_v1(index, &info) == N4M_OK);
    for (int32_t p = 0; p < info.n_params; ++p) {
        n4m_param_info_v1_t pi{};
        pi.struct_size = sizeof(pi);
        CHECK(n4m_method_param_info_v1(index, p, &pi) == N4M_OK);
        if (pi.has_default) continue;
        // N-PLS tensor modes: 12 columns = 3 x 4.
        if (std::strcmp(pi.name, "mode_j") == 0) {
            CHECK(n4m_params_set_int(params, pi.name, 3) == N4M_OK);
        } else if (std::strcmp(pi.name, "mode_k") == 0) {
            CHECK(n4m_params_set_int(params, pi.name, 4) == N4M_OK);
        } else {
            throw std::runtime_error(std::string("no test value for required parameter ") +
                                     pi.name + " @" + current_);
        }
    }
}

std::vector<unsigned char> export_bytes(n4m_context_t* ctx, const n4m_estimator_t* est) {
    size_t size = 0;
    CHECK(n4m_estimator_export_size(ctx, est, 0, &size) == N4M_OK);
    std::vector<unsigned char> bytes(size);
    size_t written = 0;
    CHECK(n4m_estimator_export_to_buffer(ctx, est, 0, bytes.data(), size, &written) == N4M_OK);
    CHECK(written == size);
    return bytes;
}

void conformance(n4m_context_t* ctx, Inputs& in, int32_t index) {
    n4m_method_info_v1_t info{};
    info.struct_size = sizeof(info);
    CHECK(n4m_method_info_v1(index, &info) == N4M_OK);
    current_ = info.method_id;
    if (info.kind != N4M_METHOD_ESTIMATOR) return;

    n4m_params_t* params = nullptr;
    CHECK(n4m_params_create(ctx, index, &params) == N4M_OK);
    fill_required(index, params);
    CHECK(n4m_params_validate(ctx, params) == N4M_OK);
    n4m_estimator_t* est = nullptr;
    CHECK(n4m_estimator_create(ctx, info.method_id, params, &est) == N4M_OK);
    n4m_params_destroy(params);

    // Not fitted yet.
    std::vector<double> pred(kTest), pred2(kTest);
    auto X_test = view(in.data.x_test.data(), kTest, kCols);
    auto P = view(pred.data(), kTest, 1);
    CHECK(n4m_estimator_predict(ctx, est, &X_test, &P) == N4M_ERR_NOT_FITTED);

    // Missing / unused inputs are refused and leave the estimator unfitted.
    n4m_fit_inputs_v1_t inputs = in.for_method(info);
    for (int k = 0; k < N4M_FIT_INPUT_COUNT; ++k) {
        if (info.inputs[k] != N4M_INPUT_REQUIRED || k == N4M_FIT_INPUT_Y) continue;
        n4m_fit_inputs_v1_t missing = inputs;
        if (k == N4M_FIT_INPUT_FEATURE_GROUPS) missing.feature_groups = nullptr;
        if (k == N4M_FIT_INPUT_BLOCKS) missing.block_sizes = nullptr;
        if (k == N4M_FIT_INPUT_TARGET_DOMAIN) missing.X_target = nullptr;
        CHECK(n4m_estimator_fit(ctx, est, &missing) == N4M_ERR_INVALID_ARGUMENT);
    }
    if (info.inputs[N4M_FIT_INPUT_GROUPS] == N4M_INPUT_NONE) {
        n4m_fit_inputs_v1_t extra = inputs;
        std::vector<int64_t> groups(kTrain, 0);
        extra.groups = groups.data();
        extra.n_groups = kTrain;
        CHECK(n4m_estimator_fit(ctx, est, &extra) == N4M_ERR_INVALID_ARGUMENT);
    }

    CHECK(n4m_estimator_fit(ctx, est, &inputs) == N4M_OK);
    int32_t fitted = 0, idx = -1;
    uint64_t caps = 0;
    CHECK(n4m_estimator_is_fitted(est, &fitted) == N4M_OK && fitted == 1);
    CHECK(n4m_estimator_info(est, &idx, &caps) == N4M_OK && idx == index);
    CHECK(caps == info.capabilities);
    // Role interfaces and fitted capabilities agree both ways.
    const bool predicts = (info.roles & (N4M_ROLE_REGRESSOR | N4M_ROLE_CLASSIFIER)) != 0;
    const bool transforms = (info.roles & (N4M_ROLE_TRANSFORMER | N4M_ROLE_SELECTOR)) != 0;
    CHECK(predicts == ((caps & N4M_CAP_PREDICT) != 0));
    CHECK(transforms == ((caps & N4M_CAP_TRANSFORM) != 0));
    CHECK(((info.roles & N4M_ROLE_CLASSIFIER) != 0) == ((caps & N4M_CAP_PREDICT_LABELS) != 0));
    CHECK(((info.roles & N4M_ROLE_SELECTOR) != 0) == ((caps & N4M_CAP_SELECTED_INDICES) != 0));
    CHECK(((info.roles & N4M_ROLE_SAMPLE_FILTER) != 0) == ((caps & N4M_CAP_APPLY_MASK) != 0));
    int64_t n_in = 0, n_out = 0;
    CHECK(n4m_estimator_n_features_in(est, &n_in) == N4M_OK && n_in == kCols);
    CHECK(n4m_estimator_n_outputs(est, &n_out) == N4M_OK && n_out == 1);

    // Out-of-sample predictions are finite and informative.
    CHECK((caps & N4M_CAP_PREDICT) != 0);
    CHECK(n4m_estimator_predict(ctx, est, &X_test, &P) == N4M_OK);
    double spread = 0.0;
    for (double v : pred) {
        CHECK(std::isfinite(v));
        spread += std::fabs(v - pred[0]);
    }
    CHECK(spread > 0.0);
    auto bad = view(pred.data(), kTest - 1, 1);
    CHECK(n4m_estimator_predict(ctx, est, &X_test, &bad) == N4M_ERR_SHAPE_MISMATCH);

    std::vector<double> scores;
    int64_t tcols = 0;
    if ((caps & N4M_CAP_TRANSFORM) != 0) {
        CHECK(n4m_estimator_transform_cols(est, &tcols) == N4M_OK && tcols > 0);
        scores.resize(static_cast<size_t>(kTest * tcols));
        auto S = view(scores.data(), kTest, tcols);
        CHECK(n4m_estimator_transform(ctx, est, &X_test, &S) == N4M_OK);
    } else {
        auto S = view(pred2.data(), kTest, 1);
        CHECK(n4m_estimator_transform(ctx, est, &X_test, &S) == N4M_ERR_UNSUPPORTED);
    }

    // N4ME round trip: identical outputs and identical re-export.
    const auto bytes = export_bytes(ctx, est);
    n4m_estimator_t* back = nullptr;
    CHECK(n4m_estimator_import_from_buffer(ctx, bytes.data(), bytes.size(), &back) == N4M_OK);
    auto P2 = view(pred2.data(), kTest, 1);
    CHECK(n4m_estimator_predict(ctx, back, &X_test, &P2) == N4M_OK);
    CHECK(std::memcmp(pred.data(), pred2.data(), sizeof(double) * kTest) == 0);
    if ((caps & N4M_CAP_TRANSFORM) != 0) {
        std::vector<double> scores2(scores.size());
        auto S2 = view(scores2.data(), kTest, tcols);
        CHECK(n4m_estimator_transform(ctx, back, &X_test, &S2) == N4M_OK);
        CHECK(std::memcmp(scores.data(), scores2.data(), sizeof(double) * scores.size()) == 0);
    }
    CHECK(export_bytes(ctx, back) == bytes);
    uint64_t back_caps = 0;
    CHECK(n4m_estimator_info(back, &idx, &back_caps) == N4M_OK && back_caps == caps);

    // Corruption and size limits fail closed.
    auto corrupt = bytes;
    corrupt[corrupt.size() / 2] ^= 0x40;
    n4m_estimator_t* none = nullptr;
    CHECK(n4m_estimator_import_from_buffer(ctx, corrupt.data(), corrupt.size(), &none) ==
          N4M_ERR_CORRUPT_BUFFER);
    CHECK(none == nullptr);
    CHECK(n4m_estimator_import_from_buffer(ctx, bytes.data(), bytes.size() - 9, &none) !=
          N4M_OK);
    CHECK(n4m_context_set_max_state_bytes(ctx, bytes.size() - 1) == N4M_OK);
    CHECK(n4m_estimator_import_from_buffer(ctx, bytes.data(), bytes.size(), &none) ==
          N4M_ERR_INVALID_ARGUMENT);
    CHECK(n4m_context_set_max_state_bytes(ctx, uint64_t{256} << 20) == N4M_OK);

    n4m_estimator_destroy(back);
    n4m_estimator_destroy(est);
}

void test_introspection_and_params(n4m_context_t* ctx) {
    current_ = "introspection";
    int32_t count = 0;
    CHECK(n4m_method_count(&count) == N4M_OK && count > 0);
    int32_t index = -1;
    CHECK(n4m_method_find("no.such.method", &index) == N4M_ERR_INVALID_ARGUMENT && index == -1);
    CHECK(n4m_method_find("models.pls.pls_regression", &index) == N4M_OK);

    // Descriptor rules: short struct_size is refused and zeroed.
    n4m_method_info_v1_t info{};
    info.struct_size = 4;
    CHECK(n4m_method_info_v1(index, &info) == N4M_ERR_INVALID_ARGUMENT);
    info.struct_size = sizeof(info);
    CHECK(n4m_method_info_v1(index, &info) == N4M_OK);
    CHECK((info.roles & N4M_ROLE_REGRESSOR) != 0 && (info.roles & N4M_ROLE_TRANSFORMER) != 0);

    n4m_params_t* params = nullptr;
    CHECK(n4m_params_create(ctx, index, &params) == N4M_OK);
    int64_t v = 0, n = 0;
    CHECK(n4m_params_get_int(params, "n_components", &v, 1, &n) == N4M_OK && v == 2 && n == 1);
    CHECK(n4m_params_get_int(params, "solver", &v, 1, &n) == N4M_OK && v == N4M_SOLVER_NIPALS);
    CHECK(n4m_params_set_enum(params, "solver", "simpls") == N4M_OK);
    CHECK(n4m_params_get_int(params, "solver", &v, 1, &n) == N4M_OK && v == N4M_SOLVER_SIMPLS);
    CHECK(n4m_params_set_enum(params, "solver", "bogus") == N4M_ERR_INVALID_ARGUMENT);
    CHECK(n4m_params_set_int(params, "n_components", 0) == N4M_ERR_INVALID_ARGUMENT);
    CHECK(n4m_params_set_int(params, "no_such", 1) == N4M_ERR_INVALID_ARGUMENT);
    CHECK(n4m_params_set_double(params, "n_components", 2.0) == N4M_ERR_INVALID_ARGUMENT);
    CHECK(n4m_params_set_bool(params, "scale_y", 2) == N4M_ERR_INVALID_ARGUMENT);
    CHECK(n4m_params_set_bool(params, "scale_y", 0) == N4M_OK);

    // Params belong to one method.
    int32_t other = -1;
    CHECK(n4m_method_find("models.pls.cppls", &other) == N4M_OK);
    n4m_estimator_t* est = nullptr;
    CHECK(n4m_estimator_create(ctx, "models.pls.cppls", params, &est) == N4M_ERR_INVALID_ARGUMENT);
    CHECK(est == nullptr);
    n4m_params_destroy(params);

    // Required parameters without default are named.
    CHECK(n4m_method_find("models.specialized.tensor_pls", &index) == N4M_OK);
    CHECK(n4m_estimator_create(ctx, "models.specialized.tensor_pls", nullptr, &est) ==
          N4M_ERR_INVALID_ARGUMENT);
    CHECK(std::strstr(n4m_context_last_error(ctx), "mode_j") != nullptr);
}

// The generic PLS estimator reproduces n4m_estimators_pls_fit.
void test_pls_fit_simple_equivalence(n4m_context_t* ctx, Inputs& in) {
    current_ = "pls_fit_simple equivalence";
    std::vector<double> coef(kCols), xm(kCols), ym(1);
    CHECK(n4m_estimators_pls_fit(in.data.x_train.data(), in.data.y_train.data(),
                                 static_cast<int32_t>(kTrain), static_cast<int32_t>(kCols), 1, 2,
                                 coef.data(), xm.data(), ym.data(), nullptr) == N4M_OK);
    n4m_estimator_t* est = nullptr;
    CHECK(n4m_estimator_create(ctx, "models.pls.pls_fit_simple", nullptr, &est) == N4M_OK);
    n4m_fit_inputs_v1_t inputs{};
    inputs.struct_size = sizeof(inputs);
    inputs.X = &in.X;
    inputs.Y = &in.Y;
    CHECK(n4m_estimator_fit(ctx, est, &inputs) == N4M_OK);
    std::vector<double> pred(kTest);
    auto X_test = view(in.data.x_test.data(), kTest, kCols);
    auto P = view(pred.data(), kTest, 1);
    CHECK(n4m_estimator_predict(ctx, est, &X_test, &P) == N4M_OK);
    for (int64_t i = 0; i < kTest; ++i) {
        double ref = ym[0];
        for (int64_t j = 0; j < kCols; ++j) {
            ref += (in.data.x_test[static_cast<size_t>(i * kCols + j)] - xm[static_cast<size_t>(j)]) *
                   coef[static_cast<size_t>(j)];
        }
        CHECK(std::fabs(ref - pred[static_cast<size_t>(i)]) <= 1e-9 * (1.0 + std::fabs(ref)));
    }
    n4m_estimator_destroy(est);
}

}  // namespace

int main() {
    n4m_context_t* ctx = nullptr;
    if (n4m_context_create(&ctx) != N4M_OK) return 1;
    int failures = 0;
    int checked = 0;
    auto run = [&](auto&& fn) {
        try {
            fn();
        } catch (const std::exception& e) {
            std::fprintf(stderr, "FAIL %s (%s)\n", e.what(), n4m_context_last_error(ctx));
            ++failures;
        }
    };
    Inputs in;
    run([&] { test_introspection_and_params(ctx); });
    run([&] { test_pls_fit_simple_equivalence(ctx, in); });
    int32_t count = 0;
    n4m_method_count(&count);
    for (int32_t i = 0; i < count; ++i) {
        run([&] { conformance(ctx, in, i); });
        ++checked;
    }
    n4m_context_destroy(ctx);
    std::printf("n4m_estimator_tests: %d methods, %d failures\n", checked, failures);
    return failures == 0 ? 0 : 1;
}
