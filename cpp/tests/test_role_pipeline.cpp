// SPDX-License-Identifier: CECILL-2.1
//
// Role pipeline (n4m/estimator.h, ABI 2.14): recipe validation, fit routing
// (filters, supervised transformers, multi-target Y, label targets), the
// feature-identity check, and N4ME export / import with its refusals. Each
// fitted pipeline is compared bitwise with the same chain of estimators run
// by hand through n4m_estimator_*.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "n4m/n4m.h"

#define CHECK(expr)                                                                   \
    do {                                                                              \
        if (!(expr)) throw std::runtime_error(std::string(#expr) + " @" + current_); \
    } while (0)

namespace {

std::string current_;
n4m_context_t* ctx_ = nullptr;

constexpr int64_t kRows = 40;
constexpr int64_t kCols = 10;

n4m_matrix_view_t view(std::vector<double>& data, int64_t rows, int64_t cols) {
    n4m_matrix_view_t v{};
    CHECK(n4m_matrix_view_init_rowmajor(&v, data.data(), rows, cols, N4M_DTYPE_F64) == N4M_OK);
    return v;
}

struct Data {
    std::vector<double> x, y, y2;
    std::vector<int64_t> labels;
    Data() {
        for (int64_t i = 0; i < kRows; ++i) {
            const double a = std::sin(0.37 * static_cast<double>(i) + 0.11);
            const double b = std::cos(0.23 * static_cast<double>(i * i % 17));
            double t0 = 0.0;
            for (int64_t j = 0; j < kCols; ++j) {
                const double u = static_cast<double>(j) / kCols;
                const double v = 1.0 + a * std::exp(-8.0 * (u - 0.3) * (u - 0.3)) +
                                 b * std::exp(-10.0 * (u - 0.7) * (u - 0.7)) +
                                 0.05 * std::sin(static_cast<double>(i) * 1.7 + static_cast<double>(j));
                x.push_back(v);
                t0 += (j % 3 == 0 ? 1.5 : -0.4) * v;
            }
            // One gross outlier in y for the target filter.
            y.push_back(i == 7 ? 60.0 : t0);
            y2.push_back(t0);
            y2.push_back(a - 2.0 * b);
            labels.push_back(a > 0.3 ? 20 : (a > -0.3 ? 10 : 30));
        }
    }
};

std::string error_text() {
    const char* msg = n4m_context_last_error(ctx_);
    return msg != nullptr ? msg : "";
}

void expect_error(n4m_status_t st, n4m_status_t expected, const char* fragment) {
    if (st != expected || error_text().find(fragment) == std::string::npos) {
        throw std::runtime_error("expected status " + std::to_string(expected) + " with '" +
                                 fragment + "', got " + std::to_string(st) + ": " +
                                 error_text() + " @" + current_);
    }
}

n4m_params_t* params_for(const char* id) {
    int32_t index = -1;
    CHECK(n4m_method_find(id, &index) == N4M_OK);
    n4m_params_t* p = nullptr;
    CHECK(n4m_params_create(ctx_, index, &p) == N4M_OK);
    return p;
}

n4m_status_t create(std::vector<const char*> ids, std::vector<const n4m_params_t*> params,
                    n4m_role_pipeline_t** out) {
    if (params.empty()) params.assign(ids.size(), nullptr);
    return n4m_role_pipeline_create(ctx_, static_cast<int32_t>(ids.size()), ids.data(),
                                    params.data(), out);
}

n4m_role_pipeline_t* make(std::vector<const char*> ids,
                          std::vector<const n4m_params_t*> params = {}) {
    n4m_role_pipeline_t* p = nullptr;
    CHECK(create(ids, params, &p) == N4M_OK);
    return p;
}

n4m_fit_inputs_v1_t inputs(n4m_matrix_view_t* X, n4m_matrix_view_t* Y) {
    n4m_fit_inputs_v1_t in{};
    in.struct_size = sizeof(in);
    in.X = X;
    in.Y = Y;
    return in;
}

// Fits one estimator by hand.
n4m_estimator_t* fit_one(const char* id, const n4m_params_t* params,
                         const n4m_fit_inputs_v1_t& in) {
    n4m_estimator_t* est = nullptr;
    CHECK(n4m_estimator_create(ctx_, id, params, &est) == N4M_OK);
    CHECK(n4m_estimator_fit(ctx_, est, &in) == N4M_OK);
    return est;
}

std::vector<double> transform_by(n4m_estimator_t* est, n4m_matrix_view_t X) {
    int64_t cols = 0;
    CHECK(n4m_estimator_transform_cols(est, &cols) == N4M_OK);
    std::vector<double> out(static_cast<size_t>(X.rows * cols));
    auto O = view(out, X.rows, cols);
    CHECK(n4m_estimator_transform(ctx_, est, &X, &O) == N4M_OK);
    return out;
}

std::vector<double> predict(const n4m_role_pipeline_t* p, n4m_matrix_view_t X) {
    int64_t q = 0;
    CHECK(n4m_role_pipeline_n_outputs(p, &q) == N4M_OK);
    std::vector<double> out(static_cast<size_t>(X.rows * q));
    auto O = view(out, X.rows, q);
    CHECK(n4m_role_pipeline_predict(ctx_, p, &X, &O) == N4M_OK);
    return out;
}

std::vector<unsigned char> export_state(const n4m_role_pipeline_t* p, int32_t state,
                                        uint32_t flags = 0) {
    size_t size = 0;
    CHECK(n4m_role_pipeline_export_state_size(ctx_, p, state, flags, &size) == N4M_OK);
    std::vector<unsigned char> bytes(size);
    size_t written = 0;
    CHECK(n4m_role_pipeline_export_state_to_buffer(ctx_, p, state, flags, bytes.data(), size,
                                                   &written) == N4M_OK);
    CHECK(written == size);
    return bytes;
}

n4m_status_t import(n4m_role_pipeline_t* p, const std::vector<std::vector<unsigned char>>& states) {
    std::vector<const void*> buffers;
    std::vector<size_t> sizes;
    for (const auto& s : states) {
        buffers.push_back(s.data());
        sizes.push_back(s.size());
    }
    return n4m_role_pipeline_import_states(ctx_, p, static_cast<int32_t>(states.size()),
                                           buffers.data(), sizes.data());
}

void test_recipe_refusals() {
    current_ = "recipe refusals";
    n4m_role_pipeline_t* p = nullptr;
    expect_error(create({}, {}, &p), N4M_ERR_INVALID_ARGUMENT, "at least one step");
    expect_error(create({"preprocessing.scatter.snv"}, {}, &p), N4M_ERR_INVALID_ARGUMENT,
                 "ends with one regressor or classifier");
    expect_error(create({"preprocessing.scatter.snv", "filters.y_outlier", "models.regularized.ridge"}, {}, &p),
                 N4M_ERR_INVALID_ARGUMENT, "step 1 (filters.y_outlier): sample filters come before");
    expect_error(create({"models.regularized.ridge", "models.regularized.ridge"}, {}, &p),
                 N4M_ERR_INVALID_ARGUMENT, "step 0 (models.regularized.ridge): only the last step");
    expect_error(create({"models.nope", "models.regularized.ridge"}, {}, &p),
                 N4M_ERR_INVALID_ARGUMENT, "unknown method 'models.nope'");
    expect_error(create({"splitters.kennard_stone", "models.regularized.ridge"}, {}, &p),
                 N4M_ERR_INVALID_ARGUMENT, "procedure");
    n4m_params_t* foreign = params_for("models.pls.cppls");
    expect_error(create({"preprocessing.scatter.snv", "models.regularized.ridge"}, {foreign, nullptr}, &p),
                 N4M_ERR_INVALID_ARGUMENT, "another method");
    n4m_params_destroy(foreign);
    CHECK(p == nullptr);

    // Role introspection: PLS plays transformer inside and regressor last.
    p = make({"filters.y_outlier", "models.pls.pls_regression", "models.pls.pls_regression"});
    int32_t n = 0;
    CHECK(n4m_role_pipeline_n_steps(p, &n) == N4M_OK && n == 3);
    CHECK(n4m_role_pipeline_n_states(p, &n) == N4M_OK && n == 2);
    const uint32_t roles[] = {N4M_ROLE_SAMPLE_FILTER, N4M_ROLE_TRANSFORMER, N4M_ROLE_REGRESSOR};
    const int32_t states[] = {-1, 0, 1};
    for (int32_t s = 0; s < 3; ++s) {
        n4m_role_pipeline_step_info_v1_t info{};
        info.struct_size = sizeof(info);
        CHECK(n4m_role_pipeline_step_info_v1(p, s, &info) == N4M_OK);
        CHECK(info.role == roles[s] && info.state_index == states[s]);
        CHECK(info.n_features_in == 0);
    }
    int32_t fitted = 1;
    CHECK(n4m_role_pipeline_is_fitted(p, &fitted) == N4M_OK && fitted == 0);
    std::vector<double> x(4);
    auto X = view(x, 2, 2);
    std::vector<double> o(2);
    auto O = view(o, 2, 1);
    CHECK(n4m_role_pipeline_predict(ctx_, p, &X, &O) == N4M_ERR_NOT_FITTED);
    n4m_role_pipeline_destroy(p);
}

// Filter (on y) -> SNV -> PLS scores -> ridge, against the same chain by hand.
void test_regression_chain(Data& d) {
    current_ = "regression chain";
    n4m_params_t* pls = params_for("models.pls.pls_regression");
    CHECK(n4m_params_set_int(pls, "n_components", 3) == N4M_OK);
    const char* ids[] = {"filters.y_outlier", "preprocessing.scatter.snv",
                         "models.pls.pls_regression", "models.regularized.ridge"};
    n4m_role_pipeline_t* p = make({ids[0], ids[1], ids[2], ids[3]}, {nullptr, nullptr, pls, nullptr});
    auto X = view(d.x, kRows, kCols);
    auto Y = view(d.y, kRows, 1);
    auto in = inputs(&X, &Y);
    CHECK(n4m_role_pipeline_fit(ctx_, p, &in) == N4M_OK);

    // By hand.
    n4m_estimator_t* filter = fit_one(ids[0], nullptr, in);
    std::vector<uint8_t> mask(kRows);
    CHECK(n4m_estimator_apply_mask(ctx_, filter, &X, &Y, mask.data(), kRows) == N4M_OK);
    CHECK(mask[7] == 0);
    std::vector<double> xs, ys;
    for (int64_t i = 0; i < kRows; ++i) {
        if (mask[static_cast<size_t>(i)] == 0) continue;
        xs.insert(xs.end(), d.x.begin() + i * kCols, d.x.begin() + (i + 1) * kCols);
        ys.push_back(d.y[static_cast<size_t>(i)]);
    }
    const auto n = static_cast<int64_t>(ys.size());
    auto Xs = view(xs, n, kCols);
    auto Ys = view(ys, n, 1);
    n4m_estimator_t* snv = fit_one(ids[1], nullptr, inputs(&Xs, nullptr));
    auto x1 = transform_by(snv, Xs);
    auto X1 = view(x1, n, kCols);
    n4m_estimator_t* scores = fit_one(ids[2], pls, inputs(&X1, &Ys));
    auto x2 = transform_by(scores, X1);
    auto X2 = view(x2, n, 3);
    n4m_estimator_t* ridge = fit_one(ids[3], nullptr, inputs(&X2, &Ys));

    auto t1 = transform_by(snv, X);
    auto T1 = view(t1, kRows, kCols);
    auto t2 = transform_by(scores, T1);
    auto T2 = view(t2, kRows, 3);
    std::vector<double> expected(kRows);
    auto E = view(expected, kRows, 1);
    CHECK(n4m_estimator_predict(ctx_, ridge, &T2, &E) == N4M_OK);
    CHECK(predict(p, X) == expected);

    // transform() = the terminal step's input.
    int64_t cols = 0;
    CHECK(n4m_role_pipeline_transform_cols(p, &cols) == N4M_OK && cols == 3);
    std::vector<double> tr(static_cast<size_t>(kRows * 3));
    auto TR = view(tr, kRows, 3);
    CHECK(n4m_role_pipeline_transform(ctx_, p, &X, &TR) == N4M_OK);
    CHECK(tr == t2);

    // Column-major input gives the same predictions (stride-aware).
    std::vector<double> xcm(d.x.size());
    for (int64_t i = 0; i < kRows; ++i)
        for (int64_t j = 0; j < kCols; ++j) xcm[static_cast<size_t>(j * kRows + i)] = d.x[static_cast<size_t>(i * kCols + j)];
    n4m_matrix_view_t Xcm{};
    CHECK(n4m_matrix_view_init_colmajor(&Xcm, xcm.data(), kRows, kCols, N4M_DTYPE_F64) == N4M_OK);
    CHECK(predict(p, Xcm) == expected);

    // Export / import round trip predicts bitwise identically.
    int32_t n_states = 0;
    CHECK(n4m_role_pipeline_n_states(p, &n_states) == N4M_OK && n_states == 3);
    std::vector<std::vector<unsigned char>> states;
    for (int32_t s = 0; s < n_states; ++s) states.push_back(export_state(p, s));
    n4m_role_pipeline_t* q = make({ids[0], ids[1], ids[2], ids[3]}, {nullptr, nullptr, pls, nullptr});
    CHECK(import(q, states) == N4M_OK);
    CHECK(predict(q, X) == expected);
    for (int32_t s = 0; s < n_states; ++s) CHECK(export_state(q, s) == states[static_cast<size_t>(s)]);

    // Import refusals.
    current_ = "import refusals";
    n4m_role_pipeline_t* r = make({ids[0], ids[1], ids[2], ids[3]}, {nullptr, nullptr, pls, nullptr});
    expect_error(import(r, {states[0], states[1]}), N4M_ERR_INVALID_ARGUMENT,
                 "3 stateful steps but 2 states");
    expect_error(import(r, {states[1], states[0], states[2]}), N4M_ERR_INVALID_ARGUMENT,
                 "state 0 (step 1 (preprocessing.scatter.snv)): the state was fitted by 'models.pls.pls_regression'");
    n4m_role_pipeline_destroy(r);
    // Recipe says 2 PLS components, the state holds 3.
    n4m_params_t* pls2 = params_for("models.pls.pls_regression");
    CHECK(n4m_params_set_int(pls2, "n_components", 2) == N4M_OK);
    r = make({ids[0], ids[1], ids[2], ids[3]}, {nullptr, nullptr, pls2, nullptr});
    expect_error(import(r, states), N4M_ERR_INVALID_ARGUMENT,
                 "parameter 'n_components' is 3 in the state but 2 in the recipe");
    int32_t fitted = 1;
    CHECK(n4m_role_pipeline_is_fitted(r, &fitted) == N4M_OK && fitted == 0);
    n4m_role_pipeline_destroy(r);
    n4m_params_destroy(pls2);
    // Width chain: ridge fitted on 3 scores cannot follow SNV directly.
    r = make({ids[1], ids[3]});
    expect_error(import(r, {states[0], states[2]}), N4M_ERR_SHAPE_MISMATCH,
                 "the state reads 3 columns but the previous step outputs 10");
    n4m_role_pipeline_destroy(r);

    for (auto* e : {filter, snv, scores, ridge}) n4m_estimator_destroy(e);
    n4m_role_pipeline_destroy(q);
    n4m_role_pipeline_destroy(p);
    n4m_params_destroy(pls);
}

// PLS(2) -> PLS(2) on a two-column Y: the supervised transformer sees both targets.
void test_multi_target(Data& d) {
    current_ = "multi-target";
    n4m_params_t* pls = params_for("models.pls.pls_regression");
    CHECK(n4m_params_set_int(pls, "n_components", 2) == N4M_OK);
    const char* id = "models.pls.pls_regression";
    n4m_role_pipeline_t* p = make({id, id}, {pls, pls});
    auto X = view(d.x, kRows, kCols);
    auto Y = view(d.y2, kRows, 2);
    auto in = inputs(&X, &Y);
    CHECK(n4m_role_pipeline_fit(ctx_, p, &in) == N4M_OK);
    int64_t q = 0;
    CHECK(n4m_role_pipeline_n_outputs(p, &q) == N4M_OK && q == 2);

    n4m_estimator_t* first = fit_one(id, pls, in);
    auto x1 = transform_by(first, X);
    auto X1 = view(x1, kRows, 2);
    n4m_estimator_t* last = fit_one(id, pls, inputs(&X1, &Y));
    std::vector<double> expected(static_cast<size_t>(kRows * 2));
    auto E = view(expected, kRows, 2);
    CHECK(n4m_estimator_predict(ctx_, last, &X1, &E) == N4M_OK);
    CHECK(predict(p, X) == expected);
    n4m_estimator_destroy(first);
    n4m_estimator_destroy(last);
    n4m_role_pipeline_destroy(p);
    n4m_params_destroy(pls);
}

// SNV -> PLS scores (fitted on the class ids) -> PLS-LDA on the labels.
void test_classification(Data& d) {
    current_ = "classification";
    const char* ids[] = {"preprocessing.scatter.snv", "models.pls.pls_regression",
                         "models.classification.pls_lda"};
    n4m_role_pipeline_t* p = make({ids[0], ids[1], ids[2]});
    auto X = view(d.x, kRows, kCols);
    n4m_fit_inputs_v1_t in = inputs(&X, nullptr);
    in.labels = d.labels.data();
    in.n_labels = kRows;
    CHECK(n4m_role_pipeline_fit(ctx_, p, &in) == N4M_OK);

    n4m_estimator_t* snv = fit_one(ids[0], nullptr, inputs(&X, nullptr));
    auto x1 = transform_by(snv, X);
    auto X1 = view(x1, kRows, kCols);
    std::vector<double> ids_as_y(d.labels.begin(), d.labels.end());
    auto Yl = view(ids_as_y, kRows, 1);
    n4m_estimator_t* pls = fit_one(ids[1], nullptr, inputs(&X1, &Yl));
    auto x2 = transform_by(pls, X1);
    int64_t k = 0;
    CHECK(n4m_estimator_transform_cols(pls, &k) == N4M_OK);
    auto X2 = view(x2, kRows, k);
    n4m_fit_inputs_v1_t lin = inputs(&X2, nullptr);
    lin.labels = d.labels.data();
    lin.n_labels = kRows;
    n4m_estimator_t* lda = fit_one(ids[2], nullptr, lin);
    std::vector<int64_t> expected(kRows), got(kRows);
    CHECK(n4m_estimator_predict_labels(ctx_, lda, &X2, expected.data(), kRows) == N4M_OK);
    CHECK(n4m_role_pipeline_predict_labels(ctx_, p, &X, got.data(), kRows) == N4M_OK);
    CHECK(got == expected);
    int64_t count = 0;
    CHECK(n4m_role_pipeline_classes(p, nullptr, 0, &count) == N4M_OK && count == 3);
    // A classifier has no regression predict.
    std::vector<double> o(kRows * 3);
    auto O = view(o, kRows, 3);
    expect_error(n4m_role_pipeline_predict(ctx_, p, &X, &O), N4M_ERR_UNSUPPORTED,
                 "step 2 (models.classification.pls_lda)");
    for (auto* e : {snv, pls, lda}) n4m_estimator_destroy(e);
    n4m_role_pipeline_destroy(p);
}

void test_input_routing(Data& d) {
    current_ = "input routing";
    auto X = view(d.x, kRows, kCols);
    auto Y = view(d.y2, kRows, 1);
    n4m_role_pipeline_t* p = make({"preprocessing.scatter.snv", "models.regularized.ridge"});
    std::vector<int64_t> groups(kRows, 1);
    n4m_fit_inputs_v1_t in = inputs(&X, &Y);
    in.groups = groups.data();
    in.n_groups = kRows;
    expect_error(n4m_role_pipeline_fit(ctx_, p, &in), N4M_ERR_INVALID_ARGUMENT,
                 "not used by any step of the pipeline 'groups'");
    in = inputs(&X, nullptr);
    expect_error(n4m_role_pipeline_fit(ctx_, p, &in), N4M_ERR_INVALID_ARGUMENT,
                 "step 1 (models.regularized.ridge): missing required fit input 'y'");
    in = inputs(&X, &Y);
    std::vector<double> y_short(kRows - 1);
    auto Ys = view(y_short, kRows - 1, 1);
    in.Y = &Ys;
    expect_error(n4m_role_pipeline_fit(ctx_, p, &in), N4M_ERR_SHAPE_MISMATCH, "'y'");
    n4m_role_pipeline_destroy(p);
    // A column input cannot reach a step after the columns changed.
    std::vector<double> axis(kCols);
    for (int64_t j = 0; j < kCols; ++j) axis[static_cast<size_t>(j)] = 1000.0 + 2.0 * static_cast<double>(j);
    p = make({"preprocessing.scatter.snv", "preprocessing.resampling.resampler", "models.regularized.ridge"});
    in = inputs(&X, &Y);
    in.axis = axis.data();
    in.n_axis = kCols;
    expect_error(n4m_role_pipeline_fit(ctx_, p, &in), N4M_ERR_INVALID_ARGUMENT,
                 "step 1 (preprocessing.resampling.resampler): fit input 'axis' describes");
    n4m_role_pipeline_destroy(p);
}

void test_feature_names(Data& d) {
    current_ = "feature names";
    n4m_role_pipeline_t* p = make({"preprocessing.scatter.snv", "models.regularized.ridge"});
    std::vector<std::string> names;
    for (int64_t j = 0; j < kCols; ++j) names.push_back("nm" + std::to_string(1000 + 2 * j));
    std::vector<const char*> ptrs;
    for (auto& s : names) ptrs.push_back(s.c_str());
    std::vector<const char*> dup = ptrs;
    dup[3] = dup[1];
    expect_error(n4m_role_pipeline_set_feature_names(ctx_, p, dup.data(), kCols),
                 N4M_ERR_INVALID_ARGUMENT, "duplicate feature name 'nm1002'");
    const char* bad_utf8[] = {"\xC3\x28"};
    expect_error(n4m_role_pipeline_set_feature_names(ctx_, p, bad_utf8, 1),
                 N4M_ERR_INVALID_ARGUMENT, "not valid UTF-8");
    CHECK(n4m_role_pipeline_set_feature_names(ctx_, p, ptrs.data(), kCols - 1) == N4M_OK);
    auto X = view(d.x, kRows, kCols);
    auto Y = view(d.y2, kRows, 1);
    auto in = inputs(&X, &Y);
    expect_error(n4m_role_pipeline_fit(ctx_, p, &in), N4M_ERR_SHAPE_MISMATCH,
                 "9 feature names for 10 input columns");
    CHECK(n4m_role_pipeline_set_feature_names(ctx_, p, ptrs.data(), kCols) == N4M_OK);
    CHECK(n4m_role_pipeline_fit(ctx_, p, &in) == N4M_OK);
    expect_error(n4m_role_pipeline_set_feature_names(ctx_, p, ptrs.data(), kCols),
                 N4M_ERR_INVALID_ARGUMENT, "before fit or import");
    int64_t n = 0;
    const char* name = nullptr;
    CHECK(n4m_role_pipeline_n_feature_names(p, &n) == N4M_OK && n == kCols);
    CHECK(n4m_role_pipeline_feature_name(p, 4, &name) == N4M_OK && std::strcmp(name, "nm1008") == 0);

    CHECK(n4m_role_pipeline_check_features(ctx_, p, kCols, ptrs.data()) == N4M_OK);
    CHECK(n4m_role_pipeline_check_features(ctx_, p, kCols, nullptr) == N4M_OK);
    std::vector<const char*> swapped = ptrs;
    std::swap(swapped[2], swapped[5]);
    expect_error(n4m_role_pipeline_check_features(ctx_, p, kCols, swapped.data()),
                 N4M_ERR_INVALID_ARGUMENT, "input column 2 is 'nm1010', fitted at column 5");
    std::vector<const char*> renamed = ptrs;
    renamed[0] = "other";
    expect_error(n4m_role_pipeline_check_features(ctx_, p, kCols, renamed.data()),
                 N4M_ERR_INVALID_ARGUMENT, "fitted with 'nm1000' there");
    expect_error(n4m_role_pipeline_check_features(ctx_, p, kCols - 1, ptrs.data()),
                 N4M_ERR_SHAPE_MISMATCH, "9 columns; the pipeline was fitted on 10");
    std::vector<double> narrow(static_cast<size_t>(kRows * (kCols - 1)));
    auto Xn = view(narrow, kRows, kCols - 1);
    std::vector<double> o(kRows);
    auto O = view(o, kRows, 1);
    expect_error(n4m_role_pipeline_predict(ctx_, p, &Xn, &O), N4M_ERR_SHAPE_MISMATCH,
                 "X has 9 columns");
    n4m_role_pipeline_destroy(p);
}

void test_training_rows(Data& d) {
    current_ = "training rows";
    n4m_role_pipeline_t* p = make({"preprocessing.scatter.snv", "models.pls.kernel"});
    auto X = view(d.x, kRows, kCols);
    auto Y = view(d.y2, kRows, 1);
    auto in = inputs(&X, &Y);
    CHECK(n4m_role_pipeline_fit(ctx_, p, &in) == N4M_OK);
    n4m_role_pipeline_step_info_v1_t info{};
    info.struct_size = sizeof(info);
    CHECK(n4m_role_pipeline_step_info_v1(p, 1, &info) == N4M_OK);
    CHECK(info.contains_training_rows == 1 && info.state_index == 1 && info.n_features_out == 1);
    CHECK(n4m_role_pipeline_step_info_v1(p, 0, &info) == N4M_OK);
    CHECK(info.contains_training_rows == 0 && info.n_features_in == kCols);
    size_t size = 0;
    expect_error(n4m_role_pipeline_export_state_size(ctx_, p, 1, 0, &size),
                 N4M_ERR_INVALID_ARGUMENT,
                 "state 1 (step 1 (models.pls.kernel)): state retains training rows");
    CHECK(!export_state(p, 1, N4M_EXPORT_ALLOW_TRAINING_ROWS).empty());
    n4m_role_pipeline_destroy(p);
}

}  // namespace

int main() {
    if (n4m_context_create(&ctx_) != N4M_OK) return 1;
    int failures = 0;
    auto run = [&](auto&& fn) {
        try {
            fn();
        } catch (const std::exception& e) {
            std::fprintf(stderr, "FAIL %s (%s)\n", e.what(), n4m_context_last_error(ctx_));
            ++failures;
        }
    };
    Data d;
    run([&] { test_recipe_refusals(); });
    run([&] { test_regression_chain(d); });
    run([&] { test_multi_target(d); });
    run([&] { test_classification(d); });
    run([&] { test_input_routing(d); });
    run([&] { test_feature_names(d); });
    run([&] { test_training_rows(d); });
    n4m_context_destroy(ctx_);
    std::printf("n4m_role_pipeline_tests: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
