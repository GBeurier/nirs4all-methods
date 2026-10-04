// SPDX-License-Identifier: CECILL-2.1
#include "n4m/multimodal.h"
#include "n4m/estimator.h"
#include "core/estimator/multimodal_pipeline.hpp"
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
#define REQUIRE(value) do { if (!(value)) throw std::runtime_error(#value); } while (false)
struct Context {
    n4m_context_t* value = nullptr;
    Context() { REQUIRE(n4m_context_create(&value) == N4M_OK); }
    ~Context() { n4m_context_destroy(value); }
};
using Owner = std::unique_ptr<n4m_multimodal_classifier_t, decltype(&n4m_multimodal_classifier_destroy)>;
struct Fixture {
    static constexpr int64_t rows = 12;
    std::array<std::string, 4> names{{"nir", "image", "series", "metadata"}};
    std::array<std::string, 4> identities{{"nir:axis-µ", "image:axes-rgb", "series:time-value", "metadata:measurement-category"}};
    const char* representations[4] = {"signal_1d", "rgb_image", "series_mv", "tabular_mixed"};
    std::array<std::vector<int64_t>, 4> shapes{{{4}, {2, 2, 2}, {3, 2}, {2}}};
    std::array<std::vector<int64_t>, 4> raw_shapes, strides;
    std::array<std::vector<double>, 4> numeric;
    std::string categories;
    std::vector<uint64_t> offsets{0};
    std::vector<double> y;
    std::array<n4m_multimodal_source_spec_v1_t, 4> specs{};
    std::array<n4m_multimodal_source_view_v1_t, 4> views{};
    n4m_multimodal_recipe_v1_t recipe{};
    Fixture() {
        for (int i = 0; i < 4; ++i) {
            int64_t width = i == 3 ? 1 : 1;
            if (i != 3) for (auto d : shapes[i]) width *= d;
            numeric[i].resize(rows * width);
            for (int64_t r = 0; r < rows; ++r)
                for (int64_t c = 0; c < width; ++c)
                    numeric[i][r * width + c] = std::sin(0.37 * (r + 1) * (c + 1)) + 0.03 * r * c;
            auto& s = specs[i]; s.struct_size = sizeof(s); s.name = names[i].c_str();
            s.representation_id = representations[i]; s.dtype = i == 3 ? "<U32" : "float64";
            s.identity_utf8 = identities[i].data(); s.identity_bytes = identities[i].size();
            s.ndim = static_cast<int32_t>(shapes[i].size()); s.shape = shapes[i].data();
            s.weight = i == 1 ? 0.5 : 1; s.numeric_column = s.categorical_column = -1;
            s.encoder = i == 0 ? N4M_MULTIMODAL_STANDARD_SCALER : i == 3 ? N4M_MULTIMODAL_COLUMN_TRANSFORMER : N4M_MULTIMODAL_TENSOR_PCA;
            if (i == 1 || i == 2) { s.n_components = 2; s.random_state = 77; }
            else { s.with_mean = 1; s.with_std = 1; }
            if (i == 3) { s.numeric_column = 0; s.categorical_column = 1; s.ignore_unknown = 1; }
            raw_shapes[i] = {rows}; raw_shapes[i].insert(raw_shapes[i].end(), shapes[i].begin(), shapes[i].end());
            strides[i].resize(raw_shapes[i].size()); int64_t stride = 1;
            for (size_t d = raw_shapes[i].size(); d-- > 0;) { strides[i][d] = stride; stride *= raw_shapes[i][d]; }
            if (i == 3) strides[i][0] = 1;
            auto& v = views[i]; v.struct_size = sizeof(v); v.name = names[i].c_str(); v.representation_id = representations[i];
            v.dtype = s.dtype; v.identity_utf8 = identities[i].data(); v.identity_bytes = identities[i].size();
            v.rank = static_cast<int32_t>(raw_shapes[i].size()); v.shape = raw_shapes[i].data(); v.strides = strides[i].data();
            v.numeric_data = numeric[i].data(); v.numeric_dtype = N4M_DTYPE_F64;
        }
        for (int64_t r = 0; r < rows; ++r) {
            const std::string label = r % 2 ? "é" : "猫";
            categories += label; offsets.push_back(categories.size()); y.push_back(0.4 * r + std::cos(0.2 * r));
        }
        views[3].categorical_utf8 = categories.data(); views[3].utf8_bytes = categories.size(); views[3].categorical_offsets = offsets.data();
        recipe.struct_size = sizeof(recipe); recipe.n_sources = 4; recipe.sources = specs.data(); recipe.alpha = 0.2;
        recipe.center_x = recipe.center_y = 1;
    }
    n4m_matrix_view_t targets() {
        n4m_matrix_view_t result{}; REQUIRE(n4m_matrix_view_init_rowmajor(&result, y.data(), rows, 1, N4M_DTYPE_F64) == N4M_OK); return result;
    }
};
struct Recipe {
    n4m_params_t* params = nullptr;
    n4m_multimodal_classifier_recipe_v1_t value{};
    Recipe(Context& ctx, Fixture& f) {
        int32_t index = -1;
        REQUIRE(n4m_method_find("models.classification.pls_logistic", &index) == N4M_OK);
        REQUIRE(n4m_params_create(ctx.value, index, &params) == N4M_OK);
        REQUIRE(n4m_params_set_int(params, "n_components", 2) == N4M_OK);
        REQUIRE(n4m_params_set_int(params, "max_iter", 1000) == N4M_OK);
        value.struct_size = sizeof(value); value.n_sources = 4; value.sources = f.specs.data();
        value.method_id = "models.classification.pls_logistic"; value.params = params;
    }
    ~Recipe() { n4m_params_destroy(params); }
};
Owner create(Context& ctx, const Recipe& r) {
    n4m_multimodal_classifier_t* p = nullptr;
    REQUIRE(n4m_multimodal_classifier_create(ctx.value, &r.value, &p) == N4M_OK);
    return Owner(p, n4m_multimodal_classifier_destroy);
}
std::vector<unsigned char> state(Context& ctx, const Owner& p) {
    size_t size = 0, written = 0;
    REQUIRE(n4m_multimodal_classifier_export_size(ctx.value, p.get(), &size) == N4M_OK);
    std::vector<unsigned char> result(size);
    REQUIRE(n4m_multimodal_classifier_export_to_buffer(ctx.value, p.get(), result.data(), size, &written) == N4M_OK);
    REQUIRE(written == size); return result;
}
std::vector<int64_t> labels(Context& ctx, const Owner& p, Fixture& f) {
    std::vector<int64_t> result(Fixture::rows);
    REQUIRE(n4m_multimodal_classifier_predict_labels(ctx.value, p.get(), 4, f.views.data(), result.data(), Fixture::rows) == N4M_OK);
    return result;
}
std::vector<double> probabilities(Context& ctx, const Owner& p, Fixture& f, int64_t classes) {
    std::vector<double> result(Fixture::rows * classes);
    n4m_matrix_view_t out{};
    REQUIRE(n4m_matrix_view_init_rowmajor(&out, result.data(), Fixture::rows, classes, N4M_DTYPE_F64) == N4M_OK);
    REQUIRE(n4m_multimodal_classifier_predict_proba(ctx.value, p.get(), 4, f.views.data(), &out) == N4M_OK);
    for (int64_t row = 0; row < Fixture::rows; ++row) {
        double sum = 0;
        for (int64_t column = 0; column < classes; ++column) {
            double value = result[row * classes + column];
            REQUIRE(std::isfinite(value) && value >= 0 && value <= 1); sum += value;
        }
        REQUIRE(std::abs(sum - 1) < 1e-12);
    }
    return result;
}
uint64_t read64(const std::vector<unsigned char>& bytes, size_t position) {
    REQUIRE(position + 8 <= bytes.size()); uint64_t value = 0;
    for (size_t i = 0; i < 8; ++i) value |= uint64_t(bytes[position + i]) << (8 * i);
    return value;
}
void put64(std::vector<unsigned char>& bytes, size_t position, uint64_t value) {
    REQUIRE(position + 8 <= bytes.size());
    for (size_t i = 0; i < 8; ++i) bytes[position + i] = static_cast<unsigned char>(value >> (8 * i));
}
void reseal(std::vector<unsigned char>& bytes) {
    uint64_t value = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < bytes.size() - 8; ++i) { value ^= bytes[i]; value *= 0x100000001b3ULL; }
    put64(bytes, bytes.size() - 8, value);
}
size_t class_table(const std::vector<unsigned char>& bytes) {
    size_t cursor = 20;
    cursor += 8 + read64(bytes, cursor);
    for (int branch = 0; branch < 4; ++branch) {
        cursor += 8 + read64(bytes, cursor);
        const auto categories = read64(bytes, cursor); cursor += 8;
        for (uint64_t i = 0; i < categories; ++i) cursor += 8 + read64(bytes, cursor);
    }
    return cursor;
}
void binary_and_multiclass() {
    for (const int64_t count : {int64_t(2), int64_t(3)}) {
        Context ctx; Fixture f; Recipe r(ctx, f); auto p = create(ctx, r);
        std::vector<int64_t> y(Fixture::rows);
        const int64_t ids[] = {-19, 5, 90};
        for (int64_t i = 0; i < Fixture::rows; ++i) y[i] = ids[i % count];
        REQUIRE(n4m_multimodal_classifier_fit(ctx.value, p.get(), 4, f.views.data(), y.data(), Fixture::rows) == N4M_OK);
        int64_t size = 0;
        REQUIRE(n4m_multimodal_classifier_classes(p.get(), nullptr, 0, &size) == N4M_OK);
        REQUIRE(size == count);
        std::vector<int64_t> classes(count);
        REQUIRE(n4m_multimodal_classifier_classes(p.get(), classes.data(), count, &size) == N4M_OK);
        for (int64_t i = 0; i < count; ++i) REQUIRE(classes[i] == ids[i]);
        auto predicted = labels(ctx, p, f); auto proba = probabilities(ctx, p, f, count);
        // Independent public native estimator receives already transformed X.
        int64_t width = 0;
        REQUIRE(n4m_multimodal_classifier_transform_cols(p.get(), &width) == N4M_OK);
        std::vector<double> z(Fixture::rows * width);
        n4m_matrix_view_t x{};
        REQUIRE(n4m_matrix_view_init_rowmajor(&x, z.data(), Fixture::rows, width, N4M_DTYPE_F64) == N4M_OK);
        REQUIRE(n4m_multimodal_classifier_transform(ctx.value, p.get(), 4, f.views.data(), &x) == N4M_OK);
        n4m_estimator_t* raw = nullptr;
        REQUIRE(n4m_estimator_create(ctx.value, r.value.method_id, r.params, &raw) == N4M_OK);
        std::unique_ptr<n4m_estimator_t, decltype(&n4m_estimator_destroy)> head(raw, n4m_estimator_destroy);
        n4m_fit_inputs_v1_t in{}; in.struct_size = sizeof(in); in.X = &x; in.labels = y.data(); in.n_labels = Fixture::rows;
        REQUIRE(n4m_estimator_fit(ctx.value, head.get(), &in) == N4M_OK);
        std::vector<int64_t> expected(Fixture::rows);
        REQUIRE(n4m_estimator_predict_labels(ctx.value, head.get(), &x, expected.data(), Fixture::rows) == N4M_OK);
        REQUIRE(expected == predicted);
        std::vector<double> expected_proba(Fixture::rows * count); n4m_matrix_view_t out{};
        REQUIRE(n4m_matrix_view_init_rowmajor(&out, expected_proba.data(), Fixture::rows, count, N4M_DTYPE_F64) == N4M_OK);
        REQUIRE(n4m_estimator_predict_proba(ctx.value, head.get(), &x, &out) == N4M_OK);
        for (size_t i = 0; i < proba.size(); ++i) REQUIRE(std::abs(proba[i] - expected_proba[i]) < 1e-13);
        const auto bytes = state(ctx, p);
        REQUIRE(std::memcmp(bytes.data(), "N4MC", 4) == 0);
        n4m_multimodal_classifier_t* imported = nullptr;
        REQUIRE(n4m_multimodal_classifier_import_from_buffer(ctx.value, &r.value, bytes.data(), bytes.size(), &imported) == N4M_OK);
        Owner replay(imported, n4m_multimodal_classifier_destroy);
        REQUIRE(labels(ctx, replay, f) == predicted);
        REQUIRE(probabilities(ctx, replay, f, count) == proba);
        REQUIRE(state(ctx, replay) == bytes);
        // Failed refit preserves fitted encoders/head and sorted IDs.
        std::fill(y.begin(), y.end(), 5);
        REQUIRE(n4m_multimodal_classifier_fit(ctx.value, p.get(), 4, f.views.data(), y.data(), Fixture::rows) != N4M_OK);
        REQUIRE(state(ctx, p) == bytes);
        REQUIRE(labels(ctx, p, f) == predicted);
    }
}
void vocabulary_and_regression_conservation() {
    Context ctx; Fixture f; Recipe r(ctx, f); auto p = create(ctx, r);
    std::vector<int64_t> y(Fixture::rows);
    for (int64_t i = 0; i < Fixture::rows; ++i) y[i] = i % 2;
    REQUIRE(n4m_multimodal_classifier_fit(ctx.value, p.get(), 4, f.views.data(), y.data(), Fixture::rows) == N4M_OK);
    n4m_multimodal_pipeline_t* raw = nullptr;
    REQUIRE(n4m_multimodal_pipeline_create(ctx.value, &f.recipe, &raw) == N4M_OK);
    std::unique_ptr<n4m_multimodal_pipeline_t, decltype(&n4m_multimodal_pipeline_destroy)> reg(raw, n4m_multimodal_pipeline_destroy);
    auto targets = f.targets();
    REQUIRE(n4m_multimodal_pipeline_fit(ctx.value, reg.get(), 4, f.views.data(), &targets) == N4M_OK);
    int64_t width = 0; REQUIRE(n4m_multimodal_classifier_transform_cols(p.get(), &width) == N4M_OK);
    REQUIRE(width == 11);
    std::string unknown; std::vector<uint64_t> offsets{0};
    for (int64_t i = 0; i < Fixture::rows; ++i) { unknown += "🚀"; offsets.push_back(unknown.size()); }
    f.views[3].categorical_utf8 = unknown.data(); f.views[3].utf8_bytes = unknown.size(); f.views[3].categorical_offsets = offsets.data();
    std::vector<double> z(Fixture::rows * width), old(Fixture::rows * width); n4m_matrix_view_t a{}, b{};
    REQUIRE(n4m_matrix_view_init_rowmajor(&a, z.data(), Fixture::rows, width, N4M_DTYPE_F64) == N4M_OK);
    REQUIRE(n4m_matrix_view_init_rowmajor(&b, old.data(), Fixture::rows, width, N4M_DTYPE_F64) == N4M_OK);
    REQUIRE(n4m_multimodal_classifier_transform(ctx.value, p.get(), 4, f.views.data(), &a) == N4M_OK);
    REQUIRE(n4m_multimodal_pipeline_transform(ctx.value, reg.get(), 4, f.views.data(), &b) == N4M_OK);
    REQUIRE(z == old);
    for (int64_t i = 0; i < Fixture::rows; ++i) REQUIRE(z[i * width + width - 2] == 0 && z[i * width + width - 1] == 0);
}
void malformed_and_expected_recipe() {
    // The pure admission predicate checks boundary/overflow without allocating
    // a Hessian or fitting a numerical surrogate.
    REQUIRE(n4m::multimodal::classifier_working_set_fits(3072, 33, 127));
    REQUIRE(!n4m::multimodal::classifier_working_set_fits(3072, 33, 128));
    REQUIRE(n4m::multimodal::classifier_working_set_fits(8388608, 2, 1));
    REQUIRE(!n4m::multimodal::classifier_working_set_fits(6000000, 2, 2));
    REQUIRE(!n4m::multimodal::classifier_working_set_fits(
        std::numeric_limits<int64_t>::max(), 2, 1));
    REQUIRE(!n4m::multimodal::classifier_working_set_fits(
        2, std::numeric_limits<uint64_t>::max(), 1));
    REQUIRE(!n4m::multimodal::classifier_working_set_fits(
        2, 2, std::numeric_limits<int64_t>::max()));
    {
        // Actual valid physical NIR descriptor: 3072x128,512 classes,128
        // components. Previously admitted onehot requires only1572864 values,
        // but its (511*129)^2 dense Hessian must be refused before any encoder.
        Context wide_ctx; Fixture wide;
        constexpr int64_t rows = 3072, columns = 128, classes = 512;
        wide.shapes[0] = {columns}; wide.specs[0].shape = wide.shapes[0].data();
        wide.numeric[0].resize(static_cast<size_t>(rows * columns));
        for (size_t i = 0; i < wide.numeric[0].size(); ++i)
            wide.numeric[0][i] = std::sin(0.017 * static_cast<double>(i));
        wide.raw_shapes[0] = {rows, columns}; wide.strides[0] = {columns, 1};
        wide.views[0].shape = wide.raw_shapes[0].data();
        wide.views[0].strides = wide.strides[0].data();
        wide.views[0].numeric_data = wide.numeric[0].data();
        Recipe wide_recipe(wide_ctx, wide);
        wide_recipe.value.n_sources = 1;
        REQUIRE(n4m_params_set_int(wide_recipe.params, "n_components", columns) == N4M_OK);
        auto pipeline = create(wide_ctx, wide_recipe);
        std::vector<int64_t> labels(static_cast<size_t>(rows));
        for (int64_t i = 0; i < rows; ++i) labels[static_cast<size_t>(i)] = i % classes;
        REQUIRE(n4m_multimodal_classifier_fit(wide_ctx.value, pipeline.get(), 1,
            wide.views.data(), labels.data(), rows) == N4M_ERR_INVALID_ARGUMENT);
        REQUIRE(std::strstr(n4m_context_last_error(wide_ctx.value), "PLS-logistic working set") != nullptr);
        int64_t width = -1;
        REQUIRE(n4m_multimodal_classifier_transform_cols(pipeline.get(), &width) == N4M_ERR_NOT_FITTED);
    }
    Context ctx; Fixture f; Recipe r(ctx, f); auto p = create(ctx, r);
    std::vector<int64_t> y(Fixture::rows);
    for (int64_t i = 0; i < Fixture::rows; ++i) y[i] = i % 2 ? 42 : -7;
    REQUIRE(n4m_multimodal_classifier_fit(ctx.value, p.get(), 4, f.views.data(), y.data(), Fixture::rows) == N4M_OK);
    const auto bytes = state(ctx, p);
    std::vector<int64_t> oversized_classes(65536);
    for (size_t i = 0; i < oversized_classes.size(); ++i) oversized_classes[i] = static_cast<int64_t>(i);
    REQUIRE(n4m_multimodal_classifier_fit(ctx.value, p.get(), 4, f.views.data(),
        oversized_classes.data(), static_cast<int64_t>(oversized_classes.size())) == N4M_ERR_INVALID_ARGUMENT);
    REQUIRE(std::strstr(n4m_context_last_error(ctx.value), "element limit") != nullptr);
    REQUIRE(state(ctx, p) == bytes);
    auto reject = [&](const std::vector<unsigned char>& corrupt) {
        n4m_multimodal_classifier_t* imported = p.get();
        REQUIRE(n4m_multimodal_classifier_import_from_buffer(ctx.value, &r.value, corrupt.data(), corrupt.size(), &imported) != N4M_OK);
        REQUIRE(imported == nullptr);
        REQUIRE(state(ctx, p) == bytes);
    };
    auto bad = bytes; bad[0] = 'X'; reseal(bad); reject(bad);
    bad = bytes; bad[4] = 2; reseal(bad); reject(bad);
    bad = bytes; bad[12] = 16; reseal(bad); reject(bad);
    bad = bytes; bad.pop_back(); reject(bad);
    bad = bytes; bad.push_back(0); reseal(bad); reject(bad);
    // Outer IDs remain sorted, but contradict actual fitted classifier classes.
    bad = bytes; put64(bad, class_table(bad) + 8, static_cast<uint64_t>(int64_t(-8))); reseal(bad); reject(bad);
    // Sortedness and uniqueness are independently required even with a valid checksum.
    bad = bytes; put64(bad, class_table(bad) + 16, static_cast<uint64_t>(int64_t(-7))); reseal(bad); reject(bad);
    REQUIRE(n4m_params_set_int(r.params, "max_iter", 999) == N4M_OK); reject(bytes);
    REQUIRE(n4m_params_set_int(r.params, "max_iter", 1000) == N4M_OK);
    f.specs[0].weight = 0.25; reject(bytes);
}
}
int main() {
    try { binary_and_multiclass(); vocabulary_and_regression_conservation(); malformed_and_expected_recipe(); }
    catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
    return 0;
}
