// SPDX-License-Identifier: CECILL-2.1
#include "n4m/multimodal.h"
#include "n4m/estimator.h"
#include <array>
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
using Owner = std::unique_ptr<n4m_multimodal_pipeline_t, decltype(&n4m_multimodal_pipeline_destroy)>;
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
        for (size_t i = 0; i < 4; ++i) {
            int64_t width = i == 3 ? 1 : 1;
            if (i != 3) for (auto d : shapes[i]) width *= d;
            numeric[i].resize(static_cast<size_t>(rows * width));
            for (int64_t r = 0; r < rows; ++r)
                for (int64_t c = 0; c < width; ++c)
                    numeric[i][static_cast<size_t>(r * width + c)] = std::sin(0.37 * static_cast<double>(r + 1) * static_cast<double>(c + 1)) + 0.03 * static_cast<double>(r) * static_cast<double>(c);
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
            categories += label; offsets.push_back(categories.size()); y.push_back(0.4 * static_cast<double>(r) + std::cos(0.2 * static_cast<double>(r)));
        }
        views[3].categorical_utf8 = categories.data(); views[3].utf8_bytes = categories.size(); views[3].categorical_offsets = offsets.data();
        recipe.struct_size = sizeof(recipe); recipe.n_sources = 4; recipe.sources = specs.data(); recipe.alpha = 0.2;
        recipe.center_x = recipe.center_y = 1;
    }
    n4m_matrix_view_t targets() {
        n4m_matrix_view_t result{}; REQUIRE(n4m_matrix_view_init_rowmajor(&result, y.data(), rows, 1, N4M_DTYPE_F64) == N4M_OK); return result;
    }
};
Owner create(Context& ctx, Fixture& fixture) {
    n4m_multimodal_pipeline_t* p = nullptr;
    REQUIRE(n4m_multimodal_pipeline_create(ctx.value, &fixture.recipe, &p) == N4M_OK);
    return Owner(p, n4m_multimodal_pipeline_destroy);
}
std::vector<unsigned char> state(Context& ctx, n4m_multimodal_pipeline_t* p) {
    size_t size = 0, written = 0; REQUIRE(n4m_multimodal_pipeline_export_size(ctx.value, p, &size) == N4M_OK);
    std::vector<unsigned char> bytes(size);
    REQUIRE(n4m_multimodal_pipeline_export_to_buffer(ctx.value, p, bytes.data(), bytes.size(), &written) == N4M_OK);
    REQUIRE(written == bytes.size()); return bytes;
}
std::vector<double> predict(Context& ctx, n4m_multimodal_pipeline_t* p, Fixture& fixture) {
    std::vector<double> result(Fixture::rows); n4m_matrix_view_t out{};
    REQUIRE(n4m_matrix_view_init_rowmajor(&out, result.data(), Fixture::rows, 1, N4M_DTYPE_F64) == N4M_OK);
    REQUIRE(n4m_multimodal_pipeline_predict(ctx.value, p, 4, fixture.views.data(), &out) == N4M_OK); return result;
}
void ownership_and_transaction() {
    Context ctx; Fixture original, borrowed;
    auto p = create(ctx, borrowed);
    // Every descriptor is copied at create; mutate and release caller storage.
    borrowed.identities[0] = "caller-released"; borrowed.names[1] = "caller-name";
    borrowed.shapes[1][0] = 999; borrowed.specs[0].weight = 50; borrowed.recipe.alpha = 8;
    borrowed.numeric = {}; borrowed.categories.clear(); borrowed.offsets.clear();
    auto y = original.targets(); REQUIRE(n4m_multimodal_pipeline_fit(ctx.value, p.get(), 4, original.views.data(), &y) == N4M_OK);
    const auto before = predict(ctx, p.get(), original);
    const auto bytes = state(ctx, p.get());
    original.numeric[2][0] = std::numeric_limits<double>::quiet_NaN();
    REQUIRE(n4m_multimodal_pipeline_fit(ctx.value, p.get(), 4, original.views.data(), &y) != N4M_OK);
    original.numeric[2][0] = std::sin(0.37);
    REQUIRE(predict(ctx, p.get(), original) == before); REQUIRE(state(ctx, p.get()) == bytes);
    // Mutating source tensors after synchronous fit does not alter learned state.
    const auto saved = original.numeric[0]; original.numeric[0].assign(saved.size(), 100);
    REQUIRE(state(ctx, p.get()) == bytes); original.numeric[0] = saved;
    original.views[0].numeric_data = original.numeric[0].data(); REQUIRE(predict(ctx, p.get(), original) == before);
}
void vocabulary_and_import() {
    Context ctx; Fixture fixture; auto p = create(ctx, fixture); auto y = fixture.targets();
    REQUIRE(n4m_multimodal_pipeline_fit(ctx.value, p.get(), 4, fixture.views.data(), &y) == N4M_OK);
    int64_t width = 0; REQUIRE(n4m_multimodal_pipeline_transform_cols(p.get(), &width) == N4M_OK); REQUIRE(width == 11);
    std::vector<double> fused(static_cast<size_t>(Fixture::rows * width)); n4m_matrix_view_t out{};
    REQUIRE(n4m_matrix_view_init_rowmajor(&out, fused.data(), Fixture::rows, width, N4M_DTYPE_F64) == N4M_OK);
    std::string unknown; std::vector<uint64_t> offsets{0};
    for (int64_t r = 0; r < Fixture::rows; ++r) { unknown += "🚀"; offsets.push_back(unknown.size()); }
    fixture.views[3].categorical_utf8 = unknown.data(); fixture.views[3].utf8_bytes = unknown.size(); fixture.views[3].categorical_offsets = offsets.data();
    REQUIRE(n4m_multimodal_pipeline_transform(ctx.value, p.get(), 4, fixture.views.data(), &out) == N4M_OK);
    for (int64_t r = 0; r < Fixture::rows; ++r) { REQUIRE(fused[static_cast<size_t>(r * width + 9)] == 0); REQUIRE(fused[static_cast<size_t>(r * width + 10)] == 0); }
    const auto bytes = state(ctx, p.get()); n4m_multimodal_pipeline_t* raw = nullptr;
    REQUIRE(n4m_multimodal_pipeline_import_from_buffer(ctx.value, &fixture.recipe, bytes.data(), bytes.size(), &raw) == N4M_OK);
    Owner restored(raw, n4m_multimodal_pipeline_destroy); REQUIRE(predict(ctx, restored.get(), fixture) == predict(ctx, p.get(), fixture));
    auto corrupt = bytes; corrupt[40] ^= 1; raw = reinterpret_cast<n4m_multimodal_pipeline_t*>(1);
    REQUIRE(n4m_multimodal_pipeline_import_from_buffer(ctx.value, &fixture.recipe, corrupt.data(), corrupt.size(), &raw) == N4M_ERR_CORRUPT_BUFFER); REQUIRE(raw == nullptr);
    fixture.specs[1].weight = 0.75;
    REQUIRE(n4m_multimodal_pipeline_import_from_buffer(ctx.value, &fixture.recipe, bytes.data(), bytes.size(), &raw) == N4M_ERR_CORRUPT_BUFFER); REQUIRE(raw == nullptr);
    fixture.specs[1].weight = 0.5;
    fixture.views[0].identity_utf8 = "wrong"; fixture.views[0].identity_bytes = 5;
    REQUIRE(n4m_multimodal_pipeline_predict(ctx.value, p.get(), 4, fixture.views.data(), &out) != N4M_OK);
}
void zero_alpha_rank_deficiency_and_transaction() {
    for (int profile = 0; profile < 3; ++profile) {
        Context ctx; Fixture fixture;
        fixture.recipe.alpha = 0;
        for (auto& spec : fixture.specs) spec.weight = 0;
        if (profile != 1) fixture.specs[3].weight = 1;
        if (profile == 0) {
            // Two centered one-hot columns are dependent. The categorical
            // target is fit exactly; numeric metadata cannot explain it alone.
            for (int64_t row = 0; row < Fixture::rows; ++row)
                fixture.y[static_cast<size_t>(row)] = row % 2 ? 2.0 : -1.0;
        } else if (profile == 2) {
            // A single learned category is a constant feature after centering.
            fixture.categories.clear(); fixture.offsets = {0};
            for (int64_t row = 0; row < Fixture::rows; ++row) {
                fixture.categories += "é"; fixture.offsets.push_back(fixture.categories.size());
                fixture.y[static_cast<size_t>(row)] = 2.0 + 3.0 * fixture.numeric[3][static_cast<size_t>(row)];
            }
            fixture.views[3].categorical_utf8 = fixture.categories.data();
            fixture.views[3].utf8_bytes = fixture.categories.size();
            fixture.views[3].categorical_offsets = fixture.offsets.data();
        }
        auto p = create(ctx, fixture); auto y = fixture.targets();
        REQUIRE(n4m_multimodal_pipeline_fit(ctx.value, p.get(), 4, fixture.views.data(), &y) == N4M_OK);
        const auto before = predict(ctx, p.get(), fixture);
        double mean = 0; for (double value : fixture.y) mean += value / Fixture::rows;
        for (int64_t row = 0; row < Fixture::rows; ++row) {
            REQUIRE(std::isfinite(before[static_cast<size_t>(row)]));
            REQUIRE(std::abs(before[static_cast<size_t>(row)] - (profile == 1 ? mean : fixture.y[static_cast<size_t>(row)])) <= 1e-10);
        }
        const auto bytes = state(ctx, p.get()); n4m_multimodal_pipeline_t* raw = nullptr;
        REQUIRE(n4m_multimodal_pipeline_import_from_buffer(ctx.value, &fixture.recipe, bytes.data(), bytes.size(), &raw) == N4M_OK);
        Owner restored(raw, n4m_multimodal_pipeline_destroy);
        REQUIRE(predict(ctx, restored.get(), fixture) == before);
        REQUIRE(state(ctx, restored.get()) == bytes);
        const double first = fixture.y[0]; fixture.y[0] = std::numeric_limits<double>::quiet_NaN();
        REQUIRE(n4m_multimodal_pipeline_fit(ctx.value, p.get(), 4, fixture.views.data(), &y) != N4M_OK);
        fixture.y[0] = first;
        REQUIRE(predict(ctx, p.get(), fixture) == before); REQUIRE(state(ctx, p.get()) == bytes);
    }
}
void profile_and_output_bounds() {
    Context ctx; Fixture fixture; n4m_multimodal_pipeline_t* raw = nullptr;
    fixture.specs[1].random_state = -1;
    REQUIRE(n4m_multimodal_pipeline_create(ctx.value, &fixture.recipe, &raw) != N4M_OK); REQUIRE(raw == nullptr);
    fixture.specs[1].random_state = 77; auto p = create(ctx, fixture); auto y = fixture.targets();
    REQUIRE(n4m_multimodal_pipeline_fit(ctx.value, p.get(), 4, fixture.views.data(), &y) == N4M_OK);
    double sentinel = 91; n4m_matrix_view_t output{};
    REQUIRE(n4m_matrix_view_init_rowmajor(&output, &sentinel, Fixture::rows, 1, N4M_DTYPE_F64) == N4M_OK);
    output.row_stride = INT64_MAX;
    REQUIRE(n4m_multimodal_pipeline_predict(ctx.value, p.get(), 4, fixture.views.data(), &output) == N4M_ERR_STRIDE_INVALID); REQUIRE(sentinel == 91);
    fixture.strides[0][0] = INT64_MAX;
    REQUIRE(n4m_multimodal_pipeline_predict(ctx.value, p.get(), 4, fixture.views.data(), &output) != N4M_OK); REQUIRE(sentinel == 91);
}
std::vector<double> encoded(Context& ctx, n4m_multimodal_pipeline_t* p, int32_t count,
                            const n4m_multimodal_source_view_v1_t* views) {
    int64_t width = 0;
    REQUIRE(n4m_multimodal_pipeline_transform_cols(p, &width) == N4M_OK);
    const int64_t rows = views[0].shape[0];
    std::vector<double> result(static_cast<size_t>(rows * width)); n4m_matrix_view_t out{};
    REQUIRE(n4m_matrix_view_init_rowmajor(&out, result.data(), rows, width, N4M_DTYPE_F64) == N4M_OK);
    REQUIRE(n4m_multimodal_pipeline_transform(ctx.value, p, count, views, &out) == N4M_OK);
    return result;
}
void ordered_subsets_match_fixed_encoders_and_one_native_ridge() {
    Context ctx; Fixture full; auto y = full.targets(); auto complete = create(ctx, full);
    REQUIRE(n4m_multimodal_pipeline_fit(ctx.value, complete.get(), 4, full.views.data(), &y) == N4M_OK);
    const auto canonical = encoded(ctx, complete.get(), 4, full.views.data());
    const int64_t widths[] = {4, 2, 2, 3}, starts[] = {0, 4, 6, 8};
    for (const auto& selection : std::vector<std::vector<int>>{{0}, {3, 0}, {2, 1, 3}, {3, 2, 1, 0}}) {
        std::vector<n4m_multimodal_source_spec_v1_t> specs;
        std::vector<n4m_multimodal_source_view_v1_t> views;
        int64_t width = 0;
        for (size_t i = 0; i < selection.size(); ++i) {
            const size_t source = static_cast<size_t>(selection[i]); specs.push_back(full.specs[source]);
            specs.back().weight *= static_cast<double>(i + 1);
            views.push_back(full.views[source]); width += widths[source];
        }
        auto recipe = full.recipe; recipe.n_sources = static_cast<int32_t>(specs.size()); recipe.sources = specs.data();
        n4m_multimodal_pipeline_t* raw = nullptr;
        REQUIRE(n4m_multimodal_pipeline_create(ctx.value, &recipe, &raw) == N4M_OK);
        Owner selected(raw, n4m_multimodal_pipeline_destroy);
        REQUIRE(n4m_multimodal_pipeline_fit(ctx.value, selected.get(), recipe.n_sources, views.data(), &y) == N4M_OK);
        const auto actual = encoded(ctx, selected.get(), recipe.n_sources, views.data());
        std::vector<double> expected(static_cast<size_t>(Fixture::rows * width));
        for (int64_t row = 0; row < Fixture::rows; ++row) {
            int64_t column = 0;
            for (size_t i = 0; i < selection.size(); ++i) {
                const size_t source = static_cast<size_t>(selection[i]);
                for (int64_t j = 0; j < widths[source]; ++j)
                    expected[static_cast<size_t>(row * width + column++)] = canonical[static_cast<size_t>(row * 11 + starts[source] + j)] * static_cast<double>(i + 1);
            }
        }
        REQUIRE(actual == expected);
        // A separate fixed Ridge receives the independently assembled features.
        int32_t index = -1; REQUIRE(n4m_method_find("models.regularized.ridge", &index) == N4M_OK);
        n4m_params_t* params_raw = nullptr; REQUIRE(n4m_params_create(ctx.value, index, &params_raw) == N4M_OK);
        std::unique_ptr<n4m_params_t, decltype(&n4m_params_destroy)> params(params_raw, n4m_params_destroy);
        REQUIRE(n4m_params_set_double(params.get(), "alpha", recipe.alpha) == N4M_OK);
        REQUIRE(n4m_params_set_bool(params.get(), "center_x", 1) == N4M_OK);
        REQUIRE(n4m_params_set_bool(params.get(), "center_y", 1) == N4M_OK);
        REQUIRE(n4m_params_set_bool(params.get(), "scale_x", 0) == N4M_OK);
        n4m_estimator_t* estimator_raw = nullptr;
        REQUIRE(n4m_estimator_create(ctx.value, "models.regularized.ridge", params.get(), &estimator_raw) == N4M_OK);
        std::unique_ptr<n4m_estimator_t, decltype(&n4m_estimator_destroy)> ridge(estimator_raw, n4m_estimator_destroy);
        n4m_matrix_view_t x{}; REQUIRE(n4m_matrix_view_init_rowmajor(&x, expected.data(), Fixture::rows, width, N4M_DTYPE_F64) == N4M_OK);
        n4m_fit_inputs_v1_t inputs{}; inputs.struct_size = sizeof(inputs); inputs.X = &x; inputs.Y = &y;
        REQUIRE(n4m_estimator_fit(ctx.value, ridge.get(), &inputs) == N4M_OK);
        std::vector<double> reference(Fixture::rows), prediction(Fixture::rows); n4m_matrix_view_t out{};
        REQUIRE(n4m_matrix_view_init_rowmajor(&out, reference.data(), Fixture::rows, 1, N4M_DTYPE_F64) == N4M_OK);
        REQUIRE(n4m_estimator_predict(ctx.value, ridge.get(), &x, &out) == N4M_OK);
        REQUIRE(n4m_matrix_view_init_rowmajor(&out, prediction.data(), Fixture::rows, 1, N4M_DTYPE_F64) == N4M_OK);
        REQUIRE(n4m_multimodal_pipeline_predict(ctx.value, selected.get(), recipe.n_sources, views.data(), &out) == N4M_OK);
        REQUIRE(prediction == reference);
        const auto bytes = state(ctx, selected.get()); raw = nullptr;
        // The unchanged format stores exactly the selected count, not dormant branches.
        REQUIRE(bytes[36] == selection.size() && bytes[37] == 0 && bytes[38] == 0 && bytes[39] == 0);
        REQUIRE(n4m_multimodal_pipeline_import_from_buffer(ctx.value, &recipe, bytes.data(), bytes.size(), &raw) == N4M_OK);
        Owner restored(raw, n4m_multimodal_pipeline_destroy); REQUIRE(state(ctx, restored.get()) == bytes);
        REQUIRE(encoded(ctx, restored.get(), recipe.n_sources, views.data()) == actual);
        specs[0].weight += 0.25; raw = nullptr;
        REQUIRE(n4m_multimodal_pipeline_import_from_buffer(ctx.value, &recipe, bytes.data(), bytes.size(), &raw) == N4M_ERR_CORRUPT_BUFFER);
        REQUIRE(raw == nullptr); specs[0].weight -= 0.25;
        specs[0].identity_utf8 = "changed-schema"; specs[0].identity_bytes = 14;
        REQUIRE(n4m_multimodal_pipeline_import_from_buffer(ctx.value, &recipe, bytes.data(), bytes.size(), &raw) == N4M_ERR_CORRUPT_BUFFER);
        REQUIRE(raw == nullptr); specs[0] = full.specs[static_cast<size_t>(selection[0])];
        if (specs.size() > 1) {
            std::swap(specs[0], specs[1]);
            REQUIRE(n4m_multimodal_pipeline_import_from_buffer(ctx.value, &recipe, bytes.data(), bytes.size(), &raw) == N4M_ERR_CORRUPT_BUFFER);
            REQUIRE(raw == nullptr); std::swap(specs[0], specs[1]);
            std::swap(views[0], views[1]);
            REQUIRE(n4m_multimodal_pipeline_predict(ctx.value, selected.get(), recipe.n_sources, views.data(), &out) == N4M_ERR_SHAPE_MISMATCH);
        }
    }
}
void excluded_encoders_are_not_fitted_and_zero_weight_still_fits() {
    Context ctx; Fixture fixture;
    fixture.recipe.n_sources = 1; fixture.recipe.sources = fixture.specs.data();
    auto selected = create(ctx, fixture);
    fixture.raw_shapes[0][0] = 1; auto y = fixture.targets(); y.rows = 1;
    // Both omitted PCA recipes require two fit rows; the selected scaler accepts one.
    REQUIRE(n4m_multimodal_pipeline_fit(ctx.value, selected.get(), 1, fixture.views.data(), &y) == N4M_OK);
    int64_t width = 0; REQUIRE(n4m_multimodal_pipeline_transform_cols(selected.get(), &width) == N4M_OK);
    REQUIRE(width == 4);
    fixture.recipe.sources = &fixture.specs[1]; fixture.specs[1].weight = 0;
    auto weighted = create(ctx, fixture); fixture.raw_shapes[1][0] = 1;
    REQUIRE(n4m_multimodal_pipeline_fit(ctx.value, weighted.get(), 1, &fixture.views[1], &y) != N4M_OK);
}
void subsets_reject_invalid_count_duplicates_and_wrong_modality_encoder() {
    Context ctx; Fixture fixture; n4m_multimodal_pipeline_t* raw = nullptr;
    for (int count : {0, -1, 5}) {
        fixture.recipe.n_sources = count;
        REQUIRE(n4m_multimodal_pipeline_create(ctx.value, &fixture.recipe, &raw) == N4M_ERR_INVALID_ARGUMENT);
        REQUIRE(raw == nullptr);
    }
    fixture.recipe.n_sources = 2; fixture.specs[1] = fixture.specs[0];
    REQUIRE(n4m_multimodal_pipeline_create(ctx.value, &fixture.recipe, &raw) == N4M_ERR_INVALID_ARGUMENT);
    fixture.recipe.n_sources = 1; fixture.specs[0].name = "other";
    REQUIRE(n4m_multimodal_pipeline_create(ctx.value, &fixture.recipe, &raw) == N4M_ERR_INVALID_ARGUMENT);
    fixture.specs[0].name = "image";
    REQUIRE(n4m_multimodal_pipeline_create(ctx.value, &fixture.recipe, &raw) == N4M_ERR_INVALID_ARGUMENT);
    REQUIRE(raw == nullptr);
}
}
int main() {
    int failures = 0;
    for (auto test : {ownership_and_transaction, vocabulary_and_import, zero_alpha_rank_deficiency_and_transaction, profile_and_output_bounds,
                     ordered_subsets_match_fixed_encoders_and_one_native_ridge,
                     excluded_encoders_are_not_fitted_and_zero_weight_still_fits,
                     subsets_reject_invalid_count_duplicates_and_wrong_modality_encoder}) {
        try { test(); } catch (const std::exception& error) { std::fprintf(stderr, "%s\n", error.what()); ++failures; }
    }
    std::printf("native multimodal: %d failures\n", failures); return failures ? 1 : 0;
}
