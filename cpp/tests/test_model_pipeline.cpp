// SPDX-License-Identifier: CECILL-2.1
//
// Focused contract for model-owned SNV -> Savitzky-Golay -> PLS and N4MM v2.

#include "n4m/n4m.h"
#include "n4m/n4m_version.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "harness.hpp"

namespace {

constexpr std::int64_t kRows = 10;
constexpr std::int64_t kFeatures = 9;

std::uint64_t fnv1a64(const std::vector<unsigned char>& bytes,
                      std::size_t length) {
    std::uint64_t hash = 14695981039346656037ull;
    for (std::size_t i = 0; i < length; ++i) {
        hash ^= static_cast<std::uint64_t>(bytes[i]);
        hash *= 1099511628211ull;
    }
    return hash;
}

void write_u32(std::vector<unsigned char>& bytes,
               std::size_t offset,
               std::uint32_t value) {
    for (std::size_t i = 0; i < 4U; ++i) {
        bytes[offset + i] =
            static_cast<unsigned char>((value >> (8U * i)) & 0xffU);
    }
}

void write_u64(std::vector<unsigned char>& bytes,
               std::size_t offset,
               std::uint64_t value) {
    for (std::size_t i = 0; i < 8U; ++i) {
        bytes[offset + i] =
            static_cast<unsigned char>((value >> (8U * i)) & 0xffU);
    }
}

void write_f64(std::vector<unsigned char>& bytes,
               std::size_t offset,
               double value) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    write_u64(bytes, offset, bits);
}

void refresh_checksum(std::vector<unsigned char>& bytes) {
    const std::size_t offset = bytes.size() - sizeof(std::uint64_t);
    write_u64(bytes, offset, fnv1a64(bytes, offset));
}

n4m_matrix_view_t view(double* values,
                       std::int64_t rows,
                       std::int64_t cols) {
    n4m_matrix_view_t out{};
    N4M_TEST_REQUIRE(n4m_matrix_view_init_rowmajor(
                         &out, values, rows, cols, N4M_DTYPE_F64) == N4M_OK);
    return out;
}

std::vector<double> array_values(const n4m_array_t* array) {
    n4m_matrix_view_t values{};
    N4M_TEST_REQUIRE(n4m_array_view(array, &values) == N4M_OK);
    const auto* begin = static_cast<const double*>(values.data);
    return std::vector<double>(
        begin, begin + static_cast<std::size_t>(values.rows * values.cols));
}

void require_close(const std::vector<double>& left,
                   const std::vector<double>& right) {
    N4M_TEST_REQUIRE(left.size() == right.size());
    for (std::size_t i = 0; i < left.size(); ++i) {
        const double scale = std::max({1.0, std::fabs(left[i]), std::fabs(right[i])});
        N4M_TEST_REQUIRE(std::fabs(left[i] - right[i]) <= 1e-12 * scale);
    }
}

n4m_pipeline_t* make_pipeline(n4m_operator_kind_t first = N4M_OP_SNV,
                              n4m_operator_kind_t second = N4M_OP_SAVGOL_SMOOTH,
                              const double* second_params = nullptr,
                              std::int32_t second_param_count = 0) {
    n4m_pipeline_t* pipeline = nullptr;
    N4M_TEST_REQUIRE(n4m_pipeline_create(&pipeline) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_pipeline_add_operator(
                         pipeline, first, nullptr, 0) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_pipeline_add_operator(
                         pipeline, second, second_params,
                         second_param_count) == N4M_OK);
    return pipeline;
}

std::vector<unsigned char> export_model(n4m_model_t* model) {
    std::size_t size = 0;
    N4M_TEST_REQUIRE(n4m_model_export_size(model, &size) == N4M_OK);
    std::vector<unsigned char> bytes(size, 0U);
    std::size_t written = 0;
    N4M_TEST_REQUIRE(n4m_model_export_to_buffer(
                         model, bytes.data(), bytes.size(), &written) == N4M_OK);
    N4M_TEST_REQUIRE(written == bytes.size());
    return bytes;
}

void test_pipeline_model_round_trip_and_parity() {
    n4m_context_t* ctx = nullptr;
    n4m_config_t* pipeline_config = nullptr;
    n4m_config_t* plain_config = nullptr;
    N4M_TEST_REQUIRE(n4m_context_create(&ctx) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_create(&pipeline_config) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_create(&plain_config) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_n_components(pipeline_config, 1) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_n_components(plain_config, 1) == N4M_OK);

    std::vector<double> x(static_cast<std::size_t>(kRows * kFeatures), 0.0);
    std::vector<double> y(static_cast<std::size_t>(kRows), 0.0);
    for (std::int64_t row = 0; row < kRows; ++row) {
        for (std::int64_t col = 0; col < kFeatures; ++col) {
            const double r = static_cast<double>(row + 1);
            const double c = static_cast<double>(col + 1);
            x[static_cast<std::size_t>(row * kFeatures + col)] =
                0.11 * r * c + 0.07 * r * r / (c + 1.0) +
                0.013 * c * c * c + 0.003 * r * c * c;
        }
        const double r = static_cast<double>(row + 1);
        y[static_cast<std::size_t>(row)] = 0.4 * r + 0.09 * r * r;
    }
    n4m_matrix_view_t x_view = view(x.data(), kRows, kFeatures);
    n4m_matrix_view_t y_view = view(y.data(), kRows, 1);

    const double sg_params[2] = {5.0, 2.0};
    n4m_pipeline_t* pipeline = make_pipeline(
        N4M_OP_SNV, N4M_OP_SAVGOL_SMOOTH, sg_params, 2);
    N4M_TEST_REQUIRE(n4m_config_set_pipeline(
                         pipeline_config, pipeline) == N4M_OK);

    n4m_model_t* pipeline_model = nullptr;
    N4M_TEST_REQUIRE(n4m_model_fit(
                         ctx, pipeline_config, &x_view, &y_view,
                         &pipeline_model) == N4M_OK);
    N4M_TEST_REQUIRE(pipeline_model != nullptr);

    std::int32_t features = 0;
    std::int32_t targets = 0;
    N4M_TEST_REQUIRE(n4m_model_get_n_features(
                         pipeline_model, &features) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_model_get_n_targets(
                         pipeline_model, &targets) == N4M_OK);
    N4M_TEST_REQUIRE(features == kFeatures);
    N4M_TEST_REQUIRE(targets == 1);

    n4m_array_t* pipeline_predictions = nullptr;
    N4M_TEST_REQUIRE(n4m_model_predict_alloc(
                         ctx, pipeline_model, &x_view,
                         &pipeline_predictions) == N4M_OK);
    const std::vector<double> predicted = array_values(pipeline_predictions);

    // Manual composition is the parity oracle: the same public pipeline
    // followed by an ordinary PLS model must produce identical predictions.
    N4M_TEST_REQUIRE(n4m_pipeline_fit(ctx, pipeline, &x_view, &y_view) == N4M_OK);
    n4m_array_t* transformed = nullptr;
    N4M_TEST_REQUIRE(n4m_pipeline_transform_alloc(
                         ctx, pipeline, &x_view, &transformed) == N4M_OK);
    n4m_matrix_view_t transformed_view{};
    N4M_TEST_REQUIRE(n4m_array_view(transformed, &transformed_view) == N4M_OK);
    n4m_model_t* plain_model = nullptr;
    N4M_TEST_REQUIRE(n4m_model_fit(
                         ctx, plain_config, &transformed_view, &y_view,
                         &plain_model) == N4M_OK);
    n4m_array_t* plain_predictions = nullptr;
    N4M_TEST_REQUIRE(n4m_model_predict_alloc(
                         ctx, plain_model, &transformed_view,
                         &plain_predictions) == N4M_OK);
    require_close(predicted, array_values(plain_predictions));

    const std::vector<unsigned char> plain_bytes = export_model(plain_model);
    std::uint32_t plain_format = 0;
    std::uint32_t major = 0;
    std::uint32_t minor = 0;
    std::uint32_t patch = 0;
    N4M_TEST_REQUIRE(n4m_serialization_inspect(
                         plain_bytes.data(), plain_bytes.size(), &plain_format,
                         &major, &minor, &patch) == N4M_OK);
    N4M_TEST_REQUIRE(plain_format == N4M_SERIALIZATION_FORMAT_VERSION_V1);

    const std::vector<unsigned char> bytes = export_model(pipeline_model);
    n4m_serialized_model_info_v1_t info{};
    N4M_TEST_REQUIRE(n4m_serialization_inspect_model_v1(
                         bytes.data(), bytes.size(), &info) == N4M_OK);
    N4M_TEST_REQUIRE(info.format_version == N4M_SERIALIZATION_FORMAT_VERSION_V2);
    N4M_TEST_REQUIRE(info.algorithm == N4M_ALGO_PLS_REGRESSION);
    N4M_TEST_REQUIRE(info.n_features == kFeatures);
    N4M_TEST_REQUIRE(info.capabilities ==
                     (N4M_SERIALIZED_MODEL_CAPABILITY_PREDICT |
                      N4M_SERIALIZED_MODEL_CAPABILITY_TRANSFORM |
                      N4M_SERIALIZED_MODEL_CAPABILITY_PIPELINE));

    n4m_model_t* restored = nullptr;
    N4M_TEST_REQUIRE(n4m_model_import_from_buffer(
                         ctx, bytes.data(), bytes.size(), &restored) == N4M_OK);
    n4m_array_t* restored_predictions = nullptr;
    N4M_TEST_REQUIRE(n4m_model_predict_alloc(
                         ctx, restored, &x_view,
                         &restored_predictions) == N4M_OK);
    require_close(predicted, array_values(restored_predictions));

    // The v2 extension is fixed-width and immediately precedes the checksum.
    constexpr std::size_t kExtensionBytes = 5U * 4U + 4U * 8U;
    const std::size_t extension = bytes.size() - 8U - kExtensionBytes;

    std::vector<unsigned char> bad_checksum = bytes;
    bad_checksum[extension + 20U] ^= 0x01U;
    std::memset(&info, 0xa5, sizeof(info));
    N4M_TEST_REQUIRE(n4m_serialization_inspect_model_v1(
                         bad_checksum.data(), bad_checksum.size(), &info) ==
                     N4M_ERR_CORRUPT_BUFFER);
    N4M_TEST_REQUIRE(info.schema_version == 0U);

    std::vector<unsigned char> bad_tag = bytes;
    write_u32(bad_tag, extension + 4U, static_cast<std::uint32_t>(N4M_OP_MSC));
    refresh_checksum(bad_tag);
    N4M_TEST_REQUIRE(n4m_serialization_inspect_model_v1(
                         bad_tag.data(), bad_tag.size(), &info) ==
                     N4M_ERR_CORRUPT_BUFFER);
    n4m_model_t* rejected = reinterpret_cast<n4m_model_t*>(0x1);
    N4M_TEST_REQUIRE(n4m_model_import_from_buffer(
                         ctx, bad_tag.data(), bad_tag.size(), &rejected) ==
                     N4M_ERR_CORRUPT_BUFFER);
    N4M_TEST_REQUIRE(rejected == nullptr);

    std::vector<unsigned char> bad_window = bytes;
    write_f64(bad_window, extension + 20U, 4.0);
    refresh_checksum(bad_window);
    N4M_TEST_REQUIRE(n4m_serialization_inspect_model_v1(
                         bad_window.data(), bad_window.size(), &info) ==
                     N4M_ERR_CORRUPT_BUFFER);

    std::vector<unsigned char> bad_dimensions = bytes;
    write_u64(bad_dimensions, 40U, 8U);
    refresh_checksum(bad_dimensions);
    N4M_TEST_REQUIRE(n4m_serialization_inspect_model_v1(
                         bad_dimensions.data(), bad_dimensions.size(), &info) ==
                     N4M_ERR_CORRUPT_BUFFER);

    n4m_array_free(restored_predictions);
    n4m_model_destroy(restored);
    n4m_array_free(plain_predictions);
    n4m_model_destroy(plain_model);
    n4m_array_free(transformed);
    n4m_array_free(pipeline_predictions);
    n4m_model_destroy(pipeline_model);
    n4m_pipeline_destroy(pipeline);
    n4m_config_destroy(plain_config);
    n4m_config_destroy(pipeline_config);
    n4m_context_destroy(ctx);
}

void test_other_pipeline_combinations_are_rejected() {
    n4m_context_t* ctx = nullptr;
    n4m_config_t* config = nullptr;
    N4M_TEST_REQUIRE(n4m_context_create(&ctx) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_create(&config) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_n_components(config, 1) == N4M_OK);

    std::vector<double> x(static_cast<std::size_t>(kRows * kFeatures), 0.0);
    std::vector<double> y(static_cast<std::size_t>(kRows), 0.0);
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = 0.1 + static_cast<double>(i % 17U) +
               0.01 * static_cast<double>(i * i);
    }
    for (std::size_t i = 0; i < y.size(); ++i) {
        y[i] = static_cast<double>(i + 1U);
    }
    n4m_matrix_view_t x_view = view(x.data(), kRows, kFeatures);
    n4m_matrix_view_t y_view = view(y.data(), kRows, 1);
    n4m_model_t* model = nullptr;

    n4m_pipeline_t* reversed = make_pipeline(
        N4M_OP_SAVGOL_SMOOTH, N4M_OP_SNV);
    N4M_TEST_REQUIRE(n4m_config_set_pipeline(config, reversed) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_model_fit(
                         ctx, config, &x_view, &y_view, &model) ==
                     N4M_ERR_UNSUPPORTED);
    N4M_TEST_REQUIRE(model == nullptr);
    n4m_pipeline_destroy(reversed);

    n4m_pipeline_t* derivative = make_pipeline(
        N4M_OP_SNV, N4M_OP_SAVGOL_DERIVATIVE);
    N4M_TEST_REQUIRE(n4m_config_set_pipeline(config, derivative) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_model_fit(
                         ctx, config, &x_view, &y_view, &model) ==
                     N4M_ERR_UNSUPPORTED);
    n4m_pipeline_destroy(derivative);

    const double invalid_sg[2] = {4.0, 2.0};
    n4m_pipeline_t* invalid = make_pipeline(
        N4M_OP_SNV, N4M_OP_SAVGOL_SMOOTH, invalid_sg, 2);
    N4M_TEST_REQUIRE(n4m_config_set_pipeline(config, invalid) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_model_fit(
                         ctx, config, &x_view, &y_view, &model) ==
                     N4M_ERR_INVALID_ARGUMENT);
    n4m_pipeline_destroy(invalid);

    const double valid_sg[2] = {5.0, 2.0};
    n4m_pipeline_t* valid = make_pipeline(
        N4M_OP_SNV, N4M_OP_SAVGOL_SMOOTH, valid_sg, 2);
    N4M_TEST_REQUIRE(n4m_config_set_pipeline(config, valid) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_set_algorithm(config, N4M_ALGO_PLS_DA) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_model_fit(
                         ctx, config, &x_view, &y_view, &model) ==
                     N4M_ERR_UNSUPPORTED);
    n4m_pipeline_destroy(valid);

    n4m_config_destroy(config);
    n4m_context_destroy(ctx);
}

}  // namespace

int main() {
    n4m_testing::Runner runner("model_pipeline");
    runner.run("fit_roundtrip_parity_tamper", test_pipeline_model_round_trip_and_parity);
    runner.run("reject_other_combinations", test_other_pipeline_combinations_are_rejected);
    return runner.finalize();
}
