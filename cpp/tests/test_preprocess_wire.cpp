// SPDX-License-Identifier: CECILL-2.1
#include "n4m/n4m.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "harness.hpp"

namespace {

constexpr std::int64_t kTrainRows = 30;
constexpr std::int64_t kFeatures = 24;
constexpr std::int64_t kHeldRows = 3;

n4m_matrix_view_t view(double* values, std::int64_t rows, std::int64_t cols) {
    n4m_matrix_view_t result{};
    N4M_TEST_REQUIRE(n4m_matrix_view_init_rowmajor(
        &result, values, rows, cols, N4M_DTYPE_F64) == N4M_OK);
    return result;
}

std::vector<double> spectra(const std::vector<double>& samples) {
    std::vector<double> out;
    for (double sample : samples) {
        for (int feature = 1; feature <= kFeatures; ++feature) {
            const double x = static_cast<double>(feature);
            out.push_back(std::sin(sample * x / 31.0) +
                          std::cos(sample + x / 11.0) + sample * x / 1000.0);
        }
    }
    return out;
}

std::uint64_t fnv1a64(const std::vector<unsigned char>& bytes, std::size_t length) {
    std::uint64_t value = UINT64_C(14695981039346656037);
    for (std::size_t i = 0; i < length; ++i) {
        value ^= static_cast<std::uint64_t>(bytes[i]);
        value *= UINT64_C(1099511628211);
    }
    return value;
}

void put_u32(std::vector<unsigned char>& bytes, std::size_t offset,
             std::uint32_t value) {
    for (std::size_t i = 0; i < 4U; ++i)
        bytes[offset + i] = static_cast<unsigned char>((value >> (8U * i)) & 0xffU);
}

void put_u64(std::vector<unsigned char>& bytes, std::size_t offset,
             std::uint64_t value) {
    for (std::size_t i = 0; i < 8U; ++i)
        bytes[offset + i] = static_cast<unsigned char>((value >> (8U * i)) & 0xffU);
}

void recalc_checksum(std::vector<unsigned char>& bytes) {
    put_u64(bytes, bytes.size() - 8U, fnv1a64(bytes, bytes.size() - 8U));
}

std::vector<unsigned char> export_bytes(const n4m_pipeline_t* pipeline) {
    std::size_t size = 0;
    N4M_TEST_REQUIRE(n4m_pipeline_export_size(pipeline, &size) == N4M_OK);
    N4M_TEST_REQUIRE(size > 40U);
    std::vector<unsigned char> bytes(size);
    std::size_t written = 0;
    N4M_TEST_REQUIRE(n4m_pipeline_export_to_buffer(
        pipeline, bytes.data(), bytes.size(), &written) == N4M_OK);
    N4M_TEST_REQUIRE(written == bytes.size());
    return bytes;
}

void run_roundtrip(const std::vector<n4m_operator_kind_t>& kinds) {
    n4m_context_t* context = nullptr;
    n4m_pipeline_t* source = nullptr;
    N4M_TEST_REQUIRE(n4m_context_create(&context) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_pipeline_create(&source) == N4M_OK);
    for (auto kind : kinds)
        N4M_TEST_REQUIRE(n4m_pipeline_add_operator(source, kind, nullptr, 0) == N4M_OK);
    std::vector<double> sample_ids;
    for (int i = 1; i <= kTrainRows; ++i) sample_ids.push_back(static_cast<double>(i));
    auto train = spectra(sample_ids);
    auto held = spectra({2.5, 11.5, 31.5});
    std::vector<double> target;
    for (int i = 1; i <= kTrainRows; ++i) target.push_back(static_cast<double>(i) / 30.0);
    auto x = view(train.data(), kTrainRows, kFeatures);
    auto y = view(target.data(), kTrainRows, 1);
    auto h = view(held.data(), kHeldRows, kFeatures);
    N4M_TEST_REQUIRE(n4m_pipeline_fit(context, source, &x, &y) == N4M_OK);
    const auto bytes = export_bytes(source);
    N4M_TEST_REQUIRE(bytes[0] == 'N' && bytes[1] == '4' &&
                     bytes[2] == 'M' && bytes[3] == 'P');
    N4M_TEST_REQUIRE(export_bytes(source) == bytes);
    n4m_pipeline_t* restored = nullptr;
    N4M_TEST_REQUIRE(n4m_pipeline_import_from_buffer(
        context, bytes.data(), bytes.size(), &restored) == N4M_OK);
    N4M_TEST_REQUIRE(restored != nullptr);
    std::int64_t width = 0;
    std::int32_t count = 0;
    N4M_TEST_REQUIRE(n4m_pipeline_get_info(restored, &width, &count) == N4M_OK);
    N4M_TEST_REQUIRE(width == kFeatures && count == static_cast<std::int32_t>(kinds.size()));
    for (std::int32_t i = 0; i < count; ++i) {
        n4m_operator_kind_t recovered = N4M_OP_GAUSSIAN;
        std::int32_t n_params = -1;
        N4M_TEST_REQUIRE(n4m_pipeline_get_operator(
            restored, i, &recovered, nullptr, 0, &n_params) == N4M_OK);
        N4M_TEST_REQUIRE(recovered == kinds[static_cast<std::size_t>(i)] && n_params == 0);
    }
    N4M_TEST_REQUIRE(export_bytes(restored) == bytes);
    std::vector<double> expected(held.size());
    std::vector<double> actual(held.size());
    auto expected_view = view(expected.data(), kHeldRows, kFeatures);
    auto actual_view = view(actual.data(), kHeldRows, kFeatures);
    N4M_TEST_REQUIRE(n4m_pipeline_transform(context, source, &h, &expected_view) == N4M_OK);
    n4m_pipeline_destroy(source);
    source = nullptr;
    N4M_TEST_REQUIRE(n4m_pipeline_transform(context, restored, &h, &actual_view) == N4M_OK);
    for (std::size_t i = 0; i < actual.size(); ++i)
        N4M_TEST_REQUIRE(std::isfinite(actual[i]) && std::fabs(actual[i] - expected[i]) < 1e-11);
    n4m_pipeline_destroy(restored);
    n4m_context_destroy(context);
}

void test_all_kinds_and_mixed() {
    for (int kind = N4M_OP_IDENTITY; kind <= N4M_OP_WAVELET_DENOISE; ++kind)
        run_roundtrip({static_cast<n4m_operator_kind_t>(kind)});
    run_roundtrip({N4M_OP_SNV, N4M_OP_MSC, N4M_OP_CENTER});
    run_roundtrip({N4M_OP_SNV, N4M_OP_OSC, N4M_OP_EPO});
}

void test_unfitted_and_corrupt() {
    n4m_context_t* context = nullptr;
    n4m_pipeline_t* source = nullptr;
    N4M_TEST_REQUIRE(n4m_context_create(&context) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_pipeline_create(&source) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_pipeline_add_operator(source, N4M_OP_CENTER, nullptr, 0) == N4M_OK);
    std::size_t size = 99;
    N4M_TEST_REQUIRE(n4m_pipeline_export_size(source, &size) == N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(size == 0);
    std::int64_t width = 99;
    std::int32_t count = 99;
    N4M_TEST_REQUIRE(n4m_pipeline_get_info(source, &width, &count) == N4M_ERR_NOT_FITTED);
    N4M_TEST_REQUIRE(width == 0 && count == 0);
    n4m_operator_kind_t op_kind = N4M_OP_GAUSSIAN;
    N4M_TEST_REQUIRE(n4m_pipeline_get_operator(
        source, 0, &op_kind, nullptr, 0, &count) == N4M_ERR_NOT_FITTED);
    N4M_TEST_REQUIRE(op_kind == N4M_OP_IDENTITY && count == 0);
    auto train = spectra({1, 2, 3, 4, 5});
    auto x = view(train.data(), 5, kFeatures);
    N4M_TEST_REQUIRE(n4m_pipeline_fit(context, source, &x, nullptr) == N4M_OK);
    const auto valid = export_bytes(source);
    N4M_TEST_REQUIRE(n4m_pipeline_get_operator(
        source, -1, &op_kind, nullptr, 0, &count) == N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(n4m_pipeline_get_operator(
        source, 1, &op_kind, nullptr, 0, &count) == N4M_ERR_INVALID_ARGUMENT);
    std::size_t written = 99;
    std::vector<unsigned char> short_buffer(valid.size() - 1U);
    N4M_TEST_REQUIRE(n4m_pipeline_export_to_buffer(
        source, short_buffer.data(), short_buffer.size(), &written) == N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(written == 0);
    auto reject = [&](std::vector<unsigned char> bytes, n4m_status_t expected) {
        n4m_pipeline_t* imported = source;
        N4M_TEST_REQUIRE(n4m_pipeline_import_from_buffer(
            context, bytes.data(), bytes.size(), &imported) == expected);
        N4M_TEST_REQUIRE(imported == nullptr);
    };
    auto corrupt = valid;
    corrupt[0] = 'X';
    reject(corrupt, N4M_ERR_CORRUPT_BUFFER);
    corrupt = valid;
    corrupt[36] ^= 1U;  // state vector data; checksum unchanged
    reject(corrupt, N4M_ERR_CORRUPT_BUFFER);
    corrupt = valid;
    put_u32(corrupt, 4U, 2U);  // unknown format, valid checksum
    recalc_checksum(corrupt);
    reject(corrupt, N4M_ERR_VERSION_INCOMPATIBLE);
    corrupt = valid;
    put_u32(corrupt, 8U, 99U);  // incompatible ABI major
    recalc_checksum(corrupt);
    reject(corrupt, N4M_ERR_VERSION_INCOMPATIBLE);
    corrupt = valid;
    put_u32(corrupt, 32U, N4M_OP_FINITE_DIFFERENCE);  // unsupported kind
    recalc_checksum(corrupt);
    reject(corrupt, N4M_ERR_CORRUPT_BUFFER);
    corrupt = valid;
    put_u64(corrupt, 20U, 1000001U);  // feature limit
    recalc_checksum(corrupt);
    reject(corrupt, N4M_ERR_CORRUPT_BUFFER);
    corrupt = valid;
    put_u32(corrupt, 32U, 0xffffffffU);  // unsupported kind
    recalc_checksum(corrupt);
    reject(corrupt, N4M_ERR_CORRUPT_BUFFER);
    corrupt = valid;
    put_u32(corrupt, 36U, 1U);  // center must have no recipe params
    recalc_checksum(corrupt);
    reject(corrupt, N4M_ERR_CORRUPT_BUFFER);
    corrupt = valid;
    put_u64(corrupt, 44U, UINT64_C(0x7ff8000000000000));  // NaN learned mean
    recalc_checksum(corrupt);
    reject(corrupt, N4M_ERR_CORRUPT_BUFFER);
    corrupt = valid;
    put_u32(corrupt, 40U, 23U);  // invalid learned-state width
    recalc_checksum(corrupt);
    reject(corrupt, N4M_ERR_CORRUPT_BUFFER);
    corrupt = valid;
    put_u32(corrupt, 28U, 257U);  // operator count limit
    recalc_checksum(corrupt);
    reject(corrupt, N4M_ERR_CORRUPT_BUFFER);
    corrupt = valid;
    corrupt.pop_back();
    reject(corrupt, N4M_ERR_CORRUPT_BUFFER);
    n4m_pipeline_destroy(source);
    n4m_context_destroy(context);
}

void test_original_parameter_plan() {
    n4m_context_t* context = nullptr;
    n4m_pipeline_t* source = nullptr;
    N4M_TEST_REQUIRE(n4m_context_create(&context) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_pipeline_create(&source) == N4M_OK);
    const double degree = 2.0;
    N4M_TEST_REQUIRE(n4m_pipeline_add_operator(source, N4M_OP_DETREND_POLY,
                                               &degree, 1) == N4M_OK);
    auto train = spectra({1, 2, 3, 4, 5});
    auto x = view(train.data(), 5, kFeatures);
    N4M_TEST_REQUIRE(n4m_pipeline_fit(context, source, &x, nullptr) == N4M_OK);
    auto bytes = export_bytes(source);
    n4m_pipeline_t* restored = nullptr;
    N4M_TEST_REQUIRE(n4m_pipeline_import_from_buffer(
        context, bytes.data(), bytes.size(), &restored) == N4M_OK);
    n4m_operator_kind_t kind = N4M_OP_IDENTITY;
    std::int32_t count = 0;
    N4M_TEST_REQUIRE(n4m_pipeline_get_operator(
        restored, 0, &kind, nullptr, 0, &count) == N4M_OK);
    N4M_TEST_REQUIRE(kind == N4M_OP_DETREND_POLY && count == 1);
    double parameter = -1.0;
    N4M_TEST_REQUIRE(n4m_pipeline_get_operator(
        restored, 0, &kind, &parameter, 0, &count) == N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(count == 1 && parameter == -1.0);
    N4M_TEST_REQUIRE(n4m_pipeline_get_operator(
        restored, 0, &kind, &parameter, 1, &count) == N4M_OK);
    N4M_TEST_REQUIRE(count == 1 && parameter == degree);
    n4m_pipeline_destroy(restored);
    n4m_pipeline_destroy(source);
    n4m_context_destroy(context);
}

}  // namespace

int main() {
    n4m_testing::Runner runner("preprocess_wire");
    runner.run("all_kinds_and_mixed", test_all_kinds_and_mixed);
    runner.run("unfitted_and_corrupt", test_unfitted_and_corrupt);
    runner.run("original_parameter_plan", test_original_parameter_plan);
    return runner.finalize();
}
