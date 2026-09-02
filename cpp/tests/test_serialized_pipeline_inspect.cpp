// SPDX-License-Identifier: CECILL-2.1
//
// Focused ABI contract for allocation-free N4MM pipeline inspection.

#include "n4m/n4m.h"
#include "n4m/n4m_version.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "harness.hpp"

namespace {

void append_u32(std::vector<unsigned char>& bytes, std::uint32_t value) {
    for (std::size_t i = 0; i < 4U; ++i) {
        bytes.push_back(static_cast<unsigned char>((value >> (8U * i)) & 0xffU));
    }
}

void append_u64(std::vector<unsigned char>& bytes, std::uint64_t value) {
    for (std::size_t i = 0; i < 8U; ++i) {
        bytes.push_back(static_cast<unsigned char>((value >> (8U * i)) & 0xffU));
    }
}

void append_f64(std::vector<unsigned char>& bytes, double value) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    append_u64(bytes, bits);
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

std::uint64_t fnv1a64(const unsigned char* bytes, std::size_t size) {
    std::uint64_t hash = 14695981039346656037ull;
    for (std::size_t i = 0; i < size; ++i) {
        hash ^= static_cast<std::uint64_t>(bytes[i]);
        hash *= 1099511628211ull;
    }
    return hash;
}

void refresh_checksum(std::vector<unsigned char>& bytes) {
    const std::size_t offset = bytes.size() - sizeof(std::uint64_t);
    write_u64(bytes, offset, fnv1a64(bytes.data(), offset));
}

void append_vector(std::vector<unsigned char>& bytes,
                   std::initializer_list<double> values) {
    append_u64(bytes, static_cast<std::uint64_t>(values.size()));
    for (double value : values) {
        append_f64(bytes, value);
    }
}

std::vector<unsigned char> model_bytes(std::uint32_t format) {
    std::vector<unsigned char> bytes{'N', '4', 'M', 'M'};
    append_u32(bytes, format);
    append_u32(bytes, N4M_ABI_VERSION_MAJOR);
    append_u32(bytes, N4M_ABI_VERSION_MINOR);
    append_u32(bytes, N4M_ABI_VERSION_PATCH);
    append_u32(bytes, static_cast<std::uint32_t>(N4M_ALGO_PLS_REGRESSION));
    append_u32(bytes, static_cast<std::uint32_t>(N4M_SOLVER_NIPALS));
    append_u32(bytes, static_cast<std::uint32_t>(N4M_DEFLATION_REGRESSION));
    append_u64(bytes, 1U);  // samples
    append_u64(bytes, 9U);  // raw/model features
    append_u64(bytes, 1U);  // targets
    append_u64(bytes, 1U);  // components
    append_u32(bytes, 1U);  // center_x
    append_u32(bytes, 0U);  // scale_x
    append_u32(bytes, 1U);  // center_y
    append_u32(bytes, 0U);  // scale_y
    append_u32(bytes, 0U);  // store_scores
    append_f64(bytes, 1e-6);
    append_u32(bytes, 500U);

    const std::array<double, 9> feature_values = {
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (int vector = 0; vector < 2; ++vector) {
        append_u64(bytes, static_cast<std::uint64_t>(feature_values.size()));
        for (double value : feature_values) append_f64(bytes, value);
    }
    append_vector(bytes, {0.0});  // y mean
    append_vector(bytes, {1.0});  // y scale
    append_u64(bytes, static_cast<std::uint64_t>(feature_values.size()));  // coefficients
    for (double value : feature_values) append_f64(bytes, value);
    for (int vector = 0; vector < 2; ++vector) {
        append_u64(bytes, static_cast<std::uint64_t>(feature_values.size()));
        for (double value : feature_values) append_f64(bytes, value);
    }
    append_vector(bytes, {0.0});  // y loadings
    append_u64(bytes, static_cast<std::uint64_t>(feature_values.size()));  // rotations
    for (double value : feature_values) append_f64(bytes, value);
    append_vector(bytes, {});  // scores
    append_vector(bytes, {});  // y scores

    if (format == N4M_SERIALIZATION_FORMAT_VERSION_V2) {
        append_u32(bytes, 2U);
        append_u32(bytes, static_cast<std::uint32_t>(N4M_OP_SNV));
        append_u32(bytes, 0U);
        append_u32(bytes, static_cast<std::uint32_t>(N4M_OP_SAVGOL_SMOOTH));
        append_u32(bytes, 4U);
        append_f64(bytes, 5.0);
        append_f64(bytes, 2.0);
        append_f64(bytes, 0.0);
        append_f64(bytes, 1.0);
    }
    append_u64(bytes, fnv1a64(bytes.data(), bytes.size()));
    return bytes;
}

bool all_zero(const n4m_serialized_pipeline_info_v1_t& info) {
    const n4m_serialized_pipeline_info_v1_t zero{};
    return std::memcmp(&info, &zero, sizeof(info)) == 0;
}

void test_v1_absent_and_sized_output() {
    const std::vector<unsigned char> bytes =
        model_bytes(N4M_SERIALIZATION_FORMAT_VERSION_V1);
    n4m_serialized_pipeline_info_v1_t info{};
    N4M_TEST_REQUIRE(n4m_serialization_inspect_pipeline_v1(
                         bytes.data(), bytes.size(), &info, sizeof(info)) == N4M_OK);
    N4M_TEST_REQUIRE(info.schema_version == N4M_SERIALIZED_PIPELINE_INFO_SCHEMA_V1);
    N4M_TEST_REQUIRE(info.struct_size == sizeof(info));
    N4M_TEST_REQUIRE(info.present == 0U);
    N4M_TEST_REQUIRE(info.operator_count == 0U);
    N4M_TEST_REQUIRE(info.fingerprint_algorithm == N4M_PIPELINE_FINGERPRINT_NONE);
    N4M_TEST_REQUIRE(info.fingerprint == 0U);

    std::memset(&info, 0xa5, sizeof(info));
    N4M_TEST_REQUIRE(n4m_serialization_inspect_pipeline_v1(
                         bytes.data(), bytes.size(), &info, sizeof(info) - 1U) ==
                     N4M_ERR_INVALID_ARGUMENT);
    const auto* raw = reinterpret_cast<const unsigned char*>(&info);
    for (std::size_t i = 0; i < sizeof(info) - 1U; ++i) {
        N4M_TEST_REQUIRE(raw[i] == 0U);
    }
    N4M_TEST_REQUIRE(raw[sizeof(info) - 1U] == 0xa5U);
}

void test_v2_plan_fingerprint_and_tamper() {
    const std::vector<unsigned char> bytes =
        model_bytes(N4M_SERIALIZATION_FORMAT_VERSION_V2);
    n4m_serialized_pipeline_info_v1_t info{};
    N4M_TEST_REQUIRE(n4m_serialization_inspect_pipeline_v1(
                         bytes.data(), bytes.size(), &info, sizeof(info)) == N4M_OK);
    N4M_TEST_REQUIRE(info.present == 1U);
    N4M_TEST_REQUIRE(info.operator_count == 2U);
    N4M_TEST_REQUIRE(info.operators[0] == N4M_OP_SNV);
    N4M_TEST_REQUIRE(info.operators[1] == N4M_OP_SAVGOL_SMOOTH);
    N4M_TEST_REQUIRE(info.savgol_window == 5);
    N4M_TEST_REQUIRE(info.savgol_poly_degree == 2);
    N4M_TEST_REQUIRE(info.savgol_derivative == 0);
    N4M_TEST_REQUIRE(info.savgol_delta == 1.0);
    N4M_TEST_REQUIRE(info.raw_n_features == 9);
    N4M_TEST_REQUIRE(info.model_n_features == 9);
    N4M_TEST_REQUIRE(info.fingerprint_algorithm == N4M_PIPELINE_FINGERPRINT_FNV1A64_V1);
    N4M_TEST_REQUIRE(info.fingerprint == UINT64_C(0x6c670bd4a3e48332));

    constexpr std::size_t kPipelineBytes = 52U;
    const std::size_t pipeline_offset = bytes.size() - 8U - kPipelineBytes;
    std::vector<unsigned char> corrupt_checksum = bytes;
    corrupt_checksum[pipeline_offset + 4U] ^= 1U;
    std::memset(&info, 0xa5, sizeof(info));
    N4M_TEST_REQUIRE(n4m_serialization_inspect_pipeline_v1(
                         corrupt_checksum.data(), corrupt_checksum.size(),
                         &info, sizeof(info)) == N4M_ERR_CORRUPT_BUFFER);
    N4M_TEST_REQUIRE(all_zero(info));

    std::vector<unsigned char> bad_tag = bytes;
    write_u32(
        bad_tag, pipeline_offset + 4U,
        static_cast<std::uint32_t>(N4M_OP_MSC));
    refresh_checksum(bad_tag);
    std::memset(&info, 0xa5, sizeof(info));
    N4M_TEST_REQUIRE(n4m_serialization_inspect_pipeline_v1(
                         bad_tag.data(), bad_tag.size(), &info, sizeof(info)) ==
                     N4M_ERR_CORRUPT_BUFFER);
    N4M_TEST_REQUIRE(all_zero(info));

    std::vector<unsigned char> negative_zero = bytes;
    write_f64(negative_zero, pipeline_offset + 36U, -0.0);
    refresh_checksum(negative_zero);
    std::memset(&info, 0xa5, sizeof(info));
    N4M_TEST_REQUIRE(n4m_serialization_inspect_pipeline_v1(
                         negative_zero.data(), negative_zero.size(),
                         &info, sizeof(info)) == N4M_ERR_CORRUPT_BUFFER);
    N4M_TEST_REQUIRE(all_zero(info));

    std::vector<unsigned char> trailing = bytes;
    trailing.insert(trailing.end() - 8, 0U);
    refresh_checksum(trailing);
    std::memset(&info, 0xa5, sizeof(info));
    N4M_TEST_REQUIRE(n4m_serialization_inspect_pipeline_v1(
                         trailing.data(), trailing.size(), &info, sizeof(info)) ==
                     N4M_ERR_CORRUPT_BUFFER);
    N4M_TEST_REQUIRE(all_zero(info));
}

}  // namespace

int main() {
    n4m_testing::Runner runner("serialized_pipeline_inspect");
    runner.run("v1_absent_sized", test_v1_absent_and_sized_output);
    runner.run("v2_plan_fingerprint_tamper", test_v2_plan_fingerprint_and_tamper);
    return runner.finalize();
}
