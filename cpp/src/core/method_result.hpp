// SPDX-License-Identifier: CECILL-2.1
//
// Universal result container exposed via the public C ABI. Holds named
// double / int arrays and named scalars so that each public fit function
// can pack method-specific outputs through one shared handle type.

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pls4all/p4a.h"

namespace n4m::core {

struct MethodResult {
    // Named double arrays. Shape is stored separately (rows, cols).
    std::unordered_map<std::string, std::vector<double>> double_arrays;
    std::unordered_map<std::string, std::pair<std::int64_t, std::int64_t>>
        double_shapes;

    // Named int32 arrays. 1-D buffers only; for matrices store them as a
    // double array instead.
    std::unordered_map<std::string, std::vector<std::int32_t>> int_arrays;

    // Named int64 arrays. 1-D buffers only; preferred for selector
    // outputs (selected_indices, removed_indices, ...) so feature counts
    // > 2^31 are not silently truncated.
    std::unordered_map<std::string, std::vector<std::int64_t>> int64_arrays;

    // Named scalars (any double-valued metric / counter).
    std::unordered_map<std::string, double> scalars;

    void set_double_matrix(const std::string& name,
                           std::vector<double> values,
                           std::int64_t rows, std::int64_t cols) {
        double_arrays[name] = std::move(values);
        double_shapes[name] = {rows, cols};
    }
    void set_int_vector(const std::string& name,
                        std::vector<std::int32_t> values) {
        int_arrays[name] = std::move(values);
    }
    void set_int64_vector(const std::string& name,
                          std::vector<std::int64_t> values) {
        int64_arrays[name] = std::move(values);
    }
    void set_scalar(const std::string& name, double value) {
        scalars[name] = value;
    }
};

}  // namespace n4m::core

struct n4m_method_result_s : public ::n4m::core::MethodResult {};
