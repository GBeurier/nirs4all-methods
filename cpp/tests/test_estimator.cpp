// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
//
// Manifest-driven conformance suite for the generic estimator surface.
// Every estimator in the compiled manifest is fitted on a train split,
// predicts held-out rows, round-trips through N4ME with bitwise-identical
// outputs, and refuses missing or unused inputs.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "n4m/n4m.h"

#include "fixtures/mb_pls_fixtures.hpp"

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
                   b * std::exp(-10.0 * (t - 0.7) * (t - 0.7)) + 0.05 * std::sin(static_cast<double>(i) * 1.7 + static_cast<double>(j));
        };
        for (int64_t i = 0; i < kTrain + kTest; ++i) {
            std::vector<double>& x = i < kTrain ? x_train : x_test;
            double y = 0.0;
            for (int64_t j = 0; j < kCols; ++j) {
                const double v = spectrum(i, j);
                x.push_back(v);
                y += (j % 3 == 0 ? 1.5 : -0.4) * v;
            }
            if (i < kTrain) y_train.push_back(y + 0.01 * std::sin(3.1 * static_cast<double>(i)));
        }
        for (int64_t i = 0; i < kTrain; ++i) {
            for (int64_t j = 0; j < kCols; ++j) x_target.push_back(spectrum(i + 101, j) + 0.2);
        }
    }
};

struct Inputs {
    Data data;
    n4m_matrix_view_t X{}, Y{}, Xt{}, Ysurv{};
    std::vector<int64_t> feature_groups, blocks, labels;
    std::vector<double> axis, surv;
    Inputs() {
        X = view(data.x_train.data(), kTrain, kCols);
        Y = view(data.y_train.data(), kTrain, 1);
        // PLS-Cox: (time, event) rows, the hazard rising with y, every
        // fourth row censored.
        for (int64_t i = 0; i < kTrain; ++i) {
            surv.push_back(std::exp(-0.5 * data.y_train[static_cast<size_t>(i)]) *
                           (1.0 + 0.01 * static_cast<double>(i)));
            surv.push_back(i % 4 == 3 ? 0.0 : 1.0);
        }
        Ysurv = view(surv.data(), kTrain, 2);
        Xt = view(data.x_target.data(), kTrain, kCols);
        for (int64_t j = 0; j < kCols; ++j) feature_groups.push_back(j / 4);
        for (int64_t j = 0; j < kCols; ++j) axis.push_back(1000.0 + 2.0 * static_cast<double>(j));
        blocks = {4, 4, 4};
        // Three classes with non-contiguous ids, balanced by response terciles.
        std::vector<double> sorted(data.y_train);
        std::sort(sorted.begin(), sorted.end());
        for (double v : data.y_train) {
            labels.push_back(v < sorted[kTrain / 3] ? 10 : v < sorted[2 * kTrain / 3] ? 20 : 30);
        }
    }
    // Inputs the method requires, plus Y.
    n4m_fit_inputs_v1_t for_method(const n4m_method_info_v1_t& info) {
        n4m_fit_inputs_v1_t in{};
        in.struct_size = sizeof(in);
        in.X = &X;
        auto wants = [&](n4m_fit_input_t k) { return info.inputs[k] != N4M_INPUT_NONE; };
        if (wants(N4M_FIT_INPUT_LABELS)) {
            in.labels = labels.data();
            in.n_labels = kTrain;
        }
        if (wants(N4M_FIT_INPUT_Y)) {
            in.Y = std::strcmp(info.method_id, "models.heads.pls_cox") == 0 ? &Ysurv : &Y;
        }
        if (wants(N4M_FIT_INPUT_FEATURE_GROUPS)) {
            in.feature_groups = feature_groups.data();
            in.n_feature_groups = kCols;
        }
        if (wants(N4M_FIT_INPUT_BLOCKS)) {
            in.block_sizes = blocks.data();
            in.n_blocks = static_cast<int64_t>(blocks.size());
        }
        if (wants(N4M_FIT_INPUT_AXIS)) {
            in.axis = axis.data();
            in.n_axis = kCols;
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
        } else if (std::strcmp(pi.name, "top_k") == 0) {
            CHECK(n4m_params_set_int(params, pi.name, 4) == N4M_OK);
        } else if (std::strcmp(pi.name, "start") == 0) {
            CHECK(n4m_params_set_int(params, pi.name, 2) == N4M_OK);
        } else if (std::strcmp(pi.name, "end") == 0) {
            CHECK(n4m_params_set_int(params, pi.name, 10) == N4M_OK);
        } else if (std::strcmp(pi.name, "num_samples") == 0) {
            CHECK(n4m_params_set_int(params, pi.name, 8) == N4M_OK);
        } else if (std::strcmp(pi.name, "thresholds") == 0) {
            const double v[] = {0.05, 0.1, 0.3};
            CHECK(n4m_params_set_double_array(params, pi.name, v, 3) == N4M_OK);
        } else if (std::strcmp(pi.name, "alpha_thresholds") == 0) {
            const double v[] = {0.95, 0.99};
            CHECK(n4m_params_set_double_array(params, pi.name, v, 2) == N4M_OK);
        } else if (std::strcmp(pi.name, "edges") == 0) {
            const double v[] = {-0.5, 0.0, 0.5};
            CHECK(n4m_params_set_double_array(params, pi.name, v, 3) == N4M_OK);
        } else if (std::strcmp(pi.name, "kernel_size") == 0) {
            CHECK(n4m_params_set_int(params, pi.name, 5) == N4M_OK);
        } else if (std::strcmp(pi.name, "alphas") == 0) {
            const double v[] = {0.0, 1.0};
            CHECK(n4m_params_set_double_array(params, pi.name, v, 2) == N4M_OK);
        } else if (std::strcmp(pi.name, "sigmas") == 0) {
            const double v[] = {1.0, 2.0};
            CHECK(n4m_params_set_double_array(params, pi.name, v, 2) == N4M_OK);
        } else if (std::strcmp(pi.name, "n_neighbors") == 0) {
            CHECK(n4m_params_set_int(params, pi.name, 10) == N4M_OK);
        } else if (std::strcmp(pi.name, "window_size") == 0) {
            CHECK(n4m_params_set_int(params, pi.name, 12) == N4M_OK);
        } else if (std::strcmp(pi.name, "chain_offsets") == 0) {
            // AOM chains: identity, then a first-order detrend (chain_params {1}).
            const int64_t v[] = {0, 1, 2};
            CHECK(n4m_params_set_int_array(params, pi.name, v, 3) == N4M_OK);
        } else if (std::strcmp(pi.name, "op_kinds") == 0) {
            const int64_t v[] = {N4M_OP_IDENTITY, N4M_OP_DETREND_POLY};
            CHECK(n4m_params_set_int_array(params, pi.name, v, 2) == N4M_OK);
        } else if (std::strcmp(pi.name, "param_offsets") == 0) {
            const int64_t v[] = {0, 0, 1};
            CHECK(n4m_params_set_int_array(params, pi.name, v, 3) == N4M_OK);
            const double order = 1.0;
            CHECK(n4m_params_set_double_array(params, "chain_params", &order, 1) == N4M_OK);
        } else if (std::strcmp(pi.name, "n_components_per_block") == 0 ||
                   std::strcmp(pi.name, "n_unique_per_block") == 0) {
            // One per test block (3 blocks of 4 columns).
            const int64_t v[] = {1, 1, 1};
            CHECK(n4m_params_set_int_array(params, pi.name, v, 3) == N4M_OK);
        } else {
            throw std::runtime_error(std::string("no test value for required parameter ") +
                                     pi.name + " @" + current_);
        }
    }
}

// States that retain training rows export only when the caller allows it.
std::vector<unsigned char> export_bytes(n4m_context_t* ctx, const n4m_estimator_t* est) {
    int32_t idx = -1;
    uint64_t caps = 0;
    CHECK(n4m_estimator_info(est, &idx, &caps) == N4M_OK);
    uint32_t flags = 0;
    size_t size = 0;
    if ((caps & N4M_CAP_RETAINS_TRAINING_ROWS) != 0) {
        CHECK(n4m_estimator_export_size(ctx, est, 0, &size) == N4M_ERR_INVALID_ARGUMENT);
        flags = N4M_EXPORT_ALLOW_TRAINING_ROWS;
    }
    CHECK(n4m_estimator_export_size(ctx, est, flags, &size) == N4M_OK);
    std::vector<unsigned char> bytes(size);
    size_t written = 0;
    CHECK(n4m_estimator_export_to_buffer(ctx, est, flags, bytes.data(), size, &written) ==
          N4M_OK);
    CHECK(written == size);
    return bytes;
}

// ---- N4ME semantic mutations ------------------------------------------------

uint64_t fnv1a64(const unsigned char* data, size_t size) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < size; ++i) {
        h ^= data[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

uint64_t get_le(const std::vector<unsigned char>& b, size_t pos, int n) {
    uint64_t v = 0;
    for (int i = 0; i < n; ++i) v |= static_cast<uint64_t>(b[pos + static_cast<size_t>(i)]) << (8 * i);
    return v;
}

void put_le(std::vector<unsigned char>& b, size_t pos, uint64_t v) {
    for (int i = 0; i < 8; ++i) b[pos + static_cast<size_t>(i)] = static_cast<unsigned char>(v >> (8 * i));
}

// Recomputes the trailing FNV-1a checksum: the mutations below test semantic
// consistency, not corruption detection.
void reseal(std::vector<unsigned char>& b) { put_le(b, b.size() - 8, fnv1a64(b.data(), b.size() - 8)); }

// Byte offset of the first value and value count of each N4ME parameter.
struct ParamSlot {
    size_t offset;
    uint64_t count;
};
std::vector<ParamSlot> param_slots(const std::vector<unsigned char>& b) {
    size_t pos = 20;  // magic, format, writer ABI
    pos += 4 + get_le(b, pos, 4);
    const uint64_t n = get_le(b, pos, 4);
    pos += 4;
    std::vector<ParamSlot> slots;
    for (uint64_t k = 0; k < n; ++k) {
        pos += 4 + get_le(b, pos, 4);  // name
        pos += 4;                      // type
        const uint64_t count = get_le(b, pos, 8);
        pos += 8;
        slots.push_back({pos, count});
        pos += 8 * count;
    }
    return slots;
}

bool in_bounds(const n4m_param_info_v1_t& pi, double v) {
    return std::isfinite(v) && (std::isnan(pi.min_value) || v >= pi.min_value) &&
           (std::isnan(pi.max_value) || v <= pi.max_value);
}

// Writes a different valid value into the first element of a parameter;
// false when the manifest leaves no other value.
bool mutate(std::vector<unsigned char>& b, const ParamSlot& slot, const n4m_param_info_v1_t& pi) {
    if (slot.count == 0) return false;
    const uint64_t raw = get_le(b, slot.offset, 8);
    if (pi.type == N4M_METHOD_PARAM_DOUBLE || pi.type == N4M_METHOD_PARAM_DOUBLE_ARRAY) {
        double v = 0.0;
        std::memcpy(&v, &raw, 8);
        for (double w : {v * 2.0 + 1.0, v / 2.0, v - 1.0}) {
            if (w != v && in_bounds(pi, w)) {
                uint64_t bits = 0;
                std::memcpy(&bits, &w, 8);
                put_le(b, slot.offset, bits);
                return true;
            }
        }
        return false;
    }
    const auto v = static_cast<int64_t>(raw);
    int64_t w = v;
    if (pi.type == N4M_METHOD_PARAM_BOOL) {
        w = 1 - v;
    } else if (pi.type == N4M_METHOD_PARAM_ENUM) {
        if (pi.n_choices < 2) return false;
        w = (v + 1) % pi.n_choices;
    } else if (in_bounds(pi, static_cast<double>(v + 1))) {
        w = v + 1;
    } else if (in_bounds(pi, static_cast<double>(v - 1))) {
        w = v - 1;
    } else {
        return false;
    }
    put_le(b, slot.offset, static_cast<uint64_t>(w));
    return true;
}

int recorded_refusals = 0;

// Every parameter the state records refuses a different value at import
// (checksum recomputed); a payload with another parameter mutated either is
// refused or round-trips unchanged.
void semantic_mutations(n4m_context_t* ctx, int32_t index, const n4m_method_info_v1_t& info,
                        const std::vector<unsigned char>& bytes) {
    std::vector<unsigned char> same = bytes;
    reseal(same);
    CHECK(same == bytes);
    const auto slots = param_slots(bytes);
    CHECK(slots.size() == static_cast<size_t>(info.n_params));
    for (int32_t p = 0; p < info.n_params; ++p) {
        n4m_param_info_v1_t pi{};
        pi.struct_size = sizeof(pi);
        CHECK(n4m_method_param_info_v1(index, p, &pi) == N4M_OK);
        std::vector<unsigned char> mutated = bytes;
        if (!mutate(mutated, slots[static_cast<size_t>(p)], pi)) {
            CHECK(pi.recorded == 0);
            continue;
        }
        reseal(mutated);
        n4m_estimator_t* back = nullptr;
        const n4m_status_t st =
            n4m_estimator_import_from_buffer(ctx, mutated.data(), mutated.size(), &back);
        if (pi.recorded != 0) {
            if (st != N4M_ERR_CORRUPT_BUFFER) {
                throw std::runtime_error(std::string("recorded parameter ") + pi.name +
                                         " accepted a contradicting value @" + current_);
            }
            ++recorded_refusals;
            continue;
        }
        if (st == N4M_OK) {
            CHECK(export_bytes(ctx, back) == mutated);
            n4m_estimator_destroy(back);
        }
    }
}

// Classifiers: labels from the fitted classes, decision width, probabilities
// summing to one when defined, bitwise N4ME round trip.
void conformance_classifier(n4m_context_t* ctx, const n4m_estimator_t* est, Inputs& in) {
    int64_t n_classes = 0, count = 0;
    CHECK(n4m_estimator_n_outputs(est, &n_classes) == N4M_OK && n_classes == 3);
    std::vector<int64_t> classes(3);
    CHECK(n4m_estimator_classes(est, classes.data(), 3, &count) == N4M_OK && count == 3);
    CHECK((classes == std::vector<int64_t>{10, 20, 30}));
    auto X_test = view(in.data.x_test.data(), kTest, kCols);
    std::vector<int64_t> labels(kTest), labels2(kTest);
    CHECK(n4m_estimator_predict_labels(ctx, est, &X_test, labels.data(), kTest) == N4M_OK);
    for (int64_t v : labels) CHECK(v == 10 || v == 20 || v == 30);
    std::vector<double> decision(static_cast<size_t>(kTest * 3)), decision2(decision.size());
    auto D = view(decision.data(), kTest, 3);
    CHECK(n4m_estimator_decision_function(ctx, est, &X_test, &D) == N4M_OK);
    std::vector<double> proba(decision.size());
    auto P = view(proba.data(), kTest, 3);
    int32_t idx = -1;
    uint64_t caps = 0;
    CHECK(n4m_estimator_info(est, &idx, &caps) == N4M_OK);
    if ((caps & N4M_CAP_PREDICT_PROBA) != 0) {
        CHECK(n4m_estimator_predict_proba(ctx, est, &X_test, &P) == N4M_OK);
        for (int64_t i = 0; i < kTest; ++i) {
            double sum = 0.0;
            for (int64_t c = 0; c < 3; ++c) sum += proba[static_cast<size_t>(i * 3 + c)];
            CHECK(std::fabs(sum - 1.0) < 1e-12);
        }
    } else {
        CHECK(n4m_estimator_predict_proba(ctx, est, &X_test, &P) == N4M_ERR_UNSUPPORTED);
    }
    std::vector<double> pred(kTest);
    auto Pr = view(pred.data(), kTest, 1);
    CHECK(n4m_estimator_predict(ctx, est, &X_test, &Pr) == N4M_ERR_UNSUPPORTED);
    const auto bytes = export_bytes(ctx, est);
    n4m_estimator_t* back = nullptr;
    CHECK(n4m_estimator_import_from_buffer(ctx, bytes.data(), bytes.size(), &back) == N4M_OK);
    CHECK(n4m_estimator_predict_labels(ctx, back, &X_test, labels2.data(), kTest) == N4M_OK);
    CHECK(labels == labels2);
    auto D2 = view(decision2.data(), kTest, 3);
    CHECK(n4m_estimator_decision_function(ctx, back, &X_test, &D2) == N4M_OK);
    CHECK(decision == decision2);
    CHECK(export_bytes(ctx, back) == bytes);
    n4m_estimator_destroy(back);
}

// Sample filters: a 0/1 keep mask that keeps rows, identical for column-major
// input, and an identical mask after an N4ME round trip when serializable.
void conformance_sample_filter(n4m_context_t* ctx, const n4m_estimator_t* est, Inputs& in,
                               const n4m_method_info_v1_t& info) {
    std::vector<uint8_t> mask(kTrain, 2), mask2(kTrain, 2);
    const n4m_matrix_view_t* Y = &in.Y;
    CHECK(n4m_estimator_apply_mask(ctx, est, &in.X, Y, mask.data(), kTrain) == N4M_OK);
    int64_t kept = 0;
    for (uint8_t m : mask) {
        CHECK(m <= 1);
        kept += m;
    }
    CHECK(kept > 0);
    if (info.inputs[N4M_FIT_INPUT_Y] == N4M_INPUT_REQUIRED) {
        CHECK(n4m_estimator_apply_mask(ctx, est, &in.X, nullptr, mask2.data(), kTrain) != N4M_OK);
    }
    std::vector<double> x_cm(static_cast<size_t>(kTrain * kCols));
    for (int64_t i = 0; i < kTrain; ++i) {
        for (int64_t j = 0; j < kCols; ++j) {
            x_cm[static_cast<size_t>(j * kTrain + i)] = in.data.x_train[static_cast<size_t>(i * kCols + j)];
        }
    }
    n4m_matrix_view_t Xc{};
    CHECK(n4m_matrix_view_init_colmajor(&Xc, x_cm.data(), kTrain, kCols, N4M_DTYPE_F64) == N4M_OK);
    CHECK(n4m_estimator_apply_mask(ctx, est, &Xc, Y, mask2.data(), kTrain) == N4M_OK);
    CHECK(mask2 == mask);
    int64_t cols = 0;
    CHECK(n4m_estimator_transform_cols(est, &cols) != N4M_OK);
    if ((info.capabilities & N4M_CAP_SERIALIZABLE) == 0) {
        size_t size = 0;
        CHECK(n4m_estimator_export_size(ctx, est, 0, &size) != N4M_OK);
        return;
    }
    const auto bytes = export_bytes(ctx, est);
    n4m_estimator_t* back = nullptr;
    CHECK(n4m_estimator_import_from_buffer(ctx, bytes.data(), bytes.size(), &back) == N4M_OK);
    std::fill(mask2.begin(), mask2.end(), uint8_t{2});
    CHECK(n4m_estimator_apply_mask(ctx, back, &in.X, Y, mask2.data(), kTrain) == N4M_OK);
    CHECK(mask2 == mask);
    CHECK(export_bytes(ctx, back) == bytes);
    n4m_estimator_destroy(back);
}

// Selectors and pure transformers: out-of-sample transform, selected columns,
// and bitwise N4ME round trip.
void conformance_transform_only(n4m_context_t* ctx, const n4m_estimator_t* est, Inputs& in,
                                const n4m_method_info_v1_t& info) {
    int64_t cols = 0;
    CHECK(n4m_estimator_transform_cols(est, &cols) == N4M_OK && cols > 0);
    CHECK((info.roles & N4M_ROLE_SELECTOR) == 0 || cols <= kCols);
    std::vector<double> out(static_cast<size_t>(kTest * cols)), out2(out.size());
    auto X_test = view(in.data.x_test.data(), kTest, kCols);
    auto O = view(out.data(), kTest, cols);
    CHECK(n4m_estimator_transform(ctx, est, &X_test, &O) == N4M_OK);
    if ((info.roles & N4M_ROLE_SELECTOR) != 0) {
        std::vector<int64_t> idx(static_cast<size_t>(cols));
        int64_t count = 0;
        CHECK(n4m_estimator_selected_indices(est, idx.data(), cols, &count) == N4M_OK);
        CHECK(count == cols);
        std::sort(idx.begin(), idx.end());
        for (int64_t i = 0; i < kTest; ++i) {
            for (int64_t j = 0; j < cols; ++j) {
                CHECK(out[static_cast<size_t>(i * cols + j)] ==
                      in.data.x_test[static_cast<size_t>(i * kCols + idx[static_cast<size_t>(j)])]);
            }
        }
    }
    // Column-major (R/MATLAB layout) input and output give identical values.
    {
        std::vector<double> x_cm(static_cast<size_t>(kTest * kCols)), o_cm(out.size());
        for (int64_t i = 0; i < kTest; ++i) {
            for (int64_t j = 0; j < kCols; ++j) {
                x_cm[static_cast<size_t>(j * kTest + i)] =
                    in.data.x_test[static_cast<size_t>(i * kCols + j)];
            }
        }
        n4m_matrix_view_t Xc{}, Oc{};
        CHECK(n4m_matrix_view_init_colmajor(&Xc, x_cm.data(), kTest, kCols, N4M_DTYPE_F64) == N4M_OK);
        CHECK(n4m_matrix_view_init_colmajor(&Oc, o_cm.data(), kTest, cols, N4M_DTYPE_F64) == N4M_OK);
        CHECK(n4m_estimator_transform(ctx, est, &Xc, &Oc) == N4M_OK);
        for (int64_t i = 0; i < kTest; ++i) {
            for (int64_t j = 0; j < cols; ++j) {
                CHECK(std::memcmp(&o_cm[static_cast<size_t>(j * kTest + i)],
                                  &out[static_cast<size_t>(i * cols + j)], sizeof(double)) == 0);
            }
        }
    }
    const auto bytes = export_bytes(ctx, est);
    n4m_estimator_t* back = nullptr;
    CHECK(n4m_estimator_import_from_buffer(ctx, bytes.data(), bytes.size(), &back) == N4M_OK);
    auto O2 = view(out2.data(), kTest, cols);
    CHECK(n4m_estimator_transform(ctx, back, &X_test, &O2) == N4M_OK);
    CHECK(out == out2);
    CHECK(export_bytes(ctx, back) == bytes);
    n4m_estimator_destroy(back);
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
    // Random frog's default initial subset (20) exceeds the 12 test columns.
    if (std::strcmp(info.method_id, "selection.random_frog") == 0) {
        CHECK(n4m_params_set_int(params, "initial_size", 6) == N4M_OK);
    }
    // The test spectra have rank 4; the calibration's PLS paths stop there.
    if (std::strcmp(info.method_id, "aom_pop.calibration") == 0) {
        CHECK(n4m_params_set_int(params, "max_components", 4) == N4M_OK);
    }
    CHECK(n4m_params_validate(ctx, params) == N4M_OK);
    n4m_estimator_t* est = nullptr;
    CHECK(n4m_estimator_create(ctx, info.method_id, params, &est) == N4M_OK);
    n4m_params_destroy(params);

    // Not fitted yet.
    int32_t unfitted_rows = -1;
    CHECK(n4m_estimator_contains_training_rows(est, &unfitted_rows) == N4M_ERR_NOT_FITTED);
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
        if (k == N4M_FIT_INPUT_AXIS) missing.axis = nullptr;
        if (k == N4M_FIT_INPUT_TARGET_DOMAIN) missing.X_target = nullptr;
        if (k == N4M_FIT_INPUT_LABELS) missing.labels = nullptr;
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
    int32_t retains = -1;
    CHECK(n4m_estimator_contains_training_rows(est, &retains) == N4M_OK);
    CHECK(retains == ((caps & N4M_CAP_RETAINS_TRAINING_ROWS) != 0 ? 1 : 0));
    if ((caps & N4M_CAP_SERIALIZABLE) != 0) {
        semantic_mutations(ctx, index, info, export_bytes(ctx, est));
    } else {
        for (int32_t p = 0; p < info.n_params; ++p) {
            n4m_param_info_v1_t pi{};
            pi.struct_size = sizeof(pi);
            CHECK(n4m_method_param_info_v1(index, p, &pi) == N4M_OK && pi.recorded == 0);
        }
    }
    // Role interfaces and fitted capabilities agree both ways.
    const bool predicts = (info.roles & N4M_ROLE_REGRESSOR) != 0;
    const bool transforms = (info.roles & (N4M_ROLE_TRANSFORMER | N4M_ROLE_SELECTOR)) != 0;
    CHECK(predicts == ((caps & N4M_CAP_PREDICT) != 0));
    CHECK(transforms == ((caps & N4M_CAP_TRANSFORM) != 0));
    CHECK(((info.roles & N4M_ROLE_CLASSIFIER) != 0) == ((caps & N4M_CAP_PREDICT_LABELS) != 0));
    CHECK(((info.roles & N4M_ROLE_SELECTOR) != 0) == ((caps & N4M_CAP_SELECTED_INDICES) != 0));
    CHECK(((info.roles & N4M_ROLE_SAMPLE_FILTER) != 0) == ((caps & N4M_CAP_APPLY_MASK) != 0));
    int64_t n_in = 0, n_out = 0;
    CHECK(n4m_estimator_n_features_in(est, &n_in) == N4M_OK && n_in == kCols);
    if ((info.roles & N4M_ROLE_CLASSIFIER) != 0) {
        conformance_classifier(ctx, est, in);
        n4m_estimator_destroy(est);
        return;
    }
    CHECK(n4m_estimator_n_outputs(est, &n_out) == N4M_OK && n_out == (predicts ? 1 : 0));
    if ((info.roles & N4M_ROLE_SAMPLE_FILTER) != 0) {
        conformance_sample_filter(ctx, est, in, info);
        n4m_estimator_destroy(est);
        return;
    }

    if (!predicts) {
        CHECK(n4m_estimator_predict(ctx, est, &X_test, &P) == N4M_ERR_UNSUPPORTED);
        conformance_transform_only(ctx, est, in, info);
        n4m_estimator_destroy(est);
        return;
    }

    // Out-of-sample predictions are finite and informative.
    CHECK(n4m_estimator_predict(ctx, est, &X_test, &P) == N4M_OK);
    double spread = 0.0;
    for (double v : pred) {
        CHECK(std::isfinite(v));
        spread += std::fabs(v - pred[0]);
    }
    CHECK(spread > 0.0);
    // Column-major (R/MATLAB layout) input and output give the same result.
    {
        std::vector<double> x_cm(static_cast<size_t>(kTest * kCols)), p_cm(kTest);
        for (int64_t i = 0; i < kTest; ++i) {
            for (int64_t j = 0; j < kCols; ++j) {
                x_cm[static_cast<size_t>(j * kTest + i)] =
                    in.data.x_test[static_cast<size_t>(i * kCols + j)];
            }
        }
        n4m_matrix_view_t Xc{}, Pc{};
        CHECK(n4m_matrix_view_init_colmajor(&Xc, x_cm.data(), kTest, kCols, N4M_DTYPE_F64) == N4M_OK);
        CHECK(n4m_matrix_view_init_colmajor(&Pc, p_cm.data(), kTest, 1, N4M_DTYPE_F64) == N4M_OK);
        CHECK(n4m_estimator_predict(ctx, est, &Xc, &Pc) == N4M_OK);
        for (int64_t i = 0; i < kTest; ++i) {
            CHECK(std::fabs(p_cm[static_cast<size_t>(i)] - pred[static_cast<size_t>(i)]) <=
                  1e-12 * (1.0 + std::fabs(pred[static_cast<size_t>(i)])));
        }
    }
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

// Estimators over kernels that also predict their own training rows
// reproduce those predictions (bitwise when the same kernel path runs).
void test_in_sample_equivalence(n4m_context_t* ctx, Inputs& in) {
    n4m_config_t* cfg = nullptr;
    CHECK(n4m_config_create(&cfg) == N4M_OK);
    std::vector<n4m_matrix_view_t> blocks;
    for (int64_t b = 0; b < 3; ++b) {
        n4m_matrix_view_t v = in.X;
        v.data = in.data.x_train.data() + 4 * b;
        v.cols = 4;
        blocks.push_back(v);
    }
    const int32_t per_block[] = {1, 1, 1};
    struct Case {
        const char* method_id;
        double tolerance;  // 0: bitwise
        n4m_status_t status;
        n4m_method_result_t* result;
    };
    Case cases[] = {
        {"models.pls.kernel", 0.0, N4M_OK, nullptr},
        {"models.local.lw_pls", 0.0, N4M_OK, nullptr},
        {"models.specialized.missing_aware_nipals", 0.0, N4M_OK, nullptr},
        // In-sample GPR uses (K + s2 I) alpha = y; predict evaluates K alpha.
        {"models.specialized.gpr_pls", 1e-9, N4M_OK, nullptr},
        // In-sample SO-PLS / ROSA accumulate scores; predict is the affine map.
        {"models.multiblock.so_pls", 1e-10, N4M_OK, nullptr},
        {"models.multiblock.rosa", 1e-10, N4M_OK, nullptr},
        {"models.heads.pls_glm", 0.0, N4M_OK, nullptr},
        {"models.heads.pls_cox", 0.0, N4M_OK, nullptr},
    };
    cases[0].status = n4m_estimators_kernel_pls_fit(ctx, cfg, 1, 0.0, 1.0, 3, &in.X, &in.Y,
                                                    &cases[0].result);
    cases[1].status = n4m_estimators_lw_pls_fit(ctx, cfg, &in.X, &in.Y, 10, &cases[1].result);
    cases[2].status =
        n4m_estimators_missing_aware_nipals_fit(ctx, cfg, &in.X, &in.Y, &cases[2].result);
    cases[3].status =
        n4m_estimators_gpr_pls_fit(ctx, cfg, &in.X, &in.Y, 1.0, 1e-3, 0, &cases[3].result);
    cases[4].status = n4m_estimators_so_pls_fit(ctx, cfg, blocks.data(), 3, &in.Y, per_block, 3,
                                                &cases[4].result);
    cases[5].status =
        n4m_estimators_rosa_fit(ctx, cfg, blocks.data(), 3, &in.Y, 2, &cases[5].result);
    cases[6].status = n4m_estimators_pls_glm_fit(ctx, cfg, &in.X, &in.Y, 0, &cases[6].result);
    std::vector<double> times;
    std::vector<int32_t> events;
    for (int64_t i = 0; i < kTrain; ++i) {
        times.push_back(in.surv[static_cast<size_t>(2 * i)]);
        events.push_back(static_cast<int32_t>(in.surv[static_cast<size_t>(2 * i + 1)]));
    }
    cases[7].status = n4m_estimators_pls_cox_fit(ctx, cfg, &in.X, times.data(), kTrain,
                                                 events.data(), kTrain, &cases[7].result);
    n4m_config_destroy(cfg);
    for (Case& c : cases) {
        current_ = std::string(c.method_id) + " in-sample equivalence";
        CHECK(c.status == N4M_OK);
        const double* ref = nullptr;
        int64_t rows = 0, cols = 0;
        CHECK(n4m_method_result_get_double_matrix(c.result, "predictions", &ref, &rows, &cols) ==
              N4M_OK);
        CHECK(rows == kTrain && cols == 1);
        int32_t index = -1;
        CHECK(n4m_method_find(c.method_id, &index) == N4M_OK);
        n4m_params_t* params = nullptr;
        CHECK(n4m_params_create(ctx, index, &params) == N4M_OK);
        fill_required(index, params);
        n4m_estimator_t* est = nullptr;
        CHECK(n4m_estimator_create(ctx, c.method_id, params, &est) == N4M_OK);
        n4m_params_destroy(params);
        n4m_method_info_v1_t info{};
        info.struct_size = sizeof(info);
        CHECK(n4m_method_info_v1(index, &info) == N4M_OK);
        n4m_fit_inputs_v1_t inputs = in.for_method(info);
        CHECK(n4m_estimator_fit(ctx, est, &inputs) == N4M_OK);
        std::vector<double> pred(kTrain);
        auto P = view(pred.data(), kTrain, 1);
        CHECK(n4m_estimator_predict(ctx, est, &in.X, &P) == N4M_OK);
        for (int64_t i = 0; i < kTrain; ++i) {
            const double r = ref[i], v = pred[static_cast<size_t>(i)];
            CHECK(c.tolerance == 0.0 ? std::memcmp(&r, &v, sizeof(double)) == 0
                                     : std::fabs(r - v) <= c.tolerance * (1.0 + std::fabs(r)));
        }
        n4m_estimator_destroy(est);
        n4m_method_result_destroy(c.result);
    }
}

// The AOM regressors predict new rows from their selected model folded into
// the input space; on the training rows that reproduces the kernel's own
// (transformed-space) in-sample predictions.
void test_aom_in_sample(n4m_context_t* ctx, Inputs& in) {
    int32_t count = 0;
    CHECK(n4m_method_count(&count) == N4M_OK);
    for (int32_t index = 0; index < count; ++index) {
        n4m_method_info_v1_t info{};
        info.struct_size = sizeof(info);
        CHECK(n4m_method_info_v1(index, &info) == N4M_OK);
        // The calibration result keeps CV scores, not in-sample predictions.
        if (std::strncmp(info.method_id, "aom_pop.", 8) != 0 ||
            (info.roles & N4M_ROLE_REGRESSOR) == 0 ||
            std::strcmp(info.method_id, "aom_pop.calibration") == 0) {
            continue;
        }
        current_ = std::string(info.method_id) + " in-sample equivalence";
        n4m_params_t* params = nullptr;
        CHECK(n4m_params_create(ctx, index, &params) == N4M_OK);
        fill_required(index, params);
        n4m_estimator_t* est = nullptr;
        CHECK(n4m_estimator_create(ctx, info.method_id, params, &est) == N4M_OK);
        n4m_params_destroy(params);
        n4m_fit_inputs_v1_t inputs = in.for_method(info);
        CHECK(n4m_estimator_fit(ctx, est, &inputs) == N4M_OK);
        const n4m_method_result_t* result = nullptr;
        CHECK(n4m_estimator_fit_result(est, &result) == N4M_OK);
        const double* ref = nullptr;
        int64_t rows = 0, cols = 0;
        CHECK(n4m_method_result_get_double_matrix(result, "predictions", &ref, &rows, &cols) ==
              N4M_OK);
        CHECK(rows == kTrain && cols == 1);
        std::vector<double> pred(kTrain);
        auto P = view(pred.data(), kTrain, 1);
        CHECK(n4m_estimator_predict(ctx, est, &in.X, &P) == N4M_OK);
        for (int64_t i = 0; i < kTrain; ++i) {
            CHECK(std::fabs(ref[i] - pred[static_cast<size_t>(i)]) <=
                  1e-10 * (1.0 + std::fabs(ref[i])));
        }
        n4m_estimator_destroy(est);
    }
}

// A failed refit keeps the previous fitted state; per-row inputs must match
// the rows of X exactly (never broadcast).
void test_refit_and_input_lengths(n4m_context_t* ctx, Inputs& in) {
    current_ = "refit";
    n4m_estimator_t* est = nullptr;
    CHECK(n4m_estimator_create(ctx, "models.pls.pls_regression", nullptr, &est) == N4M_OK);
    n4m_fit_inputs_v1_t inputs{};
    inputs.struct_size = sizeof(inputs);
    inputs.X = &in.X;
    inputs.Y = &in.Y;
    CHECK(n4m_estimator_fit(ctx, est, &inputs) == N4M_OK);
    auto X_test = view(in.data.x_test.data(), kTest, kCols);
    std::vector<double> before(kTest), after(kTest);
    auto B = view(before.data(), kTest, 1);
    CHECK(n4m_estimator_predict(ctx, est, &X_test, &B) == N4M_OK);
    // Fewer target rows than X rows, and a transposed target.
    n4m_fit_inputs_v1_t bad = inputs;
    auto short_y = view(in.data.y_train.data(), kTrain - 1, 1);
    bad.Y = &short_y;
    CHECK(n4m_estimator_fit(ctx, est, &bad) == N4M_ERR_SHAPE_MISMATCH);
    CHECK(std::strstr(n4m_context_last_error(ctx), "'y'") != nullptr);
    auto wide_y = view(in.data.y_train.data(), 1, kTrain);
    bad.Y = &wide_y;
    CHECK(n4m_estimator_fit(ctx, est, &bad) == N4M_ERR_SHAPE_MISMATCH);
    int32_t fitted = 0;
    CHECK(n4m_estimator_is_fitted(est, &fitted) == N4M_OK && fitted == 1);
    auto A = view(after.data(), kTest, 1);
    CHECK(n4m_estimator_predict(ctx, est, &X_test, &A) == N4M_OK);
    CHECK(before == after);
    // Output views must match the operation's shape exactly: a narrower or
    // wider view, or another dtype, is refused before anything is written.
    {
        std::vector<double> buf(static_cast<size_t>(kTest * 3), -7.0);
        auto wide = view(buf.data(), kTest, 2);
        CHECK(n4m_estimator_predict(ctx, est, &X_test, &wide) == N4M_ERR_SHAPE_MISMATCH);
        auto narrow = view(buf.data(), kTest, 1);
        CHECK(n4m_estimator_transform(ctx, est, &X_test, &narrow) == N4M_ERR_SHAPE_MISMATCH);
        CHECK(std::strstr(n4m_context_last_error(ctx), "output view") != nullptr);
        auto three = view(buf.data(), kTest, 3);
        CHECK(n4m_estimator_transform(ctx, est, &X_test, &three) == N4M_ERR_SHAPE_MISMATCH);
        std::vector<float> f32(static_cast<size_t>(kTest));
        n4m_matrix_view_t single{};
        CHECK(n4m_matrix_view_init_rowmajor(&single, f32.data(), kTest, 1, N4M_DTYPE_F32) == N4M_OK);
        CHECK(n4m_estimator_predict(ctx, est, &X_test, &single) == N4M_ERR_SHAPE_MISMATCH);
        for (double v : buf) CHECK(v == -7.0);
    }
    // A refit that fails inside the kernel (more components than rows allow).
    n4m_params_t* params = nullptr;
    int32_t index = -1;
    CHECK(n4m_method_find("models.pls.pls_regression", &index) == N4M_OK);
    CHECK(n4m_params_create(ctx, index, &params) == N4M_OK);
    CHECK(n4m_params_set_int(params, "n_components", 3) == N4M_OK);
    n4m_estimator_t* three = nullptr;
    CHECK(n4m_estimator_create(ctx, "models.pls.pls_regression", params, &three) == N4M_OK);
    n4m_params_destroy(params);
    CHECK(n4m_estimator_fit(ctx, three, &inputs) == N4M_OK);
    auto two_rows = view(in.data.x_train.data(), 2, kCols);
    auto two_y = view(in.data.y_train.data(), 2, 1);
    n4m_fit_inputs_v1_t tiny = inputs;
    tiny.X = &two_rows;
    tiny.Y = &two_y;
    CHECK(n4m_estimator_fit(ctx, three, &tiny) != N4M_OK);
    CHECK(n4m_estimator_is_fitted(three, &fitted) == N4M_OK && fitted == 1);
    int64_t n_in = 0;
    CHECK(n4m_estimator_n_features_in(three, &n_in) == N4M_OK && n_in == kCols);
    n4m_estimator_destroy(three);
    n4m_estimator_destroy(est);

    // Classifier: labels, weights and groups have one entry per row.
    current_ = "classifier refit";
    CHECK(n4m_estimator_create(ctx, "models.classification.pls_lda", nullptr, &est) == N4M_OK);
    n4m_fit_inputs_v1_t cls{};
    cls.struct_size = sizeof(cls);
    cls.X = &in.X;
    cls.labels = in.labels.data();
    cls.n_labels = kTrain;
    CHECK(n4m_estimator_fit(ctx, est, &cls) == N4M_OK);
    n4m_fit_inputs_v1_t one_class = cls;
    std::vector<int64_t> same(kTrain, 7);
    one_class.labels = same.data();
    CHECK(n4m_estimator_fit(ctx, est, &one_class) == N4M_ERR_INVALID_ARGUMENT);
    n4m_fit_inputs_v1_t short_labels = cls;
    short_labels.n_labels = kTrain - 1;
    CHECK(n4m_estimator_fit(ctx, est, &short_labels) == N4M_ERR_SHAPE_MISMATCH);
    std::vector<int64_t> classes(3);
    int64_t count = 0;
    CHECK(n4m_estimator_classes(est, classes.data(), 3, &count) == N4M_OK && count == 3);
    std::vector<double> scores(static_cast<size_t>(kTest * 2));
    auto two = view(scores.data(), kTest, 2);
    CHECK(n4m_estimator_decision_function(ctx, est, &X_test, &two) == N4M_ERR_SHAPE_MISMATCH);
    CHECK((classes == std::vector<int64_t>{10, 20, 30}));
    n4m_estimator_destroy(est);

    // Sample filters read an aligned target only.
    current_ = "filter y";
    CHECK(n4m_estimator_create(ctx, "filters.y_outlier", nullptr, &est) == N4M_OK);
    CHECK(n4m_estimator_fit(ctx, est, &inputs) == N4M_OK);
    std::vector<uint8_t> mask(kTrain);
    CHECK(n4m_estimator_apply_mask(ctx, est, &in.X, &short_y, mask.data(), kTrain) ==
          N4M_ERR_SHAPE_MISMATCH);
    n4m_estimator_destroy(est);
}

// Every input matrix (X, y, target domain, new rows) is validated before an
// adapter reads it: a float32 view of exactly its declared size is refused,
// never read as doubles past its extent, and so is an invalid layout.
void test_input_views(n4m_context_t* ctx, Inputs& in) {
    current_ = "input views";
    std::vector<float> x32(in.data.x_train.begin(), in.data.x_train.end());
    std::vector<float> y32(in.data.y_train.begin(), in.data.y_train.end());
    n4m_matrix_view_t X32{}, Y32{};
    CHECK(n4m_matrix_view_init_rowmajor(&X32, x32.data(), kTrain, kCols, N4M_DTYPE_F32) == N4M_OK);
    CHECK(n4m_matrix_view_init_rowmajor(&Y32, y32.data(), kTrain, 1, N4M_DTYPE_F32) == N4M_OK);
    auto expect = [&](n4m_status_t st, n4m_status_t status, const char* fragment) {
        CHECK(st == status);
        CHECK(std::strstr(n4m_context_last_error(ctx), fragment) != nullptr);
    };
    auto fit = [&](const char* id, const n4m_fit_inputs_v1_t& inputs) {
        n4m_estimator_t* est = nullptr;
        CHECK(n4m_estimator_create(ctx, id, nullptr, &est) == N4M_OK);
        const n4m_status_t st = n4m_estimator_fit(ctx, est, &inputs);
        int32_t fitted = 1;
        CHECK(n4m_estimator_is_fitted(est, &fitted) == N4M_OK);
        n4m_estimator_destroy(est);
        CHECK(st == N4M_OK || fitted == 0);
        return st;
    };
    n4m_fit_inputs_v1_t inputs{};
    inputs.struct_size = sizeof(inputs);
    inputs.X = &X32;
    expect(fit("preprocessing.scaling.simple_scale", inputs), N4M_ERR_DTYPE_MISMATCH,
           "matrix 'X' must hold float64 values");
    inputs.X = &in.X;
    inputs.Y = &Y32;
    expect(fit("models.pls.pls_regression", inputs), N4M_ERR_DTYPE_MISMATCH,
           "matrix 'y' must hold float64 values");
    inputs.Y = nullptr;
    inputs.X_target = &X32;
    expect(fit("preprocessing.transfer.direct_standardization", inputs), N4M_ERR_DTYPE_MISMATCH,
           "matrix 'target_domain' must hold float64 values");
    inputs.X_target = nullptr;
    n4m_matrix_view_t strided = in.X;
    strided.row_stride = 0;
    inputs.X = &strided;
    expect(fit("preprocessing.scaling.simple_scale", inputs), N4M_ERR_STRIDE_INVALID,
           "matrix 'X' is not a valid view");
    // Procedures share the adapters and the same validation.
    int32_t index = -1;
    CHECK(n4m_method_find("augmentation.drift.linear_drift", &index) == N4M_OK);
    inputs.X = &X32;
    n4m_method_result_t* result = nullptr;
    expect(n4m_procedure_run(ctx, index, nullptr, &inputs, &result), N4M_ERR_DTYPE_MISMATCH,
           "matrix 'X' must hold float64 values");
    CHECK(result == nullptr);

    // New rows at predict / transform / mask time.
    n4m_estimator_t* est = nullptr;
    inputs.X = &in.X;
    inputs.Y = &in.Y;
    CHECK(n4m_estimator_create(ctx, "models.pls.pls_regression", nullptr, &est) == N4M_OK);
    CHECK(n4m_estimator_fit(ctx, est, &inputs) == N4M_OK);
    std::vector<double> out(static_cast<size_t>(kTrain), -7.0);
    auto O = view(out.data(), kTrain, 1);
    expect(n4m_estimator_predict(ctx, est, &X32, &O), N4M_ERR_DTYPE_MISMATCH,
           "matrix 'X' must hold float64 values");
    for (double v : out) CHECK(v == -7.0);
    n4m_estimator_destroy(est);
    CHECK(n4m_estimator_create(ctx, "filters.y_outlier", nullptr, &est) == N4M_OK);
    CHECK(n4m_estimator_fit(ctx, est, &inputs) == N4M_OK);
    std::vector<uint8_t> mask(kTrain);
    expect(n4m_estimator_apply_mask(ctx, est, &in.X, &Y32, mask.data(), kTrain),
           N4M_ERR_DTYPE_MISMATCH, "matrix 'y' must hold float64 values");
    n4m_estimator_destroy(est);
}

// Parameters a kernel may legitimately reduce bound the effective value the
// state holds: import refuses a request below it (and a changed reference or
// absolute threshold), with the parameter named.
void test_bounded_parameters(n4m_context_t* ctx, Inputs& in) {
    struct Case {
        const char* method_id;
        const char* param;
        double fitted;   // value set before the fit (NaN: default)
        double mutated;  // value written into the payload
    };
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const Case cases[] = {
        {"models.classification.pls_qda", "n_components", nan, 1},
        {"preprocessing.feature_selection.flexible_pca", "n_components", nan, 2.0},
        {"preprocessing.wavelets.wavelet_svd", "n_components", nan, 2.0},
        {"preprocessing.orthogonalization.osc", "n_components", 2, 1},
        {"preprocessing.orthogonalization.osc", "scale", nan, 0},
        {"filters.high_leverage", "absolute_threshold", 0.5, 0.25},
        {"filters.high_leverage", "center", nan, 0},
        {"selection.vip_spa", "top_k", 4, 1},
        {"selection.cars", "min_features", 4, 12},
        {"selection.uve", "min_features", 4, 12},
        {"preprocessing.alignment.xcorr_align", "reference", 1.0, 2.0},
        {"models.transfer.pds", "window_half_width", 5, 1},
    };
    for (const Case& c : cases) {
        current_ = std::string("bounded ") + c.method_id + "." + c.param;
        int32_t index = -1;
        CHECK(n4m_method_find(c.method_id, &index) == N4M_OK);
        n4m_method_info_v1_t info{};
        info.struct_size = sizeof(info);
        CHECK(n4m_method_info_v1(index, &info) == N4M_OK);
        n4m_params_t* params = nullptr;
        CHECK(n4m_params_create(ctx, index, &params) == N4M_OK);
        fill_required(index, params);
        int32_t slot = -1;
        n4m_param_info_v1_t pi{};
        for (int32_t p = 0; p < info.n_params; ++p) {
            pi = n4m_param_info_v1_t{};
            pi.struct_size = sizeof(pi);
            CHECK(n4m_method_param_info_v1(index, p, &pi) == N4M_OK);
            if (std::strcmp(pi.name, c.param) == 0) {
                slot = p;
                break;
            }
        }
        CHECK(slot >= 0);
        const std::vector<double> reference(kCols, c.fitted);
        if (!std::isnan(c.fitted)) {
            if (pi.type == N4M_METHOD_PARAM_DOUBLE) {
                CHECK(n4m_params_set_double(params, c.param, c.fitted) == N4M_OK);
            } else if (pi.type == N4M_METHOD_PARAM_DOUBLE_ARRAY) {
                CHECK(n4m_params_set_double_array(params, c.param, reference.data(), kCols) ==
                      N4M_OK);
            } else {
                CHECK(n4m_params_set_int(params, c.param, static_cast<int64_t>(c.fitted)) == N4M_OK);
            }
        }
        n4m_estimator_t* est = nullptr;
        CHECK(n4m_estimator_create(ctx, c.method_id, params, &est) == N4M_OK);
        n4m_params_destroy(params);
        n4m_fit_inputs_v1_t inputs = in.for_method(info);
        CHECK(n4m_estimator_fit(ctx, est, &inputs) == N4M_OK);
        std::vector<unsigned char> bytes = export_bytes(ctx, est);
        n4m_estimator_destroy(est);
        const ParamSlot at = param_slots(bytes)[static_cast<size_t>(slot)];
        if (pi.type == N4M_METHOD_PARAM_DOUBLE || pi.type == N4M_METHOD_PARAM_DOUBLE_ARRAY) {
            uint64_t bits = 0;
            std::memcpy(&bits, &c.mutated, 8);
            put_le(bytes, at.offset, bits);
        } else {
            put_le(bytes, at.offset, static_cast<uint64_t>(static_cast<int64_t>(c.mutated)));
        }
        reseal(bytes);
        n4m_estimator_t* back = nullptr;
        CHECK(n4m_estimator_import_from_buffer(ctx, bytes.data(), bytes.size(), &back) ==
              N4M_ERR_CORRUPT_BUFFER);
        CHECK(back == nullptr);
    }
}

// ABI 2.14 parameter descriptors: the 2.13 layout stays accepted, recorded
// parameters are flagged, and seeds are optional (no published default).
void test_param_descriptors() {
    current_ = "param descriptors";
    int32_t index = -1;
    CHECK(n4m_method_find("models.pls.pls_regression", &index) == N4M_OK);
    n4m_param_info_v1_t pi{};
    pi.struct_size = offsetof(n4m_param_info_v1_t, recorded);
    CHECK(n4m_method_param_info_v1(index, 0, &pi) == N4M_OK);
    CHECK(std::strcmp(pi.name, "n_components") == 0 && pi.recorded == 0);
    pi.struct_size = sizeof(pi);
    CHECK(n4m_method_param_info_v1(index, 0, &pi) == N4M_OK && pi.recorded == 1);
    CHECK(n4m_method_find("augmentation.noise.gaussian_noise", &index) == N4M_OK);
    n4m_method_info_v1_t info{};
    info.struct_size = sizeof(info);
    CHECK(n4m_method_info_v1(index, &info) == N4M_OK);
    bool seen = false;
    for (int32_t p = 0; p < info.n_params; ++p) {
        pi = n4m_param_info_v1_t{};
        pi.struct_size = sizeof(pi);
        CHECK(n4m_method_param_info_v1(index, p, &pi) == N4M_OK);
        if (std::strcmp(pi.name, "seed") != 0) continue;
        seen = true;
        CHECK(pi.has_default == 1 && pi.default_length == 0);
        int64_t count = -1;
        CHECK(n4m_method_param_default_int(index, p, nullptr, 0, &count) == N4M_OK && count == 0);
    }
    CHECK(seen);
    // An unset seed runs as seed 0.
    n4m_params_t* params = nullptr;
    CHECK(n4m_params_create(nullptr, index, &params) == N4M_OK);
    int64_t seed = -1, count = 0;
    CHECK(n4m_params_get_int(params, "seed", &seed, 1, &count) == N4M_OK && seed == 0);
    n4m_params_destroy(params);
}

}  // namespace

// The generic adapter must propagate the shared SNV extent refusal. Every
// row is observed before asserting, so an old kernel cannot hide later reds.
void test_snv_parameter_domain(n4m_context_t* ctx) {
    double x[16] = {1, 3, 8, 20, 10, 11, 13, 18, -7, -3, 2, 9, 5, 5, 5, 5};
    auto xv = view(x, 4, 4);
    std::vector<std::string> failures;
    for (bool mean : {false, true})
        for (bool scale : {false, true})
            for (int64_t ddof : {int64_t{0},
                                 int64_t{1},
                                 int64_t{3},
                                 int64_t{4},
                                 int64_t{5},
                                 int64_t{2147483647}}) {
                int32_t index = -1;
                CHECK(n4m_method_find("preprocessing.scatter.snv", &index) == N4M_OK);
                n4m_params_t* params = nullptr;
                CHECK(n4m_params_create(ctx, index, &params) == N4M_OK);
                CHECK(n4m_params_set_bool(params, "with_mean", mean) == N4M_OK);
                CHECK(n4m_params_set_bool(params, "with_std", scale) == N4M_OK);
                CHECK(n4m_params_set_int(params, "ddof", ddof) == N4M_OK);
                n4m_estimator_t* est = nullptr;
                CHECK(n4m_estimator_create(ctx, "preprocessing.scatter.snv", params, &est)
                      == N4M_OK);
                n4m_params_destroy(params);
                n4m_fit_inputs_v1_t input{};
                input.struct_size = sizeof input;
                input.X = &xv;
                CHECK(n4m_estimator_fit(ctx, est, &input) == N4M_OK);
                double out[16];
                for (double& value : out)
                    value = -12345.0;
                auto ov = view(out, 4, 4);
                const auto status = n4m_estimator_transform(ctx, est, &xv, &ov);
                const bool invalid = scale && ddof >= 4;
                bool ok = status == (invalid ? N4M_ERR_INVALID_ARGUMENT : N4M_OK);
                if (invalid)
                    for (double value : out)
                        ok = ok && value == -12345.0;
                if (!invalid && !scale) {
                    const double centers[4] = {8.0, 13.0, 0.25, 5.0};
                    for (int i = 0; i < 16; ++i)
                        ok = ok && out[i] == x[i] - (mean ? centers[i / 4] : 0.0);
                }
                n4m_estimator_destroy(est);
                const auto label = "snv/mean=" + std::to_string(mean) + "/std="
                                   + std::to_string(scale) + "/ddof=" + std::to_string(ddof);
                std::printf("%s status=%d %s\n", label.c_str(), status, ok ? "PASS" : "FAIL");
                if (!ok)
                    failures.push_back(label);
            }
    for (const auto& failure : failures)
        std::fprintf(stderr, "FAIL %s\n", failure.c_str());
    CHECK(failures.empty());
}

// The generic MB-PLS role must fit the canonical NIPALS multi-block path: the
// public C API with an explicit scale_x = 0 configuration, never the legacy
// block-balanced path (scale_x = 1) that test_models_extra::test_mb_pls pins
// on the same fixture. Same engine: a configuration wiring check, not an
// independent reference. Tolerance is the fixture's MB-PLS gate (1e-8).
void test_mb_pls_role_is_canonical(n4m_context_t* ctx) {
    for (const auto& fx : ::n4m::test::fixtures::kMbPlsFixtures) {
        current_ = std::string("models.multiblock.mb_pls/canonical_role/") + fx.id;
        std::printf("%s\n", current_.c_str());
        std::vector<double> x(fx.X.values, fx.X.values + fx.X.size);
        std::vector<double> y(fx.Y.values, fx.Y.values + fx.Y.size);
        const int64_t n = fx.X.rows;
        const int64_t q = fx.Y.cols;
        auto X = view(x.data(), n, fx.X.cols);
        auto Y = view(y.data(), n, q);
        const auto n_blocks = static_cast<int64_t>(fx.block_sizes.size);
        // In-sample predictions of the explicit low-level C API.
        auto api = [&](int32_t scale_x) {
            n4m_config_t* cfg = nullptr;
            CHECK(n4m_config_create(&cfg) == N4M_OK);
            CHECK(n4m_config_set_n_components(cfg, fx.n_components) == N4M_OK);
            CHECK(n4m_config_set_scale_x(cfg, scale_x) == N4M_OK);
            n4m_method_result_t* result = nullptr;
            const n4m_status_t st = n4m_estimators_mb_pls_fit(
                ctx, cfg, &X, &Y, fx.block_sizes.values, n_blocks, &result);
            n4m_config_destroy(cfg);
            CHECK(st == N4M_OK);
            const double* data = nullptr;
            int64_t rows = 0, cols = 0;
            CHECK(n4m_method_result_get_double_matrix(result, "predictions", &data, &rows,
                                                      &cols) == N4M_OK);
            CHECK(rows == n && cols == q);
            std::vector<double> out(data, data + rows * cols);
            n4m_method_result_destroy(result);
            return out;
        };
        const auto canonical = api(0);
        const auto legacy = api(1);
        int32_t index = -1;
        CHECK(n4m_method_find("models.multiblock.mb_pls", &index) == N4M_OK);
        n4m_params_t* params = nullptr;
        CHECK(n4m_params_create(ctx, index, &params) == N4M_OK);
        CHECK(n4m_params_set_int(params, "n_components", fx.n_components) == N4M_OK);
        n4m_estimator_t* est = nullptr;
        const n4m_status_t created =
            n4m_estimator_create(ctx, "models.multiblock.mb_pls", params, &est);
        n4m_params_destroy(params);
        CHECK(created == N4M_OK);
        n4m_fit_inputs_v1_t input{};
        input.struct_size = sizeof input;
        input.X = &X;
        input.Y = &Y;
        input.block_sizes = fx.block_sizes.values;
        input.n_blocks = n_blocks;
        CHECK(n4m_estimator_fit(ctx, est, &input) == N4M_OK);
        std::vector<double> role(canonical.size()), replay(canonical.size());
        auto R = view(role.data(), n, q);
        CHECK(n4m_estimator_predict(ctx, est, &X, &R) == N4M_OK);
        double from_legacy = 0.0;
        for (size_t i = 0; i < role.size(); ++i) {
            const double diff = std::fabs(role[i] - canonical[i]);
            const double scale = std::max({std::fabs(role[i]), std::fabs(canonical[i]), 1.0});
            CHECK(diff <= 1e-8 || diff <= 1e-8 * scale);
            from_legacy = std::max(from_legacy, std::fabs(role[i] - legacy[i]));
        }
        // An unconfigured (legacy) factory cannot satisfy both checks.
        CHECK(from_legacy > 1e-6);
        const auto bytes = export_bytes(ctx, est);
        n4m_estimator_t* back = nullptr;
        CHECK(n4m_estimator_import_from_buffer(ctx, bytes.data(), bytes.size(), &back) == N4M_OK);
        auto P = view(replay.data(), n, q);
        CHECK(n4m_estimator_predict(ctx, back, &X, &P) == N4M_OK);
        CHECK(std::memcmp(role.data(), replay.data(), role.size() * sizeof(double)) == 0);
        n4m_estimator_destroy(back);
        n4m_estimator_destroy(est);
    }
}

// MI32_ESTIMATOR_V1: public ABI only; no hidden conversion helper export.
static void test_raw_i32_case(
    n4m_context_t* ctx, const char* id, const char* parameter, int64_t value, bool supervised) {
    current_ = std::string("MI32_ESTIMATOR_V1/") + id + "/" + std::to_string(value);
    std::printf("%s\n", current_.c_str());
    int32_t index = -1;
    CHECK(n4m_method_find(id, &index) == N4M_OK);
    n4m_params_t* raw_params = nullptr;
    CHECK(n4m_params_create(ctx, index, &raw_params) == N4M_OK);
    std::unique_ptr<n4m_params_t, decltype(&n4m_params_destroy)> params(raw_params,
                                                                        n4m_params_destroy);
    CHECK(n4m_params_set_int(params.get(), parameter, value) == N4M_OK);
    int64_t stored = 0, count = 0;
    CHECK(n4m_params_get_int(params.get(), parameter, &stored, 1, &count) == N4M_OK);
    CHECK(count == 1 && stored == value);
    CHECK(n4m_params_validate(ctx, params.get()) == N4M_OK);
    n4m_estimator_t* raw = nullptr;
    CHECK(n4m_estimator_create(ctx, id, params.get(), &raw) == N4M_OK);
    std::unique_ptr<n4m_estimator_t, decltype(&n4m_estimator_destroy)> est(raw,
                                                                           n4m_estimator_destroy);
    double x[] = {1, 3, 5, 2, 2, 4, 6, 9, 3, 1, 0, 4, 9, 8, 1, 2, 4, 4, 7, 1, 0, 2, 3, 8};
    const std::vector<double> original(x, x + 24);
    double y[] = {-4, -4, 3, 8, -3, -3};
    auto X = view(x, 6, 4), Y = view(y, 6, 1);
    n4m_fit_inputs_v1_t in{};
    in.struct_size = sizeof(in);
    in.X = &X;
    in.Y = supervised ? &Y : nullptr;
    CHECK(n4m_estimator_fit(ctx, est.get(), &in) == N4M_ERR_INVALID_ARGUMENT);
    CHECK(std::strcmp(n4m_context_last_error(ctx), "integer parameter exceeds int32") == 0);
    int32_t fitted = -1;
    CHECK(n4m_estimator_is_fitted(est.get(), &fitted) == N4M_OK && fitted == 0);
    double output[24];
    std::fill(output, output + 24, -777.25);
    auto out = view(output, 6, supervised ? 1 : 4);
    const auto st = supervised ? n4m_estimator_predict(ctx, est.get(), &X, &out)
                               : n4m_estimator_transform(ctx, est.get(), &X, &out);
    CHECK(st == N4M_ERR_NOT_FITTED);
    CHECK(std::all_of(output, output + 24, [](double v) {
        return v == -777.25;
    }));
    CHECK(std::equal(original.begin(), original.end(), x));
}

static void test_raw_i32_public_upper_and_wide_seed(n4m_context_t* ctx) {
    current_ = "MI32_ESTIMATOR_V1/upper-and-wide-seed";
    std::puts(current_.c_str());
    int32_t index = -1;
    CHECK(n4m_method_find("preprocessing.scatter.snv", &index) == N4M_OK);
    n4m_params_t* raw_params = nullptr;
    CHECK(n4m_params_create(ctx, index, &raw_params) == N4M_OK);
    std::unique_ptr<n4m_params_t, decltype(&n4m_params_destroy)> params(raw_params,
                                                                        n4m_params_destroy);
    CHECK(n4m_params_set_int(params.get(), "ddof", std::numeric_limits<int32_t>::max()) == N4M_OK);
    CHECK(n4m_params_set_bool(params.get(), "with_std", 0) == N4M_OK);
    n4m_estimator_t* raw = nullptr;
    CHECK(n4m_estimator_create(ctx, "preprocessing.scatter.snv", params.get(), &raw) == N4M_OK);
    std::unique_ptr<n4m_estimator_t, decltype(&n4m_estimator_destroy)> est(raw,
                                                                           n4m_estimator_destroy);
    double x[] = {1, 3, 5, 2, 4, 6}, out[6]{};
    auto X = view(x, 2, 3), Y = view(out, 2, 3);
    n4m_fit_inputs_v1_t in{};
    in.struct_size = sizeof(in);
    in.X = &X;
    CHECK(n4m_estimator_fit(ctx, est.get(), &in) == N4M_OK);
    CHECK(n4m_estimator_transform(ctx, est.get(), &X, &Y) == N4M_OK);
    const double expected[] = {-2, 0, 2, -2, 0, 2};
    CHECK(std::equal(out, out + 6, expected));
    CHECK(n4m_method_find("augmentation.noise.gaussian_noise", &index) == N4M_OK);
    n4m_params_t* seed_raw = nullptr;
    CHECK(n4m_params_create(ctx, index, &seed_raw) == N4M_OK);
    std::unique_ptr<n4m_params_t, decltype(&n4m_params_destroy)> seed(seed_raw, n4m_params_destroy);
    const int64_t wide = (int64_t{1} << 40) + 17;
    CHECK(n4m_params_set_int(seed.get(), "seed", wide) == N4M_OK);
    CHECK(n4m_params_validate(ctx, seed.get()) == N4M_OK);
    int64_t stored = 0, count = 0;
    CHECK(n4m_params_get_int(seed.get(), "seed", &stored, 1, &count) == N4M_OK);
    CHECK(stored == wide && count == 1);
    n4m_method_result_t* result = nullptr;
    CHECK(n4m_procedure_run(ctx, index, seed.get(), &in, &result) == N4M_OK);
    CHECK(result != nullptr);
    n4m_method_result_destroy(result);
}

static void test_raw_i32_failed_refit(n4m_context_t* ctx, Inputs& fixture, int64_t bad) {
    current_ = "MI32_ESTIMATOR_V1/feature-groups/" + std::to_string(bad);
    std::puts(current_.c_str());
    const char* id = "models.sparse.group_sparse_pls";
    n4m_estimator_t* raw = nullptr;
    CHECK(n4m_estimator_create(ctx, id, nullptr, &raw) == N4M_OK);
    std::unique_ptr<n4m_estimator_t, decltype(&n4m_estimator_destroy)> est(raw,
                                                                           n4m_estimator_destroy);
    auto groups = fixture.feature_groups;
    n4m_fit_inputs_v1_t in{};
    in.struct_size = sizeof(in);
    in.X = &fixture.X;
    in.Y = &fixture.Y;
    in.feature_groups = groups.data();
    in.n_feature_groups = kCols;
    CHECK(n4m_estimator_fit(ctx, est.get(), &in) == N4M_OK);
    std::vector<double> before(static_cast<size_t>(kTrain)), after(before.size(), -777.25);
    auto A = view(before.data(), kTrain, 1), Z = view(after.data(), kTrain, 1);
    CHECK(n4m_estimator_predict(ctx, est.get(), &fixture.X, &A) == N4M_OK);
    groups[0] = bad;
    CHECK(n4m_estimator_fit(ctx, est.get(), &in) == N4M_ERR_INVALID_ARGUMENT);
    CHECK(std::strcmp(n4m_context_last_error(ctx), "integer parameter exceeds int32") == 0);
    int32_t fitted = 0;
    CHECK(n4m_estimator_is_fitted(est.get(), &fitted) == N4M_OK && fitted == 1);
    CHECK(n4m_estimator_predict(ctx, est.get(), &fixture.X, &Z) == N4M_OK);
    CHECK(std::memcmp(before.data(), after.data(), before.size() * sizeof(double)) == 0);
}

// Focused case callable by the existing private narrow-driver pattern.
void test_standard_scale_ddof(n4m_context_t* ctx) {
    current_ = "standard_scale/ddof";
    using Estimator = std::unique_ptr<n4m_estimator_t, decltype(&n4m_estimator_destroy)>;
    using Parameters = std::unique_ptr<n4m_params_t, decltype(&n4m_params_destroy)>;
    int32_t index = -1;
    CHECK(n4m_method_find("preprocessing.scaling.standard_scale", &index) == N4M_OK);
    auto make = [&](int64_t ddof, bool explicit_ddof, bool mean, bool scale) {
        n4m_params_t* raw_params = nullptr;
        CHECK(n4m_params_create(ctx, index, &raw_params) == N4M_OK);
        Parameters params(raw_params, n4m_params_destroy);
        int64_t value = -1, count = 0;
        CHECK(n4m_params_get_int(params.get(), "ddof", &value, 1, &count) == N4M_OK);
        CHECK(value == 0 && count == 1);
        if (explicit_ddof) CHECK(n4m_params_set_int(params.get(), "ddof", ddof) == N4M_OK);
        CHECK(n4m_params_set_bool(params.get(), "with_mean", mean ? 1 : 0) == N4M_OK);
        CHECK(n4m_params_set_bool(params.get(), "with_std", scale ? 1 : 0) == N4M_OK);
        n4m_estimator_t* raw = nullptr;
        CHECK(n4m_estimator_create(ctx, "preprocessing.scaling.standard_scale", params.get(),
                                   &raw) == N4M_OK);
        return Estimator(raw, n4m_estimator_destroy);
    };
    const double offset = 1099511627776.0;  // 2^40, exactly representable.
    double train[] = {offset, 7, 0, offset + 2, 7, 2,
                      offset + 4, 7, 4, offset + 6, 7, 6};
    double heldout[] = {offset + 1, 7, 1, offset + 5, 7, 5};
    auto X = view(train, 4, 3), H = view(heldout, 2, 3);
    n4m_fit_inputs_v1_t inputs{};
    inputs.struct_size = sizeof(inputs);
    inputs.X = &X;
    auto transform = [&](const n4m_estimator_t* est) {
        std::vector<double> out(6);
        auto O = view(out.data(), 2, 3);
        CHECK(n4m_estimator_transform(ctx, est, &H, &O) == N4M_OK);
        return out;
    };
    for (int64_t ddof : {int64_t{0}, int64_t{1}}) {
        for (bool mean : {false, true}) {
            for (bool scale : {false, true}) {
                auto est = make(ddof, true, mean, scale);
                CHECK(n4m_estimator_fit(ctx, est.get(), &inputs) == N4M_OK);
                const auto out = transform(est.get());
                for (size_t r = 0; r < 2; ++r) {
                    for (size_t c = 0; c < 3; ++c) {
                        const double center = c == 0 ? offset + 3 : c == 1 ? 7 : 3;
                        const double sd = c == 1 ? 1 : std::sqrt(20.0 / (4 - ddof));
                        const double expected = (heldout[r * 3 + c] - (mean ? center : 0)) /
                                                (scale ? sd : 1);
                        CHECK(std::fabs(out[r * 3 + c] - expected) <=
                              1e-12 * (1 + std::fabs(expected)));
                    }
                    CHECK(out[r * 3 + 1] == (mean ? 0 : 7));
                }
                const auto bytes = export_bytes(ctx, est.get());
                n4m_estimator_t* raw_back = nullptr;
                CHECK(n4m_estimator_import_from_buffer(ctx, bytes.data(), bytes.size(),
                                                       &raw_back) == N4M_OK);
                Estimator back(raw_back, n4m_estimator_destroy);
                const auto replay = transform(back.get());
                CHECK(std::memcmp(out.data(), replay.data(), out.size() * sizeof(double)) == 0);
                CHECK(export_bytes(ctx, back.get()) == bytes);
                if (ddof == 0) {
                    auto defaults = make(0, false, mean, scale);
                    CHECK(n4m_estimator_fit(ctx, defaults.get(), &inputs) == N4M_OK);
                    const auto omitted = transform(defaults.get());
                    CHECK(std::memcmp(out.data(), omitted.data(), out.size() * sizeof(double)) == 0);
                    CHECK(export_bytes(ctx, defaults.get()) == bytes);
                }
            }
        }
    }
    double singleton[] = {0.1, 7};
    auto S = view(singleton, 1, 2);
    inputs.X = &S;
    for (bool scale : {false, true}) {
        auto est = make(0, false, true, scale);
        CHECK(n4m_estimator_fit(ctx, est.get(), &inputs) == N4M_OK);
        double zeros[] = {-1, -1};
        auto Z = view(zeros, 1, 2);
        CHECK(n4m_estimator_transform(ctx, est.get(), &S, &Z) == N4M_OK);
        CHECK(zeros[0] == 0 && zeros[1] == 0);
        for (int64_t ddof : {int64_t{1}, int64_t{2}, std::numeric_limits<int64_t>::max()}) {
            auto invalid = make(ddof, true, true, scale);
            CHECK(n4m_estimator_fit(ctx, invalid.get(), &inputs) == N4M_ERR_INVALID_ARGUMENT);
        }
    }
    n4m_params_t* raw_params = nullptr;
    CHECK(n4m_params_create(ctx, index, &raw_params) == N4M_OK);
    Parameters params(raw_params, n4m_params_destroy);
    CHECK(n4m_params_set_int(params.get(), "ddof", -1) == N4M_ERR_INVALID_ARGUMENT);
    CHECK(n4m_params_set_double(params.get(), "ddof", 0.5) == N4M_ERR_INVALID_ARGUMENT);
}

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
    run([&] { test_standard_scale_ddof(ctx); });
    run([&] {
        test_snv_parameter_domain(ctx);
    });
    run([&] { test_pls_fit_simple_equivalence(ctx, in); });
    run([&] { test_in_sample_equivalence(ctx, in); });
    run([&] { test_aom_in_sample(ctx, in); });
    run([&] { test_refit_and_input_lengths(ctx, in); });
    run([&] { test_input_views(ctx, in); });
    run([&] { test_param_descriptors(); });
    run([&] { test_bounded_parameters(ctx, in); });
    run([&] { test_mb_pls_role_is_canonical(ctx); });
    for (int64_t value :
         {int64_t{2147483648}, int64_t{4294967296}, std::numeric_limits<int64_t>::max()}) {
        run([&] {
            test_raw_i32_case(ctx, "preprocessing.scatter.snv", "ddof", value, false);
        });
        run([&] {
            test_raw_i32_case(ctx, "selection.vip_spa", "top_k", value, true);
        });
        run([&] {
            test_raw_i32_case(ctx, "models.regularized.robust_pls", "max_irls_iter", value, true);
        });
    }
    run([&] {
        test_raw_i32_public_upper_and_wide_seed(ctx);
    });
    for (int64_t value :
         {int64_t{-2147483649LL}, std::numeric_limits<int64_t>::min(), int64_t{2147483648}})
        run([&] {
            test_raw_i32_failed_refit(ctx, in, value);
        });

    int32_t count = 0;
    n4m_method_count(&count);
    for (int32_t i = 0; i < count; ++i) {
        run([&] { conformance(ctx, in, i); });
        ++checked;
    }
    n4m_context_destroy(ctx);
    if (recorded_refusals == 0) ++failures;
    std::printf("n4m_estimator_tests: %d methods, %d recorded-parameter mutations refused, "
                "%d failures\n",
                checked, recorded_refusals, failures);
    return failures == 0 ? 0 : 1;
}
