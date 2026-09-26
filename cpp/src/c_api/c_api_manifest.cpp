// SPDX-License-Identifier: CECILL-2.1
//
// n4m_method_manifest_json: the native manifest rendered once, in the
// library, so every binding and tool (CLI, Studio/Web node generators, DAG-ML
// controller specs) reads the same document.

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>
#include <string>

#include "n4m/n4m.h"

namespace {

void add_string(std::string& o, const char* s) {
    o += '"';
    for (; *s != '\0'; ++s) {
        const unsigned char c = static_cast<unsigned char>(*s);
        if (c == '"' || c == '\\') {
            o += '\\';
            o += static_cast<char>(c);
        } else if (c < 0x20) {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "\\u%04x", c);
            o += buf;
        } else {
            o += static_cast<char>(c);
        }
    }
    o += '"';
}

// NaN (unset bound, or an optional value's default) is null.
void add_number(std::string& o, double v) {
    if (v != v) {
        o += "null";
        return;
    }
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.17g", v);
    o += buf;
}

const char* param_type_name(int32_t t) {
    switch (t) {
        case N4M_METHOD_PARAM_INT: return "int";
        case N4M_METHOD_PARAM_DOUBLE: return "double";
        case N4M_METHOD_PARAM_BOOL: return "bool";
        case N4M_METHOD_PARAM_ENUM: return "enum";
        case N4M_METHOD_PARAM_INT_ARRAY: return "int_array";
        default: return "double_array";
    }
}

n4m_status_t add_default(std::string& o, int32_t method, int32_t param,
                         const n4m_param_info_v1_t& pi) {
    if (!pi.has_default) {
        o += "null";
        return N4M_OK;
    }
    const bool array = pi.type == N4M_METHOD_PARAM_INT_ARRAY ||
                       pi.type == N4M_METHOD_PARAM_DOUBLE_ARRAY;
    const auto n = static_cast<std::size_t>(pi.default_length);
    int64_t count = 0;
    if (array) o += '[';
    if (pi.type == N4M_METHOD_PARAM_DOUBLE || pi.type == N4M_METHOD_PARAM_DOUBLE_ARRAY) {
        std::string values;
        double* buf = new double[n > 0 ? n : 1];
        const n4m_status_t st =
            n4m_method_param_default_double(method, param, buf, pi.default_length, &count);
        for (std::size_t k = 0; st == N4M_OK && k < n; ++k) {
            if (k > 0) values += ',';
            add_number(values, buf[k]);
        }
        delete[] buf;
        if (st != N4M_OK) return st;
        o += values;
    } else {
        int64_t* buf = new int64_t[n > 0 ? n : 1];
        const n4m_status_t st =
            n4m_method_param_default_int(method, param, buf, pi.default_length, &count);
        for (std::size_t k = 0; st == N4M_OK && k < n; ++k) {
            if (k > 0) o += ',';
            if (pi.type == N4M_METHOD_PARAM_BOOL) {
                o += buf[k] != 0 ? "true" : "false";
            } else if (pi.type == N4M_METHOD_PARAM_ENUM) {
                add_string(o, pi.choices[buf[k]]);
            } else {
                char tmp[24];
                std::snprintf(tmp, sizeof(tmp), "%" PRId64, buf[k]);
                o += tmp;
            }
        }
        delete[] buf;
        if (st != N4M_OK) return st;
    }
    if (array) o += ']';
    return N4M_OK;
}

n4m_status_t render(std::string& o) {
    static const char* const kRoles[] = {"transformer", "regressor",  "classifier",
                                         "selector",    "sample_filter", "splitter",
                                         "augmenter",   "generic"};
    static const char* const kCaps[] = {"transform", "predict", "predict_proba",
                                        "decision_function", "predict_labels",
                                        "selected_indices", "apply_mask", "serializable",
                                        "affine", "retains_training_rows"};
    static const char* const kInputs[] = {"y",      "labels", "sample_weight",
                                          "groups", "feature_groups", "blocks",
                                          "axis",   "target_domain",  "fold_ids"};
    static const char* const kReq[] = {"none", "optional", "required"};
    int32_t count = 0;
    n4m_status_t st = n4m_method_count(&count);
    if (st != N4M_OK) return st;
    char abi[64];
    std::snprintf(abi, sizeof(abi), "{\"abi\":\"%u.%u.%u\",\"methods\":[",
                  n4m_get_abi_version_major(), n4m_get_abi_version_minor(),
                  n4m_get_abi_version_patch());
    o += abi;
    for (int32_t i = 0; i < count; ++i) {
        n4m_method_info_v1_t info{};
        info.struct_size = sizeof(info);
        st = n4m_method_info_v1(i, &info);
        if (st != N4M_OK) return st;
        o += i > 0 ? ",\n{\"method_id\":" : "\n{\"method_id\":";
        add_string(o, info.method_id);
        o += ",\"fq_name\":";
        add_string(o, info.fq_name);
        o += info.kind == N4M_METHOD_ESTIMATOR ? ",\"kind\":\"estimator\",\"roles\":["
                                               : ",\"kind\":\"procedure\",\"roles\":[";
        bool first = true;
        for (uint32_t r = 0; r < 8; ++r) {
            if ((info.roles & (1u << r)) == 0) continue;
            o += first ? "\"" : ",\"";
            o += kRoles[r];
            o += '"';
            first = false;
        }
        // Role interface -> DAG-ML node kind (docs/abi/estimator_roles_design.md, D0b);
        // generic procedures have none.
        o += "],\"node_kinds\":[";
        first = true;
        const auto kind = [&](uint32_t mask, const char* name) {
            if ((info.roles & mask) == 0) return;
            o += first ? "\"" : ",\"";
            o += name;
            o += '"';
            first = false;
        };
        kind(N4M_ROLE_REGRESSOR | N4M_ROLE_CLASSIFIER, "model");
        kind(N4M_ROLE_TRANSFORMER | N4M_ROLE_SELECTOR, "transform");
        kind(N4M_ROLE_SAMPLE_FILTER, "exclude");
        kind(N4M_ROLE_SPLITTER, "split");
        kind(N4M_ROLE_AUGMENTER, "augmentation");
        o += "],\"capabilities\":[";
        first = true;
        for (uint32_t c = 0; c < 10; ++c) {
            if ((info.capabilities & (UINT64_C(1) << c)) == 0) continue;
            o += first ? "\"" : ",\"";
            o += kCaps[c];
            o += '"';
            first = false;
        }
        o += "],\"state_format\":";
        add_string(o, info.state_format);
        o += ",\"inputs\":{";
        for (int k = 0; k < N4M_FIT_INPUT_COUNT; ++k) {
            if (k > 0) o += ',';
            add_string(o, kInputs[k]);
            o += ':';
            add_string(o, kReq[info.inputs[k]]);
        }
        o += "},\"params\":[";
        for (int32_t p = 0; p < info.n_params; ++p) {
            n4m_param_info_v1_t pi{};
            pi.struct_size = sizeof(pi);
            st = n4m_method_param_info_v1(i, p, &pi);
            if (st != N4M_OK) return st;
            o += p > 0 ? ",{\"name\":" : "{\"name\":";
            add_string(o, pi.name);
            o += ",\"type\":\"";
            o += param_type_name(pi.type);
            o += pi.has_default ? "\",\"required\":false,\"default\":"
                                : "\",\"required\":true,\"default\":";
            st = add_default(o, i, p, pi);
            if (st != N4M_OK) return st;
            o += ",\"min\":";
            add_number(o, pi.min_value);
            o += ",\"max\":";
            add_number(o, pi.max_value);
            o += ",\"choices\":[";
            for (int32_t c = 0; c < pi.n_choices; ++c) {
                if (c > 0) o += ',';
                add_string(o, pi.choices[c]);
            }
            o += "]}";
        }
        o += "]}";
    }
    o += "\n]}\n";
    return N4M_OK;
}

}  // namespace

extern "C" N4M_API n4m_status_t n4m_method_manifest_json(char* out, size_t capacity,
                                                         size_t* out_size) {
    if (out_size == nullptr) return N4M_ERR_NULL_POINTER;
    try {
        std::string json;
        const n4m_status_t st = render(json);
        if (st != N4M_OK) return st;
        *out_size = json.size();
        if (out == nullptr) return N4M_OK;
        if (capacity < json.size()) return N4M_ERR_INVALID_ARGUMENT;
        std::memcpy(out, json.data(), json.size());
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}
