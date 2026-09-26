// SPDX-License-Identifier: CECILL-2.1
//
// n4m_cli — tiny introspection tool. Phase 0 (bootstrap) supports:
//   --version      print "nirs4all-methods <project> (ABI <abi>)"
//   --abi-info     print ABI / backend / dtype metadata
//   --selfcheck    exercise context + matrix_view lifecycle, exit non-zero on
//                  any ABI inconsistency
//   --manifest-json  print the native method manifest (generic estimator
//                  roles) as JSON; bindings generate their facades from it
//
// Subsequent phases extend --selfcheck and add a --bench subcommand.
// No third-party CLI parser is used — keep runtime dependencies aligned with
// libn4m's zero-mandatory-deps promise.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "n4m/n4m.h"

namespace {

int cmd_version() {
    printf("nirs4all-methods %s (ABI %u.%u.%u)\n",
           n4m_get_version_string(),
           n4m_get_abi_version_major(),
           n4m_get_abi_version_minor(),
           n4m_get_abi_version_patch());
    return 0;
}

int cmd_abi_info() {
    printf("=== nirs4all-methods ABI info ===\n");
    printf("project_version : %s\n",   n4m_get_version_string());
    printf("abi_version     : %u.%u.%u (int=%u)\n",
           n4m_get_abi_version_major(),
           n4m_get_abi_version_minor(),
           n4m_get_abi_version_patch(),
           n4m_get_abi_version_int());
    printf("build_info      : %s\n",   n4m_get_build_info());
    printf("git_revision    : %s\n",   n4m_get_git_revision());

    printf("--- backends ---\n");
    n4m_backend_t backends[] = {
        N4M_BACKEND_AUTO,          N4M_BACKEND_REFERENCE_CPU,
        N4M_BACKEND_NATIVE_CPU,    N4M_BACKEND_BLAS,
        N4M_BACKEND_OPENMP,        N4M_BACKEND_CUDA,
        N4M_BACKEND_OPENCL,        N4M_BACKEND_METAL,
    };
    for (size_t i = 0; i < sizeof(backends)/sizeof(backends[0]); ++i) {
        printf("  %-16s available=%d\n",
               n4m_backend_to_string(backends[i]),
               n4m_backend_is_available(backends[i]));
    }

    printf("--- dtypes ---\n");
    n4m_dtype_t dtypes[] = {
        N4M_DTYPE_F64, N4M_DTYPE_F32, N4M_DTYPE_I32, N4M_DTYPE_I64,
    };
    for (size_t i = 0; i < sizeof(dtypes)/sizeof(dtypes[0]); ++i) {
        printf("  %-8s size=%zu bytes\n",
               n4m_dtype_to_string(dtypes[i]),
               n4m_dtype_size(dtypes[i]));
    }
    return 0;
}

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "selfcheck FAIL: %s (line %d)\n", #cond, __LINE__); \
        return 1; \
    } \
} while (0)

int cmd_selfcheck() {
    // ABI compatibility self-check.
    CHECK(n4m_check_abi_compatibility(N4M_ABI_VERSION_MAJOR,
                                       N4M_ABI_VERSION_MINOR) == N4M_OK);

    // Context lifecycle.
    n4m_context_t* ctx = nullptr;
    CHECK(n4m_context_create(&ctx) == N4M_OK);
    CHECK(ctx != nullptr);

    uint64_t seed = 0;
    CHECK(n4m_context_set_seed(ctx, 42) == N4M_OK);
    CHECK(n4m_context_get_seed(ctx, &seed) == N4M_OK);
    CHECK(seed == 42);

    n4m_backend_t backend = N4M_BACKEND_AUTO;
    CHECK(n4m_context_set_backend(ctx, N4M_BACKEND_REFERENCE_CPU) == N4M_OK);
    CHECK(n4m_context_get_backend(ctx, &backend) == N4M_OK);
    CHECK(backend == N4M_BACKEND_REFERENCE_CPU);

    int32_t nt = 0;
    CHECK(n4m_context_set_num_threads(ctx, 1) == N4M_OK);
    CHECK(n4m_context_get_num_threads(ctx, &nt) == N4M_OK);
    CHECK(nt == 1);

    // Matrix view validation: 3x4 row-major double matrix.
    double data[3 * 4] = {0};
    n4m_matrix_view_t view;
    CHECK(n4m_matrix_view_init_rowmajor(&view, data, 3, 4, N4M_DTYPE_F64) == N4M_OK);
    CHECK(n4m_matrix_view_validate(&view) == N4M_OK);

    // Reject negative dims.
    n4m_matrix_view_t bad;
    CHECK(n4m_matrix_view_init_rowmajor(&bad, data, -1, 4, N4M_DTYPE_F64) == N4M_ERR_INVALID_ARGUMENT);

    n4m_context_destroy(ctx);

    printf("selfcheck OK (nirs4all-methods %s, ABI %u.%u.%u)\n",
           n4m_get_version_string(),
           n4m_get_abi_version_major(),
           n4m_get_abi_version_minor(),
           n4m_get_abi_version_patch());
    return 0;
}

#undef CHECK

void json_string(const char* s) {
    putchar('"');
    for (; *s != '\0'; ++s) {
        const unsigned char c = static_cast<unsigned char>(*s);
        if (c == '"' || c == '\\') {
            printf("\\%c", c);
        } else if (c < 0x20) {
            printf("\\u%04x", c);
        } else {
            putchar(c);
        }
    }
    putchar('"');
}

void json_number(double v) {
    if (v != v) {
        printf("null");
    } else {
        printf("%.17g", v);
    }
}

const char* param_type_name(int32_t t) {
    switch (t) {
        case N4M_METHOD_PARAM_INT: return "int";
        case N4M_METHOD_PARAM_DOUBLE: return "double";
        case N4M_METHOD_PARAM_BOOL: return "bool";
        case N4M_METHOD_PARAM_ENUM: return "enum";
        case N4M_METHOD_PARAM_INT_ARRAY: return "int_array";
        case N4M_METHOD_PARAM_DOUBLE_ARRAY: return "double_array";
        default: return "unknown";
    }
}

void json_default(int32_t method, int32_t param, const n4m_param_info_v1_t& pi) {
    if (!pi.has_default) {
        printf("null");
        return;
    }
    const bool array = pi.type == N4M_METHOD_PARAM_INT_ARRAY ||
                       pi.type == N4M_METHOD_PARAM_DOUBLE_ARRAY;
    const bool doubles = pi.type == N4M_METHOD_PARAM_DOUBLE ||
                         pi.type == N4M_METHOD_PARAM_DOUBLE_ARRAY;
    int64_t n = pi.default_length;
    if (array) putchar('[');
    for (int64_t k = 0; k < n; ++k) {
        if (k > 0) putchar(',');
        if (doubles) {
            double buf[64];
            int64_t count = 0;
            if (n > 64 || n4m_method_param_default_double(method, param, buf, 64, &count) != N4M_OK) {
                printf("null");
                continue;
            }
            json_number(buf[k]);
        } else {
            int64_t buf[64];
            int64_t count = 0;
            if (n > 64 || n4m_method_param_default_int(method, param, buf, 64, &count) != N4M_OK) {
                printf("null");
                continue;
            }
            if (pi.type == N4M_METHOD_PARAM_BOOL) {
                printf(buf[k] != 0 ? "true" : "false");
            } else if (pi.type == N4M_METHOD_PARAM_ENUM) {
                json_string(pi.choices[buf[k]]);
            } else {
                printf("%lld", static_cast<long long>(buf[k]));
            }
        }
    }
    if (array) putchar(']');
}

int cmd_manifest_json() {
    static const char* const kRoles[] = {"transformer", "regressor", "classifier", "selector",
                                         "sample_filter"};
    static const char* const kCaps[] = {"transform", "predict", "predict_proba",
                                        "decision_function", "predict_labels",
                                        "selected_indices", "apply_mask", "serializable",
                                        "affine", "retains_training_rows"};
    static const char* const kInputs[] = {"y", "labels", "sample_weight", "groups",
                                          "feature_groups", "blocks", "axis", "target_domain"};
    static const char* const kReq[] = {"none", "optional", "required"};
    int32_t count = 0;
    if (n4m_method_count(&count) != N4M_OK) return 1;
    printf("{\"abi\":\"%u.%u.%u\",\"methods\":[", n4m_get_abi_version_major(),
           n4m_get_abi_version_minor(), n4m_get_abi_version_patch());
    for (int32_t i = 0; i < count; ++i) {
        n4m_method_info_v1_t info{};
        info.struct_size = sizeof(info);
        if (n4m_method_info_v1(i, &info) != N4M_OK) return 1;
        printf("%s\n{\"method_id\":", i > 0 ? "," : "");
        json_string(info.method_id);
        printf(",\"fq_name\":");
        json_string(info.fq_name);
        printf(",\"kind\":\"%s\",\"roles\":[",
               info.kind == N4M_METHOD_ESTIMATOR ? "estimator" : "procedure");
        bool first = true;
        for (uint32_t r = 0; r < 5; ++r) {
            if ((info.roles & (1u << r)) == 0) continue;
            printf("%s\"%s\"", first ? "" : ",", kRoles[r]);
            first = false;
        }
        printf("],\"node_kinds\":[");
        // Role interface -> DAG-ML node kind (docs/abi/estimator_roles_design.md, D0b).
        first = true;
        if ((info.roles & (N4M_ROLE_REGRESSOR | N4M_ROLE_CLASSIFIER)) != 0) {
            printf("\"model\"");
            first = false;
        }
        if ((info.roles & (N4M_ROLE_TRANSFORMER | N4M_ROLE_SELECTOR)) != 0) {
            printf("%s\"transform\"", first ? "" : ",");
            first = false;
        }
        if ((info.roles & N4M_ROLE_SAMPLE_FILTER) != 0) {
            printf("%s\"exclude\"", first ? "" : ",");
        }
        printf("],\"capabilities\":[");
        first = true;
        for (uint32_t c = 0; c < 10; ++c) {
            if ((info.capabilities & (UINT64_C(1) << c)) == 0) continue;
            printf("%s\"%s\"", first ? "" : ",", kCaps[c]);
            first = false;
        }
        printf("],\"state_format\":");
        json_string(info.state_format);
        printf(",\"inputs\":{");
        for (int k = 0; k < N4M_FIT_INPUT_COUNT; ++k) {
            printf("%s\"%s\":\"%s\"", k > 0 ? "," : "", kInputs[k], kReq[info.inputs[k]]);
        }
        printf("},\"params\":[");
        for (int32_t p = 0; p < info.n_params; ++p) {
            n4m_param_info_v1_t pi{};
            pi.struct_size = sizeof(pi);
            if (n4m_method_param_info_v1(i, p, &pi) != N4M_OK) return 1;
            printf("%s{\"name\":", p > 0 ? "," : "");
            json_string(pi.name);
            printf(",\"type\":\"%s\",\"required\":%s,\"default\":", param_type_name(pi.type),
                   pi.has_default ? "false" : "true");
            json_default(i, p, pi);
            printf(",\"min\":");
            json_number(pi.min_value);
            printf(",\"max\":");
            json_number(pi.max_value);
            printf(",\"choices\":[");
            for (int32_t c = 0; c < pi.n_choices; ++c) {
                if (c > 0) putchar(',');
                json_string(pi.choices[c]);
            }
            printf("]}");
        }
        printf("]}");
    }
    printf("\n]}\n");
    return 0;
}

int print_usage(FILE* out) {
    fprintf(out,
        "Usage: n4m_cli <subcommand>\n"
        "\n"
        "Subcommands:\n"
        "  --version      print library version + ABI version\n"
        "  --abi-info     print ABI / backend / dtype metadata\n"
        "  --selfcheck    run ABI smoke tests\n"
        "  --manifest-json  print the native method manifest as JSON\n"
        "  --help, -h     print this message\n"
        "\n");
    return out == stderr ? 2 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        return print_usage(stderr);
    }
    const char* cmd = argv[1];
    if (strcmp(cmd, "--version") == 0)     return cmd_version();
    if (strcmp(cmd, "--abi-info") == 0)    return cmd_abi_info();
    if (strcmp(cmd, "--selfcheck") == 0)   return cmd_selfcheck();
    if (strcmp(cmd, "--manifest-json") == 0) return cmd_manifest_json();
    if (strcmp(cmd, "--help") == 0 ||
        strcmp(cmd, "-h")     == 0)        return print_usage(stdout);
    fprintf(stderr, "Unknown subcommand: %s\n", cmd);
    return print_usage(stderr);
}
