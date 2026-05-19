// SPDX-License-Identifier: CECILL-2.1
//
// Test runner for the pls4all hand-rolled harness. Loops the registry, runs
// each test case, reports per-group totals, exits non-zero on any failure.

#include <cstdio>
#include <cstring>
#include <string>

#include "harness.hpp"

namespace {

bool matches_filter(const char* group, const char* name, const char* filter) {
    if (filter == nullptr) return true;
    std::string full = std::string(group) + "/" + name;
    return full.find(filter) != std::string::npos;
}

}  // namespace

int main(int argc, char** argv) {
    const char* filter = nullptr;
    bool verbose = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--filter") == 0 && i + 1 < argc) {
            filter = argv[++i];
        } else if (std::strcmp(argv[i], "--verbose") == 0 ||
                   std::strcmp(argv[i], "-v") == 0) {
            verbose = true;
        } else if (std::strcmp(argv[i], "--list") == 0) {
            for (const auto& tc : ::pls4all::test::Registry::instance().cases()) {
                std::printf("%s/%s\n", tc.group, tc.name);
            }
            return 0;
        } else {
            std::fprintf(stderr, "unknown arg: %s\n", argv[i]);
            return 2;
        }
    }

    int total_failures  = 0;
    int total_run       = 0;
    int total_skipped   = 0;

    for (const auto& tc : ::pls4all::test::Registry::instance().cases()) {
        if (!matches_filter(tc.group, tc.name, filter)) {
            ++total_skipped;
            continue;
        }
        ++total_run;
        int before = total_failures;
        if (verbose) {
            std::printf("[ RUN ] %s/%s\n", tc.group, tc.name);
        }
        tc.body(total_failures);
        if (total_failures == before) {
            if (verbose) {
                std::printf("[ OK  ] %s/%s\n", tc.group, tc.name);
            }
        } else {
            std::printf("[FAIL ] %s/%s\n", tc.group, tc.name);
        }
    }

    std::printf("\n=== %d run, %d failures, %d skipped ===\n",
                total_run, total_failures, total_skipped);
    return total_failures == 0 ? 0 : 1;
}
