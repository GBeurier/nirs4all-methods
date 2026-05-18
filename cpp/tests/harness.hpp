// SPDX-License-Identifier: CECILL-2.1
//
// Hand-rolled zero-dependency test harness for chemometrics4all.

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "chemometrics4all/c4a.h"

#define C4A_TEST_REQUIRE(cond) do { \
    if (!(cond)) { \
        throw c4a_testing::AssertFailed(__FILE__, __LINE__, #cond); \
    } \
} while (0)

namespace c4a_testing {

struct AssertFailed : public std::runtime_error {
    AssertFailed(const char* file, int line, const char* expr)
        : std::runtime_error(std::string(file) + ":" + std::to_string(line)
                             + ": REQUIRE(" + expr + ") failed") {}
};

class Runner {
public:
    explicit Runner(const char* suite_name) : suite_(suite_name) {}

    template <typename Fn>
    void run(const char* test_name, Fn&& fn) {
        printf("[%s] %s ... ", suite_, test_name);
        fflush(stdout);
        try {
            fn();
            ++pass_;
            printf("ok\n");
        } catch (const std::exception& e) {
            ++fail_;
            failures_.emplace_back(test_name, e.what());
            printf("FAIL\n  %s\n", e.what());
        } catch (...) {
            ++fail_;
            failures_.emplace_back(test_name, "non-std::exception thrown");
            printf("FAIL (unknown)\n");
        }
    }

    int finalize() const {
        printf("\n=== chemometrics4all_tests: %zu passed, %zu failed ===\n",
               pass_, fail_);
        for (const auto& [name, msg] : failures_) {
            printf("  FAIL %s: %s\n", name.c_str(), msg.c_str());
        }
        return fail_ == 0 ? 0 : 1;
    }

private:
    const char* suite_;
    size_t pass_ = 0;
    size_t fail_ = 0;
    std::vector<std::pair<std::string, std::string>> failures_;
};

}  // namespace c4a_testing
