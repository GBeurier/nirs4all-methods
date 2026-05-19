// SPDX-License-Identifier: CECILL-2.1
//
// Minimal zero-dependency test harness for pls4all Phase 0. Inspired by the
// hand-rolled assertion logic used in aompls; trades doctest's ergonomics
// for the right to ship without external test dependencies.
//
// Usage:
//   #include "harness.hpp"
//   TEST(group_name, test_name) {
//       CHECK(some_condition);
//       CHECK_EQ(actual, expected);
//       CHECK_STR_EQ(actual_cstr, expected_cstr);
//   }
//
// Then link the file together with cpp/tests/main.cpp which provides the
// runner.

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace pls4all::test {

struct TestCase {
    const char* group;
    const char* name;
    const char* file;
    int         line;
    void      (*body)(int& failures);
};

class Registry {
  public:
    static Registry& instance() noexcept {
        static Registry r;
        return r;
    }
    void add(const TestCase& tc) {
        cases_.push_back(tc);
    }
    const std::vector<TestCase>& cases() const noexcept { return cases_; }

  private:
    std::vector<TestCase> cases_;
};

struct AutoRegister {
    AutoRegister(const TestCase& tc) { Registry::instance().add(tc); }
};

}  // namespace pls4all::test

#define P4A_CONCAT_INNER(a, b) a##b
#define P4A_CONCAT(a, b)       P4A_CONCAT_INNER(a, b)

#define TEST(group, name)                                                       \
    static void P4A_CONCAT(p4a_test_body_, __LINE__)(int& failures);            \
    static ::pls4all::test::AutoRegister                                        \
        P4A_CONCAT(p4a_test_reg_, __LINE__){                                    \
            {#group, #name, __FILE__, __LINE__,                                  \
             &P4A_CONCAT(p4a_test_body_, __LINE__)}};                            \
    static void P4A_CONCAT(p4a_test_body_, __LINE__)(int& failures)

#define CHECK(cond)                                                             \
    do {                                                                         \
        if (!(cond)) {                                                           \
            ++failures;                                                           \
            std::fprintf(stderr, "  FAIL [%s:%d] CHECK(%s)\n",                    \
                         __FILE__, __LINE__, #cond);                              \
        }                                                                        \
    } while (0)

#define CHECK_EQ(a, b)                                                          \
    do {                                                                         \
        auto _a = (a);                                                           \
        auto _b = (b);                                                           \
        if (!(_a == _b)) {                                                       \
            ++failures;                                                           \
            std::fprintf(stderr, "  FAIL [%s:%d] CHECK_EQ(%s, %s)\n",             \
                         __FILE__, __LINE__, #a, #b);                             \
        }                                                                        \
    } while (0)

#define CHECK_NE(a, b)                                                          \
    do {                                                                         \
        auto _a = (a);                                                           \
        auto _b = (b);                                                           \
        if (!(_a != _b)) {                                                       \
            ++failures;                                                           \
            std::fprintf(stderr, "  FAIL [%s:%d] CHECK_NE(%s, %s)\n",             \
                         __FILE__, __LINE__, #a, #b);                             \
        }                                                                        \
    } while (0)

#define CHECK_STR_EQ(a, b)                                                      \
    do {                                                                         \
        const char* _a = (a);                                                    \
        const char* _b = (b);                                                    \
        if (_a == nullptr || _b == nullptr ||                                    \
            std::strcmp(_a, _b) != 0) {                                          \
            ++failures;                                                           \
            std::fprintf(stderr,                                                  \
                "  FAIL [%s:%d] CHECK_STR_EQ(%s, %s) — %s vs %s\n",                \
                __FILE__, __LINE__, #a, #b,                                       \
                _a ? _a : "(null)", _b ? _b : "(null)");                          \
        }                                                                        \
    } while (0)

#define CHECK_STR_CONTAINS(haystack, needle)                                    \
    do {                                                                         \
        const char* _h = (haystack);                                             \
        const char* _n = (needle);                                               \
        if (_h == nullptr || _n == nullptr ||                                    \
            std::strstr(_h, _n) == nullptr) {                                    \
            ++failures;                                                           \
            std::fprintf(stderr,                                                  \
                "  FAIL [%s:%d] CHECK_STR_CONTAINS(%s, %s)\n",                    \
                __FILE__, __LINE__, #haystack, #needle);                          \
        }                                                                        \
    } while (0)
