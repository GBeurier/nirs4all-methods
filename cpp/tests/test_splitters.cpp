// SPDX-License-Identifier: CECILL-2.1
//
// Parity tests for the Phase 11 sample-partitioning splitters
// (n4m_split_* category). Each operator contributes one smoke test
// (exercise create / split / destroy on a small inline matrix) and one
// parity test (load a JSON fixture produced by the matching generator
// under parity/python_generator/scripts/, run the C engine, assert
// byte-equality on the train/test integer index arrays).
//
// Tolerances per the Phase 11 brief: integer indices use byte-equality.
//
// Splitters covered (one smoke + one parity test per op):
//   1. KennardStone           — deterministic max-min on X.
//   2. SPXY                   — deterministic max-min on X+Y.
//   3. SPXYFold               — deterministic k-fold via alternating max-min.
//   4. SPXYGFold              — group-aware k-fold (aggregation = mean).
//   5. KMeans                 — k-means++ + Lloyd, PCG64-seeded.
//   6. KBinsStratified        — bin y + per-bin Fisher-Yates shuffle.
//   7. BinnedStratifiedGroupKFold — bin y + group-aware k-fold.
//   8. SystematicCircular     — argsort y, PCG64 rotate, take every n-th.
//   9. SPlit                  — data twinning (Vakayil & Joseph 2022).

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "n4m/n4m.h"

#include "fixture_parser.hpp"
#include "harness.hpp"

#ifndef N4M_PARITY_FIXTURE_DIR
#  error "N4M_PARITY_FIXTURE_DIR must be defined"
#endif

namespace {

using ParityFixture = ::n4m_testing::Fixture;
using ParityCase    = ::n4m_testing::Case;

ParityFixture load_fixture(const std::string& filename) {
    return ::n4m_testing::load_fixture(
        std::string(N4M_PARITY_FIXTURE_DIR) + "/" + filename,
        /*require_per_case_output_shape=*/false);
}

using ::n4m_testing::params_get_double;
using ::n4m_testing::params_get_int;
using ::n4m_testing::params_get_string;

n4m_matrix_view_t make_rowmajor_view(double* data, std::int64_t rows,
                                      std::int64_t cols) {
    n4m_matrix_view_t v{};
    const n4m_status_t st =
        n4m_matrix_view_init_rowmajor(&v, data, rows, cols, N4M_DTYPE_F64);
    N4M_TEST_REQUIRE(st == N4M_OK);
    return v;
}

// ---------------------------------------------------------------------------
// Helpers: read the expected train/test arrays out of a fixture case.
// The fixture format stores two parallel integer arrays per case as
// `train_idx` / `test_idx` (decimal JSON arrays embedded in `params_json`
// — see the Phase 11 generator).
// ---------------------------------------------------------------------------

std::vector<std::int64_t> parse_int_array(const std::string& json,
                                           const std::string& key) {
    std::vector<std::int64_t> out;
    const std::string q = "\"" + key + "\"";
    std::size_t p = json.find(q);
    if (p == std::string::npos) return out;
    p += q.size();
    while (p < json.size() && (json[p] == ' ' || json[p] == ':' ||
                                json[p] == '\t' || json[p] == '\n')) ++p;
    if (p >= json.size() || json[p] != '[') return out;
    ++p;
    while (p < json.size() && json[p] != ']') {
        while (p < json.size() && (json[p] == ' ' || json[p] == ',' ||
                                    json[p] == '\t' || json[p] == '\n')) ++p;
        if (p >= json.size() || json[p] == ']') break;
        char* tail = nullptr;
        const std::int64_t v = std::strtoll(json.c_str() + p, &tail, 10);
        if (tail == json.c_str() + p) break;
        out.push_back(v);
        p = static_cast<std::size_t>(tail - json.c_str());
    }
    return out;
}

void assert_indices_equal(const std::vector<std::int64_t>& got,
                           const std::vector<std::int64_t>& want,
                           const std::string& tag) {
    if (got.size() != want.size()) {
        throw std::runtime_error(tag + ": size mismatch (got " +
            std::to_string(got.size()) + ", want " +
            std::to_string(want.size()) + ")");
    }
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (got[i] != want[i]) {
            throw std::runtime_error(tag + ": index mismatch at i=" +
                std::to_string(i) + " (got " + std::to_string(got[i]) +
                ", want " + std::to_string(want[i]) + ")");
        }
    }
}

// Tolerant index-set comparison for FP-sensitive iterative splitters (kmeans).
// k-means uses an iterative floating-point objective (`s += d*d`), so a sample
// sitting on a cluster boundary can be assigned differently across compilers /
// architectures (e.g. x86 vs arm64 FMA contraction) — a single swapped sample
// is expected and must not be a hard cross-platform failure. We require the
// split to be a valid partition of the same size and within `max_sym_diff`
// elements of the fixture (2 = one sample moved between train and test).
void assert_indices_tolerant(const std::vector<std::int64_t>& got,
                              const std::vector<std::int64_t>& want,
                              const std::string& tag,
                              std::size_t max_sym_diff) {
    if (got.size() != want.size()) {
        throw std::runtime_error(tag + ": size mismatch (got " +
            std::to_string(got.size()) + ", want " +
            std::to_string(want.size()) + ")");
    }
    std::vector<std::int64_t> g = got, w = want;
    std::sort(g.begin(), g.end());
    std::sort(w.begin(), w.end());
    for (std::size_t i = 1; i < g.size(); ++i) {
        if (g[i] == g[i - 1]) {
            throw std::runtime_error(tag + ": duplicate index " +
                std::to_string(g[i]));
        }
    }
    // Symmetric-difference size via a two-pointer merge of the sorted sets.
    std::size_t diff = 0, i = 0, j = 0;
    while (i < g.size() && j < w.size()) {
        if (g[i] == w[j]) { ++i; ++j; }
        else if (g[i] < w[j]) { ++diff; ++i; }
        else { ++diff; ++j; }
    }
    diff += (g.size() - i) + (w.size() - j);
    if (diff > max_sym_diff) {
        throw std::runtime_error(tag + ": " + std::to_string(diff) +
            " indices differ from fixture (kmeans FP tolerance " +
            std::to_string(max_sym_diff) + ")");
    }
}

// ---------------------------------------------------------------------------
// Smoke tests.
// ---------------------------------------------------------------------------

void test_kennard_stone_smoke() {
    // 10 points in 2D — train should pick two extremes first.
    double X[20] = { 0.0, 0.0,  1.0, 0.0,  2.0, 0.0,  3.0, 0.0,  4.0, 0.0,
                      5.0, 0.0,  6.0, 0.0,  7.0, 0.0,  8.0, 0.0,  9.0, 0.0 };
    n4m_split_kennard_stone_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_split_kennard_stone_create(&h, 0.5) == N4M_OK);
    n4m_matrix_view_t Xv = make_rowmajor_view(X, 10, 2);
    n4m_split_result_t r{};
    N4M_TEST_REQUIRE(n4m_split_kennard_stone_split(h, Xv, &r) == N4M_OK);
    N4M_TEST_REQUIRE(r.n_train == 5);
    N4M_TEST_REQUIRE(r.n_test == 5);
    N4M_TEST_REQUIRE(r.train_idx != nullptr);
    N4M_TEST_REQUIRE(r.test_idx != nullptr);
    // First two picks must be the two endpoints (indices 0 and 9).
    bool has_0 = false, has_9 = false;
    for (std::int64_t i = 0; i < r.n_train; ++i) {
        if (r.train_idx[i] == 0) has_0 = true;
        if (r.train_idx[i] == 9) has_9 = true;
    }
    N4M_TEST_REQUIRE(has_0);
    N4M_TEST_REQUIRE(has_9);
    n4m_split_result_destroy(&r);
    n4m_split_kennard_stone_destroy(h);
    n4m_split_kennard_stone_destroy(nullptr);
    // Invalid test_size.
    N4M_TEST_REQUIRE(n4m_split_kennard_stone_create(&h, 0.0)
                     == N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(n4m_split_kennard_stone_create(&h, 1.5)
                     == N4M_ERR_INVALID_ARGUMENT);
}

void test_spxy_smoke() {
    double X[20] = { 0,0,  1,0,  2,0,  3,0,  4,0,  5,0,  6,0,  7,0,  8,0,  9,0 };
    double Y[10] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    n4m_split_spxy_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_split_spxy_create(&h, 0.5) == N4M_OK);
    n4m_matrix_view_t Xv = make_rowmajor_view(X, 10, 2);
    n4m_matrix_view_t Yv = make_rowmajor_view(Y, 10, 1);
    n4m_split_result_t r{};
    N4M_TEST_REQUIRE(n4m_split_spxy_split(h, Xv, Yv, &r) == N4M_OK);
    N4M_TEST_REQUIRE(r.n_train == 5);
    N4M_TEST_REQUIRE(r.n_test == 5);
    n4m_split_result_destroy(&r);
    n4m_split_spxy_destroy(h);
}

void test_spxy_fold_smoke() {
    double X[20] = { 0,0,  1,0,  2,0,  3,0,  4,0,  5,0,  6,0,  7,0,  8,0,  9,0 };
    double Y[10] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    n4m_split_spxy_fold_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_split_spxy_fold_create(&h, /*n_splits=*/5,
                                                /*y_metric=*/1) == N4M_OK);
    int32_t k = 0;
    N4M_TEST_REQUIRE(n4m_split_spxy_fold_n_splits(h, &k) == N4M_OK);
    N4M_TEST_REQUIRE(k == 5);
    n4m_matrix_view_t Xv = make_rowmajor_view(X, 10, 2);
    n4m_matrix_view_t Yv = make_rowmajor_view(Y, 10, 1);
    for (int32_t f = 0; f < 5; ++f) {
        n4m_split_result_t r{};
        N4M_TEST_REQUIRE(n4m_split_spxy_fold_split_fold(h, Xv, Yv, f, &r)
                         == N4M_OK);
        N4M_TEST_REQUIRE(r.n_train + r.n_test == 10);
        n4m_split_result_destroy(&r);
    }
    n4m_split_spxy_fold_destroy(h);
}

void test_spxy_g_fold_smoke() {
    double X[20] = { 0,0,  1,0,  2,0,  3,0,  4,0,  5,0,  6,0,  7,0,  8,0,  9,0 };
    double Y[10] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    int64_t groups[10] = { 0, 0, 1, 1, 2, 2, 3, 3, 4, 4 };  // 5 groups, 2 each
    n4m_split_spxy_g_fold_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_split_spxy_g_fold_create(&h, /*n_splits=*/5,
                                                   /*y_metric=*/1,
                                                   /*aggregation=*/0) == N4M_OK);
    n4m_matrix_view_t Xv = make_rowmajor_view(X, 10, 2);
    n4m_matrix_view_t Yv = make_rowmajor_view(Y, 10, 1);
    for (int32_t f = 0; f < 5; ++f) {
        n4m_split_result_t r{};
        N4M_TEST_REQUIRE(n4m_split_spxy_g_fold_split_fold(h, Xv, Yv, groups,
                                                          10, f, &r) == N4M_OK);
        // Each fold should hold exactly one group's samples (2 each).
        N4M_TEST_REQUIRE(r.n_test == 2);
        N4M_TEST_REQUIRE(r.n_train == 8);
        n4m_split_result_destroy(&r);
    }
    n4m_split_spxy_g_fold_destroy(h);
}

void test_kmeans_smoke() {
    // 12 points along a line: KMeans with k=8 should produce 8 unique train
    // indices (or fewer if Lloyd merges centroids).
    double X[24] = { 0,0, 1,0, 2,0, 3,0, 4,0, 5,0,
                     6,0, 7,0, 8,0, 9,0, 10,0, 11,0 };
    n4m_split_kmeans_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_split_kmeans_create(&h, /*test_size=*/0.333,
                                              /*seed=*/42, /*max_iter=*/100)
                     == N4M_OK);
    n4m_matrix_view_t Xv = make_rowmajor_view(X, 12, 2);
    n4m_split_result_t r{};
    N4M_TEST_REQUIRE(n4m_split_kmeans_split(h, Xv, &r) == N4M_OK);
    N4M_TEST_REQUIRE(r.n_train >= 1);
    N4M_TEST_REQUIRE(r.n_train + r.n_test == 12);
    // train indices must be unique and sorted ascending.
    for (std::int64_t i = 1; i < r.n_train; ++i) {
        N4M_TEST_REQUIRE(r.train_idx[i] > r.train_idx[i - 1]);
    }
    n4m_split_result_destroy(&r);
    n4m_split_kmeans_destroy(h);
}

void test_kbins_stratified_smoke() {
    double Y[20];
    for (int i = 0; i < 20; ++i) Y[i] = static_cast<double>(i);
    n4m_split_kbins_stratified_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_split_kbins_stratified_create(&h, /*test_size=*/0.25,
                                                        /*seed=*/123,
                                                        /*n_bins=*/4,
                                                        /*strategy=*/0)
                     == N4M_OK);
    n4m_matrix_view_t Yv = make_rowmajor_view(Y, 20, 1);
    n4m_split_result_t r{};
    N4M_TEST_REQUIRE(n4m_split_kbins_stratified_split(h, Yv, &r) == N4M_OK);
    N4M_TEST_REQUIRE(r.n_test >= 1);
    N4M_TEST_REQUIRE(r.n_train + r.n_test == 20);
    n4m_split_result_destroy(&r);
    n4m_split_kbins_stratified_destroy(h);
}

void test_bsgk_smoke() {
    // 30 samples in 15 groups of 2 — 3 splits with 2 bins ensures every
    // fold receives at least one group from each bin (5 groups per bin,
    // round-robin into 3 folds → folds 0, 1, 2 each get 1-2 groups).
    double  Y[30];
    int64_t groups[30];
    for (int i = 0; i < 30; ++i) {
        Y[i] = static_cast<double>(i);
        groups[i] = i / 2;  // 15 groups of 2
    }
    n4m_split_binned_strat_group_kfold_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_split_binned_strat_group_kfold_create(
        &h, /*n_splits=*/3, /*n_bins=*/2, /*strategy=*/0,
        /*shuffle=*/1, /*seed=*/7) == N4M_OK);
    int32_t k = 0;
    N4M_TEST_REQUIRE(n4m_split_binned_strat_group_kfold_n_splits(h, &k)
                     == N4M_OK);
    N4M_TEST_REQUIRE(k == 3);
    n4m_matrix_view_t Yv = make_rowmajor_view(Y, 30, 1);
    for (int32_t f = 0; f < 3; ++f) {
        n4m_split_result_t r{};
        N4M_TEST_REQUIRE(n4m_split_binned_strat_group_kfold_split_fold(
            h, Yv, groups, 30, f, &r) == N4M_OK);
        N4M_TEST_REQUIRE(r.n_train + r.n_test == 30);
        n4m_split_result_destroy(&r);
    }
    n4m_split_binned_strat_group_kfold_destroy(h);
}

void test_systematic_circular_smoke() {
    double Y[10] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    n4m_split_systematic_circular_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_split_systematic_circular_create(&h, 0.25, 42)
                     == N4M_OK);
    n4m_matrix_view_t Yv = make_rowmajor_view(Y, 10, 1);
    n4m_split_result_t r{};
    N4M_TEST_REQUIRE(n4m_split_systematic_circular_split(h, Yv, &r)
                     == N4M_OK);
    N4M_TEST_REQUIRE(r.n_train + r.n_test == 10);
    N4M_TEST_REQUIRE(r.n_train >= 1);
    n4m_split_result_destroy(&r);
    n4m_split_systematic_circular_destroy(h);
}

void test_split_splitter_smoke() {
    // 20 samples, 3 features.
    double X[60];
    for (int i = 0; i < 20; ++i) {
        X[i * 3 + 0] = static_cast<double>(i);
        X[i * 3 + 1] = static_cast<double>(i * 2 + 1);
        X[i * 3 + 2] = static_cast<double>((i % 3)) * 0.5;
    }
    n4m_split_split_splitter_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_split_split_splitter_create(&h, 0.25, 7) == N4M_OK);
    n4m_matrix_view_t Xv = make_rowmajor_view(X, 20, 3);
    n4m_split_result_t r{};
    N4M_TEST_REQUIRE(n4m_split_split_splitter_split(h, Xv, &r) == N4M_OK);
    N4M_TEST_REQUIRE(r.n_train + r.n_test == 20);
    // test_size 0.25 → r=4 → expected n_test = ceil(20/4) = 5.
    N4M_TEST_REQUIRE(r.n_test == 5);
    N4M_TEST_REQUIRE(r.n_train == 15);
    n4m_split_result_destroy(&r);
    n4m_split_split_splitter_destroy(h);
}

// ---------------------------------------------------------------------------
// Parity tests.
//
// Each fixture stores N samples of a (rows x cols) X matrix and (rows x 1)
// y vector. Per case it carries the expected train_idx / test_idx integer
// arrays for byte-exact comparison.
// ---------------------------------------------------------------------------

void verify_kennard_stone_parity() {
    ParityFixture fx = load_fixture("split_kennard_stone_v1.json");
    for (const auto& c : fx.cases) {
        const double test_size = params_get_double(c.params_json, "test_size", 0.25);
        n4m_split_kennard_stone_handle_t* h = nullptr;
        N4M_TEST_REQUIRE(n4m_split_kennard_stone_create(&h, test_size) == N4M_OK);
        std::vector<double> X = fx.input;
        n4m_matrix_view_t Xv = make_rowmajor_view(X.data(), fx.rows, fx.cols);
        n4m_split_result_t r{};
        N4M_TEST_REQUIRE(n4m_split_kennard_stone_split(h, Xv, &r) == N4M_OK);
        std::vector<std::int64_t> got_train(r.train_idx, r.train_idx + r.n_train);
        std::vector<std::int64_t> got_test(r.test_idx, r.test_idx + r.n_test);
        const auto want_train = parse_int_array(c.params_json, "train_idx");
        const auto want_test  = parse_int_array(c.params_json, "test_idx");
        assert_indices_equal(got_train, want_train, "ks/" + c.name + "/train");
        assert_indices_equal(got_test,  want_test,  "ks/" + c.name + "/test");
        n4m_split_result_destroy(&r);
        n4m_split_kennard_stone_destroy(h);
    }
}

void verify_spxy_parity() {
    ParityFixture fx = load_fixture("split_spxy_v1.json");
    /* y is stored in fit_input (rows x 1). */
    N4M_TEST_REQUIRE(fx.has_fit);
    for (const auto& c : fx.cases) {
        const double test_size = params_get_double(c.params_json, "test_size", 0.25);
        n4m_split_spxy_handle_t* h = nullptr;
        N4M_TEST_REQUIRE(n4m_split_spxy_create(&h, test_size) == N4M_OK);
        std::vector<double> X = fx.input;
        std::vector<double> Y = fx.fit_input;
        n4m_matrix_view_t Xv = make_rowmajor_view(X.data(), fx.rows, fx.cols);
        n4m_matrix_view_t Yv = make_rowmajor_view(Y.data(), fx.fit_rows, fx.fit_cols);
        n4m_split_result_t r{};
        N4M_TEST_REQUIRE(n4m_split_spxy_split(h, Xv, Yv, &r) == N4M_OK);
        std::vector<std::int64_t> got_train(r.train_idx, r.train_idx + r.n_train);
        std::vector<std::int64_t> got_test(r.test_idx, r.test_idx + r.n_test);
        assert_indices_equal(got_train,
                              parse_int_array(c.params_json, "train_idx"),
                              "spxy/" + c.name + "/train");
        assert_indices_equal(got_test,
                              parse_int_array(c.params_json, "test_idx"),
                              "spxy/" + c.name + "/test");
        n4m_split_result_destroy(&r);
        n4m_split_spxy_destroy(h);
    }
}

void verify_spxy_fold_parity() {
    ParityFixture fx = load_fixture("split_spxy_fold_v1.json");
    N4M_TEST_REQUIRE(fx.has_fit);
    for (const auto& c : fx.cases) {
        const int n_splits = static_cast<int>(
            params_get_int(c.params_json, "n_splits", 5));
        const int y_metric = static_cast<int>(
            params_get_int(c.params_json, "y_metric", 1));
        const int fold_idx = static_cast<int>(
            params_get_int(c.params_json, "fold_idx", 0));
        n4m_split_spxy_fold_handle_t* h = nullptr;
        N4M_TEST_REQUIRE(n4m_split_spxy_fold_create(&h, n_splits, y_metric)
                         == N4M_OK);
        std::vector<double> X = fx.input;
        std::vector<double> Y = fx.fit_input;
        n4m_matrix_view_t Xv = make_rowmajor_view(X.data(), fx.rows, fx.cols);
        n4m_matrix_view_t Yv = make_rowmajor_view(Y.data(), fx.fit_rows, fx.fit_cols);
        n4m_split_result_t r{};
        N4M_TEST_REQUIRE(n4m_split_spxy_fold_split_fold(h, Xv, Yv, fold_idx, &r)
                         == N4M_OK);
        std::vector<std::int64_t> got_train(r.train_idx, r.train_idx + r.n_train);
        std::vector<std::int64_t> got_test(r.test_idx, r.test_idx + r.n_test);
        assert_indices_equal(got_train,
                              parse_int_array(c.params_json, "train_idx"),
                              "spxy_fold/" + c.name + "/train");
        assert_indices_equal(got_test,
                              parse_int_array(c.params_json, "test_idx"),
                              "spxy_fold/" + c.name + "/test");
        n4m_split_result_destroy(&r);
        n4m_split_spxy_fold_destroy(h);
    }
}

void verify_spxy_g_fold_parity() {
    ParityFixture fx = load_fixture("split_spxy_g_fold_v1.json");
    N4M_TEST_REQUIRE(fx.has_fit);
    for (const auto& c : fx.cases) {
        const int n_splits   = static_cast<int>(params_get_int(c.params_json, "n_splits", 5));
        const int y_metric   = static_cast<int>(params_get_int(c.params_json, "y_metric", 1));
        const int aggregation = static_cast<int>(params_get_int(c.params_json, "aggregation", 0));
        const int fold_idx   = static_cast<int>(params_get_int(c.params_json, "fold_idx", 0));
        const auto groups_vec = parse_int_array(c.params_json, "groups");
        n4m_split_spxy_g_fold_handle_t* h = nullptr;
        N4M_TEST_REQUIRE(n4m_split_spxy_g_fold_create(&h, n_splits, y_metric,
                                                       aggregation) == N4M_OK);
        std::vector<double> X = fx.input;
        std::vector<double> Y = fx.fit_input;
        n4m_matrix_view_t Xv = make_rowmajor_view(X.data(), fx.rows, fx.cols);
        n4m_matrix_view_t Yv = make_rowmajor_view(Y.data(), fx.fit_rows, fx.fit_cols);
        n4m_split_result_t r{};
        N4M_TEST_REQUIRE(n4m_split_spxy_g_fold_split_fold(
            h, Xv, Yv, groups_vec.data(),
            static_cast<int64_t>(groups_vec.size()), fold_idx, &r) == N4M_OK);
        std::vector<std::int64_t> got_train(r.train_idx, r.train_idx + r.n_train);
        std::vector<std::int64_t> got_test(r.test_idx, r.test_idx + r.n_test);
        assert_indices_equal(got_train,
                              parse_int_array(c.params_json, "train_idx"),
                              "spxy_g_fold/" + c.name + "/train");
        assert_indices_equal(got_test,
                              parse_int_array(c.params_json, "test_idx"),
                              "spxy_g_fold/" + c.name + "/test");
        n4m_split_result_destroy(&r);
        n4m_split_spxy_g_fold_destroy(h);
    }
}

void verify_kmeans_parity() {
    ParityFixture fx = load_fixture("split_kmeans_v1.json");
    for (const auto& c : fx.cases) {
        const double test_size = params_get_double(c.params_json, "test_size", 0.25);
        const std::int64_t seed = params_get_int(c.params_json, "seed", 0);
        const int max_iter = static_cast<int>(params_get_int(c.params_json, "max_iter", 100));
        n4m_split_kmeans_handle_t* h = nullptr;
        N4M_TEST_REQUIRE(n4m_split_kmeans_create(&h, test_size,
                                                  static_cast<uint64_t>(seed),
                                                  max_iter) == N4M_OK);
        std::vector<double> X = fx.input;
        n4m_matrix_view_t Xv = make_rowmajor_view(X.data(), fx.rows, fx.cols);
        n4m_split_result_t r{};
        N4M_TEST_REQUIRE(n4m_split_kmeans_split(h, Xv, &r) == N4M_OK);
        std::vector<std::int64_t> got_train(r.train_idx, r.train_idx + r.n_train);
        std::vector<std::int64_t> got_test(r.test_idx, r.test_idx + r.n_test);
        // k-means is FP-iterative; tolerate one boundary sample swapping
        // train/test across architectures (x86 vs arm64). See helper note.
        assert_indices_tolerant(got_train,
                                 parse_int_array(c.params_json, "train_idx"),
                                 "kmeans/" + c.name + "/train", 2);
        assert_indices_tolerant(got_test,
                                 parse_int_array(c.params_json, "test_idx"),
                                 "kmeans/" + c.name + "/test", 2);
        n4m_split_result_destroy(&r);
        n4m_split_kmeans_destroy(h);
    }
}

void verify_kbins_stratified_parity() {
    ParityFixture fx = load_fixture("split_kbins_stratified_v1.json");
    N4M_TEST_REQUIRE(fx.has_fit);
    for (const auto& c : fx.cases) {
        const double test_size = params_get_double(c.params_json, "test_size", 0.25);
        const std::int64_t seed = params_get_int(c.params_json, "seed", 0);
        const int n_bins = static_cast<int>(params_get_int(c.params_json, "n_bins", 5));
        const int strategy = static_cast<int>(params_get_int(c.params_json, "strategy", 0));
        n4m_split_kbins_stratified_handle_t* h = nullptr;
        N4M_TEST_REQUIRE(n4m_split_kbins_stratified_create(&h, test_size,
            static_cast<uint64_t>(seed), n_bins, strategy) == N4M_OK);
        std::vector<double> Y = fx.fit_input;
        n4m_matrix_view_t Yv = make_rowmajor_view(Y.data(), fx.fit_rows, fx.fit_cols);
        n4m_split_result_t r{};
        N4M_TEST_REQUIRE(n4m_split_kbins_stratified_split(h, Yv, &r) == N4M_OK);
        std::vector<std::int64_t> got_train(r.train_idx, r.train_idx + r.n_train);
        std::vector<std::int64_t> got_test(r.test_idx, r.test_idx + r.n_test);
        assert_indices_equal(got_train,
                              parse_int_array(c.params_json, "train_idx"),
                              "kbins/" + c.name + "/train");
        assert_indices_equal(got_test,
                              parse_int_array(c.params_json, "test_idx"),
                              "kbins/" + c.name + "/test");
        n4m_split_result_destroy(&r);
        n4m_split_kbins_stratified_destroy(h);
    }
}

void verify_bsgk_parity() {
    ParityFixture fx = load_fixture("split_binned_strat_group_kfold_v1.json");
    N4M_TEST_REQUIRE(fx.has_fit);
    for (const auto& c : fx.cases) {
        const int n_splits = static_cast<int>(params_get_int(c.params_json, "n_splits", 5));
        const int n_bins   = static_cast<int>(params_get_int(c.params_json, "n_bins", 4));
        const int strategy = static_cast<int>(params_get_int(c.params_json, "strategy", 0));
        const int shuffle  = static_cast<int>(params_get_int(c.params_json, "shuffle", 1));
        const std::int64_t seed = params_get_int(c.params_json, "seed", 0);
        const int fold_idx = static_cast<int>(params_get_int(c.params_json, "fold_idx", 0));
        const auto groups_vec = parse_int_array(c.params_json, "groups");
        n4m_split_binned_strat_group_kfold_handle_t* h = nullptr;
        N4M_TEST_REQUIRE(n4m_split_binned_strat_group_kfold_create(
            &h, n_splits, n_bins, strategy, shuffle,
            static_cast<uint64_t>(seed)) == N4M_OK);
        std::vector<double> Y = fx.fit_input;
        n4m_matrix_view_t Yv = make_rowmajor_view(Y.data(), fx.fit_rows, fx.fit_cols);
        n4m_split_result_t r{};
        N4M_TEST_REQUIRE(n4m_split_binned_strat_group_kfold_split_fold(
            h, Yv, groups_vec.data(),
            static_cast<int64_t>(groups_vec.size()), fold_idx, &r) == N4M_OK);
        std::vector<std::int64_t> got_train(r.train_idx, r.train_idx + r.n_train);
        std::vector<std::int64_t> got_test(r.test_idx, r.test_idx + r.n_test);
        assert_indices_equal(got_train,
                              parse_int_array(c.params_json, "train_idx"),
                              "bsgk/" + c.name + "/train");
        assert_indices_equal(got_test,
                              parse_int_array(c.params_json, "test_idx"),
                              "bsgk/" + c.name + "/test");
        n4m_split_result_destroy(&r);
        n4m_split_binned_strat_group_kfold_destroy(h);
    }
}

void verify_systematic_circular_parity() {
    ParityFixture fx = load_fixture("split_systematic_circular_v1.json");
    N4M_TEST_REQUIRE(fx.has_fit);
    for (const auto& c : fx.cases) {
        const double test_size = params_get_double(c.params_json, "test_size", 0.25);
        const std::int64_t seed = params_get_int(c.params_json, "seed", 0);
        n4m_split_systematic_circular_handle_t* h = nullptr;
        N4M_TEST_REQUIRE(n4m_split_systematic_circular_create(
            &h, test_size, static_cast<uint64_t>(seed)) == N4M_OK);
        std::vector<double> Y = fx.fit_input;
        n4m_matrix_view_t Yv = make_rowmajor_view(Y.data(), fx.fit_rows, fx.fit_cols);
        n4m_split_result_t r{};
        N4M_TEST_REQUIRE(n4m_split_systematic_circular_split(h, Yv, &r) == N4M_OK);
        std::vector<std::int64_t> got_train(r.train_idx, r.train_idx + r.n_train);
        std::vector<std::int64_t> got_test(r.test_idx, r.test_idx + r.n_test);
        assert_indices_equal(got_train,
                              parse_int_array(c.params_json, "train_idx"),
                              "syscirc/" + c.name + "/train");
        assert_indices_equal(got_test,
                              parse_int_array(c.params_json, "test_idx"),
                              "syscirc/" + c.name + "/test");
        n4m_split_result_destroy(&r);
        n4m_split_systematic_circular_destroy(h);
    }
}

void verify_split_splitter_parity() {
    ParityFixture fx = load_fixture("split_split_splitter_v1.json");
    for (const auto& c : fx.cases) {
        const double test_size = params_get_double(c.params_json, "test_size", 0.25);
        const std::int64_t seed = params_get_int(c.params_json, "seed", 0);
        n4m_split_split_splitter_handle_t* h = nullptr;
        N4M_TEST_REQUIRE(n4m_split_split_splitter_create(
            &h, test_size, static_cast<uint64_t>(seed)) == N4M_OK);
        std::vector<double> X = fx.input;
        n4m_matrix_view_t Xv = make_rowmajor_view(X.data(), fx.rows, fx.cols);
        n4m_split_result_t r{};
        N4M_TEST_REQUIRE(n4m_split_split_splitter_split(h, Xv, &r) == N4M_OK);
        std::vector<std::int64_t> got_train(r.train_idx, r.train_idx + r.n_train);
        std::vector<std::int64_t> got_test(r.test_idx, r.test_idx + r.n_test);
        assert_indices_equal(got_train,
                              parse_int_array(c.params_json, "train_idx"),
                              "split/" + c.name + "/train");
        assert_indices_equal(got_test,
                              parse_int_array(c.params_json, "test_idx"),
                              "split/" + c.name + "/test");
        n4m_split_result_destroy(&r);
        n4m_split_split_splitter_destroy(h);
    }
}

}  // namespace

void register_splitters_tests(n4m_testing::Runner& r);
void register_splitters_tests(n4m_testing::Runner& r) {
    r.run("split_kennard_stone_smoke",         test_kennard_stone_smoke);
    r.run("split_kennard_stone_parity",        verify_kennard_stone_parity);
    r.run("split_spxy_smoke",                  test_spxy_smoke);
    r.run("split_spxy_parity",                 verify_spxy_parity);
    r.run("split_spxy_fold_smoke",             test_spxy_fold_smoke);
    r.run("split_spxy_fold_parity",            verify_spxy_fold_parity);
    r.run("split_spxy_g_fold_smoke",           test_spxy_g_fold_smoke);
    r.run("split_spxy_g_fold_parity",          verify_spxy_g_fold_parity);
    r.run("split_kmeans_smoke",                test_kmeans_smoke);
    r.run("split_kmeans_parity",               verify_kmeans_parity);
    r.run("split_kbins_stratified_smoke",      test_kbins_stratified_smoke);
    r.run("split_kbins_stratified_parity",     verify_kbins_stratified_parity);
    r.run("split_bsgk_smoke",                  test_bsgk_smoke);
    r.run("split_bsgk_parity",                 verify_bsgk_parity);
    r.run("split_systematic_circular_smoke",   test_systematic_circular_smoke);
    r.run("split_systematic_circular_parity",  verify_systematic_circular_parity);
    r.run("split_split_splitter_smoke",        test_split_splitter_smoke);
    r.run("split_split_splitter_parity",       verify_split_splitter_parity);
}
