// SPDX-License-Identifier: CECILL-2.1
//
// Parity tests for direct (closed-form) Ridge regression via the public
// n4m_ridge_fit ABI. Expected coefficients / intercepts / predictions are
// computed in numpy (closed-form primal solve on column-centered X, Y with
// intercept = y_mean - x_mean . beta) and baked in as constants.
//
//   - 6x4 X, single-output (6x1) Y, lambda in {0.1, 1.0, 10.0}
//   - 6x4 X, multi-output  (6x2) Y, lambda in {0.1, 1.0, 10.0}
//   - 4x10 wide X (p>n) -> AUTO selects the DUAL path; result must match the
//     numpy PRIMAL reference (demonstrates dual == primal to 1e-10).
//   - scale_x=True path (zero-variance-aware standardization, de-scaled coef).
//
// Tolerance: 1e-10 (the augmented-QR primal / dual-Gram solves are exact up to
// round-off; well-conditioned fixtures hold comfortably below 1e-10).

#include "n4m/n4m.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "harness.hpp"

namespace {

constexpr double kTol = 1e-10;

// ---- baked numpy references (see the generator snippet in the PR notes) ----
const double kX6x4[24] = {
    0.5, -1.2, 3.3, 0.1, 1.5, 0.7, -0.8, 2.2, -0.3, 2.1, 1.0, -1.1,
    2.0, -0.5, 0.4, 1.3, 0.9, 1.1, -2.0, 0.6, -1.0, 0.2, 1.7, -0.4};
const double kY6x1[6] = {1.0, 2.5, -0.5, 3.0, 0.8, -1.2};
const double kY6x2[12] = {1.0, 0.2, 2.5, -1.0, -0.5, 0.5,
                          3.0, 1.5, 0.8, -0.3, -1.2, 0.9};

struct SingleCase {
    double lambda;
    double coef[4];
    double intercept;
    double pred[6];
};

const SingleCase kSingle[3] = {
    {0.1,
     {1.2416853670521188, 0.1181770074383279, 0.21969847532613734,
      0.41763544915853823},
     -0.17870372719029393,
     {1.0670950609018792, 2.509587436482588, -0.5427381404336957,
      2.8763839772313347, 0.8799921301816219, -1.190320464363729}},
    {1.0,
     {0.9248660799460998, -0.06999105447358493, 0.07568216901232713,
      0.445062315634221},
     0.16072276371231142,
     {1.0014024583577648, 2.4176195046853763, -0.6776046528513628,
      2.6543043287707215, 1.0317851270987362, -0.8275067660612376}},
    {10.0,
     {0.3986775245599553, -0.16085076422099184, -0.08420017870087837,
      0.32045308231085246},
     0.6647833444664002,
     {0.8113277423297547, 1.9225610203962171, -0.2293050870084854,
      1.9254727112205636, 1.2073294827155372, -0.0373858696535877}},
};

struct MultiCase {
    double lambda;
    double coef[8];      // 4x2 row-major
    double intercept[2];
    double pred[12];     // 6x2 row-major
};

const MultiCase kMulti[3] = {
    {0.1,
     {1.2416853670521188, 0.3862620550059256, 0.11817700743832808,
      -0.3817942370878677, 0.21969847532613745, -0.05549324885806093,
      0.4176354491585383, -0.7515996579874702},
     {-0.17870372719029415, 0.5924762572407898},
     {1.0670950609018788, 0.9854726822188458, 2.509587436482588,
      -0.7045112746978148, -0.5427381404336955, 0.44609611778264624,
      2.876383977231334, 0.5566206308696392, 0.8799921301816217,
      0.18016514887310808, -1.190320464363729, 0.3361566949535752}},
    {1.0,
     {0.9248660799460999, 0.1892966029250404, -0.0699910544735848,
      -0.2191990726009228, 0.07568216901232722, 0.04886345361329885,
      0.44506231563422105, -0.41812407884256136},
     {0.1607227637123113, 0.4329394305965182},
     {1.001402458357765, 0.9100636082197757, 2.4176195046853763,
      -0.3955187521808413, -0.6776046528513626, 0.4246323375971845,
      2.654304328770722, 0.39711625169705017, 1.0317851270987362,
      0.013586038835904923, -0.8275067660612377, 0.4501205158309258}},
    {10.0,
     {0.3986775245599553, 0.007275680847670648, -0.16085076422099184,
      -0.06697861224164159, -0.08420017870087837, 0.09759425814834899,
      0.32045308231085246, -0.10376052883743782},
     {0.6647833444664002, 0.31056171947589184},
     {0.8113277423297547, 0.706258893595505, 1.9225610203962171,
      -0.031758357782793734, -0.2293050870084854, 0.3794547693836739,
      1.9254727112205636, 0.2627514030627244, 1.2073294827155372,
      -0.014011474826170989, -0.0373858696535877, 0.4973047665670613}},
};

// scale_x=True, single output, lambda=1.0
const double kScaleCoef[4] = {0.9205271731487844, -0.09283335803602975,
                              0.05361826879178879, 0.41807485362147173};
const double kScaleIntercept = 0.19784572725373883;
const double kScalePred[6] = {0.988257115846417,  2.390523199285502,
                              -0.6795265467583892, 2.6502613697939514,
                              1.0678118638373175, -0.8173270020047994};
const double kScaleXScale[4] = {1.019803902718557, 1.0708252269472673,
                                1.7039170558842742, 1.0843584893075415};

// 4x10 wide X (p>n) -> AUTO dual path
const double kXw[40] = {
    -1.4238, 1.2637, -0.8707, -0.2592, -0.0753, -0.7409, -1.3678, 0.6489,
    0.3611, -1.9529, 2.3474, 0.9685, -0.7594, 0.9022, -0.467, -0.0607,
    0.7888, -1.2567, 0.5759, 1.399, 1.3223, -0.2997, 0.9029, -1.6216,
    -0.1582, 0.4495, -1.3436, -0.0817, 1.7247, 2.6182, 0.7774, 0.8286,
    -0.959, -1.2094, -1.4123, 0.5415, 0.7519, -0.6588, -1.2287, 0.2576};
const double kYw[4] = {0.3129, -0.1308, 1.27, -0.093};

struct WideCase {
    double lambda;
    double coef[10];
    double intercept;
    double pred[4];
};

const WideCase kWide[2] = {
    {1.0,
     {-0.045174266652657794, -0.07767918398969959, 0.1117282963094475,
      -0.11933350652309917, 0.0432985456299248, 0.01645514364868452,
      -0.14571181040667125, 0.06984373892944543, 0.11503070328483436,
      0.09186479027027675},
     0.3178150254172479,
     {0.3089294368513187, -0.08513282279997275, 1.2052849969640989,
      -0.06998161101544503}},
    {10.0,
     {-0.02084708927781913, -0.04684180654572284, 0.06838047012316811,
      -0.0655865033363205, 0.0288059403533321, 0.008573441415590018,
      -0.08711504744806844, 0.03961043662686378, 0.07511922789769526,
      0.060770394438835335},
     0.3212929276737169,
     {0.29402724285165727, 0.11170128633313428, 0.8776365575339298,
      0.07573491328127854}},
};

n4m_matrix_view_t make_view(const double* values, std::int64_t rows,
                            std::int64_t cols) {
    n4m_matrix_view_t v{};
    const n4m_status_t st = n4m_matrix_view_init_rowmajor(
        &v, const_cast<double*>(values), rows, cols, N4M_DTYPE_F64);
    N4M_TEST_REQUIRE(st == N4M_OK);
    return v;
}

void check_matrix(const n4m_method_result_t* result, const char* key,
                  const double* expected, std::int64_t exp_rows,
                  std::int64_t exp_cols) {
    const double* data = nullptr;
    std::int64_t rows = 0, cols = 0;
    N4M_TEST_REQUIRE(n4m_method_result_get_double_matrix(
                         result, key, &data, &rows, &cols) == N4M_OK);
    N4M_TEST_REQUIRE(data != nullptr);
    N4M_TEST_REQUIRE(rows == exp_rows);
    N4M_TEST_REQUIRE(cols == exp_cols);
    const std::size_t n = static_cast<std::size_t>(exp_rows * exp_cols);
    for (std::size_t i = 0; i < n; ++i) {
        const double diff = std::fabs(data[i] - expected[i]);
        if (diff > kTol) {
            throw std::runtime_error(
                std::string(key) + " mismatch at i=" + std::to_string(i) +
                " got=" + std::to_string(data[i]) +
                " want=" + std::to_string(expected[i]) +
                " diff=" + std::to_string(diff));
        }
    }
}

// Fit with cfg defaults (center on, scale off) + a single lambda passed via the
// lambdas array, returning the result handle (caller destroys).
n4m_method_result_t* fit(const double* X, std::int64_t n, std::int64_t p,
                         const double* Y, std::int64_t q, double lambda,
                         int scale_x) {
    n4m_context_t* ctx = nullptr;
    n4m_config_t* cfg = nullptr;
    N4M_TEST_REQUIRE(n4m_context_create(&ctx) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_create(&cfg) == N4M_OK);
    // Default config: center_x/center_y on, scale_x/scale_y on. Direct ridge
    // ignores scale_y; flip scale_x explicitly per case via the dedicated
    // setter (scale_x defaults to 1 in Config; sklearn default is no scaling,
    // so the no-scale cases force it off).
    N4M_TEST_REQUIRE(n4m_config_set_scale_x(cfg, scale_x) == N4M_OK);

    n4m_matrix_view_t Xv = make_view(X, n, p);
    n4m_matrix_view_t Yv = make_view(Y, n, q);
    n4m_method_result_t* result = nullptr;
    N4M_TEST_REQUIRE(
        n4m_ridge_fit(ctx, cfg, &Xv, &Yv, &lambda, 1, &result) == N4M_OK);
    N4M_TEST_REQUIRE(result != nullptr);

    double got_lambda = 0.0;
    N4M_TEST_REQUIRE(
        n4m_method_result_get_scalar(result, "lambda", &got_lambda) == N4M_OK);
    N4M_TEST_REQUIRE(std::fabs(got_lambda - lambda) <= kTol);

    n4m_config_destroy(cfg);
    n4m_context_destroy(ctx);
    return result;
}

void test_single_output() {
    for (const auto& c : kSingle) {
        n4m_method_result_t* r =
            fit(kX6x4, 6, 4, kY6x1, 1, c.lambda, /*scale_x=*/0);
        check_matrix(r, "coefficients", c.coef, 4, 1);
        check_matrix(r, "intercept", &c.intercept, 1, 1);
        check_matrix(r, "predictions", c.pred, 6, 1);
        n4m_method_result_destroy(r);
    }
}

void test_multi_output() {
    for (const auto& c : kMulti) {
        n4m_method_result_t* r =
            fit(kX6x4, 6, 4, kY6x2, 2, c.lambda, /*scale_x=*/0);
        check_matrix(r, "coefficients", c.coef, 4, 2);
        check_matrix(r, "intercept", c.intercept, 1, 2);
        check_matrix(r, "predictions", c.pred, 6, 2);
        n4m_method_result_destroy(r);
    }
}

void test_scale_x() {
    n4m_method_result_t* r = fit(kX6x4, 6, 4, kY6x1, 1, 1.0, /*scale_x=*/1);
    check_matrix(r, "coefficients", kScaleCoef, 4, 1);
    check_matrix(r, "intercept", &kScaleIntercept, 1, 1);
    check_matrix(r, "predictions", kScalePred, 6, 1);
    check_matrix(r, "x_scale", kScaleXScale, 1, 4);
    n4m_method_result_destroy(r);
}

// Wide (p>n) fit: AUTO picks the DUAL path. The reference was computed with the
// PRIMAL closed form, so matching it to 1e-10 demonstrates dual == primal.
void test_dual_equals_primal() {
    for (const auto& c : kWide) {
        n4m_method_result_t* r =
            fit(kXw, 4, 10, kYw, 1, c.lambda, /*scale_x=*/0);
        check_matrix(r, "coefficients", c.coef, 10, 1);
        check_matrix(r, "intercept", &c.intercept, 1, 1);
        check_matrix(r, "predictions", c.pred, 4, 1);
        n4m_method_result_destroy(r);
    }
}

void test_invalid_lambda_rejected() {
    n4m_context_t* ctx = nullptr;
    n4m_config_t* cfg = nullptr;
    N4M_TEST_REQUIRE(n4m_context_create(&ctx) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_config_create(&cfg) == N4M_OK);
    n4m_matrix_view_t Xv = make_view(kX6x4, 6, 4);
    n4m_matrix_view_t Yv = make_view(kY6x1, 6, 1);
    n4m_method_result_t* result = nullptr;
    double bad = std::nan("");
    N4M_TEST_REQUIRE(n4m_ridge_fit(ctx, cfg, &Xv, &Yv, &bad, 1, &result) ==
                     N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(result == nullptr);
    double neg = -1.0;
    N4M_TEST_REQUIRE(n4m_ridge_fit(ctx, cfg, &Xv, &Yv, &neg, 1, &result) ==
                     N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(result == nullptr);
    n4m_config_destroy(cfg);
    n4m_context_destroy(ctx);
}

}  // namespace

void register_ridge_tests(n4m_testing::Runner& r) {
    r.run("ridge/single_output", test_single_output);
    r.run("ridge/multi_output", test_multi_output);
    r.run("ridge/scale_x", test_scale_x);
    r.run("ridge/dual_equals_primal", test_dual_equals_primal);
    r.run("ridge/invalid_lambda_rejected", test_invalid_lambda_rejected);
}
