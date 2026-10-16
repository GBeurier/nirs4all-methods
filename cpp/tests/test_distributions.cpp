// SPDX-License-Identifier: CECILL-2.1
// Chi-square and F quantiles against SciPy 1.18.1 stats.chi2.ppf / stats.f.ppf
// (parity/generators/variance_components_v1.py, "quantile_grid"), including
// tails at 1e-12 and 1 - 1e-12 and degrees of freedom from 1/2 to 1e8, and
// the domain boundary (tail 2^-64 accepted, 2^-65 refused). The two
// lower-tail points at df = 1e8 come from mpmath (SciPy is ~3e-6 off there).
#include "core/common/distributions.hpp"

#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <vector>

namespace dist = n4m::core::distributions;

static const double chi_grid[][3] = {
    {1e-12, 0.5, 1.3499395786223522e-48},
    {1e-12, 1, 1.5707963267948931e-24},
    {1e-12, 7, 0.0015031450349627595},
    {1e-12, 100, 30.084167586161836},
    {1e-12, 10000.0, 9037.257772358387},
    {1e-12, 1000000.0, 990084.0366937304},
    {1e-12, 100000000.0, 99900549.69599305},
    {1e-06, 0.5, 1.3499395786223394e-24},
    {1e-06, 1, 1.5707963267957187e-12},
    {1e-06, 7, 0.07852360099036994},
    {1e-06, 100, 46.50133071589314},
    {1e-06, 10000.0, 9342.10190723289},
    {1e-06, 1000000.0, 993292.0337421375},
    {1e-06, 100000000.0, 99932790.8248601},
    {0.025, 0.5, 5.273202591259991e-07},
    {0.025, 1, 0.0009820691171752555},
    {0.025, 7, 1.689869180677355},
    {0.025, 100, 74.22192747492373},
    {0.025, 10000.0, 9724.718377389798},
    {0.025, 1000000.0, 997230.0871432901},
    {0.025, 100000000.0, 99972283.81786752},
    {0.5, 0.5, 0.0873476047057468},
    {0.5, 1, 0.454936423119572},
    {0.5, 7, 6.345811195521515},
    {0.5, 100, 99.33412923598846},
    {0.5, 10000.0, 9999.333341235144},
    {0.5, 1000000.0, 999999.3333334123},
    {0.5, 100000000.0, 99999999.33333333},
    {0.975, 0.5, 3.433235299960771},
    {0.975, 1, 5.023886187314888},
    {0.975, 7, 16.012764274629326},
    {0.975, 100, 129.5611971858366},
    {0.975, 10000.0, 10279.07017988759},
    {0.975, 1000000.0, 1002773.701467926},
    {0.975, 100000000.0, 100027719.97074424},
    {0.999999, 0.5, 21.375635152821964},
    {0.999999, 1, 23.92812697687947},
    {0.999999, 7, 40.52183123411472},
    {0.999999, 100, 182.12677711942618},
    {0.999999, 10000.0, 10686.689829866038},
    {0.999999, 1000000.0, 1006736.7596362823},
    {0.999999, 100000000.0, 100067237.96852988},
    {0.999999999999, 0.5, 47.863796878759935},
    {0.999999999999, 1, 50.84417133244917},
    {0.999999999999, 7, 70.838475756726},
    {0.999999999999, 100, 233.81064336510855},
    {0.999999999999, 10000.0, 11027.380395779},
    {0.999999999999, 1000000.0, 1009980.6129055015},
    {0.999999999999, 100000000.0, 100099514.99294361}
};
static const double f_grid[][4] = {
    {1e-12, 1, 1, 2.4674011002723395e-24},
    {1e-12, 2, 1, 1.0000000000014999e-12},
    {1e-12, 3, 8, 7.60774342524097e-09},
    {1e-12, 8, 3, 0.0002996661723167804},
    {1e-12, 999, 2000, 0.6734255493943277},
    {1e-12, 1, 1000000.0, 1.5707971121932565e-24},
    {1e-12, 1000000.0, 1, 0.019667444781195306},
    {1e-12, 500000.0, 500000.0, 0.9802999341360005},
    {1e-06, 1, 1, 2.467401100276398e-12},
    {1e-06, 2, 1, 1.000001500002e-06},
    {1e-06, 3, 8, 7.608220904816313e-05},
    {1e-06, 8, 3, 0.009738713188269256},
    {1e-06, 999, 2000, 0.7672086120294419},
    {1e-06, 1, 1000000.0, 1.5707971121940792e-12},
    {1e-06, 1000000.0, 1, 0.041791300127639326},
    {1e-06, 500000.0, 500000.0, 0.9866452056455457},
    {0.025, 1, 1, 0.001543712508674131},
    {0.025, 2, 1, 0.025969756738987507},
    {0.025, 3, 8, 0.06877632746011883},
    {0.025, 8, 3, 0.1846393932054552},
    {0.025, 999, 2000, 0.8972566201682667},
    {0.025, 1, 1000000.0, 0.0009820696086921678},
    {0.025, 1000000.0, 1, 0.1990484956813504},
    {0.025, 500000.0, 500000.0, 0.9944717158969691},
    {0.5, 1, 1, 1.0},
    {0.5, 2, 1, 1.5},
    {0.5, 3, 8, 0.860037808522182},
    {0.5, 8, 3, 1.1627395796916387},
    {0.5, 999, 2000, 0.9996659540641715},
    {0.5, 1, 1000000.0, 0.45493675407152634},
    {0.5, 1000000.0, 1, 2.1981077392634174},
    {0.5, 500000.0, 500000.0, 1.0},
    {0.975, 1, 1, 647.7890114778442},
    {0.975, 2, 1, 799.4999999999987},
    {0.975, 3, 8, 5.415962339560236},
    {0.975, 8, 3, 14.53988657041723},
    {0.975, 999, 2000, 1.1123344200243792},
    {0.975, 1, 1000000.0, 5.023901319007523},
    {0.975, 1000000.0, 1, 1018.2577600906607},
    {0.975, 500000.0, 500000.0, 1.005559015922383},
    {0.999999, 1, 1, 405284734545.376},
    {0.999999, 2, 1, 499999999970.74426},
    {0.999999, 3, 8, 102.68297059997138},
    {0.999999, 8, 3, 13143.677247170725},
    {0.999999, 999, 2000, 1.2927398159476757},
    {0.999999, 1, 1000000.0, 23.92842522112206},
    {0.999999, 1000000.0, 1, 636619454020.8284},
    {0.999999, 500000.0, 500000.0, 1.0135355589608324},
    {0.999999999999, 1, 1, 4.053026663553072e+23},
    {0.999999999999, 2, 1, 5.00022122454199e+23},
    {0.999999999999, 3, 8, 3337.0651336352134},
    {0.999999999999, 8, 3, 131446960.31250687},
    {0.999999999999, 999, 2000, 1.459524161503745},
    {0.999999999999, 1, 1000000.0, 50.845489342512074},
    {0.999999999999, 1000000.0, 1, 6.366476212272036e+23},
    {0.999999999999, 500000.0, 500000.0, 1.0200959664519045}
};

int main() {
    int failures = 0;
    double worst = 0.0;
    auto check = [&](double got, double want, const char* what, double p, double a, double b) {
        const double err = std::abs(got - want) / std::abs(want);
        if (err > worst) worst = err;
        if (!(err <= 1e-10)) {
            std::fprintf(stderr, "%s p=%.17g a=%g b=%g got %.17g want %.17g\n", what, p, a, b, got, want);
            ++failures;
        }
    };
    // Grid points p > 1/2 are upper quantiles of the tail 1 - p (exact in double).
    auto chi = [](double p, double df) {
        return p <= 0.5 ? dist::chi2_quantile(p, df, false) : dist::chi2_quantile(1.0 - p, df, true);
    };
    auto fq = [](double p, double d1, double d2) {
        return p <= 0.5 ? dist::f_quantile(p, d1, d2, false) : dist::f_quantile(1.0 - p, d1, d2, true);
    };
    for (const auto& r : chi_grid) check(chi(r[0], r[1]), r[2], "chi2", r[0], r[1], 0);
    for (const auto& r : f_grid) check(fq(r[0], r[1], r[2]), r[3], "F", r[0], r[1], r[2]);
    // Domain boundary, inside: tail 2^-64 on both sides. SciPy 1.18.1 for the
    // chi-square and the lower F tail; mpmath 1.4.1 (40 digits) for the upper
    // F tail, where SciPy returns inf.
    const double t = 0x1p-64;
    const double chi_bound[][3] = {
        {1, 4.616155521099511e-39, 83.81940406418438},
        {7, 1.2616877861160633e-05, 106.2800472580404},
        {100, 19.400062200564484, 287.1767636672565},
        {1e4, 8769.622913389314, 11338.955350596494}};
    for (const auto& r : chi_bound) {
        check(dist::chi2_quantile(t, r[0], false), r[1], "chi2 lower 2^-64", t, r[0], 0);
        check(dist::chi2_quantile(t, r[0], true), r[2], "chi2 upper 2^-64", t, r[0], 0);
    }
    const double f_bound[][4] = {
        {3, 8, 1.0897060773801424e-13, 218886.00202842976},
        {8, 3, 4.568588172532459e-06, 9176786481765.684},
        {999, 2000, 0.5978144385529399, 1.625933698523698}};
    for (const auto& r : f_bound) {
        check(dist::f_quantile(t, r[0], r[1], false), r[2], "F lower 2^-64", t, r[0], r[1]);
        check(dist::f_quantile(t, r[0], r[1], true), r[3], "F upper 2^-64", t, r[0], r[1]);
    }
    // Closed form of F(2, nu) (review b9fix): P(F <= q) = 1 - (1 + 2q/nu)^(-nu/2),
    // so the lower quantile of p is (nu/2) expm1(-2 log1p(-p)/nu) and the upper
    // quantile of the tail p is (nu/2) expm1(-2 log(p)/nu); F(nu, 2) follows
    // by reciprocity, its lower quantile being 1 / the upper one of F(2, nu).
    // Covers nu up to 2^31 and both tails down to 2^-64.
    std::vector<double> nus{0.5, 1.0, 3.0, 10.0, 1e3, 1e6};
    for (int k = 0; k <= 31; ++k) nus.push_back(std::ldexp(1.0, k));
    for (double nu : nus) {
        for (double p : {0x1p-64, 1e-12, 1e-6, 0.025, 0.25, 0.5}) {
            const double lower = 0.5 * nu * std::expm1(-2.0 * std::log1p(-p) / nu);
            const double upper = 0.5 * nu * std::expm1(-2.0 * std::log(p) / nu);
            check(dist::f_quantile(p, 2, nu, false), lower, "F(2,nu) lower", p, 2, nu);
            check(dist::f_quantile(p, 2, nu, true), upper, "F(2,nu) upper", p, 2, nu);
            check(dist::f_quantile(p, nu, 2, false), 1.0 / upper, "F(nu,2) lower", p, nu, 2);
            check(dist::f_quantile(p, nu, 2, true), 1.0 / lower, "F(nu,2) upper", p, nu, 2);
        }
    }
    // Closed form of chi-square(2): P = 1 - e^{-x/2}; lower quantile
    // -2 log1p(-p), upper quantile -2 log(p).
    for (double p : {0x1p-64, 1e-12, 1e-6, 0.025, 0.25, 0.5}) {
        check(dist::chi2_quantile(p, 2, false), -2.0 * std::log1p(-p), "chi2(2) lower", p, 2, 0);
        check(dist::chi2_quantile(p, 2, true), -2.0 * std::log(p), "chi2(2) upper", p, 2, 0);
    }
    // Outside the domain (tail below 2^-64 or above 1/2, df outside
    // [1/2, 2^31]): NaN, never a silent value.
    const double out_tail = 0x1p-65;
    if (!std::isnan(dist::chi2_quantile(out_tail, 3, false)) ||
        !std::isnan(dist::chi2_quantile(out_tail, 3, true)) ||
        !std::isnan(dist::chi2_quantile(0.5000001, 3, false)) ||
        !std::isnan(dist::chi2_quantile(0.25, 0.25, false)) ||
        !std::isnan(dist::f_quantile(out_tail, 2, 3, true)) ||
        !std::isnan(dist::f_quantile(0.25, 3, 0x1p32, false)))
        ++failures;
    std::printf("distributions: worst relative error %.3g\n", worst);
    return failures ? 1 : 0;
}
