// SPDX-License-Identifier: CECILL-2.1
#include "n4m/metrics.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <utility>
#include <vector>

#include "core/diagnostics/preflight_pca_budget.hpp"
#include "svd_degenerate_goldens.inc"

#define REQUIRE(x) do { if (!(x)) throw std::runtime_error(#x); } while (false)
static bool near(double x, double y) { return std::abs(x-y) < 1e-9; }
static bool relative_near(double x, double y) {
    return std::abs(x-y) <= 1e-12 * std::max(1.0, std::abs(y));
}
static n4m_matrix_view_t view(double* x, int64_t n, int64_t p) {
    n4m_matrix_view_t v{};
    REQUIRE(n4m_matrix_view_init_rowmajor(&v, x, n, p, N4M_DTYPE_F64)==N4M_OK);
    return v;
}
// --- PCA magnitude / translation / constant-column invariance -------------
struct PcaRun {
    n4m_status_t status;
    std::vector<double> eigen, moments, scores, loadings;
};
static PcaRun run_pca(n4m_context_t* ctx, std::vector<double> x, int64_t n, int64_t p, int32_t k) {
    PcaRun r{N4M_OK, std::vector<double>(static_cast<size_t>(k)), std::vector<double>(2),
             std::vector<double>(static_cast<size_t>(n * k)),
             std::vector<double>(static_cast<size_t>(k * p))};
    auto X=view(x.data(),n,p), S=view(r.scores.data(),n,k), L=view(r.loadings.data(),k,p);
    r.status=n4m_preflight_pca_compute(ctx,&X,k,&S,&L,r.eigen.data(),r.moments.data());
    return r;
}
static bool rel_close(double a, double b, double tol) {
    return std::abs(a-b) <= tol*std::max(std::abs(a),std::abs(b));
}
// 6 x 5 integer design (exact under any power-of-two scaling); centred rank 5.
static const double kPcaBase[30]={3,-1,4,1,-5, 9,2,-6,5,3, -5,8,9,-7,9, 3,2,-3,8,4,
                                  -6,2,6,4,-3, 3,3,-8,3,2};
// Supported domain D of the PCA: nonzero |x| in [2^-128, 2^128] and
// max/min nonzero |x| <= 2^32 (rule 15); outside it N4M_ERR_UNSUPPORTED.
static void pca_invariance_tests(n4m_context_t* ctx) {
    const int64_t n=6, p=5; const int32_t k=3;
    const std::vector<double> base_x(kPcaBase,kPcaBase+30);
    const PcaRun base=run_pca(ctx,base_x,n,p,k);
    REQUIRE(base.status==N4M_OK && base.eigen[2]>0.0 && base.moments[1]>0.0);
    // Magnitude sweep X * 2^s (|x| in [1, 9]): inside D for s in [-128, 124]
    // with bit-exact power-of-two equivariance; outside it, refused.
    const int sweep[]={-1014,-1022,-500,-200,-129,-128,-100,0,100,124,125,200,500,1000};
    for (int s : sweep) {
        std::vector<double> x(base_x);
        for (double& v : x) v=std::ldexp(v,s);
        const PcaRun r=run_pca(ctx,x,n,p,k);
        if (s < -128 || s > 124) {
            REQUIRE(r.status==N4M_ERR_UNSUPPORTED);
            continue;
        }
        REQUIRE(r.status==N4M_OK);
        for (int32_t a=0;a<k;++a) REQUIRE(r.eigen[static_cast<size_t>(a)]==
                                          std::ldexp(base.eigen[static_cast<size_t>(a)],2*s));
        REQUIRE(r.moments[0]==std::ldexp(base.moments[0],2*s));
        REQUIRE(r.moments[1]==std::ldexp(base.moments[1],4*s));
        for (size_t i=0;i<r.scores.size();++i)
            REQUIRE(r.scores[i]==std::ldexp(base.scores[i],s));
        REQUIRE(r.loadings==base.loadings);
    }
    // D boundaries, each side: magnitude 2^128 / next above, 2^-128 / 2^-129,
    // spread 2^32 / 2^33.
    {
        const auto two=[](int e){ return std::ldexp(1.0,e); };
        const std::vector<double> hi={two(128),two(96),-two(128),two(97),two(127),-two(96),
                                      0,3*two(96),two(126),two(98),-two(127),0};
        PcaRun r=run_pca(ctx,hi,6,2,2);
        // lambda_2 / lambda_1 ~ 2^-64: below the resolution, published as 0.
        REQUIRE(r.status==N4M_OK && std::isfinite(r.moments[1]) && r.eigen[0]>0.0 && r.eigen[1]==0.0);
        std::vector<double> lo(hi);
        for (double& v : lo) v=std::ldexp(v,-224);           // [2^-128, 2^-96]
        const PcaRun rl=run_pca(ctx,lo,6,2,2);
        REQUIRE(rl.status==N4M_OK && rl.eigen[0]==std::ldexp(r.eigen[0],-448) &&
                rl.moments[1]==std::ldexp(r.moments[1],-896));
        std::vector<double> x(hi); x[0]=std::nextafter(two(128),HUGE_VAL);
        REQUIRE(run_pca(ctx,x,6,2,2).status==N4M_ERR_UNSUPPORTED);
        x=lo; for (double& v : x) v*=0.5;                    // [2^-129, 2^-97]
        REQUIRE(run_pca(ctx,x,6,2,2).status==N4M_ERR_UNSUPPORTED);
        x=hi; x[1]=two(95);                                  // spread 2^33
        REQUIRE(run_pca(ctx,x,6,2,2).status==N4M_ERR_UNSUPPORTED);
    }
    // Translation inside D: integer data + 2^40 and 16*X + 1e16 are exact.
    for (int t=0;t<2;++t) {
        std::vector<double> x(base_x);
        for (double& v : x) v = t==0 ? v+std::ldexp(1.0,40) : 16.0*v+1e16;
        const PcaRun r=run_pca(ctx,x,n,p,k);
        const double f = t==0 ? 1.0 : 256.0;
        REQUIRE(r.status==N4M_OK);
        for (int32_t a=0;a<k;++a)
            REQUIRE(rel_close(r.eigen[static_cast<size_t>(a)],f*base.eigen[static_cast<size_t>(a)],1e-12));
        REQUIRE(rel_close(r.moments[0],f*base.moments[0],1e-12));
        REQUIRE(rel_close(r.moments[1],f*f*base.moments[1],1e-12));
        for (size_t i=0;i<r.loadings.size();++i)
            REQUIRE(std::abs(r.loadings[i]-base.loadings[i])<=1e-12);
    }
    // Outside D: a 1e300 offset, a 1e300 constant column, a 2^34 column.
    {
        std::vector<double> x(base_x);
        for (double& v : x) v=std::ldexp(v,950)+1e300;
        REQUIRE(run_pca(ctx,x,n,p,k).status==N4M_ERR_UNSUPPORTED);
    }
    // Constant columns inside D (2^30, -3*2^28, 0.5, 0) are ignored: same
    // spectrum, scores and loadings, zero loading on the constant columns.
    const auto with_constants=[&](const double c[4]) {
        const int64_t pc=p+4;
        std::vector<double> x(static_cast<size_t>(n*pc));
        for (int64_t i=0;i<n;++i) {
            for (int64_t j=0;j<p;++j) x[static_cast<size_t>(i*pc+j+1)]=base_x[static_cast<size_t>(i*p+j)];
            x[static_cast<size_t>(i*pc)]=c[0];
            for (int64_t j=1;j<4;++j) x[static_cast<size_t>(i*pc+p+j)]=c[j];
        }
        return run_pca(ctx,x,n,pc,k);
    };
    {
        const double c[4]={std::ldexp(1.0,30),-3*std::ldexp(1.0,28),0.5,0.0};
        const PcaRun r=with_constants(c);
        const int64_t pc=p+4;
        REQUIRE(r.status==N4M_OK);
        for (int32_t a=0;a<k;++a)
            REQUIRE(rel_close(r.eigen[static_cast<size_t>(a)],base.eigen[static_cast<size_t>(a)],1e-12));
        REQUIRE(rel_close(r.moments[0],base.moments[0],1e-12));
        REQUIRE(rel_close(r.moments[1],base.moments[1],1e-12));
        for (int32_t a=0;a<k;++a) {
            const auto row=static_cast<size_t>(a*pc);
            REQUIRE(r.loadings[row]==0.0);
            for (int64_t j=1;j<4;++j) REQUIRE(r.loadings[row+static_cast<size_t>(p+j)]==0.0);
            for (int64_t j=0;j<p;++j)
                REQUIRE(std::abs(r.loadings[row+static_cast<size_t>(j)+1]-
                                 base.loadings[static_cast<size_t>(a*p+j)])<=1e-12);
        }
        for (size_t i=0;i<r.scores.size();++i)
            REQUIRE(std::abs(r.scores[i]-base.scores[i])<=1e-11);
        const double huge[4]={1e300,0.5,0.5,0.0}, wide[4]={std::ldexp(1.0,34),0.5,0.5,0.0};
        REQUIRE(with_constants(huge).status==N4M_ERR_UNSUPPORTED);
        REQUIRE(with_constants(wide).status==N4M_ERR_UNSUPPORTED);
    }
    // Review cases. [A; -A] with A = [[-1,1,-1],[-1,-1,-1],[1,-1,1],[-t,t,t]]:
    // analytic variances 2 sigma^2 / 7 with sigma = (sqrt17 +- 1)/2 (the
    // third, 4t^2/7 ~ 2^-61 lambda_1 for t = 2^-30, is unresolved). Spreads
    // 2^40, 2^60 and 1e40 are outside D (rule 15). The resolved weak case
    // t = 2^-20 is checked against mpmath in pca_resolution_tests.
    for (double t : {std::ldexp(1.0,-30), std::ldexp(1.0,-40), std::ldexp(1.0,-60), 1e-40}) {
        const double a4[12]={-1,1,-1, -1,-1,-1, 1,-1,1, -t,t,t};
        std::vector<double> x(24);
        for (size_t i=0;i<12;++i) { x[i]=a4[i]; x[12+i]=-a4[i]; }
        const PcaRun r=run_pca(ctx,x,8,3,3);
        if (t!=std::ldexp(1.0,-30)) { REQUIRE(r.status==N4M_ERR_UNSUPPORTED); continue; }
        REQUIRE(r.status==N4M_OK);
        const double s17=std::sqrt(17.0);
        REQUIRE(rel_close(r.eigen[0],2*(s17+1)*(s17+1)/28,1e-12));
        REQUIRE(rel_close(r.eigen[1],2*(s17-1)*(s17-1)/28,1e-12));
        // 4t^2/7 ~ 2^-61 lambda_1 is below the resolution (rule 14): the
        // 2.19 entry point publishes it as 0 (the 2.20 one also flags it).
        REQUIRE(r.eigen[2]==0.0);
    }
    {
        // diag-like 2^200 / 2^-450 rows: outside D.
        const std::vector<double> x={std::ldexp(1.0,200),0,-std::ldexp(1.0,200),0,
                                     0,std::ldexp(1.0,-450),0,-std::ldexp(1.0,-450)};
        REQUIRE(run_pca(ctx,x,4,2,2).status==N4M_ERR_UNSUPPORTED);
    }
}

// Rule 14 accuracy contract through the ABI 2.20 entry point, against the
// mpmath variances of svd_degenerate_goldens.inc, under row and column
// reversal. Families inside D (B_8(2^-30) plateau, B_8(2^-30) at 2^-97,
// B_6(2^-30) (x) 2x2, a weak but resolved variance) are computed; the
// review-round families outside D (spread > 2^32) are refused. Every resolved variance lies within the published
// resolution and is a normal double; every unresolved one is flagged with a
// zero variance and zero scores. Variances clearly above / below the bound
// must be classified accordingly.
static void pca_resolution_tests(n4m_context_t* ctx) {
    int unresolved_seen=0;
    for (const MpGolden& g : kMpGoldens) {
        const int64_t n=g.m, p=g.n; const auto k=static_cast<int32_t>(std::min(n,p));
        if (!g.preflight_domain) {
            REQUIRE(run_pca(ctx,std::vector<double>(g.A,g.A+n*p),n,p,k).status==
                    N4M_ERR_UNSUPPORTED);
            continue;
        }
        for (int perm=0; perm<4; ++perm) {
            std::vector<double> x(static_cast<size_t>(n*p)), sc(static_cast<size_t>(n*k)),
                ld(static_cast<size_t>(k*p)), ev(static_cast<size_t>(k)), mom(2);
            for (int64_t i=0;i<n;++i) for (int64_t j=0;j<p;++j)
                x[static_cast<size_t>(((perm&1)?n-1-i:i)*p+((perm&2)?p-1-j:j))]=
                    g.A[static_cast<size_t>(i*p+j)];
            std::vector<uint8_t> flag(static_cast<size_t>(k),9);
            double res=-1.0;
            auto X=view(x.data(),n,p), S=view(sc.data(),n,k), L=view(ld.data(),k,p);
            REQUIRE(n4m_preflight_pca_compute_resolved(ctx,&X,k,&S,&L,ev.data(),mom.data(),
                                                       &res,flag.data())==N4M_OK);
            REQUIRE(res==16.0*static_cast<double>(std::max(n,p))*
                         std::numeric_limits<double>::epsilon()*ev[0]);
            REQUIRE(res>=std::numeric_limits<double>::min());
            for (int32_t a=0;a<k;++a) {
                const auto ua=static_cast<size_t>(a);
                const double ref=g.variances[a];
                REQUIRE(flag[ua]<=1);
                if (ref>2.0*res) REQUIRE(flag[ua]==0);
                if (ref<0.5*res) REQUIRE(flag[ua]==1);
                if (flag[ua]==0) {
                    REQUIRE(std::abs(ev[ua]-ref)<=res && ev[ua]>=std::numeric_limits<double>::min());
                } else {
                    ++unresolved_seen;
                    REQUIRE(ev[ua]==0.0);
                    for (int64_t i=0;i<n;++i) REQUIRE(sc[static_cast<size_t>(i*k+a)]==0.0);
                }
            }
        }
    }
    // Per in-domain family and permutation: 1, 1, 2 and 0 unresolved.
    REQUIRE(unresolved_seen==4*(1+1+2+0));
}

// Every former automatic solver class of the PCA (truncated k on tall
// matrices with <= 64 columns, wide matrices with <= 256 rows, large tall
// and wide matrices that used a randomized sketch, full k) on analytic
// spectra: X = [D 0; -D 0] (2q x p, D = diag(d), zero padding to p
// columns) has centred covariance eigenvalues 2 d_i^2 / (2q - 1) and zeros.
// Review b8fix3: q = 128, d_i = 1 + i/128, k = 10 returned a first variance
// 7 % low with N4M_OK through the randomized sketch.
struct SolverCase { int64_t q, p; int32_t k; int64_t zero_tail; };
static void pca_solver_class_tests(n4m_context_t* ctx) {
    const SolverCase cases[]={
        {128,128,10,0},  // 256 x 128: former randomized (tall)
        {32,32,10,0},    // 64 x 32: former tall Gram
        {100,300,10,0},  // 200 x 300: former dual wide
        {150,400,10,0},  // 300 x 400: former randomized (wide)
        {24,24,24,4},    // 48 x 24, full k, 4 zero columns: 4 unresolved
        {10,40,20,0},    // 20 x 40, k = 20 > rank 10: wide completion
    };
    for (const SolverCase& c : cases) {
        const int64_t n=2*c.q, p=c.p; const int32_t k=c.k;
        std::vector<double> d(static_cast<size_t>(c.q));
        for (int64_t i=0;i<c.q;++i)
            d[static_cast<size_t>(i)]=i<c.q-c.zero_tail ? 1.0+static_cast<double>(i)/static_cast<double>(c.q) : 0.0;
        std::vector<double> x(static_cast<size_t>(n*p),0.0);
        for (int64_t i=0;i<c.q;++i) {
            x[static_cast<size_t>(i*p+i)]=d[static_cast<size_t>(i)];
            x[static_cast<size_t>((c.q+i)*p+i)]=-d[static_cast<size_t>(i)];
        }
        std::vector<double> ref(static_cast<size_t>(std::min(n,p)),0.0);
        for (int64_t i=0;i<c.q;++i)
            ref[static_cast<size_t>(i)]=2.0*d[static_cast<size_t>(i)]*d[static_cast<size_t>(i)]/
                                        static_cast<double>(n-1);
        std::sort(ref.begin(),ref.end(),std::greater<double>());
        std::vector<double> sc(static_cast<size_t>(n*k)), ld(static_cast<size_t>(k*p)),
            ev(static_cast<size_t>(k)), mom(2);
        std::vector<uint8_t> flag(static_cast<size_t>(k),9);
        double res=-1.0;
        auto X=view(x.data(),n,p), S=view(sc.data(),n,k), L=view(ld.data(),k,p);
        REQUIRE(n4m_preflight_pca_compute_resolved(ctx,&X,k,&S,&L,ev.data(),mom.data(),&res,
                                                   flag.data())==N4M_OK);
        REQUIRE(res==16.0*static_cast<double>(std::max(n,p))*
                     std::numeric_limits<double>::epsilon()*ev[0]);
        for (int32_t a=0;a<k;++a) {
            const double r=ref[static_cast<size_t>(a)];
            const size_t ua=static_cast<size_t>(a);
            if (r>0.0) {
                REQUIRE(flag[ua]==0 && std::abs(ev[ua]-r)<=res);
                // Scores are the centred data on the loadings (X is centred).
                for (int64_t i=0;i<n;++i) {
                    double s=0.0;
                    for (int64_t j=0;j<p;++j)
                        s+=x[static_cast<size_t>(i*p+j)]*ld[static_cast<size_t>(a*p+j)];
                    REQUIRE(std::abs(sc[static_cast<size_t>(i*k+a)]-s)<=1e-12);
                }
            } else {
                REQUIRE(flag[ua]==1 && ev[ua]==0.0);
                for (int64_t i=0;i<n;++i) REQUIRE(sc[static_cast<size_t>(i*k+a)]==0.0);
            }
        }
        // Loadings stay orthonormal, unresolved rows included.
        for (int32_t a=0;a<k;++a) for (int32_t b=0;b<=a;++b) {
            double dot=0.0;
            for (int64_t j=0;j<p;++j)
                dot+=ld[static_cast<size_t>(a*p+j)]*ld[static_cast<size_t>(b*p+j)];
            REQUIRE(std::abs(dot-(a==b?1.0:0.0))<=1e-12);
        }
    }
}

// Review b8fix4 P0-1 / P1: the unresolved loading rows of a wide input are a
// Householder completion of the resolved rows. Before, X = I_8 (k = 8)
// published a last loading of norm 0.354, and X = 0 (320 x 320, k = 320)
// cost Theta(n^4) through canonical-vector retries. Centred I_n has
// covariance (I - J/n)/(n-1): n-1 variances 1/(n-1) and one zero.
static void require_orthonormal_rows(const std::vector<double>& ld, int32_t k, int64_t p) {
    for (int32_t a=0;a<k;++a) for (int32_t b=0;b<=a;++b) {
        double dot=0.0;
        for (int64_t j=0;j<p;++j)
            dot+=ld[static_cast<size_t>(a*p+j)]*ld[static_cast<size_t>(b*p+j)];
        REQUIRE(std::abs(dot-(a==b?1.0:0.0))<=1e-12);
    }
}
static void pca_completion_tests(n4m_context_t* ctx) {
    struct Case { int64_t n; int kind; };  // 0: I_n, 1: zero, 2: [I_{n/2} 0; -I_{n/2} 0]
    for (const Case c : {Case{4,0}, Case{8,0}, Case{320,0}, Case{320,1}, Case{320,2}}) {
        const int64_t n=c.n, p=c.n; const auto k=static_cast<int32_t>(n);
        std::vector<double> x(static_cast<size_t>(n*p),0.0);
        int64_t rank=0;
        if (c.kind==0) { for (int64_t i=0;i<n;++i) x[static_cast<size_t>(i*p+i)]=1.0; rank=n-1; }
        if (c.kind==2) {
            for (int64_t i=0;i<n/2;++i) {
                x[static_cast<size_t>(i*p+i)]=1.0;
                x[static_cast<size_t>((n/2+i)*p+i)]=-1.0;
            }
            rank=n/2;
        }
        std::vector<double> sc(static_cast<size_t>(n*k)), ld(static_cast<size_t>(k*p)),
            ev(static_cast<size_t>(k)), mom(2);
        std::vector<uint8_t> flag(static_cast<size_t>(k),9);
        double res=-1.0;
        auto X=view(x.data(),n,p), S=view(sc.data(),n,k), L=view(ld.data(),k,p);
        REQUIRE(n4m_preflight_pca_compute_resolved(ctx,&X,k,&S,&L,ev.data(),mom.data(),&res,
                                                   flag.data())==N4M_OK);
        const double var=c.kind==0 ? 1.0/static_cast<double>(n-1) : 2.0/static_cast<double>(n-1);
        for (int32_t a=0;a<k;++a) {
            const auto ua=static_cast<size_t>(a);
            if (a<rank) REQUIRE(flag[ua]==0 && rel_close(ev[ua],var,1e-12));
            else REQUIRE(flag[ua]==1 && ev[ua]==0.0);
        }
        require_orthonormal_rows(ld,k,p);
    }
    // Graded wide input (12 x 16, column j scaled by 10^(-d j / 11)): the
    // rows Xc^T u / sigma of the weakest resolved components drifted up to
    // 5e-5 from orthonormality before their re-orthonormalisation.
    for (const double d : {3.0, 5.0, 6.0, 7.0}) {
        const int64_t n=12, p=16; const int32_t k=12;
        std::vector<double> x(static_cast<size_t>(n*p),0.0);
        for (int64_t i=0;i<n;++i) for (int64_t j=0;j<n;++j) {
            const double di=static_cast<double>(i), dj=static_cast<double>(j);
            x[static_cast<size_t>(i*p+j)]=(1.5+std::sin(1.0+0.7*di*dj+0.3*di*di+1.3*dj))*
                                          std::pow(10.0,-d*dj/11.0);
        }
        const PcaRun r=run_pca(ctx,x,n,p,k);
        REQUIRE(r.status==N4M_OK);
        require_orthonormal_rows(r.loadings,k,p);
    }
}

// Budget (reviews b8fix4 P1 and b8fix5 P1): guaranteed work and peak-memory
// bounds of every pass and buffer (core/diagnostics/preflight_pca_budget.hpp),
// tested by counting, without allocating the inputs they refuse.
static void pca_budget_tests(n4m_context_t* ctx) {
    using ::n4m::core::preflight_pca_budget;
    // Review b8fix5 counter-example: n = 3, p = 26e6, k = 1 was accepted at
    // 238.9 million products and 1.040 GB, although sum1, the Gram and the
    // projections alone need 312 million long-double products, and data,
    // col_exp, loadings and their Householder copy 1.144 GB.
    const auto wide = preflight_pca_budget(3, 26000000, 1);
    REQUIRE(wide.work >= 3.12e8L);
    REQUIRE(wide.doubles * 8.0L >= 1.144e9L);
    REQUIRE(wide.doubles > ::n4m::core::preflight_pca_max_doubles);
    // The QL iteration enters at its ceiling (30 s sweeps of s - 1 rotations
    // over s rows), not at a two-sweep estimate.
    const auto square = preflight_pca_budget(658, 658, 1);
    REQUIRE(square.work >= 30.0L * 658 * 657 * 4 * 658 / 16);
    REQUIRE(square.work <= ::n4m::core::preflight_pca_max_work);
    // No shape a Preflight request can produce (n p <= 1e7, k <= 10) that
    // Methods 1.3.6 (Gram-only s^2 l <= 2.5e8) or the previous estimate
    // (effdf9ad) accepted is refused. Both former counts and the new bounds
    // grow with l = max(n,p), so the largest accepted l per (s, k,
    // orientation) is the worst case.
    const auto old_accepts = [](long double n, long double p, long double k) {
        const long double s = std::min(n, p), l = std::max(n, p);
        if (s * s * l <= 2.5e8L) return true;
        const long double work = s * (s + 1) / 2 * l + k * s * l +
                                 (6 * s * s * s + (n <= p ? 3 * k * k * l : 0)) / 16;
        const long double doubles = n * p + 3 * s * s + 2 * k * p + n * k + 4 * s + k;
        return work <= 2.5e8L && doubles <= 134217728.0L;
    };
    long double worst = 0;
    for (int64_t sm = 1; sm * sm <= 10000000; ++sm)
        for (int64_t k = 1; k <= std::min<int64_t>(10, sm); ++k)
            for (const bool wide_side : {true, false}) {
                const auto shape = [&](int64_t l) {
                    return wide_side ? std::make_pair(sm, l) : std::make_pair(l, sm);
                };
                const auto accepted = [&](int64_t l) {
                    const auto [n, p] = shape(l);
                    return n >= 2 && old_accepts(static_cast<long double>(n),
                                                 static_cast<long double>(p),
                                                 static_cast<long double>(k));
                };
                int64_t lo = sm, hi = 10000000 / sm;
                if (!accepted(lo)) continue;
                while (lo < hi) {
                    const int64_t mid = lo + (hi - lo + 1) / 2;
                    if (accepted(mid)) lo = mid; else hi = mid - 1;
                }
                const auto [n, p] = shape(lo);
                const auto b = preflight_pca_budget(n, p, k);
                REQUIRE(b.work <= ::n4m::core::preflight_pca_max_work);
                REQUIRE(b.doubles <= ::n4m::core::preflight_pca_max_doubles);
                worst = std::max(worst, b.work);
            }
    REQUIRE(worst > 2.3e9L);  // 658 x 658, k = 1: the limit is tight
    // Through the ABI: 700 x 700 with k = 700 (3.2e9) is refused before
    // allocating; 600 x 600 with k = 600 (2.0e9) is now inside the bound.
    REQUIRE(preflight_pca_budget(700, 700, 700).work > ::n4m::core::preflight_pca_max_work);
    REQUIRE(preflight_pca_budget(600, 600, 600).work <= ::n4m::core::preflight_pca_max_work);
    const int64_t n=700, p=700; const int32_t k=700;
    std::vector<double> x(static_cast<size_t>(n*p),1.0), sc(static_cast<size_t>(n*k)),
        ld(static_cast<size_t>(k*p)), ev(static_cast<size_t>(k)), mom(2);
    auto X=view(x.data(),n,p), S=view(sc.data(),n,k), L=view(ld.data(),k,p);
    REQUIRE(n4m_preflight_pca_compute(ctx,&X,k,&S,&L,ev.data(),mom.data())==N4M_ERR_UNSUPPORTED);
}

// Constant columns, non-dyadic and rescaled, centre to exact zeros in the
// preflight PCA (the Hotelling centring defect of review b8fix4 P0-2 does not
// occur here): an all-constant X has no variance at all, and constant columns
// next to a varying one leave its variance and loading untouched.
static void pca_constant_column_tests(n4m_context_t* ctx) {
    for (const double s : {1.0, 3.0, 1e-7, 1e7, 0.1}) {
        for (const int64_t n : {int64_t{3}, int64_t{10}, int64_t{101}}) {
            const int64_t p=4; const int32_t k=3;
            std::vector<double> x(static_cast<size_t>(n*p));
            for (int64_t i=0;i<n;++i) {
                x[static_cast<size_t>(i*p)]=0.1*s;
                x[static_cast<size_t>(i*p+1)]=s/3.0;
                x[static_cast<size_t>(i*p+2)]=-2.7e-3*s;
                x[static_cast<size_t>(i*p+3)]=12345.678*s;
            }
            PcaRun r=run_pca(ctx,x,n,p,k);
            REQUIRE(r.status==N4M_OK && r.moments[0]==0.0 && r.moments[1]==0.0);
            for (double e : r.eigen) REQUIRE(e==0.0);
            for (double v : r.scores) REQUIRE(v==0.0);
            // Column 2 becomes i * s: variance s^2 n (n+1) / 12.
            for (int64_t i=0;i<n;++i) x[static_cast<size_t>(i*p+2)]=static_cast<double>(i)*s;
            r=run_pca(ctx,x,n,p,k);
            const double var=s*s*static_cast<double>(n)*static_cast<double>(n+1)/12.0;
            REQUIRE(r.status==N4M_OK && rel_close(r.eigen[0],var,1e-12));
            REQUIRE(r.eigen[1]==0.0 && r.eigen[2]==0.0);
            REQUIRE(std::abs(r.loadings[2]-1.0)<=1e-15);
            REQUIRE(r.loadings[0]==0.0 && r.loadings[1]==0.0 && r.loadings[3]==0.0);
        }
    }
}

// Review b8fix3 P0-2: k is validated before any allocation (k = INT_MAX on
// a 2 x 1 matrix used to request ~2 GiB, outside the exception barrier).
static void pca_component_count_tests(n4m_context_t* ctx) {
    double x[2]={1.0,2.0}, sc[2]={0,0}, ld[1]={0}, ev[1]={0}, mom[2]={0,0}, res=0.0;
    uint8_t flag[1]={0};
    auto X=view(x,2,1), S=view(sc,2,1), L=view(ld,1,1);
    const int32_t big=std::numeric_limits<int32_t>::max();
    REQUIRE(n4m_preflight_pca_compute(ctx,&X,big,&S,&L,ev,mom)==N4M_ERR_INVALID_ARGUMENT);
    REQUIRE(n4m_preflight_pca_compute_resolved(ctx,&X,big,&S,&L,ev,mom,&res,flag)==
            N4M_ERR_INVALID_ARGUMENT);
    REQUIRE(n4m_preflight_pca_compute(ctx,&X,0,&S,&L,ev,mom)==N4M_ERR_INVALID_ARGUMENT);
    REQUIRE(n4m_preflight_pca_compute(ctx,&X,1,&S,&L,ev,mom)==N4M_OK);
}

int main() {
    try {
        n4m_context_t* ctx=nullptr;
        REQUIRE(n4m_context_create(&ctx)==N4M_OK);
        // Rank-one 3x2: centered rows [-1,-2],[0,0],[1,2].
        // Covariance eigenvalues [5,0], so both moments equal 5 and 25.
        double x[6]={1,2,2,4,3,6};
        double scores[3]={}, loadings[2]={}, eigen[1]={}, moments[2]={};
        auto X=view(x,3,2), S=view(scores,3,1), L=view(loadings,1,2);
        REQUIRE(n4m_preflight_pca_compute(ctx,&X,1,&S,&L,eigen,moments)==N4M_OK);
        REQUIRE(near(eigen[0],5) && near(moments[0],5) && near(moments[1],25));
        REQUIRE(near(scores[0]*loadings[0],-1));
        REQUIRE(near(scores[2]*loadings[1],2));
        // Column-major input and strided output preserve the same answer.
        double xc[6]={1,2,3,2,4,6}, sc[6]={}, lc[4]={};
        n4m_matrix_view_t C{}, SO{}, LO{};
        REQUIRE(n4m_matrix_view_init_colmajor(&C,xc,3,2,N4M_DTYPE_F64)==N4M_OK);
        REQUIRE(n4m_matrix_view_init_strided(&SO,sc,3,1,2,1,N4M_DTYPE_F64)==N4M_OK);
        REQUIRE(n4m_matrix_view_init_strided(&LO,lc,1,2,1,2,N4M_DTYPE_F64)==N4M_OK);
        REQUIRE(n4m_preflight_pca_compute(ctx,&C,1,&SO,&LO,eigen,moments)==N4M_OK);
        REQUIRE(near(moments[0],5) && near(sc[4]*lc[2],2));
        // Truncated PCA must still include the omitted second eigenvalue
        // in both exact moments: lambda=[16/3,4/3].
        double full_rank[8]={2,1,2,-1,-2,1,-2,-1};
        double scores2[4]={}, loadings2[2]={};
        auto FR=view(full_rank,4,2), S2=view(scores2,4,1), L2=view(loadings2,1,2);
        REQUIRE(n4m_preflight_pca_compute(ctx,&FR,1,&S2,&L2,eigen,moments)==N4M_OK);
        REQUIRE(near(eigen[0],16.0/3.0));
        REQUIRE(near(moments[0],20.0/3.0) && near(moments[1],272.0/9.0));
        // +-1 spectra with row 2 the exact opposite of row 0 (rank-deficient,
        // repeated |entries|) used to return N4M_ERR_CONVERGENCE_FAILED from
        // the Jacobi SVD. Eigenvalues are test-only NumPy 2.2.6 eigvalsh
        // goldens; the centred rank is 3, so they exhaust the total variance.
        double pm1[24]={1,-1,-1,1,-1,-1, -1,-1,1,1,-1,-1, -1,1,1,-1,1,1, 1,1,-1,-1,-1,1};
        double scores3[12]={}, loadings3[18]={}, eigen3[3]={};
        auto PM=view(pm1,4,6), S3=view(scores3,4,3), L3=view(loadings3,3,6);
        REQUIRE(n4m_preflight_pca_compute(ctx,&PM,3,&S3,&L3,eigen3,moments)==N4M_OK);
        REQUIRE(near(eigen3[0],4.451454311908496) && near(eigen3[1],2.943891106099754));
        REQUIRE(near(eigen3[2],0.2713212486584137));
        REQUIRE(near(eigen3[0]+eigen3[1]+eigen3[2],moments[0]) && near(moments[0],23.0/3.0));
        REQUIRE(near(moments[1],257.0/9.0));
        pca_invariance_tests(ctx);
        pca_resolution_tests(ctx);
        pca_solver_class_tests(ctx);
        pca_component_count_tests(ctx);
        pca_completion_tests(ctx);
        pca_budget_tests(ctx);
        pca_constant_column_tests(ctx);
        // Weak decoupled variance: [A; -A; 26 zero rows], A = [[1,0,0],
        // [0,1e-9,0],[0,1e-9,1e-9]]. The second variance (NumPy 2.2.6:
        // 1.689e-19, i.e. 2.6e-18 lambda_1) is below the resolution (rule 14):
        // published as 0. The internal kernel test keeps its relative value.
        {
            std::vector<double> w(96, 0.0);
            const double a[9]={1,0,0, 0,1e-9,0, 0,1e-9,1e-9};
            for (int i=0;i<9;++i) { w[static_cast<size_t>(i)]=a[i]; w[static_cast<size_t>(9+i)]=-a[i]; }
            const PcaRun r=run_pca(ctx,w,32,3,2);
            REQUIRE(r.status==N4M_OK);
            REQUIRE(rel_close(r.eigen[0],0.06451612903225808,1e-12));
            REQUIRE(r.eigen[1]==0.0);
        }
        // Two balanced groups: within SS=4, MSW=2; between SS=16,
        // MSB=16 => ICC=(16-2)/(16+2)=7/9, pooled SD=sqrt(2).
        double repeats[4]={0,2,4,6}; int64_t ids[4]={10,10,20,20};
        auto R=view(repeats,4,1);
        double sd[1], icc[1]; int64_t ng[1]; int32_t state[1];
        REQUIRE(n4m_preflight_repeatability_compute(ctx,&R,ids,nullptr,sd,icc,ng,state)==N4M_OK);
        REQUIRE(near(sd[0],std::sqrt(2.0)) && near(icc[0],7.0/9.0));
        REQUIRE(ng[0]==2 && state[0]==0);
        // Equal group means, nonzero within variance => negative ICC=-1.
        double neg[4]={0,2,0,2}; R=view(neg,4,1);
        REQUIRE(n4m_preflight_repeatability_compute(ctx,&R,ids,nullptr,sd,icc,ng,state)==N4M_OK);
        REQUIRE(near(icc[0],-1));
        uint8_t mask[4]={1,1,1,0}; R=view(repeats,4,1);
        REQUIRE(n4m_preflight_repeatability_compute(ctx,&R,ids,mask,sd,icc,ng,state)==N4M_OK);
        REQUIRE(state[0]==2 && near(sd[0],std::sqrt(2.0)) && std::isnan(icc[0]));
        double flat[4]={3,3,3,3}; R=view(flat,4,1);
        REQUIRE(n4m_preflight_repeatability_compute(ctx,&R,ids,nullptr,sd,icc,ng,state)==N4M_OK);
        REQUIRE(state[0]==3 && near(sd[0],0) && std::isnan(icc[0]));
        double summary[8];
        REQUIRE(n4m_preflight_summary_compute(ctx,repeats,nullptr,4,summary)==N4M_OK);
        REQUIRE(near(summary[0],4) && near(summary[3],3));
        REQUIRE(near(summary[5],1.5) && near(summary[6],3) && near(summary[7],4.5));
        double extreme[3]={-1e308,0,1e308};
        REQUIRE(n4m_preflight_summary_compute(ctx,extreme,nullptr,3,summary)==N4M_OK);
        REQUIRE(near(summary[3],0) && relative_near(summary[4],1e308));
        REQUIRE(relative_near(summary[5],-5e307) && near(summary[6],0));
        REQUIRE(relative_near(summary[7],5e307));
        double unrepresentable[2]={1.7976931348623157e308,-1.7976931348623157e308};
        REQUIRE(n4m_preflight_summary_compute(ctx,unrepresentable,nullptr,2,summary)
                ==N4M_ERR_NUMERICAL_FAILURE);
        double extreme_repeat[4]={1e308,-1e308,1e308,-1e308};
        R=view(extreme_repeat,4,1);
        REQUIRE(n4m_preflight_repeatability_compute(ctx,&R,ids,nullptr,sd,icc,ng,state)==N4M_OK);
        REQUIRE(relative_near(sd[0],std::sqrt(2.0)*1e308) && near(icc[0],-1));
        double q[2]={0,0}, ref[4]={0,0,1,0};
        auto Q=view(q,1,2), F=view(ref,2,2);
        int64_t qi[1]={9}, ri[2]={9,10}, out_idx[2]; double out_dist[2];
        REQUIRE(n4m_preflight_nearest_neighbors(ctx,&Q,&F,qi,ri,2,out_idx,out_dist)==N4M_OK);
        REQUIRE(out_idx[0]==1 && near(out_dist[0],1) && out_idx[1]==-1 && std::isnan(out_dist[1]));
        double extreme_query[2]={1e308,1e308}, origin[2]={0,0};
        Q=view(extreme_query,1,2); F=view(origin,1,2);
        REQUIRE(n4m_preflight_nearest_neighbors(ctx,&Q,&F,nullptr,nullptr,1,out_idx,out_dist)
                ==N4M_OK);
        REQUIRE(relative_near(out_dist[0],std::sqrt(2.0)*1e308));
        double max_query[1]={1.7976931348623157e308};
        double min_ref[1]={-1.7976931348623157e308};
        Q=view(max_query,1,1); F=view(min_ref,1,1);
        REQUIRE(n4m_preflight_nearest_neighbors(ctx,&Q,&F,nullptr,nullptr,1,out_idx,out_dist)
                ==N4M_ERR_NUMERICAL_FAILURE);
        double continuous[3]={0,1,2}; int64_t local_idx[3]={1,2,1};
        double contrast[4]={};
        REQUIRE(n4m_preflight_local_contrast(ctx,continuous,local_idx,nullptr,3,1,0,contrast)==N4M_OK);
        REQUIRE(near(contrast[0],0.5) && near(contrast[1],1));
        REQUIRE(std::isnan(contrast[2]) && std::isnan(contrast[3]));
        // Unequal group sizes distinguish equal-row expected disagreement
        // (5/6) from pooled eligible-pair weighting (4/5).
        double classes[4]={0,0,1,1}; int64_t local_groups[4]={1,1,2,3};
        int64_t class_neighbors[4]={2,2,3,2};
        REQUIRE(n4m_preflight_local_contrast(ctx,classes,class_neighbors,local_groups,
                                             4,1,1,contrast)==N4M_OK);
        REQUIRE(near(contrast[0],0.4) && near(contrast[1],1));
        REQUIRE(near(contrast[2],0.5) && near(contrast[3],1.0/6.0));
        double one_class[3]={1,1,1};
        REQUIRE(n4m_preflight_local_contrast(ctx,one_class,local_idx,nullptr,3,1,1,contrast)==N4M_OK);
        // Factor association. Hand values: groups {1,2,3},{7,8,9}: SST=58,
        // SSB=54, eta^2=27/29, SSW=4, F=54 on (1,4); 6!/(3!3!)=20 splits.
        // Permutation counts b are test-only goldens from an independent
        // NumPy 2.2.6 / SciPy 1.15.3 reference: rng=np.random.default_rng(seed);
        // b = #{rng.permutation(n) draws with eta^2 (scipy.stats.f_oneway) or
        // Cramer V^2 (chi2_contingency, correction=False) >= observed-sqrt(eps)}.
        int64_t fa[6]={4,4,4,-9,-9,-9}; double ya[6]={1,2,3,7,8,9};
        double assoc[8]; int64_t perm_counts[2];
        REQUIRE(n4m_preflight_factor_association(ctx,fa,ya,6,0,999,17,assoc,perm_counts)==N4M_OK);
        REQUIRE(near(assoc[0],27.0/29.0) && near(assoc[1],54) && near(assoc[2],1) && near(assoc[3],4));
        REQUIRE(near(assoc[4],2) && near(assoc[5],6) && near(assoc[6],std::log10(20.0)));
        REQUIRE(perm_counts[0]==93 && perm_counts[1]==999 && near(assoc[7],94.0/1000.0));
        REQUIRE(n4m_preflight_factor_association(ctx,fa,ya,6,0,999,
                    UINT64_C(18446744073709551557),assoc,perm_counts)==N4M_OK);
        REQUIRE(perm_counts[0]==88);
        REQUIRE(n4m_preflight_factor_association(ctx,fa,ya,6,0,0,17,assoc,perm_counts)==N4M_OK);
        REQUIRE(near(assoc[0],27.0/29.0) && std::isnan(assoc[7]) && perm_counts[1]==0);
        // 2x2 table [[2,1],[1,3]]: chi2=175/144, V=5/12, df=1.
        int64_t fc[7]={0,0,0,1,1,1,1}; double yc[7]={0,0,1,1,1,1,0};
        REQUIRE(n4m_preflight_factor_association(ctx,fc,yc,7,1,999,17,assoc,perm_counts)==N4M_OK);
        REQUIRE(near(assoc[0],5.0/12.0) && near(assoc[1],175.0/144.0) && near(assoc[2],1));
        REQUIRE(std::isnan(assoc[3]) && near(assoc[5],2) && perm_counts[0]==477);
        REQUIRE(near(assoc[6],std::log10(35.0)));
        // Three levels with ties-free continuous values; F from SciPy.
        int64_t fd[12]={0,0,0,0,1,1,1,1,2,2,2,2};
        double yd[12]={-0.79,0.24,-1.9,1.4,1.44,0.51,0.49,1.1,1.33,1.37,2.32,2.11};
        REQUIRE(n4m_preflight_factor_association(ctx,fd,yd,12,0,999,5,assoc,perm_counts)==N4M_OK);
        REQUIRE(near(assoc[0],0.5318805854300557) && near(assoc[1],5.112931786078764));
        REQUIRE(perm_counts[0]==18 && near(assoc[7],19.0/1000.0));
        // Undefined effects run no permutation: one level, constant target.
        int64_t one_level[3]={1,1,1}; double spread[3]={1,2,3};
        REQUIRE(n4m_preflight_factor_association(ctx,one_level,spread,3,0,99,1,assoc,perm_counts)==N4M_OK);
        REQUIRE(std::isnan(assoc[0]) && near(assoc[4],1) && perm_counts[1]==0);
        int64_t two_levels[4]={1,1,2,2}; double zeros[4]={0.0,-0.0,0.0,-0.0};
        REQUIRE(n4m_preflight_factor_association(ctx,two_levels,zeros,4,0,99,1,assoc,perm_counts)==N4M_OK);
        REQUIRE(std::isnan(assoc[0]) && near(assoc[5],1) && std::isnan(assoc[7]));
        double fractional[4]={0,1,0.5,1};
        REQUIRE(n4m_preflight_factor_association(ctx,two_levels,fractional,4,1,0,1,assoc,perm_counts)
                ==N4M_ERR_INVALID_ARGUMENT);
        REQUIRE(n4m_preflight_factor_association(ctx,two_levels,spread,1,0,0,1,assoc,perm_counts)
                ==N4M_ERR_INVALID_ARGUMENT);
        // Extreme magnitudes are scaled before sums: eta^2 stays 27/29.
        double yh[6]={1e307,2e307,3e307,7e307,8e307,9e307};
        REQUIRE(n4m_preflight_factor_association(ctx,fa,yh,6,0,0,1,assoc,perm_counts)==N4M_OK);
        REQUIRE(near(assoc[0],27.0/29.0) && near(assoc[1],54));
        // Holm (R p.adjust "holm", hand-computed): sorted 0.005,0.01,0.03,0.04
        // -> 0.02, 0.03, 0.06, max(0.04,0.06).
        double pv[4]={0.01,0.04,0.03,0.005}, adj[4];
        REQUIRE(n4m_preflight_holm_adjust(ctx,pv,4,adj)==N4M_OK);
        REQUIRE(near(adj[0],0.03) && near(adj[1],0.06) && near(adj[2],0.06) && near(adj[3],0.02));
        double tied[3]={0.02,0.02,0.9}, adj3[3];
        REQUIRE(n4m_preflight_holm_adjust(ctx,tied,3,adj3)==N4M_OK);
        REQUIRE(near(adj3[0],0.06) && near(adj3[1],0.06) && near(adj3[2],0.9));
        double capped[2]={0.7,0.8};
        REQUIRE(n4m_preflight_holm_adjust(ctx,capped,2,adj)==N4M_OK);
        REQUIRE(near(adj[0],1) && near(adj[1],1));
        double invalid_p[1]={1.5};
        REQUIRE(n4m_preflight_holm_adjust(ctx,invalid_p,1,adj)==N4M_ERR_INVALID_ARGUMENT);
        REQUIRE(std::isnan(contrast[0]) && near(contrast[2],1) && near(contrast[3],1));
        // Noise proxy, window 5 / quadratic: SG weights (-3,12,17,12,-3)/35,
        // white-noise residual gain^2 = (18/35)^2 + 2(12/35)^2 + 2(3/35)^2
        // = 18/35. Row 0 is quadratic (zero interior residual). Row 1 is
        // linear plus a 35 impulse at band 5: residuals 3,-12,18,-12,3 at
        // bands 3..7. Edge bands 0,1,9,10 are not padded.
        double noise[22];
        for (int j=0;j<11;++j) { noise[j]=j*j; noise[11+j]=1+2*j+(j==5?35:0); }
        double band[11], pooled2[2]; int64_t counts[11];
        auto N=view(noise,2,11);
        REQUIRE(n4m_preflight_noise_proxy_compute(ctx,&N,nullptr,5,2,band,counts,pooled2)==N4M_OK);
        REQUIRE(near(pooled2[1],std::sqrt(18.0/35.0)));
        REQUIRE(near(band[5],std::sqrt(315.0)) && near(band[4],std::sqrt(140.0)));
        REQUIRE(near(band[6],std::sqrt(140.0)) && near(band[3],std::sqrt(8.75)));
        REQUIRE(near(band[2],0) && near(band[8],0) && counts[5]==2);
        for (int j : {0,1,9,10}) REQUIRE(std::isnan(band[j]) && counts[j]==0);
        REQUIRE(near(pooled2[0],std::sqrt(87.5)));
        // Column-major storage gives the same result.
        double noise_c[22];
        for (int i=0;i<2;++i) for (int j=0;j<11;++j) noise_c[j*2+i]=noise[i*11+j];
        n4m_matrix_view_t NC{};
        REQUIRE(n4m_matrix_view_init_colmajor(&NC,noise_c,2,11,N4M_DTYPE_F64)==N4M_OK);
        double band_c[11], pooled_c[2];
        REQUIRE(n4m_preflight_noise_proxy_compute(ctx,&NC,nullptr,5,2,band_c,counts,pooled_c)==N4M_OK);
        REQUIRE(near(band_c[5],band[5]) && near(pooled_c[0],pooled2[0]));
        // Masked band 9 of row 1 (value ignored) removes that row from the
        // windows of bands 7 and 8 only: pooled sum 324+288+9 over 12 cells.
        uint8_t noise_mask[22]; std::fill(noise_mask,noise_mask+22,uint8_t{1});
        noise_mask[11+9]=0; noise[11+9]=std::nan("");
        REQUIRE(n4m_preflight_noise_proxy_compute(ctx,&N,noise_mask,5,2,band,counts,pooled2)==N4M_OK);
        REQUIRE(counts[6]==2 && counts[7]==1 && counts[8]==1 && near(band[7],0));
        REQUIRE(near(pooled2[0],std::sqrt(621.0/12.0)/std::sqrt(18.0/35.0)));
        REQUIRE(n4m_preflight_noise_proxy_compute(ctx,&N,nullptr,5,2,band,counts,pooled2)
                ==N4M_ERR_INVALID_ARGUMENT);
        noise[11+9]=0;
        REQUIRE(n4m_preflight_noise_proxy_compute(ctx,&N,nullptr,4,2,band,counts,pooled2)
                ==N4M_ERR_INVALID_ARGUMENT);
        REQUIRE(n4m_preflight_noise_proxy_compute(ctx,&N,nullptr,5,4,band,counts,pooled2)
                ==N4M_ERR_INVALID_ARGUMENT);
        auto narrow=view(noise,2,3);
        REQUIRE(n4m_preflight_noise_proxy_compute(ctx,&narrow,nullptr,5,2,band,counts,pooled2)
                ==N4M_ERR_INVALID_ARGUMENT);
        // Statistical reference: zero signal plus uniform white noise of
        // variance 1 (deterministic LCG); the normalized proxy estimates 1.
        {
            const int64_t wn=2000, wp=40;
            std::vector<double> white(static_cast<size_t>(wn*wp));
            uint64_t state_lcg=12345;
            for (auto& v : white) {
                state_lcg=state_lcg*6364136223846793005ULL+1442695040888963407ULL;
                v=std::sqrt(12.0)*((state_lcg>>11)*(1.0/9007199254740992.0)-0.5);
            }
            auto W=view(white.data(),wn,wp);
            std::vector<double> wband(wp); std::vector<int64_t> wcount(wp); double wpool[2];
            REQUIRE(n4m_preflight_noise_proxy_compute(ctx,&W,nullptr,11,2,wband.data(),
                                                      wcount.data(),wpool)==N4M_OK);
            REQUIRE(std::abs(wpool[0]-1.0)<0.02 && wcount[20]==wn);
        }
        // Pairs within tolerance 0.1 (RMS over two features). Groups exclude
        // (0,4); pairs (0,1) and (1,4) have distance sqrt(0.01/2).
        double pts[10]={0,0, 0.1,0, 0,0.3, 5,5, 0,0};
        int64_t pgroups[5]={1,2,3,4,1}, pidx[4]={-7,-7,-7,-7}, pcounts[2];
        double pdist[2]={}; uint8_t pflags[5];
        auto P=view(pts,5,2);
        REQUIRE(n4m_preflight_pairs_within_tolerance(ctx,&P,nullptr,pgroups,nullptr,0.1,2,
                                                     pidx,pdist,pflags,nullptr,pcounts)==N4M_OK);
        REQUIRE(pcounts[0]==2 && pcounts[1]==9);
        REQUIRE(pidx[0]==0 && pidx[1]==1 && pidx[2]==1 && pidx[3]==4);
        REQUIRE(near(pdist[0],std::sqrt(0.005)) && near(pdist[1],std::sqrt(0.005)));
        REQUIRE(pflags[0]==1 && pflags[1]==1 && pflags[2]==0 && pflags[3]==0 && pflags[4]==1);
        // Truncated listing keeps the exact total; no groups => 10 candidates.
        REQUIRE(n4m_preflight_pairs_within_tolerance(ctx,&P,nullptr,nullptr,nullptr,0.1,1,
                                                     pidx,pdist,pflags,nullptr,pcounts)==N4M_OK);
        REQUIRE(pcounts[0]==3 && pcounts[1]==10 && pidx[0]==0 && pidx[1]==1);
        REQUIRE(n4m_preflight_pairs_within_tolerance(ctx,&P,nullptr,nullptr,nullptr,0.0,0,
                                                     nullptr,nullptr,pflags,nullptr,pcounts)
                ==N4M_ERR_INVALID_ARGUMENT);
        // Two matrices: one query, two references; only reference 0 is near.
        double one[2]={0,0}, two[4]={0,0.1, 1,1};
        auto O=view(one,1,2), T=view(two,2,2);
        uint8_t qflag[1], rflag[2];
        REQUIRE(n4m_preflight_pairs_within_tolerance(ctx,&O,&T,nullptr,nullptr,0.1,0,
                                                     nullptr,nullptr,qflag,rflag,pcounts)==N4M_OK);
        REQUIRE(pcounts[0]==1 && pcounts[1]==2 && qflag[0]==1 && rflag[0]==1 && rflag[1]==0);
        // An overflowing difference is far, not an error.
        Q=view(max_query,1,1); F=view(min_ref,1,1);
        REQUIRE(n4m_preflight_pairs_within_tolerance(ctx,&Q,&F,nullptr,nullptr,1.0,0,
                                                     nullptr,nullptr,qflag,rflag,pcounts)==N4M_OK);
        REQUIRE(pcounts[0]==0 && pcounts[1]==1 && qflag[0]==0);
        n4m_context_destroy(ctx);
        std::puts("preflight diagnostics: ok");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }
}
