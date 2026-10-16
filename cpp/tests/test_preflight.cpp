// SPDX-License-Identifier: CECILL-2.1
#include "n4m/metrics.h"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "svd_degenerate_goldens.inc"

#define REQUIRE(x) do { if (!(x)) throw std::runtime_error(#x); } while (false)
static bool near(double x, double y) { return std::abs(x-y) < 1e-9; }
// Relative error <= r against a nonzero expectation (0 and sign flips fail).
static bool rel(double x, double y, double r) {
    return std::isfinite(x) && (y == 0 ? x == 0 : std::abs(x - y) <= r * std::abs(y));
}
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
        // Domain edge 2^256: SD of [-a, 0, a] is a, quartiles -a/2 and a/2.
        const double edge=std::ldexp(1.0,256);
        double extreme[3]={-edge,0,edge};
        REQUIRE(n4m_preflight_summary_compute(ctx,extreme,nullptr,3,summary)==N4M_OK);
        REQUIRE(summary[3]==0 && summary[4]==edge);
        REQUIRE(summary[5]==-edge/2 && summary[6]==0 && summary[7]==edge/2);
        double unrepresentable[2]={1.7976931348623157e308,-1.7976931348623157e308};
        REQUIRE(n4m_preflight_summary_compute(ctx,unrepresentable,nullptr,2,summary)
                ==N4M_ERR_UNSUPPORTED);
        double extreme_repeat[4]={1e308,-1e308,1e308,-1e308};
        R=view(extreme_repeat,4,1);
        REQUIRE(n4m_preflight_repeatability_compute(ctx,&R,ids,nullptr,sd,icc,ng,state)==N4M_OK);
        REQUIRE(relative_near(sd[0],std::sqrt(2.0)*1e308) && near(icc[0],-1));
        double q[2]={0,0}, ref[4]={0,0,1,0};
        auto Q=view(q,1,2), F=view(ref,2,2);
        int64_t qi[1]={9}, ri[2]={9,10}, out_idx[2]; double out_dist[2];
        REQUIRE(n4m_preflight_nearest_neighbors(ctx,&Q,&F,qi,ri,2,out_idx,out_dist)==N4M_OK);
        REQUIRE(out_idx[0]==1 && near(out_dist[0],1) && out_idx[1]==-1 && std::isnan(out_dist[1]));
        double extreme_query[2]={edge,edge}, origin[2]={0,0};
        Q=view(extreme_query,1,2); F=view(origin,1,2);
        REQUIRE(n4m_preflight_nearest_neighbors(ctx,&Q,&F,nullptr,nullptr,1,out_idx,out_dist)
                ==N4M_OK);
        REQUIRE(rel(out_dist[0],std::sqrt(2.0)*edge,4e-16));
        double max_query[1]={1.7976931348623157e308};
        double min_ref[1]={-1.7976931348623157e308};
        Q=view(max_query,1,1); F=view(min_ref,1,1);
        REQUIRE(n4m_preflight_nearest_neighbors(ctx,&Q,&F,nullptr,nullptr,1,out_idx,out_dist)
                ==N4M_ERR_UNSUPPORTED);
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
        // Large magnitudes inside the supported domain are scaled before
        // sums: eta^2 stays 27/29; 1e307 lies outside D and is refused.
        double yh[6]={1e70,2e70,3e70,7e70,8e70,9e70};
        REQUIRE(n4m_preflight_factor_association(ctx,fa,yh,6,0,0,1,assoc,perm_counts)==N4M_OK);
        REQUIRE(near(assoc[0],27.0/29.0) && near(assoc[1],54));
        for (double& v : yh) v *= 1e237;
        REQUIRE(n4m_preflight_factor_association(ctx,fa,yh,6,0,0,1,assoc,perm_counts)==N4M_ERR_UNSUPPORTED);
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
        // Correlation test. Hand values: x=1..5, y={2,4,5,4,5}: Pearson
        // r=6/sqrt(60); Spearman on average ranks y={1,2.5,4.5,2.5,4.5}:
        // rho=7/sqrt(90); 5!/(1!2!2!)=30 distinct y arrangements. Permutation
        // counts b are test-only goldens from an independent NumPy 2.2.6 /
        // SciPy 1.15.3 reference: rng=np.random.default_rng(seed); b = #{draws
        // perm=rng.permutation(n) with |scipy.stats.pearsonr/spearmanr(x,
        // y[perm]).statistic| >= |observed| - sqrt(eps)} over B=999.
        double cx[5]={1,2,3,4,5}, cy[5]={2,4,5,4,5}, corr[5]; int64_t corr_counts[2];
        REQUIRE(n4m_stats_correlation_test(ctx,cx,cy,5,0,999,17,corr,corr_counts)==N4M_OK);
        REQUIRE(relative_near(corr[0],6.0/std::sqrt(60.0)) && near(corr[1],5) && near(corr[2],3));
        REQUIRE(near(corr[3],std::log10(30.0)));
        REQUIRE(corr_counts[0]==133 && corr_counts[1]==999 && near(corr[4],134.0/1000.0));
        REQUIRE(n4m_stats_correlation_test(ctx,cx,cy,5,1,999,17,corr,corr_counts)==N4M_OK);
        REQUIRE(relative_near(corr[0],7.0/std::sqrt(90.0)) && corr_counts[0]==201);
        REQUIRE(n4m_stats_correlation_test(ctx,cx,cy,5,0,999,
                    UINT64_C(18446744073709551557),corr,corr_counts)==N4M_OK);
        REQUIRE(corr_counts[0]==124);
        REQUIRE(n4m_stats_correlation_test(ctx,cx,cy,5,1,999,
                    UINT64_C(18446744073709551557),corr,corr_counts)==N4M_OK);
        REQUIRE(corr_counts[0]==190);
        REQUIRE(n4m_stats_correlation_test(ctx,cx,cy,5,1,0,17,corr,corr_counts)==N4M_OK);
        REQUIRE(std::isnan(corr[4]) && corr_counts[1]==0);
        // Ties in both vectors; SciPy coefficients and NumPy counts.
        double tx[10]={1,1,2,2,3,3,4,4,5,5}, ty[10]={3,1,4,1,5,9,2,6,5,3};
        REQUIRE(n4m_stats_correlation_test(ctx,tx,ty,10,0,999,17,corr,corr_counts)==N4M_OK);
        REQUIRE(relative_near(corr[0],0.33196440215590134) && corr_counts[0]==373);
        REQUIRE(n4m_stats_correlation_test(ctx,tx,ty,10,1,999,17,corr,corr_counts)==N4M_OK);
        REQUIRE(relative_near(corr[0],0.4099457958749614) && corr_counts[0]==241);
        REQUIRE(n4m_stats_correlation_test(ctx,tx,ty,10,1,999,
                    UINT64_C(18446744073709551557),corr,corr_counts)==N4M_OK);
        REQUIRE(corr_counts[0]==225);
        // Monotone with an outlier: Spearman 0.979..., Pearson 0.757...;
        // no draw reaches either observed value (b=0, p=1/1000).
        double mx[12]={0.5,1.1,1.9,2.4,3.0,3.3,4.2,4.8,5.1,6.0,6.7,7.5};
        double my[12]={0.9,1.2,1.1,2.0,2.6,2.4,3.9,3.7,5.5,7.9,9.8,30.0};
        REQUIRE(n4m_stats_correlation_test(ctx,mx,my,12,0,999,17,corr,corr_counts)==N4M_OK);
        REQUIRE(relative_near(corr[0],0.7570003200136561) && corr_counts[0]==0);
        REQUIRE(near(corr[4],1.0/1000.0));
        REQUIRE(n4m_stats_correlation_test(ctx,mx,my,12,1,999,17,corr,corr_counts)==N4M_OK);
        REQUIRE(relative_near(corr[0],0.9790209790209792) && corr_counts[0]==0);
        // Scale invariance at large and small magnitudes inside D.
        double hx[5]={1e70,2e70,3e70,4e70,5e70}, hy[5]={-2e-70,-4e-70,-5e-70,-4e-70,-5e-70};
        REQUIRE(n4m_stats_correlation_test(ctx,hx,hy,5,0,0,1,corr,corr_counts)==N4M_OK);
        REQUIRE(relative_near(corr[0],-6.0/std::sqrt(60.0)));
        // Constant vectors: undefined coefficient, no permutation.
        double zx[4]={0.0,-0.0,0.0,-0.0}, sy4[4]={1,2,3,4};
        REQUIRE(n4m_stats_correlation_test(ctx,zx,sy4,4,1,99,1,corr,corr_counts)==N4M_OK);
        REQUIRE(std::isnan(corr[0]) && near(corr[1],1) && std::isnan(corr[4]) && corr_counts[1]==0);
        REQUIRE(near(corr[3],0.0));
        double nonfinite[4]={1,2,std::nan(""),4};
        REQUIRE(n4m_stats_correlation_test(ctx,nonfinite,sy4,4,0,0,1,corr,corr_counts)
                ==N4M_ERR_INVALID_ARGUMENT);
        REQUIRE(n4m_stats_correlation_test(ctx,sy4,sy4,4,2,0,1,corr,corr_counts)
                ==N4M_ERR_INVALID_ARGUMENT);
        REQUIRE(n4m_stats_correlation_test(ctx,sy4,sy4,1,0,0,1,corr,corr_counts)
                ==N4M_ERR_INVALID_ARGUMENT);
        // Translation invariance (review P0-1): two identical groups at 1e16
        // have eta^2 = F = 0; {0,2,4} vs {6,8,10} shifted by 1e16 keep
        // eta^2 = 54/70 and F = 54/(16/4) = 13.5 (hand values).
        int64_t halves[6]={0,0,0,1,1,1};
        double same16[6]={1e16,1e16+2,1e16+4,1e16,1e16+2,1e16+4};
        REQUIRE(n4m_preflight_factor_association(ctx,halves,same16,6,0,0,1,assoc,perm_counts)==N4M_OK);
        REQUIRE(std::abs(assoc[0])<1e-15 && std::abs(assoc[1])<1e-15);
        double apart16[6]={1e16,1e16+2,1e16+4,1e16+6,1e16+8,1e16+10};
        REQUIRE(n4m_preflight_factor_association(ctx,halves,apart16,6,0,0,1,assoc,perm_counts)==N4M_OK);
        REQUIRE(relative_near(assoc[0],54.0/70.0) && relative_near(assoc[1],13.5));
        double x16[5]={1e16,1e16+2,1e16+4,1e16+6,1e16+8};
        REQUIRE(n4m_stats_correlation_test(ctx,x16,cy,5,0,0,1,corr,corr_counts)==N4M_OK);
        REQUIRE(relative_near(corr[0],6.0/std::sqrt(60.0)));
        double cy16[5]={1e16+4,1e16+8,1e16+10,1e16+8,1e16+10};
        REQUIRE(n4m_stats_correlation_test(ctx,cx,cy16,5,0,0,1,corr,corr_counts)==N4M_OK);
        REQUIRE(relative_near(corr[0],6.0/std::sqrt(60.0)));
        // Exact enumeration (review P1-1). References: brute force over all
        // n! permutations with SciPy 1.15.3 statistics (chi2_contingency,
        // pearsonr, spearmanr) and NumPy 2.2.6, as exact fractions.
        double exact[4];
        int64_t f22[8]={0,0,0,0,1,1,1,1}; double y22[8]={0,0,0,0,1,1,1,1};
        REQUIRE(n4m_stats_factor_association_exact(ctx,f22,y22,8,1,1000,exact)==N4M_OK);
        REQUIRE(near(exact[0],70) && near(exact[1],1.0/35.0) && near(exact[2],1.0/35.0));
        REQUIRE(near(exact[3],1.0));
        REQUIRE(n4m_stats_factor_association_exact(ctx,fa,ya,6,0,1000,exact)==N4M_OK);
        REQUIRE(near(exact[0],20) && near(exact[1],0.1) && near(exact[2],0.1));
        REQUIRE(near(exact[3],27.0/29.0));
        REQUIRE(n4m_stats_factor_association_exact(ctx,fc,yc,7,1,1000,exact)==N4M_OK);
        REQUIRE(near(exact[0],35) && near(exact[1],17.0/35.0) && near(exact[2],1.0/35.0));
        REQUIRE(n4m_stats_correlation_exact(ctx,cx,cy,5,0,1000,exact)==N4M_OK);
        REQUIRE(near(exact[0],30) && near(exact[1],2.0/15.0) && near(exact[2],1.0/15.0));
        REQUIRE(near(exact[3],0.9036961141150641));
        REQUIRE(n4m_stats_correlation_exact(ctx,cx,cy,5,1,1000,exact)==N4M_OK);
        REQUIRE(near(exact[1],0.2) && near(exact[2],1.0/15.0) && near(exact[3],0.9486832980505138));
        REQUIRE(n4m_stats_correlation_exact(ctx,cx,cy,5,1,29,exact)==N4M_ERR_UNSUPPORTED);
        REQUIRE(n4m_stats_correlation_exact(ctx,cx,cy,5,1,0,exact)==N4M_ERR_INVALID_ARGUMENT);
        REQUIRE(n4m_stats_factor_association_exact(ctx,one_level,spread,3,0,10,exact)==N4M_OK);
        REQUIRE(near(exact[0],1) && std::isnan(exact[1]) && std::isnan(exact[3]));
        // Oversized categorical spaces (review P0): 65536 levels x 65536
        // classes = 2^32 cells would wrap to 0 in 32-bit size_t. Both APIs
        // refuse before any allocation or statistic.
        {
            const int64_t big = 65536;
            std::vector<int64_t> codes(static_cast<size_t>(big));
            std::vector<double> classes(static_cast<size_t>(big));
            for (int64_t i = 0; i < big; ++i) {
                codes[static_cast<size_t>(i)] = i;
                classes[static_cast<size_t>(i)] = static_cast<double>(i);
            }
            REQUIRE(n4m_stats_factor_association_exact(ctx,codes.data(),classes.data(),big,1,
                        100000,exact)==N4M_ERR_UNSUPPORTED);
            REQUIRE(n4m_preflight_factor_association(ctx,codes.data(),classes.data(),big,1,0,1,
                        assoc,perm_counts)==N4M_ERR_UNSUPPORTED);
            // Continuous: no table; 65536! arrangements exceed the cap.
            REQUIRE(n4m_stats_factor_association_exact(ctx,codes.data(),classes.data(),big,0,
                        100000,exact)==N4M_ERR_UNSUPPORTED);
            // Within the table budget, the Monte Carlo work budget still refuses.
            REQUIRE(n4m_preflight_factor_association(ctx,codes.data(),classes.data(),3000,1,999,1,
                        assoc,perm_counts)==N4M_ERR_UNSUPPORTED);
        }
        // Supported numerical domain D (rules 13 and 15): finite, nonzero |x|
        // in [2^-256, 2^256], max/min nonzero |x| <= 2^32 per input array,
        // exact zeros allowed; p-values only need the magnitude bounds. Inside D results are checked against exact
        // references; outside D every kernel returns N4M_ERR_UNSUPPORTED.
        {
            const auto rel = [](double a, double b) {
                return std::abs(a - b) <= 1e-12 * std::abs(b);
            };
            double xz4[4]={1,2,3,4};
            const double lo = std::ldexp(1.0,-256), hi = std::ldexp(1.0,256);
            const double below = std::nextafter(lo, 0.0), above = std::nextafter(hi, HUGE_VAL);
            const double sp = std::ldexp(1.0,N4M_DOMAIN_MAX_SPREAD_LOG2), sp_above = std::nextafter(sp, HUGE_VAL);
            const double sp_inv = std::ldexp(1.0,-64), sp_inv_below = std::nextafter(sp_inv, 0.0);
            // Boundaries on both sides, for every kernel that reads magnitudes.
            int64_t f4b[4]={0,0,1,1};
            const double inside[][4] = {{lo, 2*lo, 3*lo, 0.0}, {hi/4, hi/2, hi, 0.0},
                                        {1.0, 2.0, sp, 0.0}, {-lo, 0.0, lo*sp, -lo}};
            REQUIRE(sp == 4294967296.0);  // rule 15: spread 2^32
            const double outside[][4] = {{below, 2*lo, 3*lo, 0.0}, {hi/4, hi/2, above, 0.0},
                                         {1.0, 2.0, sp_above, 0.0}, {-lo, 0.0, 2*lo*sp, -lo},
                                         {1.0, 0.0, std::ldexp(1.0,33), 0.0}};
            for (const auto& v : inside) {
                double y4[4] = {v[0], v[1], v[2], v[3]};
                REQUIRE(n4m_preflight_factor_association(ctx,f4b,y4,4,0,0,1,assoc,perm_counts)==N4M_OK);
                REQUIRE(n4m_stats_factor_association_exact(ctx,f4b,y4,4,0,100,exact)==N4M_OK);
                REQUIRE(n4m_stats_correlation_test(ctx,y4,xz4,4,0,0,1,corr,corr_counts)==N4M_OK);
                REQUIRE(n4m_stats_correlation_test(ctx,xz4,y4,4,1,0,1,corr,corr_counts)==N4M_OK);
                REQUIRE(n4m_stats_correlation_exact(ctx,y4,xz4,4,0,100,exact)==N4M_OK);
            }
            for (const auto& v : outside) {
                double y4[4] = {v[0], v[1], v[2], v[3]};
                REQUIRE(n4m_preflight_factor_association(ctx,f4b,y4,4,0,0,1,assoc,perm_counts)==N4M_ERR_UNSUPPORTED);
                REQUIRE(n4m_stats_factor_association_exact(ctx,f4b,y4,4,0,100,exact)==N4M_ERR_UNSUPPORTED);
                REQUIRE(n4m_stats_correlation_test(ctx,y4,xz4,4,0,0,1,corr,corr_counts)==N4M_ERR_UNSUPPORTED);
                REQUIRE(n4m_stats_correlation_test(ctx,xz4,y4,4,1,0,1,corr,corr_counts)==N4M_ERR_UNSUPPORTED);
                REQUIRE(n4m_stats_correlation_exact(ctx,y4,xz4,4,0,100,exact)==N4M_ERR_UNSUPPORTED);
            }
            // Holm/BH have no spread limit (no subtraction): p = 1 next to
            // 1e-25 or 2^-1022 is accepted; only subnormal p are refused.
            double pin[2]={sp_inv_below,1.0}, padj[2];
            REQUIRE(n4m_preflight_holm_adjust(ctx,pin,2,padj)==N4M_OK && padj[0]==2*sp_inv_below);
            REQUIRE(n4m_stats_bh_adjust(ctx,pin,2,padj)==N4M_OK && padj[0]==2*sp_inv_below);
            double wide_p[2]={1e-25,1.0};
            REQUIRE(n4m_preflight_holm_adjust(ctx,wide_p,2,padj)==N4M_OK && padj[0]==2e-25 && padj[1]==1.0);
            REQUIRE(n4m_stats_bh_adjust(ctx,wide_p,2,padj)==N4M_OK && padj[0]==2e-25 && padj[1]==1.0);
            double pmin[2]={DBL_MIN,1.0}, psub[2]={std::nextafter(DBL_MIN,0.0),1.0};
            REQUIRE(n4m_preflight_holm_adjust(ctx,pmin,2,padj)==N4M_OK && padj[0]==2*DBL_MIN);
            REQUIRE(n4m_stats_bh_adjust(ctx,pmin,2,padj)==N4M_OK && padj[0]==2*DBL_MIN);
            REQUIRE(n4m_preflight_holm_adjust(ctx,psub,2,padj)==N4M_ERR_UNSUPPORTED);
            REQUIRE(n4m_stats_bh_adjust(ctx,psub,2,padj)==N4M_ERR_UNSUPPORTED);
            double pzero[2]={0.0,1.0};
            REQUIRE(n4m_preflight_holm_adjust(ctx,pzero,2,padj)==N4M_OK && padj[0]==0.0);
            // Categorical targets are labels: codes are not domain-checked.
            int64_t fcat[4]={0,0,1,1}; double cbig[4]={0.0,9007199254740991.0,0.0,1.0};
            REQUIRE(n4m_preflight_factor_association(ctx,fcat,cbig,4,1,0,1,assoc,perm_counts)==N4M_OK);

            // Scale sweep inside D: 2^k data keep eta^2 = 27/29, F = 54,
            // r = 6/sqrt(60), rho = 7/sqrt(90), the NumPy counts and the exact
            // distributions; the same data outside D are refused.
            for (int k : {-240, -100, 0, 100, 240}) {
                double ys[6], xs5[5], ys5[5];
                for (int i = 0; i < 6; ++i) ys[i] = std::ldexp(ya[i], k);
                for (int i = 0; i < 5; ++i) {
                    xs5[i] = std::ldexp(cx[i], k);
                    ys5[i] = std::ldexp(cy[i], -k / 2);
                }
                REQUIRE(n4m_preflight_factor_association(ctx,fa,ys,6,0,999,17,assoc,perm_counts)==N4M_OK);
                REQUIRE(rel(assoc[0],27.0/29.0) && rel(assoc[1],54.0) && perm_counts[0]==93);
                REQUIRE(n4m_stats_factor_association_exact(ctx,fa,ys,6,0,1000,exact)==N4M_OK);
                REQUIRE(near(exact[0],20) && near(exact[1],0.1) && near(exact[2],0.1));
                REQUIRE(n4m_stats_correlation_test(ctx,xs5,ys5,5,0,999,17,corr,corr_counts)==N4M_OK);
                REQUIRE(rel(corr[0],6.0/std::sqrt(60.0)) && corr_counts[0]==133);
                REQUIRE(n4m_stats_correlation_test(ctx,xs5,ys5,5,1,999,17,corr,corr_counts)==N4M_OK);
                REQUIRE(rel(corr[0],7.0/std::sqrt(90.0)) && corr_counts[0]==201);
                REQUIRE(n4m_stats_correlation_exact(ctx,xs5,ys5,5,0,1000,exact)==N4M_OK);
                REQUIRE(near(exact[1],2.0/15.0) && near(exact[2],1.0/15.0));
            }
            for (int k : {-1060, -1022, -300, 300, 1000}) {
                double ys[6];
                for (int i = 0; i < 6; ++i) ys[i] = std::ldexp(ya[i], k);
                REQUIRE(n4m_preflight_factor_association(ctx,fa,ys,6,0,999,17,assoc,perm_counts)==N4M_ERR_UNSUPPORTED);
                REQUIRE(n4m_stats_factor_association_exact(ctx,fa,ys,6,0,1000,exact)==N4M_ERR_UNSUPPORTED);
                REQUIRE(n4m_stats_correlation_test(ctx,ys,ya,6,1,999,17,corr,corr_counts)==N4M_ERR_UNSUPPORTED);
            }
            // Translations inside D: 1e16 + 2y (ulp 2) and 2^200 + y 2^150
            // (ulp 2^148); 1e300 is outside D.
            const double offsets[2][2] = {{1e16, 1.0}, {std::ldexp(1.0, 200), std::ldexp(1.0, 149)}};
            for (const auto& o : offsets) {
                double ys[6], xs5[5];
                for (int i = 0; i < 6; ++i) ys[i] = o[0] + 2.0 * o[1] * ya[i];
                for (int i = 0; i < 5; ++i) xs5[i] = o[0] + 2.0 * o[1] * cx[i];
                REQUIRE(n4m_preflight_factor_association(ctx,fa,ys,6,0,999,17,assoc,perm_counts)==N4M_OK);
                REQUIRE(rel(assoc[0],27.0/29.0) && rel(assoc[1],54.0) && perm_counts[0]==93);
                REQUIRE(n4m_stats_correlation_test(ctx,xs5,cy,5,0,999,17,corr,corr_counts)==N4M_OK);
                REQUIRE(rel(corr[0],6.0/std::sqrt(60.0)) && corr_counts[0]==133);
            }
            double far[6];
            for (int i = 0; i < 6; ++i) far[i] = 1e300 + std::ldexp(ya[i], 950);
            REQUIRE(n4m_preflight_factor_association(ctx,fa,far,6,0,0,1,assoc,perm_counts)==N4M_ERR_UNSUPPORTED);

            // Mixed exponents inside D, exact references. Levels {0, 2^-30}
            // | {1}: SSW = 2^-61 from per-level residuals; F = SSB/SSW =
            // 1.5372286713774735e18 (exact rational, Python fractions).
            int64_t f3[3]={0,0,1};
            double gap[3]={0.0,std::ldexp(1.0,-30),1.0};
            REQUIRE(n4m_preflight_factor_association(ctx,f3,gap,3,0,0,1,assoc,perm_counts)==N4M_OK);
            REQUIRE(rel(assoc[1],1.5372286713774735e18) && rel(assoc[0],1.0));
            // Signed mixture {-2^-15, 2^-15} | {2^15}: SSW = 2^-29,
            // SSB = (2/3) 2^30, F = (2/3) 2^59.
            double mixed[3]={-std::ldexp(1.0,-15),std::ldexp(1.0,-15),std::ldexp(1.0,15)};
            REQUIRE(n4m_preflight_factor_association(ctx,f3,mixed,3,0,0,1,assoc,perm_counts)==N4M_OK);
            REQUIRE(rel(assoc[1],std::ldexp(2.0/3.0,59)) && rel(assoc[0],1.0));
            // The review's extreme mixtures lie outside D: explicit refusal.
            double huge[3]={0.0,1e-100,1e300}, denorm[3]={0.0,std::ldexp(1.0,-1074),1.0};
            double wide[3]={0.0,std::ldexp(1.0,-60),1.0};
            double wide_mixed[3]={-std::ldexp(1.0,-30),std::ldexp(1.0,-30),std::ldexp(1.0,30)};
            REQUIRE(n4m_preflight_factor_association(ctx,f3,wide_mixed,3,0,0,1,assoc,perm_counts)==N4M_ERR_UNSUPPORTED);
            REQUIRE(n4m_preflight_factor_association(ctx,f3,huge,3,0,0,1,assoc,perm_counts)==N4M_ERR_UNSUPPORTED);
            REQUIRE(n4m_preflight_factor_association(ctx,f3,denorm,3,0,0,1,assoc,perm_counts)==N4M_ERR_UNSUPPORTED);
            REQUIRE(n4m_preflight_factor_association(ctx,f3,wide,3,0,0,1,assoc,perm_counts)==N4M_ERR_UNSUPPORTED);
            // Truly zero within-level variance: F undefined (NaN), status OK.
            double flat[3]={5.0,5.0,7.0};
            REQUIRE(n4m_preflight_factor_association(ctx,f3,flat,3,0,0,1,assoc,perm_counts)==N4M_OK);
            REQUIRE(rel(assoc[0],1.0) && std::isnan(assoc[1]));
            // Near-zero effect: {0, 2} | {d, 2+d}, d = 2^-20: eta^2 =
            // d^2/(4+d^2), F = d^2/2 (rational references).
            const double dl = std::ldexp(1.0,-20);
            double near0[4]={0.0,2.0,dl,2.0+dl};
            REQUIRE(n4m_preflight_factor_association(ctx,f4b,near0,4,0,0,1,assoc,perm_counts)==N4M_OK);
            REQUIRE(rel(assoc[0],dl*dl/(4.0+dl*dl)) && rel(assoc[1],dl*dl/2.0));
            // Exactly zero effect: equal level means give eta^2 = F = 0.
            double zero_effect[4]={0.0,2.0,2.0,0.0};
            REQUIRE(n4m_preflight_factor_association(ctx,f4b,zero_effect,4,0,0,1,assoc,perm_counts)==N4M_OK);
            REQUIRE(assoc[0]==0.0 && assoc[1]==0.0);
            // Cramer's V at independence: a 2x9 table with one unit per cell
            // has chi-square = V = 0 exactly.
            int64_t rows18[18]; double cols18[18];
            for (int i = 0; i < 18; ++i) { rows18[i] = i / 9; cols18[i] = i % 9; }
            REQUIRE(n4m_preflight_factor_association(ctx,rows18,cols18,18,1,0,1,assoc,perm_counts)==N4M_OK);
            REQUIRE(assoc[0]==0.0 && assoc[1]==0.0);
            // Near independence: [[m, m+1], [m+1, m]] has chi-square =
            // 4/(4m+2) and V = 1/(2m+1) exactly (rational reference).
            for (int64_t m : {50, 1000}) {
                std::vector<int64_t> rr; std::vector<double> cc;
                const int64_t cells[4] = {m, m + 1, m + 1, m};
                for (int k = 0; k < 4; ++k)
                    for (int64_t j = 0; j < cells[k]; ++j) { rr.push_back(k / 2); cc.push_back(k % 2); }
                const auto nn = static_cast<int64_t>(rr.size());
                REQUIRE(n4m_preflight_factor_association(ctx,rr.data(),cc.data(),nn,1,0,1,assoc,perm_counts)==N4M_OK);
                const double md = static_cast<double>(m);
                REQUIRE(rel(assoc[0],1.0/(2.0*md+1.0)) && rel(assoc[1],4.0/(4.0*md+2.0)));
            }
            // Correlation with mixed exponents inside D: x = {1, 2, 2^20},
            // y = {1, 2, 3}: r = 0.8660258167378264859... (exact rational
            // r^2 evaluated with Python fractions); ranks give rho = 1.
            double xm[3]={1.0,2.0,std::ldexp(1.0,20)}, ym[3]={1,2,3};
            REQUIRE(n4m_stats_correlation_test(ctx,xm,ym,3,0,0,1,corr,corr_counts)==N4M_OK);
            REQUIRE(rel(corr[0],0.8660258167378265));
            REQUIRE(n4m_stats_correlation_test(ctx,xm,ym,3,1,0,1,corr,corr_counts)==N4M_OK);
            REQUIRE(rel(corr[0],1.0));
            // Exactly uncorrelated (dyadic): r = rho = 0.
            double yz[4]={1,-1,-1,1};
            REQUIRE(n4m_stats_correlation_test(ctx,xz4,yz,4,0,0,1,corr,corr_counts)==N4M_OK);
            REQUIRE(corr[0]==0.0);
            REQUIRE(n4m_stats_correlation_test(ctx,xz4,yz,4,1,0,1,corr,corr_counts)==N4M_OK);
            REQUIRE(corr[0]==0.0);
            // Large offset next to a small step inside D: y = 2^200 +
            // {0, 0, 2^150, 2^150} against x = {1, 2, 3, 4}: r = 2/sqrt(5).
            const double o200 = std::ldexp(1.0,200), s150 = std::ldexp(1.0,150);
            double yo[4]={o200,o200,o200+s150,o200+s150};
            REQUIRE(n4m_stats_correlation_test(ctx,xz4,yo,4,0,0,1,corr,corr_counts)==N4M_OK);
            REQUIRE(rel(corr[0],2.0/std::sqrt(5.0)));
            // Holm and BH: small normal p-values scale exactly; subnormal
            // ones are refused.
            for (int k : {-100, -200, -260, -500, -1000}) {
                double tiny[4], h[4], b[4];
                for (int i = 0; i < 4; ++i) tiny[i] = std::ldexp(pv[i], k);
                REQUIRE(n4m_preflight_holm_adjust(ctx,tiny,4,h)==N4M_OK);
                REQUIRE(n4m_stats_bh_adjust(ctx,tiny,4,b)==N4M_OK);
                const double holm_ref[4]={0.03,0.06,0.06,0.02}, bh_ref[4]={0.02,0.04,0.04,0.02};
                for (int i = 0; i < 4; ++i) {
                    REQUIRE(rel(std::ldexp(h[i],-k),holm_ref[i]) && rel(std::ldexp(b[i],-k),bh_ref[i]));
                    REQUIRE(b[i] <= h[i]);
                }
            }
            for (int k : {-1060}) {
                double tiny[4], h[4];
                for (int i = 0; i < 4; ++i) tiny[i] = std::ldexp(pv[i], k);
                REQUIRE(n4m_preflight_holm_adjust(ctx,tiny,4,h)==N4M_ERR_UNSUPPORTED);
                REQUIRE(n4m_stats_bh_adjust(ctx,tiny,4,h)==N4M_ERR_UNSUPPORTED);
            }
        }
        // Benjamini-Hochberg: statsmodels 0.15.0 multipletests(fdr_bh) and
        // SciPy 1.15.3 false_discovery_control(method="bh") reference values.
        double bh[6];
        REQUIRE(n4m_stats_bh_adjust(ctx,pv,4,bh)==N4M_OK);
        REQUIRE(near(bh[0],0.02) && near(bh[1],0.04) && near(bh[2],0.04) && near(bh[3],0.02));
        REQUIRE(n4m_stats_bh_adjust(ctx,tied,3,bh)==N4M_OK);
        REQUIRE(near(bh[0],0.03) && near(bh[1],0.03) && near(bh[2],0.9));
        REQUIRE(n4m_stats_bh_adjust(ctx,capped,2,bh)==N4M_OK);
        REQUIRE(near(bh[0],0.8) && near(bh[1],0.8));
        double six[6]={0.001,0.2,0.5,0.03,0.04,0.049};
        REQUIRE(n4m_stats_bh_adjust(ctx,six,6,bh)==N4M_OK);
        REQUIRE(near(bh[0],0.006) && near(bh[1],0.24) && near(bh[2],0.5));
        REQUIRE(near(bh[3],0.0735) && near(bh[4],0.0735) && near(bh[5],0.0735));
        REQUIRE(n4m_stats_bh_adjust(ctx,invalid_p,1,bh)==N4M_ERR_INVALID_ARGUMENT);
        REQUIRE(n4m_stats_bh_adjust(ctx,pv,0,bh)==N4M_ERR_INVALID_ARGUMENT);
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
        // Values outside the supported domain are refused, never misread.
        Q=view(max_query,1,1); F=view(min_ref,1,1);
        REQUIRE(n4m_preflight_pairs_within_tolerance(ctx,&Q,&F,nullptr,nullptr,1.0,0,
                                                     nullptr,nullptr,qflag,rflag,pcounts)
                ==N4M_ERR_UNSUPPORTED);
        // Regression (review b1 #3, b1fix #2, b14fix #2/#8) inside the domain
        // (nonzero |x| in [2^-256, 2^256], per-row spread <= 2^32). Distances use
        // a relative tolerance on the expected value (so 0 never passes) and
        // decisions are checked on both sides of each threshold.
        {
            const double A=std::ldexp(1.0,200), a=std::ldexp(1.0,-200);
            const double lo=std::ldexp(1.0,-256), hi=std::ldexp(1.0,256);
            double hd[1]={}; int64_t hi_idx[2]={};
            const auto status = [&](double q0, double q1, double r0, double r1, double tol) {
                double q[2]={q0,q1}, r[2]={r0,r1};
                auto Q2=view(q,1,2), R2=view(r,1,2);
                hd[0]=-1.0;
                return n4m_preflight_pairs_within_tolerance(ctx,&Q2,&R2,nullptr,nullptr,tol,1,
                                                            hi_idx,hd,qflag,rflag,pcounts);
            };
            const auto pair = [&](double q0, double r0, double tol) {
                REQUIRE(status(q0,0,r0,0,tol)==N4M_OK);
                return pcounts[0]==1;
            };
            // q - r = 2A on one of two features: RMS sqrt(2) A.
            REQUIRE(pair(A,-A,1.5*A) && rel(hd[0],std::sqrt(2.0)*A,4e-16));
            REQUIRE(!pair(A,-A,1.414*A));
            // RMS a/sqrt(2) at every tolerance (tolerance-free distance).
            for (double tol : {0.70711*a, 1e-30, 1.0, A, hi}) {
                REQUIRE(pair(a,0,tol));
                REQUIRE(rel(hd[0],a/std::sqrt(2.0),4e-16));
            }
            REQUIRE(!pair(a,0,0.70710*a));
            // At the lower edge: [2 lo, 0] vs 0 has RMS sqrt(2) lo > lo.
            REQUIRE(!pair(2*lo,0,lo));
            REQUIRE(pair(2*lo,0,2*lo) && rel(hd[0],std::sqrt(2.0)*lo,4e-16));
            // Domain boundaries on both sides: magnitudes and per-row spread.
            REQUIRE(status(lo,0,hi,0,1.0)==N4M_OK);
            REQUIRE(status(std::nextafter(lo,0.0),0,0,0,1.0)==N4M_ERR_UNSUPPORTED);
            REQUIRE(status(std::nextafter(hi,INFINITY),0,0,0,1.0)==N4M_ERR_UNSUPPORTED);
            REQUIRE(status(1.0,std::ldexp(1.0,32),0,0,1.0)==N4M_OK);
            REQUIRE(status(1.0,std::nextafter(std::ldexp(1.0,32),INFINITY),0,0,1.0)
                    ==N4M_ERR_UNSUPPORTED);
            REQUIRE(status(0,0,std::ldexp(1.0,-100),std::ldexp(1.0,-30),1.0)
                    ==N4M_ERR_UNSUPPORTED);
        }
        // SNV with per-row rounding bounds (ABI 2.20). Rows: constant (exact
        // zeros, bound 0), [a,0,-a,0] at the domain edges 2^-256 and 2^256
        // (mean 0, SD a/sqrt(2): [sqrt(2),0,-sqrt(2),0]), [1,2,3,4]
        // (z = (j-2.5)/sqrt(1.25)), and an offset row at 2^60.
        {
            const double lo=std::ldexp(1.0,-256), hi=std::ldexp(1.0,256), off=std::ldexp(1.0,60);
            // Row 4 is [c, c, c, c + 256] with c = 2^60 (ulp 256).
            double sx[20]={0.1,0.1,0.1,0.1, lo,0,-lo,0, hi,0,-hi,0, 1,2,3,4,
                           off,off,off,off+256};
            double so[20]={}, sb[5]={};
            auto SX=view(sx,5,4), SO=view(so,5,4);
            REQUIRE(n4m_preflight_snv_compute(ctx,&SX,nullptr,5,&SO,sb)==N4M_OK);
            for (int j=0;j<4;++j) REQUIRE(so[j]==0.0);
            REQUIRE(sb[0]==0.0);
            for (int i : {1,2}) {
                REQUIRE(rel(so[4*i],std::sqrt(2.0),1e-15) && so[4*i+1]==0.0 &&
                        rel(so[4*i+2],-std::sqrt(2.0),1e-15) && so[4*i+3]==0.0);
                REQUIRE(sb[i]>0.0 && sb[i]<1e-14);
            }
            for (int j=0;j<4;++j) REQUIRE(rel(so[12+j],(j+1-2.5)/std::sqrt(1.25),1e-15));
            // [c,c,c,c+256]: z = [-1,-1,-1,3]/sqrt(3) whatever c (2^60 offset).
            for (int j=0;j<3;++j) REQUIRE(rel(so[16+j],-1/std::sqrt(3.0),1e-15));
            REQUIRE(rel(so[19],3/std::sqrt(3.0),1e-15) && sb[4]<1e-14);
            // The bound covers the actual error of an offset row of 100,000
            // bands (99,999 values then one larger), whose exact SNV is
            // -1/sqrt(99999) and sqrt(99999) (review b1fix #3).
            const int64_t wide=100000;
            std::vector<double> wx(static_cast<size_t>(2*wide)), wo(wx.size()); double wb[2];
            for (int64_t j=0;j<wide;++j) {
                const double v = j+1<wide ? 1.0 : 2.0;
                wx[static_cast<size_t>(j)] = 1.0*v + 0.0;
                wx[static_cast<size_t>(wide+j)] = 1.1*v + 100.0;
            }
            auto WX=view(wx.data(),2,wide), WO=view(wo.data(),2,wide);
            REQUIRE(n4m_preflight_snv_compute(ctx,&WX,nullptr,2,&WO,wb)==N4M_OK);
            const double z_lo=-1.0/std::sqrt(99999.0), hi_z=std::sqrt(99999.0);
            for (int i=0;i<2;++i) {
                double err=0;
                for (int64_t j=0;j<wide;++j)
                    err=std::max(err,std::abs(wo[static_cast<size_t>(i*wide+j)]-(j+1<wide?z_lo:hi_z)));
                REQUIRE(err<=wb[i] && wb[i]<1e-12);
            }
            // Aggregation (b14fix #3): a [1,2,4] with a = 1, 2, 4, 8 and
            // 1, 10000, 1, 1 identical acquisitions. Every analytic SNV row is
            // [1,2,4] standardized; group means must stay within their bounds.
            const int64_t reps[4]={1,10000,1,1};
            std::vector<double> gx; std::vector<int64_t> gid;
            for (int64_t g=0; g<4; ++g)
                for (int64_t r=0; r<reps[g]; ++r) {
                    for (double v : {1.0,2.0,4.0}) gx.push_back(std::ldexp(v,static_cast<int>(g)));
                    gid.push_back(g);
                }
            const int64_t gn=static_cast<int64_t>(gid.size());
            double go[12], gb[4];
            auto GX=view(gx.data(),gn,3), GO=view(go,4,3);
            REQUIRE(n4m_preflight_snv_compute(ctx,&GX,gid.data(),4,&GO,gb)==N4M_OK);
            const double mu=7.0/3.0, sd=std::sqrt(((1-mu)*(1-mu)+(2-mu)*(2-mu)+(4-mu)*(4-mu))/3.0);
            for (int g=0;g<4;++g) {
                REQUIRE(gb[g]>0.0 && gb[g]<1e-14);
                int j=0;
                for (double v : {1.0,2.0,4.0}) {
                    REQUIRE(std::abs(go[g*3+j]-(v-mu)/sd)<=gb[g]+4e-16*std::abs((v-mu)/sd));
                    ++j;
                }
            }
            int64_t badgid[2]={0,2};
            double two[4]={1,2,3,5}, tout[4]={};
            auto TX=view(two,2,2), TO=view(tout,2,2);
            REQUIRE(n4m_preflight_snv_compute(ctx,&TX,badgid,2,&TO,gb)==N4M_ERR_INVALID_ARGUMENT);
            int64_t onegid[2]={0,0};
            REQUIRE(n4m_preflight_snv_compute(ctx,&TX,onegid,2,&TO,gb)==N4M_ERR_INVALID_ARGUMENT);
            // Domain boundaries: 2^256 next to 1 (spread 2^256), 2^-257, and a
            // spread just above 2^32 are refused; a spread of exactly 2^32 is not.
            for (const auto& row : {std::vector<double>{1.0, std::ldexp(1.0,256)},
                                    std::vector<double>{std::ldexp(1.0,-257), 0.0},
                                    std::vector<double>{1.0, std::nextafter(std::ldexp(1.0,32),INFINITY)}}) {
                std::vector<double> rx(row), ro(2); double rb[1];
                auto RX=view(rx.data(),1,2), RO=view(ro.data(),1,2);
                REQUIRE(n4m_preflight_snv_compute(ctx,&RX,nullptr,1,&RO,rb)==N4M_ERR_UNSUPPORTED);
            }
            {
                double rx[2]={1.0,std::ldexp(1.0,32)}, ro[2]; double rb[1];
                auto RX=view(rx,1,2), RO=view(ro,1,2);
                REQUIRE(n4m_preflight_snv_compute(ctx,&RX,nullptr,1,&RO,rb)==N4M_OK);
                REQUIRE(rel(ro[0],-1.0,1e-15) && rel(ro[1],1.0,1e-15));
            }
            double bad[2]={1,std::nan("")};
            auto BX=view(bad,1,2), BO=view(so,1,2);
            REQUIRE(n4m_preflight_snv_compute(ctx,&BX,nullptr,1,&BO,sb)==N4M_ERR_INVALID_ARGUMENT);
            auto narrow1=view(sx,4,1), nout=view(so,4,1);
            REQUIRE(n4m_preflight_snv_compute(ctx,&narrow1,nullptr,4,&nout,sb)==N4M_ERR_INVALID_ARGUMENT);
        }
        // Regression (review b1 #2): orders outside the validated domain are
        // refused (window 41 / order 38 gave a constant-row residual of 0.83),
        // and a constant row has zero residual at the domain edge (order 10).
        {
            std::vector<double> ones(41, 1.0); double cb[41], cp[2]; int64_t cc[41];
            auto Cst=view(ones.data(),1,41);
            REQUIRE(n4m_preflight_noise_proxy_compute(ctx,&Cst,nullptr,41,38,cb,cc,cp)
                    ==N4M_ERR_INVALID_ARGUMENT);
            REQUIRE(n4m_preflight_noise_proxy_compute(ctx,&Cst,nullptr,41,11,cb,cc,cp)
                    ==N4M_ERR_INVALID_ARGUMENT);
            REQUIRE(n4m_preflight_noise_proxy_compute(ctx,&Cst,nullptr,41,10,cb,cc,cp)==N4M_OK);
            REQUIRE(std::abs(cp[0])<1e-12 && std::abs(cb[20])<1e-12 && cc[20]==1);
            std::vector<double> wide(1003, 1.0); double wb[1003]; int64_t wc[1003];
            auto Wd=view(wide.data(),1,1003);
            REQUIRE(n4m_preflight_noise_proxy_compute(ctx,&Wd,nullptr,1003,2,wb,wc,cp)
                    ==N4M_ERR_INVALID_ARGUMENT);
            REQUIRE(n4m_preflight_noise_proxy_compute(ctx,&Wd,nullptr,1001,2,wb,wc,cp)==N4M_OK);
        }
        n4m_context_destroy(ctx);
        std::puts("preflight diagnostics: ok");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }
}
