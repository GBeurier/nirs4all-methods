// SPDX-License-Identifier: CECILL-2.1
#include "n4m/metrics.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

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
        // Values outside the supported domain are refused, never misread.
        Q=view(max_query,1,1); F=view(min_ref,1,1);
        REQUIRE(n4m_preflight_pairs_within_tolerance(ctx,&Q,&F,nullptr,nullptr,1.0,0,
                                                     nullptr,nullptr,qflag,rflag,pcounts)
                ==N4M_ERR_UNSUPPORTED);
        // Regression (review b1 #3, b1fix #2, b14fix #2/#8) inside the domain
        // (nonzero |x| in [2^-256, 2^256], per-row spread <= 2^64). Distances use
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
            REQUIRE(status(1.0,std::ldexp(1.0,64),0,0,1.0)==N4M_OK);
            REQUIRE(status(1.0,std::nextafter(std::ldexp(1.0,64),INFINITY),0,0,1.0)
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
            // spread just above 2^64 are refused; a spread of exactly 2^64 is not.
            for (const auto& row : {std::vector<double>{1.0, std::ldexp(1.0,256)},
                                    std::vector<double>{std::ldexp(1.0,-257), 0.0},
                                    std::vector<double>{1.0, std::nextafter(std::ldexp(1.0,64),INFINITY)}}) {
                std::vector<double> rx(row), ro(2); double rb[1];
                auto RX=view(rx.data(),1,2), RO=view(ro.data(),1,2);
                REQUIRE(n4m_preflight_snv_compute(ctx,&RX,nullptr,1,&RO,rb)==N4M_ERR_UNSUPPORTED);
            }
            {
                double rx[2]={1.0,std::ldexp(1.0,64)}, ro[2]; double rb[1];
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
