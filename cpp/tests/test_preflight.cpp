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
        // Scale invariance at extreme magnitudes and signed zeros.
        double hx[5]={1e307,2e307,3e307,4e307,5e307}, hy[5]={-2e-300,-4e-300,-5e-300,-4e-300,-5e-300};
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
        // Magnitude sweep (robustness rule): scaling by 2^k, k from the
        // subnormal range to 2^1000, and large translations leave eta^2, F,
        // r, rho, the permutation counts and the exact distributions
        // unchanged (relative 1e-12, counts exact). Data are chosen so every
        // scaled or shifted input is exactly representable.
        {
            const int ks[] = {-1060, -1022, -500, -100, 0, 100, 500, 1000};
            const auto rel = [](double a, double b) {
                return std::abs(a - b) <= 1e-12 * std::abs(b);
            };
            for (int k : ks) {
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
            // Translations: 1e16 + 2y (spacing 2 = ulp) and 1e300 + y * 2^950
            // (1e300 is a multiple of its ulp 2^944).
            const double offsets[2][2] = {{1e16, 1.0}, {1e300, std::ldexp(1.0, 950)}};
            for (const auto& o : offsets) {
                double ys[6], xs5[5];
                for (int i = 0; i < 6; ++i) ys[i] = o[0] + 2.0 * o[1] * ya[i];
                for (int i = 0; i < 5; ++i) xs5[i] = o[0] + 2.0 * o[1] * cx[i];
                REQUIRE(n4m_preflight_factor_association(ctx,fa,ys,6,0,999,17,assoc,perm_counts)==N4M_OK);
                REQUIRE(rel(assoc[0],27.0/29.0) && rel(assoc[1],54.0) && perm_counts[0]==93);
                REQUIRE(n4m_stats_correlation_test(ctx,xs5,cy,5,0,999,17,corr,corr_counts)==N4M_OK);
                REQUIRE(rel(corr[0],6.0/std::sqrt(60.0)) && corr_counts[0]==133);
            }
            // A small within-level spread next to a large between-level gap:
            // levels {0, d} and {1}. SSW = d^2/2 is accumulated at its own
            // scale; with d = 2^-510, F = (2/3) 2^1021 (hand value, the d terms
            // of SSB are below rounding). With d = 2^-520, F = (2/3) 2^1041 is
            // not representable: explicit error, never Inf or NaN with OK.
            int64_t fl[3]={0,0,1};
            double gap[3]={0.0,std::ldexp(1.0,-510),1.0};
            REQUIRE(n4m_preflight_factor_association(ctx,fl,gap,3,0,0,1,assoc,perm_counts)==N4M_OK);
            REQUIRE(assoc[1] > 0.0 && rel(assoc[1],std::ldexp(2.0/3.0,1021)));
            gap[1]=std::ldexp(1.0,-520);
            REQUIRE(n4m_preflight_factor_association(ctx,fl,gap,3,0,0,1,assoc,perm_counts)
                    ==N4M_ERR_NUMERICAL_FAILURE);
            // Holm and BH: tiny p-values (subnormal included) stay positive,
            // keep their order and scale exactly when no cap applies.
            for (int k : {-1060, -1022, -500, -100}) {
                double tiny[4], h[4], b[4];
                for (int i = 0; i < 4; ++i) tiny[i] = std::ldexp(pv[i], k);
                REQUIRE(n4m_preflight_holm_adjust(ctx,tiny,4,h)==N4M_OK);
                REQUIRE(n4m_stats_bh_adjust(ctx,tiny,4,b)==N4M_OK);
                const double holm_ref[4]={0.03,0.06,0.06,0.02}, bh_ref[4]={0.02,0.04,0.04,0.02};
                for (int i = 0; i < 4; ++i) {
                    REQUIRE(h[i] > 0.0 && b[i] > 0.0 && h[i] >= tiny[i] && b[i] >= tiny[i]);
                    REQUIRE(b[i] <= h[i]);
                    if (k >= -1022) {
                        REQUIRE(std::abs(std::ldexp(h[i],-k) - holm_ref[i]) <= 1e-12);
                        REQUIRE(std::abs(std::ldexp(b[i],-k) - bh_ref[i]) <= 1e-12);
                    }
                }
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
