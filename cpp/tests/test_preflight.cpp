// SPDX-License-Identifier: CECILL-2.1
#include "n4m/metrics.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>

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
        REQUIRE(std::isnan(contrast[0]) && near(contrast[2],1) && near(contrast[3],1));
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
        n4m_context_destroy(ctx);
        std::puts("preflight diagnostics: ok");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }
}
