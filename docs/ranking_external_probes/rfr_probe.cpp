#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
#include "rfr/data_containers/default_data_container.hpp"
#include "rfr/splits/binary_split_one_feature_rss_loss.hpp"
#include "rfr/forests/fanova_forest.hpp"
using Rng = std::mt19937;
using Split = rfr::splits::binary_split_one_feature_rss_loss<double,double,unsigned,Rng>;
using Forest = rfr::forests::fANOVA_forest<Split,double,double,unsigned,Rng>;
extern "C" double ranking_probe_marginal() {
    rfr::data_containers::default_container<double,double,unsigned> data(2);
    data.set_type_of_feature(0,2); data.set_type_of_feature(1,2);
    for(unsigned i=0;i<4;++i) data.add_data_point({double(i&1),double(i>>1)},double(i&1));
    Forest f; f.options.num_trees=1; f.options.do_bootstrapping=false;
    f.options.num_data_points_per_tree=4; f.options.tree_opts.max_features=2;
    f.options.tree_opts.max_depth=4; Rng rng(17); f.fit(data,rng); f.precompute_marginals();
    auto v=f.get_trees_total_variances();
    double marginal=f.marginal_mean_prediction({1.0,NAN});
    std::cout << "prediction=" << f.predict({1,0}) << " marginal=" << marginal << " total_variance=" << v.at(0) << '\n';
    if(std::abs(marginal-1)>1e-12 || std::abs(v.at(0)-0.25)>1e-12) return -1;
    return marginal;
}
#ifndef PROBE_LIBRARY
int main(){return ranking_probe_marginal()==1.0 ? 0:1;}
#endif
