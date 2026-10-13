// SPDX-License-Identifier: CECILL-2.1
//
// HDBSCAN* (Euclidean) — C++ port of scikit-learn 1.9.1
// sklearn/cluster/_hdbscan/{hdbscan.py,_linkage.pyx,_tree.pyx} (BSD-3-Clause),
// plus the GLOSH outlier scores and cluster persistence of
// scikit-learn-contrib/hdbscan 0.8.44 (BSD-3-Clause). See
// LICENSE.scikit_learn.txt in this directory for the attribution.
//
// Frozen semantics (scikit-learn `HDBSCAN(metric="euclidean", alpha=1,
// cluster_selection_epsilon=0, max_cluster_size=None)`):
//   * core distance = distance to the min_samples-th nearest row, the row
//     itself included (distance 0), i.e. scikit-learn `min_samples`
//     (scikit-learn-contrib/hdbscan `min_samples` + 1);
//   * mutual reachability = max(core_i, core_j, d_ij);
//   * MST by Prim's algorithm on the implicit mutual-reachability graph,
//     exactly as `mst_from_data_matrix` (first index wins on equal weights);
//   * single linkage from the MST edges sorted by weight with a STABLE sort
//     (ties keep Prim insertion order). scikit-learn uses an unspecified
//     `np.argsort` order here, so only tie order may differ from it;
//   * condensed tree with `min_cluster_size`, lambda = 1/distance
//     (+inf at distance 0), stability = sum (lambda_p - lambda_birth) * size;
//   * EOM (a node is replaced by its children only when their summed
//     stability is strictly greater) or leaf selection, `allow_single_cluster`;
//   * labels numbered by increasing condensed-tree node index, noise = -1;
//   * membership = scikit-learn `probabilities_` (a membership strength,
//     not a probability), outlier score = GLOSH, persistence = contrib
//     `cluster_persistence_`.
#pragma once

#include <cstdint>
#include <vector>

namespace n4m::core::hdbscan {

enum class Selection : std::int32_t { eom = 0, leaf = 1 };

struct Params {
    std::int64_t min_cluster_size = 5;
    std::int64_t min_samples = 5;
    Selection selection = Selection::eom;
    bool allow_single_cluster = false;
};

// One condensed-tree cluster node. Node 0 is the root.
struct TreeNode {
    std::int64_t parent = -1;   // parent node index, -1 for the root
    std::int64_t size = 0;      // points when the node is born
    double lambda_birth = 0.0;  // 0 for the root
    double lambda_death = 0.0;  // largest lambda at which a point/child leaves
    double stability = 0.0;     // excess of mass before selection
    std::int64_t label = -1;    // selected cluster label or -1
};

struct Result {
    std::vector<std::int64_t> labels;
    std::vector<double> membership;
    std::vector<double> outlier_scores;
    std::vector<double> persistence;  // one value per selected label
    std::vector<TreeNode> tree;
};

// `data` is a row-major, contiguous, finite n x p block with n >= 2,
// 1 <= min_samples <= n and min_cluster_size >= 2 (validated by the caller).
Result fit(const double* data, std::int64_t n, std::int64_t p, const Params& params);

}  // namespace n4m::core::hdbscan
