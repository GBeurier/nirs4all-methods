// SPDX-License-Identifier: CECILL-2.1
//
// HDBSCAN* port. The algorithm, control flow and floating-point operation
// order follow scikit-learn 1.9.1 (`_hdbscan_prims`, `mst_from_data_matrix`,
// `make_single_linkage`, `_condense_tree`, `_compute_stability`,
// `_get_clusters`, `_do_labelling`, `get_probabilities`) and
// scikit-learn-contrib/hdbscan 0.8.44 (`outlier_scores`,
// `get_stability_scores`). Both are BSD-3-Clause; this is an independent C++
// rewrite, see LICENSE.scikit_learn.txt. Only the fixed configuration
// documented in hdbscan.hpp is ported (no epsilon, max size, alpha, metrics).
#include "core/clustering/_vendored/hdbscan.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <utility>

namespace n4m::core::hdbscan {
namespace {

constexpr double infinity = std::numeric_limits<double>::infinity();

struct Edge {
    std::size_t current;
    std::size_t next;
    double distance;
};

struct Merge {
    std::size_t left;
    std::size_t right;
    double distance;
    std::size_t size;
};

struct Row {
    std::size_t parent;
    std::size_t child;
    double lambda;
    std::size_t size;
};

// The C API admits only the numerical domain of rules 13/15 (finite values,
// nonzero |x| in [2^-256, 2^256], spread <= 2^32). There every nonzero
// coordinate difference lies in [2^-308, 2^257], so every squared distance is
// a normal double and the scikit-learn sum of squares is kept unchanged.
double euclidean(const double* x, const double* y, std::size_t p) noexcept {
    double d = 0.0;
    for (std::size_t k = 0; k < p; ++k) {
        const double t = x[k] - y[k];
        d += t * t;
    }
    return std::sqrt(d);
}

// Distance to the k-th nearest row, the row itself counted (sklearn
// `NearestNeighbors.kneighbors(X, min_samples)[:, -1]`).
std::vector<double> core_distances(const double* x, std::size_t n, std::size_t p,
                                   std::size_t k) {
    std::vector<double> core(n), row(n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) row[j] = euclidean(x + i * p, x + j * p, p);
        const auto kth = row.begin() + static_cast<std::ptrdiff_t>(k - 1);
        std::nth_element(row.begin(), kth, row.end());
        core[i] = *kth;
    }
    return core;
}

// Prim's algorithm on the implicit mutual-reachability graph
// (`mst_from_data_matrix`, alpha = 1).
std::vector<Edge> prim_mst(const double* x, std::size_t n, std::size_t p,
                           const std::vector<double>& core) {
    std::vector<Edge> mst(n - 1);
    std::vector<unsigned char> in_tree(n, 0);
    std::vector<double> min_reachability(n, infinity);
    std::vector<std::size_t> sources(n, 1);
    std::size_t current = 0;
    for (std::size_t i = 0; i + 1 < n; ++i) {
        in_tree[current] = 1;
        const double current_core = core[current];
        double new_reachability = DBL_MAX;
        std::size_t source = 0, new_node = 0;
        for (std::size_t j = 0; j < n; ++j) {
            if (in_tree[j]) continue;
            const double next_min = min_reachability[j];
            const std::size_t next_source = sources[j];
            const double pair = euclidean(x + current * p, x + j * p, p);
            const double mutual = std::max(std::max(current_core, core[j]), pair);
            if (mutual < next_min) {
                min_reachability[j] = mutual;
                sources[j] = current;
                if (mutual < new_reachability) {
                    new_reachability = mutual;
                    source = current;
                    new_node = j;
                }
            } else if (next_min < new_reachability) {
                new_reachability = next_min;
                source = next_source;
                new_node = j;
            }
        }
        mst[i] = {source, new_node, new_reachability};
        current = new_node;
    }
    return mst;
}

// Sorted MST -> scipy-style linkage (`make_single_linkage`).
std::vector<Merge> single_linkage(std::vector<Edge> mst, std::size_t n) {
    std::stable_sort(mst.begin(), mst.end(),
                     [](const Edge& a, const Edge& b) { return a.distance < b.distance; });
    std::vector<std::size_t> parent(2 * n - 1, SIZE_MAX), size(2 * n - 1, 0);
    std::fill(size.begin(), size.begin() + static_cast<std::ptrdiff_t>(n), std::size_t{1});
    auto find = [&parent](std::size_t v) {
        std::size_t root = v;
        while (parent[root] != SIZE_MAX) root = parent[root];
        while (parent[v] != SIZE_MAX && parent[v] != root) {
            const std::size_t up = parent[v];
            parent[v] = root;
            v = up;
        }
        return root;
    };
    std::vector<Merge> merges(n - 1);
    std::size_t next_label = n;
    for (std::size_t i = 0; i + 1 < n; ++i) {
        const std::size_t a = find(mst[i].current), b = find(mst[i].next);
        if (a == b) return {};  // a cycle: the caller rejects the empty linkage
        merges[i] = {a, b, mst[i].distance, size[a] + size[b]};
        parent[a] = parent[b] = next_label;
        size[next_label] = size[a] + size[b];
        ++next_label;
    }
    return merges;
}

std::vector<std::size_t> bfs_from_hierarchy(const std::vector<Merge>& h, std::size_t root,
                                            std::size_t n) {
    std::vector<std::size_t> result, queue{root}, next;
    while (!queue.empty()) {
        result.insert(result.end(), queue.begin(), queue.end());
        next.clear();
        for (const std::size_t node : queue) {
            if (node < n) continue;
            next.push_back(h[node - n].left);
            next.push_back(h[node - n].right);
        }
        queue.swap(next);
    }
    return result;
}

// `_condense_tree`: rows in breadth-first order, cluster ids n, n+1, ...
std::vector<Row> condense(const std::vector<Merge>& h, std::size_t n, std::size_t min_size,
                          std::size_t& n_nodes) {
    const std::size_t root = 2 * (n - 1);
    std::size_t next_label = n + 1;
    std::vector<std::size_t> relabel(root + 1, 0);
    relabel[root] = n;
    std::vector<unsigned char> ignore(root + 1, 0);
    std::vector<Row> rows;
    auto drop = [&](std::size_t parent_label, std::size_t subtree, double lambda) {
        for (const std::size_t sub : bfs_from_hierarchy(h, subtree, n)) {
            if (sub < n) rows.push_back({parent_label, sub, lambda, 1});
            ignore[sub] = 1;
        }
    };
    for (const std::size_t node : bfs_from_hierarchy(h, root, n)) {
        if (ignore[node] || node < n) continue;
        const Merge& m = h[node - n];
        const double lambda = m.distance > 0.0 ? 1.0 / m.distance : infinity;
        const std::size_t left_count = m.left >= n ? h[m.left - n].size : 1;
        const std::size_t right_count = m.right >= n ? h[m.right - n].size : 1;
        const std::size_t label = relabel[node];
        if (left_count >= min_size && right_count >= min_size) {
            relabel[m.left] = next_label++;
            rows.push_back({label, relabel[m.left], lambda, left_count});
            relabel[m.right] = next_label++;
            rows.push_back({label, relabel[m.right], lambda, right_count});
        } else if (left_count < min_size && right_count < min_size) {
            drop(label, m.left, lambda);
            drop(label, m.right, lambda);
        } else if (left_count < min_size) {
            relabel[m.right] = label;
            drop(label, m.left, lambda);
        } else {
            relabel[m.left] = label;
            drop(label, m.right, lambda);
        }
    }
    n_nodes = next_label - n;
    return rows;
}

}  // namespace

Result fit(const double* data, std::int64_t n_rows, std::int64_t n_cols, const Params& params) {
    const auto n = static_cast<std::size_t>(n_rows);
    const auto p = static_cast<std::size_t>(n_cols);
    const auto min_size = static_cast<std::size_t>(params.min_cluster_size);
    const std::vector<double> core =
        core_distances(data, n, p, static_cast<std::size_t>(params.min_samples));
    std::vector<Edge> mst = prim_mst(data, n, p, core);
    const auto failure = [](Error error) {
        Result failed;
        failed.error = error;
        return failed;
    };
    for (const Edge& e : mst) {
        if (e.current >= n || e.next >= n || e.current == e.next)
            return failure(Error::invalid_tree);
    }
    const std::vector<Merge> merges = single_linkage(std::move(mst), n);
    if (merges.size() != n - 1) return failure(Error::invalid_tree);
    std::size_t k = 0;
    const std::vector<Row> rows = condense(merges, n, min_size, k);

    // Node bookkeeping (`_compute_stability`, max lambda per parent).
    std::vector<double> birth(k, 0.0), death(k, 0.0), stability(k, 0.0);
    std::vector<std::size_t> node_parent(k, SIZE_MAX), node_size(k, n);
    std::vector<std::vector<std::size_t>> children(k);
    std::vector<double> point_lambda(n, 0.0);
    std::vector<std::size_t> point_parent(n, 0);
    double max_lambda = 0.0;
    for (const Row& r : rows) {
        const std::size_t parent = r.parent - n;
        death[parent] = std::max(death[parent], r.lambda);
        max_lambda = std::max(max_lambda, r.lambda);
        if (r.child >= n) {
            const std::size_t child = r.child - n;
            birth[child] = r.lambda;
            node_parent[child] = parent;
            node_size[child] = r.size;
            children[parent].push_back(child);
        } else {
            point_lambda[r.child] = r.lambda;
            point_parent[r.child] = parent;
        }
    }
    for (const Row& r : rows) {
        const std::size_t parent = r.parent - n;
        // A node born at zero distance whose points leave at zero distance
        // carries no excess of mass (scikit-learn would form inf - inf = NaN).
        const double excess =
            std::isinf(r.lambda) && std::isinf(birth[parent]) ? 0.0 : r.lambda - birth[parent];
        stability[parent] += excess * static_cast<double>(r.size);
    }

    // Cluster selection (`_get_clusters`). Children have larger node indices
    // than their parent, so a descending sweep visits children first.
    std::vector<unsigned char> selected(k, 0);
    const std::size_t lowest = params.allow_single_cluster ? 0 : 1;
    if (params.selection == Selection::eom) {
        std::vector<double> mass = stability;
        for (std::size_t node = k; node-- > lowest;) selected[node] = 1;
        for (std::size_t node = k; node-- > lowest;) {
            double subtree = 0.0;
            for (const std::size_t child : children[node]) subtree += mass[child];
            if (subtree > mass[node]) {
                selected[node] = 0;
                mass[node] = subtree;
            } else {
                std::vector<std::size_t> stack(children[node]);
                while (!stack.empty()) {
                    const std::size_t sub = stack.back();
                    stack.pop_back();
                    selected[sub] = 0;
                    stack.insert(stack.end(), children[sub].begin(), children[sub].end());
                }
            }
        }
    } else {
        // Leaves of the cluster tree; with no cluster tree, nothing is selected.
        for (std::size_t node = 1; node < k; ++node) selected[node] = children[node].empty();
    }

    Result out;
    out.tree.resize(k);
    std::vector<std::size_t> label_node;
    for (std::size_t node = 0; node < k; ++node) {
        TreeNode& t = out.tree[node];
        t.parent = node_parent[node] == SIZE_MAX ? -1 : static_cast<std::int64_t>(node_parent[node]);
        t.size = static_cast<std::int64_t>(node_size[node]);
        t.lambda_birth = birth[node];
        t.lambda_death = death[node];
        t.stability = stability[node];
        if (selected[node]) {
            t.label = static_cast<std::int64_t>(label_node.size());
            label_node.push_back(node);
        }
    }

    // `_do_labelling`: a point belongs to its nearest selected ancestor.
    std::vector<std::size_t> owner(k, 0);
    for (std::size_t node = 1; node < k; ++node)
        owner[node] = selected[node] ? node : owner[node_parent[node]];
    out.labels.assign(n, -1);
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t cluster = owner[point_parent[i]];
        if (cluster != 0) {
            out.labels[i] = out.tree[cluster].label;
        } else if (selected[0] && point_lambda[i] >= death[0]) {
            out.labels[i] = out.tree[0].label;
        }
    }

    // `get_probabilities` (scikit-learn) and `outlier_scores` (GLOSH).
    std::vector<double> subtree_death = death;
    for (std::size_t node = k; node-- > 1;)
        if (subtree_death[node] > subtree_death[node_parent[node]])
            subtree_death[node_parent[node]] = subtree_death[node];
    out.membership.assign(n, 0.0);
    out.outlier_scores.assign(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        const double lambda = point_lambda[i];
        if (out.labels[i] >= 0) {
            const double cluster_death =
                death[label_node[static_cast<std::size_t>(out.labels[i])]];
            out.membership[i] = (cluster_death == 0.0 || std::isinf(lambda))
                                    ? 1.0
                                    : std::min(lambda, cluster_death) / cluster_death;
        }
        const double glosh_death = subtree_death[point_parent[i]];
        out.outlier_scores[i] = (glosh_death == 0.0 || !std::isfinite(lambda))
                                    ? 0.0
                                    : (glosh_death - lambda) / glosh_death;
    }

    // contrib `get_stability_scores` with the global maximum lambda.
    std::vector<std::size_t> members(label_node.size(), 0);
    for (const std::int64_t label : out.labels)
        if (label >= 0) ++members[static_cast<std::size_t>(label)];
    out.persistence.resize(label_node.size());
    for (std::size_t c = 0; c < label_node.size(); ++c) {
        const double denominator = static_cast<double>(members[c]) * max_lambda;
        out.persistence[c] =
            (std::isinf(max_lambda) || max_lambda == 0.0 || members[c] == 0)
                ? 1.0
                : stability[label_node[c]] / denominator;
    }
    return out;
}

}  // namespace n4m::core::hdbscan
