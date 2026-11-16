// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
//
// Path-dependent TreeSHAP (Lundberg et al. 2020, "From local explanations to
// global understanding with explainable AI for trees", Algorithm 2), written
// after the reference shap/cext/tree_shap.h (extend_path, unwind_path,
// unwound_path_sum, tree_shap_recursive) with an explicit stack instead of
// recursion so deep trees cannot exhaust small WebAssembly stacks.
//
// Zero fractions are cover[child] / (cover[left] + cover[right]) and the
// expected value is the matching recursive cover-weighted mean of the leaves,
// so sum(phi) + expected == prediction for every tree, including trees whose
// stored parent cover differs from the sum of the child covers.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "core/common/context.hpp"
#include "core/explain/explain_internal.hpp"

namespace n4m::explain {
namespace {

struct PathElement {
    int64_t feature = -1;
    double zero_fraction = 0.0;
    double one_fraction = 0.0;
    double pweight = 0.0;
};

void extend_path(PathElement* path, std::size_t depth, double zero_fraction, double one_fraction,
                 int64_t feature) {
    path[depth].feature = feature;
    path[depth].zero_fraction = zero_fraction;
    path[depth].one_fraction = one_fraction;
    path[depth].pweight = depth == 0 ? 1.0 : 0.0;
    const auto l1 = static_cast<double>(depth + 1);
    for (std::size_t k = depth; k-- > 0;) {
        path[k + 1].pweight += one_fraction * path[k].pweight * static_cast<double>(k + 1) / l1;
        path[k].pweight = zero_fraction * path[k].pweight * static_cast<double>(depth - k) / l1;
    }
}

void unwind_path(PathElement* path, std::size_t depth, std::size_t index) {
    const double one_fraction = path[index].one_fraction;
    const double zero_fraction = path[index].zero_fraction;
    const auto l1 = static_cast<double>(depth + 1);
    double next_one_portion = path[depth].pweight;
    for (std::size_t k = depth; k-- > 0;) {
        if (one_fraction != 0.0) {
            const double tmp = path[k].pweight;
            path[k].pweight = next_one_portion * l1 / (static_cast<double>(k + 1) * one_fraction);
            next_one_portion =
                tmp - path[k].pweight * zero_fraction * static_cast<double>(depth - k) / l1;
        } else {
            path[k].pweight =
                path[k].pweight * l1 / (zero_fraction * static_cast<double>(depth - k));
        }
    }
    for (std::size_t k = index; k < depth; ++k) {
        path[k].feature = path[k + 1].feature;
        path[k].zero_fraction = path[k + 1].zero_fraction;
        path[k].one_fraction = path[k + 1].one_fraction;
    }
}

double unwound_path_sum(const PathElement* path, std::size_t depth, std::size_t index) {
    const double one_fraction = path[index].one_fraction;
    const double zero_fraction = path[index].zero_fraction;
    const auto l1 = static_cast<double>(depth + 1);
    double next_one_portion = path[depth].pweight;
    double total = 0.0;
    for (std::size_t k = depth; k-- > 0;) {
        if (one_fraction != 0.0) {
            const double tmp = next_one_portion * l1 / (static_cast<double>(k + 1) * one_fraction);
            total += tmp;
            next_one_portion =
                path[k].pweight - tmp * zero_fraction * static_cast<double>(depth - k) / l1;
        } else {
            total += path[k].pweight * l1 / (zero_fraction * static_cast<double>(depth - k));
        }
    }
    return total;
}

struct Tree {
    std::size_t base = 0;   // global index of local node 0
    std::size_t nodes = 0;  // node count
    std::size_t max_depth = 0;
};

struct Frame {
    std::size_t node;
    std::size_t depth;          // unique_depth argument of the reference recursion
    std::size_t parent_offset;  // parent unique path offset in the path buffer
    double zero_fraction;
    double one_fraction;
    int64_t feature;
};

class Explainer {
  public:
    Explainer(const TreeEnsembleView& trees, std::size_t n_outputs)
        : t_(trees), n_outputs_(n_outputs) {}

    bool goes_left(std::size_t k, double x) const noexcept {
        if (std::isnan(x)) {
            return t_.default_left[k] != 0;
        }
        return t_.decision_type == 0 ? x <= t_.threshold[k] : x < t_.threshold[k];
    }

    // Leaf reached by x in the tree (global index).
    std::size_t leaf(const Tree& tree, const double* x) const noexcept {
        std::size_t k = tree.base;
        while (t_.left[k] >= 0) {
            const double v = x[static_cast<std::size_t>(t_.feature[k])];
            const int64_t child = goes_left(k, v) ? t_.left[k] : t_.right[k];
            k = tree.base + static_cast<std::size_t>(child);
        }
        return k;
    }

    // Accumulates unscaled per-feature Shapley values of one tree into phi (F x O).
    void shap(const Tree& tree, const double* x, std::vector<PathElement>& buffer,
              std::vector<Frame>& stack, double* phi) const {
        const std::size_t O = n_outputs_;
        stack.clear();
        stack.push_back(Frame{0, 0, 0, 1.0, 1.0, -1});
        while (!stack.empty()) {
            const Frame frame = stack.back();
            stack.pop_back();
            std::size_t depth = frame.depth;
            const std::size_t offset = frame.parent_offset + depth + 1;
            PathElement* path = &buffer[offset];
            std::copy(&buffer[frame.parent_offset], &buffer[frame.parent_offset] + depth + 1,
                      path);
            extend_path(path, depth, frame.zero_fraction, frame.one_fraction, frame.feature);
            const std::size_t k = tree.base + frame.node;
            if (t_.left[k] < 0) {
                for (std::size_t i = 1; i <= depth; ++i) {
                    const double w = unwound_path_sum(path, depth, i);
                    const PathElement& element = path[i];
                    const double scale = w * (element.one_fraction - element.zero_fraction);
                    double* target = phi + static_cast<std::size_t>(element.feature) * O;
                    for (std::size_t o = 0; o < O; ++o) {
                        target[o] += scale * t_.value[k * O + o];
                    }
                }
                continue;
            }
            const int64_t split = t_.feature[k];
            const auto left = static_cast<std::size_t>(t_.left[k]);
            const auto right = static_cast<std::size_t>(t_.right[k]);
            const bool hot_left = goes_left(k, x[static_cast<std::size_t>(split)]);
            const std::size_t hot = hot_left ? left : right;
            const std::size_t cold = hot_left ? right : left;
            const double cover_sum = t_.cover[tree.base + left] + t_.cover[tree.base + right];
            const double hot_zero = t_.cover[tree.base + hot] / cover_sum;
            const double cold_zero = t_.cover[tree.base + cold] / cover_sum;
            double incoming_zero = 1.0;
            double incoming_one = 1.0;
            std::size_t index = 0;
            while (index <= depth && path[index].feature != split) {
                ++index;
            }
            if (index <= depth) {
                // Undo the earlier split on the same feature and redo it here.
                incoming_zero = path[index].zero_fraction;
                incoming_one = path[index].one_fraction;
                unwind_path(path, depth, index);
                depth -= 1;
            }
            stack.push_back(Frame{cold, depth + 1, offset, cold_zero * incoming_zero, 0.0, split});
            stack.push_back(
                Frame{hot, depth + 1, offset, hot_zero * incoming_zero, incoming_one, split});
        }
    }

  private:
    const TreeEnsembleView& t_;
    std::size_t n_outputs_;
};

}  // namespace

n4m_status_t tree_shap_path_dependent(n4m_context_t* ctx, const n4m_matrix_view_t* X,
                                      const TreeEnsembleView& t, const int64_t* feature_group,
                                      int64_t n_groups, double* attributions,
                                      int64_t attributions_size, double* expected,
                                      int64_t expected_size, double* predictions,
                                      int64_t predictions_size) {
    n4m_status_t status = validate_f64_view(ctx, X, "X", 1);
    if (status != N4M_OK) {
        return status;
    }
    if (t.tree_offsets == nullptr || t.feature == nullptr || t.threshold == nullptr
        || t.left == nullptr || t.right == nullptr || t.cover == nullptr || t.value == nullptr
        || attributions == nullptr || expected == nullptr) {
        return fail(ctx, N4M_ERR_NULL_POINTER,
                    "tree arrays (except default_left/base_offset), attributions and expected "
                    "must be non-NULL");
    }
    if (t.n_trees < 1 || t.n_outputs < 1) {
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "n_trees and n_outputs must be >= 1");
    }
    if (t.decision_type != 0 && t.decision_type != 1) {
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "decision_type must be 0 (<=) or 1 (<)");
    }
    if (!std::isfinite(t.tree_scale)) {
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "tree_scale must be finite");
    }
    const int64_t n_features = X->cols;
    std::vector<std::size_t> group_of_feature;
    status = validate_groups(ctx, feature_group, n_features, n_groups, group_of_feature);
    if (status != N4M_OK) {
        return status;
    }
    int64_t n_attr = 0;
    int64_t n_out = 0;
    if (!checked_mul(X->rows, n_groups, n_attr) || !checked_mul(n_attr, t.n_outputs, n_attr)
        || !checked_mul(X->rows, t.n_outputs, n_out)) {
        return fail(ctx, N4M_ERR_UNSUPPORTED, "output size overflows int64");
    }
    if (attributions_size < n_attr || expected_size < n_out
        || (predictions != nullptr && predictions_size < n_out)) {
        return fail(ctx, N4M_ERR_SHAPE_MISMATCH,
                    "output buffers are smaller than n_rows * n_groups * n_outputs "
                    "(attributions) or n_rows * n_outputs (expected, predictions)");
    }
    const auto O = static_cast<std::size_t>(t.n_outputs);
    if (t.base_offset != nullptr) {
        for (std::size_t o = 0; o < O; ++o) {
            if (!std::isfinite(t.base_offset[o])) {
                return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "base_offset must be finite");
            }
        }
    }

    // Structural validation of every tree reachable from its root.
    if (t.tree_offsets[0] != 0) {
        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "tree_offsets[0] must be 0");
    }
    const auto n_trees = static_cast<std::size_t>(t.n_trees);
    std::vector<Tree> trees(n_trees);
    for (std::size_t i = 0; i < n_trees; ++i) {
        if (t.tree_offsets[i + 1] <= t.tree_offsets[i]) {
            return fail(ctx, N4M_ERR_INVALID_ARGUMENT,
                        "tree_offsets must be strictly increasing (every tree has a node)");
        }
        trees[i].base = static_cast<std::size_t>(t.tree_offsets[i]);
        trees[i].nodes = static_cast<std::size_t>(t.tree_offsets[i + 1] - t.tree_offsets[i]);
    }
    int64_t n_values = 0;
    if (!checked_mul(t.tree_offsets[n_trees], t.n_outputs, n_values)) {
        return fail(ctx, N4M_ERR_UNSUPPORTED, "node value array size overflows int64");
    }
    std::vector<double> tree_expected(n_trees * O, 0.0);
    std::vector<unsigned char> visited;
    std::vector<std::pair<std::size_t, std::size_t>> todo;
    std::vector<std::size_t> order;
    std::vector<double> node_expected;
    for (std::size_t i = 0; i < n_trees; ++i) {
        Tree& tree = trees[i];
        visited.assign(tree.nodes, 0);
        order.clear();
        todo.assign(1, {0, 0});
        while (!todo.empty()) {
            const auto [local, depth] = todo.back();
            todo.pop_back();
            if (visited[local] != 0) {
                return fail(ctx, N4M_ERR_INVALID_ARGUMENT,
                            "tree nodes must form a tree (a node is reached twice or a cycle)");
            }
            visited[local] = 1;
            order.push_back(local);
            tree.max_depth = std::max(tree.max_depth, depth);
            const std::size_t k = tree.base + local;
            const double cover = t.cover[k];
            if (!std::isfinite(cover) || !(cover > 0.0)) {
                return fail(ctx, N4M_ERR_INVALID_ARGUMENT,
                            "cover must be finite and > 0 on every reachable node");
            }
            const int64_t l = t.left[k];
            const int64_t r = t.right[k];
            if (l == -1 && r == -1) {
                for (std::size_t o = 0; o < O; ++o) {
                    if (!std::isfinite(t.value[k * O + o])) {
                        return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "leaf values must be finite");
                    }
                }
                continue;
            }
            const auto n_local = static_cast<int64_t>(tree.nodes);
            if (l < 0 || r < 0 || l >= n_local || r >= n_local || l == r) {
                return fail(ctx, N4M_ERR_INVALID_ARGUMENT,
                            "internal nodes need two distinct local children; leaves need -1/-1");
            }
            if (t.feature[k] < 0 || t.feature[k] >= n_features) {
                return fail(ctx, N4M_ERR_INVALID_ARGUMENT,
                            "split feature index must lie in [0, n_features)");
            }
            if (!std::isfinite(t.threshold[k])) {
                return fail(ctx, N4M_ERR_INVALID_ARGUMENT, "split thresholds must be finite");
            }
            todo.emplace_back(static_cast<std::size_t>(l), depth + 1);
            todo.emplace_back(static_cast<std::size_t>(r), depth + 1);
        }
        // Recursive cover-weighted mean of the leaves (children follow parents in `order`).
        node_expected.assign(tree.nodes * O, 0.0);
        for (std::size_t n = order.size(); n-- > 0;) {
            const std::size_t local = order[n];
            const std::size_t k = tree.base + local;
            if (t.left[k] < 0) {
                for (std::size_t o = 0; o < O; ++o) {
                    node_expected[local * O + o] = t.value[k * O + o];
                }
                continue;
            }
            const auto l = static_cast<std::size_t>(t.left[k]);
            const auto r = static_cast<std::size_t>(t.right[k]);
            const double cl = t.cover[tree.base + l];
            const double cr = t.cover[tree.base + r];
            for (std::size_t o = 0; o < O; ++o) {
                node_expected[local * O + o] =
                    (cl * node_expected[l * O + o] + cr * node_expected[r * O + o]) / (cl + cr);
            }
        }
        for (std::size_t o = 0; o < O; ++o) {
            tree_expected[i * O + o] = node_expected[o];
        }
    }

    // Input rows: NaN follows default_left when provided, otherwise it is rejected.
    const auto R = static_cast<std::size_t>(X->rows);
    const auto F = static_cast<std::size_t>(n_features);
    const auto G = static_cast<std::size_t>(n_groups);
    std::vector<double> x(R * F);
    for (std::size_t r = 0; r < R; ++r) {
        for (std::size_t j = 0; j < F; ++j) {
            const double v =
                view_at(*X, static_cast<int64_t>(r), static_cast<int64_t>(j));
            if (std::isnan(v) && t.default_left == nullptr) {
                return fail(ctx, N4M_ERR_INVALID_ARGUMENT,
                            "X contains NaN but default_left is NULL");
            }
            x[r * F + j] = v;
        }
    }

    std::vector<double> base(O, 0.0);
    std::vector<double> e(O, 0.0);
    for (std::size_t o = 0; o < O; ++o) {
        base[o] = t.base_offset != nullptr ? t.base_offset[o] : 0.0;
        double sum = 0.0;
        for (std::size_t i = 0; i < n_trees; ++i) {
            sum += tree_expected[i * O + o];
        }
        e[o] = base[o] + t.tree_scale * sum;
    }
    std::size_t max_depth = 0;
    for (const Tree& tree : trees) {
        max_depth = std::max(max_depth, tree.max_depth);
    }
    std::vector<PathElement> buffer((max_depth + 3) * (max_depth + 4) / 2);
    std::vector<Frame> stack;
    stack.reserve(2 * (max_depth + 1));
    const Explainer explainer(t, O);
    std::vector<double> phi(R * G * O, 0.0);
    std::vector<double> pred(R * O, 0.0);
    std::vector<double> phi_feature(F * O, 0.0);
    for (std::size_t r = 0; r < R; ++r) {
        const double* xr = &x[r * F];
        std::fill(phi_feature.begin(), phi_feature.end(), 0.0);
        std::vector<double> leaf_sum(O, 0.0);
        for (const Tree& tree : trees) {
            explainer.shap(tree, xr, buffer, stack, phi_feature.data());
            const std::size_t k = explainer.leaf(tree, xr);
            for (std::size_t o = 0; o < O; ++o) {
                leaf_sum[o] += t.value[k * O + o];
            }
        }
        for (std::size_t j = 0; j < F; ++j) {
            const std::size_t g = group_of_feature[j];
            for (std::size_t o = 0; o < O; ++o) {
                phi[(r * G + g) * O + o] += t.tree_scale * phi_feature[j * O + o];
            }
        }
        for (std::size_t o = 0; o < O; ++o) {
            pred[r * O + o] = base[o] + t.tree_scale * leaf_sum[o];
        }
    }
    std::copy(phi.begin(), phi.end(), attributions);
    for (std::size_t r = 0; r < R; ++r) {
        std::copy(e.begin(), e.end(), expected + r * O);
    }
    if (predictions != nullptr) {
        std::copy(pred.begin(), pred.end(), predictions);
    }
    return N4M_OK;
}

}  // namespace n4m::explain
