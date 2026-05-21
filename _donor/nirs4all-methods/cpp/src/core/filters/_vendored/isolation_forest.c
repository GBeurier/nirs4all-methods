/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Isolation Forest — vendored minimal implementation. See header.
 *
 * Tree representation: implicit binary tree stored as a flat node array,
 * left child at ``left[i]`` and right child at ``right[i]``. Internal nodes
 * carry (split_feature, split_value); leaves carry (-1, n_samples_at_leaf).
 *
 * Memory: each tree allocates its own node vector. Re-allocations follow
 * a doubling strategy starting at 31 nodes (~ ceil(log2(256)) * 2 + 1 = 17,
 * doubled for safety).
 */

#include "isolation_forest.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "core/common/rng_pcg64.h"

#define N4M_IFOREST_INITIAL_NODES 31

typedef struct iforest_node_t {
    int32_t  feature;       /* -1 for leaf, otherwise split column index    */
    int32_t  n_samples;     /* leaf: sample count; internal: unused         */
    double   split_value;
    int32_t  left;
    int32_t  right;
    int32_t  depth;
} iforest_node_t;

typedef struct iforest_tree_t {
    iforest_node_t* nodes;
    int32_t         n_nodes;
    int32_t         cap;
    int32_t         max_depth;
} iforest_tree_t;

struct n4m_iforest_t {
    iforest_tree_t* trees;
    int             n_trees;
    int64_t         max_samples;
    int64_t         cols;
};

/* Expected average path length of an unsuccessful search in a BST of n
 * elements. c(1) = 0 by definition. */
static double iforest_path_length_c(int64_t n) {
    if (n <= 1) return 0.0;
    const double H = log((double)(n - 1)) + 0.5772156649015329;
    return 2.0 * H - 2.0 * (double)(n - 1) / (double)n;
}

static int tree_ensure_capacity(iforest_tree_t* t, int32_t needed) {
    if (needed <= t->cap) return 0;
    int32_t new_cap = (t->cap == 0) ? N4M_IFOREST_INITIAL_NODES : t->cap;
    while (new_cap < needed) {
        new_cap = new_cap * 2;
        if (new_cap <= 0) return -1;
    }
    iforest_node_t* p = (iforest_node_t*)realloc(
        t->nodes, (size_t)new_cap * sizeof(iforest_node_t));
    if (p == NULL) return -1;
    t->nodes = p;
    t->cap = new_cap;
    return 0;
}

/* Append a node and return its index. */
static int32_t tree_add_node(iforest_tree_t* t) {
    if (tree_ensure_capacity(t, t->n_nodes + 1) != 0) return -1;
    const int32_t idx = t->n_nodes++;
    iforest_node_t* nd = &t->nodes[idx];
    nd->feature = -1;
    nd->n_samples = 0;
    nd->split_value = 0.0;
    nd->left = -1;
    nd->right = -1;
    nd->depth = 0;
    return idx;
}

static double rng_uniform(n4m_rng_pcg64* rng) {
    return n4m_pcg64_engine_next_double(rng);
}

static int64_t rng_int_below(n4m_rng_pcg64* rng, int64_t upper) {
    /* Lemire-style rejection-free range (acceptable for small upper). */
    if (upper <= 1) return 0;
    const uint64_t mask = (uint64_t)upper;
    uint64_t x = n4m_pcg64_engine_next_uint64(rng) & 0x7FFFFFFFFFFFFFFFULL;
    return (int64_t)(x % mask);
}

/* Build the tree recursively on a sub-sample of indices.
 *
 * indices[lo..hi) is the current partition of indices into X.
 * Returns the index of the created node, or -1 on allocation failure. */
static int32_t build_node(iforest_tree_t* t,
                          const double* X, int64_t cols,
                          int32_t* indices, int32_t lo, int32_t hi,
                          int32_t depth,
                          n4m_rng_pcg64* rng) {
    const int32_t idx = tree_add_node(t);
    if (idx < 0) return -1;
    t->nodes[idx].depth = depth;
    const int32_t n_here = hi - lo;
    if (n_here <= 1 || depth >= t->max_depth) {
        t->nodes[idx].feature = -1;
        t->nodes[idx].n_samples = n_here;
        return idx;
    }
    /* Pick a random column with non-degenerate range. Up to ``cols`` retries
     * before falling back to a leaf node (mirrors sklearn's behaviour when
     * every feature is constant in the partition). */
    int32_t feat = -1;
    double min_v = 0.0, max_v = 0.0;
    for (int retry = 0; retry < (int)cols; ++retry) {
        const int32_t cand = (int32_t)rng_int_below(rng, cols);
        double lo_v = X[(size_t)indices[lo] * (size_t)cols + (size_t)cand];
        double hi_v = lo_v;
        for (int32_t k = lo + 1; k < hi; ++k) {
            const double v = X[(size_t)indices[k] * (size_t)cols + (size_t)cand];
            if (v < lo_v) lo_v = v;
            if (v > hi_v) hi_v = v;
        }
        if (hi_v > lo_v) {
            feat = cand;
            min_v = lo_v;
            max_v = hi_v;
            break;
        }
    }
    if (feat < 0) {
        t->nodes[idx].feature = -1;
        t->nodes[idx].n_samples = n_here;
        return idx;
    }
    const double u = rng_uniform(rng);
    const double split = min_v + u * (max_v - min_v);
    /* Partition indices[lo..hi) by < split / >= split. */
    int32_t left_cursor = lo, right_cursor = hi - 1;
    while (left_cursor <= right_cursor) {
        const double v = X[(size_t)indices[left_cursor] * (size_t)cols
                            + (size_t)feat];
        if (v < split) {
            ++left_cursor;
        } else {
            const int32_t tmp = indices[left_cursor];
            indices[left_cursor] = indices[right_cursor];
            indices[right_cursor] = tmp;
            --right_cursor;
        }
    }
    const int32_t mid = left_cursor;
    /* Guard against degenerate splits where every element fell on one side
     * (e.g. all values equal the chosen split). Promote to leaf. */
    if (mid == lo || mid == hi) {
        t->nodes[idx].feature = -1;
        t->nodes[idx].n_samples = n_here;
        return idx;
    }
    /* Set internal node fields _before_ recursing because t->nodes may move
     * when capacity grows. */
    t->nodes[idx].feature = feat;
    t->nodes[idx].split_value = split;
    const int32_t left = build_node(t, X, cols, indices, lo, mid,
                                      depth + 1, rng);
    if (left < 0) return -1;
    /* Re-read because realloc may have moved t->nodes. */
    t->nodes[idx].left = left;
    const int32_t right = build_node(t, X, cols, indices, mid, hi,
                                       depth + 1, rng);
    if (right < 0) return -1;
    t->nodes[idx].right = right;
    return idx;
}

static n4m_status_t build_tree(iforest_tree_t* t,
                                const double* X, int64_t rows, int64_t cols,
                                int64_t sub_n,
                                n4m_rng_pcg64* rng) {
    t->nodes = NULL;
    t->n_nodes = 0;
    t->cap = 0;
    t->max_depth = (int32_t)ceil(log2((double)sub_n));
    if (t->max_depth < 1) t->max_depth = 1;
    /* Sub-sample rows without replacement via partial Fisher-Yates. */
    int32_t* perm = (int32_t*)malloc((size_t)rows * sizeof(int32_t));
    if (perm == NULL) return N4M_ERR_OUT_OF_MEMORY;
    for (int32_t i = 0; i < (int32_t)rows; ++i) perm[i] = i;
    for (int64_t i = 0; i < sub_n; ++i) {
        const int64_t j = i + rng_int_below(rng, rows - i);
        const int32_t tmp = perm[i];
        perm[i] = perm[j];
        perm[j] = tmp;
    }
    const int32_t root = build_node(t, X, cols, perm, 0, (int32_t)sub_n,
                                       0, rng);
    free(perm);
    if (root < 0) {
        free(t->nodes);
        t->nodes = NULL;
        return N4M_ERR_OUT_OF_MEMORY;
    }
    return N4M_OK;
}

/* Walk one tree for one sample; return path length with leaf correction. */
static double tree_path_length(const iforest_tree_t* t,
                                const double* row, int64_t cols) {
    int32_t idx = 0;
    while (idx >= 0 && t->nodes[idx].feature >= 0) {
        const iforest_node_t* nd = &t->nodes[idx];
        const double v = row[(size_t)nd->feature];
        idx = (v < nd->split_value) ? nd->left : nd->right;
    }
    (void)cols;
    if (idx < 0) return 0.0;
    const iforest_node_t* leaf = &t->nodes[idx];
    return (double)leaf->depth + iforest_path_length_c(leaf->n_samples);
}

n4m_iforest_t* n4m_iforest_new(void) {
    n4m_iforest_t* f = (n4m_iforest_t*)malloc(sizeof(n4m_iforest_t));
    if (f == NULL) return NULL;
    f->trees = NULL;
    f->n_trees = 0;
    f->max_samples = 0;
    f->cols = 0;
    return f;
}

void n4m_iforest_free(n4m_iforest_t* f) {
    if (f == NULL) return;
    for (int i = 0; i < f->n_trees; ++i) {
        free(f->trees[i].nodes);
    }
    free(f->trees);
    free(f);
}

n4m_status_t n4m_iforest_fit(n4m_iforest_t* f,
                              const double* X, int64_t rows, int64_t cols,
                              int n_estimators, int64_t max_samples,
                              uint64_t seed) {
    if (f == NULL) return N4M_ERR_NULL_POINTER;
    if (rows < 1 || cols < 1) return N4M_ERR_INVALID_ARGUMENT;
    if (X == NULL) return N4M_ERR_NULL_POINTER;
    if (n_estimators < 1) return N4M_ERR_INVALID_ARGUMENT;
    if (max_samples < 1) return N4M_ERR_INVALID_ARGUMENT;
    /* Free any existing trees (defensive — caller normally fits once). */
    for (int i = 0; i < f->n_trees; ++i) free(f->trees[i].nodes);
    free(f->trees);
    f->trees = (iforest_tree_t*)calloc((size_t)n_estimators,
                                         sizeof(iforest_tree_t));
    if (f->trees == NULL) return N4M_ERR_OUT_OF_MEMORY;
    f->n_trees = n_estimators;
    f->max_samples = (max_samples < rows) ? max_samples : rows;
    f->cols = cols;
    n4m_rng_pcg64 rng;
    n4m_pcg64_engine_seed(&rng, seed);
    for (int i = 0; i < n_estimators; ++i) {
        const n4m_status_t st = build_tree(&f->trees[i], X, rows, cols,
                                              f->max_samples, &rng);
        if (st != N4M_OK) return st;
    }
    return N4M_OK;
}

n4m_status_t n4m_iforest_score(const n4m_iforest_t* f,
                                const double* X, int64_t rows, int64_t cols,
                                double* scores) {
    if (f == NULL || scores == NULL) return N4M_ERR_NULL_POINTER;
    if (rows < 0 || cols < 0) return N4M_ERR_INVALID_ARGUMENT;
    if (rows == 0) return N4M_OK;
    if (X == NULL) return N4M_ERR_NULL_POINTER;
    if (cols != f->cols) return N4M_ERR_SHAPE_MISMATCH;
    if (f->n_trees == 0) return N4M_ERR_NOT_FITTED;
    const double cn = iforest_path_length_c(f->max_samples);
    const double denom = (cn > 0.0) ? cn : 1.0;
    for (int64_t r = 0; r < rows; ++r) {
        const double* row = X + (size_t)r * (size_t)cols;
        double sum = 0.0;
        for (int t = 0; t < f->n_trees; ++t) {
            sum += tree_path_length(&f->trees[t], row, cols);
        }
        const double mean_h = sum / (double)f->n_trees;
        scores[r] = mean_h / denom;
    }
    return N4M_OK;
}
