#!/usr/bin/env python3
# SPDX-License-Identifier: CECILL-2.1
"""Generate Phase 11 parity fixtures for the sample-partitioning splitters.

Nine operators in the new ``c4a_split_*`` category:

  * KennardStone           — deterministic max-min Euclidean on X.
  * SPXY                   — deterministic max-min on X + Y (joint distance).
  * SPXYFold               — k-fold alternating max-min on X + Y.
  * SPXYGFold              — group-aware k-fold (mean / median aggregation).
  * KMeans                 — k-means++ + Lloyd, PCG64-seeded.
  * KBinsStratified        — sklearn KBinsDiscretizer + StratifiedShuffleSplit.
  * BinnedStratifiedGroupKFold — bin y + group-aware k-fold.
  * SystematicCircular     — argsort y, PCG64 rotation, take every n-th.
  * SPlit                  — data twinning (Vakayil & Joseph 2022) with
                             PCG64-drawn start index.

The fixture for each splitter stores a (rows x cols) X matrix and (rows x 1)
y vector. Per case it carries the expected ``train_idx`` and ``test_idx``
integer arrays for byte-exact comparison against the C engine.

We deliberately re-implement each algorithm in pure NumPy here (instead of
calling sklearn / nirs4all directly) because:

  * Most C splitters use PCG64 for randomness. Re-implementing locally lets us
    seed bit-exactly via ``numpy.random.default_rng(seed)`` (which is PCG64
    under the hood).
  * KBinsStratified is the exception: it intentionally mirrors nirs4all's
    sklearn stack, including RandomState(MT19937) index ordering.
  * Some sklearn behaviour (e.g. StratifiedGroupKFold) carries complex
    constraint-satisfaction heuristics that the C engine simplifies. We
    document the simplification by matching it here.
"""

from __future__ import annotations

import json
import sys
from math import ceil, floor
from pathlib import Path
from struct import pack
from typing import Any, Callable

import numpy as np
from scipy.spatial.distance import cdist


REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Encoding helpers.
# ---------------------------------------------------------------------------

def double_to_hex(value: float) -> str:
    return pack(">d", float(value)).hex()


def array_to_hex(arr: np.ndarray) -> list[str]:
    flat = np.ascontiguousarray(arr, dtype=np.float64).ravel(order="C")
    return [double_to_hex(float(v)) for v in flat]


# ---------------------------------------------------------------------------
# Synthetic dataset.
#
# 60 samples x 20 features. The X distribution mixes a deterministic linear
# pattern (so KennardStone has a well-defined max-distance pair) with
# moderate Gaussian noise. y is a continuous regression target.
# ---------------------------------------------------------------------------

SEED_DATA = 20260518
ROWS = 60
COLS_X = 20


def synthesize_dataset(seed: int = SEED_DATA) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = np.empty((ROWS, COLS_X), dtype=np.float64)
    for i in range(ROWS):
        base = (i / (ROWS - 1)) * 2.0 - 1.0       # in [-1, 1]
        X[i, :] = base + rng.normal(0.0, 0.1, size=COLS_X)
    y = np.linspace(0.0, 1.0, ROWS, dtype=np.float64) + rng.normal(0.0, 0.05, size=ROWS)
    return X, y


# ---------------------------------------------------------------------------
# Shared helpers — mirrored from the C engine.
# ---------------------------------------------------------------------------

def compute_train_test_count(n_samples: int, test_size: float) -> tuple[int, int]:
    n_test = int(ceil(test_size * n_samples))
    if n_test < 1:
        n_test = 1
    if n_test >= n_samples:
        n_test = n_samples - 1
    return n_samples - n_test, n_test


def max_min_selection(D: np.ndarray, train_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Replicates c4a_split_max_min_selection. Returns (train_idx, test_idx)."""
    n = D.shape[0]
    flat_arg = int(np.argmax(D))    # row-major first-occurrence on ties
    first  = flat_arg // n
    second = flat_arg %  n
    if first == second:
        first, second = 0, 1
    in_train = np.zeros(n, dtype=bool)
    in_train[first] = True
    in_train[second] = True
    train = [first, second]
    min_dist = np.minimum(D[first], D[second])
    for _ in range(train_size - 2):
        best_j = -1
        best_d = -1.0
        for j in range(n):
            if in_train[j]:
                continue
            if best_j == -1 or min_dist[j] > best_d:
                best_j = j
                best_d = min_dist[j]
        train.append(best_j)
        in_train[best_j] = True
        for j in range(n):
            if in_train[j]:
                continue
            nd = D[best_j, j]
            if nd < min_dist[j]:
                min_dist[j] = nd
    test = [j for j in range(n) if not in_train[j]]
    return np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64)


def normalise_distance(D: np.ndarray) -> np.ndarray:
    dmax = D.max()
    if dmax <= 0.0:
        return D
    return D / dmax


def euclidean_distance_matrix(M: np.ndarray) -> np.ndarray:
    return cdist(M, M, metric="euclidean").astype(np.float64)


def rng_uniform01(rng: np.random.Generator) -> float:
    """Mirror the C `c4a_split_rng_uniform01`. Both consume one uint64."""
    u = int(rng.bit_generator.random_raw())
    return (u >> 11) * (1.0 / (1 << 53))


def fisher_yates(arr: np.ndarray, rng: np.random.Generator) -> None:
    """In-place Fisher-Yates matching the C engine's bounded draw."""
    for i in range(len(arr) - 1, 0, -1):
        u = rng_uniform01(rng)
        j = int(floor(u * (i + 1)))
        if j > i:
            j = i
        if j < 0:
            j = 0
        arr[i], arr[j] = arr[j], arr[i]


# ---------------------------------------------------------------------------
# Splitter implementations (must match the C engines bit-for-bit).
# ---------------------------------------------------------------------------

def split_kennard_stone(X: np.ndarray, test_size: float) -> tuple[np.ndarray, np.ndarray]:
    n_train, _ = compute_train_test_count(X.shape[0], test_size)
    D = euclidean_distance_matrix(X)
    return max_min_selection(D, n_train)


def split_spxy(X: np.ndarray, y: np.ndarray, test_size: float) -> tuple[np.ndarray, np.ndarray]:
    n_train, _ = compute_train_test_count(X.shape[0], test_size)
    DX = normalise_distance(euclidean_distance_matrix(X))
    Y2 = y.reshape(-1, 1) if y.ndim == 1 else y
    DY = normalise_distance(euclidean_distance_matrix(Y2))
    return max_min_selection(DX + DY, n_train)


def _compute_fold_distance(X: np.ndarray, y: np.ndarray | None, use_y: int) -> np.ndarray:
    D = normalise_distance(euclidean_distance_matrix(X))
    if use_y == 1:
        Y2 = y.reshape(-1, 1) if y.ndim == 1 else y
        D = D + normalise_distance(euclidean_distance_matrix(Y2))
    elif use_y == 2:
        # hamming: any(y[i, k] != y[j, k])
        Y2 = y.reshape(-1, 1) if y.ndim == 1 else y
        DY = np.any(Y2[:, None, :] != Y2[None, :, :], axis=2).astype(np.float64)
        D = D + DY
    return D


def assign_to_folds(D: np.ndarray, n_splits: int) -> np.ndarray:
    """Replicates c4a_split_spxy_fold_assign."""
    n = D.shape[0]
    fa = np.full(n, -1, dtype=np.int32)
    centroid = D.mean(axis=1)
    sorted_idx = np.argsort(centroid, kind="stable")
    init = sorted_idx[n - n_splits:]
    for k, s in enumerate(init):
        fa[int(s)] = k
    assigned = np.zeros(n, dtype=bool)
    for s in init:
        assigned[int(s)] = True
    target = n // n_splits
    max_size = target + (1 if n % n_splits > 0 else 0)
    fold_sizes = np.ones(n_splits, dtype=np.int32)
    fold_members = [[int(init[k])] for k in range(n_splits)]
    min_dist = np.empty((n_splits, n), dtype=np.float64)
    for k in range(n_splits):
        min_dist[k] = D[fold_members[k][0]]
    remaining = n - n_splits
    while remaining > 0:
        for k in range(n_splits):
            if remaining == 0:
                break
            if fold_sizes[k] >= max_size:
                continue
            best_r = -1
            best_d = -1.0
            for r in range(n):
                if assigned[r]:
                    continue
                if best_r == -1 or min_dist[k, r] > best_d:
                    best_r = r
                    best_d = min_dist[k, r]
            if best_r < 0:
                break
            fa[best_r] = k
            fold_members[k].append(best_r)
            fold_sizes[k] += 1
            assigned[best_r] = True
            for r in range(n):
                if assigned[r]:
                    continue
                nd = D[best_r, r]
                if nd < min_dist[k, r]:
                    min_dist[k, r] = nd
            remaining -= 1
        if not any(fold_sizes[k] < max_size for k in range(n_splits)):
            break
    return fa


def split_spxy_fold(X: np.ndarray, y: np.ndarray, n_splits: int,
                     y_metric: int, fold_idx: int) -> tuple[np.ndarray, np.ndarray]:
    D = _compute_fold_distance(X, y, y_metric)
    fa = assign_to_folds(D, n_splits)
    test = np.where(fa == fold_idx)[0].astype(np.int64)
    train = np.where(fa != fold_idx)[0].astype(np.int64)
    return train, test


def split_spxy_g_fold(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                       n_splits: int, y_metric: int, aggregation: int,
                       fold_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Mirrors c4a_split_spxy_g_fold_assign."""
    n = X.shape[0]
    unique_groups: list[int] = []
    sample_to_group = np.zeros(n, dtype=np.int64)
    for i in range(n):
        g = int(groups[i])
        if g not in unique_groups:
            unique_groups.append(g)
        sample_to_group[i] = unique_groups.index(g)
    n_groups = len(unique_groups)
    X_rep = np.zeros((n_groups, X.shape[1]), dtype=np.float64)
    y_arr = y.reshape(-1, 1) if y.ndim == 1 else y
    Y_rep: np.ndarray | None = None
    if y_metric != 0:
        Y_rep = np.zeros((n_groups, y_arr.shape[1]), dtype=np.float64)
    for g in range(n_groups):
        idx = np.where(sample_to_group == g)[0]
        Xg = X[idx]
        if aggregation == 0:
            X_rep[g] = Xg.mean(axis=0)
        else:
            X_rep[g] = np.median(Xg, axis=0)
        if y_metric != 0 and Y_rep is not None:
            yg = y_arr[idx]
            if aggregation == 0:
                Y_rep[g] = yg.mean(axis=0)
            else:
                Y_rep[g] = np.median(yg, axis=0)
    D = _compute_fold_distance(X_rep, Y_rep, y_metric)
    group_fa = assign_to_folds(D, n_splits)
    fa = group_fa[sample_to_group]
    test = np.where(fa == fold_idx)[0].astype(np.int64)
    train = np.where(fa != fold_idx)[0].astype(np.int64)
    return train, test


def _kmeans_sq_dist(row: np.ndarray, cent: np.ndarray) -> float:
    """Bit-exact mirror of the C engine's sq_dist (sequential accumulation)."""
    s = 0.0
    n = row.shape[0]
    for kk in range(n):
        diff = row[kk] - cent[kk]
        s += diff * diff
    return s


def split_kmeans(X: np.ndarray, test_size: float, seed: int,
                  max_iter: int) -> tuple[np.ndarray, np.ndarray]:
    n_train, _ = compute_train_test_count(X.shape[0], test_size)
    k = n_train
    rng = np.random.default_rng(seed)
    n, d = X.shape

    # k-means++ init (row-order accumulation, matches the C engine).
    centroids = np.empty((k, d), dtype=np.float64)
    closest_sq = np.empty(n, dtype=np.float64)
    first = int(floor(rng_uniform01(rng) * n))
    if first >= n:
        first = n - 1
    for kk in range(d):
        centroids[0, kk] = X[first, kk]
    for i in range(n):
        closest_sq[i] = _kmeans_sq_dist(X[i], centroids[0])
    for c in range(1, k):
        total = 0.0
        for i in range(n):
            total += closest_sq[i]
        u = rng_uniform01(rng) * total
        acc = 0.0
        pick = n - 1
        for i in range(n):
            acc += closest_sq[i]
            if acc >= u:
                pick = i
                break
        for kk in range(d):
            centroids[c, kk] = X[pick, kk]
        for i in range(n):
            d2 = _kmeans_sq_dist(X[i], centroids[c])
            if d2 < closest_sq[i]:
                closest_sq[i] = d2

    # Lloyd iterations.
    assignments = -np.ones(n, dtype=np.int64)
    for _ in range(max_iter):
        changed = False
        new_assign = np.empty(n, dtype=np.int64)
        for i in range(n):
            best_c = 0
            best_d = _kmeans_sq_dist(X[i], centroids[0])
            for c in range(1, k):
                dd = _kmeans_sq_dist(X[i], centroids[c])
                if dd < best_d:
                    best_d = dd
                    best_c = c
            new_assign[i] = best_c
            if assignments[i] != best_c:
                changed = True
        assignments = new_assign
        # Recompute centroids: row-order accumulation, multiply-by-reciprocal.
        new_centroids = np.zeros_like(centroids)
        counts = np.zeros(k, dtype=np.int64)
        for i in range(n):
            c = int(assignments[i])
            for kk in range(d):
                new_centroids[c, kk] += X[i, kk]
            counts[c] += 1
        for c in range(k):
            if counts[c] > 0:
                inv = 1.0 / counts[c]
                for kk in range(d):
                    centroids[c, kk] = new_centroids[c, kk] * inv
        if not changed:
            break

    # Argmin per centroid (first on ties).
    picked = np.empty(k, dtype=np.int64)
    for c in range(k):
        best_i = 0
        best_d = _kmeans_sq_dist(X[0], centroids[c])
        for i in range(1, n):
            dd = _kmeans_sq_dist(X[i], centroids[c])
            if dd < best_d:
                best_d = dd
                best_i = i
        picked[c] = best_i

    in_train = np.zeros(n, dtype=bool)
    for p in picked:
        in_train[int(p)] = True
    train = np.where(in_train)[0].astype(np.int64)
    test  = np.where(~in_train)[0].astype(np.int64)
    return train, test


def split_kbins_stratified(y: np.ndarray, test_size: float, seed: int,
                            n_bins: int, strategy: int) -> tuple[np.ndarray, np.ndarray]:
    from sklearn import model_selection, preprocessing

    y2 = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    X_dummy = np.zeros((y2.shape[0], 1), dtype=np.float64)
    strategy_name = "uniform" if strategy == 0 else "quantile"
    kwargs = {"n_bins": n_bins, "encode": "ordinal", "strategy": strategy_name}
    try:
        binner = preprocessing.KBinsDiscretizer(**kwargs, subsample=None)
    except TypeError:
        binner = preprocessing.KBinsDiscretizer(**kwargs)
    y_disc = binner.fit_transform(y2).ravel()
    splitter = model_selection.StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=seed,
    )
    train, test = next(splitter.split(X_dummy, y_disc))
    return train.astype(np.int64), test.astype(np.int64)


def split_bsgk(y: np.ndarray, groups: np.ndarray, n_splits: int,
                n_bins: int, strategy: int, shuffle: int, seed: int,
                fold_idx: int) -> tuple[np.ndarray, np.ndarray]:
    n = y.shape[0]
    # Enumerate unique groups in first-seen order.
    unique_groups: list[int] = []
    sample_to_group = np.zeros(n, dtype=np.int64)
    for i in range(n):
        g = int(groups[i])
        if g not in unique_groups:
            unique_groups.append(g)
        sample_to_group[i] = unique_groups.index(g)
    n_groups = len(unique_groups)
    # Bin assignment.
    if strategy == 0:
        ymin = y.min()
        ymax = y.max()
        span = ymax - ymin
        if span <= 0.0:
            bin_of = np.zeros(n, dtype=np.int32)
        else:
            inv_w = n_bins / span
            bin_of = np.empty(n, dtype=np.int32)
            for i in range(n):
                b = int(floor((y[i] - ymin) * inv_w))
                if b < 0: b = 0
                if b >= n_bins: b = n_bins - 1
                bin_of[i] = b
    else:
        order = np.lexsort((np.arange(n), y))
        bin_of = np.empty(n, dtype=np.int32)
        for pos in range(n):
            b = (pos * n_bins) // n
            if b >= n_bins: b = n_bins - 1
            bin_of[int(order[pos])] = b
    # Group bin = bin of first-seen sample.
    group_bin = np.zeros(n_groups, dtype=np.int32)
    seen = np.zeros(n_groups, dtype=bool)
    for i in range(n):
        g = int(sample_to_group[i])
        if not seen[g]:
            group_bin[g] = bin_of[i]
            seen[g] = True
    # Build per-bin group lists.
    bin_groups: list[list[int]] = [[] for _ in range(n_bins)]
    for g in range(n_groups):
        bin_groups[int(group_bin[g])].append(g)
    if shuffle:
        rng = np.random.default_rng(seed)
        for b in range(n_bins):
            if len(bin_groups[b]) > 1:
                arr = np.asarray(bin_groups[b], dtype=np.int64)
                fisher_yates(arr, rng)
                bin_groups[b] = arr.tolist()
    # Round-robin assign groups to folds within each bin.
    group_fold = np.zeros(n_groups, dtype=np.int32)
    for b in range(n_bins):
        for p, g in enumerate(bin_groups[b]):
            group_fold[g] = p % n_splits
    fa = group_fold[sample_to_group]
    test = np.where(fa == fold_idx)[0].astype(np.int64)
    train = np.where(fa != fold_idx)[0].astype(np.int64)
    return train, test


def split_systematic_circular(y: np.ndarray, test_size: float,
                               seed: int) -> tuple[np.ndarray, np.ndarray]:
    n = y.shape[0]
    n_train, _ = compute_train_test_count(n, test_size)
    ordered = np.lexsort((np.arange(n), y))   # stable ascending by y
    rng = np.random.default_rng(seed)
    offset = int(floor(rng_uniform01(rng) * n))
    if offset < 0: offset = 0
    if offset >= n: offset = n - 1
    rotated = np.empty(n, dtype=np.int64)
    for i in range(n):
        src = (i - offset) % n
        rotated[i] = ordered[src]
    step = n / n_train
    picked_positions: list[int] = []
    picked_mask = np.zeros(n, dtype=bool)
    for k in range(n_train):
        p = int(np.rint(step * k))
        if p < 0: p = 0
        if p >= n: p = n - 1
        picked_positions.append(p)
        picked_mask[p] = True
    train_indices: list[int] = []
    consumed = np.zeros(n, dtype=bool)
    for p in picked_positions:
        if not consumed[p]:
            train_indices.append(int(rotated[p]))
            consumed[p] = True
    test_indices = [int(rotated[p]) for p in range(n) if not picked_mask[p]]
    return (np.asarray(train_indices, dtype=np.int64),
            np.asarray(test_indices,  dtype=np.int64))


def split_split_splitter(X: np.ndarray, test_size: float,
                          seed: int) -> tuple[np.ndarray, np.ndarray]:
    n, d = X.shape
    r = int(1.0 / test_size)
    # Preprocess: drop constant columns, z-score.
    keep_mask = np.array([not np.all(X[:, c] == X[0, c]) for c in range(d)])
    Xk = X[:, keep_mask]
    if Xk.shape[1] == 0:
        n_test = ceil(n / r)
        test = np.arange(n_test, dtype=np.int64)
        train = np.arange(n_test, n, dtype=np.int64)
        return train, test
    mean = Xk.mean(axis=0)
    sd = Xk.std(axis=0, ddof=0)
    Xz = (Xk - mean) / np.where(sd > 0, sd, 1.0)
    Xz = np.ascontiguousarray(Xz, dtype=np.float64)

    rng = np.random.default_rng(seed)
    current = int(floor(rng_uniform01(rng) * n))
    if current < 0: current = 0
    if current >= n: current = n - 1
    n_twin = ceil(n / r)
    twin = np.empty(n_twin, dtype=np.int64)
    active = np.ones(n, dtype=bool)
    count = 0
    while active.sum() >= r:
        active_idx = np.flatnonzero(active)
        diffs = Xz[active_idx] - Xz[current]
        sq = np.einsum("ij,ij->i", diffs, diffs)
        # Find r nearest, sorted by (distance, sample-index).
        # Use lexsort to break ties stably.
        order = np.lexsort((active_idx, sq))[:r]
        neighbours_local = order
        neighbours = active_idx[neighbours_local]
        twin[count] = neighbours[0]
        count += 1
        active[neighbours] = False
        if active.any():
            farthest = neighbours[-1]
            active_idx2 = np.flatnonzero(active)
            diffs2 = Xz[active_idx2] - Xz[farthest]
            sq2 = np.einsum("ij,ij->i", diffs2, diffs2)
            # nearest (tie -> lower index)
            order2 = np.lexsort((active_idx2, sq2))
            current = int(active_idx2[order2[0]])
    if count < n_twin:
        twin[count] = current
        count += 1
    twin = twin[:count]
    in_test = np.zeros(n, dtype=bool)
    for t in twin:
        in_test[int(t)] = True
    test = np.array([int(t) for t in twin], dtype=np.int64)
    train = np.where(~in_test)[0].astype(np.int64)
    return train, test


# ---------------------------------------------------------------------------
# Case lists.
# ---------------------------------------------------------------------------

def kennard_stone_cases(X: np.ndarray):
    return [
        ("ts_0_25", {"test_size": 0.25}, lambda: split_kennard_stone(X, 0.25)),
        ("ts_0_33", {"test_size": 1.0 / 3.0}, lambda: split_kennard_stone(X, 1.0 / 3.0)),
        ("ts_0_50", {"test_size": 0.5}, lambda: split_kennard_stone(X, 0.5)),
    ]


def spxy_cases(X: np.ndarray, y: np.ndarray):
    return [
        ("ts_0_25", {"test_size": 0.25}, lambda: split_spxy(X, y, 0.25)),
        ("ts_0_33", {"test_size": 1.0 / 3.0}, lambda: split_spxy(X, y, 1.0 / 3.0)),
    ]


def spxy_fold_cases(X: np.ndarray, y: np.ndarray):
    cases = []
    for n_splits in (3, 5):
        for fold_idx in (0, n_splits - 1):
            cases.append((
                f"k{n_splits}_f{fold_idx}",
                {"n_splits": n_splits, "y_metric": 1, "fold_idx": fold_idx},
                lambda ns=n_splits, fi=fold_idx: split_spxy_fold(X, y, ns, 1, fi),
            ))
    return cases


def spxy_g_fold_cases(X: np.ndarray, y: np.ndarray):
    # 10 groups of size 6.
    groups = np.repeat(np.arange(10), 6).astype(np.int64)
    cases = [
        ("k5_mean_f0",
         {"n_splits": 5, "y_metric": 1, "aggregation": 0, "fold_idx": 0,
          "groups": groups.tolist()},
         lambda: split_spxy_g_fold(X, y, groups, 5, 1, 0, 0)),
        ("k5_mean_f2",
         {"n_splits": 5, "y_metric": 1, "aggregation": 0, "fold_idx": 2,
          "groups": groups.tolist()},
         lambda: split_spxy_g_fold(X, y, groups, 5, 1, 0, 2)),
        ("k2_median_f1",
         {"n_splits": 2, "y_metric": 1, "aggregation": 1, "fold_idx": 1,
          "groups": groups.tolist()},
         lambda: split_spxy_g_fold(X, y, groups, 2, 1, 1, 1)),
    ]
    return cases


def kmeans_cases(X: np.ndarray):
    return [
        ("seed42_ts_0_25",
         {"test_size": 0.25, "seed": 42, "max_iter": 50},
         lambda: split_kmeans(X, 0.25, 42, 50)),
        ("seed7_ts_0_33",
         {"test_size": 1.0 / 3.0, "seed": 7, "max_iter": 50},
         lambda: split_kmeans(X, 1.0 / 3.0, 7, 50)),
    ]


def kbins_stratified_cases(y: np.ndarray):
    return [
        ("uniform_b5_s7",
         {"test_size": 0.25, "seed": 7, "n_bins": 5, "strategy": 0},
         lambda: split_kbins_stratified(y, 0.25, 7, 5, 0)),
        ("quantile_b4_s11",
         {"test_size": 0.3, "seed": 11, "n_bins": 4, "strategy": 1},
         lambda: split_kbins_stratified(y, 0.3, 11, 4, 1)),
    ]


def bsgk_cases(y: np.ndarray):
    groups = np.repeat(np.arange(15), 4).astype(np.int64)
    cases = []
    for fold_idx in (0, 2):
        cases.append((
            f"k5_b4_uniform_shuffle_f{fold_idx}",
            {"n_splits": 5, "n_bins": 4, "strategy": 0,
             "shuffle": 1, "seed": 13,
             "fold_idx": fold_idx, "groups": groups.tolist()},
            lambda fi=fold_idx: split_bsgk(y, groups, 5, 4, 0, 1, 13, fi),
        ))
    cases.append((
        "k3_b3_quantile_noshuffle_f0",
        {"n_splits": 3, "n_bins": 3, "strategy": 1,
         "shuffle": 0, "seed": 0,
         "fold_idx": 0, "groups": groups.tolist()},
        lambda: split_bsgk(y, groups, 3, 3, 1, 0, 0, 0),
    ))
    return cases


def systematic_circular_cases(y: np.ndarray):
    return [
        ("ts_0_25_s42",
         {"test_size": 0.25, "seed": 42},
         lambda: split_systematic_circular(y, 0.25, 42)),
        ("ts_0_33_s7",
         {"test_size": 1.0 / 3.0, "seed": 7},
         lambda: split_systematic_circular(y, 1.0 / 3.0, 7)),
    ]


def split_splitter_cases(X: np.ndarray):
    return [
        ("ts_0_25_s7",
         {"test_size": 0.25, "seed": 7},
         lambda: split_split_splitter(X, 0.25, 7)),
        ("ts_0_20_s42",
         {"test_size": 0.20, "seed": 42},
         lambda: split_split_splitter(X, 0.20, 42)),
    ]


# ---------------------------------------------------------------------------
# Fixture writer.
# ---------------------------------------------------------------------------

def write_fixture(name: str, X: np.ndarray, y: np.ndarray | None,
                  cases: list[tuple[str, dict, Callable[[], tuple[np.ndarray, np.ndarray]]]],
                  out_dir: Path, embed_y: bool) -> None:
    fixture: dict[str, Any] = {
        "format": f"c4a_split_{name}_v1",
        "numpy_version": np.__version__,
        "encoding": "ieee754_binary64_be_hex",
        "rows": int(X.shape[0]),
        "cols": int(X.shape[1]),
        "input_hex": array_to_hex(X),
        "cases": [],
    }
    if embed_y and y is not None:
        yc = y.reshape(-1, 1)
        fixture["fit_rows"] = int(yc.shape[0])
        fixture["fit_cols"] = int(yc.shape[1])
        fixture["fit_input_hex"] = array_to_hex(yc)
    for case_name, params, fn in cases:
        train_idx, test_idx = fn()
        params_with_idx = dict(params)
        params_with_idx["train_idx"] = train_idx.tolist()
        params_with_idx["test_idx"]  = test_idx.tolist()
        fixture["cases"].append({
            "name": case_name,
            "params": params_with_idx,
            "output_rows": 0,
            "output_cols": 0,
            "output_hex": [],
        })
    out = out_dir / f"split_{name}_v1.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(fixture, f, indent=None)
        f.write("\n")
    print(f"wrote {out} ({out.stat().st_size:,} bytes, {len(cases)} cases)")


def main() -> None:
    out_dir = REPO_ROOT / "parity" / "fixtures"
    out_dir.mkdir(parents=True, exist_ok=True)
    X, y = synthesize_dataset()
    print(f"X shape {X.shape}, y shape {y.shape}, y range [{y.min():.4f}, {y.max():.4f}]")

    # KennardStone: X only.
    write_fixture("kennard_stone", X, None, kennard_stone_cases(X), out_dir, embed_y=False)
    # SPXY: X + y.
    write_fixture("spxy", X, y, spxy_cases(X, y), out_dir, embed_y=True)
    write_fixture("spxy_fold", X, y, spxy_fold_cases(X, y), out_dir, embed_y=True)
    write_fixture("spxy_g_fold", X, y, spxy_g_fold_cases(X, y), out_dir, embed_y=True)
    write_fixture("kmeans", X, None, kmeans_cases(X), out_dir, embed_y=False)
    # KBins / SystematicCircular: y only. We still ship X as a placeholder
    # (1 x 1 column of zeros to keep the fixture parser happy).
    placeholder_X = np.zeros((ROWS, 1), dtype=np.float64)
    write_fixture("kbins_stratified", placeholder_X, y, kbins_stratified_cases(y),
                  out_dir, embed_y=True)
    write_fixture("binned_strat_group_kfold", placeholder_X, y, bsgk_cases(y),
                  out_dir, embed_y=True)
    write_fixture("systematic_circular", placeholder_X, y,
                  systematic_circular_cases(y), out_dir, embed_y=True)
    write_fixture("split_splitter", X, None, split_splitter_cases(X), out_dir,
                  embed_y=False)


if __name__ == "__main__":
    main()
