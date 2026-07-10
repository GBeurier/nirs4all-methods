# SPDX-License-Identifier: CECILL-2.1
"""External / independent references for HPO parity.

Two kinds of check, matching the tiers used across the rest of the project:

- **Tier A (bit-exact external):** the Sobol sampler must equal
  ``scipy.stats.qmc.Sobol(scramble=False)``.
- **Decision-level self-reference:** pure-Python reimplementations of the
  documented pruner rules (median / ASHA / Hyperband). These are a SECOND,
  independent implementation of the same spec — if the native pruner and this
  reference disagree on a canned rung history, one of them is wrong. (Optuna is
  intentionally NOT used as a hard gate: its pruners share the intent but not the
  exact peer-set/percentile conventions, so agreement is only approximate.)
"""
from __future__ import annotations

from statistics import median as _median


def sobol_reference(dim: int, n: int):
    """First ``n`` points of the unscrambled Sobol sequence via scipy."""
    from scipy.stats import qmc  # local import: only needed for the Sobol cell

    return qmc.Sobol(d=dim, scramble=False).random(n)


# ---- pure-Python pruner rules (mirror cpp/src/core/optimization/pruners.cpp) ----

def median_should_prune(history: dict[int, list[tuple[int, float]]], trial_id: int,
                        step: int, score: float, min_peers: int,
                        maximize: bool = False) -> bool:
    """Vizier median stopping rule. ``history`` maps trial_id -> [(step, score)]."""
    peers = []
    for tid, rungs in history.items():
        if tid == trial_id:
            continue
        for st, sc in rungs:
            if st == step:
                peers.append(sc)
                break
    if len(peers) < min_peers:
        return False
    med = _median(peers)  # statistics.median == true 50th percentile
    return score < med if maximize else score > med


def asha_should_prune(history: dict[int, list[tuple[int, float]]], trial_id: int,
                      step: int, score: float, eta: int, maximize: bool = False) -> bool:
    """Asynchronous successive halving: survive iff in the top 1/eta at the rung."""
    peers = []
    for tid, rungs in history.items():
        if tid == trial_id:
            continue
        for st, sc in rungs:
            if st == step:
                peers.append(sc)
                break
    n = len(peers) + 1
    if n < eta:
        return False
    n_promote = max(1, n // eta)
    better = sum(1 for p in peers if (p > score if maximize else p < score))
    return better >= n_promote


def _rung_index(resource: int, eta: int) -> int:
    if resource < 1:
        return -1
    k, v = 0, resource
    while v > 1:
        if v % eta != 0:
            return -1
        v //= eta
        k += 1
    return k


def hyperband_should_prune(order: list[int], history: dict[int, list[tuple[int, float]]],
                           trial_id: int, step: int, score: float, eta: int,
                           max_resource: int, maximize: bool = False) -> bool:
    """Bracketed successive halving. ``order`` is the ask order of trial ids
    (defines the stable index → bracket assignment); ``max_resource`` 0 = derive.
    """
    k = _rung_index(step + 1, eta)
    if k < 0:
        return False
    if max_resource <= 0:
        raise ValueError("hyperband requires max_resource > 0")
    R = max_resource  # fixed budget → stable brackets (mirrors the native rule)
    k_max, v = 0, R
    while v >= eta:
        v //= eta
        k_max += 1
    if k > k_max:  # above the top rung → never prune
        return False
    n_brackets = k_max + 1
    idx = {tid: i for i, tid in enumerate(order)}
    s = idx[trial_id] % n_brackets
    if k < s:
        return False
    peers = []
    for tid, rungs in history.items():
        if tid == trial_id or idx.get(tid, -1) % n_brackets != s:
            continue
        for st, sc in rungs:
            if st == step:
                peers.append(sc)
                break
    n = len(peers) + 1
    if n < eta:
        return False
    n_promote = max(1, n // eta)
    better = sum(1 for p in peers if (p > score if maximize else p < score))
    return better >= n_promote
