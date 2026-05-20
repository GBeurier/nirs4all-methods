# SPDX-License-Identifier: CECILL-2.1
"""ChannelDropout parity reference."""

from __future__ import annotations

import numpy as np


def channel_dropout(X: np.ndarray, rng: np.random.Generator, *,
                    dropout_prob: float, mode: str) -> np.ndarray:
    """`mode` ∈ {"zero", "interp"}."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    n_samples, n_features = X.shape
    mask = rng.random(size=X.shape) < dropout_prob
    out = X.copy()
    if mode == "zero":
        out[mask] = 0.0
        return out
    if mode != "interp":
        raise ValueError(f"unknown mode {mode}")
    for i in range(n_samples):
        dropped = np.where(mask[i])[0]
        if dropped.size == 0:
            continue
        kept = np.where(~mask[i])[0]
        if kept.size == 0:
            continue
        out[i, dropped] = np.interp(dropped, kept, X[i, kept])
    return out
