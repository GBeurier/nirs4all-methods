# SPDX-License-Identifier: CECILL-2.1
"""Haar operator reference: wavelet_transform with family=haar, mode=periodization."""

from __future__ import annotations

import numpy as np

from .wavelet import wavelet_transform


def haar_transform(X: np.ndarray) -> np.ndarray:
    return wavelet_transform(X, family="haar", mode="periodization")
