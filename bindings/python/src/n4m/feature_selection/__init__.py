# SPDX-License-Identifier: CECILL-2.1
"""n4m.feature_selection role package."""

from . import (
    filter,
    interval,
    ranking,
    wrapper,
)
from .generic import SELECTOR_METHODS, Selector

__all__ = [
    "SELECTOR_METHODS",
    "Selector",
    "filter",
    "interval",
    "ranking",
    "wrapper",
]
