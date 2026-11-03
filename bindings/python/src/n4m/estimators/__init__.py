# SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
"""n4m.estimators role package."""

from . import (
    classification,
    multiblock,
    regression,
    survival,
)

__all__ = [
    "classification",
    "multiblock",
    "regression",
    "survival",
]
