# SPDX-License-Identifier: CECILL-2.1

from n4m._impl import (
    IntervalGenerator,
)
from n4m._impl import native as _native

interval_generator = _native.interval_generator

__all__ = [
    "IntervalGenerator",
    "interval_generator",
]
