# SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
"""Private scalar admission before host conversions can lose meaning."""

from __future__ import annotations

import numbers
import operator
from typing import Any

import numpy as np


def _int32_parameter(name: str, value: object) -> int:
    """Admit an integral scalar representable by the signed int32 ABI."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer, not a boolean")
    try:
        integer = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer index") from exc
    if not -(2**31) <= integer <= 2**31 - 1:
        raise ValueError(f"{name} must be in [INT32_MIN, INT32_MAX]")
    return integer


def _snv_parameters(
    with_mean: object, with_std: object, ddof: object
) -> tuple[bool, bool, int]:
    for name, value in (("with_mean", with_mean), ("with_std", with_std)):
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be a boolean")
    if isinstance(ddof, (bool, np.bool_)) or not isinstance(ddof, numbers.Integral):
        raise TypeError("ddof must be an integer, not a boolean")
    try:
        degree = operator.index(ddof)
    except TypeError as exc:
        # NumPy timedelta64 inherits Integral but is not an integer index.
        raise TypeError("ddof must be an integer index") from exc
    if not 0 <= degree <= 2**31 - 1:
        raise ValueError("ddof must be in [0, INT32_MAX]")
    return bool(with_mean), bool(with_std), degree


def _validate_snv_role(method: Any) -> None:
    if method._method_id == "preprocessing.scatter.snv":
        _snv_parameters(method.with_mean, method.with_std, method.ddof)
