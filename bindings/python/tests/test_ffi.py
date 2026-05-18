# SPDX-License-Identifier: CECILL-2.1
"""Smoke tests for the ctypes Tier-1 layer."""
from __future__ import annotations

import ctypes

import numpy as np
import pytest


def test_library_loads():
    """Loading the package binds 402 ABI symbols."""
    from chemometrics4all._ffi import lib
    from chemometrics4all._ffi_decls import SYMBOLS

    # All 402 declared symbols must be present on the loaded library.
    missing = [name for name, _, _ in SYMBOLS if not hasattr(lib, name)]
    assert not missing, f"Missing symbols: {missing[:5]} (and {len(missing) - 5} more)"
    assert len(SYMBOLS) == 402


def test_abi_version_matches():
    """The header version reported by libc4a matches our constants."""
    from chemometrics4all._ffi import (
        ABI_VERSION_MAJOR,
        ABI_VERSION_MINOR,
        ABI_VERSION_PATCH,
        lib,
    )

    assert lib.c4a_get_abi_version_major() == ABI_VERSION_MAJOR
    assert lib.c4a_get_abi_version_minor() == ABI_VERSION_MINOR
    assert lib.c4a_get_abi_version_patch() == ABI_VERSION_PATCH


def test_status_to_string():
    """c4a_status_to_string returns a static C string."""
    from chemometrics4all._ffi import lib
    from chemometrics4all._types import Status

    s = ctypes.c_char_p(lib.c4a_status_to_string(Status.OK)).value
    assert s is not None
    # Should contain "ok" (case-insensitive) for status code 0.
    assert b"ok" in s.lower()


def test_context_lifecycle():
    """A context can be created and destroyed without leaks."""
    from chemometrics4all._context import Context

    ctx = Context()
    assert ctx.handle.value
    ctx.set_seed(42)
    ctx.close()


def test_pcg64_lifecycle():
    """The PCG64 RNG wrapper can be created and destroyed."""
    from chemometrics4all._rng import PCG64

    rng = PCG64(seed=20260518)
    assert rng.handle.value
    rng.close()


def test_matrix_view_round_trip():
    """numpy_to_view produces a valid c4a_matrix_view_t."""
    from chemometrics4all._ffi import lib
    from chemometrics4all._matrix import numpy_to_view

    X = np.arange(20, dtype=np.float64).reshape(4, 5)
    view = numpy_to_view(X)
    status = lib.c4a_matrix_view_validate(ctypes.byref(view))
    assert status == 0, f"matrix_view_validate failed with status {status}"
    assert view.rows == 4
    assert view.cols == 5
    assert view.row_stride == 5
    assert view.col_stride == 1
