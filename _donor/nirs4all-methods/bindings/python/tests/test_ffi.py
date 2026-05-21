# SPDX-License-Identifier: CECILL-2.1
"""Smoke tests for the ctypes Tier-1 layer."""
from __future__ import annotations

import ctypes

import numpy as np
import pytest


def test_library_loads():
    """Loading the package binds all declared ABI symbols."""
    from n4m._ffi import lib
    from n4m._ffi_decls import SYMBOLS

    # All declared symbols must be present on the loaded library.
    missing = [name for name, _, _ in SYMBOLS if not hasattr(lib, name)]
    assert not missing, f"Missing symbols: {missing[:5]} (and {len(missing) - 5} more)"
    assert len(SYMBOLS) == 498


def test_abi_version_matches():
    """The header version reported by libn4m matches our constants."""
    from n4m._ffi import (
        ABI_VERSION_MAJOR,
        ABI_VERSION_MINOR,
        ABI_VERSION_PATCH,
        lib,
    )

    assert lib.n4m_get_abi_version_major() == ABI_VERSION_MAJOR
    assert lib.n4m_get_abi_version_minor() == ABI_VERSION_MINOR
    assert lib.n4m_get_abi_version_patch() == ABI_VERSION_PATCH


def test_public_version_api():
    """Top-level version helpers expose package and ABI versions."""
    import n4m

    assert n4m.__version__ == "0.1.0"
    assert n4m.abi_version() == (1, 9, 0)
    assert n4m.version().startswith("0.1.0+abi.1.9.0")


def test_status_to_string():
    """n4m_status_to_string returns a static C string."""
    from n4m._ffi import lib
    from n4m._types import Status

    s = ctypes.c_char_p(lib.n4m_status_to_string(Status.OK)).value
    assert s is not None
    # Should contain "ok" (case-insensitive) for status code 0.
    assert b"ok" in s.lower()


def test_context_lifecycle():
    """A context can be created and destroyed without leaks."""
    from n4m._context import Context

    ctx = Context()
    assert ctx.handle.value
    ctx.set_seed(42)
    ctx.close()


def test_pcg64_lifecycle():
    """The PCG64 RNG wrapper can be created and destroyed."""
    from n4m._rng import PCG64

    rng = PCG64(seed=20260518)
    assert rng.handle.value
    rng.close()


def test_matrix_view_round_trip():
    """numpy_to_view produces a valid n4m_matrix_view_t."""
    from n4m._ffi import lib
    from n4m._matrix import numpy_to_view

    X = np.arange(20, dtype=np.float64).reshape(4, 5)
    view = numpy_to_view(X)
    status = lib.n4m_matrix_view_validate(ctypes.byref(view))
    assert status == 0, f"matrix_view_validate failed with status {status}"
    assert view.rows == 4
    assert view.cols == 5
    assert view.row_stride == 5
    assert view.col_stride == 1


# The 32 cross-binding method IDs maintained by the benchmark orchestrator.
# Each entry maps a method id -> the corresponding ``n4m.python`` callable name.
_CROSS_BINDING_PYTHON_NAMES: dict[str, str] = {
    "log_transform": "log_transform",
    "derivate": "derivate",
    "wavelet_pca": "wavelet_pca",
    "wavelet_svd": "wavelet_svd",
    "aug_gaussian_noise": "aug_gaussian_noise",
    "aug_multiplicative_noise": "aug_multiplicative_noise",
    "aug_spike_noise": "aug_spike_noise",
    "aug_hetero_noise": "aug_hetero_noise",
    "aug_linear_drift": "aug_linear_drift",
    "aug_poly_drift": "aug_poly_drift",
    "aug_path_length": "aug_path_length",
    "aug_wavelength_spectral": "aug_wavelength_spectral",
    "aug_mixup": "aug_mixup",
    "aug_local_mixup": "aug_local_mixup",
    "aug_scatter_sim": "aug_scatter_sim",
    "aug_particle_size": "aug_particle_size",
    "aug_emsc_distort": "aug_emsc_distort",
    "aug_batch_effect": "aug_batch_effect",
    "aug_instrument_broaden": "aug_instrument_broaden",
    "aug_dead_band": "aug_dead_band",
    "aug_temperature": "aug_temperature",
    "aug_moisture": "aug_moisture",
    "high_leverage": "high_leverage_filter",
    "spectral_quality": "spectral_quality_filter",
    "composite_filter": "composite_filter",
    "hotelling_t2": "hotelling_t2",
    "q_residuals": "q_residuals",
    "nirs_metrics": "nirs_metrics",
    "signal_type_detector": "signal_type_detector",
    "transfer_metrics": "transfer_metrics",
    "rng_pcg64": "rng_pcg64",
}


def test_cross_binding_methods_present_in_python_surface():
    """All cross-binding method IDs must resolve on ``n4m.python``."""
    from n4m import python as n4m_python

    missing = [
        mid
        for mid, name in _CROSS_BINDING_PYTHON_NAMES.items()
        if not callable(getattr(n4m_python, name, None))
    ]
    assert not missing, f"missing n4m.python callables: {missing}"
    assert len(_CROSS_BINDING_PYTHON_NAMES) == 31


@pytest.mark.parametrize(
    "method_id",
    [
        "log_transform",
        "derivate",
        "aug_gaussian_noise",
        "aug_wavelength_spectral",
        "high_leverage",
        "spectral_quality",
        "composite_filter",
        "nirs_metrics",
        "signal_type_detector",
        "transfer_metrics",
        "rng_pcg64",
    ],
)
def test_cross_binding_python_surface_runs_smoke(method_id):
    """Execute a representative cheap subset of the 32 cross-binding methods."""
    from n4m import python as n4m_python

    rng = np.random.default_rng(2026)
    x = rng.uniform(0.1, 0.9, size=(40, 24)).astype(np.float64)
    wl = np.linspace(1100.0, 1600.0, 24)
    func = getattr(n4m_python, _CROSS_BINDING_PYTHON_NAMES[method_id])

    if method_id == "nirs_metrics":
        y = np.arange(8, dtype=np.float64)
        out = func(y, y + 0.05)
        assert out.shape == (8,)
        assert np.isfinite(out).all()
        return
    if method_id == "transfer_metrics":
        out = func(x, x, n_components=3, k_neighbors=4, seed=1)
        assert out.ndim == 1
        return
    if method_id == "rng_pcg64":
        out = func(seed=1, n=64)
        assert out.shape == (64,)
        return
    if method_id == "signal_type_detector":
        out = func(x, wavelengths=wl)
        assert out.shape == (2,)
        return
    if method_id in {"high_leverage", "spectral_quality", "composite_filter"}:
        mask = func(x)
        assert mask.shape == (x.shape[0],)
        return
    if method_id == "aug_wavelength_spectral":
        out = func(x, wavelengths=wl, seed=1)
    elif method_id == "aug_gaussian_noise":
        out = func(x, seed=1)
    elif method_id == "derivate":
        out = func(x, order=1)
        assert out.shape[1] == x.shape[1] - 1
        return
    else:
        out = func(x)
    assert out.shape[0] == x.shape[0]
    assert np.isfinite(out).all()
