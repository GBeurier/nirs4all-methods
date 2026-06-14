# SPDX-License-Identifier: CECILL-2.1
"""Python binding for libn4m.

The public surface is organised into role-based subpackages that mirror the
scikit-learn / DL pipeline conventions::

    from n4m.transform.scatter import SNV, MSC
    from n4m.estimators.regression.regularized import Ridge, RidgePLS
    from n4m.feature_selection.wrapper import CARS
    from n4m.model_selection.splitters import KennardStone
    from n4m.ensemble import AOMRidgeBlender

The top-level package only exposes metadata and a handful of shared helpers
(:func:`version`, :func:`abi_version`, :class:`Context`, :class:`N4MError`,
:class:`MatrixView`, :class:`PCG64`) plus the role subpackages. There are no
top-level class/function re-exports.
"""
import ctypes

from ._context import Context
from ._errors import N4MError
from ._ffi import (
    ABI_VERSION_MAJOR,
    ABI_VERSION_MINOR,
    ABI_VERSION_PATCH,
    ABI_VERSION_STRING,
    lib,
    library_path,
)
from ._rng import PCG64
from ._types import MatrixView

# Role subpackages (the public surface).
from . import (  # noqa: F401
    augmentation,
    compose,
    decomposition,
    domain_adaptation,
    ensemble,
    estimators,
    feature_selection,
    lowlevel,
    metrics,
    model_selection,
    outlier_detection,
    transform,
)


def version() -> str:
    """Return the runtime libn4m version string, e.g. ``2.0.0+abi.2.0.0``."""
    raw = ctypes.c_char_p(lib.n4m_get_version_string()).value
    if raw is None:  # pragma: no cover - defensive ABI guard
        raise RuntimeError("libn4m returned a null version string")
    return raw.decode("utf-8")


def abi_version() -> tuple[int, int, int]:
    """Return the loaded libn4m ABI version as ``(major, minor, patch)``."""
    return (
        int(lib.n4m_get_abi_version_major()),
        int(lib.n4m_get_abi_version_minor()),
        int(lib.n4m_get_abi_version_patch()),
    )


__all__ = [
    "ABI_VERSION_MAJOR",
    "ABI_VERSION_MINOR",
    "ABI_VERSION_PATCH",
    "ABI_VERSION_STRING",
    "Context",
    "MatrixView",
    "N4MError",
    "PCG64",
    "abi_version",
    "augmentation",
    "compose",
    "decomposition",
    "domain_adaptation",
    "ensemble",
    "estimators",
    "feature_selection",
    "library_path",
    "lowlevel",
    "metrics",
    "model_selection",
    "outlier_detection",
    "transform",
    "version",
    "__version__",
]

# Derive __version__ from the loaded native library so it stays in sync with the
# C ABI version_string (shape "X.Y.Z+abi.A.B.C") — strip the "+abi.*" suffix.
__version__ = version().split("+", 1)[0]
