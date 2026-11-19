# SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
"""n4m.transform role package, with submodules loaded on first access."""

from importlib import import_module
from types import ModuleType

__all__ = [
    "alignment",
    "baseline",
    "orthogonalization",
    "resampling",
    "scaling",
    "scatter",
    "signal_conversion",
    "smoothing",
    "specialized",
    "wavelet",
]


def __getattr__(name: str) -> ModuleType:
    # A concurrent first import can already hold a sibling submodule's lock
    # while waiting for this package. Do not import siblings during package
    # initialization; let each requested module use its own import lock.
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
