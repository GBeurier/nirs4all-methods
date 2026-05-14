"""pls4all — Python binding for the pls4all C ABI.

Current scope is small: version / status / dtype / backend introspection,
a Context and a Config wrapper. Full Python model wrappers land in Phase 2
on top of the live NIPALS, orthogonal-scores, SIMPLS, kernel, wide-kernel,
SVD, power, randomized-SVD, PLSCanonical, PLS-DA and PCR C ABI.
"""

from ._ffi import lib  # noqa: F401  — eagerly loads libp4a
from ._errors import Pls4allError
from ._types import Algorithm, Backend, Deflation, Dtype, Solver, Status
from ._context import Context
from ._config import Config


def version() -> str:
    """Return the runtime library version string, e.g. '0.12.0+abi.1.0.0'."""
    return lib.p4a_get_version_string().decode("utf-8")


def abi_version() -> tuple[int, int, int]:
    return (
        int(lib.p4a_get_abi_version_major()),
        int(lib.p4a_get_abi_version_minor()),
        int(lib.p4a_get_abi_version_patch()),
    )


def build_info() -> str:
    return lib.p4a_get_build_info().decode("utf-8")


__all__ = [
    "Pls4allError",
    "Backend",
    "Dtype",
    "Deflation",
    "Algorithm",
    "Status",
    "Solver",
    "Context",
    "Config",
    "version",
    "abi_version",
    "build_info",
]
__version__ = "0.12.0"
