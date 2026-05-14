"""pls4all — Python binding for the pls4all C ABI.

Phase 0 scope is small: version / status / dtype / backend introspection,
a Context and a Config wrapper. Full `fit / predict / transform` lands in
Phase 2 once Phase 1 supplies a real NIPALS implementation behind the
C ABI.
"""

from ._ffi import lib  # noqa: F401  — eagerly loads libp4a
from ._errors import Pls4allError
from ._types import Backend, Dtype, Status
from ._context import Context
from ._config import Config


def version() -> str:
    """Return the runtime library version string, e.g. '0.1.0+abi.1.0.0'."""
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
    "Status",
    "Context",
    "Config",
    "version",
    "abi_version",
    "build_info",
]
__version__ = "0.1.0"
