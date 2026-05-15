"""pls4all — Python binding for the pls4all C ABI.

Current scope: version / status / dtype / backend introspection, a Context
and a Config wrapper, an OperatorBank wrapper, a ValidationPlan wrapper, and
the AOM global / POP per-component selection result handles
(`aom_global_select`, `aom_per_component_select`). Full Python model
wrappers (NIPALS, orthogonal-scores, SIMPLS, kernel, wide-kernel, SVD,
power, randomized-SVD, PLSCanonical, PLSSVD, PLS-DA, OPLS/OPLS-DA, PCR and
the cross-validation / metrics / variable-selection surface) land in
Phase 2 on top of the live C core.
"""

from ._ffi import lib  # noqa: F401  — eagerly loads libp4a
from ._errors import Pls4allError
from ._types import Algorithm, Backend, Deflation, Dtype, Solver, Status
from ._context import Context
from ._config import Config
from ._aom import (
    AomGlobalResult,
    AomPerComponentResult,
    OperatorBank,
    OperatorKind,
    ValidationPlan,
    aom_global_select,
    aom_per_component_select,
)
from ._model import Model, ModelArrayKind
from ._methods import (
    MethodResult,
    sparse_simpls_fit,
    di_pls_fit,
    recursive_pls_run,
)


def version() -> str:
    """Return the runtime library version string, e.g. '0.58.0+abi.1.0.0'."""
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
    "OperatorKind",
    "OperatorBank",
    "ValidationPlan",
    "AomGlobalResult",
    "AomPerComponentResult",
    "aom_global_select",
    "aom_per_component_select",
    "Model",
    "ModelArrayKind",
    "MethodResult",
    "sparse_simpls_fit",
    "di_pls_fit",
    "recursive_pls_run",
    "version",
    "abi_version",
    "build_info",
]
__version__ = "0.71.0"
