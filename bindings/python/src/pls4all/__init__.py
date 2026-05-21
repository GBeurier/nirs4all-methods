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

from ._ffi import lib  # noqa: F401  — eagerly loads libn4m
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
    cppls_fit,
    weighted_pls_fit,
    robust_pls_fit,
    ridge_pls_fit,
    continuum_regression_fit,
    n_pls_fit,
    kernel_pls_fit,
    o2pls_fit,
    approximate_press_compute,
    pls_diagnostics_compute,
    sparse_pls_da_fit,
    group_sparse_pls_fit,
    fused_sparse_pls_fit,
    pds_fit,
    ds_fit,
    mir_pls_fit,
    missing_aware_nipals_fit,
    pls_glm_fit,
    pls_qda_fit,
    pls_cox_fit,
    bagging_pls_fit,
    gpr_pls_fit,
    boosting_pls_fit,
    random_subspace_pls_fit,
    so_pls_fit,
    on_pls_fit,
    rosa_fit,
    pls_monitoring_run,
    one_se_rule_compute,
    # §17 fit shims
    mb_pls_fit,
    lw_pls_fit,
    pls_lda_fit,
    pls_logistic_fit,
    aom_preprocess_fit,
    # §18 selector shims
    variable_select_rank,
    interval_select,
    stability_select,
    uve_select,
    spa_select,
    cars_select,
    random_frog_select,
    scars_select,
    ga_select,
    pso_select,
    vissa_select,
    shaving_select,
    bve_select,
    t2_select,
    wvc_select,
    wvc_threshold_select,
    emcuve_select,
    randomization_select,
    bipls_select,
    sipls_select,
    rep_select,
    ipw_select,
    st_select,
    # §19 Phase 50+ numerical methods
    ecr_fit,
    iriv_select,
    irf_select,
    vip_spa_select,
)


def version() -> str:
    """Return the runtime library version string, e.g. 'X.Y.Z+abi.A.B.C'."""
    return lib.n4m_get_version_string().decode("utf-8")


def abi_version() -> tuple[int, int, int]:
    return (
        int(lib.n4m_get_abi_version_major()),
        int(lib.n4m_get_abi_version_minor()),
        int(lib.n4m_get_abi_version_patch()),
    )


def build_info() -> str:
    return lib.n4m_get_build_info().decode("utf-8")


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
    "cppls_fit",
    "weighted_pls_fit",
    "robust_pls_fit",
    "ridge_pls_fit",
    "continuum_regression_fit",
    "n_pls_fit",
    "kernel_pls_fit",
    "o2pls_fit",
    "approximate_press_compute",
    "pls_diagnostics_compute",
    "sparse_pls_da_fit",
    "group_sparse_pls_fit",
    "fused_sparse_pls_fit",
    "pds_fit",
    "ds_fit",
    "mir_pls_fit",
    "missing_aware_nipals_fit",
    "pls_glm_fit",
    "pls_qda_fit",
    "pls_cox_fit",
    "bagging_pls_fit",
    "gpr_pls_fit",
    "boosting_pls_fit",
    "random_subspace_pls_fit",
    "so_pls_fit",
    "on_pls_fit",
    "rosa_fit",
    "pls_monitoring_run",
    "one_se_rule_compute",
    # §17 fit shims
    "mb_pls_fit",
    "lw_pls_fit",
    "pls_lda_fit",
    "pls_logistic_fit",
    "aom_preprocess_fit",
    # §18 selector shims
    "variable_select_rank",
    "interval_select",
    "stability_select",
    "uve_select",
    "spa_select",
    "cars_select",
    "random_frog_select",
    "scars_select",
    "ga_select",
    "pso_select",
    "vissa_select",
    "shaving_select",
    "bve_select",
    "t2_select",
    "wvc_select",
    "wvc_threshold_select",
    "emcuve_select",
    "randomization_select",
    "bipls_select",
    "sipls_select",
    "rep_select",
    "ipw_select",
    "st_select",
    # §19 Phase 50+ numerical methods
    "ecr_fit",
    "iriv_select",
    "irf_select",
    "vip_spa_select",
    "version",
    "abi_version",
    "build_info",
]

# Derive __version__ from the loaded native library so it stays in sync with
# the C ABI version_string without any manual maintenance. The expected shape
# of version() is "X.Y.Z+abi.A.B.C" — we strip the "+abi.*" suffix.
__version__ = version().split("+", 1)[0]
