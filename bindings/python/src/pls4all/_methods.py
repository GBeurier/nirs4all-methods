"""Python wrappers for the universal method-result handle and the
per-method fit entry points exposed in p4a.h §16.

Each fit function returns a `MethodResult` object whose `.matrix(name)`,
`.vector(name)` and `.scalar(name)` accessors yield NumPy arrays /
scalars copied out of the C-owned storage. The underlying handle is
freed when the wrapper is closed or garbage-collected.
"""

from __future__ import annotations

import ctypes
from typing import Any

import numpy as np

from ._config import Config
from ._context import Context
from ._errors import Pls4allError
from ._ffi import MatrixView, lib
from ._model import _matrix_view, _as_float64_contiguous
from ._types import Status


def _check(status_int: int, ctx: Context | None = None) -> None:
    if status_int == Status.OK:
        return
    msg = None
    if ctx is not None:
        raw = lib.n4m_context_last_error(ctx.handle)
        if raw:
            msg = raw.decode("utf-8")
    raise Pls4allError(status_int, msg)


# ---- FFI signatures for the universal handle + fit entry points ----------

lib.n4m_method_result_destroy.restype = None
lib.n4m_method_result_destroy.argtypes = [ctypes.c_void_p]

lib.n4m_method_result_get_double_matrix.restype = ctypes.c_int
lib.n4m_method_result_get_double_matrix.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
    ctypes.POINTER(ctypes.c_int64),
    ctypes.POINTER(ctypes.c_int64),
]

lib.n4m_method_result_get_int_vector.restype = ctypes.c_int
lib.n4m_method_result_get_int_vector.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)),
    ctypes.POINTER(ctypes.c_int32),
]

lib.n4m_method_result_get_int64_vector.restype = ctypes.c_int
lib.n4m_method_result_get_int64_vector.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_int64)),
    ctypes.POINTER(ctypes.c_int64),
]

lib.n4m_method_result_get_scalar.restype = ctypes.c_int
lib.n4m_method_result_get_scalar.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_double),
]

lib.n4m_sparse_simpls_fit.restype = ctypes.c_int
lib.n4m_sparse_simpls_fit.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.c_double, ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_di_pls_fit.restype = ctypes.c_int
lib.n4m_di_pls_fit.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.POINTER(MatrixView),
    ctypes.c_double, ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_recursive_pls_run.restype = ctypes.c_int
lib.n4m_recursive_pls_run.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_cppls_fit.restype = ctypes.c_int
lib.n4m_cppls_fit.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.c_double, ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_weighted_pls_fit.restype = ctypes.c_int
lib.n4m_weighted_pls_fit.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.POINTER(ctypes.c_double), ctypes.c_int64,
    ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_robust_pls_fit.restype = ctypes.c_int
lib.n4m_robust_pls_fit.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.c_double, ctypes.c_int32,
    ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_ridge_pls_fit.restype = ctypes.c_int
lib.n4m_ridge_pls_fit.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.c_double, ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_continuum_regression_fit.restype = ctypes.c_int
lib.n4m_continuum_regression_fit.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.c_double, ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_ecr_fit.restype = ctypes.c_int
lib.n4m_ecr_fit.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.c_double, ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_n_pls_fit.restype = ctypes.c_int
lib.n4m_n_pls_fit.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView),
    ctypes.c_int32, ctypes.c_int32,
    ctypes.POINTER(MatrixView),
    ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_kernel_pls_fit.restype = ctypes.c_int
lib.n4m_kernel_pls_fit.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int32, ctypes.c_double, ctypes.c_double, ctypes.c_int32,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_o2pls_fit.restype = ctypes.c_int
lib.n4m_o2pls_fit.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_approximate_press_compute.restype = ctypes.c_int
lib.n4m_approximate_press_compute.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_pls_diagnostics_compute.restype = ctypes.c_int
lib.n4m_pls_diagnostics_compute.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_sparse_pls_da_fit.restype = ctypes.c_int
lib.n4m_sparse_pls_da_fit.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView),
    ctypes.POINTER(ctypes.c_int32), ctypes.c_int64,
    ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_group_sparse_pls_fit.restype = ctypes.c_int
lib.n4m_group_sparse_pls_fit.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.POINTER(ctypes.c_int32), ctypes.c_int64,
    ctypes.c_double, ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_fused_sparse_pls_fit.restype = ctypes.c_int
lib.n4m_fused_sparse_pls_fit.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.c_double, ctypes.c_double,
    ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_pds_fit.restype = ctypes.c_int
lib.n4m_pds_fit.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_ds_fit.restype = ctypes.c_int
lib.n4m_ds_fit.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_mir_pls_fit.restype = ctypes.c_int
lib.n4m_mir_pls_fit.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_missing_aware_nipals_fit.restype = ctypes.c_int
lib.n4m_missing_aware_nipals_fit.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_pls_glm_fit.restype = ctypes.c_int
lib.n4m_pls_glm_fit.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_bagging_pls_fit.restype = ctypes.c_int
lib.n4m_bagging_pls_fit.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.c_int32, ctypes.c_uint64,
    ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_gpr_pls_fit.restype = ctypes.c_int
lib.n4m_gpr_pls_fit.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.c_double, ctypes.c_double, ctypes.c_uint64,
    ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_pls_monitoring_run.restype = ctypes.c_int
lib.n4m_pls_monitoring_run.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.c_double, ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_one_se_rule_compute.restype = ctypes.c_int
lib.n4m_one_se_rule_compute.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int32, ctypes.c_int32,
    ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_so_pls_fit.restype = ctypes.c_int
lib.n4m_so_pls_fit.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.c_int32,
    ctypes.POINTER(MatrixView),
    ctypes.POINTER(ctypes.c_int32), ctypes.c_int64,
    ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_on_pls_fit.restype = ctypes.c_int
lib.n4m_on_pls_fit.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.c_int32,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int32), ctypes.c_int64,
    ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_rosa_fit.restype = ctypes.c_int
lib.n4m_rosa_fit.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.c_int32,
    ctypes.POINTER(MatrixView), ctypes.c_int32,
    ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_boosting_pls_fit.restype = ctypes.c_int
lib.n4m_boosting_pls_fit.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.c_int32, ctypes.c_double,
    ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_random_subspace_pls_fit.restype = ctypes.c_int
lib.n4m_random_subspace_pls_fit.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.c_int32, ctypes.c_int32, ctypes.c_uint64,
    ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_pls_qda_fit.restype = ctypes.c_int
lib.n4m_pls_qda_fit.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView),
    ctypes.POINTER(ctypes.c_int32), ctypes.c_int64,
    ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_pls_cox_fit.restype = ctypes.c_int
lib.n4m_pls_cox_fit.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView),
    ctypes.POINTER(ctypes.c_double), ctypes.c_int64,
    ctypes.POINTER(ctypes.c_int32), ctypes.c_int64,
    ctypes.POINTER(ctypes.c_void_p),
]

# ---- §17 fit shims (Phase 4r/4s/4p/4q/6a) -------------------------------

lib.n4m_mb_pls_fit.restype = ctypes.c_int
lib.n4m_mb_pls_fit.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.POINTER(ctypes.c_int64), ctypes.c_int64,
    ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_lw_pls_fit.restype = ctypes.c_int
lib.n4m_lw_pls_fit.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_pls_lda_fit.restype = ctypes.c_int
lib.n4m_pls_lda_fit.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView),
    ctypes.POINTER(ctypes.c_int32), ctypes.c_int64,
    ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_pls_logistic_fit.restype = ctypes.c_int
lib.n4m_pls_logistic_fit.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView),
    ctypes.POINTER(ctypes.c_int32), ctypes.c_int64,
    ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_aom_preprocess_fit.restype = ctypes.c_int
lib.n4m_aom_preprocess_fit.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.POINTER(ctypes.c_void_p),
]

# ---- §18 selector shims (Phase 5 variable selection) --------------------

lib.n4m_variable_select_rank.restype = ctypes.c_int
lib.n4m_variable_select_rank.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView),
    ctypes.c_int32, ctypes.c_int32,
    ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_interval_select.restype = ctypes.c_int
lib.n4m_interval_select.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.c_void_p,
    ctypes.c_int32, ctypes.c_int32,
    ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_stability_select.restype = ctypes.c_int
lib.n4m_stability_select.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_uve_select.restype = ctypes.c_int
lib.n4m_uve_select.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.c_void_p,
    ctypes.c_int32, ctypes.c_uint64,
    ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_spa_select.restype = ctypes.c_int
lib.n4m_spa_select.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_cars_select.restype = ctypes.c_int
lib.n4m_cars_select.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.c_void_p,
    ctypes.c_int32, ctypes.c_int32,
    ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_random_frog_select.restype = ctypes.c_int
lib.n4m_random_frog_select.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.c_void_p,
    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ctypes.c_int32, ctypes.c_int32, ctypes.c_uint64,
    ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_scars_select.restype = ctypes.c_int
lib.n4m_scars_select.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.c_void_p,
    ctypes.c_int32, ctypes.c_int32,
    ctypes.c_double, ctypes.c_uint64,
    ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_ga_select.restype = ctypes.c_int
lib.n4m_ga_select.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.c_void_p,
    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ctypes.c_double, ctypes.c_uint64,
    ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_pso_select.restype = ctypes.c_int
lib.n4m_pso_select.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.c_void_p,
    ctypes.c_int32, ctypes.c_int32,
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
    ctypes.c_uint64,
    ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_vissa_select.restype = ctypes.c_int
lib.n4m_vissa_select.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.c_void_p,
    ctypes.c_int32, ctypes.c_int32,
    ctypes.c_double, ctypes.c_double, ctypes.c_double,
    ctypes.c_uint64,
    ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_shaving_select.restype = ctypes.c_int
lib.n4m_shaving_select.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.c_void_p,
    ctypes.c_int32, ctypes.c_int32, ctypes.c_double,
    ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_bve_select.restype = ctypes.c_int
lib.n4m_bve_select.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.c_void_p,
    ctypes.c_int32, ctypes.c_int32,
    ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_iriv_select.restype = ctypes.c_int
lib.n4m_iriv_select.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.c_void_p,
    ctypes.c_int32, ctypes.c_uint64,
    ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_irf_select.restype = ctypes.c_int
lib.n4m_irf_select.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.c_void_p,
    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ctypes.c_uint64,
    ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_vip_spa_select.restype = ctypes.c_int
lib.n4m_vip_spa_select.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.c_double, ctypes.c_int32,
    ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_t2_select.restype = ctypes.c_int
lib.n4m_t2_select.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double), ctypes.c_int64,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_wvc_select.restype = ctypes.c_int
lib.n4m_wvc_select.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_wvc_threshold_select.restype = ctypes.c_int
lib.n4m_wvc_threshold_select.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.c_int32, ctypes.c_int32,
    ctypes.c_double, ctypes.c_double, ctypes.c_int32,
    ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_emcuve_select.restype = ctypes.c_int
lib.n4m_emcuve_select.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.c_void_p,
    ctypes.c_int32, ctypes.c_uint64,
    ctypes.c_int32, ctypes.c_double,
    ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_randomization_select.restype = ctypes.c_int
lib.n4m_randomization_select.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.c_int32, ctypes.c_uint64, ctypes.c_double,
    ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_bipls_select.restype = ctypes.c_int
lib.n4m_bipls_select.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.c_void_p,
    ctypes.c_int32, ctypes.c_int32,
    ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_sipls_select.restype = ctypes.c_int
lib.n4m_sipls_select.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.c_void_p,
    ctypes.c_int32, ctypes.c_int32,
    ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_rep_select.restype = ctypes.c_int
lib.n4m_rep_select.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.c_void_p,
    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_ipw_select.restype = ctypes.c_int
lib.n4m_ipw_select.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.c_void_p,
    ctypes.c_int32, ctypes.c_int32,
    ctypes.c_double, ctypes.c_double,
    ctypes.POINTER(ctypes.c_void_p),
]

lib.n4m_st_select.restype = ctypes.c_int
lib.n4m_st_select.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double), ctypes.c_int64,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_void_p),
]


class MethodResult:
    """Owning wrapper around `n4m_method_result_t`."""

    def __init__(self, handle: ctypes.c_void_p) -> None:
        self._h = handle

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __copy__(self):
        raise TypeError("MethodResult is not copyable")

    def __deepcopy__(self, _memo):
        raise TypeError("MethodResult is not copyable")

    def close(self) -> None:
        if self._h:
            lib.n4m_method_result_destroy(self._h)
            self._h = ctypes.c_void_p(0)

    def matrix(self, name: str) -> np.ndarray:
        data_ptr = ctypes.POINTER(ctypes.c_double)()
        rows = ctypes.c_int64(0)
        cols = ctypes.c_int64(0)
        _check(lib.n4m_method_result_get_double_matrix(
            self._h, name.encode("utf-8"),
            ctypes.byref(data_ptr), ctypes.byref(rows), ctypes.byref(cols),
        ))
        n = int(rows.value) * int(cols.value)
        if n == 0:
            return np.empty((int(rows.value), int(cols.value)),
                            dtype=np.float64)
        buf_type = ctypes.c_double * n
        buffer = buf_type.from_address(ctypes.addressof(data_ptr.contents))
        arr = np.frombuffer(buffer, dtype=np.float64, count=n)
        return np.array(arr.reshape(int(rows.value), int(cols.value)),
                        copy=True)

    def vector_int(self, name: str) -> np.ndarray:
        data_ptr = ctypes.POINTER(ctypes.c_int32)()
        size = ctypes.c_int32(0)
        _check(lib.n4m_method_result_get_int_vector(
            self._h, name.encode("utf-8"),
            ctypes.byref(data_ptr), ctypes.byref(size),
        ))
        n = int(size.value)
        if n == 0:
            return np.empty(0, dtype=np.int32)
        buf_type = ctypes.c_int32 * n
        buffer = buf_type.from_address(ctypes.addressof(data_ptr.contents))
        arr = np.frombuffer(buffer, dtype=np.int32, count=n)
        return np.array(arr, copy=True)

    def vector_int64(self, name: str) -> np.ndarray:
        data_ptr = ctypes.POINTER(ctypes.c_int64)()
        size = ctypes.c_int64(0)
        _check(lib.n4m_method_result_get_int64_vector(
            self._h, name.encode("utf-8"),
            ctypes.byref(data_ptr), ctypes.byref(size),
        ))
        n = int(size.value)
        if n == 0:
            return np.empty(0, dtype=np.int64)
        buf_type = ctypes.c_int64 * n
        buffer = buf_type.from_address(ctypes.addressof(data_ptr.contents))
        arr = np.frombuffer(buffer, dtype=np.int64, count=n)
        return np.array(arr, copy=True)

    def scalar(self, name: str) -> float:
        out = ctypes.c_double(0.0)
        _check(lib.n4m_method_result_get_scalar(
            self._h, name.encode("utf-8"), ctypes.byref(out),
        ))
        return float(out.value)


def _resolve_handle(out: ctypes.c_void_p, ctx: Context, fn_name: str) -> MethodResult:
    if not out:
        raise Pls4allError(int(Status.ERR_INTERNAL),
                           f"{fn_name} returned a NULL handle")
    try:
        return MethodResult(out)
    except BaseException:
        lib.n4m_method_result_destroy(out)
        raise


def sparse_simpls_fit(ctx: Context, cfg: Config,
                       X: Any, Y: Any,
                       sparsity_lambda: float) -> MethodResult:
    X_arr = _as_float64_contiguous(X)
    Y_arr = _as_float64_contiguous(Y)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    out = ctypes.c_void_p(0)
    status = lib.n4m_sparse_simpls_fit(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view), ctypes.byref(y_view),
        ctypes.c_double(float(sparsity_lambda)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_sparse_simpls_fit")


def di_pls_fit(ctx: Context, cfg: Config,
                X_source: Any, Y_source: Any, X_target: Any,
                di_lambda: float) -> MethodResult:
    Xs = _as_float64_contiguous(X_source)
    Ys = _as_float64_contiguous(Y_source)
    Xt = _as_float64_contiguous(X_target)
    xs_view = _matrix_view(Xs)
    ys_view = _matrix_view(Ys)
    xt_view = _matrix_view(Xt)
    out = ctypes.c_void_p(0)
    status = lib.n4m_di_pls_fit(
        ctx.handle, cfg.handle,
        ctypes.byref(xs_view), ctypes.byref(ys_view), ctypes.byref(xt_view),
        ctypes.c_double(float(di_lambda)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_di_pls_fit")


def recursive_pls_run(ctx: Context, cfg: Config,
                       X: Any, Y: Any,
                       window_size: int) -> MethodResult:
    X_arr = _as_float64_contiguous(X)
    Y_arr = _as_float64_contiguous(Y)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    out = ctypes.c_void_p(0)
    status = lib.n4m_recursive_pls_run(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view), ctypes.byref(y_view),
        ctypes.c_int32(int(window_size)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_recursive_pls_run")


def cppls_fit(ctx: Context, cfg: Config,
               X: Any, Y: Any,
               gamma: float) -> MethodResult:
    X_arr = _as_float64_contiguous(X)
    Y_arr = _as_float64_contiguous(Y)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    out = ctypes.c_void_p(0)
    status = lib.n4m_cppls_fit(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view), ctypes.byref(y_view),
        ctypes.c_double(float(gamma)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_cppls_fit")


def weighted_pls_fit(ctx: Context, cfg: Config,
                      X: Any, Y: Any,
                      sample_weights: Any) -> MethodResult:
    X_arr = _as_float64_contiguous(X)
    Y_arr = _as_float64_contiguous(Y)
    w_arr = np.ascontiguousarray(sample_weights, dtype=np.float64).reshape(-1)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    out = ctypes.c_void_p(0)
    status = lib.n4m_weighted_pls_fit(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view), ctypes.byref(y_view),
        w_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int64(int(w_arr.size)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_weighted_pls_fit")


def robust_pls_fit(ctx: Context, cfg: Config,
                    X: Any, Y: Any,
                    huber_k: float,
                    max_irls_iter: int = 5) -> MethodResult:
    X_arr = _as_float64_contiguous(X)
    Y_arr = _as_float64_contiguous(Y)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    out = ctypes.c_void_p(0)
    status = lib.n4m_robust_pls_fit(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view), ctypes.byref(y_view),
        ctypes.c_double(float(huber_k)),
        ctypes.c_int32(int(max_irls_iter)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_robust_pls_fit")


def ridge_pls_fit(ctx: Context, cfg: Config,
                   X: Any, Y: Any,
                   ridge_lambda: float) -> MethodResult:
    X_arr = _as_float64_contiguous(X)
    Y_arr = _as_float64_contiguous(Y)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    out = ctypes.c_void_p(0)
    status = lib.n4m_ridge_pls_fit(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view), ctypes.byref(y_view),
        ctypes.c_double(float(ridge_lambda)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_ridge_pls_fit")


def continuum_regression_fit(ctx: Context, cfg: Config,
                              X: Any, Y: Any,
                              tau: float) -> MethodResult:
    X_arr = _as_float64_contiguous(X)
    Y_arr = _as_float64_contiguous(Y)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    out = ctypes.c_void_p(0)
    status = lib.n4m_continuum_regression_fit(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view), ctypes.byref(y_view),
        ctypes.c_double(float(tau)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_continuum_regression_fit")


def n_pls_fit(ctx: Context, cfg: Config,
               X_flat: Any, mode_j: int, mode_k: int,
               Y: Any) -> MethodResult:
    X_arr = _as_float64_contiguous(X_flat)
    Y_arr = _as_float64_contiguous(Y)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    out = ctypes.c_void_p(0)
    status = lib.n4m_n_pls_fit(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view),
        ctypes.c_int32(int(mode_j)), ctypes.c_int32(int(mode_k)),
        ctypes.byref(y_view),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_n_pls_fit")


def kernel_pls_fit(ctx: Context, cfg: Config,
                    X: Any, Y: Any,
                    kernel_type: int,
                    gamma: float = 0.0,
                    coef0: float = 1.0,
                    degree: int = 3) -> MethodResult:
    X_arr = _as_float64_contiguous(X)
    Y_arr = _as_float64_contiguous(Y)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    out = ctypes.c_void_p(0)
    status = lib.n4m_kernel_pls_fit(
        ctx.handle, cfg.handle,
        ctypes.c_int32(int(kernel_type)),
        ctypes.c_double(float(gamma)), ctypes.c_double(float(coef0)),
        ctypes.c_int32(int(degree)),
        ctypes.byref(x_view), ctypes.byref(y_view),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_kernel_pls_fit")


def o2pls_fit(ctx: Context, cfg: Config,
               X: Any, Y: Any,
               n_predictive: int,
               n_x_orthogonal: int = 0,
               n_y_orthogonal: int = 0) -> MethodResult:
    X_arr = _as_float64_contiguous(X)
    Y_arr = _as_float64_contiguous(Y)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    out = ctypes.c_void_p(0)
    status = lib.n4m_o2pls_fit(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view), ctypes.byref(y_view),
        ctypes.c_int32(int(n_predictive)),
        ctypes.c_int32(int(n_x_orthogonal)),
        ctypes.c_int32(int(n_y_orthogonal)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_o2pls_fit")


def approximate_press_compute(ctx: Context, cfg: Config,
                                X: Any, Y: Any,
                                max_components: int) -> MethodResult:
    X_arr = _as_float64_contiguous(X)
    Y_arr = _as_float64_contiguous(Y)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    out = ctypes.c_void_p(0)
    status = lib.n4m_approximate_press_compute(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view), ctypes.byref(y_view),
        ctypes.c_int32(int(max_components)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_approximate_press_compute")


def sparse_pls_da_fit(ctx: Context, cfg: Config,
                        X: Any, y_labels: Any) -> MethodResult:
    X_arr = _as_float64_contiguous(X)
    labels = np.ascontiguousarray(y_labels, dtype=np.int32).reshape(-1)
    x_view = _matrix_view(X_arr)
    out = ctypes.c_void_p(0)
    status = lib.n4m_sparse_pls_da_fit(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view),
        labels.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int64(int(labels.size)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_sparse_pls_da_fit")


def group_sparse_pls_fit(ctx: Context, cfg: Config,
                          X: Any, Y: Any,
                          group_assignment: Any,
                          group_lambda: float) -> MethodResult:
    X_arr = _as_float64_contiguous(X)
    Y_arr = _as_float64_contiguous(Y)
    groups = np.ascontiguousarray(group_assignment, dtype=np.int32).reshape(-1)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    out = ctypes.c_void_p(0)
    status = lib.n4m_group_sparse_pls_fit(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view), ctypes.byref(y_view),
        groups.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int64(int(groups.size)),
        ctypes.c_double(float(group_lambda)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_group_sparse_pls_fit")


def fused_sparse_pls_fit(ctx: Context, cfg: Config,
                          X: Any, Y: Any,
                          l1_lambda: float,
                          fusion_lambda: float) -> MethodResult:
    X_arr = _as_float64_contiguous(X)
    Y_arr = _as_float64_contiguous(Y)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    out = ctypes.c_void_p(0)
    status = lib.n4m_fused_sparse_pls_fit(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view), ctypes.byref(y_view),
        ctypes.c_double(float(l1_lambda)),
        ctypes.c_double(float(fusion_lambda)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_fused_sparse_pls_fit")


def pls_monitoring_run(ctx: Context,
                         model: Any,
                         X_reference: Any,
                         X_monitor: Any,
                         alpha: float = 0.05) -> MethodResult:
    Xr = _as_float64_contiguous(X_reference)
    Xm = _as_float64_contiguous(X_monitor)
    xr_view = _matrix_view(Xr)
    xm_view = _matrix_view(Xm)
    model_handle = getattr(model, "handle", model)
    out = ctypes.c_void_p(0)
    status = lib.n4m_pls_monitoring_run(
        ctx.handle, model_handle,
        ctypes.byref(xr_view), ctypes.byref(xm_view),
        ctypes.c_double(float(alpha)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_pls_monitoring_run")


def one_se_rule_compute(ctx: Context,
                          fold_rmse_matrix: Any,
                          max_components: int | None = None,
                          n_folds: int | None = None) -> MethodResult:
    arr = np.ascontiguousarray(fold_rmse_matrix, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("fold_rmse_matrix must be 2-D")
    if max_components is None:
        max_components = int(arr.shape[0])
    if n_folds is None:
        n_folds = int(arr.shape[1])
    out = ctypes.c_void_p(0)
    status = lib.n4m_one_se_rule_compute(
        ctx.handle,
        arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int32(int(max_components)),
        ctypes.c_int32(int(n_folds)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_one_se_rule_compute")


def _blocks_to_views(blocks):
    views = (MatrixView * len(blocks))()
    refs = []
    for i, b in enumerate(blocks):
        arr = _as_float64_contiguous(b)
        refs.append(arr)
        v = _matrix_view(arr)
        views[i] = v
    return views, refs


def so_pls_fit(ctx: Context, cfg: Config,
                X_blocks: list,
                Y: Any,
                n_components_per_block: Any) -> MethodResult:
    views, _refs = _blocks_to_views(X_blocks)
    Y_arr = _as_float64_contiguous(Y)
    y_view = _matrix_view(Y_arr)
    comps = np.ascontiguousarray(n_components_per_block,
                                  dtype=np.int32).reshape(-1)
    out = ctypes.c_void_p(0)
    status = lib.n4m_so_pls_fit(
        ctx.handle, cfg.handle,
        views, ctypes.c_int32(len(X_blocks)),
        ctypes.byref(y_view),
        comps.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int64(int(comps.size)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_so_pls_fit")


def on_pls_fit(ctx: Context, cfg: Config,
                X_blocks: list,
                n_joint: int,
                n_unique_per_block: Any) -> MethodResult:
    views, _refs = _blocks_to_views(X_blocks)
    uniques = np.ascontiguousarray(n_unique_per_block,
                                    dtype=np.int32).reshape(-1)
    out = ctypes.c_void_p(0)
    status = lib.n4m_on_pls_fit(
        ctx.handle, cfg.handle,
        views, ctypes.c_int32(len(X_blocks)),
        ctypes.c_int32(int(n_joint)),
        uniques.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int64(int(uniques.size)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_on_pls_fit")


def rosa_fit(ctx: Context, cfg: Config,
              X_blocks: list,
              Y: Any,
              n_components: int) -> MethodResult:
    views, _refs = _blocks_to_views(X_blocks)
    Y_arr = _as_float64_contiguous(Y)
    y_view = _matrix_view(Y_arr)
    out = ctypes.c_void_p(0)
    status = lib.n4m_rosa_fit(
        ctx.handle, cfg.handle,
        views, ctypes.c_int32(len(X_blocks)),
        ctypes.byref(y_view), ctypes.c_int32(int(n_components)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_rosa_fit")


def bagging_pls_fit(ctx: Context, cfg: Config,
                     X: Any, Y: Any,
                     n_estimators: int,
                     seed: int = 0) -> MethodResult:
    X_arr = _as_float64_contiguous(X)
    Y_arr = _as_float64_contiguous(Y)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    out = ctypes.c_void_p(0)
    status = lib.n4m_bagging_pls_fit(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view), ctypes.byref(y_view),
        ctypes.c_int32(int(n_estimators)),
        ctypes.c_uint64(int(seed)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_bagging_pls_fit")


def gpr_pls_fit(ctx: Context, cfg: Config,
                X: Any, Y: Any,
                n_components: int,
                length_scale: float,
                noise_level: float,
                seed: int = 0) -> MethodResult:
    """GPR-on-PLS (§47).

    Two-stage regression: SIMPLS to obtain `n_components` latent
    components, then a Gaussian Process with RBF kernel on the training
    scores. `noise_level` is the diagonal noise **variance** (sigma^2,
    not sigma), matching sklearn `WhiteKernel(noise_level=...)`.
    `seed` is reserved for ABI symmetry; the fit is fully deterministic.
    """
    X_arr = _as_float64_contiguous(X)
    Y_arr = _as_float64_contiguous(Y)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    cfg.n_components = int(n_components)
    out = ctypes.c_void_p(0)
    status = lib.n4m_gpr_pls_fit(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view), ctypes.byref(y_view),
        ctypes.c_double(float(length_scale)),
        ctypes.c_double(float(noise_level)),
        ctypes.c_uint64(int(seed)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_gpr_pls_fit")


def boosting_pls_fit(ctx: Context, cfg: Config,
                      X: Any, Y: Any,
                      n_estimators: int,
                      learning_rate: float = 0.1) -> MethodResult:
    X_arr = _as_float64_contiguous(X)
    Y_arr = _as_float64_contiguous(Y)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    out = ctypes.c_void_p(0)
    status = lib.n4m_boosting_pls_fit(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view), ctypes.byref(y_view),
        ctypes.c_int32(int(n_estimators)),
        ctypes.c_double(float(learning_rate)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_boosting_pls_fit")


def random_subspace_pls_fit(ctx: Context, cfg: Config,
                              X: Any, Y: Any,
                              n_estimators: int,
                              features_per_subspace: int,
                              seed: int = 0) -> MethodResult:
    X_arr = _as_float64_contiguous(X)
    Y_arr = _as_float64_contiguous(Y)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    out = ctypes.c_void_p(0)
    status = lib.n4m_random_subspace_pls_fit(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view), ctypes.byref(y_view),
        ctypes.c_int32(int(n_estimators)),
        ctypes.c_int32(int(features_per_subspace)),
        ctypes.c_uint64(int(seed)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_random_subspace_pls_fit")


def pls_glm_fit(ctx: Context, cfg: Config,
                 X: Any, Y: Any,
                 poisson: bool = False) -> MethodResult:
    X_arr = _as_float64_contiguous(X)
    Y_arr = _as_float64_contiguous(Y)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    out = ctypes.c_void_p(0)
    status = lib.n4m_pls_glm_fit(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view), ctypes.byref(y_view),
        ctypes.c_int32(1 if poisson else 0),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_pls_glm_fit")


def pls_qda_fit(ctx: Context, cfg: Config,
                 X: Any, y_labels: Any) -> MethodResult:
    X_arr = _as_float64_contiguous(X)
    labels = np.ascontiguousarray(y_labels, dtype=np.int32).reshape(-1)
    x_view = _matrix_view(X_arr)
    out = ctypes.c_void_p(0)
    status = lib.n4m_pls_qda_fit(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view),
        labels.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int64(int(labels.size)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_pls_qda_fit")


def pls_cox_fit(ctx: Context, cfg: Config,
                 X: Any,
                 survival_times: Any,
                 event_indicators: Any) -> MethodResult:
    X_arr = _as_float64_contiguous(X)
    times = np.ascontiguousarray(survival_times, dtype=np.float64).reshape(-1)
    events = np.ascontiguousarray(event_indicators, dtype=np.int32).reshape(-1)
    x_view = _matrix_view(X_arr)
    out = ctypes.c_void_p(0)
    status = lib.n4m_pls_cox_fit(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view),
        times.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int64(int(times.size)),
        events.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int64(int(events.size)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_pls_cox_fit")


def pds_fit(ctx: Context, X_source: Any, X_target: Any,
              window_half_width: int) -> MethodResult:
    Xs = _as_float64_contiguous(X_source)
    Xt = _as_float64_contiguous(X_target)
    xs_view = _matrix_view(Xs)
    xt_view = _matrix_view(Xt)
    out = ctypes.c_void_p(0)
    status = lib.n4m_pds_fit(
        ctx.handle,
        ctypes.byref(xs_view), ctypes.byref(xt_view),
        ctypes.c_int32(int(window_half_width)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_pds_fit")


def ds_fit(ctx: Context, X_source: Any, X_target: Any) -> MethodResult:
    Xs = _as_float64_contiguous(X_source)
    Xt = _as_float64_contiguous(X_target)
    xs_view = _matrix_view(Xs)
    xt_view = _matrix_view(Xt)
    out = ctypes.c_void_p(0)
    status = lib.n4m_ds_fit(
        ctx.handle,
        ctypes.byref(xs_view), ctypes.byref(xt_view),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_ds_fit")


def mir_pls_fit(ctx: Context, cfg: Config,
                 X: Any, Y: Any) -> MethodResult:
    X_arr = _as_float64_contiguous(X)
    Y_arr = _as_float64_contiguous(Y)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    out = ctypes.c_void_p(0)
    status = lib.n4m_mir_pls_fit(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view), ctypes.byref(y_view),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_mir_pls_fit")


def missing_aware_nipals_fit(ctx: Context, cfg: Config,
                               X: Any, Y: Any) -> MethodResult:
    X_arr = _as_float64_contiguous(X)
    Y_arr = _as_float64_contiguous(Y)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    out = ctypes.c_void_p(0)
    status = lib.n4m_missing_aware_nipals_fit(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view), ctypes.byref(y_view),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_missing_aware_nipals_fit")


def pls_diagnostics_compute(ctx: Context,
                              model: Any,
                              X: Any,
                              X_reference: Any | None = None,
                              ) -> MethodResult:
    X_arr = _as_float64_contiguous(X)
    x_view = _matrix_view(X_arr)
    ref_view_ptr = None
    if X_reference is not None:
        Xr_arr = _as_float64_contiguous(X_reference)
        ref_view = _matrix_view(Xr_arr)
        ref_view_ptr = ctypes.byref(ref_view)
    out = ctypes.c_void_p(0)
    # model is a pls4all.Model wrapper exposing .handle.
    model_handle = getattr(model, "handle", model)
    status = lib.n4m_pls_diagnostics_compute(
        ctx.handle, model_handle,
        ctypes.byref(x_view),
        ref_view_ptr if ref_view_ptr is not None else None,
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_pls_diagnostics_compute")


# ============================================================================
#  §17 — Phase 4r/4s/4p/4q + 6a ABI shims (internal-only core algos)
# ============================================================================

def mb_pls_fit(ctx: Context, cfg: Config,
                X: Any, Y: Any,
                block_sizes: Any) -> MethodResult:
    X_arr = _as_float64_contiguous(X)
    Y_arr = _as_float64_contiguous(Y)
    sizes = np.ascontiguousarray(block_sizes, dtype=np.int64).reshape(-1)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    out = ctypes.c_void_p(0)
    status = lib.n4m_mb_pls_fit(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view), ctypes.byref(y_view),
        sizes.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int64(int(sizes.size)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_mb_pls_fit")


def lw_pls_fit(ctx: Context, cfg: Config,
                X: Any, Y: Any,
                n_neighbors: int) -> MethodResult:
    X_arr = _as_float64_contiguous(X)
    Y_arr = _as_float64_contiguous(Y)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    out = ctypes.c_void_p(0)
    status = lib.n4m_lw_pls_fit(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view), ctypes.byref(y_view),
        ctypes.c_int32(int(n_neighbors)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_lw_pls_fit")


def pls_lda_fit(ctx: Context, cfg: Config,
                 X: Any, y_labels: Any,
                 n_classes: int) -> MethodResult:
    X_arr = _as_float64_contiguous(X)
    labels = np.ascontiguousarray(y_labels, dtype=np.int32).reshape(-1)
    x_view = _matrix_view(X_arr)
    out = ctypes.c_void_p(0)
    status = lib.n4m_pls_lda_fit(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view),
        labels.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int64(int(labels.size)),
        ctypes.c_int32(int(n_classes)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_pls_lda_fit")


def pls_logistic_fit(ctx: Context, cfg: Config,
                      X: Any, y_labels: Any,
                      n_classes: int) -> MethodResult:
    X_arr = _as_float64_contiguous(X)
    labels = np.ascontiguousarray(y_labels, dtype=np.int32).reshape(-1)
    x_view = _matrix_view(X_arr)
    out = ctypes.c_void_p(0)
    status = lib.n4m_pls_logistic_fit(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view),
        labels.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int64(int(labels.size)),
        ctypes.c_int32(int(n_classes)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_pls_logistic_fit")


def aom_preprocess_fit(ctx: Context,
                        bank: Any,
                        gate: Any,
                        X: Any,
                        Y: Any | None = None) -> MethodResult:
    X_arr = _as_float64_contiguous(X)
    x_view = _matrix_view(X_arr)
    y_view_ptr = None
    _y_arr = None
    if Y is not None:
        _y_arr = _as_float64_contiguous(Y)
        y_view = _matrix_view(_y_arr)
        y_view_ptr = ctypes.byref(y_view)
    bank_handle = getattr(bank, "handle", bank)
    gate_handle = getattr(gate, "handle", gate)
    out = ctypes.c_void_p(0)
    status = lib.n4m_aom_preprocess_fit(
        ctx.handle, bank_handle, gate_handle,
        ctypes.byref(x_view),
        y_view_ptr,
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_aom_preprocess_fit")


# ============================================================================
#  §18 — Phase 5 variable-selection method shims
# ============================================================================

def variable_select_rank(ctx: Context,
                          model: Any,
                          X: Any,
                          method: int,
                          top_k: int) -> MethodResult:
    """Rank features by VIP (0), coefficient magnitude (1), or selectivity ratio (2)."""
    X_arr = _as_float64_contiguous(X)
    x_view = _matrix_view(X_arr)
    model_handle = getattr(model, "handle", model)
    out = ctypes.c_void_p(0)
    status = lib.n4m_variable_select_rank(
        ctx.handle, model_handle,
        ctypes.byref(x_view),
        ctypes.c_int32(int(method)),
        ctypes.c_int32(int(top_k)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_variable_select_rank")


def interval_select(ctx: Context, cfg: Config,
                     X: Any, Y: Any,
                     plan: Any,
                     interval_width: int,
                     step: int) -> MethodResult:
    X_arr = _as_float64_contiguous(X)
    Y_arr = _as_float64_contiguous(Y)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    plan_handle = getattr(plan, "handle", plan)
    out = ctypes.c_void_p(0)
    status = lib.n4m_interval_select(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view), ctypes.byref(y_view),
        plan_handle,
        ctypes.c_int32(int(interval_width)),
        ctypes.c_int32(int(step)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_interval_select")


def stability_select(ctx: Context, cfg: Config,
                      X: Any, Y: Any,
                      plan: Any,
                      top_k: int) -> MethodResult:
    X_arr = _as_float64_contiguous(X)
    Y_arr = _as_float64_contiguous(Y)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    plan_handle = getattr(plan, "handle", plan)
    out = ctypes.c_void_p(0)
    status = lib.n4m_stability_select(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view), ctypes.byref(y_view),
        plan_handle,
        ctypes.c_int32(int(top_k)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_stability_select")


def uve_select(ctx: Context, cfg: Config,
                X: Any, Y: Any,
                plan: Any,
                noise_features: int,
                noise_seed: int = 0) -> MethodResult:
    X_arr = _as_float64_contiguous(X)
    Y_arr = _as_float64_contiguous(Y)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    plan_handle = getattr(plan, "handle", plan)
    out = ctypes.c_void_p(0)
    status = lib.n4m_uve_select(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view), ctypes.byref(y_view),
        plan_handle,
        ctypes.c_int32(int(noise_features)),
        ctypes.c_uint64(int(noise_seed)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_uve_select")


def spa_select(ctx: Context, cfg: Config,
                X: Any, Y: Any,
                top_k: int) -> MethodResult:
    X_arr = _as_float64_contiguous(X)
    Y_arr = _as_float64_contiguous(Y)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    out = ctypes.c_void_p(0)
    status = lib.n4m_spa_select(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view), ctypes.byref(y_view),
        ctypes.c_int32(int(top_k)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_spa_select")


def cars_select(ctx: Context, cfg: Config,
                 X: Any, Y: Any,
                 plan: Any,
                 *, n_iterations: int,
                 min_features: int) -> MethodResult:
    X_arr = _as_float64_contiguous(X)
    Y_arr = _as_float64_contiguous(Y)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    plan_handle = getattr(plan, "handle", plan)
    out = ctypes.c_void_p(0)
    status = lib.n4m_cars_select(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view), ctypes.byref(y_view),
        plan_handle,
        ctypes.c_int32(int(n_iterations)),
        ctypes.c_int32(int(min_features)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_cars_select")


def random_frog_select(ctx: Context, cfg: Config,
                        X: Any, Y: Any,
                        plan: Any,
                        *, n_iterations: int,
                        initial_size: int,
                        min_size: int,
                        max_size: int,
                        top_k: int,
                        seed: int = 0) -> MethodResult:
    X_arr = _as_float64_contiguous(X)
    Y_arr = _as_float64_contiguous(Y)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    plan_handle = getattr(plan, "handle", plan)
    out = ctypes.c_void_p(0)
    status = lib.n4m_random_frog_select(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view), ctypes.byref(y_view),
        plan_handle,
        ctypes.c_int32(int(n_iterations)),
        ctypes.c_int32(int(initial_size)),
        ctypes.c_int32(int(min_size)),
        ctypes.c_int32(int(max_size)),
        ctypes.c_int32(int(top_k)),
        ctypes.c_uint64(int(seed)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_random_frog_select")


def scars_select(ctx: Context, cfg: Config,
                  X: Any, Y: Any,
                  plan: Any,
                  *, n_iterations: int,
                  min_features: int,
                  sample_fraction: float,
                  seed: int = 0) -> MethodResult:
    X_arr = _as_float64_contiguous(X)
    Y_arr = _as_float64_contiguous(Y)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    plan_handle = getattr(plan, "handle", plan)
    out = ctypes.c_void_p(0)
    status = lib.n4m_scars_select(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view), ctypes.byref(y_view),
        plan_handle,
        ctypes.c_int32(int(n_iterations)),
        ctypes.c_int32(int(min_features)),
        ctypes.c_double(float(sample_fraction)),
        ctypes.c_uint64(int(seed)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_scars_select")


def ga_select(ctx: Context, cfg: Config,
               X: Any, Y: Any,
               plan: Any,
               *, n_generations: int,
               population_size: int,
               min_features: int,
               max_features: int,
               mutation_rate: float,
               seed: int = 0) -> MethodResult:
    X_arr = _as_float64_contiguous(X)
    Y_arr = _as_float64_contiguous(Y)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    plan_handle = getattr(plan, "handle", plan)
    out = ctypes.c_void_p(0)
    status = lib.n4m_ga_select(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view), ctypes.byref(y_view),
        plan_handle,
        ctypes.c_int32(int(n_generations)),
        ctypes.c_int32(int(population_size)),
        ctypes.c_int32(int(min_features)),
        ctypes.c_int32(int(max_features)),
        ctypes.c_double(float(mutation_rate)),
        ctypes.c_uint64(int(seed)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_ga_select")


def pso_select(ctx: Context, cfg: Config,
                X: Any, Y: Any,
                plan: Any,
                *, n_swarm: int,
                n_iterations: int,
                w: float = 0.729,
                c1: float = 1.494,
                c2: float = 1.494,
                v_max: float = 4.0,
                seed: int = 0) -> MethodResult:
    """PSO-PLS (§48): Binary Particle Swarm variable selection.

    Reference: Kennedy & Eberhart (1997). Defaults follow the
    Clerc-Kennedy constriction-coefficient analysis (2002).
    """
    X_arr = _as_float64_contiguous(X)
    Y_arr = _as_float64_contiguous(Y)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    plan_handle = getattr(plan, "handle", plan)
    out = ctypes.c_void_p(0)
    status = lib.n4m_pso_select(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view), ctypes.byref(y_view),
        plan_handle,
        ctypes.c_int32(int(n_swarm)),
        ctypes.c_int32(int(n_iterations)),
        ctypes.c_double(float(w)),
        ctypes.c_double(float(c1)),
        ctypes.c_double(float(c2)),
        ctypes.c_double(float(v_max)),
        ctypes.c_uint64(int(seed)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_pso_select")


def vissa_select(ctx: Context, cfg: Config,
                  X: Any, Y: Any,
                  plan: Any,
                  *, n_iterations: int = 20,
                  n_submodels: int = 100,
                  ratio_kept: float = 0.1,
                  threshold: float = 0.5,
                  floor_probability: float = 0.01,
                  seed: int = 0) -> MethodResult:
    """VISSA-PLS (§49): Variable Iterative Space Shrinkage Approach.

    Paper-only reference: Deng B. et al. (2014) Anal. Chim. Acta
    838:27-40. WBMS-based iterative shrinkage of variable inclusion
    probabilities.
    """
    X_arr = _as_float64_contiguous(X)
    Y_arr = _as_float64_contiguous(Y)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    plan_handle = getattr(plan, "handle", plan)
    out = ctypes.c_void_p(0)
    status = lib.n4m_vissa_select(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view), ctypes.byref(y_view),
        plan_handle,
        ctypes.c_int32(int(n_iterations)),
        ctypes.c_int32(int(n_submodels)),
        ctypes.c_double(float(ratio_kept)),
        ctypes.c_double(float(threshold)),
        ctypes.c_double(float(floor_probability)),
        ctypes.c_uint64(int(seed)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_vissa_select")


def shaving_select(ctx: Context, cfg: Config,
                    X: Any, Y: Any,
                    plan: Any,
                    *, n_steps: int,
                    min_features: int,
                    shave_fraction: float) -> MethodResult:
    X_arr = _as_float64_contiguous(X)
    Y_arr = _as_float64_contiguous(Y)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    plan_handle = getattr(plan, "handle", plan)
    out = ctypes.c_void_p(0)
    status = lib.n4m_shaving_select(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view), ctypes.byref(y_view),
        plan_handle,
        ctypes.c_int32(int(n_steps)),
        ctypes.c_int32(int(min_features)),
        ctypes.c_double(float(shave_fraction)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_shaving_select")


def bve_select(ctx: Context, cfg: Config,
                X: Any, Y: Any,
                plan: Any,
                *, n_steps: int,
                min_features: int) -> MethodResult:
    X_arr = _as_float64_contiguous(X)
    Y_arr = _as_float64_contiguous(Y)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    plan_handle = getattr(plan, "handle", plan)
    out = ctypes.c_void_p(0)
    status = lib.n4m_bve_select(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view), ctypes.byref(y_view),
        plan_handle,
        ctypes.c_int32(int(n_steps)),
        ctypes.c_int32(int(min_features)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_bve_select")


def t2_select(ctx: Context, cfg: Config,
               X: Any, Y: Any,
               plan: Any,
               *, alpha_thresholds: Any,
               min_selected: int) -> MethodResult:
    X_arr = _as_float64_contiguous(X)
    Y_arr = _as_float64_contiguous(Y)
    alphas = np.ascontiguousarray(alpha_thresholds, dtype=np.float64).reshape(-1)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    plan_handle = getattr(plan, "handle", plan)
    out = ctypes.c_void_p(0)
    status = lib.n4m_t2_select(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view), ctypes.byref(y_view),
        plan_handle,
        alphas.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int64(int(alphas.size)),
        ctypes.c_int32(int(min_selected)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_t2_select")


def wvc_select(ctx: Context,
                X: Any, Y: Any,
                *, n_components: int,
                top_k: int,
                normalize: bool = True) -> MethodResult:
    X_arr = _as_float64_contiguous(X)
    Y_arr = _as_float64_contiguous(Y)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    out = ctypes.c_void_p(0)
    status = lib.n4m_wvc_select(
        ctx.handle,
        ctypes.byref(x_view), ctypes.byref(y_view),
        ctypes.c_int32(int(n_components)),
        ctypes.c_int32(int(top_k)),
        ctypes.c_int32(1 if normalize else 0),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_wvc_select")


def wvc_threshold_select(ctx: Context,
                          X: Any, Y: Any,
                          *, n_components: int,
                          normalize: bool = True,
                          score_threshold: float = 0.0,
                          threshold_factor: float = 1.0,
                          min_selected: int = 1) -> MethodResult:
    X_arr = _as_float64_contiguous(X)
    Y_arr = _as_float64_contiguous(Y)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    out = ctypes.c_void_p(0)
    status = lib.n4m_wvc_threshold_select(
        ctx.handle,
        ctypes.byref(x_view), ctypes.byref(y_view),
        ctypes.c_int32(int(n_components)),
        ctypes.c_int32(1 if normalize else 0),
        ctypes.c_double(float(score_threshold)),
        ctypes.c_double(float(threshold_factor)),
        ctypes.c_int32(int(min_selected)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_wvc_threshold_select")


def emcuve_select(ctx: Context, cfg: Config,
                   X: Any, Y: Any,
                   plan: Any,
                   *, noise_features: int,
                   noise_seed: int = 0,
                   n_ensembles: int,
                   vote_threshold: float) -> MethodResult:
    X_arr = _as_float64_contiguous(X)
    Y_arr = _as_float64_contiguous(Y)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    plan_handle = getattr(plan, "handle", plan)
    out = ctypes.c_void_p(0)
    status = lib.n4m_emcuve_select(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view), ctypes.byref(y_view),
        plan_handle,
        ctypes.c_int32(int(noise_features)),
        ctypes.c_uint64(int(noise_seed)),
        ctypes.c_int32(int(n_ensembles)),
        ctypes.c_double(float(vote_threshold)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_emcuve_select")


def randomization_select(ctx: Context, cfg: Config,
                          X: Any, Y: Any,
                          *, n_permutations: int,
                          randomization_seed: int = 0,
                          alpha: float = 0.05) -> MethodResult:
    X_arr = _as_float64_contiguous(X)
    Y_arr = _as_float64_contiguous(Y)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    out = ctypes.c_void_p(0)
    status = lib.n4m_randomization_select(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view), ctypes.byref(y_view),
        ctypes.c_int32(int(n_permutations)),
        ctypes.c_uint64(int(randomization_seed)),
        ctypes.c_double(float(alpha)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_randomization_select")


def bipls_select(ctx: Context, cfg: Config,
                  X: Any, Y: Any,
                  plan: Any,
                  *, interval_width: int,
                  min_intervals: int) -> MethodResult:
    X_arr = _as_float64_contiguous(X)
    Y_arr = _as_float64_contiguous(Y)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    plan_handle = getattr(plan, "handle", plan)
    out = ctypes.c_void_p(0)
    status = lib.n4m_bipls_select(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view), ctypes.byref(y_view),
        plan_handle,
        ctypes.c_int32(int(interval_width)),
        ctypes.c_int32(int(min_intervals)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_bipls_select")


def sipls_select(ctx: Context, cfg: Config,
                  X: Any, Y: Any,
                  plan: Any,
                  *, interval_width: int,
                  combination_size: int) -> MethodResult:
    X_arr = _as_float64_contiguous(X)
    Y_arr = _as_float64_contiguous(Y)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    plan_handle = getattr(plan, "handle", plan)
    out = ctypes.c_void_p(0)
    status = lib.n4m_sipls_select(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view), ctypes.byref(y_view),
        plan_handle,
        ctypes.c_int32(int(interval_width)),
        ctypes.c_int32(int(combination_size)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_sipls_select")


def rep_select(ctx: Context, cfg: Config,
                X: Any, Y: Any,
                plan: Any,
                *, n_steps: int,
                min_features: int,
                remove_count: int) -> MethodResult:
    X_arr = _as_float64_contiguous(X)
    Y_arr = _as_float64_contiguous(Y)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    plan_handle = getattr(plan, "handle", plan)
    out = ctypes.c_void_p(0)
    status = lib.n4m_rep_select(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view), ctypes.byref(y_view),
        plan_handle,
        ctypes.c_int32(int(n_steps)),
        ctypes.c_int32(int(min_features)),
        ctypes.c_int32(int(remove_count)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_rep_select")


def ipw_select(ctx: Context, cfg: Config,
                X: Any, Y: Any,
                plan: Any,
                *, n_iterations: int,
                top_k: int,
                damping: float,
                weight_floor: float) -> MethodResult:
    X_arr = _as_float64_contiguous(X)
    Y_arr = _as_float64_contiguous(Y)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    plan_handle = getattr(plan, "handle", plan)
    out = ctypes.c_void_p(0)
    status = lib.n4m_ipw_select(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view), ctypes.byref(y_view),
        plan_handle,
        ctypes.c_int32(int(n_iterations)),
        ctypes.c_int32(int(top_k)),
        ctypes.c_double(float(damping)),
        ctypes.c_double(float(weight_floor)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_ipw_select")


def st_select(ctx: Context, cfg: Config,
               X: Any, Y: Any,
               plan: Any,
               *, thresholds: Any,
               min_selected: int) -> MethodResult:
    X_arr = _as_float64_contiguous(X)
    Y_arr = _as_float64_contiguous(Y)
    thr = np.ascontiguousarray(thresholds, dtype=np.float64).reshape(-1)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    plan_handle = getattr(plan, "handle", plan)
    out = ctypes.c_void_p(0)
    status = lib.n4m_st_select(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view), ctypes.byref(y_view),
        plan_handle,
        thr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int64(int(thr.size)),
        ctypes.c_int32(int(min_selected)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_st_select")


def ecr_fit(ctx: Context, cfg: Config,
             X: Any, Y: Any,
             alpha: float) -> MethodResult:
    """ECR — Elastic Component Regression (Phase 50).

    Bridge between PCR (alpha=0) and PLS (alpha=1) introduced by Liu et al.
    (2009/2010). For each component the dominant eigenvector of
    ``(1-alpha) * X'X + alpha * X' y y' X`` is extracted via power iteration,
    scores deflated, and a regression coefficient matrix is reassembled
    via the SIMPLS-style ``Wstar = W (P' W)^{-1}`` relation.
    """
    X_arr = _as_float64_contiguous(X)
    Y_arr = _as_float64_contiguous(Y)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    out = ctypes.c_void_p(0)
    status = lib.n4m_ecr_fit(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view), ctypes.byref(y_view),
        ctypes.c_double(float(alpha)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_ecr_fit")


def iriv_select(ctx: Context, cfg: Config,
                 X: Any, Y: Any,
                 plan: Any,
                 *, max_rounds: int,
                 seed: int = 0) -> MethodResult:
    """IRIV — Iteratively Retains Informative Variables (Phase 51).

    Yun et al. (2014). Each round samples a binary mask matrix, runs
    PLS-CV on each row and on per-variable bit-flipped variants, then
    drops variables flagged uninformative by a Mann-Whitney U test on
    the include/exclude RMSECV populations. Iterates until no variable
    is flagged or ``max_rounds`` is reached.
    """
    X_arr = _as_float64_contiguous(X)
    Y_arr = _as_float64_contiguous(Y)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    plan_handle = getattr(plan, "handle", plan)
    out = ctypes.c_void_p(0)
    status = lib.n4m_iriv_select(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view), ctypes.byref(y_view),
        plan_handle,
        ctypes.c_int32(int(max_rounds)),
        ctypes.c_uint64(int(seed)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_iriv_select")


def irf_select(ctx: Context, cfg: Config,
                X: Any, Y: Any,
                plan: Any,
                *, n_iterations: int,
                window_size: int,
                initial_intervals: int,
                top_k: int,
                seed: int = 0) -> MethodResult:
    """IRF — Interval Random Frog (Phase 52).

    Yun et al. (2013). Random Frog applied to fixed-width sliding
    spectral intervals; the final feature selection is the union of the
    features under the top-K inclusion-frequency intervals.
    """
    X_arr = _as_float64_contiguous(X)
    Y_arr = _as_float64_contiguous(Y)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    plan_handle = getattr(plan, "handle", plan)
    out = ctypes.c_void_p(0)
    status = lib.n4m_irf_select(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view), ctypes.byref(y_view),
        plan_handle,
        ctypes.c_int32(int(n_iterations)),
        ctypes.c_int32(int(window_size)),
        ctypes.c_int32(int(initial_intervals)),
        ctypes.c_int32(int(top_k)),
        ctypes.c_uint64(int(seed)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_irf_select")


def vip_spa_select(ctx: Context, cfg: Config,
                    X: Any, Y: Any,
                    *, vip_threshold: float = 0.3,
                    top_k: int) -> MethodResult:
    """VIP_SPA — VIP-then-SPA hybrid variable selection (Phase 53).

    Composes Favilla 2013 VIP scoring with Araujo 2001 Successive
    Projections greedy collinearity-minimising pick over features
    surviving ``VIP > vip_threshold``. Mirrors auswahl.VIP_SPA
    (LSX-UniWue). The SPA start is taken as argmax(VIP) within the
    surviving subset (deterministic) rather than auswahl's per-seed
    enumeration.
    """
    X_arr = _as_float64_contiguous(X)
    Y_arr = _as_float64_contiguous(Y)
    x_view = _matrix_view(X_arr)
    y_view = _matrix_view(Y_arr)
    out = ctypes.c_void_p(0)
    status = lib.n4m_vip_spa_select(
        ctx.handle, cfg.handle,
        ctypes.byref(x_view), ctypes.byref(y_view),
        ctypes.c_double(float(vip_threshold)),
        ctypes.c_int32(int(top_k)),
        ctypes.byref(out),
    )
    _check(status, ctx)
    return _resolve_handle(out, ctx, "n4m_vip_spa_select")


__all__ = [
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
    # §17 ABI shims
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
]
