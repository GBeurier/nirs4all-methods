"""ctypes loader + raw symbol signatures."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_DIR.parents[3] if _PACKAGE_DIR.parents[3].name == "pls4all" else None


def _candidate_paths() -> list[Path]:
    paths: list[Path] = []

    env = os.environ.get("PLS4ALL_LIB_PATH")
    if env:
        paths.append(Path(env))

    # Wheel layout: bindings/python/src/pls4all/lib/libp4a*
    for p in (_PACKAGE_DIR / "lib").glob("libp4a*"):
        paths.append(p)
    for p in (_PACKAGE_DIR / "lib").glob("p4a*"):
        paths.append(p)

    # Developer convenience: repo-root build directory.
    if _REPO_ROOT is not None:
        for preset in ("dev-release", "dev-debug"):
            for name in ("libp4a.so", "libp4a.dylib", "libp4a.so.0.66.0", "libp4a.so.0.65.0", "libp4a.so.0.64.0", "libp4a.so.0.63.0", "libp4a.so.0.62.0", "libp4a.so.0.61.0", "libp4a.so.0.60.0", "libp4a.so.0.59.0", "libp4a.so.0.58.0", "libp4a.so.0.57.0", "libp4a.so.0.56.0", "libp4a.so.0.55.0", "libp4a.so.0.54.0", "libp4a.so.0.53.0", "libp4a.so.0.52.0", "libp4a.so.0.51.0", "libp4a.so.0.50.0", "libp4a.so.0.49.0", "libp4a.so.0.48.0", "libp4a.so.0.47.0", "libp4a.so.0.46.0", "libp4a.so.0.45.0", "libp4a.so.0.44.0", "libp4a.so.0.43.0", "libp4a.so.0.42.0", "libp4a.so.0.41.0", "libp4a.so.0.40.0", "libp4a.so.0.39.0", "libp4a.so.0.38.0", "libp4a.so.0.37.0", "libp4a.so.0.36.0", "libp4a.so.0.35.0", "libp4a.so.0.34.0", "libp4a.so.0.33.0", "libp4a.so.0.32.0", "libp4a.so.0.31.0", "libp4a.so.0.30.0", "libp4a.so.0.29.0", "libp4a.so.0.28.0", "libp4a.so.0.27.0", "libp4a.so.0.26.0", "libp4a.so.0.25.0", "libp4a.so.0.24.0", "libp4a.so.0.23.0", "libp4a.so.0.22.0", "libp4a.so.0.21.0", "libp4a.so.0.20.0", "libp4a.so.0.19.0", "libp4a.so.0.18.0", "libp4a.so.0.17.0", "libp4a.so.0.16.0", "libp4a.so.0.15.0", "libp4a.so.0.14.0", "libp4a.so.0.13.0", "libp4a.so.0.12.0", "libp4a.so.0.11.0", "libp4a.so.0.10.0", "libp4a.so.0.9.0", "libp4a.so.0.8.0", "libp4a.so.0.7.0", "libp4a.so.0.6.0", "libp4a.so.0.5.0", "libp4a.so.0.4.0", "libp4a.so.0.3.0", "libp4a.so.0.2.0", "libp4a.so.0.1.0", "p4a.dll"):
                p = _REPO_ROOT / "build" / preset / "cpp" / "src" / name
                if p.exists():
                    paths.append(p)

    # System search path.
    if sys.platform.startswith("linux"):
        sys_path = ctypes.util.find_library("p4a")
        if sys_path:
            paths.append(Path(sys_path))
    elif sys.platform == "darwin":
        sys_path = ctypes.util.find_library("p4a")
        if sys_path:
            paths.append(Path(sys_path))
    elif sys.platform == "win32":
        for name in ("p4a.dll", "libp4a.dll"):
            paths.append(Path(name))
    return paths


def _load_library() -> ctypes.CDLL:
    errors = []
    for path in _candidate_paths():
        try:
            return ctypes.CDLL(str(path))
        except OSError as exc:
            errors.append((path, exc))
    msg = ["could not locate libp4a. Tried:"]
    for p, e in errors:
        msg.append(f"  - {p}: {e}")
    msg.append(
        "Set the PLS4ALL_LIB_PATH environment variable to the absolute path of "
        "libp4a.so / libp4a.dylib / p4a.dll."
    )
    raise ImportError("\n".join(msg))


lib = _load_library()

# ---- p4a_get_* (version queries) ----
lib.p4a_get_abi_version_major.restype = ctypes.c_uint32
lib.p4a_get_abi_version_minor.restype = ctypes.c_uint32
lib.p4a_get_abi_version_patch.restype = ctypes.c_uint32
lib.p4a_get_abi_version_int.restype   = ctypes.c_uint32
lib.p4a_get_version_string.restype    = ctypes.c_char_p
lib.p4a_get_build_info.restype        = ctypes.c_char_p
lib.p4a_get_git_revision.restype      = ctypes.c_char_p

# ---- p4a_status / dtype / backend ----
lib.p4a_status_to_string.restype  = ctypes.c_char_p
lib.p4a_status_to_string.argtypes = [ctypes.c_int]

lib.p4a_dtype_size.restype  = ctypes.c_size_t
lib.p4a_dtype_size.argtypes = [ctypes.c_int]
lib.p4a_dtype_to_string.restype  = ctypes.c_char_p
lib.p4a_dtype_to_string.argtypes = [ctypes.c_int]

lib.p4a_backend_is_available.restype  = ctypes.c_int
lib.p4a_backend_is_available.argtypes = [ctypes.c_int]
lib.p4a_backend_to_string.restype  = ctypes.c_char_p
lib.p4a_backend_to_string.argtypes = [ctypes.c_int]

# ---- p4a_context_* ----
lib.p4a_context_create.restype  = ctypes.c_int
lib.p4a_context_create.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
lib.p4a_context_destroy.restype  = None
lib.p4a_context_destroy.argtypes = [ctypes.c_void_p]
lib.p4a_context_set_seed.restype  = ctypes.c_int
lib.p4a_context_set_seed.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
lib.p4a_context_get_seed.restype  = ctypes.c_int
lib.p4a_context_get_seed.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
lib.p4a_context_set_backend.restype  = ctypes.c_int
lib.p4a_context_set_backend.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.p4a_context_get_backend.restype  = ctypes.c_int
lib.p4a_context_get_backend.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
lib.p4a_context_set_num_threads.restype  = ctypes.c_int
lib.p4a_context_set_num_threads.argtypes = [ctypes.c_void_p, ctypes.c_int32]
lib.p4a_context_get_num_threads.restype  = ctypes.c_int
lib.p4a_context_get_num_threads.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32)]
lib.p4a_context_last_error.restype  = ctypes.c_char_p
lib.p4a_context_last_error.argtypes = [ctypes.c_void_p]
lib.p4a_context_clear_error.restype  = None
lib.p4a_context_clear_error.argtypes = [ctypes.c_void_p]

# ---- p4a_config_* (a representative subset; Phase 2 fills out the rest) ----
lib.p4a_config_create.restype  = ctypes.c_int
lib.p4a_config_create.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
lib.p4a_config_destroy.restype  = None
lib.p4a_config_destroy.argtypes = [ctypes.c_void_p]
lib.p4a_config_set_n_components.restype  = ctypes.c_int
lib.p4a_config_set_n_components.argtypes = [ctypes.c_void_p, ctypes.c_int32]
lib.p4a_config_get_n_components.restype  = ctypes.c_int
lib.p4a_config_get_n_components.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32)]
lib.p4a_config_set_tol.restype  = ctypes.c_int
lib.p4a_config_set_tol.argtypes = [ctypes.c_void_p, ctypes.c_double]
lib.p4a_config_get_tol.restype  = ctypes.c_int
lib.p4a_config_get_tol.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
lib.p4a_config_set_algorithm.restype  = ctypes.c_int
lib.p4a_config_set_algorithm.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.p4a_config_get_algorithm.restype  = ctypes.c_int
lib.p4a_config_get_algorithm.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
lib.p4a_config_set_solver.restype  = ctypes.c_int
lib.p4a_config_set_solver.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.p4a_config_get_solver.restype  = ctypes.c_int
lib.p4a_config_get_solver.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
lib.p4a_config_set_deflation.restype  = ctypes.c_int
lib.p4a_config_set_deflation.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.p4a_config_get_deflation.restype  = ctypes.c_int
lib.p4a_config_get_deflation.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
lib.p4a_config_set_center_x.restype  = ctypes.c_int
lib.p4a_config_set_center_x.argtypes = [ctypes.c_void_p, ctypes.c_int32]
lib.p4a_config_get_center_x.restype  = ctypes.c_int
lib.p4a_config_get_center_x.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32)]
lib.p4a_config_set_scale_x.restype   = ctypes.c_int
lib.p4a_config_set_scale_x.argtypes  = [ctypes.c_void_p, ctypes.c_int32]
lib.p4a_config_get_scale_x.restype   = ctypes.c_int
lib.p4a_config_get_scale_x.argtypes  = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32)]
lib.p4a_config_set_center_y.restype  = ctypes.c_int
lib.p4a_config_set_center_y.argtypes = [ctypes.c_void_p, ctypes.c_int32]
lib.p4a_config_get_center_y.restype  = ctypes.c_int
lib.p4a_config_get_center_y.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32)]
lib.p4a_config_set_scale_y.restype   = ctypes.c_int
lib.p4a_config_set_scale_y.argtypes  = [ctypes.c_void_p, ctypes.c_int32]
lib.p4a_config_get_scale_y.restype   = ctypes.c_int
lib.p4a_config_get_scale_y.argtypes  = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32)]
lib.p4a_config_set_store_scores.restype  = ctypes.c_int
lib.p4a_config_set_store_scores.argtypes = [ctypes.c_void_p, ctypes.c_int32]
lib.p4a_config_get_store_scores.restype  = ctypes.c_int
lib.p4a_config_get_store_scores.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32)]

# ---- p4a_operator_bank_* ----
lib.p4a_operator_bank_create.restype  = ctypes.c_int
lib.p4a_operator_bank_create.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
lib.p4a_operator_bank_destroy.restype  = None
lib.p4a_operator_bank_destroy.argtypes = [ctypes.c_void_p]
lib.p4a_operator_bank_add.restype  = ctypes.c_int
lib.p4a_operator_bank_add.argtypes = [
    ctypes.c_void_p, ctypes.c_int,
    ctypes.POINTER(ctypes.c_double), ctypes.c_int32,
]
lib.p4a_operator_bank_size.restype  = ctypes.c_int
lib.p4a_operator_bank_size.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32)]

# ---- p4a_validation_plan_* ----
lib.p4a_validation_plan_create.restype  = ctypes.c_int
lib.p4a_validation_plan_create.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
lib.p4a_validation_plan_destroy.restype  = None
lib.p4a_validation_plan_destroy.argtypes = [ctypes.c_void_p]
lib.p4a_validation_plan_set_n_samples.restype  = ctypes.c_int
lib.p4a_validation_plan_set_n_samples.argtypes = [ctypes.c_void_p, ctypes.c_int64]
lib.p4a_validation_plan_add_fold.restype  = ctypes.c_int
lib.p4a_validation_plan_add_fold.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int64), ctypes.c_int64,
    ctypes.POINTER(ctypes.c_int64), ctypes.c_int64,
]
lib.p4a_validation_plan_get_n_samples.restype  = ctypes.c_int
lib.p4a_validation_plan_get_n_samples.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64),
]
lib.p4a_validation_plan_get_n_folds.restype  = ctypes.c_int
lib.p4a_validation_plan_get_n_folds.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
]

# ---- p4a_matrix_view_t layout (must mirror cpp/include/pls4all/p4a.h) ----


class MatrixView(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("rows", ctypes.c_int64),
        ("cols", ctypes.c_int64),
        ("row_stride", ctypes.c_int64),
        ("col_stride", ctypes.c_int64),
        ("dtype", ctypes.c_int),
        ("reserved0", ctypes.c_int32),
    ]


# ---- p4a_aom_global_* ----
lib.p4a_aom_global_select.restype  = ctypes.c_int
lib.p4a_aom_global_select.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p),
]
lib.p4a_aom_global_result_destroy.restype  = None
lib.p4a_aom_global_result_destroy.argtypes = [ctypes.c_void_p]
lib.p4a_aom_global_result_get_n_operators.restype  = ctypes.c_int
lib.p4a_aom_global_result_get_n_operators.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
]
lib.p4a_aom_global_result_get_max_components.restype  = ctypes.c_int
lib.p4a_aom_global_result_get_max_components.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
]
lib.p4a_aom_global_result_get_selected_operator_index.restype  = ctypes.c_int
lib.p4a_aom_global_result_get_selected_operator_index.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
]
lib.p4a_aom_global_result_get_selected_n_components.restype  = ctypes.c_int
lib.p4a_aom_global_result_get_selected_n_components.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
]
lib.p4a_aom_global_result_get_best_score.restype  = ctypes.c_int
lib.p4a_aom_global_result_get_best_score.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_double),
]
lib.p4a_aom_global_result_get_operator_kinds.restype  = ctypes.c_int
lib.p4a_aom_global_result_get_operator_kinds.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
    ctypes.POINTER(ctypes.c_int32),
]
lib.p4a_aom_global_result_get_operator_scores.restype  = ctypes.c_int
lib.p4a_aom_global_result_get_operator_scores.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
    ctypes.POINTER(ctypes.c_int32),
]
lib.p4a_aom_global_result_get_rmse_curves.restype  = ctypes.c_int
lib.p4a_aom_global_result_get_rmse_curves.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
    ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
]
lib.p4a_aom_global_result_get_predictions.restype  = ctypes.c_int
lib.p4a_aom_global_result_get_predictions.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
    ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64),
]

# ---- p4a_aom_per_component_* ----
lib.p4a_aom_per_component_select.restype  = ctypes.c_int
lib.p4a_aom_per_component_select.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p),
]
lib.p4a_aom_per_component_result_destroy.restype  = None
lib.p4a_aom_per_component_result_destroy.argtypes = [ctypes.c_void_p]
lib.p4a_aom_per_component_result_get_n_operators.restype  = ctypes.c_int
lib.p4a_aom_per_component_result_get_n_operators.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
]
lib.p4a_aom_per_component_result_get_max_components.restype  = ctypes.c_int
lib.p4a_aom_per_component_result_get_max_components.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
]
lib.p4a_aom_per_component_result_get_selected_n_components.restype  = ctypes.c_int
lib.p4a_aom_per_component_result_get_selected_n_components.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
]
lib.p4a_aom_per_component_result_get_best_score.restype  = ctypes.c_int
lib.p4a_aom_per_component_result_get_best_score.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_double),
]
lib.p4a_aom_per_component_result_get_operator_kinds.restype  = ctypes.c_int
lib.p4a_aom_per_component_result_get_operator_kinds.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
    ctypes.POINTER(ctypes.c_int32),
]
lib.p4a_aom_per_component_result_get_selected_operator_indices.restype  = ctypes.c_int
lib.p4a_aom_per_component_result_get_selected_operator_indices.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)),
    ctypes.POINTER(ctypes.c_int32),
]
lib.p4a_aom_per_component_result_get_component_scores.restype  = ctypes.c_int
lib.p4a_aom_per_component_result_get_component_scores.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
    ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
]
lib.p4a_aom_per_component_result_get_prefix_scores.restype  = ctypes.c_int
lib.p4a_aom_per_component_result_get_prefix_scores.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
    ctypes.POINTER(ctypes.c_int32),
]
lib.p4a_aom_per_component_result_get_predictions.restype  = ctypes.c_int
lib.p4a_aom_per_component_result_get_predictions.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
    ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64),
]
