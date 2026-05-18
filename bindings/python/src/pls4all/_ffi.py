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

    # Developer convenience: repo-root build directory. We glob so we pick up
    # whatever versioned SONAME the build produced (libp4a.so.${ABI_MAJOR}
    # and libp4a.so.${ABI_VERSION}) without hard-coding any version string.
    if _REPO_ROOT is not None:
        for preset in ("dev-release", "dev-debug"):
            build_lib = _REPO_ROOT / "build" / preset / "cpp" / "src"
            if not build_lib.is_dir():
                continue
            for pattern in ("libp4a.so*", "libp4a*.dylib", "p4a.dll", "libp4a.dll"):
                paths.extend(sorted(build_lib.glob(pattern)))

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

# ---- p4a_model_* ----
lib.p4a_model_fit.restype  = ctypes.c_int
lib.p4a_model_fit.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.POINTER(ctypes.c_void_p),
]
lib.p4a_model_destroy.restype  = None
lib.p4a_model_destroy.argtypes = [ctypes.c_void_p]
lib.p4a_model_predict.restype  = ctypes.c_int
lib.p4a_model_predict.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
]
lib.p4a_model_transform.restype  = ctypes.c_int
lib.p4a_model_transform.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
]
lib.p4a_model_predict_alloc.restype  = ctypes.c_int
lib.p4a_model_predict_alloc.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(ctypes.c_void_p),
]
lib.p4a_model_transform_alloc.restype  = ctypes.c_int
lib.p4a_model_transform_alloc.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(ctypes.c_void_p),
]
lib.p4a_model_get_array.restype  = ctypes.c_int
lib.p4a_model_get_array.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
    ctypes.POINTER(ctypes.c_void_p),
]
lib.p4a_model_get_n_components.restype  = ctypes.c_int
lib.p4a_model_get_n_components.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
]
lib.p4a_model_get_n_features.restype  = ctypes.c_int
lib.p4a_model_get_n_features.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
]
lib.p4a_model_get_n_targets.restype  = ctypes.c_int
lib.p4a_model_get_n_targets.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
]

# ---- p4a_model_* serialization (P4A_SERIALIZATION_FORMAT_VERSION = 1) ----
lib.p4a_model_export_size.restype  = ctypes.c_int
lib.p4a_model_export_size.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
]
lib.p4a_model_export_to_buffer.restype  = ctypes.c_int
lib.p4a_model_export_to_buffer.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_size_t),
]
lib.p4a_model_import_from_buffer.restype  = ctypes.c_int
lib.p4a_model_import_from_buffer.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_void_p),
]
lib.p4a_serialization_inspect.restype  = ctypes.c_int
lib.p4a_serialization_inspect.argtypes = [
    ctypes.c_void_p, ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
    ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
]

# ---- p4a_array_* ----
lib.p4a_array_dtype.restype  = ctypes.c_int
lib.p4a_array_dtype.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
]
lib.p4a_array_shape.restype  = ctypes.c_int
lib.p4a_array_shape.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64),
]
lib.p4a_array_view.restype  = ctypes.c_int
lib.p4a_array_view.argtypes = [ctypes.c_void_p, ctypes.POINTER(MatrixView)]
lib.p4a_array_free.restype  = None
lib.p4a_array_free.argtypes = [ctypes.c_void_p]

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
