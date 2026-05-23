"""ctypes loader + raw symbol signatures.

Loader search order (highest priority first):

1.  $PLS4ALL_LIB_PATH                                  — explicit override.
2.  <site-packages>/pls4all.libs/libn4m-*.so*          — auditwheel-grafted
    sibling directory (Linux wheels). delocate creates an equivalent
    layout on macOS, and delvewheel uses pls4all.libs/ as well on Windows
    (since delvewheel 1.5).
3.  <site-packages>/pls4all/lib/libn4m*                — wheel layout when
    libn4m is bundled directly as package data (the cibuildwheel
    fallback when no repair tool runs).
4.  <repo-root>/build/{dev-release,dev-debug}/cpp/src  — developer
    convenience.
5.  System search path (ctypes.util.find_library / DLL search path).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent
_SIBLING_LIBS = _PACKAGE_DIR.parent / f"{_PACKAGE_DIR.name}.libs"
_REPO_ROOT = _PACKAGE_DIR.parents[3] if _PACKAGE_DIR.parents[3].name == "pls4all" else None

# auditwheel renames the bundled library to ``libn4m-<8hexhash>.so.1.16.0``,
# so we must accept the form ``libn4m*.so*`` in addition to ``libn4m.so*``.
_LIB_PATTERNS = ("libn4m*.so*", "libn4m*.dylib", "p4a*.dll", "libn4m*.dll")


def _glob_libs(directory: Path) -> list[Path]:
    """Return all libn4m candidates under `directory`, sorted for determinism."""
    found: list[Path] = []
    if not directory.is_dir():
        return found
    for pattern in _LIB_PATTERNS:
        found.extend(sorted(directory.glob(pattern)))
    return found


def _candidate_paths() -> list[Path]:
    paths: list[Path] = []

    # (1) Explicit override always wins.
    env = os.environ.get("PLS4ALL_LIB_PATH")
    if env:
        paths.append(Path(env))

    # (2) Auditwheel / delocate / delvewheel place the bundled library in a
    #     sibling `<package>.libs/` directory at install time. The library
    #     is content-addressed (e.g. libn4m-7c4d2a1f.so.1) — match by glob.
    paths.extend(_glob_libs(_SIBLING_LIBS))

    # (3) Direct wheel layout: package_data ships libn4m under pls4all/lib/.
    paths.extend(_glob_libs(_PACKAGE_DIR / "lib"))

    # (4) Developer convenience: repo-root build directory. Glob so any
    #     versioned SONAME the build produced is picked up without
    #     hard-coding a version string.
    if _REPO_ROOT is not None:
        for preset in ("dev-release", "dev-debug"):
            paths.extend(_glob_libs(_REPO_ROOT / "build" / preset / "cpp" / "src"))

    # (5) System search path (ctypes.util.find_library or DLL search path).
    if sys.platform.startswith("linux") or sys.platform == "darwin":
        sys_path = ctypes.util.find_library("p4a")
        if sys_path:
            paths.append(Path(sys_path))
    elif sys.platform == "win32":
        for name in ("p4a.dll", "libn4m.dll"):
            paths.append(Path(name))
    return paths


def _load_library() -> ctypes.CDLL:
    errors = []
    for path in _candidate_paths():
        try:
            return ctypes.CDLL(str(path))
        except OSError as exc:
            errors.append((path, exc))
    msg = ["could not locate libn4m. Tried:"]
    for p, e in errors:
        msg.append(f"  - {p}: {e}")
    msg.append(
        "Set the PLS4ALL_LIB_PATH environment variable to the absolute path of "
        "libn4m.so / libn4m.dylib / p4a.dll."
    )
    raise ImportError("\n".join(msg))


lib = _load_library()

# ---- n4m_get_* (version queries) ----
lib.n4m_get_abi_version_major.restype = ctypes.c_uint32
lib.n4m_get_abi_version_minor.restype = ctypes.c_uint32
lib.n4m_get_abi_version_patch.restype = ctypes.c_uint32
lib.n4m_get_abi_version_int.restype   = ctypes.c_uint32
lib.n4m_get_version_string.restype    = ctypes.c_char_p
lib.n4m_get_build_info.restype        = ctypes.c_char_p
lib.n4m_get_git_revision.restype      = ctypes.c_char_p

# ---- n4m_status / dtype / backend ----
lib.n4m_status_to_string.restype  = ctypes.c_char_p
lib.n4m_status_to_string.argtypes = [ctypes.c_int]

lib.n4m_dtype_size.restype  = ctypes.c_size_t
lib.n4m_dtype_size.argtypes = [ctypes.c_int]
lib.n4m_dtype_to_string.restype  = ctypes.c_char_p
lib.n4m_dtype_to_string.argtypes = [ctypes.c_int]

lib.n4m_backend_is_available.restype  = ctypes.c_int
lib.n4m_backend_is_available.argtypes = [ctypes.c_int]
lib.n4m_backend_to_string.restype  = ctypes.c_char_p
lib.n4m_backend_to_string.argtypes = [ctypes.c_int]

# ---- n4m_context_* ----
lib.n4m_context_create.restype  = ctypes.c_int
lib.n4m_context_create.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
lib.n4m_context_destroy.restype  = None
lib.n4m_context_destroy.argtypes = [ctypes.c_void_p]
lib.n4m_context_set_seed.restype  = ctypes.c_int
lib.n4m_context_set_seed.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
lib.n4m_context_get_seed.restype  = ctypes.c_int
lib.n4m_context_get_seed.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
lib.n4m_context_set_backend.restype  = ctypes.c_int
lib.n4m_context_set_backend.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.n4m_context_get_backend.restype  = ctypes.c_int
lib.n4m_context_get_backend.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
lib.n4m_context_set_num_threads.restype  = ctypes.c_int
lib.n4m_context_set_num_threads.argtypes = [ctypes.c_void_p, ctypes.c_int32]
lib.n4m_context_get_num_threads.restype  = ctypes.c_int
lib.n4m_context_get_num_threads.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32)]
lib.n4m_context_last_error.restype  = ctypes.c_char_p
lib.n4m_context_last_error.argtypes = [ctypes.c_void_p]
lib.n4m_context_clear_error.restype  = None
lib.n4m_context_clear_error.argtypes = [ctypes.c_void_p]

# ---- n4m_config_* (a representative subset; Phase 2 fills out the rest) ----
lib.n4m_config_create.restype  = ctypes.c_int
lib.n4m_config_create.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
lib.n4m_config_destroy.restype  = None
lib.n4m_config_destroy.argtypes = [ctypes.c_void_p]
lib.n4m_config_set_n_components.restype  = ctypes.c_int
lib.n4m_config_set_n_components.argtypes = [ctypes.c_void_p, ctypes.c_int32]
lib.n4m_config_get_n_components.restype  = ctypes.c_int
lib.n4m_config_get_n_components.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32)]
lib.n4m_config_set_tol.restype  = ctypes.c_int
lib.n4m_config_set_tol.argtypes = [ctypes.c_void_p, ctypes.c_double]
lib.n4m_config_get_tol.restype  = ctypes.c_int
lib.n4m_config_get_tol.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
lib.n4m_config_set_algorithm.restype  = ctypes.c_int
lib.n4m_config_set_algorithm.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.n4m_config_get_algorithm.restype  = ctypes.c_int
lib.n4m_config_get_algorithm.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
lib.n4m_config_set_solver.restype  = ctypes.c_int
lib.n4m_config_set_solver.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.n4m_config_get_solver.restype  = ctypes.c_int
lib.n4m_config_get_solver.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
lib.n4m_config_set_deflation.restype  = ctypes.c_int
lib.n4m_config_set_deflation.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.n4m_config_get_deflation.restype  = ctypes.c_int
lib.n4m_config_get_deflation.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
lib.n4m_config_set_center_x.restype  = ctypes.c_int
lib.n4m_config_set_center_x.argtypes = [ctypes.c_void_p, ctypes.c_int32]
lib.n4m_config_get_center_x.restype  = ctypes.c_int
lib.n4m_config_get_center_x.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32)]
lib.n4m_config_set_scale_x.restype   = ctypes.c_int
lib.n4m_config_set_scale_x.argtypes  = [ctypes.c_void_p, ctypes.c_int32]
lib.n4m_config_get_scale_x.restype   = ctypes.c_int
lib.n4m_config_get_scale_x.argtypes  = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32)]
lib.n4m_config_set_center_y.restype  = ctypes.c_int
lib.n4m_config_set_center_y.argtypes = [ctypes.c_void_p, ctypes.c_int32]
lib.n4m_config_get_center_y.restype  = ctypes.c_int
lib.n4m_config_get_center_y.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32)]
lib.n4m_config_set_scale_y.restype   = ctypes.c_int
lib.n4m_config_set_scale_y.argtypes  = [ctypes.c_void_p, ctypes.c_int32]
lib.n4m_config_get_scale_y.restype   = ctypes.c_int
lib.n4m_config_get_scale_y.argtypes  = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32)]
lib.n4m_config_set_store_scores.restype  = ctypes.c_int
lib.n4m_config_set_store_scores.argtypes = [ctypes.c_void_p, ctypes.c_int32]
lib.n4m_config_get_store_scores.restype  = ctypes.c_int
lib.n4m_config_get_store_scores.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32)]
lib.n4m_config_set_sparse_simpls_legacy.restype  = ctypes.c_int
lib.n4m_config_set_sparse_simpls_legacy.argtypes = [ctypes.c_void_p, ctypes.c_int32]
lib.n4m_config_get_sparse_simpls_legacy.restype  = ctypes.c_int
lib.n4m_config_get_sparse_simpls_legacy.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32)]
lib.n4m_config_set_robust_pls_legacy.restype  = ctypes.c_int
lib.n4m_config_set_robust_pls_legacy.argtypes = [ctypes.c_void_p, ctypes.c_int32]
lib.n4m_config_get_robust_pls_legacy.restype  = ctypes.c_int
lib.n4m_config_get_robust_pls_legacy.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32)]
lib.n4m_config_set_approximate_press_legacy.restype  = ctypes.c_int
lib.n4m_config_set_approximate_press_legacy.argtypes = [ctypes.c_void_p, ctypes.c_int32]
lib.n4m_config_get_approximate_press_legacy.restype  = ctypes.c_int
lib.n4m_config_get_approximate_press_legacy.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32)]

# ---- n4m_operator_bank_* ----
lib.n4m_operator_bank_create.restype  = ctypes.c_int
lib.n4m_operator_bank_create.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
lib.n4m_operator_bank_destroy.restype  = None
lib.n4m_operator_bank_destroy.argtypes = [ctypes.c_void_p]
lib.n4m_operator_bank_add.restype  = ctypes.c_int
lib.n4m_operator_bank_add.argtypes = [
    ctypes.c_void_p, ctypes.c_int,
    ctypes.POINTER(ctypes.c_double), ctypes.c_int32,
]
lib.n4m_operator_bank_size.restype  = ctypes.c_int
lib.n4m_operator_bank_size.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32)]

# ---- n4m_validation_plan_* ----
lib.n4m_validation_plan_create.restype  = ctypes.c_int
lib.n4m_validation_plan_create.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
lib.n4m_validation_plan_destroy.restype  = None
lib.n4m_validation_plan_destroy.argtypes = [ctypes.c_void_p]
lib.n4m_validation_plan_set_n_samples.restype  = ctypes.c_int
lib.n4m_validation_plan_set_n_samples.argtypes = [ctypes.c_void_p, ctypes.c_int64]
lib.n4m_validation_plan_add_fold.restype  = ctypes.c_int
lib.n4m_validation_plan_add_fold.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int64), ctypes.c_int64,
    ctypes.POINTER(ctypes.c_int64), ctypes.c_int64,
]
lib.n4m_validation_plan_get_n_samples.restype  = ctypes.c_int
lib.n4m_validation_plan_get_n_samples.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64),
]
lib.n4m_validation_plan_get_n_folds.restype  = ctypes.c_int
lib.n4m_validation_plan_get_n_folds.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
]

# ---- n4m_matrix_view_t layout (must mirror cpp/include/pls4all/p4a.h) ----


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


# ---- n4m_aom_global_* ----
lib.n4m_aom_global_select.restype  = ctypes.c_int
lib.n4m_aom_global_select.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p),
]
lib.n4m_aom_global_result_destroy.restype  = None
lib.n4m_aom_global_result_destroy.argtypes = [ctypes.c_void_p]
lib.n4m_aom_global_result_get_n_operators.restype  = ctypes.c_int
lib.n4m_aom_global_result_get_n_operators.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
]
lib.n4m_aom_global_result_get_max_components.restype  = ctypes.c_int
lib.n4m_aom_global_result_get_max_components.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
]
lib.n4m_aom_global_result_get_selected_operator_index.restype  = ctypes.c_int
lib.n4m_aom_global_result_get_selected_operator_index.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
]
lib.n4m_aom_global_result_get_selected_n_components.restype  = ctypes.c_int
lib.n4m_aom_global_result_get_selected_n_components.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
]
lib.n4m_aom_global_result_get_best_score.restype  = ctypes.c_int
lib.n4m_aom_global_result_get_best_score.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_double),
]
lib.n4m_aom_global_result_get_operator_kinds.restype  = ctypes.c_int
lib.n4m_aom_global_result_get_operator_kinds.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
    ctypes.POINTER(ctypes.c_int32),
]
lib.n4m_aom_global_result_get_operator_scores.restype  = ctypes.c_int
lib.n4m_aom_global_result_get_operator_scores.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
    ctypes.POINTER(ctypes.c_int32),
]
lib.n4m_aom_global_result_get_rmse_curves.restype  = ctypes.c_int
lib.n4m_aom_global_result_get_rmse_curves.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
    ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
]
lib.n4m_aom_global_result_get_predictions.restype  = ctypes.c_int
lib.n4m_aom_global_result_get_predictions.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
    ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64),
]

# ---- n4m_model_* ----
lib.n4m_model_fit.restype  = ctypes.c_int
lib.n4m_model_fit.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.POINTER(ctypes.c_void_p),
]
lib.n4m_model_destroy.restype  = None
lib.n4m_model_destroy.argtypes = [ctypes.c_void_p]
lib.n4m_model_predict.restype  = ctypes.c_int
lib.n4m_model_predict.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
]
lib.n4m_model_transform.restype  = ctypes.c_int
lib.n4m_model_transform.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
]
lib.n4m_model_predict_alloc.restype  = ctypes.c_int
lib.n4m_model_predict_alloc.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(ctypes.c_void_p),
]
lib.n4m_model_transform_alloc.restype  = ctypes.c_int
lib.n4m_model_transform_alloc.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(ctypes.c_void_p),
]
lib.n4m_model_get_array.restype  = ctypes.c_int
lib.n4m_model_get_array.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
    ctypes.POINTER(ctypes.c_void_p),
]
lib.n4m_model_get_n_components.restype  = ctypes.c_int
lib.n4m_model_get_n_components.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
]
lib.n4m_model_get_n_features.restype  = ctypes.c_int
lib.n4m_model_get_n_features.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
]
lib.n4m_model_get_n_targets.restype  = ctypes.c_int
lib.n4m_model_get_n_targets.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
]

# ---- n4m_model_* serialization (N4M_SERIALIZATION_FORMAT_VERSION = 1) ----
lib.n4m_model_export_size.restype  = ctypes.c_int
lib.n4m_model_export_size.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
]
lib.n4m_model_export_to_buffer.restype  = ctypes.c_int
lib.n4m_model_export_to_buffer.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_size_t),
]
lib.n4m_model_import_from_buffer.restype  = ctypes.c_int
lib.n4m_model_import_from_buffer.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_void_p),
]
lib.n4m_serialization_inspect.restype  = ctypes.c_int
lib.n4m_serialization_inspect.argtypes = [
    ctypes.c_void_p, ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
    ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
]

# ---- n4m_array_* ----
lib.n4m_array_dtype.restype  = ctypes.c_int
lib.n4m_array_dtype.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
]
lib.n4m_array_shape.restype  = ctypes.c_int
lib.n4m_array_shape.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64),
]
lib.n4m_array_view.restype  = ctypes.c_int
lib.n4m_array_view.argtypes = [ctypes.c_void_p, ctypes.POINTER(MatrixView)]
lib.n4m_array_free.restype  = None
lib.n4m_array_free.argtypes = [ctypes.c_void_p]

# ---- n4m_aom_per_component_* ----
lib.n4m_aom_per_component_select.restype  = ctypes.c_int
lib.n4m_aom_per_component_select.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MatrixView), ctypes.POINTER(MatrixView),
    ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p),
]
lib.n4m_aom_per_component_result_destroy.restype  = None
lib.n4m_aom_per_component_result_destroy.argtypes = [ctypes.c_void_p]
lib.n4m_aom_per_component_result_get_n_operators.restype  = ctypes.c_int
lib.n4m_aom_per_component_result_get_n_operators.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
]
lib.n4m_aom_per_component_result_get_max_components.restype  = ctypes.c_int
lib.n4m_aom_per_component_result_get_max_components.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
]
lib.n4m_aom_per_component_result_get_selected_n_components.restype  = ctypes.c_int
lib.n4m_aom_per_component_result_get_selected_n_components.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
]
lib.n4m_aom_per_component_result_get_best_score.restype  = ctypes.c_int
lib.n4m_aom_per_component_result_get_best_score.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_double),
]
lib.n4m_aom_per_component_result_get_operator_kinds.restype  = ctypes.c_int
lib.n4m_aom_per_component_result_get_operator_kinds.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
    ctypes.POINTER(ctypes.c_int32),
]
lib.n4m_aom_per_component_result_get_selected_operator_indices.restype  = ctypes.c_int
lib.n4m_aom_per_component_result_get_selected_operator_indices.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)),
    ctypes.POINTER(ctypes.c_int32),
]
lib.n4m_aom_per_component_result_get_component_scores.restype  = ctypes.c_int
lib.n4m_aom_per_component_result_get_component_scores.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
    ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
]
lib.n4m_aom_per_component_result_get_prefix_scores.restype  = ctypes.c_int
lib.n4m_aom_per_component_result_get_prefix_scores.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
    ctypes.POINTER(ctypes.c_int32),
]
lib.n4m_aom_per_component_result_get_predictions.restype  = ctypes.c_int
lib.n4m_aom_per_component_result_get_predictions.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
    ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64),
]
