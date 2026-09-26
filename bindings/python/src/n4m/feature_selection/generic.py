# SPDX-License-Identifier: CECILL-2.1
"""Closed, native selector dispatcher for portable n4m recipes.

The argument order and defaults mirror the R ``n4m_method`` C dispatcher.
Each fit calls the public libn4m C ABI; no sklearn selector fallback or
host-language numerical selection is involved.
"""

from __future__ import annotations

import ctypes
import math
from typing import Any

import numpy as np

from n4m._impl import native as _native
from n4m._impl.selection import _BaseSelector, _check_X_y

_WIDTH = "input_width"
_COMPONENTS = "n_components"
_REQUIRED = "required"

# Ordered ABI arguments: (name, scalar type, native dispatcher default).
# ``required`` is stricter than the C API for top_k and stochastic seeds:
# portable recipes must state these values explicitly.
_SPECS: dict[str, tuple[tuple[str, str, Any], ...]] = {
    "spa_select": (("top_k", "int", _REQUIRED),),
    "cars_select": (("n_iterations", "int", 50), ("min_features", "int", 5)),
    "interval_select": (("interval_width", "int", 10), ("step", "int", 1)),
    "stability_select": (("top_k", "int", _REQUIRED),),
    "uve_select": (
        ("noise_features", "int", _WIDTH),
        ("noise_seed", "seed", _REQUIRED),
    ),
    "random_frog_select": (
        ("n_iterations", "int", 100),
        ("initial_size", "int", 30),
        ("min_size", "int", _COMPONENTS),
        ("max_size", "int", _WIDTH),
        ("top_k", "int", _REQUIRED),
        ("seed", "seed", _REQUIRED),
    ),
    "scars_select": (
        ("n_iterations", "int", 50),
        ("min_features", "int", 5),
        ("sample_fraction", "float", 0.8),
        ("seed", "seed", _REQUIRED),
    ),
    "ga_select": (
        ("n_generations", "int", 50),
        ("population_size", "int", 50),
        ("min_features", "int", _COMPONENTS),
        ("max_features", "int", _WIDTH),
        ("mutation_rate", "float", 0.01),
        ("seed", "seed", _REQUIRED),
    ),
    "pso_select": (
        ("n_swarm", "int", 30),
        ("n_iterations", "int", 50),
        ("w", "float", 0.729),
        ("c1", "float", 1.494),
        ("c2", "float", 1.494),
        ("v_max", "float", 4.0),
        ("seed", "seed", _REQUIRED),
    ),
    "vissa_select": (
        ("n_iterations", "int", 20),
        ("n_submodels", "int", 100),
        ("ratio_kept", "float", 0.1),
        ("threshold", "float", 0.5),
        ("floor_probability", "float", 0.01),
        ("seed", "seed", _REQUIRED),
    ),
    "shaving_select": (
        ("n_steps", "int", 10),
        ("min_features", "int", 5),
        ("shave_fraction", "float", 0.1),
    ),
    "bve_select": (("n_steps", "int", 10), ("min_features", "int", 5)),
    "t2_select": (
        ("alpha_thresholds", "vector", _REQUIRED),
        ("min_selected", "int", _COMPONENTS),
    ),
    "wvc_select": (("top_k", "int", _REQUIRED), ("normalize", "bool", True)),
    "wvc_threshold_select": (
        ("normalize", "bool", True),
        ("threshold", "float", 0.0),
        ("threshold_factor", "float", 1.0),
        ("min_selected", "int", 1),
    ),
    "emcuve_select": (
        ("noise_features", "int", _WIDTH),
        ("noise_seed", "seed", _REQUIRED),
        ("n_ensembles", "int", 5),
        ("vote_threshold", "float", 0.5),
    ),
    "randomization_select": (
        ("n_permutations", "int", 100),
        ("randomization_seed", "seed", _REQUIRED),
        ("alpha", "float", 0.05),
    ),
    "bipls_select": (("interval_width", "int", 10), ("min_intervals", "int", 1)),
    "sipls_select": (("interval_width", "int", 10), ("combination_size", "int", 2)),
    "rep_select": (
        ("n_steps", "int", 10),
        ("min_features", "int", 5),
        ("remove_count", "int", 1),
    ),
    "ipw_select": (
        ("n_iterations", "int", 10),
        ("top_k", "int", _REQUIRED),
        ("damping", "float", 0.5),
        ("weight_floor", "float", 1e-6),
    ),
    "st_select": (
        ("thresholds", "vector", _REQUIRED),
        ("min_selected", "int", _COMPONENTS),
    ),
    "iriv_select": (("max_rounds", "int", 20), ("seed", "seed", _REQUIRED)),
    "irf_select": (
        ("n_iterations", "int", 100),
        ("window_size", "int", 10),
        ("initial_intervals", "int", 10),
        ("top_k", "int", _REQUIRED),
        ("seed", "seed", _REQUIRED),
    ),
    "vip_spa_select": (("vip_threshold", "float", 0.3), ("top_k", "int", _REQUIRED)),
}

_NO_PLAN = {
    "spa_select",
    "wvc_select",
    "wvc_threshold_select",
    "randomization_select",
    "vip_spa_select",
}
_NO_CONFIG = {"wvc_select", "wvc_threshold_select"}
_SEED_MAX = 2**31 - 1  # R's current public constructor stores signed integers.


def _validate_value(name: str, kind: str, value: Any) -> Any:
    if kind in {"int", "seed"}:
        if (
            type(value) is not int
            or value < (0 if kind == "seed" else 1)
            or value > _SEED_MAX
        ):
            raise ValueError(
                f"{name} must be a bounded {'nonnegative' if kind == 'seed' else 'positive'} integer"
            )
        return value
    if kind == "bool":
        if type(value) is not bool:
            raise ValueError(f"{name} must be a boolean")
        return value
    if kind == "float":
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite numeric")
        return float(value)
    if kind == "vector":
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError(f"{name} must be a nonempty numeric array")
        return [_validate_value(name, "float", item) for item in value]
    raise RuntimeError(f"unrecognized selector parameter type {kind}")


class Selector(_BaseSelector):
    """Native n4m selector with a closed, portable method vocabulary.

    ``method_params`` keys match the C ABI argument names. Every stochastic
    seed and top-k/threshold input is explicit; omitted optional arguments
    use the same C dispatcher defaults as the R ``n4m_method`` binding.
    Three contiguous native validation folds are used where required.
    """

    def __init__(
        self,
        method: str,
        n_components: int = 2,
        method_params: dict[str, Any] | None = None,
    ) -> None:
        self.method = method
        self.n_components = n_components
        self.method_params = method_params

    def _arguments(
        self, width: int
    ) -> tuple[dict[str, Any], list[Any], list[np.ndarray]]:
        if type(self.method) is not str or self.method not in _SPECS:
            raise ValueError(f"unsupported n4m selector method: {self.method!r}")
        if (
            type(self.n_components) is not int
            or self.n_components < 1
            or self.n_components > min(width, _SEED_MAX)
        ):
            raise ValueError("n_components exceeds the input width")
        params = {} if self.method_params is None else self.method_params
        if not isinstance(params, dict) or any(type(key) is not str for key in params):
            raise ValueError("method_params must be a named mapping")
        spec = _SPECS[self.method]
        allowed = {name for name, _, _ in spec}
        if set(params) - allowed:
            raise ValueError("unsupported n4m selector parameter")
        resolved: dict[str, Any] = {}
        arguments: list[Any] = []
        buffers: list[np.ndarray] = []
        for name, kind, default in spec:
            if name in params:
                value = params[name]
            elif default == _REQUIRED:
                raise ValueError(f"{self.method} requires explicit {name}")
            elif default == _WIDTH:
                value = width
            elif default == _COMPONENTS:
                value = self.n_components
            else:
                value = default
            value = _validate_value(name, kind, value)
            resolved[name] = value
            if kind == "vector":
                buffer = np.ascontiguousarray(value, dtype=np.float64)
                buffers.append(buffer)
                arguments.extend(
                    (
                        buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        ctypes.c_int64(buffer.size),
                    )
                )
            elif kind == "float":
                arguments.append(ctypes.c_double(value))
            elif kind == "seed":
                arguments.append(ctypes.c_uint64(value))
            else:
                arguments.append(ctypes.c_int32(int(value)))
        if self.method in _NO_CONFIG:
            arguments.insert(0, ctypes.c_int32(self.n_components))
        for key in (
            "top_k",
            "min_features",
            "min_size",
            "max_size",
            "max_features",
            "initial_size",
            "min_selected",
            "interval_width",
        ):
            if key in resolved and resolved[key] > width:
                raise ValueError(f"{key} exceeds input width")
        if (
            "min_size" in resolved
            and "max_size" in resolved
            and resolved["min_size"] > resolved["max_size"]
        ):
            raise ValueError("min_size exceeds max_size")
        if (
            "min_features" in resolved
            and "max_features" in resolved
            and resolved["min_features"] > resolved["max_features"]
        ):
            raise ValueError("min_features exceeds max_features")
        return resolved, arguments, buffers

    def fit(self, X: Any, y: Any) -> Selector:
        X_arr, y_arr = _check_X_y(self, X, y)
        if (
            y_arr.ndim != 1
            or y_arr.shape[0] != X_arr.shape[0]
            or not np.isfinite(X_arr).all()
            or not np.isfinite(y_arr).all()
        ):
            raise ValueError("selector requires finite aligned one-dimensional targets")
        _, arguments, buffers = self._arguments(X_arr.shape[1])
        fields = _native._run_select(
            f"n4m_feature_selection_{self.method}",
            X_arr,
            y_arr,
            *arguments,
            n_components=self.n_components,
            plan_folds=None if self.method in _NO_PLAN else 3,
            with_config=self.method not in _NO_CONFIG,
        )
        del buffers  # Keep vector argument backing arrays alive through the ABI call.
        indices = _native._selected_indices(fields)
        if (
            indices.ndim != 1
            or indices.size < 1
            or len(np.unique(indices)) != indices.size
            or np.any(indices < 0)
            or np.any(indices >= X_arr.shape[1])
        ):
            raise ValueError("native selector returned invalid selected_indices")
        self._commit_fit(indices=indices)
        return self


SELECTOR_METHODS = tuple(_SPECS)

__all__ = ["SELECTOR_METHODS", "Selector"]
