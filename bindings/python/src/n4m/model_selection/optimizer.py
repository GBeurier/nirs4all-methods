# SPDX-License-Identifier: CECILL-2.1
"""Idiomatic Python wrapper over the native ask/tell HPO optimizer (libn4m
``optimization`` role, ABI 2.1).

The search algorithm runs in ``libn4m`` (identical across bindings); this module
is a thin marshaller. Typical use::

    from n4m.model_selection.optimizer import Optimizer, SearchSpace, Sampler

    space = SearchSpace()
    space.add_int("n_components", 1, 30)
    space.add_float("alpha", 1e-4, 1.0, log=True)

    opt = Optimizer(space, sampler=Sampler.TPE, seed=42)
    for _ in range(60):
        trial = opt.ask()
        score = evaluate(trial.get_int("n_components"), trial.get_float("alpha"))
        opt.tell(trial.id, score)
    best, best_score = opt.best()
"""
from __future__ import annotations

import ctypes
from ctypes import POINTER, byref, c_char_p, c_double, c_int, c_int32, c_int64, c_void_p, cast
from enum import IntEnum

from .._context import Context
from .._errors import check
from .._ffi import lib
from .._types import OptimizerOptions

_N4M_OK = 0
_N4M_ERR_NOT_FITTED = 6


class Sampler(IntEnum):
    RANDOM = 0
    SOBOL = 1
    LHS = 2
    TERNARY = 3
    GA = 4
    PSO = 5
    CMAES = 6
    TPE = 7
    GP_EI = 8


class Pruner(IntEnum):
    NONE = 0
    MEDIAN = 1
    ASHA = 2
    HYPERBAND = 3
    RACING = 4


class Direction(IntEnum):
    AUTO = 0
    MINIMIZE = 1
    MAXIMIZE = 2


class EvalMode(IntEnum):
    BEST = 0
    MEAN = 1
    ROBUST_BEST = 2


class Metric(IntEnum):
    RMSE = 0
    MSE = 1
    MAE = 2
    R2 = 3
    ACCURACY = 16
    BALANCED_ACCURACY = 17
    F1 = 18
    LOGLOSS = 19


class TrialStatus(IntEnum):
    RUNNING = 0
    COMPLETED = 1
    PRUNED = 2
    FAILED = 3


class _CatType(IntEnum):
    STR = 0
    INT = 1
    FLOAT = 2
    BOOL = 3


class ConstraintKind(IntEnum):
    MUTEX_GROUP = 0
    REQUIRES = 1
    EXCLUDE = 2
    CONDITION_IN = 3
    CONDITION_NOT_IN = 4


class SearchSpace:
    """A typed hyperparameter search space."""

    def __init__(self) -> None:
        handle = c_void_p()
        check(lib.n4m_search_space_create(byref(handle)), "n4m_search_space_create")
        self._handle = handle

    def add_int(self, name: str, low: int, high: int, step: int = 1, log: bool = False) -> "SearchSpace":
        check(lib.n4m_search_space_add_int(self._handle, name.encode(), low, high, step, int(log)),
              "n4m_search_space_add_int")
        return self

    def add_float(self, name: str, low: float, high: float, step: float = 0.0,
                  log: bool = False) -> "SearchSpace":
        check(lib.n4m_search_space_add_float(self._handle, name.encode(), low, high, step, int(log)),
              "n4m_search_space_add_float")
        return self

    def add_categorical(self, name: str, choices) -> "SearchSpace":
        choices = list(choices)
        if choices and all(isinstance(c, bool) for c in choices):
            ctype, arr = _CatType.BOOL, (c_int32 * len(choices))(*[1 if c else 0 for c in choices])
        elif choices and all(isinstance(c, int) for c in choices):
            ctype, arr = _CatType.INT, (c_int64 * len(choices))(*choices)
        elif choices and all(isinstance(c, (int, float)) for c in choices):
            ctype, arr = _CatType.FLOAT, (c_double * len(choices))(*[float(c) for c in choices])
        else:
            ctype = _CatType.STR
            arr = (c_char_p * len(choices))(*[str(c).encode() for c in choices])
        check(lib.n4m_search_space_add_categorical(self._handle, name.encode(), int(ctype),
                                                    cast(arr, c_void_p), len(choices)),
              "n4m_search_space_add_categorical")
        return self

    def add_ordinal(self, name: str, values) -> "SearchSpace":
        values = [float(v) for v in values]
        arr = (c_double * len(values))(*values)
        check(lib.n4m_search_space_add_ordinal(self._handle, name.encode(), arr, len(values)),
              "n4m_search_space_add_ordinal")
        return self

    def add_sorted_tuple(self, name: str, length: int, low: float, high: float,
                         element_is_int: bool = False) -> "SearchSpace":
        check(lib.n4m_search_space_add_sorted_tuple(self._handle, name.encode(), length, low, high,
                                                    int(element_is_int)),
              "n4m_search_space_add_sorted_tuple")
        return self

    def add_constraint(self, kind: ConstraintKind, param_refs, label_refs=None) -> "SearchSpace":
        n = len(param_refs)
        prefs = (c_char_p * n)(*[p.encode() for p in param_refs])
        labels = list(label_refs) if label_refs is not None else [""] * n
        lrefs = (c_char_p * n)(*[(l or "").encode() for l in labels])
        check(lib.n4m_search_space_add_constraint(self._handle, int(kind), prefs, lrefs, n),
              "n4m_search_space_add_constraint")
        return self

    def num_params(self) -> int:
        out = c_int32()
        check(lib.n4m_search_space_num_params(self._handle, byref(out)), "n4m_search_space_num_params")
        return int(out.value)

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle:
            lib.n4m_search_space_destroy(handle)
            self._handle = None


class Trial:
    """A proposed configuration (borrowed; owned by the optimizer)."""

    def __init__(self, handle: c_void_p) -> None:
        self._handle = handle

    @property
    def id(self) -> int:
        out = c_int64()
        check(lib.n4m_trial_get_id(self._handle, byref(out)), "n4m_trial_get_id")
        return int(out.value)

    def get_int(self, name: str) -> int:
        out = c_int64()
        check(lib.n4m_trial_get_int(self._handle, name.encode(), byref(out)), "n4m_trial_get_int")
        return int(out.value)

    def get_float(self, name: str) -> float:
        out = c_double()
        check(lib.n4m_trial_get_float(self._handle, name.encode(), byref(out)), "n4m_trial_get_float")
        return float(out.value)

    def get_category(self, name: str):
        idx = c_int32()
        label = c_char_p()
        check(lib.n4m_trial_get_category(self._handle, name.encode(), byref(idx), byref(label)),
              "n4m_trial_get_category")
        return int(idx.value), (label.value.decode() if label.value else None)

    def is_active(self, name: str) -> bool:
        out = c_int32()
        check(lib.n4m_trial_is_active(self._handle, name.encode(), byref(out)), "n4m_trial_is_active")
        return bool(out.value)

    @property
    def status(self) -> TrialStatus:
        out = c_int()
        check(lib.n4m_trial_get_status(self._handle, byref(out)), "n4m_trial_get_status")
        return TrialStatus(out.value)

    @property
    def rung(self) -> int:
        out = c_int32()
        check(lib.n4m_trial_get_rung(self._handle, byref(out)), "n4m_trial_get_rung")
        return int(out.value)

    @property
    def duration(self) -> float:
        out = c_double()
        check(lib.n4m_trial_get_duration(self._handle, byref(out)), "n4m_trial_get_duration")
        return float(out.value)


class Optimizer:
    """Native ask/tell hyperparameter optimizer."""

    def __init__(self, space: SearchSpace, *, sampler: Sampler = Sampler.RANDOM,
                 pruner: Pruner = Pruner.NONE, direction: Direction = Direction.AUTO,
                 metric: Metric = Metric.RMSE, eval_mode: EvalMode = EvalMode.MEAN,
                 n_startup_trials: int = 10, seed: int = 0, timeout_seconds: float = 0.0,
                 max_resource: int = 0, reduction_factor: int = 0,
                 context: Context | None = None) -> None:
        self._ctx = context or Context()
        self._space = space  # keep a reference alive
        opts = OptimizerOptions()
        lib.n4m_optimizer_options_init(byref(opts))
        opts.sampler = int(sampler)
        opts.pruner = int(pruner)
        opts.direction = int(direction)
        opts.metric = int(metric)
        opts.eval_mode = int(eval_mode)
        opts.n_startup_trials = int(n_startup_trials)
        opts.seed = int(seed)
        opts.timeout_seconds = float(timeout_seconds)
        opts.max_resource = int(max_resource)          # hyperband top rung (0 = derive)
        opts.reduction_factor = int(reduction_factor)  # asha/hyperband eta (0 = default 3)
        self._opts = opts
        handle = c_void_p()
        check(lib.n4m_optimizer_create(self._ctx._handle, space._handle, byref(opts), byref(handle)),
              "n4m_optimizer_create")
        self._handle = handle

    def ask(self) -> Trial:
        th = c_void_p()
        check(lib.n4m_optimizer_ask(self._handle, byref(th)), "n4m_optimizer_ask")
        return Trial(th)

    def tell(self, trial_id: int, score: float) -> None:
        check(lib.n4m_optimizer_tell(self._handle, trial_id, score), "n4m_optimizer_tell")

    def tell_result(self, trial_id: int, status: TrialStatus, score: float = 0.0,
                    error: str | None = None) -> None:
        check(lib.n4m_optimizer_tell_result(self._handle, trial_id, int(status), score,
                                            error.encode() if error else None),
              "n4m_optimizer_tell_result")

    def tell_intermediate(self, trial_id: int, step: int, score: float) -> bool:
        out = c_int32()
        check(lib.n4m_optimizer_tell_intermediate(self._handle, trial_id, step, score, byref(out)),
              "n4m_optimizer_tell_intermediate")
        return bool(out.value)

    def enqueue(self, params: dict) -> None:
        names = list(params.keys())
        n = len(names)
        cnames = (c_char_p * n)(*[k.encode() for k in names])
        cvals = (c_double * n)(*[float(params[k]) for k in names])
        check(lib.n4m_optimizer_enqueue(self._handle, cnames, cvals, n), "n4m_optimizer_enqueue")

    def best(self):
        th = c_void_p()
        score = c_double()
        status = lib.n4m_optimizer_best(self._handle, byref(th), byref(score))
        if status == _N4M_ERR_NOT_FITTED:
            return None
        check(status, "n4m_optimizer_best")
        return Trial(th), float(score.value)

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle:
            lib.n4m_optimizer_destroy(handle)
            self._handle = None


__all__ = [
    "SearchSpace", "Trial", "Optimizer",
    "Sampler", "Pruner", "Direction", "EvalMode", "Metric", "TrialStatus", "ConstraintKind",
]
