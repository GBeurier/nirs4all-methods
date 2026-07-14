# SPDX-License-Identifier: CECILL-2.1
"""Idiomatic Python wrapper over the native ask/tell HPO optimizer (libn4m
``optimization`` role, ABI 2.2).

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

import math
from ctypes import (
    POINTER,
    byref,
    c_char_p,
    c_double,
    c_int,
    c_int32,
    c_int64,
    c_uint8,
    c_void_p,
    cast,
    string_at,
)
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from .._context import Context
from .._errors import N4MError, PartialBatchError, check
from .._ffi import lib
from .._matrix import numpy_to_view
from .._types import Dtype, MatrixView, OptimizerOptions, Status
from .._validation import ValidationPlan

_N4M_ERR_NOT_FITTED = 6
_TRIAL_ERROR_PREFIX = "n4m.error.v1"
_TRIAL_TRACE_FORMAT_VERSION = 1


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


class Algorithm(IntEnum):
    """Native ``n4m_algorithm_t`` values (exact C ABI numbering).

    :func:`finetune_estimator` drives only the generic regression routes
    ``PLS_REGRESSION``, ``PLS_CANONICAL``, ``PLS_SVD``, ``OPLS``, ``SPARSE_PLS``
    and ``PCR``; every other member is rejected with ``UNSUPPORTED``.
    """

    PLS_REGRESSION = 0
    PLS_CANONICAL = 1
    PLS_SVD = 2
    PLS_DA = 3
    OPLS = 4
    OPLS_DA = 5
    SPARSE_PLS = 6
    MB_PLS = 7
    LW_PLS = 8
    AOM_PLS = 9
    PCR = 10


class TrialStatus(IntEnum):
    RUNNING = 0
    COMPLETED = 1
    PRUNED = 2
    FAILED = 3
    CANCELLED = 4


class ParameterKind(IntEnum):
    INT = 0
    FLOAT = 1
    LOG_INT = 2
    LOG_FLOAT = 3
    CATEGORICAL = 4
    ORDINAL = 5
    SORTED_TUPLE = 6


class CategoryType(IntEnum):
    STR = 0
    INT = 1
    FLOAT = 2
    BOOL = 3


@dataclass(frozen=True)
class TrialError:
    """Stable diagnostic persisted for a failed or cancelled trial."""

    code: str
    message: str
    retryable: bool = False

    def _wire(self) -> bytes:
        if not isinstance(self.code, str) or not _valid_error_code(self.code):
            raise ValueError(
                "TrialError.code must contain 2-64 uppercase letters, digits, or '_'"
            )
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("TrialError.message must not be empty")
        if "\x00" in self.message:
            raise ValueError("TrialError.message must not contain NUL")
        if not isinstance(self.retryable, bool):
            raise TypeError("TrialError.retryable must be a bool")
        retryable = "1" if self.retryable else "0"
        return f"{_TRIAL_ERROR_PREFIX}|{self.code}|{retryable}|{self.message}".encode()


@dataclass(frozen=True)
class TrialParameter:
    """One parameter value in an owning :class:`TrialRecord` snapshot."""

    value: float
    kind: ParameterKind
    category_index: int | None
    category_label: str | None
    category_type: CategoryType | None
    integer: bool
    active: bool

    @property
    def decoded_value(self) -> bool | int | float | str:
        """Return the typed value declared by the ordered search space."""

        if self.kind is ParameterKind.CATEGORICAL:
            if self.category_type is CategoryType.STR:
                return self.category_label or ""
            if self.category_type is CategoryType.INT:
                return int(self.value)
            if self.category_type is CategoryType.BOOL:
                return bool(int(self.value))
            return self.value
        if self.integer:
            return int(self.value)
        return self.value


@dataclass(frozen=True)
class IntermediateValue:
    """One immutable, strictly ordered intermediate objective report."""

    sequence: int
    step: int
    score: float
    should_prune: bool


@dataclass(frozen=True)
class TrialRecord:
    """Owning trial snapshot returned by :meth:`Optimizer.get_trials`."""

    id: int
    ask_sequence: int
    terminal_sequence: int | None
    params: dict[str, bool | int | float | str]
    param_details: dict[str, TrialParameter]
    status: TrialStatus
    score: float | None
    rung: int
    duration: float
    intermediates: tuple[IntermediateValue, ...]
    error: TrialError | None


@dataclass(frozen=True)
class FinetuneResult:
    """Owning result of :func:`finetune_estimator`.

    Every field is copied out of the native ``n4m_method_result_t`` before it is
    destroyed, so the result stays valid independently of any native handle.
    ``best_params`` carries the tuned axes of the best trial (``n_components`` as
    an ``int``, ``sparsity_lambda`` as a ``float`` when it was sampled), and
    ``trials`` is the full ordered trial trace.
    """

    best_score: float
    best_params: dict[str, int | float]
    metric: Metric
    estimator: Algorithm
    trials: tuple[TrialRecord, ...]
    timed_out: bool
    requested_trials: int


def _valid_error_code(code: str) -> bool:
    return (
        2 <= len(code) <= 64
        and "A" <= code[0] <= "Z"
        and all(
            char == "_" or "A" <= char <= "Z" or "0" <= char <= "9" for char in code
        )
    )


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

    def add_int(
        self, name: str, low: int, high: int, step: int = 1, log: bool = False
    ) -> "SearchSpace":
        self._require_open()
        check(
            lib.n4m_search_space_add_int(
                self._handle, name.encode(), low, high, step, int(log)
            ),
            "n4m_search_space_add_int",
        )
        return self

    def add_float(
        self, name: str, low: float, high: float, step: float = 0.0, log: bool = False
    ) -> "SearchSpace":
        self._require_open()
        check(
            lib.n4m_search_space_add_float(
                self._handle, name.encode(), low, high, step, int(log)
            ),
            "n4m_search_space_add_float",
        )
        return self

    def add_categorical(self, name: str, choices) -> "SearchSpace":
        self._require_open()
        choices = list(choices)
        if choices and all(isinstance(c, bool) for c in choices):
            ctype, arr = (
                CategoryType.BOOL,
                (c_int32 * len(choices))(*[1 if c else 0 for c in choices]),
            )
        elif choices and all(isinstance(c, int) for c in choices):
            ctype, arr = CategoryType.INT, (c_int64 * len(choices))(*choices)
        elif choices and all(isinstance(c, (int, float)) for c in choices):
            ctype, arr = (
                CategoryType.FLOAT,
                (c_double * len(choices))(*[float(c) for c in choices]),
            )
        else:
            ctype = CategoryType.STR
            arr = (c_char_p * len(choices))(*[str(c).encode() for c in choices])
        check(
            lib.n4m_search_space_add_categorical(
                self._handle,
                name.encode(),
                int(ctype),
                cast(arr, c_void_p),
                len(choices),
            ),
            "n4m_search_space_add_categorical",
        )
        return self

    def add_ordinal(self, name: str, values) -> "SearchSpace":
        self._require_open()
        values = [float(v) for v in values]
        arr = (c_double * len(values))(*values)
        check(
            lib.n4m_search_space_add_ordinal(
                self._handle, name.encode(), arr, len(values)
            ),
            "n4m_search_space_add_ordinal",
        )
        return self

    def add_sorted_tuple(
        self,
        name: str,
        length: int,
        low: float,
        high: float,
        element_is_int: bool = False,
    ) -> "SearchSpace":
        self._require_open()
        check(
            lib.n4m_search_space_add_sorted_tuple(
                self._handle, name.encode(), length, low, high, int(element_is_int)
            ),
            "n4m_search_space_add_sorted_tuple",
        )
        return self

    def add_constraint(
        self, kind: ConstraintKind, param_refs, label_refs=None
    ) -> "SearchSpace":
        self._require_open()
        n = len(param_refs)
        prefs = (c_char_p * n)(*[p.encode() for p in param_refs])
        labels = list(label_refs) if label_refs is not None else [""] * n
        lrefs = (c_char_p * n)(*[(label or "").encode() for label in labels])
        check(
            lib.n4m_search_space_add_constraint(
                self._handle, int(kind), prefs, lrefs, n
            ),
            "n4m_search_space_add_constraint",
        )
        return self

    def num_params(self) -> int:
        self._require_open()
        out = c_int32()
        check(
            lib.n4m_search_space_num_params(self._handle, byref(out)),
            "n4m_search_space_num_params",
        )
        return int(out.value)

    def _require_open(self) -> None:
        if not getattr(self, "_handle", None) or not self._handle.value:
            raise RuntimeError("SearchSpace is closed")

    def __copy__(self):
        raise TypeError("SearchSpace is not copyable")

    def __deepcopy__(self, _memo):
        raise TypeError("SearchSpace is not copyable")

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle and handle.value:
            lib.n4m_search_space_destroy(handle)
            self._handle = c_void_p()

    def __enter__(self) -> "SearchSpace":
        self._require_open()
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class Trial:
    """A proposed configuration (borrowed; owned by the optimizer)."""

    def __init__(self, handle: c_void_p, optimizer: "Optimizer") -> None:
        self._handle = handle
        self._optimizer = optimizer

    def _live_handle(self) -> c_void_p:
        self._optimizer._require_open()
        return self._handle

    @property
    def id(self) -> int:
        out = c_int64()
        check(lib.n4m_trial_get_id(self._live_handle(), byref(out)), "n4m_trial_get_id")
        return int(out.value)

    def get_int(self, name: str) -> int:
        out = c_int64()
        check(
            lib.n4m_trial_get_int(self._live_handle(), name.encode(), byref(out)),
            "n4m_trial_get_int",
        )
        return int(out.value)

    def get_float(self, name: str) -> float:
        out = c_double()
        check(
            lib.n4m_trial_get_float(self._live_handle(), name.encode(), byref(out)),
            "n4m_trial_get_float",
        )
        return float(out.value)

    def get_category(self, name: str):
        idx = c_int32()
        label = c_char_p()
        check(
            lib.n4m_trial_get_category(
                self._live_handle(), name.encode(), byref(idx), byref(label)
            ),
            "n4m_trial_get_category",
        )
        return int(idx.value), (label.value.decode() if label.value else None)

    def is_active(self, name: str) -> bool:
        out = c_int32()
        check(
            lib.n4m_trial_is_active(self._live_handle(), name.encode(), byref(out)),
            "n4m_trial_is_active",
        )
        return bool(out.value)

    @property
    def status(self) -> TrialStatus:
        out = c_int()
        check(
            lib.n4m_trial_get_status(self._live_handle(), byref(out)),
            "n4m_trial_get_status",
        )
        return TrialStatus(out.value)

    @property
    def rung(self) -> int:
        out = c_int32()
        check(
            lib.n4m_trial_get_rung(self._live_handle(), byref(out)),
            "n4m_trial_get_rung",
        )
        return int(out.value)

    @property
    def duration(self) -> float:
        out = c_double()
        check(
            lib.n4m_trial_get_duration(self._live_handle(), byref(out)),
            "n4m_trial_get_duration",
        )
        return float(out.value)


def _result_double_matrix(result: c_void_p, name: str) -> tuple[list[float], int, int]:
    data = POINTER(c_double)()
    rows = c_int64()
    cols = c_int64()
    check(
        lib.n4m_method_result_get_double_matrix(
            result, name.encode(), byref(data), byref(rows), byref(cols)
        ),
        f"n4m_method_result_get_double_matrix({name})",
    )
    size = int(rows.value * cols.value)
    return (
        [float(data[index]) for index in range(size)],
        int(rows.value),
        int(cols.value),
    )


def _result_int_vector(result: c_void_p, name: str) -> list[int]:
    data = POINTER(c_int32)()
    size = c_int32()
    check(
        lib.n4m_method_result_get_int_vector(
            result, name.encode(), byref(data), byref(size)
        ),
        f"n4m_method_result_get_int_vector({name})",
    )
    return [int(data[index]) for index in range(size.value)]


def _result_int64_vector(result: c_void_p, name: str) -> list[int]:
    data = POINTER(c_int64)()
    size = c_int64()
    check(
        lib.n4m_method_result_get_int64_vector(
            result, name.encode(), byref(data), byref(size)
        ),
        f"n4m_method_result_get_int64_vector({name})",
    )
    return [int(data[index]) for index in range(size.value)]


def _result_scalar(result: c_void_p, name: str) -> float:
    value = c_double()
    check(
        lib.n4m_method_result_get_scalar(result, name.encode(), byref(value)),
        f"n4m_method_result_get_scalar({name})",
    )
    return float(value.value)


def _maybe_result_scalar(result: c_void_p, name: str) -> float | None:
    """Return a named scalar, or ``None`` when the result does not carry it."""
    value = c_double()
    status = lib.n4m_method_result_get_scalar(result, name.encode(), byref(value))
    if status == Status.OK:
        return float(value.value)
    if status == Status.ERR_INVALID_ARGUMENT:  # documented missing-key status
        return None
    check(status, f"n4m_method_result_get_scalar({name})")
    raise AssertionError("unreachable")


def _result_enum_scalar(result: c_void_p, name: str, enum_type):
    """Decode an enum transported through the scalar result channel."""
    raw = _result_scalar(result, name)
    if not math.isfinite(raw) or raw != int(raw):
        raise RuntimeError(f"corrupt native finetune result: invalid {name}")
    try:
        return enum_type(int(raw))
    except ValueError as exc:
        raise RuntimeError(f"corrupt native finetune result: unknown {name}") from exc


def _result_bounded_int_scalar(
    result: c_void_p, name: str, *, low: int, high: int
) -> int:
    """Decode an exactly integral scalar within a closed range."""
    raw = _result_scalar(result, name)
    if not math.isfinite(raw) or raw != int(raw) or raw < low or raw > high:
        raise RuntimeError(f"corrupt native finetune result: invalid {name}")
    return int(raw)


def _decode_utf8_records(
    raw_bytes: list[int], offsets: list[int], expected_records: int, name: str
) -> list[str]:
    if len(offsets) != expected_records + 1 or not offsets or offsets[0] != 0:
        raise RuntimeError(f"corrupt native HPO trace: invalid {name} offsets")
    if offsets[-1] != len(raw_bytes) or any(
        left > right for left, right in zip(offsets, offsets[1:])
    ):
        raise RuntimeError(f"corrupt native HPO trace: inconsistent {name} offsets")
    if any(byte < 0 or byte > 255 for byte in raw_bytes):
        raise RuntimeError(f"corrupt native HPO trace: invalid {name} byte")
    try:
        return [
            bytes(raw_bytes[offsets[index] : offsets[index + 1]]).decode("utf-8")
            for index in range(expected_records)
        ]
    except UnicodeDecodeError as exc:
        raise RuntimeError(
            f"corrupt native HPO trace: invalid UTF-8 in {name}"
        ) from exc


def _decode_trials_result(result: c_void_p, since_id: int) -> list[TrialRecord]:
    """Decode the owning rich-trace ``n4m_method_result_t`` into ordered records.

    Shared by :meth:`Optimizer.get_trials` and :func:`finetune_estimator`. The
    caller owns ``result`` and is responsible for destroying it; this helper
    never frees it.
    """
    version = _result_scalar(result, "trace_format_version")
    if version != float(_TRIAL_TRACE_FORMAT_VERSION):
        raise RuntimeError(f"unsupported native HPO trace version: {version!r}")
    declared_events = _result_scalar(result, "n_events")
    if (
        not math.isfinite(declared_events)
        or declared_events < 0
        or declared_events != int(declared_events)
    ):
        raise RuntimeError("corrupt native HPO trace: invalid event count")

    ids = _result_int64_vector(result, "trial_ids_i64")
    n_trials = len(ids)
    if any(left >= right for left, right in zip(ids, ids[1:])) or any(
        trial_id < since_id for trial_id in ids
    ):
        raise RuntimeError("corrupt native HPO trace: invalid trial id order")
    ask_sequences = _result_int64_vector(result, "trial_ask_sequence")
    terminal_sequences = _result_int64_vector(result, "trial_terminal_sequence")
    if len(ask_sequences) != n_trials or len(terminal_sequences) != n_trials:
        raise RuntimeError("corrupt native HPO trace: event sequence width")

    scores, score_rows, score_cols = _result_double_matrix(result, "trial_scores")
    status_values, status_rows, status_cols = _result_double_matrix(
        result, "trial_status"
    )
    rungs, rung_rows, rung_cols = _result_double_matrix(result, "trial_rung")
    durations, duration_rows, duration_cols = _result_double_matrix(
        result, "trial_duration"
    )
    shapes = {
        (score_rows, score_cols),
        (status_rows, status_cols),
        (rung_rows, rung_cols),
        (duration_rows, duration_cols),
    }
    if shapes != {(1, n_trials)}:
        raise RuntimeError("corrupt native HPO trace: lifecycle shape mismatch")

    param_values, param_rows, n_params = _result_double_matrix(
        result, "trial_param_values"
    )
    if param_rows != n_trials:
        raise RuntimeError("corrupt native HPO trace: parameter row mismatch")
    param_category_index = _result_int_vector(result, "trial_param_category_index")
    param_active = _result_int_vector(result, "trial_param_active")
    param_cells = n_trials * n_params
    if not (
        len(param_values)
        == len(param_category_index)
        == len(param_active)
        == param_cells
    ):
        raise RuntimeError("corrupt native HPO trace: parameter width mismatch")
    if (
        any(not math.isfinite(value) for value in param_values)
        or any(index < -1 for index in param_category_index)
        or any(active not in (0, 1) for active in param_active)
    ):
        raise RuntimeError("corrupt native HPO trace: invalid parameter value")
    param_names = _decode_utf8_records(
        _result_int_vector(result, "trial_param_name_utf8"),
        _result_int64_vector(result, "trial_param_name_offsets"),
        n_params,
        "parameter name",
    )
    if len(set(param_names)) != len(param_names):
        raise RuntimeError("corrupt native HPO trace: duplicate parameter name")
    raw_param_kinds = _result_int_vector(result, "trial_param_kind")
    raw_category_types = _result_int_vector(result, "trial_param_category_type")
    param_integer = _result_int_vector(result, "trial_param_integer")
    if (
        not (
            len(raw_param_kinds)
            == len(raw_category_types)
            == len(param_integer)
            == n_params
        )
        or any(value not in (0, 1) for value in param_integer)
        or any(value < -1 for value in raw_category_types)
    ):
        raise RuntimeError("corrupt native HPO trace: parameter metadata width")
    try:
        param_kinds = [ParameterKind(value) for value in raw_param_kinds]
        category_types = [
            CategoryType(value) if value >= 0 else None for value in raw_category_types
        ]
    except ValueError as exc:
        raise RuntimeError(
            "corrupt native HPO trace: unknown parameter metadata"
        ) from exc
    for kind, category_type in zip(param_kinds, category_types, strict=True):
        if (kind is ParameterKind.CATEGORICAL) != (category_type is not None):
            raise RuntimeError(
                "corrupt native HPO trace: inconsistent category metadata"
            )
    param_labels = _decode_utf8_records(
        _result_int_vector(result, "trial_param_label_utf8"),
        _result_int64_vector(result, "trial_param_label_offsets"),
        param_cells,
        "parameter label",
    )

    intermediate_offsets = _result_int64_vector(result, "trial_intermediate_offsets")
    intermediate_sequences = _result_int64_vector(result, "trial_intermediate_sequence")
    intermediate_steps = _result_int_vector(result, "trial_intermediate_steps")
    intermediate_scores, intermediate_rows, intermediate_cols = _result_double_matrix(
        result, "trial_intermediate_scores"
    )
    intermediate_prune = _result_int_vector(result, "trial_intermediate_should_prune")
    if (
        len(intermediate_offsets) != n_trials + 1
        or not intermediate_offsets
        or intermediate_offsets[0] != 0
        or intermediate_offsets[-1] != len(intermediate_steps)
        or len(intermediate_sequences) != len(intermediate_steps)
        or intermediate_rows != 1
        or intermediate_cols != len(intermediate_steps)
        or len(intermediate_scores) != len(intermediate_steps)
        or len(intermediate_prune) != len(intermediate_steps)
        or any(not math.isfinite(score) for score in intermediate_scores)
        or any(decision not in (0, 1) for decision in intermediate_prune)
        or any(
            left > right
            for left, right in zip(intermediate_offsets, intermediate_offsets[1:])
        )
    ):
        raise RuntimeError("corrupt native HPO trace: intermediate offsets")

    error_codes = _decode_utf8_records(
        _result_int_vector(result, "trial_error_code_utf8"),
        _result_int64_vector(result, "trial_error_code_offsets"),
        n_trials,
        "error code",
    )
    error_messages = _decode_utf8_records(
        _result_int_vector(result, "trial_error_message_utf8"),
        _result_int64_vector(result, "trial_error_message_offsets"),
        n_trials,
        "error message",
    )
    error_retryable = _result_int_vector(result, "trial_error_retryable")
    if len(error_retryable) != n_trials or any(
        value not in (0, 1) for value in error_retryable
    ):
        raise RuntimeError("corrupt native HPO trace: error retryable width")

    records: list[TrialRecord] = []
    seen_event_sequences: set[int] = set()
    for trial_index, trial_id in enumerate(ids):
        details: dict[str, TrialParameter] = {}
        values: dict[str, bool | int | float | str] = {}
        for param_index, name in enumerate(param_names):
            cell = trial_index * n_params + param_index
            category_index = param_category_index[cell]
            kind = param_kinds[param_index]
            category_type = category_types[param_index]
            expects_category_index = kind in {
                ParameterKind.CATEGORICAL,
                ParameterKind.ORDINAL,
            }
            if expects_category_index != (category_index >= 0):
                raise RuntimeError(
                    "corrupt native HPO trace: inconsistent category index"
                )
            category_label = param_labels[cell] if category_index >= 0 else None
            integer = bool(param_integer[param_index])
            value = param_values[cell]
            if integer and value != int(value):
                raise RuntimeError(
                    "corrupt native HPO trace: non-integral integer parameter"
                )
            if category_type is CategoryType.BOOL and value not in (0.0, 1.0):
                raise RuntimeError(
                    "corrupt native HPO trace: invalid boolean parameter"
                )
            detail = TrialParameter(
                value=value,
                kind=kind,
                category_index=(category_index if category_index >= 0 else None),
                category_label=category_label,
                category_type=category_type,
                integer=integer,
                active=bool(param_active[cell]),
            )
            details[name] = detail
            values[name] = detail.decoded_value

        start = intermediate_offsets[trial_index]
        stop = intermediate_offsets[trial_index + 1]
        steps = intermediate_steps[start:stop]
        if any(left >= right for left, right in zip(steps, steps[1:])):
            raise RuntimeError(
                "corrupt native HPO trace: intermediate steps are not increasing"
            )
        sequences = intermediate_sequences[start:stop]
        if any(left >= right for left, right in zip(sequences, sequences[1:])):
            raise RuntimeError("corrupt native HPO trace: intermediate event order")
        intermediates = tuple(
            IntermediateValue(
                sequence=intermediate_sequences[index],
                step=intermediate_steps[index],
                score=intermediate_scores[index],
                should_prune=bool(intermediate_prune[index]),
            )
            for index in range(start, stop)
        )

        code = error_codes[trial_index]
        message = error_messages[trial_index]
        if bool(code) != bool(message):
            raise RuntimeError("corrupt native HPO trace: partial trial error")
        if code and not _valid_error_code(code):
            raise RuntimeError("corrupt native HPO trace: invalid error code")
        error_value = (
            TrialError(code, message, bool(error_retryable[trial_index]))
            if code
            else None
        )
        raw_status = status_values[trial_index]
        if not math.isfinite(raw_status) or raw_status != int(raw_status):
            raise RuntimeError("corrupt native HPO trace: invalid trial status")
        try:
            status_value = TrialStatus(int(raw_status))
        except ValueError as exc:
            raise RuntimeError(
                "corrupt native HPO trace: unknown trial status"
            ) from exc
        score_value = scores[trial_index]
        rung_value = rungs[trial_index]
        duration_value = durations[trial_index]
        if (
            not math.isfinite(rung_value)
            or rung_value != int(rung_value)
            or rung_value < 0
            or not math.isfinite(duration_value)
            or duration_value < 0.0
        ):
            raise RuntimeError("corrupt native HPO trace: lifecycle value")
        ask_sequence = ask_sequences[trial_index]
        terminal_sequence = terminal_sequences[trial_index]
        if ask_sequence < 0 or (
            terminal_sequence >= 0 and terminal_sequence <= ask_sequence
        ):
            raise RuntimeError("corrupt native HPO trace: invalid terminal order")
        if sequences and sequences[0] <= ask_sequence:
            raise RuntimeError("corrupt native HPO trace: invalid intermediate order")
        if terminal_sequence >= 0 and sequences and terminal_sequence <= sequences[-1]:
            raise RuntimeError("corrupt native HPO trace: invalid terminal order")
        terminal_status = status_value is not TrialStatus.RUNNING
        if terminal_status != (terminal_sequence >= 0):
            raise RuntimeError("corrupt native HPO trace: terminal state mismatch")
        if status_value in {TrialStatus.FAILED, TrialStatus.CANCELLED}:
            if error_value is None:
                raise RuntimeError("corrupt native HPO trace: missing trial error")
        elif error_value is not None:
            raise RuntimeError("corrupt native HPO trace: unexpected trial error")
        if status_value is TrialStatus.COMPLETED:
            if not math.isfinite(score_value):
                raise RuntimeError("corrupt native HPO trace: completed score")
        elif not math.isnan(score_value):
            raise RuntimeError("corrupt native HPO trace: unexpected trial score")
        prune_indices = [
            index for index, item in enumerate(intermediates) if item.should_prune
        ]
        if prune_indices and (
            status_value is not TrialStatus.PRUNED
            or prune_indices != [len(intermediates) - 1]
        ):
            raise RuntimeError("corrupt native HPO trace: invalid prune decision")
        trial_events = [ask_sequence, *sequences]
        if terminal_sequence >= 0:
            trial_events.append(terminal_sequence)
        if seen_event_sequences.intersection(trial_events):
            raise RuntimeError("corrupt native HPO trace: duplicate event sequence")
        seen_event_sequences.update(trial_events)
        records.append(
            TrialRecord(
                id=trial_id,
                ask_sequence=ask_sequence,
                terminal_sequence=(
                    terminal_sequence if terminal_sequence >= 0 else None
                ),
                params=values,
                param_details=details,
                status=status_value,
                score=None if math.isnan(score_value) else score_value,
                rung=int(rung_value),
                duration=duration_value,
                intermediates=intermediates,
                error=error_value,
            )
        )
    if len(seen_event_sequences) != int(declared_events):
        raise RuntimeError("corrupt native HPO trace: event count mismatch")
    return records


def _validate_finetune_result(
    *,
    best_score: float,
    best_params: dict[str, int | float],
    metric: Metric,
    estimator: Algorithm,
    trials: tuple[TrialRecord, ...],
    timed_out: bool,
    requested_trials: int,
    expected_metric: Metric,
    expected_estimator: Algorithm,
    direction: Direction,
) -> None:
    """Fail closed if the native result bag contradicts its owning trace."""

    if not math.isfinite(best_score):
        raise RuntimeError("corrupt native finetune result: non-finite best_score")
    if metric is not expected_metric or estimator is not expected_estimator:
        raise RuntimeError("corrupt native finetune result: request metadata mismatch")
    if len(trials) > requested_trials:
        raise RuntimeError("corrupt native finetune result: too many trials")
    if timed_out != (len(trials) < requested_trials):
        raise RuntimeError(
            "corrupt native finetune result: timeout/trial count mismatch"
        )
    if any(
        trial.status not in {TrialStatus.COMPLETED, TrialStatus.FAILED}
        for trial in trials
    ):
        raise RuntimeError("corrupt native finetune result: non-terminal driver trial")

    completed = [
        trial
        for trial in trials
        if trial.status is TrialStatus.COMPLETED and trial.score is not None
    ]
    if not completed:
        raise RuntimeError("corrupt native finetune result: no completed trial")
    effective_direction = direction
    if effective_direction is Direction.AUTO:
        effective_direction = (
            Direction.MAXIMIZE if metric is Metric.R2 else Direction.MINIMIZE
        )
    scores = [float(trial.score) for trial in completed if trial.score is not None]
    expected_score = (
        max(scores) if effective_direction is Direction.MAXIMIZE else min(scores)
    )
    if best_score != expected_score:
        raise RuntimeError("corrupt native finetune result: best score mismatch")

    def active_params(trial: TrialRecord) -> dict[str, int | float]:
        params: dict[str, int | float] = {}
        for name, detail in trial.param_details.items():
            if not detail.active:
                continue
            try:
                value = trial.params[name]
            except KeyError as exc:
                raise RuntimeError(
                    "corrupt native finetune result: missing active parameter"
                ) from exc
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RuntimeError(
                    "corrupt native finetune result: non-numeric active parameter"
                )
            params[name] = value
        return params

    if not any(
        trial.score == best_score and active_params(trial) == best_params
        for trial in completed
    ):
        raise RuntimeError("corrupt native finetune result: best parameters mismatch")


class Optimizer:
    """Native ask/tell hyperparameter optimizer."""

    def __init__(
        self,
        space: SearchSpace,
        *,
        sampler: Sampler = Sampler.RANDOM,
        pruner: Pruner = Pruner.NONE,
        direction: Direction = Direction.AUTO,
        metric: Metric = Metric.RMSE,
        eval_mode: EvalMode = EvalMode.MEAN,
        n_startup_trials: int = 10,
        seed: int = 0,
        timeout_seconds: float = 0.0,
        max_resource: int = 0,
        reduction_factor: int = 0,
        context: Context | None = None,
    ) -> None:
        space._require_open()
        self._owns_context = context is None
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
        opts.max_resource = int(max_resource)  # hyperband requires a positive top rung
        opts.reduction_factor = int(
            reduction_factor
        )  # asha/hyperband eta (0 = default 3)
        self._opts = opts
        handle = c_void_p()
        try:
            check(
                lib.n4m_optimizer_create(
                    self._ctx._handle, space._handle, byref(opts), byref(handle)
                ),
                "n4m_optimizer_create",
            )
        except BaseException:
            if self._owns_context:
                self._ctx.close()
            raise
        self._handle = handle

    def _require_open(self) -> None:
        if not getattr(self, "_handle", None) or not self._handle.value:
            raise RuntimeError("Optimizer is closed")

    def __copy__(self):
        raise TypeError("Optimizer is not copyable")

    def __deepcopy__(self, _memo):
        raise TypeError("Optimizer is not copyable")

    def ask(self) -> Trial:
        self._require_open()
        th = c_void_p()
        check(lib.n4m_optimizer_ask(self._handle, byref(th)), "n4m_optimizer_ask")
        return Trial(th, self)

    def ask_batch(self, n: int) -> list[Trial]:
        """Best-effort batch of up to ``n`` asks, returned in ask order.

        ``n`` must be an ``int`` in the signed ``int32`` range (``bool`` is
        rejected); a negative value reaches the native contract and raises
        :class:`N4MError` with ``INVALID_ARGUMENT``. A short list is normal and
        benign: population samplers (GA/PSO/CMA-ES) stop at a generation boundary
        and a configured ``timeout_seconds`` stops mid-batch; both commit exactly
        what they produced. Returned trials are borrowed and valid until the
        optimizer is closed.

        ``ask_batch(n)`` is exactly ``n`` sequential :meth:`ask` calls with no
        intervening tell (the constant-liar policy stays ``NONE``), so an adaptive
        sampler sees the same history it would one ask at a time. For the same
        ordered search space and seed, reproducible continuation requires the
        same ordered ``ASK`` / ``INTERMEDIATE`` / ``TERMINAL`` event stream with
        the same payloads; it does not cover an arbitrary wall-clock completion
        order in parallel hosts.

        Raises:
            N4MError: the very first ask failed, so nothing was committed — a
                generation boundary at zero capacity (``INVALID_ARGUMENT``) or a
                timeout (``CANCELLED``).
            PartialBatchError: at least one trial was committed and then a
                non-benign error stopped the batch. ``partial_trials`` holds the
                committed (``RUNNING``, borrowed) trials; the caller must
                terminalize them (``tell`` / ``tell_result``) or recover them via
                :meth:`get_trials`. They are never rolled back.
        """
        self._require_open()
        if isinstance(n, bool) or not isinstance(n, int):
            raise TypeError("n must be an int")
        if not (-(2**31) <= n <= 2**31 - 1):
            raise OverflowError("n must fit in an int32")
        count = c_int32()
        if n <= 0:
            # n == 0 -> OK/empty; n < 0 -> native INVALID_ARGUMENT (zero progress).
            # out_trials may be NULL for n == 0, and the shim rejects n < 0 before
            # dereferencing it, so pass NULL either way and allocate nothing.
            status = lib.n4m_optimizer_ask_batch(self._handle, n, None, byref(count))
            if status != Status.OK:
                raise N4MError(status, "n4m_optimizer_ask_batch")
            return []
        buf = (c_void_p * n)()  # allocate exactly n pointer slots
        status = lib.n4m_optimizer_ask_batch(self._handle, n, buf, byref(count))
        committed = [Trial(c_void_p(buf[i]), self) for i in range(count.value)]
        if status == Status.OK:
            return committed
        if count.value == 0:
            raise N4MError(status, "n4m_optimizer_ask_batch")
        raise PartialBatchError(status, committed, "n4m_optimizer_ask_batch")

    def save(self) -> bytes:
        """Return a portable native optimizer checkpoint.

        The returned ``bytes`` contains N4MOPT format v1, including the ordered
        search space, options, lifecycle/error/intermediate trace, queued warm
        starts, RNG stream and adaptive sampler state. It can be restored with
        :meth:`load` on any supported host with a compatible libn4m.
        """

        self._require_open()
        array = c_void_p()
        check(lib.n4m_optimizer_save(self._handle, byref(array)), "n4m_optimizer_save")
        try:
            view = MatrixView()
            check(lib.n4m_array_view(array, byref(view)), "n4m_array_view(checkpoint)")
            if (
                view.dtype != Dtype.I64  # owning byte words
                or view.rows != 1
                or view.cols <= 0
                or not view.data
            ):
                raise RuntimeError("corrupt native optimizer checkpoint array")
            return string_at(view.data, int(view.cols) * 8)
        finally:
            lib.n4m_array_free(array)

    @classmethod
    def load(
        cls,
        checkpoint: bytes | bytearray | memoryview,
        *,
        context: Context | None = None,
    ) -> "Optimizer":
        """Restore an optimizer from N4MOPT checkpoint bytes.

        Decode is transactional in libn4m: an exception never exposes a partial
        optimizer. ``checkpoint`` is copied, so the caller may immediately
        mutate or release its input buffer after this method returns.
        """

        if not isinstance(checkpoint, (bytes, bytearray, memoryview)):
            raise TypeError("checkpoint must be bytes-like")
        raw = bytes(checkpoint)
        if not raw:
            raise ValueError("checkpoint must not be empty")
        owns_context = context is None
        native_context = context or Context()
        handle = c_void_p()
        buffer = (c_uint8 * len(raw)).from_buffer_copy(raw)
        try:
            check(
                lib.n4m_optimizer_load(
                    native_context._handle, buffer, len(raw), byref(handle)
                ),
                "n4m_optimizer_load",
            )
        except BaseException:
            if owns_context:
                native_context.close()
            raise
        optimizer = cls.__new__(cls)
        optimizer._owns_context = owns_context
        optimizer._ctx = native_context
        optimizer._space = None  # native checkpoint owns the decoded SearchSpace
        optimizer._opts = None  # options are owned by the decoded native optimizer
        optimizer._handle = handle
        return optimizer

    def tell(self, trial_id: int, score: float) -> None:
        self._require_open()
        check(
            lib.n4m_optimizer_tell(self._handle, trial_id, score), "n4m_optimizer_tell"
        )

    def tell_result(
        self,
        trial_id: int,
        status: TrialStatus,
        score: float = 0.0,
        error: str | TrialError | None = None,
    ) -> None:
        if isinstance(error, TrialError):
            error_bytes = error._wire()
        elif isinstance(error, str):
            if "\x00" in error:
                raise ValueError("error must not contain NUL")
            error_bytes = error.encode()
        elif error is None:
            error_bytes = None
        else:
            raise TypeError("error must be a string, TrialError, or None")
        self._require_open()
        check(
            lib.n4m_optimizer_tell_result(
                self._handle,
                trial_id,
                int(status),
                score,
                error_bytes,
            ),
            "n4m_optimizer_tell_result",
        )

    def tell_intermediate(self, trial_id: int, step: int, score: float) -> bool:
        self._require_open()
        out = c_int32()
        check(
            lib.n4m_optimizer_tell_intermediate(
                self._handle, trial_id, step, score, byref(out)
            ),
            "n4m_optimizer_tell_intermediate",
        )
        return bool(out.value)

    def get_trials(self, since_id: int = 0) -> list[TrialRecord]:
        """Return owning trial snapshots in ascending id order.

        ``since_id`` is inclusive. Unlike :class:`Trial`, returned records own
        all decoded data and remain usable after the optimizer is closed.
        """

        self._require_open()
        if (
            isinstance(since_id, bool)
            or not isinstance(since_id, int)
            or since_id < 0
            or since_id > 2**63 - 1
        ):
            raise ValueError("since_id must be a non-negative int64")
        result = c_void_p()
        check(
            lib.n4m_optimizer_get_trials(self._handle, since_id, byref(result)),
            "n4m_optimizer_get_trials",
        )
        try:
            return _decode_trials_result(result, since_id)
        finally:
            lib.n4m_method_result_destroy(result)

    def enqueue(self, params: dict) -> None:
        self._require_open()
        names = list(params.keys())
        n = len(names)
        cnames = (c_char_p * n)(*[k.encode() for k in names])
        cvals = (c_double * n)(*[float(params[k]) for k in names])
        check(
            lib.n4m_optimizer_enqueue(self._handle, cnames, cvals, n),
            "n4m_optimizer_enqueue",
        )

    def best(self):
        self._require_open()
        th = c_void_p()
        score = c_double()
        status = lib.n4m_optimizer_best(self._handle, byref(th), byref(score))
        if status == _N4M_ERR_NOT_FITTED:
            return None
        check(status, "n4m_optimizer_best")
        return Trial(th, self), float(score.value)

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle and handle.value:
            lib.n4m_optimizer_destroy(handle)
            self._handle = c_void_p()
        if getattr(self, "_owns_context", False):
            self._ctx.close()

    def __enter__(self) -> "Optimizer":
        self._require_open()
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def finetune_estimator(
    estimator: Algorithm,
    X,
    Y,
    plan: ValidationPlan,
    space: SearchSpace,
    *,
    n_trials: int,
    sampler: Sampler = Sampler.RANDOM,
    pruner: Pruner = Pruner.NONE,
    direction: Direction = Direction.AUTO,
    metric: Metric = Metric.RMSE,
    eval_mode: EvalMode = EvalMode.MEAN,
    n_startup_trials: int = 10,
    seed: int = 0,
    timeout_seconds: float = 0.0,
    max_resource: int = 0,
    reduction_factor: int = 0,
    context: Context | None = None,
) -> FinetuneResult:
    """Tune a generic native regression estimator over a single validation plan.

    Thin marshaller over ``n4m_finetune_estimator``: the ask/tell study and the
    cross-validation objective run entirely in libn4m. ``estimator`` must be one
    of the eligible generic regression routes (``PLS_REGRESSION``,
    ``PLS_CANONICAL``, ``PLS_SVD``, ``OPLS``, ``SPARSE_PLS``, ``PCR``); anything
    else raises :class:`N4MError` with ``UNSUPPORTED`` before a study is created.

    ``X`` is coerced to a contiguous ``(n_samples, n_features)`` float64 array and
    ``Y`` to ``(n_samples, n_targets)`` (a 1-D target vector becomes a single
    column). Build the owning ``plan`` with :class:`ValidationPlan`; ``space``
    declares the tuned axes (``n_components`` and, for ``SPARSE_PLS``,
    ``sparsity_lambda``). The keyword options mirror :class:`Optimizer`.

    Returns an owning :class:`FinetuneResult`; every field is copied out before
    the native result handle is destroyed. This convenience driver performs
    selection only: it does not refit a final estimator on all rows. A timeout
    after at least one completed trial returns the best partial result with
    ``timed_out=True``; a timeout before any completion raises ``CANCELLED``.
    """
    if isinstance(n_trials, bool) or not isinstance(n_trials, int):
        raise TypeError("n_trials must be an int")
    if n_trials < 1 or n_trials > 2**31 - 1:
        raise ValueError("n_trials must be in [1, INT32_MAX]")
    if isinstance(estimator, bool) or not isinstance(estimator, (Algorithm, int)):
        raise TypeError("estimator must be an Algorithm or int enum value")
    if estimator < -(2**31) or estimator > 2**31 - 1:
        raise ValueError("estimator enum value must fit int32")
    if not isinstance(plan, ValidationPlan):
        raise TypeError("plan must be an owning n4m.model_selection.ValidationPlan")
    if not isinstance(space, SearchSpace):
        raise TypeError("space must be an owning n4m.model_selection.SearchSpace")
    if context is not None and not isinstance(context, Context):
        raise TypeError("context must be an n4m.Context or None")
    space._require_open()
    plan._require_open()
    if context is not None and not context._handle.value:
        raise RuntimeError("Context is closed")

    x_arr = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    if x_arr.ndim != 2:
        raise ValueError("X must be a 2-D (n_samples, n_features) array")
    y_arr = np.ascontiguousarray(np.asarray(Y, dtype=np.float64))
    if y_arr.ndim == 1:
        y_arr = y_arr.reshape(-1, 1)
    elif y_arr.ndim != 2:
        raise ValueError(
            "Y must be a 1-D (n_samples,) or 2-D (n_samples, n_targets) array"
        )
    if y_arr.shape[0] != x_arr.shape[0]:
        raise ValueError("X and Y must contain the same number of samples")
    if plan.n_samples != x_arr.shape[0]:
        raise ValueError("plan.n_samples must match X.shape[0]")
    x_view = numpy_to_view(x_arr)
    y_view = numpy_to_view(y_arr)

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
    opts.max_resource = int(max_resource)
    opts.reduction_factor = int(reduction_factor)

    owns_context = context is None
    ctx = context or Context()
    try:
        result = c_void_p()
        check(
            lib.n4m_finetune_estimator(
                ctx._handle,
                int(estimator),
                byref(x_view),
                byref(y_view),
                plan._handle,
                space._handle,
                byref(opts),
                int(n_trials),
                byref(result),
            ),
            "n4m_finetune_estimator",
        )
        # x_arr / y_arr must outlive the native call above; they remain referenced
        # here until the whole block returns, so the views stay valid throughout.
        try:
            best_score = _result_scalar(result, "best_score")
            metric_out = _result_enum_scalar(result, "metric", Metric)
            estimator_out = _result_enum_scalar(result, "estimator", Algorithm)
            best_params: dict[str, int | float] = {}
            n_components = _maybe_result_scalar(result, "best.n_components")
            if n_components is not None:
                if (
                    not math.isfinite(n_components)
                    or n_components != int(n_components)
                    or n_components < 1
                    or n_components > 2**31 - 1
                ):
                    raise RuntimeError(
                        "corrupt native finetune result: invalid best.n_components"
                    )
                best_params["n_components"] = int(n_components)
            sparsity_lambda = _maybe_result_scalar(result, "best.sparsity_lambda")
            if sparsity_lambda is not None:
                if (
                    not math.isfinite(sparsity_lambda)
                    or sparsity_lambda < 0.0
                    or sparsity_lambda >= 1.0
                ):
                    raise RuntimeError(
                        "corrupt native finetune result: invalid best.sparsity_lambda"
                    )
                best_params["sparsity_lambda"] = float(sparsity_lambda)
            raw_timed_out = _result_scalar(result, "timed_out")
            if raw_timed_out not in (0.0, 1.0):
                raise RuntimeError("corrupt native finetune result: invalid timed_out")
            timed_out = bool(int(raw_timed_out))
            requested_trials = _result_bounded_int_scalar(
                result, "requested_trials", low=1, high=2**31 - 1
            )
            if requested_trials != n_trials:
                raise RuntimeError(
                    "corrupt native finetune result: requested_trials mismatch"
                )
            trials = tuple(_decode_trials_result(result, 0))
            try:
                expected_estimator = Algorithm(int(estimator))
                expected_metric = Metric(int(metric))
                requested_direction = Direction(int(direction))
            except ValueError as exc:  # native success makes this an ABI contradiction
                raise RuntimeError(
                    "corrupt native finetune result: invalid request metadata"
                ) from exc
            _validate_finetune_result(
                best_score=best_score,
                best_params=best_params,
                metric=metric_out,
                estimator=estimator_out,
                trials=trials,
                timed_out=timed_out,
                requested_trials=requested_trials,
                expected_metric=expected_metric,
                expected_estimator=expected_estimator,
                direction=requested_direction,
            )
        finally:
            lib.n4m_method_result_destroy(result)
    finally:
        if owns_context:
            ctx.close()

    return FinetuneResult(
        best_score=best_score,
        best_params=best_params,
        metric=metric_out,
        estimator=estimator_out,
        trials=trials,
        timed_out=timed_out,
        requested_trials=requested_trials,
    )


__all__ = [
    "SearchSpace",
    "Trial",
    "TrialRecord",
    "TrialParameter",
    "TrialError",
    "IntermediateValue",
    "FinetuneResult",
    "ValidationPlan",
    "ParameterKind",
    "CategoryType",
    "Optimizer",
    "Algorithm",
    "Sampler",
    "Pruner",
    "Direction",
    "EvalMode",
    "Metric",
    "TrialStatus",
    "ConstraintKind",
    "finetune_estimator",
]
