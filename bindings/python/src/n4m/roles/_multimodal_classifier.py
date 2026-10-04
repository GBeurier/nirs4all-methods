# SPDX-License-Identifier: CECILL-2.1
"""Raw encoders and PLS-logistic classification owned by native ABI 2.17."""

from __future__ import annotations

import copy
import ctypes as ct
import numbers
from typing import Any

import numpy as np

from .._errors import check
from .._ffi import lib
from .._matrix import numpy_to_view
from .._types import MatrixView
from ._base import _Context, method_class
from ._multimodal import (
    _SourceSpec,
    _SourceView,
    _integer,
    _keys,
    _source_configuration,
    _source_order,
    _views,
)

_METHOD = "models.classification.pls_logistic"
_I64P = ct.POINTER(ct.c_int64)
_MAX_ELEMENTS = 16 * 1024 * 1024


def _bounded_shape(rows: int, cols: int) -> None:
    if rows < 0 or cols <= 0 or cols > _MAX_ELEMENTS or rows > _MAX_ELEMENTS // cols:
        raise ValueError("classifier matrix exceeds the native 16777216-element bound")


def _bounded_working_set(rows: int, classes: int, components: int) -> None:
    message = "multimodal classifier PLS-logistic working set exceeds 16777216-element matrix limit"
    if rows < 2 or not 2 <= classes <= 65536 or components < 1:
        raise ValueError(message)
    terms = components + 1
    if rows > _MAX_ELEMENTS // classes or terms > _MAX_ELEMENTS or rows > _MAX_ELEMENTS // terms:
        raise ValueError(message)
    if classes - 1 > _MAX_ELEMENTS // terms:
        raise ValueError(message)
    parameters = (classes - 1) * terms
    if parameters > _MAX_ELEMENTS // parameters:
        raise ValueError(message)


class _ClassifierRecipe(ct.Structure):
    _fields_ = [
        ("struct_size", ct.c_uint32),
        ("n_sources", ct.c_int32),
        ("sources", ct.POINTER(_SourceSpec)),
        ("method_id", ct.c_char_p),
        ("params", ct.c_void_p),
    ]


def _api() -> None:
    check(lib.n4m_check_abi_compatibility(2, 17), "multimodal classifier requires ABI 2.17")
    handle = ct.c_void_p
    views = [handle, handle, ct.c_int32, ct.POINTER(_SourceView)]
    specs = {
        "create": [handle, ct.POINTER(_ClassifierRecipe), ct.POINTER(handle)],
        "destroy": [handle],
        "fit": [*views, _I64P, ct.c_int64],
        "predict_labels": [*views, _I64P, ct.c_int64],
        "decision_function": [*views, ct.POINTER(MatrixView)],
        "predict_proba": [*views, ct.POINTER(MatrixView)],
        "classes": [handle, _I64P, ct.c_int64, _I64P],
        "n_outputs": [handle, _I64P],
        "transform_cols": [handle, _I64P],
        "transform": [*views, ct.POINTER(MatrixView)],
        "export_size": [handle, handle, ct.POINTER(ct.c_size_t)],
        "export_to_buffer": [handle, handle, handle, ct.c_size_t, ct.POINTER(ct.c_size_t)],
        "import_from_buffer": [
            handle, ct.POINTER(_ClassifierRecipe), handle, ct.c_size_t, ct.POINTER(handle)
        ],
    }
    for name, arguments in specs.items():
        try:
            function = getattr(lib, f"n4m_multimodal_classifier_{name}")
        except AttributeError as exc:
            raise RuntimeError("native multimodal classifier ABI is unavailable") from exc
        function.argtypes = arguments
        function.restype = None if name == "destroy" else ct.c_int


def _configuration(recipe: Any, schemas: Any, context: _Context):
    sources, keep, recipe = _source_configuration(recipe, schemas)
    model = _keys(recipe["model"], {"method_id", "params"}, "classifier model")
    if model["method_id"] != _METHOD:
        raise ValueError("MultimodalClassifierPipeline requires native PLS-logistic")
    params = _keys(model["params"], {"n_components", "max_iter"}, "PLS-logistic params")
    values = {key: _integer(value, key) for key, value in params.items()}
    native = method_class(_METHOD)(**values)._native_params(context)
    config = _ClassifierRecipe(
        ct.sizeof(_ClassifierRecipe), len(sources), sources, _METHOD.encode(), native
    )
    return config, keep, native


def _label_table(values: Any) -> np.ndarray:
    labels = np.asarray(values, dtype=object)
    if labels.ndim != 1 or not len(labels):
        raise ValueError("class_names must be a nonempty one-dimensional table")
    items = labels.tolist()
    if all(isinstance(value, str) for value in items):
        for value in items:
            value.encode("utf-8")
        dtype = object
    elif all(isinstance(value, numbers.Integral) and not isinstance(value, (bool, np.bool_)) for value in items):
        if any(not -(2**63) <= int(value) < 2**63 for value in items):
            raise ValueError("integer labels must fit signed int64")
        items = [int(value) for value in items]
        dtype = np.int64
    else:
        raise TypeError("labels must be homogeneous strings or signed int64 integers")
    if len(set(items)) != len(items):
        raise ValueError("class_names must contain unique labels")
    return np.asarray(items, dtype=dtype)


def _encode_labels(y: Any) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(y, dtype=object)
    if values.ndim != 1 or not len(values):
        raise ValueError("labels must be a nonempty one-dimensional vector")
    items = values.tolist()
    # Validate the type of every element before a NumPy cast can erase mixed types.
    if not (
        all(isinstance(value, str) for value in items)
        or all(isinstance(value, numbers.Integral) and not isinstance(value, (bool, np.bool_)) for value in items)
    ):
        raise TypeError("labels must be homogeneous strings or signed int64 integers")
    names = _label_table(sorted(set(items)))
    positions = {value: index for index, value in enumerate(names.tolist())}
    return np.ascontiguousarray([positions[value] for value in items], dtype=np.int64), names


class MultimodalClassifierPipeline:
    """Native raw-source classifier with an explicit external typed label table.

    Probability and decision columns follow classes_. Imported arbitrary native
    IDs are mapped by class-column position. Without class_names, native IDs are
    exposed directly. Native portable state contains fitted encoders and head;
    the original typed label table is separately signed by archive owners.
    """

    def __init__(self, recipe: Any, source_schemas: Any):
        _api()
        self.recipe = copy.deepcopy(recipe)
        self.source_schemas = copy.deepcopy(source_schemas)
        self._handle = ct.c_void_p()
        self._closed = False
        self._label_names: np.ndarray | None = None
        self._order = _source_order(self.recipe["source_order"])
        with _Context() as context:
            config, keep, params = _configuration(self.recipe, self.source_schemas, context)
            try:
                context.check(
                    lib.n4m_multimodal_classifier_create(context.handle, ct.byref(config), ct.byref(self._handle)),
                    "multimodal classifier create",
                )
                self._n_components = _integer(self.recipe["model"]["params"]["n_components"], "n_components")
            finally:
                lib.n4m_params_destroy(params)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("MultimodalClassifierPipeline is closed")

    def fit(self, blocks: Any, y: Any, *, source_schemas: Any = None):
        self._require_open()
        labels, names = _encode_labels(y)
        if len(names) >= 2:
            _bounded_working_set(len(labels), len(names), self._n_components)
        views, keep, rows = _views(blocks, self.source_schemas if source_schemas is None else source_schemas, self._order)
        if rows != len(labels):
            raise ValueError("raw source and label row counts differ")
        _bounded_shape(rows, len(names))
        with _Context() as context:
            context.check(
                lib.n4m_multimodal_classifier_fit(context.handle, self._handle, len(views), views, labels.ctypes.data_as(_I64P), rows),
                "multimodal classifier fit",
            )
        self._label_names = names
        return self

    def _class_ids(self) -> np.ndarray:
        self._require_open()
        count = ct.c_int64()
        check(lib.n4m_multimodal_classifier_classes(self._handle, None, 0, ct.byref(count)), "classifier classes")
        if not 2 <= count.value <= 65536:
            raise ValueError("native classifier class count exceeds the supported bound")
        values = np.empty(count.value, dtype=np.int64)
        check(lib.n4m_multimodal_classifier_classes(self._handle, values.ctypes.data_as(_I64P), len(values), ct.byref(count)), "classifier classes")
        return values

    @property
    def classes_(self) -> np.ndarray:
        ids = self._class_ids()
        return ids if self._label_names is None else self._label_names.copy()

    @property
    def label_names_(self) -> np.ndarray | None:
        self._require_open()
        return None if self._label_names is None else self._label_names.copy()

    def predict(self, blocks: Any, *, source_schemas: Any = None) -> np.ndarray:
        self._require_open()
        views, keep, rows = _views(blocks, self.source_schemas if source_schemas is None else source_schemas, self._order)
        cols = ct.c_int64()
        check(lib.n4m_multimodal_classifier_n_outputs(self._handle, ct.byref(cols)), "classifier prediction width")
        _bounded_shape(rows, cols.value)
        result = np.empty(rows, dtype=np.int64)
        with _Context() as context:
            context.check(lib.n4m_multimodal_classifier_predict_labels(context.handle, self._handle, len(views), views, result.ctypes.data_as(_I64P), rows), "classifier predict")
        if self._label_names is None:
            return result
        positions = {int(value): index for index, value in enumerate(self._class_ids())}
        return self._label_names[[positions[int(value)] for value in result]]

    def _matrix(self, operation: str, blocks: Any, source_schemas: Any) -> np.ndarray:
        self._require_open()
        views, keep, rows = _views(blocks, self.source_schemas if source_schemas is None else source_schemas, self._order)
        cols = ct.c_int64()
        query = lib.n4m_multimodal_classifier_transform_cols if operation == "transform" else lib.n4m_multimodal_classifier_n_outputs
        check(query(self._handle, ct.byref(cols)), "classifier matrix width")
        _bounded_shape(rows, cols.value)
        result = np.empty((rows, cols.value), dtype=np.float64)
        view = numpy_to_view(result)
        with _Context() as context:
            function = getattr(lib, f"n4m_multimodal_classifier_{operation}")
            context.check(function(context.handle, self._handle, len(views), views, ct.byref(view)), f"classifier {operation}")
        return result

    def predict_proba(self, blocks: Any, *, source_schemas: Any = None) -> np.ndarray:
        return self._matrix("predict_proba", blocks, source_schemas)

    def decision_function(self, blocks: Any, *, source_schemas: Any = None) -> np.ndarray:
        return self._matrix("decision_function", blocks, source_schemas)

    def transform(self, blocks: Any, *, source_schemas: Any = None) -> np.ndarray:
        return self._matrix("transform", blocks, source_schemas)

    def export_state(self) -> bytes:
        self._require_open()
        with _Context() as context:
            size = ct.c_size_t()
            context.check(lib.n4m_multimodal_classifier_export_size(context.handle, self._handle, ct.byref(size)), "classifier export size")
            buffer = ct.create_string_buffer(size.value)
            written = ct.c_size_t()
            context.check(lib.n4m_multimodal_classifier_export_to_buffer(context.handle, self._handle, buffer, size.value, ct.byref(written)), "classifier export")
            return buffer.raw[:written.value]

    @classmethod
    def from_state(cls, state: bytes, *, recipe: Any, source_schemas: Any, class_names: Any = None):
        if not isinstance(state, bytes):
            raise TypeError("state must be bytes")
        if not 0 < len(state) <= 64 * 1024 * 1024:
            raise ValueError("state exceeds native byte bounds")
        pipeline = cls(recipe, source_schemas)
        try:
            with _Context() as context:
                config, keep, params = _configuration(pipeline.recipe, pipeline.source_schemas, context)
                imported = ct.c_void_p()
                buffer = ct.create_string_buffer(state)
                try:
                    context.check(lib.n4m_multimodal_classifier_import_from_buffer(context.handle, ct.byref(config), buffer, len(state), ct.byref(imported)), "classifier import")
                finally:
                    lib.n4m_params_destroy(params)
                lib.n4m_multimodal_classifier_destroy(pipeline._handle)
                pipeline._handle = imported
            if class_names is not None:
                names = _label_table(class_names)
                if len(names) != len(pipeline._class_ids()):
                    raise ValueError("class_names must match native class column count")
                pipeline._label_names = names
            return pipeline
        except BaseException:
            pipeline.close()
            raise

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle and not getattr(self, "_closed", False):
            lib.n4m_multimodal_classifier_destroy(handle)
        self._handle = ct.c_void_p()
        self._closed = True

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()
