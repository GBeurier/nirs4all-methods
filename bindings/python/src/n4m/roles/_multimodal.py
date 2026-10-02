# SPDX-License-Identifier: CECILL-2.1
"""Thin binding for the complete native raw multimodal predictor (ABI 2.16).

Sources are already aligned by the caller. Learned encoders, categorical
vocabulary, weighted fusion, Ridge and portable N4MF state live in libn4m.
"""

from __future__ import annotations

import copy
import ctypes as ct
from collections.abc import Mapping
from typing import Any

import numpy as np

from .._ffi import lib
from .._matrix import numpy_to_view
from .._types import MatrixView
from ._base import _Context

_ORDER = ("nir", "image", "series", "metadata")
_KINDS = {"standard_scaler": 1, "tensor_pca": 2, "column_transformer": 3}
_I64P = ct.POINTER(ct.c_int64)


class _SourceSpec(ct.Structure):
    _fields_ = [
        ("struct_size", ct.c_uint32),
        ("name", ct.c_char_p),
        ("representation_id", ct.c_char_p),
        ("dtype", ct.c_char_p),
        ("identity_utf8", ct.c_void_p),
        ("identity_bytes", ct.c_size_t),
        ("ndim", ct.c_int32),
        ("shape", _I64P),
        ("encoder", ct.c_uint32),
        ("weight", ct.c_double),
        ("n_components", ct.c_int64),
        ("random_state", ct.c_int64),
        ("with_mean", ct.c_int32),
        ("with_std", ct.c_int32),
        ("whiten", ct.c_int32),
        ("ignore_unknown", ct.c_int32),
        ("numeric_column", ct.c_int64),
        ("categorical_column", ct.c_int64),
    ]


class _Recipe(ct.Structure):
    _fields_ = [
        ("struct_size", ct.c_uint32),
        ("n_sources", ct.c_int32),
        ("sources", ct.POINTER(_SourceSpec)),
        ("alpha", ct.c_double),
        ("center_x", ct.c_int32),
        ("center_y", ct.c_int32),
        ("scale_x", ct.c_int32),
    ]


class _SourceView(ct.Structure):
    _fields_ = [
        ("struct_size", ct.c_uint32),
        ("name", ct.c_char_p),
        ("representation_id", ct.c_char_p),
        ("dtype", ct.c_char_p),
        ("identity_utf8", ct.c_void_p),
        ("identity_bytes", ct.c_size_t),
        ("rank", ct.c_int32),
        ("shape", _I64P),
        ("strides", _I64P),
        ("numeric_data", ct.c_void_p),
        ("numeric_dtype", ct.c_int),
        ("categorical_utf8", ct.c_void_p),
        ("utf8_bytes", ct.c_size_t),
        ("categorical_offsets", ct.POINTER(ct.c_uint64)),
    ]


def _api() -> None:
    """Bind additive functions on use, preserving imports on older runtimes."""
    pointer = ct.c_void_p
    signatures = {
        "create": [pointer, ct.POINTER(_Recipe), ct.POINTER(pointer)],
        "destroy": [pointer],
        "fit": [
            pointer,
            pointer,
            ct.c_int32,
            ct.POINTER(_SourceView),
            ct.POINTER(MatrixView),
        ],
        "predict": [
            pointer,
            pointer,
            ct.c_int32,
            ct.POINTER(_SourceView),
            ct.POINTER(MatrixView),
        ],
        "transform": [
            pointer,
            pointer,
            ct.c_int32,
            ct.POINTER(_SourceView),
            ct.POINTER(MatrixView),
        ],
        "transform_cols": [pointer, ct.POINTER(ct.c_int64)],
        "export_size": [pointer, pointer, ct.POINTER(ct.c_size_t)],
        "export_to_buffer": [
            pointer,
            pointer,
            pointer,
            ct.c_size_t,
            ct.POINTER(ct.c_size_t),
        ],
        "import_from_buffer": [
            pointer,
            ct.POINTER(_Recipe),
            pointer,
            ct.c_size_t,
            ct.POINTER(pointer),
        ],
    }
    for name, arguments in signatures.items():
        try:
            function = getattr(lib, "n4m_multimodal_pipeline_" + name)
        except AttributeError as exc:
            raise RuntimeError(
                "MultimodalPipeline requires libn4m ABI 2.16 with native encoder support"
            ) from exc
        function.argtypes = arguments
        function.restype = None if name == "destroy" else ct.c_int


def _keys(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{label} must have exactly {sorted(expected)}")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{label} must be an integer")
    result = int(value)
    if not -(1 << 63) <= result < (1 << 63):
        raise ValueError(f"{label} is outside int64")
    return result


def _boolean(value: Any, label: str) -> int:
    if type(value) is not bool:
        raise TypeError(f"{label} must be a boolean")
    return int(value)


def _cstring(value: Any) -> bytes:
    if not isinstance(value, str) or "\0" in value:
        raise TypeError(
            "schema names, representations and dtypes must be strings without NUL"
        )
    return value.encode("utf-8")


def _schema(schema: Any, item: Any, keep: list[Any]) -> None:
    schema = _keys(
        schema,
        {"representation_id", "input_shape", "dtype", "identity"},
        "source schema",
    )
    item.representation_id = _cstring(schema["representation_id"])
    item.dtype = _cstring(schema["dtype"])
    if not isinstance(schema["identity"], str):
        raise TypeError(
            "source identity must be the existing canonical descriptor string"
        )
    identity = schema["identity"].encode("utf-8")
    buffer = ct.create_string_buffer(identity)
    keep.append(buffer)
    item.identity_utf8 = ct.cast(buffer, ct.c_void_p)
    item.identity_bytes = len(identity)


def _configuration(recipe: Any, schemas: Any) -> tuple[_Recipe, list[Any]]:
    recipe = _keys(
        recipe,
        {
            "schema_version",
            "fusion",
            "source_order",
            "encoders",
            "source_weights",
            "model",
        },
        "recipe",
    )
    if (
        type(recipe["schema_version"]) is not int
        or recipe["schema_version"] != 1
        or recipe["fusion"] != "early"
        or list(recipe["source_order"]) != list(_ORDER)
    ):
        raise ValueError(
            "MultimodalPipeline requires recipe v1 with canonical ordered early fusion"
        )
    encoders = _keys(recipe["encoders"], set(_ORDER), "encoders")
    weights = _keys(recipe["source_weights"], set(_ORDER), "source_weights")
    schemas = _keys(schemas, set(_ORDER), "source_schemas")
    model = _keys(recipe["model"], {"method_id", "params"}, "model")
    if model["method_id"] != "models.regularized.ridge":
        raise ValueError("MultimodalPipeline requires the native Ridge head")
    params = _keys(
        model["params"], {"alpha", "center_x", "center_y", "scale_x"}, "Ridge params"
    )
    sources = (_SourceSpec * 4)()
    keep: list[Any] = [sources]
    for index, name in enumerate(_ORDER):
        spec = sources[index]
        spec.struct_size = ct.sizeof(_SourceSpec)
        spec.name = name.encode()
        _schema(schemas[name], spec, keep)
        shape_values = [
            _integer(value, "source shape") for value in schemas[name]["input_shape"]
        ]
        shape = (ct.c_int64 * len(shape_values))(*shape_values)
        keep.append(shape)
        spec.ndim, spec.shape = len(shape_values), shape
        if isinstance(weights[name], (bool, np.bool_)):
            raise TypeError("source weights must be numbers")
        spec.weight = float(weights[name])
        encoder = encoders[name]
        if not isinstance(encoder, Mapping) or encoder.get("kind") not in _KINDS:
            raise ValueError("unsupported encoder kind")
        kind = encoder["kind"]
        spec.encoder = _KINDS[kind]
        spec.numeric_column = spec.categorical_column = -1
        if kind == "standard_scaler":
            _keys(encoder, {"kind", "with_mean", "with_std"}, "standard_scaler")
            spec.with_mean = _boolean(encoder["with_mean"], "with_mean")
            spec.with_std = _boolean(encoder["with_std"], "with_std")
        elif kind == "tensor_pca":
            _keys(
                encoder,
                {"kind", "n_components", "whiten", "random_state"},
                "tensor_pca",
            )
            spec.n_components = _integer(encoder["n_components"], "n_components")
            spec.random_state = _integer(encoder["random_state"], "random_state")
            spec.whiten = _boolean(encoder["whiten"], "whiten")
        else:
            _keys(
                encoder,
                {
                    "kind",
                    "numeric_columns",
                    "categorical_columns",
                    "with_mean",
                    "with_std",
                    "handle_unknown",
                    "sparse_output",
                    "drop",
                },
                "column_transformer",
            )
            if (
                encoder["numeric_columns"] != [0]
                or encoder["categorical_columns"] != [1]
                or encoder["handle_unknown"] != "ignore"
                or encoder["sparse_output"] is not False
                or encoder["drop"] is not None
            ):
                raise ValueError(
                    "mixed encoder requires numeric0/category1, dense output and unknown-ignore"
                )
            spec.numeric_column, spec.categorical_column = 0, 1
            spec.with_mean = _boolean(encoder["with_mean"], "with_mean")
            spec.with_std = _boolean(encoder["with_std"], "with_std")
            spec.ignore_unknown = 1
    if isinstance(params["alpha"], (bool, np.bool_)):
        raise TypeError("alpha must be a number")
    config = _Recipe(
        ct.sizeof(_Recipe),
        4,
        sources,
        float(params["alpha"]),
        _boolean(params["center_x"], "center_x"),
        _boolean(params["center_y"], "center_y"),
        _boolean(params["scale_x"], "scale_x"),
    )
    return config, keep


def _views(blocks: Any, schemas: Any) -> tuple[Any, list[Any], int]:
    blocks = _keys(blocks, set(_ORDER), "blocks")
    schemas = _keys(schemas, set(_ORDER), "source_schemas")
    views = (_SourceView * 4)()
    keep: list[Any] = [views]
    row_count = -1
    for index, name in enumerate(_ORDER):
        values = np.asarray(blocks[name])
        if values.ndim < 2:
            raise ValueError(
                "raw blocks must retain sample axis and non-sample dimensions"
            )
        if list(values.shape[1:]) != list(schemas[name]["input_shape"]):
            raise ValueError(
                "raw source shape differs from its independently supplied schema"
            )
        view = views[index]
        view.struct_size, view.name = ct.sizeof(_SourceView), name.encode()
        _schema(schemas[name], view, keep)
        view.rank = values.ndim
        shape = (ct.c_int64 * values.ndim)(*values.shape)
        keep.append(shape)
        view.shape = shape
        if row_count == -1:
            row_count = values.shape[0]
        if row_count != values.shape[0]:
            raise ValueError("raw sources must already have aligned equal row counts")
        if name == "metadata":
            if values.ndim != 2 or values.shape[1] != 2:
                raise ValueError("mixed metadata must be a two-column raw table")
            if str(values.dtype) != schemas[name]["dtype"]:
                raise ValueError(
                    "mixed metadata dtype differs from its captured source schema"
                )
            numeric = np.ascontiguousarray(values[:, 0], dtype=np.float64).reshape(
                -1, 1
            )
            cells = []
            offsets = [0]
            for cell in values[:, 1]:
                if not isinstance(cell, str):
                    raise TypeError(
                        "categorical cells must be strings; no host integer codes are accepted"
                    )
                encoded = cell.encode("utf-8")
                if (
                    len(encoded) > 1024 * 1024
                    or offsets[-1] + len(encoded) > 64 * 1024 * 1024
                ):
                    raise ValueError(
                        "categorical UTF-8 cells exceed native byte bounds"
                    )
                cells.append(encoded)
                offsets.append(offsets[-1] + len(encoded))
            utf8 = b"".join(cells)
            buffer = ct.create_string_buffer(utf8)
            boundaries = (ct.c_uint64 * len(offsets))(*offsets)
            keep.extend([buffer, boundaries])
            view.categorical_utf8, view.utf8_bytes = (
                ct.cast(buffer, ct.c_void_p),
                len(utf8),
            )
            view.categorical_offsets = boundaries
        else:
            if values.dtype not in (np.dtype("float32"), np.dtype("float64")):
                raise TypeError(
                    "numeric raw tensors must have float32 or float64 dtype"
                )
            numeric = (
                np.ascontiguousarray(values)
                if any(stride < 0 for stride in values.strides)
                else values
            )
        strides = (ct.c_int64 * values.ndim)(
            *(stride // numeric.itemsize for stride in numeric.strides),
            *([0] * (values.ndim - numeric.ndim)),
        )
        keep.extend([numeric, strides])
        view.strides = strides
        view.numeric_data = numeric.ctypes.data
        view.numeric_dtype = 2 if numeric.dtype == np.dtype("float32") else 1
    return views, keep, row_count


class MultimodalPipeline:
    """One native complete-source early-fusion predictor, portable as N4MF.

    ``recipe`` and ``source_schemas`` are versioned public declarations.
    ``identity`` is the existing canonical IO descriptor text, carried
    unchanged between hosts. Rows must be aligned before calling ``fit``.
    ``predict`` may supply independently captured new-input schemas; native
    code checks their exact identity against the fitted source contracts.
    """

    def __init__(self, recipe: Mapping[str, Any], source_schemas: Mapping[str, Any]):
        _api()
        self.recipe = copy.deepcopy(recipe)
        self.source_schemas = copy.deepcopy(source_schemas)
        self._handle = ct.c_void_p()
        self._closed = False
        config, keep = _configuration(self.recipe, self.source_schemas)
        with _Context() as context:
            context.check(
                lib.n4m_multimodal_pipeline_create(
                    context.handle, ct.byref(config), ct.byref(self._handle)
                ),
                "MultimodalPipeline.create",
            )
        del keep

    def _require_open(self) -> None:
        if self._closed or not self._handle:
            raise RuntimeError("MultimodalPipeline is closed")

    def fit(self, blocks: Mapping[str, Any], y: Any) -> MultimodalPipeline:
        """Learn native encoders and Ridge from exactly the supplied rows."""
        self._require_open()
        views, keep, _ = _views(blocks, self.source_schemas)
        target = np.ascontiguousarray(y, dtype=np.float64)
        if target.ndim == 1:
            target = target.reshape(-1, 1)
        if target.ndim != 2 or target.shape[1] != 1:
            raise ValueError("MultimodalPipeline requires one numeric target")
        target_view = numpy_to_view(target)
        with _Context() as context:
            context.check(
                lib.n4m_multimodal_pipeline_fit(
                    context.handle, self._handle, 4, views, ct.byref(target_view)
                ),
                "MultimodalPipeline.fit",
            )
        del keep
        return self

    def _operation(
        self,
        blocks: Mapping[str, Any],
        schemas: Mapping[str, Any] | None,
        transform: bool,
    ) -> np.ndarray:
        self._require_open()
        views, keep, rows = _views(
            blocks, self.source_schemas if schemas is None else schemas
        )
        cols = ct.c_int64(1)
        with _Context() as context:
            if transform:
                context.check(
                    lib.n4m_multimodal_pipeline_transform_cols(
                        self._handle, ct.byref(cols)
                    ),
                    "MultimodalPipeline.transform_cols",
                )
            result = np.empty((rows, cols.value), dtype=np.float64)
            view = numpy_to_view(result)
            operation = (
                lib.n4m_multimodal_pipeline_transform
                if transform
                else lib.n4m_multimodal_pipeline_predict
            )
            context.check(
                operation(context.handle, self._handle, 4, views, ct.byref(view)),
                "MultimodalPipeline.transform"
                if transform
                else "MultimodalPipeline.predict",
            )
        del keep
        return result if transform else result[:, 0]

    def predict(
        self,
        blocks: Mapping[str, Any],
        *,
        source_schemas: Mapping[str, Any] | None = None,
    ) -> np.ndarray:
        """Predict raw new inputs without learning or retaining their rows."""
        return self._operation(blocks, source_schemas, False)

    def transform(
        self,
        blocks: Mapping[str, Any],
        *,
        source_schemas: Mapping[str, Any] | None = None,
    ) -> np.ndarray:
        """Return the actual native weighted encoded features without fitting."""
        return self._operation(blocks, source_schemas, True)

    def export_state(self) -> bytes:
        """Export the bounded complete native N4MF state; no training rows."""
        self._require_open()
        size, written = ct.c_size_t(), ct.c_size_t()
        with _Context() as context:
            context.check(
                lib.n4m_multimodal_pipeline_export_size(
                    context.handle, self._handle, ct.byref(size)
                ),
                "MultimodalPipeline.export_size",
            )
            buffer = ct.create_string_buffer(size.value)
            context.check(
                lib.n4m_multimodal_pipeline_export_to_buffer(
                    context.handle, self._handle, buffer, size.value, ct.byref(written)
                ),
                "MultimodalPipeline.export_state",
            )
        return buffer.raw[: written.value]

    @classmethod
    def from_state(
        cls,
        state: bytes,
        *,
        recipe: Mapping[str, Any],
        source_schemas: Mapping[str, Any],
    ) -> MultimodalPipeline:
        """Hydrate only after native recipe/schema/integrity validation; no FIT."""
        pipeline = cls(recipe, source_schemas)
        try:
            if not isinstance(state, (bytes, bytearray, memoryview)):
                raise TypeError("state must be bytes, bytearray or memoryview")
            size = state.nbytes if isinstance(state, memoryview) else len(state)
            if size > 64 * 1024 * 1024:
                raise ValueError("N4MF payload exceeds 64MiB")
            payload = bytes(state)
            if len(payload) > 64 * 1024 * 1024:
                raise ValueError("N4MF payload exceeds 64MiB")
            buffer = ct.create_string_buffer(payload)
            config, keep = _configuration(pipeline.recipe, pipeline.source_schemas)
            handle = ct.c_void_p()
            with _Context() as context:
                context.check(
                    lib.n4m_multimodal_pipeline_import_from_buffer(
                        context.handle,
                        ct.byref(config),
                        buffer,
                        len(payload),
                        ct.byref(handle),
                    ),
                    "MultimodalPipeline.from_state",
                )
            lib.n4m_multimodal_pipeline_destroy(pipeline._handle)
            pipeline._handle = handle
            del keep
            return pipeline
        except BaseException:
            pipeline.close()
            raise

    def close(self) -> None:
        """Release owned native state; repeated close is harmless."""
        handle = getattr(self, "_handle", None)
        if handle:
            lib.n4m_multimodal_pipeline_destroy(handle)
            self._handle = ct.c_void_p()
        self._closed = True

    def __enter__(self) -> MultimodalPipeline:
        self._require_open()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
