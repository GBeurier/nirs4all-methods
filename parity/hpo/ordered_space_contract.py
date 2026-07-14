"""Independent test oracle for OrderedSearchSpaceSpec v1.

This module is parity infrastructure, not a production binding. Production
validation and fingerprinting will live in libn4m once the wire contract is
frozen.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import struct
import unicodedata
from typing import Any

_I64_RE = re.compile(r"^(0|-?[1-9][0-9]*)$")
_F64_RE = re.compile(r"^[0-9a-f]{16}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")
_I64_MIN = -(2**63)
_I64_MAX = 2**63 - 1
_INT32_MAX = 2**31 - 1
_MAX_SAFE_JSON_INT = 2**53 - 1
_MAX_EXACT_BINARY64_INT = 2**53
_AXIS_KINDS = {"int", "float", "categorical", "ordinal", "sorted_tuple"}
_NAMESPACES = {"operator", "fit", "control", "structural"}
_CONSTRAINT_KINDS = {"mutex_group", "requires", "exclude"}


class ContractError(ValueError):
    """Ordered search-space contract violation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _require_unicode_scalar_text(value: str, label: str) -> None:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ContractError(
            f"{label} must contain valid Unicode scalar values"
        ) from exc


def _text(value: Any, label: str, *, axis_name: bool = False) -> str:
    _require(
        isinstance(value, str) and bool(value), f"{label} must be a non-empty string"
    )
    _require_unicode_scalar_text(value, label)
    _require("\x00" not in value, f"{label} contains NUL")
    _require(
        unicodedata.normalize("NFC", value) == value, f"{label} must be NFC-normalized"
    )
    if axis_name:
        _require("#" not in value, f"{label} contains reserved '#'")
    return value


def _typed_scalar(value: Any, label: str) -> tuple[str, Any]:
    _require(
        isinstance(value, dict) and len(value) == 1, f"{label} must be one typed scalar"
    )
    kind, encoded = next(iter(value.items()))
    if kind == "i64":
        _require(
            isinstance(encoded, str) and _I64_RE.fullmatch(encoded) is not None,
            f"{label}.i64 is not canonical",
        )
        parsed = int(encoded)
        _require(_I64_MIN <= parsed <= _I64_MAX, f"{label}.i64 is outside int64")
        return kind, parsed
    if kind == "f64_bits":
        _require(
            isinstance(encoded, str) and _F64_RE.fullmatch(encoded) is not None,
            f"{label}.f64_bits is not canonical",
        )
        _require(encoded != "8000000000000000", f"{label} must normalize negative zero")
        parsed = struct.unpack(">d", bytes.fromhex(encoded))[0]
        _require(math.isfinite(parsed), f"{label} must be finite")
        return kind, parsed
    if kind == "string":
        _require(isinstance(encoded, str), f"{label}.string must be a string")
        _require(bool(encoded), f"{label}.string must be non-empty")
        _require_unicode_scalar_text(encoded, f"{label}.string")
        _require("\x00" not in encoded, f"{label}.string contains NUL")
        _require(
            unicodedata.normalize("NFC", encoded) == encoded,
            f"{label}.string must be NFC-normalized",
        )
        return kind, encoded
    if kind == "bool":
        _require(type(encoded) is bool, f"{label}.bool must be boolean")
        return kind, encoded
    raise ContractError(f"{label} has unknown scalar kind {kind!r}")


def _require_exact_binary64_int(value: int, label: str) -> None:
    _require(
        -_MAX_EXACT_BINARY64_INT <= value <= _MAX_EXACT_BINARY64_INT,
        f"{label} cannot be represented exactly by the current binary64 C ABI",
    )


def _structural_int(
    value: Any,
    label: str,
    *,
    minimum: int = 0,
    maximum: int = _MAX_SAFE_JSON_INT,
) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{label} must be an integer",
    )
    _require(minimum <= value <= maximum, f"{label} is outside its portable range")
    return value


def _validate_target(target: Any, label: str) -> tuple[str, str, tuple[str, ...]]:
    _require(isinstance(target, dict), f"{label} must be an object")
    _require(
        set(target) == {"node_id", "namespace", "path"}, f"{label} has invalid fields"
    )
    node_id = _text(target["node_id"], f"{label}.node_id")
    _require(
        len(node_id) <= 128 and _IDENTIFIER_RE.fullmatch(node_id) is not None,
        f"{label}.node_id must be a portable DAG-ML identifier",
    )
    namespace = target["namespace"]
    _require(namespace in _NAMESPACES, f"{label}.namespace is invalid")
    path = target["path"]
    _require(isinstance(path, list) and bool(path), f"{label}.path must be non-empty")
    return node_id, namespace, tuple(_text(part, f"{label}.path") for part in path)


def _validate_domain(axis: dict[str, Any], index: int) -> int | None:
    name = axis["name"]
    kind = axis["kind"]
    domain = axis["domain"]
    _require(isinstance(domain, dict), f"axis {name!r} domain must be an object")
    if kind in {"int", "float"}:
        _require(
            set(domain) == {"low", "high", "step", "log"},
            f"axis {name!r} numeric domain fields are invalid",
        )
        low_kind, low = _typed_scalar(domain["low"], f"axes[{index}].domain.low")
        high_kind, high = _typed_scalar(domain["high"], f"axes[{index}].domain.high")
        step_kind, step = _typed_scalar(domain["step"], f"axes[{index}].domain.step")
        expected = "i64" if kind == "int" else "f64_bits"
        _require(
            {low_kind, high_kind, step_kind} == {expected},
            f"axis {name!r} numeric scalar types do not match kind",
        )
        _require(low <= high, f"axis {name!r} low exceeds high")
        _require(type(domain["log"]) is bool, f"axis {name!r} log must be boolean")
        if kind == "int":
            _require_exact_binary64_int(low, f"axis {name!r} low")
            _require_exact_binary64_int(high, f"axis {name!r} high")
            _require_exact_binary64_int(step, f"axis {name!r} step")
            _require(step > 0, f"axis {name!r} integer step must be positive")
            if domain["log"]:
                _require(step == 1, f"axis {name!r} log-integer step must be one")
        else:
            _require(step >= 0, f"axis {name!r} step is negative")
            if domain["log"]:
                _require(step == 0, f"axis {name!r} log-float step must be zero")
        if domain["log"]:
            _require(low > 0 and high > 0, f"axis {name!r} log domain must be positive")
        return None
    if kind in {"categorical", "ordinal"}:
        _require(
            set(domain) == {"choices"},
            f"axis {name!r} choice domain fields are invalid",
        )
        choices = domain["choices"]
        _require(
            isinstance(choices, list) and bool(choices),
            f"axis {name!r} choices must be non-empty",
        )
        canonical = [
            _typed_scalar(choice, f"axes[{index}].domain.choices") for choice in choices
        ]
        choice_kinds = {choice_kind for choice_kind, _ in canonical}
        _require(
            len(choice_kinds) == 1,
            f"axis {name!r} choices must use one homogeneous scalar kind",
        )
        _require(
            len(set(canonical)) == len(canonical),
            f"axis {name!r} has duplicate choices",
        )
        if kind == "ordinal":
            _require(
                all(choice_kind in {"i64", "f64_bits"} for choice_kind, _ in canonical),
                f"axis {name!r} ordinal choices must be numeric",
            )
        if next(iter(choice_kinds)) == "i64":
            for _, choice in canonical:
                _require_exact_binary64_int(choice, f"axis {name!r} choice")
        return len(choices)
    _require(kind == "sorted_tuple", f"axis {name!r} kind is invalid")
    _require(
        set(domain) == {"length", "element_kind", "low", "high"},
        f"axis {name!r} tuple domain fields are invalid",
    )
    _structural_int(
        domain["length"],
        f"axis {name!r} tuple length",
        minimum=1,
        maximum=_INT32_MAX,
    )
    element_kind = domain["element_kind"]
    _require(
        element_kind in {"int", "float"}, f"axis {name!r} tuple element_kind is invalid"
    )
    low_kind, low = _typed_scalar(domain["low"], f"axes[{index}].domain.low")
    high_kind, high = _typed_scalar(domain["high"], f"axes[{index}].domain.high")
    expected = "i64" if element_kind == "int" else "f64_bits"
    _require(
        low_kind == expected and high_kind == expected,
        f"axis {name!r} tuple bounds do not match element_kind",
    )
    _require(low <= high, f"axis {name!r} tuple low exceeds high")
    if element_kind == "int":
        _require_exact_binary64_int(low, f"axis {name!r} tuple low")
        _require_exact_binary64_int(high, f"axis {name!r} tuple high")
    return None


def validate_ordered_search_space(spec: Any) -> None:
    """Validate the semantic V1 contract or raise :class:`ContractError`."""

    _require(isinstance(spec, dict), "contract must be an object")
    _require(
        set(spec) == {"schema", "schema_version", "axes", "constraints"},
        "contract fields are invalid",
    )
    _require(spec["schema"] == "nirs4all.ordered-search-space", "schema id is invalid")
    _require(
        isinstance(spec["schema_version"], int)
        and not isinstance(spec["schema_version"], bool)
        and spec["schema_version"] == 1,
        "schema_version must be integer 1",
    )
    axes = spec["axes"]
    _require(
        isinstance(axes, list) and bool(axes), "axes must be a non-empty ordered list"
    )

    by_name: dict[str, dict[str, Any]] = {}
    choice_counts: dict[str, int | None] = {}
    targets: set[tuple[str, str, tuple[str, ...]]] = set()
    parents: dict[str, str] = {}
    for index, axis in enumerate(axes):
        _require(isinstance(axis, dict), f"axes[{index}] must be an object")
        required = {"name", "kind", "target", "domain", "activation", "inactive_policy"}
        _require(set(axis) == required, f"axes[{index}] fields are invalid")
        name = _text(axis["name"], f"axes[{index}].name", axis_name=True)
        _require(name not in by_name, f"duplicate axis name {name!r}")
        _require(axis["kind"] in _AXIS_KINDS, f"axis {name!r} kind is invalid")
        target = _validate_target(axis["target"], f"axes[{index}].target")
        _require(
            target not in targets, f"duplicate ParameterPatch target for axis {name!r}"
        )
        _require(
            axis["inactive_policy"] == "sample_then_omit",
            f"axis {name!r} inactive_policy is invalid",
        )
        choice_counts[name] = _validate_domain(axis, index)
        activation = axis["activation"]
        _require(
            isinstance(activation, dict), f"axis {name!r} activation must be an object"
        )
        mode = activation.get("mode")
        if mode == "always":
            _require(
                set(activation) == {"mode"},
                f"axis {name!r} always activation has extra fields",
            )
        else:
            _require(
                mode in {"when", "when_not"},
                f"axis {name!r} activation mode is invalid",
            )
            _require(
                set(activation) == {"mode", "parent", "choice_index"},
                f"axis {name!r} conditional activation fields are invalid",
            )
            parents[name] = _text(
                activation["parent"], f"axis {name!r} activation parent", axis_name=True
            )
            _structural_int(
                activation["choice_index"],
                f"axis {name!r} activation choice_index",
            )
        by_name[name] = axis
        targets.add(target)

    for child, parent in parents.items():
        _require(parent in by_name, f"axis {child!r} has unknown parent {parent!r}")
        count = choice_counts[parent]
        _require(
            count is not None,
            f"axis {child!r} parent {parent!r} is not categorical or ordinal",
        )
        choice_index = by_name[child]["activation"]["choice_index"]
        _require(
            choice_index < count, f"axis {child!r} parent choice_index is out of range"
        )

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visited:
            return
        _require(name not in visiting, f"activation cycle contains axis {name!r}")
        visiting.add(name)
        if name in parents:
            visit(parents[name])
        visiting.remove(name)
        visited.add(name)

    for name in by_name:
        visit(name)

    constraints = spec["constraints"]
    _require(isinstance(constraints, list), "constraints must be an ordered list")
    for index, constraint in enumerate(constraints):
        _require(
            isinstance(constraint, dict) and set(constraint) == {"kind", "refs"},
            f"constraints[{index}] fields are invalid",
        )
        kind = constraint["kind"]
        _require(kind in _CONSTRAINT_KINDS, f"constraints[{index}] kind is invalid")
        refs = constraint["refs"]
        expected = 2 if kind in {"requires", "exclude"} else None
        _require(
            isinstance(refs, list) and len(refs) >= 2,
            f"constraints[{index}] needs at least two refs",
        )
        if expected is not None:
            _require(
                len(refs) == expected,
                f"constraints[{index}] {kind} needs exactly two refs",
            )
        canonical_refs: list[tuple[str, int | None]] = []
        for ref_index, ref in enumerate(refs):
            _require(
                isinstance(ref, dict)
                and set(ref) in ({"axis"}, {"axis", "choice_index"}),
                f"constraints[{index}].refs[{ref_index}] fields are invalid",
            )
            axis_name = _text(
                ref["axis"],
                f"constraints[{index}].refs[{ref_index}].axis",
                axis_name=True,
            )
            _require(
                axis_name in by_name,
                f"constraints[{index}] references unknown axis {axis_name!r}",
            )
            choice_index = ref.get("choice_index")
            if choice_index is not None:
                _structural_int(
                    choice_index,
                    f"constraints[{index}] choice_index",
                )
                count = choice_counts[axis_name]
                _require(
                    count is not None and choice_index < count,
                    f"constraints[{index}] choice_index is out of range",
                )
            canonical_refs.append((axis_name, choice_index))
        _require(
            len(set(canonical_refs)) == len(canonical_refs),
            f"constraints[{index}] has duplicate refs",
        )


def _binary64_bits(value: float) -> str:
    return struct.pack(">d", value).hex()


def _native_choice_label(choice: Any, label: str) -> str:
    kind, value = _typed_scalar(choice, label)
    if kind == "string":
        return value
    if kind == "bool":
        return "true" if value else "false"
    if kind == "i64":
        return str(value)
    return format(value, ".17g")


def c_abi_projection(spec: Any) -> dict[str, Any]:
    """Project the full host contract onto the current ordered C builders.

    ``target`` and ``inactive_policy`` intentionally remain host-owned and are
    still covered by :func:`space_fingerprint`; this projection is not itself a
    portable search-space wire format.
    """

    validate_ordered_search_space(spec)
    axes = spec["axes"]
    by_name = {axis["name"]: axis for axis in axes}
    params: list[dict[str, Any]] = []
    constraints: list[dict[str, Any]] = []

    for index, axis in enumerate(axes):
        name = axis["name"]
        kind = axis["kind"]
        domain = axis["domain"]
        if kind == "int":
            params.append(
                {
                    "call": "n4m_search_space_add_int",
                    "name": name,
                    "low": _typed_scalar(domain["low"], f"axes[{index}].low")[1],
                    "high": _typed_scalar(domain["high"], f"axes[{index}].high")[1],
                    "step": _typed_scalar(domain["step"], f"axes[{index}].step")[1],
                    "log": domain["log"],
                }
            )
        elif kind == "float":
            params.append(
                {
                    "call": "n4m_search_space_add_float",
                    "name": name,
                    "low_f64_bits": domain["low"]["f64_bits"],
                    "high_f64_bits": domain["high"]["f64_bits"],
                    "step_f64_bits": domain["step"]["f64_bits"],
                    "log": domain["log"],
                }
            )
        elif kind == "categorical":
            choices = domain["choices"]
            scalar_kind = next(iter(choices[0]))
            params.append(
                {
                    "call": "n4m_search_space_add_categorical",
                    "name": name,
                    "cat_type": scalar_kind,
                    "values": copy.deepcopy(choices),
                }
            )
        elif kind == "ordinal":
            ordinal_bits = []
            for choice_index, choice in enumerate(domain["choices"]):
                _, value = _typed_scalar(
                    choice, f"axes[{index}].choices[{choice_index}]"
                )
                ordinal_bits.append(_binary64_bits(float(value)))
            params.append(
                {
                    "call": "n4m_search_space_add_ordinal",
                    "name": name,
                    "values_f64_bits": ordinal_bits,
                }
            )
        else:
            low_kind, low = _typed_scalar(domain["low"], f"axes[{index}].low")
            _, high = _typed_scalar(domain["high"], f"axes[{index}].high")
            params.append(
                {
                    "call": "n4m_search_space_add_sorted_tuple",
                    "name": name,
                    "length": domain["length"],
                    "element_is_int": domain["element_kind"] == "int",
                    "low": low if low_kind == "i64" else _binary64_bits(low),
                    "high": high if low_kind == "i64" else _binary64_bits(high),
                    "bound_encoding": "i64" if low_kind == "i64" else "f64_bits",
                }
            )

        activation = axis["activation"]
        if activation["mode"] != "always":
            parent = by_name[activation["parent"]]
            parent_choice = parent["domain"]["choices"][activation["choice_index"]]
            constraints.append(
                {
                    "kind": (
                        "N4M_CONSTRAINT_CONDITION_IN"
                        if activation["mode"] == "when"
                        else "N4M_CONSTRAINT_CONDITION_NOT_IN"
                    ),
                    "param_refs": [name, parent["name"]],
                    "label_refs": [
                        None,
                        _native_choice_label(
                            parent_choice, f"axis {name!r} activation choice"
                        ),
                    ],
                }
            )

    kind_map = {
        "mutex_group": "N4M_CONSTRAINT_MUTEX_GROUP",
        "requires": "N4M_CONSTRAINT_REQUIRES",
        "exclude": "N4M_CONSTRAINT_EXCLUDE",
    }
    for constraint_index, constraint in enumerate(spec["constraints"]):
        param_refs: list[str] = []
        label_refs: list[str | None] = []
        for ref_index, ref in enumerate(constraint["refs"]):
            param_refs.append(ref["axis"])
            choice_index = ref.get("choice_index")
            if choice_index is None:
                label_refs.append(None)
            else:
                choice = by_name[ref["axis"]]["domain"]["choices"][choice_index]
                label_refs.append(
                    _native_choice_label(
                        choice,
                        f"constraints[{constraint_index}].refs[{ref_index}] choice",
                    )
                )
        constraints.append(
            {
                "kind": kind_map[constraint["kind"]],
                "param_refs": param_refs,
                "label_refs": label_refs,
            }
        )

    return {"params": params, "constraints": constraints}


def _restricted_jcs_encode(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, list):
        return "[" + ",".join(_restricted_jcs_encode(item) for item in value) + "]"
    if isinstance(value, dict):
        _require(
            all(isinstance(key, str) for key in value),
            "canonical JSON object keys must be strings",
        )
        # RFC 8785 sorts object property names by their UTF-16 code units, not
        # by Unicode scalar value or UTF-8 bytes.
        keys = sorted(value, key=lambda key: key.encode("utf-16-be"))
        return (
            "{"
            + ",".join(
                f"{_restricted_jcs_encode(key)}:{_restricted_jcs_encode(value[key])}"
                for key in keys
            )
            + "}"
        )
    raise ContractError(
        f"canonical JSON contains unsupported value type {type(value).__name__}"
    )


def canonical_json_bytes(spec: Any) -> bytes:
    """Return RFC 8785/JCS bytes for the restricted V1 value domain.

    V1 deliberately represents all tunable numeric values as tagged strings.
    Consequently its only JSON number tokens are non-negative structural
    integers, and the complicated ECMAScript float-rendering part of JCS is not
    needed by this independent test oracle.
    """

    validate_ordered_search_space(spec)
    return _restricted_jcs_encode(spec).encode("utf-8")


def space_fingerprint(spec: Any) -> str:
    """Return the pinned SHA-256 fingerprint for a validated space."""

    return "sha256:" + hashlib.sha256(canonical_json_bytes(spec)).hexdigest()
