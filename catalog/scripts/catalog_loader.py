#!/usr/bin/env python3
"""Shared catalog IO helpers for Phase B scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from validate_catalog import parse_yaml

REPO = Path(__file__).resolve().parents[2]
CATALOG = REPO / "catalog"
METHODS_DIR = CATALOG / "methods"
LEGACY_METHODS = CATALOG / "methods.yaml"
CATALOG_VERSION = 1

METHOD_FIELD_ORDER = [
    "catalog_version",
    "method_id",
    "family",
    "category",
    "subcategory",
    "since_abi",
    "deprecated_in_abi",
    "removed_in_abi",
    "abi_symbols",
    "tu",
    "headers",
    "carry_along_tu",
    "parity",
    "bench",
    "feature_flags",
    "vendored_licenses",
    "bindings",
    "publications",
    "notes",
]


def method_filename(method_id: str) -> str:
    return f"{method_id}.yaml"


def load_yaml_file(path: Path) -> Any:
    return parse_yaml(path.read_text(encoding="utf-8"))


def load_legacy_methods(path: Path = LEGACY_METHODS) -> list[dict[str, Any]]:
    doc = load_yaml_file(path)
    if not isinstance(doc, dict) or not isinstance(doc.get("methods"), list):
        raise ValueError(f"{path}: expected top-level mapping with methods list")
    methods = []
    for method in doc["methods"]:
        if not isinstance(method, dict):
            raise ValueError(f"{path}: methods list contains non-mapping entry")
        methods.append(dict(method))
    return methods


def normalize_method(method: dict[str, Any]) -> dict[str, Any]:
    out = dict(method)
    out.setdefault("catalog_version", CATALOG_VERSION)
    parity = out.get("parity")
    if isinstance(parity, dict):
        parity = dict(parity)
        parity.setdefault("tolerances", [])
        out["parity"] = parity
    return ordered_method(out)


def ordered_method(method: dict[str, Any]) -> dict[str, Any]:
    ordered: dict[str, Any] = {}
    for key in METHOD_FIELD_ORDER:
        if key in method:
            ordered[key] = method[key]
    for key in sorted(k for k in method if k not in ordered):
        ordered[key] = method[key]
    return ordered


def load_method_files(methods_dir: Path = METHODS_DIR) -> list[dict[str, Any]]:
    files = sorted(methods_dir.glob("*.yaml"))
    methods: list[dict[str, Any]] = []
    for path in files:
        doc = load_yaml_file(path)
        if not isinstance(doc, dict):
            raise ValueError(f"{path}: expected method mapping")
        methods.append(ordered_method(doc))
    return methods


def load_methods(prefer_split: bool = True) -> list[dict[str, Any]]:
    if prefer_split and METHODS_DIR.is_dir() and any(METHODS_DIR.glob("*.yaml")):
        return load_method_files(METHODS_DIR)
    return [normalize_method(method) for method in load_legacy_methods()]


def _quote_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def _scalar(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, str):
        return _quote_string(value)
    raise TypeError(f"unsupported scalar type: {type(value).__name__}")


def dump_yaml(data: Any, indent: int = 0) -> str:
    spaces = " " * indent
    if isinstance(data, dict):
        if not data:
            return "{}"
        lines: list[str] = []
        for key, value in data.items():
            if isinstance(value, (dict, list)) and value:
                lines.append(f"{spaces}{key}:")
                lines.append(dump_yaml(value, indent + 2))
            else:
                lines.append(f"{spaces}{key}: {dump_yaml(value, 0)}")
        return "\n".join(lines)
    if isinstance(data, list):
        if not data:
            return "[]"
        lines = []
        for value in data:
            if isinstance(value, (dict, list)) and value:
                rendered = dump_yaml(value, indent + 2).splitlines()
                lines.append(f"{spaces}- {rendered[0].lstrip()}")
                lines.extend(rendered[1:])
            else:
                lines.append(f"{spaces}- {dump_yaml(value, 0)}")
        return "\n".join(lines)
    return _scalar(data)


def render_method(method: dict[str, Any]) -> str:
    header = (
        "# Auto-generated from catalog/methods.yaml during the Phase B transition.\n"
        "# Edit catalog/methods.yaml, then rerun split_legacy_methods.py.\n"
    )
    return header + dump_yaml(normalize_method(method)) + "\n"
