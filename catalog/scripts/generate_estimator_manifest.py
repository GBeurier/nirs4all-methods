#!/usr/bin/env python3
"""Generate the compiled estimator manifest from catalog `estimator` blocks.

The catalog entry is the authoring source; the generated C++ table is the
runtime authority exposed by `n4m_method_*` introspection.

    python catalog/scripts/generate_estimator_manifest.py --write
    python catalog/scripts/generate_estimator_manifest.py --check
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import Any

from catalog_loader import METHODS_DIR, REPO, load_yaml_file

OUTPUT = REPO / "cpp" / "src" / "core" / "estimator" / "generated_manifest.inc"
FACTORIES = REPO / "cpp" / "src" / "core" / "estimator" / "generated_factories.hpp"

ROLES = {
    "transformer": "N4M_ROLE_TRANSFORMER",
    "regressor": "N4M_ROLE_REGRESSOR",
    "classifier": "N4M_ROLE_CLASSIFIER",
    "selector": "N4M_ROLE_SELECTOR",
    "sample_filter": "N4M_ROLE_SAMPLE_FILTER",
}
KINDS = {"estimator": "N4M_METHOD_ESTIMATOR", "procedure": "N4M_METHOD_PROCEDURE"}
INPUTS = [
    "y",
    "labels",
    "sample_weight",
    "groups",
    "feature_groups",
    "blocks",
    "axis",
    "target_domain",
    "fold_ids",
]
REQUIREMENTS = {
    "none": "N4M_INPUT_NONE",
    "optional": "N4M_INPUT_OPTIONAL",
    "required": "N4M_INPUT_REQUIRED",
}
PARAM_TYPES = {
    "int": "N4M_METHOD_PARAM_INT",
    "double": "N4M_METHOD_PARAM_DOUBLE",
    "bool": "N4M_METHOD_PARAM_BOOL",
    "enum": "N4M_METHOD_PARAM_ENUM",
    "int_array": "N4M_METHOD_PARAM_INT_ARRAY",
    "double_array": "N4M_METHOD_PARAM_DOUBLE_ARRAY",
}
INT_TYPES = {"int", "bool", "enum", "int_array"}


class ManifestError(ValueError):
    pass


def inline_list(value: Any) -> list[str]:
    """Parse the catalog parser's `[a, b]` scalar into a list of strings."""
    if isinstance(value, list):
        return [str(v) for v in value]
    text = str(value).strip()
    if not (text.startswith("[") and text.endswith("]")):
        return [text]
    body = text[1:-1].strip()
    return [item.strip().strip('"') for item in body.split(",")] if body else []


def c_ident(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text)


def c_string(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def c_double(value: float) -> str:
    if math.isnan(value):
        return "kNaN"
    return repr(float(value))


def parse_param(method_id: str, raw: dict[str, Any]) -> dict[str, Any]:
    name = str(raw.get("name", ""))
    ptype = str(raw.get("type", ""))
    if not name or ptype not in PARAM_TYPES:
        raise ManifestError(f"{method_id}: invalid param {raw!r}")
    choices = inline_list(raw["choices"]) if "choices" in raw else []
    if (ptype == "enum") != bool(choices):
        raise ManifestError(
            f"{method_id}.{name}: enum params need choices, others must not have them"
        )
    default: list[Any] | None = None
    if "default" in raw:
        values = (
            inline_list(raw["default"])
            if ptype.endswith("_array")
            else [str(raw["default"]).strip('"')]
        )
        if ptype == "bool":
            if values[0] not in ("true", "false", "True", "False"):
                raise ManifestError(
                    f"{method_id}.{name}: bool default must be true/false"
                )
            default = [1 if values[0] in ("true", "True") else 0]
        elif ptype == "enum":
            if values[0] not in choices:
                raise ManifestError(
                    f"{method_id}.{name}: default {values[0]!r} not in choices"
                )
            default = [choices.index(values[0])]
        elif ptype in INT_TYPES:
            default = [int(v) for v in values]
        else:
            default = [float(v) for v in values]
    lo = float(raw["min"]) if "min" in raw else math.nan
    hi = float(raw["max"]) if "max" in raw else math.nan
    if default is not None and ptype in ("int", "double"):
        v = default[0]
        if (not math.isnan(lo) and v < lo) or (not math.isnan(hi) and v > hi):
            raise ManifestError(f"{method_id}.{name}: default {v} outside [{lo}, {hi}]")
    return {
        "name": name,
        "type": ptype,
        "default": default,
        "min": lo,
        "max": hi,
        "choices": choices,
    }


def load_specs() -> list[dict[str, Any]]:
    specs = []
    for path in sorted(METHODS_DIR.glob("*.yaml")):
        doc = load_yaml_file(path)
        block = doc.get("estimator")
        if block is None:
            continue
        method_id = doc["method_id"]
        kind = str(block.get("kind", ""))
        if kind not in KINDS:
            raise ManifestError(
                f"{method_id}: estimator.kind must be one of {sorted(KINDS)}"
            )
        roles = inline_list(block.get("roles", "[]"))
        if kind == "estimator" and (not roles or any(r not in ROLES for r in roles)):
            raise ManifestError(f"{method_id}: invalid roles {roles}")
        adapter = str(block.get("adapter", ""))
        if not adapter.isidentifier():
            raise ManifestError(f"{method_id}: estimator.adapter must be an identifier")
        inputs = block.get("inputs") or {}
        if not isinstance(inputs, dict) or any(k not in INPUTS for k in inputs):
            raise ManifestError(f"{method_id}: unknown inputs {inputs}")
        if any(str(v) not in REQUIREMENTS for v in inputs.values()):
            raise ManifestError(f"{method_id}: invalid input requirement in {inputs}")
        params = [parse_param(method_id, p) for p in (block.get("params") or [])]
        names = [p["name"] for p in params]
        if len(names) != len(set(names)):
            raise ManifestError(f"{method_id}: duplicate parameter names")
        specs.append(
            {
                "method_id": method_id,
                "fq_name": doc.get("fq_name", method_id),
                "kind": kind,
                "roles": roles,
                "adapter": adapter,
                "state": str(block.get("state", "")),
                "inputs": {k: str(v) for k, v in inputs.items()},
                "params": params,
            }
        )
    return specs


def render_factories(specs: list[dict[str, Any]]) -> str:
    out = [
        "// Generated by catalog/scripts/generate_estimator_manifest.py. Do not edit.",
        "// Adapter factories named by catalog `estimator.adapter` entries.",
        "",
        "#pragma once",
        "",
        "#include <memory>",
        "",
        '#include "core/estimator/spec.hpp"',
        "",
        "namespace n4m::estimator {",
        "",
    ]
    for adapter in sorted({s["adapter"] for s in specs}):
        out.append(f"std::unique_ptr<Adapter> make_{adapter}(const MethodSpec& spec);")
    out += ["", "}  // namespace n4m::estimator", ""]
    return "\n".join(out)


def render(specs: list[dict[str, Any]]) -> str:
    out = [
        "// Generated by catalog/scripts/generate_estimator_manifest.py. Do not edit.",
        "// Source: catalog/methods/<method_id>.yaml `estimator` blocks.",
        "",
    ]
    for s in specs:
        ident = c_ident(s["method_id"])
        for p in s["params"]:
            if p["default"] is not None:
                ctype = "std::int64_t" if p["type"] in INT_TYPES else "double"
                vals = (
                    ", ".join(
                        str(v) if ctype != "double" else c_double(v)
                        for v in p["default"]
                    )
                    or "0"
                )
                out.append(
                    f"const {ctype} kDefault_{ident}_{p['name']}[] = {{{vals}}};"
                )
            if p["choices"]:
                labels = ", ".join(c_string(c) for c in p["choices"])
                out.append(
                    f"const char* const kChoices_{ident}_{p['name']}[] = {{{labels}}};"
                )
        if s["params"]:
            out.append(f"const ParamSpec kParams_{ident}[] = {{")
            for p in s["params"]:
                d = p["default"]
                is_int = p["type"] in INT_TYPES
                d_int = (
                    f"kDefault_{ident}_{p['name']}"
                    if d is not None and is_int
                    else "nullptr"
                )
                d_dbl = (
                    f"kDefault_{ident}_{p['name']}"
                    if d is not None and not is_int
                    else "nullptr"
                )
                choices = f"kChoices_{ident}_{p['name']}" if p["choices"] else "nullptr"
                out.append(
                    f"    {{{c_string(p['name'])}, {PARAM_TYPES[p['type']]}, {'true' if d is not None else 'false'}, "
                    f"{d_int}, {d_dbl}, {len(d) if d is not None else 0}, {c_double(p['min'])}, "
                    f"{c_double(p['max'])}, {choices}, {len(p['choices'])}}},"
                )
            out.append("};")
    out.append("")
    out.append("const MethodSpec kMethods[] = {")
    for s in specs:
        ident = c_ident(s["method_id"])
        roles = " | ".join(ROLES[r] for r in s["roles"]) or "0u"
        inputs = ", ".join(
            REQUIREMENTS[s["inputs"].get(name, "none")] for name in INPUTS
        )
        params = f"kParams_{ident}" if s["params"] else "nullptr"
        out.append(
            f"    {{{c_string(s['method_id'])}, {c_string(s['fq_name'])}, {KINDS[s['kind']]}, {roles}, "
            f"{params}, {len(s['params'])}, {{{inputs}}}, {c_string(s['state'])}, &make_{s['adapter']}}},"
        )
    out.append("};")
    out.append("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        specs = load_specs()
    except ManifestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    outputs = {OUTPUT: render(specs), FACTORIES: render_factories(specs)}
    stale = False
    for path, text in outputs.items():
        if args.write:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            print(f"wrote {path.relative_to(REPO)}")
        elif (path.read_text(encoding="utf-8") if path.exists() else "") != text:
            print(
                f"error: {path.relative_to(REPO)} is stale; rerun with --write",
                file=sys.stderr,
            )
            stale = True
    if not args.write and not stale:
        print("estimator manifest up to date")
    return 1 if stale else 0


if __name__ == "__main__":
    raise SystemExit(main())
