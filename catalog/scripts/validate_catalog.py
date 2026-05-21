#!/usr/bin/env python3
"""Catalog validation gate.

Runs in CI on every PR. Failure modes (in order of severity):

1. methods.yaml or any subset YAML doesn't validate against the JSON schema.
2. Two methods share the same method_id.
3. A method's TU path doesn't exist in the working tree.
4. An ACTIVE subset (status: active) refers to a category that has zero methods
   in methods.yaml, or to a method_id that doesn't exist, or excludes everything
   it included.
5. A PLANNED subset (status: planned) breaks the schema. (Other contents are
   advisory until the subset activates.)

Exits 0 on success, 1 on failure. Prints a structured report.

Stdlib-only — no PyYAML / jsonschema runtime dependency. The YAML parsing is
deliberately minimal (matches the conservative emission style of
discover_methods.py + the hand-written subset YAMLs). If we ever need full
YAML, swap to `import yaml` here only and document the dep.
"""

from __future__ import annotations
import fnmatch
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
CATALOG = REPO / "catalog"
SCHEMA_DIR = CATALOG / "schema"


# ----------- minimal YAML parser (whitespace-significant, no anchors / flow) -----------
# Supports: top-level scalar:value, scalar: <newline> indented block, list items "- ", inline empty {} / [].
# Sufficient for methods.yaml (emitted by us) + the subset YAMLs (hand-written but disciplined).

def parse_yaml(text: str) -> Any:
    lines = text.splitlines()
    pos = [0]

    def peek_indent(i: int) -> int:
        s = lines[i]
        return len(s) - len(s.lstrip(" "))

    def skip_blank_and_comments() -> None:
        while pos[0] < len(lines):
            s = lines[pos[0]]
            stripped = s.strip()
            if not stripped or stripped.startswith("#"):
                pos[0] += 1
            else:
                return

    def strip_inline_comment(token: str) -> str:
        # Strip trailing "# ..." but not inside a quoted string.
        # Simple state machine: walk chars, drop everything after the first
        # unquoted '#' preceded by whitespace.
        in_dq = False
        in_sq = False
        prev_ws = True   # start-of-line counts as whitespace boundary
        for i, c in enumerate(token):
            if c == '"' and not in_sq:
                in_dq = not in_dq
            elif c == "'" and not in_dq:
                in_sq = not in_sq
            if c == "#" and not in_dq and not in_sq and prev_ws:
                return token[:i].rstrip()
            prev_ws = c.isspace()
        return token

    def parse_scalar(token: str) -> Any:
        token = strip_inline_comment(token).strip()
        if not token:
            return ""
        if token in ("null", "~"):
            return None
        if token == "true":
            return True
        if token == "false":
            return False
        if token == "[]":
            return []
        if token == "{}":
            return {}
        # quoted string
        if (token.startswith('"') and token.endswith('"')) or (token.startswith("'") and token.endswith("'")):
            inner = token[1:-1]
            # minimal unescape
            if token.startswith('"'):
                inner = inner.replace('\\"', '"').replace("\\\\", "\\")
            return inner
        # block-literal start
        if token == "|":
            return ("__BLOCK_LITERAL__",)
        # integer / float
        try:
            if "." not in token and "e" not in token.lower():
                return int(token)
        except (ValueError, TypeError):
            pass
        try:
            return float(token)
        except (ValueError, TypeError):
            pass
        return token

    def parse_block_literal(base_indent: int) -> str:
        out_lines: list[str] = []
        while pos[0] < len(lines):
            s = lines[pos[0]]
            if not s.strip():
                out_lines.append("")
                pos[0] += 1
                continue
            ind = peek_indent(pos[0])
            if ind <= base_indent:
                break
            out_lines.append(s[base_indent + 2:] if ind >= base_indent + 2 else s[base_indent:].lstrip())
            pos[0] += 1
        return "\n".join(out_lines).rstrip() + "\n"

    def parse_block(parent_indent: int) -> Any:
        skip_blank_and_comments()
        if pos[0] >= len(lines):
            return None
        first = lines[pos[0]]
        ind = peek_indent(pos[0])
        if ind <= parent_indent:
            return None
        stripped = first.lstrip(" ")
        if stripped.startswith("- "):
            return parse_list(ind)
        if ":" in stripped:
            return parse_map(ind)
        # plain scalar?
        pos[0] += 1
        return parse_scalar(stripped)

    def parse_list(indent: int) -> list[Any]:
        out: list[Any] = []
        while True:
            skip_blank_and_comments()
            if pos[0] >= len(lines):
                break
            s = lines[pos[0]]
            ind = peek_indent(pos[0])
            if ind < indent:
                break
            if ind > indent:
                # nested element — shouldn't happen at list root
                pos[0] += 1
                continue
            stripped = s.lstrip(" ")
            if not stripped.startswith("- "):
                break
            rest = stripped[2:]
            if rest.strip() == "":
                # nested mapping/list on next lines
                pos[0] += 1
                child = parse_block(indent + 1)
                out.append(child)
            elif ":" in rest and not (rest.strip().startswith('"') or rest.strip().startswith("'")):
                # inline first key of a mapping list item
                # Treat rest as the first line of a mapping, then continue parsing more keys at indent+2
                key, _, val = rest.partition(":")
                key = key.strip()
                val_raw = val.strip()
                inline_value: Any
                if val_raw == "":
                    pos[0] += 1
                    inline_value = parse_block(indent + 1)
                else:
                    pos[0] += 1
                    inline_value = parse_scalar(val_raw)
                # Continue collecting more keys for this mapping at indent+2
                mapping: dict[str, Any] = {key: inline_value}
                child_indent = indent + 2
                while pos[0] < len(lines):
                    skip_blank_and_comments()
                    if pos[0] >= len(lines):
                        break
                    s2 = lines[pos[0]]
                    ind2 = peek_indent(pos[0])
                    if ind2 < child_indent:
                        break
                    if ind2 > child_indent:
                        pos[0] += 1
                        continue
                    stripped2 = s2.lstrip(" ")
                    if stripped2.startswith("- "):
                        break
                    if ":" not in stripped2:
                        break
                    k2, _, v2 = stripped2.partition(":")
                    k2 = k2.strip()
                    v2 = v2.strip()
                    if v2 == "":
                        pos[0] += 1
                        mapping[k2] = parse_block(child_indent)
                    else:
                        pos[0] += 1
                        sc = parse_scalar(v2)
                        if isinstance(sc, tuple) and sc and sc[0] == "__BLOCK_LITERAL__":
                            mapping[k2] = parse_block_literal(child_indent)
                        else:
                            mapping[k2] = sc
                out.append(mapping)
            else:
                # scalar list item
                pos[0] += 1
                out.append(parse_scalar(rest))
        return out

    def parse_map(indent: int) -> dict[str, Any]:
        out: dict[str, Any] = {}
        while True:
            skip_blank_and_comments()
            if pos[0] >= len(lines):
                break
            s = lines[pos[0]]
            ind = peek_indent(pos[0])
            if ind < indent:
                break
            if ind > indent:
                pos[0] += 1
                continue
            stripped = s.lstrip(" ")
            if stripped.startswith("- "):
                break
            if ":" not in stripped:
                break
            key, _, val = stripped.partition(":")
            key = key.strip()
            val_raw = val.strip()
            if val_raw == "":
                pos[0] += 1
                out[key] = parse_block(indent)
            else:
                pos[0] += 1
                sc = parse_scalar(val_raw)
                if isinstance(sc, tuple) and sc and sc[0] == "__BLOCK_LITERAL__":
                    out[key] = parse_block_literal(indent)
                else:
                    out[key] = sc
        return out

    skip_blank_and_comments()
    if pos[0] >= len(lines):
        return None
    first_ind = peek_indent(pos[0])
    stripped = lines[pos[0]].lstrip(" ")
    if stripped.startswith("- "):
        return parse_list(first_ind)
    return parse_map(first_ind)


# ----------- minimal JSON-schema validation -----------
# Covers what we use in method_v1 and subset_v1: type, enum, required, properties,
# additionalProperties, pattern, items, minItems, default. Skips $ref / oneOf / allOf.

def validate_schema(obj: Any, schema: dict, path: str = "$") -> list[str]:
    errs: list[str] = []
    # Type-Match shortcut for nullable
    typ = schema.get("type")
    if typ:
        types = typ if isinstance(typ, list) else [typ]
        type_ok = False
        for t in types:
            if t == "string"  and isinstance(obj, str):  type_ok = True
            if t == "integer" and isinstance(obj, int) and not isinstance(obj, bool): type_ok = True
            if t == "number"  and isinstance(obj, (int, float)) and not isinstance(obj, bool): type_ok = True
            if t == "boolean" and isinstance(obj, bool): type_ok = True
            if t == "null"    and obj is None: type_ok = True
            if t == "object"  and isinstance(obj, dict): type_ok = True
            if t == "array"   and isinstance(obj, list): type_ok = True
        if not type_ok:
            errs.append(f"{path}: type {typ} expected, got {type(obj).__name__}")
            return errs
    if obj is None:
        return errs
    if "enum" in schema and obj not in schema["enum"]:
        errs.append(f"{path}: value {obj!r} not in enum {schema['enum']}")
    if "pattern" in schema and isinstance(obj, str):
        if not re.search(schema["pattern"], obj):
            errs.append(f"{path}: value {obj!r} does not match pattern {schema['pattern']!r}")
    if "minItems" in schema and isinstance(obj, list):
        if len(obj) < schema["minItems"]:
            errs.append(f"{path}: array has {len(obj)} items, minItems={schema['minItems']}")
    if isinstance(obj, dict):
        for req in schema.get("required", []):
            if req not in obj:
                errs.append(f"{path}: required key {req!r} missing")
        if schema.get("additionalProperties") is False:
            for k in obj:
                if k not in schema.get("properties", {}):
                    errs.append(f"{path}: additional property {k!r} not allowed")
        for k, sub in schema.get("properties", {}).items():
            if k in obj:
                errs.extend(validate_schema(obj[k], sub, f"{path}.{k}"))
    if isinstance(obj, list) and "items" in schema:
        for i, item in enumerate(obj):
            errs.extend(validate_schema(item, schema["items"], f"{path}[{i}]"))
    return errs


def load_yaml(path: Path) -> Any:
    return parse_yaml(path.read_text(encoding="utf-8"))


def main() -> int:
    print(f"== validate_catalog.py ==  repo={REPO}")
    # 1) load schemas
    try:
        method_schema = json.loads((SCHEMA_DIR / "method_v1.json").read_text())
        subset_schema = json.loads((SCHEMA_DIR / "subset_v1.json").read_text())
    except Exception as e:
        print(f"FAIL: cannot load schemas: {e}")
        return 1

    fail_count = 0

    # 2) methods.yaml
    methods_path = CATALOG / "methods.yaml"
    if not methods_path.exists():
        print("FAIL: catalog/methods.yaml is missing")
        return 1
    methods_doc = load_yaml(methods_path)
    if not isinstance(methods_doc, dict) or "methods" not in methods_doc:
        print(f"FAIL: methods.yaml top-level must be a mapping with 'methods'; got {type(methods_doc).__name__}")
        return 1
    methods = methods_doc["methods"]
    print(f"  methods.yaml — {len(methods)} entries")

    seen_ids: set[str] = set()
    method_index: dict[str, dict] = {}
    for i, m in enumerate(methods):
        if not isinstance(m, dict):
            print(f"FAIL: methods[{i}] is not a mapping")
            fail_count += 1
            continue
        errs = validate_schema(m, method_schema, f"$methods[{i}]")
        for e in errs:
            print(f"  SCHEMA: {e}")
            fail_count += 1
        mid = m.get("method_id")
        if not mid:
            print(f"  FAIL: methods[{i}] has no method_id")
            fail_count += 1
            continue
        if mid in seen_ids:
            print(f"  FAIL: duplicate method_id {mid!r} at index {i}")
            fail_count += 1
        seen_ids.add(mid)
        method_index[mid] = m
        # TU existence
        for tu in m.get("tu", []):
            # strip any ":symbol_pattern" suffix
            tu_path = tu.split(":", 1)[0]
            if not (REPO / tu_path).exists():
                print(f"  FAIL: methods[{i}] ({mid}) TU path missing: {tu_path}")
                fail_count += 1

    if fail_count == 0:
        print(f"  methods.yaml OK ({len(methods)} entries, {len(seen_ids)} unique ids)")

    # 3) subsets
    subset_dir = CATALOG / "subsets"
    subset_files = sorted(subset_dir.glob("*.yaml"))
    if not subset_files:
        print("FAIL: catalog/subsets/ contains no YAML")
        return 1
    print(f"  subsets — {len(subset_files)} descriptors")

    method_categories: dict[str, set[str]] = {}
    for m in methods:
        cat = m.get("category")
        if cat:
            method_categories.setdefault(cat, set()).add(m["method_id"])

    for sf in subset_files:
        s = load_yaml(sf)
        if not isinstance(s, dict):
            print(f"  FAIL: {sf.name} top-level is not a mapping")
            fail_count += 1
            continue
        errs = validate_schema(s, subset_schema, f"${sf.name}")
        for e in errs:
            print(f"  SCHEMA: {sf.name}: {e}")
            fail_count += 1
        status = s.get("status")
        if status not in {"active", "planned", "deprecated"}:
            # already caught by schema, but be sure
            continue
        if status != "active":
            print(f"  {sf.name}: status={status} (closure check skipped)")
            continue

        # Active subset closure check
        included_ids: set[str] = set()
        inc = s.get("includes") or {}
        for cat in inc.get("categories") or []:
            ids = method_categories.get(cat, set())
            if not ids:
                print(f"  FAIL: {sf.name} active subset includes category {cat!r} but methods.yaml has 0 entries in it")
                fail_count += 1
            included_ids.update(ids)
        for mid in inc.get("method_ids") or []:
            if mid not in method_index:
                print(f"  FAIL: {sf.name} active subset includes method_id {mid!r} not in methods.yaml")
                fail_count += 1
            else:
                included_ids.add(mid)
        for pat in inc.get("patterns") or []:
            matched = [mid for mid in method_index if fnmatch.fnmatch(mid, pat)]
            included_ids.update(matched)

        exc = s.get("excludes") or {}
        excluded_ids: set[str] = set()
        for cat in exc.get("categories") or []:
            excluded_ids.update(method_categories.get(cat, set()))
        for mid in exc.get("method_ids") or []:
            excluded_ids.add(mid)
        for pat in exc.get("patterns") or []:
            excluded_ids.update(mid for mid in method_index if fnmatch.fnmatch(mid, pat))

        final_ids = included_ids - excluded_ids
        if not final_ids:
            print(f"  FAIL: {sf.name} active subset closure is EMPTY after excludes")
            fail_count += 1
        else:
            print(f"  {sf.name}: closure OK ({len(final_ids)} methods after excludes)")

    print("--")
    if fail_count == 0:
        print("PASS: catalog validation green")
        return 0
    print(f"FAIL: {fail_count} catalog validation error(s)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
