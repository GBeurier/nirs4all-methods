#!/usr/bin/env python3
# SPDX-License-Identifier: CECILL-2.1
"""Generate ``bindings/python/src/n4m/_ffi_decls.py`` from the public C ABI headers.

The generator parses every ``N4M_API`` function prototype declared in
``cpp/include/n4m/**/*.h``, maps each C type to the ctypes vocabulary used by
``bindings/python/src/n4m/_types.py``, and emits a single ``SYMBOLS`` table of
``(name, argtypes, restype)`` triples plus an ``apply_argtypes(lib)`` binder.

The emitted symbol names are validated **exactly** against
``cpp/abi/expected_symbols_linux.txt`` (minus the leading ``N4M_2`` version-script
node): the header surface and the exported-symbol surface must be identical. The
old→new rename map (``proposals/namespace/_rename_map.tsv``) is used only as a
migration backstop — every old symbol referenced by the binding must resolve to a
header symbol — never to derive signatures (those come from the headers, never the
built ``.so``).

Usage::

    python scripts/generate_ffi_decls.py            # regenerate in place
    python scripts/generate_ffi_decls.py --check     # fail (exit 1) on drift
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INCLUDE_DIR = REPO_ROOT / "cpp" / "include" / "n4m"
ABI_SNAPSHOT = REPO_ROOT / "cpp" / "abi" / "expected_symbols_linux.txt"
RENAME_MAP = REPO_ROOT / "proposals" / "namespace" / "_rename_map.tsv"
OUTPUT = REPO_ROOT / "bindings" / "python" / "src" / "n4m" / "_ffi_decls.py"

# Concrete ctypes value structs that cross the ABI by pointer. Everything else
# named ``n4m_*_t`` is an opaque/forward-declared handle -> c_void_p.
VALUE_STRUCTS = {
    "n4m_matrix_view_t": "MatrixView",
    "n4m_linear_predictor_spec_t": "LinearPredictorSpec",
    "n4m_serialized_model_info_v1_t": "SerializedModelInfoV1",
    "n4m_serialized_pipeline_info_v1_t": "SerializedPipelineInfoV1",
    "n4m_filter_stats_t": "FilterStats",
    "n4m_optimizer_options_t": "OptimizerOptions",
    "n4m_split_result_t": "SplitResult",
    "n4m_transfer_metrics_t": "TransferMetrics",
}

# Populated at generation time from ``typedef enum ... n4m_*_t`` declarations.
# Enums are 32-bit ints in this ABI: by value -> c_int; as a pointer the binding
# convention wraps c_int32 (e.g. ``const n4m_operator_kind_t**`` -> POINTER(POINTER(c_int32))).
ENUM_TYPES: set[str] = set()

# Plain C scalar types -> ctypes (pointer level handled separately).
SCALAR_MAP = {
    "int8_t": "c_int8",
    "uint8_t": "c_uint8",
    "int16_t": "c_int16",
    "uint16_t": "c_uint16",
    "int32_t": "c_int32",
    "uint32_t": "c_uint32",
    "int64_t": "c_int64",
    "uint64_t": "c_uint64",
    "size_t": "c_size_t",
    "double": "c_double",
    "float": "c_float",
    "int": "c_int",
    "unsigned": "c_uint",
    "char": "c_char",  # only meaningful behind a pointer; see _ctype_for
    "void": "c_void_p",  # only valid as void* (level >= 1); guarded below
}

# Header. Kept byte-stable so ``--check`` diffs cleanly across regenerations.
FILE_HEADER = '''# SPDX-License-Identifier: CECILL-2.1
"""Auto-generated ctypes argtypes/restype declarations for libn4m.

Do not edit by hand. Regenerate with::

    python scripts/generate_ffi_decls.py
"""
from __future__ import annotations

from ctypes import (
    POINTER, c_char_p, c_double, c_int, c_int32,
    c_int64, c_size_t, c_uint8, c_uint32, c_uint64, c_void_p,
)

from ._types import MatrixView, LinearPredictorSpec, SerializedModelInfoV1, SerializedPipelineInfoV1, FilterStats, OptimizerOptions, SplitResult, TransferMetrics

# Symbols: tuple of (symbol_name, argtypes_tuple, restype).
# restype = None for void; argtypes = () for parameter-less.
SYMBOLS = (
'''

FILE_FOOTER = ''')


def apply_argtypes(lib):
    """Bind argtypes/restype for every exported function on ``lib``."""
    for name, argtypes, restype in SYMBOLS:
        fn = getattr(lib, name, None)
        if fn is None:
            continue
        fn.argtypes = list(argtypes)
        fn.restype = restype
'''


class GeneratorError(RuntimeError):
    pass


def _strip_comments(text: str) -> str:
    """Remove C block and line comments so prototype parsing is comment-safe."""
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", " ", text)
    return text


def _split_top_level(args: str) -> list[str]:
    """Split a parameter list on top-level commas (ignoring nested parens)."""
    parts: list[str] = []
    depth = 0
    cur = ""
    for ch in args:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur)
    return [p.strip() for p in parts if p.strip()]


def _base_type(decl: str) -> tuple[str, int]:
    """Return ``(base_c_type, pointer_level)`` for a parameter or return decl.

    Strips ``const``/``struct``/``enum`` qualifiers and the trailing identifier;
    treats a trailing ``[...]`` array as one pointer level.
    """
    decl = decl.strip()
    pointer_level = decl.count("*")
    if re.search(r"\[\s*\d*\s*\]\s*$", decl):
        pointer_level += 1
    decl = re.sub(r"\[\s*\d*\s*\]", " ", decl)
    decl = decl.replace("*", " ")
    tokens = [t for t in decl.split() if t not in ("const", "struct", "enum", "restrict")]
    if not tokens:
        raise GeneratorError(f"could not parse type from decl {decl!r}")
    # Heuristic: the last token may be a parameter name; the type is the run of
    # known type tokens. Build the type from the leading tokens, dropping a
    # trailing identifier when more than one token remains.
    if len(tokens) == 1:
        base = tokens[0]
    else:
        # If the last token is not itself a type token (it's an identifier), drop it.
        last = tokens[-1]
        if last in SCALAR_MAP or last in VALUE_STRUCTS or _looks_like_type(last):
            base = tokens[-1]
        else:
            base = tokens[-2]
    return base, pointer_level


def _looks_like_type(token: str) -> bool:
    return token.endswith("_t") and token.startswith("n4m_")


def _ctype_for(base: str, level: int, *, is_return: bool) -> str:
    """Map a parsed ``(base, pointer_level)`` to a ctypes expression string."""
    # void / void* -------------------------------------------------------------
    if base == "void":
        if level == 0:
            if is_return:
                return "None"
            raise GeneratorError("bare 'void' parameter is not expected")
        # void* (any level) -> c_void_p
        return "c_void_p"

    # char* / char[] -> c_char_p as a parameter (ctypes accepts bytes). As a
    # RETURN type the binding convention is c_void_p: callers wrap the raw
    # pointer with ``ctypes.c_char_p(ptr).value`` so a NULL stays falsy and the
    # core-owned string is never auto-freed by ctypes. bare char only behind ptr.
    if base == "char":
        if level >= 1:
            return "c_void_p" if is_return else "c_char_p"
        return "c_char"

    # n4m_matrix_view_t and the other concrete value structs -------------------
    if base in VALUE_STRUCTS:
        struct = VALUE_STRUCTS[base]
        if level == 0:
            return struct  # passed by value
        if level == 1:
            return f"POINTER({struct})"
        return _wrap_pointer(f"POINTER({struct})", level - 1)

    # n4m_*_t enums: by value -> c_int; behind a pointer wrap c_int32 (the
    # 32-bit storage the ABI fixes for these enums) to match the binding's call
    # sites that pass ``byref`` over c_int32 buffers.
    if base in ENUM_TYPES:
        if level == 0:
            return "c_int"
        return _wrap_pointer("c_int32", level)

    # Any other n4m_*_t is an opaque/forward-declared handle. It only ever
    # crosses the ABI behind a pointer:
    #   level 1  -> opaque handle*    -> c_void_p
    #   level 2  -> opaque handle**   -> POINTER(c_void_p)
    if _looks_like_type(base):
        if level == 0:
            # A by-value opaque struct would be a real ABI design error; flag it.
            raise GeneratorError(f"by-value opaque type {base!r} (not an enum)")
        if level == 1:
            return "c_void_p"
        return _wrap_pointer("c_void_p", level - 1)

    # Plain scalars ------------------------------------------------------------
    if base in SCALAR_MAP:
        ct = SCALAR_MAP[base]
        if level == 0:
            return ct
        return _wrap_pointer(ct, level)

    raise GeneratorError(f"unmapped C type: base={base!r} level={level}")


def _wrap_pointer(inner: str, level: int) -> str:
    for _ in range(level):
        inner = f"POINTER({inner})"
    return inner


def _parse_enum_types(headers: list[Path]) -> set[str]:
    """Collect every ``typedef enum ... n4m_*_t`` name across the headers."""
    enums: set[str] = set()
    for header in headers:
        text = _strip_comments(header.read_text())
        # typedef enum [Tag] { ... } n4m_name_t;
        for match in re.finditer(
            r"typedef\s+enum\b[^{};]*\{[^}]*\}\s*(n4m_[a-z0-9_]+)\s*;", text, re.DOTALL
        ):
            enums.add(match.group(1))
        # typedef enum n4m_name_t name_alias;  (rare; forward-style)
        for match in re.finditer(r"typedef\s+enum\s+(n4m_[a-z0-9_]+)\b[^{]*;", text):
            enums.add(match.group(1))
    return enums


def _parse_prototypes(headers: list[Path]) -> list[tuple[str, list[str], str]]:
    """Parse every ``N4M_API`` function prototype across ``headers``."""
    decls: list[tuple[str, list[str], str]] = []
    seen: set[str] = set()
    proto_re = re.compile(r"N4M_API\s+(.*?)\)\s*;", re.DOTALL)
    for header in headers:
        text = _strip_comments(header.read_text())
        for match in proto_re.finditer(text):
            body = re.sub(r"\s+", " ", match.group(1)).strip()
            open_paren = body.find("(")
            if open_paren < 0:
                continue
            head = body[:open_paren].strip()
            name_match = re.search(r"(n4m_[A-Za-z0-9_]+)$", head)
            if name_match is None:
                # Not a function prototype (e.g. a macro / typedef artifact).
                continue
            name = name_match.group(1)
            ret_decl = head[: name_match.start()].strip()
            args = body[open_paren + 1:].strip()
            if name in seen:
                raise GeneratorError(f"duplicate prototype for {name} in {header}")
            seen.add(name)

            restype = _ctype_for(*_base_type(ret_decl or "void"), is_return=True)

            if args in ("", "void"):
                argtypes: list[str] = []
            else:
                argtypes = [
                    _ctype_for(*_base_type(p), is_return=False)
                    for p in _split_top_level(args)
                ]
            decls.append((name, argtypes, restype))
    return decls


def _load_expected_symbols() -> set[str]:
    symbols = {
        line.strip()
        for line in ABI_SNAPSHOT.read_text().splitlines()
        if line.strip() and line.strip() != "N4M_2"
    }
    return symbols


def _load_rename_targets() -> set[str]:
    if not RENAME_MAP.exists():
        return set()
    targets: set[str] = set()
    for line in RENAME_MAP.read_text().splitlines()[1:]:
        if not line.strip():
            continue
        cols = line.split("\t")
        if len(cols) >= 2:
            targets.add(cols[1])
    return targets


def _validate(names: list[str]) -> None:
    name_set = set(names)
    if len(name_set) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise GeneratorError(f"duplicate emitted symbols: {dupes}")

    expected = _load_expected_symbols()
    missing = sorted(expected - name_set)
    extra = sorted(name_set - expected)
    if missing or extra:
        msg = ["emitted symbols do not match cpp/abi/expected_symbols_linux.txt:"]
        if missing:
            msg.append(f"  missing ({len(missing)}): {missing[:20]}")
        if extra:
            msg.append(f"  extra ({len(extra)}): {extra[:20]}")
        raise GeneratorError("\n".join(msg))

    # Migration backstop: every rename target must be a real exported symbol.
    targets = _load_rename_targets()
    stale = sorted(targets - expected)
    if stale:
        raise GeneratorError(
            "rename map targets absent from the ABI surface (stale migration map): "
            f"{stale[:20]}"
        )


def _render(decls: list[tuple[str, list[str], str]]) -> str:
    lines = [FILE_HEADER]
    for name, argtypes, restype in decls:
        if len(argtypes) == 1:
            args_repr = f"({argtypes[0]},)"
        else:
            args_repr = "(" + ", ".join(argtypes) + ")"
        lines.append(f'    ("{name}", {args_repr}, {restype}),\n')
    lines.append(FILE_FOOTER)
    return "".join(lines)


def generate() -> str:
    headers = sorted(INCLUDE_DIR.rglob("*.h"))
    if not headers:
        raise GeneratorError(f"no headers found under {INCLUDE_DIR}")
    ENUM_TYPES.clear()
    ENUM_TYPES.update(_parse_enum_types(headers))
    decls = _parse_prototypes(headers)
    _validate([d[0] for d in decls])
    # Stable ordering: by name keeps the file deterministic across header edits.
    decls.sort(key=lambda d: d[0])
    return _render(decls)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="regenerate to memory and fail (exit 1) if it differs from the file on disk",
    )
    args = parser.parse_args(argv)

    rendered = generate()

    if args.check:
        current = OUTPUT.read_text() if OUTPUT.exists() else ""
        if current != rendered:
            sys.stderr.write(
                f"{OUTPUT.relative_to(REPO_ROOT)} is out of date; "
                "run: python scripts/generate_ffi_decls.py\n"
            )
            return 1
        return 0

    OUTPUT.write_text(rendered)
    sys.stdout.write(f"wrote {OUTPUT.relative_to(REPO_ROOT)} ({len(rendered.splitlines())} lines)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
