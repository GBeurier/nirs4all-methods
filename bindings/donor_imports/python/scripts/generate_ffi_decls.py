#!/usr/bin/env python3
# SPDX-License-Identifier: CECILL-2.1
"""Auto-generate ctypes declarations for libn4m.

Parses ``cpp/include/n4m/n4m.h`` and emits a python module
``_ffi_decls.py`` containing two helper functions:

    apply_argtypes(lib): bind argtypes/restype for every exported function.
    SYMBOLS: a tuple of (symbol_name, argtypes, restype) for all ABI exports.

The parser is intentionally simple: every ``N4M_API <return> <name>(<args>);``
declaration that fits on one or more lines (delimited by ``;``) is recognised.
Unknown handle struct pointers are mapped to ``ctypes.c_void_p`` since the
binding never dereferences them.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_HEADER = HERE / ".." / ".." / ".." / "cpp" / "include" / "n4m" / "n4m.h"
DEFAULT_OUT = HERE / ".." / "src" / "n4m" / "_ffi_decls.py"

# Base type-name -> ctypes expression (single tokens, no pointer / const).
PRIMITIVE_MAP = {
    "void": "None",
    "int": "c_int",
    "int32_t": "c_int32",
    "int64_t": "c_int64",
    "uint8_t": "c_uint8",
    "uint32_t": "c_uint32",
    "uint64_t": "c_uint64",
    "size_t": "c_size_t",
    "double": "c_double",
    "float": "c_float",
    "char": "c_char",
    "n4m_status_t": "c_int",  # enum -> int
    "n4m_backend_t": "c_int",
    "n4m_dtype_t": "c_int",
    "n4m_pp_savgol_mode_t": "c_int",
    "n4m_pp_gaussian_mode_t": "c_int",
    "n4m_pp_lsnv_pad_mode_t": "c_int",
    "n4m_pp_area_method_t": "c_int",
    "n4m_signal_type_t": "c_int",
    "n4m_split_kbins_strategy_t": "c_int",
    "n4m_split_y_metric_t": "c_int",
    "n4m_split_aggregation_t": "c_int",
    "n4m_y_outlier_method_t": "c_int",
    "n4m_filter_x_outlier_method_t": "c_int",
    "n4m_composite_mode_t": "c_int",
    "n4m_pp_wavelet_family_t": "c_int",
    "n4m_pp_wavelet_boundary_t": "c_int",
    "n4m_pp_wavelet_threshold_t": "c_int",
    "n4m_pp_wavelet_noise_t": "c_int",
    "n4m_matrix_view_t": "MatrixView",
    "n4m_filter_stats_t": "FilterStats",
    "n4m_split_result_t": "SplitResult",
    "n4m_transfer_metrics_t": "TransferMetrics",
}

DECL_RE = re.compile(
    r"^N4M_API\s+(?P<rest>.*?)\s*\((?P<args>.*?)\)\s*;",
    re.DOTALL | re.MULTILINE,
)


def _strip_comments(src: str) -> str:
    """Strip /* ... */ and // ... comments."""
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.DOTALL)
    src = re.sub(r"//[^\n]*", "", src)
    return src


def _split_args(args: str) -> list[str]:
    """Split a parameter list on top-level commas only (no template brackets here)."""
    parts: list[str] = []
    depth = 0
    cur = []
    for ch in args:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur))
    return [p.strip() for p in parts if p.strip()]


def _normalise_param(p: str) -> str:
    """Drop ``const``, parameter name and array-of-N markers; keep only the type."""
    p = re.sub(r"\bconst\b", "", p).strip()
    # array dimensions: T name[256] -> T*
    p = re.sub(r"\[[^\]]*\]", "*", p)
    # collapse whitespace
    p = re.sub(r"\s+", " ", p).strip()
    # split off the parameter name (last identifier without *)
    tokens = p.split(" ")
    if not tokens:
        return p
    # The last token may be `name` or `*name` or `name`. Strip if it doesn't
    # look like a pointer-only suffix.
    last = tokens[-1]
    # If last token is purely identifier -> parameter name; drop it.
    if re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", last):
        tokens = tokens[:-1]
    else:
        # last token may be "*name" -> split into "*" and "name"
        m = re.match(r"^(\*+)([A-Za-z_][A-Za-z_0-9]*)$", last)
        if m:
            tokens[-1] = m.group(1)
        # otherwise leave (rare; e.g. function pointer types — none in our header)
    return " ".join(tokens).strip()


def _ctype_of(typestr: str) -> str:
    """Map a normalised C type string to a ctypes expression."""
    t = typestr.strip()
    if not t or t == "void":
        return "None"
    # count pointer asterisks
    n_stars = t.count("*")
    base = t.replace("*", "").strip()
    if base == "void" and n_stars >= 1:
        # void* / void** -> POINTER(c_void_p) / c_void_p (best practice for opaque double pointer)
        if n_stars == 1:
            return "c_void_p"
        else:
            return "POINTER(c_void_p)"
    if base == "char" and n_stars >= 1:
        if n_stars == 1:
            return "c_char_p"
        return "POINTER(c_char_p)"
    # struct-by-value (no asterisks)
    if n_stars == 0:
        if base in PRIMITIVE_MAP:
            return PRIMITIVE_MAP[base]
        # unknown -> treat as enum (most enums) - default to c_int
        return "c_int"
    # pointer types
    if base in PRIMITIVE_MAP:
        inner = PRIMITIVE_MAP[base]
        if inner == "None":
            inner = "None"
        # build POINTER chain
        for _ in range(n_stars):
            inner = f"POINTER({inner})"
        return inner
    # Opaque handle structs (n4m_*_handle_t) - we never deref. Use c_void_p
    # for single-pointer, POINTER(c_void_p) for **
    if n_stars == 1:
        return "c_void_p"
    return "POINTER(c_void_p)"


def parse_header(header_text: str) -> list[dict]:
    src = _strip_comments(header_text)
    src = src.replace("\r\n", "\n")
    decls = []
    for m in DECL_RE.finditer(src):
        rest = m.group("rest").strip()
        args_str = m.group("args").strip()
        # rest = "<return type> <name>"
        tokens = rest.split()
        # The last token is the function name; everything before is the return type.
        if len(tokens) < 2:
            continue
        name = tokens[-1].strip("*")
        ret_type = " ".join(tokens[:-1])
        # If name has leading *, return type absorbs the *
        leading_stars = tokens[-1].count("*")
        if leading_stars > 0:
            ret_type = ret_type + "*" * leading_stars

        if args_str == "void":
            args = []
        else:
            args = [_normalise_param(p) for p in _split_args(args_str)]

        decls.append({
            "name": name,
            "return": ret_type.strip(),
            "args": args,
        })
    return decls


def render_module(decls: list[dict]) -> str:
    lines: list[str] = []
    lines.append("# SPDX-License-Identifier: CECILL-2.1")
    lines.append('"""Auto-generated ctypes argtypes/restype declarations for libn4m.')
    lines.append("")
    lines.append("Do not edit by hand. Regenerate with::")
    lines.append("")
    lines.append("    python scripts/generate_ffi_decls.py")
    lines.append('"""')
    lines.append("from __future__ import annotations")
    lines.append("")
    lines.append("from ctypes import (")
    lines.append("    POINTER, c_char, c_char_p, c_double, c_float, c_int, c_int32,")
    lines.append("    c_int64, c_size_t, c_uint8, c_uint32, c_uint64, c_void_p,")
    lines.append(")")
    lines.append("")
    lines.append("from ._types import MatrixView, FilterStats, SplitResult, TransferMetrics")
    lines.append("")
    lines.append("# Symbols: tuple of (symbol_name, argtypes_tuple, restype).")
    lines.append("# restype = None for void; argtypes = () for parameter-less.")
    lines.append("SYMBOLS = (")
    for d in decls:
        rest = _ctype_of(d["return"])
        args = tuple(_ctype_of(a) for a in d["args"])
        if not args:
            argline = "()"
        elif len(args) == 1:
            argline = f"({args[0]},)"
        else:
            argline = "(" + ", ".join(args) + ")"
        lines.append(f'    ("{d["name"]}", {argline}, {rest}),')
    lines.append(")")
    lines.append("")
    lines.append("")
    lines.append("def apply_argtypes(lib):")
    lines.append('    """Bind argtypes/restype for every exported function on ``lib``."""')
    lines.append("    for name, argtypes, restype in SYMBOLS:")
    lines.append("        fn = getattr(lib, name, None)")
    lines.append("        if fn is None:")
    lines.append("            continue")
    lines.append("        fn.argtypes = list(argtypes)")
    lines.append("        fn.restype = restype")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--header", type=Path, default=DEFAULT_HEADER)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    header_text = args.header.read_text()
    decls = parse_header(header_text)
    print(f"Parsed {len(decls)} declarations from {args.header}", file=sys.stderr)
    rendered = render_module(decls)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered)
    print(f"Wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
