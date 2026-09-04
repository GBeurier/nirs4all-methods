"""Header-derived ctypes types must preserve pointer indirection."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

GENERATOR = Path(__file__).resolve().parents[3] / "scripts" / "generate_ffi_decls.py"


@pytest.fixture(scope="module")
def generator():
    spec = importlib.util.spec_from_file_location("generate_ffi_decls", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("declaration", "expected"),
    [
        ("const char* name", "c_char_p"),
        ("char name[]", "c_char_p"),
        ("const char** names", "POINTER(c_char_p)"),
        ("const char* const names[]", "POINTER(c_char_p)"),
        ("char*** output", "POINTER(POINTER(c_char_p))"),
    ],
)
def test_string_parameter_indirection(generator, declaration, expected):
    base, level = generator._base_type(declaration)
    assert generator._ctype_for(base, level, is_return=False) == expected


def test_string_returns_keep_raw_pointer_contract(generator):
    assert generator._ctype_for("char", 1, is_return=True) == "c_void_p"
    assert generator._ctype_for("char", 2, is_return=True) == "POINTER(c_char_p)"


def test_generated_declarations_match_headers(generator):
    import ast

    # Formatting is not part of the ABI contract.
    assert ast.dump(ast.parse(generator.generate())) == ast.dump(
        ast.parse(generator.OUTPUT.read_text())
    )
