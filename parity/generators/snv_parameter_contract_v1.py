#!/usr/bin/env python3
"""Independent literal/NumPy SNV contract references; never imports libn4m."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MATRIX = [
    [1.0, 3.0, 8.0, 20.0],
    [10.0, 11.0, 13.0, 18.0],
    [-7.0, -3.0, 2.0, 9.0],
    [5.0, 5.0, 5.0, 5.0],
]
NATIVE_SHA = "e7f53f7b3859988a17a9eed9beeda2e6321aabe3b3c6f43ef08e63c1b383d11c"
WASM_SHA = "8b048f1cb30881b48599dbe697bafa18bc73ecf27e4eea790ce58f09cfbd9bd1"
DEFAULT_BITS = {
    "native_e7": "eb62bec8a657eebf3a6b638f52ace5bf000000000000000012e710acfc01fa3f2eb7d9dd8125efbf74cf3be9abc3e4bf000000000000000051c38ae396f4f93fb0b8f1a7f46af3bf4f4d5409b868e1bf903fbd1d8bbfd23f748f2ce56d6ff73f0000000000000000000000000000000000000000000000000000000000000000",
    "wasm_8b0": "eb62bec8a657eebf3a6b638f52ace5bf000000000000000012e710acfc01fa3f2eb7d9dd8125efbf74cf3be9abc3e4bf000000000000000051c38ae396f4f93fb0b8f1a7f46af3bf4f4d5409b868e1bf903fbd1d8bbfd23f748f2ce56d6ff73f0000000000000000000000000000000000000000000000000000000000000000",
}
ORIGIN = {
    "probes/native.stdout": "09092fdea9d2ae4c826b829fc3d50ab4ce6ed6e01aedf2e894ace2529c42b795",
    "probes/native.receipt.json": "fb771fd90f8d9517d3c3e026d9b4b97485a19c09d4caad2055a288c2c82284f0",
    "probes/v22.21.1.stdout": "3e23b97221d92345287441f82101989f24c0f43d61146fe78506b691ee71a622",
    "probes/v22.21.1.receipt.json": "6862af0dddb920eec373940a3ca26a80d066acbf11fa50224b7cfe739a89637c",
}


def generate():
    x = np.asarray(MATRIX, dtype=np.float64)
    refs = []
    for mean, std, ddof in [
        (True, True, 0),
        (True, True, 1),
        (True, True, 3),
        (False, True, 0),
        (False, True, 1),
        (True, False, 0),
        (True, False, 4),
        (True, False, 2147483647),
        (False, False, 5),
    ]:
        y = x - x.mean(axis=1, keepdims=True) if mean else x.copy()
        if std:
            sd = x.std(axis=1, ddof=ddof, keepdims=True)
            sd[sd == 0] = 1.0
            y = y / sd
        refs.append(
            {
                "id": f"mean{int(mean)}-std{int(std)}-ddof{ddof}",
                "params": {"with_mean": mean, "with_std": std, "ddof": ddof},
                "values": y.tolist(),
            }
        )
    for bits in DEFAULT_BITS.values():
        baseline = np.frombuffer(bytes.fromhex(bits), dtype="<f8").reshape(4, 4)
        np.testing.assert_allclose(baseline, refs[0]["values"], rtol=0, atol=1e-12)
    domains = []
    for mean in (False, True):
        for std in (False, True):
            for ddof in (-1, 0, 1, 3, 4, 5, 2147483647):
                domains.append(
                    {
                        "id": f"mean{int(mean)}-std{int(std)}-ddof{ddof}",
                        "params": {"with_mean": mean, "with_std": std, "ddof": ddof},
                        "create_status": 1 if ddof < 0 else 0,
                        "transform_status": 1 if std and ddof >= 4 else 0,
                    }
                )
    # Host tags express values JSON itself cannot carry; no product wire format.
    types = []

    def row(name, parameter, tag, value, python, js, r, octave):
        types.append(
            {
                "id": name,
                "parameter": parameter,
                "tag": tag,
                "value": value,
                "admit": {"python": python, "javascript": js, "r": r, "octave": octave},
            }
        )

    row("string-false", "with_std", "literal", "false", False, False, False, False)
    row("string-true", "with_mean", "literal", "true", False, False, False, False)
    for flag in (0, 1, 2):
        row(
            f"flag-{flag}",
            "with_std",
            "literal",
            flag,
            False,
            False,
            flag < 2,
            flag < 2,
        )
    for key in ("with_mean", "with_std", "ddof"):
        row(f"null-{key}", key, "literal", None, False, False, False, False)
    for name, tag, value, py, js, r, octa in [
        ("fraction", "literal", 1.9, False, False, False, False),
        ("whole-float", "literal", 1.0, False, True, True, True),
        ("bool-ddof", "literal", True, False, False, False, False),
        ("string-ddof", "literal", "1", False, False, False, False),
        ("negative", "literal", -1, False, False, False, False),
        ("overflow-i32", "literal", 2147483648, False, False, False, False),
        ("wrap-i32", "literal", 4294967296, False, False, False, False),
        ("nan", "nan", None, False, False, False, False),
        ("infinity", "infinity", None, False, False, False, False),
        ("timedelta-unitless", "np.timedelta64", None, False, None, None, None),
        ("timedelta-seconds", "np.timedelta64[s]", None, False, None, None, None),
        ("integer-enum", "IntEnum", 1, True, None, None, None),
        ("numpy-integer", "np.int64", 1, True, None, None, None),
        ("numpy-unsigned", "np.uint32", 1, True, None, None, None),
        ("bigint", "bigint", 1, None, False, None, None),
        ("boxed-number", "boxed-number", 1, None, False, None, None),
        ("factor", "factor", "z", None, None, False, None),
        ("difftime", "difftime", 1, None, None, False, None),
    ]:
        row(name, "ddof", tag, value, py, js, r, octa)
    row("numpy-boolean", "with_std", "np.bool_", False, True, None, None, None)
    row("boxed-boolean", "with_std", "boxed-boolean", False, None, False, None, None)
    row("missing-r-flag", "with_std", "NA", None, None, None, False, None)
    return {
        "schema": "n4m.snv-parameter-contract.v1",
        "matrix": MATRIX,
        "references": refs,
        "domains": domains,
        "typed_values": types,
        "default_bits": DEFAULT_BITS,
        "provenance": {
            "numpy_version": np.__version__,
            "native_sha256": NATIVE_SHA,
            "wasm_sha256": WASM_SHA,
            "original_receipts": ORIGIN,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(generate(), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
