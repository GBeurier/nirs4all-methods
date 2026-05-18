#!/usr/bin/env python3
# SPDX-License-Identifier: CECILL-2.1
"""Generate the PCG64 parity fixture against ``numpy.random.default_rng``.

The fixture captures, for each seed in :data:`SEEDS`:

* ``uint64``           — 4096 raw uint64 draws from ``rng.bit_generator.random_raw``.
* ``standard_normal``  — 4096 ``standard_normal`` doubles from a *fresh* generator
  with the same seed. NumPy's ``Generator`` consumes underlying PCG64 draws
  whenever a method is invoked, so the standard-normal stream must be measured
  against a freshly seeded generator, not after the uint64 fill — otherwise the
  fixture would describe a different starting state than the C engine sees.

The fixture format is intentionally trivial JSON (no third-party tooling on the
C++ side); the load path in ``cpp/tests/test_rng_pcg64.cpp`` is a hand-rolled
parser.
"""

from __future__ import annotations

import json
from pathlib import Path
from struct import pack, unpack

import numpy as np

SEEDS: list[int] = [0, 1, 42, 12345, 0xDEADBEEF]
N_SAMPLES: int = 4096


def double_to_hex(value: float) -> str:
    """Round-trip a double through its IEEE-754 binary64 little-endian bytes.

    JSON's default float formatting loses bits on irrational values, so we
    encode each double via its 8-byte big-endian hex representation. The C++
    test parser reverses this to reconstruct the exact bit pattern.
    """
    return pack(">d", value).hex()


def hex_to_double(text: str) -> float:
    """Inverse of :func:`double_to_hex` (used for self-validation)."""
    return float(unpack(">d", bytes.fromhex(text))[0])


def build_fixture() -> dict:
    fixture: dict = {
        "format": "c4a_rng_pcg64_stream_v1",
        "numpy_version": np.__version__,
        "n_samples": N_SAMPLES,
        "standard_normal_encoding": "ieee754_binary64_be_hex",
        "seeds": {},
    }
    for seed in SEEDS:
        rng_uint = np.random.default_rng(seed=seed)
        uint64_draws = [
            int(rng_uint.bit_generator.random_raw()) for _ in range(N_SAMPLES)
        ]

        rng_norm = np.random.default_rng(seed=seed)
        normal_draws = rng_norm.standard_normal(N_SAMPLES)

        fixture["seeds"][str(seed)] = {
            "uint64": uint64_draws,
            "standard_normal": [double_to_hex(float(v)) for v in normal_draws],
        }
    return fixture


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    out_path = repo_root / "parity" / "fixtures" / "_rng_pcg64_stream_v1.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fixture = build_fixture()

    # Self-validate the bit-exact round-trip before writing.
    for seed in SEEDS:
        expected = np.random.default_rng(seed=seed).standard_normal(8)
        decoded = [
            hex_to_double(h)
            for h in fixture["seeds"][str(seed)]["standard_normal"][:8]
        ]
        for ref, dec in zip(expected.tolist(), decoded):
            assert ref == dec, (
                f"hex round-trip lost a bit for seed={seed}: "
                f"ref={ref!r} dec={dec!r}"
            )

    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(fixture, handle, indent=None)
        handle.write("\n")
    print(f"wrote {out_path} ({out_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
