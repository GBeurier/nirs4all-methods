#!/usr/bin/env python3
# SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
"""Check frozen PCA goldens with mpmath 1.3.0, without loading Methods."""

import hashlib
import json
import struct
from pathlib import Path

import mpmath as mp


def main():
    """Recompute the independent oracle from the exact frozen binary64 input."""
    assert mp.__version__ == "1.3.0", "Reference generation requires mpmath 1.3.0"
    mp.mp.dps = 100
    path = Path(__file__).resolve().parents[1] / "fixtures/preflight_pca_weak_scores_v1.json"
    fixture = json.loads(path.read_text())

    def decode(key):
        return [mp.mpf(struct.unpack(">d", bytes.fromhex(value))[0]) for value in fixture[key]]

    n, p = fixture["rows"], fixture["cols"]
    values = decode("input_hex")
    raw = b"".join(struct.pack("<d", float(value)) for value in values)
    assert hashlib.sha256(raw).hexdigest() == fixture["input_f64_le_sha256"]
    means = [mp.fsum(values[i * p + j] for i in range(n)) / n for j in range(p)]
    centered = mp.matrix([[values[i * p + j] - means[j] for j in range(p)] for i in range(n)])
    u, s, v = mp.svd(centered)
    a = fixture["reference"]["component"] - 1
    sign = 1 if v[a, max(range(p), key=lambda j: abs(v[a, j]))] > 0 else -1
    scores = [sign * u[i, a] * s[a] for i in range(n)]
    factor = decode("factor_hex")
    mean = mp.fsum(scores) / n
    factor_mean = mp.fsum(factor) / n
    ss = mp.fsum((value - mean) ** 2 for value in scores)
    pearson = mp.fsum((scores[i] - mean) * (factor[i] - factor_mean) for i in range(n)) / mp.sqrt(
        ss * mp.fsum((value - factor_mean) ** 2 for value in factor)
    )
    groups = fixture["group_codes"]
    between = mp.mpf(0)
    for group in sorted(set(groups)):
        members = [scores[i] for i in range(n) if groups[i] == group]
        between += len(members) * (mp.fsum(members) / len(members) - mean) ** 2
    references = {
        "reference_pc5_scores_hex": scores,
        "reference_pc5_loadings_hex": [sign * v[a, j] for j in range(p)],
        "reference_eigenvalues_hex": [s[j] ** 2 / (n - 1) for j in range(fixture["components"])],
        "reference_pearson_hex": [pearson],
        "reference_eta_squared_hex": [between / ss],
    }
    for key, reference in references.items():
        assert [struct.pack(">d", float(value)).hex() for value in reference] == fixture[key], key
    print("PCA weak-score oracle: all frozen binary64 goldens match the 100-digit direct SVD")


if __name__ == "__main__":
    main()
