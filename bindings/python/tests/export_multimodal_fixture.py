# SPDX-License-Identifier: CECILL-2.1
"""Export the binding-only raw diagnostic fixture after native qualification.

This is test tooling, not a dataset product or a replacement for canonical U07.
The reference predictions come from the independent sklearn encoder/Ridge
oracle in test_multimodal_pipeline.py; the N4MF bytes come from native fit.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np

from n4m import MultimodalPipeline
from test_multimodal_pipeline import oracle, raw_case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    args = parser.parse_args()
    blocks, y, recipe, schemas = raw_case(np.dtype(args.dtype))
    heldout = {name: value[:3].copy() for name, value in blocks.items()}
    heldout["metadata"][:, 1] = ["🚀", "A", "é"]
    encode, ridge, _ = oracle(blocks, y, recipe)

    def transport(values):
        return {
            name: value.tolist()
            if name == "metadata"
            else {"shape": list(value.shape), "data": value.ravel(order="C").tolist()}
            for name, value in values.items()
        }

    with MultimodalPipeline(recipe, schemas) as native:
        native.fit(blocks, y)
        result = {
            "recipe": recipe,
            "source_schemas": schemas,
            "train": transport(blocks),
            "heldout": transport(heldout),
            "y": y.tolist(),
            "expected": ridge.predict(encode(heldout)).tolist(),
            "state": base64.b64encode(native.export_state()).decode("ascii"),
        }
    args.output.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
