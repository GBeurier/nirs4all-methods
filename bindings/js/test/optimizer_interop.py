"""Bidirectional Python/JS N4MOPT continuation over one ordered event stream."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

from n4m.model_selection.optimizer import Optimizer, SearchSpace


def new_optimizer() -> Optimizer:
    space = SearchSpace()
    space.add_int("x", 1, 5)
    space.add_float("a", 0.1, 0.9)
    return Optimizer(space, seed=42)


def decision(trial) -> dict[str, int | float]:
    return {"id": trial.id, "x": trial.get_int("x"), "a": trial.get_float("a")}


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="n4m-js-opt-") as directory:
        root = Path(directory)
        with new_optimizer() as optimizer:
            for _ in range(3):
                trial = optimizer.ask()
                optimizer.tell(trial.id, trial.get_int("x") + trial.get_float("a"))
            (root / "python.n4mopt").write_bytes(optimizer.save())
            expected = decision(optimizer.ask())
        (root / "expected.json").write_text(json.dumps(expected), encoding="utf-8")

        node = os.environ.get("N4M_NODE", "node")
        subprocess.run(
            [node, str(Path(__file__).with_name("run_optimization_interop.mjs")), str(root)],
            check=True,
        )

        with Optimizer.load((root / "js.n4mopt").read_bytes()) as optimizer:
            assert decision(optimizer.ask()) == expected
    print("Python resumed JS/WASM N4MOPT checkpoint exactly")


if __name__ == "__main__":
    main()
