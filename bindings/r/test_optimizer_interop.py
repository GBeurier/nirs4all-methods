"""Two-way R/Python N4MOPT continuation gate (run after R package install)."""

from __future__ import annotations

import sys
from pathlib import Path

from n4m.model_selection.optimizer import Optimizer, SearchSpace


def new_optimizer() -> Optimizer:
    space = SearchSpace()
    space.add_int("x", 1, 5)
    space.add_float("a", 0.1, 0.9)
    return Optimizer(space, seed=42)


def result(trial) -> tuple[int, int, float]:
    return trial.id, trial.get_int("x"), trial.get_float("a")


def run(mode: str, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    expected_file = directory / "expected.txt"
    if mode == "write":
        optimizer = new_optimizer()
        for _ in range(3):
            trial = optimizer.ask()
            optimizer.tell(trial.id, trial.get_int("x") + trial.get_float("a"))
        (directory / "python.n4mopt").write_bytes(optimizer.save())
        expected = result(optimizer.ask())
        expected_file.write_text(f"{expected[0]} {expected[1]} {expected[2]:.17g}\n")
    elif mode == "read":
        optimizer = Optimizer.load((directory / "r.n4mopt").read_bytes())
        expected = expected_file.read_text().split()
        actual = result(optimizer.ask())
        assert actual == (int(expected[0]), int(expected[1]), float(expected[2])), (
            actual,
            expected,
        )
        print("Python resumed R N4MOPT checkpoint exactly")
    else:
        raise ValueError("mode must be write or read")


if __name__ == "__main__":
    run(sys.argv[1], Path(sys.argv[2]))
