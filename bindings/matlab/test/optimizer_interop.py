"""Bidirectional Python/Octave N4MOPT continuation over one ordered event stream."""

from __future__ import annotations

import sys
from pathlib import Path

from n4m.model_selection.optimizer import Optimizer, SearchSpace


def new_optimizer() -> Optimizer:
    space = SearchSpace()
    space.add_int("x", 1, 5)
    space.add_float("a", 0.1, 0.9)
    return Optimizer(space, seed=42)


def decision(trial) -> tuple[int, int, float]:
    return trial.id, trial.get_int("x"), trial.get_float("a")


def main(mode: str, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    expected_file = directory / "expected.txt"
    if mode == "write":
        with new_optimizer() as optimizer:
            for _ in range(3):
                trial = optimizer.ask()
                optimizer.tell(trial.id, trial.get_int("x") + trial.get_float("a"))
            (directory / "python.n4mopt").write_bytes(optimizer.save())
            expected = decision(optimizer.ask())
        expected_file.write_text(
            f"{expected[0]} {expected[1]} {expected[2]:.17g}\n", encoding="utf-8"
        )
    elif mode == "read":
        with Optimizer.load((directory / "octave.n4mopt").read_bytes()) as optimizer:
            actual = decision(optimizer.ask())
        expected = expected_file.read_text(encoding="utf-8").split()
        assert actual == (int(expected[0]), int(expected[1]), float(expected[2])), (
            actual,
            expected,
        )
        print("Python resumed MATLAB/Octave N4MOPT checkpoint exactly")
    else:
        raise ValueError("mode must be write or read")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: optimizer_interop.py write|read DIRECTORY")
    main(sys.argv[1], Path(sys.argv[2]))
