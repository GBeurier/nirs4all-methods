# SPDX-License-Identifier: CECILL-2.1
"""Deterministic N4MOPT checkpoint/resume parity tapes.

The tape is deliberately independent of the serialized bytes (which contain
elapsed monotonic durations). It freezes the observable continuation after a
checkpoint and proves that direct and restored optimizers consume exactly the
same sampler trajectory.
"""

from __future__ import annotations

from typing import Any

from n4m.model_selection.optimizer import (
    Direction,
    Optimizer,
    Sampler,
    SearchSpace,
)


def _score(x: float, k: int) -> float:
    return (x - 0.25) ** 2 + (k - 2) ** 2


def run_resume_tape(sampler: Sampler) -> dict[str, Any]:
    """Run one canonical pre-checkpoint prefix and four-event continuation."""

    with SearchSpace().add_float("x", -2.0, 2.0).add_int("k", 1, 7) as space:
        original = Optimizer(
            space,
            sampler=sampler,
            direction=Direction.MINIMIZE,
            n_startup_trials=2,
            seed=0x123456789,
        )
        try:
            prefix: list[dict[str, int | float]] = []
            for _ in range(8):
                trial = original.ask()
                x = trial.get_float("x")
                k = trial.get_int("k")
                score = _score(x, k)
                prefix.append({"id": trial.id, "x": x, "k": k, "score": score})
                original.tell(trial.id, score)

            restored = Optimizer.load(original.save())
            try:
                continuation: list[dict[str, int | float]] = []
                for _ in range(4):
                    direct = original.ask()
                    resumed = restored.ask()
                    direct_values = (
                        direct.id,
                        direct.get_float("x"),
                        direct.get_int("k"),
                    )
                    resumed_values = (
                        resumed.id,
                        resumed.get_float("x"),
                        resumed.get_int("k"),
                    )
                    if direct_values != resumed_values:
                        raise AssertionError(
                            f"checkpoint continuation diverged: {direct_values!r} != "
                            f"{resumed_values!r}"
                        )
                    trial_id, x, k = direct_values
                    score = _score(x, k)
                    continuation.append(
                        {"id": trial_id, "x": x, "k": k, "score": score}
                    )
                    original.tell(trial_id, score)
                    restored.tell(trial_id, score)
            finally:
                restored.close()
        finally:
            original.close()
    return {
        "sampler": sampler.name.lower(),
        "seed": 0x123456789,
        "checkpoint_after": 8,
        "prefix": prefix,
        "continuation": continuation,
    }


def generate_resume_pack() -> dict[str, Any]:
    """Return the canonical random + adaptive-TPE resume tape pack."""

    return {
        "format": "n4m.hpo.checkpoint_resume.v1",
        "tapes": [
            run_resume_tape(Sampler.RANDOM),
            run_resume_tape(Sampler.TPE),
        ],
    }


__all__ = ["generate_resume_pack", "run_resume_tape"]
