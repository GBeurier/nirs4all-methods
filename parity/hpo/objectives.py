# SPDX-License-Identifier: CECILL-2.1
"""Deterministic closed-form objectives for HPO cross-binding parity.

The HPO parity contract is: for a fixed (sampler, pruner, search space, seed),
every binding must produce the SAME sequence of proposed configurations when the
tell() score is a deterministic function of the proposed parameters. These
objectives are that function. They are intentionally simple closed forms so that
the R / MATLAB-Octave / JS-WASM bindings can implement byte-identical versions
and reproduce the committed golden traces (see README.md).

Each objective maps a resolved parameter dict -> float (lower is better unless the
spec's direction says otherwise). Pruner specs additionally expose an
``intermediate(params, step)`` learning-curve value reported at each rung.

Keep these EXACT — changing a formula invalidates every golden trace.
"""

from __future__ import annotations

import math
from typing import Callable


def _num(params: dict, name: str, default: float = 0.0) -> float:
    v = params.get(name, default)
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    return float(v)


def sphere(params: dict) -> float:
    """Sum of squared numeric params about fixed targets; categorical adds an
    index penalty. Target for a param named x* is 2.0; for others it is 0.
    """
    total = 0.0
    for name, value in params.items():
        if isinstance(value, str):
            # categorical: penalty = position of first char in "abcdef..." (0-based)
            total += float(max(0, ord(value[0]) - ord("a"))) if value else 0.0
        elif isinstance(value, bool):
            total += 0.0 if value else 1.0
        else:
            target = 2.0 if name.startswith("x") else 0.0
            total += (float(value) - target) ** 2
    return total


def weighted_ramp(params: dict) -> float:
    """A monotone, mildly multi-scale objective: sum_i (i+1) * |v_i - i*0.3|."""
    total = 0.0
    for i, (_, value) in enumerate(sorted(params.items())):
        if isinstance(value, str):
            v = float(max(0, ord(value[0]) - ord("a"))) if value else 0.0
        elif isinstance(value, bool):
            v = 1.0 if value else 0.0
        else:
            v = float(value)
        total += (i + 1) * abs(v - i * 0.3)
    return total


def learning_curve(params: dict, step: int) -> float:
    """Deterministic decreasing learning curve for pruner specs. The asymptote is
    the sphere value; the curve approaches it as 1/(step+1). Lower is better.
    """
    asymptote = sphere(params)
    return asymptote + 5.0 / (step + 1.0)


def compatibility_curve(params: dict, step: int) -> float:
    """Adversarial deterministic curve for the 9 x 5 compatibility matrix.

    The terminal objective minimizes ``sphere``.  Negating that asymptote makes
    later, terminally-better proposals deliberately non-trivial for every
    pruner instead of letting an adaptive sampler monotonically outrun its
    pruning history.  This is native conformance evidence, not an external
    numerical reference.
    """
    return -sphere(params) + 5.0 / (step + 1.0)


def racing_observation(params: dict, step: int) -> float:
    """Deterministic repeated observations for the Hoeffding racing cell.

    The bounded, zero-mean eight-step offset makes each trial expose repeated
    fold-like observations while preserving the ordering of the ``sphere``
    objective.  The sequence is repeated for longer racing tapes.
    """
    offsets = (-0.21, 0.08, 0.17, -0.04, 0.13, -0.19, 0.02, 0.04)
    return sphere(params) + offsets[step % len(offsets)]


# Registry of objective functions by name (portable identifiers used in specs).
OBJECTIVES: dict[str, Callable[[dict], float]] = {
    "sphere": sphere,
    "weighted_ramp": weighted_ramp,
}

INTERMEDIATE: dict[str, Callable[[dict, int], float]] = {
    "learning_curve": learning_curve,
    "compatibility_curve": compatibility_curve,
    "racing_observation": racing_observation,
}


def objective_formula_doc() -> str:
    """Human-readable spec of the exact formulas (for the binding authors)."""
    return (
        "sphere(params)        = Σ over params: numeric→(v-target)^2 "
        "(target=2 if name startswith 'x' else 0); str→(ord(c0)-'a', min 0); "
        "bool→(0 if true else 1)\n"
        "weighted_ramp(params) = Σ_i (i+1)*|v_i - i*0.3|, i over params sorted by name\n"
        "learning_curve(p,s)   = sphere(p) + 5/(s+1)\n"
        "compatibility_curve(p,s) = -sphere(p) + 5/(s+1)\n"
        "racing_observation(p,s) = sphere(p) + "
        "[-.21,.08,.17,-.04,.13,-.19,.02,.04][s mod 8]\n"
        f"pi={math.pi!r} (unused; kept for reference)"
    )
