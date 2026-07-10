# SPDX-License-Identifier: CECILL-2.1
"""HpoSpec registry — the declarative catalog of HPO cross-binding parity cells.

Each ``HpoSpec`` fully determines a study: sampler, pruner, search space, seed,
trial budget, and the deterministic objective. Running a spec through any binding
must reproduce the committed golden trace (``golden/<id>.json``) — that is the
cross-binding guarantee, analogous to the numeric fixtures for the linalg methods.

This is the HPO analogue of ``benchmarks/parity_timing/registry.py`` but tiny and
purpose-built: the space is typed via the same primitives every binding exposes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class ParamDecl:
    """One search-space axis, in binding-neutral terms."""

    kind: str          # int | float | log_int | log_float | categorical | ordinal | sorted_tuple
    name: str
    args: tuple = ()   # (low, high[, step]) for numeric; choices for categorical/ordinal
    length: int = 0    # sorted_tuple length


@dataclass(frozen=True)
class HpoSpec:
    id: str
    sampler: str                 # random | sobol | lhs | ternary | ga | pso | cmaes | tpe | gp_ei
    space: tuple[ParamDecl, ...]
    objective: str               # key into objectives.OBJECTIVES
    n_trials: int
    seed: int = 0
    pruner: str = "none"         # none | median | asha | hyperband | racing
    direction: str = "minimize"  # minimize | maximize
    n_startup_trials: int = 10
    intermediate: str = ""       # key into objectives.INTERMEDIATE (pruner cells only)
    intermediate_steps: int = 0  # rungs reported per trial when intermediate is set
    max_resource: int = 0
    reduction_factor: int = 0
    tol: float = 0.0             # 0 = exact; >0 = abs tolerance on params/scores
    notes: str = ""


def _floats(*names, low=-5.0, high=5.0):
    return tuple(ParamDecl("float", n, (low, high)) for n in names)


# The registry. Kept small and representative: every sampler at least once, plus
# mixed/conditional and pruner cells. Ids are stable (they name the golden files).
REGISTRY: list[HpoSpec] = [
    # --- samplers over a plain continuous space (cross-binding determinism) ---
    HpoSpec("random_sphere2", "random", _floats("x", "y"), "sphere", 40, seed=123),
    HpoSpec("lhs_sphere3", "lhs", _floats("x", "y", "z"), "sphere", 30, seed=7,
            n_startup_trials=30),
    HpoSpec("ternary_int", "ternary",
            (ParamDecl("int", "k", (1, 30)),), "sphere", 25, seed=1),
    HpoSpec("ga_sphere2", "ga", _floats("x", "y"), "sphere", 48, seed=5),
    HpoSpec("pso_sphere2", "pso", _floats("x", "y"), "sphere", 48, seed=5),
    HpoSpec("cmaes_sphere2", "cmaes", _floats("x", "y"), "sphere", 60, seed=5),
    HpoSpec("gp_ei_sphere2", "gp_ei", _floats("x", "y"), "sphere", 40, seed=7,
            n_startup_trials=8),
    # --- Sobol: has a Tier-A external reference (scipy), checked separately ---
    HpoSpec("sobol_unit3", "sobol",
            (ParamDecl("float", "a", (0.0, 1.0)), ParamDecl("float", "b", (0.0, 1.0)),
             ParamDecl("float", "c", (0.0, 1.0))), "sphere", 32, seed=0,
            notes="Tier-A: params must equal scipy.stats.qmc.Sobol(scramble=False)"),
    # --- mixed continuous + categorical (TPE's home turf) ---
    HpoSpec("tpe_mixed", "tpe",
            (ParamDecl("float", "x", (0.0, 10.0)),
             ParamDecl("categorical", "c", ("a", "b", "c"))),
            "sphere", 60, seed=2, n_startup_trials=15),
    # --- pruner decision cells (fixed sampler, learning-curve intermediates) ---
    HpoSpec("median_prune", "random", (ParamDecl("int", "k", (1, 10)),), "sphere", 20,
            seed=1, pruner="median", n_startup_trials=3,
            intermediate="learning_curve", intermediate_steps=5),
    HpoSpec("asha_prune", "random", (ParamDecl("int", "k", (1, 10)),), "sphere", 20,
            seed=1, pruner="asha", reduction_factor=3,
            intermediate="learning_curve", intermediate_steps=9),
    HpoSpec("hyperband_prune", "random", (ParamDecl("int", "k", (1, 10)),), "sphere", 21,
            seed=1, pruner="hyperband", max_resource=9, reduction_factor=3,
            intermediate="learning_curve", intermediate_steps=9),
]


def by_id(spec_id: str) -> HpoSpec:
    for s in REGISTRY:
        if s.id == spec_id:
            return s
    raise KeyError(f"unknown HpoSpec id: {spec_id}")
