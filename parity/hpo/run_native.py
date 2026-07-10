# SPDX-License-Identifier: CECILL-2.1
"""Run an HpoSpec through the native libn4m optimizer (Python binding) and emit a
StudyTrace — the binding-neutral record every other binding must reproduce.

Requires a built libn4m; point ``N4M_LIB_PATH`` at it (e.g. .../libn4m.so.2.1.0).
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "bindings" / "python" / "src"))

from n4m.model_selection.optimizer import (  # noqa: E402
    Direction,
    Optimizer,
    Pruner,
    Sampler,
    SearchSpace,
    TrialStatus,
)

from objectives import INTERMEDIATE, OBJECTIVES  # noqa: E402
from specs import HpoSpec, ParamDecl  # noqa: E402

_SAMPLER = {s.name.lower(): s for s in Sampler}
_PRUNER = {p.name.lower(): p for p in Pruner}
_DIRECTION = {"minimize": Direction.MINIMIZE, "maximize": Direction.MAXIMIZE, "auto": Direction.AUTO}


def _build_space(space: tuple[ParamDecl, ...]) -> SearchSpace:
    s = SearchSpace()
    for p in space:
        if p.kind == "int":
            s.add_int(p.name, int(p.args[0]), int(p.args[1]),
                      int(p.args[2]) if len(p.args) > 2 else 1)
        elif p.kind == "log_int":
            s.add_int(p.name, int(p.args[0]), int(p.args[1]), 1, log=True)
        elif p.kind == "float":
            s.add_float(p.name, float(p.args[0]), float(p.args[1]),
                        float(p.args[2]) if len(p.args) > 2 else 0.0)
        elif p.kind == "log_float":
            s.add_float(p.name, float(p.args[0]), float(p.args[1]), 0.0, log=True)
        elif p.kind == "categorical":
            s.add_categorical(p.name, list(p.args))
        elif p.kind == "ordinal":
            s.add_ordinal(p.name, list(p.args))
        else:
            raise ValueError(f"unsupported ParamDecl kind in parity spec: {p.kind}")
    return s


def _resolve(trial, space: tuple[ParamDecl, ...]) -> dict:
    out: dict = {}
    for p in space:
        if p.kind in ("int", "log_int"):
            out[p.name] = trial.get_int(p.name)
        elif p.kind in ("float", "log_float", "ordinal"):
            out[p.name] = trial.get_float(p.name)
        elif p.kind == "categorical":
            _idx, label = trial.get_category(p.name)
            out[p.name] = label
    return out


def run_spec(spec: HpoSpec) -> list[dict]:
    """Execute a spec and return its StudyTrace (one record per asked trial)."""
    space = _build_space(spec.space)
    opt = Optimizer(
        space,
        sampler=_SAMPLER[spec.sampler],
        pruner=_PRUNER[spec.pruner],
        direction=_DIRECTION[spec.direction],
        n_startup_trials=spec.n_startup_trials,
        seed=spec.seed,
        max_resource=spec.max_resource,
        reduction_factor=spec.reduction_factor,
    )
    obj = OBJECTIVES[spec.objective]
    inter = INTERMEDIATE[spec.intermediate] if spec.intermediate else None

    trace: list[dict] = []
    for _ in range(spec.n_trials):
        t = opt.ask()
        params = _resolve(t, spec.space)
        rec: dict = {"id": t.id, "params": params}
        if inter is not None:
            pruned_at = -1
            for step in range(spec.intermediate_steps):
                if opt.tell_intermediate(t.id, step, inter(params, step)):
                    pruned_at = step
                    break
            rec["pruned_at"] = pruned_at
            if pruned_at < 0:
                opt.tell(t.id, obj(params))
        else:
            score = obj(params)
            rec["score"] = score
            opt.tell(t.id, score)
        rec["status"] = int(t.status) if t.status is not None else int(TrialStatus.RUNNING)
        trace.append(rec)
    return trace
