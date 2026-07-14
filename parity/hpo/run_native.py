# SPDX-License-Identifier: CECILL-2.1
"""Run an HpoSpec through the native libn4m optimizer (Python binding) and emit a
StudyTrace — the binding-neutral record every other binding must reproduce.

Requires a built libn4m; point ``N4M_LIB_PATH`` at it (e.g. .../libn4m.so.2.2.0).
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
    TrialError,
    TrialStatus,
)

from objectives import INTERMEDIATE, OBJECTIVES  # noqa: E402
from specs import HpoSpec, ParamDecl  # noqa: E402

_SAMPLER = {s.name.lower(): s for s in Sampler}
_PRUNER = {p.name.lower(): p for p in Pruner}
_DIRECTION = {
    "minimize": Direction.MINIMIZE,
    "maximize": Direction.MAXIMIZE,
    "auto": Direction.AUTO,
}


def _build_space(space: tuple[ParamDecl, ...]) -> SearchSpace:
    s = SearchSpace()
    for p in space:
        if p.kind == "int":
            s.add_int(
                p.name,
                int(p.args[0]),
                int(p.args[1]),
                int(p.args[2]) if len(p.args) > 2 else 1,
            )
        elif p.kind == "log_int":
            s.add_int(p.name, int(p.args[0]), int(p.args[1]), 1, log=True)
        elif p.kind == "float":
            s.add_float(
                p.name,
                float(p.args[0]),
                float(p.args[1]),
                float(p.args[2]) if len(p.args) > 2 else 0.0,
            )
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


def run_spec(spec: HpoSpec, *, checkpoint_after: int | None = None) -> list[dict]:
    """Execute a spec and return its StudyTrace (one record per asked trial).

    When ``checkpoint_after`` is provided, the owning optimizer is saved and
    restored after that many terminal trials.  The split must align with the
    logical batch size.  This keeps the event tape unchanged while exercising
    portable continuation through the public Python binding.
    """
    space = _build_space(spec.space)
    opt: Optimizer | None = None
    try:
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

        if spec.max_in_flight < 1:
            raise ValueError("max_in_flight must be >= 1")
        if spec.tell_order and sorted(spec.tell_order) != list(
            range(spec.max_in_flight)
        ):
            raise ValueError(
                "tell_order must be a permutation of batch-local positions"
            )
        if checkpoint_after is not None and (
            checkpoint_after <= 0
            or checkpoint_after >= spec.n_trials
            or checkpoint_after % spec.max_in_flight != 0
        ):
            raise ValueError(
                "checkpoint_after must be inside the study and align with max_in_flight"
            )

        trace: list[dict] = []
        host_errors: dict[int, dict] = {}
        resumed = False
        batch_size = spec.max_in_flight
        for batch_start in range(0, spec.n_trials, batch_size):
            batch: list[tuple[object, dict, dict]] = []
            current_size = min(batch_size, spec.n_trials - batch_start)
            for _ in range(current_size):
                t = opt.ask()
                params = _resolve(t, spec.space)
                rec: dict = {"id": t.id, "params": params}
                batch.append((t, params, rec))
                trace.append(rec)

            order = spec.tell_order or tuple(range(current_size))
            order = tuple(pos for pos in order if pos < current_size)
            if inter is not None:
                for _t, _params, rec in batch:
                    rec["intermediates"] = []
                    rec["pruned_at"] = -1
                for step in range(spec.intermediate_steps):
                    for pos in order:
                        t, params, rec = batch[pos]
                        if rec["pruned_at"] >= 0:
                            continue
                        score = inter(params, step)
                        should_prune = opt.tell_intermediate(t.id, step, score)
                        rec["intermediates"].append(
                            {
                                "step": step,
                                "score": score,
                                "should_prune": should_prune,
                            }
                        )
                        if should_prune:
                            rec["pruned_at"] = step

            for pos in order:
                t, params, rec = batch[pos]
                if rec.get("pruned_at", -1) >= 0:
                    continue
                if t.id in spec.failed_trial_ids:
                    error = {
                        "category": "objective",
                        "code": "OBJECTIVE_EVALUATION_FAILED",
                        "details": {"phase": "evaluate", "trial_id": t.id},
                        "message": "deterministic objective failure fixture",
                        "retryable": False,
                    }
                    opt.tell_result(
                        t.id,
                        TrialStatus.FAILED,
                        error=TrialError(
                            code=error["code"],
                            message=error["message"],
                            retryable=error["retryable"],
                        ),
                    )
                    host_errors[t.id] = error
                else:
                    score = obj(params)
                    rec["score"] = score
                    opt.tell(t.id, score)

            if checkpoint_after is not None and len(trace) == checkpoint_after:
                checkpoint_records = opt.get_trials()
                if any(
                    record.status is TrialStatus.RUNNING
                    for record in checkpoint_records
                ):
                    raise RuntimeError(
                        "checkpoint split must be a terminal logical-batch boundary"
                    )
                checkpoint = opt.save()
                opt.close()
                opt = Optimizer.load(checkpoint)
                resumed = True

        if checkpoint_after is not None and not resumed:
            raise RuntimeError("checkpoint split was not reached")

        # Build the published parity tape from the owning native rich trace, not
        # from host-side counters. This makes every golden exercise the actual
        # parameter decoder, event sequences, lifecycle and structured-error wire.
        records = opt.get_trials()
        if [record.id for record in records] != [rec["id"] for rec in trace]:
            raise RuntimeError("native rich trace changed trial order")
        for rec, record in zip(trace, records, strict=True):
            if rec["params"] != record.params:
                raise RuntimeError(
                    f"native rich trace changed params for trial {record.id}"
                )
            rec["params"] = dict(record.params)
            rec["asked_at"] = record.ask_sequence
            rec["status"] = int(record.status)
            if record.terminal_sequence is not None:
                rec["terminal_at"] = record.terminal_sequence
            if record.status is TrialStatus.COMPLETED:
                rec["score"] = record.score
            else:
                rec.pop("score", None)

            if inter is not None:
                rec["intermediates"] = [
                    {
                        "sequence": value.sequence,
                        "step": value.step,
                        "score": value.score,
                        "should_prune": value.should_prune,
                    }
                    for value in record.intermediates
                ]
                prune_steps = [
                    value.step for value in record.intermediates if value.should_prune
                ]
                rec["pruned_at"] = prune_steps[0] if prune_steps else -1

            if record.error is not None:
                error = host_errors.get(record.id)
                if error is None:
                    raise RuntimeError(
                        f"missing host error metadata for trial {record.id}"
                    )
                if (
                    record.error.code != error["code"]
                    or record.error.message != error["message"]
                    or record.error.retryable != error["retryable"]
                ):
                    raise RuntimeError(
                        f"native rich trace changed structured error for trial {record.id}"
                    )
                rec["error"] = {
                    **error,
                    "code": record.error.code,
                    "message": record.error.message,
                    "retryable": record.error.retryable,
                }
        return trace
    finally:
        if opt is not None:
            opt.close()
        space.close()
