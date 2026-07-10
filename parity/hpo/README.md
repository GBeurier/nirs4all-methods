# HPO cross-binding parity (Track Q)

Parity harness for the native ask/tell hyperparameter optimizer (`optimization`
role, ABI 2.1). It is the HPO analogue of the numeric fixtures under
`parity/fixtures/`: it pins the **behaviour** of every sampler and pruner so that
the R / MATLAB-Octave / JS-WASM bindings — which call the same `libn4m` — can be
proven to reproduce the Python binding byte-for-byte.

## What it checks

1. **Golden-trace stability (the cross-binding contract).** Each `HpoSpec`
   (`specs.py`) fully determines a study: sampler, pruner, typed search space,
   seed, trial budget, and a deterministic closed-form objective. Running it
   through a binding produces a **StudyTrace** — the ordered list of proposed
   parameters (+ score / prune decisions). The trace is committed under
   `golden/<id>.json`. Any binding must reproduce it exactly (float axes may use
   `HpoSpec.tol`, but the samplers are deterministic at a fixed seed, so the
   default is exact equality). This is the real guarantee: *same seed → same
   proposals, everywhere.*

2. **Sobol Tier-A (external reference).** The Sobol cell's parameters must equal
   `scipy.stats.qmc.Sobol(scramble=False)` bit-for-bit.

3. **Pruner decisions (independent reimplementation).** `references.py` contains
   pure-Python reimplementations of the documented median / ASHA / Hyperband
   rules. The native pruner's prune/keep verdicts must match them on the exact
   rung history each study produces — a second implementation guarding the native
   one (they were written from the spec, not copied from each other).

## Why the objective is closed-form

The cross-binding contract only holds if `tell()` receives the same score in every
language. So the objective is a **portable closed form** of the proposed
parameters, not a model fit. The exact formulas live in `objectives.py`
(`objective_formula_doc()`); a binding author reimplements those three functions
(`sphere`, `weighted_ramp`, `learning_curve`) and then the golden traces are
reachable. The trace comparison is on the *proposals*, so the sampler/pruner logic
— not the objective — is what is actually being pinned.

## Usage

```bash
# Check (CI gate): trace stability + Sobol + pruner decisions
N4M_LIB_PATH=/path/to/libn4m.so.2.1.0 python parity/hpo/run.py

# Regenerate goldens after an intentional algorithm change (review the diff!)
N4M_LIB_PATH=/path/to/libn4m.so.2.1.0 python parity/hpo/run.py --update

# One spec
N4M_LIB_PATH=... python parity/hpo/run.py --only sobol_unit3
```

Wired into `.github/workflows/parity-gate.yml` (the `fixture-determinism` job,
which already builds `libn4m`). Skips cleanly when `N4M_LIB_PATH` is unset.

## Adding a binding to the contract

An R / MATLAB / WASM binding validates by: build its search space + optimizer from
each `HpoSpec`, run the same ask/tell loop with the reimplemented objective, emit
the StudyTrace in the same JSON shape, and diff against `golden/<id>.json`. A pass
means that binding is numerically identical to the C++ core and the Python binding.

## Files

| File | Role |
|---|---|
| `specs.py` | `HpoSpec` dataclass + `REGISTRY` (the parity cells). |
| `objectives.py` | Portable closed-form objectives (`OBJECTIVES`, `INTERMEDIATE`). |
| `run_native.py` | Run a spec through the Python binding → StudyTrace. |
| `comparators.py` | Trace comparison + golden load/dump. |
| `references.py` | scipy Sobol + pure-Python pruner rules. |
| `run.py` | CLI gate (check / `--update`). |
| `golden/` | Committed golden traces (the cross-binding contract). |
| `tests/test_hpo_parity.py` | pytest wrapper (skips without a build). |
