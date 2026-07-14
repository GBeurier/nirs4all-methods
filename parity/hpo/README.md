# HPO cross-binding parity (Track Q)

Parity harness for the native ask/tell hyperparameter optimizer (`optimization`
role, ABI 2.2), currently driven through the Python binding. It is the HPO
analogue of the numeric fixtures under `parity/fixtures/`: it commits selected
native traces that future R, MATLAB-Octave and JS-WASM runners must reproduce.
Those optimizer wrappers and runners are not implemented by this harness, so the
goldens are acceptance targets rather than proof that every binding already
matches. A separate versioned compatibility contract executes all 45 native
sampler-pruner compositions; that exhaustive native/Python gate is not a set of
45 independent references and is not cross-binding evidence.

## What it checks

1. **Golden-trace stability (the binding acceptance target).** Each registered `HpoSpec`
   (`specs.py`) fully determines a study: sampler, pruner, typed search space,
   seed, trial budget, and a deterministic closed-form objective. Running it
   through a binding produces a **StudyTrace** — the ordered list of proposed
   parameters (+ score / prune decisions). The trace is committed under
   `golden/<id>.json`. Any binding must satisfy the same comparator (float axes
   may use `HpoSpec.tol`, but the current cells default to exact equality). Today this
   proves stability of the native/Python runner. A second binding earns a parity
   claim only after its own runner produces the same trace with the same ordered
   ask/tell schedule and score tape.

2. **Sobol Tier-A (external reference).** The Sobol cell's parameters must equal
   `scipy.stats.qmc.Sobol(scramble=False)` bit-for-bit.

3. **Selected pruner decisions (independent reimplementation).** `references.py` contains
   pure-Python reimplementations of the documented median, ASHA, Hyperband and
   racing rules. The native pruner's prune/keep verdicts must match them on the exact
   rung history each study produces — a second implementation guarding the native
   one (they were written from the spec, not copied from each other). These cells
   invoke `tell_intermediate()` from the host loop; they do not use
   `n4m_finetune_estimator`, which accepts only `pruner = none`.

4. **Exhaustive native composition conformance.**
   `contracts/sampler_pruner_compatibility.v1.json` declares exactly nine
   samplers × five pruners. `compatibility.py` fails closed on a missing,
   duplicate or unknown cell and runs every declared cell through the public
   Python optimizer three times: direct, fresh exact replay, and save/load
   continuation. It also executes every declared transverse refusal and compares
   the stable C ABI status code, never the diagnostic message.

## Exact coverage and non-coverage

There are two deliberately different coverage sets.

The selected frozen registry contains 14 specs, plus one external comparison
gate:

- one native golden trace for each of the nine samplers (`random`, `lhs`,
  `ternary`, `ga`, `pso`, `cmaes`, `tpe`, `sobol`, `gp_ei`);
- one `random` + `median`, one `random` + `asha`, and one `random` +
  `hyperband` decision cell;
- one logically parallel `random` + `racing` decision cell with a fixed tell
  order;
- one random-sampler lifecycle cell with deterministic `FAILED` trials and
  structured errors;
- one external Sobol comparison (3 dimensions, 32 points).

The composition contract additionally contains all 45 pairs. Every pair creates
on the same ordered `int` + `float` space, executes 20 terminal trials, reports
at least one intermediate value per trial (including `pruner=none`), and produces
at least one actual prune decision for every non-`none` pruner. The tape crosses:

- the 16-member first generation and asks a second generation for GA and PSO;
- the six-member CMA-ES generation (`lambda=6` for two continuous axes);
- three completed startup trials followed by a model-driven ASK for TPE and
  GP-EI. The startup proposals equal the random fallback, and the first eligible
  post-startup proposal must differ from it.

Racing uses two logically in-flight trials and a fixed `(1, 0)` reporting order.
This fits the 6/16 population boundaries; requesting a logical batch that crosses
one of those boundaries is instead handled by the documented partial-batch API.

The contract also freezes typed creation refusals for unknown sampler/pruner
enums, invalid startup/resource/reduction options, reserved constant-liar and
evaluation modes, unsupported hard constraints, and hard constraints over a
`sorted_tuple` root. Existing lifecycle tests cover runtime misuse such as
unknown trial ids, score rewrites and illegal terminal transitions.

The 45-cell gate does **not** prove independent sampler correctness (except the
separate SciPy Sobol gate), independent pruner correctness (except the four
selected decision references), R/MATLAB/WASM parity, every
typed/conditional/constrained space, every persistence-state cross-product, or
the pure-native finetune driver. In particular, no cross-binding 45/45 claim is
valid until another binding executes the same contract.

## Ordered events, population schedules and replay

The reproducibility input is the complete ordered event stream, not wall-clock
timing:

1. `ASK` fixes the trial id and proposed parameters.
2. Zero or more `INTERMEDIATE` events carry `(step, score)` and the native prune
   verdict in their global sequence order.
3. Exactly one `TERMINAL` event closes the trial as completed, pruned, failed or
   cancelled.

For sequential-history samplers, each terminal report is visible to the next
ASK. GA, PSO and CMA-ES are generation-synchronous: candidates in the current
generation are fixed together, and the next generation cannot be sampled until
every current member is terminal. Trials may still be dispensed and terminalized
one at a time; "generation-synchronous" describes the learning boundary, not a
requirement that the host keep the whole population concurrently in flight.
Pruned/failed/cancelled members resolve the boundary, but only completed scored
members enter the population update.

`max_in_flight` and `tell_order` pin logical concurrency. Thread scheduling and
arbitrary wall-clock completion order are not interchangeable with that logical
tape. The exact comparison includes ids, parameters, statuses, scores,
intermediates, prune decisions, errors and global event sequences. It
deliberately excludes `trial_duration` and all other wall-clock measurements.

For each of the 45 cells, the third execution saves after 14 terminal trials at
a logical-batch boundary, asserts that no trial is `RUNNING`, restores via
`Optimizer.load()`, and must reproduce the direct continuation exactly. This
proves a terminal-prefix checkpoint for every pair. It does not claim coverage
of checkpoints containing running trials or queued warm starts; those states
remain covered by their dedicated native tests rather than this matrix.

## Why the objective is closed-form

The cross-binding contract only holds if `tell()` receives the same score in every
language. So the objective is a **portable closed form** of the proposed
parameters, not a model fit. The exact formulas live in `objectives.py`
(`objective_formula_doc()`). The current 14 specs use `sphere` as the terminal
objective and `learning_curve` / `racing_observation` for intermediate tapes;
`weighted_ramp` is defined but is not currently registered. A
binding runner must reimplement the formulas used by each selected spec. The
trace comparison is on the *proposals* and recorded prune points, so the selected
sampler/pruner paths — not model-fitting numerics — are what is pinned.

The exhaustive matrix additionally uses
`compatibility_curve(p, s) = -sphere(p) + 5/(s+1)`. Its intentionally inverted
asymptote makes every pruner exercise non-trivial keep/prune decisions even after
an adaptive sampler improves the terminal objective. It is a deterministic test
stimulus, not an external numerical reference.

## Ordered search-space contract

`contracts/ordered_search_space.v1.json` pins the portable axis order, typed
domains, structured parameter targets, activation semantics and SHA-256 space
fingerprint used before a study starts. It also pins the explicit projection to
the current ordered C builder calls without pretending that host-only patch
targets exist in `n4m_search_space_t`. `ordered_space_contract.py` is an
independent test-only validator for that wire contract. It is deliberately not
imported by the Python binding: production validation will eventually be native
so every binding sees the same rejection and fingerprint behavior.

## Usage

```bash
# Check (CI gate): selected traces + references + 45 cells + typed refusals
N4M_LIB_PATH=/path/to/libn4m.so.2.2.0 python parity/hpo/run.py

# Regenerate goldens after an intentional algorithm change (review the diff!)
N4M_LIB_PATH=/path/to/libn4m.so.2.2.0 python parity/hpo/run.py --update

# One selected golden spec (the exhaustive 45-cell gate still runs)
N4M_LIB_PATH=... python parity/hpo/run.py --only sobol_unit3
```

Wired into `.github/workflows/parity-gate.yml` (the `fixture-determinism` job,
which already builds `libn4m`). Skips cleanly when `N4M_LIB_PATH` is unset.

## Adding a binding to the contract

An R / MATLAB / WASM binding validates the selected golden contract by adding a
binding-specific runner that
builds the ordered search space and optimizer for each `HpoSpec`, runs the same
ask/tell loop with the reimplemented objective, emits the StudyTrace in the same
JSON shape, and diffs it against `golden/<id>.json`. Passing those cells means the
binding matches the covered native traces; it is not evidence for unregistered
sampler/pruner or search-space combinations. It may then implement the versioned
45-cell compatibility contract with the same canonical tape. Only the results of
that second runner can establish cross-binding 45/45 parity; the current
Python-driven native gate alone cannot.

## Files

| File | Role |
|---|---|
| `specs.py` | `HpoSpec` dataclass + `REGISTRY` (the parity cells). |
| `objectives.py` | Portable closed-form objectives (`OBJECTIVES`, `INTERMEDIATE`). |
| `run_native.py` | Run a spec, then build StudyTrace from the owning native `Optimizer.get_trials()` snapshot. |
| `comparators.py` | Trace comparison + golden load/dump. |
| `references.py` | scipy Sobol + pure-Python pruner rules. |
| `compatibility.py` | Semantic validator, 45-cell executor, transition/checkpoint assertions and typed refusal probes. |
| `ordered_space_contract.py` | Test-only semantic validator and fingerprint oracle. |
| `run.py` | CLI gate (check / `--update`). |
| `contracts/sampler_pruner_compatibility.v1.json` | Machine-readable 9 × 5 matrix, tape and transverse refusal catalogue. |
| `contracts/sampler_pruner_compatibility.v1.schema.json` | Local JSON Schema 2020-12 shape for that contract. |
| `contracts/` | Versioned StudyTrace and ordered search-space contracts. |
| `golden/` | Committed native golden traces (binding acceptance targets). |
| `checkpoint_resume.py` | Random + adaptive-TPE N4MOPT continuation tape generator. |
| `golden/checkpoint_resume.v1.json` | Frozen pre-checkpoint and resumed proposals/scores. |
| `tests/test_hpo_parity.py` | pytest wrapper (skips without a build). |
| `tests/test_checkpoint_resume.py` | Bit-exact direct-vs-restored continuation gate. |
| `tests/test_sampler_pruner_matrix.py` | Exact 45-cell and typed-refusal pytest gate. |
