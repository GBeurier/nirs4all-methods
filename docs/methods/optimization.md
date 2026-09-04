# Optimization role — native hyperparameter finetuning

The `optimization` role (C ABI header [`n4m/optimization.h`](../../cpp/include/n4m/optimization.h), ABI 2.2) is a **portable ask/tell hyperparameter optimizer**: the search algorithm lives once in `libn4m` and can be reused by every binding. The native core and Python binding are implemented. R, MATLAB-Octave and WASM optimizer wrappers and their cross-binding gates are still pending, so cross-language reproducibility is a target backed by the shared C ABI and Track-Q fixtures, not yet a release claim for every binding. Design rationale and the full plan are in [developer documentation](../dev/documentation.md) and [`NATIVE_FINETUNING.md`](../NATIVE_FINETUNING.md); the ABI freeze is detailed in [developer documentation](../dev/documentation.md).

## Model

Three objects, all opaque C-ABI handles:

- **`n4m_search_space_t`** — a typed, ordered set of parameters (`int`, `float`, `log_int`, `log_float`, `categorical` with typed values, `ordinal`, `sorted_tuple`) plus declarative constraints (`mutex_group`, `requires`, `exclude`, `condition_in`/`_not_in`). Built with `n4m_search_space_add_*`; final cross-axis and sampler-compatibility validation happens in `n4m_optimizer_create`.
- **`n4m_optimizer_t`** — the stateful search. `ask()` proposes a trial; the host evaluates it however it likes; `tell()` / `tell_result()` reports the outcome (`completed` / `pruned` / `failed` / `cancelled`). `tell_intermediate()` reports a fidelity-rung score for pruning. `best()` returns the incumbent; `get_trials()` snapshots the trace; `enqueue()` warm-starts a known configuration.
- **`n4m_trial_t`** — one proposed configuration (borrowed; owned by the optimizer). Read parameters with `n4m_trial_get_int/float/category`, activation with `n4m_trial_is_active`.

The **host drives the loop** (`ask → evaluate → tell`); there is no C→host objective callback, so the pattern is safe under R's non-reentrant evaluator, WASM's synchronous runtime, and Python's GIL. Pruning is host-driven too: the host reports each rung through `tell_intermediate()` and acts on the returned decision.

## Pure-native estimator selection

`n4m_finetune_estimator` is the closed, single-level convenience driver for a
small set of generic native regression estimators. It uses the same optimizer,
trial lifecycle and rich trace as the host-driven loop, but evaluates every
candidate through native regression cross-validation. It invokes no host
callback, creates no second DAG and accepts no classification route.

Eligibility and tuned axes are registry-driven:

| `n4m_algorithm_t` | Required search-space schema | Omitted-value behavior |
|---|---|---|
| `N4M_ALGO_PLS_REGRESSION` | exactly `n_components` | — |
| `N4M_ALGO_PLS_CANONICAL` | exactly `n_components` | — |
| `N4M_ALGO_PLS_SVD` | exactly `n_components` | — |
| `N4M_ALGO_OPLS` | exactly `n_components` | — |
| `N4M_ALGO_PCR` | exactly `n_components` | — |
| `N4M_ALGO_SPARSE_PLS` | any non-empty subset of `n_components`, `sparsity_lambda` | omitted `n_components = 2`; omitted `sparsity_lambda = 0.0` |

`PLS_DA`, `OPLS_DA`, `MB_PLS`, `LW_PLS`, `AOM_PLS` and unknown enum values
return `N4M_ERR_UNSUPPORTED`. The specialized estimators are not silently
approximated by a generic fit route.

The two recognized keywords are validated before the optimizer is created:

- `n_components` must be one non-log `N4M_PARAM_INT` axis with integral
  `1 <= low <= high <= INT32_MAX` and integral `step >= 1`.
- `sparsity_lambda` must satisfy `0 <= low <= high < 1`. A linear
  `N4M_PARAM_FLOAT` axis accepts `step = 0` for continuous sampling or a
  positive, representable grid step. A `N4M_PARAM_LOG_FLOAT` axis additionally
  requires `low > 0` and `step = 0`.

Unknown or duplicate axes, a missing required axis, a wrong kind/log flag, an
invalid bound or step, and `condition_in` / `condition_not_in` all return the
stable status `N4M_ERR_UNSUPPORTED` before the first `ask()`. Conditional axes
are deliberately refused because both registered keywords are numeric and
cannot serve as the categorical/ordinal condition parent. Other search-space
and sampler restrictions still pass through the normal optimizer validation.

Only regression metrics `RMSE`, `MSE`, `MAE` and `R2` are available. `AUTO`
direction minimizes the first three and maximizes `R2`. The driver requires
`pruner = NONE`: it produces no intermediate scores, so configuring a pruner
returns `N4M_ERR_UNSUPPORTED` rather than pretending pruning occurred.

`X`, `Y` and the complete validation plan are globally preflighted before the
first trial. Matrices must be non-empty float32/float64 views with the same
row count, finite values, and column counts representable by the C ABI. `X` and
`Y` may independently use float32 or float64. Every fold must have non-empty,
unique, in-range, disjoint train/test indices, and every sample must appear in
exactly one test fold. A malformed input therefore returns its data/plan error
without creating a misleading failed trial. Candidate-specific fit failures
remain in the owning trace as `FAILED`; later valid candidates may still
complete and win.

This API performs **selection only**. It does not fit or return a final model on
all rows, enforce nested-CV leakage rules, select an outer-fold candidate, or
refit a pipeline. Those responsibilities remain with dag-ml or the host
controller. The returned `n4m_method_result_t` contains:

- `best_score`, `metric`, `estimator`, `timed_out` and `requested_trials`;
- `best.n_components` and/or `best.sparsity_lambda`, but only for axes active in
  the winning trial;
- the same owning rich trial trace documented below.

The timeout is checked when the driver asks for the next trial. If the deadline
is reached before any candidate completes, the call returns
`N4M_ERR_CANCELLED` and no result. If at least one candidate completed, it
returns `N4M_OK` with the best partial result, `timed_out = 1`, and fewer trace
rows than `requested_trials`. A study that completes its requested final trial
before another `ask()` is not retrospectively marked timed out.

The public Python binding owns both the plan and the returned snapshot:

```python
import numpy as np

from n4m.model_selection import (
    Algorithm,
    Metric,
    Sampler,
    SearchSpace,
    ValidationPlan,
    finetune_estimator,
)

fold_ids = np.arange(X.shape[0], dtype=np.int32) % 5
with ValidationPlan.from_fold_ids(fold_ids) as plan:
    with SearchSpace() as space:
        space.add_int("n_components", 1, 20)
        result = finetune_estimator(
            Algorithm.PLS_REGRESSION,
            X,
            y,
            plan,
            space,
            n_trials=40,
            sampler=Sampler.TPE,
            metric=Metric.RMSE,
            seed=42,
            timeout_seconds=30.0,
        )

print(result.best_params, result.best_score, result.timed_out)
```

`ValidationPlan.from_fold_ids(ids, cv=...)` accepts one zero-based integer test
fold id per sample; omit `cv` to infer `max(ids) + 1`. If `cv` is supplied it
must be an integer in `[2, n_samples]`. Use
`ValidationPlan.from_splits(n_samples, splits)` for explicit ordered
`(train_indices, test_indices)` pairs. Both forms copy their indices into an
owning native handle and reject use after `close()`.

## Samplers and pruners

Selected via `n4m_optimizer_options_t.sampler` / `.pruner`. Algorithms sit behind reserved enum values; a value not yet implemented returns `N4M_ERR_NOT_IMPLEMENTED` at `n4m_optimizer_create`, so activating one in a later phase adds **no** new ABI symbol.

| Kind | Status | Phase | Page |
|---|---|---|---|
| sampler `random` | ✅ implemented | F0 | [random.md](random.md) |
| sampler `ternary` | ✅ implemented | F1 | [ternary.md](ternary.md) |
| sampler `lhs` | ✅ implemented | F1 | [lhs.md](lhs.md) |
| sampler `ga` | ✅ implemented | F3 | [ga_search.md](ga_search.md) |
| sampler `pso` | ✅ implemented | F3 | [pso_search.md](pso_search.md) |
| sampler `cmaes` | ✅ implemented | F4 | [cmaes.md](cmaes.md) |
| sampler `tpe` | ✅ implemented | F4 | [tpe.md](tpe.md) |
| sampler `sobol` | ✅ implemented | F1 | [sobol.md](sobol.md) |
| sampler `gp_ei` | ✅ implemented | F4 | [gp_ei.md](gp_ei.md) |
| pruner `none` | ✅ implemented | F0 | — |
| pruner `median` | ✅ implemented | F2 | [median_pruner.md](median_pruner.md) |
| pruner `asha` | ✅ implemented | F2 | [asha.md](asha.md) |
| pruner `racing` | ✅ implemented | F2 | [racing.md](racing.md) |
| pruner `hyperband` | ✅ implemented | F5 | [hyperband.md](hyperband.md) |

The following matrix describes the current native behavior. “Hard constraints”
means `mutex_group`, `requires` and `exclude`; `condition_in` and
`condition_not_in` are activation rules and remain supported by every sampler.

| Sampler | Conditional activation | Hard constraints | `sorted_tuple` axis | `enqueue()` | Ask lifecycle |
|---|---|---|---|---|---|
| `random` | supported | retry, at most 200 draws | independently sampled | supported, except tuple values | independent asks |
| `lhs` | supported | retry, at most 200 draws | independently sampled | supported, except tuple values | numeric LHS startup, otherwise random |
| `ternary` | supported | retry, at most 200 draws | independently sampled | supported, except tuple values | completed-history ternary decision |
| `tpe` | supported | retry, at most 200 draws | independently sampled, not modelled | supported, except tuple values | completed-history model |
| `sobol` | supported | **rejected at create** | independently sampled, not Sobol | supported, except tuple values | one Sobol point per ask |
| `ga` | supported | **rejected at create** | base RNG; genome coordinate unused | unsupported | synchronous generations of 16 |
| `pso` | supported | **rejected at create** | base RNG; particle coordinate unused | unsupported | synchronous iterations of 16 |
| `cmaes` | supported | **rejected at create** | independently sampled, not modelled | unsupported | synchronous CMA generations |
| `gp_ei` | supported | **rejected at create** | independently sampled, not modelled | unsupported | completed-history model |

A hard constraint that references a `sorted_tuple` root is rejected for **every**
sampler: tuple components are exposed as `name#0`, `name#1`, … and the tuple root
does not yet have trial-level presence semantics. No sampler converts an
unsupported hard constraint into a poor host-provided fitness. Unsupported
sampler/space combinations return `N4M_ERR_UNSUPPORTED` from
`n4m_optimizer_create`, before the first `ask()`.

For `condition_in` and `condition_not_in`, the first reference is the unlabeled
child and the second is a labeled categorical/ordinal parent. A `sorted_tuple`
is supported as the child: when its branch is inactive, every component with
the `name#` prefix is inactive. A tuple cannot be a labeled parent; that malformed
condition returns `N4M_ERR_INVALID_ARGUMENT` at optimizer creation.

Pruners compose with samplers through the options struct, but only in the
host-driven ask/tell loop. `hyperband` requires `max_resource > 0`; all other
pruners require `max_resource = 0`. `reduction_factor` is accepted only by
`asha` and `hyperband` (`0` means the default 3, otherwise it must be at least
2). API composition alone is not the proof: Track-Q separately executes all 45
sampler/pruner pairs through the native/Python compatibility gate. That result
proves deterministic conformance on the contracted plain numeric space, not
cross-binding parity or 45 independent algorithm-oracle results.

The portable, ordered wire format for future DAG-level tuning is specified in
[`ordered_search_space.md`](../architecture/ordered_search_space.md). Axis and
categorical-choice order are fingerprinted. A binding may provide convenient
maps or keyword aliases, but it must compile them to that ordered form.

## Options, defaults and fail-closed validation

Always call `n4m_optimizer_options_init()` before overriding fields. It zeroes
the full struct, sets `struct_size = sizeof(n4m_optimizer_options_t)`, and applies
these ABI 2.1 defaults:

| Option | Default | Meaning or restriction |
|---|---|---|
| `sampler` | `random` | There is no implicit `auto` sampler policy in the C ABI. |
| `pruner` | `none` | Intermediate reports are retained but never prune. |
| `direction` | `auto` | Derived from the metric. |
| `eval_mode` | `mean` | The only implemented value. |
| `metric` | `rmse` | Also determines `auto` direction. |
| `liar` | `none` | The only implemented value. |
| `n_startup_trials` | `10` | Must be positive for TPE, GP-EI and the median pruner. |
| `seed` | `0` | Seeds the native RNG; Sobol itself is unscrambled. |
| `timeout_seconds` | `0` | No timeout; otherwise finite and non-negative. |
| `max_resource` | `0` | Must be positive only for Hyperband. |
| `reduction_factor` | `0` | Default factor 3 for ASHA/Hyperband. |
| `reserved` | all zero | Any non-zero current reserved byte is invalid. |

ABI 2.1 is the first published options layout and does **not** accept a truncated
prefix: `struct_size < sizeof(n4m_optimizer_options_t)` (including zero) returns
`N4M_ERR_INVALID_ARGUMENT`. A larger future layout is accepted; the current
library reads only the known ABI 2.1 prefix and ignores its tail. This is why a
zero-initialized struct is not a substitute for `n4m_optimizer_options_init()`.

Validation is deliberately fail-closed: malformed field values and non-extensible
enum values return `N4M_ERR_INVALID_ARGUMENT`; unknown sampler/pruner values
return `N4M_ERR_NOT_IMPLEMENTED` because those two enums are extensible.
Recognized but unavailable behavior also returns `N4M_ERR_NOT_IMPLEMENTED`, and
recognized unsupported combinations return `N4M_ERR_UNSUPPORTED`. In particular,
`eval_mode = best` or `robust_best`, and
`liar = min`, `mean` or `max`, return `N4M_ERR_NOT_IMPLEMENTED` at optimizer
creation. Pending trials are therefore never assigned an implicit constant-liar
score.

Linear stepped integer/float axes are sampled uniformly by grid index. A high
bound that is not aligned to `low + k * step` is never manufactured by clamping;
the last legal point is the greatest aligned value below the high bound. The
same ULP-bounded grid rule is used by `enqueue()`, including for very small
domains.

## Sampler-pruner compatibility

On a valid plain numeric search space, every exported pruner composes with every
exported sampler: 9 × 5 = 45 supported pairs. There is no hidden pair-specific
fallback. The option preconditions still apply:

| Pruner keyword | Sampler coverage | Required options and effect |
|---|---|---|
| `none` | all nine | Intermediate values are recorded and always return `should_prune=false`. |
| `median` | all nine | `n_startup_trials > 0`; compares the current rung with completed history. |
| `asha` | all nine | `reduction_factor` is `0` (default 3) or at least 2; may prune at successive rungs. |
| `hyperband` | all nine | `max_resource > 0`; `reduction_factor` is `0` or at least 2; assigns deterministic brackets/rungs. |
| `racing` | all nine | Consumes repeated bounded observations in their recorded order. |

Compatibility is conditional on the search-space contract rather than only on
the pair name. Sobol, GA, PSO, CMA-ES and GP-EI reject flat hard constraints with
`N4M_ERR_UNSUPPORTED`; every sampler rejects a hard constraint that references a
`sorted_tuple` root. Random, LHS, ternary and TPE enforce supported flat hard
constraints with bounded rejection sampling. Conditional activation rules remain
available to every sampler. The executable matrix and its stable refusal reasons
live in `parity/hpo/contracts/sampler_pruner_compatibility.v1.json`.

## Best-effort batch ask

`n4m_optimizer_ask_batch(opt, n, out_trials, out_count)` dispenses up to `n`
trials in one call. It is exactly `n` sequential `ask()` calls with no
intervening `tell()` — the constant-liar policy stays `none` — so an adaptive
sampler sees the same ordered history it would one ask at a time. For the same
ordered search space and seed, normative reproducibility requires the same
ordered `ASK` / `INTERMEDIATE` / `TERMINAL` event stream with the same payloads;
this includes intermediate reports because pruner decisions depend on them. It
is not a claim about arbitrary wall-clock completion order in parallel hosts.
The buffer is a caller precondition:
`out_count` is required and is set to `0` whenever possible; for `n > 0` the
caller must provide storage for `n` pointers, and the library initializes all
`n` slots to `NULL` before the first ask. `out_trials` may be `NULL` only when
`n == 0`. `n < 0` returns `N4M_ERR_INVALID_ARGUMENT`.

Every committed slot `[0, *out_count)` is a borrowed `RUNNING` trial exactly as
if produced by `ask()`, in ask order. Committed trials are **never rolled back**;
the caller must eventually `tell`/cancel each one. The status distinguishes
these outcomes:

| Outcome | Status | `*out_count` |
|---|---|---|
| Full batch | `N4M_OK` | `n` |
| Benign partial — population (GA/PSO/CMA-ES) generation boundary or timeout, after ≥1 commit | `N4M_OK` | `0 < count < n` |
| Zero-progress boundary at zero capacity | `N4M_ERR_INVALID_ARGUMENT` | `0` |
| Zero-progress timeout (deadline already reached) | `N4M_ERR_CANCELLED` | `0` |
| Any other failure (unsatisfiable constraints, invalid queued warm-start, allocation/internal) after ≥1 commit | its exact non-OK status | `count > 0`, remaining slots `NULL` |

A benign partial is the normal way population samplers hand back a generation:
the batch stops at the synchronous boundary and returns `N4M_OK` with a short
count. Asking again at zero capacity surfaces the stable boundary status
(`N4M_ERR_INVALID_ARGUMENT`) so the host knows to score the outstanding
generation first. A generation boundary is detected privately (no new public
status); a genuine `N4M_ERR_INVALID_ARGUMENT` — an unsatisfiable constraint or
an invalid queued warm-start — is never swallowed and is returned with the
already-committed trials still valid.

GA, PSO and CMA-ES are generation-synchronous at this boundary. Their candidate
population is fixed when the generation starts; terminal reports from an early
member do not rewrite later candidates in that same generation. The host may
still ask and terminalize members sequentially. A new generation is the point
that consumes the completed scores, and it cannot start until every member of
the previous generation is terminal. Pruned, failed and cancelled members count
as resolved, while only completed members with finite scores participate in the
population update.

Python exposes this as `Optimizer.ask_batch(n) -> list[Trial]`. `n` must be an
`int` in the signed `int32` range (`bool` is rejected); negative values raise a
zero-progress `N4MError` with `N4M_ERR_INVALID_ARGUMENT`. A benign partial simply
returns a shorter list. Any other zero-progress native error raises `N4MError`.
A non-benign error after a partial batch raises `PartialBatchError` (a subclass
of `N4MError`) whose `partial_trials` holds the committed, borrowed trials and
whose `status` is the exact native code; the caller must terminalize those trials
(`tell` / `tell_result`) or recover them via `get_trials`. `PartialBatchError` is
exported from both `n4m` and `n4m.model_selection`.

## Trial lifecycle and structured errors

`n4m_trial_status_t` has one open state and four terminal states. Their stable
numeric values are `RUNNING=0`, `COMPLETED=1`, `PRUNED=2`, `FAILED=3`, and
`CANCELLED=4`. Only a finite score reported as `COMPLETED` participates in
`best()`. `FAILED`, `CANCELLED`, and `PRUNED` never become incumbents.

Do not confuse the two cancellation channels: `N4M_ERR_CANCELLED` returned by
`ask()` means the optimizer deadline rejected that operation; it does not
silently rewrite already-running trials. The host explicitly terminalizes each
such trial with `tell_result(..., N4M_TRIAL_CANCELLED, ...)` when appropriate.

Terminal history is immutable. Repeating the exact same terminal report is
idempotent, but changing its status, completed score, or structured error
returns `N4M_ERR_INVALID_ARGUMENT`. Unknown trial ids are also rejected. A
terminal trial rejects further intermediate reports. Intermediate steps must be
non-negative and strictly increasing; an exact replay of the latest
`(step, score)` is accepted and returns the original prune decision (including
after that decision terminalized the trial), while an out-of-order step or
score rewrite is rejected.

Intermediate insertion is transactional with respect to native pruner
evaluation: if the pruner raises an allocation/internal error, the staged value
and event sequence are rolled back, so retrying cannot silently turn a lost
prune decision into an accepted non-pruning replay. Likewise, an enqueued
warm-start is consumed only after `ask()` commits its trial; an allocation or
sampler exception leaves that queued payload available to retry.

`FAILED` and `CANCELLED` always retain an optimizer-owned error with three
stable fields: `code`, `message`, and `retryable`. The existing C signature is
unchanged. A plain UTF-8 `error` passed to `n4m_optimizer_tell_result` maps to
`OBJECTIVE_ERROR` for `FAILED` or `BUDGET_CANCELLED` for `CANCELLED`, with
`retryable=false`. A null or empty error receives the corresponding default
message. Bindings that need explicit fields use this versioned wire form:

```text
n4m.error.v1|UPPER_SNAKE_CODE|0-or-1|arbitrary UTF-8 message
```

The message is the remainder of the record and may itself contain `|`.
Codes contain 2–64 ASCII uppercase letters, digits, or underscores. An unknown
wire version, invalid UTF-8, invalid code, invalid retry flag, or empty structured
message fails closed without changing the trial. Parameter names and string
categorical labels are validated as UTF-8 at their C builder boundaries too.
`COMPLETED` and `PRUNED` reject a
non-empty error. The `n4m.error.v` prefix namespace is reserved for versioned
records, so legacy plain messages must not start with that prefix.

Python exposes the structure directly:

```python
from n4m.model_selection.optimizer import (
    Optimizer,
    SearchSpace,
    TrialError,
    TrialStatus,
)

space = SearchSpace().add_int("n_components", 1, 30)
with Optimizer(space, seed=42) as optimizer:
    trial = optimizer.ask()
    optimizer.tell_result(
        trial.id,
        TrialStatus.CANCELLED,
        error=TrialError(
            code="BUDGET_CANCELLED",
            message="study deadline reached",
            retryable=True,
        ),
    )
    records = optimizer.get_trials(since_id=trial.id)

assert records[0].status is TrialStatus.CANCELLED
assert records[0].error.retryable is True
```

The legacy Python form `error="objective failed"` remains supported and uses
the same default-code mapping as C.

## Rich trial trace v1

`n4m_optimizer_get_trials(opt, since_id, &result)` returns trials with
`id >= since_id`, in ascending id order. The filter is inclusive. The returned
`n4m_method_result_t` owns a point-in-time copy of every buffer: it remains valid
after the optimizer, search space, and context are destroyed, and must be freed
with `n4m_method_result_destroy`. Before the first `ask()`, the snapshot has zero
trial rows but still publishes the ordered parameter names and width from the
validated search space.

The five original ABI 2.1 matrices are preserved so existing consumers keep
working. New consumers should use `trial_ids_i64`, because the compatibility
`trial_ids` double matrix cannot exactly represent ids above 2^53. Rich trace v1
uses row-major matrices plus flattened streams and offsets:

| Key | Method-result type and shape | Meaning |
|---|---|---|
| `trace_format_version` | scalar, `N4M_TRIAL_TRACE_FORMAT_VERSION` (`1`) | Decoder discriminator. |
| `n_trials`, `n_params`, `n_intermediates`, `n_events` | scalars | Selected row/parameter/intermediate/event counts. |
| `trial_ids` | double matrix `1 × n` | Compatibility ids; retained unchanged. |
| `trial_scores` | double matrix `1 × n` | Completed score, otherwise NaN. |
| `trial_status`, `trial_rung`, `trial_duration` | double matrices `1 × n` | Compatibility lifecycle columns. |
| `trial_ids_i64` | int64 vector `n` | Exact trial ids. |
| `trial_ask_sequence`, `trial_terminal_sequence` | int64 vectors `n` | Absolute event order; terminal is `-1` while running. |
| `trial_param_values` | double matrix `n × p` | Numeric value; string categoricals carry their choice index. |
| `trial_param_category_index` | int32 vector `n*p` | Row-major category index, `-1` for numeric. |
| `trial_param_active` | int32 vector `n*p` | Row-major activation bits. |
| `trial_param_kind` | int32 vector `p` | Ordered `n4m_param_kind_t` metadata. |
| `trial_param_category_type` | int32 vector `p` | `n4m_cat_type_t`, or `-1` when not categorical. |
| `trial_param_integer` | int32 vector `p` | Integer-axis/tuple-element bit. |
| `trial_param_name_utf8` / `_offsets` | int32 bytes + int64 offsets `p+1` | Ordered parameter names. |
| `trial_param_label_utf8` / `_offsets` | int32 bytes + int64 offsets `n*p+1` | Row-major category labels; empty for numeric. |
| `trial_intermediate_offsets` | int64 vector `n+1` | Per-trial slices into the intermediate streams. |
| `trial_intermediate_sequence` | int64 vector `m` | Absolute event order for each report. |
| `trial_intermediate_steps` | int32 vector `m` | Strictly increasing step per trial. |
| `trial_intermediate_scores` | double matrix `1 × m` | Finite intermediate scores. |
| `trial_intermediate_should_prune` | int32 vector `m` | Native pruner decision for each report. |
| `trial_error_code_utf8` / `_offsets` | int32 bytes + int64 offsets `n+1` | Per-trial stable code, empty without error. |
| `trial_error_message_utf8` / `_offsets` | int32 bytes + int64 offsets `n+1` | Per-trial message, empty without error. |
| `trial_error_retryable` | int32 vector `n` | Per-trial retry policy bit. |

Every UTF-8 pool stores unsigned byte values in the range 0–255 inside the
existing int32 method-result vector type. Offsets start at zero, are
non-decreasing, and end at the byte/element stream length. Sorted-tuple
components appear under their deterministic expanded names (`name#0`,
`name#1`, …), exactly as they do on `n4m_trial_t`.

Python's `Optimizer.get_trials()` decodes this payload into owning
`TrialRecord` objects. Each record exposes `params`, lossless
`param_details` (`value`, `kind`, `category_index`, `category_label`,
`category_type`, `integer`, `active`), a tuple of `IntermediateValue`,
`ask_sequence` / `terminal_sequence`, and an optional
`TrialError`. Sequence numbers are optimizer-global, monotonic, and are not
renumbered by `since_id`, so consumers can reconstruct deterministic ASK,
INTERMEDIATE, and TERMINAL order across a filtered page. These records remain
usable after `Optimizer.close()`. In contrast, the lightweight `Trial` returned
by `ask()` remains a borrowed handle and deliberately refuses access after its
optimizer closes.

`TrialRecord.params` preserves declared scalar types: integer/log-integer and
integer tuple elements decode as Python `int`; string/int/float/bool
categoricals decode to the corresponding Python type; ordinal and continuous
axes remain `float`. Inactive values stay present for auditability and carry
`active=False` in `param_details`.

## Reproducibility

The optimizer owns a seeded `n4m_rng`; the same ordered search space and seed,
followed by the same ordered `ASK` / `INTERMEDIATE` / `TERMINAL` event stream
with the same payloads, reproduce the exact native ask continuation. The stream
is normative: `ASK` precedes every intermediate for that trial, intermediate
steps are strictly ordered, and one `TERMINAL` closes it. Sequential-history
samplers consume each terminal history update before the next ask. GA, PSO and
CMA-ES instead consume updates at the generation boundary described above.

Track-Q commits 14 selected native golden specs: one per sampler, four
random-sampler pruner decision cells (median, ASHA, Hyperband and racing), and a
failed-trial lifecycle cell. Those traces are acceptance targets for future
bindings. In addition,
`parity/hpo/contracts/sampler_pruner_compatibility.v1.json` declares and executes
the exhaustive 9 × 5 composition matrix on a plain ordered numeric space. Every
cell must terminalize 20 trials, cross the relevant startup/generation boundary,
replay exactly, and continue exactly through a public checkpoint. Its transverse
refusal catalogue checks stable C ABI status codes rather than error messages.

This exhaustive result is **native/Python deterministic conformance**, not 45
independent algorithm references and not cross-binding parity. Only Sobol has an
external sampler oracle; the four selected pruner cells have independent
decision-rule implementations. R, MATLAB-Octave or WASM earns a parity claim
only after its own runner reproduces the same covered tapes. Parallel execution
is reproducible only when the binding records and replays the same logical event
stream and fixed tell order. Arbitrary wall-clock completion order is not
interchangeable with event order, and `trial_duration` is intentionally excluded
from exact replay comparisons. The exact proof boundaries are documented in
`parity/hpo/README.md`.

## Portable checkpoint and resume

`n4m_optimizer_save` writes **N4MOPT format v1** and
`n4m_optimizer_load` restores it transactionally. The checkpoint contains the
ordered search space, normalized optimizer options, every trial and parameter,
terminal errors, intermediate values, global event sequences, queued warm
starts, the SplitMix64 stream, and the mutable sampler state required by Sobol,
GA, PSO, CMA-ES, and GP-EI. Random, LHS, ternary and TPE need only their base
RNG, options and history. The five pruners are pure functions of options and
the stored intermediate history, so they have no additional hidden state.

All nine samplers and all five pruners are checkpointable. After a successful
load, continuation is bit-identical to the source optimizer when both sides
receive the same ordered `ASK` / `INTERMEDIATE` / `TERMINAL` events with the
same payloads; intermediate order matters because pruners consume it.
The parity gate freezes random and adaptive-TPE continuation tapes. Its exhaustive
9 × 5 matrix also saves and restores every pair after a 14-trial terminal prefix,
asserting that no trial is running at that logical-batch boundary before comparing
the exact continuation. This matrix does not claim a running/pending checkpoint
state. The C++ gate independently crosses an adaptive boundary for every sampler
and covers the broader native checkpoint state machine.

Python uses bytes rather than exposing the C word container:

```python
from n4m.model_selection import Direction, Optimizer, Sampler, SearchSpace

space = SearchSpace().add_float("alpha", 1e-6, 1.0, log=True)
optimizer = Optimizer(
    space,
    sampler=Sampler.TPE,
    direction=Direction.MINIMIZE,
    n_startup_trials=10,
    seed=42,
)

# ... ask / tell ...
checkpoint: bytes = optimizer.save()
resumed = Optimizer.load(checkpoint)
next_trial = resumed.ask()
```

`Optimizer.load()` copies its input and owns the decoded search space. Closing
the original `SearchSpace` or releasing the input bytes is therefore safe.
Decode errors raise `N4MError`; no partial optimizer handle is exposed.

At the C ABI, `save` returns an owning `1 x n_words` `N4M_DTYPE_I64`
`n4m_array_t`. Its backing storage is the byte stream verbatim. Obtain it with
`n4m_array_view`, pass `view.data` and `view.cols * 8` to `load`, then release it
with `n4m_array_free`. This word container preserves the already-frozen ABI even
though `n4m_dtype_t` has no byte dtype.

N4MOPT is little-endian, size-bounded to 64 MiB, and carries independent
canonical SearchSpace/options fingerprints plus a whole-message FNV-1a checksum.
Load rejects bad magic/checksum/padding, truncation or trailing bytes, excessive
counts, malformed UTF-8, a future format version, and any option/space/state
mismatch. FNV-1a detects accidental corruption; it is not authentication, so an
application must authenticate untrusted checkpoints separately. No pointer,
native object layout, ABI-sized integer, or steady-clock epoch is serialized.
Elapsed optimizer/trial time is restored relative to the load instant, so time
spent with the process stopped is paused rather than charged to the timeout.

There is deliberately no native `save(path)` operation: persistence policy and
filesystem authority stay in the host. If Python writes the returned bytes to a
durable path, write a temporary file in the same directory, flush/fsync as
required by the application, and atomically replace the destination; a direct
write is not crash-atomic. Format v1 has no lossy migration path. A future
format version is rejected with `N4M_ERR_VERSION_INCOMPATIBLE` until an explicit
reader/migration is added.

## Current ABI 2.2 limitations

- non-`mean` `eval_mode` and non-`none` constant-liar values are exported enum
  reservations but are explicitly rejected with `N4M_ERR_NOT_IMPLEMENTED`;
- `n4m_finetune_estimator` is the closed six-route regression/no-pruner driver
  described above, not an arbitrary estimator, final-refit or DAG callback
  runner;
- Python `Trial` objects are borrowed from their parent `Optimizer`; closing the
  optimizer invalidates them and later access raises an error.

The rich `get_trials` result remains an in-memory reporting contract; persist
N4MOPT bytes from `save()` rather than serializing that result's implementation
layout.
