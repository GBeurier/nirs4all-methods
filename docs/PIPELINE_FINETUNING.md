# Pipeline finetuning — tuning operators & sub-pipelines, not just scalars

**Status:** design + phased plan (companion to [`FINETUNING_ROADMAP.md`](FINETUNING_ROADMAP.md) and
[`NATIVE_FINETUNING.md`](NATIVE_FINETUNING.md)). Written after the Phase-1 native HPO
engine landed. **Nothing here is implemented yet** — it plans the step from
"finetune scalar params" to "finetune the pipeline structure itself".

## 1. The goal (what the user wants)

Finetune a **pipeline directly**, with the **operators themselves in the search
space** — going beyond primitive `int/float/categorical` to tune **sub-pipelines
or whole pipelines**, addressed in the Optuna / scikit-learn `object__attribute`
style:

```python
pipeline.finetune({
    "preproc_0": ["snv", "msc", ["sg", {"window": (5, 21, 2), "order": (2, 3)}]],
    "model":     ["pls", "ridge"],
    "model__pls__n_components": ("int", 1, 30),        # active only when model == pls
    "model__ridge__alpha":      ("float", 1e-4, 1e2, "log"),  # active only when model == ridge
}, n_trials=100, sampler="tpe", pruner="asha")
```

A list = a **choice of operator**; a tuple = a **numeric range**; nesting = a
**sub-pipeline**. `a__b__c` addresses a nested slot. Each trial should
**materialise a concrete pipeline**, be fit + CV-scored, and the best pipeline
refit — identically in Python / R / MATLAB / WASM.

## 2. What works TODAY (verified in the native substrate)

The native ask/tell optimizer (ABI 2.1) already has the pieces for a **flat
pipeline, one level of operator-choice per slot**:

| Capability | Native primitive | Evidence |
|---|---|---|
| "which operator" choice | `n4m_search_space_add_categorical` (str/int/float/bool) | `test_space_build` |
| attribute active only for a chosen operator | `N4M_CONSTRAINT_CONDITION_IN` / `_NOT_IN` (child active iff parent label ∈ set) | `test_conditional_activation` (`gamma` active iff `kernel=="rbf"`) |
| deactivating a slot cascades to its attributes | `#` name cascade — deactivating `X` deactivates all `X#*` | `apply_conditions` in `optimizer.cpp` |
| host reads what to build | `n4m_trial_is_active(name)` | `test_conditional_activation` |
| a sampler that groks conditional/tree spaces | `N4M_SAMPLER_TPE` (Tree-structured Parzen) | `test_tpe_converges_mixed` |

**So today, by hand, a host could already compile** a one-level space like the
example's `model` + `model__pls__n_components` + `model__ridge__alpha`: declare the
`model` categorical, then `n_components` with a `CONDITION_IN(model=="pls")` and
`alpha` with a `CONDITION_IN(model=="ridge")`, ask/tell, and read `is_active` to
build the chosen branch. `object__attribute` is just a host naming convention the
optimizer already tolerates (native cascade separator is `#`; a controller can
present `__` and translate).

## 3. What's missing (the gap between "works by hand" and the goal)

### 3a. The pipeline ⇄ search-space compiler + materialiser (the bulk — Phase 2)

Nobody turns a **pipeline-with-tunable-slots** into the flat conditional space and
back:

- **compile**: walk the pipeline spec → emit the categorical slot params + each
  operator's attribute params + the conditional-activation constraints + the
  hierarchical names. Recurse for sub-pipelines.
- **materialise**: given a proposed trial, read the active slots/attrs
  (`is_active` + getters) and rebuild a **concrete pipeline/graph** to fit.
- **eval loop**: host-drives-eval (already the ABI model) — for each trial,
  materialise → CV → `tell()` (+ optional `tell_intermediate` for pruning on the
  `n_components` fidelity axis) → refit the best.

Per the North Star this is **coordination/orchestration**, so it belongs in
**`dag-ml`** (graph compile/plan/materialise/refit — cross-language, so R/WASM/
MATLAB get pipeline finetuning too) plus a **`nirs4all` Python controller** for the
`object__attribute` UX. It is **not** numerical-kernel work.

### 3b. Native ABI gaps that make deep/branchy spaces robust (small — Phase 1.5)

The substrate handles one level cleanly but is fragile for the fully-general case:

- **Single parent per condition.** A child may be gated by exactly one parent
  (`optimization.h:43`; a 2nd parent → `N4M_ERR_UNSUPPORTED`). So "attr active when
  op==sg **and** mode==derivative" (multi-gate) is unexpressible.
- **Label-only cascade, not active-state-aware.** `apply_conditions` is a single
  pass keying off the parent's *label*, with the `#`-prefix carrying "parent
  inactive ⇒ children inactive". Correct for one level; for **deep nesting** (a
  child that is itself a parent) it relies entirely on careful `#`-naming and is
  not validated topologically.
- **No first-class "choice / branch sub-space".** Declaring "slot = one of {branch
  specs}" today means hand-wiring N conditions + a naming discipline. A first-class
  node would remove that boilerplate and let the sampler reason about the branch
  structure directly.
- **No variable-length structure.** "A pipeline of length 1–5, each step one of N
  operators" has no primitive. A **bounded** version flattens to K conditional
  slots (pure convention, no ABI); a genuinely dynamic/unbounded space is a bigger
  research item, out of scope for v1.

## 4. Design — `object__attribute` → flat conditional space

```
USER SPEC (nirs4all Python controller, Optuna-style)
    "model": ["pls", "ridge"]
    "model__pls__n_components": ("int", 1, 30)
    "model__ridge__alpha": ("float", 1e-4, 1e2, "log")
    "preproc_0": ["snv", "msc", ["sg", {"window": (5,21,2), "order": (2,3)}]]
        │
        ▼  COMPILE  (dag-ml, cross-language)
NATIVE SEARCH SPACE
    categorical  model {pls, ridge}
    categorical  preproc_0 {snv, msc, sg}
    int          model#pls#n_components [1,30]     COND_IN(model, "pls")
    log_float    model#ridge#alpha [1e-4,1e2]      COND_IN(model, "ridge")
    int          preproc_0#sg#window [5,21] step 2 COND_IN(preproc_0, "sg")
    int          preproc_0#sg#order  [2,3]         COND_IN(preproc_0, "sg")
        │
        ▼  ASK  (native optimizer, any binding)
TRIAL  {model="pls", model#pls#n_components=12, preproc_0="snv", … (sg#* inactive)}
        │
        ▼  MATERIALISE  (dag-ml)  — read is_active + getters
CONCRETE PIPELINE  [SNV] → [PLSRegression(n_components=12)]
        │
        ▼  EVAL (host loop)  materialise → CV → tell(score)  [+ tell_intermediate → prune]
        ▼  best trial → refit
```

The compiler is a pure function `pipeline_template → (search_space, name_map)`;
the materialiser is `(trial, template) → concrete_pipeline`. Both are deterministic
and live once in `dag-ml`, so every binding gets identical behaviour — the same
cross-binding guarantee the Track-Q golden traces already enforce for scalars.

## 5. Phased plan

### Phase 1.5 — native enablers (this repo, `nirs4all-methods`) — OPTIONAL, ABI-additive

Only needed for **deep/branchy** spaces; one-level pipelines work without them.
Recommended order (each is a small, Codex-reviewed, parity-gated block like the
Phase-1 samplers):

| ID | Enabler | ABI impact | Why |
|---|---|---|---|
| **E2** | Topological, active-state-aware conditional cascade in `apply_conditions` — a child is active iff parent is active **and** label matches, resolved in dependency order. | Behavioural only (no symbol) | Makes deep nesting correct without relying on `#`-naming discipline. Do this regardless. |
| **E1** | Multi-parent conditions (`CONDITION_ALL_IN`, or allow repeated parents with AND semantics). | Enum value (ABI minor, reserved slot) | Enables multi-gate attributes (op==sg AND mode==derivative). |
| **E3** | First-class choice/branch node: `n4m_search_space_add_choice(slot, branch_labels…)` + branch-scoped param helpers. | New symbols (ABI minor) | Removes host boilerplate; lets samplers see branch structure. |
| **E4** | Bounded variable-length convention (K slots each "operator-or-none") documented + a helper. | Convention (no ABI) or one helper | "Pipeline of length ≤ K" without dynamic structure. |

Each ships with: C++ core + test, a `parity/hpo/` spec exercising the conditional
structure (extend `HpoSpec` with a `structural` cell + golden trace), doc + bib,
ABI snapshot regen if E1/E3 add a symbol. **These bump the ABI minor**, so they
need your green light (the standing rule: no ABI change without sign-off).

### Phase 2 — the actual pipeline finetuning (`dag-ml` + `nirs4all-core` + `nirs4all` Python)

The substance. **Touches other repos → blocked on your green light + Phase-2
start** (and an agent may be active in dag-ml/nirs4all-core — coordinate).

| ID | Deliverable | Repo |
|---|---|---|
| **P1** | Tunable-graph annotation: mark graph nodes/params tunable (choice / range / nested). | `dag-ml` |
| **P2** | `compile(template) → native search space` + `materialise(trial, template) → graph`. | `dag-ml` (calls the native search-space + optimizer via its C-ABI) |
| **P3** | Host-drives-eval loop over materialised graphs (CV, `tell`, optional pruning on `n_components`). | `dag-ml` |
| **P4** | `pipeline.finetune({...}, sampler=…, pruner=…)` — the `object__attribute` UX; wraps P1–P3 + the native optimizer binding; Optuna stays selectable. | `nirs4all` (Python controller) |
| **P5** | Idiomatic thin wrappers so R / MATLAB / WASM get the same `finetune(template, …)`. | `nirs4all-core` bindings |
| **P6** | Extend Track-Q: golden **pipeline-space** traces (same template + seed → same materialised pipeline sequence, every binding). | `nirs4all-methods` `parity/hpo/` (contract) + per-repo runners |

### Dependencies

```
E2 ─┬─► E1 ─► E3 ─► (E4)        [Phase 1.5, nirs4all-methods, ABI minor]
    └───────────────► P2 ──► P3 ──► P4 ──► P5
                 P1 ─┘                 └──► P6 (contract back in nirs4all-methods)
```

P2 can start against **today's** substrate (one-level spaces) and adopt E1/E3 as
they land — so Phase 2 is **not blocked** on Phase 1.5; the enablers upgrade it
from "flat pipelines" to "arbitrary nested sub-pipelines".

## 6. Verdict

- **Flat pipeline, one operator-choice per slot with gated attributes:** possible
  **now**, no native change — a host compiler over the existing conditional space.
- **Sub-pipelines / whole-pipeline / deep nesting in `object__attribute` form:**
  **not yet** — needs the Phase-2 compiler+materialiser+controller (the bulk) and,
  for robustness/ergonomics, the Phase-1.5 native enablers E1–E3.
- **None of it requires new numerical kernels** — the optimizer, samplers and
  pruners are done. The remaining work is *space construction* (small, native) and
  *orchestration* (large, dag-ml + Python), exactly the split the North Star
  predicts.
