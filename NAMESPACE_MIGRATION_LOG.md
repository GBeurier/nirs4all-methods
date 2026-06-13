# Namespace migration — work log

**Goal:** reorganize all 208 `libn4m` methods under the **ML/DL pipeline namespace (Scheme B)**
across the C ABI, the C++ core public surface, every language binding, both web dashboards, and
all docs — then propagate the change to every downstream repo that depends on `nirs4all-methods`,
updating code + docs and proving no regression.

**Decision:** Scheme B (ML/DL) chosen by the maintainer. Reference: `proposals/namespace/NAMESPACE_PROPOSALS.md`
(`# Scheme B — ML / DL PIPELINE`), flat mapping `proposals/namespace/_ml_table.md`, ground-truth
inventory `proposals/namespace/_method_inventory.json` (208 ids, validated 208/208 mapped, no collisions).

**Review discipline (maintainer's rule):** every major step is reviewed by **Codex** *before*
(roadmap gate) and *after* (code / doc / tests / e2e gate). Multiple agents are used for execution.
The agent updating the **alternative web page** is explicitly briefed ("notified") with the final
namespace spec.

---

## Target namespace (Scheme B) — top level

`n4m.{ transform, decomposition, feature_selection, augmentation, estimators, ensemble,
model_selection, compose, domain_adaptation, outlier_detection, metrics, utils }`
— 12 top-level nodes, 39 leaf namespaces. Symbol convention `n4m_<role>_<method>_<verb>`;
Python mirrors sklearn (`from n4m.transform.scatter import SNV`,
`from n4m.estimators.regression.linear import Ridge`, ...).

---

## Phase plan

| # | Phase | Codex-before (roadmap) | Codex-after (review) | Status |
|---|-------|:---:|:---:|--------|
| 0 | Review the chosen ML table; finalize target spec | n/a | ✅ ml-table review | **in progress** |
| 1 | Master migration roadmap + downstream dependency map | ☐ | n/a | pending |
| 2 | Migrate `nirs4all-methods` core: C ABI headers + symbols, C++ registry, catalog | ☐ | ☐ | pending |
| 3 | Migrate bindings: Python, R, MATLAB/Octave, Julia, JS-WASM, JNI/Android | ☐ | ☐ | pending |
| 4 | Update both web dashboards (current + alternative `dashboard_v2`) + docs (Sphinx) | ☐ | ☐ | pending |
| 5 | Propagate to downstream repos (code + docs) + regression tests | ☐ | ☐ | pending |

> Execution of phases 2–5 is **held** until the running visualization workflow (`dashboard_v2`)
> completes, per maintainer instruction ("attends le workflow"). Phase 0 (namespace review) is
> independent of that workflow and proceeds now.

---

## Timeline

### 2026-06-13

- **Namespace proposals delivered** (prior workflow): `NAMESPACE_PROPOSALS.md` with Scheme A (stats)
  + Scheme B (ML/DL), both mapping all 208 methods 1:1; maintainer selected **Scheme B**.
- **Visualization workflow** (`wmff0y9nk`) still running — building the alternative parity/benchmark
  page `proposals/dashboard_v2/`. Migration execution is gated on its completion.
- **Visualization workflow result:** `build:dashboard` **FAILED** — *"Claude Fable 5 is currently
  unavailable."* The alternative page `proposals/dashboard_v2/` was never created. The namespace
  track of the same workflow **succeeded** (proposal + inventory complete, 208/208 validated).
- **Transport fix:** the Codex **MCP** call hung (~no response). Switched all Codex reviews to the
  documented `codex exec` **CLI** path (works: `codex-cli 0.139.0`), run **backgrounded with a hard
  timeout** so a stalled reviewer can never block the session again.
- **Model fix:** Fable is unavailable — **all agents now run on Opus** (dashboard included).

### Resume (2026-06-14) — two independent background tracks

- **Track A (namespace gate):** Codex reviewing Scheme B (the ML table) via `codex exec`
  → `/tmp/codex_ml_review.md`. Verdict + change list folded in here on return, then target spec is
  finalized and the migration roadmap written (Codex roadmap gate before any code mutation).
- **Track B (visualization):** rebuilding the alternative dashboard `proposals/dashboard_v2/` on
  **Opus** (self-contained, offline, no CDN), independent of the namespace work. Phase 4 will later
  relabel its method grouping to the final Scheme-B namespace — this is the point at which the
  "alternative web page" work is briefed ("notified") with the final namespace.

<!-- CODEX_ML_REVIEW -->
