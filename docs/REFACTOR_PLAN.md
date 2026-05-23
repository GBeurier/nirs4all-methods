# Refactor plan — `nirs4all-methods` (n4m)

> Founding document for the post-merge refactor (`pls4all` → `nirs4all-methods`).
> Status: draft — to be validated before the backlog starts executing.
> Date: 2026-05-23.

---

## 1. Library objectives

### 1.1 One-sentence statement

`nirs4all-methods` (n4m) is a scientific library of ML/NIRS methods (PLS, preprocessing, augmentation, variable selection, splitters, etc.) implemented in **optimized C++**, exposed through a **stable C ABI**, and distributed into every scientific ecosystem (Python, R, MATLAB/Octave, Julia, JS/WASM, etc.) — both as a **raw low-level API** and as **idiomatic API wrappers** compatible with the dominant framework of each ecosystem (scikit-learn, R formula, mdatools, etc.).

Each method must be:
- **documented** scientifically and technically,
- **verified** numerically against an external reference implementation (or, when no external implementation exists, against documented self-consistency properties — see §1.3 Pillar 2),
- **verified** identical across all of its bindings,
- **measured** for execution time across varied scenarios.

All of this information feeds, automatically, into the per-method documentation and into a **public web dashboard** (the "benchmark matrix").

### 1.1bis Library scope and extensibility model

n4m is a **closed library** in the sense that **the set of methods is curated and extensible only via Pull Request** to this repository. There is no plugin API, no user-side method registration, no runtime injection of custom operators. This is a deliberate choice:

- every method ships with mandatory documentation, parity validation against a reference, and inter-binding verification — none of these can be enforced on out-of-tree code,
- the catalog (`catalog/methods/*.yaml`) is the single source of truth for what the library exposes — guaranteeing that PyPI/CRAN/npm packages match the C++ ABI surface bit-for-bit,
- the same constraint also guarantees that the dashboard accurately reflects the entire library: no hidden out-of-tree methods can pollute the benchmark matrix.

Users who want to add a method submit a PR following the `CONTRIBUTING.md` checklist (cf. backlog Phase A17). Users who want to compose methods do so **inside their host language** (sklearn `Pipeline`, R formulas, MATLAB scripts) — composition is not an n4m primitive.

**User-driven requests are channelled through structured GitHub issue templates** (`.github/ISSUE_TEMPLATE/`) so the maintainer can triage cleanly:

- **`method-request.yml`** — request a new method (publication DOI, expected external reference if any, dataset suggestion, expected inputs/outputs).
- **`method-update.yml`** — request an update to an existing method's behaviour / parameters (with rationale and reference to the spec change).
- **`external-reference-request.yml`** — request adding a new external reference implementation (`alternates`) for an existing method, or pinning a newer version of an existing reference.
- **`binding-request.yml`** — request a new language binding (target language, target framework / idiomatic wrapper, expected userbase, packaging registry).
- **`binding-update.yml`** — request an update to an existing binding's idiomatic wrapper (e.g. add a new idiomatic wrapper for an already-supported language, like a TensorFlow.js layer on top of the existing JS binding).
- **`subset-request.yml`** — request a new packaging subset (`pls-diagnostics`, `pls-ensembles`, etc.).
- **`bug-report.yml`** and **`parity-discrepancy.yml`** — standard bug reports plus a dedicated form for numerical parity issues (with minimal reproducer, expected reference values, observed n4m output, environment).

The maintainer triages, decides whether to accept, then either drafts the PR themselves or invites the requester to submit one (with the issue acting as the design pre-discussion). This keeps the library closed while giving users a structured way to drive its evolution.

**Each issue template has a matching pull-request template** under `.github/PULL_REQUEST_TEMPLATE/` so the work submitted in the PR is reviewable against the original request, and the PR carries the exact contributor checklist that `CONTRIBUTING.md` would otherwise re-prescribe verbatim:

- **`method-add.md`** — new method PR (links the `method-request.yml` issue if any; checklist: C++ impl + `extern "C"` wrapper, `cpp/abi/` snapshot regen, `catalog/methods/<id>.yaml` + `parity/scenarios/<id>.yaml`, tolerances calibrated, all bindings render clean via `render_api.py`, conformance suite green).
- **`method-update.md`** — behavioural / signature change PR (parameter migration notes, `catalog_version` bump if schema impacted, parity re-snapshot, dashboard delta noted).
- **`external-reference.md`** — `alternates` addition or version bump (env recipe in `parity/environments/`, `parity-snapshots.yml` run output linked, drift assessment vs previous pinned version).
- **`binding-add.md`** — new language binding (skeleton from `make new-binding LANG=...`, conformance runner, at least one profile declared, single CI matrix row added).
- **`binding-update.md`** — new idiomatic profile or wrapper change (profile YAML + Jinja template, render_api regen output committed, conformance hook for the framework).
- **`subset.md`** — new subset package (manifest in `catalog/subsets/`, render output committed, release workflow row).
- **`parity-fix.md`** — fix for a parity discrepancy (root cause analysis, before/after metrics, tolerance change justified if relaxed).
- **`PULL_REQUEST_TEMPLATE.md`** at the repo root — generic fallback for PRs that don't match any of the above (build/CI/refactor work).

PR templates are selected via the GitHub query-string convention (`?template=method-add.md`) and linked from the matching issue template's `description:` so a contributor following the request → PR flow lands on the right form automatically. Each PR template embeds an "external-impact" section that's checked against §2.10 invariants (ABI symbol snapshot, env_lock_hash freshness, no hand-written per-method per-profile code, etc.) — so the reviewer (human + Codex per existing workflow) sees the same gates the CI enforces.

### 1.1quater Concrete template schemas

The two highest-detail templates are shown below to anchor the contract. The remaining templates (`method-update`, `external-reference`, `binding-add`, `binding-update`, `subset`, `parity-fix`, `parity-discrepancy`, `bug-report`) follow the same pattern with fields adapted to their change type — all listed in full in `.github/ISSUE_TEMPLATE/` and `.github/PULL_REQUEST_TEMPLATE/` once Phases A18/A19 land.

#### Issue: `method-request.yml`

```yaml
# .github/ISSUE_TEMPLATE/method-request.yml
name: Method request
description: Request a new method to be added to n4m
title: "[Method] <short name>"
labels: [method-request, triage]
body:
  - type: markdown
    attributes:
      value: |
        Before submitting: please check `catalog/methods/` and the dashboard to confirm
        the method isn't already covered or in draft. If accepted, you'll be invited to
        submit a PR using the `method-add.md` template.

  - type: input
    id: name
    attributes: {label: Method canonical name, placeholder: "e.g. Recursive Sparse PLS"}
    validations: {required: true}

  - type: dropdown
    id: category
    attributes:
      label: Target category
      options: [pls, preprocessing, augmentation, selection, splitters, filters, diagnostics, other]
    validations: {required: true}

  - type: input
    id: publication_doi
    attributes: {label: Reference publication DOI, placeholder: "10.xxxx/xxxxx"}
    validations: {required: true}

  - type: textarea
    id: publication_meta
    attributes: {label: "Title, authors, year", placeholder: "de Jong, S. (1993). SIMPLS — ..."}
    validations: {required: true}

  - type: textarea
    id: math_summary
    attributes: {label: Math summary, description: "Either inline LaTeX or a link to the relevant section of the paper."}
    validations: {required: true}

  - type: dropdown
    id: external_reference_exists
    attributes:
      label: Does an external reference implementation exist?
      options: ["Yes — packaged in a public library", "No — paper-only (will need self-consistency block)"]
    validations: {required: true}

  - type: textarea
    id: external_reference_details
    attributes:
      label: External reference details (if any)
      description: |
        Framework (python / r / matlab / octave / julia / ...), package name, symbol,
        pinned version, invocation hint (class / pipeline / script).
      placeholder: |
        framework: python
        package: scikit-learn
        symbol: sklearn.cross_decomposition.PLSRegression
        pinned_version: 1.5.0
        invocation: class

  - type: textarea
    id: parameters
    attributes:
      label: Expected parameters
      description: List parameters with types, defaults, valid ranges, and short descriptions.

  - type: textarea
    id: inputs_outputs
    attributes:
      label: Expected inputs and outputs
      description: "Input shapes/types (e.g. X: (n_samples, n_features)) and output shapes/types."

  - type: textarea
    id: scenarios
    attributes:
      label: Suggested test scenarios / datasets
      description: |
        Real datasets (corn / beer / alpine) and/or synthetic generators with sizes and seeds.
        These will become entries in `parity/scenarios/<method_id>.yaml`.

  - type: textarea
    id: bindings_priority
    attributes:
      label: Priority bindings (optional)
      description: "Which language bindings / idiomatic profiles should expose this method first?"

  - type: textarea
    id: use_case
    attributes:
      label: Real-world use case
      description: "Why do you need this method? What problem does it solve in your domain?"
    validations: {required: true}

  - type: textarea
    id: subset_targeting
    attributes:
      label: Target subset packages (optional)
      description: "Should this method be exposed in any specific subset (pls4all, nirs4all-preprocessing, ...)?"

  - type: markdown
    attributes:
      value: |
        Once accepted by triage, submit your PR using
        **[method-add.md](../../compare/main...HEAD?template=method-add.md)**.
```

#### PR: `method-add.md`

```markdown
<!-- .github/PULL_REQUEST_TEMPLATE/method-add.md -->

## New method: <method_id>

Closes #<issue_number> (mandatory if a `method-request.yml` issue exists)

### Catalog entry

- [ ] `catalog/methods/<method_id>.yaml` created
- [ ] `catalog_version: <N>` matches the current schema
- [ ] `publication:` filled with DOI, authors, year, title
- [ ] `math.file: catalog/methods/<method_id>.math.md` created with derivation + complexity
- [ ] `parameters:` exhaustively typed and ranged
- [ ] `status:` is `production` (or `paper_only` with `self_consistency:` block populated)

<details>
<summary>Paste the full YAML here for review convenience</summary>

```yaml
# (paste catalog/methods/<method_id>.yaml)
```
</details>

### C++ implementation

- [ ] Source under `cpp/src/core/<category>/<method_id>.{cpp,hpp}`
- [ ] `extern "C"` wrapper under `cpp/src/c_api/c_api_<category>.cpp`
- [ ] Unit tests under `cpp/tests/<category>/test_<method_id>.cpp` (doctest)
- [ ] `cpp/abi/expected_symbols_{linux,macos,windows}.txt` regenerated (auto by CI)
- [ ] `docs/abi/changes_log.md` updated with the new symbols and a one-line justification

### Reference & parity (skip if `paper_only`)

- [ ] `reference.canonical` resolved to an existing or new `env_id`
- [ ] If new external framework: env recipe added under `parity/environments/<env_id>/` with Dockerfile + lock; `parity-env-build.yml` ran clean
- [ ] `parity/scenarios/<method_id>.yaml` written (≥ 3 scenarios: small / medium / one real dataset)
- [ ] Reference snapshots regenerated via `make snapshot METHOD=<method_id> REF=<ref_alias>` — paste the verdict below
- [ ] Per-`(method, reference)` tolerance calibrated in `parity.tolerances:`; rationale captured if any tolerance is looser than `1e-8 rmse_rel`
- [ ] `make parity METHOD=<method_id>` green; verdict report attached

<details>
<summary>Parity verdict output</summary>

```
(paste `make parity METHOD=<method_id>` output: per-scenario rmse_rel, max_abs, sign_agreement, verdict)
```
</details>

### Self-consistency (`paper_only` only)

- [ ] `self_consistency:` block declared in YAML (determinism / refit_idempotence / invariants)
- [ ] `make parity-paper-only METHOD=<method_id>` green; verdict attached

### Bindings

- [ ] `bindings.<lang>.raw` and `bindings.<lang>.idiomatic[]` declared in the YAML for every supported language
- [ ] `python catalog/scripts/render_api.py` run; generated wrappers committed
- [ ] No hand-written per-method per-profile code (§2.10 invariant #2 + §2.11)
- [ ] `bindings/conformance/` fixtures regenerated if the method introduces new types (rare)
- [ ] `bindings.yml` matrix green across all active languages
- [ ] Inter-binding parity (`rmse_rel < 1e-12` vs C++ raw) green

### Examples & docs

- [ ] `examples.canonical_scenario: examples/canonical/<method_id>.yaml` written
- [ ] Per-binding per-profile examples auto-generated; spot-check confirms they compile & run
- [ ] Sphinx page for the method renders; parity badge + timing mini-table embed correctly
- [ ] `CHANGELOG.md` entry added under the next unreleased section

### Reviewer checklist — §2.10 invariants

- [ ] Catalog is the only place this method's facts live (no duplication in code/doc)
- [ ] All wrappers come from `render_api.py` + per-language Jinja templates
- [ ] ABI snapshot regeneration was automatic (not hand-edited)
- [ ] Every committed snapshot carries `env_lock_hash` + `host_card_id`
- [ ] Conformance suite passes in `bindings.yml` matrix
```

The other PR templates mirror this structure, scoped to their change:

- **`external-reference.md`** — checklist focused on `alternates[]` entry, env recipe build/reuse, `make snapshot REF=<new>` verdict, cross-reference parity diff vs the previous pinned version, tolerance for the new pair.
- **`binding-add.md`** — checklist focused on `make new-binding LANG=...` skeleton output committed, `bindings/SPEC.md` conformance per-entry, first framework profile + Jinja template, `bindings.yml` matrix row, native packaging metadata, inter-binding parity verdict for the new binding.
- **`binding-update.md`** — checklist focused on new profile YAML + Jinja template, render_api regen committed, conformance hook (e.g. sklearn `check_estimator`) green, examples per the new profile.
- **`subset.md`** — checklist on `catalog/subsets/<id>.yaml`, render output, release workflow matrix row, smoke `pip install <subset>` / `R CMD INSTALL <subset>` from a fresh env.
- **`parity-fix.md`** — root cause analysis, before/after metrics, tolerance change justification if any, regression test added.
- **`PULL_REQUEST_TEMPLATE.md`** (generic) — minimal: what changed, why, how to verify, CHANGELOG entry, link to relevant invariants checklist.

The schemas above are normative for what Phase A18 and A19 ship.

### 1.1ter MATLAB / Octave policy

n4m supports the MATLAB ecosystem but **CI runs Octave** — GitHub-hosted runners do not provide MATLAB licences, and self-hosting a MATLAB runner is not cost-effective for the current project size. The `bindings/matlab/` folder ships code that targets the **intersection of MATLAB and Octave** (classdefs, MEX C ABI, struct-based APIs). Differences are documented per-symbol in `bindings/matlab/COMPAT.md`. Releases to MATLAB File Exchange happen on a periodic manual cadence by a maintainer with a MATLAB licence; CI is responsible only for the Octave-runnable surface.

### 1.2 Layered architecture

```
                        ┌──────────────────────────────────┐
                        │     C++ core (cpp/src/core/)      │
                        │   Optimized numerical methods     │
                        └────────────────┬──────────────────┘
                                         │
                                ┌────────▼────────┐
                                │  Stable C ABI    │  (libn4m.so)
                                │  cpp/include/    │
                                │  + cpp/src/c_api/│
                                └────────┬─────────┘
                                         │
              ┌──────────────────────────┼──────────────────────────┐
              │                          │                          │
        ┌─────▼──────┐            ┌──────▼──────┐            ┌──────▼──────┐
        │  Raw FFI   │            │  Raw FFI    │            │  Raw FFI    │
        │ bindings/  │            │ bindings/   │            │ bindings/   │
        │  python/   │            │  r/         │            │  matlab/    │ ...
        └─────┬──────┘            └──────┬──────┘            └──────┬──────┘
              │                          │                          │
   ┌──────────┴──────────┐    ┌─────────┴──────────┐    ┌──────────┴──────────┐
   │ Idiomatic wrappers  │    │ Idiomatic wrappers │    │ Idiomatic wrappers  │
   │ - sklearn-compat    │    │ - base R formula   │    │ - classdef factory  │
   │ - numpy/pandas      │    │ - pls-compat       │    │ - struct-based      │
   │ - SHAP-ready        │    │ - mdatools-compat  │    │ - parallel toolbox  │
   └──────────┬──────────┘    └──────────┬─────────┘    └──────────┬──────────┘
              │                          │                          │
        ┌─────▼──────────────────────────▼──────────────────────────▼─────┐
        │       Publishable packages (packaging-only subsets)              │
        │                                                                   │
        │   PyPI: nirs4all-methods (full)  · pls4all (PLS-only) · ...      │
        │   CRAN: nirs4all-methods         · pls4all                       │
        │   npm:  @nirs4all/methods        · ...                           │
        │   MATLAB FX, Octave Forge, Julia General registry, etc.           │
        └───────────────────────────────────────────────────────────────────┘
```

**Golden rule**: one single `libn4m.so` everywhere. Subset packages (`pls4all`, etc.) **are not separate binaries** — they ship the same `libn4m.so` and only re-export a subset of the API in the host language. Decision settled on 2026-05-23, against the current `render_subset.py` approach that builds smaller `libpls4all.so` artefacts (to be retired — see backlog).

### 1.3 The four pillars of validity

#### Pillar 1 — Documentation

Each method ships with a structured documentation file that contains:

| Section                          | Content                                                                                                                |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Identity                         | canonical name, aliases, category (pls / preprocessing / selection / …)                                                |
| Parameters                       | exhaustive signature (types, defaults, valid ranges, constraints)                                                      |
| Scientific reference             | source publication (title, authors, DOI, year)                                                                         |
| Mathematical explanation         | equations, algorithmic schema, complexity                                                                              |
| Reference implementation         | external lib (`sklearn.cross_decomposition.PLSRegression`, `R::pls::plsr`, …) or `nirs4all` if no external exists, or `null` if the method only lives in a paper |
| Reference traceability           | pinned version, invocation options, RNG/seed, source (CRAN, PyPI, …), validation hash                                  |
| Per-binding examples             | usage snippet in `cpp`, `python` (raw + idiomatic), `r` (raw + formula + mdatools), `matlab`, etc.                     |
| Parity status vs reference        | auto-inserted table (see pillar 2)                                                                                     |
| Inter-binding parity status       | auto-inserted table (see pillar 3)                                                                                     |
| Timing benchmark                  | auto-inserted table (see pillar 4)                                                                                     |

The structure is **machine-readable** (YAML front-matter + standardized markdown sections) so that doc pages and dashboard cells can be generated automatically.

#### Pillar 2 — Parity check vs reference

**Definition**: a *scenario* is a tuple `(method, dataset, parameters, RNG/seed, n_threads, OS, BLAS backend)`. For every scenario, we compute the output of the external reference method in its native framework, archive it with its full **provenance ID card** (library version, OS, validation hash, date), then statistically compare n4m C++ output against this snapshot.

```
   ┌──────────────────────┐         ┌──────────────────────┐
   │  Scenario S          │         │  Scenario S          │
   │  (PLS, X 100x50, k=5)│────┬────│  (PLS, X 100x50, k=5)│
   └──────────┬───────────┘    │    └──────────┬───────────┘
              ▼                │               ▼
   ┌──────────────────────┐    │    ┌──────────────────────┐
   │  sklearn 1.5.0       │    │    │  n4m C++ libn4m 1.9  │
   │  PLSRegression       │    │    │  n4m_pls_fit (BLAS)  │
   │  → output snapshot   │    │    │  → output            │
   │  + provenance JSON   │    │    └──────────┬───────────┘
   └──────────┬───────────┘    │               │
              │                │               │
              ▼                │               ▼
   ┌──────────────────────────────────────────────────────┐
   │  Comparator : RMSE, R², max_abs_diff, sign(loadings) │
   │  Tolerances per (method, reference) pair             │
   │  declared in catalog/methods/<id>.yaml               │
   └──────────────────────┬───────────────────────────────┘
                          ▼
              ┌─────────────────────┐
              │  Verdict: OK/KO     │
              │  + metrics          │
              │  archived per scen. │
              └─────────────────────┘
```

**Third-party snapshots**: when a method has multiple external implementations (sklearn, R.pls, MATLAB plsregress), we pick **one canonical reference** (declared in the catalog), BUT we also archive and benchmark the other external implementations against this canonical reference (`sklearn vs R.pls`, `R.pls vs MATLAB`). This quantifies numerical drift between external frameworks — a scientifically meaningful data point.

**Pipeline references**: some reference implementations are not a single class call but a *short pipeline* in the host framework (e.g. `StandardScaler → PLSRegression` in sklearn, or a 2-step `ropls` flow in R). This is **not** an n4m composition feature — composition stays out of n4m's scope (§1.1bis). It only means the catalog YAML's `reference.canonical.invocation` field accepts a `pipeline` form alongside the single-class form (see §2.2). The generator runs the pipeline as a black box; n4m C++ output is compared against the pipeline's final stage output.

**Methods with no external reference** (`status: paper_only`): some methods are this project's scientific differentiators (AOM-PLS, POP-PLS) or only exist in the literature. For these, `reference.canonical: null`. Parity is replaced by **self-consistency validation** declared in the YAML's `self_consistency` block: re-fit determinism (same seed → same output), idempotence (transform twice = transform once, where applicable), declared mathematical invariants (e.g. orthogonality of loadings, sign convention). The comparator treats these as first-class verdicts on the dashboard, with a distinct badge.

**Snapshot storage**:
- **In-repo** (`parity/snapshots/current/`): exactly **one** snapshot per scenario, at the currently pinned version. Small (~a few MB), git-versioned, CI is fast.
- **External** (GitHub Releases or HF Hub): historical multi-version snapshots (sklearn drift 1.4 → 1.5 → 1.6, etc.). Lazy-fetched by a dedicated monthly CI.

#### Pillar 3 — Inter-binding parity check

Simpler: the C++ raw API is the reference, the same scenario is replayed through **every binding** (raw FFI + idiomatic wrappers), and we check that the result is bit-identical (or within `rmse_rel < 1e-12`).

```
                     n4m C++ raw (binding reference)
                              │
            ┌─────────────┬───┴────┬─────────────┐
            ▼             ▼        ▼             ▼
       py raw FFI    py sklearn   R raw      R formula
            │             │        │             │
            └─────────────┴───┬────┴─────────────┘
                              ▼
                  Inter-binding comparator
                  (tolerance: rmse_rel < 1e-12)
```

#### Pillar 4 — Timing benchmarks

Same scenarios, but we measure execution time. Strictly controlled conditions (threads, BLAS backend, dataset, params). Measurements use warmup + adaptive run count (already implemented in `benchmarks/cross_binding/orchestrator.py`).

Results are aggregated by `(method, binding, dataset_size, n_threads, BLAS backend, OS, CPU)`.

### 1.4 Aggregated outputs

All this data converges into two auto-generated artefacts:

1. **Per-method documentation pages** (Sphinx): the static doc embeds parity verdicts and timing tables for the current version.
2. **Public SPA dashboard** (gh-pages): an interactive interface consuming a canonical JSON, allowing filtering by method/binding/reference/dataset/threads, scenario drill-down, multi-version comparisons.

---

## 2. Design — proposal

### 2.1 Final repo layout

```
nirs4all-methods/
├── catalog/                     # ★ Single source of truth
│   ├── methods/                 # One YAML per method
│   │   ├── pls.yaml
│   │   ├── pls_simpls.yaml
│   │   ├── aom_pls.yaml
│   │   └── ...
│   ├── subsets/                 # Sub-package manifests
│   │   ├── pls4all.yaml         # PLS-only package
│   │   └── nirs4all_methods.yaml  # full
│   ├── scripts/
│   │   ├── validate.py          # Lints catalog (all refs resolved, etc.)
│   │   └── render_api.py        # Generates __init__.py / NAMESPACE / +n4m classdefs
│   └── README.md                # Catalog contract
│
├── cpp/                         # ★ Engine
│   ├── include/n4m/             # Public headers, extern "C" ONLY
│   │   ├── n4m.h                # Umbrella
│   │   ├── n4m_version.h        # Version source of truth
│   │   ├── n4m_types.h
│   │   ├── n4m_pls.h
│   │   ├── n4m_preprocessing.h
│   │   ├── n4m_selection.h
│   │   └── ...                  # One header per category
│   ├── src/
│   │   ├── core/                # C++17 implementations (preserved from current repo)
│   │   │   ├── pls/
│   │   │   ├── preprocessing/
│   │   │   ├── augmentation/
│   │   │   ├── selection/
│   │   │   ├── splitters/
│   │   │   ├── filters/
│   │   │   ├── diagnostics/
│   │   │   └── common/          # linalg, RNG, IO
│   │   └── c_api/               # extern "C" wrappers, organized by category
│   │       ├── c_api_pls.cpp
│   │       ├── c_api_preprocessing.cpp
│   │       └── ...
│   ├── cli/                     # n4m_cli — version/abi-info/selfcheck
│   ├── tests/                   # doctest (one file per method + ABI tests)
│   └── abi/                     # Symbol snapshots + changes log
│       ├── expected_symbols_linux.txt
│       ├── expected_symbols_macos.txt
│       ├── expected_symbols_windows.txt
│       └── changes_log.md
│
├── bindings/                    # One folder per language, never duplicated
│   ├── python/
│   │   ├── src/n4m/             # Raw FFI + idiomatic wrappers
│   │   │   ├── _ffi.py
│   │   │   ├── raw.py
│   │   │   ├── sklearn/         # sklearn-compatible API
│   │   │   └── _api.py
│   │   ├── tests/
│   │   ├── pyproject.toml       # n4m package (full)
│   │   └── packaging/
│   │       ├── pls4all/         # pls4all wheel: re-exports a subset of n4m
│   │       │   └── pyproject.toml
│   │       └── nirs4all/        # idem
│   ├── r/
│   │   ├── n4m/                 # CRAN n4m package (full)
│   │   └── pls4all/             # CRAN pls4all (PLS-only re-export)
│   ├── matlab/
│   ├── js/
│   ├── julia/
│   └── _archive/                # Frozen PoCs (go, rust, dotnet, lua, nim, ruby)
│
├── parity/                      # ★ Scientific validation
│   ├── scenarios/               # Scenario definitions (one YAML per method)
│   ├── snapshots/
│   │   ├── current/             # Reference snapshots (in-repo, pinned)
│   │   └── manifest.json        # Hashes + provenance
│   ├── generators/              # Generates reference snapshots
│   │   ├── python/              # sklearn, ikpls, diPLSlib, etc.
│   │   ├── r/                   # pls, ropls, mixOmics, plsVarSel
│   │   └── matlab/              # plsregress, libPLS
│   ├── comparator/              # Compares n4m output vs snapshot
│   │   └── tolerances.yaml      # Tolerances per (method, reference) pair
│   └── schema/                  # JSON schemas for snapshots + provenance
│
├── benchmarks/                  # ★ Timing — contains NO parity logic
│   ├── scenarios/               # Reuses parity/scenarios + adds size sweeps
│   ├── runners/                 # One runner per binding
│   ├── orchestrator.py          # Drives runs: threads, warmup, adaptive n_runs
│   └── results/                 # CSV/JSON results, versioned
│
├── docs/                        # Sphinx
│   ├── source/
│   │   ├── methods/             # Auto-generated from catalog/ + parity/ + benchmarks/
│   │   ├── architecture/
│   │   ├── parity/
│   │   ├── bindings/
│   │   └── dev/
│   └── _extras/                 # Build scripts (per-method page generation)
│
├── dashboard/                   # ★ gh-pages SPA
│   ├── src/                     # Vue/Svelte/React — to be decided
│   ├── public/
│   └── package.json
│
├── ci/                          # GitHub Actions workflows
│   └── (see .github/workflows/)
│
├── scripts/
│   ├── bump_version.sh
│   └── (other tooling)
│
├── CMakeLists.txt
├── CMakePresets.json
├── README.md
├── CONTRIBUTING.md
└── LICENSE
```

### 2.2 Catalog — single source of truth

Today the catalog is scattered across:
- `benchmarks/parity_timing/registry.py` (~10 000 LOC) — mixes logic and data
- `catalog/methods.yaml` + `methods.discovered.yaml` (~155 KB, identical)
- `parity/tolerances.md` (manual markdown)
- `roadmap/phase-*.md` (free-form text)

**Proposal**: one YAML file **per method** in `catalog/methods/<method_id>.yaml`. Single format:

```yaml
# catalog/methods/pls_simpls.yaml
catalog_version: 1            # schema version (bump on incompatible field changes)
id: pls_simpls
name: PLS SIMPLS
category: pls
status: production            # draft | production | paper_only | deprecated
abi_symbols:
  - n4m_pls_simpls_fit
  - n4m_pls_simpls_predict
publication:
  title: SIMPLS — an alternative approach to partial least squares regression
  authors: [de Jong, S.]
  year: 1993
  doi: 10.1016/0169-7439(93)85002-X
math:
  file: catalog/methods/pls_simpls.math.md
parameters:
  n_components:
    type: int
    default: null
    range: [1, min(n_samples, n_features)]
    description: Number of latent components.
  scale:
    type: bool
    default: false
reference:
  canonical:
    framework: python
    package: scikit-learn
    symbol: sklearn.cross_decomposition.PLSRegression
    pinned_version: 1.5.0
    env_id: env-py-sklearn-1.5     # ↳ environment recipe in parity/environments/
    invocation:
      kind: class                  # class | pipeline | script
      class: PLSRegression
      kwargs: {scale: false}
  alternates:
    - {framework: r, package: pls, symbol: plsr, pinned_version: 2.8-3, env_id: env-r-pls-2.8}
    - {framework: octave, package: statistics, symbol: plsregress, pinned_version: 6.4.0, env_id: env-octave-6.4}
    - {framework: python, package: ikpls, symbol: NpalsPLS, pinned_version: 1.2.0, env_id: env-py-ikpls-1.2}
parity:
  scenarios: parity/scenarios/pls_simpls.yaml
  tolerances:
    sklearn:        {rmse_rel: 1e-10, max_abs: 1e-12}
    R.pls:          {rmse_rel: 1e-8,  max_abs: 1e-10}
    octave.plsregress: {rmse_rel: 1e-8, max_abs: 1e-10}
    ikpls:          {rmse_rel: 1e-10, max_abs: 1e-12}
bindings:
  python:
    raw: n4m.raw.pls_simpls_fit
    idiomatic:
      - {profile: python_sklearn, symbol: n4m.sklearn.PLSRegression}
      - {profile: python_keras,   symbol: n4m.keras.PLSLayer}
  r:
    raw: "n4m::pls_simpls_fit"
    idiomatic:
      - {profile: r_formula,         symbol: "n4m::pls"}
      - {profile: r_pls_compat,      symbol: "n4m::plsr"}
      - {profile: r_mdatools_compat, symbol: "n4m::pls.matrix"}
  matlab:                              # tested via Octave in CI (§1.1ter)
    raw: "n4m_mex('pls_simpls_fit', X, y)"
    idiomatic:
      - {profile: matlab_classdef, symbol: "n4m.pls.SimplsRegression"}
      - {profile: matlab_factory,  symbol: "n4m.fit('pls_simpls', X, y)"}
  julia:
    raw: "N4M.pls_simpls_fit(X, y)"
    idiomatic:
      - {profile: julia_mlj, symbol: "N4M.MLJ.PLSRegressor"}
  js:
    raw: "n4m.cwrap('pls_simpls_fit')(X, y)"
    idiomatic:
      - {profile: js_tfjs, symbol: "n4m.tfjs.PLSLayer"}
  # rust, go, dotnet, ruby, swift, kotlin: raw-only at v1; idiomatic wrappers added on request
examples:
  # Auto-generated per (binding, idiomatic-profile) by render_api from a single canonical scenario,
  # so a method ships with one logical example per wrapper without 30 hand-maintained snippets.
  canonical_scenario: examples/canonical/pls_simpls.yaml
  cpp: docs/source/methods/_examples/pls_simpls.cpp   # cpp written by hand (no profile)
```

**Pipeline reference example** — when the canonical reference is necessarily a multi-step composition in the host framework:

```yaml
reference:
  canonical:
    framework: python
    package: scikit-learn
    env_id: env-py-sklearn-1.5
    invocation:
      kind: pipeline
      steps:
        - {class: sklearn.preprocessing.StandardScaler}
        - {class: sklearn.cross_decomposition.PLSRegression, kwargs: {n_components: 5}}
      compare_against: final     # final | named:<step_id>
```

**Paper-only example** — when no external implementation exists (e.g. AOM-PLS):

```yaml
status: paper_only
reference:
  canonical: null
self_consistency:
  - {check: determinism, seed: 42, tolerance: 0}        # same seed → bit-identical
  - {check: refit_idempotence, tolerance: 1e-14}
  - {check: invariant, expression: "norm(loadings, axis=1) == 1", tolerance: 1e-12}
parity:
  tolerances:
    self: {rmse_rel: 1e-12, max_abs: 1e-14}             # used by inter-binding parity
```

Every downstream artefact (Sphinx page, dashboard cell, parity scenario, ABI symbol validation, tolerance table) is derived from this file. `benchmarks/parity_timing/registry.py` is deleted.

### 2.3 Scenarios and snapshots — formats

**`parity/scenarios/pls_simpls.yaml`**:

```yaml
method_id: pls_simpls
scenarios:
  - id: small_random_seed42
    dataset:
      generator: synthetic_regression
      n_samples: 100
      n_features: 50
      seed: 42
    params:
      n_components: 5
      scale: false
    expected_in_snapshot: [coefficients, predictions, scores, loadings]
  - id: medium_random_seed42
    dataset: {generator: synthetic_regression, n_samples: 1000, n_features: 200, seed: 42}
    params:  {n_components: 10, scale: false}
    expected_in_snapshot: [coefficients, predictions, scores, loadings]
  - id: corn_dataset_classic
    dataset: {ref: corn}
    params:  {n_components: 8, scale: true}
    expected_in_snapshot: [coefficients, predictions, scores, loadings]
```

**`parity/snapshots/current/pls_simpls/small_random_seed42.json`**:

```json
{
  "method_id": "pls_simpls",
  "scenario_id": "small_random_seed42",
  "reference": {
    "framework": "python",
    "package": "scikit-learn",
    "version": "1.5.0",
    "symbol": "sklearn.cross_decomposition.PLSRegression",
    "invocation": {"class": "PLSRegression", "kwargs": {"n_components": 5, "scale": false}}
  },
  "provenance": {
    "generated_at": "2026-05-23T14:32:11Z",
    "host_card_id": "bench-01-2026",
    "host": {
      "os": "linux-6.6.87",
      "cpu": "AMD Ryzen 9 7950X",
      "ram_gb": 64,
      "blas": "openblas-0.3.27",
      "n_threads_pinned": 1
    },
    "env_id": "env-py-sklearn-1.5",
    "env_lock_hash": "sha256:...",
    "generator_version": "0.1.0",
    "generator_git_rev": "abc123",
    "rng_seed": 42
  },
  "outputs": {
    "coefficients": [[...]],
    "predictions": [...],
    "scores": [[...]],
    "loadings": [[...]]
  },
  "hash": "sha256:..."
}
```

### 2.4 Unified comparator

One single Python comparator (reused by C++ tests via JSON IO) computes, for every `(n4m_output, reference_snapshot)` pair:
- `rmse_rel`, `rmse_abs`, `max_abs_diff`, `r2`, `sign_agreement` (for loadings),
- OK/KO verdict using tolerances declared in `catalog/methods/<id>.yaml`,
- metrics archived in `parity/results/<method_id>/<scenario_id>.json`.

For `paper_only` methods, the comparator instead runs the checks declared in the `self_consistency` block (determinism, idempotence, mathematical invariants — cf. §1.3 Pillar 2 and §2.2). Same output format, distinct verdict category surfaced on the dashboard.

Parity results are **git-versioned** (negligible size) so we keep history.

### 2.5 Measurement discipline (timing benchmarks)

Python is the orchestrator, **never the timer**. Three invariants make timing measurements trustworthy:

1. **The clock lives in the target language**, not around the FFI call. Each runner (`benchmarks/runners/bench_cpp.cpp`, `bench_r.R`, `bench_octave.m`, `bench_python.py`, `bench_js.mjs`, …) measures with its own high-resolution clock (`std::chrono::steady_clock`, `bench::mark()`, `tic/toc`, `time.perf_counter_ns()`, `performance.now()`) and emits a JSON `{"elapsed_ns": ..., "n_repeats": ..., "warmup_ns": ...}`. The orchestrator only reads and aggregates. Wrapping subprocess launches in a Python timer would add 50–500 ms per run from fork+import — fatal for sub-100 ms methods.
2. **Loop inside the runner for fast methods** (< 1 ms). The orchestrator passes `n_repeats=N` to the runner, the runner runs the hot path N times, returns median + p95 + std. Pure measurement code stays in the fast language; orchestrator overhead is paid once.
3. **Controlled environment via env vars + CPU pinning**, set by the orchestrator before spawning the runner: `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `RAYON_NUM_THREADS`; `taskset -c <core>` on Linux. The runner reports the values it saw back into the JSON for cross-checking.

**Parallelization policy** — parity and timing have opposite rules:
- **Parity runs** are embarrassingly parallel and have no inter-run interaction. Default to `ProcessPoolExecutor(max_workers=os.cpu_count())`. Linear speedup.
- **Timing runs are serial by default on a single machine.** Parallel runs contend for CPU/cache/BLAS and pollute measurements (a single-thread method sharing L3 with another run reports 20–40 % slower). Opt-in mode `--parallel-isolated` pins each process to a disjoint core set and forces `OMP_NUM_THREADS=1` (accepts residual cache noise) — the maintainer decides when to use it. The horizontal scaling option (matrix CI across machines) is deferred until the project has more than one canonical host card (§2.6.2).

### 2.6 Reproducibility & environments

A snapshot is scientifically meaningful only if the next contributor can regenerate it. n4m treats reproducibility as a first-class concern on three layers:

#### 2.6.1 Reference environments — Docker images per (framework, version)

Each `env_id` referenced from the catalog corresponds to a recipe under `parity/environments/<env_id>/`:

```
parity/environments/
├── env-py-sklearn-1.5/
│   ├── Dockerfile           # FROM python:3.12-slim, pinned wheel set
│   ├── requirements-lock.txt
│   └── README.md
├── env-py-ikpls-1.2/
├── env-r-pls-2.8/
│   ├── Dockerfile           # FROM r-base:4.4.0, renv lockfile
│   └── renv.lock
├── env-r-ropls-1.36/
├── env-octave-6.4/
│   └── Dockerfile           # FROM octave/octave:9.2, statistics pkg pinned
└── env-py-n4m-runtime/      # used by parity-fast in CI (no externals)
    └── Dockerfile
```

Rules:
- Each Dockerfile is built once, tagged `n4m-env:<env_id>-<short_lock_hash>`, pushed to GHCR.
- Snapshot generators (`parity/generators/python/<framework>.py`) run **inside the matching container**: `docker run --rm n4m-env:env-py-sklearn-1.5 -v $PWD:/work generator.py …`.
- The container's lock hash goes into `provenance.env_lock_hash`. If the recipe changes, snapshots must be regenerated — checked by `parity-fast` (refusal if `env_lock_hash` in committed snapshot differs from the current recipe hash).
- **For local dev**: contributors run the outer dev shell described in §2.12 (devcontainer + Docker Compose) and call into these inner parity images via the Docker socket. A `parity/environments/dev/` conda recipe is shipped as a non-Docker fallback covering only the Python/R generator surface — not used in CI; only the per-`env_id` Docker images are authoritative.

#### 2.6.2 Benchmark host policy — host cards

Timing measurements are valid **only for the host they were taken on**. n4m formalizes this:

```
benchmarks/hosts/
├── bench-01-2026.yaml       # the maintainer's machine (current canonical)
└── README.md                # how to add a new host card
```

A host card YAML:
```yaml
id: bench-01-2026
purpose: canonical timing host for n4m benchmark matrix
maintainer: gbeurier
cpu: {model: "AMD Ryzen 9 7950X", cores: 16, base_ghz: 4.5, smt: false}
ram: {gb: 64, type: DDR5-5200}
os: {distro: ubuntu, version: "24.04", kernel: "6.6.87"}
blas: {impl: openblas, version: "0.3.27", threads_pinned: 1}
created_at: 2026-05-23
deprecated_at: null
```

Rules:
- Every benchmark result JSON carries the `host_card_id` (cf. §2.3).
- The dashboard surfaces the current host card prominently. Comparing timings across host cards is disabled in the UI (or shown with a "different host" warning).
- **Changing the canonical host** = new host card (`bench-02-YYYY.yaml`), parallel publication for one cycle, then deprecation of the old card. The drift between cards is published as a one-off benchmark note.
- Until the project grows, **timing is maintained on one machine** (current: `bench-01-2026`). This is documented as a known limitation, not a defect. Adding a second host card is straightforward (just a YAML + a runner with Docker) but requires a maintainer to commit to the cadence.

#### 2.6.3 Manual execution model — `Makefile` and `workflow_dispatch`

**Heavy workloads are never scheduled automatically.** Parity snapshot regeneration and timing benchmarks run only when the maintainer launches them, either via a local command or via a `workflow_dispatch` GitHub Action with explicit scenario parameters. Two reasons drive this:

1. Reference snapshots and timing measurements are physically tied to one machine and to pinned external library versions (cf. §2.6.1, §2.6.2). They are scientific outputs, not health checks — re-running them blindly produces stale artefacts when neither inputs nor environments have changed.
2. The current project has one maintainer and one host card. Cron schedules at that scale produce burnt-out CI runners and notification fatigue without buying reproducibility.

A repo-wide `Makefile` is the **primary** entry point for the maintainer:

```make
make env-build                                # builds all parity Docker environment images locally
make env-build ENV=env-py-sklearn-1.5         # rebuilds one image

make snapshot METHOD=pls_simpls REF=sklearn   # regenerates one reference snapshot via Docker
make snapshot METHOD=all REF=all              # regenerates everything (long)

make parity METHOD=pls_simpls                 # runs C++ vs in-repo snapshot for one method
make parity METHOD=all                        # full parity sweep (all methods, all references)
make parity-paper-only                         # self-consistency checks for paper_only methods

make bench METHOD=pls_simpls                  # runs the timing matrix for one method on the current host card
make bench METHOD=all THREADS="1 4 8"         # full timing sweep with thread variations

make dashboard-data                            # rebuilds dashboard.json from parity/results/ + benchmarks/results/
make dashboard-serve                           # local Vite dev server for the SPA
```

The same operations are also exposed as **manually-dispatchable GitHub Actions** (`workflow_dispatch` with scenario / method / reference inputs) for the cases where running on a clean GitHub-hosted environment is preferable. Neither path runs on a schedule.

### 2.7 SPA Dashboard

Chosen stack:
- **Framework**: Svelte (settled 2026-05-23 — lighter than Vue/React for this use case, and the dashboard JSON-to-DOM mapping is straightforward enough that the heavier frameworks add no value).
- **Build**: Vite.
- **Data source**: a single `dashboard.json` produced by `make dashboard-data` (locally) or by the dispatched workflow that wraps the same script. Committed to `main` by the maintainer alongside the runs that produced it, then deployed to `gh-pages`.
- **Views**:
  1. **Matrix**: methods × bindings table, parity badges (green/yellow/red), median timing at 100×50.
  2. **Method drill-down**: per-method timing curves vs size (lin and log), multi-binding comparison, parity table, reference snapshot metadata.
  3. **Drift**: historical curve of parity verdicts over the last n versions of sklearn/R.pls/etc. (consumes external snapshots lazy-fetched from GH Releases).
  4. **Catalog**: navigation by category, status, fuzzy search.

### 2.8 CI/CD

Two tiers, with a hard split between **mechanical gates** (auto-triggered on PR / push / tag — fast, deterministic, no external dependencies beyond the toolchain) and **scientific runs** (manually dispatched by the maintainer when there's a reason — slow, depend on external libraries and a canonical host).

#### 2.8.1 Auto-triggered workflows (mechanical gates)

```
.github/workflows/
├── cpp.yml                  # PR — ctest on 11 platforms (existing — keep)
├── abi-check.yml            # PR — symbol diff vs cpp/abi/expected_*.txt (existing)
├── catalog-validate.yml     # PR — lint catalog/methods/*.yaml (HARD blocker)
├── parity-env-build.yml     # push to parity/environments/** — rebuild + push Docker images to GHCR
├── bindings-py.yml          # PR — bindings/python/ unit tests (no parity, no externals)
├── bindings-r.yml           # PR — bindings/r/ unit tests
├── bindings-octave.yml      # PR — bindings/matlab/ tested via Octave (no MATLAB on CI)
├── bindings-js.yml          # PR — bindings/js/ tests (Emscripten)
├── version-sync.yml         # PR — header + manifest version sync
├── docs.yml                 # PR — Sphinx build (no deploy on PR); push to main → deploy
├── release-py.yml           # on tag — n4m + pls4all + … wheels to PyPI (trusted publisher)
└── release-r.yml            # on tag — CRAN tarballs (manual submission)
```

#### 2.8.2 Manually-dispatched workflows (`workflow_dispatch` only)

```
.github/workflows/
├── parity-snapshots.yml     # Regenerate reference snapshots. Inputs: method, reference, env_id
├── parity-run.yml           # Run parity verdicts (C++ vs in-repo snapshots). Inputs: method scope
├── parity-paper-only.yml    # Self-consistency checks for paper_only methods. Inputs: method scope
├── benchmarks.yml           # Timing matrix on the canonical host card. Inputs: method, threads, sizes
└── dashboard-publish.yml    # Build SPA + push to gh-pages. Inputs: which dashboard.json to use
```

Every dispatched workflow takes explicit scenario inputs (`method`, `reference`, `host_card_id`, `n_threads`, …). None of them run on `schedule:`. The maintainer launches them when something changed that warrants regeneration (new external lib version pinned, new method merged, new host card commissioned, etc.) — never as a heartbeat.

The same operations are mirrored in the `Makefile` (§2.6.3) for local execution; the GHA path is for cases where a clean ephemeral runner is preferable to the maintainer's workstation.

### 2.9 Versioning

Six independent axes, already partially in place in `cpp/include/n4m/n4m_version.h`:

| Axis                       | Symbol                                                  | Cadence                                                                            |
| -------------------------- | ------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| ABI                        | `N4M_ABI_VERSION_*`                                     | Bumps only when the C surface changes                                              |
| Project (libn4m)           | `N4M_PROJECT_VERSION_*`                                 | Application semver                                                                 |
| Packages                   | `n4m/pyproject.toml`, etc.                              | Follows the project; local revs allowed                                            |
| Method catalog (per file)  | `version` in the YAML                                   | Bumps when the method signature changes                                            |
| Catalog **schema**         | `catalog_version` in every YAML + `catalog/schema/method.json` | Bumps on incompatible field changes; `catalog/scripts/migrate_schema.py` rewrites every YAML |
| Reference snapshot         | `provenance.generator_version` + `env_lock_hash`        | Bumps if format changes; bumps automatically with the Docker recipe hash           |
| Host card                  | `benchmarks/hosts/<id>.yaml` id                         | New id on hardware/OS/BLAS change; deprecation flow in §2.6.2                       |

### 2.10 Extensibility model — what's automated, what's manual

Adding things to n4m has a deliberately uneven cost — the design optimises for the two most common operations (adding a method, adding an idiomatic wrapper) and accepts higher bootstrap cost for adding a brand-new language. Target scale at mid-term: **~10 languages × ~2 idiomatic wrappers each = ~20-30 wrappers total**. The numbers below assume the §2.11 binding scaling infrastructure is in place.

| Operation                                          | Cost                                  | What's auto-generated                                                                                              | What's manual                                                                                          |
| -------------------------------------------------- | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------ |
| Add a method (reference exists externally)         | 1 PR (~½ day for an internal contributor) | All binding re-exports across **all** languages and idiomatic wrappers, Sphinx page, dashboard cell, parity verdict, examples per wrapper | C++ impl + extern "C" wrapper, one YAML, one scenarios YAML, tolerances (first-pass calibration)        |
| Add a method (`paper_only`)                        | 1 PR (~½ day)                         | Same as above + paper-only badge on dashboard                                                                      | C++ impl + extern "C" wrapper, one YAML with `self_consistency` block, scenarios YAML                  |
| Add an external reference for an existing method   | Tiny PR (~1 h)                        | Cross-reference parity automatically wired                                                                          | Add `alternates` entry in the YAML, possibly a new env recipe, tolerance for the new pair               |
| Add an external framework (whole new generator)    | ~1 day                                | Snapshot regeneration runs via Docker container                                                                     | One file in `parity/generators/<framework>/`, one Docker recipe in `parity/environments/<env_id>/`     |
| Add an idiomatic wrapper for an existing binding   | ~½–1 day                              | Per-method wrapper code across all methods, examples, conformance test fixtures                                     | One framework profile YAML in `bindings/profiles/<profile_id>.yaml` + one Jinja template               |
| Add a brand-new language binding                   | ~1-2 days (with the §2.11 skeleton)   | FFI scaffolding, project layout, conformance test runner, CI workflow row, `Makefile` entries — all from a single `make new-binding LANG=...` | Language-specific FFI quirks, one raw runner file, one example raw idiomatic profile to confirm the contract |
| Add a subset package                               | Trivial (~1 h)                        | `pyproject.toml`, `__init__.py` re-exports, R `NAMESPACE`                                                          | One YAML in `catalog/subsets/`                                                                          |
| Add a dashboard view                               | ~½ day                                | Reads existing `dashboard.json`                                                                                    | Svelte component, possibly new fields in the dashboard JSON schema                                      |
| Add a benchmark host                               | ~½ day                                | Dashboard surfaces the new card, runs land in the matrix                                                            | YAML in `benchmarks/hosts/`, self-hosted runner registration                                            |

**Out of scope by design** (re-iterating §1.1bis):

- No plugin API, no `register_method()` at runtime, no user-side extensibility.
- No method composition / pipeline DSL in n4m itself (composition stays in the host language: sklearn `Pipeline`, R formulas, MATLAB scripts).
- Pipelines appear only as a YAML-declared form of an external reference (§2.2), strictly to handle the few cases where the canonical reference is itself a multi-step composition in its native framework.

**Invariants the design depends on** — violating any of them undoes the extensibility properties above:

1. **Catalog is the single source of truth.** No fact about a method may be duplicated between YAML and code/doc. `catalog-validate.yml` is a hard CI blocker.
2. **render_api.py uses per-language template files from day one** (`catalog/scripts/templates/python.j2`, `r.j2`, `octave.j2`, …). One monolithic Python file that string-formats N languages becomes unmaintainable at ~5 languages.
3. **ABI symbol snapshots are regenerated automatically** by a CI job that builds on Linux/macOS/Windows and commits the snapshot to the PR branch — a developer should never have to touch three files by hand.
4. **Snapshots and environments are co-versioned via `env_lock_hash`.** A snapshot that doesn't carry its environment's lock hash is invalid and refused by `parity-run`.
5. **Every binding implements the §2.11 binding spec and passes the conformance test suite.** No PR is merged that adds an in-tree binding which skips the conformance run.

### 2.11 Binding scaling architecture

The target scale (10 languages × ~2 idiomatic wrappers each) means hand-crafted FFI plumbing per language and hand-crafted per-method wrappers per idiomatic style would explode into thousands of files. The design pushes everything possible into **four layers of abstraction** so adding a binding becomes a small, mostly declarative change.

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Layer 1: bindings/SPEC.md                                                │
│  The binding contract — every binding must implement these N entry        │
│  points (matrix marshalling, status enum, error string, version probe).   │
│  No exceptions. Validated by the conformance suite.                        │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Layer 2: bindings/conformance/                                           │
│  Cross-language conformance fixtures (JSON: input matrices, expected     │
│  outputs, error scenarios). Every binding ships a runner that ingests    │
│  these and asserts pass/fail. CI matrix runs the same suite per binding.  │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Layer 3: bindings/profiles/<profile_id>.yaml                             │
│  Framework profiles. Each describes a single idiomatic convention        │
│  (sklearn / Keras / R-formula / MLJ.jl / TensorFlow.js / linfa / ...).   │
│  Contains: base class, required methods, naming conventions, parameter   │
│  name mapping, Jinja template path, conformance hooks.                    │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Layer 4: render_api.py + per-profile Jinja templates                     │
│  For each (method × profile) pair declared in catalog/methods/*.yaml,    │
│  emit the wrapper file by combining: the method's catalog YAML + the     │
│  profile YAML + the profile's Jinja template. Zero hand-written per-     │
│  method per-binding code.                                                  │
└──────────────────────────────────────────────────────────────────────────┘
```

#### 2.11.1 Binding spec (`bindings/SPEC.md`)

A short, normative document. Every in-tree binding must implement:

- **Library loading** — locate and load `libn4m.{so,dll,dylib}` (or call into a JS Module instance for WASM).
- **ABI version probe** — call `n4m_check_abi_compatibility(header_major, header_minor)` before any other call.
- **Matrix marshalling** — accept the language's native matrix type (NumPy array, R matrix, MATLAB array, Julia Array, JS TypedArray, Rust ndarray, Go gonum/mat) and construct `n4m_matrix_view_t` with explicit `row_stride` / `col_stride`. **Never** force a copy.
- **Status / error handling** — read `n4m_status_t` from every fallible call; on non-OK, copy the error string out of the context-owned buffer before any other call on the same context.
- **Memory ownership** — never free a buffer the core allocated; never expect the core to free a buffer the binding allocated. Two output APIs (caller-allocated, core-allocated via `n4m_array_free`).
- **Symbol enumeration** — at startup, the binding queries the catalog (compiled-in YAML or a runtime descriptor) to discover which methods this `libn4m` build exposes, then emits the language-native error if a method is requested but missing.

A binding is "spec-compliant" when its conformance runner passes all `bindings/conformance/` fixtures. Non-compliance is a CI hard fail.

#### 2.11.2 Conformance test suite (`bindings/conformance/`)

A language-agnostic set of JSON fixtures + assertions. Every binding ships a thin **runner** (~100-200 LOC) that:

1. Reads a fixture JSON (input matrices, method id, parameters, expected output, optional expected error).
2. Calls the appropriate raw-FFI entry point in its language.
3. Compares output against expected (within `rmse_rel < 1e-12`).
4. Emits a JSON verdict.

The CI matrix (`bindings.yml` — single workflow, matrix over languages) runs all binding runners against all fixtures. Adding a new binding means adding one row to the matrix and shipping a runner.

#### 2.11.3 Framework profiles (`bindings/profiles/<profile_id>.yaml`)

A profile is a declarative description of one idiomatic convention. Example:

```yaml
# bindings/profiles/python_sklearn.yaml
id: python_sklearn
language: python
package_dep: "scikit-learn >= 1.0"
description: scikit-learn BaseEstimator-compliant wrappers
api_contract:
  base_class: sklearn.base.BaseEstimator
  mixins:
    regressor:   sklearn.base.RegressorMixin
    classifier:  sklearn.base.ClassifierMixin
    transformer: sklearn.base.TransformerMixin
  required_methods: [fit, predict, get_params, set_params]
  conformance_hook: sklearn.utils.estimator_checks.check_estimator
parameter_mapping:
  default_strategy: snake_case_pass_through
  overrides:
    # Map n4m-native names → sklearn convention when they differ
    ncomp: n_components
naming:
  class_suffix_by_role:
    regressor:  Regression
    classifier: Classifier
    transformer: Transformer
template: bindings/profiles/templates/python_sklearn.j2
example_template: bindings/profiles/templates/python_sklearn.example.j2
```

A few anticipated profiles (non-exhaustive — list grows as bindings land):

| Language       | Anticipated profiles                                                       |
| -------------- | -------------------------------------------------------------------------- |
| Python         | `python_sklearn`, `python_keras`, `python_torch`, `python_pandas_xform`    |
| R              | `r_formula`, `r_pls_compat`, `r_mdatools_compat`, `r_tidymodels`           |
| MATLAB/Octave  | `matlab_classdef`, `matlab_factory`, `matlab_struct_functional`            |
| Julia          | `julia_mlj`, `julia_dataframes`                                            |
| JS/TS          | `js_raw`, `js_tfjs`                                                        |
| Rust           | `rust_raw`, `rust_linfa`                                                   |
| Go             | `go_raw`, `go_gonum`                                                       |
| .NET           | `dotnet_raw`, `dotnet_mlnet`                                               |
| Ruby           | `ruby_raw`                                                                 |
| Swift          | `swift_raw`, `swift_coreml`                                                |
| Kotlin / JNI   | `kotlin_jni`, `kotlin_smile`                                               |

Adding a profile = one YAML + one Jinja template + (optional) one conformance hook. Once the profile is in place, **every existing catalog method automatically gets a wrapper for it** at the next `render_api.py` run, with no per-method human input beyond declaring `{profile: <id>, symbol: <symbol>}` in the method's `bindings.<lang>.idiomatic[]` list.

#### 2.11.4 Skeleton generator for new bindings (`make new-binding`)

```bash
make new-binding LANG=rust    # creates bindings/rust/ from bindings/skeletons/rust/
```

Each skeleton ships:
- `Cargo.toml` / `pyproject.toml` / `package.json` / `Package.swift` / `DESCRIPTION` / etc. — language-native project metadata, pre-filled.
- `src/ffi.*` — FFI loader + matrix marshalling + status handling, implementing the `bindings/SPEC.md` contract.
- `src/raw.*` — auto-generated raw API surface (from catalog).
- `tests/conformance/` — runner that ingests `bindings/conformance/` fixtures.
- A CI matrix row stub for `.github/workflows/bindings.yml`.
- A `Makefile` block declaring `test`, `build`, `package` targets.

Result: a new language binding goes from "1-2 weeks of bootstrap from scratch" (where we are today) to "1-2 days" of language-specific FFI work, the rest being mechanical adaptation of the skeleton.

#### 2.11.5 Single CI matrix workflow

```yaml
# .github/workflows/bindings.yml
jobs:
  bindings:
    strategy:
      matrix:
        binding:
          - python
          - r
          - octave
          - julia
          - js
          - rust
          - go
          - dotnet
          - ruby
          - swift
          - kotlin
    runs-on: ${{ matrix.binding.runs_on || 'ubuntu-latest' }}
    steps:
      - uses: actions/checkout@v4
      - name: Setup ${{ matrix.binding }} toolchain
        uses: ./.github/actions/setup-${{ matrix.binding }}
      - name: Build binding
        run: make -C bindings/${{ matrix.binding }} build
      - name: Run conformance suite
        run: make -C bindings/${{ matrix.binding }} conformance
      - name: Run idiomatic-profile unit tests
        run: make -C bindings/${{ matrix.binding }} test
```

One workflow, N matrix rows. Adding a binding adds one matrix entry and one `./.github/actions/setup-<lang>` action. No per-language workflow file to maintain.

#### 2.11.6 Versioning and release matrix

Each language has its own package versioning (`pyproject.toml`, `DESCRIPTION`, `Cargo.toml`, …) but **all in-tree bindings track the n4m project version** with a `>=X.Y.0,<X.(Y+1).0` constraint on `libn4m`. The release workflow (`release-bindings.yml`, manual `workflow_dispatch`) loops over the same matrix as the test workflow and publishes each binding to its native registry (PyPI, CRAN, npm, Maven Central, NuGet, RubyGems, crates.io, Julia General, Swift Package Index).

### 2.12 Local development environment

n4m's developer toolchain is unusually broad: C++17 + CMake + Ninja + BLAS for the core, Python+uv for orchestration/catalog/Sphinx, R + many CRAN packages for the R binding, Octave + statistics for the MATLAB binding, Node.js + Vite for the dashboard SPA, Emscripten for the JS/WASM binding, plus (eventually) Rust / Go / .NET / Julia / Ruby / Swift / Kotlin for the F-rollout bindings, plus Docker for the parity environments (§2.6.1). Asking a contributor to install all of this bare-metal is unrealistic.

The repo ships **two layers** of local environment, with one clear primary path:

| Layer                                          | Built on                       | Purpose                                                                                                   | Authoritative for…                          |
| ---------------------------------------------- | ------------------------------ | --------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| **Outer dev shell** (this section)             | Devcontainer + Docker Compose  | Single container with all toolchains. The shell you `make` from.                                          | Build, test, render, orchestration          |
| **Inner parity environments** (§2.6.1)         | Per-framework Docker images    | One image per `(framework, pinned-version)` used to regenerate reference snapshots.                       | Reference snapshot bit-reproducibility      |

The outer shell **calls into** the inner images via Docker-in-Docker (or socket-mount on Linux/macOS). The maintainer's bare-metal install is treated as a documented fallback, not the primary path.

#### 2.12.1 Options considered

| Option                                          | Pros                                                                          | Cons                                                                                          | Verdict       |
| ----------------------------------------------- | ----------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | ------------- |
| **A. Devcontainer + Docker Compose** (chosen)   | One descriptor, works in VSCode / JetBrains / `devcontainer` CLI / plain `docker compose`. Single source of truth for tooling. Inner parity images reusable as base layers. | Heavy image (~3-5 GB), requires Docker locally. Docker-in-Docker setup needed for parity work. | **Primary**   |
| **B. Nix flake (`flake.nix` + `devshell`)**     | Hermetic, deterministic, exact-version pinning of every binary, no container overhead. Excellent for the C++/Python/R/Node mix. | Steep Nix learning curve. Smaller contributor pool. Doesn't replace Docker for parity envs. | **Documented as opt-in for Nix users**, second-class |
| **C. Conda mega-env (`environment.yml`)**       | Cross-platform, no Docker needed.                                             | conda-forge doesn't cover Rust / Go / .NET / Swift / Octave statistics well; env resolution is slow at this scale. | **Available as `parity/environments/dev/`** (already in §2.6.1) — fallback for non-Docker users on the Python/R surface only |
| **D. Bare-metal `scripts/bootstrap-<os>.sh`**   | Closest to the maintainer's actual machine, no overhead.                      | Not hermetic, OS-specific (apt vs brew vs choco), drifts silently.                            | **Documented as advanced** for hardware-tied work (timing benchmarks on `bench-01-2026`) |

Reasons option A wins: the contributor cohort skews scientific Python / R rather than systems / Nix; the project already commits to Docker for the inner parity images so adding it to the outer shell adds zero new dependency; and `devcontainer.json` is portable across VSCode, JetBrains Gateway, GitHub Codespaces, and the standalone `devcontainer` CLI — covering most realistic contributor setups out of one config.

#### 2.12.2 Layout

```
.devcontainer/
├── devcontainer.json       # VSCode / Codespaces / JetBrains descriptor
├── Dockerfile              # Outer dev shell: C++17 + CMake + Ninja + OpenBLAS, Python 3.12 + uv,
│                           #   R 4.4 + system deps, Octave 9 + statistics, Node 22 + npm,
│                           #   Emscripten 3.x, Docker CLI (for in-shell `docker run` of parity envs)
├── docker-compose.yml      # For non-VSCode use: `docker compose run --rm dev bash`
└── post-create.sh          # Installs rustup (stable), Julia (juliaup), additional toolchains
                            #   that aren't in the base image; pulls parity env images

.github/
└── codespaces/             # Optional Codespaces prebuild config — mirrors the devcontainer

scripts/
├── doctor.sh               # Print versions of every required tool, flag missing deps with install hints
├── bootstrap-bare-metal-linux.sh    # Best-effort apt-based install (Ubuntu 22.04 / 24.04 reference)
├── bootstrap-bare-metal-macos.sh    # brew-based install
└── bootstrap-bare-metal-windows.ps1 # Chocolatey-based install (the C++/CMake/Ninja/Python/R/Node stack only)
```

Plus three `Makefile` entry points wired to the above:

```make
make doctor                 # diagnose: print tool versions, flag gaps, suggest fixes
make bootstrap              # interactive: prompts to use devcontainer (recommended) or bare-metal
make shell                  # opens a bash inside the dev container (proxy for `docker compose run --rm dev bash`)
```

#### 2.12.3 The contributor journey

The "fresh clone to first PR" path:

```bash
git clone https://github.com/GBeurier/nirs4all-methods.git
cd nirs4all-methods
make doctor               # See what's already on your machine
make bootstrap            # If you have Docker + VSCode: opens the devcontainer
                          # Otherwise: prints next steps for your OS
# Inside the dev shell (or on bare metal):
make env-build            # Pull/build the inner parity Docker images (§2.6.1)
cmake --preset dev-release && cmake --build --preset dev-release --parallel
ctest --preset dev-release --output-on-failure
make parity METHOD=pls_simpls     # Run one parity check end-to-end
```

Everything past `make bootstrap` is identical inside the devcontainer or on bare metal — the `Makefile` is the single API surface a contributor learns.

#### 2.12.4 Caching and persistence

The devcontainer mounts named Docker volumes for the high-churn caches so a `docker compose down` doesn't blow away an hour of pip / R / npm / cmake / ccache work:

- `n4m-ccache` → `/root/.ccache` (C++ compile cache)
- `n4m-uv-cache` → `~/.cache/uv`
- `n4m-r-libs` → `~/R/x86_64-pc-linux-gnu-library/4.4`
- `n4m-npm-cache` → `~/.npm`
- `n4m-cargo-cache` → `~/.cargo/registry` (for F-rollout-2 onward)

The inner parity images (per §2.6.1) live in the **host** Docker daemon — they're not nested inside the devcontainer image — and are reached via Docker-socket mount, so they're cached and shared across re-creations of the outer shell.

#### 2.12.5 Limits — what the dev shell does NOT cover

- **Timing benchmarks.** `make bench` should always run **bare-metal on the canonical host card** (`bench-01-2026`, §2.6.2). Containers add cache/scheduling noise that invalidates timing measurements. The devcontainer can run `make bench` and will produce numbers, but they get a "non-canonical host" badge and are not committed to `benchmarks/results/`.
- **Hardware-tied work** (GPU CUDA testing, ARM-specific runners, MATLAB licence-bound features). Bare-metal on the right machine, documented per-target.
- **Releases.** The release workflows run on GitHub-hosted runners with their own clean images, not the devcontainer.

---

## 3. Backlog

Organized into **6 sequential phases** (each phase can take multiple PRs). Respects the user constraint: **wipe duplicated ABI wrappers, binding stubs, old CLI, obsolete docs without mercy**; **absolutely preserve** the numerical code (`cpp/src/core/`) and the parity fixtures (`parity/fixtures/`).

### Phase A — Rename cleanup (non-destructive big-bang)

**Goal**: a single `n4m` namespace, no remaining `p4a` paths or symbols, no duplicated folders.

| # | Task | Files / folders | Decision |
|---|------|------------------|----------|
| A1 | Run `scripts/migrate_p4a_to_n4m.py --apply` across the whole repo | ~600 files, ~18.5k replacements | preserve code, change symbols only |
| A2 | Delete legacy header dir | `cpp/include/pls4all/p4a.h`, `cpp/include/pls4all/n4m_version.h`, `cpp/include/pls4all/n4m_export.h.in` | wipe |
| A3 | Delete legacy ABI wrappers | `cpp/src/c_api/` (whole folder, after confirming no missing symbol vs `c_api_n4m/`) | wipe |
| A4 | Delete old CLI | `cpp/cli/n4m_cli.cpp` | wipe; rename `cpp/cli_n4m/` → `cpp/cli/` |
| A5 | Delete old test tree | `cpp/tests/` (118 files) | wipe; rename `cpp/tests_n4m/` → `cpp/tests/` |
| A6 | Rename ABI snapshot dir | `cpp/abi_n4m/` → `cpp/abi/` | rename |
| A7 | Delete Python binding stubs | `bindings/python_nirs4all_methods/`, `bindings/python_pls4all/` | wipe |
| A8 | Consolidate Python bindings | Merge `bindings/python/` (mature) with `bindings/python_n4m/` (new) into a single `bindings/python/` | merge |
| A9 | Delete R binding stubs | `bindings/r_n4m/`, `bindings/r_pls4all/` (stubs) | wipe |
| A10 | Keep `bindings/r/` (mature) as baseline | Rename internal package `pls4all` → `n4m` | rename |
| A11 | Freeze non-priority binding PoCs | Move `bindings/{go,rust,dotnet,lua,nim,ruby}` → `bindings/_archive/` | archive (not wipe — PoCs can be revived) |
| A12 | Rewrite CMake targets | `cpp/src/CMakeLists.txt`, `cpp/src/n4m_targets.cmake`: single `libn4m` (SHARED + STATIC), no more `pls4all_*` | rename targets |
| A13 | Rename CMake presets | `CMakePresets.json`: variables `PLS4ALL_*` → `N4M_*` and any preset name containing `pls4all` | rename |
| A14 | Delete root-level obsolete docs | (already deleted in git status) `ARCHITECTURE.md`, `Backlog.md`, `Direction_Technique.md`, `Overview.md`, `ROADMAP.md` — commit the deletion | wipe (already exist under `docs/`) |
| A15 | Delete merge artefacts | `pls4all.Rcheck/`, `bindings/_catalog/`, `bindings/python_n4m/_original_donor_python/`, `report.log`, `objectifs.txt` (empty) | wipe |
| A16 | Update root `README.md` | Reflect the `nirs4all-methods` name + new quickstarts | rewrite |
| A17 | Update `CONTRIBUTING.md` | Document the closed-lib + PR-only extensibility model (§1.1bis), the user request flow (issue templates), and the method-addition checklist | rewrite |
| A18 | Create GitHub issue templates under `.github/ISSUE_TEMPLATE/` | One YAML form per change type per §1.1bis: `method-request.yml`, `method-update.yml`, `external-reference-request.yml`, `binding-request.yml`, `binding-update.yml`, `subset-request.yml`, `bug-report.yml`, `parity-discrepancy.yml`. Plus `config.yml` to disable blank issues. **Schemas per §1.1quater** — `method-request.yml` shown in full there; the others mirror the structure. Each issue form links the matching PR template in its closing `description:` via a `?template=...` URL | new files |
| A19 | Create PR templates under `.github/PULL_REQUEST_TEMPLATE/` | One template per change type per §1.1bis: `method-add.md`, `method-update.md`, `external-reference.md`, `binding-add.md`, `binding-update.md`, `subset.md`, `parity-fix.md`. Plus a top-level `.github/PULL_REQUEST_TEMPLATE.md` generic fallback. **Schemas per §1.1quater** — `method-add.md` shown in full there; the others mirror the structure with scope-adapted checklists. Every PR template ends with the §2.10 invariants checklist the reviewer (human + Codex) must verify | new files |
| A20 | Build the outer dev shell (§2.12) — devcontainer + Docker Compose | `.devcontainer/Dockerfile` ships C++17 + CMake + Ninja + OpenBLAS, Python 3.12 + uv, R 4.4 + system deps, Octave 9 + statistics, Node 22, Emscripten 3.x, Docker CLI. Mounted named volumes for ccache / uv / R libs / npm. `.devcontainer/docker-compose.yml` for non-VSCode users. `post-create.sh` installs rustup + juliaup | new files |
| A21 | Implement `scripts/doctor.sh` + `make doctor` | Prints version + path for every required tool (cmake, ninja, cc/cxx, blas, python, uv, R, octave, node, docker, optionally rustc/cargo/julia/go/dotnet). Flags missing deps with OS-specific install hints | new files |
| A22 | Implement `scripts/bootstrap-bare-metal-{linux,macos}.sh` + `bootstrap-bare-metal-windows.ps1` | Best-effort native install for contributors who don't want containers. Linux=apt (Ubuntu 22.04/24.04), macOS=brew, Windows=Chocolatey. Covers the C++/CMake/Ninja/Python/R/Node/Docker stack; Rust/Julia/.NET deferred to the contributor | new files |
| A23 | Implement `make bootstrap` + `make shell` | Interactive `bootstrap`: if VSCode + Docker are detected, prints the "Reopen in Container" prompt; else falls back to `bootstrap-bare-metal-<os>`. `shell` is a proxy for `docker compose run --rm dev bash` so contributors don't have to memorize compose | Makefile additions |
| A24 | Optional `.github/codespaces/` config | Mirrors the devcontainer for GitHub Codespaces with prebuild config, so contributors can spin up an env in 30 s without local Docker | optional |
| A25 | Document MATLAB/Octave policy | Create `bindings/matlab/COMPAT.md` listing surface differences; clarify in README that CI is Octave | new file |
| A26 | Update `CLAUDE.md` | Reflect the post-Phase-A layout + dev shell entry points | rewrite |
| A27 | Final audit: `grep -rE "\\b(p4a\|pls4all)\\b"` must only return historical doc (changelog); `make doctor` is green on a fresh devcontainer; `cmake --preset dev-release && ctest` is green inside the dev shell | whole repo | verify |

**Phase A exit criterion**: `cmake --preset dev-release && ctest` is green inside the devcontainer (and on bare metal for the maintainer), `nm` on `libn4m.so` contains no `n4m_*`, the repo `tree -L 2` shows no legacy names, `make doctor` runs clean in a fresh devcontainer, every method-/binding-/subset-related issue and PR template renders correctly on GitHub.

### Phase B — Unified catalog + auto-generation

**Goal**: replace `benchmarks/parity_timing/registry.py` (10k LOC) and the monolithic YAMLs with `catalog/methods/<id>.yaml` (one file per method) as the single source of truth for parity, benchmarks, doc, and packaging.

| # | Task | Details |
|---|------|---------|
| B1 | Define the YAML schema for a method file | JSON Schema at `catalog/schema/method.json`, doc at `catalog/README.md`. See section 2.2 for the contract |
| B2 | Split the current `methods.yaml` into one file per method | One-shot script `catalog/scripts/split_legacy_methods.py`. Generates `catalog/methods/<id>.yaml` |
| B3 | Migrate `benchmarks/parity_timing/registry.py` data into the catalog | Everything that is *data* (canonical reference, alternates, params) goes into the YAMLs; everything that is *logic* (resolver, comparator) stays in Python but reads the YAMLs |
| B4 | Migrate `parity/tolerances.md` → `parity.tolerances` field in each YAML | Tolerances by `(method, reference)` pair |
| B5 | Implement `catalog/scripts/validate.py` | Check that: (1) each `abi_symbols` entry exists in `cpp/abi/expected_symbols_linux.txt`, (2) each `reference.canonical` is resolvable OR `status: paper_only` with a `self_consistency` block, (3) each `bindings.<lang>.raw` and each `bindings.<lang>.idiomatic[].symbol` points to an existing symbol, (4) each `bindings.<lang>.idiomatic[].profile` matches a profile YAML in `bindings/profiles/`, (5) `examples.canonical_scenario` exists, (6) each `reference.canonical.env_id` and `alternates[].env_id` matches a recipe in `parity/environments/`, (7) `catalog_version` matches the current schema |
| B6 | Implement `catalog/scripts/render_api.py` **with per-language Jinja templates from day one** | Templates live in `catalog/scripts/templates/<lang>.j2` — one per binding/language. From the catalog, generate: `bindings/python/src/n4m/__init__.py` (re-exports), `bindings/r/n4m/NAMESPACE`, `bindings/matlab/+n4m/Contents.m`, etc. Never re-export by hand again. **Hard rule**: no language-specific string formatting in `render_api.py` itself |
| B7 | CI workflow `catalog-validate.yml` | `python catalog/scripts/validate.py` on every PR. **Hard blocker** — no merge if it fails |
| B8 | Implement `catalog/scripts/migrate_schema.py` | One-shot schema migration tool. Used whenever `catalog_version` is bumped. Rewrites every `catalog/methods/*.yaml` from version N to N+1 |
| B9 | Delete `catalog/methods.yaml`, `methods.discovered.yaml` (duplicate), `benchmarks/parity_timing/registry.py` | Once B2-B5 are stable |
| B10 | Delete `roadmap/phase-*.md` files that describe implemented methods | Replaced by the catalog YAMLs |

**Phase B exit criterion**: `python catalog/scripts/validate.py` returns 0 errors, `render_api.py` reproduces existing APIs identically, `registry.py` is gone.

### Phase C — Parity infra rebuild (scenario-based)

**Goal**: separate scenarios / snapshots / comparator / verdicts, and wire everything to the catalog.

| # | Task | Details |
|---|------|---------|
| C1 | Build `parity/environments/<env_id>/` recipes | One Dockerfile + lock per (framework, version): `env-py-sklearn-1.5`, `env-py-ikpls-1.2`, `env-r-pls-2.8`, `env-r-ropls-1.36`, `env-octave-6.4`, `env-py-n4m-runtime`, etc. Plus a single conda recipe `dev/environment.yml` for local convenience |
| C2 | CI workflow `parity-env-build.yml` | On change to any `parity/environments/*/`: rebuild image, push to GHCR with tag `<env_id>-<short_lock_hash>` |
| C3 | Create `parity/scenarios/<method_id>.yaml` | Format from §2.3. For each method, canonical scenarios (small/medium/large, real datasets) |
| C4 | Migrate `parity/fixtures/*.json` (202 files) → `parity/snapshots/current/<method_id>/<scenario_id>.json` | Format from §2.3. One-shot script. **Preserve the data** — only the layout changes. Backfill missing `env_id` and `host_card_id` where derivable |
| C5 | Implement `parity/comparator/` in Python | Reads a snapshot JSON + an n4m output JSON + the method YAML → verdict. Output goes into `parity/results/`. Handles both numerical-tolerance verdicts and `self_consistency` verdicts for `paper_only` methods |
| C6 | Rewrite external generators cleanly | `parity/generators/python/` (sklearn, ikpls, diPLSlib, OnPLS, tensorly), `parity/generators/r/` (pls, ropls, mixOmics, plsVarSel, chemometrics, plsRglm), `parity/generators/octave/` (plsregress, libPLS-ports). Each generator reads `parity/scenarios/<method_id>.yaml`, runs the external method **inside its matching Docker container**, writes a snapshot with full provenance including `env_lock_hash` and `host_card_id` |
| C7 | Support pipeline references in generators | When `reference.canonical.invocation.kind == "pipeline"`, the generator assembles the pipeline in the host framework, runs it, captures the output of the stage declared in `compare_against`. Minimal generic implementation — no per-pipeline custom code |
| C8 | Implement `self_consistency` checks for `paper_only` methods | `parity/comparator/self_consistency.py` handles `determinism`, `refit_idempotence`, declared `invariant` expressions. Used by AOM-PLS, POP-PLS, and any future method with no external reference |
| C9 | Write `parity/generators/run.py` | Orchestrates snapshot regeneration: resolves `env_id` → docker image, runs the right generator inside, collects snapshots. Parallel via `ProcessPoolExecutor` (parity is embarrassingly parallel per §2.5) |
| C10 | Cross-reference snapshots | For each method with `alternates`, also compute `sklearn vs R.pls`, `R.pls vs Octave`, etc. Stored in `parity/snapshots/current/<method_id>/_cross_reference/` |
| C11 | Inter-binding parity | For each method, after the C++ raw result, replay the scenario through every other binding (python raw, python sklearn, r raw, r formula, octave, js, …) and verify `rmse_rel < 1e-12` vs C++ raw. Results in `parity/results/<method_id>/_bindings/` |
| C12 | Define benchmark host card format | Create `benchmarks/hosts/bench-01-2026.yaml` (current maintainer machine) + `benchmarks/hosts/README.md` documenting the host-card lifecycle (creation, deprecation, multi-host policy) |
| C13 | CI workflow `parity-snapshots.yml` (manual `workflow_dispatch`) | Regenerates external snapshots inside their Docker images. Inputs: `method` (one id or `all`), `reference` (one alias or `all`), `env_id`. Writes back into `parity/snapshots/current/` on a branch, opens a PR with the diff |
| C14 | CI workflow `parity-run.yml` (manual `workflow_dispatch`) | Runs C++ vs in-repo snapshots. Inputs: `method` scope. Refuses any snapshot whose `env_lock_hash` doesn't match the current recipe. Output is a verdict report attached to the run |
| C15 | CI workflow `parity-paper-only.yml` (manual `workflow_dispatch`) | Runs `self_consistency` checks for `paper_only` methods. Inputs: `method` scope |
| C16 | `Makefile` entries `parity`, `snapshot`, `parity-paper-only` | Same operations as C13–C15, runnable locally with the same Docker images. Primary execution path for the maintainer (§2.6.3) |
| C17 | External storage (historical drift, manual archive) | When the maintainer decides to archive a snapshot generation (after a noteworthy external lib bump), `make archive-snapshots TAG=YYYY-MM` packages `parity/snapshots/current/` into `snapshots-YYYY-MM.tar.zst` and uploads it as a GH Releases asset. Not scheduled — only invoked when there's something worth archiving |
| C18 | Delete `parity/fixtures/` and `parity/PARITY_FIXES_SUMMARY.md` (one-shot done) | After C4 |
| C19 | Delete `parity/python_generator/` and `parity/r_generator/` legacy | After C6 |
| C20 | Delete `benchmarks/parity_timing/` (the folder) | After B9 + C6 |

**Phase C exit criterion**: `make parity METHOD=all` green on the current host card, all in-repo snapshots carry `env_lock_hash` + `host_card_id`, `make archive-snapshots TAG=...` produces a readable GH Releases asset, and the same three operations are runnable via `workflow_dispatch`.

### Phase D — Benchmark + SPA dashboard

**Goal**: turn the current static markdown table into an interactive SPA consuming a single JSON.

| # | Task | Details |
|---|------|---------|
| D1 | Refactor `benchmarks/cross_binding/orchestrator.py` (1306 LOC) | Consume `parity/scenarios/`, never carry its own data again. **Apply §2.5 measurement discipline rigorously**: timer lives in per-language runners, repeat-loops inside the runner for sub-1ms methods, `OMP_NUM_THREADS` + `taskset` set by orchestrator before spawning, parallel for parity / serial for timing by default. Output: `benchmarks/results/<run_id>/dashboard.json` |
| D2 | Per-language runner skeleton with internal timer | `benchmarks/runners/bench_{cpp,r,octave,python,js}.{cpp,R,m,py,mjs}`. Each emits `{"elapsed_ns": ..., "n_repeats": ..., "warmup_ns": ..., "env_observed": {...}}` for the orchestrator to ingest |
| D3 | Define the `dashboard.json` schema | `dashboard/schema/dashboard.schema.json`. Includes: methods (catalog summary), parity (verdicts incl. paper_only), timings (per scenario, per binding), provenance (`host_card_id`, `env_id`s, library versions) |
| D4 | Bootstrap `dashboard/` with Svelte + Vite | `npm create vite@latest dashboard -- --template svelte-ts`. Stack: Svelte 4, Vite, TailwindCSS, vanilla d3 (no heavy wrapper). Decision settled (§2.7) |
| D5 | "Matrix" view | Methods × bindings table + filters + parity badges (green/yellow/red, distinct `paper_only` badge) + median timing at a reference dataset size on the current host card |
| D6 | "Method drill-down" view | Per method: timing curves vs size (lin + log), multi-reference parity table, snapshot metadata (incl. `env_lock_hash`, `host_card_id`) |
| D7 | "Drift" view | Historical curve of parity verdicts over n versions (consumes the `snapshots-YYYY-MM.tar.zst` assets, lazy-loaded). Sparse by design — populated only with the manually-archived snapshots from C17 |
| D8 | "Catalog" view | Navigation by category, status, fuzzy search on name/symbol |
| D9 | "Host" panel | Surfaces the current `host_card_id` with hardware details. Cross-host comparison disabled / warning-gated |
| D10 | `Makefile` entries `dashboard-data`, `dashboard-serve`, `dashboard-build` | Local generation and preview. `dashboard-data` reads `parity/results/` + `benchmarks/results/` → writes `dashboard/public/dashboard.json` |
| D11 | CI workflow `dashboard-publish.yml` (manual `workflow_dispatch`) | Build SPA + push to `gh-pages`. Input: ref to the commit containing the desired `dashboard.json` |
| D12 | CI workflow `benchmarks.yml` (manual `workflow_dispatch`) | Run timing matrix on a self-hosted runner pinned to the current host card. Inputs: `method` (or `all`), `n_threads` (list), `sizes` (list). Commits the run output to `benchmarks/results/<run_id>/` on a branch and opens a PR |
| D13 | Stale-data badge in the SPA | If `dashboard.json.generated_at` is older than a configurable threshold relative to the latest `main` commit touching a method, the matrix shows a "stale" badge — surfaces that the maintainer hasn't re-run the bench yet, without falsely claiming current numbers |
| D14 | Delete `docs/landing/`, `docs/_extras/build_landing.py`, `docs/_static/bench-data.json`, `docs/_templates/landing.html`, `docs/benchmarks/cross_binding.md` (generated) | All replaced by the SPA |
| D15 | Sphinx hook: embed parity badge + mini timing table per method page | Read `dashboard.json` at build time to enrich method pages |

**Phase D exit criterion**: `https://gbeurier.github.io/nirs4all-methods/` serves the SPA after a manual `dashboard-publish` dispatch, `make dashboard-data && make dashboard-serve` works locally, drill-down works on at least 5 pilot methods, the stale-badge logic is wired.

### Phase E — Sub-packages (packaging-only)

**Goal**: ship curated subset packages (PyPI + CRAN) that re-export part of `n4m` without recompiling `libn4m.so`. Each subset is a thin wrapper depending on `n4m == X.Y.*`.

**Proposed subset catalog** (final list to be confirmed at start of Phase E):

| Subset id | Description | Audience | Languages |
|-----------|-------------|----------|-----------|
| `nirs4all-methods` (full) | The full n4m surface | Researchers, power users | PyPI, CRAN, npm, MATLAB FX |
| `pls4all` | PLS family only: PLS, SIMPLS, kernel PLS, OPLS/OPLS-DA, PLS-DA/QDA/LDA/Logistic, sparse PLS, CPPLS, MB-PLS, N-PLS, LW-PLS, AOM-PLS, POP-PLS, PCR, ridge/robust/weighted/continuum PLS, ECR | Generic ML / chemometrics users who don't need spectroscopy preprocessing | PyPI, CRAN |
| `nirs4all-preprocessing` | SNV, MSC, EMSC, derivatives (1st/2nd/Norris-Williams), SG smoothing, Detrend, ASLS, Haar/wavelet, OSC, EPO, ReflectanceToAbsorbance, Resampler | Standalone preprocessing for users who model with sklearn/keras | PyPI |
| `nirs4all-selection` (a.k.a. `pls-varsel`) | Variable selection methods: VIP, CARS, SPA, GA, IRIV, IRF, VISSA, PSO, shaving, BVE, REP, IPW, ST-PLS, T2, WVC, EMCUVE, randomization, biPLS, siPLS, interval, stability, UVE, random frog, SCARS, VIP-SPA | Chemometrics users who want feature selection without committing to n4m's modeling stack | PyPI, CRAN |
| `n4m-mobile` | Predict-only surface: bundle loaders + `predict()` paths for the PLS family + preprocessing pipelines. Excludes training, CV, augmentation, validation, selection | Embedded / Android / iOS deployments. Same `libn4m.so` (packaging-only per §1.2) but a stripped Python/JS API | PyPI, npm |
| `nirs4all-augmentation` | Spectral augmentation operators: noise, baseline drift, wavelength shift/stretch/warp, mixup, batch effect, instrumental broadening, spline perturbations, scatter simulation | Deep learning users (Keras/PyTorch) needing physics-aware NIR data augmentation | PyPI |

`pls-transfer` (PDS / DS calibration transfer), `pls-ensembles` (Bagging / Boosting / Random Subspace / GPR-on-PLS), `nirs4all-classification` and `pls-diagnostics` are noted but **deferred** until a real user request — narrow audiences relative to the per-subset packaging maintenance cost (CI matrix expansion, release pipelines, doc surface).

| # | Task | Details |
|---|------|---------|
| E1 | Define `catalog/subsets/<subset_id>.yaml` | List of `method_id`s to expose + package metadata (PyPI/CRAN name, description, license, deps) |
| E2 | Implement the new `catalog/scripts/render_subset.py` | Generates `bindings/python/packaging/<subset>/pyproject.toml` + `<subset>/__init__.py` that re-exports only the subset's classes/functions from `n4m`. Same template approach for R. **Uses the per-language Jinja templates from B6** |
| E3 | Delete the old `catalog/scripts/render_subset.py` (which generates separate `libpls4all.so` binaries) | Per the §1.2 decision |
| E4 | First operational subset: `pls4all` (PyPI + CRAN) | Wheel/tarball depending on `n4m == X.Y.*`, re-exports the PLS surface |
| E5 | Second operational subset: `nirs4all-preprocessing` (PyPI) | Re-exports preprocessing only |
| E6 | Third operational subset: `nirs4all-selection` (PyPI + CRAN) | Re-exports variable selection only |
| E7 | Fourth operational subset: `n4m-mobile` (PyPI + npm) | Predict-only surface. Validate that the resulting wheel imports only the predict code paths (size budget < 5 MB Python wheel) |
| E8 | Remaining subsets (`nirs4all-classification`, `nirs4all-augmentation`, `pls-diagnostics`) | Same pattern, one PR per subset |
| E9 | Update `release-py.yml` to build wheels for `n4m` + all subsets | Single workflow, matrix over `catalog/subsets/` |
| E10 | Update `release-r.yml` to build tarballs for `n4m` + R subsets | Manual CRAN submission step kept |
| E11 | Delete obsolete subset manifests | `catalog/subsets/aompls.yaml`, `n4m_transfer.yaml`, `nirs4all_methods.yaml` (renamed `nirs4all-methods.yaml` to match PyPI), etc. Only the agreed subsets remain |

**Phase E exit criterion**: `pip install pls4all` and `pip install nirs4all-preprocessing` work, expose their respective surfaces, depend on the matching `n4m` version. All subsets documented under `docs/source/packages/`.

### Phase F — Binding scaling infrastructure + rollout

**Goal**: build the §2.11 layered infrastructure (spec → conformance → profiles → render → skeletons) **before** scaling to 10+ languages, then bootstrap the first 4 bindings (Python, R, Octave, JS) as templates that validate the design, then roll out the remaining bindings each as a thin PR.

Three sub-phases, gated by exit criteria.

#### F-prep — Binding scaling infrastructure (must complete before F-bootstrap)

| # | Task | Details |
|---|------|---------|
| F-prep-1  | Write `bindings/SPEC.md` (§2.11.1) | The normative binding contract: library loading, ABI version probe, matrix marshalling (stride-aware), status/error handling, memory ownership, symbol enumeration. Short, prescriptive |
| F-prep-2  | Build the conformance suite `bindings/conformance/` (§2.11.2) | Language-agnostic JSON fixtures (input matrices, method id, params, expected output, optional expected error). Covers happy path + error path for at least 3 methods of each category |
| F-prep-3  | Write framework-profile schema | `bindings/profiles/schema.json` defining the YAML structure (id, language, package_dep, api_contract, parameter_mapping, naming, template, example_template). Validated by `catalog-validate.yml` |
| F-prep-4  | Extend `catalog/scripts/render_api.py` to consume profiles + templates (§2.11.3) | For each `(method × profile)` pair, emit the wrapper file from `bindings/profiles/templates/<profile_id>.j2`. **Zero hand-written per-method per-binding code from this point forward** |
| F-prep-5  | Implement skeleton generator (§2.11.4) | `bindings/skeletons/<lang>/` directories + `make new-binding LANG=<lang>` that copies the skeleton, fills in project metadata, and registers the binding in the CI matrix |
| F-prep-6  | Single CI matrix workflow `bindings.yml` (§2.11.5) | Replaces all per-language workflows. Matrix over `binding`. Per-binding setup actions under `.github/actions/setup-<lang>/` |
| F-prep-7  | Release matrix workflow `release-bindings.yml` (manual `workflow_dispatch`) | Single workflow looping over the same matrix, publishing each binding to its native registry |

**F-prep exit criterion**: `bindings/SPEC.md` is complete, conformance suite covers ≥ 3 methods, schema validation runs in `catalog-validate.yml`, `render_api.py` can emit per-profile wrappers, `make new-binding LANG=dummy` produces a buildable skeleton, `bindings.yml` matrix passes for a "dummy" binding that does nothing but pass conformance.

#### F-bootstrap — First 4 bindings as templates

Goal: validate the F-prep design by bringing 4 bindings to full conformance + multi-profile coverage. These four are templates for the rest.

| # | Task | Details |
|---|------|---------|
| F-boot-1  | Python: raw FFI compliant with `SPEC.md` | Migrated from existing `bindings/python/_ffi.py`. Implements all SPEC entry points |
| F-boot-2  | Python: profile `python_sklearn` + Jinja template | Wraps the existing sklearn wrappers under the profile system. `BaseEstimator`-compliant, `check_estimator` clean |
| F-boot-3  | Python: profile `python_keras` + Jinja template | New idiomatic wrapper. `tf.keras.layers.Layer` style for predict-only inline use |
| F-boot-4  | R: raw FFI compliant with `SPEC.md` | Migrated from existing `bindings/r/` (PLS-only) to full catalog |
| F-boot-5  | R: profiles `r_formula`, `r_pls_compat`, `r_mdatools_compat` + Jinja templates | Three idiomatic wrappers (§1.2 + user confirmation 2026-05-23) |
| F-boot-6  | Octave: raw FFI + MEX dispatcher compliant with `SPEC.md` | Built from existing `bindings/matlab/`. Tested via Octave; MATLAB-specific surface gated by `bindings/matlab/COMPAT.md` |
| F-boot-7  | Octave: profiles `matlab_classdef`, `matlab_factory` + Jinja templates | Existing classdefs migrated under the profile system |
| F-boot-8  | JS/WASM: raw FFI compliant with `SPEC.md` | Migrated from existing `bindings/js/` (9 files) to full catalog via `cwrap` |
| F-boot-9  | JS/WASM: profile `js_tfjs` + Jinja template | `tf.Layer`-style wrapper for browser-side predict |
| F-boot-10 | Per-binding conformance runners | Thin (~100-200 LOC) runner per binding that ingests `bindings/conformance/` fixtures |
| F-boot-11 | Inter-binding parity tests (consumes C11) | Each bootstrap binding runs the inter-binding parity assertion against C++ raw |
| F-boot-12 | Maintain `bindings/matlab/COMPAT.md` | Per-symbol MATLAB-vs-Octave divergence table; `catalog-validate.yml` fails if a divergence is undeclared |

**F-bootstrap exit criterion**: the 4 bootstrap bindings (Python, R, Octave, JS) each pass conformance + inter-binding parity for the full catalog; the 9 profiles listed above (2+3+2+1+raw counted separately) are operational; adding one more method to the catalog auto-generates wrappers for all 9 profiles with zero hand-written code.

#### F-rollout — Additional language bindings (each ~1-2 days)

Each of these is a thin PR after F-prep + F-bootstrap: run `make new-binding LANG=<lang>`, fill the language-specific FFI quirks (1-2 days), declare profiles on demand. Listed in priority order.

| # | Task | Notes |
|---|------|-------|
| F-roll-1 | `julia` binding + profile `julia_mlj` | High-value scientific ecosystem |
| F-roll-2 | `rust` binding (raw only at first) | Foundation for embedded use cases |
| F-roll-3 | `rust` profile `rust_linfa` | When/if there's user demand |
| F-roll-4 | `go` binding (raw only) | Foundation for backend services |
| F-roll-5 | `dotnet` binding + profile `dotnet_mlnet` | ML.NET integration |
| F-roll-6 | `ruby` binding (raw only) | Data science niche |
| F-roll-7 | `swift` binding + profile `swift_coreml` | iOS / macOS deployments |
| F-roll-8 | `kotlin` / `jni` binding + profile `kotlin_smile` | Android deployments (pairs with `n4m-mobile` subset) |

Languages can be added/skipped/reordered based on user requests via `binding-request.yml` issues (cf. §1.1bis, Phase A18).

**F-rollout exit criterion**: every active binding passes the conformance suite; each ships at least one example per declared profile; `bindings.yml` matrix is green across all rows.

---

## 4. Refactor success metrics

| Metric                                                       | Target                              |
| ------------------------------------------------------------ | ----------------------------------- |
| `grep -rE "\b(p4a\|pls4all)\b" cpp/ bindings/ catalog/ parity/ benchmarks/ docs/source/` | 0 matches outside `_archive/` and `CHANGELOG.md` |
| Duplicated folders (`*_n4m` next to `*`)                     | 0                                   |
| Empty binding stubs                                          | 0 (all archived or implemented)     |
| LOC of `benchmarks/parity_timing/registry.py`                | 0 (deleted)                         |
| Methods with non-null `parity.canonical` or `status: paper_only` with self-consistency block | 100 % of the catalog               |
| Methods green on the last `make parity METHOD=all` recorded on main | 100 % of those with snapshots  |
| Methods with examples in 4 bindings                          | ≥ 50 methods                        |
| SPA dashboard deployed, reading `dashboard.json`              | yes                                 |
| `pip install pls4all` works and imports                      | yes                                 |
| `make parity METHOD=<single>` runtime on the canonical host   | < 30 s for a typical method         |
| `make parity METHOD=all` runtime on the canonical host        | documented per-release; no scheduled enforcement |
| Every committed snapshot carries `env_lock_hash` + `host_card_id` | 100 %                          |
| All `parity/environments/*/` recipes build clean in `parity-env-build.yml` | 100 %                  |
| Octave covers all `bindings/matlab/` examples in CI            | 100 % (MATLAB-only divergences declared in `COMPAT.md`) |
| Scheduled (`schedule:`/cron) workflows in `.github/workflows/` | 0 — all heavy runs are manual (`workflow_dispatch` + `Makefile`) |
| In-tree bindings passing the §2.11 conformance suite           | 100 %                                                       |
| Hand-written per-(method × idiomatic-profile) wrapper files    | 0 — all wrappers come from `render_api.py` + templates       |
| Per-language CI workflow files                                 | 0 — single `bindings.yml` matrix                            |
| Languages supported via `make new-binding LANG=<lang>` skeleton | ≥ 8 at v1, target ≥ 10 at mid-term                          |
| Active framework profiles in `bindings/profiles/`              | ≥ 9 at v1 (the F-bootstrap set), target ≥ 20 at mid-term    |
| Fresh-clone-to-first-`make-parity` time in the devcontainer    | ≤ 30 min (image pull + `env-build` + first parity run)      |
| `make doctor` runs clean on (a) the devcontainer, (b) the maintainer's bare metal | yes                                       |

