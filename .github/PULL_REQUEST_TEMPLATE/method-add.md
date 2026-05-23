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
