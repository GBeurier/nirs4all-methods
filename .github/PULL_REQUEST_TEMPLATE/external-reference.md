<!-- .github/PULL_REQUEST_TEMPLATE/external-reference.md -->

## External reference: <method_id> ← <framework>:<package>@<version>

Closes #<issue_number>

### Catalog change

- [ ] Entry added to `reference.alternates[]` (or `reference.canonical` updated) in `catalog/methods/<method_id>.yaml`
- [ ] `env_id` resolved (existing or new)
- [ ] `parity.tolerances:` extended with the new `(method, reference)` pair

### Environment recipe (if new env_id)

- [ ] `parity/environments/<env_id>/Dockerfile` + lockfile added
- [ ] `parity-env-build.yml` ran clean (image pushed to GHCR with `<env_id>-<short_lock_hash>` tag)
- [ ] Recipe reuses an upstream base image where possible (no hand-crafted deps)

### Snapshot regeneration

- [ ] `make snapshot METHOD=<method_id> REF=<ref_alias>` ran clean
- [ ] New snapshots committed under `parity/snapshots/current/<method_id>/`
- [ ] Every committed snapshot carries `env_lock_hash` + `host_card_id`

### Cross-reference verification

- [ ] Where this is an `alternates[]` addition, the cross-reference verdict
      vs the canonical was computed and committed under
      `parity/snapshots/current/<method_id>/_cross_reference/`
- [ ] For a `pinned_version` bump, the drift vs the previous pinned version is
      summarised in the PR description below

<details>
<summary>Drift / cross-reference report</summary>

```
(paste comparator output: rmse_rel, max_abs, sign_agreement vs canonical / previous version)
```
</details>

### Reviewer checklist — §2.10 invariants

- [ ] No new external symbol introduced outside the catalog YAML
- [ ] No tolerance loosened without an explicit rationale
- [ ] `CHANGELOG.md` entry added under the next unreleased section
