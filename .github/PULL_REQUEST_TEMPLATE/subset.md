<!-- .github/PULL_REQUEST_TEMPLATE/subset.md -->

## New subset: <subset_id>

Closes #<issue_number>

### Subset manifest

- [ ] `catalog/subsets/<subset_id>.yaml` created with the exhaustive list of
      `method_id`s the subset re-exports
- [ ] Package metadata complete (PyPI / CRAN / npm name, description, license,
      deps including `n4m == X.Y.*`)
- [ ] All listed `method_id`s exist in `catalog/methods/`

### Render output

- [ ] `python catalog/scripts/render_subset.py` ran clean
- [ ] Generated re-export module(s) committed under
      `bindings/<lang>/packaging/<subset_id>/`
- [ ] Generated `pyproject.toml` / `DESCRIPTION` / `package.json` validate

### Release workflow

- [ ] `release-py.yml` (or the matching language workflow) matrix updated
      to include the new subset
- [ ] Smoke install from a fresh env documented in the PR description below

<details>
<summary>Smoke install transcript</summary>

```
(paste `pip install <subset_id>` from a fresh venv, then a smoke import + 1-line API call)
```
</details>

### Reviewer checklist — §2.10 invariants

- [ ] Subset ships the same `libn4m` binary (no separate compiled artefact)
- [ ] Re-export module is generated, not hand-written
- [ ] `CHANGELOG.md` entry added under the next unreleased section
