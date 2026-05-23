<!-- .github/PULL_REQUEST_TEMPLATE/binding-update.md -->

## Binding update: <language>: <profile_id>

Closes #<issue_number>

### Profile definition

- [ ] `bindings/profiles/<profile_id>.yaml` added/updated (id, language, package_dep,
      api_contract, parameter_mapping, naming, template, example_template)
- [ ] Jinja template under `bindings/profiles/templates/<profile_id>.j2`
- [ ] Profile validates against `bindings/profiles/schema.json` (enforced by
      `catalog-validate.yml`)

### Catalog impact

- [ ] Affected `catalog/methods/<id>.yaml` entries gain a matching
      `bindings.<lang>.idiomatic[]` block
- [ ] `render_api.py` regenerated wrappers for every affected method; output
      committed

### Conformance hook

- [ ] If the profile targets a framework with its own contract (e.g. sklearn's
      `check_estimator`, R formula S3 dispatch), the conformance hook is wired
      under `bindings/<lang>/conformance/`
- [ ] Hook passes on the matrix CI row

### Examples & docs

- [ ] At least one runnable example per (method × profile) auto-generated
- [ ] Spot-check confirms generated examples compile and run
- [ ] Docs page under `docs/source/bindings/<lang>/<profile_id>.md` (or equivalent) added

### Reviewer checklist — §2.10 invariants

- [ ] No hand-written per-(method × profile) code
- [ ] Profile YAML is the only place per-profile facts live
- [ ] `CHANGELOG.md` entry added under the next unreleased section
