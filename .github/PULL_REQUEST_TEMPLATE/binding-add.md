<!-- .github/PULL_REQUEST_TEMPLATE/binding-add.md -->

## New binding: <language>

Closes #<issue_number>

### Skeleton

- [ ] Skeleton generated from `make new-binding LANG=<lang>`
- [ ] Project metadata filled (registry name, license, dependency on `libn4m`)
- [ ] Binding registered in the `bindings.yml` CI matrix

### SPEC.md conformance

Per [`bindings/SPEC.md`](../../bindings/SPEC.md), the new binding must implement:

- [ ] Library loading + ABI version probe (`n4m_check_abi_compatibility`)
- [ ] Stride-aware matrix marshalling (no implicit copies for transposes / slices)
- [ ] Status / error-string mapping
- [ ] Memory ownership semantics (caller-allocated + core-allocated paths)
- [ ] Symbol enumeration helper

### Conformance suite

- [ ] Conformance runner under `bindings/<lang>/conformance/` ingests
      `bindings/conformance/*.json` fixtures
- [ ] Happy-path and error-path fixtures pass for ≥ 3 methods of each category
- [ ] `bindings.yml` matrix row passes for this binding

### First idiomatic profile

- [ ] First framework profile declared in `bindings/profiles/<profile_id>.yaml`
- [ ] Per-method wrapper template under `bindings/profiles/templates/<profile_id>.j2`
- [ ] `render_api.py` regenerates wrappers for every catalog method that declares
      `bindings.<lang>.idiomatic[*].profile == <profile_id>`
- [ ] At least one runnable example per declared profile under `examples/<lang>/`

### Inter-binding parity

- [ ] `rmse_rel < 1e-12` vs C++ raw across the catalog
- [ ] Verdict report committed under `parity/results/_bindings/<lang>/`

### Native packaging

- [ ] Native package metadata complete (PyPI / CRAN / npm / Maven / NuGet / crates / ...)
- [ ] Release workflow row added to `release-bindings.yml`

### Reviewer checklist — §2.10 invariants

- [ ] No hand-written per-(method × profile) code — all wrappers from templates
- [ ] No new symbol exported outside the public C ABI
- [ ] `CHANGELOG.md` entry added under the next unreleased section
- [ ] `bindings/SPEC.md` reference up to date (if the binding introduces any new spec clarification)
