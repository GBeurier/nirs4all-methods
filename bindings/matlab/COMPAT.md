# MATLAB / Octave compatibility table

`bindings/matlab/` targets the **intersection of MATLAB and Octave** so a
single binding ships to both ecosystems. CI exercises the **Octave**
path only — GitHub-hosted runners do not provide MATLAB licences and
self-hosting a MATLAB runner is not cost-effective at the project's
current scale (per `docs/REFACTOR_PLAN.md` §1.1ter).

Releases to MATLAB File Exchange happen on a periodic manual cadence
performed by a maintainer with a MATLAB licence; CI is responsible only
for the Octave-runnable surface.

The catalog's `catalog-validate.yml` workflow **fails closed** if a
divergence between MATLAB and Octave is observed by the conformance
runner but not declared in this file.

---

## How to read this file

Each row documents one symbol or feature that behaves differently on
MATLAB versus Octave. The columns are:

| Field | Meaning |
|-------|---------|
| **Symbol** | Public name in the `+n4m` namespace (or `n4m.fit("…")` factory key) |
| **MATLAB**  | Behaviour on MATLAB R2024a+ (the minimum we target) |
| **Octave**  | Behaviour on Octave 9.x (the version pinned in `.devcontainer/Dockerfile`) |
| **Resolution** | What the binding does at runtime to keep the contract observable |

---

## Declared divergences

> _None at v1 — the binding currently surfaces only MEX-level calls that
> behave identically on both runtimes. As we add idiomatic profiles
> (`matlab_classdef`, `matlab_factory`) in Phase F-bootstrap, divergences
> discovered by the conformance runner land here._

### Template for future entries

```
### `<symbol>`

- **MATLAB**: <how it behaves; cite the rN year if version-dependent>
- **Octave**: <how it behaves; cite octave-statistics version if relevant>
- **Resolution**: <e.g. "binding throws on Octave with a clear message">
- **Conformance hook**: `bindings/matlab/conformance/test_<symbol>.m`
- **Tracking issue**: #<n>
```

---

## Build and test on each runtime

```bash
# Octave (what CI runs)
cd bindings/matlab
octave --no-gui --eval "addpath(genpath('.')); n4m_smoke_test"

# MATLAB (manual; maintainer machine)
matlab -batch "addpath(genpath('bindings/matlab')); n4m_smoke_test"
```

The smoke test must pass identically on both runtimes; differences in
output that aren't explained by an entry in this file are a CI failure.
