# MATLAB / Octave compatibility table

`bindings/matlab/` targets the **intersection of MATLAB and Octave** so one
source binding ships to both ecosystems. The V1 namespace is `+n4m`. The Octave path is CI-runnable:
`.github/workflows/cross-binding-parity.yml` builds `libn4m`, compiles the MEX
shims with `build_mex.m`, and runs `bindings/matlab/test/test_parity.m`,
`test_n4m_namespace.m`, `test_optimizer.m`, all 14 HPO golden traces and a
bidirectional Python checkpoint continuation. The CI runner currently uses
Octave 6.4 from Ubuntu 22.04; the devcontainer installs the Octave package
available on Ubuntu 24.04.

MATLAB uses the same source package but remains a manual release/runtime check
because GitHub-hosted runners do not provide a MATLAB license.

Releases to MATLAB File Exchange happen on a periodic manual cadence performed
by a maintainer with a MATLAB licence.

Divergences between MATLAB and Octave **must** be declared in the table below
and covered by the relevant conformance test.

---

## How to read this file

Each row documents one symbol or feature that behaves differently on
MATLAB versus Octave. The columns are:

| Field | Meaning |
|-------|---------|
| **Symbol** | Public name in the `+n4m` namespace |
| **MATLAB**  | Behaviour on MATLAB R2024a+ (the minimum we target) |
| **Octave**  | Behaviour on Octave 6.4 CI and the Ubuntu 24.04 devcontainer package |
| **Resolution** | What the binding does at runtime to keep the contract observable |

---

## Declared divergences

> _None at v1 - the shared public subset currently surfaces only MEX-level
> calls and thin function wrappers that behave identically on both runtimes.
> As we add idiomatic profiles (`matlab_classdef`, `matlab_factory`),
> divergences discovered by the conformance runner land here._

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
octave --no-gui --no-history --eval "cd bindings/matlab; build_mex"
octave --no-gui --no-history --eval "addpath('bindings/matlab'); cd bindings/matlab/test; test_parity"

# MATLAB (manual; maintainer machine)
matlab -batch "cd bindings/matlab; build_mex"
matlab -batch "addpath('bindings/matlab'); cd bindings/matlab/test; test_parity"
```

The smoke test must pass identically on both runtimes; differences in
output that aren't explained by an entry in this file are a CI failure.
