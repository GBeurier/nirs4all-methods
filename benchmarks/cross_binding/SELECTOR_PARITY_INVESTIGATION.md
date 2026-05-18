# Selector Cross-Binding Parity — Investigation

Branch: `release/m2-cran`.

## Root Cause

Selector parity depends on two deterministic inputs:

1. the `p4a_validation_plan_t*` used for cross-validated scoring;
2. the explicit selector RNG seed passed through the method parameters.

The C++ kernel does not create or shuffle validation folds internally.
Every selector that scores candidates by cross-validation consumes the
folds exactly as supplied by the binding through
`p4a_validation_plan_add_fold`.

The remaining selector RNGs are seeded explicitly from the registry cell
parameters. No selector binding should depend on the host language RNG.

## Chosen Invariant

The canonical binding plan is now a simple 3-fold contiguous split:

- `fold_size = n_samples // 3`
- folds 0 and 1 get `fold_size` samples
- fold 2 gets the remainder
- no shuffle

This was chosen over serialising shuffled indices into every R/MATLAB
benchmark call. It keeps the ABI and benchmark JSON payloads small while
making `registry_pls4all`, `cpp`, Python tier 1, Python sklearn tier 2,
R tier 1, and MATLAB tier 1 build the same plan.

## Implemented Alignment

- `benchmarks/parity_timing/registry.py::_build_default_plan`
  builds the canonical contiguous plan.
- `bindings/python/src/pls4all/sklearn/_selection.py::_default_plan`
  builds the same plan; its `seed` argument is retained for API
  compatibility but no longer changes fold composition.
- `bindings/r/pls4all/src/r_dispatch.c::make_default_plan`
  uses the same floor-sized 3-fold split.
- `bindings/matlab/mex/p4a_method_fit_mex.c::make_default_plan`
  uses the same floor-sized 3-fold split.

R and MATLAB selector dispatch call sites now request `make_default_plan(n, 3)`.

## Expected Gate Semantics

Binding parity for pls4all selector bindings should be strict: selected
masks, indices, and scores are expected to match the C++/registry cell
unless a binding surface is genuinely unavailable.

Unavailable idiomatic wrappers are reported as `not_available`, not as
failed parity:

- R formula tier 2 does not expose selector formula wrappers.
- MATLAB classdef tier 2 does not expose selector factory entries.
- Python sklearn only marks a selector missing when no estimator exists.

External reference parity is a separate gate. For selectors with
stochastic or convention-heavy external libraries, the registry tolerance
captures algorithm-family agreement and should not be described as binary
identity.
