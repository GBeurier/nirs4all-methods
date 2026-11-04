# Static component-analysis qualification

Validated on Linux x86-64 with optimized C++/Rust artifacts, ABI 2.22.0.
The mixed-effects and fANOVA kernels execute in the shared native library, an
installed standalone static C consumer, an installed Python wheel, Node 24 WASM,
and Chrome 143 WASM. Source builds without either optional backend expose the
same ABI and explicit unavailable results. Other desktop targets are covered by
the component-analysis CI matrix; local Linux results alone do not qualify them.

## Gates

- Full native CTest suite: 34/34 targets, including seven fANOVA scientific suites,
  mixed-effects strided/atomic/refusal contracts and existing engine regressions.
- Full Python binding suite: 1,655 passed, 9 optional skips. Binding generation,
  catalog and ABI checks pass; strict method-science/catalog tests: 33 passed.
- Mixed-effects adapter: seven Rust tests, formatting and clippy with warnings as
  errors. Independent statsmodels 0.14.6 fixtures cover crossed/nested ML and REML;
  analytic cases cover coefficients, variances and conditional covariance.
- Existing linked Rust binding: 36 tests plus doc tests pass.
- Complete JS/WASM suite passes, including actual analytic mixed fits, main and
  interaction fANOVA and refusal cases. Both engines also execute inside Chrome,
  where the browser consumes the built package through its public ES module.
- Installed static C11 consumer uses only exported `find_package(N4m)` targets;
  analytic REML, XOR fANOVA and constant-response refusal pass without libn4m.so.
- Four same-source optimized native builds and built wheels isolate each backend's
  increment. The combined wheel is installed into a fresh environment and fits
  both engines. See `component_analysis_sizes.linux-x86_64.json`.

## Reproduce

```sh
cmake --preset analysis-all
cmake --build --preset analysis-all --parallel 3
ctest --test-dir build/analysis-all --output-on-failure
python scripts/test_component_backends.py

# In an activated Emscripten 3.1.74 + Rust 1.96.0 target environment:
cmake --preset emscripten-analysis
cmake --build --preset emscripten-analysis --target n4m_wasm --parallel 3
cd bindings/js
npm ci
npm run build
N4M_WASM_PRESET=emscripten-analysis npm run stage:wasm
npm test
```

The installed-static consumer has its own presets and instructions under
`cpp/tests/installed_component_analysis/`. Artifact measurement uses
`scripts/measure_component_artifacts.py` after building all four `analysis-*`
and Emscripten feature combinations. Release jobs enable both backends and run
actual binding fits; end users need no Rust, R or additional analysis service.

## Counterexamples and interpretation

Rank-deficient fixed designs, confounded random covariance directions, insufficient
group structure, constant/perfectly explained responses and exhausted optimizers
withhold mixed estimates. Genuine boundary fits remain explicitly flagged. SEs
are conditional plug-in values, with no claimed variance-parameter uncertainty.

fANOVA tests include exact two/three-factor categorical decompositions, unreported
higher orders, widened declared domains, incomplete/correlated observations,
missing category support, extreme/subnormal responses, unaligned and strided
inputs, constant trees and bounded marginal budgets. Statistical refusals expose
no effect estimates; technical errors preserve caller buffers. Importance is
surrogate variance attribution under a declared independent uniform domain, not
a signed gain or a causal effect. Tree dispersion is not a confidence interval.
