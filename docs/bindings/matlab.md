# MATLAB / Octave binding

The current MATLAB/Octave binding lives in `bindings/matlab` and exposes the
`+n4m` namespace over the public ABI-2 `libn4m` C interface. `+n4m` is the
current package namespace, not the retired `+pls4all` route.

The checked public surface includes:

- `n4m.version`
- `n4m.pls_fit`
- `n4m.snv_transform`
- `n4m.savgol_transform`
- `n4m.kennard_stone_split`
- generated method/model wrappers backed by `n4m_method_fit_mex` and
  `n4m_model_fit_mex`, including `gpr_pls`, `mb_pls`, `pls_diagnostics`,
  selection functions, and corresponding regression classes

The Octave path is CI-gated by `.github/workflows/cross-binding-parity.yml`,
which builds `libn4m`, compiles the MEX shims with
`bindings/matlab/build_mex.m`, then runs `bindings/matlab/test/test_parity.m`
and `bindings/matlab/test/test_n4m_namespace.m`.
MATLAB uses the same source package and is checked manually for release because
GitHub-hosted runners do not provide a MATLAB license.

See `bindings/matlab/README.md` for build commands and examples.

Generated method pages name a MATLAB function only after its `.m` signature
was found under `bindings/matlab/+n4m`; they do not derive a MATLAB route from
an old Python or R binding name.
