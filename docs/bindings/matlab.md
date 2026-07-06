# MATLAB / Octave binding

The MATLAB/Octave binding lives in `bindings/matlab` and exposes the V1
`+n4m` namespace over the public `libn4m` C ABI. The older `+pls4all`
namespace remains available for compatibility.

Current public surface:

- `n4m.version`
- `n4m.pls_fit`
- `n4m.snv_transform`
- `n4m.savgol_transform`
- `n4m.kennard_stone_split`
- generated method/model wrappers backed by `n4m_method_fit_mex` and
  `n4m_model_fit_mex`

The Octave path is CI-gated by `.github/workflows/cross-binding-parity.yml`,
which builds `libn4m`, compiles the MEX shims with
`bindings/matlab/build_mex.m`, then runs `bindings/matlab/test/test_parity.m`
and `bindings/matlab/test/test_n4m_alias.m`.
MATLAB uses the same source package and is checked manually for release because
GitHub-hosted runners do not provide a MATLAB license.

See `bindings/matlab/README.md` for build commands and examples.
