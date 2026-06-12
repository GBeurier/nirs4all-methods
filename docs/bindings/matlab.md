# MATLAB / Octave binding

The MATLAB/Octave binding lives in `bindings/matlab` and exposes the `+pls4all`
namespace over the public `libn4m` C ABI.

Current public surface:

- `pls4all.version`
- `pls4all.pls_fit`
- `pls4all.snv_transform`
- `pls4all.savgol_transform`
- `pls4all.kennard_stone_split`
- generated method/model wrappers backed by `n4m_method_fit_mex` and
  `n4m_model_fit_mex`

The Octave path is CI-gated by `.github/workflows/cross-binding-parity.yml`,
which builds `libn4m`, compiles the MEX shims with
`bindings/matlab/build_mex.m`, and runs `bindings/matlab/test/test_parity.m`.
MATLAB uses the same source package and is checked manually for release because
GitHub-hosted runners do not provide a MATLAB license.

See `bindings/matlab/README.md` for build commands and examples.
