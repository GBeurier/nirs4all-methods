# RC-P2 Methods Binding Gates

Date: 2026-07-02

## Scope

Audit and small gate hardening for `nirs4all-methods` bindings/release checks:
Python installed-wheel smoke, JS/WASM presence, R/Octave/MATLAB presence, and
native ABI freshness.

## Files modified

- `Makefile`
  - Added explicit local gate targets:
    - `test-abi-freshness`
    - `test-js-wasm`
    - `test-r-binding`
    - `test-octave-mex`
    - `test-matlab-binding`
- `scripts/check_native_abi_freshness.sh`
  - New reusable local native ABI gate:
    - committed ABI snapshot diff via `regen_abi_snapshots.sh --check`
    - exported-symbol namespace guard
    - `n4m_check_abi_compatibility` loadability check
    - Linux SONAME/RPATH/RUNPATH check
    - forbidden runtime dependency audit

## Gate audit

- Python installed-wheel smoke: present and now locally exposed via existing
  `make test-python-install`; verified.
- JS/WASM: binding present under `bindings/js`; CI already runs Emscripten +
  `npm test`. Added local `make test-js-wasm`.
- R: binding present under `bindings/r/n4m`; CI already runs `R CMD INSTALL` +
  `bindings/r/test_parity.R`. Added local `make test-r-binding`.
- Octave: shared MATLAB/Octave binding present under `bindings/matlab`; CI
  already builds MEX with Octave and runs `test_parity`. Added local
  `make test-octave-mex`.
- MATLAB: binding present and documented as manual due licensed runtime. Added
  local `make test-matlab-binding`.
- ABI/native freshness: CI had inline ABI checks and snapshot diff; added a
  reusable local script and Makefile target so the same release-blocking checks
  can run before CI.

## Tests run

- `scripts/check_native_abi_freshness.sh --help`
  - PASS
- `make help`
  - PASS; new targets render.
- `env FC=/nonexistent cmake --preset dev-release --fresh && cmake --build --preset dev-release --target n4m_c --parallel`
  - PASS; used to bypass the host `~/.local/bin/gfortran`, which is broken in
    this workspace and links against missing `/lib64/libm.so.6` /
    `/lib64/libmvec.so.1`.
- `make test-abi-freshness PRESET=dev-release`
  - PASS after the `dev-release` reconfigure above.
- `make test-python-install`
  - PASS; built `nirs4all_methods-1.0.2-...whl`, installed into a clean venv,
    imported `n4m`, loaded bundled `libn4m.so.2.0.0`, ABI `(2, 0, 0)`.
- `python3 -m pytest bindings/python/tests/test_installed_nirs4all_methods_smoke.py -q`
  - PASS, `3 passed`.
- `bash -n scripts/check_native_abi_freshness.sh`
  - PASS.

## Not run

- `make test-js-wasm`
  - Not runnable end-to-end in this environment: Node/npm are available through
    the local nvm install, but `emcc` is absent and `EMSDK` is unset, so the
    Emscripten preset cannot build the WASM artifact.
- `make test-r-binding`
  - Not runnable: no `R` or `Rscript` in PATH.
- `make test-octave-mex`
  - Not runnable: no `octave` in PATH.
- `make test-matlab-binding`
  - Not runnable: no `matlab` in PATH/licensed runtime.

## Decisions

- Kept CI behavior intact. The change adds local entry points and one reusable
  native ABI checker; it does not alter numerical kernels, ABI snapshots, or
  release workflows.
- MATLAB remains a manual gate because the repo already documents the licensed
  runner constraint.
- JS/R/Octave targets intentionally mirror existing CI commands instead of
  inventing separate smoke logic.

## Risks

- The host Fortran toolchain issue can make a plain fresh `dev-release` build
  attempt FITPACK and fail before gates run. This is environmental in this
  workspace; CI installs system toolchains explicitly. A future hardening could
  add an explicit CMake option for `AUTO/ON/OFF` FITPACK selection.
- Local JS/R/Octave/MATLAB targets were added but not executed here because the
  required runtimes are absent.
