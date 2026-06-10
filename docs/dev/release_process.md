# Development — Release Process

How each binding of `libn4m` is versioned, gated, and published. Some paths are
automated (PyPI, CRAN-tarball build); the JS / MATLAB / Octave bindings are
**published manually** and are documented in full below.

## Binding → registry → automation

| Binding | Package | Registry | Automation | Trigger |
|---------|---------|----------|------------|---------|
| Python (full) | `nirs4all-methods` | PyPI | **Automated** — `release-wheels.yml` (cibuildwheel matrix + Trusted Publishing) publishes the `nirs4all-methods` project | push tag `v*` (non-`-rc`) → PyPI; `workflow_dispatch` + `publish=true` |
| Python (slim) | `pls4all` | PyPI | **Automated** — `release-python.yml` (its own Trusted Publisher) publishes the `pls4all` project: sdist + cibuildwheel + retag-to-py3 + TestPyPI/PyPI + post-publish smoke. The two Python workflows are split one-per-PyPI-project (no collision), each with its own one-to-one Trusted Publisher. | push tag `v*` (non-`-rc`) → PyPI; `workflow_dispatch` + `publish=true` |
| R | `n4m` | CRAN | **Semi-automated** — `release-r.yml` vendors libn4m into `src/vendor/`, runs `R CMD check --as-cran` on the Linux/macOS/Windows + release/devel matrix, and (on tag push) attaches the tarball to the GitHub Release. **Submission is the irreducible manual web form.** | `workflow_dispatch`; tag push attaches the tarball |
| R | `pls4all` (slim) | CRAN | **Semi-automated** — same `release-r.yml`, the matrix has a `pkg: [n4m, pls4all]` leg. | `workflow_dispatch`; tag push attaches the tarball |
| JS / WASM | `@nirs4all/methods-wasm` | npm | **Build CI-automated** in `cross-binding-parity.yml` (emsdk pinned, `npm test` parity); **publish manual** (this doc) | — |
| MATLAB | `+pls4all` | File Exchange | **Manual** (no licensed runner; build/test described in `bindings/matlab/COMPAT.md`) | — |
| Octave | `pls4all` (pkg) | — / Octave Forge | **Build CI-automated** in `cross-binding-parity.yml` (apt octave + `build_mex.m` + `test_parity`); **publish manual** | — |

## Pre-release gates (release blockers)

These three CI checks block any release; run them before tagging or publishing
anything (see `CLAUDE.md` → "Release / ABI gates"):

1. **Version sync** — `scripts/bump_version.sh --check`. The canonical version
   lives in `cpp/include/n4m/n4m_version.h`; the script syncs it into every
   active downstream manifest (`bindings/python/pyproject.toml`,
   `parity/python_generator/pyproject.toml`, `bindings/r/n4m/DESCRIPTION`,
   `bindings/js/package.json` + `package-lock.json`). MATLAB/Octave read the
   version from libn4m at runtime, so they have no manifest to sync; archived
   bindings under `bindings/_archive/` are frozen and excluded. **Bump with**
   `bump_version.sh --bump X.Y.Z`.
2. **ABI symbol surface** — the exported `n4m_*` set must match
   `cpp/abi/expected_symbols_{linux,macos,windows}.txt` exactly.
3. **SONAME / linkage** — `readelf -d` sanity on the shared object.

Also confirm the cross-binding parity gate (`cross-binding-parity.yml`) and the
self-consistency gate (`make parity-paper-only`) are green for the build you
ship.

## Automated paths (summary)

Each PyPI project gets its own workflow — one workflow per package, so PyPI
Trusted Publishing stays one-to-one and the previous dual-publish collision
on the `pls4all` name is structurally impossible:

- **PyPI (`pls4all`)** — tag `vX.Y.Z` (non-`-rc`) → `release-python.yml`
  builds the cibuildwheel matrix from `bindings/python/` directly, retags
  wheels to `py3-none-${platform}` (pure-Python over ctypes-loaded libn4m;
  no per-cpython ABI), runs the sklearn parity gate + installed-wheel smoke,
  publishes via Trusted Publishing, then re-installs from PyPI to verify
  propagation. `-rcN` tags route to TestPyPI on workflow_dispatch.
- **PyPI (`nirs4all-methods`)** — tag `vX.Y.Z` (non-`-rc`) → `release-wheels.yml`
  builds the cibuildwheel matrix (Linux x86_64 + aarch64, macOS x86_64 +
  arm64, Windows x86_64 across cp310–cp313) from the generated
  `bindings/python_nirs4all_methods/` dir (`make_python_package.py --name
  nirs4all-methods` writes the dir; `prepare_wheel_packages.sh` builds + stages
  libn4m into `n4m/lib/` inside each cibuildwheel env so the bundled lib
  matches the wheel). Repairs use auditwheel / delocate / delvewheel
  `--analyze-existing`. Publishes via Trusted Publishing.
- **CRAN (both `n4m` and `pls4all`)** — `release-r.yml` (`workflow_dispatch`,
  also attaches on tag push) vendors the full libn4m C/C++/Fortran core +
  static `n4m_export.h` into `src/vendor/` via `N4M_R_VENDOR=1 ./configure`,
  then runs `R CMD check --as-cran` across the `{pkg: n4m, pls4all} ×
  {linux-release, linux-devel, macos-arm64-release, windows-release}` matrix
  and produces a self-contained source tarball. **CRAN submission itself is
  the irreducible manual step**: download the tarball from the run (or the
  attached Release asset) and submit it at
  <https://cran.r-project.org/submit.html>.

---

## Manual binding publication

The JS, MATLAB, and Octave bindings have **no publish workflow**. Their native
glue (WASM module, MEX dispatcher) is built locally and published by hand. Each
binding's artifact is built from the **same `libn4m` source** at the released
version; always run the pre-release gates first.

### JS → npm (`@nirs4all/methods-wasm`)

The npm package ships the Emscripten WASM module + the TypeScript wrappers; the
`npm run build` step only runs `tsc`, so the WASM artifacts must be built first.

```bash
# 0. Gates: scripts/bump_version.sh --check   (syncs bindings/js/package.json)

# 1. Build the WASM module (requires the Emscripten SDK on PATH).
source /path/to/emsdk/emsdk_env.sh
cmake --preset emscripten
cmake --build --preset emscripten --target pls4all_wasm
#    → build/emscripten/bindings/js/{n4m.js,n4m.wasm}

cd bindings/js
npm run build           # tsc -p .  → dist/index.js + dist/index.d.ts (TS only)

# 2. STAGE the WASM artifacts into dist/ — `npm run build` runs only tsc, and
#    package.json ships `files: ["dist/"]`. The `stage:wasm` script copies the
#    Emscripten artifacts in (and `prepack` runs build + stage:wasm for you):
npm run stage:wasm      # → dist/n4m.js + dist/n4m.wasm

# 3. Verify the tarball actually contains index.js + n4m.wasm + n4m.js BEFORE
#    publishing, and run the smoke test:
npm pack --dry-run      # inspect the file list
npm test                # node test/run_smoke.mjs — must pass

# 4. Publish (scoped public package; needs npm login + 2FA OTP).
npm publish --access public
```

Notes: the version is already correct if `bump_version.sh --check` passed (it
edits `package.json` + `package-lock.json`). The scope `@nirs4all` must exist on
the npm org and the publisher must be a member.

#### One-time npm registration (required for the automated `release-npm.yml`)

`release-npm.yml` publishes `@nirs4all/methods-wasm` automatically on a non-`-rc`
`v*` tag — it builds the WASM (pinned `setup-emsdk`), then runs `npm publish`
with `NODE_AUTH_TOKEN=${{ secrets.NPM_TOKEN }}` + provenance. **Until the scope +
token below exist, that one leg fails** (the PyPI / R / source legs are
independent and still succeed). To enable it:

1. **Own the `@nirs4all` scope** on [npmjs.com](https://www.npmjs.com) — sign in
   as the maintainer, *Add Organization* → create the free org `nirs4all` (so the
   scope `@nirs4all` is yours). The publishing account must be a member, and the
   package name `@nirs4all/methods-wasm` must be free/owned.
2. **Generate an automation token** — npmjs.com → *Access Tokens* → *Generate New
   Token* → **Granular Access** (or **Automation**), granting **Read and write**
   on the `@nirs4all` scope / `@nirs4all/methods-wasm` package. Copy it.
3. **Add it as a GitHub Actions secret** — repo *Settings → Secrets and variables
   → Actions → New repository secret*, name **`NPM_TOKEN`**, value = the token.
4. **Publish** — either re-run `release-npm.yml` (*Run workflow* → `publish=true`)
   for the already-cut `v0.99.0`, or it publishes automatically on the next tag.

`release-npm.yml` requests `id-token: write`, so once the scope + token exist the
package publishes with a verified npm provenance attestation. The WASM staging is
handled by the package's `prepack` script in CI — no manual step.

### MATLAB → File Exchange (`+pls4all`)

MATLAB has no package registry with a CLI publish; distribution is a **MATLAB
File Exchange** listing (typically linked to the GitHub repo) and/or a packaged
`.mltbx` toolbox.

```matlab
% 0. Build the MEX dispatcher against libn4m (see bindings/matlab/README.md).
cd bindings/matlab
build_mex            % build_mex.m — compiles the n4m_*_mex entry points
```

Then either link the GitHub repository from a File Exchange entry, or package a
redistributable `.mltbx`. There is **no** committed Toolbox project file — to
build a `.mltbx` you first create one (MATLAB **Add-Ons → Package Toolbox**,
which writes a `.prj`), then `matlab.addons.toolbox.packageToolbox('<that>.prj',
'pls4all.mltbx')`. CI does **not** run MATLAB (no licensed runner); confirm
MATLAB-specific behaviour against `bindings/matlab/COMPAT.md` before publishing.
The `+pls4all` package reads its version from libn4m at runtime.

### Octave (`pls4all`)

The Octave surface is the same `bindings/matlab/` binding (built for the
MATLAB ∩ Octave intersection); Octave builds the MEX via `build_mex.m` /
`mkoctfile` — there is **no** dedicated CMake Octave target. The build is
**CI-tested on every push** (the `octave-mex` job in
`cross-binding-parity.yml` installs apt Octave, runs `build_mex.m`, and gates
on `test_parity` with `rmse_rel <= 1e-12`; locally observed ~4e-16). What is
still manual is **publication** (no Octave Forge submission is wired): ship
the built binding with the GitHub Release, or have users build it locally
against the released libn4m.

---

## Why JS / MATLAB / Octave are manual today

Their **builds** are CI-automated in `cross-binding-parity.yml` (JS via
emsdk + `npm test`, Octave via apt octave + `build_mex.m` + `test_parity`).
What is still manual is **publication**: an `npm publish` job, a `.mltbx`
build job, or an Octave Forge submission. None of those are required for the
cross-binding promise (one libn4m → identical numbers in every binding) — they
are distribution-channel ergonomics, tracked but not blocking. MATLAB itself
stays out of CI entirely (no licensed runner); the Octave job validates the
MATLAB ∩ Octave intersection per `bindings/matlab/COMPAT.md`.
