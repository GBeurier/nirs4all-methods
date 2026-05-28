# Development — Release Process

How each binding of `libn4m` is versioned, gated, and published. Some paths are
automated (PyPI, CRAN-tarball build); the JS / MATLAB / Octave bindings are
**published manually** and are documented in full below.

## Binding → registry → automation

| Binding | Package | Registry | Automation | Trigger |
|---------|---------|----------|------------|---------|
| Python (full) | `nirs4all-methods` | PyPI | **Automated** — `release-wheels.yml` (cibuildwheel matrix + Trusted Publishing) | push tag `v*` (non-`-rc`) → PyPI; `workflow_dispatch` + `publish=true` |
| Python (slim) | `pls4all` | PyPI | **Automated** — same `release-wheels.yml` builds both packages in one matrix; the legacy `release-python.yml` is PR-only (smoke + sdist) and no longer publishes | push tag `v*` (non-`-rc`) → PyPI; `workflow_dispatch` + `publish=true` |
| R | `n4m` | CRAN | **Semi-automated** — `release-r.yml` vendors libn4m into `src/vendor/`, runs `R CMD check --as-cran` on the Linux/macOS/Windows + release/devel matrix, and (on tag push) attaches the tarball to the GitHub Release. **Submission is the irreducible manual web form.** | `workflow_dispatch`; tag push attaches the tarball |
| R | `pls4all` (slim) | CRAN | **Semi-automated** — same `release-r.yml`, the matrix has a `pkg: [n4m, pls4all]` leg. | `workflow_dispatch`; tag push attaches the tarball |
| JS / WASM | `@pls4all/wasm` | npm | **Build CI-automated** in `cross-binding-parity.yml` (emsdk pinned, `npm test` parity); **publish manual** (this doc) | — |
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

### JS → npm (`@pls4all/wasm`)

The npm package ships the Emscripten WASM module + the TypeScript wrappers; the
`npm run build` step only runs `tsc`, so the WASM artifacts must be built first.

```bash
# 0. Gates: scripts/bump_version.sh --check   (syncs bindings/js/package.json)

# 1. Build the WASM module (requires the Emscripten SDK on PATH).
source /path/to/emsdk/emsdk_env.sh
cmake --preset emscripten
cmake --build --preset emscripten --target pls4all_wasm
#    → build/emscripten/bindings/js/{p4a.js,p4a.wasm}

cd bindings/js
npm run build           # tsc -p .  → dist/index.js + dist/index.d.ts (TS only)

# 2. STAGE the WASM artifacts into dist/ — `npm run build` runs only tsc, and
#    package.json ships `files: ["dist/"]`, so the WASM must be copied in by
#    hand (there is no copy/prepublish script yet — this staging is unwired):
cp ../../build/emscripten/bindings/js/p4a.wasm dist/
cp ../../build/emscripten/bindings/js/p4a.js   dist/

# 3. Verify the tarball actually contains index.js + p4a.wasm + p4a.js BEFORE
#    publishing, and run the smoke test:
npm pack --dry-run      # inspect the file list
npm test                # node test/run_smoke.mjs — must pass

# 4. Publish (scoped public package; needs npm login + 2FA OTP).
npm publish --access public
```

Notes: the version is already correct if `bump_version.sh --check` passed (it
edits `package.json` + `package-lock.json`). The scope `@pls4all` must exist on
the npm org and the publisher must be a member. **The WASM-staging step (2) is
not yet scripted** — a `prepublishOnly` copy (or a CMake install rule) should be
added before this path is automated; until then verify step 3's file list every
time.

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
