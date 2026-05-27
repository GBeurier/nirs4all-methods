# Development — Release Process

How each binding of `libn4m` is versioned, gated, and published. Some paths are
automated (PyPI, CRAN-tarball build); the JS / MATLAB / Octave bindings are
**published manually** and are documented in full below.

## Binding → registry → automation

| Binding | Package | Registry | Automation | Trigger |
|---------|---------|----------|------------|---------|
| Python (slim) | `pls4all` | PyPI | **Automated** — `release-python.yml` (cibuildwheel + Trusted Publishing/OIDC) | push tag `v*` (TestPyPI on `-rc*`) |
| Python (full) | `nirs4all-methods` | PyPI | Scaffolded, **inactive** — `release-wheels.yml` (needs the full package's build glue) | `workflow_dispatch` |
| R | `n4m` | CRAN | **Semi-automated** — `release-r.yml` builds + `R CMD check`s the tarball; **submission is a manual web form** | `workflow_dispatch` (manual-only until the package builds standalone — see the note atop `release-r.yml`) |
| JS / WASM | `@pls4all/wasm` | npm | **Manual** (this doc) | — |
| MATLAB | `+pls4all` | File Exchange | **Manual** (this doc) | — |
| Octave | `pls4all` (pkg) | — / Octave Forge | **Manual** (this doc) | — |

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

- **PyPI (`pls4all`)** — tag `vX.Y.Z` → `release-python.yml` builds wheels with
  cibuildwheel, repairs them (auditwheel/delocate/delvewheel), smoke-tests, and
  publishes via PyPI Trusted Publishing (no stored token). `-rcN` tags go to
  TestPyPI. One-time Trusted-Publisher config is documented in the workflow header.
- **CRAN (`n4m`)** — `release-r.yml` (`workflow_dispatch`) builds the source
  tarball and runs `R CMD check --as-cran` across the CRAN matrix, then attaches
  the tarball to the GitHub Release. **CRAN submission itself is the irreducible
  manual step**: download the tarball from the run and submit it at
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
`mkoctfile` — there is **no** dedicated CMake Octave target. Distribution is
manual (no Octave Forge submission is wired — Forge packaging/review is a
heavyweight manual process): ship the built binding with the GitHub Release, or
have users build it locally against the released libn4m. Validate with the
`n4m_smoke_test` smoke documented in `bindings/matlab/COMPAT.md` — run manually;
it is **not** in CI today (despite COMPAT.md describing it as the intended CI
path).

---

## Why JS / MATLAB / Octave are manual today

Automating these (an `npm publish` job, a `.mltbx` build job) is tracked as
binding-CI work, not done here: their CI-buildable packaging is itself
unfinished (the R package's `Makevars` still references the pre-rename vendored
tree; the Octave/MATLAB MEX needs an equivalent CI build step; the JS WASM build
needs the Emscripten SDK provisioned in CI). Until those land, the artifacts are
built locally and published by hand as above. See also the cross-binding parity
gate's scope note in `.github/workflows/cross-binding-parity.yml`.
