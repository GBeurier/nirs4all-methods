# Development — Release Process

How each binding of `libn4m` is versioned, gated, and published. PyPI, npm,
CRAN-tarball build, source archives and the MATLAB/Octave GitHub Release archive
are automated; CRAN submission and optional registry/listing pages remain manual.

## Binding → registry → automation

| Binding | Package | Registry | Automation | Trigger |
|---------|---------|----------|------------|---------|
| Python (full) | `nirs4all-methods` | PyPI | **Automated** — `release-wheels.yml` (cibuildwheel matrix + Trusted Publishing) publishes the `nirs4all-methods` project | push tag `v*` (non-`-rc`) → PyPI; `workflow_dispatch` + `publish=true` |
| Python (slim) | `pls4all` | PyPI | **Automated** — `release-python.yml` (its own Trusted Publisher) publishes the `pls4all` project: sdist + cibuildwheel + retag-to-py3 + TestPyPI/PyPI + post-publish smoke. The two Python workflows are split one-per-PyPI-project (no collision), each with its own one-to-one Trusted Publisher. | push tag `v*` (non-`-rc`) → PyPI; `workflow_dispatch` + `publish=true` |
| R | `n4m` | CRAN | **Semi-automated** — `release-r.yml` vendors libn4m into `src/vendor/`, runs `R CMD check --as-cran` on the Linux/macOS/Windows + release/devel matrix, and (on tag push) attaches the tarball to the GitHub Release. **Submission is the irreducible manual web form.** | `workflow_dispatch`; tag push attaches the tarball |
| R | `pls4all` (slim) | CRAN | **Semi-automated** — same `release-r.yml`, the matrix has a `pkg: [n4m, pls4all]` leg. | `workflow_dispatch`; tag push attaches the tarball |
| JS / WASM | `@nirs4all/methods` | npm | **Automated** — `release-npm.yml` builds the pinned emsdk package, runs `npm test`, stages `dist/`, and publishes with npm provenance once `NPM_TOKEN`/scope are configured | tag push or `workflow_dispatch` with `publish=true` |
| MATLAB / Octave | `+n4m` | GitHub Release | **Automated** — `release-matlab.yml` attaches `nirs4all-methods-matlab-octave-<version>.zip` to the Release. ONE binding serves both: users build the MEX with `build_mex.m` (MATLAB) or `mkoctfile` via `build_mex.m` (Octave); see `bindings/matlab/COMPAT.md`. A File Exchange / Octave Forge listing is optional + manual. | tag push attaches the zip |
| Rust | `n4m` (official pre-release) | none — crates.io publication deferred | **Qualified, not published** — `rust-binding.yml` runs Linux/macOS/Windows C-ABI gates, declared-MSRV testing, and Linux sanitizer checks; it also verifies the packed license texts | push / pull request |

## Exact release artifacts — what each binding ships, and where to upload it

Every artifact below is also attached to the **GitHub Release** for the release
tag (`v<version>`) — so all of them are downloadable from one place.

| Binding | Registry | Exact file(s) | Upload |
|---|---|---|---|
| Python `nirs4all-methods` | PyPI | `nirs4all_methods-<version>-*.whl` (cibuildwheel matrix: Linux x86_64/aarch64, macOS x86_64/arm64, Windows x86_64) | **Automated** — Trusted Publishing, *no manual upload* |
| Python `pls4all` | PyPI | `pls4all-<version>-py3-none-*.whl` + `pls4all-<version>.tar.gz` (sdist) | **Automated** — Trusted Publishing, *no manual upload* |
| R `n4m` | CRAN | **`n4m_<version>.tar.gz`** (source tarball) | **Manual** — web form (see *R → CRAN* below) |
| R `pls4all` | CRAN | **`pls4all_<version>.tar.gz`** (source tarball) | **Manual** — web form |
| R `n4m` + `pls4all` | R-universe | — (built from Git, no upload) | **Automated** — registry repo + app (see *R → R-universe*) |
| JS / WASM `@nirs4all/methods` | npm | the staged `dist/` package (via `npm publish`) | **Automated** — `release-npm.yml` (needs `NPM_TOKEN` — see *JS → npm*) |
| MATLAB / Octave `+n4m` | GitHub Release | **`nirs4all-methods-matlab-octave-<version>.zip`** — the `bindings/matlab/` source (`+n4m` package + `build_mex.m` + MEX sources + tests). ONE package for both; download + run `build_mex.m`. | **Automated** — `release-matlab.yml` attaches it (File Exchange / Octave Forge optional + manual) |
| Source + provenance | GitHub Release | `nirs4all-methods-<version>-src.tar.gz` · `...-src.zip` · `nirs4all-methods-<version>.cdx.json` (SBOM) · `SHA256SUMS` | **Automated** — `release-source.yml` |

**For R/CRAN, upload the source `.tar.gz` only** — never a binary, the GitHub repo
zip, or the Python artifacts. The PyPI and npm files publish from CI (no manual
upload); they are listed here only so the GitHub Release carries every artifact.

## Pre-release gates (release blockers)

These three CI checks block any release; run them before tagging or publishing
anything (see `CLAUDE.md` → "Release / ABI gates"):

1. **Version sync** — `scripts/bump_version.sh --check`. The canonical version
   lives in `cpp/include/n4m/n4m_version.h`; the script syncs it into every
   active downstream manifest (`bindings/python/pyproject.toml`,
   `parity/python_generator/pyproject.toml`,
   `bindings/r/{n4m,pls4all}/DESCRIPTION`,
   `bindings/r/{n4m,pls4all}/cran-comments.md`, `bindings/js/package.json`,
   `bindings/js/package-lock.json`, `CITATION.cff`, `docs/conf.py`, and the
   README citation block). It also verifies the Python loader ABI constants.
   MATLAB/Octave read the version from libn4m at runtime, so they have no
   manifest to sync; archived bindings under `bindings/_archive/` are frozen
   and excluded. **Bump with**
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

## Publication prerequisites and manual channels

The **MATLAB/Octave** package is attached to the GitHub Release automatically by
`release-matlab.yml` (`nirs4all-methods-matlab-octave-<version>.zip`); the optional
File Exchange / Octave Forge *listing* is still manual. The **JS/WASM** package is
built + published to npm by `release-npm.yml` (`@nirs4all/methods`). Each
binding's artifact is built from the **same `libn4m` source** at the released
version; always run the pre-release gates first.

### JS → npm (`@nirs4all/methods`)

The npm package ships the Emscripten WASM module + the TypeScript wrappers; the
`npm run build` step only runs `tsc`, so the WASM artifacts must be built first.

```bash
# 0. Gates: scripts/bump_version.sh --check   (syncs bindings/js/package.json)

# 1. Build the WASM module (requires the Emscripten SDK on PATH).
source /path/to/emsdk/emsdk_env.sh
cmake --preset emscripten
cmake --build --preset emscripten --target n4m_wasm
#    → build/emscripten/bindings/js/{n4m.js,n4m.wasm}

cd bindings/js
npm run build           # tsc -p .  → dist/index.js + dist/index.d.ts (TS only)

# 2. STAGE the WASM artifacts into dist/ — `npm run build` runs only tsc, and
#    package.json ships `files: ["dist/"]`. The `stage:wasm` script copies the
#    Emscripten artifacts in (and `prepack` runs build + stage:wasm for you):
npm run stage:wasm      # → dist/n4m.js + dist/n4m.wasm

# 3. Verify the tarball actually contains index.js + n4m.wasm + n4m.js BEFORE
#    publishing, and run the JS/WASM smoke suite:
npm pack --dry-run      # inspect the file list
npm test                # PLS parity + API/generic/AOM/new-pack smokes — must pass

# 4. Publish manually only for an emergency/local dry-run path.
#    The release path is release-npm.yml with NPM_TOKEN + provenance.
npm publish --access public
```

Notes: the version is already correct if `bump_version.sh --check` passed (it
edits `package.json` + `package-lock.json`). The scope `@nirs4all` must exist on
the npm org and the publisher must be a member.

#### One-time npm registration (required for the automated `release-npm.yml`)

`release-npm.yml` publishes `@nirs4all/methods` automatically on a non-`-rc`
`v*` tag — it builds the WASM (pinned `setup-emsdk`), then runs `npm publish`
with `NODE_AUTH_TOKEN=${{ secrets.NPM_TOKEN }}` + provenance. **Until the scope +
token below exist, that one leg fails** (the PyPI / R / source legs are
independent and still succeed). To enable it:

1. **Own the `@nirs4all` scope** on [npmjs.com](https://www.npmjs.com) — sign in
   as the maintainer, *Add Organization* → create the free org `nirs4all` (so the
   scope `@nirs4all` is yours). The publishing account must be a member, and the
   package name `@nirs4all/methods` must be free/owned.
2. **Generate an automation token** — npmjs.com → *Access Tokens* → *Generate New
   Token* → **Granular Access** (or **Automation**), granting **Read and write**
   on the `@nirs4all` scope / `@nirs4all/methods` package. Copy it.
3. **Add it as a GitHub Actions secret** — repo *Settings → Secrets and variables
   → Actions → New repository secret*, name **`NPM_TOKEN`**, value = the token.
4. **Publish** — either dispatch `release-npm.yml` (*Run workflow* →
   `publish=true`) for the current release tag, or let it publish automatically
   on the next non-`-rc` tag.

`release-npm.yml` requests `id-token: write`, so once the scope + token exist the
package publishes with a verified npm provenance attestation. The WASM staging is
handled by the package's `prepack` script in CI — no manual step.

### MATLAB → File Exchange (`+n4m`)

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
'n4m.mltbx')`. CI does **not** run MATLAB (no licensed runner); confirm
MATLAB-specific behaviour against `bindings/matlab/COMPAT.md` before publishing.
The `+n4m` package reads its version from libn4m at runtime.

### Octave (`n4m`)

The Octave surface is the same `bindings/matlab/` binding (built for the
MATLAB ∩ Octave intersection); Octave builds the MEX via `build_mex.m` /
`mkoctfile` — there is **no** dedicated CMake Octave target. The build is
**CI-tested on every push** (the `octave-mex` job in
`cross-binding-parity.yml` installs apt Octave, runs `build_mex.m`, and gates
on `test_parity` and `test_n4m_namespace`; locally observed `test_parity`
`rmse_rel <= 1e-12`. What is
still manual is **publication** (no Octave Forge submission is wired): ship
the built binding with the GitHub Release, or have users build it locally
against the released libn4m.

---

## What still needs manual action

The remaining manual work is external-channel administration: CRAN web-form
submission for the generated R tarballs, owning/configuring the npm `@nirs4all`
scope and `NPM_TOKEN`, and optional MATLAB File Exchange / Octave Forge listings
or `.mltbx` packaging. The build/test/publish workflows for JS and the GitHub
Release archive for MATLAB/Octave are automated. MATLAB itself stays out of CI
entirely (no licensed runner); the Octave job validates the MATLAB/Octave
intersection per `bindings/matlab/COMPAT.md`.

---

## R → R-universe (registration)

R-universe builds binaries (Windows/macOS/Linux) straight from Git — no review,
no submission. Users then
`install.packages("n4m", repos = "https://gbeurier.r-universe.dev")`.

- **Registry repo** (done): public
  **<https://github.com/GBeurier/GBeurier.r-universe.dev>** with `packages.json`
  listing both packages by `subdir`:
  ```json
  [
    { "package": "n4m",     "url": "https://github.com/GBeurier/nirs4all-methods", "subdir": "bindings/r/n4m" },
    { "package": "pls4all", "url": "https://github.com/GBeurier/nirs4all-methods", "subdir": "bindings/r/pls4all" }
  ]
  ```
  No `branch` field → it tracks `main` (the R packages + `.prepare` hooks are on
  `main` post-merge).
- **GitHub App** (one manual browser step — cannot be scripted): install
  **<https://github.com/apps/r-universe>** on the `GBeurier` account (grant *All
  repositories*, or at least `GBeurier.r-universe.dev` + `nirs4all-methods`). It is
  required — it lets R-universe build the universe and post status. Builds
  (re)trigger on any commit to the registry repo and auto-rebuild every 30 days.
- **Self-contained build** (already wired): R-universe clones the whole monorepo
  and runs `bindings/r/{n4m,pls4all}/.prepare` *before* `R CMD build` (while the
  top-level `cpp/` is on disk); the hook runs `N4M_R_VENDOR=1 ./configure` to
  vendor the core into `src/vendor/` → a self-contained ~1 MB tarball (233 TUs,
  far under R-universe's 100 MB cap).
- **Verify**: watch <https://gbeurier.r-universe.dev> (it *shows* the `R CMD check`
  result but, unlike CRAN, does not block on a NOTE/WARNING).

## R → CRAN (submission)

CRAN is the canonical R repo (`install.packages("n4m")` with default repos);
submission is a **manual web form** with human review.

### Build + check (automated)
`release-r.yml` (tag push, or `workflow_dispatch`) runs `R CMD check --as-cran`
for both packages across `{n4m, pls4all} × {linux-release, linux-devel,
macos-arm64-release, windows-release}`, re-vendors fresh via `.prepare`, and — on
a non-`-rc` `v*` tag — attaches `n4m_<version>.tar.gz` + `pls4all_<version>.tar.gz` to
the GitHub Release. Build/attach are **gated on `--as-cran` passing**. Verified
locally: both packages **0 ERROR / 0 WARNING** (4 expected NOTEs).

Optional pre-submission smoke (recommended for a first submission):
```r
devtools::check_win_devel("bindings/r/n4m")   # and bindings/r/pls4all
devtools::check_mac_release("bindings/r/n4m")
rhub::rhub_check("bindings/r/n4m")
```

### Exactly what to upload
The **R source tarball only** — one submission per package:
- **`n4m_<version>.tar.gz`**
- **`pls4all_<version>.tar.gz`**

Get them from the GitHub Release assets for `v<version>`:
the files are named exactly **`n4m_<version>.tar.gz`** and **`pls4all_<version>.tar.gz`**
(attached by `release-r.yml`). There is **no `-cran-tarball` suffix on the Release** —
download `n4m_<version>.tar.gz` and `pls4all_<version>.tar.gz` directly and upload those.

> `<pkg>-cran-tarball` is only the GitHub **Actions artifact** name that wraps the
> *same* `<pkg>_<version>.tar.gz` on a `release-r.yml` *Run workflow* run — relevant
> only if you download from the Actions run instead of the Release.

You can also build the tarball locally
(`cd bindings/r/n4m && sh .prepare && R CMD build .`).

### The submission form — exact values
At **<https://cran.r-project.org/submit.html>** (submit `n4m` first, then `pls4all`):

| Field | Value |
|---|---|
| Your name | `Gregory Beurier` |
| Your email | `gregory.beurier@cirad.fr` (matches the `Maintainer:` in DESCRIPTION) |
| Upload | the package's `.tar.gz` |
| Optional comment | paste the matching block below |

`cran-comments.md` is `.Rbuildignore`d (not in the tarball), so the comment box is
where these notes go:

**— paste for `n4m` —**
```text
New submission.

n4m <version> — a portable Partial Least Squares (PLS) and Near-Infrared
Spectroscopy (NIRS) engine. The C++17/C/Fortran numerical core (233 vendored
translation units under src/vendor/) is compiled from source at install time;
no external system library is required. License: CeCILL-2.1 (a GPL-compatible
French free-software license, in R's license database). Imports: stats only.

SystemRequirements: GNU make — the Makevars use $(shell find ...) to enumerate
the vendored sources without hard-coding each filename.

Test environments:
- R CMD check --as-cran, R 4.3.3 (Ubuntu): 0 ERRORs, 0 WARNINGs (vignettes built,
  checkbashisms installed); 4 expected NOTEs only.
- CI matrix (R release + devel) via GitHub Actions: Ubuntu 22.04, macOS 14
  (arm64), Windows Server 2022.
- win-builder (devel + release) and R-hub v2 are run before submission.

Notes (all expected):
- New submission (first upload).
- GNU make is declared in SystemRequirements.
- Any 'compilation flags' NOTE on a local build comes from the host R's Makeconf
  (e.g. conda's -march=nocona), not from the package Makevars, which use no
  -O3 / -march=native / -Werror.
- The optional CUDA backend (cuda_dispatch.cpp) is intentionally excluded from
  the R build; the package exposes the portable scalar code path only.

Maintainer: Gregory Beurier <gregory.beurier@cirad.fr> (CIRAD).
```

**— paste for `pls4all` —**
```text
New submission.

pls4all <version> — a portable Partial Least Squares engine for chemometrics: the
slim, PLS-focused distribution carved from the nirs4all-methods library. The
C++17/C/Fortran numerical core (233 vendored translation units under src/vendor/)
is compiled from source at install time; no external system library is required.
License: CeCILL-2.1 (a GPL-compatible French free-software license, in R's
license database). Imports: stats only.

SystemRequirements: GNU make — the Makevars use $(shell find ...) to enumerate
the vendored sources without hard-coding each filename.

Test environments:
- R CMD check --as-cran, R 4.3.3 (Ubuntu): 0 ERRORs, 0 WARNINGs (vignettes built,
  checkbashisms installed); 4 expected NOTEs only.
- CI matrix (R release + devel) via GitHub Actions: Ubuntu 22.04, macOS 14
  (arm64), Windows Server 2022.
- win-builder (devel + release) and R-hub v2 are run before submission.

Notes (all expected):
- New submission (first upload).
- GNU make is declared in SystemRequirements.
- Any 'compilation flags' NOTE on a local build comes from the host R's Makeconf
  (e.g. conda's -march=nocona), not from the package Makevars, which use no
  -O3 / -march=native / -Werror.
- The optional CUDA backend (cuda_dispatch.cpp) is intentionally excluded from
  the R build; the package exposes the portable scalar code path only.

Maintainer: Gregory Beurier <gregory.beurier@cirad.fr> (CIRAD).
```

After uploading, CRAN emails a confirmation link — click it to complete. For a new
package, expect the automated incoming checks then a human reviewer.

## Operational notes (lessons from the 0.99.0 multi-registry release)

- **npm needs an Automation token.** A regular/Granular token without "bypass 2FA"
  fails `npm publish` with `403 — 2FA required` (or `E404` on the scoped PUT if it
  lacks `@nirs4all` org write). Use a **classic Automation** token (or a Granular
  token with read+write on the `@nirs4all` org) as the `NPM_TOKEN` secret.
- **`release-npm.yml` dispatch defaults to a dry run.** Its `publish` input is
  `default: false` — pass `-f publish=true` to actually publish:
  `gh workflow run release-npm.yml --ref main -f publish=true`.
- **Re-run a publish on `main`, not by replaying an old tag.** `gh run rerun <id>`
  on a tag-triggered run replays the *tagged* commit; re-running the `v0.99.0` npm
  run failed with `No such build preset: emscripten` because that tag predates the
  preset. Dispatch the workflow on `main` instead. (`gh run rerun` also has no
  `--failed` flag in current gh.)
- **`pls4all` Trusted Publisher** = repo `nirs4all-methods`, workflow
  `release-python.yml`, env `pypi` (NOT the historical `GBeurier/pls4all` repo).
- **`vX.Y.Z` tags publish only if non-pre-release** (no `-`); the PyPI/Release jobs
  gate on `!contains(ref, '-')`.
