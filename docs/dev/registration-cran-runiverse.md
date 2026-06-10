# Registering the R packages on R-universe and CRAN

Step-by-step for publishing the two R packages. Do **R-universe first** (minutes,
no human review, validates the cross-platform build), then **CRAN** (the
canonical repo, with review).

## Package map

| Distribution | Full engine | Slim PLS subset |
|---|---|---|
| **R** (CRAN / R-universe) | **`n4m`** | **`pls4all`** |
| Python (PyPI — already configured) | `nirs4all-methods` | `pls4all` |

Both R packages live in subdirectories of the monorepo and are **self-contained**:
their `./configure` vendors the libn4m C/C++/Fortran core into `src/vendor/` and
compiles a standalone `n4m.so`/`.dll` — no external libn4m at install or runtime.
Source of truth for versions: `cpp/include/n4m/n4m_version.h` (currently 0.99.0).

---

## 1. R-universe (`https://GBeurier.r-universe.dev`)

R-universe builds binaries (Windows/macOS/Linux) straight from Git — no review,
no submission. Users then `install.packages(..., repos = "https://GBeurier.r-universe.dev")`.

### 1a. Create the registry repository — ✓ DONE
Created (public): **https://github.com/GBeurier/GBeurier.r-universe.dev**, with
`packages.json` + `README.md` already pushed. (The name must be
`{ghuser}.r-universe.dev`.)

### 1b. `packages.json` — ✓ DONE
Already committed at the registry repo root:
```json
[
  { "package": "n4m",     "url": "https://github.com/GBeurier/nirs4all-methods", "subdir": "bindings/r/n4m" },
  { "package": "pls4all", "url": "https://github.com/GBeurier/nirs4all-methods", "subdir": "bindings/r/pls4all" }
]
```
- No `branch` field → it tracks the repo's **default branch** (`main`), so the R
  packages + the `.prepare` hooks must be **merged to `main`** before R-universe
  can build. To validate from the feature branch *before* merging, add
  `"branch": "release-readiness-fixes"` to each entry (and remove it after merge).

### 1c. Install the R-universe GitHub App — 👤 YOU (required; cannot be automated)
This is the **one manual step that must be done in the browser** — installing a
GitHub App is an interactive authorization, so it cannot be scripted. It is
**required**: the app is what lets R-universe build your universe and post the
build status. Steps:
1. Go to **https://github.com/apps/r-universe** → **Install** (or **Configure**).
2. Choose the **GBeurier** account.
3. Grant access to **All repositories** (simplest), or at minimum to both
   `GBeurier.r-universe.dev` *and* `nirs4all-methods`.
4. Confirm. Within a few minutes the universe appears at
   **https://gbeurier.r-universe.dev** and the first build starts (builds are
   (re)triggered by any commit to the registry repo, and auto-rebuild every 30 days).

> If `github.com/apps/r-universe` 404s, search "r-universe" in the GitHub
> Marketplace, or follow the install link from
> [docs.r-universe.dev](https://docs.r-universe.dev/publish/set-up.html).

### 1d. How the build is made self-contained (already wired)
R-universe clones the **whole monorepo** and runs a pre-build hook,
`bindings/r/{n4m,pls4all}/.prepare`, **before `R CMD build`** while the top-level
`cpp/` tree is still on disk. That hook runs `N4M_R_VENDOR=1 ./configure`, which
vendors the core into `src/vendor/`; the resulting source tarball is fully
self-contained (validated locally: 233 vendored TUs, ~1 MB tarball — far under
R-universe's 100 MB cap). **Nothing else to configure** — the hook is committed.

### 1e. Verify and install
Watch the build at `https://GBeurier.r-universe.dev` (each package gets a status
page with the `R CMD check` result — note R-universe **shows** check results but,
unlike CRAN, does **not** block publication on a NOTE/WARNING). Then:
```r
install.packages("n4m",     repos = c("https://GBeurier.r-universe.dev", "https://cloud.r-project.org"))
install.packages("pls4all", repos = c("https://GBeurier.r-universe.dev", "https://cloud.r-project.org"))
```

---

## 2. CRAN (`n4m` and `pls4all`)

CRAN is the canonical R repository (users just `install.packages("n4m")`), but
submission is a **manual web form** with human review.

### 2a. Build the tarballs + run the CRAN check matrix
Trigger the **`release-r.yml`** workflow (`workflow_dispatch`, or a `v0.99.0`
tag push). It runs `R CMD check --as-cran` for **both** packages across
Linux/macOS/Windows, builds `n4m_0.99.0.tar.gz` + `pls4all_0.99.0.tar.gz` (each
re-vendored fresh via `N4M_R_VENDOR=1 ./configure`), uploads them as run
artifacts, and — on a non-`-rc` `v*.*.*` tag — attaches them to the GitHub
Release. **The build-tarball/attach jobs are gated on the `--as-cran` check
passing**, so a check failure ⇒ no tarball: watch that job (it is the first
authoritative `--as-cran` run of the new `coef.n4m_fit` + the `r_p4a_→r_n4m_`
rename).

Independent pre-submission smoke (recommended for a first submission):
```r
# win-builder (returns a check log by email)
devtools::check_win_devel("bindings/r/n4m")      # and bindings/r/pls4all
# macOS builder
devtools::check_mac_release("bindings/r/n4m")
# or R-hub v2
rhub::rhub_check("bindings/r/n4m")
```

### 2b. cran-comments
Each package ships `bindings/r/<pkg>/cran-comments.md` — update the check-results
summary (platforms tested, 0 ERRORs/0 WARNINGs, any expected NOTEs) before submitting.

### 2c. Exactly what to upload
The **R source tarball** — one submission per package, the `.tar.gz` only (NOT a
binary, NOT the GitHub repo zip, NOT the PyPI/Python artifacts):

- `n4m_0.99.0.tar.gz`
- `pls4all_0.99.0.tar.gz`

Get each from any of:
- the **GitHub Release `v0.99.0`** assets (attached automatically by `release-r.yml`), or
- a manual **`release-r.yml` → "Run workflow"** run → download the
  `n4m-cran-tarball` / `pls4all-cran-tarball` artifacts, or
- locally: `cd bindings/r/n4m && sh .prepare && R CMD build .` → `n4m_0.99.0.tar.gz`
  (same in `bindings/r/pls4all`).

### 2d. The submission form — exact values
At **https://cran.r-project.org/submit.html** (submit `n4m` first, then `pls4all`
as a separate submission):

| Form field | Value |
|---|---|
| Your name | `Gregory Beurier` |
| Your email | `gregory.beurier@cirad.fr` — must match the `Maintainer:` in DESCRIPTION (it does) |
| Upload package | the package's `.tar.gz` |
| Optional comment | paste the matching block below |

`cran-comments.md` is **not** shipped inside the tarball (it is `.Rbuildignore`d),
so the form's **comment box is where these notes must go**:

**— paste for `n4m` —**
```text
New submission.

n4m 0.99.0 — a portable Partial Least Squares (PLS) and Near-Infrared
Spectroscopy (NIRS) engine. The C++17/C/Fortran numerical core (233 vendored
translation units under src/vendor/) is compiled from source at install time;
no external system library is required. License: CeCILL-2.1 (a GPL-compatible
French free-software license, in R's license database). Imports: stats only.

SystemRequirements: GNU make — the Makevars use $(shell find ...) to enumerate
the vendored sources without hard-coding each filename.

Test environments:
- R CMD check --as-cran, R 4.3.3 (Ubuntu): 0 ERRORs. The only WARNINGs seen
  locally are 'checkbashisms' not being installed and a skipped-vignette
  artifact; both are absent on the CRAN build farm.
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

pls4all 0.99.0 — a portable Partial Least Squares engine for chemometrics: the
slim, PLS-focused distribution carved from the nirs4all-methods library. The
C++17/C/Fortran numerical core (233 vendored translation units under src/vendor/)
is compiled from source at install time; no external system library is required.
License: CeCILL-2.1 (a GPL-compatible French free-software license, in R's
license database). Imports: stats only.

SystemRequirements: GNU make — the Makevars use $(shell find ...) to enumerate
the vendored sources without hard-coding each filename.

Test environments:
- R CMD check --as-cran, R 4.3.3 (Ubuntu): 0 ERRORs. The only WARNINGs seen
  locally are 'checkbashisms' not being installed and a skipped-vignette
  artifact; both are absent on the CRAN build farm.
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

After uploading, CRAN emails a confirmation link — click it to complete the
submission. For a new package, expect the automated incoming checks, then a human
reviewer.

> CRAN ≠ the GitHub Release: the tag produces the tarballs as downloadable Release
> assets, but **you** upload them to CRAN via this form — there is no automated
> CRAN submission.

---

## Recommended order

1. Merge `release-readiness-fixes` → `main` (so `main` carries the R packages + the `.prepare` hooks).
2. **R-universe** (1a–1e) — fast, validates the Win/macOS/Linux builds publicly.
3. Tag `v0.99.0` (also drives PyPI + the R tarball assets + source archive).
4. **CRAN** (2a–2c) — submit the two tarballs once R-universe + the `--as-cran` matrix are green.

See also: [`release_process.md`](release_process.md) (full cut-day sequence),
[`DISTRIBUTION.md`](DISTRIBUTION.md) (distribution architecture).
