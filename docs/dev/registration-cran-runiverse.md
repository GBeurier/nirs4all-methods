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

### 1a. Create the registry repository
Create a **public** GitHub repo named **exactly** `GBeurier.r-universe.dev` (the
name must be `{ghuser}.r-universe.dev`).

### 1b. Add `packages.json` at its root
```json
[
  { "package": "n4m",     "url": "https://github.com/GBeurier/nirs4all-methods", "subdir": "bindings/r/n4m" },
  { "package": "pls4all", "url": "https://github.com/GBeurier/nirs4all-methods", "subdir": "bindings/r/pls4all" }
]
```
- `branch` is optional and defaults to the repo's **default branch** (`main`) — so
  the R packages (and the `.prepare` hook below) must be **merged to `main`** first.
  To test from a feature branch before merging, add e.g. `"branch": "release-readiness-fixes"`.

### 1c. Install the R-universe GitHub App
On your GitHub account, install the **R-universe** app (it only needs
read/write commit-status permission, so it can post the green build check).

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

### 2c. Submit
Upload the tarball at **https://cran.r-project.org/submit.html** — `n4m` first,
then `pls4all`. Both are **new packages**, so expect extra scrutiny: respond to
the automated incoming checks, then the manual reviewer. A new submission that
declares a self-contained native build (no `SystemRequirements: libn4m`) and
passes the `--as-cran` matrix is the smooth path.

> CRAN ≠ the GitHub Release: the tag produces the tarballs as downloadable Release
> assets, but **you** upload them to CRAN via the web form — there is no automated
> CRAN submission.

---

## Recommended order

1. Merge `release-readiness-fixes` → `main` (so `main` carries the R packages + the `.prepare` hooks).
2. **R-universe** (1a–1e) — fast, validates the Win/macOS/Linux builds publicly.
3. Tag `v0.99.0` (also drives PyPI + the R tarball assets + source archive).
4. **CRAN** (2a–2c) — submit the two tarballs once R-universe + the `--as-cran` matrix are green.

See also: [`release_process.md`](release_process.md) (full cut-day sequence),
[`DISTRIBUTION.md`](DISTRIBUTION.md) (distribution architecture).
