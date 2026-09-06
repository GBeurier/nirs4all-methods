# cran-comments.md

## Submission summary

* This is a **new submission** to CRAN.
* `pls4all` 1.0.19 — a portable Partial Least Squares engine for chemometrics:
  the slim, PLS-focused distribution carved from the nirs4all-methods library
  (every method built on the shared PLS core). The C++17/C numerical
  core (222 translation units) is vendored under `src/vendor/` and compiled
  from source at install time. The package contains **no Fortran**: it is a
  pure C/C++ build, so no Fortran toolchain is needed and no Fortran flags are
  set. No external system library is required.
* License: `CeCILL (== 2.1)` (a GPL-compatible French free-software
  license, listed in R's license database).
* The same numerical core powers the project's Python (PyPI),
  JavaScript / WebAssembly (npm), Julia, MATLAB and other bindings.
  Each binding is parity-gated against shared `parity_fixture.json`
  fixtures to `rmse_rel < 1e-12`.

## Test environments

* Local development check (Ubuntu/WSL2, conda R 4.6.0): `R CMD check --as-cran
  --no-manual` → **0 errors, 0 warnings, 2 NOTEs**. The two NOTEs are
  (a) "New submission" and (b) a single non-portable compilation flag
  `-march=nocona` that comes from conda-forge R's own `Makeconf`, not from the
  package — neither originates from the package. See "Known notes" below.
* Submission-grade checks are run on **current R (release + devel)** before
  upload, via the GitHub Actions CI matrix (`.github/workflows/release-r.yml`):
  - Ubuntu 22.04 (R release + devel)
  - macOS 14 (R release, arm64)
  - Windows Server 2022 (R release)
* win-builder (`devtools::check_win_devel()` / `check_win_release()`) and
  R-hub v2 (`rhub::rhub_check()`) are run manually before each CRAN
  submission; their results are attached to the matching GitHub Release.

## Known notes (all expected)

* **GNU make is a SystemRequirements** — the package's Makevars uses
  `$(shell find ...)` and pattern substitution to enumerate the 222
  vendored C/C++ sources without hard-coding each filename. GNU make
  is declared in `SystemRequirements`.
* **New submission** — first upload.
* **No Fortran, no non-default compilation flags** — the package is a pure
  C/C++ build. It vendors and compiles only C/C++ sources and sets no
  non-default compilation flag (no `PKG_FFLAGS`, no `-std=legacy`). The
  spline-smoothing augmenter uses a from-scratch C cubic smoothing-spline
  (Reinsch) implementation, so the build needs no Fortran toolchain. The only
  non-portable flag in the install log, `-march=nocona`, is injected by
  conda-forge R's own `Makeconf`, not by the package.

## Compile time

The package vendors and compiles 222 C/C++ translation units. On a
typical CRAN check farm the install takes ~3–5 minutes. Per CRAN policy this
is within the acceptable budget.

## Anti-patterns avoided

* No `-O3`, `-march=native`, `-Werror`, LTO, or `PKG_FFLAGS` flags in the
  Makevars. The package sets no non-default compilation flag at all (it is a
  pure C/C++ build with no Fortran).
* No `printf` / `cout` / `Rprintf` from C++ paths during default
  execution.
* No internet, filesystem write outside `tempdir()`, or shell
  invocation during examples / tests / vignettes.
* No `:::` calls to private functions of other packages.
* No reverse-dependency footprint — the package is a leaf in the CRAN
  dependency graph (only imports `stats`).

## Tier-2 integrations parked outside CRAN

The original draft of this package also shipped optional tidymodels
(`parsnip`) and mlr3 (`R6`) integrations. Both are archived under
`bindings/r/archive/parsnip-mlr3/` and are not part of this submission.
The active R surface is PLS/chemometrics-first: base formula/S3, `pls`-style
`plsr()` / `pcr()`, and `mdatools`-style matrix `pls(x, y, ...)`.

## Reviewer-facing notes

* The vendored C/C++ sources are a textual copy of `cpp/include/`
  and `cpp/src/` from the project's GitHub repository at the tag
  matching this version, with one directory omitted: the optional vendored
  FITPACK Fortran (`cpp/src/core/common/_vendored/fitpack/`) is **not**
  shipped, because the R build selects the package's own from-scratch C cubic
  smoothing-spline (Reinsch) path. The sync is automated via
  `scripts/bump_version.sh` and verified by the
  `.github/workflows/version-sync.yml` workflow on every PR.
* The `CUDA` backend file `cuda_dispatch.cpp` is intentionally
  excluded from the R-package build (filtered out in `src/Makevars` on
  Unix and `src/Makevars.win` on Windows).
  CUDA is an optional accelerated backend in the standalone library,
  not required for the reference scalar code path that the R package
  exposes.
* Author / maintainer is Grégory Beurier (CIRAD,
  `gregory.beurier@cirad.fr`). The "pls4all contributors" `ctb` entry
  refers to upstream contributors listed in the project's
  `CITATION.cff` and CONTRIBUTORS.md (in the source tree).

Thank you for the review!
