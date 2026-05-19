# cran-comments.md

## Submission summary

* This is a **new submission** to CRAN.
* `pls4all` 0.97.0 — a portable Partial Least Squares (PLS) and
  Near-Infrared Spectroscopy (NIRS) engine. The C++17 numerical core is
  vendored under `src/libp4a/` and compiled from source at install time.
  No external system library is required.
* License: `CeCILL (== 2.1)` (a GPL-compatible French free-software
  license, listed in R's license database).
* The same numerical core powers the project's Python (PyPI),
  JavaScript / WebAssembly (npm), Julia, MATLAB and other bindings.
  Each binding is parity-gated against shared `parity_fixture.json`
  fixtures to `rmse_rel < 1e-12`.

## Test environments

* local Ubuntu 22.04, R 4.6.0, GCC 15.2 (conda-forge): `R CMD check
  --as-cran` → **0 errors, 0 warnings, 3 notes** (see "Known notes"
  below for the remaining notes).
* GitHub Actions CI (`.github/workflows/release-r.yml`):
  - Ubuntu 22.04 (R release + devel)
  - macOS 14 (R release, arm64)
  - Windows latest (R release)
* Pre-submission R-hub v2 / win-builder / macbuilder runs are
  invoked manually before each CRAN submission via
  `rhub::rhub_check()`, `devtools::check_win_devel()` and
  `devtools::check_mac_release()`. Results attached to the matching
  GitHub Release.

## Known notes (all expected)

* **GNU make is a SystemRequirements** — the package's Makevars uses
  `$(wildcard ...)` and pattern substitution to enumerate the ~130
  vendored C++ sources without hard-coding each filename. GNU make is
  declared in `SystemRequirements`.
* **New submission** — first upload.
* **Compilation flags** — local conda-forge R sets `-march=nocona`;
  this comes from R's own Makeconf, not from the package's Makevars.
  Not present in CRAN-farm builds.
* **HTML tidy unavailable locally** — the local check host does not
  have the optional `tidy` executable, so HTML validation is skipped.
  The package manual and PDF manual both build successfully.

## Compile time

The package vendors and compiles ~130 C++ files (about 1500 SLOC
each on average). On a typical CRAN check farm the install takes
~3–5 minutes. Per CRAN policy this is within the acceptable budget.

## Anti-patterns avoided

* No `-O3`, `-march=native`, `-Werror`, or other non-portable compiler
  flags in the package's Makevars.
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
The active R surface is now NIRS-first: base formula/S3, `pls`-style
`plsr()` / `pcr()`, and `mdatools`-style matrix `pls(x, y, ...)`.

## Reviewer-facing notes

* The vendored C++ sources are an exact textual copy of `cpp/include/`
  and `cpp/src/` from the project's GitHub repository at the tag
  matching this version. The sync is automated via
  `scripts/bump_version.sh` and verified by the
  `.github/workflows/version-sync.yml` workflow on every PR.
* The `CUDA` backend file `cuda_dispatch.cpp` is intentionally
  excluded from the R-package build (filtered out in `src/Makevars`).
  CUDA is an optional accelerated backend in the standalone library,
  not required for the reference scalar code path that the R package
  exposes.
* Author / maintainer is Grégory Beurier (CIRAD,
  `gregory.beurier@cirad.fr`). The "pls4all contributors" `ctb` entry
  refers to upstream contributors listed in the project's
  `CITATION.cff` and CONTRIBUTORS.md (in the source tree).

Thank you for the review!
