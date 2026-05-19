# Distribution Plan — pls4all

> **Scope.** This document is the single source of truth for *where*
> `pls4all` is published, *what* gets shipped to each surface, and
> *how* a release reaches it. It complements `ROADMAP.md` (what we
> build) and `docs/dev/release_process.md` (which steps a maintainer
> walks through on cut-day). All technical specifications below refer
> to the current ABI (`P4A_ABI_VERSION = 1.16.0`) and project version
> (`0.97.0` at the time of writing — see `cpp/include/pls4all/p4a_version.h`).
>
> **Audience.** Maintainers + future contributors. If you are reading
> this to add a *new* binding, jump to [§6 New-channel checklist](#6-new-channel-checklist).
>
> **License gate.** Everything ships under `CeCILL-2.1` (SPDX:
> `CECILL-2.1`, present on the SPDX License List since v3.0).
> Registries surveyed below accept `CECILL-2.1` either as an SPDX
> license expression or, where expression validation is not available,
> by packaging the full license text as a license file. Use the
> **canonical SPDX casing `CECILL-2.1`** in every non-R manifest
> (Cargo, npm, NuGet, vcpkg, RubyGems, conda recipes, SPDX headers).
> Do **not** use NuGet `licenseUrl`; it is deprecated — use
> `<PackageLicenseExpression>CECILL-2.1</PackageLicenseExpression>`.
> We do **not** dual-license: CeCILL-2.1 is already GPL-compatible
> and recognised by French law.
>
> **Casing audit (done 2026-05-18 in M0.4).** All SPDX headers and
> manifest license expressions across the tree have been normalised
> to `CECILL-2.1`. The R `DESCRIPTION` file keeps the human-readable
> `CeCILL-2.1` form per R's license database. The C++ `LICENSE` file
> at the repo root is the canonical license text (no SPDX field
> there) and stays untouched.

---

## 0. TL;DR distribution matrix

Three classes of artifact go out per release tag. Read this table as
"a green release ships X to Y."

| Class | Asset | Channel(s) | Cadence | Trigger |
|------|-------|------------|---------|---------|
| **A. Source** | git tag + signed source archive (`pls4all-src-${VER}.tar.gz`) + SBOM + provenance | GitHub Releases, Zenodo (DOI), Software Heritage | every tag | tag push |
| **B. C/C++ binaries** | `libp4a.{so,dylib,dll}` + `pls4all_cli` + headers, per OS/arch | GitHub Releases (canonical), Homebrew tap (macOS/Linux), vcpkg port, Conan recipe (CCI), Docker `ghcr.io/gbeurier/pls4all` | every tag | tag push |
| **C. Language packages** | per-binding distributables (see §3) | language-specific registries (PyPI, CRAN, npm/JSR, crates.io, Maven Central, NuGet, RubyGems, Julia General, LuaRocks, Nimble, MATLAB FileExchange, conda-forge, etc.) | every tag (after A+B succeed) | post-tag workflow |

Channel ownership is single-maintainer (currently `@GBeurier`). API
tokens, GPG keys, and 2FA-recovery codes are documented in the
private `SECURITY.md` companion ledger (not in-tree).

---

## 1. Repositories you must own / be listed on

These are *accounts and namespaces* — register them, point them at
this repo, and they stop being a future risk.

### 1.1 Core source-of-truth

| Repository | URL | Role |
|------------|-----|------|
| **GitHub** primary | `github.com/GBeurier/pls4all` | source, CI, Releases, Pages docs |
| **GitHub Pages** | `gbeurier.github.io/pls4all/` | docs deployed by `.github/workflows/docs.yml` on every push to `main` (live benchmark CSV → Sphinx) |
| **Read the Docs** | `pls4all.readthedocs.io/` | docs built by RTD webhook (uses committed `bench-data.json` fallback) — see §3.bis |
| **Zenodo** | `zenodo.org/account/settings/github/` enabled for `GBeurier/pls4all` | DOI per tag, archival, citable |
| **Software Heritage** | `archive.softwareheritage.org/browse/origin/?origin_url=https://github.com/GBeurier/pls4all` | passive mirror — register the origin once |
| **HAL** (optional, French academia) | `hal.science` deposit + Software Heritage swhid | only when a paper is associated |
| **JOSS** (eventually) | submission once paper drafted | software paper for the engine + AOM/POP |

### 1.2 Mirrors (defensive, low-cost)

- **Codeberg** (`codeberg.org/GBeurier/pls4all`) — read-only mirror via Codeberg's pull-mirror feature. Insurance against GitHub outages.
- **GitLab.com** (`gitlab.com/GBeurier/pls4all`) — same.
- **CIRAD GitLab** (`gitlab.cirad.fr/...`) — institutional mirror; required for some funder reporting.

### 1.3 Per-binding namespaces to claim *now* (squatters are the enemy)

Even before publishing v1.0, claim the names — empty placeholder
packages with a "see GitHub" README are fine.

| Registry | Namespace to reserve | Status (2026-05) |
|----------|---------------------|------------------|
| PyPI | `pls4all` | reserve via empty 0.0.0 upload + `Development Status :: 1 - Planning` |
| TestPyPI | `pls4all` | already used implicitly by `cibuildwheel` test publish |
| conda-forge feedstock | `pls4all-feedstock` (created by staged-recipes) | pending v0.97 |
| CRAN | `pls4all` | submission post-stabilisation; pre-flight via R-universe |
| R-universe | `gbeurier.r-universe.dev/pls4all` | pre-CRAN rolling build (free, mandatory before CRAN) |
| crates.io | `pls4all` | reserve via initial 0.0.0 publish |
| npm | `@pls4all/wasm`, `@pls4all/node`, `@pls4all/types` | scope `@pls4all` already in use; claim org |
| JSR (`jsr.io`) | `@pls4all/wasm` | mirror of npm — new since 2024, free |
| pkg.go.dev | `github.com/GBeurier/pls4all/bindings/go` | indexed automatically on first `GOPROXY` hit |
| NuGet | `Pls4all` (managed) + `Pls4all.Native` (multi-RID native runtime) | reserve via Microsoft account |
| RubyGems | `pls4all` | reserve via `gem push` of 0.0.0 placeholder |
| Julia General | `Pls4all` (UUID `f04a4a11-…`) | already in `Project.toml`; not yet registered |
| LuaRocks | `pls4all` | rockspec + `luarocks upload` |
| Nimble | `pls4all` | PR to `nim-lang/packages` |
| Maven Central | groupId `io.github.gbeurier`, artifactId `pls4all-jni` (desktop), `pls4all-android` (AAR) | via Sonatype Central Portal; needs domain ownership proof |
| MATLAB File Exchange | `pls4all` page on `mathworks.com/matlabcentral/fileexchange` | needs MathWorks account; `.mltbx` from GitHub Release auto-pulled |
| Octave Packages | `pls4all` on `gnu-octave/packages` index | PR to `gnu-octave/packages/index.yaml` |
| MSYS2 / MinGW-packages | `mingw-w64-pls4all` PKGBUILD in `msys2/MINGW-packages` | PR review by MSYS2 maintainers |
| MacPorts | `science/pls4all` Portfile in `macports/macports-ports` | PR review |
| FreeBSD Ports | `math/pls4all` or `science/pls4all` | PR to `freebsd/freebsd-ports` |
| GitHub Packages | `maven.pkg.github.com/GBeurier/pls4all`, `nuget.pkg.github.com/GBeurier`, `npm.pkg.github.com/@pls4all` | free; auto on GH org; RC channel |
| Hugging Face | `huggingface.co/pls4all` (org) | only for pretrained PLS models if we ever ship any |
| Docker / GHCR | `ghcr.io/gbeurier/pls4all` and `docker.io/gbeurier/pls4all` | claim both |
| Quay.io | `quay.io/gbeurier/pls4all` | OCI mirror for OpenShift / HPC sites |

> **Action item.** Reserve every empty namespace above on day 1 of
> v1.0 prep. A squatter takes 5 minutes; recovering a squatted name
> takes weeks (or never).

---

## 2. GitHub Release asset matrix

Every tag (`v0.X.Y` and `vX.Y.Z`) produces the **same canonical asset
bundle**, attached to the GitHub Release page. All other registries
either pull from this bundle or are built fresh from the same source
tree in the publication workflow.

### 2.1 Source artifacts

| Asset | Filename | Notes |
|------|----------|-------|
| Source tarball | `pls4all-src-${VER}.tar.gz` | `git archive` reproducible, contains `VERSION` file (renamed to avoid collision with Python sdist `pls4all-${VER}.tar.gz`) |
| Source zip | `pls4all-src-${VER}.zip` | Windows convenience |
| SBOM (CycloneDX) | `pls4all-${VER}.cdx.json` | generated by `cyclonedx-conan` / `cyclonedx-cli` for C++ + per-binding SBOMs |
| Build provenance attestation | `pls4all-${VER}.intoto.jsonl` | `actions/attest-build-provenance@v2` — useful in-toto/Sigstore provenance, **not** automatic SLSA L3 (achieving L3 requires a hermetic builder we do not yet have) |
| SBOM attestation | embedded in the release | `actions/attest-sbom@v2` |
| Sigstore signature | `*.sig`, `*.crt` | keyless OIDC via `actions/attest-build-provenance@v2` (which signs by default) |
| Checksums | `SHA256SUMS`, `SHA256SUMS.asc` | GPG-signed by maintainer key fingerprint `<published-in-SECURITY.md>` |

### 2.2 C/C++ prebuilt binaries (B class)

One archive per **target triple**. Naming follows the Rust-style
triple convention so cross-language tooling can parse it:

```
pls4all-${VER}-${TRIPLE}.tar.xz       # Linux/macOS
pls4all-${VER}-${TRIPLE}.zip          # Windows
```

Each archive is `tar -xJf` extractable to a self-contained prefix:

```
pls4all-${VER}-x86_64-unknown-linux-gnu/
├── bin/
│   └── pls4all_cli
├── include/pls4all/
│   ├── p4a.h
│   ├── p4a_export.h
│   └── p4a_version.h
├── lib/
│   ├── libp4a.so → libp4a.so.1            (SONAME symlink)
│   ├── libp4a.so.1 → libp4a.so.1.16.0     (ABI MAJOR symlink)
│   └── libp4a.so.1.16.0                   (real file)
├── lib/cmake/pls4all/
│   ├── pls4allConfig.cmake
│   └── pls4allConfigVersion.cmake
├── lib/pkgconfig/
│   └── pls4all.pc
├── share/doc/pls4all/
│   ├── README.md, LICENSE, CITATION.cff, CHANGELOG.md
│   └── abi/ (expected_symbols_*.txt for the release)
└── share/licenses/pls4all/LICENSE
```

#### Target triples shipped per release

| Triple | Toolchain | OS minimum | Notes |
|--------|-----------|------------|-------|
| `x86_64-unknown-linux-gnu` | gcc 12 | glibc 2.31 (Ubuntu 20.04) | manylinux_2_31 compatible |
| `x86_64-unknown-linux-musl` | clang 16 + musl | static glibc-free | for Alpine, distroless, scratch |
| `aarch64-unknown-linux-gnu` | gcc 13 cross | glibc 2.31 | ARM servers, RPi 4 64-bit |
| `aarch64-unknown-linux-musl` | clang 16 + musl | — | ARM Alpine |
| `x86_64-apple-darwin` | Apple clang | macOS 12 | universal2 also offered |
| `aarch64-apple-darwin` | Apple clang | macOS 12 | Apple Silicon |
| `universal2-apple-darwin` | `lipo` of the two above | macOS 12 | preferred for downstream Python wheels |
| `x86_64-pc-windows-msvc` | MSVC 19.4x | Windows 10 | `p4a.dll` + `p4a.lib` + `p4a.pdb` |
| `x86_64-pc-windows-mingw` | MinGW UCRT64 | Windows 10 | for MinGW/MSYS downstream tooling |
| `aarch64-pc-windows-msvc` | MSVC ARM64 | Windows 11 | reserved (Snapdragon X) — produced when CI has the runner |
| `wasm32-unknown-emscripten` | Emscripten 3.x | — | `p4a.wasm` + `p4a.js` glue |
| `wasm32-wasi` | wasi-sdk 22+ | — | server-side WASI runtimes |
| `aarch64-linux-android` | NDK r26+ | API 26 (Android 8.0, matches `CMakePresets.json` `android-*` presets) | Android AAR target |
| `x86_64-linux-android` | NDK r26+ | API 26 | emulator |

Each archive is reproducible (`SOURCE_DATE_EPOCH=$(git log -1 --format=%ct)`)
and verified by re-running the build twice in CI and diff-checking
the `*.so` / `*.dll`. Both BLAS and OpenMP are *optional* runtime
plugins — the canonical Release ships the **dev-release** (no BLAS,
no OpenMP) preset; the **blas-omp** preset is shipped as an
*additional* archive per OS suffixed `-blas-omp` (`pls4all-${VER}-x86_64-unknown-linux-gnu-blas-omp.tar.xz`).

#### Binary ABI identity gate (release preflight)

**Current state (audited 2026-05-18):** the previous SOVERSION mismatch
has been fixed. Linux builds now produce the ABI-major chain
`libp4a.so -> libp4a.so.1 -> libp4a.so.1.16.0`, derived from
`P4A_ABI_VERSION_MAJOR` and the full ABI version, not from the project
semver. The release preflight is therefore:

1. Build from a clean tree, not from a reused local build directory. Old
   symlinks such as `libp4a.so.0` can remain in dirty build dirs and must
   not be copied into packages.
2. Check that the produced Linux SONAME matches
   `libp4a.so.${P4A_ABI_VERSION_MAJOR}`.
3. Do the equivalent on macOS via `install_name_tool`: the install name is
   `@rpath/libp4a.${ABI_MAJOR}.dylib`.
4. Windows DLLs do not carry a SONAME; the import lib should carry the ABI
   in its filename for clarity (`p4a-1.lib`).
5. Diff the exported `p4a_*` symbol set against
   `cpp/abi/expected_symbols_linux.txt`. Any added symbol is a public ABI
   decision and must be documented in `docs/abi/changes_log.md`.

#### Forbidden runtime dependencies (audited per archive)

The ABI-check workflow already gates `ldd` / `otool -L` / `dumpbin`
against a deny-list (`libopenblas`, `libmkl`, `libpython`, `libR.*`,
`librcpp`, `libboost`, `libeigen`, `libpybind11`, `libembind`,
`libnlohmann`, `libyaml-cpp`). The Release workflow re-runs that
check against the shipped archives before the upload step.

### 2.3 Language packages (C class) attached to the Release

The package files themselves are uploaded to the registry, but a
*copy* is attached to the GitHub Release for forensic/audit purposes:

| Binding | File(s) attached to Release |
|---------|----------------------------|
| Python | `pls4all-${VER}-*.whl` (per-platform, manylinux/macos/win), `pls4all-${VER}.tar.gz` (sdist) |
| R | `pls4all_${VER}.tar.gz` (CRAN source), `pls4all_${VER}.tgz` (macOS), `pls4all_${VER}.zip` (Windows) |
| MATLAB | `pls4all-${VER}.mltbx` (FileExchange-ready toolbox) |
| Julia | n/a — registered by Project.toml SHA; tag-only |
| JS | `pls4all-wasm-${VER}.tgz` (`npm pack` output) |
| Go | n/a — go modules resolve from git tag directly |
| Rust | `pls4all-${VER}.crate` (output of `cargo package`) |
| .NET | `Pls4all.${VER}.nupkg`, `Pls4all.Native.${VER}.nupkg` (multi-RID) |
| Ruby | `pls4all-${VER}.gem` |
| Lua | `pls4all-${VER}-1.rockspec` + `pls4all-${VER}-1.src.rock` |
| Nim | n/a — Nimble pulls from git tag |
| JNI / JVM | `pls4all-jni-${VER}.jar`, `pls4all-jni-${VER}-sources.jar`, `pls4all-jni-${VER}-javadoc.jar`, all `.asc` signed |
| Android | `pls4all-android-${VER}.aar` |

---

## 3. Per-binding publication channels

Each subsection follows the same shape: **registry → identity →
required assets → prereqs → publish step → smoke test → caveats**.

### 3.1 Python — `pls4all` on PyPI (+ conda-forge mirror)

| Field | Value |
|-------|-------|
| Registry | https://pypi.org/project/pls4all/ |
| Identity | name: `pls4all`, distribution: `wheel` + `sdist` |
| Tier 1 | ctypes wrapper around `libp4a` (currently shipped) |
| Tier 2 | `pls4all.sklearn` (67 estimators) — Phase 7b live |

**Required assets (built by `cibuildwheel` in CI):**

The binding is **ctypes-only** (no CPython C extension), so wheels
are interpreter-agnostic *but* platform-specific. We ship
`py3-none-${platform}` platform wheels, **not** `cp310-cp310-*` ABI
wheels (which would falsely advertise an extension matching that
specific CPython ABI):

- `pls4all-${VER}-py3-none-manylinux_2_31_x86_64.whl`
- `pls4all-${VER}-py3-none-manylinux_2_31_aarch64.whl`
- `pls4all-${VER}-py3-none-musllinux_1_2_x86_64.whl`
- `pls4all-${VER}-py3-none-musllinux_1_2_aarch64.whl`
- `pls4all-${VER}-py3-none-macosx_12_0_universal2.whl`
- `pls4all-${VER}-py3-none-win_amd64.whl`
- `pls4all-${VER}-py3-none-win_arm64.whl`
- `pls4all-${VER}.tar.gz` (sdist with vendored `cpp/` for from-source builds)

There is **no `py3-none-any` downloader wheel**: every PyPI wheel
either bundles `libp4a` (platform wheel) or builds it from source
(sdist). Wheels do not have a reliable post-install hook, and
import-time binary downloads are hostile to reproducibility and
offline installs.

Wheels **bundle `libp4a` inside the wheel** via `auditwheel repair`
(Linux) / `delocate-wheel` (macOS) / `delvewheel` (Windows). The
shared library is renamed and placed under `pls4all.libs/` (Linux
convention) with RPATH `$ORIGIN/../pls4all.libs` patched into the
loader stub. To convince `cibuildwheel` to repair a ctypes-only
wheel (which it normally treats as pure-Python and skips), we use
`CIBW_REPAIR_WHEEL_COMMAND_*` explicitly *and* force a platform tag
via a `setup.py` shim that calls
`from setuptools import Distribution; Distribution(...).has_ext_modules = lambda: True`.
This keeps installs pip-only (no `apt install libp4a-dev`).

> **License note** (re: cibuildwheel warning): bundling libp4a is
> fine because we *own* libp4a and it's the same CeCILL-2.1 as the
> wheel. The `LICENSE` file gets copied to `pls4all-${VER}.dist-info/`.

**Prereqs:**
1. PyPI API token in GitHub Actions secret `PYPI_API_TOKEN` (per-project token, scope = `pls4all` only).
2. TestPyPI token in `TESTPYPI_API_TOKEN` for pre-release dry runs.
3. Trusted publishing (OIDC) configured for `GBeurier/pls4all` → PyPI (preferred over long-lived tokens since 2024).
4. `pyproject.toml` updated to use `setuptools-scm` so `pip install` from the sdist deduces the version from the git tag.

**Publish step (CI):**
```yaml
# .github/workflows/release.yml — python job
- uses: pypa/cibuildwheel@v3
  env:
    # Build once per platform/arch — the ctypes binding is interpreter-agnostic.
    CIBW_BUILD: "cp312-*"            # any modern CPython picks up the platform wheel
    CIBW_SKIP: "*-musllinux_i686 *-win32"
    CIBW_BEFORE_ALL_LINUX: bash bindings/python/scripts/build_libp4a_in_wheel.sh
    CIBW_REPAIR_WHEEL_COMMAND_LINUX: "auditwheel repair -w {dest_dir} {wheel}"
    CIBW_REPAIR_WHEEL_COMMAND_MACOS: "delocate-wheel -w {dest_dir} -v {wheel}"
    CIBW_REPAIR_WHEEL_COMMAND_WINDOWS: "delvewheel repair -w {dest_dir} {wheel}"
    CIBW_TEST_COMMAND: "python -m pytest {project}/bindings/python/tests"
- name: Re-tag wheels py3-none-${platform}
  run: python bindings/python/scripts/retag_wheels.py wheelhouse/
- uses: pypa/gh-action-pypi-publish@release/v1
  with:
    skip-existing: true
    attestations: true
```

**Smoke test (in CI, after publish):**
```bash
pip install --index-url https://pypi.org/simple/ pls4all==${VER}
python -c "import pls4all; assert pls4all.version().startswith('${VER}')"
python -m pls4all --selfcheck
```

**conda-forge mirror.** Submit `pls4all-feedstock` to
`conda-forge/staged-recipes` once PyPI has a stable release. The
recipe `meta.yaml` declares `host: libp4a` + `run: libp4a`: the
conda Python package depends on the conda `libp4a` package. PyPI
wheels do **not** depend on conda packages — they bundle their own
copy of `libp4a` and are independently usable.

Two feedstocks, in order:
1. `libp4a-feedstock` (the C/C++ library — no Python) → migrators handle BLAS variants.
2. `pls4all-feedstock` (the Python package) → `run: libp4a {{ pin_compatible('libp4a') }}`.

### 3.2 R — `pls4all` on CRAN (+ R-universe rolling)

| Field | Value |
|-------|-------|
| Registry | https://cran.r-project.org/package=pls4all |
| Identity | Package: `pls4all`, License: `CeCILL-2.1` (R license database short form) |
| Tier 1 | `.Call` gateway over the C ABI (no Rcpp) |
| Tier 2 | base-R formula+S3 · parsnip · mlr3 — all live |

**Required assets:**
- `pls4all_${VER}.tar.gz` (CRAN source) — `R CMD build bindings/r/pls4all`.
- `pls4all_${VER}.tgz` (binary for macOS, built per R minor: 4.4, 4.5).
- `pls4all_${VER}.zip` (binary for Windows, built per R minor).

CRAN itself rebuilds binaries automatically from the submitted
source — but R-universe lets us **publish those binaries before
CRAN does**, which is the practical mainstream.

**The `libp4a` dilemma.** CRAN source packages cannot contain
prebuilt binary libraries, but they can include C/C++ source and
build it during package installation. Two viable strategies:

1. **System dependency**: declare `SystemRequirements: libp4a (>=
   1.16)` and rely on the user (or the conda-forge R installation,
   or `r-spelling`-style autobrew) to provide `libp4a`. This is
   what the current `DESCRIPTION` does. CRAN policy permits this,
   but only realistic if `libp4a` is widely available on CRAN's
   build machines (currently it is not).
2. **Vendored source + static build**: the `src/` of the R package
   contains the `cpp/` tree as a git subtree and a `Makevars` that
   builds the static `libp4a.a` at install time, then links the R
   package against it. Heavier (compile time ~5 min on CRAN's
   slowest farm) but no system dependency. CRAN policy explicitly
   permits this: "if a system library is unavailable, it is
   desirable to include the library sources in the package and
   compile them." **This is the practical first-submission path
   for `pls4all`.** We switch to it for v1.0.

**Prereqs:**
1. CRAN maintainer email = `gregory.beurier@cirad.fr` (already in `DESCRIPTION`).
2. R-universe organisation: `https://gbeurier.r-universe.dev/`.
3. The package passes `R CMD check --as-cran --no-manual` with zero `NOTE`/`WARNING`/`ERROR` on:
   - Linux release + devel
   - macOS arm64 + intel
   - Windows
   - R-hub `windows-x86_64-devel`, `fedora-clang-devel`, `debian-clang-devel`
4. WRE compliance: no internet access in tests, no parallel writes to `/tmp`, no `sudo`, no `:::` from other packages.

**Publish step (manual, CRAN is human-reviewed):**
```bash
R CMD build bindings/r/pls4all
R CMD check --as-cran pls4all_${VER}.tar.gz
# Then via web form
open "https://cran.r-project.org/submit.html"
# Upload tarball, paste cran-comments.md, wait 1-3 days for human review.
```

**R-universe (rolling, no human review):**
- One file: `packages.json` in the GitHub universe config repo
  `gbeurier/gbeurier.r-universe.dev`, pointing at this git URL +
  the `bindings/r/pls4all` subdir.
- R-universe rebuilds on every commit; users install via:
  ```r
  install.packages("pls4all",
    repos = c("https://gbeurier.r-universe.dev",
              "https://cloud.r-project.org"))
  ```

**Smoke test:**
```r
library(pls4all)
stopifnot(grepl("^${VER}", pls4all::pls4all_version()))
fit <- pls(y ~ ., data = iris[,-5], ncomp = 2)
predict(fit, newdata = iris[1:5, -5])
```

### 3.3 MATLAB — `+pls4all` package on File Exchange + Add-On Explorer

| Field | Value |
|-------|-------|
| Registry | https://www.mathworks.com/matlabcentral/fileexchange/ |
| Identity | submission name `pls4all`, package format: `.mltbx` Toolbox |
| Tier 1 | single MEX dispatcher (`p4a_method_fit_mex`) — live |
| Tier 2 | 18 classdefs + factory — live |

**Required assets:**
- `pls4all-${VER}.mltbx` (MATLAB Toolbox installable file).
- The same toolbox files for **Octave**, packaged as `pls4all-octave-${VER}.zip` (Octave Forge sees this).
- Per-OS-per-MATLAB-version prebuilt MEX files baked into the `.mltbx`:
  | MEX file | OS × MATLAB |
  |---|---|
  | `p4a_method_fit_mex.mexa64` | Linux x86_64, R2023b+ |
  | `p4a_method_fit_mex.mexmaci64` | macOS x86_64, R2023b+ |
  | `p4a_method_fit_mex.mexmaca64` | macOS arm64, R2024a+ |
  | `p4a_method_fit_mex.mexw64` | Windows x86_64, R2023b+ |
  | `p4a_method_fit_mex.oct` | Octave 9.x (all platforms) |

The MEX must dynamically link `libp4a` shipped *inside* the toolbox:
the `.mltbx` archive contains `+pls4all/private/libp4a/` and the
MATLAB code prepends that directory to the dynamic-library search
path before any MEX call.

**Prereqs:**
1. MathWorks account = `gregory.beurier@cirad.fr` (matlabcentral profile).
2. MATLAB R2024b on at least one CI runner to package the `.mltbx`
   (use `matlab.addons.toolbox.packageToolbox(...)`).
3. `bindings/matlab/build_mex.m` produces the per-OS MEX files;
   the `release.yml` workflow runs that on a matrix of OS runners.

**Publish step (semi-automated):**
- MATLAB GitHub Action: `matlab-actions/run-build@v2` invokes
  `bindings/matlab/release.m`, which calls
  `matlab.addons.toolbox.packageToolbox('bindings/matlab/toolbox.prj',
  'pls4all-${VER}.mltbx')`.
- The resulting `.mltbx` is attached to the GitHub Release.
- File Exchange has a **GitHub integration**: bind the FX submission
  to `GBeurier/pls4all`. When a new release lands, FX picks up the
  `.mltbx` automatically from the Release asset and updates the FX
  page. (FX itself does not gate on test-runs; our own MATLAB CI
  runs the smoke test below before the GitHub Release is promoted
  out of draft.)

**Smoke test (MATLAB CI):**
```matlab
matlab.addons.install("pls4all-${VER}.mltbx");
assert(strncmp(pls4all.version(), "${VER}", strlength("${VER}")));
[X, y] = pls4all.synthetic(50, 5, 1);
mdl = pls4all.fit("simpls", X, y, "NumComponents", 3);
yhat = predict(mdl, X);
assert(size(yhat, 1) == 50);
```

> **Forge mirror.** Octave Forge accepts source-only packages
> (`pls4all-octave-${VER}.tar.gz`). After v1.0 we also upload to
> [Octave Packages](https://gnu-octave.github.io/packages/) via the
> `index.yaml` PR pattern. Naming is `pls4all` (no Octave suffix).

### 3.4 JavaScript / TypeScript — `@pls4all/*` on npm + JSR

| Field | Value |
|-------|-------|
| Registry | https://www.npmjs.com/org/pls4all + https://jsr.io/@pls4all |
| Identity | `@pls4all/wasm` (current), planned: `@pls4all/node`, `@pls4all/types` |
| Tier 1 | `p4a.wasm` + thin TS API — live |
| Tier 2 | `@pls4all/sklearn-like` (idiomatic class) — live |

**Required assets:**
- `@pls4all/wasm` (browser + Deno + Bun + Node universal ESM):
  - `dist/index.js` (ESM)
  - `dist/index.d.ts`
  - `dist/p4a.wasm` (single binary, ~120 KB after strip)
  - `dist/p4a.wasm.map` (source map, optional)
- `@pls4all/node` (Node-only with native bindings via `node-gyp` or
  `napi-rs`, for users who don't want WASM): includes prebuilt
  `pls4all.node` binaries via `prebuildify` for:
  | Triple | Node ABI | OS |
  |---|---|---|
  | `linux-x64`, `linux-arm64`, `linux-x64-musl`, `linux-arm64-musl` | napi-9 (Node 20+) | Linux |
  | `darwin-x64`, `darwin-arm64` | napi-9 | macOS |
  | `win32-x64`, `win32-arm64` | napi-9 | Windows |
- `@pls4all/types` (pure `.d.ts`, for projects mixing both): one
  package keeps the type surface single-source.

**Prereqs:**
1. npm org `@pls4all` claimed; CI scope tokens stored as `NPM_TOKEN`.
2. JSR org `@pls4all` claimed; OIDC trust to GitHub Actions.
3. Emscripten 3.x available in the JS build job (currently in
   `bindings/js/CMakeLists.txt`).

**Publish step:**
```bash
# npm
cd bindings/js
npm version ${VER} --no-git-tag-version
npm publish --access public --provenance     # provenance since npm v9
# JSR mirror
npx jsr publish
```

**Smoke test:**
```bash
deno run --allow-all -r https://jsr.io/@pls4all/wasm/${VER}/smoke.ts
node -e "import('@pls4all/wasm').then(m => console.log(m.version()))"
```

### 3.5 Go — `github.com/GBeurier/pls4all/bindings/go` on pkg.go.dev

Go modules are special — there is no central upload, only the git
tag + module path.

| Field | Value |
|-------|-------|
| Registry | https://pkg.go.dev/github.com/GBeurier/pls4all/bindings/go/pls4all |
| Identity | module path `github.com/GBeurier/pls4all/bindings/go` |
| Tier | cgo wrapper (current) |

**Required assets:** none in the GitHub Release. Go fetches from the
git tag directly. The constraint is that the `go.mod` module path
must include the binding subdirectory: `github.com/GBeurier/pls4all/bindings/go`.

**Prereqs:**
1. The git tag must follow the **submodule pattern** for sub-tree
   modules: `bindings/go/v0.97.0` (not just `v0.97.0`). pkg.go.dev
   indexes that automatically.
2. `bindings/go/go.mod` declares `go 1.22`.
3. cgo is enabled by default; users must have `libp4a` installed
   on their `LD_LIBRARY_PATH`. Document the Homebrew tap (§4.1) and
   Debian package (§4.4) as the canonical install paths.

**Publish step:**
```bash
git tag bindings/go/v${VER}
git push origin bindings/go/v${VER}
# Force a fetch to populate pkg.go.dev
GOPROXY=proxy.golang.org go list -m github.com/GBeurier/pls4all/bindings/go@v${VER}
```

**Smoke test:**
```bash
mkdir /tmp/p4a-smoke && cd /tmp/p4a-smoke
go mod init smoke
go get github.com/GBeurier/pls4all/bindings/go@v${VER}
cat > main.go <<'EOF'
package main

import (
    "fmt"
    "github.com/GBeurier/pls4all/bindings/go/pls4all"
)

func main() { fmt.Println(pls4all.Version()) }
EOF
LD_LIBRARY_PATH=/usr/local/lib go run .
```

### 3.6 Rust — `pls4all` on crates.io

| Field | Value |
|-------|-------|
| Registry | https://crates.io/crates/pls4all |
| Identity | `pls4all` crate, `lib` crate-type rlib (current) |
| Tier | hand-rolled `extern "C"` over `libp4a` (current) |

**License field.** `crates.io` validates the `license =` field as a
valid SPDX expression. `CECILL-2.1` is on the SPDX list, so this
works:
```toml
license = "CECILL-2.1"
```
Cargo supports either `license` (SPDX expression) **or** `license-file`
(arbitrary text file), but Cargo warns if **both** are set in the
same manifest. For pls4all we use `license = "CECILL-2.1"` and rely
on Cargo's automatic inclusion of `LICENSE` in the crate.

**Required assets:**
- `pls4all-${VER}.crate` (output of `cargo package --list`).

The crate **does not bundle `libp4a`**. It links dynamically via
`build.rs`, which reads `PLS4ALL_LIB_DIR` / `PLS4ALL_INCLUDE_DIR` and
falls back to `pkg-config --libs pls4all`. The README documents
Homebrew/Debian/conda-forge as the canonical install routes.

**Future: `pls4all-sys`.** A companion `pls4all-sys` crate that
*does* bundle libp4a as a build dep is on the roadmap. It's the
standard Rust pattern (`openssl-sys`, `libsqlite3-sys`). For v1.0
the single `pls4all` crate is enough.

**Prereqs:**
1. `crates.io` API token in `CARGO_REGISTRY_TOKEN` secret.
2. `cargo publish --dry-run` clean from CI before tag day.

**Publish step:**
```bash
cd bindings/rust/pls4all
cargo publish --dry-run
cargo publish
```

**Smoke test:**
```bash
cargo new --bin /tmp/p4a-smoke && cd /tmp/p4a-smoke
cargo add pls4all@${VER}
cargo run
```

### 3.7 .NET — `Pls4all` + native runtime packages on NuGet

The .NET ecosystem distinguishes the **managed assembly** (your C#)
from the **native dependency** (`libp4a`). The .NET-standard pattern
is to ship them as **two NuGet packages**, with all native runtimes
bundled in one multi-RID package:

| Package | Type | Content |
|---------|------|---------|
| `Pls4all` | managed | `lib/net9.0/Pls4all.dll` (any-CPU), targets `net9.0` (matches current `Pls4all.csproj`); depends on `Pls4all.Native` |
| `Pls4all.Native` | native multi-RID | `runtimes/linux-x64/native/libp4a.so`, `runtimes/linux-arm64/native/libp4a.so`, `runtimes/osx-x64/native/libp4a.dylib`, `runtimes/osx-arm64/native/libp4a.dylib`, `runtimes/win-x64/native/p4a.dll`, `runtimes/win-arm64/native/p4a.dll` |

The native filenames are the **unversioned loadable names** that
match the `[DllImport("p4a")]` attribute used by `Pls4all.cs`. The
ABI-versioned filenames (`libp4a.so.1.16.0`) ship inside the C++
archive (§2.2) but **not** inside the NuGet native package, because
the .NET SDK/runtime selects native assets by RID-directory matching,
not by SONAME.

A consumer just adds `Pls4all` and `Pls4all.Native` is pulled in as
a dependency. The .NET SDK/runtime selects the matching files under
`runtimes/{rid}/native/` at build/publish/runtime — no `Pls4all.Native.all`
meta-package needed.

**Optional split.** If the multi-RID package grows too large
(>50 MB), split into per-RID packages following the
`Microsoft.Data.SqlClient.SNI.runtime` pattern: `Pls4all.Native.linux-x64`,
`Pls4all.Native.linux-arm64`, etc., with a meta package `Pls4all.Native`
that takes a hard dependency on all of them. Defer this until the
size actually becomes a problem; today libp4a is ~2 MB per RID, so
the single package totals ~12 MB which is fine.

**Prereqs:**
1. NuGet API key in `NUGET_API_KEY` secret (or trusted publishing
   via NuGet's GitHub OIDC integration).
2. `Pls4all.csproj` has `<GeneratePackageOnBuild>true</GeneratePackageOnBuild>`
   and `<PackageLicenseExpression>CECILL-2.1</PackageLicenseExpression>`
   (do **not** use the deprecated `licenseUrl`).
3. `Pls4all.Native.csproj` is a `Microsoft.NET.Sdk` project with
   `<IncludeBuildOutput>false</IncludeBuildOutput>` and a `<Content>`
   manifest that maps each prebuilt `libp4a` to `runtimes/${rid}/native/`.

**Publish step:**
```bash
dotnet pack bindings/dotnet/Pls4all -c Release -o out/
dotnet pack bindings/dotnet/Pls4all.Native -c Release -o out/
dotnet nuget push out/*.nupkg --source https://api.nuget.org/v3/index.json \
  --api-key $NUGET_API_KEY --skip-duplicate
```

**Smoke test:**
```bash
dotnet new console -o /tmp/p4a-smoke && cd /tmp/p4a-smoke
dotnet add package Pls4all --version ${VER}
# Pls4all.Native is pulled in transitively.
dotnet run
```

### 3.8 Julia — `Pls4all.jl` on the General registry (+ `Pls4all_jll`)

Julia separates the binding from the binary even more strictly than
.NET, via the **JLL wrapper** pattern.

| Package | Role |
|---------|------|
| `Pls4all_jll` | auto-generated by BinaryBuilder.jl, ships `libp4a` for every platform Julia supports |
| `Pls4all.jl` | user-facing API; depends on `Pls4all_jll` for the binary |

Two different registration flows:

- `Pls4all.jl` is registered to **General** via JuliaRegistrator.jl
  (a GitHub App that opens a PR against the General registry on a
  `@JuliaRegistrator register` comment).
- `Pls4all_jll` is generated by **BinaryBuilder** when a recipe
  PR (`P/Pls4all/build_tarballs.jl`) is merged into Yggdrasil; the
  Yggdrasil pipeline then opens its own PR against General. We do
  not call Registrator for the JLL.

**Required assets:** none in our Release. Everything is built from
the git tag (Pls4all.jl) and from the source tarball (Pls4all_jll
via Yggdrasil's `build_tarballs.jl`).

**Prereqs:**
1. UUID locked in `bindings/julia/Pls4all.jl/Project.toml` (already done).
2. PR to Yggdrasil with `P/Pls4all/build_tarballs.jl`. The platforms
   actually produced are those declared in the recipe (typically
   `expand_cxxstring_abis(supported_platforms())`); not "every
   platform Julia supports" unless we explicitly target all of them.
3. JuliaRegistrator bot installed on `GBeurier/pls4all` *or* manual
   web UI at `juliahub.com → Register packages`.

**Publish step:**
```
# As an issue comment on the right commit:
@JuliaRegistrator register subdir=bindings/julia/Pls4all.jl
```

**Smoke test:**
```julia
using Pkg; Pkg.add(name="Pls4all", version="${VER}")
using Pls4all
@assert startswith(Pls4all.version(), "${VER}")
```

### 3.9 Ruby — `pls4all` gem on RubyGems

| Field | Value |
|-------|-------|
| Registry | https://rubygems.org/gems/pls4all |
| Identity | `pls4all`, Fiddle-based (stdlib FFI, no `mkmf` C extension) |

**Required assets:**
- `pls4all-${VER}.gem` (output of `gem build`). Pure-Ruby gem — no
  native code in the gem itself, relies on system-installed `libp4a`.
- Optional: a **fat gem** per platform that ships `libp4a` inside
  (Ruby's `Gem::Platform.local` convention, e.g. `pls4all-${VER}-x86_64-linux.gem`).
  Add this when v1.0 stabilises; it's how `nokogiri` and `grpc` do it.

**Missing artifact today:** `bindings/ruby/pls4all.gemspec`. Must be
created (see [§6 New-channel checklist](#6-new-channel-checklist)).

**Prereqs:**
1. RubyGems API key in `RUBYGEMS_API_KEY` secret.
2. `bindings/ruby/pls4all.gemspec` with name, version (auto from
   `lib/pls4all/version.rb`), authors, license = `CECILL-2.1`.

**Publish step:**
```bash
cd bindings/ruby
gem build pls4all.gemspec
gem push pls4all-${VER}.gem
```

### 3.10 Lua — `pls4all` on LuaRocks

| Field | Value |
|-------|-------|
| Registry | https://luarocks.org/modules/gbeurier/pls4all |
| Identity | `pls4all` (rockspec format) |
| Targets | **LuaJIT 2.1 only** (standard Lua 5.1+ has no built-in FFI; the binding uses LuaJIT's `ffi.*` API directly) |

**Required assets:**
- `pls4all-${VER}-1.rockspec`
- `pls4all-${VER}-1.src.rock` (source rock)

**Missing artifact today:** the rockspec. Must be created.

**Prereqs:**
1. LuaRocks account; API key in `LUAROCKS_API_KEY` secret.
2. `pls4all-${VER}-1.rockspec` with `external_dependencies = { LIBP4A = { library = "p4a" } }`.

**Publish step:**
```bash
cd bindings/lua
luarocks pack pls4all-${VER}-1.rockspec
luarocks upload pls4all-${VER}-1.rockspec --api-key=$LUAROCKS_API_KEY
```

### 3.11 Nim — `pls4all` in Nimble packages

| Field | Value |
|-------|-------|
| Registry | https://nimble.directory/pkg/pls4all (mirror of `nim-lang/packages_official`) |
| Identity | `pls4all` |

**Required assets:**
- `bindings/nim/pls4all.nimble` — package metadata.
- Nim packages are git-fetched at install; no upload needed.

**Missing artifact today:** the `.nimble` file. Must be created.

**Prereqs:**
1. PR to [`nim-lang/packages`](https://github.com/nim-lang/packages)
   adding a JSON entry pointing at this repo's `bindings/nim` subdir.

**Publish step:**
```bash
# One-time
gh repo clone nim-lang/packages /tmp/nim-packages
# Add an entry, open a PR; nim merges within ~days.
```

### 3.12 JVM / Android — Maven Central (`io.github.gbeurier:pls4all-{jni,android}`)

Two artifacts, two flavours:

| GroupId : artifactId : packaging | Target |
|--------|--------|
| `io.github.gbeurier:pls4all-jni:${VER}` (jar) | Desktop JVM — bundles `libp4a.so` + `libp4a_jni.so` per OS in `META-INF/native/<os>/<arch>/` |
| `io.github.gbeurier:pls4all-android:${VER}` (aar) | Android — includes `jniLibs/{arm64-v8a,x86_64}/libp4a.so` + Kotlin wrapper |
| `io.github.gbeurier:pls4all-jni:${VER}:sources` (jar) | source jar (required for Maven Central) |
| `io.github.gbeurier:pls4all-jni:${VER}:javadoc` (jar) | javadoc jar (required for Maven Central) |
| All of the above `.asc` | GPG signatures (required for Maven Central) |

**Prereqs:**
1. Sonatype Central Portal account. The namespace `io.github.gbeurier`
   must be **verified** in Central Portal. If the maintainer signs
   up with the `GBeurier` GitHub account, Sonatype may provision
   it automatically; otherwise use the code-hosting verification
   flow (create a temporary public repo with a Sonatype-supplied
   verification token).
2. GPG key registered with `keys.openpgp.org` + `keyserver.ubuntu.com`,
   private key + passphrase in `MAVEN_GPG_PRIVATE_KEY` /
   `MAVEN_GPG_PASSPHRASE` secrets.
3. Android NDK r26d+ available in the CI runner for `pls4all-android`
   (the current Phase 36 deferral resolves when this is wired).
4. AAR built via the planned `bindings/android/pls4all-android/build.gradle.kts`.
5. A Gradle Central Portal plugin — **Sonatype does not provide an
   official Gradle plugin** for Central Portal publishing. Pick one of:
   - [JReleaser](https://jreleaser.org/) — most flexible, handles
     full release orchestration (recommended).
   - [`com.gradleup.nmcp`](https://github.com/GradleUp/nmcp).
   - [`com.vanniktech.maven.publish`](https://vanniktech.github.io/gradle-maven-publish-plugin/).
   - Call the Publisher API directly via `curl` + the Central Portal
     bearer token.

**Publish step (assuming JReleaser):**
```bash
./gradlew publish              # stages signed artifacts into build/staging-deploy
./gradlew jreleaserDeploy      # uploads to Central Portal and waits for validation
# JReleaser auto-publishes once validation passes; or set
# jreleaser.deploy.maven.mavenCentral.<name>.stage = UPLOAD for a manual UI promote.
```

**Smoke test (Android):**
```kotlin
// Android instrumentation test
val v = io.pls4all.Pls4all.version()
assertTrue(v.startsWith("${VER}"))
```

### 3.13 WebAssembly / browser — npm + JSR + CDN

In addition to the npm publication in §3.4, we expose the WASM blob
to CDNs for zero-tooling browser use:

- `https://cdn.jsdelivr.net/npm/@pls4all/wasm@${VER}/dist/p4a.wasm` (auto on npm publish)
- `https://unpkg.com/@pls4all/wasm@${VER}/dist/p4a.wasm` (auto on npm publish)
- `https://esm.sh/@pls4all/wasm@${VER}` (Deno-friendly imports)

Nothing to do beyond §3.4 — CDNs mirror npm automatically.

### 3.14 Documentation — GitHub Pages + Read the Docs

The Sphinx site in `docs/` is published in parallel to **two** independent surfaces. Both build the same Sphinx config (`docs/conf.py`) from the same git history; they differ in trigger, hosting, and benchmark-data source.

| Target | URL | Triggered by | Benchmark data source |
|---|---|---|---|
| **GitHub Pages** | `https://gbeurier.github.io/pls4all/` | `.github/workflows/docs.yml` on push to `main` | live `benchmarks/cross_binding/results/full_matrix.csv` if present, else `docs/_static/bench-data.json` |
| **Read the Docs** | `https://pls4all.readthedocs.io/` | RTD webhook on every push (all branches + tags) | committed `docs/_static/bench-data.json` only — RTD's build env doesn't carry the full benchmark cache |

**Why both?**
- **GitHub Pages** is the canonical site (rolls forward with `main`, gets fresh benchmark data from CI).
- **Read the Docs** adds (a) versioned snapshots (`/en/v0.97.0/`, `/en/v1.0.0/`, …), (b) PDF + ePub builds (the `.readthedocs.yaml` declares `formats: [htmlzip, pdf]`), (c) a PR-preview build per pull-request — useful when reviewing doc changes before merging.

**Configuration files**:

| File | Owner | Status |
|---|---|---|
| `.readthedocs.yaml` (repo root) | RTD build pipeline | committed at the same time as M1 |
| `docs/conf.py` | both builds | maintained by M0 + ongoing doc work |
| `docs/requirements.txt` | both builds (Sphinx + myst-parser) | committed |
| `docs/dev/readthedocs.md` | maintainer-facing setup doc | committed |

**One-time RTD setup (maintainer action)**:

| # | Who | Action |
|---|-----|--------|
| RTD.1 | 👤 | Sign in at https://readthedocs.org/ with the GitHub OAuth provider (no separate password needed). |
| RTD.2 | 👤 | **Import a Project** → pick `GBeurier/pls4all`. RTD reads `.readthedocs.yaml` automatically. |
| RTD.3 | 👤 | In **Admin → Advanced settings** set: *Documentation type*=Sphinx HTML, *Privacy level*=Public, *Default branch*=`main`. |
| RTD.4 | 👤 | (Optional) **Admin → Domains** → add CNAME `docs.pls4all.io` (or skip and live with `pls4all.readthedocs.io`). |
| RTD.5 | 🤖 | Verify the first build at `https://readthedocs.org/projects/pls4all/builds/`. If it's red, I'll patch `docs/requirements.txt` / `docs/conf.py` based on the RTD log. |

**Recurring cost**: zero. Both Pages and RTD build automatically on push. No secrets to manage. The two sites can drift (Pages has fresh benchmark data, RTD doesn't) but `docs/conf.py:setup()` auto-falls-back to the committed JSON so the RTD build never errors on missing benchmark inputs.

**Version policy**:
- Pages: always reflects `main` (one rolling site).
- RTD: each git tag becomes a versioned snapshot at `/en/${TAG}/`. The "stable" alias points at the most recent non-`-rc` tag, "latest" follows `main`. RTD picks these up automatically from the tags pushed by M0/M1.

---

## 4. C/C++ ecosystem channels (where the .so/.dll lives)

These channels target users who don't go through a language binding
but install the C/C++ library directly.

### 4.1 Homebrew tap — `gbeurier/pls4all`

| Field | Value |
|-------|-------|
| Tap repo | `GBeurier/homebrew-pls4all` |
| Formula | `Formula/pls4all.rb` |
| Bottles | `gbeurier-pls4all-${VER}.{x86_64,arm64}_{ventura,sonoma,sequoia}.bottle.tar.gz` |
| Bottle host | GitHub Releases of the **tap repo** (not this repo — Homebrew convention) |

User installs:
```bash
brew tap gbeurier/pls4all
brew install pls4all
```

**Prereqs:**
1. A second repo `homebrew-pls4all`.
2. A `brew tap-new` skeleton with the bottle-build workflow.

We do **not** push to homebrew-core. The barrier (popularity check,
review) is too high for a research-grade chemometrics library at
v1.0. Re-evaluate at v2.0.

### 4.2 vcpkg port — `pls4all` in microsoft/vcpkg

| Field | Value |
|-------|-------|
| Registry | https://github.com/microsoft/vcpkg/tree/master/ports/pls4all |
| Identity | `pls4all` port + version DB entry |
| Maintainer | `GBeurier` |

**Required assets:**
- `ports/pls4all/portfile.cmake` (download + CMake invocation)
- `ports/pls4all/vcpkg.json` (manifest)
- `versions/p-/pls4all.json` (version DB entry)
- `versions/baseline.json` update

**Publish step:** PR against `microsoft/vcpkg` adding the four files
above. The vcpkg curated registry's PR review enforces:
- portfile pinned to a release tag with sha512.
- usage file `usage` printed after install.
- license expression matches `CECILL-2.1`.

### 4.3 Conan recipe — Conan Center Index (CCI)

| Field | Value |
|-------|-------|
| Registry | https://conan.io/center/recipes/pls4all |
| Identity | `pls4all/${VER}@` (no user/channel — CCI convention) |

**Required assets:**
- `recipes/pls4all/all/conanfile.py`
- `recipes/pls4all/all/conandata.yml` (with sha256 pinned to GitHub Release tarball)
- `recipes/pls4all/config.yml`

**Publish step:** PR against
`conan-io/conan-center-index`. CCI's CI builds across the full matrix
(Linux gcc/clang, macOS, Windows MSVC/MinGW, x86_64 + arm64, multiple
shared/static/with_blas/with_openmp/with_cuda permutations) before merging.

### 4.4 Linux distro packages (long tail, low priority)

| Distro | Channel | Format |
|--------|---------|--------|
| Debian / Ubuntu | private PPA `ppa:gbeurier/pls4all` → eventually debian official | `.deb` |
| Arch | `aur.archlinux.org/packages/pls4all` (community-driven) | PKGBUILD |
| Fedora / RHEL | COPR `@gbeurier/pls4all` → eventually Fedora official | `.rpm` |
| openSUSE | OBS `home:gbeurier:pls4all` | `.rpm`/`.deb` |
| Nix | `nixpkgs/pkgs/by-name/pl/pls4all/package.nix` | derivation |
| Spack | `var/spack/repos/builtin/packages/pls4all/package.py` (used by HPC sites) | spack recipe |
| Guix | `gnu/packages/chemometrics.scm` | guile recipe |

These are *low priority* (most users install via the language
binding, which already pulls libp4a transitively via conda-forge or
the bundled wheel). We support them only when a downstream maintainer
asks. Each follows the upstream contribution policy of the distro.

### 4.5 Docker / OCI images on GHCR + Docker Hub

| Image | Tags | Content |
|-------|------|---------|
| `ghcr.io/gbeurier/pls4all:${VER}` | `latest`, `${VER}`, `${VER}-cuda` | `pls4all_cli` + `libp4a` runtime |
| `ghcr.io/gbeurier/pls4all:${VER}-dev` | same | adds headers + `cmake`/`gcc` for FROM-as |
| `ghcr.io/gbeurier/pls4all-runner:${VER}` | same | base for benchmark runs; preloads Python+R+MATLAB-mcr |
| `docker.io/gbeurier/pls4all:${VER}` | mirror | optional, helps users behind GHCR-blocked firewalls |

All images are **multi-arch** (`linux/amd64`, `linux/arm64`) via
`docker buildx`. SBOM and provenance attestation are embedded via
`docker buildx --attest type=sbom,generator=docker/scout-sbom-indexer`
and `--attest type=provenance,mode=max`.

---

## 5. Release workflow (CI orchestration)

The publication of A + B + C classes is driven by a single workflow,
`.github/workflows/release.yml`, triggered on `push: tags: ['v*.*.*']`.

```mermaid
graph LR
  TAG[Tag v0.97.0] --> SRC[1. Source bundle + SBOM + provenance]
  TAG --> BLD[2. Multi-OS build matrix\n→ Triple archives]
  SRC --> REL[3. Create draft GH Release\nupload SRC + binaries]
  BLD --> REL
  REL --> PY[4a. cibuildwheel → PyPI + TestPyPI]
  REL --> RP[4b. R-universe push + CRAN tarball attached]
  REL --> ML[4c. .mltbx packager (matlab-actions)]
  REL --> NPM[4d. npm publish + JSR mirror]
  REL --> CR[4e. cargo publish]
  REL --> NUG[4f. dotnet pack + nuget push]
  REL --> JL[4g. JuliaRegistrator comment]
  REL --> GE[4h. gem push (Ruby)]
  REL --> LR[4i. luarocks upload]
  REL --> NB[4j. Nimble repo PR / autoupdate]
  REL --> MV[4k. gradle publishToCentralPortal]
  REL --> DK[4l. docker buildx push (multi-arch)]
  REL --> HB[4m. homebrew tap formula bump PR]
  REL --> VC[4n. vcpkg port version bump PR (manual)]
  REL --> CC[4o. conan-center-index PR (manual)]
  PY --> SMOKE[5. Post-publish smoke test matrix]
  RP --> SMOKE
  ML --> SMOKE
  NPM --> SMOKE
  CR --> SMOKE
  NUG --> SMOKE
  JL --> SMOKE
  GE --> SMOKE
  LR --> SMOKE
  NB --> SMOKE
  MV --> SMOKE
  DK --> SMOKE
  SMOKE --> PROMOTE[6. Promote GH Release from draft → published]
  PROMOTE --> ANNOUNCE[7. Announce: docs site, mailing list, GH Discussion]
```

### 5.1 Idempotency

All publish steps use `--skip-existing` (PyPI), `--skip-duplicate`
(NuGet), `skip-existing: true` (gh-action-pypi-publish), so the
workflow is safe to re-run if a single step fails.

### 5.2 Manual-review channels (CRAN, vcpkg, conan-center, Maven
Central staging, MATLAB FX)

These produce a **release-candidate PR or staging upload** in the
workflow; a human flips the final switch. Specifically:

- **CRAN**: the tarball is attached to the GH Release. Maintainer uploads via the web form after running `cran-comments.md`.
- **vcpkg / conan-center**: the workflow opens a PR to the upstream registry. Maintainer responds to CI feedback.
- **Maven Central**: artifacts are *staged* via the Central Portal API. Maintainer clicks "Publish" once smoke-tests pass.
- **MATLAB FX**: automatic on GH Release if FX integration is bound; otherwise click "Submit" in the FX dashboard.

### 5.3 Version-bump source of truth

The single canonical version is `cpp/include/pls4all/p4a_version.h`.
A pre-tag script (`scripts/bump_version.sh ${NEW_VER}`) regenerates
every downstream manifest:

| File | Generator clause |
|------|------------------|
| `CMakeLists.txt` `project(... VERSION ${X})` | sed by `bump_version.sh` |
| `bindings/python/pyproject.toml` | sed |
| `bindings/r/pls4all/DESCRIPTION` | sed |
| `bindings/julia/Pls4all.jl/Project.toml` | sed |
| `bindings/rust/pls4all/Cargo.toml` | `cargo set-version ${X}` |
| `bindings/js/package.json` | `npm version ${X} --no-git-tag-version` |
| `bindings/dotnet/Pls4all/Pls4all.csproj` `<Version>` | sed |
| `bindings/ruby/lib/pls4all/version.rb` | sed |
| `bindings/lua/pls4all-${VER}-1.rockspec` | filename rename + content sed |
| `bindings/nim/pls4all.nimble` | sed |
| `bindings/android/pls4all-android/build.gradle.kts` `version =` | sed |
| `CITATION.cff` `version:` | sed |
| `docs/abi/changelog.md` | hand-edited per phase |

The CI check `.github/workflows/version-sync.yml` and
`scripts/bump_version.sh` are now present. Before any registry
publication, run:

```bash
scripts/bump_version.sh --check
```

and make the workflow fail PRs when any manifest drifts from
`p4a_version.h`. If another manifest family is added, extend the script
first and then add the new package.

### 5.4 Today's CI workflows — gaps to fill

What we already have:
- `.github/workflows/ci.yml` — multi-OS C++ build matrix.
- `.github/workflows/abi-check.yml` — ABI symbol diff + dep deny-list.
- `.github/workflows/parity-gate.yml` — fixture determinism + numerical parity.
- `.github/workflows/sanitizers.yml` — ASan/UBSan/TSan.
- `.github/workflows/coverage.yml` — lcov via Codecov.
- `.github/workflows/docs.yml` — Sphinx → GitHub Pages.

**Net new workflows we still need to add** (this is the actionable
gap list):

| Workflow | Purpose | Trigger |
|----------|---------|---------|
| `release.yml` | the orchestration described in §5 | `push: tags` |
| `release-source.yml` | source tarball + SBOM + provenance | called from `release.yml` |
| `release-binaries.yml` | the 14-triple cross-build matrix | called from `release.yml` |
| `release-python.yml` | `cibuildwheel` × PyPI + TestPyPI | called from `release.yml` |
| `release-r.yml` | `R CMD build` + R-universe push + CRAN tarball | called from `release.yml` |
| `release-matlab.yml` | `matlab-actions` → `.mltbx` | called from `release.yml` |
| `release-js.yml` | npm + JSR + CDN | called from `release.yml` |
| `release-rust.yml` | `cargo publish` | called from `release.yml` |
| `release-dotnet.yml` | `dotnet pack` + NuGet push (managed + 6 native RIDs) | called from `release.yml` |
| `release-julia.yml` | nothing — Yggdrasil and Registrator are external; this workflow only attaches Project.toml diff to the release notes for transparency | called from `release.yml` |
| `release-ruby.yml` | `gem build` + `gem push` | called from `release.yml` |
| `release-lua.yml` | `luarocks upload` | called from `release.yml` |
| `release-nim.yml` | PR to `nim-lang/packages` | called from `release.yml` |
| `release-maven.yml` | gradle publishToCentralPortal | called from `release.yml` |
| `release-docker.yml` | `docker buildx` multi-arch | called from `release.yml` |
| `release-homebrew.yml` | PR to `GBeurier/homebrew-pls4all` | called from `release.yml` |
| `prerelease-nightly.yml` | every-day cross-binding parity gate on `main` against an unbumped `v0.97.0-dev.${SHORTSHA}` | `schedule: cron 'nightly'` |
| `smoke-installed.yml` | weekly pulls the last released `pls4all` from every registry and reruns the parity fixture — catches registry drift / outages | `schedule: cron 'weekly'` |

---

## 6. New-channel checklist

When adding a new binding or registry, follow this 10-step gate. The
checklist is reproduced inside each `bindings/${LANG}/CHECKLIST.md`.

1. **License**: confirm SPDX `CECILL-2.1` or accepted equivalent. If not, halt.
2. **Identity**: claim the name on the registry (empty placeholder OK).
3. **Manifest**: add the language's package manifest (`pyproject.toml`, `gemspec`, `nimble`, `rockspec`, etc.) with a `version` field tied to the canonical version source.
4. **Source layout**: respect `bindings/${LANG}/` with the README, the manifest, `src/`, and `tests/`.
5. **Parity gate**: implement the SIMPLS parity test against `bindings/_catalog/parity_fixture.json` so the binding fails CI if any numerical drift exceeds `rmse_rel < 1e-12`.
6. **Loader rules**: documented prioritised path discovery for `libp4a` — `${LIB}_LIB_PATH` env var first, then bundled, then system.
7. **Smoke test**: in CI on every PR.
8. **Release wiring**: a `release-${LANG}.yml` workflow callable from `release.yml`.
9. **Docs**: a one-page `docs/bindings/${LANG}.md` linked from the README binding-matrix table.
10. **README badge**: add the registry's version badge to the project README's matrix.

The checklist is enforced by `scripts/check_binding_complete.sh
bindings/${LANG}/`, which the `version-sync.yml` workflow runs.

---

## 7. Proactive additions worth considering

Beyond what the user explicitly mentioned, these channels would
extend pls4all's reach with disproportionate ROI:

### 7.1 Channels we should add now (v1.0)

- **conda-forge** (Python + standalone `libp4a`) — single largest reach into the scientific Python world. ~70% of conda users install through conda-forge.
- **R-universe** — mandatory pre-CRAN proving ground; users get binaries hours after a tag.
- **JSR** (https://jsr.io) — Deno/Bun's preferred registry, gradually displacing npm for new JS work. Zero cost to publish alongside npm.
- **Maven Central** — the only JVM registry serious consumers will use. Required before Android, IDE plugins, or Kotlin DataFrame integrations make sense.
- **Homebrew tap** + **vcpkg** + **Conan** — the three places C/C++ users actually look.
- **Docker GHCR** — free, GitHub-native, no rate limits for our scale.
- **MSYS2 / MinGW-packages** — recipe path `mingw-w64-pls4all/PKGBUILD` in
  the `msys2/MINGW-packages` repo. Targets: `mingw-w64-ucrt-x86_64-pls4all`,
  `mingw-w64-clang-x86_64-pls4all`, `mingw-w64-clang-aarch64-pls4all`.
  This is where Windows C/C++ users actually get MinGW/UCRT libraries —
  the doc already ships `x86_64-pc-windows-mingw` triple archives but
  MSYS2 is the natural distribution surface for those.
- **Octave Packages** — package name `pls4all`, archive
  `pls4all-octave-${VER}.tar.gz`, PR to `gnu-octave/packages/index.yaml`.
  Treated as its own channel, not as a MATLAB sub-item. The Octave
  code paths and MEX `.oct` files live next to the MATLAB ones in
  `bindings/matlab/` but the distribution surface is independent.
- **MacPorts** — `Portfile` at `dports/science/pls4all/Portfile` in
  `macports/macports-ports`. Captures macOS scientific users who do
  not use Homebrew, especially institutional machines.
- **FreeBSD Ports** — `math/pls4all` or `science/pls4all`, package
  `pkg install pls4all`. Low-cost C/C++ surface; aligns with our
  Unix-portability story if FreeBSD becomes a Julia JLL target.
- **GitHub Packages** (fallback / RC channel) — Maven
  `maven.pkg.github.com/GBeurier/pls4all`, NuGet
  `https://nuget.pkg.github.com/GBeurier/index.json`, npm
  `npm.pkg.github.com` under `@pls4all/wasm`. Not a public-primary
  channel, but useful for release candidates, private downstream
  consumers, and testing immutable package layouts before Central /
  NuGet / npm.
- **Quay.io OCI mirror** — `quay.io/gbeurier/pls4all:${VER}` and
  `quay.io/gbeurier/pls4all-runner:${VER}`. Useful for OpenShift /
  HPC sites where GHCR or Docker Hub access is blocked.

### 7.2 Channels we should add at v1.1 / v1.2

- **Hugging Face Hub** — if we publish pretrained PLS models (calibrations from public NIRS datasets), this is the canonical model registry. `pls4all` could ship a `PretrainedModel` class that pulls a calibration from `huggingface.co/pls4all/wheat-protein-v1`.
- **Quarto / Posit Cloud** — for R + Python notebooks that demonstrate AOM-PLS / POP-PLS. RStudio's Posit Cloud has a free tier for open-source teaching material.
- **Observable / Observable Plot** — for the WASM binding's web-native demo notebooks. JSR-mirrored package works out of the box.
- **Pyodide registry** — pls4all already builds for `wasm32-unknown-emscripten`, so JupyterLite users get the library in-browser for free. A small PR to `pyodide/pyodide` registers it.
- **Spack** — HPC sites with no internet access download tarballs from a Spack mirror. The recipe is ~30 lines.
- **CRAN Task View: ChemPhys** — adding `pls4all` to the [ChemPhys task view](https://cran.r-project.org/web/views/ChemPhys.html) is a single PR and exposes the package to every chemometrician browsing CRAN.

### 7.3 Channels we explicitly defer

These are not worth the maintenance burden at v1.0 — re-evaluate as
consumer demand surfaces:

- **PHP / Perl / Tcl / Common Lisp / Crystal / Zig / Dart / OCaml / Haskell** — already covered in `roadmap/phase-36-deferred-bindings.md`.
- **WinGet / Chocolatey** (Windows): the `pls4all_cli` is binary-only; Windows users typically install via `pip` or `nuget`, not via a CLI package manager.
- **Snap / Flatpak** (Linux desktop app stores): pls4all is a library, not a GUI app. Wrong audience.
- **Microsoft Store**, **App Store** — same reason.

### 7.4 Channels we keep as future-defensive

- **CodeArtifact** (AWS), **Artifactory** (JFrog), **Cloudsmith** — only when an enterprise downstream needs an air-gapped mirror.
- **CITATION.cff registries** (Zotero, ORCID) — partially automatic from `CITATION.cff`. No action required as long as the file stays current.

---

## 8. Open questions / decisions still owed

These are points where the document has *picked a default* but the
maintainer may want to revisit:

1. **CRAN's vendored static build vs. SystemRequirements**. The doc
   commits to the vendored static build (lower install friction).
   Trade-off: ~5 min compile on CRAN's slowest farm. If CRAN
   complains, fall back to the SystemRequirements path.
2. **`pls4all-sys` companion crate.** The doc defers it (single
   `pls4all` crate is enough for v1.0). Revisit when a Rust
   consumer asks for build-time `libp4a` bundling.
3. **Android NDK packaging timing.** The doc treats it as in-scope
   for the Maven Central artifact; current ROADMAP defers it to
   "phase 36 unfreeze". The two have to converge before v1.0.
4. **Single `libp4a` build vs. `libp4a-blas-omp` separate binary.**
   The doc ships both as separate archives. Alternative: a single
   archive with the BLAS/OpenMP backend dlopened at runtime. The
   latter halves the asset count but adds a new abstraction layer.
5. **Trusted publishing (OIDC) vs. PATs everywhere.** The doc
   prefers OIDC (PyPI, npm, JSR, NuGet, RubyGems all support it).
   Maintainer must wire trust relationships per registry before
   v1.0 cut.
6. **Reproducibility scope.** The doc requires `SOURCE_DATE_EPOCH`
   for the C++ archives. Should we extend that to the Python wheel
   (cibuildwheel supports it) and the .NET nupkg? Default: yes,
   chase reproducibility on every artifact we control.

---

## 9. Quick reference — registry → manifest path

```
PyPI               → bindings/python/pyproject.toml
conda-forge        → external feedstock: conda-forge/pls4all-feedstock + libp4a-feedstock
CRAN               → bindings/r/pls4all/DESCRIPTION
R-universe         → external repo: gbeurier/universe → packages.json
MATLAB File Ex.    → bindings/matlab/toolbox.prj (.mltbx generator)
npm                → bindings/js/package.json
JSR                → bindings/js/jsr.json (mirror of package.json)
crates.io          → bindings/rust/pls4all/Cargo.toml
NuGet              → bindings/dotnet/Pls4all/Pls4all.csproj (managed)
                   + bindings/dotnet/Pls4all.Native/Pls4all.Native.csproj (multi-RID native, TO CREATE)
Julia General      → bindings/julia/Pls4all.jl/Project.toml + Yggdrasil PR for the JLL
RubyGems           → bindings/ruby/pls4all.gemspec        (TO CREATE)
LuaRocks           → bindings/lua/pls4all-${VER}-1.rockspec (TO CREATE)
Nimble             → bindings/nim/pls4all.nimble           (TO CREATE)
Maven Central      → bindings/jni/pom.xml + bindings/android/build.gradle.kts (TO CREATE)
Homebrew tap       → external repo: GBeurier/homebrew-pls4all/Formula/pls4all.rb (TO CREATE)
vcpkg              → external PR: microsoft/vcpkg/ports/pls4all (TO CREATE)
Conan Center       → external PR: conan-io/conan-center-index/recipes/pls4all (TO CREATE)
GHCR Docker        → docker/Dockerfile + .github/workflows/release-docker.yml (TO CREATE)
GitHub Releases    → .github/workflows/release.yml (TO CREATE)
Zenodo             → no manifest; activated once via GitHub-Zenodo integration
Software Heritage  → no manifest; passive
```

---

## 10. Status (as of 2026-05-18)

| Surface | Status | Blocker |
|---------|--------|---------|
| GitHub Releases (binaries) | ⬜ not yet wired | needs `release.yml` + multi-OS asset matrix |
| PyPI | ⬜ name not reserved | needs first 0.0.0 placeholder upload |
| CRAN | ⬜ never submitted | needs zero-warning `R CMD check --as-cran`, vendored-core sync, and first R-universe run |
| R-universe | ⬜ universe not created | needs `gbeurier/universe` repo + packages.json |
| MATLAB File Exchange | ⬜ no FX submission | needs `.mltbx` packager job + FX account |
| npm `@pls4all/wasm` | 🟡 package.json present, never published | needs npm scope + first publish |
| JSR | ⬜ not configured | mirror of npm — add after first npm publish |
| pkg.go.dev | 🟡 module exists, not tagged in subdir convention | needs `bindings/go/v*` tag |
| crates.io | ⬜ never published | needs first `cargo publish` |
| NuGet | ⬜ never published | needs `Pls4all` (managed) + `Pls4all.Native` (multi-RID native) |
| Julia General | ⬜ never registered | needs Registrator comment + Yggdrasil PR for JLL |
| RubyGems | ⬜ no gemspec | needs `pls4all.gemspec` + first `gem push` |
| LuaRocks | ⬜ no rockspec | needs `*.rockspec` + first `luarocks upload` |
| Nimble | ⬜ no nimble file | needs `pls4all.nimble` + PR to nim-lang/packages |
| Maven Central JNI | ⬜ never published | needs Sonatype Central Portal account + GPG key + Gradle wiring |
| Maven Central Android | 🔒 deferred per roadmap | unblocked when NDK is in the build host |
| Homebrew tap | ⬜ tap repo not created | needs second repo `homebrew-pls4all` |
| vcpkg port | ⬜ no port | needs PR to microsoft/vcpkg |
| Conan Center | ⬜ no recipe | needs PR to conan-io/conan-center-index |
| Docker GHCR | ⬜ no images | needs Dockerfile + `release-docker.yml` |
| Quay.io mirror | ⬜ no images | optional; mirrors GHCR via `skopeo copy` |
| MSYS2 PKGBUILD | ⬜ no PKGBUILD | low-priority PR to `msys2/MINGW-packages` |
| Octave Packages | ⬜ no submission | PR to `gnu-octave/packages` once `.oct` baked into the release |
| MacPorts Portfile | ⬜ no Portfile | low-priority PR to `macports/macports-ports` |
| FreeBSD Ports | ⬜ no port | low-priority PR to `freebsd/freebsd-ports` |
| GitHub Packages (RC) | ⬜ not wired | reuse `release-*.yml` workflows with `--source github` |
| Zenodo DOI | ⬜ not enabled | one-click toggle in zenodo.org settings |
| Software Heritage | ⬜ not registered | one-time origin save |
| Version-sync CI | ✅ present | keep `scripts/bump_version.sh --check` authoritative |
| SOVERSION fix | ✅ fixed | verify clean release builds produce `libp4a.so.1 -> libp4a.so.1.16.0` |

Every ⬜ is a small, isolated unit of work. The first cut should
land in this order:

0. **Operational risk-zero step**: keep ABI identity checked from a clean
   build and refresh the expected symbol snapshot intentionally. Several
   registries produce immutable packages once pushed; a stale local
   symlink or accidental new public symbol can poison downstream packages.
1. Keep `version-sync.yml` + `bump_version.sh --check` green before every
   tag.
2. `release.yml` skeleton + `release-binaries.yml` (gives us the
   GitHub-Release-as-source-of-truth, with the binary-ABI gate from
   §2.2 enforcing `libp4a.so.${ABI_MAJOR}`).
3. Wire OIDC trusted publishing where supported (PyPI, npm, JSR, NuGet,
   RubyGems). Do **not** reserve names with placeholder versions until
   step 0+1+2 land — placeholder packages on immutable registries
   create their own cleanup problems.
4. Reserve every name in §1.3 with **real** 0.97.0 (or `0.97.0-rc.1`)
   uploads driven by the new release workflow.
5. PyPI + conda-forge + R-universe (the three highest-reach channels).
6. CRAN + Maven Central (highest reach but human-review cost).
7. Everything else in parallel.

### Single biggest threat to a smooth v1.0 cut

Publishing immutable packages with the wrong version and/or wrong ABI
identity. The remaining risks are (a) manifest drift if
`bump_version.sh --check` is bypassed, (b) stale local build artifacts
being copied into wheels or release archives, and (c) accidental new
public `p4a_*` exports that are not reflected in the ABI snapshot.
Registries cannot be cleanly overwritten, so every release job must smoke
test the built artifact rather than the checkout.

---

## 11. Responsibilities split — what Claude does vs what you (maintainer) must do

Every distribution channel splits into three classes of action:

- **🤖 Claude alone** — code, manifests, CI workflows, scripts, ports, recipes, smoke tests, PR drafts. No credentials needed.
- **👤 You alone** — account creation, password/2FA, generating API tokens, configuring OIDC trust, GPG keys, web-form submissions, replying to human reviewers.
- **🤖👤 Joint** — I prepare the artifact; you click "Publish" / approve a PR / answer a reviewer mail.

### 11.1 Universal one-time chores (do these first, they unlock everything)

| # | Who | Action |
|---|-----|--------|
| U1 | 👤 | Generate a GPG key (`gpg --full-gen-key`, ed25519), publish on `keys.openpgp.org`. Needed by Maven Central, signed git tags, signed Releases. |
| U2 | 👤 | Enable 2FA on every account: GitHub (recovery codes saved), PyPI, npm, crates.io, RubyGems, NuGet, Maven Central, MathWorks, RubyGems, JuliaHub, Docker Hub. Most registries now refuse publication without it. |
| U3 | 👤 | (Optional) issue a fine-grained PAT to me (`gh auth login --with-token`) with scope `contents:write, packages:write, pull-requests:write` if you want me to open upstream PRs (vcpkg, conan-center, etc.) from your account. Otherwise I deliver branches/diffs and you push. |
| U4 | 🤖 | Maintain `scripts/bump_version.sh` and `.github/workflows/version-sync.yml`; extend them whenever a new manifest appears. |
| U5 | 🤖 | Keep the SOVERSION preflight green from clean builds and prevent stale `libp4a.so.0` artifacts from entering packages. |
| U6 | 🤖 | Write `.github/workflows/release.yml` + `release-binaries.yml` (14-triple cross-build) — gives the canonical GitHub-Release-as-source-of-truth. |

### 11.2 Per-channel split

Legend: 🤖 = Claude can do it alone · 👤 = you only · 🤖👤 = joint.

#### Tier A — your immediate focus (CRAN + PyPI)

| Channel | Code / manifests / CI | Account, tokens, web actions | Recurring human cost |
|---------|----------------------|------------------------------|----------------------|
| **CRAN `pls4all`** | 🤖 vendored static build in `bindings/r/pls4all/src/`, normalised `DESCRIPTION`, roxygen2 docs with `@return` everywhere, `tests/testthat/`, runnable `@examples`, `cran-comments.md`, `release-r.yml` | 👤 (1) R-hub v2: either set up GitHub-Actions checks via `rhub::rhub_setup()` / `rhub::rhub_check()` (needs a GitHub PAT in your local git credential store), or use R Consortium runners via `rhub::rc_new_token()` / `rhub::rc_submit()` (the legacy `validate_email()` / `check_for_cran()` flow is R-hub v1 and superseded), (2) submit tarball at `https://cran.r-project.org/submit.html` (5 min, manual web form), (3) confirm submission via the e-mail link CRAN sends | 🤖👤 CRAN review typically takes a few days; allow up to **10 working days** for a first submission. You forward each reviewer e-mail; I patch; resubmission uses the same web form with the *optional comment* field updated. |
| **R-universe `gbeurier.r-universe.dev`** | 🤖 the `packages.json` file contents | 👤 create the GitHub repo `gbeurier/gbeurier.r-universe.dev` (5 min, no special perms beyond your normal GH account) | none |
| **PyPI `pls4all`** | 🤖 normalised `pyproject.toml`, `setup.py` shim, `bindings/python/scripts/retag_wheels.py`, `bindings/python/scripts/build_libp4a_in_wheel.sh`, `release-python.yml` with `cibuildwheel` matrix, `environment: pypi` + `permissions: id-token: write` in the job | 👤 (1) PyPI account (https://pypi.org/account/register/), (2) reserve name `pls4all` — **TestPyPI does not reserve PyPI names**; reserve on real PyPI via the first `0.97.0rc0` Trusted-Publishing upload, or by uploading an explicit placeholder, or by claiming via PyPI support if taken, (3) enable **Trusted Publishing** at `https://pypi.org/manage/account/publishing/` — five fields for a *new* project: PyPI Project Name=`pls4all`, owner=`GBeurier`, repo=`pls4all`, workflow filename=`release-python.yml`, environment=`pypi` (optional but recommended). Replaces any need for long-lived tokens. | none (OIDC handles auth on every tag) |
| **TestPyPI `pls4all`** | 🤖 same workflow with `--repository testpypi` | 👤 separate account on `https://test.pypi.org/` + identical Trusted Publishing setup | none |

#### Tier B — high-reach, plug onto the same release pipeline

| Channel | Code | Account / tokens | Recurring |
|---------|------|------------------|-----------|
| **conda-forge** (two feedstocks: `libp4a` + `pls4all`) | 🤖 both `meta.yaml` recipes + PR to `conda-forge/staged-recipes` | 👤 conda-forge-admin GitHub app authorisation on your account (1 click during PR) | 🤖👤 reviewer comments, you forward to me |
| **npm `@pls4all/wasm`** | 🤖 `release-js.yml`; **provenance is automatic** under npm Trusted Publishing (no need to pass `--provenance`) | 👤 (1) create npm org `@pls4all` (free, 5 min), (2) configure npm Trusted Publishing OIDC → repo `GBeurier/pls4all`. Requires **npm >= 11.5.1**, **Node >= 22.14.0**, and GitHub-hosted runners. | none |
| **JSR `@pls4all/wasm`** | 🤖 `jsr.json` mirror of npm manifest + workflow | 👤 create JSR scope `@pls4all` at `https://jsr.io/new`; bind to GitHub repo via OIDC (1 click) | none |
| **crates.io `pls4all`** | 🤖 `Cargo.toml` cleanup + `release-rust.yml` | 👤 crates.io login (GitHub OAuth, 30 sec) + paste API token in repo secret `CARGO_REGISTRY_TOKEN`. crates.io has no public OIDC yet. | none |

#### Tier C — JVM, .NET, niche-scientific (later wave)

| Channel | Code | Account / tokens | Recurring |
|---------|------|------------------|-----------|
| **NuGet `Pls4all` + `Pls4all.Native`** | 🤖 managed csproj + native multi-RID csproj + `release-dotnet.yml` | 👤 NuGet.org account + Trusted Publishing OIDC **if available on the account** (Microsoft documents a gradual rollout — keep an API-key fallback in `NUGET_API_KEY` until confirmed). | none, or `NUGET_API_KEY` fallback |
| **Maven Central JNI** | 🤖 `build.gradle.kts` + JReleaser config | 👤 (1) Sonatype Central Portal account, (2) **verify namespace** `io.github.gbeurier` — may auto-provision if you sign up via GitHub `GBeurier`, otherwise create a temporary public verification repo per Sonatype's instructions, (3) GPG key from U1 published, (4) paste 4 secrets: `MAVEN_GPG_PRIVATE_KEY`, `MAVEN_GPG_PASSPHRASE`, `CENTRAL_USERNAME`, `CENTRAL_TOKEN` | 🤖👤 click "Publish" in Central Portal UI for the first 1-2 releases; switch to auto-publish in JReleaser config afterwards |
| **Maven Central Android AAR** | 🔒 deferred (Android NDK not present on host — `roadmap/phase-36-deferred-bindings.md`) | — | — |
| **Julia General `Pls4all.jl`** | 🤖 keep `Project.toml` in sync | 👤 install GitHub App `JuliaRegistrator` on `GBeurier/pls4all` (1 click) | 🤖👤 comment `@JuliaRegistrator register subdir=bindings/julia/Pls4all.jl` on the release commit each tag (I can do it via `gh` if U3 is granted) |
| **`Pls4all_jll` via Yggdrasil** | 🤖 write `P/Pls4all/build_tarballs.jl` + PR to `JuliaPackaging/Yggdrasil` | 👤 nothing beyond the GitHub OAuth on your account (PR is from your fork) | 🤖👤 Yggdrasil reviewer comments |
| **RubyGems `pls4all`** | 🤖 `bindings/ruby/pls4all.gemspec` + `release-ruby.yml` | 👤 RubyGems account + Trusted Publishing OIDC (RubyGems supports it since 2024) | none |
| **LuaRocks `pls4all`** | 🤖 `pls4all-${VER}-1.rockspec` + `release-lua.yml` | 👤 LuaRocks account + API key in secret `LUAROCKS_API_KEY` (no OIDC) | none |
| **Nimble `pls4all`** | 🤖 `bindings/nim/pls4all.nimble` + PR to `nim-lang/packages` | 👤 nothing — PR from your GH fork | 🤖👤 nim-lang reviewer |
| **MATLAB File Exchange** | 🤖 `bindings/matlab/toolbox.prj` + `release.m` + `release-matlab.yml` using `matlab-actions/setup-matlab@v3` + `matlab-actions/run-build@v3` | 👤 (1) MathWorks account (covered by your CIRAD address), (2) bind FX submission to `GBeurier/pls4all` via MatlabCentral UI (one-time), (3) licensing: `matlab-actions/setup-matlab` **automatically licenses supported products on public GitHub projects** — no secret needed for the typical case. For private repos *or* transformation products (Coder/Compiler), use a **batch licensing token** in secret `MLM_LICENSE_TOKEN`. External language interfaces (calling MEX) sometimes need a self-hosted normally-licensed runner. | none (FX auto-pulls `.mltbx` from new GH Release) |
| **Octave Packages** | 🤖 PR to `gnu-octave/packages/index.yaml` | 👤 nothing | 🤖👤 Octave Packages reviewer |

#### Tier D — C/C++ ecosystem surfaces

| Channel | Code | Account / tokens | Recurring |
|---------|------|------------------|-----------|
| **GitHub Releases (binaries)** | 🤖 `release-binaries.yml` (14 triples) | 👤 nothing — uses native `GITHUB_TOKEN` | 👤 approve/push the tag |
| **Homebrew tap `gbeurier/pls4all`** | 🤖 `Formula/pls4all.rb` + bottle-build workflow inside the tap | 👤 **create the second repo `GBeurier/homebrew-pls4all`** (1 min, public) | none — formula bump PRs are automated |
| **vcpkg port** | 🤖 `portfile.cmake` + `vcpkg.json` + `versions/p-/pls4all.json` + PR to `microsoft/vcpkg` | 👤 nothing | 🤖👤 vcpkg reviewer comments |
| **Conan Center Index** | 🤖 `recipes/pls4all/all/conanfile.py` + PR to `conan-io/conan-center-index` | 👤 nothing | 🤖👤 CCI reviewer comments |
| **Docker GHCR `ghcr.io/gbeurier/pls4all`** | 🤖 `Dockerfile` + `release-docker.yml` (multi-arch buildx) | 👤 nothing — uses `GITHUB_TOKEN` | none |
| **Docker Hub mirror** | 🤖 parallel push in same workflow | 👤 Docker Hub account + token `DOCKERHUB_TOKEN` | none |
| **Quay.io mirror** | 🤖 `skopeo copy` step | 👤 Quay.io account + robot token | none |
| **MSYS2 `mingw-w64-pls4all`** | 🤖 PKGBUILD + PR to `msys2/MINGW-packages` | 👤 nothing | 🤖👤 MSYS2 reviewer |
| **MacPorts `science/pls4all`** | 🤖 Portfile + PR to `macports/macports-ports` | 👤 nothing | 🤖👤 MacPorts reviewer |
| **FreeBSD Ports** | 🤖 Makefile + PR to `freebsd/freebsd-ports` | 👤 may need a FreeBSD Bugzilla account if their workflow demands it | 🤖👤 FreeBSD ports reviewer |

#### Tier E — defensive / archival / mirroring

| Channel | Who | Action |
|---------|-----|--------|
| **Zenodo DOI** | 👤 | one toggle at `https://zenodo.org/account/settings/github/` to enable `GBeurier/pls4all`; every future Release auto-generates a DOI |
| **Software Heritage** | 👤 | one visit to `https://archive.softwareheritage.org/save/origin/?origin_url=https://github.com/GBeurier/pls4all` (~30 sec) |
| **Codeberg / GitLab mirror** | 👤 | create the accounts + repos + enable pull-mirror in their UIs (5 min each, optional) |
| **GitHub Packages (RC channel)** | 🤖 | reuse the per-language workflows with `--source github`; uses native `GITHUB_TOKEN`, no new secret |

### 11.3 Quick map of secrets you'll have collected by v1.0

The `Settings → Secrets and variables → Actions` of `GBeurier/pls4all` will hold, **at most**:

| Secret | Tier | Notes |
|--------|------|-------|
| (none for PyPI) | A | Trusted Publishing OIDC replaces this |
| (none for npm) | B | Trusted Publishing OIDC |
| (none for JSR) | B | OIDC |
| `CARGO_REGISTRY_TOKEN` | B | crates.io has no OIDC yet |
| (none for RubyGems) | C | OIDC |
| `LUAROCKS_API_KEY` | C | no OIDC |
| `MAVEN_GPG_PRIVATE_KEY`, `MAVEN_GPG_PASSPHRASE`, `CENTRAL_USERNAME`, `CENTRAL_TOKEN` | C | Maven Central |
| (none for NuGet) | C | Trusted Publishing OIDC |
| `MLM_LICENSE_TOKEN` *(none needed for supported products on public GitHub Actions)* | C | MathWorks batch licensing token |
| `NUGET_API_KEY` *(fallback only)* | C | NuGet — until Trusted Publishing OIDC is confirmed available on the account |
| `DOCKERHUB_TOKEN` | D | Docker Hub mirror |
| `QUAY_ROBOT_TOKEN` | D | Quay.io mirror |

GitHub-native tokens (`GITHUB_TOKEN` for GHCR + GH Packages + GH Releases) are never stored — they're injected per-workflow-run automatically.

---

## 12. Fast track to CRAN + PyPI

Target: **`install.packages("pls4all")` on CRAN + `pip install pls4all` on PyPI**, both green, both reproducible from a tag. Three milestones; each is a discrete piece of work you can pause on.

### 12.1 Milestone M0 — foundations verification (~0.5 working day)

Most M0 foundation work has landed: SOVERSION follows the ABI major,
`scripts/bump_version.sh` exists, and `version-sync.yml` is present.
Before any registry sees the project, verify those guard rails from a
clean checkout and close the remaining ABI snapshot/doc gaps.

| # | Who | Step | Output |
|---|-----|------|--------|
| M0.1 | 🤖 | Verify clean Linux builds produce `libp4a.so -> libp4a.so.1 -> libp4a.so.1.16.0` and no stale `libp4a.so.0` is staged into packages. | ABI identity checked |
| M0.2 | 🤖 | Run `scripts/bump_version.sh --check` and ensure `version-sync.yml` covers every active manifest. | version guard green |
| M0.3 | 🤖 | Refresh `cpp/abi/expected_symbols_linux.txt` only if the extra exported symbols are intentionally public. | ABI snapshot intentional |
| M0.4 | 🤖 | Re-run `ctest --preset dev-release --output-on-failure`. | C++ gate green |
| M0.5 | 👤 | Approve the verification diff, especially ABI symbols and package metadata. | tag-ready |

**Exit criteria.** `git tag -a v0.97.0 -m "M0 baseline"` produces a tree
where every binding manifest, the C++ project, the C ABI version, the ABI
symbol snapshot and the documentation agree.

### 12.2 Milestone M1 — PyPI publication (target: 2–3 working days)

Independent of M2. Can run in parallel.

| # | Who | Step | Output |
|---|-----|------|--------|
| M1.1 | 🤖 | Refactor `bindings/python/src/pls4all/_ffi.py` to also look in `<package>/.libs/` (wheel layout) | loader works for both editable and wheel install |
| M1.2 | 🤖 | Add `bindings/python/setup.py` shim with `Distribution.has_ext_modules = lambda self: True` **and** `package_data={"pls4all": [".libs/libp4a*", ".libs/p4a*"]}` + a `MANIFEST.in` that ships the native lib under `pls4all/.libs/`. Constrain `cibuildwheel` to **one CPython per platform/arch** (`CIBW_BUILD=cp312-*`) so retag does not collide. | non-pure platform wheel that *contains* `libp4a` |
| M1.3 | 🤖 | Add `bindings/python/scripts/build_libp4a_in_wheel.sh` (invoked by `CIBW_BEFORE_ALL_LINUX`/`MACOS`/`WINDOWS`) — builds the C++ core then copies `libp4a.so.1.16.0` / `libp4a.1.16.0.dylib` / `p4a.dll` into `bindings/python/src/pls4all/.libs/` | native blob present at wheel-build time |
| M1.4 | 🤖 | Add `bindings/python/scripts/retag_wheels.py` — runs **after** repair, rewrites three things in lockstep: the filename `cp312-cp312-*.whl` → `py3-none-*.whl`, the `*.dist-info/WHEEL` `Tag:` metadata lines, and `*.dist-info/RECORD` hashes/paths. Renaming only the file produces a wheel `pip` rejects. | valid `py3-none-${platform}` wheels |
| M1.5 | 🤖 | Verify `.github/workflows/release-python.yml`: `permissions: id-token: write`, `environment: pypi`, `cibuildwheel` matrix, repair commands for all three OSes, retag step after repair, and an installed-wheel smoke job that runs `pip install ./wheelhouse/pls4all-*.whl` in a clean venv and checks `pls4all.abi_version() == (1, 16, 0)` | reproducible wheel matrix + green smoke |
| M1.6 | 🤖 | Local dry run: build a single Linux wheel + run `pls4all.tests.test_smoke` | passes locally |
| M1.7 | 👤 | Create a PyPI account if needed; verify e-mail; **enable 2FA** | unblocks publishing |
| M1.8 | 👤 | Configure **Trusted Publishing** on PyPI at `https://pypi.org/manage/account/publishing/`: owner=`GBeurier`, repo=`pls4all`, workflow=`release-python.yml`, environment=`pypi`. Repeat on TestPyPI. | OIDC link |
| M1.9 | 👤 | Reserve the name `pls4all` on **real PyPI** — TestPyPI does *not* reserve PyPI names. Either rely on the pending-publisher + first real `0.97.0rc0` upload (M1.10/M1.12), or upload an explicit placeholder, or claim via PyPI support if the name is taken. | name claimed |
| M1.10 | 🤖 + 👤 | Trigger a release-candidate workflow against **TestPyPI** first: tag `v0.97.0rc0`, watch the run, confirm wheels land on `https://test.pypi.org/project/pls4all/0.97.0rc0/` | green RC |
| M1.11 | 🤖 + 👤 | Smoke test (note `--extra-index-url` for `numpy` etc. that don't live on TestPyPI): `pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple pls4all==0.97.0rc0 && python -c "import pls4all; pls4all.selftest()"` on Linux + macOS + Windows | parity OK |
| M1.12 | 👤 | Promote to PyPI: tag `v0.97.0`, the workflow publishes to the real `pypi.org` | live |
| M1.13 | 🤖 | Verify `pip install pls4all==0.97.0` from a clean venv on Linux/macOS/Windows in CI | green |

**Exit criteria.** `pip install pls4all` works in a fresh venv on every supported platform, the wheel actually bundles `libp4a` (verify with `unzip -l pls4all-0.97.0-py3-none-*.whl | grep libp4a`), the `WHEEL` metadata's `Tag:` lines match the filename, and `python -c "import pls4all; print(pls4all.abi_version())"` prints `(1, 16, 0)` from the *uploaded* artifact (not from a local dev install).

### 12.3 Milestone M2 — CRAN publication (target: 5–10 working days, gated by CRAN reviewer)

CRAN has a human reviewer in the loop. M2.4–M2.7 are the *only* irreducible humans-in-loop steps; everything before is code.

| # | Who | Step | Output |
|---|-----|------|--------|
| M2.1 | 🤖 | Audit the current **vendored static** R mode: regenerate the vendored `libp4a` subset from `cpp/`, remove divergent PCR code, use R compiler macros only (`CXX_STD = CXX17`, `PKG_CPPFLAGS`, `PKG_CXXFLAGS`, `PKG_LIBS`), remove non-portable flags such as `-march=*`, avoid CMake/internet at install, and keep example/test runtime under CRAN's per-check budget. | self-contained CRAN source package |
| M2.2 | 🤖 | Document every **exported** R function via `roxygen2`, including `@return` (mandatory for `\value{}` in the `.Rd`) and a runnable `@examples` block (any non-runnable example must be in `\dontrun{}` or `\donttest{}` with justification). Update `NAMESPACE` via `devtools::document()`. Add `tests/testthat.R`, `tests/testthat/test-*.R`, `Suggests: testthat (>= 3.0.0)`, `Config/testthat/edition: 3` in `DESCRIPTION`. Add a minimal vignette `vignettes/pls4all.Rmd` (or at least a strong `README.Rmd` rendered to `README.md`) — scientific packages without a vignette draw reviewer questions. | CRAN-clean docs + tests + (optional) vignette |
| M2.3 | 🤖 | Write `bindings/r/pls4all/cran-comments.md`: "New submission" line, justification of vendored sources (mention CECILL-2.1, vendored from `cpp/`, no external libs), compile-time budget, list of platforms tested via R-hub v2 / win-builder / macbuilder, list of `R CMD check --as-cran` results. Update the optional comment field on each resubmission instead of editing this file. | submission text |
| M2.4 | 🤖 | Verify `.github/workflows/release-r.yml`: `R CMD check --as-cran` on **the built tarball** across the matrix, `rcmdcheck::rcmdcheck()` artifacts, and fail on any `error`, `warning` or unintended `note`. | CI gate |
| M2.5 | 👤 | R-hub v2 onboarding — choose one path: **(a)** GitHub-Actions runners via `rhub::rhub_setup()` then `rhub::rhub_check()` (needs a GitHub PAT in your local git credential store); **(b)** R Consortium runners via `rhub::rc_new_token()` then `rhub::rc_submit()` (mail-token confirmation flow). The legacy `validate_email()` / `check_for_cran()` you may remember from R-hub v1 is superseded. | rhub unlocked |
| M2.6 | 🤖 | Run CRAN-like checks on the **built tarball** through R-hub v2 (selected platforms incl. `linux`, `macos-arm64`, `windows`), plus **win-builder** (`devtools::check_win_devel()`) and **macbuilder** (`devtools::check_mac_release()`). All must be green before submission. | CRAN-platform clean |
| M2.7 | 👤 | Build the final tarball: `R CMD build bindings/r/pls4all` → `pls4all_0.97.0.tar.gz` (I run the command; this row is your sign-off after eyeballing the diff). | tarball |
| M2.8 | 👤 | **Upload to CRAN**: open `https://cran.r-project.org/submit.html` → upload `pls4all_0.97.0.tar.gz` + paste `cran-comments.md` → submit (5 min web form). CRAN sends a confirmation e-mail; click the link. | submission queued |
| M2.9 | 👤 + 🤖 | Wait for CRAN reviewer — **a few days typically; allow up to 10 working days for a first submission**. CRAN policy forbids further submissions while one is pending. Forward each mail to me; I patch; **if code changes**, bump **patch** version `0.97.0 → 0.97.1` (CRAN convention) and resubmit via the web form with an updated *optional comment* field. If we want to revise *before* CRAN has reviewed (e.g. self-spotted issue), the same tarball name with bumped patch is the safest path; in-place `0.97.0` revisions are only acceptable if never actually queued. | accepted |
| M2.10 | 🤖 | Once accepted, R-universe rolling release lands automatically (M2.11 prerequisite). Verify `install.packages("pls4all", repos = c("https://cloud.r-project.org"))` works on macOS/Windows binaries (CRAN builds those itself within a few days). | live |
| M2.11 | 👤 | Create the GH repo `gbeurier/gbeurier.r-universe.dev` (5 min). I write the `packages.json` content. R-universe will then have rolling binaries ahead of CRAN's. | rolling RC channel |

**Exit criteria.** `install.packages("pls4all")` works in vanilla R on Linux / macOS / Windows. Source tarball and macOS / Windows binary builds are visible on the CRAN package page.

### 12.4 Critical path summary

```
[ M0 (1d) ] ─┬─► [ M1: PyPI       (2–3d)  ] ──► PyPI live
             │
             └─► [ M2: CRAN code  (2–3d)  ] ──► [ CRAN review (1–5d, human) ] ──► CRAN live
```

If you focus on **what only you can do** (the `👤` rows above):

- M0.5 — approve the foundation verification diff (~10 min review).
- M1.7–M1.9 — PyPI account, 2FA, Trusted Publishing config (×2 for PyPI + TestPyPI), name reservation, environment protection rules (~30–45 min if no account exists, ~15 min if accounts exist).
- M2.5 — R-hub v2 onboarding (~10 min if GitHub PAT path; longer for token-flow).
- M2.7 — sign-off on the built tarball after eyeballing the diff (~10 min).
- M2.8 — CRAN submission form + confirmation e-mail click (~10 min).
- M2.9 — answer the reviewer mails: typically 1–3 round trips, each ~10–20 min of your time forwarding/answering.
- Plus the *one-time* universal chores from §11.1 (U1 GPG key generation if not done, U2 enable 2FA everywhere). Realistic add: 30–60 min.

**Your total irreducible human time: realistically 90–180 minutes of action + the CRAN-review wait.** The 40-minute floor is only achievable if every account already exists, OIDC binds on the first try, and CRAN comes back with no substantive questions. Plan for the higher end. Everything *else* I can carry from inside this repo.

### 12.5 Highest-risk item — read this before clicking "Publish"

The **single highest-risk item** across the M0 → M1 → M2 path is
native-artifact metadata correctness: never publish a wheel or a
CRAN tarball whose metadata claims support for a platform/ABI
unless the bundled `libp4a` actually installs, loads, and reports
ABI `(1, 16, 0)` from the **exact uploaded artifact** (not from a
local dev install).

Concrete failure modes that all force a yank-or-`0.97.1`:
- A `py3-none-${platform}` wheel whose `WHEEL` `Tag:` metadata does
  not match the filename → `pip` rejects on install.
- A wheel that lacks `pls4all/.libs/libp4a*` because `MANIFEST.in`
  forgot to include it → `import pls4all` raises `OSError: cannot
  load libp4a` at runtime, on the user's machine, after a
  successful `pip install`.
- A CRAN source tarball whose `Makevars` references a non-portable
  flag (`-march=native`, `-Werror`, hard `clang++`) → fails on at
  least one CRAN platform → reviewer rejects.
- A vendored static build that pulls in `libgomp` or `libgfortran`
  by accident → CRAN check flags new system dependencies → reviewer
  rejects.

PyPI files are **immutable** — a bad wheel cannot be replaced under
the same version. The practical fix is `pip yank` plus a re-publish
under `0.97.1`. CRAN-side, a bad tarball is "unsubmit + resubmit
with bumped patch." Both cost ~24h of your time. The M2.6 / M1.11
smoke tests on installed-from-registry artifacts are there
specifically to prevent this.

### 12.6 Ready-to-start checklist

When you give the green light, I open work in this order, committing in small reviewable chunks:

- [ ] Branch `release/m0-verification`: M0.1 → M0.4, ABI snapshot decision, version-sync check, clean-build SOVERSION verification. Single PR.
- [ ] Branch `release/m1-pypi`: M1.1 → M1.6, then tag `v0.97.0rc0` to dry-run on TestPyPI. Verify installed-wheel smoke before promoting.
- [ ] Branch `release/m2-cran`: M2.1 → M2.4, tarball built and attached to a draft GH Release for your inspection. Verify `R CMD check --as-cran` on the *built tarball* across the matrix.
- [ ] Once M2.6 is green on R-hub v2 + win-builder + macbuilder: I hand you `pls4all_0.97.0.tar.gz` + `cran-comments.md` → you submit (M2.8).
- [ ] In parallel of CRAN review: M2.11 (create r-universe repo) + I drop `packages.json` into it.

Say "go" and I start on M0.
