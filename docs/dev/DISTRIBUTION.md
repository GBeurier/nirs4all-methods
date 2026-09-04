# Distribution Plan — nirs4all-methods

> **Scope.** This document is the single source of truth for *where*
> `nirs4all-methods` is published, *what* gets shipped to each surface, and
> *how* a release reaches it. It complements `ROADMAP.md` (what we
> build) and `docs/dev/release_process.md` (which steps a maintainer
> walks through on cut-day). All technical specifications below refer
> to the current ABI (`N4M_ABI_VERSION = 1.22.0`) and project version
> (`0.98.0` at the time of writing — see `cpp/include/n4m/n4m_version.h`).
>
> **Audience.** Maintainers + future contributors. If you are reading
> this to add a *new* binding, jump to [§6 New-channel checklist](#6-new-channel-checklist).
>
> **License gate.** The authoritative repository policy in
> [`LICENSING.md`](../../LICENSING.md) is the dual SPDX expression
> `CECILL-2.1 OR AGPL-3.0-or-later`, at the recipient's option. Registry
> metadata must use that expression where supported, and self-contained
> packages must include the complete applicable texts from
> `LICENSES/CeCILL-2.1.txt` and `LICENSES/AGPL-3.0-or-later.txt`. Do **not**
> describe the repository-root `LICENSE` as the CeCILL text: it contains the
> AGPL-3.0-or-later text only. Use the canonical SPDX casing `CECILL-2.1`; do
> not use NuGet `licenseUrl` (deprecated).

---

## 0. TL;DR distribution matrix

Three classes of artifact go out per release tag. Read this table as
"a green release ships X to Y."

| Class | Asset | Channel(s) | Cadence | Trigger |
|------|-------|------------|---------|---------|
| **A. Source** | git tag + signed source archive (`nirs4all-methods-src-${VER}.tar.gz`) + SBOM + provenance | GitHub Releases, Zenodo (DOI), Software Heritage | every tag | tag push |
| **B. C/C++ binaries** | `libn4m.{so,dylib,dll}` + `n4m_cli` + headers, per OS/arch | GitHub Releases (canonical), Homebrew tap (macOS/Linux), vcpkg port, Conan recipe (CCI), Docker `ghcr.io/gbeurier/nirs4all-methods` | every tag | tag push |
| **C. Language packages** | per-binding distributables (see §3) | language-specific registries (PyPI, CRAN, npm/JSR, Maven Central, MATLAB FileExchange, conda-forge, etc.) | every tag (after A+B succeed) | post-tag workflow |

> **Active release targets, official pre-release, and frozen PoCs.** Per the project
> `CLAUDE.md`, the **active** release targets are: **Python** (PyPI
> `nirs4all-methods` full binding + the slim `pls4all` subset),
> **R** (CRAN `n4m` full binding + the slim `pls4all` subset),
> **JS / WASM** (npm `@nirs4all/methods`), **MATLAB**, and
> **Octave**. `bindings/rust/n4m` is an **official pre-release binding**:
> it has model fit/predict, serialization, and native HPO/selection scope,
> but no crates.io publication and no cross-language prediction fixture or
> parity claim. The **Go / .NET / Ruby / Lua / Nim** bindings are frozen
> proof-of-concept bindings under `bindings/_archive/`; their historical
> channel rows remain frozen and are not live release targets.

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
| **GitHub** primary | `github.com/GBeurier/nirs4all-methods` | source, CI, Releases, Pages docs |
| **GitHub Pages** | `methods.nirs4all.org/` | docs deployed by `.github/workflows/docs.yml` on every push to `main` (live benchmark CSV → Sphinx) |
| **Read the Docs** | `nirs4all-methods.readthedocs.io/` | docs built by RTD webhook (uses committed `bench-data.json` fallback) — see §3.bis |
| **Zenodo** | `zenodo.org/account/settings/github/` enabled for `GBeurier/nirs4all-methods` | DOI per tag, archival, citable |
| **Software Heritage** | `archive.softwareheritage.org/browse/origin/?origin_url=https://github.com/GBeurier/nirs4all-methods` | passive mirror — register the origin once |
| **HAL** (optional, French academia) | `hal.science` deposit + Software Heritage swhid | only when a paper is associated |
| **JOSS** (eventually) | submission once paper drafted | software paper for the engine + AOM/POP |

### 1.2 Mirrors (defensive, low-cost)

- **Codeberg** (`codeberg.org/GBeurier/nirs4all-methods`) — read-only mirror via Codeberg's pull-mirror feature. Insurance against GitHub outages.
- **GitLab.com** (`gitlab.com/GBeurier/nirs4all-methods`) — same.
- **CIRAD GitLab** (`gitlab.cirad.fr/...`) — institutional mirror; required for some funder reporting.

### 1.3 Per-binding namespaces to claim *now* (squatters are the enemy)

Even before publishing v1.0, claim the names — empty placeholder
packages with a "see GitHub" README are fine.

Names marked **(frozen PoC)** map to bindings under `bindings/_archive/`
and are **not** current release targets — reserve them defensively only.

| Registry | Namespace to reserve | Status (2026-05) |
|----------|---------------------|------------------|
| PyPI | `nirs4all-methods` (full binding) + `pls4all` (slim PLS-only subset) | reserve via empty 0.0.0 upload + `Development Status :: 1 - Planning` |
| TestPyPI | `nirs4all-methods` + `pls4all` | already used implicitly by `cibuildwheel` test publish |
| conda-forge feedstock | `nirs4all-methods-feedstock` (created by staged-recipes) | pending v0.98 |
| CRAN | `n4m` (full binding) + `pls4all` (slim PLS-only subset) | submission post-stabilisation; pre-flight via R-universe |
| R-universe | `gbeurier.r-universe.dev/n4m` (+ slim `pls4all`) | pre-CRAN rolling build (free, mandatory before CRAN) |
| crates.io | `n4m` *(official local pre-release)* | publication is intentionally deferred; do not reserve or upload a placeholder |
| npm | `@nirs4all/methods` | scope `@nirs4all` — claim org |
| JSR (`jsr.io`) | `@nirs4all/methods` | mirror of npm — new since 2024, free |
| pkg.go.dev | `github.com/GBeurier/nirs4all-methods/bindings/_archive/go` *(frozen PoC)* | indexed automatically on first `GOPROXY` hit |
| NuGet | `Nirs4allMethods` (managed) + `Nirs4allMethods.Native` (multi-RID native runtime) *(frozen PoC)* | reserve via Microsoft account |
| RubyGems | `nirs4all-methods` *(frozen PoC)* | reserve via `gem push` of 0.0.0 placeholder |
| Julia General | `Nirs4allMethods` (UUID `f04a4a11-…`) *(frozen PoC)* | already in `Project.toml`; not yet registered |
| LuaRocks | `nirs4all-methods` *(frozen PoC)* | rockspec + `luarocks upload` |
| Nimble | `nirs4all-methods` *(frozen PoC)* | PR to `nim-lang/packages` |
| Maven Central | groupId `io.github.gbeurier`, artifactId `nirs4all-methods-jni` (desktop), `nirs4all-methods-android` (AAR) | via Sonatype Central Portal; needs domain ownership proof |
| MATLAB / Octave GitHub Release | `nirs4all-methods-matlab-octave-${VER}.zip` (`+n4m` source package) | automated by `release-matlab.yml`; users run `build_mex.m` locally |
| MATLAB File Exchange | optional `n4m` page on `mathworks.com/matlabcentral/fileexchange` (`+n4m` package) | manual listing; `.mltbx` packaging is not wired in CI |
| Octave Packages | `n4m` on `gnu-octave/packages` index (`+n4m` package) | PR to `gnu-octave/packages/index.yaml` |
| MSYS2 / MinGW-packages | `mingw-w64-nirs4all-methods` PKGBUILD in `msys2/MINGW-packages` | PR review by MSYS2 maintainers |
| MacPorts | `science/nirs4all-methods` Portfile in `macports/macports-ports` | PR review |
| FreeBSD Ports | `math/nirs4all-methods` or `science/nirs4all-methods` | PR to `freebsd/freebsd-ports` |
| GitHub Packages | `maven.pkg.github.com/GBeurier/nirs4all-methods`, `nuget.pkg.github.com/GBeurier`, `npm.pkg.github.com/@nirs4all` | free; auto on GH org; RC channel |
| Hugging Face | `huggingface.co/nirs4all-methods` (org) | only for pretrained PLS models if we ever ship any |
| Docker / GHCR | `ghcr.io/gbeurier/nirs4all-methods` and `docker.io/gbeurier/nirs4all-methods` | claim both |
| Quay.io | `quay.io/gbeurier/nirs4all-methods` | OCI mirror for OpenShift / HPC sites |

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
| Source tarball | `nirs4all-methods-src-${VER}.tar.gz` | `git archive` reproducible, contains `VERSION` file (renamed to avoid collision with Python sdist `nirs4all-methods-${VER}.tar.gz`) |
| Source zip | `nirs4all-methods-src-${VER}.zip` | Windows convenience |
| SBOM (CycloneDX) | `nirs4all-methods-${VER}.cdx.json` | generated by `cyclonedx-conan` / `cyclonedx-cli` for C++ + per-binding SBOMs |
| Build provenance attestation | `nirs4all-methods-${VER}.intoto.jsonl` | `actions/attest-build-provenance@v2` — useful in-toto/Sigstore provenance, **not** automatic SLSA L3 (achieving L3 requires a hermetic builder we do not yet have) |
| SBOM attestation | embedded in the release | `actions/attest-sbom@v2` |
| Sigstore signature | `*.sig`, `*.crt` | keyless OIDC via `actions/attest-build-provenance@v2` (which signs by default) |
| Checksums | `SHA256SUMS`, `SHA256SUMS.asc` | GPG-signed by maintainer key fingerprint `<published-in-SECURITY.md>` |

### 2.2 C/C++ prebuilt binaries (B class)

One archive per **target triple**. Naming follows the Rust-style
triple convention so cross-language tooling can parse it:

```
nirs4all-methods-${VER}-${TRIPLE}.tar.xz       # Linux/macOS
nirs4all-methods-${VER}-${TRIPLE}.zip          # Windows
```

Each archive is `tar -xJf` extractable to a self-contained prefix:

```
nirs4all-methods-${VER}-x86_64-unknown-linux-gnu/
├── bin/
│   └── n4m_cli
├── include/n4m/
│   ├── n4m.h
│   ├── pls.h
│   ├── n4m_export.h
│   └── n4m_version.h
├── lib/
│   ├── libn4m.so → libn4m.so.1             (SONAME symlink)
│   ├── libn4m.so.1 → libn4m.so.1.22.0      (ABI MAJOR symlink)
│   └── libn4m.so.1.22.0                    (real file)
├── lib/cmake/n4m/
│   ├── n4mConfig.cmake
│   └── n4mConfigVersion.cmake
├── lib/pkgconfig/
│   └── n4m.pc
├── share/doc/nirs4all-methods/
│   ├── README.md, LICENSE, CITATION.cff, CHANGELOG.md
│   └── abi/ (expected_symbols_*.txt for the release)
└── share/licenses/nirs4all-methods/LICENSE
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
| `x86_64-pc-windows-msvc` | MSVC 19.4x | Windows 10 | `n4m.dll` + `n4m.lib` + `n4m.pdb` |
| `x86_64-pc-windows-mingw` | MinGW UCRT64 | Windows 10 | for MinGW/MSYS downstream tooling |
| `aarch64-pc-windows-msvc` | MSVC ARM64 | Windows 11 | reserved (Snapdragon X) — produced when CI has the runner |
| `wasm32-unknown-emscripten` | Emscripten 3.1.74 | — | `n4m.wasm` + `n4m.js` glue |
| `wasm32-wasi` | wasi-sdk 22+ | — | server-side WASI runtimes |
| `aarch64-linux-android` | NDK r26+ | API 26 (Android 8.0, matches `CMakePresets.json` `android-*` presets) | Android AAR target |
| `x86_64-linux-android` | NDK r26+ | API 26 | emulator |

Each archive is reproducible (`SOURCE_DATE_EPOCH=$(git log -1 --format=%ct)`)
and verified by re-running the build twice in CI and diff-checking
the `*.so` / `*.dll`. Both BLAS and OpenMP are *optional* runtime
plugins — the canonical Release ships the **dev-release** (no BLAS,
no OpenMP) preset; the **blas-omp** preset is shipped as an
*additional* archive per OS suffixed `-blas-omp` (`nirs4all-methods-${VER}-x86_64-unknown-linux-gnu-blas-omp.tar.xz`).

#### Binary ABI identity gate (release preflight)

**Current state (audited 2026-05-18):** the previous SOVERSION mismatch
has been fixed. Linux builds now produce the ABI-major chain
`libn4m.so -> libn4m.so.1 -> libn4m.so.1.22.0`, derived from
`N4M_ABI_VERSION_MAJOR` and the full ABI version, not from the project
semver. The release preflight is therefore:

1. Build from a clean tree, not from a reused local build directory. Old
   symlinks such as `libn4m.so.0` can remain in dirty build dirs and must
   not be copied into packages.
2. Check that the produced Linux SONAME matches
   `libn4m.so.${N4M_ABI_VERSION_MAJOR}`.
3. Do the equivalent on macOS via `install_name_tool`: the install name is
   `@rpath/libn4m.${ABI_MAJOR}.dylib`.
4. Windows DLLs do not carry a SONAME; the import lib should carry the ABI
   in its filename for clarity (`n4m-1.lib`).
5. Diff the exported `n4m_*` symbol set against
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
| Python (full) | `nirs4all_methods-${VER}-*.whl` (per-platform, manylinux/macos/win), `nirs4all-methods-${VER}.tar.gz` (sdist) |
| Python (slim subset) | `pls4all-${VER}-*.whl`, `pls4all-${VER}.tar.gz` (sdist) |
| R (full) | `n4m_${VER}.tar.gz` (CRAN source), `n4m_${VER}.tgz` (macOS), `n4m_${VER}.zip` (Windows) |
| R (slim subset) | `pls4all_${VER}.tar.gz` / `.tgz` / `.zip` |
| MATLAB / Octave | `nirs4all-methods-matlab-octave-${VER}.zip` (source package with `+n4m`, `build_mex.m`, MEX sources, and tests) |
| JS | `nirs4all-methods-js-${VER}.tgz` (`npm pack` output) |
| Julia *(frozen PoC)* | n/a — registered by Project.toml SHA; tag-only |
| Go *(frozen PoC)* | n/a — go modules resolve from git tag directly |
| Rust *(official pre-release)* | no release asset or registry publication; CI verifies `cargo package --list` for the local `n4m` crate |
| .NET *(frozen PoC)* | `Nirs4allMethods.${VER}.nupkg`, `Nirs4allMethods.Native.${VER}.nupkg` (multi-RID) |
| Ruby *(frozen PoC)* | `nirs4all-methods-${VER}.gem` |
| Lua *(frozen PoC)* | `nirs4all-methods-${VER}-1.rockspec` + `nirs4all-methods-${VER}-1.src.rock` |
| Nim *(frozen PoC)* | n/a — Nimble pulls from git tag |
| JNI / JVM | `nirs4all-methods-jni-${VER}.jar`, `nirs4all-methods-jni-${VER}-sources.jar`, `nirs4all-methods-jni-${VER}-javadoc.jar`, all `.asc` signed |
| Android | `nirs4all-methods-android-${VER}.aar` |

---

## 3. Per-binding publication channels

Each subsection follows the same shape: **registry → identity →
required assets → prereqs → publish step → smoke test → caveats**.

### 3.1 Python — `pls4all` on PyPI (+ conda-forge mirror)

| Field | Value |
|-------|-------|
| Registry | https://pypi.org/project/pls4all/ |
| Identity | name: `pls4all` (slim PLS-only subset; the full binding is `nirs4all-methods`), distribution: `wheel` + `sdist` |
| Tier 1 | ctypes wrapper around `libn4m` (currently shipped) |
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
either bundles `libn4m` (platform wheel) or builds it from source
(sdist). Wheels do not have a reliable post-install hook, and
import-time binary downloads are hostile to reproducibility and
offline installs.

Wheels **bundle `libn4m` inside the wheel** via `auditwheel repair`
(Linux) / `delocate-wheel` (macOS) / `delvewheel` (Windows). The
shared library is renamed and placed under `pls4all.libs/` (Linux
convention) with RPATH `$ORIGIN/../pls4all.libs` patched into the
loader stub. To convince `cibuildwheel` to repair a ctypes-only
wheel (which it normally treats as pure-Python and skips), we use
`CIBW_REPAIR_WHEEL_COMMAND_*` explicitly *and* force a platform tag
via a `setup.py` shim that calls
`from setuptools import Distribution; Distribution(...).has_ext_modules = lambda: True`.
This keeps installs pip-only (no `apt install libn4m-dev`).

> **License note** (re: cibuildwheel warning): bundling libn4m is
> permitted under the repository's dual `CECILL-2.1 OR AGPL-3.0-or-later`
> policy. Distribution assets must preserve the applicable license notices;
> the root `LICENSE` alone is the AGPL text, not a substitute for CeCILL.

**Prereqs:**
1. PyPI API token in GitHub Actions secret `PYPI_API_TOKEN` (per-project token, scope = `pls4all` only).
2. TestPyPI token in `TESTPYPI_API_TOKEN` for pre-release dry runs.
3. Trusted publishing (OIDC) configured for `GBeurier/nirs4all-methods` → PyPI (preferred over long-lived tokens since 2024).
4. `pyproject.toml` updated to use `setuptools-scm` so `pip install` from the sdist deduces the version from the git tag.

**Publish step (CI):**
```yaml
# .github/workflows/release.yml — python job
- uses: pypa/cibuildwheel@v3
  env:
    # Build once per platform/arch — the ctypes binding is interpreter-agnostic.
    CIBW_BUILD: "cp312-*"            # any modern CPython picks up the platform wheel
    CIBW_SKIP: "*-musllinux_i686 *-win32"
    CIBW_BEFORE_ALL_LINUX: bash bindings/python/scripts/build_libn4m_in_wheel.sh
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
recipe `meta.yaml` declares `host: libn4m` + `run: libn4m`: the
conda Python package depends on the conda `libn4m` package. PyPI
wheels do **not** depend on conda packages — they bundle their own
copy of `libn4m` and are independently usable.

Two feedstocks, in order:
1. `libn4m-feedstock` (the C/C++ library — no Python) → migrators handle BLAS variants.
2. `pls4all-feedstock` (the Python package) → `run: libn4m {{ pin_compatible('libn4m') }}`.

### 3.2 R — `pls4all` on CRAN (+ R-universe rolling)

| Field | Value |
|-------|-------|
| Registry | https://cran.r-project.org/package=pls4all |
| Identity | Package: `pls4all`, License: `CeCILL (== 2.1)` (R license database form for CeCILL-2.1) |
| Tier 1 | `.Call` gateway over the C ABI (no Rcpp) |
| Tier 2 | base-R formula+S3 · `pls`-compatible formula API · `mdatools`-compatible matrix API. parsnip/mlr3 adapters are archived under `bindings/r/archive/parsnip-mlr3/` for later revival. |

**Required assets:**
- `pls4all_${VER}.tar.gz` (CRAN source) — `R CMD build bindings/r/pls4all`.
- `pls4all_${VER}.tgz` (binary for macOS, built per R minor: 4.4, 4.5).
- `pls4all_${VER}.zip` (binary for Windows, built per R minor).

CRAN itself rebuilds binaries automatically from the submitted
source — but R-universe lets us **publish those binaries before
CRAN does**, which is the practical mainstream.

**The `libn4m` dilemma.** CRAN source packages cannot contain
prebuilt binary libraries, but they can include C/C++ source and
build it during package installation. Two viable strategies:

1. **System dependency**: declare `SystemRequirements: libn4m (>=
   1.10)` and rely on the user (or the conda-forge R installation,
   or `r-spelling`-style autobrew) to provide `libn4m`. This is
   what the current `DESCRIPTION` does. CRAN policy permits this,
   but only realistic if `libn4m` is widely available on CRAN's
   build machines (currently it is not).
2. **Vendored source + static build**: the `src/` of the R package
   contains the `cpp/` tree as a git subtree and a `Makevars` that
   builds the static `libn4m.a` at install time, then links the R
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

### 3.3 MATLAB / Octave — `+n4m` source package on GitHub Releases

| Field | Value |
|-------|-------|
| Registry | GitHub Releases (automated); MATLAB File Exchange / Octave Packages optional manual mirrors |
| Identity | `+n4m`, artifact `nirs4all-methods-matlab-octave-${VER}.zip` |
| Tier 1 | single MEX dispatcher (`n4m_method_fit_mex`) — live |
| Tier 2 | 18 classdefs + factory — live |

**Required assets:**
- `nirs4all-methods-matlab-octave-${VER}.zip` containing the `bindings/matlab/`
  source tree under a versioned top-level directory.
- The zip includes `+n4m`, `build_mex.m`, MEX C++ sources, docs, and tests.
- Users build the MEX files locally against the released or local `libn4m`.
  `build_mex.m` honors `N4M_INCLUDE_DIR`, `N4M_GENERATED_DIR`, and `N4M_LIB_DIR`.

**Prereqs:**
1. `release-matlab.yml` enabled on tags.
2. `bindings/matlab/build_mex.m` remains compatible with MATLAB and Octave.
3. Optional mirrors: MathWorks account for File Exchange, and/or Octave Packages
   index PR for a source package listing.

**Publish step (automated for GitHub Release):**
- `release-matlab.yml` reads `N4M_PROJECT_VERSION_STRING` from
  `cpp/include/n4m/n4m_version.h`.
- It archives `HEAD:bindings/matlab` as
  `nirs4all-methods-matlab-octave-${VER}.zip`.
- On non-RC release tags, the zip is attached to the matching GitHub Release.
- `.mltbx`, MATLAB File Exchange, and Octave Packages are manual optional
  mirrors until a licensed MATLAB packaging workflow is intentionally added.

**Smoke test after local build:**
```matlab
cd bindings/matlab
build_mex
assert(strncmp(n4m.version(), "${VER}", strlength("${VER}")));
X = randn(50, 5);
y = X(:, 1) - 0.25 * X(:, 3) + 0.01 * randn(50, 1);
mdl = n4m.fit("simpls", X, y, "NumComponents", 3);
yhat = predict(mdl, X);
assert(size(yhat, 1) == 50);
```

> **Octave mirror.** Octave Packages can list the same source distribution via
> the `index.yaml` PR pattern. Naming is `n4m`; there is no Octave-specific
> legacy package name.

### 3.4 JavaScript / TypeScript — `@nirs4all/methods` on npm + JSR

| Field | Value |
|-------|-------|
| Registry | https://www.npmjs.com/org/nirs4all + https://jsr.io/@nirs4all |
| Identity | `@nirs4all/methods` (current), planned: `@nirs4all/methods-node`, `@nirs4all/methods-types` |
| Tier 1 | `n4m.wasm` + thin TS API — live |
| Tier 2 | `@nirs4all/methods-sklearn-like` (idiomatic class) — live |

**Required assets:**
- `@nirs4all/methods` (browser + Deno + Bun + Node universal ESM):
  - `dist/index.js` (ESM)
  - `dist/index.d.ts`
  - `dist/n4m.js` (Emscripten JS glue)
  - `dist/n4m.wasm` (single binary, ~1.5 MB raw / ~497 KB gzip)
  - `dist/n4m.wasm.map` (source map, optional)
- `@nirs4all/methods-node` (Node-only with native bindings via `node-gyp` or
  `napi-rs`, for users who don't want WASM): includes prebuilt
  `n4m.node` binaries via `prebuildify` for:
  | Triple | Node ABI | OS |
  |---|---|---|
  | `linux-x64`, `linux-arm64`, `linux-x64-musl`, `linux-arm64-musl` | napi-9 (Node 20+) | Linux |
  | `darwin-x64`, `darwin-arm64` | napi-9 | macOS |
  | `win32-x64`, `win32-arm64` | napi-9 | Windows |
- `@nirs4all/methods-types` (pure `.d.ts`, for projects mixing both): one
  package keeps the type surface single-source.

The `npm pack` output for the wasm package is
`nirs4all-methods-js-${VER}.tgz`.

**Prereqs:**
1. npm org `@nirs4all` claimed; CI scope tokens stored as `NPM_TOKEN`.
2. JSR org `@nirs4all` claimed; OIDC trust to GitHub Actions.
3. Emscripten 3.1.74 available in the JS build job (currently in
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
deno run --allow-all -r https://jsr.io/@nirs4all/methods/${VER}/smoke.ts
node -e "import('@nirs4all/methods').then(m => console.log(m.version()))"
```

### 3.5 Go — `github.com/GBeurier/nirs4all-methods/bindings/_archive/go` on pkg.go.dev

> **FROZEN PoC.** The Go binding lives under `bindings/_archive/` and is
> **not a current release target.** The plan below is retained for the
> day it is unfrozen; the channel is not live.

Go modules are special — there is no central upload, only the git
tag + module path.

| Field | Value |
|-------|-------|
| Registry | https://pkg.go.dev/github.com/GBeurier/nirs4all-methods/bindings/_archive/go/n4m |
| Identity | module path `github.com/GBeurier/nirs4all-methods/bindings/_archive/go` |
| Tier | cgo wrapper (frozen) |

**Required assets:** none in the GitHub Release. Go fetches from the
git tag directly. The constraint is that the `go.mod` module path
must include the binding subdirectory: `github.com/GBeurier/nirs4all-methods/bindings/_archive/go`.

**Prereqs:**
1. The git tag must follow the **submodule pattern** for sub-tree
   modules: `bindings/_archive/go/v0.98.0` (not just `v0.98.0`). pkg.go.dev
   indexes that automatically.
2. `bindings/_archive/go/go.mod` declares `go 1.22`.
3. cgo is enabled by default; users must have `libn4m` installed
   on their `LD_LIBRARY_PATH`. Document the Homebrew tap (§4.1) and
   Debian package (§4.4) as the canonical install paths.

**Publish step:**
```bash
git tag bindings/_archive/go/v${VER}
git push origin bindings/_archive/go/v${VER}
# Force a fetch to populate pkg.go.dev
GOPROXY=proxy.golang.org go list -m github.com/GBeurier/nirs4all-methods/bindings/_archive/go@v${VER}
```

**Smoke test:**
```bash
mkdir /tmp/n4m-smoke && cd /tmp/n4m-smoke
go mod init smoke
go get github.com/GBeurier/nirs4all-methods/bindings/_archive/go@v${VER}
cat > main.go <<'EOF'
package main

import (
    "fmt"
    "github.com/GBeurier/nirs4all-methods/bindings/_archive/go/n4m"
)

func main() { fmt.Println(n4m.Version()) }
EOF
LD_LIBRARY_PATH=/usr/local/lib go run .
```

### 3.6 Rust — official pre-release `n4m` crate

> **Official pre-release; not a registry release.** The current binding is
> `bindings/rust/n4m`, not the archived Rust PoC. It is qualified for real
> C-ABI model fit/predict, N4MM/N4MOPT serialization, native optimizer HPO,
> and `n4m_finetune_estimator` selection-only HPO. The latter deliberately
> does not refit a final model. No cross-language Rust prediction fixture is
> committed, so this binding does **not** claim numerical parity with the
> cross-binding matrix.

| Field | Value |
|-------|-------|
| Registry | none yet — crates.io publication is intentionally deferred |
| Identity | workspace crate `n4m` (`bindings/rust/n4m`), `rlib` |
| Native dependency | prebuilt dynamically linked `libn4m`; `N4M_LIB_DIR` is required |
| Qualification | Linux/macOS/Windows binding gates; Linux ASan/UBSan/combined sanitizer runtime gate |

The manifest declares `license = "CECILL-2.1 OR AGPL-3.0-or-later"` and the
crate packages both complete texts under `LICENSES/`. This avoids presenting
the repository-root `LICENSE` (which is the AGPL text) as though it were the
CeCILL text.
The release check is:

```bash
cargo package -p n4m --list --allow-dirty
```

Its package list contains `Cargo.lock`, `Cargo.toml`, `README.md`, both
`LICENSES/*.txt` texts, `abi_probe.c`, `build.rs`,
`scripts/run_sanitized_tests.sh`, and `src/lib.rs`, plus Cargo-generated
`.cargo_vcs_info.json` and `Cargo.toml.orig`. It does not bundle `libn4m`.

Build the native library before testing the crate:

```bash
cmake --preset dev-debug && cmake --build --preset dev-debug --parallel
N4M_LIB_DIR="$PWD/build/dev-debug/cpp/src" \
N4M_RUNTIME_RPATH="$PWD/build/dev-debug/cpp/src" \
cargo test -p n4m
```

### 3.7 .NET — `Nirs4allMethods` + native runtime packages on NuGet

> **FROZEN PoC.** The .NET binding lives under `bindings/_archive/` and is
> **not a current release target.** The plan below is retained for the
> day it is unfrozen; the channel is not live.

The .NET ecosystem distinguishes the **managed assembly** (your C#)
from the **native dependency** (`libn4m`). The .NET-standard pattern
is to ship them as **two NuGet packages**, with all native runtimes
bundled in one multi-RID package:

| Package | Type | Content |
|---------|------|---------|
| `Nirs4allMethods` | managed | `lib/net9.0/Nirs4allMethods.dll` (any-CPU), targets `net9.0` (matches current `Nirs4allMethods.csproj`); depends on `Nirs4allMethods.Native` |
| `Nirs4allMethods.Native` | native multi-RID | `runtimes/linux-x64/native/libn4m.so`, `runtimes/linux-arm64/native/libn4m.so`, `runtimes/osx-x64/native/libn4m.dylib`, `runtimes/osx-arm64/native/libn4m.dylib`, `runtimes/win-x64/native/n4m.dll`, `runtimes/win-arm64/native/n4m.dll` |

The native filenames are the **unversioned loadable names** that
match the `[DllImport("n4m")]` attribute used by `Nirs4allMethods.cs`. The
ABI-versioned filenames (`libn4m.so.1.22.0`) ship inside the C++
archive (§2.2) but **not** inside the NuGet native package, because
the .NET SDK/runtime selects native assets by RID-directory matching,
not by SONAME.

A consumer just adds `Nirs4allMethods` and `Nirs4allMethods.Native` is pulled in as
a dependency. The .NET SDK/runtime selects the matching files under
`runtimes/{rid}/native/` at build/publish/runtime — no `Nirs4allMethods.Native.all`
meta-package needed.

**Optional split.** If the multi-RID package grows too large
(>50 MB), split into per-RID packages following the
`Microsoft.Data.SqlClient.SNI.runtime` pattern: `Nirs4allMethods.Native.linux-x64`,
`Nirs4allMethods.Native.linux-arm64`, etc., with a meta package `Nirs4allMethods.Native`
that takes a hard dependency on all of them. Defer this until the
size actually becomes a problem; today libn4m is ~2 MB per RID, so
the single package totals ~12 MB which is fine.

**Prereqs:**
1. NuGet API key in `NUGET_API_KEY` secret (or trusted publishing
   via NuGet's GitHub OIDC integration).
2. `Nirs4allMethods.csproj` has `<GeneratePackageOnBuild>true</GeneratePackageOnBuild>`
   and `<PackageLicenseExpression>CECILL-2.1 OR AGPL-3.0-or-later</PackageLicenseExpression>`
   (do **not** use the deprecated `licenseUrl`).
3. `Nirs4allMethods.Native.csproj` is a `Microsoft.NET.Sdk` project with
   `<IncludeBuildOutput>false</IncludeBuildOutput>` and a `<Content>`
   manifest that maps each prebuilt `libn4m` to `runtimes/${rid}/native/`.

**Publish step:**
```bash
dotnet pack bindings/_archive/dotnet/Nirs4allMethods -c Release -o out/
dotnet pack bindings/_archive/dotnet/Nirs4allMethods.Native -c Release -o out/
dotnet nuget push out/*.nupkg --source https://api.nuget.org/v3/index.json \
  --api-key $NUGET_API_KEY --skip-duplicate
```

**Smoke test:**
```bash
dotnet new console -o /tmp/n4m-smoke && cd /tmp/n4m-smoke
dotnet add package Nirs4allMethods --version ${VER}
# Nirs4allMethods.Native is pulled in transitively.
dotnet run
```

### 3.8 Julia — `Nirs4allMethods.jl` on the General registry (+ `Nirs4allMethods_jll`)

> **FROZEN PoC.** The Julia binding lives under `bindings/_archive/` and is
> **not a current release target.** The plan below is retained for the
> day it is unfrozen; the channel is not live.

Julia separates the binding from the binary even more strictly than
.NET, via the **JLL wrapper** pattern.

| Package | Role |
|---------|------|
| `Nirs4allMethods_jll` | auto-generated by BinaryBuilder.jl, ships `libn4m` for every platform Julia supports |
| `Nirs4allMethods.jl` | user-facing API; depends on `Nirs4allMethods_jll` for the binary |

Two different registration flows:

- `Nirs4allMethods.jl` is registered to **General** via JuliaRegistrator.jl
  (a GitHub App that opens a PR against the General registry on a
  `@JuliaRegistrator register` comment).
- `Nirs4allMethods_jll` is generated by **BinaryBuilder** when a recipe
  PR (`N/Nirs4allMethods/build_tarballs.jl`) is merged into Yggdrasil; the
  Yggdrasil pipeline then opens its own PR against General. We do
  not call Registrator for the JLL.

**Required assets:** none in our Release. Everything is built from
the git tag (Nirs4allMethods.jl) and from the source tarball (Nirs4allMethods_jll
via Yggdrasil's `build_tarballs.jl`).

**Prereqs:**
1. UUID locked in `bindings/_archive/julia/Nirs4allMethods.jl/Project.toml` (already done).
2. PR to Yggdrasil with `N/Nirs4allMethods/build_tarballs.jl`. The platforms
   actually produced are those declared in the recipe (typically
   `expand_cxxstring_abis(supported_platforms())`); not "every
   platform Julia supports" unless we explicitly target all of them.
3. JuliaRegistrator bot installed on `GBeurier/nirs4all-methods` *or* manual
   web UI at `juliahub.com → Register packages`.

**Publish step:**
```
# As an issue comment on the right commit:
@JuliaRegistrator register subdir=bindings/_archive/julia/Nirs4allMethods.jl
```

**Smoke test:**
```julia
using Pkg; Pkg.add(name="Nirs4allMethods", version="${VER}")
using Nirs4allMethods
@assert startswith(Nirs4allMethods.version(), "${VER}")
```

### 3.9 Ruby — `nirs4all-methods` gem on RubyGems

> **FROZEN PoC.** The Ruby binding lives under `bindings/_archive/` and is
> **not a current release target.** The plan below is retained for the
> day it is unfrozen; the channel is not live.

| Field | Value |
|-------|-------|
| Registry | https://rubygems.org/gems/nirs4all-methods |
| Identity | `nirs4all-methods`, Fiddle-based (stdlib FFI, no `mkmf` C extension) |

**Required assets:**
- `nirs4all-methods-${VER}.gem` (output of `gem build`). Pure-Ruby gem — no
  native code in the gem itself, relies on system-installed `libn4m`.
- Optional: a **fat gem** per platform that ships `libn4m` inside
  (Ruby's `Gem::Platform.local` convention, e.g. `nirs4all-methods-${VER}-x86_64-linux.gem`).
  Add this when v1.0 stabilises; it's how `nokogiri` and `grpc` do it.

**Missing artifact today:** `bindings/_archive/ruby/nirs4all-methods.gemspec`. Must be
created (see [§6 New-channel checklist](#6-new-channel-checklist)).

**Prereqs:**
1. RubyGems API key in `RUBYGEMS_API_KEY` secret.
2. `bindings/_archive/ruby/nirs4all-methods.gemspec` with name, version (auto from
   `lib/nirs4all_methods/version.rb`), authors, license = `CECILL-2.1`.

**Publish step:**
```bash
cd bindings/_archive/ruby
gem build nirs4all-methods.gemspec
gem push nirs4all-methods-${VER}.gem
```

### 3.10 Lua — `nirs4all-methods` on LuaRocks

> **FROZEN PoC.** The Lua binding lives under `bindings/_archive/` and is
> **not a current release target.** The plan below is retained for the
> day it is unfrozen; the channel is not live.

| Field | Value |
|-------|-------|
| Registry | https://luarocks.org/modules/gbeurier/nirs4all-methods |
| Identity | `nirs4all-methods` (rockspec format) |
| Targets | **LuaJIT 2.1 only** (standard Lua 5.1+ has no built-in FFI; the binding uses LuaJIT's `ffi.*` API directly) |

**Required assets:**
- `nirs4all-methods-${VER}-1.rockspec`
- `nirs4all-methods-${VER}-1.src.rock` (source rock)

**Missing artifact today:** the rockspec. Must be created.

**Prereqs:**
1. LuaRocks account; API key in `LUAROCKS_API_KEY` secret.
2. `nirs4all-methods-${VER}-1.rockspec` with `external_dependencies = { LIBN4M = { library = "n4m" } }`.

**Publish step:**
```bash
cd bindings/_archive/lua
luarocks pack nirs4all-methods-${VER}-1.rockspec
luarocks upload nirs4all-methods-${VER}-1.rockspec --api-key=$LUAROCKS_API_KEY
```

### 3.11 Nim — `nirs4all-methods` in Nimble packages

> **FROZEN PoC.** The Nim binding lives under `bindings/_archive/` and is
> **not a current release target.** The plan below is retained for the
> day it is unfrozen; the channel is not live.

| Field | Value |
|-------|-------|
| Registry | https://nimble.directory/pkg/nirs4all-methods (mirror of `nim-lang/packages_official`) |
| Identity | `nirs4all-methods` |

**Required assets:**
- `bindings/_archive/nim/nirs4all-methods.nimble` — package metadata.
- Nim packages are git-fetched at install; no upload needed.

**Missing artifact today:** the `.nimble` file. Must be created.

**Prereqs:**
1. PR to [`nim-lang/packages`](https://github.com/nim-lang/packages)
   adding a JSON entry pointing at this repo's `bindings/_archive/nim` subdir.

**Publish step:**
```bash
# One-time
gh repo clone nim-lang/packages /tmp/nim-packages
# Add an entry, open a PR; nim merges within ~days.
```

### 3.12 JVM / Android — Maven Central (`io.github.gbeurier:nirs4all-methods-{jni,android}`)

Two artifacts, two flavours:

| GroupId : artifactId : packaging | Target |
|--------|--------|
| `io.github.gbeurier:nirs4all-methods-jni:${VER}` (jar) | Desktop JVM — bundles `libn4m.so` + `libn4m_jni.so` per OS in `META-INF/native/<os>/<arch>/` |
| `io.github.gbeurier:nirs4all-methods-android:${VER}` (aar) | Android — includes `jniLibs/{arm64-v8a,x86_64}/libn4m.so` + Kotlin wrapper |
| `io.github.gbeurier:nirs4all-methods-jni:${VER}:sources` (jar) | source jar (required for Maven Central) |
| `io.github.gbeurier:nirs4all-methods-jni:${VER}:javadoc` (jar) | javadoc jar (required for Maven Central) |
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
3. Android NDK r26d+ available in the CI runner for `nirs4all-methods-android`
   (the current Phase 36 deferral resolves when this is wired).
4. AAR built via the planned `bindings/android/nirs4all-methods-android/build.gradle.kts`.
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
val v = io.nirs4all.Nirs4allMethods.version()
assertTrue(v.startsWith("${VER}"))
```

### 3.13 WebAssembly / browser — npm + JSR + CDN

In addition to the npm publication in §3.4, we expose the WASM blob
to CDNs for zero-tooling browser use:

- `https://cdn.jsdelivr.net/npm/@nirs4all/methods@${VER}/dist/n4m.wasm` (auto on npm publish)
- `https://unpkg.com/@nirs4all/methods@${VER}/dist/n4m.wasm` (auto on npm publish)
- `https://esm.sh/@nirs4all/methods@${VER}` (Deno-friendly imports)

Nothing to do beyond §3.4 — CDNs mirror npm automatically.

### 3.14 Documentation — GitHub Pages + Read the Docs

The Sphinx site in `docs/` is published in parallel to **two** independent surfaces. Both build the same Sphinx config (`docs/conf.py`) from the same git history; they differ in trigger, hosting, and benchmark-data source.

| Target | URL | Triggered by | Benchmark data source |
|---|---|---|---|
| **GitHub Pages** | `https://methods.nirs4all.org/` | `.github/workflows/docs.yml` on push to `main` | live `benchmarks/cross_binding/results/full_matrix.csv` if present, else `docs/_static/bench-data.json` |
| **Read the Docs** | `https://nirs4all-methods.readthedocs.io/` | RTD webhook on every push (all branches + tags) | committed `docs/_static/bench-data.json` only — RTD's build env doesn't carry the full benchmark cache |

**Why both?**
- **GitHub Pages** is the canonical site (rolls forward with `main`, gets fresh benchmark data from CI).
- **Read the Docs** adds (a) versioned snapshots (`/en/v0.98.0/`, `/en/v1.0.0/`, …), (b) PDF + ePub builds (the `.readthedocs.yaml` declares `formats: [htmlzip, pdf]`), (c) a PR-preview build per pull-request — useful when reviewing doc changes before merging.

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
| RTD.2 | 👤 | **Import a Project** → pick `GBeurier/nirs4all-methods`. RTD reads `.readthedocs.yaml` automatically. |
| RTD.3 | 👤 | In **Admin → Advanced settings** set: *Documentation type*=Sphinx HTML, *Privacy level*=Public, *Default branch*=`main`. |
| RTD.4 | 👤 | (Optional) **Admin → Domains** → add CNAME `docs.nirs4all-methods.io` (or skip and live with `nirs4all-methods.readthedocs.io`). |
| RTD.5 | 🤖 | Verify the first build at `https://readthedocs.org/projects/nirs4all-methods/builds/`. If it's red, I'll patch `docs/requirements.txt` / `docs/conf.py` based on the RTD log. |

**Recurring cost**: zero. Both Pages and RTD build automatically on push. No secrets to manage. The two sites can drift (Pages has fresh benchmark data, RTD doesn't) but `docs/conf.py:setup()` auto-falls-back to the committed JSON so the RTD build never errors on missing benchmark inputs.

**Version policy**:
- Pages: always reflects `main` (one rolling site).
- RTD: each git tag becomes a versioned snapshot at `/en/${TAG}/`. The "stable" alias points at the most recent non-`-rc` tag, "latest" follows `main`. RTD picks these up automatically from the tags pushed by M0/M1.

---

## 4. C/C++ ecosystem channels (where the .so/.dll lives)

These channels target users who don't go through a language binding
but install the C/C++ library directly.

### 4.1 Homebrew tap — `gbeurier/nirs4all-methods`

| Field | Value |
|-------|-------|
| Tap repo | `GBeurier/homebrew-nirs4all-methods` |
| Formula | `Formula/nirs4all-methods.rb` |
| Bottles | `gbeurier-nirs4all-methods-${VER}.{x86_64,arm64}_{ventura,sonoma,sequoia}.bottle.tar.gz` |
| Bottle host | GitHub Releases of the **tap repo** (not this repo — Homebrew convention) |

User installs:
```bash
brew tap gbeurier/nirs4all-methods
brew install nirs4all-methods
```

**Prereqs:**
1. A second repo `homebrew-nirs4all-methods`.
2. A `brew tap-new` skeleton with the bottle-build workflow.

We do **not** push to homebrew-core. The barrier (popularity check,
review) is too high for a research-grade chemometrics library at
v1.0. Re-evaluate at v2.0.

### 4.2 vcpkg port — `nirs4all-methods` in microsoft/vcpkg

| Field | Value |
|-------|-------|
| Registry | https://github.com/microsoft/vcpkg/tree/master/ports/nirs4all-methods |
| Identity | `nirs4all-methods` port + version DB entry |
| Maintainer | `GBeurier` |

**Required assets:**
- `ports/nirs4all-methods/portfile.cmake` (download + CMake invocation)
- `ports/nirs4all-methods/vcpkg.json` (manifest)
- `versions/n-/nirs4all-methods.json` (version DB entry)
- `versions/baseline.json` update

**Publish step:** PR against `microsoft/vcpkg` adding the four files
above. The vcpkg curated registry's PR review enforces:
- portfile pinned to a release tag with sha512.
- usage file `usage` printed after install.
- license expression matches `CECILL-2.1 OR AGPL-3.0-or-later`.

### 4.3 Conan recipe — Conan Center Index (CCI)

| Field | Value |
|-------|-------|
| Registry | https://conan.io/center/recipes/nirs4all-methods |
| Identity | `nirs4all-methods/${VER}@` (no user/channel — CCI convention) |

**Required assets:**
- `recipes/nirs4all-methods/all/conanfile.py`
- `recipes/nirs4all-methods/all/conandata.yml` (with sha256 pinned to GitHub Release tarball)
- `recipes/nirs4all-methods/config.yml`

**Publish step:** PR against
`conan-io/conan-center-index`. CCI's CI builds across the full matrix
(Linux gcc/clang, macOS, Windows MSVC/MinGW, x86_64 + arm64, multiple
shared/static/with_blas/with_openmp/with_cuda permutations) before merging.

### 4.4 Linux distro packages (long tail, low priority)

| Distro | Channel | Format |
|--------|---------|--------|
| Debian / Ubuntu | private PPA `ppa:gbeurier/nirs4all-methods` → eventually debian official | `.deb` |
| Arch | `aur.archlinux.org/packages/nirs4all-methods` (community-driven) | PKGBUILD |
| Fedora / RHEL | COPR `@gbeurier/nirs4all-methods` → eventually Fedora official | `.rpm` |
| openSUSE | OBS `home:gbeurier:nirs4all-methods` | `.rpm`/`.deb` |
| Nix | `nixpkgs/pkgs/by-name/ni/nirs4all-methods/package.nix` | derivation |
| Spack | `var/spack/repos/builtin/packages/nirs4all-methods/package.py` (used by HPC sites) | spack recipe |
| Guix | `gnu/packages/chemometrics.scm` | guile recipe |

These are *low priority* (most users install via the language
binding, which already pulls libn4m transitively via conda-forge or
the bundled wheel). We support them only when a downstream maintainer
asks. Each follows the upstream contribution policy of the distro.

### 4.5 Docker / OCI images on GHCR + Docker Hub

| Image | Tags | Content |
|-------|------|---------|
| `ghcr.io/gbeurier/nirs4all-methods:${VER}` | `latest`, `${VER}`, `${VER}-cuda` | `n4m_cli` + `libn4m` runtime |
| `ghcr.io/gbeurier/nirs4all-methods:${VER}-dev` | same | adds headers + `cmake`/`gcc` for FROM-as |
| `ghcr.io/gbeurier/nirs4all-methods-runner:${VER}` | same | base for benchmark runs; preloads Python+R+MATLAB-mcr |
| `docker.io/gbeurier/nirs4all-methods:${VER}` | mirror | optional, helps users behind GHCR-blocked firewalls |

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
  TAG[Tag v0.98.0] --> SRC[1. Source bundle + SBOM + provenance]
  TAG --> BLD[2. Multi-OS build matrix\n→ Triple archives]
  SRC --> REL[3. Create draft GH Release\nupload SRC + binaries]
  BLD --> REL
  REL --> PY[4a. cibuildwheel → PyPI + TestPyPI]
  REL --> RP[4b. R-universe push + CRAN tarball attached]
  REL --> ML[4c. MATLAB/Octave source zip]
  REL --> NPM[4d. npm publish + JSR mirror]
  REL --> NUG[4e. dotnet pack + nuget push]
  REL --> JL[4f. JuliaRegistrator comment]
  REL --> GE[4g. gem push (Ruby)]
  REL --> LR[4h. luarocks upload]
  REL --> NB[4i. Nimble repo PR / autoupdate]
  REL --> MV[4j. gradle publishToCentralPortal]
  REL --> DK[4k. docker buildx push (multi-arch)]
  REL --> HB[4l. homebrew tap formula bump PR]
  REL --> VC[4m. vcpkg port version bump PR (manual)]
  REL --> CC[4n. conan-center-index PR (manual)]
  PY --> SMOKE[5. Post-publish smoke test matrix]
  RP --> SMOKE
  ML --> SMOKE
  NPM --> SMOKE
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

The single canonical version is `cpp/include/n4m/n4m_version.h`.
A pre-tag script (`scripts/bump_version.sh ${NEW_VER}`) regenerates
every downstream manifest (the `bindings/_archive/*` rows are the frozen
PoC bindings — kept for completeness, not part of the active release set):

| File | Generator clause |
|------|------------------|
| `CMakeLists.txt` `project(... VERSION ${X})` | sed by `bump_version.sh` |
| `bindings/python/pyproject.toml` | sed |
| `bindings/r/n4m/DESCRIPTION` | sed |
| `bindings/_archive/julia/Nirs4allMethods.jl/Project.toml` | sed |
| `bindings/rust/n4m/Cargo.toml` | pre-release crate version is maintained independently until a registry-release policy is approved |
| `bindings/js/package.json` | `npm version ${X} --no-git-tag-version` |
| `bindings/_archive/dotnet/Nirs4allMethods/Nirs4allMethods.csproj` `<Version>` | sed |
| `bindings/_archive/ruby/lib/nirs4all_methods/version.rb` | sed |
| `bindings/_archive/lua/nirs4all-methods-${VER}-1.rockspec` | filename rename + content sed |
| `bindings/_archive/nim/nirs4all-methods.nimble` | sed |
| `bindings/android/nirs4all-methods-android/build.gradle.kts` `version =` | sed |
| `CITATION.cff` `version:` | sed |
| `docs/abi/changelog.md` | hand-edited per phase |

The CI check `.github/workflows/version-sync.yml` and
`scripts/bump_version.sh` are now present. Before any registry
publication, run:

```bash
scripts/bump_version.sh --check
```

and make the workflow fail PRs when any manifest drifts from
`n4m_version.h`. If another manifest family is added, extend the script
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
| `release-matlab.yml` | deterministic `bindings/matlab/` zip for MATLAB/Octave | tag push / workflow dispatch |
| `release-js.yml` | npm + JSR + CDN | called from `release.yml` |
| `rust-binding.yml` | build, MSRV, ABI probe, package-list, and sanitizer qualification for the pre-release crate | push / pull request |
| `release-dotnet.yml` | `dotnet pack` + NuGet push (managed + 6 native RIDs) | called from `release.yml` |
| `release-julia.yml` | nothing — Yggdrasil and Registrator are external; this workflow only attaches Project.toml diff to the release notes for transparency | called from `release.yml` |
| `release-ruby.yml` | `gem build` + `gem push` | called from `release.yml` |
| `release-lua.yml` | `luarocks upload` | called from `release.yml` |
| `release-nim.yml` | PR to `nim-lang/packages` | called from `release.yml` |
| `release-maven.yml` | gradle publishToCentralPortal | called from `release.yml` |
| `release-docker.yml` | `docker buildx` multi-arch | called from `release.yml` |
| `release-homebrew.yml` | PR to `GBeurier/homebrew-nirs4all-methods` | called from `release.yml` |
| `prerelease-nightly.yml` | every-day cross-binding parity gate on `main` against an unbumped `v0.98.0-dev.${SHORTSHA}` | `schedule: cron 'nightly'` |
| `smoke-installed.yml` | weekly pulls the last released `nirs4all-methods` from every registry and reruns the parity fixture — catches registry drift / outages | `schedule: cron 'weekly'` |

---

## 6. New-channel checklist

When adding a new binding or registry, follow this 10-step gate. The
checklist is reproduced inside each `bindings/${LANG}/CHECKLIST.md`.

1. **License**: confirm SPDX `CECILL-2.1 OR AGPL-3.0-or-later` (or a
   registry-specific representation that preserves the recipient's choice).
   If not, halt.
2. **Identity**: claim the name on the registry (empty placeholder OK).
3. **Manifest**: add the language's package manifest (`pyproject.toml`, `gemspec`, `nimble`, `rockspec`, etc.) with a `version` field tied to the canonical version source.
4. **Source layout**: respect `bindings/${LANG}/` with the README, the manifest, `src/`, and `tests/`.
5. **Parity gate**: implement the SIMPLS parity test against `bindings/_catalog/parity_fixture.json` so the binding fails CI if any numerical drift exceeds `rmse_rel < 1e-12`.
6. **Loader rules**: documented prioritised path discovery for `libn4m` — `${LIB}_LIB_PATH` env var first, then bundled, then system.
7. **Smoke test**: in CI on every PR.
8. **Release wiring**: a `release-${LANG}.yml` workflow callable from `release.yml`.
9. **Docs**: a one-page `docs/bindings/${LANG}.md` linked from the README binding-matrix table.
10. **README badge**: add the registry's version badge to the project README's matrix.

The checklist is enforced by `scripts/check_binding_complete.sh
bindings/${LANG}/`, which the `version-sync.yml` workflow runs.

---

## 7. Proactive additions worth considering

Beyond what the user explicitly mentioned, these channels would
extend nirs4all-methods' reach with disproportionate ROI:

### 7.1 Channels we should add now (v1.0)

- **conda-forge** (Python + standalone `libn4m`) — single largest reach into the scientific Python world. ~70% of conda users install through conda-forge.
- **R-universe** — mandatory pre-CRAN proving ground; users get binaries hours after a tag.
- **JSR** (https://jsr.io) — Deno/Bun's preferred registry, gradually displacing npm for new JS work. Zero cost to publish alongside npm.
- **Maven Central** — the only JVM registry serious consumers will use. Required before Android, IDE plugins, or Kotlin DataFrame integrations make sense.
- **Homebrew tap** + **vcpkg** + **Conan** — the three places C/C++ users actually look.
- **Docker GHCR** — free, GitHub-native, no rate limits for our scale.
- **MSYS2 / MinGW-packages** — recipe path `mingw-w64-nirs4all-methods/PKGBUILD` in
  the `msys2/MINGW-packages` repo. Targets: `mingw-w64-ucrt-x86_64-nirs4all-methods`,
  `mingw-w64-clang-x86_64-nirs4all-methods`, `mingw-w64-clang-aarch64-nirs4all-methods`.
  This is where Windows C/C++ users actually get MinGW/UCRT libraries —
  the doc already ships `x86_64-pc-windows-mingw` triple archives but
  MSYS2 is the natural distribution surface for those.
- **Octave Packages** — package name `n4m` (`+n4m` package),
  source archive derived from `nirs4all-methods-matlab-octave-${VER}.zip`, PR to
  `gnu-octave/packages/index.yaml`.
  Treated as its own channel, not as a MATLAB sub-item. The Octave
  code paths and MEX `.oct` files live next to the MATLAB ones in
  `bindings/matlab/` but the distribution surface is independent.
- **MacPorts** — `Portfile` at `dports/science/nirs4all-methods/Portfile` in
  `macports/macports-ports`. Captures macOS scientific users who do
  not use Homebrew, especially institutional machines.
- **FreeBSD Ports** — `math/nirs4all-methods` or `science/nirs4all-methods`, package
  `pkg install nirs4all-methods`. Low-cost C/C++ surface; aligns with our
  Unix-portability story if FreeBSD becomes a Julia JLL target.
- **GitHub Packages** (fallback / RC channel) — Maven
  `maven.pkg.github.com/GBeurier/nirs4all-methods`, NuGet
  `https://nuget.pkg.github.com/GBeurier/index.json`, npm
  `npm.pkg.github.com` under `@nirs4all/methods`. Not a public-primary
  channel, but useful for release candidates, private downstream
  consumers, and testing immutable package layouts before Central /
  NuGet / npm.
- **Quay.io OCI mirror** — `quay.io/gbeurier/nirs4all-methods:${VER}` and
  `quay.io/gbeurier/nirs4all-methods-runner:${VER}`. Useful for OpenShift /
  HPC sites where GHCR or Docker Hub access is blocked.

### 7.2 Channels we should add at v1.1 / v1.2

- **Hugging Face Hub** — if we publish pretrained PLS models (calibrations from public NIRS datasets), this is the canonical model registry. `nirs4all-methods` could ship a `PretrainedModel` class that pulls a calibration from `huggingface.co/nirs4all-methods/wheat-protein-v1`.
- **Quarto / Posit Cloud** — for R + Python notebooks that demonstrate AOM-PLS / POP-PLS. RStudio's Posit Cloud has a free tier for open-source teaching material.
- **Observable / Observable Plot** — for the WASM binding's web-native demo notebooks. JSR-mirrored package works out of the box.
- **Pyodide registry** — nirs4all-methods already builds for `wasm32-unknown-emscripten`, so JupyterLite users get the library in-browser for free. A small PR to `pyodide/pyodide` registers it.
- **Spack** — HPC sites with no internet access download tarballs from a Spack mirror. The recipe is ~30 lines.
- **CRAN Task View: ChemPhys** — adding `pls4all` (slim CRAN subset) to the [ChemPhys task view](https://cran.r-project.org/web/views/ChemPhys.html) is a single PR and exposes the package to every chemometrician browsing CRAN.

### 7.3 Channels we explicitly defer

These are not worth the maintenance burden at v1.0 — re-evaluate as
consumer demand surfaces:

- **PHP / Perl / Tcl / Common Lisp / Crystal / Zig / Dart / OCaml / Haskell** — already covered in `roadmap/phase-36-deferred-bindings.md`.
- **WinGet / Chocolatey** (Windows): the `n4m_cli` is binary-only; Windows users typically install via `pip` or `nuget`, not via a CLI package manager.
- **Snap / Flatpak** (Linux desktop app stores): nirs4all-methods is a library, not a GUI app. Wrong audience.
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
2. **Rust companion crate and registry policy.** The official local
   pre-release is the single `n4m` crate; crates.io publication and any
   `n4m-sys` companion are deferred. Revisit both only when Rust distribution
   policy and build-time `libn4m` bundling are approved.
3. **Android NDK packaging timing.** The doc treats it as in-scope
   for the Maven Central artifact; current ROADMAP defers it to
   "phase 36 unfreeze". The two have to converge before v1.0.
4. **Single `libn4m` build vs. `libn4m-blas-omp` separate binary.**
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
conda-forge        → external feedstock: conda-forge/pls4all-feedstock + libn4m-feedstock
CRAN               → bindings/r/n4m/DESCRIPTION
R-universe         → external repo: gbeurier/universe → packages.json
MATLAB / Octave    → release-matlab.yml → nirs4all-methods-matlab-octave-${VER}.zip
MATLAB File Ex.    → optional manual .mltbx project, not wired in CI
npm                → bindings/js/package.json
JSR                → bindings/js/jsr.json (mirror of package.json)
Rust pre-release   → bindings/rust/n4m/Cargo.toml (local qualification only; no crates.io publication)
NuGet              → bindings/_archive/dotnet/Nirs4allMethods/Nirs4allMethods.csproj (managed, frozen PoC)
                   + bindings/_archive/dotnet/Nirs4allMethods.Native/Nirs4allMethods.Native.csproj (multi-RID native, frozen PoC)
Julia General      → bindings/_archive/julia/Nirs4allMethods.jl/Project.toml + Yggdrasil PR for the JLL (frozen PoC)
RubyGems           → bindings/_archive/ruby/nirs4all-methods.gemspec        (frozen PoC, TO CREATE)
LuaRocks           → bindings/_archive/lua/nirs4all-methods-${VER}-1.rockspec (frozen PoC, TO CREATE)
Nimble             → bindings/_archive/nim/nirs4all-methods.nimble           (frozen PoC, TO CREATE)
Maven Central      → bindings/jni/pom.xml + bindings/android/build.gradle.kts (TO CREATE)
Homebrew tap       → external repo: GBeurier/homebrew-nirs4all-methods/Formula/nirs4all-methods.rb (TO CREATE)
vcpkg              → external PR: microsoft/vcpkg/ports/nirs4all-methods (TO CREATE)
Conan Center       → external PR: conan-io/conan-center-index/recipes/nirs4all-methods (TO CREATE)
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
| MATLAB / Octave GitHub Release | ✅ source zip wired | optional File Exchange / Octave Packages mirrors remain manual |
| npm `@nirs4all/methods` | 🟡 package.json present, never published | needs npm scope + first publish |
| JSR | ⬜ not configured | mirror of npm — add after first npm publish |
| pkg.go.dev *(frozen PoC)* | 🟡 module exists, not tagged in subdir convention | frozen in `bindings/_archive/`; not a release target |
| Rust `n4m` *(official pre-release)* | 🟡 locally qualified, not published | `cargo package --list` includes both `LICENSES/*.txt` texts; no crates.io publication or cross-language prediction fixture/parity claim |
| NuGet *(frozen PoC)* | ⬜ never published | frozen in `bindings/_archive/`; would need `Nirs4allMethods` (managed) + `Nirs4allMethods.Native` (multi-RID native) |
| Julia General *(frozen PoC)* | ⬜ never registered | frozen in `bindings/_archive/`; not a release target |
| RubyGems *(frozen PoC)* | ⬜ no gemspec | frozen in `bindings/_archive/`; not a release target |
| LuaRocks *(frozen PoC)* | ⬜ no rockspec | frozen in `bindings/_archive/`; not a release target |
| Nimble *(frozen PoC)* | ⬜ no nimble file | frozen in `bindings/_archive/`; not a release target |
| Maven Central JNI | ⬜ never published | needs Sonatype Central Portal account + GPG key + Gradle wiring |
| Maven Central Android | 🔒 deferred per roadmap | unblocked when NDK is in the build host |
| Homebrew tap | ⬜ tap repo not created | needs second repo `homebrew-nirs4all-methods` |
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
| SOVERSION fix | ✅ fixed | verify clean release builds produce `libn4m.so.1 -> libn4m.so.1.22.0` |

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
   §2.2 enforcing `libn4m.so.${ABI_MAJOR}`).
3. Wire OIDC trusted publishing where supported (PyPI, npm, JSR, NuGet,
   RubyGems). Do **not** reserve names with placeholder versions until
   step 0+1+2 land — placeholder packages on immutable registries
   create their own cleanup problems.
4. Reserve every name in §1.3 with **real** 0.98.0 (or `0.98.0-rc.1`)
   uploads driven by the new release workflow.
5. PyPI + conda-forge + R-universe (the three highest-reach channels).
6. CRAN + Maven Central (highest reach but human-review cost).
7. Everything else in parallel.

### Single biggest threat to a smooth v1.0 cut

Publishing immutable packages with the wrong version and/or wrong ABI
identity. The remaining risks are (a) manifest drift if
`bump_version.sh --check` is bypassed, (b) stale local build artifacts
being copied into wheels or release archives, and (c) accidental new
public `n4m_*` exports that are not reflected in the ABI snapshot.
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
| U2 | 👤 | Enable 2FA on every account: GitHub (recovery codes saved), PyPI, npm, RubyGems, NuGet, Maven Central, MathWorks, RubyGems, JuliaHub, Docker Hub. Most registries now refuse publication without it. Rust has no registry account requirement while publication is deferred. |
| U3 | 👤 | (Optional) issue a fine-grained PAT to me (`gh auth login --with-token`) with scope `contents:write, packages:write, pull-requests:write` if you want me to open upstream PRs (vcpkg, conan-center, etc.) from your account. Otherwise I deliver branches/diffs and you push. |
| U4 | 🤖 | Maintain `scripts/bump_version.sh` and `.github/workflows/version-sync.yml`; extend them whenever a new manifest appears. |
| U5 | 🤖 | Keep the SOVERSION preflight green from clean builds and prevent stale `libn4m.so.0` artifacts from entering packages. |
| U6 | 🤖 | Write `.github/workflows/release.yml` + `release-binaries.yml` (14-triple cross-build) — gives the canonical GitHub-Release-as-source-of-truth. |

### 11.2 Per-channel split

Legend: 🤖 = Claude can do it alone · 👤 = you only · 🤖👤 = joint.

#### Tier A — your immediate focus (CRAN + PyPI)

| Channel | Code / manifests / CI | Account, tokens, web actions | Recurring human cost |
|---------|----------------------|------------------------------|----------------------|
| **CRAN `pls4all`** | 🤖 vendored static build in `bindings/r/pls4all/src/`, normalised `DESCRIPTION`, roxygen2 docs with `@return` everywhere, `tests/testthat/`, runnable `@examples`, `cran-comments.md`, `release-r.yml` | 👤 (1) R-hub v2: either set up GitHub-Actions checks via `rhub::rhub_setup()` / `rhub::rhub_check()` (needs a GitHub PAT in your local git credential store), or use R Consortium runners via `rhub::rc_new_token()` / `rhub::rc_submit()` (the legacy `validate_email()` / `check_for_cran()` flow is R-hub v1 and superseded), (2) submit tarball at `https://cran.r-project.org/submit.html` (5 min, manual web form), (3) confirm submission via the e-mail link CRAN sends | 🤖👤 CRAN review typically takes a few days; allow up to **10 working days** for a first submission. You forward each reviewer e-mail; I patch; resubmission uses the same web form with the *optional comment* field updated. |
| **R-universe `gbeurier.r-universe.dev`** | 🤖 the `packages.json` file contents | 👤 create the GitHub repo `gbeurier/gbeurier.r-universe.dev` (5 min, no special perms beyond your normal GH account) | none |
| **PyPI `pls4all`** | 🤖 normalised `pyproject.toml`, `setup.py` shim, `bindings/python/scripts/retag_wheels.py`, `bindings/python/scripts/build_libn4m_in_wheel.sh`, `release-python.yml` with `cibuildwheel` matrix, `environment: pypi` + `permissions: id-token: write` in the job | 👤 (1) PyPI account (https://pypi.org/account/register/), (2) reserve name `pls4all` — **TestPyPI does not reserve PyPI names**; reserve on real PyPI via the first `0.98.0rc0` Trusted-Publishing upload, or by uploading an explicit placeholder, or by claiming via PyPI support if taken, (3) enable **Trusted Publishing** at `https://pypi.org/manage/account/publishing/` — five fields for a *new* project: PyPI Project Name=`pls4all`, owner=`GBeurier`, repo=`nirs4all-methods`, workflow filename=`release-python.yml`, environment=`pypi` (optional but recommended). Replaces any need for long-lived tokens. | none (OIDC handles auth on every tag) |
| **TestPyPI `pls4all`** | 🤖 same workflow with `--repository testpypi` | 👤 separate account on `https://test.pypi.org/` + identical Trusted Publishing setup | none |

#### Tier B — high-reach, plug onto the same release pipeline

| Channel | Code | Account / tokens | Recurring |
|---------|------|------------------|-----------|
| **conda-forge** (two feedstocks: `libn4m` + `pls4all`) | 🤖 both `meta.yaml` recipes + PR to `conda-forge/staged-recipes` | 👤 conda-forge-admin GitHub app authorisation on your account (1 click during PR) | 🤖👤 reviewer comments, you forward to me |
| **npm `@nirs4all/methods`** | 🤖 `release-js.yml`; **provenance is automatic** under npm Trusted Publishing (no need to pass `--provenance`) | 👤 (1) create npm org `@nirs4all` (free, 5 min), (2) configure npm Trusted Publishing OIDC → repo `GBeurier/nirs4all-methods`. Requires **npm >= 11.5.1**, **Node >= 22.14.0**, and GitHub-hosted runners. | none |
| **JSR `@nirs4all/methods`** | 🤖 `jsr.json` mirror of npm manifest + workflow | 👤 create JSR scope `@nirs4all` at `https://jsr.io/new`; bind to GitHub repo via OIDC (1 click) | none |
| **Rust `n4m`** *(official pre-release)* | 🤖 maintain `bindings/rust/n4m`, its ABI probe, `cargo package --list`, and Linux sanitizer-runtime gate | no registry account or token while publication remains deferred | no publication; no cross-language prediction parity claim |

#### Tier C — JVM, .NET, niche-scientific (later wave)

| Channel | Code | Account / tokens | Recurring |
|---------|------|------------------|-----------|
| **NuGet `Nirs4allMethods` + `Nirs4allMethods.Native`** *(frozen PoC)* | 🤖 managed csproj + native multi-RID csproj + `release-dotnet.yml` | 👤 NuGet.org account + Trusted Publishing OIDC **if available on the account** (Microsoft documents a gradual rollout — keep an API-key fallback in `NUGET_API_KEY` until confirmed). | none, or `NUGET_API_KEY` fallback |
| **Maven Central JNI** | 🤖 `build.gradle.kts` + JReleaser config | 👤 (1) Sonatype Central Portal account, (2) **verify namespace** `io.github.gbeurier` — may auto-provision if you sign up via GitHub `GBeurier`, otherwise create a temporary public verification repo per Sonatype's instructions, (3) GPG key from U1 published, (4) paste 4 secrets: `MAVEN_GPG_PRIVATE_KEY`, `MAVEN_GPG_PASSPHRASE`, `CENTRAL_USERNAME`, `CENTRAL_TOKEN` | 🤖👤 click "Publish" in Central Portal UI for the first 1-2 releases; switch to auto-publish in JReleaser config afterwards |
| **Maven Central Android AAR** | 🔒 deferred (Android NDK not present on host — `roadmap/phase-36-deferred-bindings.md`) | — | — |
| **Julia General `Nirs4allMethods.jl`** *(frozen PoC)* | 🤖 keep `Project.toml` in sync | 👤 install GitHub App `JuliaRegistrator` on `GBeurier/nirs4all-methods` (1 click) | 🤖👤 comment `@JuliaRegistrator register subdir=bindings/_archive/julia/Nirs4allMethods.jl` on the release commit each tag (I can do it via `gh` if U3 is granted) |
| **`Nirs4allMethods_jll` via Yggdrasil** *(frozen PoC)* | 🤖 write `N/Nirs4allMethods/build_tarballs.jl` + PR to `JuliaPackaging/Yggdrasil` | 👤 nothing beyond the GitHub OAuth on your account (PR is from your fork) | 🤖👤 Yggdrasil reviewer comments |
| **RubyGems `nirs4all-methods`** *(frozen PoC)* | 🤖 `bindings/_archive/ruby/nirs4all-methods.gemspec` + `release-ruby.yml` | 👤 RubyGems account + Trusted Publishing OIDC (RubyGems supports it since 2024) | none |
| **LuaRocks `nirs4all-methods`** *(frozen PoC)* | 🤖 `nirs4all-methods-${VER}-1.rockspec` + `release-lua.yml` | 👤 LuaRocks account + API key in secret `LUAROCKS_API_KEY` (no OIDC) | none |
| **Nimble `nirs4all-methods`** *(frozen PoC)* | 🤖 `bindings/_archive/nim/nirs4all-methods.nimble` + PR to `nim-lang/packages` | 👤 nothing — PR from your GH fork | 🤖👤 nim-lang reviewer |
| **MATLAB / Octave source zip** | 🤖 `release-matlab.yml` archives `bindings/matlab/` as `nirs4all-methods-matlab-octave-${VER}.zip` | 👤 none for GitHub Release; optional File Exchange / `.mltbx` requires MathWorks account and a manually created Toolbox project | users build MEX locally with `build_mex.m` |
| **Octave Packages** | 🤖 PR to `gnu-octave/packages/index.yaml` | 👤 nothing | 🤖👤 Octave Packages reviewer |

#### Tier D — C/C++ ecosystem surfaces

| Channel | Code | Account / tokens | Recurring |
|---------|------|------------------|-----------|
| **GitHub Releases (binaries)** | 🤖 `release-binaries.yml` (14 triples) | 👤 nothing — uses native `GITHUB_TOKEN` | 👤 approve/push the tag |
| **Homebrew tap `gbeurier/nirs4all-methods`** | 🤖 `Formula/nirs4all-methods.rb` + bottle-build workflow inside the tap | 👤 **create the second repo `GBeurier/homebrew-nirs4all-methods`** (1 min, public) | none — formula bump PRs are automated |
| **vcpkg port** | 🤖 `portfile.cmake` + `vcpkg.json` + `versions/n-/nirs4all-methods.json` + PR to `microsoft/vcpkg` | 👤 nothing | 🤖👤 vcpkg reviewer comments |
| **Conan Center Index** | 🤖 `recipes/nirs4all-methods/all/conanfile.py` + PR to `conan-io/conan-center-index` | 👤 nothing | 🤖👤 CCI reviewer comments |
| **Docker GHCR `ghcr.io/gbeurier/nirs4all-methods`** | 🤖 `Dockerfile` + `release-docker.yml` (multi-arch buildx) | 👤 nothing — uses `GITHUB_TOKEN` | none |
| **Docker Hub mirror** | 🤖 parallel push in same workflow | 👤 Docker Hub account + token `DOCKERHUB_TOKEN` | none |
| **Quay.io mirror** | 🤖 `skopeo copy` step | 👤 Quay.io account + robot token | none |
| **MSYS2 `mingw-w64-nirs4all-methods`** | 🤖 PKGBUILD + PR to `msys2/MINGW-packages` | 👤 nothing | 🤖👤 MSYS2 reviewer |
| **MacPorts `science/nirs4all-methods`** | 🤖 Portfile + PR to `macports/macports-ports` | 👤 nothing | 🤖👤 MacPorts reviewer |
| **FreeBSD Ports** | 🤖 Makefile + PR to `freebsd/freebsd-ports` | 👤 may need a FreeBSD Bugzilla account if their workflow demands it | 🤖👤 FreeBSD ports reviewer |

#### Tier E — defensive / archival / mirroring

| Channel | Who | Action |
|---------|-----|--------|
| **Zenodo DOI** | 👤 | one toggle at `https://zenodo.org/account/settings/github/` to enable `GBeurier/nirs4all-methods`; every future Release auto-generates a DOI |
| **Software Heritage** | 👤 | one visit to `https://archive.softwareheritage.org/save/origin/?origin_url=https://github.com/GBeurier/nirs4all-methods` (~30 sec) |
| **Codeberg / GitLab mirror** | 👤 | create the accounts + repos + enable pull-mirror in their UIs (5 min each, optional) |
| **GitHub Packages (RC channel)** | 🤖 | reuse the per-language workflows with `--source github`; uses native `GITHUB_TOKEN`, no new secret |

### 11.3 Quick map of secrets you'll have collected by v1.0

The `Settings → Secrets and variables → Actions` of `GBeurier/nirs4all-methods` will hold, **at most**:

| Secret | Tier | Notes |
|--------|------|-------|
| (none for PyPI) | A | Trusted Publishing OIDC replaces this |
| (none for npm) | B | Trusted Publishing OIDC |
| (none for JSR) | B | OIDC |
| (none for Rust) | B | the official local pre-release is not published to crates.io, so no registry token is needed |
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
| M0.1 | 🤖 | Verify clean Linux builds produce `libn4m.so -> libn4m.so.1 -> libn4m.so.1.22.0` and no stale `libn4m.so.0` is staged into packages. | ABI identity checked |
| M0.2 | 🤖 | Run `scripts/bump_version.sh --check` and ensure `version-sync.yml` covers every active manifest. | version guard green |
| M0.3 | 🤖 | Refresh `cpp/abi/expected_symbols_linux.txt` only if the extra exported symbols are intentionally public. | ABI snapshot intentional |
| M0.4 | 🤖 | Re-run `ctest --preset dev-release --output-on-failure`. | C++ gate green |
| M0.5 | 👤 | Approve the verification diff, especially ABI symbols and package metadata. | tag-ready |

**Exit criteria.** `git tag -a v0.98.0 -m "M0 baseline"` produces a tree
where every binding manifest, the C++ project, the C ABI version, the ABI
symbol snapshot and the documentation agree.

### 12.2 Milestone M1 — PyPI publication (target: 2–3 working days)

Independent of M2. Can run in parallel.

| # | Who | Step | Output |
|---|-----|------|--------|
| M1.1 | 🤖 | Refactor `bindings/python/src/pls4all/_ffi.py` to also look in `<package>/.libs/` (wheel layout) | loader works for both editable and wheel install |
| M1.2 | 🤖 | Add `bindings/python/setup.py` shim with `Distribution.has_ext_modules = lambda self: True` **and** `package_data={"pls4all": [".libs/libn4m*", ".libs/n4m*"]}` + a `MANIFEST.in` that ships the native lib under `pls4all/.libs/`. Constrain `cibuildwheel` to **one CPython per platform/arch** (`CIBW_BUILD=cp312-*`) so retag does not collide. | non-pure platform wheel that *contains* `libn4m` |
| M1.3 | 🤖 | Add `bindings/python/scripts/build_libn4m_in_wheel.sh` (invoked by `CIBW_BEFORE_ALL_LINUX`/`MACOS`/`WINDOWS`) — builds the C++ core then copies `libn4m.so.1.22.0` / `libn4m.1.22.0.dylib` / `n4m.dll` into `bindings/python/src/pls4all/.libs/` | native blob present at wheel-build time |
| M1.4 | 🤖 | Add `bindings/python/scripts/retag_wheels.py` — runs **after** repair, rewrites three things in lockstep: the filename `cp312-cp312-*.whl` → `py3-none-*.whl`, the `*.dist-info/WHEEL` `Tag:` metadata lines, and `*.dist-info/RECORD` hashes/paths. Renaming only the file produces a wheel `pip` rejects. | valid `py3-none-${platform}` wheels |
| M1.5 | 🤖 | Verify `.github/workflows/release-python.yml`: `permissions: id-token: write`, `environment: pypi`, `cibuildwheel` matrix, repair commands for all three OSes, retag step after repair, and an installed-wheel smoke job that runs `pip install ./wheelhouse/pls4all-*.whl` in a clean venv and checks `pls4all.abi_version() == (1, 10, 0)` | reproducible wheel matrix + green smoke |
| M1.6 | 🤖 | Local dry run: build a single Linux wheel + run `pls4all.tests.test_smoke` | passes locally |
| M1.7 | 👤 | Create a PyPI account if needed; verify e-mail; **enable 2FA** | unblocks publishing |
| M1.8 | 👤 | Configure **Trusted Publishing** on PyPI at `https://pypi.org/manage/account/publishing/`: owner=`GBeurier`, repo=`nirs4all-methods`, workflow=`release-python.yml`, environment=`pypi`. Repeat on TestPyPI. | OIDC link |
| M1.9 | 👤 | Reserve the name `pls4all` on **real PyPI** — TestPyPI does *not* reserve PyPI names. Either rely on the pending-publisher + first real `0.98.0rc0` upload (M1.10/M1.12), or upload an explicit placeholder, or claim via PyPI support if the name is taken. | name claimed |
| M1.10 | 🤖 + 👤 | Trigger a release-candidate workflow against **TestPyPI** first: tag `v0.98.0rc0`, watch the run, confirm wheels land on `https://test.pypi.org/project/pls4all/0.98.0rc0/` | green RC |
| M1.11 | 🤖 + 👤 | Smoke test (note `--extra-index-url` for `numpy` etc. that don't live on TestPyPI): `pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple pls4all==0.98.0rc0 && python -c "import pls4all; pls4all.selftest()"` on Linux + macOS + Windows | parity OK |
| M1.12 | 👤 | Promote to PyPI: tag `v0.98.0`, the workflow publishes to the real `pypi.org` | live |
| M1.13 | 🤖 | Verify `pip install pls4all==0.98.0` from a clean venv on Linux/macOS/Windows in CI | green |

**Exit criteria.** `pip install pls4all` works in a fresh venv on every supported platform, the wheel actually bundles `libn4m` (verify with `unzip -l pls4all-0.98.0-py3-none-*.whl | grep libn4m`), the `WHEEL` metadata's `Tag:` lines match the filename, and `python -c "import pls4all; print(pls4all.abi_version())"` prints `(1, 10, 0)` from the *uploaded* artifact (not from a local dev install).

### 12.3 Milestone M2 — CRAN publication (target: 5–10 working days, gated by CRAN reviewer)

CRAN has a human reviewer in the loop. M2.4–M2.7 are the *only* irreducible humans-in-loop steps; everything before is code.

| # | Who | Step | Output |
|---|-----|------|--------|
| M2.1 | 🤖 | Audit the current **vendored static** R mode: regenerate the vendored `libn4m` subset from `cpp/`, remove divergent PCR code, use R compiler macros only (`CXX_STD = CXX17`, `PKG_CPPFLAGS`, `PKG_CXXFLAGS`, `PKG_LIBS`), remove non-portable flags such as `-march=*`, avoid CMake/internet at install, and keep example/test runtime under CRAN's per-check budget. | self-contained CRAN source package |
| M2.2 | 🤖 | Document every **exported** R function via `roxygen2`, including `@return` (mandatory for `\value{}` in the `.Rd`) and a runnable `@examples` block (any non-runnable example must be in `\dontrun{}` or `\donttest{}` with justification). Update `NAMESPACE` via `devtools::document()`. Add `tests/testthat.R`, `tests/testthat/test-*.R`, `Suggests: testthat (>= 3.0.0)`, `Config/testthat/edition: 3` in `DESCRIPTION`. Add a minimal vignette `vignettes/pls4all.Rmd` (or at least a strong `README.Rmd` rendered to `README.md`) — scientific packages without a vignette draw reviewer questions. | CRAN-clean docs + tests + (optional) vignette |
| M2.3 | 🤖 | Write `bindings/r/pls4all/cran-comments.md`: "New submission" line, justification of vendored sources (mention CECILL-2.1, vendored from `cpp/`, no external libs), compile-time budget, list of platforms tested via R-hub v2 / win-builder / macbuilder, list of `R CMD check --as-cran` results. Update the optional comment field on each resubmission instead of editing this file. | submission text |
| M2.4 | 🤖 | Verify `.github/workflows/release-r.yml`: `R CMD check --as-cran` on **the built tarball** across the matrix, `rcmdcheck::rcmdcheck()` artifacts, and fail on any `error`, `warning` or unintended `note`. | CI gate |
| M2.5 | 👤 | R-hub v2 onboarding — choose one path: **(a)** GitHub-Actions runners via `rhub::rhub_setup()` then `rhub::rhub_check()` (needs a GitHub PAT in your local git credential store); **(b)** R Consortium runners via `rhub::rc_new_token()` then `rhub::rc_submit()` (mail-token confirmation flow). The legacy `validate_email()` / `check_for_cran()` you may remember from R-hub v1 is superseded. | rhub unlocked |
| M2.6 | 🤖 | Run CRAN-like checks on the **built tarball** through R-hub v2 (selected platforms incl. `linux`, `macos-arm64`, `windows`), plus **win-builder** (`devtools::check_win_devel()`) and **macbuilder** (`devtools::check_mac_release()`). All must be green before submission. | CRAN-platform clean |
| M2.7 | 👤 | Build the final tarball: `R CMD build bindings/r/pls4all` → `pls4all_0.98.0.tar.gz` (I run the command; this row is your sign-off after eyeballing the diff). | tarball |
| M2.8 | 👤 | **Upload to CRAN**: open `https://cran.r-project.org/submit.html` → upload `pls4all_0.98.0.tar.gz` + paste `cran-comments.md` → submit (5 min web form). CRAN sends a confirmation e-mail; click the link. | submission queued |
| M2.9 | 👤 + 🤖 | Wait for CRAN reviewer — **a few days typically; allow up to 10 working days for a first submission**. CRAN policy forbids further submissions while one is pending. Forward each mail to me; I patch; **if code changes**, bump **patch** version `0.98.0 → 0.98.1` (CRAN convention) and resubmit via the web form with an updated *optional comment* field. If we want to revise *before* CRAN has reviewed (e.g. self-spotted issue), the same tarball name with bumped patch is the safest path; in-place `0.98.0` revisions are only acceptable if never actually queued. | accepted |
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
unless the bundled `libn4m` actually installs, loads, and reports
ABI `(1, 10, 0)` from the **exact uploaded artifact** (not from a
local dev install).

Concrete failure modes that all force a yank-or-`0.98.1`:
- A `py3-none-${platform}` wheel whose `WHEEL` `Tag:` metadata does
  not match the filename → `pip` rejects on install.
- A wheel that lacks `pls4all/.libs/libn4m*` because `MANIFEST.in`
  forgot to include it → `import pls4all` raises `OSError: cannot
  load libn4m` at runtime, on the user's machine, after a
  successful `pip install`.
- A CRAN source tarball whose `Makevars` references a non-portable
  flag (`-march=native`, `-Werror`, hard `clang++`) → fails on at
  least one CRAN platform → reviewer rejects.
- A vendored static build that pulls in `libgomp` or `libgfortran`
  by accident → CRAN check flags new system dependencies → reviewer
  rejects.

PyPI files are **immutable** — a bad wheel cannot be replaced under
the same version. The practical fix is `pip yank` plus a re-publish
under `0.98.1`. CRAN-side, a bad tarball is "unsubmit + resubmit
with bumped patch." Both cost ~24h of your time. The M2.6 / M1.11
smoke tests on installed-from-registry artifacts are there
specifically to prevent this.

### 12.6 Ready-to-start checklist

When you give the green light, I open work in this order, committing in small reviewable chunks:

- [ ] Branch `release/m0-verification`: M0.1 → M0.4, ABI snapshot decision, version-sync check, clean-build SOVERSION verification. Single PR.
- [ ] Branch `release/m1-pypi`: M1.1 → M1.6, then tag `v0.98.0rc0` to dry-run on TestPyPI. Verify installed-wheel smoke before promoting.
- [ ] Branch `release/m2-cran`: M2.1 → M2.4, tarball built and attached to a draft GH Release for your inspection. Verify `R CMD check --as-cran` on the *built tarball* across the matrix.
- [ ] Once M2.6 is green on R-hub v2 + win-builder + macbuilder: I hand you `pls4all_0.98.0.tar.gz` + `cran-comments.md` → you submit (M2.8).
- [ ] In parallel of CRAN review: M2.11 (create r-universe repo) + I drop `packages.json` into it.

Say "go" and I start on M0.
