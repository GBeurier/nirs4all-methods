# Disposable external ranking probes

These are feasibility harnesses, not production adapters or scientific parity
tests. Sources are fixed by `fetch_sources.py`; no external source is vendored
here. Python 3.12+ is needed for the tar extraction safety filter. Network access
is needed only to fetch sources and Cargo dependencies.

From the nirs4all-methods root, copy these files to a disposable location:

```bash
cp -a docs/ranking_external_probes /tmp/n4m-ranking-external-replay
cd /tmp/n4m-ranking-external-replay
python3 fetch_sources.py
```

The Rust manifest `path` dependencies resolve against the fetched source trees.
The preserved Cargo.lock files pin transitive package versions. To isolate new
Cargo caches as well, set `CARGO_HOME` to a disposable directory before commands;
this assessment used the existing user Cargo download cache but installed no
shared binaries or packages.

## C++ rfr static archive and numerical smoke

```bash
g++ -std=c++17 -O1 -fPIC -fvisibility=hidden -I rfr/include rfr_probe.cpp -o rfr_probe
./rfr_probe
g++ -std=c++17 -O1 -fPIC -fvisibility=hidden -DPROBE_LIBRARY -I rfr/include -c rfr_probe.cpp -o rfr_probe.o
ar rcs librfr_probe.a rfr_probe.o
g++ driver.cpp librfr_probe.a -o rfr_static_probe
./rfr_static_probe
wc -c librfr_probe.a rfr_static_probe
readelf -d rfr_static_probe
nm -g --defined-only librfr_probe.a
# Include closure, excluding system headers:
g++ -std=c++17 -MM -I rfr/include rfr_probe.cpp
```

Observed output: `prediction=1 marginal=1 total_variance=0.25`, exit 0.
The only static library on the explicit link line is `librfr_probe.a`; it contains
`rfr_probe.o`. The compiler links its platform C/C++ runtime normally. The reached
upstream headers are listed in `rfr-header-manifest.txt`. The archive also holds
instantiated C++ template/standard-library symbols; a future shared n4m link must
keep those hidden with the existing export controls. The harness symbol is not
part of the n4m ABI. `-fvisibility=hidden` by itself is not proof that an eventual
shared-library export list is correct.

## lme4 deliberately minimal predictor-only compile

```bash
g++ -std=c++17 -I lme4/src -c lme4_probe.cpp -o lme4_probe.o
```

On the assessed machine this fails at `RcppEigen.h` because no RcppEigen headers
are supplied. This is an environment-limited compile, not proof that lme4 cannot
be built. The stronger extraction concern comes from its SEXP constructor and
Rcpp-owned response/data interfaces in the pinned source.

## mixeff static archive and C++ caller

```bash
CARGO_TARGET_DIR="$PWD/rust-target" cargo build --locked --manifest-path mixeff-probe/Cargo.toml -j 1
g++ mixeff_driver.cpp rust-target/debug/libn4m_ranking_mixeff_probe.a -ldl -lpthread -lm -o mixeff_static_probe
./mixeff_static_probe
wc -c rust-target/debug/libn4m_ranking_mixeff_probe.a mixeff_static_probe
readelf -d mixeff_static_probe
CARGO_TARGET_DIR="$PWD/rust-target" cargo metadata --locked --manifest-path mixeff-probe/Cargo.toml --format-version 1 > mixeff-deps.json
```

Observed output: `slope=0.75`, exit 0. The assertion tolerates 0.02 because this
smoke only checks a known synthetic effect. It is not an lme4 oracle comparison.
Native archive: 81,899,562 bytes; executable: 29,900,672 bytes, debug profile with
`debug=0`, otherwise unoptimized, no stripping or LTO. Dependencies shown by
`readelf`: libstdc++, libm, libgcc_s, libc and the platform loader. This static
subset requires no R/Python/NLopt shared runtime. A production C adapter would
validate all input, catch panics/errors and return owned native statuses; the
constant-data probe is not that adapter.

The following additional compile-only probe also passed (165.34 s wall,
1,169,936 KiB peak RSS):

```bash
CARGO_TARGET_DIR="$PWD/rust-target" cargo build --locked --manifest-path mixeff-probe/Cargo.toml --target wasm32-unknown-emscripten -j 1
```

This is not a JS/WASM binding test or proof that Rust archives and the installed
Emscripten C++ runtime can be linked together. No WASM execution was performed.

## Dependency inventories without compiling other candidates

```bash
cargo metadata --locked --manifest-path metadata-probe/Cargo.toml --format-version 1 > other-rust-deps.json
```

`metadata-probe` jointly requests formula-free glmm and default-free OpenBLUP core.
`other-rust-license-summary.json` records normal/build dependency closures: 145
packages reachable from glmm and 175 from OpenBLUP core, including the candidate
itself. These are conservative source-package inventories under the joint
consumer's feature union and target conditions, not production-linked library
counts. They include architecture-specific packages; no glmm/OpenBLUP compilation
or executed fit is asserted. `mixeff-license-summary.json` records 73 packages
including its wrapper. Inventory counting follows `resolve.nodes[].deps`, keeping
normal and build edges and dropping dev-only edges; actual target pruning is not
performed.

The fetched upstream LICENSE files, license-file fields, target build scripts
and all final copied/linked notices must be reviewed before distribution. An SPDX
inventory alone does not establish notice compliance.
