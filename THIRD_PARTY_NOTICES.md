# Third-Party Notices — nirs4all-methods

`nirs4all-methods` is distributed under `CeCILL-2.1 OR AGPL-3.0-or-later` (plus an optional
commercial license; see [`LICENSING.md`](LICENSING.md)). nirs4all-methods does **not** vendor the
components below — they are pulled from their official distributions — but their licenses are
acknowledged here as a courtesy and for compliance. Licenses are reported on a best-effort
basis; the authoritative text always ships with each upstream project.

`nirs4all-methods` (`libn4m`) has a C++17 core with **zero mandatory third-party dependencies** — it links only the C++ standard library. The components below are **optional** (dev/test) or belong to the **language bindings**, which only translate native objects and add their host ecosystem's dependencies. No third-party source is vendored into the shipped library.

| Component | License (SPDX) | Upstream |
|---|---|---|
| GoogleTest (dev/test) | BSD-3-Clause | https://github.com/google/googletest |
| pybind11 (Python binding) | BSD-3-Clause | https://github.com/pybind/pybind11 |
| Rcpp (R binding) | GPL-2.0-or-later | https://github.com/RcppCore/Rcpp |
| Emscripten (WASM binding) | MIT / University of Illinois | https://github.com/emscripten-core/emscripten |

For the exhaustive, version-pinned dependency tree and its licenses, run:

```
cmake --build build   # then inspect the binding toolchains' own SBOMs
```

License-family texts are bundled under [`LICENSES/`](LICENSES/): MIT, Apache-2.0, BSD-3-Clause.
