# Donor bindings — archive of the original nirs4all-methods Python + R bindings

**Status**: parked here by M9 of the merge plan. These are the donor's
historical bindings, lifted out of `_donor/` so the working tree no
longer has any donor-origin material under that prefix beyond the
remaining docs/roadmap archeology.

When the new package scaffolds at:

- `bindings/python_nirs4all_methods/`  (full umbrella)
- `bindings/python_pls4all/`            (slim PLS subset)
- `bindings/r_n4m/`                     (full R)
- `bindings/r_pls4all/`                 (slim R)

are activated by the M9 / M10 / M11 focused sessions, the donor copies
under `bindings/donor_imports/{python,r}/` will be deleted. Until then
they serve two roles:

1. **Reference**: the ctypes loader, sklearn classes, parsnip-style
   helpers, and registry-driven function dispatch the donor wrote are
   the starting point for the new packages' code generators.
2. **Smoke test**: existing donor parity tests under
   `bindings/donor_imports/python/tests/` can be run against
   `libn4m.so` once the build wires the unified sources.

## Layout

| Path | Origin | Notes |
|---|---|---|
| `bindings/donor_imports/python/` | `_donor/nirs4all-methods/bindings/python/` | ctypes + sklearn-compat. Seed for `bindings/python_nirs4all_methods/`. |
| `bindings/donor_imports/r/` | `_donor/nirs4all-methods/bindings/r/` | `.Call` dispatch + `Makevars` for vendored libn4m build. Seed for `bindings/r_n4m/`. |

## Pls4all-side bindings (NOT parked here)

- `bindings/python/` — pls4all's existing Python binding (with `pls4all.sklearn` ~68 classes)
- `bindings/r/pls4all/` — pls4all's existing R package (formula+S3 + pls-compat + mdatools-compat)
- `bindings/{matlab,octave,js,julia,go,rust,dotnet,ruby,lua,nim,jni,android}/` — pls4all's existing 12 secondary bindings

The merge plan's M9–M13 generates four NEW packages (above) and
refreshes the 12 secondary bindings to use the renamed `n4m_*` ABI
(after M5+M6 land).

## Original donor note

> Bindings will be rebuilt in Phase 22 (Python), Phase 23 (R),
> Phase 24 (MATLAB), Phase 25 (JS/WASM).
