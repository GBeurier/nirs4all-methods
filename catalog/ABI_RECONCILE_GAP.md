# Catalog ↔ ABI reconciliation — COMPLETE (Phase B)

`reconcile_abi.py --check` and `validate.py --strict-abi` are GREEN: every exported `n4m_*` symbol is accounted for.

- **548 method symbols** → **188 catalog methods** (longest-prefix assignment from the reviewed `abi_method_map.yaml` + `abi_missing_methods.yaml`).
- **121 infra/support symbols** → explicit `abi_infra.yaml` (context/config/array/status/version/rng/matrix + the shared model & split-result handles).
- union = 669/669 exported n4m_* symbols; 0 orphans, 0 unreal.

## What the reconciliation did
- Replaced the M2 auto-discovered GUESS symbols (coverage was 1.9%) with the REAL exported symbols, 100% now.
- ADDED 45 public methods that libn4m exports but M2 missed (POP-PLS, ~25 model variants, ~15 preprocessing variants incl. calibration-transfer aligners + DS family, the feature filters, the combined edge-artifacts augmenter, approximate-PRESS).
- REMOVED 17 helper/concept stubs that had no public ABI symbol (internal aggregates, config-side handles, symbol-less diagnostic concepts).

## Gates (in catalog-validate.yml)
- `validate.py --strict-abi`: every catalog `abi_symbols` entry is real, no symbol is claimed by two methods, and union(method, infra) covers the full snapshot.
- `reconcile_abi.py --check`: same coverage invariant from the reconciliation side.

To re-reconcile after an ABI change: edit `abi_method_map.yaml` / `abi_missing_methods.yaml`, then `reconcile_abi.py --add-missing && reconcile_abi.py --write && split_legacy_methods.py --write && validate.py --strict-abi`.
