# Method Catalog

Phase B makes `catalog/methods/<method_id>.yaml` the working catalog format.
Each method file carries `catalog_version: 1` and validates against
`catalog/schema/method.json`.

The legacy `catalog/methods.yaml` and `catalog/methods.discovered.yaml` are
kept during the transition so existing tooling and review diffs remain
auditable. New scripts prefer the split files and fall back to the legacy file
only when `catalog/methods/` has not been generated yet.

Until `catalog/methods.yaml` is retired, edit the legacy file and regenerate
the split files with `split_legacy_methods.py`. The write mode also prunes
stale split files left by renamed or removed method ids.

## Common Commands

```bash
# Regenerate per-method files from the legacy catalog.
python catalog/scripts/split_legacy_methods.py --write

# Verify the split files are current.
python catalog/scripts/split_legacy_methods.py --check

# Run the Phase B validator.
python catalog/scripts/validate.py

# Exercise the parser/emitter round-trip used by split/migration scripts.
python catalog/scripts/selftest.py

# Keep the old validation entry point green during the transition.
python catalog/scripts/validate_catalog.py

# Render catalog-derived API metadata without touching hand-written bindings.
python catalog/scripts/render_api.py --write

# Check or rewrite catalog schema versions.
python catalog/scripts/migrate_schema.py --from 1 --to 1
python catalog/scripts/migrate_schema.py --from 1 --to 1 --write
```

`catalog/scripts/validate.py --strict-abi` turns ABI symbol snapshot warnings
into errors. The current auto-discovered catalog still contains scaffolded
symbol names, so the default CI gate reports those as warnings until Phase B
finishes the ABI-symbol reconciliation.
