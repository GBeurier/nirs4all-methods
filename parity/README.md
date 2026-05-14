# Parity gate

`pls4all` is validated against multiple reference PLS implementations. The parity gate exists so that any numerical change in the C core surfaces immediately as a test failure, with a documented tolerance per pair of implementations.

## Layout

```
parity/
├── schema/
│   ├── fixture_schema_v1.json     JSON Schema for the fixture file format.
│   └── fixture_schema_v1.md       Human description of every field.
├── fixtures/
│   ├── manifest.json              Per-fixture SHA-256 + generator git rev.
│   ├── synthetic_small_pls1_v1.json
│   ├── synthetic_small_pls2_v1.json
│   └── synthetic_tiny_centered_v1.json
├── tolerances.md                  Pair-wise abs / rel tolerance table.
├── python_generator/              Pinned scikit-learn + nirs4all adapters.
└── r_generator/                   Pinned pls / ropls / mixOmics adapters.
```

## Regenerating the fixtures

```bash
cd parity/python_generator
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-lock.txt
python generate_fixtures.py --check       # asserts bit-identity with checked-in fixtures
python generate_fixtures.py --regenerate  # rewrites the JSONs from scratch
```

R-side regeneration uses `renv` and produces the same three canonical fixtures:

```bash
cd parity/r_generator
Rscript -e 'renv::restore()'
Rscript generate_fixtures.R --check       # asserts bit-identity
```

## Status at Phase 0

The Phase 0 fixtures are **red on the numerical side by design**: every
`p4a_model_fit` call returns `P4A_ERR_NOT_IMPLEMENTED`. PR-1 (Phase 1) plugs
NIPALS into the chassis and flips the loader test from "asserts
NOT_IMPLEMENTED" to "asserts coefficients agree within `tolerances.md`".
