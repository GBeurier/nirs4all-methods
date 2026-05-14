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
│   ├── synthetic_tiny_centered_v1.json
│   ├── synthetic_simpls_tiny_pls1_v1.json
│   └── synthetic_simpls_small_pls2_v1.json
├── tolerances.md                  Pair-wise abs / rel tolerance table.
├── python_generator/              Pinned scikit-learn + NumPy SIMPLS adapters.
└── r_generator/                   Pinned pls / ropls / mixOmics adapters.
```

## Regenerating the fixtures

```bash
cd parity/python_generator
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-lock.txt
pip install -e .
generate-fixtures --check       # asserts bit-identity with checked-in fixtures
generate-fixtures --regenerate  # rewrites the JSONs from scratch
```

R-side regeneration uses `renv` and produces the canonical cross-implementation
fixtures once the R generator lands:

```bash
cd parity/r_generator
Rscript -e 'renv::restore()'
Rscript generate_fixtures.R --check       # asserts bit-identity
```

## Current status

The checked-in fixtures are green in CI. NIPALS fixtures are generated from
scikit-learn `PLSRegression`; SIMPLS fixtures are generated from a deterministic
NumPy port of the local `nirs4all`/AOM SIMPLS reference. C++ parity tests assert
predictions, coefficients, preprocessing statistics and latent arrays within
`tolerances.md`.
