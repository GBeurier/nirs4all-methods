# Parity Gates

`chemometrics4all` validates non-PLS NIRS operators against two different
contracts:

| Gate | Comparator | Purpose | CI Stage |
| --- | --- | --- | --- |
| Binding parity | `parity/binding_parity.py` | Every language binding must match the same `libc4a` call path. | Stage 4 |
| Reference parity | `parity/reference_parity.py` | `libc4a` fixtures must agree with the canonical external implementation where one exists. | Stage 2 |

The unified runner is `parity/python_generator/scripts/run_parity_gate.py`.
It executes:

1. Stage 0, generator environment: the active Python environment must match
   `parity/python_generator/requirements-lock.txt` before fixtures can be
   regenerated.
2. Stage 1, fixture determinism: generators must reproduce every committed
   fixture byte-for-byte against `parity/fixtures/manifest.json`.
3. Stage 2, reference parity: `parity/run_reference_parity.py` compares
   fixtures with NumPy, SciPy, scikit-learn, PyWavelets, pybaselines, and
   selected `nirs4all` homemade references.
4. Stage 3, C++ parity: `chemometrics4all_tests` validates `libc4a` against
   the frozen fixtures.
5. Stage 4, binding parity: each binding validates itself against the built
   `libc4a`.

Stage 2 is a hard gate by default. Expected reference gaps live in
`parity/divergences.json`; any skipped suite or skipped case that is not listed
there fails the gate. Use `--skip-policy fail` to reject all skips, or
`--skip-policy legacy` only when reproducing old reports.

## Regenerating Fixtures

Use the locked generator environment. Running fixture generation from a newer
NumPy/SciPy stack can produce byte drift and must not update committed fixtures.

```bash
cd parity/python_generator
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install -e .
cd ../..
python parity/python_generator/scripts/run_parity_gate.py --build-dir build/dev-debug
```

After an intentional dependency/reference bump, refresh fixtures and the
manifest in the same locked environment:

```bash
python parity/python_generator/scripts/run_parity_gate.py \
  --build-dir build/dev-debug \
  --skip-cross-validation \
  --skip-cpp-tests \
  --update-manifest
```

For local debugging without touching fixtures:

```bash
python parity/run_reference_parity.py --cache-dir /tmp/c4a-reference-cache --quiet-cases
python parity/python_generator/scripts/run_parity_gate.py --skip-fixture-regen --build-dir build/dev-debug
```

## Reference Gaps

`parity/divergences.json` is the machine-readable reference contract. It
records stochastic operators whose upstream library cannot reproduce the exact
`c4a` PCG64 draw order, wavelet fixtures that still rely on a frozen layout
reference, and weak tolerance zones that need future tightening.

The file is intentionally separate from the implementation so dashboards and
release checks can distinguish:

- hard reference failures;
- expected skips;
- weak but currently accepted reference parity;
- pure binding parity against `libc4a`.
