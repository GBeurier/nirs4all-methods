# Landing dashboard

The public landing page renders the generated cross-binding benchmark matrix
from `docs/_static/bench-data.json`.

## Current behavior

- Rows are `algorithm x dataset size x thread count`.
- Columns include the direct C++ ABI, the ABI-close Python function layer
  (`C4A.python`), the scikit-learn-compatible estimator layer
  (`C4A.sklearn`), public R bindings, and external reference libraries.
- A sticky `reference` column lists the external reference used for each
  method when one is registered.
- Internal binding cells report binding parity against the C++ ABI.
- External reference cells report reference parity against the native
  chemometrics4all output.
- Candidate external references are visible as `? candidate` cells when a
  library or paper implementation exists but the parity contract is not aligned.
- Methods without an executable external reference are visible as
  `⊘ reference needed` cells instead of disappearing from the external view.
- Cell tooltips expose the raw parity token, gate type, binding parity,
  reference parity, reference library, and reason.
- Presets expose `All`, `chemometrics4all`, and `externals` views; library
  groups separate internal C++/Python/R bindings from external Python and
  external R.

## Source layout

| File | Role |
|---|---|
| `benchmarks/cross_binding/orchestrator.py` | Generates `full_matrix.csv` timings and parity metadata. |
| `benchmarks/cross_binding/results/full_matrix.csv` | Generated timing matrix consumed by the dashboard builder. |
| `docs/_extras/build_landing.py` | Converts `full_matrix.csv` into the compact dashboard JSON payload. |
| `docs/_static/bench-data.json` | Committed dashboard payload. |
| `docs/_templates/landing.html` | HTML/CSS/JS shell for filtering, timing views, and parity display. |

## Refreshing the dashboard

```bash
PYTHONPATH=bindings/python/src \
CHEMOMETRICS4ALL_LIB_PATH=build/dev-release/cpp/src/libc4a.so \
python benchmarks/cross_binding/orchestrator.py \
  --repeat 3 --sizes 32x64 128x256 --threads 1 4

python docs/_extras/build_landing.py \
  --results benchmarks/cross_binding/results \
  --out docs/_static/bench-data.json
```

The dashboard intentionally keeps external divergences visible. A divergent
external reference is not hidden or coerced into a pass; it remains a
reference-parity failure so users can see where third-party implementations
do not match the selected oracle.
