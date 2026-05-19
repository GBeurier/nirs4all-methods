# Read the Docs

The documentation builds in two environments:

| Target | Triggered by | Source of truth for benchmark data |
|---|---|---|
| **GitHub Pages** (`https://gbeurier.github.io/chemometrics4all/`) | Push to `main` via `.github/workflows/docs.yml` | Scaffold `docs/_static/bench-data.json` until a real benchmark runner is implemented |
| **Read the Docs** (`chemometrics4all.readthedocs.io`) | RTD webhook on every push | `docs/_static/bench-data.json` (the committed scaffold fallback) |

The two builds share the same Sphinx config (`docs/conf.py`).

## One-time RTD setup

1. **Push** `.readthedocs.yaml` and `docs/_static/bench-data.json` to
   `main`. Both already live in this repo.
2. Sign in at <https://readthedocs.org/> and **Import a Project** →
   GitHub provider → pick `GBeurier/chemometrics4all`.
3. In **Admin → Advanced settings**:
   - *Documentation type*: Sphinx HTML.
   - *Privacy level*: Public.
   - *Default branch*: `main`.
4. Optional: in **Admin → Domains** add a custom CNAME
   (e.g. `docs.chemometrics4all.io`).
5. Run the first build (it auto-triggers after import). Logs land at
   `https://readthedocs.org/projects/chemometrics4all/builds/`.

After the first green build the URL is
`https://chemometrics4all.readthedocs.io/`. Versioned snapshots live at
`/en/<branch-or-tag>/`.

## Build behaviour

The RTD container clones the repo, installs `docs/requirements.txt`
(Sphinx + myst-parser + linkify), then runs:

```
sphinx-build -b html docs $READTHEDOCS_OUTPUT/html
```

Fresh clones and RTD builds use `docs/_static/bench-data.json`, which is
currently a scaffold payload with no timing or parity results.

## Refreshing the bundled benchmark payload

```bash
# With no result CSV, this refreshes the scaffold placeholder:
python docs/_extras/build_landing.py \
    --results benchmarks/cross_binding/results \
    --out    docs/_static/bench-data.json

git add docs/_static/bench-data.json
git commit -m "Refresh benchmark dashboard payload"
```

The next push triggers the RTD webhook which rebuilds the docs against
the fresh JSON.

## When RTD builds fail

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: yaml` while building methods | `pyyaml` not in `docs/requirements.txt` | RTD doesn't run `build_methods.py`; only commit the *generated* pages |
| `toctree contains reference to nonexisting document` | A docs page was linked before it was committed | Add the page or remove the toctree entry |
| `landing.html: bench_data_json is undefined` | `docs/_static/bench-data.json` not committed | Commit the JSON snapshot |
| Build times out | A generated benchmark CSV or heavy result artifact was committed by accident | Re-check `.gitignore`; keep the JSON fallback only |

## Local mirror of an RTD build

```bash
# Reproduce the RTD container build locally:
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
xdg-open docs/_build/html/index.html
```

This is the *same* command RTD runs, so anything that fails locally
will also fail on RTD.
