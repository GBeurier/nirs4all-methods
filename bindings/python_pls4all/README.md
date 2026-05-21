# pls4all (Python slim) — M10 scaffold

**Status**: scaffold only. Vendored-source build comes from
catalog/scripts/render_subset.py in the M10 focused session.

Generated from catalog/subsets/pls4all.yaml (68 methods after closure).

When activated:
```bash
pip install pls4all
```
exposes:
```python
from pls4all.sklearn import PLSRegression  # legacy alias preserved
from pls4all.sklearn import SIMPLS
from pls4all.sklearn import AOMPLSRegressor, POPPLSRegressor
# ... 68 sklearn classes
```

Backwards-compatible with the legacy pls4all 0.97.x sklearn surface
via contract tests (M10 deliverable: tests/legacy_imports.py).
