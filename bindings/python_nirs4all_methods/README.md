# nirs4all-methods (Python full umbrella) — M9 scaffold

**Status**: scaffold only. Package metadata declared; the actual
extension build (vendored sources + ctypes loader + sklearn classes)
lands in the M9 focused session.

Generated from `catalog/subsets/nirs4all_methods.yaml` (active subset
covering 10 categories, 160 methods).

When activated:
```bash
pip install nirs4all-methods
```
exposes:
```python
from nirs4all_methods.preprocessing.scatter import SNV
from nirs4all_methods.models.pls import SIMPLS
from nirs4all_methods.selection import CARS
# ... full catalog
```

Until M9 lands, use the legacy `pls4all` package or build from the
unified repo source directly.
