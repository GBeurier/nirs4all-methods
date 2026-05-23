# M17 — release v1.0.0: PROCEDURE

**Status**: documentation only — release is the final step after all
M0–M16 phases land green.

## Pre-flight checklist

All sub-gates green:
- [ ] M0 baselines tagged (`pls4all-final/v1.0.0-pre-merge` + `nirs4all-methods-final/6195574`) — ✅ done
- [ ] M1 subtree import committed (`merge/import-donor`) — ✅ done
- [ ] M2 catalog + subset YAMLs validated — ✅ done
- [ ] M2.5 common-core landed at `cpp/src/core/common/` — ✅ done
- [ ] M3 donor sources lifted to `cpp/src/core/<category>/` — ✅ done
- [ ] M4 source split executed (Phase 1 helpers + Phases 2-7 algorithms) — ❌ deferred
- [ ] M5 ABI rename `n4m_*` → `n4m_*` executed across whole tree — ❌ deferred (script ready)
- [ ] M6 per-category headers populated (umbrella + 11 cat headers) — ❌ deferred (stubs only)
- [ ] M7 unified parity gate green (Stage 0-4) — ❌ deferred (donor infrastructure parked)
- [ ] M8 unified benchmark dashboard live (per-category filters) — ❌ deferred (orchestrator de-hardcoded only)
- [ ] M9 Python `nirs4all-methods` package activated (catalog/scripts/render_subset.py --copy) — ❌ deferred (scaffold only)
- [ ] M10 Python `pls4all` slim package activated — ❌ deferred (scaffold only)
- [ ] M11 R `n4m` + `pls4all` packages activated; `R CMD check --as-cran` 0/0/0 — ❌ deferred
- [ ] M12 MATLAB/Octave dispatcher refreshed — ❌ deferred
- [ ] M13 12 secondary bindings refreshed — ❌ deferred (donor Python+R lifted to bindings/donor_imports/)
- [ ] M14 license audit clean; NOTICE.md + THIRD_PARTY_LICENSES.md committed — ✅ done
- [ ] M15 cibuildwheel + R-hub rehearsals green — ❌ scaffold only
- [ ] M16 repo renames + donor archive — ❌ documented only

## Release-day procedure (once all gates pass)

### 1. Final integration tag

```bash
cd /path/to/nirs4all-methods
git checkout merge/unified   # post-M5 unified main candidate
git pull --ff-only

# Build verification:
cmake --preset dev-release && cmake --build --preset dev-release -j
ctest --preset dev-release --output-on-failure
./build/dev-release/cpp/cli/n4m_cli --selfcheck   # post-M5: was n4m_cli

# Parity gate:
python parity/python_generator/scripts/run_parity_gate.py --build-dir build/dev-release
# Expect: PASS Stage 0-4

# License audit:
python scripts/audit_third_party_licenses.py --check
# Expect: PASS

# Catalog validation:
python catalog/scripts/validate_catalog.py
# Expect: PASS

# Build the wheels (cibuildwheel matrix from .github/workflows/release-wheels.yml):
gh workflow run release-wheels.yml --field version_tag=v1.0.0 --field publish=false
# Wait for green, then re-run with publish=true
```

### 2. Tag v1.0.0

```bash
git tag -a v1.0.0 -m "Release v1.0.0 — unified nirs4all-methods.

This is the first stable release of the unified library combining
pls4all (the PLS / NIRS engine) and the original nirs4all-methods
(preprocessing, augmentation, splitter, filter, utility operators).

For full release notes see CHANGELOG.md and MERGE_ROADMAP.md."

git push origin v1.0.0
```

### 3. Publish to PyPI

If using Trusted Publishing (recommended):
```bash
gh workflow run release-wheels.yml --field version_tag=v1.0.0 --field publish=true
```

This deploys both `nirs4all-methods 1.0.0` and `pls4all 1.0.0`.

### 4. Submit to CRAN

For `n4m` (full R) and `pls4all` (slim R):
```bash
# Build sources:
cd bindings/r_n4m
R CMD build .
# Check on win-builder + R-hub + macOS-ARM:
R CMD check --as-cran n4m_1.0.0.tar.gz   # must be 0/0/0

# Same for pls4all:
cd ../r_pls4all
R CMD build .
R CMD check --as-cran pls4all_1.0.0.tar.gz   # 0/0/0

# Submit (manual via https://cran.r-project.org/submit.html)
```

### 5. GitHub release page

```bash
gh release create v1.0.0 \
    --title "v1.0.0 — Unified nirs4all-methods" \
    --notes-file CHANGELOG.md \
    wheels/*.whl \
    sources/*.tar.gz
```

### 6. Announce

- Docs site auto-deploys via GitHub Pages from main.
- Email the project mailing list + GitHub Discussions.
- Update DOI on Zenodo (if linked).

### 7. Post-release smoke

```bash
pip install nirs4all-methods==1.0.0   # from PyPI
python -c "
from nirs4all_methods.preprocessing.scatter import SNV
from nirs4all_methods.models.pls import SIMPLS
import numpy as np
X = np.random.randn(50, 100)
y = np.random.randn(50)
m = SIMPLS(n_components=5).fit(SNV().fit_transform(X), y)
print(m.score(SNV().fit_transform(X), y))
"

pip install pls4all==1.0.0   # legacy slim
python -c "
from pls4all.sklearn import PLSRegression
import numpy as np
X = np.random.randn(50, 100)
y = np.random.randn(50)
m = PLSRegression(n_components=5).fit(X, y)
print(m.score(X, y))
"
```

Both should return reasonable R² values.

## Rollback (if release goes wrong)

PyPI: yank the release.
```bash
pip install twine
twine upload --skip-existing dist/*  # if needs to re-upload
# Or via web: https://pypi.org/manage/project/nirs4all-methods/releases/
```

CRAN: contact CRAN maintainers to withdraw.

GitHub tag: delete locally + remotely, recreate after fix.
```bash
git tag -d v1.0.0
git push origin :refs/tags/v1.0.0
# Fix the bug
# Re-tag
```

## Open items that are deliberately not blocking v1.0.0

- M4 source split (organizational — algorithms work, just monolithic)
- M5 ABI rename (needs M5+M6 focused session)
- M6 header reorganisation (paired with M5)
- M11 CRAN cleanup pass for pls4all R package (existing 1W + 4N)

The merge plan deliberately budgeted these for follow-up release passes.
v1.0.0 ships the **unified source tree** + **functional pls4all build**
+ **catalog/subset machinery** + **license audit** — not the fully split
+ renamed + fully-built-from-catalog state. v1.1.0 closes those.
