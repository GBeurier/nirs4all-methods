<!-- .github/PULL_REQUEST_TEMPLATE/method-update.md -->

## Method update: <method_id>

Closes #<issue_number>

### Summary of behavioural change

<!-- Before / after for the affected parameters, outputs, or algorithm steps. -->

### Catalog impact

- [ ] `catalog/methods/<method_id>.yaml` updated to reflect new behaviour
- [ ] `catalog_version` bumped if schema impacted; `migrate_schema.py` ran on the entire catalog
- [ ] Parameter migration notes added inline in the YAML

### Parity impact

- [ ] Tolerances under `parity.tolerances:` re-calibrated where the change shifts numerics
- [ ] Justification for any tolerance loosening captured in PR description
- [ ] `parity/scenarios/<method_id>.yaml` updated if new scenarios are needed to characterise the change
- [ ] Reference snapshots re-generated where required (`make snapshot`)
- [ ] `make parity METHOD=<method_id>` green; verdict diff attached below

<details>
<summary>Parity verdict — before vs after</summary>

```
(paste before/after rmse_rel, max_abs, sign_agreement for each affected scenario)
```
</details>

### Bindings impact

- [ ] `render_api.py` re-run; generated wrappers committed
- [ ] No hand-written per-(method × profile) code introduced
- [ ] Inter-binding parity green
- [ ] Examples regenerated where the API signature changed

### Dashboard impact

- [ ] Stale-badge logic acknowledged (next bench run will refresh the timing column)
- [ ] If the change affects benchmark scenarios, `benchmarks.yml` re-dispatched and the new numbers committed

### Reviewer checklist — §2.10 invariants

- [ ] `catalog_version` bump (if any) is reflected in every YAML
- [ ] ABI snapshot regen was automatic
- [ ] `CHANGELOG.md` entry added under the next unreleased section
