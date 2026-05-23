<!-- .github/PULL_REQUEST_TEMPLATE/parity-fix.md -->

## Parity fix: <method_id> vs <reference>

Closes #<issue_number>

### Root cause

<!--
  Explain what was wrong in n4m (or in the reference snapshot) and why the
  comparator was flagging the discrepancy. Cite the relevant code path /
  algorithm step. If the root cause is in the reference (e.g. upstream bug),
  link the upstream tracker.
-->

### Before / after

<details>
<summary>Comparator output</summary>

```
(paste before/after rmse_rel, max_abs, sign_agreement for each scenario)
```
</details>

### Tolerance change (if any)

- [ ] **No tolerance change** — the fix brings n4m back inside the existing tolerance
- [ ] **Tighter tolerance** — explain below
- [ ] **Looser tolerance** — explain *and* justify below; needs reviewer sign-off

<!-- justification for tolerance change goes here -->

### Regression coverage

- [ ] Added scenario(s) under `parity/scenarios/<method_id>.yaml` that would
      have caught the bug before the fix
- [ ] `make parity METHOD=<method_id>` green on the canonical host card
- [ ] Inter-binding parity re-verified if the fix is in a C ABI wrapper

### Reviewer checklist — §2.10 invariants

- [ ] Fix is in the right layer (core algorithm vs C ABI wrapper vs binding adapter)
- [ ] No hand-written per-method per-profile code introduced as a side-effect
- [ ] `CHANGELOG.md` entry added under the next unreleased section
