<!--
  Generic fallback PR template. For change-specific templates (new method,
  binding, subset, etc.), append ?template=<name>.md to the PR URL or pick
  the matching template from .github/PULL_REQUEST_TEMPLATE/. Available:
    - method-add.md       (new method)
    - method-update.md    (existing method behavioural change)
    - external-reference.md  (alternate / pinned version of an external ref)
    - binding-add.md      (new language binding)
    - binding-update.md   (new profile / wrapper update on existing binding)
    - subset.md           (new packaging subset)
    - parity-fix.md       (numerical parity discrepancy fix)
-->

## Summary

<!-- What changed and why, in 1–3 bullet points. -->

## How to verify

<!--
  Concrete commands to run / files to look at. e.g.
    cmake --preset dev-release && ctest --preset dev-release
    make parity METHOD=<id>
-->

## Linked issues

Closes #

## Reviewer checklist — Phase A §2.10 invariants

- [ ] No new hand-written per-(method × idiomatic-profile) code (all wrappers from `render_api.py`)
- [ ] No new entry duplicates a fact that lives in `catalog/methods/`
- [ ] `cpp/abi/expected_symbols_*.txt` snapshots regenerated automatically (not hand-edited) if the ABI surface changed
- [ ] CHANGELOG entry added under the next unreleased section
- [ ] CI matrix green
