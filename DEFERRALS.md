# Deferrals — deliberate, documented non-implementations

This file is the canonical record of features that are **intentionally not
implemented** in the current ABI, with a stable public surface reserved for
them and a concrete trigger for when they land. A deferral here is **not
release debt** — it is a decided scope boundary. Code comments that say
"see DEFERRALS.md" point here.

A deferral is legitimate only when all of these hold:

- the public surface is **reserved** (ABI symbols exist; `create`/`destroy`
  work; the operative call returns a clear, tested error);
- there is a **documented reason** the work is gated;
- there is a **concrete trigger** that unblocks it;
- tests **assert the deferred behaviour** so it cannot regress silently.

---

## Dashboard — interactive SPA (D-SPA)

The interactive Svelte/Vite dashboard (Phase D, D1–D15) is **deferred, not
cancelled**. D-min shipped the load-bearing half: a schema-validated
`dashboard.json` **contract** + automated per-method score cards, which the SPA
will consume **unchanged**.

- **Reserved surface:** the `make dashboard-data` / `dashboard-serve` /
  `dashboard-build` targets are stubs that print "not yet bootstrapped (Phase D)"
  and exit 0; the `dashboard/` app does not exist.
- **Reason:** the static Sphinx/landing dashboard + the `method_scores` cards
  already make the parity/timing signal visible and maintainable. The SPA is UX
  polish, made cheap by the fixed contract.
- **Trigger:** when the static dashboard becomes the limiting factor for
  consuming the signal (rich filtering, drift history, multi-host).
- **Full scope:** [`docs/dashboard_contract.md`](docs/dashboard_contract.md) → "D-SPA (deferred): scope".

## Binding-scaling infrastructure (Phase F-prep)

The 10+-language binding-scaling infra is **reduced and deferred** by the
4-language target + the hand-written-idiomatic decision in
[`bindings/SPEC.md`](bindings/SPEC.md).

- **Cancelled (over-engineering at 4-language scale):** the framework-profile
  schema (F-prep-3) and the per-`(method × profile)` template/codegen engine
  (F-prep-4).
- **Deferred, low value:** the skeleton generator (`make new-binding LANG=`,
  a stub) and the unified `bindings.yml` / `release-bindings.yml` matrices
  (a CI/release consolidation, not new capability).
- **Trigger:** a request for a 5th+ target language.
- **Full scope:** [`bindings/SPEC.md`](bindings/SPEC.md) → "F-prep scope".
