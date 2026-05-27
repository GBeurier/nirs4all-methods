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

## Augmentation — spline simplification (v2)

Two augmenters are reserved public surface that returns
`N4M_ERR_NOT_IMPLEMENTED`:

| Operator | C ABI (reserved) | Python binding |
|----------|------------------|----------------|
| `Spline_X_Simplification` | `n4m_aug_spline_x_simplify_{create,apply,destroy}` | `n4m.sklearn.SplineXSimplificationAugmenter` |
| `Spline_Curve_Simplification` | `n4m_aug_spline_curve_simplify_{create,apply,destroy}` | `n4m.sklearn.SplineCurveSimplificationAugmenter` |

**Status (current ABI 1.9):**

- The six symbols are present in `cpp/abi/expected_symbols_{linux,macos,windows}.txt`
  — the surface is **reserved**, not missing. Implementing them later is a
  **behaviour-only** change (no ABI surface change, no snapshot regeneration).
- `*_create` / `*_destroy` work (state lifecycle is real).
- `*_apply` returns `N4M_ERR_NOT_IMPLEMENTED`.
- Asserted by `cpp/tests/test_augmenters_edge_splines_random.cpp`
  (`test_spline_x_simplification_stub`, `test_spline_curve_simplification_stub`)
  — these tests **must keep** asserting `N4M_ERR_NOT_IMPLEMENTED` until the
  trigger below is met.

**What already exists (so this is a small future task, not a rewrite):**

- The cubic B-spline primitive (`cpp/src/core/common/bspline.h`,
  `n4m_bspline_build_not_a_knot_cubic`) — the "splrep surrogate" — is
  implemented and already reused by `spline_x_perturbations.c`.

**Why deferred:** the reference behaviour
(`nirs4all.operators.augmentation.splines.Spline_X_Simplification` /
`Spline_Curve_Simplification`) draws a random control subset with
`numpy.random.Generator.choice(..., replace=False)` over a PCG64 stream.
Bit-exact parity with NumPy's choice-**without-replacement** algorithm
(its permutation/Floyd path) is materially harder than the uniform-draw
parity the existing fixtures replicate (`parity/fixtures/_rng_pcg64_stream_v1.json`),
and reimplementing it bit-for-bit is a disproportionate, brittle effort for
two low-priority operators. (Codex review, thread 019e69b7.)

**Trigger to implement (v2):** a tested, PCG64-backed
`choice(replace=False)` primitive lands in the core with NumPy parity fixtures.
Once that exists, fill in the two `*_apply` bodies (reusing the existing cubic
B-spline), add per-operator parity fixtures, and flip the stub tests to parity
assertions. No ABI bump is required.

**Explicitly NOT chosen:** shipping a deterministic-but-non-NumPy subsampling
as "expected divergence". Expected divergence is for *unavoidable* reference
differences, not for knowingly choosing different core semantics — that would
create permanent semantic debt and muddy the "one numerical core" guarantee.

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
