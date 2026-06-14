# nirs4all-methods · parity & benchmark explorer — proposal v2

An alternative, production-grade dashboard for the libn4m parity + benchmark
matrix. Built to be opened **side-by-side** with the current
`build/cross_binding_dashboard/index.html` so the maintainer can pick.

## How to open

**Double-click `index.html`.** That's it.

- No build step, no dev server, no network, no CDN.
- Works from `file://` — `app.js` reads the data from the global
  `window.BENCH_DATA` (defined in `bench-data.js`); it never `fetch()`es,
  because `fetch()` of a local file is blocked under `file://`.
- Fonts are a pure system stack (the brand uses Inter / IBM Plex Sans /
  JetBrains Mono via Google Fonts; offline we degrade to the system fallbacks
  of the same families). No `<link href=http…>`, no `<script src=http…>`.

Optional, to mimic the deployed site over HTTP:

```bash
cd proposals/dashboard_v2 && python3 -m http.server 8077
# then open http://localhost:8077/
```

## Files

| File | Role |
|------|------|
| `index.html` | Self-contained shell. Double-click to open. |
| `styles.css` | Design system cloned from the nirs4all web family (warm paper, teal/cyan/indigo). |
| `app.js`     | All logic: model build, fuzzy search, filtering, drawer drill-down, hand-rolled inline-SVG timing charts, parity table. Vanilla JS, zero deps. |
| `bench-data.js` | The full canonical `docs/_static/bench-data.json` assigned to `window.BENCH_DATA`. Regenerate with the one-liner below. |

Regenerate the data after a new benchmark run:

```bash
printf 'window.BENCH_DATA = ' > bench-data.js \
  && cat ../../docs/_static/bench-data.json >> bench-data.js \
  && printf ';' >> bench-data.js
```

## Data contract

Renders the **real** canonical payload (`docs/_static/bench-data.json`,
semantics in `docs/dashboard_contract.md` / `docs/dashboard.schema.json`).
Nothing is mocked. Stable keys consumed: `columns[]`, `rows[]` (one per
`algo,n,p,threads` with `cells` keyed by column id), `method_scores{}`,
`stats{}`, `host`, `versions`, `algo_groups`, `algo_to_group`, `algo_origin`,
`algo_has_doc`, `algo_display`, `algo_fq`.

## Design rationale

- **Two questions, two visible answers.** The single most important fact this
  data carries is that *binding parity* (does each Python/R/MATLAB binding match
  the C++ core?) and *reference parity* (does the C++ core match the canonical
  external library?) are **different gates**. The old matrix smears them into
  one colored cell. Here every method card shows **both** as a labelled dual
  meter, and the drawer breaks each into its own verdict histogram and its own
  divergence basis (`reference` vs `binding`).
- **Colorblind-safe, never color-alone.** The verdict palette is Okabe–Ito-derived
  and every badge pairs **color + glyph + text** (`✓ exact`, `◐ cross-check`,
  `▲ divergent`, `∅ n/a`, `· NR`, `✕ error`). NR badges are also dashed. A
  legend is always on screen.
- **Brand-faithful.** Palette, radii, shadows, gradient headings, paper canvas,
  and pill/badge/card components are lifted verbatim from
  `datasets.nirs4all.org` / `nirs4all.org`, then adapted to a dense
  data-exploration layout.
- **Honest about absence.** `not_run` (NR) and `not_available` (n/a) are kept
  strictly distinct and are **never coerced into a divergence number**. The
  divergence column shows an explained em-dash (`—`) for those cells. The score
  meters compute percentages over *comparable* cells only, and still surface the
  NR count so missing coverage is visible, not hidden.
- **Metric-correct divergence.** Numeric rows show relative-RMSE Δ (tagged
  `ΔRMSE`); selector rows show Jaccard set-overlap (tagged `J`, `1.00` =
  identical feature mask). The two are never conflated, and the gate basis is
  printed alongside.

## Pain points of the old table that this solves

1. **CDN dependency / offline breakage.** The old single file pulls Google Fonts
   from `fonts.googleapis.com` (3 references). This proposal has **zero**
   network requests and is verified to open from `file://`.
2. **Binding vs reference parity were indistinguishable.** Now they are two
   labelled gates everywhere — card meters, drawer tiles, table columns,
   divergence basis.
3. **No search / no fast navigation.** A 183-method × 46-column grid is a wall.
   This adds **fuzzy search** (name **and** ABI/doc handle **and** category),
   category grouping with per-category accents and summaries, origin/parity
   filters, and four sort modes (incl. "needs attention" and "slowest").
   Categories are the 12 top-level `n4m.<role>` namespace roles (ABI 2.0):
   `transform`, `augmentation`, `estimators`, `feature_selection`,
   `model_selection`, `domain_adaptation`, `outlier_detection`, `ensemble`,
   `compose`, `metrics`, `decomposition`, `lowlevel`. The grouping is fully
   data-driven from the payload's `algo_groups` / `algo_to_group`, so it stays
   in lockstep with the catalog namespace.
4. **Timing was a number in a cell, not a trend.** The drawer renders
   **timing-vs-size curves** (hand-rolled SVG) with a linear/log toggle and
   hover tooltips, so the scaling behaviour of each implementation is legible.
5. **NR faked into / confused with zero or n/a.** Made explicit and never
   numeric.
6. **No provenance at a glance.** Host CPU/RAM/OS, generation time, and key
   library versions are surfaced in the hero strip and per-method drawer.
7. **Accessibility.** Full keyboard nav (`/` to search, `Esc` to close, Tab
   focus-trap in the drawer), ARIA roles/labels, visible focus rings, and
   sufficient contrast.
8. **Methods still wore their legacy registry prefixes** (`pp_`, `aug_`,
   `split_`, `filter_`, `*_select`). Now every card shows the **ABI-2 namespace
   identity**: the catalog leaf as the label (`pp_savgol` → `savitzky_golay`,
   `cars_select` → `cars`, `split_kennard_stone` → `kennard_stone`) and the
   fully-qualified `n4m.<role>…<leaf>` path as the handle
   (`n4m.transform.smoothing.savitzky_golay`). The mapping is **catalog-driven**
   (legacy symbol → `_rename_map.tsv` → ABI-2 symbol → catalog `c_surface`), not
   a string-strip, so abbreviations resolve correctly (179/183 carry a full
   `fq_name`; the rest display cleanly without one). Search still matches the
   legacy id. *(The underlying benchmark IDs / doc-page filenames intentionally
   keep their legacy names for parity-data lineage — nothing user-facing shows them.)*
9. **No links to the documentation.** Every card now has a **📄 docs ↗** link and
   the drawer an **Open documentation ↗** link to the published page
   (`https://methods.nirs4all.org/methods/<page>.html`, set via the `DOCS_BASE`
   constant in `app.js`). These are navigational links — the page still loads
   fully offline. The page content itself renders ABI-2 symbols (Phase-4 docs).

## Known gaps / future polish

- Drift-over-versions view (the contract's archived snapshots) is out of scope
  for this single-payload proposal.
- The timing chart picks one point per (column, size); for thread-swept methods
  it plots each thread count as separate points rather than a 3-D thread axis.
- Reduced-motion is honored; a full high-contrast theme toggle is not yet added.
