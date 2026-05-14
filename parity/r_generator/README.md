# R parity generator

Phase 0 ships only a stub. The full R-side generator (mirroring the Python
producers, using `pls`, `ropls`, `mixOmics`) lands in PR-1 alongside the
Phase 1 NIPALS implementation.

Planned layout once filled in:

```
r_generator/
├── DESCRIPTION
├── renv.lock
├── generate_fixtures.R
└── adapters/
    ├── pls_adapter.R         # pls::plsr (kernelpls + simpls)
    ├── ropls_adapter.R       # ropls::opls
    ├── mixOmics_adapter.R    # mixOmics::pls (regression + canonical)
    └── plsVarSel_adapter.R   # (deferred to Phase 5)
```

The R generator produces the **same three canonical fixtures** as the Python
generator (`synthetic_small_pls1_v1`, `synthetic_small_pls2_v1`,
`synthetic_tiny_centered_v1`) but written under `r_generator/fixtures_r/`
so the two pipelines can be diffed before the comparator merges them into
the shared `parity/fixtures/` directory.
