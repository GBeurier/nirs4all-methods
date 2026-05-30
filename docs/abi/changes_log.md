# ABI — Changes Log

## 2026-05-18 — Linux export baseline for ABI 1.16.0

`build/dev-release/cpp/src/libn4m.so.1.16.0` exports 27 additional
`n4m_*` symbols compared with the previous Linux baseline. Each added symbol is
declared with `N4M_API` in the public header `cpp/include/pls4all/p4a.h`, so the
Linux ABI gate now treats them as intentional public additions:

- `n4m_method_result_get_int64_vector`
- `n4m_mb_pls_fit`, `n4m_lw_pls_fit`, `n4m_pls_lda_fit`,
  `n4m_pls_logistic_fit`, `n4m_aom_preprocess_fit`
- `n4m_variable_select_rank`, `n4m_interval_select`,
  `n4m_stability_select`, `n4m_uve_select`, `n4m_spa_select`,
  `n4m_cars_select`, `n4m_random_frog_select`, `n4m_scars_select`,
  `n4m_ga_select`, `n4m_shaving_select`, `n4m_bve_select`,
  `n4m_t2_select`, `n4m_wvc_select`, `n4m_wvc_threshold_select`,
  `n4m_emcuve_select`, `n4m_randomization_select`, `n4m_bipls_select`,
  `n4m_sipls_select`, `n4m_rep_select`, `n4m_ipw_select`, `n4m_st_select`

## 2026-05-30 — ABI 1.10.0: additive RNG-kind config selector

Two additive public symbols (ABI MINOR bump 1.9.0 → 1.10.0), backward-compatible
(no signature/layout change, nothing removed):

- `n4m_config_set_rng_kind(n4m_config_t*, n4m_rng_kind_t)`
- `n4m_config_get_rng_kind(const n4m_config_t*, n4m_rng_kind_t*)`

New enum `n4m_rng_kind_t` { `N4M_RNG_SPLITMIX64`=0 (default), `N4M_RNG_PCG64`=1,
`N4M_RNG_MT_R`=2, `N4M_RNG_NUMPY_MT`=3 } selects the RNG engine a stochastic
method draws from, so its output can match an external reference library's exact
RNG (numpy default_rng / base R / numpy RandomState) for parity. Default
SPLITMIX64 reproduces n4m's historical streams bit-for-bit — leaving it unset
changes nothing. Snapshots regenerated for all three platforms
(expected_symbols_{linux,macos,windows}.txt). Engines verified bit-exact:
docs/dev/RNG_TIER0_INVENTORY.md, cpp/tests/test_rng_engine.cpp.
