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
