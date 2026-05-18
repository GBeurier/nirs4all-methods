# ABI — Changes Log

## 2026-05-18 — Linux export baseline for ABI 1.16.0

`build/dev-release/cpp/src/libp4a.so.1.16.0` exports 27 additional
`p4a_*` symbols compared with the previous Linux baseline. Each added symbol is
declared with `P4A_API` in the public header `cpp/include/pls4all/p4a.h`, so the
Linux ABI gate now treats them as intentional public additions:

- `p4a_method_result_get_int64_vector`
- `p4a_mb_pls_fit`, `p4a_lw_pls_fit`, `p4a_pls_lda_fit`,
  `p4a_pls_logistic_fit`, `p4a_aom_preprocess_fit`
- `p4a_variable_select_rank`, `p4a_interval_select`,
  `p4a_stability_select`, `p4a_uve_select`, `p4a_spa_select`,
  `p4a_cars_select`, `p4a_random_frog_select`, `p4a_scars_select`,
  `p4a_ga_select`, `p4a_shaving_select`, `p4a_bve_select`,
  `p4a_t2_select`, `p4a_wvc_select`, `p4a_wvc_threshold_select`,
  `p4a_emcuve_select`, `p4a_randomization_select`, `p4a_bipls_select`,
  `p4a_sipls_select`, `p4a_rep_select`, `p4a_ipw_select`, `p4a_st_select`
