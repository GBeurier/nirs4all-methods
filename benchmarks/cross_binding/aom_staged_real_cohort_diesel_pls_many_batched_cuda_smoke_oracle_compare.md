# AOM staged oracle comparison

- datasets with target: 1
- datasets in union: 61
- aom_pls_oracle: paired=1, target_wins=0, median_ratio=2.81636
- aom_ridge_oracle: paired=1, target_wins=0, median_ratio=3.1579
- tabpfn_oracle: paired=1, target_wins=0, median_ratio=2.0836

Winner counts:
- tabpfn_oracle: 30
- aom_ridge_oracle: 19
- aom_pls_oracle: 12

## Target Paired Rows

| Dataset | Target | AOM-PLS oracle | ratio | AOM-Ridge oracle | ratio | TabPFN oracle | ratio | Winner |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| DIESEL/DIESEL_bp50_246_b-a | 8.63519 | 3.06608 | 2.816 | 2.73447 | 3.158 | 4.14435 | 2.084 | aom_ridge_oracle |
