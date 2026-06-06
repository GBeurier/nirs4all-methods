# AOM staged oracle comparison

- datasets with target: 8
- datasets in union: 61
- aom_pls_oracle: paired=7, target_wins=1, median_ratio=1.03079
- aom_ridge_oracle: paired=8, target_wins=0, median_ratio=1.08068
- tabpfn_oracle: paired=8, target_wins=4, median_ratio=0.978527

Winner counts:
- tabpfn_oracle: 30
- aom_ridge_oracle: 19
- aom_pls_oracle: 12

## Target Paired Rows

| Dataset | Target | AOM-PLS oracle | ratio | AOM-Ridge oracle | ratio | TabPFN oracle | ratio | Winner |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| BEEFMARBLING/Beef_Marbling_RandomSplit | 73.4547 | 72.5384 | 1.013 | 71.2688 | 1.031 | 61.6615 | 1.191 | tabpfn_oracle |
| DIESEL/DIESEL_bp50_246_b-a | 3.37902 | 3.06608 | 1.102 | 2.73447 | 1.236 | 4.14435 | 0.8153 | aom_ridge_oracle |
| DIESEL/DIESEL_bp50_246_hla-b | 2.81322 | 2.72992 | 1.031 | 2.49228 | 1.129 | 4.20467 | 0.6691 | aom_ridge_oracle |
| ECOSIS_LeafTraits/Chla+b_spxyG_block2deg | 41.3984 |  |  | 14.0404 | 2.949 | 59.898 | 0.6911 | aom_ridge_oracle |
| GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_MicroNIR | 3.5307 | 3.42524 | 1.031 | 3.42431 | 1.031 | 3.73477 | 0.9454 | aom_ridge_oracle |
| MANURE21/All_manure_CaO_SPXY_strat_Manure_type | 7.75806 | 7.23552 | 1.072 | 6.77427 | 1.145 | 5.56168 | 1.395 | tabpfn_oracle |
| MANURE21/All_manure_P2O5_SPXY_strat_Manure_type | 2.54165 | 2.33476 | 1.089 | 2.46144 | 1.033 | 2.2951 | 1.107 | tabpfn_oracle |
| WOOD_density/WOOD_N_402_Olale | 0.0478478 | 0.0494642 | 0.9673 | 0.0472564 | 1.013 | 0.0472946 | 1.012 | aom_ridge_oracle |
