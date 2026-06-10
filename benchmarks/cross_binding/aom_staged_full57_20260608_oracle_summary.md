# AOM staged oracle comparison

- datasets with target: 51
- datasets in union: 61
- aom_pls_oracle: paired=49, target_wins=15, median_ratio=1.03725
- aom_ridge_oracle: paired=51, target_wins=5, median_ratio=1.09688
- tabpfn_oracle: paired=51, target_wins=15, median_ratio=1.10882

Winner counts:
- tabpfn_oracle: 30
- aom_ridge_oracle: 19
- aom_pls_oracle: 11
- target: 1

## Target Paired Rows

| Dataset | Target | AOM-PLS oracle | ratio | AOM-Ridge oracle | ratio | TabPFN oracle | ratio | Winner |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| ALPINE/ALPINE_P_291_KS | 0.0586343 | 0.0598845 | 0.9791 | 0.0567969 | 1.032 | 0.0434172 | 1.35 | tabpfn_oracle |
| AMYLOSE/Rice_Amylose_313_YbasedSplit | 2.59539 | 1.88725 | 1.375 | 1.90252 | 1.364 | 1.63247 | 1.59 | tabpfn_oracle |
| BEEFMARBLING/Beef_Marbling_RandomSplit | 73.4547 | 72.5384 | 1.013 | 71.2688 | 1.031 | 61.6615 | 1.191 | tabpfn_oracle |
| BEER/Beer_OriginalExtract_60_KS | 0.233411 | 0.204183 | 1.143 | 0.155666 | 1.499 | 0.129906 | 1.797 | tabpfn_oracle |
| BEER/Beer_OriginalExtract_60_YbaseSplit | 0.252663 | 0.210712 | 1.199 | 0.241646 | 1.046 | 0.151806 | 1.664 | tabpfn_oracle |
| BERRY/brix_groupSampleID_stratDateVar_balRows | 4.02244 | 4.1629 | 0.9663 | 3.59104 | 1.12 | 2.85317 | 1.41 | tabpfn_oracle |
| BERRY/ph_groupSampleID_stratDateVar_balRows | 0.331077 | 0.338459 | 0.9782 | 0.2979 | 1.111 | 0.23114 | 1.432 | tabpfn_oracle |
| BERRY/ta_groupSampleID_stratDateVar_balRows | 1.88989 | 1.87331 | 1.009 | 1.78459 | 1.059 | 1.52325 | 1.241 | tabpfn_oracle |
| BISCUIT/Biscuit_Fat_40_RandomSplit | 0.399271 | 0.27403 | 1.457 | 0.200384 | 1.993 | 0.402531 | 0.9919 | aom_ridge_oracle |
| BISCUIT/Biscuit_Sucrose_40_RandomSplit | 1.28201 | 0.881698 | 1.454 | 1.05329 | 1.217 | 0.95636 | 1.341 | aom_pls_oracle |
| COLZA/C_woOutlier | 1.91477 | 2.32753 | 0.8227 | 1.6833 | 1.138 | 0.793406 | 2.413 | tabpfn_oracle |
| COLZA/N_wOutlier | 0.350366 | 0.342932 | 1.022 | 0.268476 | 1.305 | 0.175339 | 1.998 | tabpfn_oracle |
| COLZA/N_woOutlier | 0.28606 | 0.31343 | 0.9127 | 0.220739 | 1.296 | 0.177459 | 1.612 | tabpfn_oracle |
| CORN/Corn_Oil_80_ZhengChenPelegYbaseSplit | 0.019417 | 0.0221768 | 0.8756 | 0.0174273 | 1.114 | 0.0247324 | 0.7851 | aom_ridge_oracle |
| CORN/Corn_Starch_80_ZhengChenPelegYbaseSplit | 0.084921 | 0.138825 | 0.6117 | 0.109062 | 0.7786 | 0.0640284 | 1.326 | tabpfn_oracle |
| DIESEL/DIESEL_bp50_246_b-a | 3.37902 | 3.06608 | 1.102 | 2.73447 | 1.236 | 4.14435 | 0.8153 | aom_ridge_oracle |
| DIESEL/DIESEL_bp50_246_hla-b | 2.81322 | 2.72992 | 1.031 | 2.49228 | 1.129 | 4.20467 | 0.6691 | aom_ridge_oracle |
| DIESEL/DIESEL_bp50_246_hlb-a | 3.44863 | 2.95566 | 1.167 | 2.69623 | 1.279 | 2.94169 | 1.172 | aom_ridge_oracle |
| DarkResp/Rd25_CBtestSite | 0.222769 | 0.222655 | 1.001 | 0.22545 | 0.9881 | 0.276801 | 0.8048 | aom_pls_oracle |
| DarkResp/Rd25_GTtestSite | 0.190349 | 0.191829 | 0.9923 | 0.192192 | 0.9904 | 0.283656 | 0.6711 | target |
| DarkResp/Rd25_XSBNtestSite | 0.25199 | 0.263662 | 0.9557 | 0.238417 | 1.057 | 0.255539 | 0.9861 | aom_ridge_oracle |
| DarkResp/Rd25_spxy70 | 0.17668 | 0.178122 | 0.9919 | 0.167088 | 1.057 | 0.166234 | 1.063 | tabpfn_oracle |
| ECOSIS_LeafTraits/Ccar_spxyG_block2deg | 12.4524 | 39.1586 | 0.318 | 11.5827 | 1.075 | 30.4005 | 0.4096 | aom_ridge_oracle |
| ECOSIS_LeafTraits/Chla+b_spxyG_block2deg | 41.3984 |  |  | 14.0404 | 2.949 | 59.898 | 0.6911 | aom_ridge_oracle |
| ECOSIS_LeafTraits/Chla+b_spxyG_species | 33.3124 |  |  | 15.6447 | 2.129 | 49.0026 | 0.6798 | aom_ridge_oracle |
| FUSARIUM/Fv_Fm_grp70_30 | 0.0303829 | 0.029413 | 1.033 | 0.0286809 | 1.059 | 0.0255441 | 1.189 | tabpfn_oracle |
| GRAPEVINES/grapevine_chloride_556_KS | 941.754 | 946.433 | 0.9951 | 933.641 | 1.009 | 771.275 | 1.221 | tabpfn_oracle |
| GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_ASD | 4.16122 | 3.42571 | 1.215 | 3.24311 | 1.283 | 3.41959 | 1.217 | aom_ridge_oracle |
| GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_MicroNIR | 3.5307 | 3.42524 | 1.031 | 3.42431 | 1.031 | 3.73477 | 0.9454 | aom_ridge_oracle |
| GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_MicroNIR_NeoSpectra | 4.0948 | 3.49178 | 1.173 | 3.78182 | 1.083 | 3.62373 | 1.13 | aom_pls_oracle |
| GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_NeoSpectra | 4.8506 | 4.34707 | 1.116 | 4.4067 | 1.101 | 4.37457 | 1.109 | aom_pls_oracle |
| GRAPEVINE_LeafTraits/LMA_spxyG70_30_byCultivar_ASD | 0.302122 | 0.303214 | 0.9964 | 0.301787 | 1.001 | 0.302785 | 0.9978 | aom_ridge_oracle |
| GRAPEVINE_LeafTraits/WUEinst_spxyG70_30_byCultivar_MicroNIR_NeoSpectra | 1.39471 | 1.3114 | 1.064 | 1.43453 | 0.9722 | 1.48328 | 0.9403 | aom_pls_oracle |
| IncombustibleMaterial/TIC_spxy70 | 4.31338 | 2.95006 | 1.462 | 3.32584 | 1.297 | 2.96182 | 1.456 | aom_pls_oracle |
| MANURE21/All_manure_CaO_SPXY_strat_Manure_type | 7.75806 | 7.23552 | 1.072 | 6.77427 | 1.145 | 5.56168 | 1.395 | tabpfn_oracle |
| MANURE21/All_manure_K2O_SPXY_strat_Manure_type | 2.62492 | 2.29076 | 1.146 | 2.13821 | 1.228 | 2.65835 | 0.9874 | aom_ridge_oracle |
| MANURE21/All_manure_MgO_SPXY_strat_Manure_type | 0.786902 | 0.716858 | 1.098 | 0.73762 | 1.067 | 0.750899 | 1.048 | aom_pls_oracle |
| MANURE21/All_manure_P2O5_SPXY_strat_Manure_type | 2.54165 | 2.33476 | 1.089 | 2.46144 | 1.033 | 2.2951 | 1.107 | tabpfn_oracle |
| MANURE21/All_manure_Total_N_SPXY_strat_Manure_type | 1.71462 | 1.54858 | 1.107 | 1.53293 | 1.119 | 1.59419 | 1.076 | aom_ridge_oracle |
| MILK/Milk_Fat_1224_KS | 0.096817 | 0.0926869 | 1.045 | 0.0816594 | 1.186 | 0.0804811 | 1.203 | tabpfn_oracle |
| MILK/Milk_Lactose_1224_KS | 0.0579605 | 0.0568685 | 1.019 | 0.0528411 | 1.097 | 0.04768 | 1.216 | tabpfn_oracle |
| MILK/Milk_Urea_1224_KS | 4.10359 | 4.18171 | 0.9813 | 4.09869 | 1.001 | 3.92768 | 1.045 | tabpfn_oracle |
| PHOSPHORUS/LP_spxyG | 0.168762 | 0.148218 | 1.139 | 0.16709 | 1.01 | 0.165718 | 1.018 | aom_pls_oracle |
| PHOSPHORUS/MP_spxyG | 0.0214841 | 0.0190536 | 1.128 | 0.0198951 | 1.08 | 0.0199866 | 1.075 | aom_pls_oracle |
| PHOSPHORUS/NP_spxyG | 0.107394 | 0.101966 | 1.053 | 0.100044 | 1.073 | 0.115738 | 0.9279 | aom_ridge_oracle |
| PHOSPHORUS/Pi_spxyG | 0.16801 | 0.146759 | 1.145 | 0.172448 | 0.9743 | 0.157163 | 1.069 | aom_pls_oracle |
| PHOSPHORUS/V25_spxyG | 0.294737 | 0.252191 | 1.169 | 0.238548 | 1.236 | 0.263201 | 1.12 | aom_ridge_oracle |
| PLUMS/Firmness_spxy70 | 0.35174 | 0.264871 | 1.328 | 0.230195 | 1.528 | 0.257266 | 1.367 | aom_ridge_oracle |
| TABLET/Escitalopramt_310_Zhao | 0.375384 | 0.372232 | 1.008 | 0.336737 | 1.115 | 0.328421 | 1.143 | tabpfn_oracle |
| WOOD_density/WOOD_Density_402_Olale | 0.133492 | 0.128697 | 1.037 | 0.127735 | 1.045 | 0.121667 | 1.097 | tabpfn_oracle |
| WOOD_density/WOOD_N_402_Olale | 0.0478478 | 0.0494642 | 0.9673 | 0.0472564 | 1.013 | 0.0472946 | 1.012 | aom_ridge_oracle |
