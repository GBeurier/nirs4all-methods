# n4m 1.0.21.9004

- `group_sparse_pls` now applies groupwise proximal shrinkage to the
  coefficients used for prediction. Earlier builds accepted `group_lambda`
  but left predictions unchanged.
- `fused_sparse_pls` consequently applies `l1_lambda` to predictive
  coefficients before its existing fusion smoothing. Predictions with a
  positive penalty can differ from 1.0.21.9003.
- These are documented post-SIMPLS approximations, not the `sgPLS::gPLS`
  estimator or a full fused-lasso optimization.
