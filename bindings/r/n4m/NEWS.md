# n4m 1.0.21.9005

- Affine `MethodResult` regressors can be promoted by the C++ core to a
  predict-only, N4MM-serializable model. Sixteen verified methods use this
  common path in the Python and R estimator interfaces; the JavaScript/WASM
  binding can replay the native model.
- R formula estimators use native prediction for qualified affine results and
  retain N4MM bytes for `saveRDS()`/`readRDS()` across processes.
- R exposes a generic native preprocessing fit/transform interface for the
  fifteen operator kinds supported by the C pipeline. Its fitted state is
  process-local; no general trained-state interchange is claimed.

# n4m 1.0.21.9004

- `group_sparse_pls` now applies groupwise proximal shrinkage to the
  coefficients used for prediction. Earlier builds accepted `group_lambda`
  but left predictions unchanged.
- `fused_sparse_pls` consequently applies `l1_lambda` to predictive
  coefficients before its existing fusion smoothing. Predictions with a
  positive penalty can differ from 1.0.21.9003.
- These are documented post-SIMPLS approximations, not the `sgPLS::gPLS`
  estimator or a full fused-lasso optimization.
