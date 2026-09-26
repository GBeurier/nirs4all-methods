# n4m 1.0.21.9006

- Added native dispatch for nine sample splitters and 22 seeded, train-only
  X-to-X augmentations. The augmentation contract preserves sample order and
  feature width; it does not transform targets or serialize fitted state.
- This development package carries libn4m ABI 2.11. The `pls4all` R subset is
  also versioned 1.0.21.9006 so R-universe does not reuse an ABI-2.10 package
  version for a newly built binary.

# n4m 1.0.21.9005

- Affine `MethodResult` regressors can be promoted by the C++ core to a
  predict-only, N4MM-serializable model. Sixteen verified methods use this
  common path in the Python and R estimator interfaces; the JavaScript/WASM
  binding can replay the native model.
- R formula estimators use native prediction for qualified affine results and
  retain N4MM bytes for `saveRDS()`/`readRDS()` across processes.
- R and Python expose the same generic native preprocessing pipeline for the
  fifteen supported operator kinds. Fitted linear-chain state can be exchanged
  as versioned N4MP bytes, with native plan inspection and strict decoding.
  Branching, variable selection and product-level trained pipeline archives
  are not implied by this low-level format.

# n4m 1.0.21.9004

- `group_sparse_pls` now applies groupwise proximal shrinkage to the
  coefficients used for prediction. Earlier builds accepted `group_lambda`
  but left predictions unchanged.
- `fused_sparse_pls` consequently applies `l1_lambda` to predictive
  coefficients before its existing fusion smoothing. Predictions with a
  positive penalty can differ from 1.0.21.9003.
- These are documented post-SIMPLS approximations, not the `sgPLS::gPLS`
  estimator or a full fused-lasso optimization.
