# `multibranch_regressor` — n4m.estimators.neural.multibranch_regressor

_Namespace_: **`n4m.estimators.neural`** · _Fully-qualified_: `n4m.estimators.neural.multibranch_regressor` · _Catalog id_: `models.neural.multibranch_regressor`

## API surface

**C ABI:** no standalone exported symbol is declared for this method.

**Python (verified public re-export):** `from n4m.roles import MultibranchRegressor`

**Signature:** [`MultibranchRegressor(*, branch_kinds = (), branch_shapes = (), filters = 4, kernel_size = 5, pool_size = 2, embedding_dim = 8, head_hidden = 8, epochs = 100, batch_size = 16, learning_rate = 0.01, weight_decay = 0.0001, seed = None)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/roles/_generated.py#L2752)

**R (source-verified):** [`n4m_multibranch_regressor(branch_kinds = c(), branch_shapes = c(), filters = 4L, kernel_size = 5L, pool_size = 2L, embedding_dim = 8L, head_hidden = 8L, epochs = 100L, batch_size = 16L, learning_rate = 0.01, weight_decay = 0.0001, seed = NULL)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/r/n4m/R/estimator_roles_generated.R).

```r
library(n4m)
result <- n4m_multibranch_regressor()
```

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

### Parameters

| Name | Type | Default |
|---|---|---|
| `branch_kinds` | `—` | `()` |
| `branch_shapes` | `—` | `()` |
| `filters` | `—` | `4` |
| `kernel_size` | `—` | `5` |
| `pool_size` | `—` | `2` |
| `embedding_dim` | `—` | `8` |
| `head_hidden` | `—` | `8` |
| `epochs` | `—` | `100` |
| `batch_size` | `—` | `16` |
| `learning_rate` | `—` | `0.01` |
| `weight_decay` | `—` | `0.0001` |
| `seed` | `—` | `None` |

## Explanations

### Bibliographic source

Implementation-specific compact architecture; no single canonical paper is asserted. Its components follow standard references: LeCun, Bottou, Bengio & Haffner (1998), *Gradient-based learning applied to document recognition*, Proceedings of the IEEE 86, 2278-2324, https://doi.org/10.1109/5.726791 (convolution and backpropagation); Kingma & Ba (2015), *Adam: a method for stochastic optimization*, https://arxiv.org/abs/1412.6980; Loshchilov & Hutter (2019), *Decoupled weight decay regularization*, https://arxiv.org/abs/1711.05101. No deep-learning framework parity is claimed.

### Mathematical principle

X concatenates ordered blocks; each block feeds one branch. A dense branch maps its standardized block to an embedding e_b = act(W_b x_b + c_b). A conv1d branch applies a valid convolution with F filters of length K, a leaky ReLU act (slope 0.01), non-overlapping average pooling of width P and a dense embedding. A conv2d branch does the same on an H x W x C block with F kernels C x K x K and P x P pooling. Embeddings are concatenated in block order and a common head (optional hidden layer with act, then a linear output) predicts the standardized targets. All kernels, embeddings and head weights are trained jointly by backpropagation of the mean squared error with Adam and decoupled weight decay for a fixed number of epochs; the model after the last epoch is returned.

### Appropriate uses

Teaching and small-sample studies of intermediate (mid-level) fusion of several measurements of the same samples, for example spectra plus marker dosages or spectra plus small images, and per-source CNN baselines with a single branch.

### Limits and validation

Small fixed architecture without early stopping or internal validation: epochs and every other hyperparameter must be fixed a priori or chosen inside a training scope, never on a retained test set. Networks can overfit or underperform linear models on small or linear data. Marker order is not a physical convolution axis, so markers should use a dense branch. Fit refuses NaN or infinity (no imputation). Images are not resized; the declared H x W x C must match the block. Bounded to 2,000,000 parameters; CPU only, no pretrained backbone. Results are deterministic for a build without floating-point contraction; other builds agree to rounding.

### Implementation

The native models.neural.multibranch_regressor estimator (regressor and transformer) runs cpp/src/core/neural/multibranch_network.cpp through the adapter in cpp/src/core/estimator/neural_adapters.cpp and the shared n4m_estimator_* lifecycle; strided inputs are copied to row-major. branch_kinds (0 dense, 1 conv1d, 2 conv2d) and branch_shapes (3 integers per branch) declare the branches against the optional blocks fit input. Standardization is fitted on training rows only (dense per column, conv1d per block, conv2d per channel, y per target). PCG64 seeds He/Glorot initialization and the per-epoch minibatch permutation. transform returns the learned embeddings; the fit result holds loss_history. The MBN1 N4ME block stores architecture, scaling and weights so import predicts without refitting. ABI 2.22.0 is unchanged.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/neural/multibranch_network.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/estimator/neural_adapters.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/catalog/methods/models.neural.multibranch_regressor.yaml; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/tests/test_multibranch_regressor.cpp

## Catalog note

Jointly trained multibranch neural regressor for intermediate (mid-level) multisource fusion through the shared n4m_estimator_* role API, ABI 2.22.0 unchanged. X is the concatenation of ordered blocks (optional blocks fit input; absent = one block). Each block feeds one branch declared by branch_kinds (0 dense, 1 conv1d, 2 conv2d; empty = all dense) and branch_shapes (3 ints per branch: dense/conv1d [width,1,1], conv2d HWC [H,W,C]). Branches: dense layer, or valid convolution + leaky ReLU + average pooling + dense embedding; embeddings are concatenated and a common head (optional hidden layer) regresses standardized y. All kernels, embeddings and head weights are trained jointly by backpropagation of MSE with Adam and decoupled weight decay, PCG64-seeded He/Glorot initialization and per-epoch Fisher-Yates minibatch order, for a fixed number of epochs (no early stopping, no validation split). Input/target standardization is fitted on training rows only (dense per column, conv1d per block, conv2d per channel). Transform returns the learned embeddings; the fit result holds loss_history. Fit refuses NaN/Inf (complete-data profile). No GPU, pretrained backbone or deep-learning framework parity claim.


_See also_: [methods index](index.md).