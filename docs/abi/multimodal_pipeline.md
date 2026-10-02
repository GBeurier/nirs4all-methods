# Complete raw multimodal pipeline (ABI 2.16)

`n4m_multimodal_pipeline_*` owns the complete learned early-fusion predictor.
Python exposes `n4m.MultimodalPipeline`; JavaScript/WASM exposes
`MultimodalPipeline`; R exposes `n4m_multimodal_pipeline`; Octave exposes
`n4m.MultimodalPipeline`. Rust exposes the same C ABI through its typed facade.

The closed profile takes an explicit ordered subset of one to four distinct
IO-aligned sources named `nir`, `image`, `series`, `metadata`. The historical
canonical profile selects all four in that order. Their representations are `signal_1d`,
`rgb_image`, `series_mv`, and `tabular_mixed`. Numeric arrays retain their raw
sample-first dimensions. Native code reshapes only the non-sample dimensions
for the learned encoders. Metadata remains a two-column raw table: declared
numeric column 0 is converted to float64, categorical column 1 is UTF-8 bytes
with one offset per row. Bindings never derive category codes or a vocabulary.

The native stages are population StandardScaler for NIR, learned PCA for image
and series, and a numeric population scaler plus dense one-hot encoding for
metadata. Category strings are sorted and learned from fit rows alone; unseen
categories give zero categorical features. Each encoded source is multiplied
by its nonnegative declared weight, then concatenated in source order. One
Ridge regressor follows with centered X/y and **unscaled X**. Alpha zero
uses the existing native compact SVD for minimum-norm least squares, including
rank-deficient one-hot and all-zero weighted designs. Positive alpha retains
the existing augmented-QR/dual solver paths. Categorical cells are length-
delimited UTF-8 and may contain U+0000; recipe/schema identifiers still refuse
NUL. The Octave/MATLAB binding delegates Unicode conversion to `unicode2native`
and preserves the complete returned byte count.

StandardScaler reuses the existing native population statistics. Translation
by an observed column value before mean/scale accumulation preserves constant
columns and limits loss from a large shared offset. A zero scale becomes one.
All numeric inputs and encoded outputs must be finite. PCA reuses the native
flexible-PCA fitted kernel and its N4ME state. Native SVD signs may differ from
sklearn: subspace/Gram comparisons and complete predictor outputs are the
parity contract. This profile does not promise bitwise equality with sklearn
randomized/auto solver choices on larger inputs. The recorded uint32 seed is
recipe identity; the existing deterministic native PCA kernel controls its
solver and does not consume that seed as a host algorithm.

`source_order` alone selects branches: encoder, weight and schema mappings have
exactly those names. Only selected sources are passed to the C ABI, fitted and
serialized. A selected source of weight zero still fits its encoder and keeps
its fitted state, preserving the historical weight semantics. Exclusion is
declared by absence. The surrounding SDK/DAG graph keeps the complete signed
four-source raw contract and projects selected schemas/views for Methods;
native Methods owns no graph, catalogue or row alignment.

Source schemas contain exactly `representation_id`, `input_shape`, `dtype`,
and `identity`. `identity` is the original canonical IO descriptor text
including axes/coordinates/features; every host carries it unchanged. Fit,
prediction and import check captured source order, raw shape, dtype and schema
identity. Python takes native NumPy arrays; JS takes `{data,shape,strides?}`
raw tensors; R and Octave take native column-major arrays. These layout
translations do not learn features. R's native array storage is float64;
float32 raw sources need a host with actual float32 storage.

Rank is 2–8 including the sample axis; positive fixed non-sample dimensions
have product at most 1,048,576. A numeric raw/encoded/fused matrix has at most
16,777,216 elements. PCA count is positive and does not exceed feature count
or actual fit-row count; native code refuses instead of silently clamping.
Seed is 0–UINT32_MAX. Each schema identity or categorical cell is at most 1MiB;
the learned vocabulary has at most 65,536 entries. The complete state has a
64MiB cap, further reduced by the native context state limit.

N4MF format 1 starts with `N4MF`, then little-endian u32 format version and
writer ABI major/minor/patch (20-byte prefix). The body stores the complete
normalized recipe and selected source schemas, one native fitted encoder state
per selected source,
training-only UTF-8 vocabulary and one native Ridge state. The final eight
bytes are an FNV-1a integrity checksum. Import requires independently expected
recipe and schemas, validates nested methods/parameters/widths/state integrity,
and publishes only a fully valid predictor. It never fits and contains no
training rows. Export/import buffers are caller owned; the appropriate native
destroy function releases the handle.

The existing recipe count/order and N4MF format 1 already encode this selection.
Canonical four-source payloads retain the same layout and bytes; import also
validates selected count, order, weights and schema identity before publishing.
The C structures and exported symbols are unchanged.

The canonical public recipe declaration is:

```json
{
  "schema_version": 1,
  "fusion": "early",
  "source_order": ["nir", "image", "series", "metadata"],
  "encoders": {
    "nir": {"kind": "standard_scaler", "with_mean": true, "with_std": true},
    "image": {"kind": "tensor_pca", "n_components": 2, "whiten": false, "random_state": 17},
    "series": {"kind": "tensor_pca", "n_components": 2, "whiten": false, "random_state": 17},
    "metadata": {
      "kind": "column_transformer", "numeric_columns": [0], "categorical_columns": [1],
      "with_mean": true, "with_std": true, "handle_unknown": "ignore",
      "sparse_output": false, "drop": null
    }
  },
  "source_weights": {"nir": 1.0, "image": 1.0, "series": 1.0, "metadata": 1.0},
  "model": {
    "method_id": "models.regularized.ridge",
    "params": {"alpha": 1.0, "center_x": true, "center_y": true, "scale_x": false}
  }
}
```

For already aligned raw `blocks` and their captured IO `source_schemas`, Python
uses `with n4m.MultimodalPipeline(recipe, source_schemas) as model`,
`model.fit(blocks, y)`, `model.predict(new_blocks, source_schemas=new_schemas)`,
and `state = model.export_state()`. A fresh process uses
`n4m.MultimodalPipeline.from_state(state, recipe=recipe,
source_schemas=source_schemas)` followed by prediction; targets are unnecessary.

C descriptors are copied by `create`; raw tensors and UTF-8 views are borrowed
only for the duration of synchronous fit/predict/transform calls. Successful
fit replaces state transactionally; failed fit leaves the previous predictor
usable. Handles and contexts are not shared concurrently across threads.

Octave builds only the required shims with
`build_mex({'n4m_multimodal_pipeline_mex','n4m_version_mex'})`, after setting
`N4M_INCLUDE_DIR`, `N4M_GENERATED_DIR` and `N4M_LIB_DIR` to matching ABI 2.16
headers, generated export header and library. The old RolePipeline surface
and its seven-field numeric archive profile are unchanged.
