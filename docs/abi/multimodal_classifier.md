# Raw multimodal classifier — ABI 2.17 (unreleased)

The additive classifier API in n4m/multimodal.h accepts the same ordered selected
raw sources and encoders as MultimodalPipeline. It fits encoders on supplied
training rows, fuses their weighted features and fits the registered
models.classification.pls_logistic native estimator. Encoders do not fit a
placeholder Ridge. The closed classifier profile has n_components and max_iter;
other classifier methods and parameters are refused.

The recipe struct fields are struct_size (u32), n_sources (i32), sources pointer,
method_id pointer and native params pointer. LP64 offsets are 0/4/8/16/24, size32;
wasm32 offsets are 0/4/8/12/16, size20. Source and input descriptors are unchanged.
Fit borrows one signed int64 label per aligned raw row. Native class IDs are
strictly sorted and unique; decision/probability columns follow that order.
Classes query supports a null buffer/count-only call. A failed fit/import never
publishes partial fitted state. Handles own their encoders/head until destroy.
Fit and prediction refuse a rows-by-classes product above16777216 before the
native head allocates its label/probability matrices. Python also checks native
matrix dimensions/product before allocating output arrays, including transform,
and refuses class tables outside2..65536 before allocating their host arrays.
Before fitting any encoder or classifier, the raw classification profile also
bounds the registered PLS-logistic working matrices using requested components P,
observed classes K and training rows N: N*K, N*(P+1), and
((K-1)*(P+1))^2 must each be at most16777216 elements. Products are checked by
division before multiplication. The dense Hessian and its solver copy therefore
each meet that per-matrix limit; this is not an aggregate memory limit. Requested
P conservatively bounds fitted components. Refusal leaves the previous fitted
owner intact, without fitting encoders/head. Generic preexisting Role APIs and
the numerical kernel are unchanged; DAG separately admits new meta-head fits.

Python exports MultimodalClassifierPipeline(recipe, source_schemas), with fit,
predict, decision_function, predict_proba, transform, classes_, label_names_,
export_state, from_state, close and context-manager ownership. Bindings only
marshal arrays/UTF-8 and map labels. Python fit accepts homogeneous strings or
signed int64 integers, records their sorted unique original values and sends
contiguous native codes. from_state without class_names exposes native IDs.
An explicit class_names table must be homogeneous, unique and match native class
column count. Its positions map sorted native IDs, including noncontiguous IDs;
native IDs are never used as external-table array indices. Archive owners must
sign/validate this external typed table. N4MC alone carries native IDs.

## N4MC format 1

All integers and IEEE f64 values below are little endian, without alignment.
A block is u64 byte length followed by those bytes; a string is a UTF-8 block.

| Offset | Value |
| --- | --- |
| 0 | ASCII N4MC |
| 4 | u32 format version 1 |
| 8 | u32 writer ABI major 2 |
| 12 | u32 writer ABI minor 17 |
| 16 | u32 writer ABI patch 0 |
| 20 | block canonical expected classifier recipe |

Canonical recipe, in order:

1. string models.classification.pls_logistic; u64 n_components; u64 max_iter;
   u32 selected source count.
2. For each ordered source: strings name, representation_id, dtype, identity;
   u32 number of nonsample dimensions; u64 dimensions; u32 encoder; f64 weight;
   u64 n_components, random_state, numeric_column, categorical_column;
   u32 with_mean, with_std, whiten, ignore_unknown.

Column sentinel -1 uses two's-complement u64. Source fields are validated and
serialized exactly; there is no default inference at import. Standard-scaler
fields are mean=std=1, PCA fields are n_components>0/random_state>=0/whiten=0;
irrelevant fields are zero except numeric/categorical columns -1. Mixed fields
are numeric_column=0, categorical_column=1, mean=std=ignore_unknown=1, others0.

The body after the recipe contains one block encoder-N4ME followed by u64
category count and that many strings for each source. Mixed categories are
strictly sorted unique UTF-8; other sources have zero categories. Then u64 class
count, that many signed int64 class IDs, block classifier-N4ME, and a final u64
FNV1a64 checksum of every preceding byte (offset basis cbf29ce484222325,
prime100000001b3).

Encoder N4ME methods are preprocessing.scaling.standard_scale and
preprocessing.feature_selection.flexible_pca; mixed numeric scaling uses the
first method. Import checks method, resolved params, input/output widths, no
embedded training rows and exact canonical expected recipe bytes. The head must
be PLS-logistic with resolved matching n_components/max_iter, input width equal
to the fused encoder width, output width equal to class count, capabilities156
(labels16/decision8/proba4/serializable128), no training-row retention bit512,
and native head classes exactly equal to the outer class table.

The existing head StateBlock tag is 0x31534c43 (CLS1). Its bytes begin int64 input
width, u64 class count and sorted signed int64 classes; the remaining existing
PLS-logistic fitted kernel state is validated by the native N4ME importer.
Classifier import rejects N4MF, checksum/length/trailing-byte errors, unsorted or
duplicate classes and mismatched nested states. Budgets are at most min(context
state budget,64MiB), with at most65536 categories/classes. N4MF regression format
and historical fixtures remain intact; new exports stamp current writer ABI.
