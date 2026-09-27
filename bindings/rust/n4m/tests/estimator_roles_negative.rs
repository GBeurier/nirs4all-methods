//! Refusals of the generic estimator roles: the shared negative fixture
//! (written by the Python binding, also replayed by R and JS/WASM) gives the
//! same refusals and the same surviving state in Rust. Lengths and shapes are
//! checked natively; Rust slices and matrix views carry their own sizes.

use base64::{engine::general_purpose::STANDARD, Engine as _};
use n4m::roles::{self, Estimator, FitInputs, ParamValue, Params};
use n4m::{Context, MatrixRef};
use serde_json::Value;
use std::ffi::{c_char, c_void, CStr};
use std::ptr;

fn fixture() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../parity/fixtures/estimator_roles_negative.json"
    );
    serde_json::from_str(&std::fs::read_to_string(path).expect("fixture")).expect("fixture JSON")
}

fn floats(value: &Value) -> Vec<f64> {
    match value {
        Value::Array(items) => items.iter().flat_map(floats).collect(),
        other => vec![other.as_f64().expect("number")],
    }
}
fn ints(value: &Value) -> Vec<i64> {
    let list = value.as_array().expect("int list");
    list.iter().map(|v| v.as_i64().expect("int")).collect()
}
fn rows_cols(value: &Value) -> (usize, usize) {
    let rows = value.as_array().expect("matrix");
    (rows.len(), rows[0].as_array().map_or(1, Vec::len))
}

/// Raw ABI view (`n4m_matrix_view_t`): the safe API always allocates the
/// native width, so a narrower or wider caller view needs the raw call.
#[repr(C)]
struct RawView {
    data: *mut c_void,
    rows: i64,
    cols: i64,
    row_stride: i64,
    col_stride: i64,
    dtype: i32,
    reserved0: i32,
}
impl RawView {
    fn row_major(data: &mut [f64], rows: usize, cols: usize) -> Self {
        Self {
            data: data.as_mut_ptr().cast(),
            rows: rows as i64,
            cols: cols as i64,
            row_stride: cols.max(1) as i64,
            col_stride: 1,
            dtype: 1,
            reserved0: 0,
        }
    }
}
type Op = unsafe extern "C" fn(*mut c_void, *const c_void, *const RawView, *mut RawView) -> i32;
extern "C" {
    fn n4m_context_create(out: *mut *mut c_void) -> i32;
    fn n4m_context_destroy(ctx: *mut c_void);
    fn n4m_context_last_error(ctx: *const c_void) -> *const c_char;
    fn n4m_estimator_import_from_buffer(
        ctx: *mut c_void,
        buffer: *const c_void,
        size: usize,
        out: *mut *mut c_void,
    ) -> i32;
    fn n4m_estimator_destroy(est: *mut c_void);
    fn n4m_estimator_transform(
        ctx: *mut c_void,
        est: *const c_void,
        x: *const RawView,
        out: *mut RawView,
    ) -> i32;
    fn n4m_estimator_predict(
        ctx: *mut c_void,
        est: *const c_void,
        x: *const RawView,
        out: *mut RawView,
    ) -> i32;
}

/// A caller's output view of another width is refused before any write.
fn output_views(case: &Value, x_test: &[f64], p: usize) {
    let id = case["id"].as_str().unwrap();
    let payload = STANDARD
        .decode(case["n4me_base64"].as_str().unwrap())
        .unwrap();
    let ctx = Context::new().unwrap();
    let est = Estimator::from_n4me(&ctx, &payload).unwrap();
    let tcols = case["transform_cols"].as_u64().unwrap() as usize;
    let pcols = case["predict_cols"].as_u64().unwrap() as usize;
    assert_eq!(est.transform_cols().unwrap(), tcols, "{id}");
    let rows = x_test.len() / p;
    let mut x = x_test.to_vec();
    let xv = RawView::row_major(&mut x, rows, p);
    let ops: [(Op, usize); 3] = [
        (n4m_estimator_transform, tcols - 1),
        (n4m_estimator_transform, tcols + 1),
        (n4m_estimator_predict, pcols + 1),
    ];
    unsafe {
        let mut raw_ctx = ptr::null_mut();
        assert_eq!(n4m_context_create(&mut raw_ctx), 0);
        let mut raw = ptr::null_mut();
        assert_eq!(
            n4m_estimator_import_from_buffer(
                raw_ctx,
                payload.as_ptr().cast(),
                payload.len(),
                &mut raw
            ),
            0
        );
        for (op, cols) in ops {
            let mut out = vec![-7.0; rows * cols];
            let mut ov = RawView::row_major(&mut out, rows, cols);
            assert_eq!(op(raw_ctx, raw, &xv, &mut ov), 3, "{id}: SHAPE_MISMATCH");
            let message = CStr::from_ptr(n4m_context_last_error(raw_ctx)).to_string_lossy();
            assert!(
                message.contains(case["mentions"].as_str().unwrap()),
                "{id}: {message}"
            );
            assert!(out.iter().all(|&v| v == -7.0), "{id}: nothing written");
        }
        n4m_estimator_destroy(raw);
        n4m_context_destroy(raw_ctx);
    }
}

fn estimator(ctx: &Context, case: &Value) -> Estimator {
    let method_id = case["method_id"].as_str().unwrap();
    let mut params = Params::new(ctx, method_id).unwrap();
    if let Some(values) = case.get("params") {
        for (name, value) in values.as_object().unwrap() {
            params
                .set(name, &ParamValue::Int(value.as_i64().unwrap()))
                .unwrap();
        }
    }
    Estimator::new(ctx, method_id, Some(&params)).unwrap()
}

#[test]
fn replays_the_shared_negative_fixture() {
    let ctx = Context::new().unwrap();
    let fx = fixture();
    let (n, p) = rows_cols(&fx["x_train"]);
    let x_data = floats(&fx["x_train"]);
    let x = MatrixRef::row_major(&x_data, n, p).unwrap();
    let t_data = floats(&fx["x_test"]);
    let x_test = MatrixRef::row_major(&t_data, t_data.len() / p, p).unwrap();
    let cases = fx["cases"].as_array().unwrap();
    for case in cases {
        let id = case["id"].as_str().unwrap();
        let mentions = case["mentions"].as_str().unwrap();
        let refused = |err: Option<n4m::Error>| {
            let err = err.unwrap_or_else(|| panic!("{id}: accepted"));
            assert!(err.message.contains(mentions), "{id}: {err}");
        };
        match case["step"].as_str().unwrap() {
            "fit" => {
                let mut est = estimator(&ctx, case);
                let (y_data, y_shape) = match case.get("y_matrix") {
                    Some(m) => (floats(m), rows_cols(m)),
                    None => case
                        .get("y")
                        .map_or((Vec::new(), (0, 0)), |y| (floats(y), (floats(y).len(), 1))),
                };
                let labels = case.get("labels").map(ints);
                let weights = case.get("sample_weight").map(floats);
                let folds = case.get("fold_ids").map(ints);
                let mut inputs = FitInputs::new(x);
                if !y_data.is_empty() {
                    inputs = inputs.y(MatrixRef::row_major(&y_data, y_shape.0, y_shape.1).unwrap());
                }
                if let Some(labels) = &labels {
                    inputs = inputs.labels(labels);
                }
                if let Some(weights) = &weights {
                    inputs = inputs.sample_weight(weights);
                }
                if let Some(folds) = &folds {
                    inputs = inputs.fold_ids(folds);
                }
                refused(est.fit(&ctx, &inputs).err());
            }
            "import" => {
                let bytes = STANDARD
                    .decode(case["n4me_base64"].as_str().unwrap())
                    .unwrap();
                refused(Estimator::from_n4me(&ctx, &bytes).err());
                let control = STANDARD
                    .decode(case["control_base64"].as_str().unwrap())
                    .unwrap();
                let est = Estimator::from_n4me(&ctx, &control).unwrap();
                let pred = est.predict(&ctx, x_test).unwrap();
                for (a, e) in pred.data.iter().zip(floats(&case["control_predict"])) {
                    assert!((a - e).abs() <= 1e-12 * (1.0 + e.abs()), "{id}");
                }
            }
            "output_view" => output_views(case, &t_data, p),
            "export" => {
                let mut est = estimator(&ctx, case);
                let y = floats(&case["y"]);
                let inputs = FitInputs::new(x).y(MatrixRef::row_major(&y, n, 1).unwrap());
                est.fit(&ctx, &inputs).unwrap();
                assert_eq!(
                    est.contains_training_rows().unwrap(),
                    case["contains_training_rows"].as_bool().unwrap(),
                    "{id}"
                );
                refused(est.to_n4me(&ctx, false).err());
                assert_eq!(
                    est.to_n4me(&ctx, true).unwrap().len() as u64,
                    case["export_size_with_opt_in"].as_u64().unwrap(),
                    "{id}"
                );
            }
            step => {
                assert_eq!(step, "refit", "{id}");
                let mut est = estimator(&ctx, case);
                let labels = ints(&case["labels"]);
                est.fit(&ctx, &FitInputs::new(x).labels(&labels)).unwrap();
                let refit = ints(&case["refit_labels"]);
                refused(est.fit(&ctx, &FitInputs::new(x).labels(&refit)).err());
                assert!(est.is_fitted().unwrap(), "{id}");
                assert_eq!(est.classes().unwrap(), ints(&case["classes"]), "{id}");
                assert_eq!(
                    est.predict_labels(&ctx, x_test).unwrap(),
                    ints(&case["predict_labels"]),
                    "{id}"
                );
            }
        }
    }
    // Seeds are optional in the manifest; parameters the state records are flagged.
    let noise = roles::method_info("augmentation.noise.gaussian_noise").unwrap();
    let seed = noise.params.iter().find(|p| p.name == "seed").unwrap();
    assert!(!seed.required && seed.default.is_none() && !seed.recorded);
    let pls = roles::method_info("models.pls.pls_regression").unwrap();
    assert!(
        pls.params
            .iter()
            .find(|p| p.name == "n_components")
            .unwrap()
            .recorded
    );
}
