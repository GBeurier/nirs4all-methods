//! Generic estimator roles (ABI 2.13): typed manifest, one life cycle per
//! role, and the shared N4ME fixture written by the Python binding (also
//! replayed by R and JS/WASM) predicting identically in Rust.

use base64::{engine::general_purpose::STANDARD, Engine as _};
use n4m::roles::{
    self, Estimator, FitInput, FitInputs, InputRequirement, MethodKind, ParamType, ParamValue,
    Params, ResultEntryKind,
};
use n4m::{Context, ErrorKind, MatrixRef};
use serde_json::Value;
use std::collections::BTreeSet;

static_assertions::assert_not_impl_any!(Estimator: Send, Sync);
static_assertions::assert_not_impl_any!(Params: Send, Sync);
static_assertions::assert_not_impl_any!(roles::MethodResult: Send, Sync);

fn fixture() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../parity/fixtures/estimator_roles_n4me.json"
    );
    serde_json::from_str(&std::fs::read_to_string(path).expect("fixture")).expect("fixture JSON")
}

/// Row-major owned matrix from nested JSON rows, or a column from a flat list.
struct Dense {
    data: Vec<f64>,
    rows: usize,
    cols: usize,
}
impl Dense {
    fn new(value: &Value) -> Self {
        let rows = value.as_array().expect("matrix");
        match rows[0].as_array() {
            Some(first) => Self {
                data: rows.iter().flat_map(floats).collect(),
                rows: rows.len(),
                cols: first.len(),
            },
            None => Self {
                data: floats(value),
                rows: rows.len(),
                cols: 1,
            },
        }
    }
    fn view(&self) -> MatrixRef<'_> {
        MatrixRef::row_major(&self.data, self.rows, self.cols).unwrap()
    }
}
fn floats(value: &Value) -> Vec<f64> {
    match value {
        Value::Array(items) => items.iter().flat_map(floats).collect(),
        other => vec![other.as_f64().expect("number")],
    }
}
fn ints(value: &Value) -> Vec<i64> {
    value
        .as_array()
        .expect("int list")
        .iter()
        .map(|v| v.as_i64().expect("int"))
        .collect()
}
/// Fixture outputs were computed on Linux x86-64; iterative kernels drift by a
/// few ulps elsewhere (FMA contraction), so replays compare at 1e-9.
const REPLAY_TOL: f64 = 1e-9;

fn close(actual: &[f64], expected: &[f64], tol: f64, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tol * (1.0 + e.abs()),
            "{label}[{i}]: {a} vs {e}"
        );
    }
}

/// Parameters from fixture JSON, typed by the native manifest.
fn params_for(ctx: &Context, method_id: &str, values: &Value) -> Params {
    let info = roles::method_info(method_id).unwrap();
    let mut params = Params::new(ctx, method_id).unwrap();
    for (name, value) in values.as_object().expect("params") {
        let declared = info.params.iter().find(|p| &p.name == name).unwrap();
        let typed = match declared.param_type {
            ParamType::Int => ParamValue::Int(value.as_i64().unwrap()),
            ParamType::Double => ParamValue::Double(value.as_f64().unwrap()),
            ParamType::Bool => ParamValue::Bool(value.as_bool().unwrap()),
            ParamType::Enum => ParamValue::Enum(value.as_str().unwrap().to_owned()),
            ParamType::IntArray => ParamValue::IntArray(ints(value)),
            ParamType::DoubleArray => ParamValue::DoubleArray(floats(value)),
        };
        params.set(name, &typed).unwrap();
    }
    params
}

/// Optional fit inputs of the fixture, added by their manifest names.
struct ExtraInputs {
    feature_groups: Vec<i64>,
    blocks: Vec<i64>,
    axis: Vec<f64>,
    groups: Vec<i64>,
    x_target: Dense,
    y_train: Dense,
}
impl ExtraInputs {
    fn new(fx: &Value) -> Self {
        Self {
            feature_groups: ints(&fx["feature_groups"]),
            blocks: ints(&fx["blocks"]),
            axis: floats(&fx["axis"]),
            groups: ints(&fx["groups"]),
            x_target: Dense::new(&fx["x_target"]),
            y_train: Dense::new(&fx["y_train"]),
        }
    }
    fn add<'a>(&'a self, mut inputs: FitInputs<'a>, names: &Value) -> FitInputs<'a> {
        for name in names.as_array().unwrap() {
            inputs = match name.as_str().unwrap() {
                "feature_groups" => inputs.feature_groups(&self.feature_groups),
                "blocks" => inputs.blocks(&self.blocks),
                "axis" => inputs.axis(&self.axis),
                "X_target" => inputs.x_target(self.x_target.view()),
                "y" => inputs.y(self.y_train.view()),
                "groups" => inputs.groups(&self.groups),
                other => panic!("fixture input {other}"),
            };
        }
        inputs
    }
}

fn regression_data() -> (Vec<f64>, Vec<f64>) {
    let (n, p) = (24, 8);
    let x: Vec<f64> = (0..n * p)
        .map(|k| {
            let (i, j) = ((k / p) as f64, (k % p) as f64);
            (0.37 * i + 0.11 * j).sin() + 0.05 * (1.3 * i * j).cos() + 1.0
        })
        .collect();
    let y = (0..n)
        .map(|i| x[i * p] - 0.5 * x[i * p + 3] + 0.25 * x[i * p + 6])
        .collect();
    (x, y)
}

#[test]
fn manifest_is_typed_and_matches_json() {
    let json = roles::manifest_json().unwrap();
    assert!(json.starts_with("{\"abi\":\"2.14"), "{}", &json[..40]);
    let methods = roles::methods().unwrap();
    for method in &methods {
        assert!(json.contains(&format!("\"method_id\":\"{}\"", method.method_id)));
    }
    let cppls = roles::method_info("models.pls.cppls").unwrap();
    assert_eq!(cppls.kind, MethodKind::Estimator);
    assert_ne!(cppls.roles & roles::ROLE_REGRESSOR, 0);
    assert_eq!(cppls.input(FitInput::Y), InputRequirement::Required);
    assert_eq!(cppls.input(FitInput::Groups), InputRequirement::None);
    let n_components = cppls
        .params
        .iter()
        .find(|p| p.name == "n_components")
        .unwrap();
    assert_eq!(n_components.param_type, ParamType::Int);
    assert!(matches!(n_components.default, Some(ParamValue::Int(_))));

    let pls = roles::method_info("models.pls.pls_regression").unwrap();
    let solver = pls.params.iter().find(|p| p.name == "solver").unwrap();
    assert_eq!(solver.param_type, ParamType::Enum);
    match &solver.default {
        Some(ParamValue::Enum(label)) => assert!(solver.choices.contains(label)),
        other => panic!("solver default {other:?}"),
    }
    let ks = roles::method_info("splitters.kennard_stone").unwrap();
    assert_eq!(
        (ks.kind, ks.roles),
        (MethodKind::Procedure, roles::ROLE_SPLITTER)
    );

    let err = roles::method_info("models.pls.missing").unwrap_err();
    assert_eq!(err.kind, ErrorKind::InvalidArgument);
    assert!(err.message.contains("models.pls.missing"), "{err}");
}

#[test]
fn regressor_fit_predict_and_n4me_round_trip() {
    let ctx = Context::new().unwrap();
    let (x, y) = regression_data();
    let (xv, yv) = (
        MatrixRef::row_major(&x, 24, 8).unwrap(),
        MatrixRef::row_major(&y, 24, 1).unwrap(),
    );
    let mut params = Params::new(&ctx, "models.pls.cppls").unwrap();
    params.set_int("n_components", 3).unwrap();
    let mut est = Estimator::new(&ctx, "models.pls.cppls", Some(&params)).unwrap();
    assert!(!est.is_fitted().unwrap());
    assert_eq!(
        est.predict(&ctx, xv).unwrap_err().kind,
        ErrorKind::NotFitted
    );
    est.fit(&ctx, &FitInputs::new(xv).y(yv)).unwrap();
    assert!(est.is_fitted().unwrap());
    assert_eq!(est.method_id().unwrap(), "models.pls.cppls");
    assert_eq!(
        (est.n_features_in().unwrap(), est.n_outputs().unwrap()),
        (8, 1)
    );
    assert_ne!(est.capabilities().unwrap() & roles::CAP_PREDICT, 0);
    assert_eq!(
        est.params(&ctx)
            .unwrap()
            .int_values("n_components")
            .unwrap(),
        vec![3]
    );
    let predicted = est.predict(&ctx, xv).unwrap();
    assert_eq!((predicted.rows, predicted.cols), (24, 1));
    let rmse = (predicted
        .data
        .iter()
        .zip(&y)
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f64>()
        / 24.0)
        .sqrt();
    assert!(rmse < 0.1, "rmse {rmse}");

    // A column-major copy of X, read through strides, predicts identically.
    let col_major: Vec<f64> = (0..8)
        .flat_map(|j| (0..24).map(move |i| (i, j)))
        .map(|(i, j)| x[i * 8 + j])
        .collect();
    let strided = MatrixRef::strided(&col_major, 24, 8, 1, 24).unwrap();
    assert_eq!(est.predict(&ctx, strided).unwrap(), predicted);
    assert!(MatrixRef::strided(&col_major, 24, 8, 1, 25).is_err());

    let bytes = est.to_n4me(&ctx, false).unwrap();
    assert_eq!(&bytes[..4], b"N4ME");
    let restored = Estimator::from_n4me(&ctx, &bytes).unwrap();
    assert_eq!(restored.predict(&ctx, xv).unwrap(), predicted);
    assert_eq!(restored.to_n4me(&ctx, false).unwrap(), bytes);
    assert_eq!(
        restored
            .params(&ctx)
            .unwrap()
            .int_values("n_components")
            .unwrap(),
        vec![3]
    );

    // Roles and inputs are enforced natively.
    assert_eq!(
        est.transform(&ctx, xv).unwrap_err().kind,
        ErrorKind::Unsupported
    );
    let groups = vec![1i64; 24];
    let err = est
        .fit(&ctx, &FitInputs::new(xv).y(yv).groups(&groups))
        .unwrap_err();
    assert!(err.message.contains("not used"), "{err}");
    let err = Estimator::new(&ctx, "models.sparse.group_sparse_pls", None)
        .unwrap()
        .fit(&ctx, &FitInputs::new(xv).y(yv))
        .unwrap_err();
    assert!(err.message.contains("feature_groups"), "{err}");
    let mut bogus = Params::new(&ctx, "models.pls.pls_regression").unwrap();
    let err = bogus.set_enum("solver", "bogus").err().unwrap();
    assert!(err.message.contains("solver"), "{err}");
    let err = bogus.set_int("no_such_parameter", 1).err().unwrap();
    assert!(err.message.contains("no_such_parameter"), "{err}");
    ctx.set_max_state_bytes(8).unwrap();
    assert!(Estimator::from_n4me(&ctx, &bytes).is_err());
}

#[test]
fn transformer_selector_classifier_and_filter() {
    let ctx = Context::new().unwrap();
    let fx = fixture();
    let x_train = Dense::new(&fx["x_train"]);
    let x_test = Dense::new(&fx["x_test"]);
    let y_train = Dense::new(&fx["y_train"]);
    let y_test = Dense::new(&fx["y_test"]);
    let labels = ints(&fx["labels_train"]);

    let mut snv = Estimator::new(&ctx, "preprocessing.scatter.snv", None).unwrap();
    snv.fit(&ctx, &FitInputs::new(x_train.view())).unwrap();
    let out = snv.transform(&ctx, x_test.view()).unwrap();
    assert_eq!((out.rows, out.cols), (x_test.rows, x_test.cols));
    for row in out.data.chunks(out.cols) {
        let mean = row.iter().sum::<f64>() / row.len() as f64;
        assert!(mean.abs() < 1e-12);
    }
    assert_eq!(
        snv.predict(&ctx, x_test.view()).unwrap_err().kind,
        ErrorKind::Unsupported
    );

    let mut params = Params::new(&ctx, "filters.correlation").unwrap();
    params.set_int("top_k", 4).unwrap();
    let mut selector = Estimator::new(&ctx, "filters.correlation", Some(&params)).unwrap();
    selector
        .fit(&ctx, &FitInputs::new(x_train.view()).y(y_train.view()))
        .unwrap();
    let selected = selector.selected_indices().unwrap();
    assert_eq!(selected.len(), 4);
    let mut ascending = selected.clone();
    ascending.sort_unstable();
    let subset = selector.transform(&ctx, x_test.view()).unwrap();
    assert_eq!(subset.cols, 4);
    for (row, out_row) in subset.data.chunks(4).enumerate() {
        let expected: Vec<f64> = ascending
            .iter()
            .map(|&j| x_test.data[row * x_test.cols + j as usize])
            .collect();
        assert_eq!(out_row, &expected[..]);
    }

    let mut qda = Estimator::new(&ctx, "models.classification.pls_qda", None).unwrap();
    qda.fit(&ctx, &FitInputs::new(x_train.view()).labels(&labels))
        .unwrap();
    let classes = qda.classes().unwrap();
    let distinct: Vec<i64> = labels
        .iter()
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    assert_eq!(classes, distinct);
    let proba = qda.predict_proba(&ctx, x_test.view()).unwrap();
    assert_eq!(proba.cols, classes.len());
    let predicted = qda.predict_labels(&ctx, x_test.view()).unwrap();
    for (row, label) in proba.data.chunks(proba.cols).zip(&predicted) {
        assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        let best = (0..row.len()).fold(0, |b, k| if row[k] > row[b] { k } else { b });
        assert_eq!(classes[best], *label);
    }
    assert_eq!(
        qda.decision_function(&ctx, x_test.view()).unwrap().cols,
        classes.len()
    );

    let mut filter = Estimator::new(&ctx, "filters.high_leverage", None).unwrap();
    filter.fit(&ctx, &FitInputs::new(x_train.view())).unwrap();
    let mask = filter
        .apply_mask(&ctx, x_test.view(), Some(y_test.view()))
        .unwrap();
    assert_eq!(mask.len(), x_test.rows);
    assert_eq!(filter.apply_mask(&ctx, x_test.view(), None).unwrap(), mask);
}

#[test]
fn procedures_split_augment_and_run() {
    let ctx = Context::new().unwrap();
    let fx = fixture();
    let x_train = Dense::new(&fx["x_train"]);
    let x_predictions = Dense::new(&fx["x_predictions"]);
    let y_train = Dense::new(&fx["y_train"]);

    let split = roles::run_procedure(
        &ctx,
        "splitters.kennard_stone",
        None,
        &FitInputs::new(x_train.view()),
    )
    .unwrap();
    let folds = split.folds().unwrap();
    assert_eq!(folds.len(), 1);
    let mut rows: Vec<i64> = folds[0]
        .train
        .iter()
        .chain(&folds[0].test)
        .copied()
        .collect();
    rows.sort_unstable();
    assert_eq!(rows, (0..x_train.rows as i64).collect::<Vec<_>>());

    let mut params = Params::new(&ctx, "augmentation.noise.gaussian_noise").unwrap();
    params.set_int("seed", 3).unwrap();
    let augment = |params: &Params| {
        roles::run_procedure(
            &ctx,
            "augmentation.noise.gaussian_noise",
            Some(params),
            &FitInputs::new(x_train.view()),
        )
        .unwrap()
        .double_matrix("X")
        .unwrap()
    };
    let noisy = augment(&params);
    assert_eq!((noisy.rows, noisy.cols), (x_train.rows, x_train.cols));
    assert_ne!(noisy.data, x_train.data);
    assert_eq!(augment(&params), noisy);

    let metrics = roles::run_procedure(
        &ctx,
        "diagnostics.regression_metrics",
        None,
        &FitInputs::new(x_predictions.view()).y(y_train.view()),
    )
    .unwrap();
    let entries = metrics.entries().unwrap();
    assert!(entries.iter().all(|e| e.kind == ResultEntryKind::Scalar));
    let rmse = metrics.scalar("rmse").unwrap();
    let expected = &fx["procedures"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["method_id"] == "diagnostics.regression_metrics")
        .unwrap()["outputs"]["rmse"];
    close(&[rmse], &[expected.as_f64().unwrap()], 1e-12, "rmse");
    assert!(metrics
        .scalar("missing")
        .unwrap_err()
        .message
        .contains("missing"));
    assert_eq!(
        roles::run_procedure(
            &ctx,
            "models.pls.cppls",
            None,
            &FitInputs::new(x_train.view())
        )
        .err()
        .unwrap()
        .kind,
        ErrorKind::InvalidArgument
    );
}

#[test]
fn replays_the_cross_language_fixture() {
    let ctx = Context::new().unwrap();
    let fx = fixture();
    let x_train = Dense::new(&fx["x_train"]);
    let x_test = Dense::new(&fx["x_test"]);
    let y_train = Dense::new(&fx["y_train"]);
    let y_test = Dense::new(&fx["y_test"]);
    let labels = ints(&fx["labels_train"]);
    let extra = ExtraInputs::new(&fx);
    let (xt, xtest) = (x_train.view(), x_test.view());

    let cases = fx["cases"].as_array().unwrap();
    let procedures = fx["procedures"].as_array().unwrap();
    let methods = roles::methods().unwrap();
    assert_eq!(methods.len(), cases.len() + procedures.len());
    let listed: BTreeSet<&str> = cases
        .iter()
        .chain(procedures)
        .map(|c| c["method_id"].as_str().unwrap())
        .collect();
    assert_eq!(
        listed,
        methods
            .iter()
            .map(|m| m.method_id.as_str())
            .collect::<BTreeSet<_>>()
    );

    let check = |est: &Estimator, case: &Value, tol: f64, label: &str| {
        let caps = est.capabilities().unwrap();
        assert_eq!(
            case.get("predict").is_some(),
            caps & roles::CAP_PREDICT != 0,
            "{label}"
        );
        if let Some(expected) = case.get("predict") {
            close(
                &est.predict(&ctx, xtest).unwrap().data,
                &floats(expected),
                tol,
                label,
            );
        } else {
            assert_eq!(
                est.predict(&ctx, xtest).unwrap_err().kind,
                ErrorKind::Unsupported
            );
        }
        assert_eq!(
            case.get("transform").is_some(),
            caps & roles::CAP_TRANSFORM != 0,
            "{label}"
        );
        if let Some(expected) = case.get("transform") {
            close(
                &est.transform(&ctx, xtest).unwrap().data,
                &floats(expected),
                tol,
                label,
            );
        }
        if let Some(expected) = case.get("selected_indices") {
            assert_eq!(est.selected_indices().unwrap(), ints(expected), "{label}");
        }
        if let Some(expected) = case.get("mask") {
            let mask: Vec<bool> = ints(expected).into_iter().map(|v| v == 1).collect();
            assert_eq!(
                est.apply_mask(&ctx, xtest, Some(y_test.view())).unwrap(),
                mask,
                "{label} mask"
            );
        }
        if let Some(expected) = case.get("classes") {
            assert_eq!(est.classes().unwrap(), ints(expected), "{label}");
            assert_eq!(
                est.predict_labels(&ctx, xtest).unwrap(),
                ints(&case["predict_labels"]),
                "{label}"
            );
            close(
                &est.decision_function(&ctx, xtest).unwrap().data,
                &floats(&case["decision_function"]),
                tol,
                label,
            );
            match case.get("predict_proba") {
                Some(expected) => close(
                    &est.predict_proba(&ctx, xtest).unwrap().data,
                    &floats(expected),
                    tol,
                    label,
                ),
                None => assert_eq!(
                    est.predict_proba(&ctx, xtest).unwrap_err().kind,
                    ErrorKind::Unsupported,
                    "{label} must not define probabilities"
                ),
            }
        }
    };

    let mut imported = 0;
    for case in cases {
        let method_id = case["method_id"].as_str().unwrap();
        let params = params_for(&ctx, method_id, &case["params"]);
        let mut fitted = Estimator::new(&ctx, method_id, Some(&params)).unwrap();
        let response = case
            .get("y")
            .map(|name| Dense::new(&fx[name.as_str().unwrap()]));
        let base = if case.get("classes").is_some() {
            FitInputs::new(xt).labels(&labels)
        } else if let Some(response) = &response {
            FitInputs::new(xt).y(response.view())
        } else {
            FitInputs::new(xt).y(y_train.view())
        };
        fitted
            .fit(&ctx, &extra.add(base, &case["fit_inputs"]))
            .unwrap_or_else(|e| panic!("{method_id} fit: {e}"));
        check(&fitted, case, 1e-9, &format!("{method_id} Rust fit"));

        let Some(encoded) = case["n4me_base64"].as_str() else {
            continue; // train-only filter without a serializable state
        };
        let payload = STANDARD.decode(encoded).unwrap();
        let est = Estimator::from_n4me(&ctx, &payload).unwrap();
        assert_eq!(est.method_id().unwrap(), method_id);
        check(&est, case, REPLAY_TOL, method_id);
        assert_eq!(
            est.to_n4me(&ctx, true).unwrap(),
            payload,
            "{method_id} re-export"
        );
        imported += 1;
    }

    let x_predictions = Dense::new(&fx["x_predictions"]);
    let (mut folds, mut augmented, mut runs) = (0, 0, 0);
    for case in procedures {
        let method_id = case["method_id"].as_str().unwrap();
        let x = match case["x"].as_str().unwrap() {
            "x_train" => xt,
            "x_predictions" => x_predictions.view(),
            other => panic!("fixture matrix {other}"),
        };
        let params = case.get("params").map(|p| params_for(&ctx, method_id, p));
        let result = roles::run_procedure(
            &ctx,
            method_id,
            params.as_ref(),
            &extra.add(FitInputs::new(x), &case["inputs"]),
        )
        .unwrap_or_else(|e| panic!("{method_id}: {e}"));
        if let Some(expected) = case.get("folds") {
            let got: Vec<(Vec<i64>, Vec<i64>)> = result
                .folds()
                .unwrap()
                .into_iter()
                .map(|f| (f.train, f.test))
                .collect();
            let want: Vec<(Vec<i64>, Vec<i64>)> = expected
                .as_array()
                .unwrap()
                .iter()
                .map(|f| (ints(&f[0]), ints(&f[1])))
                .collect();
            assert_eq!(got, want, "{method_id} folds");
            folds += 1;
        }
        if let Some(expected) = case.get("X") {
            let got = result.double_matrix("X").unwrap();
            assert_eq!((got.rows, got.cols), (x.rows(), x.cols()));
            close(&got.data, &floats(expected), REPLAY_TOL, method_id);
            augmented += 1;
        }
        if let Some(expected) = case.get("outputs") {
            let expected = expected.as_object().unwrap();
            let entries = result.entries().unwrap();
            assert_eq!(
                entries
                    .iter()
                    .map(|e| e.name.as_str())
                    .collect::<BTreeSet<_>>(),
                expected.keys().map(String::as_str).collect::<BTreeSet<_>>(),
                "{method_id} outputs"
            );
            for entry in entries {
                let name = &entry.name;
                let got: Vec<f64> = match entry.kind {
                    ResultEntryKind::DoubleMatrix => result.double_matrix(name).unwrap().data,
                    ResultEntryKind::IntVector => result
                        .int_vector(name)
                        .unwrap()
                        .into_iter()
                        .map(f64::from)
                        .collect(),
                    ResultEntryKind::Int64Vector => result
                        .int64_vector(name)
                        .unwrap()
                        .into_iter()
                        .map(|v| v as f64)
                        .collect(),
                    ResultEntryKind::Scalar => vec![result.scalar(name).unwrap()],
                };
                close(
                    &got,
                    &floats(&expected[name]),
                    1e-9,
                    &format!("{method_id}.{name}"),
                );
            }
            runs += 1;
        }
    }
    assert_eq!(folds + augmented + runs, procedures.len());
    println!(
        "estimator roles: {} estimators ({imported} N4ME states) and {} procedures \
         ({folds} splitters, {augmented} augmenters, {runs} generic) reproduced in Rust",
        cases.len(),
        procedures.len()
    );
}
