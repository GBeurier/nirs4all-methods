//! Native role pipeline (ABI 2.14): the shared fixture written by the Python
//! binding (also replayed by R and JS/WASM). Its pipelines predict identically
//! in Rust, Rust refits reproduce the Python fits, and the negative cases are
//! refused with the same native status and message.

use base64::{engine::general_purpose::STANDARD, Engine as _};
use n4m::roles::{self, FitInputs, ParamType, ParamValue, Params, RolePipeline};
use n4m::{Context, MatrixRef};
use serde_json::Value;

static_assertions::assert_not_impl_any!(RolePipeline: Send, Sync);

fn fixture() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../parity/fixtures/role_pipeline_negative.json"
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
    fn first_columns(&self, cols: usize) -> Self {
        Self {
            data: self
                .data
                .chunks(self.cols)
                .flat_map(|row| row[..cols].iter().copied())
                .collect(),
            rows: self.rows,
            cols,
        }
    }
}
fn floats(value: &Value) -> Vec<f64> {
    match value {
        Value::Array(items) => items.iter().flat_map(floats).collect(),
        other => vec![other.as_f64().expect("number")],
    }
}
fn strings(value: &Value) -> Vec<String> {
    value
        .as_array()
        .expect("string list")
        .iter()
        .map(|v| v.as_str().expect("string").to_owned())
        .collect()
}

/// Fixture outputs were computed on Linux x86-64; kernels drift by a few ulps
/// elsewhere, so replays compare at 1e-9.
const REPLAY_TOL: f64 = 1e-9;

fn close(actual: &[f64], expected: &[f64], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= REPLAY_TOL * (1.0 + e.abs()),
            "{label}[{i}]: {a} vs {e}"
        );
    }
}

/// Recipe tokens `{"class": "n4m:<id>", "params": {...}}` as native steps.
struct Recipe {
    ids: Vec<String>,
    params: Vec<Params>,
}
impl Recipe {
    fn new(ctx: &Context, steps: &Value) -> Self {
        let mut ids = Vec::new();
        let mut params = Vec::new();
        for step in steps.as_array().expect("steps") {
            let id = step["class"].as_str().unwrap().trim_start_matches("n4m:");
            let info = roles::method_info(id).unwrap();
            let mut p = Params::new(ctx, id).unwrap();
            for (name, value) in step["params"].as_object().expect("params") {
                let declared = info.params.iter().find(|q| &q.name == name).unwrap();
                let typed = match declared.param_type {
                    ParamType::Int => ParamValue::Int(value.as_i64().unwrap()),
                    ParamType::Double => ParamValue::Double(value.as_f64().unwrap()),
                    ParamType::Bool => ParamValue::Bool(value.as_bool().unwrap()),
                    ParamType::Enum => ParamValue::Enum(value.as_str().unwrap().to_owned()),
                    other => panic!("fixture parameter type {other:?}"),
                };
                p.set(name, &typed).unwrap();
            }
            ids.push(id.to_owned());
            params.push(p);
        }
        Self { ids, params }
    }
    fn pipeline(&self, ctx: &Context) -> Result<RolePipeline, n4m::Error> {
        let steps: Vec<(&str, Option<&Params>)> = self
            .ids
            .iter()
            .zip(&self.params)
            .map(|(id, p)| (id.as_str(), Some(p)))
            .collect();
        RolePipeline::new(ctx, &steps)
    }
}

fn decode(states: &Value) -> Vec<Vec<u8>> {
    states
        .as_array()
        .unwrap()
        .iter()
        .map(|s| {
            let text = s.get("n4me_base64").unwrap_or(s).as_str().unwrap();
            STANDARD.decode(text).unwrap()
        })
        .collect()
}

fn import(
    ctx: &Context,
    steps: &Value,
    states: &Value,
    names: Option<&[&str]>,
) -> Result<RolePipeline, n4m::Error> {
    let mut pipeline = Recipe::new(ctx, steps).pipeline(ctx)?;
    if let Some(names) = names {
        pipeline.set_feature_names(ctx, names)?;
    }
    let payloads = decode(states);
    let slices: Vec<&[u8]> = payloads.iter().map(Vec::as_slice).collect();
    pipeline.import_states(ctx, &slices)?;
    Ok(pipeline)
}

/// Class ids of string labels, in sorted order (as every binding encodes them).
fn label_ids(labels: &[String], classes: &[String]) -> Vec<i64> {
    labels
        .iter()
        .map(|l| classes.iter().position(|c| c == l).unwrap() as i64)
        .collect()
}

#[test]
fn shared_fixture_pipelines_replay_and_refit() {
    let fx = fixture();
    let ctx = Context::new().unwrap();
    let owned_names = strings(&fx["feature_names"]);
    let names: Vec<&str> = owned_names.iter().map(String::as_str).collect();
    let (x_train, x_test) = (Dense::new(&fx["x_train"]), Dense::new(&fx["x_test"]));

    let reg = &fx["regression"];
    let pipeline = import(&ctx, &reg["steps"], &reg["states"], Some(&names)).unwrap();
    let predicted = pipeline.predict(&ctx, x_test.view(), Some(&names)).unwrap();
    close(
        &predicted.data,
        &floats(&reg["predict"]),
        "regression predict",
    );
    let transformed = pipeline.transform(&ctx, x_test.view(), None).unwrap();
    close(
        &transformed.data,
        &floats(&reg["transform"]),
        "regression transform",
    );
    assert_eq!(
        pipeline.export_states(&ctx, false).unwrap(),
        decode(&reg["states"])
    );
    assert_eq!(pipeline.feature_names().unwrap(), owned_names);
    let steps = pipeline.steps().unwrap();
    assert_eq!(steps[0].role, roles::ROLE_SAMPLE_FILTER);
    assert_eq!(steps[0].state_index, None);
    assert_eq!(steps[3].role, roles::ROLE_REGRESSOR);

    let y_train = Dense::new(&fx["y_train"]);
    let mut refit = Recipe::new(&ctx, &reg["steps"]).pipeline(&ctx).unwrap();
    refit
        .fit(&ctx, &FitInputs::new(x_train.view()).y(y_train.view()))
        .unwrap();
    close(
        &refit.predict(&ctx, x_test.view(), None).unwrap().data,
        &floats(&reg["predict"]),
        "regression refit",
    );

    let cls = &fx["classification"];
    let class_names = strings(&cls["class_names"]);
    let pipeline = import(&ctx, &cls["steps"], &cls["states"], Some(&names)).unwrap();
    let expected = label_ids(&strings(&cls["predict"]), &class_names);
    assert_eq!(
        pipeline.predict_labels(&ctx, x_test.view(), None).unwrap(),
        expected
    );
    assert_eq!(pipeline.classes().unwrap(), vec![0, 1, 2]);
    close(
        &pipeline
            .decision_function(&ctx, x_test.view(), None)
            .unwrap()
            .data,
        &floats(&cls["decision_function"]),
        "decision function",
    );
    let labels = label_ids(&strings(&fx["labels_train"]), &class_names);
    let mut refit = Recipe::new(&ctx, &cls["steps"]).pipeline(&ctx).unwrap();
    refit
        .fit(&ctx, &FitInputs::new(x_train.view()).labels(&labels))
        .unwrap();
    assert_eq!(
        refit.predict_labels(&ctx, x_test.view(), None).unwrap(),
        expected
    );
}

#[test]
fn shared_fixture_negative_cases_are_refused_alike() {
    let fx = fixture();
    let ctx = Context::new().unwrap();
    let owned_names = strings(&fx["feature_names"]);
    let names: Vec<&str> = owned_names.iter().map(String::as_str).collect();
    let (x_train, x_test) = (Dense::new(&fx["x_train"]), Dense::new(&fx["x_test"]));
    let reg = &fx["regression"];
    let fitted = import(&ctx, &reg["steps"], &reg["states"], Some(&names)).unwrap();
    for case in fx["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let stage = case["stage"].as_str().unwrap();
        let fit = || {
            let y = Dense::new(&fx[case["y"].as_str().unwrap()]);
            let mut pipeline = Recipe::new(&ctx, &case["steps"]).pipeline(&ctx)?;
            pipeline.fit(&ctx, &FitInputs::new(x_train.view()).y(y.view()))?;
            Ok::<_, n4m::Error>(pipeline)
        };
        let result = match stage {
            "fit" => {
                let pipeline = fit().unwrap();
                let predicted = pipeline.predict(&ctx, x_test.view(), None).unwrap();
                close(&predicted.data, &floats(&case["predict"]), name);
                continue;
            }
            "create" => Recipe::new(&ctx, &case["steps"]).pipeline(&ctx).map(drop),
            "import" => import(&ctx, &case["steps"], &case["states"], None).map(drop),
            "predict" => {
                let columns: Vec<String> = if case["drop_last_column"].as_bool() == Some(true) {
                    owned_names[..owned_names.len() - 1].to_vec()
                } else {
                    strings(&case["feature_names"])
                };
                let cols: Vec<&str> = columns.iter().map(String::as_str).collect();
                let x = x_test.first_columns(cols.len());
                fitted.predict(&ctx, x.view(), Some(&cols)).map(drop)
            }
            "export" => fit().and_then(|p| p.export_states(&ctx, false)).map(drop),
            other => panic!("fixture stage {other}"),
        };
        let error = result.expect_err(name);
        assert_eq!(
            i64::from(error.status),
            case["status"].as_i64().unwrap(),
            "{name}"
        );
        let message = case["message"].as_str().unwrap();
        assert!(
            error.message.contains(message),
            "{name}: '{}' lacks '{message}'",
            error.message
        );
    }
}

#[test]
fn training_rows_need_an_opt_in() {
    let fx = fixture();
    let ctx = Context::new().unwrap();
    let (x_train, y_train) = (Dense::new(&fx["x_train"]), Dense::new(&fx["y_train"]));
    let mut pipeline = RolePipeline::new(
        &ctx,
        &[
            ("preprocessing.scatter.snv", None),
            ("models.pls.kernel", None),
        ],
    )
    .unwrap();
    pipeline
        .fit(&ctx, &FitInputs::new(x_train.view()).y(y_train.view()))
        .unwrap();
    let rows: Vec<bool> = pipeline
        .steps()
        .unwrap()
        .iter()
        .map(|s| s.contains_training_rows)
        .collect();
    assert_eq!(rows, vec![false, true]);
    let error = pipeline.export_states(&ctx, false).unwrap_err();
    assert!(error.message.contains("state retains training rows"));
    assert_eq!(pipeline.export_states(&ctx, true).unwrap().len(), 2);
}
