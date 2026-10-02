//! Complete native fit/state/replay and safe tensor marshalling, not an encoder
//! implementation. The SDK gate covers the canonical U07 numerical oracle.

use n4m::multimodal::{Encoder, MultimodalPipeline, Recipe, SourceSpec, SourceView};
use n4m::{Context, ErrorKind, MatrixRef};

static_assertions::assert_not_impl_any!(MultimodalPipeline: Send, Sync);

fn recipe() -> Recipe {
    let names = ["nir", "image", "series", "metadata"];
    let representations = ["signal_1d", "rgb_image", "series_mv", "tabular_mixed"];
    let shapes = [vec![3], vec![2, 2, 1], vec![3, 1], vec![2]];
    let encoders = [
        Encoder::StandardScaler,
        Encoder::TensorPca {
            n_components: 2,
            random_state: 42,
        },
        Encoder::TensorPca {
            n_components: 1,
            random_state: 42,
        },
        Encoder::ColumnTransformer,
    ];
    Recipe {
        alpha: 0.5,
        sources: (0..4)
            .map(|index| SourceSpec {
                name: names[index].into(),
                representation_id: representations[index].into(),
                dtype: if index == 3 { "object" } else { "float64" }.into(),
                identity: format!(
                    "{{\"source_id\":\"{}\",\"axis_order\":\"canonical\"}}",
                    names[index]
                ),
                input_shape: shapes[index].clone(),
                encoder: encoders[index].clone(),
                weight: 1.0,
            })
            .collect(),
    }
}
struct Data {
    nir: Vec<f64>,
    image: Vec<f64>,
    series: Vec<f64>,
    metadata: Vec<f64>,
    categories: Vec<String>,
    y: Vec<f64>,
    rows: usize,
}
impl Data {
    fn training() -> Self {
        Self::new(1, 6, false)
    }
    fn heldout() -> Self {
        Self::new(7, 2, true)
    }
    fn new(start: usize, rows: usize, heldout: bool) -> Self {
        let values = (start..start + rows).map(|v| v as f64).collect::<Vec<_>>();
        Self {
            nir: values.iter().flat_map(|&v| [v, 7.0, -v]).collect(),
            image: values
                .iter()
                .flat_map(|&v| [v, v * v, -v, v + 1.0])
                .collect(),
            series: values.iter().flat_map(|&v| [v, v * v, v.sin()]).collect(),
            metadata: values
                .iter()
                .map(|&v| v + if heldout { 13.0 } else { 9.0 })
                .collect(),
            categories: (0..rows)
                .map(|row| {
                    if heldout {
                        if row == 0 {
                            "未見"
                        } else {
                            "\0"
                        }
                    } else if row % 2 == 0 {
                        "A"
                    } else {
                        "é"
                    }
                    .into()
                })
                .collect(),
            y: values
                .iter()
                .enumerate()
                .map(|(row, &v)| 2.0 * v + if row % 2 == 0 { 0.0 } else { 0.5 })
                .collect(),
            rows,
        }
    }
    fn views<'a>(&'a self, declaration: &'a Recipe) -> Vec<SourceView<'a>> {
        vec![
            SourceView::numeric(&declaration.sources[0], &self.nir, &[self.rows, 3]).unwrap(),
            SourceView::numeric(&declaration.sources[1], &self.image, &[self.rows, 2, 2, 1])
                .unwrap(),
            SourceView::numeric(&declaration.sources[2], &self.series, &[self.rows, 3, 1]).unwrap(),
            SourceView::mixed(
                &declaration.sources[3],
                &self.metadata,
                &self
                    .categories
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
        ]
    }
    fn y(&self) -> MatrixRef<'_> {
        MatrixRef::row_major(&self.y, self.rows, 1).unwrap()
    }
}

#[test]
fn complete_state_replays_in_a_fresh_context_without_fit_or_training_buffers() {
    let declaration = recipe();
    let ctx = Context::new().unwrap();
    let train = Data::training();
    let heldout = Data::heldout();
    let mut pipeline = MultimodalPipeline::new(&ctx, &declaration).unwrap();
    pipeline
        .fit(&ctx, &train.views(&declaration), train.y())
        .unwrap();
    let expected = pipeline
        .predict(&ctx, &heldout.views(&declaration))
        .unwrap();
    let features = pipeline
        .transform(&ctx, &heldout.views(&declaration))
        .unwrap();
    assert_eq!(features.cols, 9);
    for row in features.data.chunks_exact(9) {
        assert_eq!(row[1], 0.0, "constant NIR column");
        assert_eq!(&row[7..9], &[0.0, 0.0], "unknown categories ignored");
    }
    assert!(
        (features.data[6] - (20.0 - 12.5) / (35.0_f64 / 12.0).sqrt()).abs() < 1e-12,
        "metadata population scaling uses train-only statistics"
    );
    let state = pipeline.export_state(&ctx).unwrap();
    assert_eq!(&state[..4], b"N4MF");
    drop(pipeline);
    drop(ctx);
    drop(train);
    let fresh_ctx = Context::new().unwrap();
    let replay = MultimodalPipeline::from_state(&fresh_ctx, &declaration, &state).unwrap();
    assert_eq!(
        replay
            .predict(&fresh_ctx, &heldout.views(&declaration))
            .unwrap(),
        expected
    );
    assert_eq!(
        replay
            .transform(&fresh_ctx, &heldout.views(&declaration))
            .unwrap(),
        features
    );
    assert_eq!(replay.export_state(&fresh_ctx).unwrap(), state);
}

#[test]
fn native_import_refuses_recipe_drift_and_corrupt_state() {
    let declaration = recipe();
    let ctx = Context::new().unwrap();
    let train = Data::training();
    let mut pipeline = MultimodalPipeline::new(&ctx, &declaration).unwrap();
    pipeline
        .fit(&ctx, &train.views(&declaration), train.y())
        .unwrap();
    let mut state = pipeline.export_state(&ctx).unwrap();
    let mut changed = declaration.clone();
    changed.alpha = 1.0;
    assert!(MultimodalPipeline::from_state(&ctx, &changed, &state).is_err());
    let middle = state.len() / 2;
    state[middle] ^= 1;
    assert!(MultimodalPipeline::from_state(&ctx, &declaration, &state).is_err());
}

#[test]
fn empty_utf8_vocabulary_and_embedded_nul_unknown_replay_exactly() {
    let declaration = recipe();
    let ctx = Context::new().unwrap();
    let mut train = Data::training();
    train.categories.iter_mut().for_each(String::clear);
    let mut heldout = Data::heldout();
    heldout.categories = vec![String::new(), "\0".into()];
    let mut pipeline = MultimodalPipeline::new(&ctx, &declaration).unwrap();
    pipeline
        .fit(&ctx, &train.views(&declaration), train.y())
        .unwrap();
    let features = pipeline
        .transform(&ctx, &heldout.views(&declaration))
        .unwrap();
    assert_eq!(features.cols, 8, "one learned empty-string category");
    assert_eq!(
        features.data[7], 1.0,
        "empty string remains a known category"
    );
    assert_eq!(
        features.data[15], 0.0,
        "embedded NUL is an unknown category"
    );
    let predictions = pipeline
        .predict(&ctx, &heldout.views(&declaration))
        .unwrap();
    let state = pipeline.export_state(&ctx).unwrap();
    drop(pipeline);
    drop(train);
    let restored = MultimodalPipeline::from_state(&ctx, &declaration, &state).unwrap();
    assert_eq!(
        restored
            .transform(&ctx, &heldout.views(&declaration))
            .unwrap(),
        features
    );
    assert_eq!(
        restored
            .predict(&ctx, &heldout.views(&declaration))
            .unwrap(),
        predictions
    );
}

#[test]
fn native_prediction_refuses_independently_declared_schema_drift() {
    let declaration = recipe();
    let ctx = Context::new().unwrap();
    let train = Data::training();
    let heldout = Data::heldout();
    let mut pipeline = MultimodalPipeline::new(&ctx, &declaration).unwrap();
    pipeline
        .fit(&ctx, &train.views(&declaration), train.y())
        .unwrap();
    let mut changed = declaration.clone();
    changed.sources[2].identity.push_str("different-axis");
    assert_eq!(
        pipeline
            .predict(&ctx, &heldout.views(&changed))
            .unwrap_err()
            .kind,
        ErrorKind::ShapeMismatch
    );
}

#[test]
fn failed_fit_preserves_previous_predictor_and_exact_state() {
    let declaration = recipe();
    let ctx = Context::new().unwrap();
    let train = Data::training();
    let mut pipeline = MultimodalPipeline::new(&ctx, &declaration).unwrap();
    pipeline
        .fit(&ctx, &train.views(&declaration), train.y())
        .unwrap();
    let before = pipeline.export_state(&ctx).unwrap();
    let bad_target = MatrixRef::row_major(&train.y[..5], 5, 1).unwrap();
    assert!(pipeline
        .fit(&ctx, &train.views(&declaration), bad_target)
        .is_err());
    assert_eq!(pipeline.export_state(&ctx).unwrap(), before);
    assert!(pipeline.predict(&ctx, &train.views(&declaration)).is_ok());
}

#[test]
fn column_major_tensor_view_fits_the_same_native_predictor() {
    let declaration = recipe();
    let ctx = Context::new().unwrap();
    let train = Data::training();
    let mut row_major = MultimodalPipeline::new(&ctx, &declaration).unwrap();
    row_major
        .fit(&ctx, &train.views(&declaration), train.y())
        .unwrap();
    let train_ref = &train;
    let transposed = (0..3)
        .flat_map(|col| (0..train.rows).map(move |row| train_ref.nir[row * 3 + col]))
        .collect::<Vec<_>>();
    let mut views = train.views(&declaration);
    views[0] = SourceView::strided(&declaration.sources[0], &transposed, &[6, 3], &[1, 6]).unwrap();
    let mut column_major = MultimodalPipeline::new(&ctx, &declaration).unwrap();
    column_major.fit(&ctx, &views, train.y()).unwrap();
    assert_eq!(
        column_major.export_state(&ctx).unwrap(),
        row_major.export_state(&ctx).unwrap()
    );
}

#[test]
fn safe_views_refuse_out_of_bounds_spans_and_rank_mismatch() {
    let declaration = recipe();
    assert!(SourceView::numeric(&declaration.sources[0], &[1.0], &[2, 3]).is_err());
    assert!(
        SourceView::strided(&declaration.sources[0], &[1.0], &[usize::MAX, 2], &[2, 1]).is_err()
    );
    assert!(SourceView::strided(&declaration.sources[0], &[1.0], &[1, 1], &[1]).is_err());
    assert!(SourceView::mixed(&declaration.sources[3], &[1.0], &[]).is_err());
}

#[test]
fn f32_numeric_source_retains_native_dtype_and_predicts() {
    let ctx = Context::new().unwrap();
    let train = Data::training();
    let reference_declaration = recipe();
    let mut reference = MultimodalPipeline::new(&ctx, &reference_declaration).unwrap();
    reference
        .fit(&ctx, &train.views(&reference_declaration), train.y())
        .unwrap();
    let expected = reference
        .predict(&ctx, &train.views(&reference_declaration))
        .unwrap();
    let mut declaration = reference_declaration.clone();
    declaration.sources[0].dtype = "float32".into();
    let f32_values = train
        .nir
        .iter()
        .map(|&value| value as f32)
        .collect::<Vec<_>>();
    let mut views = train.views(&declaration);
    views[0] =
        SourceView::numeric_f32(&declaration.sources[0], &f32_values, &[train.rows, 3]).unwrap();
    let mut pipeline = MultimodalPipeline::new(&ctx, &declaration).unwrap();
    pipeline.fit(&ctx, &views, train.y()).unwrap();
    assert_eq!(pipeline.predict(&ctx, &views).unwrap(), expected);
    let state = pipeline.export_state(&ctx).unwrap();
    let replay = MultimodalPipeline::from_state(&ctx, &declaration, &state).unwrap();
    assert_eq!(replay.predict(&ctx, &views).unwrap(), expected);
}
