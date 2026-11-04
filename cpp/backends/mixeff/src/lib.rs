// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
//! Restricted matrix-only adapter; the statistical engine is pinned mixeff-rs.
use mixeff_rs::formula::{FixedTerm, Formula, GroupingFactor, RandomCovariance, RandomTerm};
use mixeff_rs::model::{
    DataFrame, FitOptions, FitToleranceOverrides, LinearMixedModel, MixedModelFit, OptimizerControl,
};
use mixeff_rs::types::{ConvergenceStatus, Optimizer};
use nalgebra::{DMatrix, DVector};
use std::collections::BTreeMap;
use std::panic::{catch_unwind, AssertUnwindSafe};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Options {
    pub struct_size: u64,
    pub reml: i32,
    pub reserved: i32,
    pub max_evaluations: i64,
    pub tolerance: f64,
    pub rank_tolerance: f64,
    pub boundary_tolerance: f64,
}
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Summary {
    pub struct_size: u64,
    pub fit_state: i32,
    pub reason: i32,
    pub converged: i32,
    pub singular: i32,
    pub boundary: i32,
    pub optimizer_status: i32,
    pub observations: i64,
    pub fixed_effects: i64,
    pub group_terms: i64,
    pub design_rank: i64,
    pub total_group_levels: i64,
    pub evaluations: i64,
    pub objective: f64,
    pub log_likelihood: f64,
    pub residual_df: f64,
    pub condition_number: f64,
}
struct Outcome {
    beta: Vec<f64>,
    covariance: Vec<f64>,
    standard_errors: Vec<f64>,
    variances: Vec<f64>,
    summary: Summary,
}
impl Outcome {
    fn new(n: usize, p: usize, k: usize) -> Self {
        Self {
            beta: vec![f64::NAN; p],
            covariance: vec![f64::NAN; p * p],
            standard_errors: vec![f64::NAN; p],
            variances: vec![f64::NAN; k + 1],
            summary: Summary {
                struct_size: std::mem::size_of::<Summary>() as u64,
                fit_state: 5,
                reason: 11,
                converged: 0,
                singular: 0,
                boundary: 0,
                optimizer_status: 0,
                observations: n as i64,
                fixed_effects: p as i64,
                group_terms: k as i64,
                design_rank: 0,
                total_group_levels: 0,
                evaluations: 0,
                objective: f64::NAN,
                log_likelihood: f64::NAN,
                residual_df: f64::NAN,
                condition_number: f64::NAN,
            },
        }
    }
    fn refuse(mut self, state: i32, reason: i32) -> Self {
        self.summary.fit_state = state;
        self.summary.reason = reason;
        self
    }
}

/// Check linear independence of residual and random covariance directions in
/// the fixed-effect residual space. This is structural identifiability, not a
/// post-fit information/uncertainty or causal assessment. Uses traces without
/// constructing an n*n covariance matrix.
fn identifiable(groups: &[Vec<usize>], levels: &[usize], u: &DMatrix<f64>, tol: f64) -> bool {
    let (n, p) = (u.nrows(), u.ncols());
    let mut sums = Vec::new();
    let mut fixed_grams = Vec::new();
    for (ids, &size) in groups.iter().zip(levels) {
        let mut m = DMatrix::<f64>::zeros(size, p);
        for (i, &level) in ids.iter().enumerate() {
            for j in 0..p {
                m[(level, j)] += u[(i, j)];
            }
        }
        fixed_grams.push(m.transpose() * &m);
        sums.push(m);
    }
    let k = groups.len();
    let mut gram = DMatrix::<f64>::zeros(k + 1, k + 1);
    gram[(0, 0)] = (n - p) as f64;
    for g in 0..k {
        gram[(0, g + 1)] = n as f64 - sums[g].norm_squared();
        gram[(g + 1, 0)] = gram[(0, g + 1)];
        for h in 0..=g {
            let mut counts = BTreeMap::<(usize, usize), usize>::new();
            for (&a, &b) in groups[g].iter().zip(&groups[h]) {
                *counts.entry((a, b)).or_default() += 1;
            }
            let mut inner = 0.0;
            let mut projected = 0.0;
            for ((a, b), count) in counts {
                inner += (count * count) as f64;
                let dot: f64 = (0..p).map(|j| sums[g][(a, j)] * sums[h][(b, j)]).sum();
                projected += count as f64 * dot;
            }
            let fixed: f64 = fixed_grams[g]
                .iter()
                .zip(fixed_grams[h].iter())
                .map(|(a, b)| a * b)
                .sum();
            let value = inner - 2.0 * projected + fixed;
            gram[(g + 1, h + 1)] = value;
            gram[(h + 1, g + 1)] = value;
        }
    }
    let scale: Vec<f64> = (0..=k).map(|j| gram[(j, j)].max(0.0).sqrt()).collect();
    if scale.iter().any(|s| !s.is_finite() || *s <= tol.sqrt()) {
        return false;
    }
    for i in 0..=k {
        for j in 0..=k {
            gram[(i, j)] /= scale[i] * scale[j];
        }
    }
    let eigen = gram.symmetric_eigen().eigenvalues;
    let largest = eigen.iter().copied().fold(0.0, f64::max);
    eigen.iter().all(|v| v.is_finite() && *v > tol * largest)
}

fn fit(
    x: &[f64],
    y: &[f64],
    group_data: &[i64],
    n: usize,
    p: usize,
    k: usize,
    o: Options,
) -> Result<Outcome, i32> {
    let mut result = Outcome::new(n, p, k);
    let mut groups = Vec::new();
    let mut levels = Vec::new();
    for j in 0..k {
        let mut map = BTreeMap::new();
        let mut ids = Vec::with_capacity(n);
        for i in 0..n {
            let next = map.len();
            let level = *map.entry(group_data[i * k + j]).or_insert(next);
            ids.push(level);
        }
        levels.push(map.len());
        groups.push(ids);
    }
    let total: usize = levels.iter().sum();
    result.summary.total_group_levels = total as i64;
    if total > 1024 || n * total > 4194304 {
        return Err(10);
    }
    let maximum = x.iter().copied().map(f64::abs).fold(0.0, f64::max);
    if maximum == 0.0 {
        result.summary.residual_df = n as f64;
        result.summary.condition_number = f64::INFINITY;
        return Ok(result.refuse(2, 1));
    }
    let matrix = DMatrix::from_row_slice(n, p, x).map(|v| v / maximum);
    let Some(svd) = matrix.try_svd(true, false, 1e-14, 1000) else {
        return Ok(result.refuse(5, 10));
    };
    let largest = svd.singular_values.iter().copied().fold(0.0, f64::max);
    let rank = svd
        .singular_values
        .iter()
        .filter(|s| **s > largest * o.rank_tolerance)
        .count();
    result.summary.design_rank = rank as i64;
    result.summary.residual_df = (n - rank) as f64;
    result.summary.condition_number = if rank == p {
        largest / svd.singular_values[p - 1]
    } else {
        f64::INFINITY
    };
    if rank < p {
        return Ok(result.refuse(2, 1));
    }
    if n <= rank {
        return Ok(result.refuse(3, 2));
    }
    if y.iter().all(|v| *v == y[0]) {
        return Ok(result.refuse(3, 3));
    }
    if levels.iter().any(|size| *size < 2) {
        return Ok(result.refuse(3, 4));
    }
    let u = svd.u.expect("SVD requested fixed-effect left vectors");
    let y_max = y.iter().copied().map(f64::abs).fold(0.0, f64::max);
    let scaled_y = DVector::from_iterator(n, y.iter().map(|v| v / y_max));
    let residual = &scaled_y - &u * (u.transpose() * &scaled_y);
    if residual.norm_squared() <= o.rank_tolerance.powi(2) * scaled_y.norm_squared() {
        return Ok(result.refuse(3, 6));
    }
    if !identifiable(&groups, &levels, &u, o.rank_tolerance) {
        return Ok(result.refuse(3, 5));
    }

    let mut data = DataFrame::new();
    data.add_numeric("n4m_response", y.to_vec())
        .map_err(|_| 255)?;
    let names: Vec<String> = (0..p).map(|j| format!("n4m_x{j:02}")).collect();
    let mut fixed = vec![FixedTerm::NoIntercept];
    for (j, name) in names.iter().enumerate() {
        data.add_numeric(name, (0..n).map(|i| x[i * p + j]).collect())
            .map_err(|_| 255)?;
        fixed.push(FixedTerm::Column(name.clone()));
    }
    let mut random = Vec::new();
    let group_names: Vec<String> = (0..k).map(|j| format!("n4m_group{j:02}")).collect();
    for (j, name) in group_names.iter().enumerate() {
        data.add_categorical(name, groups[j].iter().map(|v| v.to_string()).collect())
            .map_err(|_| 255)?;
        random.push(RandomTerm {
            terms: vec![FixedTerm::Intercept],
            grouping: GroupingFactor::Single(name.clone()),
            zerocorr: false,
            covariance: RandomCovariance::Full,
            source: None,
        });
    }
    let formula = Formula {
        response: "n4m_response".into(),
        fixed_terms: fixed,
        random_terms: random,
        derived: vec![],
    };
    let mut model = match LinearMixedModel::new(formula, &data, None) {
        Ok(m) => m,
        Err(_) => return Ok(result.refuse(5, 11)),
    };
    // Upstream may apply a different rank tolerance; never accept dropped terms.
    if model.fixed_effect_rank() != p {
        result.summary.design_rank = model.fixed_effect_rank() as i64;
        result.summary.residual_df = (n - model.fixed_effect_rank()) as f64;
        return Ok(result.refuse(2, 1));
    }
    let control = OptimizerControl::auto()
        .with_optimizer(Optimizer::TrustBq)
        .with_max_feval(o.max_evaluations as usize)
        .with_tolerances(FitToleranceOverrides::default().with_ftol_rel(o.tolerance));
    let options = (if o.reml == 1 {
        FitOptions::reml()
    } else {
        FitOptions::ml()
    })
    .with_optimizer_control(control);
    let fitted = model.fit_with_options(options).is_ok();
    result.summary.evaluations = model.optsum().feval.max(0);
    let status = match model.convergence_status() {
        ConvergenceStatus::Converged => 1,
        ConvergenceStatus::BudgetExhausted => 2,
        ConvergenceStatus::RoundoffLimited => 3,
        _ => 4,
    };
    result.summary.optimizer_status = status;
    if !fitted || status != 1 {
        return Ok(result.refuse(
            4,
            match status {
                2 => 7,
                3 => 8,
                _ => 9,
            },
        ));
    }
    result.summary.converged = 1;
    result.summary.objective = model.objective();
    result.summary.log_likelihood = model.loglikelihood();
    let coef_names = model.coef_names();
    let ordering: Option<Vec<usize>> = names
        .iter()
        .map(|name| coef_names.iter().position(|s| s == name))
        .collect();
    let Some(order) = ordering else {
        return Ok(result.refuse(5, 11));
    };
    let beta = model.coef();
    let covariance = model.vcov();
    if beta.len() != p || covariance.nrows() != p || covariance.ncols() != p {
        return Ok(result.refuse(5, 11));
    }
    let components = model.varcorr();
    let mut variances = vec![model.sigma().powi(2)];
    for name in &group_names {
        let Some(component) = components.components.iter().find(|c| &c.group == name) else {
            return Ok(result.refuse(5, 11));
        };
        if component.std_dev.len() != 1 {
            return Ok(result.refuse(5, 11));
        }
        variances.push(component.std_dev[0].powi(2));
    }
    let mut cov = Vec::with_capacity(p * p);
    let mut se = Vec::with_capacity(p);
    let mut coefficients = Vec::with_capacity(p);
    for &i in &order {
        coefficients.push(beta[i]);
        se.push(covariance[(i, i)].sqrt());
        for &j in &order {
            cov.push(covariance[(i, j)]);
        }
    }
    if !coefficients
        .iter()
        .chain(&cov)
        .chain(&se)
        .chain(&variances)
        .all(|v| v.is_finite())
        || variances[0] <= 0.0
        || !result.summary.objective.is_finite()
        || !result.summary.log_likelihood.is_finite()
    {
        return Ok(result.refuse(5, 10));
    }
    let boundary = variances[1..]
        .iter()
        .any(|v| *v / variances[0] <= o.boundary_tolerance);
    result.summary.fit_state = if boundary { 1 } else { 0 };
    result.summary.reason = 0;
    result.summary.boundary = i32::from(boundary);
    result.summary.singular = i32::from(boundary);
    result.beta = coefficients;
    result.covariance = cov;
    result.standard_errors = se;
    result.variances = variances;
    Ok(result)
}

#[cfg(test)]
static TEST_PANIC_NEXT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Internal symbol: C++ owns view validation and final atomic output commit.
///
/// # Safety
/// Pointers must refer to disjoint writable output regions or readable packed
/// row-major input arrays of the declared sizes. This is not the public ABI.
#[no_mangle]
pub unsafe extern "C" fn n4m_mixeff_fit_bridge(
    x: *const f64,
    y: *const f64,
    groups: *const i64,
    n: usize,
    p: usize,
    k: usize,
    options: *const Options,
    beta: *mut f64,
    covariance: *mut f64,
    se: *mut f64,
    variances: *mut f64,
    summary: *mut Summary,
) -> i32 {
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        #[cfg(test)]
        if TEST_PANIC_NEXT.swap(false, std::sync::atomic::Ordering::SeqCst) {
            panic!("controlled boundary panic");
        }
        let x = std::slice::from_raw_parts(x, n * p);
        let y = std::slice::from_raw_parts(y, n);
        let groups = std::slice::from_raw_parts(groups, n * k);
        fit(x, y, groups, n, p, k, *options)
    }));
    match outcome {
        Ok(Ok(result)) => {
            std::ptr::copy_nonoverlapping(result.beta.as_ptr(), beta, p);
            std::ptr::copy_nonoverlapping(result.covariance.as_ptr(), covariance, p * p);
            std::ptr::copy_nonoverlapping(result.standard_errors.as_ptr(), se, p);
            std::ptr::copy_nonoverlapping(result.variances.as_ptr(), variances, k + 1);
            *summary = result.summary;
            0
        }
        Ok(Err(status)) => status,
        Err(_) => 255,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn options() -> Options {
        Options {
            struct_size: 48,
            reml: 1,
            reserved: 0,
            max_evaluations: 1000,
            tolerance: 1e-9,
            rank_tolerance: 1e-10,
            boundary_tolerance: 1e-8,
        }
    }
    fn balanced() -> (Vec<f64>, Vec<f64>, Vec<i64>) {
        let mut y = vec![];
        let mut g = vec![];
        for group in 0..5 {
            for e in [-0.3, -0.1, 0.1, 0.3] {
                y.push(2.0 + (group as f64 - 2.0) + e);
                g.push(group);
            }
        }
        (vec![1.0; y.len()], y, g)
    }
    #[test]
    fn balanced_random_intercept_analytic_reml_and_ml() {
        let (x, y, g) = balanced();
        for reml in [0, 1] {
            let mut o = options();
            o.reml = reml;
            let r = fit(&x, &y, &g, y.len(), 1, 1, o).unwrap();
            assert_eq!(r.summary.fit_state, 0, "{:?}", r.summary);
            assert!((r.beta[0] - 2.0).abs() < 1e-8);
            assert!(
                (r.variances[0] - 1.0 / 15.0).abs() < 1e-5,
                "{:?}",
                r.variances
            );
            let group = if reml == 1 {
                149.0 / 60.0
            } else {
                119.0 / 60.0
            };
            assert!((r.variances[1] - group).abs() < 1e-4, "{:?}", r.variances);
            assert!((r.covariance[0] - if reml == 1 { 0.5 } else { 0.4 }).abs() < 1e-4);
            assert!(r.summary.evaluations > 0 && r.summary.evaluations <= 1000);
        }
    }
    #[test]
    fn rank_saturation_constant_and_group_confounding_refuse() {
        let (x, y, g) = balanced();
        let duplicate: Vec<f64> = x.iter().flat_map(|v| [*v, *v]).collect();
        let r = fit(&duplicate, &y, &g, y.len(), 2, 1, options()).unwrap();
        assert_eq!(r.summary.fit_state, 2);
        assert!(r.beta.iter().all(|v| v.is_nan()));
        let constant = vec![3.0; y.len()];
        assert_eq!(
            fit(&x, &constant, &g, y.len(), 1, 1, options())
                .unwrap()
                .summary
                .reason,
            3
        );
        let duplicate_groups: Vec<i64> = g.iter().flat_map(|v| [*v, *v + 100]).collect();
        assert_eq!(
            fit(&x, &y, &duplicate_groups, y.len(), 1, 2, options())
                .unwrap()
                .summary
                .reason,
            5
        );
        let singleton: Vec<i64> = (0..y.len() as i64).collect();
        assert_eq!(
            fit(&x, &y, &singleton, y.len(), 1, 1, options())
                .unwrap()
                .summary
                .reason,
            5
        );
    }
    #[test]
    fn exhausted_budget_is_not_a_fit_and_repeated_calls_are_stable() {
        let (x, y, g) = balanced();
        let mut o = options();
        o.max_evaluations = 1;
        let r = fit(&x, &y, &g, y.len(), 1, 1, o).unwrap();
        assert_eq!(r.summary.fit_state, 4, "{:?}", r.summary);
        assert!(r.beta[0].is_nan());
        assert!(r.summary.evaluations <= 1, "{:?}", r.summary);
        let a = fit(&x, &y, &g, y.len(), 1, 1, options()).unwrap();
        let b = fit(&x, &y, &g, y.len(), 1, 1, options()).unwrap();
        assert_eq!(a.beta, b.beta);
        assert_eq!(a.variances, b.variances);
    }
    #[test]
    fn crossed_nested_independent_statsmodels_reference() {
        let reference: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../parity/fixtures/mixed_effects_reference_v1.json"
        ))
        .unwrap();
        for case in reference["cases"].as_array().unwrap() {
            let rows = case["X"].as_array().unwrap();
            let n = rows.len();
            let p = rows[0].as_array().unwrap().len();
            let x: Vec<f64> = rows
                .iter()
                .flat_map(|r| r.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
                .collect();
            let y: Vec<f64> = case["y"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect();
            let g: Vec<i64> = case["groups"]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|r| r.as_array().unwrap().iter().map(|v| v.as_i64().unwrap()))
                .collect();
            let mut o = options();
            o.reml = i32::from(case["reml"].as_bool().unwrap());
            o.max_evaluations = 3000;
            o.tolerance = 1e-10;
            let result = fit(&x, &y, &g, n, p, 2, o).unwrap();
            assert_eq!(
                result.summary.fit_state, 0,
                "{} {:?}",
                case["name"], result.summary
            );
            let expected = &case["expected"];
            for (actual, wanted) in result.beta.iter().zip(expected["beta"].as_array().unwrap()) {
                let wanted = wanted.as_f64().unwrap();
                assert!((actual - wanted).abs() <= 1e-5 + 1e-4 * wanted.abs());
            }
            for (actual, wanted) in result
                .variances
                .iter()
                .zip(expected["variances"].as_array().unwrap())
            {
                let wanted = wanted.as_f64().unwrap();
                assert!((actual - wanted).abs() <= 3e-5 + 2e-3 * wanted.abs());
            }
            let expected_cov: Vec<f64> = expected["covariance"]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|r| r.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
                .collect();
            for (actual, wanted) in result.covariance.iter().zip(expected_cov) {
                assert!((actual - wanted).abs() <= 3e-5 + 2e-3 * wanted.abs());
            }
            assert!(
                (result.summary.log_likelihood - expected["log_likelihood"].as_f64().unwrap())
                    .abs()
                    < 1e-4
            );
        }
    }
    #[test]
    fn genuine_boundary_and_fixed_absorbed_group() {
        let (x, _, g) = balanced();
        let y: Vec<f64> = (0..20)
            .map(|i| 2.0 + [-0.3, -0.1, 0.1, 0.3][i % 4])
            .collect();
        let r = fit(&x, &y, &g, 20, 1, 1, options()).unwrap();
        assert_eq!(r.summary.fit_state, 1, "{:?}", r.summary);
        assert_eq!(r.summary.boundary, 1);
        assert_eq!(r.summary.singular, 1);
        assert!(r.beta[0].is_finite());
        let dummy: Vec<f64> = g
            .iter()
            .flat_map(|id| (0..5).map(move |j| f64::from(*id == j)))
            .collect();
        let r = fit(&dummy, &y, &g, 20, 5, 1, options()).unwrap();
        assert_eq!(r.summary.reason, 5);
        let r = fit(
            &[1.0, 0.0, 0.0, 1.0],
            &[1.0, 2.0],
            &[0, 1],
            2,
            2,
            1,
            options(),
        )
        .unwrap();
        assert_eq!(r.summary.reason, 2);
    }
    #[test]
    fn rust_panic_never_writes_caller_buffers() {
        let (x, y, g) = balanced();
        let o = options();
        let mut beta = [77.0];
        let mut covariance = [77.0];
        let mut se = [77.0];
        let mut variance = [77.0, 77.0];
        let mut summary = Outcome::new(20, 1, 1).summary;
        summary.reason = 77;
        TEST_PANIC_NEXT.store(true, std::sync::atomic::Ordering::SeqCst);
        let code = unsafe {
            n4m_mixeff_fit_bridge(
                x.as_ptr(),
                y.as_ptr(),
                g.as_ptr(),
                20,
                1,
                1,
                &o,
                beta.as_mut_ptr(),
                covariance.as_mut_ptr(),
                se.as_mut_ptr(),
                variance.as_mut_ptr(),
                &mut summary,
            )
        };
        assert_eq!(code, 255);
        assert_eq!(beta, [77.0]);
        assert_eq!(covariance, [77.0]);
        assert_eq!(se, [77.0]);
        assert_eq!(variance, [77.0, 77.0]);
        assert_eq!(summary.reason, 77);
    }
    #[test]
    fn wire_layout_sizes() {
        assert_eq!(std::mem::size_of::<Options>(), 48);
        assert_eq!(std::mem::size_of::<Summary>(), 112);
    }
}
