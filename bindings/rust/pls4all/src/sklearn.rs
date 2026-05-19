// SPDX-License-Identifier: CECILL-2.1
//
// Idiomatic Rust tier-2 wrapper around `pls4all::pls_fit`. Provides a
// `PLSRegression` struct with a `fit` / `predict` / `score` surface
// consistent with the `linfa` ecosystem's `Fit` / `Predict` traits,
// without taking a hard dependency on linfa itself.
//
// Example:
//
// ```ignore
// use pls4all::sklearn::PLSRegression;
// let model = PLSRegression::new(5).fit(&x, &y, n, p)?;
// let preds = model.predict(&x_new, n_new, p)?;
// let r2 = model.score(&x_test, &y_test, n_test, p)?;
// ```

use crate::{pls_fit, FitError};

#[derive(Debug, Clone)]
pub struct PLSRegression {
    pub n_components: usize,

    coefficients: Option<Vec<f64>>,
    x_mean: Option<Vec<f64>>,
    y_mean: Option<Vec<f64>>,
    n_features_in: usize,
    n_targets: usize,
}

impl PLSRegression {
    /// Create an unfitted PLS regression with the given component count.
    pub fn new(n_components: usize) -> Self {
        PLSRegression {
            n_components,
            coefficients: None,
            x_mean: None,
            y_mean: None,
            n_features_in: 0,
            n_targets: 0,
        }
    }

    /// Train the model in-place. `x` is row-major `(n, p)`; `y` is
    /// row-major `(n, q)` or a length-`n` vector when `q == 1`.
    pub fn fit(mut self, x: &[f64], y: &[f64], n: usize,
                p: usize) -> Result<Self, FitError> {
        if n == 0 || p == 0 {
            return Err(FitError::NComponents(0));
        }
        if x.len() != n * p {
            return Err(FitError::XLength { got: x.len(), expected: n * p });
        }
        if y.is_empty() || y.len() % n != 0 {
            return Err(FitError::YLength { got: y.len(), expected: n });
        }
        let q = y.len() / n;
        let res = pls_fit(x, y, n, p, q, self.n_components)?;
        self.coefficients = Some(res.coefficients);
        self.x_mean = Some(res.x_mean);
        self.y_mean = Some(res.y_mean);
        self.n_features_in = p;
        self.n_targets = q;
        Ok(self)
    }

    /// Predict on new X (row-major `(n_pred, p)`).
    pub fn predict(&self, x: &[f64], n_pred: usize,
                    p: usize) -> Result<Vec<f64>, FitError> {
        let coef = self.coefficients.as_ref().ok_or(FitError::Native(-1))?;
        let xmean = self.x_mean.as_ref().unwrap();
        let ymean = self.y_mean.as_ref().unwrap();
        if p != self.n_features_in {
            return Err(FitError::Native(-2));
        }
        let q = self.n_targets;
        let mut preds = vec![0.0_f64; n_pred * q];
        for i in 0..n_pred {
            for t in 0..q {
                let mut sum = ymean[t];
                for j in 0..p {
                    let centered = x[i * p + j] - xmean[j];
                    sum += centered * coef[j * q + t];
                }
                preds[i * q + t] = sum;
            }
        }
        Ok(preds)
    }

    /// R² coefficient of determination on (X, y), uniform-averaged
    /// across targets.
    pub fn score(&self, x: &[f64], y: &[f64], n: usize,
                  p: usize) -> Result<f64, FitError> {
        let preds = self.predict(x, n, p)?;
        let q = self.n_targets;
        let mut r2_sum = 0.0_f64;
        for t in 0..q {
            let mean: f64 = (0..n).map(|i| y[i * q + t]).sum::<f64>()
                / n as f64;
            let mut ss_res = 0.0_f64;
            let mut ss_tot = 0.0_f64;
            for i in 0..n {
                let dr = preds[i * q + t] - y[i * q + t];
                let dt = y[i * q + t] - mean;
                ss_res += dr * dr;
                ss_tot += dt * dt;
            }
            if ss_tot != 0.0 && ss_tot.is_finite() {
                r2_sum += 1.0 - ss_res / ss_tot;
            }
        }
        Ok(r2_sum / q as f64)
    }

    pub fn coefficients(&self) -> Option<&[f64]> {
        self.coefficients.as_deref()
    }
    pub fn x_mean(&self) -> Option<&[f64]> {
        self.x_mean.as_deref()
    }
    pub fn y_mean(&self) -> Option<&[f64]> {
        self.y_mean.as_deref()
    }
}
