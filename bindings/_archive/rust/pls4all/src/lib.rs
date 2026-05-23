// SPDX-License-Identifier: CECILL-2.1
//
//! Safe Rust wrappers around libn4m's `n4m_pls_fit_simple` C ABI helper.
//!
//! Parity-gated against the shared cross-binding fixture
//! (`bindings/js/test/parity_fixture.json`) at machine epsilon. See
//! `examples/test_parity.rs` for the gate.

#![deny(unsafe_op_in_unsafe_fn)]

use std::ffi::CStr;
use std::os::raw::{c_char, c_double, c_int};

#[link(name = "p4a")]
unsafe extern "C" {
    fn n4m_get_version_string() -> *const c_char;
    fn n4m_get_abi_version_major() -> u32;
    fn n4m_get_abi_version_minor() -> u32;
    fn n4m_get_abi_version_patch() -> u32;
    fn n4m_pls_fit_simple(
        x: *const c_double,
        y: *const c_double,
        n: i32,
        p: i32,
        q: i32,
        n_components: i32,
        coefficients_out: *mut c_double,
        x_mean_out: *mut c_double,
        y_mean_out: *mut c_double,
        predictions_out: *mut c_double,
    ) -> c_int;
}

/// Returns the libn4m runtime version, e.g. `"0.89.0+abi.1.13.0"`.
pub fn version() -> String {
    // SAFETY: `n4m_get_version_string` always returns a non-null
    // NUL-terminated static string owned by libn4m.
    unsafe {
        let raw = n4m_get_version_string();
        if raw.is_null() {
            String::new()
        } else {
            CStr::from_ptr(raw).to_string_lossy().into_owned()
        }
    }
}

/// Returns the libn4m ABI `(major, minor, patch)` version.
pub fn abi_version() -> (u32, u32, u32) {
    // SAFETY: these are pure leaf getters returning POD values.
    unsafe {
        (
            n4m_get_abi_version_major(),
            n4m_get_abi_version_minor(),
            n4m_get_abi_version_patch(),
        )
    }
}

/// Bundle of outputs returned by [`pls_fit`].
///
/// All matrices are stored row-major:
/// * `coefficients` length = `p * q`
/// * `x_mean`       length = `p`
/// * `y_mean`       length = `q`
/// * `predictions`  length = `n * q`
#[derive(Debug, Clone)]
pub struct FitResult {
    pub n: usize,
    pub p: usize,
    pub q: usize,
    pub n_components: usize,
    pub coefficients: Vec<f64>,
    pub x_mean: Vec<f64>,
    pub y_mean: Vec<f64>,
    pub predictions: Vec<f64>,
}

/// Error returned by [`pls_fit`] when input validation fails or the
/// underlying C call returns a non-zero status.
#[derive(Debug, Clone)]
pub enum FitError {
    XLength { got: usize, expected: usize },
    YLength { got: usize, expected: usize },
    NComponents(usize),
    Native(i32),
}

impl std::fmt::Display for FitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FitError::XLength { got, expected } => {
                write!(f, "X length {got} but n*p = {expected}")
            }
            FitError::YLength { got, expected } => {
                write!(f, "Y length {got} but n*q = {expected}")
            }
            FitError::NComponents(k) => {
                write!(f, "n_components must be >= 1, got {k}")
            }
            FitError::Native(s) => {
                write!(f, "n4m_pls_fit_simple failed with status {s}")
            }
        }
    }
}

impl std::error::Error for FitError {}

/// Fit a SIMPLS PLS regression on row-major matrices `x` and `y`.
///
/// `x` must have length `n * p`; `y` must have length `n * q`. The
/// returned [`FitResult`] owns all output buffers.
pub fn pls_fit(
    x: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    q: usize,
    n_components: usize,
) -> Result<FitResult, FitError> {
    let expected_x = n.checked_mul(p).unwrap_or(usize::MAX);
    if x.len() != expected_x {
        return Err(FitError::XLength { got: x.len(), expected: expected_x });
    }
    let expected_y = n.checked_mul(q).unwrap_or(usize::MAX);
    if y.len() != expected_y {
        return Err(FitError::YLength { got: y.len(), expected: expected_y });
    }
    if n_components < 1 {
        return Err(FitError::NComponents(n_components));
    }

    let mut coefficients = vec![0.0; p * q];
    let mut x_mean = vec![0.0; p];
    let mut y_mean = vec![0.0; q];
    let mut predictions = vec![0.0; n * q];

    // SAFETY: all six pointers point to valid slice memory whose
    // length we just checked. n4m_pls_fit_simple writes into the
    // four `_out` buffers up to (p*q, p, q, n*q) doubles respectively,
    // which matches the allocations above.
    let status = unsafe {
        n4m_pls_fit_simple(
            x.as_ptr(),
            y.as_ptr(),
            n as i32,
            p as i32,
            q as i32,
            n_components as i32,
            coefficients.as_mut_ptr(),
            x_mean.as_mut_ptr(),
            y_mean.as_mut_ptr(),
            predictions.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(FitError::Native(status));
    }
    Ok(FitResult {
        n, p, q, n_components,
        coefficients, x_mean, y_mean, predictions,
    })
}

pub mod sklearn;
pub use sklearn::PLSRegression;
