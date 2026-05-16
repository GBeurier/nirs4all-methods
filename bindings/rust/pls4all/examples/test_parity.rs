// SPDX-License-Identifier: CeCILL-2.1
//
// Cross-binding parity gate for the Rust binding. Builds the same
// deterministic (X, Y) the Python generator writes, fits the model,
// and compares against `bindings/js/test/parity_fixture.json`.
// Exits non-zero on failure.
//
// Run from the repo root:
//
//   cargo run --manifest-path bindings/rust/pls4all/Cargo.toml \
//             --example test_parity --release

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

fn rms_rel(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    let mut diff = 0.0;
    let mut reff = 0.0;
    for (av, bv) in a.iter().zip(b.iter()) {
        let d = av - bv;
        diff += d * d;
        reff += bv * bv;
    }
    let denom = reff.sqrt().max(f64::MIN_POSITIVE);
    diff.sqrt() / denom
}

fn parse_int(json: &str, key: &str) -> i64 {
    let needle = format!("\"{key}\"");
    let i = json.find(&needle).expect("missing key");
    let tail = &json[i + needle.len()..];
    let after_colon = tail.find(':').expect("missing :") + 1;
    let s = &tail[after_colon..];
    let s = s.trim_start();
    let end = s
        .find(|c: char| !c.is_ascii_digit() && c != '-')
        .unwrap_or(s.len());
    s[..end].parse::<i64>().expect("int parse")
}

fn parse_string(json: &str, key: &str) -> String {
    let needle = format!("\"{key}\"");
    let i = json.find(&needle).expect("missing key");
    let tail = &json[i + needle.len()..];
    let start = tail.find('"').expect("opening quote") + 1;
    let s = &tail[start..];
    let end = s.find('"').expect("closing quote");
    s[..end].to_string()
}

fn parse_array(json: &str, key: &str) -> Vec<f64> {
    let needle = format!("\"{key}\"");
    let i = json.find(&needle).expect("missing key");
    let tail = &json[i + needle.len()..];
    let start = tail.find('[').expect("opening bracket") + 1;
    let s = &tail[start..];
    let end = s.find(']').expect("closing bracket");
    s[..end]
        .split(',')
        .map(|tok| tok.trim().parse::<f64>().expect("float parse"))
        .collect()
}

fn main() -> ExitCode {
    let repo_root = env::var("REPO_ROOT")
        .map(PathBuf::from)
        .or_else(|_| -> Result<_, std::io::Error> {
            // Walk up from the example until we find bindings/js/test.
            let mut cur = env::current_dir()?;
            loop {
                if cur.join("bindings/js/test/parity_fixture.json").exists() {
                    return Ok(cur);
                }
                if !cur.pop() {
                    return Err(std::io::Error::other(
                        "could not locate repo root containing \
                         bindings/js/test/parity_fixture.json",
                    ));
                }
            }
        })
        .expect("repo root");

    let fixture_path = repo_root.join("bindings/js/test/parity_fixture.json");
    let raw = fs::read_to_string(&fixture_path).expect("read fixture");

    let n = parse_int(&raw, "n") as usize;
    let p = parse_int(&raw, "p") as usize;
    let q = parse_int(&raw, "q") as usize;
    let n_components = parse_int(&raw, "n_components") as usize;
    let fixture_version = parse_string(&raw, "pls4all_version");
    let fx_coefs = parse_array(&raw, "coefficients");
    let fx_x_mean = parse_array(&raw, "x_mean");
    let fx_y_mean = parse_array(&raw, "y_mean");
    let fx_preds = parse_array(&raw, "predictions");

    let mut x = vec![0.0; n * p];
    let mut y = vec![0.0; n * q];
    for i in 0..n {
        for j in 0..p {
            x[i * p + j] = (((i + 1) * (j + 1)) as f64 * 0.3).sin();
        }
        y[i * q + 0] =
            x[i * p + 0] + 0.5 * x[i * p + 1] - 0.3 * x[i * p + 2];
    }

    let fit = pls4all::pls_fit(&x, &y, n, p, q, n_components)
        .expect("pls_fit");

    let err_c = rms_rel(&fit.coefficients, &fx_coefs);
    let err_x = rms_rel(&fit.x_mean, &fx_x_mean);
    let err_y = rms_rel(&fit.y_mean, &fx_y_mean);
    let err_p = rms_rel(&fit.predictions, &fx_preds);

    let (major, minor, patch) = pls4all::abi_version();
    println!("pls4all::version()       = {}", pls4all::version());
    println!("pls4all::abi_version()   = {major}.{minor}.{patch}");
    println!("fixture pls4all_version  = {fixture_version}");
    println!("rmse_rel coefficients    = {err_c:.3e}");
    println!("rmse_rel x_mean          = {err_x:.3e}");
    println!("rmse_rel y_mean          = {err_y:.3e}");
    println!("rmse_rel predictions     = {err_p:.3e}");

    let tol = 1e-12;
    if err_c > tol || err_x > tol || err_y > tol || err_p > tol {
        eprintln!("PARITY GATE FAIL (tol = {tol:.1e})");
        return ExitCode::from(1);
    }
    println!("PARITY GATE PASS");
    ExitCode::SUCCESS
}
