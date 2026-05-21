// SPDX-License-Identifier: CECILL-2.1
//
// Build script for the pls4all Rust binding. Tells cargo where to
// find libn4m's shared object. The PLS4ALL_LIB_DIR env var overrides
// the default location (the CMake `dev-release` preset under the
// repo root). Setting `RUSTFLAGS="-L /path/to/libn4m"` works too.

use std::env;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=PLS4ALL_LIB_DIR");
    println!("cargo:rerun-if-env-changed=PLS4ALL_RUNTIME_RPATH");

    let lib_dir = resolve_lib_dir();
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=p4a");

    // Bake an rpath so consumers don't need LD_LIBRARY_PATH at runtime
    // when picking up the build-time libn4m path. Override with
    // PLS4ALL_RUNTIME_RPATH for downstream packaging.
    let rpath = env::var("PLS4ALL_RUNTIME_RPATH")
        .unwrap_or_else(|_| lib_dir.display().to_string());
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", rpath);
}

fn resolve_lib_dir() -> PathBuf {
    if let Ok(p) = env::var("PLS4ALL_LIB_DIR") {
        return PathBuf::from(p);
    }
    // Walk up from CARGO_MANIFEST_DIR to find <repo>/build/dev-release/cpp/src.
    let manifest = env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR not set");
    let mut cur: &Path = Path::new(&manifest);
    while let Some(parent) = cur.parent() {
        let candidate = parent.join("build/dev-release/cpp/src");
        if candidate.is_dir() {
            return candidate;
        }
        cur = parent;
    }
    panic!(
        "could not locate libn4m. Set PLS4ALL_LIB_DIR or build the \
         CMake dev-release preset before running cargo build."
    );
}
