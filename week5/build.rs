//! Compile the Enzyme kernel as an isolated `no_std` static library.
//!
//! The autodiff pass must not see the whole program (CLI, JSON, allocation),
//! so the differentiated timestep lives in `src/kernel.rs` and is built here
//! by the pinned nightly with `-Zautodiff=Enable`. The main crate only links
//! the resulting archive and calls it through a C ABI.

use std::{env, path::PathBuf, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=src/kernel.rs");
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set"));
    let status = Command::new(env::var_os("RUSTC").expect("RUSTC is set"))
        .args([
            "--edition=2021",
            "--crate-name",
            "seismic_kernel",
            "--crate-type",
            "staticlib",
            "-C",
            "opt-level=3",
            "-C",
            "lto=fat",
            "-C",
            "panic=abort",
            "-Zautodiff=Enable",
            "src/kernel.rs",
            "-o",
        ])
        .arg(out.join("libseismic_kernel.a"))
        .status()
        .expect("run the pinned rustc on the Enzyme kernel");
    assert!(status.success(), "Enzyme kernel compilation failed");
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=seismic_kernel");
}
