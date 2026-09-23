//! `perturb` reads a velocity field that `field` wrote to stdout, adds a
//! vorticity ripple, and writes the perturbed field as the same JSON shape.
//!
//! The ripple is
//!
//! ```text
//!   delta omega(x, y) = -7e-5 * M * cos(3x) cos(4y)
//! ```
//!
//! where `M` is the largest absolute value among all initial `u` and `v`
//! values of the input field.
//!
//! The streamfunction is adjusted so the spectral curl of the new `u` and `v`
//! exceeds the old one by exactly this ripple:
//! `delta psi_hat = delta omega_hat / |k|^2`, `delta u_hat = i ky delta psi_hat`,
//! `delta v_hat = -i kx delta psi_hat`.
//!
//! Usage: field <case> | perturb
//! Writes one JSON object to stdout: case, n, seed, k_band, u, v.

use advection::vorticity::{Grid2d, Spectral};
use std::io::{self, Read, Write};

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).expect("read stdin");
    let parsed: serde_json::Value =
        serde_json::from_str(&input).expect("parse stdin JSON");

    let n = parsed
        .get("n")
        .and_then(|v| v.as_u64())
        .expect("stdin JSON is missing 'n'") as usize;
    let grid = Grid2d::new(n);
    let spectral = Spectral::new(grid, 0.0);
    let u = read_array(&parsed, "u", grid.len());
    let v = read_array(&parsed, "v", grid.len());

    // M is the largest |u| or |v| of this case's initial field.
    let m = u
        .iter()
        .chain(v.iter())
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);

    let mut delta_omega = vec![0.0; grid.len()];
    for iy in 0..n {
        for ix in 0..n {
            let (x, y) = (grid.x(ix), grid.y(iy));
            delta_omega[grid.flat(iy, ix)] = -7.0e-5 * m * (3.0 * x).cos() * (4.0 * y).cos();
        }
    }
    let delta_omega_hat = spectral.forward_real(&delta_omega);

    let u_hat = spectral.forward_real(&u);
    let v_hat = spectral.forward_real(&v);
    let mut new_u_hat = u_hat.clone();
    let mut new_v_hat = v_hat.clone();
    for i in 0..grid.len() {
        let (kx, ky) = (spectral.kx_at(i), spectral.ky_at(i));
        let k2 = kx * kx + ky * ky;
        if k2 == 0.0 {
            continue;
        }
        let delta_psi = delta_omega_hat[i] / k2;
        new_u_hat[i] += rustfft::num_complex::Complex::new(0.0, ky) * delta_psi;
        new_v_hat[i] -= rustfft::num_complex::Complex::new(0.0, kx) * delta_psi;
    }

    let new_u = spectral.inverse_real(&new_u_hat);
    let new_v = spectral.inverse_real(&new_v_hat);

    let case = parsed.get("case").cloned().unwrap_or(serde_json::Value::Null);
    let seed = parsed.get("seed").cloned().unwrap_or(serde_json::Value::Null);
    let k_band = parsed
        .get("k_band")
        .cloned()
        .unwrap_or(serde_json::Value::Null);

    let mut json = String::new();
    json.push_str("{\n");
    json.push_str(&format!("  \"case\": {case},\n"));
    json.push_str(&format!("  \"n\": {n},\n"));
    json.push_str(&format!("  \"seed\": {seed},\n"));
    json.push_str(&format!("  \"k_band\": {k_band},\n"));
    push_array(&mut json, "u", &new_u);
    json.push_str(",\n");
    push_array(&mut json, "v", &new_v);
    json.push_str("\n}\n");

    let stdout = io::stdout();
    let mut handle = stdout.lock();
    handle.write_all(json.as_bytes()).expect("write stdout");
    eprintln!("M = {m:.10}, ripple amplitude = {:.10e}", 7.0e-5 * m);
}

fn read_array(parsed: &serde_json::Value, key: &str, expected: usize) -> Vec<f64> {
    let values = parsed
        .get(key)
        .and_then(|v| v.as_array())
        .unwrap_or_else(|| panic!("stdin JSON is missing '{key}'"));
    assert_eq!(values.len(), expected, "'{key}' has the wrong length");
    values
        .iter()
        .map(|value| value.as_f64().expect("non-numeric entry"))
        .collect()
}

fn push_array(out: &mut String, name: &str, values: &[f64]) {
    out.push_str(&format!("  \"{name}\": ["));
    for (i, value) in values.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(&format!("{value:.12}"));
    }
    out.push(']');
}
