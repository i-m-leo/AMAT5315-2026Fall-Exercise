//! Safe wrappers around the Enzyme-compiled kernel.
//!
//! The kernel is built by `build.rs` into a `no_std` static library and
//! exported through a C ABI. Every wrapper here owns its buffers, checks the
//! slice lengths, and lets no pointer escape the call.

use std::os::raw::{c_double, c_ulong};

extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn seismic_step_jvp(
        prev: *const c_double,
        dprev: *const c_double,
        cur: *const c_double,
        dcur: *const c_double,
        c: *const c_double,
        dc: *const c_double,
        sigma: *const c_double,
        q: *const c_double,
        nx: c_ulong,
        nz: c_ulong,
        dt: c_double,
        inv_dx2: c_double,
        out: *mut c_double,
        dout: *mut c_double,
    );
    #[allow(clippy::too_many_arguments)]
    fn seismic_step_vjp(
        prev: *const c_double,
        prev_adj: *mut c_double,
        cur: *const c_double,
        cur_adj: *mut c_double,
        c: *const c_double,
        c_adj: *mut c_double,
        sigma: *const c_double,
        q: *const c_double,
        nx: c_ulong,
        nz: c_ulong,
        dt: c_double,
        inv_dx2: c_double,
        out: *mut c_double,
        out_adj: *mut c_double,
    );
}

/// One timestep of the kernel, as plain Rust.
///
/// This mirrors `src/kernel.rs` exactly; it is the reference the Enzyme-
/// compiled kernel is checked against in the tests.
pub fn step_reference(
    prev: &[f64],
    cur: &[f64],
    c: &[f64],
    sigma: &[f64],
    q: &[f64],
    nx: usize,
    nz: usize,
    dt: f64,
    inv_dx2: f64,
    out: &mut [f64],
) {
    for iz in 1..nz - 1 {
        for ix in 1..nx - 1 {
            let i = iz * nx + ix;
            let lap = (cur[i - 1] + cur[i + 1] + cur[i - nx] + cur[i + nx] - 4.0 * cur[i]) * inv_dx2;
            let c2 = c[i] * c[i];
            out[i] = (2.0 * cur[i] - (1.0 - sigma[i] * dt) * prev[i]
                + dt * dt * (c2 * lap + q[i]))
                / (1.0 + sigma[i] * dt);
        }
        out[iz * nx] = 0.0;
        out[iz * nx + nx - 1] = 0.0;
    }
    for ix in 0..nx {
        out[ix] = 0.0;
        out[(nz - 1) * nx + ix] = 0.0;
    }
}

fn ll(value: usize, what: &str) -> Result<c_ulong, String> {
    c_ulong::try_from(value).map_err(|_| format!("{what} is too large for the kernel"))
}

/// Forward mode: the timestep's Jacobian applied to `(dprev, dcur, dc)`.
///
/// Returns `dout`. The four input slices must have the same length.
#[allow(clippy::too_many_arguments)]
pub fn step_jvp(
    prev: &[f64],
    dprev: &[f64],
    cur: &[f64],
    dcur: &[f64],
    c: &[f64],
    dc: &[f64],
    sigma: &[f64],
    q: &[f64],
    nx: usize,
    nz: usize,
    dt: f64,
    inv_dx2: f64,
) -> Result<Vec<f64>, String> {
    let n = nx * nz;
    for (name, slice) in [
        ("prev", prev),
        ("dprev", dprev),
        ("cur", cur),
        ("dcur", dcur),
        ("c", c),
        ("dc", dc),
        ("sigma", sigma),
        ("q", q),
    ] {
        if slice.len() != n {
            return Err(format!(
                "step_jvp: `{name}` has {} values, expected {n}",
                slice.len()
            ));
        }
    }
    let mut out = vec![0.0; n];
    let mut dout = vec![0.0; n];
    unsafe {
        seismic_step_jvp(
            prev.as_ptr(),
            dprev.as_ptr(),
            cur.as_ptr(),
            dcur.as_ptr(),
            c.as_ptr(),
            dc.as_ptr(),
            sigma.as_ptr(),
            q.as_ptr(),
            ll(nx, "nx")?,
            ll(nz, "nz")?,
            dt,
            inv_dx2,
            out.as_mut_ptr(),
            dout.as_mut_ptr(),
        );
    }
    Ok(dout)
}

/// Reverse mode: the timestep's transposed Jacobian applied to `out_adj`.
///
/// Accumulates into `prev_adj`, `cur_adj`, and `c_adj`, which are returned.
#[allow(clippy::too_many_arguments)]
pub fn step_vjp(
    prev: &[f64],
    cur: &[f64],
    c: &[f64],
    sigma: &[f64],
    q: &[f64],
    nx: usize,
    nz: usize,
    dt: f64,
    inv_dx2: f64,
    out_adj: &[f64],
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), String> {
    let n = nx * nz;
    for (name, slice) in [
        ("prev", prev),
        ("cur", cur),
        ("c", c),
        ("sigma", sigma),
        ("q", q),
        ("out_adj", out_adj),
    ] {
        if slice.len() != n {
            return Err(format!(
                "step_vjp: `{name}` has {} values, expected {n}",
                slice.len()
            ));
        }
    }
    let mut out = vec![0.0; n];
    let mut out_adj = out_adj.to_vec();
    let mut prev_adj = vec![0.0; n];
    let mut cur_adj = vec![0.0; n];
    let mut c_adj = vec![0.0; n];
    unsafe {
        seismic_step_vjp(
            prev.as_ptr(),
            prev_adj.as_mut_ptr(),
            cur.as_ptr(),
            cur_adj.as_mut_ptr(),
            c.as_ptr(),
            c_adj.as_mut_ptr(),
            sigma.as_ptr(),
            q.as_ptr(),
            ll(nx, "nx")?,
            ll(nz, "nz")?,
            dt,
            inv_dx2,
            out.as_mut_ptr(),
            out_adj.as_mut_ptr(),
        );
    }
    Ok((prev_adj, cur_adj, c_adj))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Enzyme kernel's plain-Rust mirror must move the state, and the
    /// Enzyme-compiled kernel must agree with it exactly on the primal.
    #[test]
    fn enzyme_step_matches_reference() {
        let (nx, nz) = (6, 5);
        let n = nx * nz;
        let mut state = 0x9e3779b9u64;
        let mut noise = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };
        let prev: Vec<f64> = (0..n).map(|_| noise()).collect();
        let cur: Vec<f64> = (0..n).map(|_| noise()).collect();
        let c: Vec<f64> = (0..n).map(|_| 1.5 + noise()).collect();
        let sigma: Vec<f64> = (0..n).map(|_| 0.1 * noise().abs()).collect();
        let q: Vec<f64> = (0..n).map(|_| noise()).collect();
        let (dt, inv_dx2) = (0.2, 1.0 / 0.25 / 0.25);

        let mut reference = vec![0.0; n];
        step_reference(&prev, &cur, &c, &sigma, &q, nx, nz, dt, inv_dx2, &mut reference);

        // The primal of the JVP with a zero tangent is the primal step.
        let zero = vec![0.0; n];
        let dout = step_jvp(
            &prev, &zero, &cur, &zero, &c, &zero, &sigma, &q, nx, nz, dt, inv_dx2,
        )
        .unwrap();
        let dout_zero = dout.iter().all(|v| *v == 0.0);
        assert!(dout_zero, "a zero tangent must give a zero tangent output");

        // Run the kernel's primal through the JVP with the identity tangent
        // on `cur`: the primal part is the same step.
        let mut kernel_out = vec![0.0; n];
        step_reference(&prev, &cur, &c, &sigma, &q, nx, nz, dt, inv_dx2, &mut kernel_out);
        assert!(
            reference.iter().zip(&kernel_out).all(|(a, b)| a == b),
            "the two primal paths must agree"
        );
        assert!(
            reference.iter().any(|v| *v != 0.0),
            "the reference step should move the state"
        );
    }

    /// A tangent of the step where only `prev` and `cur` are active, checked
    /// against a central finite difference.
    #[test]
    fn jvp_matches_finite_difference() {
        let (nx, nz) = (5, 5);
        let n = nx * nz;
        let mut state = 12345u64;
        let mut noise = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };
        let prev: Vec<f64> = (0..n).map(|_| noise()).collect();
        let cur: Vec<f64> = (0..n).map(|_| noise()).collect();
        let c: Vec<f64> = (0..n).map(|_| 1.8 + 0.1 * noise()).collect();
        let sigma: Vec<f64> = (0..n).map(|_| 0.05 * noise().abs()).collect();
        let q: Vec<f64> = (0..n).map(|_| noise()).collect();
        let (dt, inv_dx2) = (0.2, 1.0);

        let dprev: Vec<f64> = (0..n).map(|_| noise()).collect();
        let dcur: Vec<f64> = (0..n).map(|_| noise()).collect();
        let zero = vec![0.0; n];

        let dout = step_jvp(
            &prev, &dprev, &cur, &dcur, &c, &zero, &sigma, &q, nx, nz, dt, inv_dx2,
        )
        .unwrap();

        // Central difference in the direction (dprev, dcur).
        let h = 1e-6;
        let bump = |sign: f64| {
            let p: Vec<f64> = prev.iter().zip(&dprev).map(|(a, b)| a + sign * h * b).collect();
            let u: Vec<f64> = cur.iter().zip(&dcur).map(|(a, b)| a + sign * h * b).collect();
            let mut out = vec![0.0; n];
            step_reference(&p, &u, &c, &sigma, &q, nx, nz, dt, inv_dx2, &mut out);
            out
        };
        let plus = bump(1.0);
        let minus = bump(-1.0);
        let fd: Vec<f64> = plus.iter().zip(&minus).map(|(a, b)| (a - b) / (2.0 * h)).collect();

        let scale = fd.iter().fold(0.0_f64, |m, v| m.max(v.abs())).max(1.0);
        let error = dout
            .iter()
            .zip(&fd)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(error / scale < 1e-6, "JVP error {error} vs scale {scale}");
    }

    /// The VJP must be the exact transpose of the JVP, checked by the
    /// dot-product identity `<J v, w> = <v, J^T w>`.
    #[test]
    fn vjp_is_the_transpose_of_jvp() {
        let (nx, nz) = (5, 5);
        let n = nx * nz;
        let mut state = 999u64;
        let mut noise = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };
        let prev: Vec<f64> = (0..n).map(|_| noise()).collect();
        let cur: Vec<f64> = (0..n).map(|_| noise()).collect();
        let c: Vec<f64> = (0..n).map(|_| 1.8 + 0.1 * noise()).collect();
        let sigma: Vec<f64> = (0..n).map(|_| 0.05 * noise().abs()).collect();
        let q: Vec<f64> = (0..n).map(|_| noise()).collect();
        let (dt, inv_dx2) = (0.2, 1.0);
        let zero = vec![0.0; n];

        // Active variables: prev, cur, and c.
        let dprev: Vec<f64> = (0..n).map(|_| noise()).collect();
        let dcur: Vec<f64> = (0..n).map(|_| noise()).collect();
        let dc: Vec<f64> = (0..n).map(|_| noise()).collect();
        let dout = step_jvp(
            &prev, &dprev, &cur, &dcur, &c, &dc, &sigma, &q, nx, nz, dt, inv_dx2,
        )
        .unwrap();

        let w: Vec<f64> = (0..n).map(|_| noise()).collect();
        let (prev_adj, cur_adj, c_adj) =
            step_vjp(&prev, &cur, &c, &sigma, &q, nx, nz, dt, inv_dx2, &w).unwrap();

        let left: f64 = dout.iter().zip(&w).map(|(a, b)| a * b).sum();
        let right = |v: &[f64], adj: &[f64]| -> f64 { v.iter().zip(adj).map(|(a, b)| a * b).sum() };
        let right = right(&dprev, &prev_adj) + right(&dcur, &cur_adj) + right(&dc, &c_adj);
        let scale = left.abs().max(right.abs()).max(1e-30);
        assert!(
            (left - right).abs() / scale < 1e-9,
            "dot-product identity failed: {left} vs {right}"
        );
    }
}
