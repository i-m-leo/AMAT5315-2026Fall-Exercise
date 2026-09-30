//! The differentiated timestep, isolated for Enzyme.
//!
//! This file is an input to `build.rs` only and is never part of the main
//! crate: the autodiff pass must see only the timestep, never the CLI, JSON,
//! or allocation. It is `no_std`, has no `Vec` and no formatting, and the
//! wrappers below move data across a C ABI through raw pointers and lengths.
//!
//! One update, with the boundary ring zeroed:
//!
//! ```text
//! out = [2 cur - (1 - sigma dt) prev + dt^2 (c^2 lap(cur) + q)] / (1 + sigma dt)
//! ```

#![no_std]
#![feature(autodiff)]

use core::autodiff::{autodiff_forward, autodiff_reverse};

/// One timestep. `prev` is `u^{n-1}`, `cur` is `u^n`, `c` is the wave speed,
/// `sigma` the sponge, `q` the source, and `out` receives `u^{n+1}`.
#[autodiff_forward(step_fwd, Dual, Dual, Dual, Const, Const, Const, Const, Const, Const, Dual)]
#[autodiff_reverse(step_rev, Duplicated, Duplicated, Duplicated, Const, Const, Const, Const, Const, Const, Duplicated)]
fn step(
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

/// Forward mode: `dout = J step`, one tangent pushed through the update.
/// The tangent buffers (`dprev`, `dcur`, `dc`) are inputs; `dout` is written.
///
/// # Safety
/// Every pointer must reference `nx * nz` readable or writable `f64`s, and
/// the output buffers must be distinct from the inputs.
#[no_mangle]
pub unsafe extern "C" fn seismic_step_jvp(
    prev: *const f64,
    dprev: *const f64,
    cur: *const f64,
    dcur: *const f64,
    c: *const f64,
    dc: *const f64,
    sigma: *const f64,
    q: *const f64,
    nx: usize,
    nz: usize,
    dt: f64,
    inv_dx2: f64,
    out: *mut f64,
    dout: *mut f64,
) {
    let n = nx * nz;
    let s = |p: *const f64| unsafe { core::slice::from_raw_parts(p, n) };
    let m = |p: *mut f64| unsafe { core::slice::from_raw_parts_mut(p, n) };
    let (dprev, dcur, dc) = (s(dprev), s(dcur), s(dc));
    let (mut out, mut dout) = (m(out), m(dout));
    step_fwd(
        s(prev),
        dprev,
        s(cur),
        dcur,
        s(c),
        dc,
        s(sigma),
        s(q),
        nx,
        nz,
        dt,
        inv_dx2,
        &mut out,
        &mut dout,
    );
}

/// Reverse mode: `(prev, cur, c)` adjoints from the output adjoint `out_adj`.
/// The input adjoints are accumulated into the buffers passed in.
///
/// # Safety
/// Every pointer must reference `nx * nz` readable or writable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn seismic_step_vjp(
    prev: *const f64,
    prev_adj: *mut f64,
    cur: *const f64,
    cur_adj: *mut f64,
    c: *const f64,
    c_adj: *mut f64,
    sigma: *const f64,
    q: *const f64,
    nx: usize,
    nz: usize,
    dt: f64,
    inv_dx2: f64,
    out: *mut f64,
    out_adj: *mut f64,
) {
    let n = nx * nz;
    let s = |p: *const f64| unsafe { core::slice::from_raw_parts(p, n) };
    let m = |p: *mut f64| unsafe { core::slice::from_raw_parts_mut(p, n) };
    let (mut prev_adj, mut cur_adj, mut c_adj) = (m(prev_adj), m(cur_adj), m(c_adj));
    let (mut out, mut out_adj) = (m(out), m(out_adj));
    step_rev(
        s(prev),
        &mut prev_adj,
        s(cur),
        &mut cur_adj,
        s(c),
        &mut c_adj,
        s(sigma),
        s(q),
        nx,
        nz,
        dt,
        inv_dx2,
        &mut out,
        &mut out_adj,
    );
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
