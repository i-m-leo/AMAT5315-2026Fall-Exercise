use advection::{mode, wave_exact, CentredDifferenceRhs, FourierRhs, PeriodicGrid, Scheme};

fn l2_error(a: &[f64], b: &[f64]) -> f64 {
    let s: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
    (s / a.len() as f64).sqrt()
}

fn main() {
    let grid = PeriodicGrid::new(64);
    let (c, nu) = (0.5, 0.01);
    let k = 1.0;
    let t_end = 1.0;
    let steps = 400;
    let dt = t_end / steps as f64;
    let initial = mode(&grid, k, 1.0, 0.0);
    let exact = wave_exact(&grid, k, 1.0, 0.0, c, nu, t_end);

    println!("single wave: k={k}, c={c}, nu={nu}, t_end={t_end}, steps={steps}");
    println!("{:<18} {:>12} {:>12}", "integrator", "spectral", "centred");
    for scheme in Scheme::all() {
        let spectral = FourierRhs::new(grid, c, nu);
        let centred = CentredDifferenceRhs::new(grid, c, nu);
        let from_spectral = scheme.advance(&initial, dt, steps, |s| spectral.rate(s));
        let from_centred = scheme.advance(&initial, dt, steps, |s| centred.rate(s));
        println!(
            "{:<18} {:>12.3e} {:>12.3e}",
            scheme.name(),
            l2_error(&from_spectral, &exact),
            l2_error(&from_centred, &exact)
        );
    }
}
