use std::f64::consts::PI;

#[derive(Clone, Copy, Debug)]
pub struct PeriodicGrid {
    n: usize,
    length: f64,
}

impl PeriodicGrid {
    pub fn new(n: usize) -> Self {
        assert!(n >= 2, "grid needs at least two points");
        assert!(n % 2 == 0, "grid size must be even for the Nyquist mode");
        Self {
            n,
            length: 2.0 * PI,
        }
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn length(&self) -> f64 {
        self.length
    }

    pub fn spacing(&self) -> f64 {
        self.length / self.n as f64
    }

    pub fn point(&self, j: usize) -> f64 {
        j as f64 * self.spacing()
    }

    pub fn points(&self) -> Vec<f64> {
        (0..self.n).map(|j| self.point(j)).collect()
    }

    pub fn wavenumber(&self, j: usize) -> f64 {
        let n = self.n as isize;
        let j = j as isize;
        if j < n / 2 {
            j as f64
        } else {
            (j - n) as f64
        }
    }

    pub fn wavenumbers(&self) -> Vec<f64> {
        (0..self.n).map(|j| self.wavenumber(j)).collect()
    }

    pub fn is_nyquist(&self, j: usize) -> bool {
        j == self.n / 2
    }

    pub fn nyquist_index(&self) -> usize {
        self.n / 2
    }
}

pub fn mode(grid: &PeriodicGrid, k: f64, amplitude: f64, phase: f64) -> Vec<f64> {
    grid.points()
        .into_iter()
        .map(|x| amplitude * (k * x + phase).cos())
        .collect()
}

pub fn wave_exact(
    grid: &PeriodicGrid,
    k: f64,
    amplitude: f64,
    phase: f64,
    c: f64,
    nu: f64,
    t: f64,
) -> Vec<f64> {
    let decay = (-nu * k * k * t).exp();
    grid.points()
        .into_iter()
        .map(|x| amplitude * decay * (k * (x - c * t) + phase).cos())
        .collect()
}

pub fn gaussian(grid: &PeriodicGrid, sigma: f64, x0: f64) -> Vec<f64> {
    let length = grid.length();
    grid.points()
        .into_iter()
        .map(|x| {
            let raw = (x - x0).abs();
            let d = raw.min(length - raw);
            (-0.5 * (d / sigma).powi(2)).exp()
        })
        .collect()
}
