use md::{lennard_jones_pair_energy, lennard_jones_pair_force};

const LIMIT: f64 = 3.0;
const ENERGY_SAMPLES: usize = 301;
const ARROW_SAMPLES: usize = 15;

fn main() {
    println!("energy,{ENERGY_SAMPLES},{LIMIT}");
    for row in 0..ENERGY_SAMPLES {
        let y = coordinate(row, ENERGY_SAMPLES);
        for column in 0..ENERGY_SAMPLES {
            let x = coordinate(column, ENERGY_SAMPLES);
            let distance = x.hypot(y);
            let energy = if distance == 0.0 {
                f64::INFINITY
            } else {
                lennard_jones_pair_energy(distance)
            };
            println!("e,{row},{column},{energy}");
        }
    }

    println!("arrows,{ARROW_SAMPLES},{LIMIT}");
    for row in 0..ARROW_SAMPLES {
        let y = coordinate(row, ARROW_SAMPLES);
        for column in 0..ARROW_SAMPLES {
            let x = coordinate(column, ARROW_SAMPLES);
            let distance = x.hypot(y);
            if (0.7..=LIMIT).contains(&distance) {
                let radial_force = lennard_jones_pair_force(distance);
                let force_x = radial_force * x / distance;
                let force_y = radial_force * y / distance;
                println!("f,{x},{y},{force_x},{force_y}");
            }
        }
    }
}

fn coordinate(index: usize, samples: usize) -> f64 {
    -LIMIT + 2.0 * LIMIT * index as f64 / (samples - 1) as f64
}
