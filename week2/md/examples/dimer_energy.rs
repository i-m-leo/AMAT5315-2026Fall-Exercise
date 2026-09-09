use md::{
    run_dimer_experiment, DimerState, EnergySample, ForwardEuler, TimeIntegrator, VelocityVerlet,
};

const TIME_STEP: f64 = 0.01;

fn print_experiment<I: TimeIntegrator>(name: &str, integrator: &I, steps: usize) {
    let samples = run_dimer_experiment(integrator, DimerState::stationary(1.2), TIME_STEP, steps);
    for EnergySample {
        step,
        time,
        total_energy,
        relative_energy_error,
    } in samples
    {
        println!("{name},{step},{time},{total_energy},{relative_energy_error}");
    }
}

fn main() {
    println!("method,step,time,total_energy,relative_energy_error");
    print_experiment("Forward Euler (500)", &ForwardEuler, 500);
    print_experiment("Velocity-Verlet (500)", &VelocityVerlet, 500);
    print_experiment("Velocity-Verlet (5000)", &VelocityVerlet, 5_000);
}
