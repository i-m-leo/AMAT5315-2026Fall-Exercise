//! End-to-end check of the `seismic` binary on a tiny generated experiment.

use std::path::PathBuf;
use std::process::Command;

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("seismic-test-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A 9 x 9 homogeneous grid, one shot at the centre, one receiver off-axis.
fn experiment() -> String {
    let n = 9;
    let background: Vec<Vec<f64>> = vec![vec![1.8; n]; n];
    let perturbation: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
    serde_json::json!({
        "schema": "week5-seismic-experiment-v1",
        "name": "tiny",
        "nx": n, "nz": n, "dx": 1.0, "dt": 0.1, "steps": 40,
        "source_frequency": 0.08, "source_peak_time": 15.0, "source_amplitude": 1.0,
        "shots": [[4.0, 4.0], [6.0, 2.0]], "receivers": [[4, 5], [6, 2]],
        "sponge_width": 2.0, "sponge_strength": 0.4,
        "length_unit_m": 100.0, "time_unit_s": 0.1,
        "background": background, "perturbation": perturbation,
    })
    .to_string()
}

fn run(name: &str, extra: &[&str]) -> (String, PathBuf) {
    let dir = temp_dir(name);
    let path = dir.join("tiny.json");
    std::fs::write(&path, experiment()).unwrap();
    let out = dir.join("out");
    let mut args = vec![
        "--experiment".to_string(),
        path.display().to_string(),
        "--mode".to_string(),
        "forward".to_string(),
        "--out".to_string(),
        out.display().to_string(),
    ];
    args.extend(extra.iter().map(|s| s.to_string()));
    let output = Command::new(env!("CARGO_BIN_EXE_seismic")).args(&args).output().unwrap();
    assert!(
        output.status.success(),
        "seismic failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    (String::from_utf8(output.stdout).unwrap(), out)
}

#[test]
fn forward_run_writes_every_artifact() {
    let (stdout, out) = run("forward", &["--every", "5"]);

    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines[0], "shot\tmode\tl2");
    assert_eq!(lines.len(), 3);
    for (shot, line) in lines[1..].iter().enumerate() {
        let mut fields = line.split('\t');
        assert_eq!(fields.next().unwrap(), shot.to_string());
        assert_eq!(fields.next().unwrap(), "forward");
        let l2: f64 = fields.next().unwrap().parse().expect("l2 is a number");
        assert!(l2 > 0.0 && l2.is_finite(), "l2 should be positive, got {l2}");
    }
    // The two shots sit at different offsets from their receiver, so the
    // recorded energies must differ.
    assert_ne!(lines[1], lines[2]);

    for name in ["run.json", "result.json", "traces.npy", "wavefield.npy", "echo.npy"] {
        let path = out.join(name);
        assert!(path.exists(), "missing {}", path.display());
        assert!(std::fs::metadata(&path).unwrap().len() > 0, "{name} is empty");
    }

    let run: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("run.json")).unwrap()).unwrap();
    assert_eq!(run["experiment"]["name"], "tiny");
    assert!(run["experiment"].get("background").is_none());
    assert!(run["experiment"].get("perturbation").is_none());
    assert_eq!(run["recording"]["steps"], serde_json::json!([5, 10, 15, 20, 25, 30, 35, 40]));

    let result: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("result.json")).unwrap()).unwrap();
    assert_eq!(result["mode"], "forward");
    assert_eq!(result["steps"], 40);
    // Integers stay integers, floats stay floats.
    assert_eq!(result["shots"], serde_json::json!([[4.0, 4.0], [6.0, 2.0]]));
    assert_eq!(result["receivers"], serde_json::json!([[4, 5], [6, 2]]));
}

#[test]
fn traces_are_a_valid_npy_without_recording() {
    let (_, out) = run("no-recording", &[]);

    let raw = std::fs::read(out.join("traces.npy")).unwrap();
    assert_eq!(&raw[..6], b"\x93NUMPY");
    assert_eq!(&raw[6..8], &[1, 0], "version 1.0");
    let header_len = u16::from_le_bytes([raw[8], raw[9]]) as usize;
    assert_eq!((10 + header_len) % 64, 0, "data starts 64-byte aligned");
    let header = std::str::from_utf8(&raw[10..10 + header_len - 1]).unwrap();
    assert!(header.contains("'descr': '<f8'"));
    assert!(header.contains("'fortran_order': False"));
    assert!(header.contains("'shape': (2, 40, 2)"));
    assert_eq!(raw.len(), 10 + header_len + 2 * 40 * 2 * 8);

    assert!(!out.join("wavefield.npy").exists(), "recording should be off");
    let run: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("run.json")).unwrap()).unwrap();
    assert!(run.get("recording").is_none());
}

#[test]
fn bad_arguments_fail_loudly() {
    let dir = temp_dir("errors");
    let path = dir.join("tiny.json");
    std::fs::write(&path, experiment()).unwrap();
    let out = dir.join("out");
    let base = vec![
        "--experiment".to_string(),
        path.display().to_string(),
        "--out".to_string(),
        out.display().to_string(),
    ];

    // The adjoint needs the receiver weights from a Born run.
    let output = Command::new(env!("CARGO_BIN_EXE_seismic"))
        .args(&base)
        .args(["--mode", "adjoint"])
        .output()
        .unwrap();
    assert!(!output.status.success(), "adjoint without --data should fail");
    assert!(String::from_utf8_lossy(&output.stderr).contains("--data"));

    // A data file of the wrong shape is rejected.
    let wrong = dir.join("wrong.npy");
    std::fs::write(&wrong, b"not an npy").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_seismic"))
        .args(&base)
        .args(["--mode", "adjoint", "--data", &wrong.display().to_string()])
        .output()
        .unwrap();
    assert!(!output.status.success(), "a bad --data file should fail");

    let output = Command::new(env!("CARGO_BIN_EXE_seismic"))
        .args(&base)
        .args(["--mode", "forward", "--every", "0"])
        .output()
        .unwrap();
    assert!(!output.status.success(), "--every 0 should fail");
    assert!(String::from_utf8_lossy(&output.stderr).contains("at least 1"));
}

#[test]
fn reference_experiment_reproduces_course_numbers() {
    let experiment = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("inputs/reflector.json");
    if !experiment.exists() {
        eprintln!("skipping: {} is not present (inputs/ is untracked)", experiment.display());
        return;
    }
    let (stdout, _) = run_inputs(&experiment, "reflector");
    let l2: Vec<f64> = stdout
        .lines()
        .skip(1)
        .map(|line| line.split('\t').nth(2).unwrap().parse().unwrap())
        .collect();
    let reference = [6.623960807913, 6.798645121807, 6.623960807913];
    for (got, want) in l2.iter().zip(reference) {
        assert!((got - want).abs() < 1e-9, "per-shot L2 {got} vs reference {want}");
    }
}

/// Like `run_inputs`, but with an explicit mode and extra arguments.
fn run_mode(
    experiment: &std::path::Path,
    mode: &str,
    extra: &[String],
    name: &str,
) -> (String, PathBuf) {
    let out = temp_dir(name).join("out");
    let mut args = vec![
        "--experiment".to_string(),
        experiment.display().to_string(),
        "--mode".to_string(),
        mode.to_string(),
        "--out".to_string(),
        out.display().to_string(),
    ];
    args.extend(extra.iter().cloned());
    let output = Command::new(env!("CARGO_BIN_EXE_seismic"))
        .args(&args)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "seismic {mode} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    (String::from_utf8(output.stdout).unwrap(), out)
}

/// The Born/adjoint pair on the reference experiment: the Born data norm,
/// and the adjoint-image dot-product test `<K v, K v> = <v, K^T K v>`.
#[test]
fn born_and_adjoint_satisfy_the_dot_product_identity() {
    let experiment = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("inputs/reflector.json");
    if !experiment.exists() {
        eprintln!("skipping: {} is not present (inputs/ is untracked)", experiment.display());
        return;
    }
    let (_, born_out) = run_mode(&experiment, "born", &[], "born");
    let data = std::fs::read(born_out.join("born_data.npy")).unwrap();

    let (_, adjoint_out) = run_mode(
        &experiment,
        "adjoint",
        &[
            "--data".to_string(),
            born_out.join("born_data.npy").display().to_string(),
        ],
        "adjoint",
    );

    // Read the two arrays back through the crate's public readers.
    let born = seismic::npy::read_f64(&born_out.join("born_data.npy"), &[3, 240, 14]).unwrap();
    let image = seismic::npy::read_f64(&adjoint_out.join("image.npy"), &[41, 41]).unwrap();
    assert!(!data.is_empty());

    // The reference Born norm, and the reflector at z = 21.
    let norm = born.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(
        (norm - 0.18667589618149358).abs() < 1e-12,
        "Born norm {norm} vs reference 0.18667589618149358"
    );

    // dot_left is the data energy; dot_right pairs the image with the
    // perturbation the data came from. They must agree to round-off.
    let dot_left: f64 = born.iter().map(|v| v * v).sum();
    let perturbation =
        seismic::Experiment::load(&experiment).unwrap().perturbation;
    let dot_right: f64 = image.iter().zip(&perturbation).map(|(a, b)| a * b).sum();
    let relative = (dot_left - dot_right).abs() / dot_left.abs();
    assert!(
        relative < 1e-12,
        "dot-product identity: {dot_left} vs {dot_right}, relative {relative}"
    );

    // The depth profile peaks on the reflector, two cells wide about z = 21.
    let peak = (0..41)
        .map(|z| image[z * 41..(z + 1) * 41].iter().map(|v| v.abs()).sum::<f64>())
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();
    assert_eq!(peak.0, 21, "image depth profile peaks at z = {}", peak.0);
}
fn run_inputs(experiment: &std::path::Path, name: &str) -> (String, PathBuf) {
    let out = temp_dir(name).join("out");
    let output = Command::new(env!("CARGO_BIN_EXE_seismic"))
        .args([
            "--experiment",
            &experiment.display().to_string(),
            "--mode",
            "forward",
            "--out",
            &out.display().to_string(),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "seismic failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    (String::from_utf8(output.stdout).unwrap(), out)
}
