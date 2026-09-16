use std::fs;
use std::process::Command;

use serde_json::Value;
use tempfile::tempdir;

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_ising")
}

#[test]
fn metropolis_writes_contract_outputs() {
    let dir = tempdir().unwrap();
    let output = Command::new(binary())
        .args([
            "--update",
            "metropolis",
            "--l",
            "2",
            "--t-from",
            "1.5",
            "--t-to",
            "1.6",
            "--t-step",
            "0.1",
            "--discard",
            "1",
            "--measure",
            "2",
            "--every",
            "1",
            "--seed",
            "2026",
            "--out",
            dir.path().to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.lines().next().unwrap().contains("T"));
    assert_eq!(stdout.lines().count(), 3);

    let run: Value =
        serde_json::from_str(&fs::read_to_string(dir.path().join("run.json")).unwrap()).unwrap();
    assert_eq!(run["L"], 2);
    assert_eq!(run["update"], "metropolis");
    assert_eq!(run["t_grid"], serde_json::json!([1.5, 1.6]));
    assert_eq!(run["sample_every"], 1);
    assert_eq!(run["time_unit"], "sweep");

    let series: Vec<Value> = fs::read_to_string(dir.path().join("series.jsonl"))
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(series.len(), 4);
    assert_eq!(series[0]["L"], 2);
    assert_eq!(series[0]["sweep"], 1);
    assert_eq!(series[2]["sweep"], 1);
    assert!(series
        .iter()
        .all(|row| row["M"].is_number() && row["E"].is_number()));

    let spins: Vec<Value> = fs::read_to_string(dir.path().join("spins.jsonl"))
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(spins.len(), 4);
    assert_eq!(spins[0]["sweep"], 2);
    assert_eq!(spins[2]["sweep"], 5);
    assert_eq!(spins[0]["spins"].as_array().unwrap().len(), 4);
}

#[test]
fn wolff_writes_cluster_flip_outputs() {
    let output = Command::new(binary())
        .args([
            "--update",
            "wolff",
            "--l",
            "2",
            "--t-from",
            "1.5",
            "--t-to",
            "1.5",
            "--t-step",
            "0.1",
            "--discard",
            "0",
            "--measure",
            "0",
            "--seed",
            "1",
            "--out",
            "out",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("mean_cluster_size"));
    let dir = tempfile::tempdir().unwrap();
    let output = Command::new(binary())
        .args([
            "--update",
            "wolff",
            "--l",
            "4",
            "--t-from",
            "1.5",
            "--t-to",
            "1.5",
            "--t-step",
            "0.1",
            "--discard",
            "1",
            "--measure",
            "2",
            "--every",
            "1",
            "--seed",
            "1",
            "--out",
            dir.path().to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let rows: Vec<Value> = fs::read_to_string(dir.path().join("series.jsonl"))
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(rows.len(), 2);
    assert!(rows
        .iter()
        .all(|row| row["cluster_size"].as_u64().unwrap() >= 1));
    let run: Value =
        serde_json::from_str(&fs::read_to_string(dir.path().join("run.json")).unwrap()).unwrap();
    assert_eq!(run["time_unit"], "cluster_flip");
}
