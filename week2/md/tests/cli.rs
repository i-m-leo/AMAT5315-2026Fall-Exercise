use serde_json::Value;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn temporary_directory(label: &str) -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("md-{label}-{}-{stamp}", std::process::id()))
}

#[test]
fn run_writes_viewer_compatible_artifacts_and_check_passes() {
    let output = temporary_directory("cli");
    let binary = env!("CARGO_BIN_EXE_md");
    let run = Command::new(binary)
        .args([
            "run",
            "--n",
            "36",
            "--eq-steps",
            "10",
            "--steps",
            "20",
            "--sample-every",
            "5",
            "--out",
        ])
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );

    let metadata: Value =
        serde_json::from_slice(&fs::read(output.join("run.json")).unwrap()).unwrap();
    assert_eq!(metadata["format"], "amat5315-md-v1");
    assert_eq!(metadata["saved_frames"], 4);
    assert!(metadata["box"].is_array());
    let lines = fs::read_to_string(output.join("traj.jsonl")).unwrap();
    let frames = lines
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(frames.len(), 4);
    assert_eq!(frames[0]["step"], 5);
    for key in ["t", "pos", "vel", "E_pot", "E_kin"] {
        assert!(frames[0].get(key).is_some(), "missing {key}");
    }

    let check = Command::new(binary)
        .args(["check", "--input"])
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        check.status.success(),
        "{}",
        String::from_utf8_lossy(&check.stderr)
    );
    assert!(String::from_utf8_lossy(&check.stdout).contains("Stored energy integrity"));
    fs::remove_dir_all(output).unwrap();
}
