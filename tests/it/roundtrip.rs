#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Integration tests for the public `checkpoint` API: save/load round trips,
//! truncated and malformed input, atomic replacement, and compatibility.

use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use ::checkpoint::{
    CHECKPOINTVERSION, checkcompat, checkpoint, checkpointerror, checkversion, loadcheckpoint,
    savecheckpoint, tensorspec, trainingstate,
};
use tensor::{dtype, shape};

fn scratch(tag: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = std::env::temp_dir().join(format!(
        "llm-checkpoint-it-{}-{tag}-{nanos}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn dims(values: &[usize]) -> shape {
    shape::new(values.to_vec()).unwrap()
}

fn sample() -> checkpoint {
    let mut ckpt = checkpoint::new(CHECKPOINTVERSION);
    ckpt.settraining(trainingstate::new(128, 4));
    ckpt.addmetadata("source".to_string(), "integration".to_string());
    ckpt.addtensor("embed".to_string(), dtype::f32, dims(&[2, 4]), &[0u8; 32])
        .unwrap();
    ckpt.addtensor("norm".to_string(), dtype::f32, dims(&[4]), &[0u8; 16])
        .unwrap();
    ckpt
}

#[test]
fn save_load_roundtrip_preserves_model_and_training_state() {
    let dir = scratch("roundtrip");
    let path = dir.join("model.bin");
    let ckpt = sample();
    savecheckpoint(&path, &ckpt).unwrap();

    let loaded = loadcheckpoint(&path).unwrap();
    assert_eq!(loaded, ckpt);
    assert_eq!(
        loaded.getmanifest().gettraining(),
        Some(trainingstate::new(128, 4))
    );
    assert_eq!(loaded.readtensor("embed").unwrap().len(), 32);
    assert_eq!(loaded.readtensor("norm").unwrap().len(), 16);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn truncated_file_reports_a_malformed_error() {
    let dir = scratch("truncated");
    let path = dir.join("model.bin");
    savecheckpoint(&path, &sample()).unwrap();

    let mut bytes = std::fs::read(&path).unwrap();
    bytes.truncate(bytes.len() - 8);
    let corrupted = dir.join("truncated.bin");
    std::fs::write(&corrupted, &bytes).unwrap();

    let err = loadcheckpoint(&corrupted).unwrap_err();
    assert!(matches!(err, checkpointerror::offsetoutofrange { .. }));
    assert!(!err.isrecoverable());
    assert!(!err.isincompatibility());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn corrupt_magic_reports_badmagic() {
    let dir = scratch("magic");
    let path = dir.join("model.bin");
    savecheckpoint(&path, &sample()).unwrap();
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[0] = b'Z';
    let corrupted = dir.join("corrupt.bin");
    std::fs::write(&corrupted, &bytes).unwrap();

    let err = loadcheckpoint(&corrupted).unwrap_err();
    assert!(matches!(err, checkpointerror::badmagic));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn missing_file_reports_a_recoverable_io_error() {
    let dir = scratch("missing");
    let path = dir.join("absent.bin");
    let err = loadcheckpoint(&path).unwrap_err();
    assert!(err.isrecoverable());
    assert!(!err.isincompatibility());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn compatibility_accepts_a_match_and_rejects_mismatches() {
    let ckpt = sample();
    assert!(checkversion(&ckpt, CHECKPOINTVERSION).is_ok());

    let matching = [
        tensorspec::new("embed".to_string(), dtype::f32, dims(&[2, 4])),
        tensorspec::new("norm".to_string(), dtype::f32, dims(&[4])),
    ];
    assert!(checkcompat(&ckpt, &matching).is_ok());

    assert!(
        checkcompat(
            &ckpt,
            &[tensorspec::new(
                "absent".to_string(),
                dtype::f32,
                dims(&[1])
            )]
        )
        .is_err()
    );
    assert!(
        checkcompat(
            &ckpt,
            &[tensorspec::new(
                "embed".to_string(),
                dtype::f64,
                dims(&[2, 4])
            )]
        )
        .is_err()
    );
    assert!(
        checkcompat(
            &ckpt,
            &[tensorspec::new("embed".to_string(), dtype::f32, dims(&[8]))]
        )
        .is_err()
    );
    assert!(checkversion(&ckpt, CHECKPOINTVERSION + 1).is_err());
    assert!(
        checkversion(&ckpt, CHECKPOINTVERSION + 1)
            .unwrap_err()
            .isincompatibility()
    );
}

#[test]
fn atomic_save_replaces_content_without_partial_artifacts() {
    let dir = scratch("atomic");
    let path = dir.join("model.bin");

    let mut first = checkpoint::new(CHECKPOINTVERSION);
    first
        .addtensor("a".to_string(), dtype::f32, dims(&[1]), &[1u8; 4])
        .unwrap();
    savecheckpoint(&path, &first).unwrap();
    let initial = std::fs::read(&path).unwrap();

    let mut second = checkpoint::new(CHECKPOINTVERSION);
    second
        .addtensor("a".to_string(), dtype::f32, dims(&[2]), &[2u8; 8])
        .unwrap();
    savecheckpoint(&path, &second).unwrap();
    let replaced = std::fs::read(&path).unwrap();

    assert_ne!(initial, replaced);
    assert_eq!(loadcheckpoint(&path).unwrap(), second);
    // No staging file survived the operation.
    let entries: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(entries, vec!["model.bin".to_string()]);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn failed_save_leaves_the_previous_checkpoint_intact() {
    let dir = scratch("failedsave");
    let path = dir.join("model.bin");
    savecheckpoint(&path, &sample()).unwrap();
    let before = std::fs::read(&path).unwrap();

    // A missing parent directory makes the staging file uncreatable.
    let bad = dir.join("missing").join("model.bin");
    assert!(savecheckpoint(&bad, &sample()).is_err());

    assert_eq!(std::fs::read(&path).unwrap(), before);
    let _ = std::fs::remove_dir_all(&dir);
}
