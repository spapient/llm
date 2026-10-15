//! Relocated unit tests for the `checkpoint` crate.
//!
//! Formerly inline `#[cfg(test)] mod tests` blocks, moved here so that
//! production sources under `crates/checkpoint/src` hold no test code. Each source
//! module's tests are wrapped in a nested module to keep helper and test names
//! isolated. Every assertion reaches the crate through its public interface
//! (see `docs/engineering.md`).
#![allow(clippy::unwrap_used, clippy::expect_used, dead_code)]

/// Tests relocated from `crates/checkpoint/src/atomicwrite.rs`.
mod atomicwrite {
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use ::checkpoint::checkpointerror;
    use ::checkpoint::writeatomic;

    fn scratch(tag: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!(
            "llm-checkpoint-atomic-{}-{tag}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn writes_and_replaces_content() {
        let dir = scratch("replace");
        let target = dir.join("artifact.bin");
        writeatomic(&target, b"first").unwrap();
        assert_eq!(std::fs::read(&target).unwrap(), b"first");
        writeatomic(&target, b"second").unwrap();
        assert_eq!(std::fs::read(&target).unwrap(), b"second");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn leaves_no_staging_files_behind() {
        let dir = scratch("cleanup");
        let target = dir.join("artifact.bin");
        writeatomic(&target, b"data").unwrap();
        let entries: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(entries, vec!["artifact.bin".to_string()]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn failed_rename_preserves_existing_target_and_cleans_staging() {
        let dir = scratch("failrename");
        // The target is an existing directory, so renaming a file onto it fails.
        let target = dir.join("occupied");
        std::fs::create_dir(&target).unwrap();
        let result = writeatomic(&target, b"data");
        assert!(matches!(result, Err(checkpointerror::io { .. })));
        assert!(target.is_dir());
        let entries: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(entries, vec!["occupied".to_string()]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_parent_directory_is_reported_as_io() {
        let dir = scratch("noparent");
        let target = dir.join("does-not-exist").join("artifact.bin");
        assert!(matches!(
            writeatomic(&target, b"data"),
            Err(checkpointerror::io { .. })
        ));
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// Tests relocated from `crates/checkpoint/src/compatibility.rs`.
mod compatibility {
    use crate::support::shapeof;
    use ::checkpoint::checkcompat;
    use ::checkpoint::checkpoint;
    use ::checkpoint::checkpointerror;
    use ::checkpoint::checkversion;
    use ::checkpoint::tensorspec;
    use tensor::dtype;

    fn ckpt() -> checkpoint {
        let mut ckpt = checkpoint::new(1);
        ckpt.addtensor("w".to_string(), dtype::f32, shapeof(&[2, 2]), &[0u8; 16])
            .unwrap();
        ckpt
    }

    #[test]
    fn matching_tensor_specs_pass() {
        let specs = [tensorspec::new(
            "w".to_string(),
            dtype::f32,
            shapeof(&[2, 2]),
        )];
        assert!(checkcompat(&ckpt(), &specs).is_ok());
    }

    #[test]
    fn missing_tensor_is_an_incompatibility() {
        let specs = [tensorspec::new(
            "b".to_string(),
            dtype::f32,
            shapeof(&[2, 2]),
        )];
        let err = checkcompat(&ckpt(), &specs).unwrap_err();
        assert!(matches!(err, checkpointerror::missingtensor { .. }));
        assert!(err.isincompatibility());
        assert!(!err.isrecoverable());
    }

    #[test]
    fn dtype_mismatch_is_an_incompatibility() {
        let specs = [tensorspec::new(
            "w".to_string(),
            dtype::f64,
            shapeof(&[2, 2]),
        )];
        let err = checkcompat(&ckpt(), &specs).unwrap_err();
        assert!(matches!(err, checkpointerror::dtypemismatch { .. }));
        assert!(err.isincompatibility());
    }

    #[test]
    fn shape_mismatch_is_an_incompatibility() {
        let specs = [tensorspec::new("w".to_string(), dtype::f32, shapeof(&[4]))];
        let err = checkcompat(&ckpt(), &specs).unwrap_err();
        assert!(matches!(err, checkpointerror::shapemismatch { .. }));
        assert!(err.isincompatibility());
    }

    #[test]
    fn version_check_reports_both_versions() {
        let err = checkversion(&ckpt(), 2).unwrap_err();
        assert!(matches!(
            err,
            checkpointerror::unsupportedversion {
                found: 1,
                supported: 2
            }
        ));
        assert!(checkversion(&ckpt(), 1).is_ok());
    }
}

/// Tests relocated from `crates/checkpoint/src/crc32c.rs`.
mod crc32c {
    use ::checkpoint::crc32c;

    #[test]
    fn empty_input_has_zero_checksum() {
        assert_eq!(crc32c(b""), 0);
    }

    #[test]
    fn matches_the_castagnoli_known_answer() {
        assert_eq!(crc32c(b"123456789"), 0xE306_9283);
    }

    #[test]
    fn detects_a_single_bit_flip() {
        let mut data = vec![0u8; 64];
        let before = crc32c(&data);
        data[37] ^= 1;
        assert_ne!(crc32c(&data), before);
    }
}

/// Tests relocated from `crates/checkpoint/src/error.rs`.
mod error {
    use ::checkpoint::checkpointerror;

    #[test]
    fn io_failure_is_recoverable_and_not_incompatible() {
        let e = checkpointerror::io {
            path: "ckpt.bin".into(),
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "missing"),
        };
        assert!(e.isrecoverable());
        assert!(!e.isincompatibility());
    }

    #[test]
    fn structural_failures_are_incompatible_and_not_recoverable() {
        let cases = [
            checkpointerror::unsupportedversion {
                found: 2,
                supported: 1,
            },
            checkpointerror::missingtensor {
                name: "w".to_string(),
            },
            checkpointerror::dtypemismatch {
                name: "w".to_string(),
                expected: "F32".to_string(),
                actual: "F64".to_string(),
            },
            checkpointerror::shapemismatch {
                name: "w".to_string(),
                expected: vec![2, 2],
                actual: vec![4],
            },
        ];
        for e in cases {
            assert!(e.isincompatibility(), "{e:?} should be incompatible");
            assert!(!e.isrecoverable(), "{e:?} should not be recoverable");
        }
    }

    #[test]
    fn malformed_failures_are_neither() {
        let e = checkpointerror::badmagic;
        assert!(!e.isrecoverable());
        assert!(!e.isincompatibility());
    }

    #[test]
    fn display_reports_version_bounds() {
        let e = checkpointerror::unsupportedversion {
            found: 9,
            supported: 1,
        };
        assert_eq!(
            e.to_string(),
            "unsupported checkpoint version 9 (supported 1)"
        );
    }
}

/// Tests relocated from `crates/checkpoint/src/hash.rs`.
mod hash {
    use ::checkpoint::blake3hash;
    use ::checkpoint::contenthash;

    #[test]
    fn empty_input_matches_the_blake3_known_answer() {
        let digest = blake3hash(b"");
        let expected = [
            0xAF, 0x13, 0x49, 0xB9, 0xF5, 0xF9, 0xA1, 0xA6, 0xA0, 0x40, 0x4D, 0xEA, 0x36, 0xDC,
            0xC9, 0x49, 0x9B, 0xCB, 0x25, 0xC9, 0xAD, 0xC1, 0x12, 0xB7, 0xCC, 0x9A, 0x93, 0xCA,
            0xE4, 0x1F, 0x32, 0x62,
        ];
        assert_eq!(digest, expected);
    }

    #[test]
    fn contenthash_is_the_leading_digest_bytes() {
        let digest = blake3hash(b"abc");
        let content = contenthash(b"abc");
        assert_eq!(content.as_slice(), &digest[..16]);
    }

    #[test]
    fn contenthash_has_sixteen_bytes() {
        assert_eq!(contenthash(b"anything").len(), 16);
    }

    #[test]
    fn distinct_inputs_produce_distinct_hashes() {
        assert_ne!(blake3hash(b"a"), blake3hash(b"b"));
    }
}

/// Tests relocated from `crates/checkpoint/src/manifests.rs`.
mod manifests {
    use crate::support::shapeof;
    use ::checkpoint::checkpoint;
    use ::checkpoint::checkpointerror;
    use ::checkpoint::manifest;
    use ::checkpoint::trainingstate;
    use tensor::dtype;

    #[test]
    fn encode_decode_roundtrip_preserves_all_state() {
        let mut ckpt = checkpoint::new(1);
        ckpt.settraining(trainingstate::new(42, 2));
        ckpt.addmetadata("source".to_string(), "test".to_string());
        ckpt.addtensor("w".to_string(), dtype::f32, shapeof(&[2, 2]), &[0u8; 16])
            .unwrap();
        ckpt.addtensor("b".to_string(), dtype::i64, shapeof(&[2]), &[0u8; 16])
            .unwrap();
        let bytes = ckpt.encode().unwrap();
        let back = checkpoint::decode(&bytes).unwrap();
        assert_eq!(back, ckpt);
        assert_eq!(back.getmanifest().getversion(), 1);
        assert_eq!(
            back.getmanifest().gettraining(),
            Some(trainingstate::new(42, 2))
        );
        assert_eq!(
            back.getmanifest()
                .getmetadata()
                .get("source")
                .map(String::as_str),
            Some("test")
        );
        assert_eq!(back.readtensor("w").unwrap().len(), 16);
    }

    #[test]
    fn addtensor_rejects_length_mismatch() {
        let mut ckpt = checkpoint::new(1);
        assert!(matches!(
            ckpt.addtensor("w".to_string(), dtype::f32, shapeof(&[2, 2]), &[0u8; 8]),
            Err(checkpointerror::lengthmismatch {
                expected: 16,
                actual: 8,
                ..
            })
        ));
        assert!(ckpt.getdata().is_empty());
    }

    #[test]
    fn addtensor_rejects_duplicate_names() {
        let mut ckpt = checkpoint::new(1);
        ckpt.addtensor("w".to_string(), dtype::f32, shapeof(&[1]), &[0u8; 4])
            .unwrap();
        assert!(matches!(
            ckpt.addtensor("w".to_string(), dtype::f32, shapeof(&[1]), &[0u8; 4]),
            Err(checkpointerror::duplicatetensor { .. })
        ));
        assert_eq!(ckpt.getdata().len(), 4);
    }

    #[test]
    fn readtensor_returns_none_for_missing_name() {
        let ckpt = checkpoint::new(1);
        assert!(ckpt.readtensor("nope").is_none());
    }

    #[test]
    fn decode_rejects_bad_magic() {
        let mut bytes = checkpoint::new(1).encode().unwrap();
        bytes[0] = b'X';
        assert!(matches!(
            checkpoint::decode(&bytes),
            Err(checkpointerror::badmagic)
        ));
    }

    #[test]
    fn decode_rejects_truncated_prefix() {
        assert!(matches!(
            checkpoint::decode(b"LLMC"),
            Err(checkpointerror::truncated {
                needed: 8,
                available: 4
            })
        ));
    }

    #[test]
    fn decode_rejects_truncated_manifest() {
        let mut bytes = checkpoint::new(1).encode().unwrap();
        bytes.truncate(12);
        assert!(matches!(
            checkpoint::decode(&bytes),
            Err(checkpointerror::truncated { .. })
        ));
    }

    #[test]
    fn decode_rejects_invalid_manifest_json() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"LLMCKPT1");
        bytes.extend_from_slice(&3u64.to_le_bytes());
        bytes.extend_from_slice(b"{x}");
        assert!(matches!(
            checkpoint::decode(&bytes),
            Err(checkpointerror::invalidmanifestjson { .. })
        ));
    }

    #[test]
    fn decode_rejects_offset_past_data_section() {
        let mut ckpt = checkpoint::new(1);
        ckpt.addtensor("w".to_string(), dtype::f32, shapeof(&[2]), &[0u8; 8])
            .unwrap();
        let mut bytes = ckpt.encode().unwrap();
        // Truncate the data section, keeping the manifest intact.
        bytes.truncate(bytes.len() - 4);
        assert!(matches!(
            checkpoint::decode(&bytes),
            Err(checkpointerror::offsetoutofrange { .. })
        ));
    }

    #[test]
    fn manifest_roundtrip_is_stable() {
        let mut m = manifest::new(1);
        m.settraining(trainingstate::new(1, 0));
        m.addmetadata("k".to_string(), "v".to_string());
        let bytes = m.tobytes().unwrap();
        let back = manifest::frombytes(&bytes).unwrap();
        assert_eq!(back, m);
    }

    #[test]
    fn manifest_rejects_unknown_dtype_token() {
        let json = br#"{"version":1,"tensors":[{"name":"w","dtype":"BOOL","shape":[1],"offset":0,"nbytes":1}]}"#;
        assert!(matches!(
            manifest::frombytes(json),
            Err(checkpointerror::dtypeunsupported { .. })
        ));
    }

    #[test]
    fn manifest_rejects_duplicate_names() {
        let json = br#"{"version":1,"tensors":[
            {"name":"w","dtype":"F32","shape":[1],"offset":0,"nbytes":4},
            {"name":"w","dtype":"F32","shape":[1],"offset":4,"nbytes":4}
        ]}"#;
        assert!(matches!(
            manifest::frombytes(json),
            Err(checkpointerror::duplicatetensor { .. })
        ));
    }
}

/// Tests relocated from `crates/checkpoint/src/training.rs`.
mod training {
    use ::checkpoint::trainingstate;

    #[test]
    fn new_and_getters_roundtrip() {
        let s = trainingstate::new(120, 3);
        assert_eq!(s.getstep(), 120);
        assert_eq!(s.getepoch(), 3);
    }

    #[test]
    fn json_roundtrip_preserves_state() {
        let s = trainingstate::new(7, 1);
        let text = serde_json::to_string(&s).unwrap();
        let back: trainingstate = serde_json::from_str(&text).unwrap();
        assert_eq!(back, s);
    }
}
