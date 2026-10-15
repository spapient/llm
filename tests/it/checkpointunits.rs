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

/// Tests relocated from `crates/checkpoint/src/blocks.rs`.
mod blocks {
    use ::checkpoint::{chunk, BLOCKSIZE, MAXBLOCK, MINBLOCK};

    #[test]
    fn empty_input_yields_no_blocks() {
        assert!(chunk(b"").is_empty());
    }

    #[test]
    fn single_short_block() {
        let blocks = chunk(b"hello");
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].getlen(), 5);
    }

    #[test]
    fn blocks_cover_data_and_respect_bounds() {
        let data = vec![0u8; BLOCKSIZE * 4 + 123];
        let blocks = chunk(&data);
        let total: usize = blocks.iter().map(|b| b.getlen() as usize).sum();
        assert_eq!(total, data.len());
        for block in &blocks[..blocks.len() - 1] {
            assert!(block.getlen() as usize >= MINBLOCK);
            assert!(block.getlen() as usize <= MAXBLOCK);
        }
    }

    #[test]
    fn identical_content_has_identical_hash() {
        let a = chunk(&[0u8; BLOCKSIZE]);
        let b = chunk(&[0u8; BLOCKSIZE]);
        assert_eq!(a[0].gethash(), b[0].gethash());
    }
}

/// Tests relocated from `crates/checkpoint/src/uformat.rs`.
mod uformat {
    use crate::support::shapeof;
    use ::checkpoint::{checkpointerror, snapshot, BLOCKSIZE};
    use tensor::dtype;

    fn onesnap() -> snapshot {
        let mut snap = snapshot::new(1);
        snap.addtensor("w".to_string(), dtype::f32, shapeof(&[2, 2]), &[0u8; 16])
            .unwrap();
        snap
    }

    #[test]
    fn encode_decode_roundtrip() {
        let snap = onesnap();
        let bytes = snap.encode().unwrap();
        let back = snapshot::decode(&bytes).unwrap();
        assert_eq!(back, snap);
    }

    #[test]
    fn block_boundary_falls_mid_tensor() {
        let mut snap = snapshot::new(1);
        snap.addtensor("a".to_string(), dtype::u8, shapeof(&[1000]), &vec![1u8; 1000])
            .unwrap();
        snap.addtensor(
            "b".to_string(),
            dtype::u8,
            shapeof(&[BLOCKSIZE]),
            &vec![2u8; BLOCKSIZE],
        )
        .unwrap();
        let bytes = snap.encode().unwrap();
        let back = snapshot::decode(&bytes).unwrap();
        assert_eq!(back, snap);
        assert_eq!(back.readtensor("a").unwrap().len(), 1000);
        assert_eq!(back.readtensor("b").unwrap().len(), BLOCKSIZE);
    }

    #[test]
    fn single_bit_flip_reports_crc_mismatch() {
        let snap = onesnap();
        let mut bytes = snap.encode().unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 1;
        let err = snapshot::decode(&bytes).unwrap_err();
        assert!(matches!(err, checkpointerror::blockcrcmismatch { .. }));
        assert!(err.iscorruption());
    }

    #[test]
    fn truncated_input_reports_truncation() {
        let snap = onesnap();
        let bytes = snap.encode().unwrap();
        let cut = &bytes[..bytes.len() - 3];
        assert!(matches!(
            snapshot::decode(cut),
            Err(checkpointerror::truncated { .. })
                | Err(checkpointerror::truncatedblock { .. })
        ));
    }

    #[test]
    fn zero_length_tensor() {
        let mut snap = snapshot::new(1);
        snap.addtensor("empty".to_string(), dtype::u8, shapeof(&[0]), &[])
            .unwrap();
        let bytes = snap.encode().unwrap();
        let back = snapshot::decode(&bytes).unwrap();
        assert_eq!(back.readtensor("empty").unwrap().len(), 0);
    }

    #[test]
    fn multi_block_large_tensor() {
        let len = BLOCKSIZE * 3 + 100;
        let data = vec![7u8; len];
        let mut snap = snapshot::new(1);
        snap.addtensor("big".to_string(), dtype::u8, shapeof(&[len]), &data)
            .unwrap();
        let bytes = snap.encode().unwrap();
        let back = snapshot::decode(&bytes).unwrap();
        assert_eq!(back.readtensor("big").unwrap(), data);
    }

    #[test]
    fn decode_rejects_bad_magic() {
        let snap = onesnap();
        let mut bytes = snap.encode().unwrap();
        bytes[0] = b'X';
        assert!(matches!(
            snapshot::decode(&bytes),
            Err(checkpointerror::badmagic)
        ));
    }

    #[test]
    fn decode_rejects_manifest_root_tamper() {
        let snap = onesnap();
        let mut bytes = snap.encode().unwrap();
        bytes[16] ^= 1;
        let err = snapshot::decode(&bytes).unwrap_err();
        assert!(matches!(err, checkpointerror::rootmismatch { .. }));
        assert!(err.iscorruption());
    }
}

/// Tests relocated from `crates/checkpoint/src/delta.rs`.
mod delta {
    use crate::support::shapeof;
    use ::checkpoint::{chunk, delta, snapshot, BLOCKSIZE};
    use tensor::dtype;

    fn pseudorandom(len: usize) -> Vec<u8> {
        let mut data = vec![0u8; len];
        let mut state = 0x1234_5678_9ABC_DEF0u64;
        for byte in &mut data {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *byte = (state >> 56) as u8;
        }
        data
    }

    #[test]
    fn unchanged_tensor_stores_no_blocks() {
        let mut base = snapshot::new(1);
        base.addtensor("w".to_string(), dtype::u8, shapeof(&[100]), &vec![1u8; 100])
            .unwrap();
        let mut new = snapshot::new(1);
        new.addtensor("w".to_string(), dtype::u8, shapeof(&[100]), &vec![1u8; 100])
            .unwrap();
        let d = delta::fromsnapshots(&base, &new);
        assert!(d.getstored().is_empty());
        assert_eq!(d.getbaseid(), base.roothash());
        assert_eq!(d.getroothash(), new.roothash());
    }

    #[test]
    fn prefix_change_dedups_the_unchanged_suffix() {
        let len = BLOCKSIZE * 8;
        let base_data = pseudorandom(len);
        let mut new_data = base_data.clone();
        for byte in &mut new_data[..100] {
            *byte ^= 0xFF;
        }
        let mut base = snapshot::new(1);
        base.addtensor("w".to_string(), dtype::u8, shapeof(&[len]), &base_data)
            .unwrap();
        let mut new = snapshot::new(1);
        new.addtensor("w".to_string(), dtype::u8, shapeof(&[len]), &new_data)
            .unwrap();
        let d = delta::fromsnapshots(&base, &new);
        let total = chunk(&new.getdata()).len();
        assert!(
            d.getstored().len() < total / 2,
            "stored {} vs total {}",
            d.getstored().len(),
            total
        );
    }

    #[test]
    fn insertion_shift_dedups_the_unchanged_suffix() {
        let len = BLOCKSIZE * 8;
        let data = pseudorandom(len);
        let mut base = snapshot::new(1);
        base.addtensor("w".to_string(), dtype::u8, shapeof(&[len]), &data)
            .unwrap();
        let mut new = snapshot::new(1);
        new.addtensor("prefix".to_string(), dtype::u8, shapeof(&[1000]), &vec![0u8; 1000])
            .unwrap();
        new.addtensor("w".to_string(), dtype::u8, shapeof(&[len]), &data)
            .unwrap();
        let d = delta::fromsnapshots(&base, &new);
        let total = chunk(&new.getdata()).len();
        assert!(
            d.getstored().len() < total / 2,
            "stored {} vs total {}",
            d.getstored().len(),
            total
        );
    }

    #[test]
    fn unchanged_large_tensor_stores_no_blocks() {
        let data = pseudorandom(BLOCKSIZE * 4);
        let mut base = snapshot::new(1);
        base.addtensor("w".to_string(), dtype::u8, shapeof(&[data.len()]), &data)
            .unwrap();
        let mut new = snapshot::new(1);
        new.addtensor("w".to_string(), dtype::u8, shapeof(&[data.len()]), &data)
            .unwrap();
        let d = delta::fromsnapshots(&base, &new);
        assert!(d.getstored().is_empty());
    }
}

/// Tests relocated from `crates/checkpoint/src/merge.rs`.
mod merge {
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::support::shapeof;
    use ::checkpoint::{checkpointerror, delta, merge, mergefiles, snapshot, BLOCKSIZE};
    use tensor::dtype;

    fn base() -> snapshot {
        let mut base = snapshot::new(1);
        base.addtensor("a".to_string(), dtype::u8, shapeof(&[BLOCKSIZE]), &vec![1u8; BLOCKSIZE])
            .unwrap();
        base.addtensor("b".to_string(), dtype::u8, shapeof(&[BLOCKSIZE]), &vec![2u8; BLOCKSIZE])
            .unwrap();
        base
    }

    fn new() -> snapshot {
        let mut new = snapshot::new(1);
        new.addtensor("a".to_string(), dtype::u8, shapeof(&[BLOCKSIZE]), &vec![9u8; BLOCKSIZE])
            .unwrap();
        new.addtensor("b".to_string(), dtype::u8, shapeof(&[BLOCKSIZE]), &vec![2u8; BLOCKSIZE])
            .unwrap();
        new
    }

    fn scratch(tag: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!(
            "llm-merge-{}-{tag}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn merge_reconstructs_the_full_snapshot() {
        let base = base();
        let new = new();
        let d = delta::fromsnapshots(&base, &new);
        let merged = merge(&base, &d).unwrap();
        assert_eq!(merged.roothash(), d.getroothash());
        assert_eq!(merged.getdata(), new.getdata());
        assert_eq!(merged, new);
    }

    #[test]
    fn wrong_base_is_an_incompatibility() {
        let base = base();
        let new = new();
        let d = delta::fromsnapshots(&base, &new);
        let other = snapshot::new(1);
        let err = merge(&other, &d).unwrap_err();
        assert!(matches!(err, checkpointerror::basehashmismatch { .. }));
        assert!(err.isincompatibility());
        assert!(!err.isrecoverable());
    }

    #[test]
    fn mergefiles_writes_and_roundtrips() {
        let dir = scratch("roundtrip");
        let base_path = dir.join("base.u");
        let delta_path = dir.join("delta.u");
        let out_path = dir.join("out.u");
        let base = base();
        let new = new();
        std::fs::write(&base_path, base.encode().unwrap()).unwrap();
        std::fs::write(&delta_path, delta::fromsnapshots(&base, &new).encode().unwrap()).unwrap();
        mergefiles(&base_path, &delta_path, &out_path).unwrap();
        let merged = snapshot::decode(&std::fs::read(&out_path).unwrap()).unwrap();
        assert_eq!(merged, new);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_base_file_reports_missingbase() {
        let dir = scratch("missing");
        let base_path = dir.join("absent.u");
        let delta_path = dir.join("delta.u");
        let out_path = dir.join("out.u");
        let base = base();
        let new = new();
        std::fs::write(&delta_path, delta::fromsnapshots(&base, &new).encode().unwrap()).unwrap();
        let err = mergefiles(&base_path, &delta_path, &out_path).unwrap_err();
        assert!(matches!(err, checkpointerror::missingbase { .. }));
        assert!(!out_path.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// Tests relocated from `crates/checkpoint/src/reader.rs`.
mod reader {
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::support::shapeof;
    use ::checkpoint::{checkpointerror, checkpointreader, delta, snapshot, BLOCKSIZE, MAXBLOCK};
    use tensor::dtype;

    fn base() -> snapshot {
        let mut snap = snapshot::new(1);
        snap.addtensor("a".to_string(), dtype::u8, shapeof(&[BLOCKSIZE]), &vec![1u8; BLOCKSIZE])
            .unwrap();
        snap.addtensor("b".to_string(), dtype::u8, shapeof(&[BLOCKSIZE]), &vec![2u8; BLOCKSIZE])
            .unwrap();
        snap
    }

    fn changed() -> snapshot {
        let mut snap = snapshot::new(1);
        snap.addtensor("a".to_string(), dtype::u8, shapeof(&[BLOCKSIZE]), &vec![9u8; BLOCKSIZE])
            .unwrap();
        snap.addtensor("b".to_string(), dtype::u8, shapeof(&[BLOCKSIZE]), &vec![2u8; BLOCKSIZE])
            .unwrap();
        snap
    }

    #[test]
    fn reads_from_a_full_snapshot() {
        let snap = base();
        let reader = checkpointreader::openfull(snap.encode().unwrap()).unwrap();
        assert_eq!(reader.readtensor("a").unwrap().unwrap(), vec![1u8; BLOCKSIZE]);
        assert_eq!(reader.readtensor("b").unwrap().unwrap(), vec![2u8; BLOCKSIZE]);
        assert!(reader.readtensor("absent").unwrap().is_none());
    }

    #[test]
    fn reads_from_a_delta_without_merging() {
        let base = base();
        let new = changed();
        let d = delta::fromsnapshots(&base, &new);
        let reader =
            checkpointreader::opendelta(base.encode().unwrap(), d.encode().unwrap()).unwrap();
        assert_eq!(reader.readtensor("a").unwrap().unwrap(), vec![9u8; BLOCKSIZE]);
        assert_eq!(reader.readtensor("b").unwrap().unwrap(), vec![2u8; BLOCKSIZE]);
    }

    #[test]
    fn touches_only_the_requested_blocks() {
        let mut snap = snapshot::new(1);
        // "a" is MAXBLOCK so the forced cut puts it in its own block; corrupting
        // "b"'s byte must not affect reading "a".
        snap.addtensor("a".to_string(), dtype::u8, shapeof(&[MAXBLOCK]), &vec![1u8; MAXBLOCK])
            .unwrap();
        snap.addtensor("b".to_string(), dtype::u8, shapeof(&[BLOCKSIZE]), &vec![2u8; BLOCKSIZE])
            .unwrap();
        let mut bytes = snap.encode().unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 1;
        let reader = checkpointreader::openfull(bytes).unwrap();
        assert_eq!(reader.readtensor("a").unwrap().unwrap(), vec![1u8; MAXBLOCK]);
        let err = reader.readtensor("b").unwrap_err();
        assert!(matches!(err, checkpointerror::blockcrcmismatch { .. }));
    }

    fn scratch(tag: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!(
            "llm-reader-{}-{tag}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn openfullfile_reads_tensors() {
        let dir = scratch("full");
        let path = dir.join("snap.u");
        let snap = base();
        std::fs::write(&path, snap.encode().unwrap()).unwrap();
        let reader = checkpointreader::openfullfile(&path).unwrap();
        assert_eq!(reader.readtensor("a").unwrap().unwrap(), vec![1u8; BLOCKSIZE]);
        assert_eq!(reader.readtensor("b").unwrap().unwrap(), vec![2u8; BLOCKSIZE]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn opendeltafile_reads_tensors() {
        let dir = scratch("delta");
        let base_path = dir.join("base.u");
        let delta_path = dir.join("delta.u");
        let base = base();
        let new = changed();
        std::fs::write(&base_path, base.encode().unwrap()).unwrap();
        std::fs::write(&delta_path, delta::fromsnapshots(&base, &new).encode().unwrap()).unwrap();
        let reader = checkpointreader::opendeltafile(&base_path, &delta_path).unwrap();
        assert_eq!(reader.readtensor("a").unwrap().unwrap(), vec![9u8; BLOCKSIZE]);
        assert_eq!(reader.readtensor("b").unwrap().unwrap(), vec![2u8; BLOCKSIZE]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// Tests relocated from `crates/checkpoint/src/compact.rs`.
mod compact {
    use crate::support::shapeof;
    use ::checkpoint::{checkpointerror, compact, delta, snapshot, verifyprovenance, BLOCKSIZE};
    use tensor::dtype;

    fn full(v: u8) -> snapshot {
        let mut snap = snapshot::new(1);
        snap.addtensor("a".to_string(), dtype::u8, shapeof(&[BLOCKSIZE]), &vec![v; BLOCKSIZE])
            .unwrap();
        snap.addtensor("b".to_string(), dtype::u8, shapeof(&[BLOCKSIZE]), &vec![2u8; BLOCKSIZE])
            .unwrap();
        snap
    }

    #[test]
    fn compact_materializes_a_chain() {
        let f0 = full(1);
        let f1 = full(3);
        let f2 = full(5);
        let d1 = delta::fromsnapshots(&f0, &f1);
        let d2 = delta::fromsnapshots(&f1, &f2);
        let result = compact(&f0, &[&d1, &d2]).unwrap();
        assert_eq!(result, f2);
        assert_eq!(result.roothash(), f2.roothash());
    }

    #[test]
    fn verifyprovenance_accepts_a_valid_chain() {
        let f0 = full(1);
        let f1 = full(3);
        let f2 = full(5);
        let d1 = delta::fromsnapshots(&f0, &f1);
        let d2 = delta::fromsnapshots(&f1, &f2);
        verifyprovenance(&f0, &[&d1, &d2]).unwrap();
    }

    #[test]
    fn verifyprovenance_rejects_a_broken_chain() {
        let f0 = full(1);
        let f1 = full(3);
        let f2 = full(5);
        let d1 = delta::fromsnapshots(&f0, &f1);
        let d2 = delta::fromsnapshots(&f0, &f2);
        let err = verifyprovenance(&f0, &[&d1, &d2]).unwrap_err();
        assert!(matches!(err, checkpointerror::basehashmismatch { .. }));
    }
}

/// Tests for per-block LZ4 compression of `.u` snapshots.
mod compression {
    use crate::support::shapeof;
    use ::checkpoint::{checkpointerror, checkpointreader, snapshot, BLOCKSIZE};
    use tensor::dtype;

    fn snapwith(data: &[u8]) -> snapshot {
        let mut snap = snapshot::new(1);
        snap.addtensor("w".to_string(), dtype::u8, shapeof(&[data.len()]), data)
            .unwrap();
        snap
    }

    #[test]
    fn compressible_data_shrinks_and_roundtrips() {
        let data = vec![7u8; BLOCKSIZE * 4];
        let snap = snapwith(&data);
        let encoded = snap.encode().unwrap();
        assert!(
            encoded.len() < data.len() / 8,
            "compressed {} vs raw {}",
            encoded.len(),
            data.len()
        );
        let back = snapshot::decode(&encoded).unwrap();
        assert_eq!(back.readtensor("w").unwrap(), data);
    }

    #[test]
    fn incompressible_data_roundtrips() {
        let mut data = vec![0u8; BLOCKSIZE];
        let mut state = 0x1234_5678u32;
        for byte in &mut data {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            *byte = (state >> 24) as u8;
        }
        let snap = snapwith(&data);
        let encoded = snap.encode().unwrap();
        // Random data is stored raw, so the artifact is only slightly larger
        // than the raw bytes.
        assert!(encoded.len() < data.len() + 4096);
        let back = snapshot::decode(&encoded).unwrap();
        assert_eq!(back.readtensor("w").unwrap(), data);
    }

    #[test]
    fn corruption_in_stored_data_is_detected() {
        let data = vec![7u8; BLOCKSIZE * 2];
        let snap = snapwith(&data);
        let mut encoded = snap.encode().unwrap();
        let last = encoded.len() - 1;
        encoded[last] ^= 1;
        let err = snapshot::decode(&encoded).unwrap_err();
        assert!(matches!(err, checkpointerror::blockcrcmismatch { .. }));
        assert!(err.iscorruption());
    }

    #[test]
    fn reader_reads_compressed_blocks() {
        let data = vec![5u8; BLOCKSIZE * 3];
        let snap = snapwith(&data);
        let reader = checkpointreader::openfull(snap.encode().unwrap()).unwrap();
        assert_eq!(reader.readtensor("w").unwrap().unwrap(), data);
    }
}

/// Tests for the quantization descriptor and scale-as-tensor.
mod quant {
    use crate::support::shapeof;
    use ::checkpoint::{delta, quantdesc, snapshot};
    use tensor::dtype;

    #[test]
    fn quantized_tensor_roundtrips_with_its_scale() {
        let mut snap = snapshot::new(1);
        snap.addtensor("w".to_string(), dtype::i8, shapeof(&[4096]), &vec![1u8; 4096])
            .unwrap();
        snap.addtensor("w_scale".to_string(), dtype::f32, shapeof(&[32]), &[0u8; 128])
            .unwrap();
        snap.addtensor("w_zp".to_string(), dtype::i8, shapeof(&[32]), &[0u8; 32])
            .unwrap();
        snap.setquant(
            "w",
            quantdesc::new(128, "w_scale".to_string()).withzeropoint("w_zp".to_string()),
        )
        .unwrap();
        let bytes = snap.encode().unwrap();
        let back = snapshot::decode(&bytes).unwrap();
        let entry = back
            .gettensors()
            .iter()
            .find(|t| t.getname() == "w")
            .unwrap();
        let q = entry.getquant().unwrap();
        assert_eq!(q.getgroupsize(), 128);
        assert_eq!(q.getscale(), "w_scale");
        assert_eq!(q.getzeropoint(), Some("w_zp"));
        assert_eq!(back.readtensor("w_scale").unwrap().len(), 128);
    }

    #[test]
    fn quantdesc_survives_a_delta() {
        let mut base = snapshot::new(1);
        base.addtensor("w".to_string(), dtype::i8, shapeof(&[4096]), &vec![1u8; 4096])
            .unwrap();
        base.addtensor("w_scale".to_string(), dtype::f32, shapeof(&[32]), &[0u8; 128])
            .unwrap();
        base.setquant("w", quantdesc::new(128, "w_scale".to_string()))
            .unwrap();
        let mut new = snapshot::new(1);
        new.addtensor("w".to_string(), dtype::i8, shapeof(&[4096]), &vec![2u8; 4096])
            .unwrap();
        new.addtensor("w_scale".to_string(), dtype::f32, shapeof(&[32]), &[0u8; 128])
            .unwrap();
        new.setquant("w", quantdesc::new(128, "w_scale".to_string()))
            .unwrap();
        let d = delta::fromsnapshots(&base, &new);
        let back = delta::decode(&d.encode().unwrap()).unwrap();
        let entry = back
            .gettensors()
            .iter()
            .find(|t| t.getname() == "w")
            .unwrap();
        assert_eq!(entry.getquant().unwrap().getscale(), "w_scale");
    }
}
