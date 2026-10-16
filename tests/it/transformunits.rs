//! Tests for the bit-shuffle + delta lossless pre-pass (`FLAGTRANSFORMED`):
//! exact round trips through the public `checkpoint` API for f32 and f16
//! weight profiles, mixed-dtype snapshots, encryption, deltas, the on-demand
//! reader, and the direct inverse property of the transform itself.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use ::checkpoint::blocks::{bitshuffle_delta, unbitshuffle_delta};
use ::checkpoint::checkpointreader;
use ::checkpoint::deltas::delta;
use ::checkpoint::uformat::snapshot;
use tensor::{dtype, shape};

fn shapeof(values: &[usize]) -> shape {
    shape::new(values.to_vec()).unwrap()
}

fn xorshift64(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

/// Correlated random walk: the realistic weight structure the transform is
/// designed for. `f16` selects the 2-byte little-endian storage layout.
fn gen_weights(seed: &mut u64, count: usize, f16: bool) -> Vec<u8> {
    let mut out = Vec::with_capacity(count * if f16 { 2 } else { 4 });
    let mut val: f32 = 0.0;
    for _ in 0..count {
        let r = xorshift64(seed) % 100;
        let delta: f32 = if r < 70 {
            (xorshift64(seed) % 2000) as f32 / 1e6 - 1e-3
        } else if r < 90 {
            (xorshift64(seed) % 200_000) as f32 / 1e6 - 0.1
        } else {
            0.0
        };
        val += delta;
        if val > 1.0 {
            val -= 2.0;
        }
        if val < -1.0 {
            val += 2.0;
        }
        if f16 {
            let bits = val.to_bits();
            let sign = (bits >> 16) & 0x8000;
            let exp = ((bits >> 23) & 0xff) as i32;
            let mant = bits & 0x7fffff;
            let f16 = if exp >= 0x71 {
                (sign | 0x7c00) as u16
            } else if exp == 0 {
                sign as u16
            } else {
                (sign | (((exp - 127 + 15) as u32) << 10) | (mant >> 13)) as u16
            };
            out.extend_from_slice(&f16.to_le_bytes());
        } else {
            out.extend_from_slice(&val.to_le_bytes());
        }
    }
    out
}

/// Builds a single-tensor snapshot whose bytes span several blocks.
fn weight_snapshot(name: &str, dtype: dtype, bytes: &[u8]) -> snapshot {
    let mut snap = snapshot::new(1);
    let count = bytes.len() / dtype.sizeinbytes();
    snap.addtensor(name.to_string(), dtype, shapeof(&[count]), bytes)
        .unwrap();
    snap
}

mod transform_units {
    use super::*;

    #[test]
    fn bitshuffle_delta_inverts_exactly() {
        let mut seed = 0x1234_5678_9abc_def0u64;
        for f16 in [false, true] {
            let data = gen_weights(&mut seed, 300_000, f16);
            let elem = if f16 { 2 } else { 4 };
            let transformed = bitshuffle_delta(&data, elem);
            assert_eq!(transformed.len(), data.len());
            assert_eq!(unbitshuffle_delta(&transformed, elem), data);
        }
    }

    #[test]
    fn bitshuffle_delta_inverts_non_aligned_lengths() {
        // CDC block boundaries are arbitrary byte offsets; the transform must
        // invert blocks whose length is not a multiple of the element size.
        let mut seed = 0x0ddb_00d5_0000_0001u64;
        let data = gen_weights(&mut seed, 100_000, false);
        for len in 1..=8 {
            let block = &data[..data.len() - len];
            for elem in [2usize, 4] {
                let transformed = bitshuffle_delta(block, elem);
                assert_eq!(transformed.len(), block.len());
                assert_eq!(
                    unbitshuffle_delta(&transformed, elem),
                    block,
                    "len % {elem} = {}",
                    block.len() % elem
                );
            }
        }
    }

    #[test]
    fn bitshuffle_delta_is_deterministic() {
        let mut seed = 0x1234_5678_9abc_def0u64;
        let data = gen_weights(&mut seed, 100_000, false);
        assert_eq!(bitshuffle_delta(&data, 4), bitshuffle_delta(&data, 4));
        assert_ne!(bitshuffle_delta(&data, 4), data, "pre-pass must reorder");
    }

    #[test]
    fn f32_weights_snapshot_roundtrips_and_shrinks() {
        let mut seed = 0x1234_5678_9abc_def0u64;
        let bytes = gen_weights(&mut seed, 400_000, false); // ~1.6 MiB, several blocks
        let snap = weight_snapshot("w", dtype::f32, &bytes);
        let encoded = snap.encode().unwrap();
        let decoded = snapshot::decode(&encoded).unwrap();
        assert_eq!(decoded.readtensor("w").unwrap(), bytes);
        // The transform + codec picker must shrink correlated weights well
        // below what plain Zstd alone achieves (measured ~17% vs ~39% saved).
        assert!(
            encoded.len() * 100 < bytes.len() * 70,
            "encoded {} bytes should be well under 70% of input {} bytes",
            encoded.len(),
            bytes.len()
        );
    }

    #[test]
    fn f16_weights_snapshot_roundtrips() {
        let mut seed = 0x1111_2222_3333_4444u64;
        let bytes = gen_weights(&mut seed, 500_000, true); // ~1 MiB
        let snap = weight_snapshot("w", dtype::f16, &bytes);
        let encoded = snap.encode().unwrap();
        let decoded = snapshot::decode(&encoded).unwrap();
        assert_eq!(decoded.readtensor("w").unwrap(), bytes);
        assert!(encoded.len() < bytes.len(), "f16 weights must compress");
    }

    #[test]
    fn mixed_dtype_snapshot_roundtrips_across_block_boundaries() {
        let mut seed = 0x9999_0000_aaaa_bbbbu64;
        let f32 = gen_weights(&mut seed, 70_001, false); // 280_004 bytes
        let f16 = gen_weights(&mut seed, 33_333, true); // 66_666 bytes
        let u8 = vec![7u8; 7]; // odd length: total stream is not a multiple of 4
        let mut snap = snapshot::new(1);
        snap.addtensor("a".to_string(), dtype::f32, shapeof(&[70_001]), &f32)
            .unwrap();
        snap.addtensor("b".to_string(), dtype::f16, shapeof(&[33_333]), &f16)
            .unwrap();
        snap.addtensor("c".to_string(), dtype::u8, shapeof(&[7]), &u8)
            .unwrap();
        let encoded = snap.encode().unwrap();
        let decoded = snapshot::decode(&encoded).unwrap();
        assert_eq!(decoded.readtensor("a").unwrap(), f32);
        assert_eq!(decoded.readtensor("b").unwrap(), f16);
        assert_eq!(decoded.readtensor("c").unwrap(), u8);
    }

    #[test]
    fn encrypted_transformed_snapshot_roundtrips() {
        let mut seed = 0x5555_6666_7777_8888u64;
        let bytes = gen_weights(&mut seed, 300_000, false);
        let snap = weight_snapshot("w", dtype::f32, &bytes);
        let key = [0x42u8; 32];
        let encoded = snap.encode_encrypted(&key).unwrap();
        assert!(snapshot::decode(&encoded).is_err(), "missing key must fail");
        let decoded = snapshot::decode_encrypted(&encoded, &key).unwrap();
        assert_eq!(decoded.readtensor("w").unwrap(), bytes);
    }

    #[test]
    fn delta_between_transformed_snapshots_roundtrips_through_reader() {
        let mut seed = 0xabcd_ef01_2345_6789u64;
        let base_bytes = gen_weights(&mut seed, 300_000, false);
        let mut new_bytes = base_bytes.clone();
        // Modify a middle region so only a few blocks change.
        for i in 100_000..120_000 {
            new_bytes[i * 4] = new_bytes[i * 4].wrapping_add(7);
        }
        let base = weight_snapshot("w", dtype::f32, &base_bytes);
        let new = weight_snapshot("w", dtype::f32, &new_bytes);
        let delta = delta::fromsnapshots(&base, &new);
        let base_encoded = base.encode().unwrap();
        let delta_encoded = delta.encode().unwrap();
        let decoded_delta = delta::decode(&delta_encoded).unwrap();
        assert_eq!(decoded_delta.getroothash(), new.roothash());
        let reader = checkpointreader::opendelta(base_encoded, delta_encoded).unwrap();
        assert_eq!(reader.readtensor("w").unwrap().unwrap(), new_bytes);
    }

    #[test]
    fn encrypted_delta_roundtrips_through_reader() {
        let mut seed = 0x0fed_cba9_8765_4321u64;
        let base_bytes = gen_weights(&mut seed, 200_000, false);
        let mut new_bytes = base_bytes.clone();
        for i in 60_000..70_000 {
            new_bytes[i * 4] = new_bytes[i * 4].wrapping_add(3);
        }
        let base = weight_snapshot("w", dtype::f32, &base_bytes);
        let new = weight_snapshot("w", dtype::f32, &new_bytes);
        let delta = delta::fromsnapshots(&base, &new);
        let key = [0x24u8; 32];
        let base_encoded = base.encode_encrypted(&key).unwrap();
        let delta_encoded = delta.encode_encrypted(&key).unwrap();
        let reader =
            checkpointreader::opendelta_encrypted(base_encoded, delta_encoded, &key).unwrap();
        assert_eq!(reader.readtensor("w").unwrap().unwrap(), new_bytes);
    }

    #[test]
    fn transformed_reader_reads_tensor_exactly() {
        let mut seed = 0x3141_5926_5358_9793u64;
        let bytes = gen_weights(&mut seed, 250_000, false);
        let snap = weight_snapshot("w", dtype::f32, &bytes);
        let encoded = snap.encode().unwrap();
        let reader = checkpointreader::openfull(encoded).unwrap();
        let cow = reader.readtensorcow("w").unwrap().unwrap();
        assert_eq!(cow.as_ref(), bytes.as_slice());
    }

    #[test]
    fn signed_transformed_snapshot_verifies() {
        let mut seed = 0x8888_9999_aaaa_bbbbu64;
        let bytes = gen_weights(&mut seed, 150_000, false);
        let snap = weight_snapshot("w", dtype::f32, &bytes);
        let seed_bytes = [0x11u8; 32];
        let encoded = snap.encode_signed(&seed_bytes).unwrap();
        let decoded = snapshot::decode(&encoded).unwrap();
        assert_eq!(decoded.readtensor("w").unwrap(), bytes);
        let public = ::checkpoint::signature::publickey(&seed_bytes).unwrap();
        decoded.verifysignature(&public).unwrap();
    }

    #[test]
    fn encode_is_deterministic() {
        let mut seed = 0x7777_8888_9999_0000u64;
        let bytes = gen_weights(&mut seed, 120_000, false);
        let snap = weight_snapshot("w", dtype::f32, &bytes);
        assert_eq!(snap.encode().unwrap(), snap.encode().unwrap());
    }
}
