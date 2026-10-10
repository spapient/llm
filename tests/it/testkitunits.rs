//! Relocated unit tests for the `testkit` crate.
//!
//! Formerly inline `#[cfg(test)] mod tests` blocks, moved here so that
//! production sources under `crates/testkit/src` hold no test code. Each source
//! module's tests are wrapped in a nested module to keep helper and test names
//! isolated. Every assertion reaches the crate through its public interface
//! (see `docs/engineering.md`).
#![allow(clippy::unwrap_used, clippy::expect_used, dead_code)]

/// Tests relocated from `crates/testkit/src/assertions.rs`.
mod assertions {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use testkit::assertions::assertallclose;

    #[test]
    fn identical_slices_pass() {
        assertallclose(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], 0.0, 0.0);
    }

    #[test]
    fn within_tolerance_passes() {
        assertallclose(&[1.0, 2.0], &[1.0 + 1e-7, 2.0 - 1e-7], 1e-6, 1e-6);
    }

    #[test]
    #[should_panic(expected = "mismatch at index")]
    fn beyond_tolerance_panics() {
        assertallclose(&[1.0], &[1.1], 1e-6, 1e-6);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn length_mismatch_panics() {
        assertallclose(&[1.0], &[1.0, 2.0], 0.0, 0.0);
    }

    #[test]
    #[should_panic(expected = "abstol must be finite")]
    fn negative_tolerance_is_rejected() {
        assertallclose(&[1.0], &[1.0], -1.0, 0.0);
    }
}

/// Tests relocated from `crates/testkit/src/fixtures.rs`.
mod fixtures {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use {
        testkit::fixtures::constant, testkit::fixtures::matrix, testkit::fixtures::ramp,
        testkit::fixtures::sequential,
    };

    #[test]
    fn sequential_is_row_major_ramp() {
        let t = sequential(2, 3).unwrap();
        assert_eq!(t.getshape().getdims(), &[2, 3]);
        assert_eq!(t.getdata(), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn ramp_matches_length() {
        let t = ramp(4).unwrap();
        assert_eq!(t.getdata(), &[0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn constant_repeats_value() {
        let t = constant(3, 0.5).unwrap();
        assert_eq!(t.getdata(), &[0.5, 0.5, 0.5]);
    }

    #[test]
    fn matrix_fills_every_element() {
        let t = matrix(2, 2, 2.0).unwrap();
        assert_eq!(t.getdata(), &[2.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn overflow_is_reported_not_panicked() {
        assert!(matrix(usize::MAX, 2, 0.0).is_err());
    }
}

/// Tests relocated from `crates/testkit/src/reference.rs`.
mod reference {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use {
        testkit::reference::embedding, testkit::reference::gelutanh, testkit::reference::map,
        testkit::reference::matmul, testkit::reference::relu, testkit::reference::rmsnorm,
        testkit::reference::rope, testkit::reference::silu, testkit::reference::softmax,
    };

    #[test]
    fn relu_clamps_negatives() {
        assert_eq!(relu(-2.0), 0.0);
        assert_eq!(relu(3.0), 3.0);
    }

    #[test]
    fn silu_of_zero_is_zero() {
        assert_eq!(silu(0.0), 0.0);
    }

    #[test]
    fn gelutanh_matches_known_constants() {
        assert!((gelutanh(1.0) - 0.841_192).abs() < 1e-6);
        assert!((gelutanh(-1.0) + 0.158_808).abs() < 1e-6);
        assert!((gelutanh(2.0) - 1.954_597).abs() < 1e-5);
    }

    #[test]
    fn softmax_rows_sum_to_one() {
        let out = softmax(&[1.0, 2.0, 3.0]);
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn softmax_is_stable_for_extreme_logits() {
        let out = softmax(&[1000.0, 1000.0]);
        assert!((out[0] - 0.5).abs() < 1e-6);
        assert!((out[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn map_applies_function_elementwise() {
        assert_eq!(map(&[-1.0, 0.0, 1.0], relu), vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn matmul_matches_hand_computation() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        assert_eq!(matmul(&a, &b, 2, 3, 2), vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn rmsnorm_scales_by_inverse_root_mean_square() {
        let out = rmsnorm(&[3.0, 4.0], &[1.0, 1.0], 0.0);
        assert!((out[0] - 0.848_528_1).abs() < 1e-6);
        assert!((out[1] - 1.131_370_8).abs() < 1e-6);
    }

    #[test]
    fn rope_rotates_pairs_by_position() {
        let out = rope(&[1.0, 0.0], 1, 1, 2, 10000.0);
        assert!((out[0] - 1.0_f32.cos()).abs() < 1e-6);
        assert!((out[1] - 1.0_f32.sin()).abs() < 1e-6);
    }

    #[test]
    fn embedding_gathers_rows_in_index_order() {
        let table = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(embedding(&table, 2, &[2, 0]), vec![4.0, 5.0, 0.0, 1.0]);
    }
}
