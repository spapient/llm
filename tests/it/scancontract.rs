//! Contract tests for the semigroup scan scalar reference, backing the β-channel
//! scan operator (`docs/beta-scan-plan.md`).
//!
//! `compute::scan`/`scan_cached` and the `submission` `record*` methods are
//! added additively and stubbed in the three backends; their numerical
//! implementations land in later stages. These tests pin the scalar reference's
//! correctness and its agreement with an independent semigroup prefix, so the
//! parallel scan has an oracle to match.

use testkit::assertions::assertallclose;
use testkit::reference::scan;

/// Independent semigroup-prefix reference: accumulates `(A, b)` pairs with the
/// semigroup combine `(A2 A1, A2 b1 + b2)` and applies the prefix to `init`.
/// Shares no code with [`scan`].
#[allow(clippy::too_many_arguments)]
fn semigroupprefix(
    input: &[f32],
    init: &[f32],
    a: &[f32],
    b: &[f32],
    c: &[f32],
    skip: &[f32],
    n: usize,
    d: usize,
    statewidth: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut acum = vec![1.0_f32; d * statewidth];
    let mut bcum = vec![0.0_f32; d * statewidth];
    let mut out = vec![0.0_f32; n * d];
    for t in 0..n {
        for i in 0..d {
            let x = input[t * d + i];
            for j in 0..statewidth {
                let idx = i * statewidth + j;
                bcum[idx] = a[idx] * bcum[idx] + b[idx] * x;
                acum[idx] *= a[idx];
            }
        }
        for i in 0..d {
            let mut acc = 0.0_f32;
            for j in 0..statewidth {
                let idx = i * statewidth + j;
                acc += c[idx] * (acum[idx] * init[idx] + bcum[idx]);
            }
            out[t * d + i] = acc + skip[i] * input[t * d + i];
        }
    }
    let mut state = vec![0.0_f32; d * statewidth];
    for i in 0..d {
        for j in 0..statewidth {
            let idx = i * statewidth + j;
            state[idx] = acum[idx] * init[idx] + bcum[idx];
        }
    }
    (out, state)
}

#[test]
fn scan_reference_matches_hand_computed() {
    // d = 1, N = 1, exact decimal values so every operation is bit-exact.
    let (out, state) = scan(&[2.0, 3.0], &[1.0], &[0.5], &[1.0], &[2.0], &[0.0], 2, 1, 1);
    assert_eq!(out, vec![5.0, 8.5]);
    assert_eq!(state, vec![4.25]);
}

#[test]
fn scan_reference_matches_semigroup_prefix_on_64_points() {
    let n = 64;
    let d = 3;
    let statewidth = 4;
    let input: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.37).sin()).collect();
    let init: Vec<f32> = (0..d * statewidth)
        .map(|i| (i as f32 * 0.13).cos())
        .collect();
    let a: Vec<f32> = (0..d * statewidth)
        .map(|i| 0.5 + 0.5 * (i as f32 * 0.71).cos().abs())
        .collect();
    let b: Vec<f32> = (0..d * statewidth)
        .map(|i| (i as f32 * 0.29).sin())
        .collect();
    let c: Vec<f32> = (0..d * statewidth)
        .map(|i| (i as f32 * 0.19).cos())
        .collect();
    let skip: Vec<f32> = (0..d).map(|i| (i as f32 * 0.11).sin()).collect();

    let (got, gotstate) = scan(&input, &init, &a, &b, &c, &skip, n, d, statewidth);
    let (want, wantstate) = semigroupprefix(&input, &init, &a, &b, &c, &skip, n, d, statewidth);

    assertallclose(&got, &want, 1e-6, 1e-5);
    assertallclose(&gotstate, &wantstate, 1e-6, 1e-5);
}

#[test]
fn scan_reference_empty_sequence_returns_init_unchanged() {
    let init = [7.0_f32, 8.0_f32];
    let (out, state) = scan(
        &[],
        &init,
        &[0.5, 0.5],
        &[1.0, 1.0],
        &[1.0, 1.0],
        &[0.0, 0.0],
        0,
        2,
        1,
    );
    assert!(out.is_empty());
    assert_eq!(state, init);
}

#[test]
fn scan_reference_handles_non_power_of_two_length() {
    let n = 5;
    let d = 2;
    let statewidth = 2;
    let input: Vec<f32> = (0..n * d).map(|i| i as f32 + 0.5).collect();
    let init: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let a: Vec<f32> = vec![0.5, 0.25, 0.75, 0.125];
    let b: Vec<f32> = vec![1.0, 0.5, 2.0, 0.25];
    let c: Vec<f32> = vec![0.5, 1.0, 1.5, 2.0];
    let skip: Vec<f32> = vec![0.0, 0.0];

    let (got, gotstate) = scan(&input, &init, &a, &b, &c, &skip, n, d, statewidth);
    let (want, wantstate) = semigroupprefix(&input, &init, &a, &b, &c, &skip, n, d, statewidth);

    assertallclose(&got, &want, 1e-6, 1e-5);
    assertallclose(&gotstate, &wantstate, 1e-6, 1e-5);
}
