//! Cross-crate tests for the CPU serial scan kernel through its public API.
//!
//! The serial path is the deterministic reference the parallel scan must match,
//! so these tests pin bit-for-bit agreement with `testkit::reference::scan`
//! and the streaming-vs-full consistency (M11) on the serial path.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use crate::support::{build, wide};
use api::{backenderror, compute};
use cpu::cpuexecutor;
use testkit::reference;

fn exec() -> cpuexecutor {
    cpuexecutor::new(wide())
}

/// Deterministic scan fixture: `n = 3`, `d = 2`, `N = 2`, with `|a| <= 1`.
#[allow(clippy::type_complexity)]
fn fixture() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let input = vec![1.0, 2.0, 0.5, -1.0, 3.0, 0.25];
    let init = vec![0.0, 1.0, 2.0, 0.5];
    let a = vec![0.5, -0.25, 0.75, 0.125];
    let b = vec![1.0, 0.5, 2.0, 0.25];
    let c = vec![0.5, 1.0, 1.5, 2.0];
    let skip = vec![0.25, -0.5];
    (input, init, a, b, c, skip)
}

#[test]
fn scan_matches_reference_bit_for_bit() {
    let exec = exec();
    let (input, init, a, b, c, skip) = fixture();
    let n = 3;
    let d = 2;
    let statewidth = 2;
    let input_t = build(vec![n, d], input.clone());
    let init_t = build(vec![d, statewidth], init.clone());
    let a_t = build(vec![d, statewidth], a.clone());
    let b_t = build(vec![d, statewidth], b.clone());
    let c_t = build(vec![d, statewidth], c.clone());
    let skip_t = build(vec![d], skip.clone());
    let (out, state) = exec
        .scan(
            &input_t.getview(),
            &init_t.getview(),
            &a_t.getview(),
            &b_t.getview(),
            &c_t.getview(),
            &skip_t.getview(),
        )
        .unwrap();
    let (want, wantstate) = reference::scan(&input, &init, &a, &b, &c, &skip, n, d, statewidth);
    assert_eq!(out.getdata(), &want[..]);
    assert_eq!(state.getdata(), &wantstate[..]);
}

#[test]
fn scan_cached_matches_full_scan_step_by_step() {
    let exec = exec();
    let (input, init, a, b, c, skip) = fixture();
    let n = 3;
    let d = 2;
    let statewidth = 2;
    let a_t = build(vec![d, statewidth], a.clone());
    let b_t = build(vec![d, statewidth], b.clone());
    let c_t = build(vec![d, statewidth], c.clone());
    let skip_t = build(vec![d], skip.clone());

    let init_t = build(vec![d, statewidth], init.clone());
    let input_full_t = build(vec![n, d], input.clone());
    let (full_out, full_state) = exec
        .scan(
            &input_full_t.getview(),
            &init_t.getview(),
            &a_t.getview(),
            &b_t.getview(),
            &c_t.getview(),
            &skip_t.getview(),
        )
        .unwrap();

    let mut state = init;
    let mut stepped = Vec::new();
    for t in 0..n {
        let token = vec![input[t * d], input[t * d + 1]];
        let token_t = build(vec![1, d], token);
        let state_t = build(vec![d, statewidth], state.clone());
        let (out, newstate) = exec
            .scan_cached(
                &token_t.getview(),
                &state_t.getview(),
                &a_t.getview(),
                &b_t.getview(),
                &c_t.getview(),
                &skip_t.getview(),
            )
            .unwrap();
        stepped.extend_from_slice(out.getdata());
        state = newstate.getdata().to_vec();
    }
    assert_eq!(&stepped[..], full_out.getdata());
    assert_eq!(&state[..], full_state.getdata());
}

#[test]
fn scan_rejects_rank_mismatch() {
    let exec = exec();
    let input = build(vec![2], vec![1.0, 2.0]);
    let init = build(vec![2, 1], vec![0.0, 0.0]);
    let a = build(vec![2, 1], vec![0.5, 0.5]);
    let b = build(vec![2, 1], vec![1.0, 1.0]);
    let c = build(vec![2, 1], vec![1.0, 1.0]);
    let skip = build(vec![2], vec![0.0, 0.0]);
    assert_eq!(
        exec.scan(
            &input.getview(),
            &init.getview(),
            &a.getview(),
            &b.getview(),
            &c.getview(),
            &skip.getview(),
        )
        .map(|_| ()),
        Err(backenderror::shapemismatch { op: "scan" })
    );
}

#[test]
fn scan_rejects_channel_mismatch() {
    let exec = exec();
    let input = build(vec![3, 2], vec![0.0; 6]);
    let init = build(vec![2, 1], vec![0.0, 0.0]);
    let a = build(vec![3, 1], vec![0.5, 0.5, 0.5]);
    let b = build(vec![2, 1], vec![1.0, 1.0]);
    let c = build(vec![2, 1], vec![1.0, 1.0]);
    let skip = build(vec![2], vec![0.0, 0.0]);
    assert_eq!(
        exec.scan(
            &input.getview(),
            &init.getview(),
            &a.getview(),
            &b.getview(),
            &c.getview(),
            &skip.getview(),
        )
        .map(|_| ()),
        Err(backenderror::shapemismatch { op: "scan" })
    );
}

#[test]
fn scan_rejects_state_width_mismatch() {
    let exec = exec();
    let input = build(vec![3, 2], vec![0.0; 6]);
    let init = build(vec![2, 2], vec![0.0; 4]);
    let a = build(vec![2, 1], vec![0.5, 0.5]);
    let b = build(vec![2, 2], vec![1.0; 4]);
    let c = build(vec![2, 2], vec![1.0; 4]);
    let skip = build(vec![2], vec![0.0, 0.0]);
    assert_eq!(
        exec.scan(
            &input.getview(),
            &init.getview(),
            &a.getview(),
            &b.getview(),
            &c.getview(),
            &skip.getview(),
        )
        .map(|_| ()),
        Err(backenderror::shapemismatch { op: "scan" })
    );
}

#[test]
fn scan_rejects_spectral_violation() {
    let exec = exec();
    let input = build(vec![1, 2], vec![1.0, 1.0]);
    let init = build(vec![2, 1], vec![0.0, 0.0]);
    let a = build(vec![2, 1], vec![0.5, -1.5]);
    let b = build(vec![2, 1], vec![1.0, 1.0]);
    let c = build(vec![2, 1], vec![1.0, 1.0]);
    let skip = build(vec![2], vec![0.0, 0.0]);
    assert_eq!(
        exec.scan(
            &input.getview(),
            &init.getview(),
            &a.getview(),
            &b.getview(),
            &c.getview(),
            &skip.getview(),
        )
        .map(|_| ()),
        Err(backenderror::invalidparameter { op: "scan" })
    );
}

#[test]
fn scan_empty_sequence_returns_init_unchanged() {
    let exec = exec();
    let input = build(vec![0, 2], Vec::new());
    let init = build(vec![2, 1], vec![7.0, 8.0]);
    let a = build(vec![2, 1], vec![0.5, 0.5]);
    let b = build(vec![2, 1], vec![1.0, 1.0]);
    let c = build(vec![2, 1], vec![1.0, 1.0]);
    let skip = build(vec![2], vec![0.0, 0.0]);
    let (out, state) = exec
        .scan(
            &input.getview(),
            &init.getview(),
            &a.getview(),
            &b.getview(),
            &c.getview(),
            &skip.getview(),
        )
        .unwrap();
    assert!(out.isempty());
    assert_eq!(state.getdata(), &[7.0, 8.0]);
}

#[test]
fn scan_cached_rejects_multi_token() {
    let exec = exec();
    let input = build(vec![2, 2], vec![1.0; 4]);
    let state = build(vec![2, 1], vec![0.0, 0.0]);
    let a = build(vec![2, 1], vec![0.5, 0.5]);
    let b = build(vec![2, 1], vec![1.0, 1.0]);
    let c = build(vec![2, 1], vec![1.0, 1.0]);
    let skip = build(vec![2], vec![0.0, 0.0]);
    assert_eq!(
        exec.scan_cached(
            &input.getview(),
            &state.getview(),
            &a.getview(),
            &b.getview(),
            &c.getview(),
            &skip.getview(),
        )
        .map(|_| ()),
        Err(backenderror::shapemismatch { op: "scan" })
    );
}
