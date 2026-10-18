//! Contract tests for the SYCL backend (Intel oneAPI DPC++), exercised only
//! through its public API from outside the crate.
//!
//! Linux and Windows hosts with the oneAPI toolkit: on every other target the
//! `sycl` module is compiled out, so the whole suite is compiled out there. The
//! `sycl_toolkit` cfg is emitted by the root build script when the oneAPI
//! driver exists.
//!
//! # Environment assumptions
//!
//! The numerical suites require an Intel GPU visible to the oneAPI runtime
//! (`sycl-ls` reports it). Per docs/engineering.md, a test that cannot run in
//! its environment must be reported as *not executed*, never as passed:
//!
//! - The device probe is `sycl-ls`, shared with every GPU backend through
//!   [`testkit::probe`]. When it is absent or reports no device, the suite
//!   prints `E4 未执行` and returns without asserting the numerical result.
//! - Setting `LLM_SYCL_REQUIRE_DEVICE=1` turns the same condition into a hard
//!   failure, so a missing device can never be mistaken for a passing
//!   numerical check.
//!
//! # What runs without a device
//!
//! The identity and capability declarations settle on the host and run
//! unconditionally when the module compiles in.
#![cfg(all(any(target_os = "linux", target_os = "windows"), sycl_toolkit))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use crate::support::{build, wide};
use api::{backend, backenderror, compute, operand, session, stepref, submission};
use gpu::sycl::executor as syclexecutor;
use lib::limits;
use testkit::{assertions::assertallclose, probe::gpuprobe, reference};

/// The device gate shared by every numerical case.
fn hasdevice(test: &str) -> bool {
    testkit::probe::requiredevice(test, gpuprobe::sycl)
}

#[test]
fn backend_exposes_identity_and_capabilities() {
    let exec = syclexecutor::new(wide());
    let b = exec.getbackend();
    assert_eq!(b.getname(), "sycl");
    assert!(b.getdevice().issycl());
    assert_eq!(b.getdevice(), api::device::sycl());
}

#[test]
fn add_matches_the_scalar_reference() {
    if !hasdevice("add_matches_the_scalar_reference") {
        return;
    }
    let exec = syclexecutor::new(wide());
    let a = build(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = build(vec![2, 3], vec![0.5, -1.0, 2.0, 1.5, -0.25, 0.75]);
    let out = exec.add(&a.getview(), &b.getview()).unwrap();
    let want: Vec<f32> = a
        .getdata()
        .iter()
        .zip(b.getdata().iter())
        .map(|(&x, &y)| x + y)
        .collect();
    assertallclose(out.getdata(), &want, 1.0e-6, 1.0e-6);
}

#[test]
fn matmul_matches_the_scalar_reference() {
    if !hasdevice("matmul_matches_the_scalar_reference") {
        return;
    }
    let exec = syclexecutor::new(wide());
    let a = build(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = build(vec![3, 2], vec![0.5, -1.0, 2.0, 1.5, -0.25, 0.75]);
    let out = exec.matmul(&a.getview(), &b.getview()).unwrap();
    assertallclose(
        out.getdata(),
        &reference::matmul(a.getdata(), b.getdata(), 2, 3, 2),
        1.0e-5,
        1.0e-5,
    );
}

#[test]
fn rmsnorm_matches_the_scalar_reference_per_row() {
    if !hasdevice("rmsnorm_matches_the_scalar_reference_per_row") {
        return;
    }
    let exec = syclexecutor::new(wide());
    let x = build(
        vec![4, 8],
        vec![
            0.5, 1.0, -1.5, 2.0, 0.25, -0.75, 1.25, -2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
            -0.5, 0.5, -1.0, 1.0, -2.0, 2.0, -3.0, 3.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 5.0, -5.0,
        ],
    );
    let w = build(vec![8], vec![1.0; 8]);
    let out = exec.rmsnorm(&x.getview(), &w.getview(), 1.0e-5).unwrap();
    let mut want = Vec::new();
    for row in x.getdata().chunks(8) {
        want.extend(reference::rmsnorm(row, &[1.0; 8], 1.0e-5));
    }
    assertallclose(out.getdata(), &want, 1.0e-4, 1.0e-4);
}

#[test]
fn matmul_rejects_mismatched_shapes() {
    let exec = syclexecutor::new(limits::new(16, 64, 8, 2));
    let a = build(vec![2, 3], vec![0.0; 6]);
    let b = build(vec![2, 2], vec![0.0; 4]);
    assert_eq!(
        exec.matmul(&a.getview(), &b.getview()).map(|_| ()),
        Err(api::backenderror::shapemismatch { op: "matmul" })
    );
}

#[test]
fn attention_head_dim_beyond_the_launch_ceiling_is_rejected_identically() {
    if !hasdevice("attention_head_dim_beyond_the_launch_ceiling_is_rejected_identically") {
        return;
    }
    let exec = syclexecutor::new(wide());
    // 4097 is inside the shared `MAXHEADDIM` (8192) but beyond the launch
    // guard `ATTNLIMIT` (4096) shared by every GRID backend; the direct and
    // recorded paths must report the same error variant.
    let headdim = 4097;
    let q = build(vec![1, headdim], vec![0.0; headdim]);
    let k = build(vec![1, headdim], vec![0.0; headdim]);
    let v = build(vec![1, headdim], vec![0.0; headdim]);

    assert_eq!(
        exec.attention(&q.getview(), &k.getview(), &v.getview(), 1, 1, true)
            .map(|_| ()),
        Err(backenderror::sizeoverflow)
    );

    let mut sub = exec.newsubmission().unwrap();
    let err = sub
        .recordattention(
            operand::input(q.getview()),
            operand::input(k.getview()),
            operand::input(v.getview()),
            1,
            1,
            true,
        )
        .map(|_| ())
        .unwrap_err();
    assert_eq!(err, backenderror::sizeoverflow);
}

#[test]
fn batched_matmul_matches_the_scalar_reference() {
    if !hasdevice("batched_matmul_matches_the_scalar_reference") {
        return;
    }
    let exec = syclexecutor::new(wide());
    let batch = 2;
    let (m, k, n) = (2, 3, 2);
    let adata: Vec<f32> = (1..=12).map(|v| v as f32 * 0.5).collect();
    let bdata: Vec<f32> = (1..=12).map(|v| (v as f32).sin()).collect();
    let a = build(vec![batch, m, k], adata.clone());
    let b = build(vec![batch, k, n], bdata.clone());

    let out = exec.matmul(&a.getview(), &b.getview()).unwrap();

    let mut want = Vec::new();
    for bi in 0..batch {
        want.extend(reference::matmul(
            &adata[bi * m * k..(bi + 1) * m * k],
            &bdata[bi * k * n..(bi + 1) * k * n],
            m,
            k,
            n,
        ));
    }
    assertallclose(out.getdata(), &want, 1.0e-5, 1.0e-5);
}

#[test]
fn matmul_handles_partial_tiles_and_unaligned_strides() {
    if !hasdevice("matmul_handles_partial_tiles_and_unaligned_strides") {
        return;
    }
    let exec = syclexecutor::new(wide());
    // The tiled kernel splits the output into 64x64 tiles and stages 16
    // rows of K at a time. These shapes exercise: a row tile that is not a
    // multiple of 64 (65), a K that is not a multiple of 4 (17), a column
    // tile that is not a multiple of 64 (70), the smallest legal shape, a K
    // that is a multiple of 4 but not 16 (20), and an exactly full tile.
    let shapes = [
        (65u32, 17u32, 70u32, 2u32),
        (1, 1, 1, 1),
        (3, 20, 66, 1),
        (64, 16, 64, 3),
    ];
    for (m, k, n, batch) in shapes {
        let (m, k, n, batch) = (m as usize, k as usize, n as usize, batch as usize);
        let adata: Vec<f32> = (0..batch * m * k)
            .map(|i| ((i % 29) as f32 - 14.0) * 0.125)
            .collect();
        let bdata: Vec<f32> = (0..batch * k * n)
            .map(|i| ((i % 31) as f32 - 15.0) * 0.0625)
            .collect();
        let a = build(vec![batch, m, k], adata.clone());
        let b = build(vec![batch, k, n], bdata.clone());

        let out = exec.matmul(&a.getview(), &b.getview()).unwrap();

        let mut want = Vec::new();
        for bi in 0..batch {
            let abase = bi * m * k;
            let bbase = bi * k * n;
            want.extend(reference::matmul(
                &adata[abase..abase + m * k],
                &bdata[bbase..bbase + k * n],
                m,
                k,
                n,
            ));
        }
        assert_eq!(out.getdata().len(), want.len(), "m={m} k={k} n={n}");
        assertallclose(out.getdata(), &want, 1.0e-4, 1.0e-4);
    }
}

#[test]
fn submission_reproduces_a_compute_chain_on_the_device() {
    if !hasdevice("submission_reproduces_a_compute_chain_on_the_device") {
        return;
    }
    let exec = syclexecutor::new(wide());
    let a = build(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = build(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let c = build(vec![2, 2], vec![0.5, -0.5, 1.5, -1.5]);

    let mut sub = exec.newsubmission().unwrap();
    let matmul = sub
        .recordmatmul(operand::input(a.getview()), operand::input(b.getview()))
        .unwrap();
    let sum = sub
        .recordadd(operand::result(matmul), operand::input(c.getview()))
        .unwrap();
    let silu = sub.recordsilu(operand::result(sum)).unwrap();
    let got = sub.apply(&[silu]).unwrap();

    let want_matmul = exec.matmul(&a.getview(), &b.getview()).unwrap();
    let want_sum = exec.add(&want_matmul.getview(), &c.getview()).unwrap();
    let want_silu = exec.silu(&want_sum.getview()).unwrap();

    assert_eq!(got.len(), 1);
    assertallclose(got[0].getdata(), want_silu.getdata(), 1.0e-6, 1.0e-6);
}

#[test]
fn batched_matmul_records_through_a_submission() {
    if !hasdevice("batched_matmul_records_through_a_submission") {
        return;
    }
    let exec = syclexecutor::new(wide());
    let batch = 2;
    let (m, k, n) = (2, 3, 2);
    let a = build(
        vec![batch, m, k],
        (1..=12).map(|v| v as f32 * 0.5).collect(),
    );
    let b = build(
        vec![batch, k, n],
        (1..=12).map(|v| (v as f32).sin()).collect(),
    );

    let mut sub = exec.newsubmission().unwrap();
    let step = sub
        .recordmatmul(operand::input(a.getview()), operand::input(b.getview()))
        .unwrap();
    let got = sub.apply(&[step]).unwrap();

    let want = exec.matmul(&a.getview(), &b.getview()).unwrap();
    assert_eq!(got.len(), 1);
    assertallclose(got[0].getdata(), want.getdata(), 1.0e-6, 1.0e-6);
}

#[test]
fn session_reuses_one_device_state_across_applies() {
    if !hasdevice("session_reuses_one_device_state_across_applies") {
        return;
    }
    let exec = syclexecutor::new(wide());
    let a = build(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = build(vec![2, 2], vec![0.5, -0.5, 1.5, -1.5]);

    let mut sess = exec.newsession().unwrap();
    for _ in 0..2 {
        let mul = api::recorder::recordmul(
            &mut sess,
            operand::input(a.getview()),
            operand::input(b.getview()),
        )
        .unwrap();
        let sum =
            api::recorder::recordadd(&mut sess, operand::result(mul), operand::input(a.getview()))
                .unwrap();
        let got = sess.apply(&[sum]).unwrap();

        let want_mul = exec.mul(&a.getview(), &b.getview()).unwrap();
        let want = exec.add(&want_mul.getview(), &a.getview()).unwrap();
        assertallclose(got[0].getdata(), want.getdata(), 1.0e-6, 1.0e-6);
    }

    sess.reset();
    let sum = api::recorder::recordadd(
        &mut sess,
        operand::input(a.getview()),
        operand::input(b.getview()),
    )
    .unwrap();
    let got = sess.apply(&[sum]).unwrap();
    let want = exec.add(&a.getview(), &b.getview()).unwrap();
    assertallclose(got[0].getdata(), want.getdata(), 1.0e-6, 1.0e-6);
}

#[test]
fn session_reset_discards_a_chain_recorded_before_apply() {
    if !hasdevice("session_reset_discards_a_chain_recorded_before_apply") {
        return;
    }
    let exec = syclexecutor::new(wide());
    let a = build(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = build(vec![2, 2], vec![0.5, -0.5, 1.5, -1.5]);

    let mut sess = exec.newsession().unwrap();
    // Record, then discard without applying: reset must drain the in-order
    // queue before the step buffers are released (shared USM has no implicit
    // synchronisation on free). The session must remain usable afterwards.
    api::recorder::recordmul(
        &mut sess,
        operand::input(a.getview()),
        operand::input(b.getview()),
    )
    .unwrap();
    sess.reset();

    let sum = api::recorder::recordadd(
        &mut sess,
        operand::input(a.getview()),
        operand::input(b.getview()),
    )
    .unwrap();
    let got = sess.apply(&[sum]).unwrap();
    let want = exec.add(&a.getview(), &b.getview()).unwrap();
    assertallclose(got[0].getdata(), want.getdata(), 1.0e-6, 1.0e-6);
}

#[test]
fn submission_rejects_an_unrecorded_result_reference() {
    if !hasdevice("submission_rejects_an_unrecorded_result_reference") {
        return;
    }
    let exec = syclexecutor::new(wide());
    let mut sub = exec.newsubmission().unwrap();
    let a = build(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let bogus = stepref::fromindex(99);

    let err = sub
        .recordadd(operand::result(bogus), operand::input(a.getview()))
        .unwrap_err();

    assert_eq!(err, backenderror::indexoutofrange { index: 99, len: 0 });
}
