//! Relocated unit tests for the `bench` crate.
//!
//! Formerly an inline `#[cfg(test)] mod tests` block, moved here so that
//! `crates/bench/src` holds no test code. The tests reach only the crate's
//! public workload constructors and constants; the crate-private `VOCAB` and
//! `OPLIMITS` are intentionally not exported (A4 interface minimality), so the
//! embedding row count is asserted with the fixed literal the workload uses
//! (see `docs/engineering.md`).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use ::bench::{
    ELEMENTWISE, GATHER, HEADDIM, HEADS, MATMULDIM, NORMCOLS, NORMROWS, SEQLEN, SOFTMAXCOLS,
    SOFTMAXROWS, attentioninput, cpuexec, elementwiseinput, embeddinginput, matmulbatchinput,
    matmulinput, norminput, ropeinput, softmaxinput,
};

/// Row count of the fixed embedding workload (`bench` keeps this crate-private).
const EMBEDDINGVOCAB: usize = 4096;

/// Block count of the fixed batched matmul workload (`bench` keeps it
/// crate-private).
const MATMULBATCH: usize = 8;

#[test]
fn elementwise_input_has_fixed_shape() {
    let t = elementwiseinput().unwrap();
    assert_eq!(t.getshape().getdims(), &[ELEMENTWISE]);
    assert_eq!(t.getlen(), ELEMENTWISE);
}

#[test]
fn softmax_input_has_fixed_shape() {
    let t = softmaxinput().unwrap();
    assert_eq!(t.getshape().getdims(), &[SOFTMAXROWS, SOFTMAXCOLS]);
    assert_eq!(t.getlen(), SOFTMAXROWS * SOFTMAXCOLS);
}

#[test]
fn operator_inputs_have_fixed_shapes() {
    let (a, b) = matmulinput().unwrap();
    assert_eq!(a.getshape().getdims(), &[MATMULDIM, MATMULDIM]);
    assert_eq!(b.getshape().getdims(), &[MATMULDIM, MATMULDIM]);

    let (ba, bb) = matmulbatchinput().unwrap();
    assert_eq!(
        ba.getshape().getdims(),
        &[MATMULBATCH, MATMULDIM, MATMULDIM]
    );
    assert_eq!(
        bb.getshape().getdims(),
        &[MATMULBATCH, MATMULDIM, MATMULDIM]
    );

    let (x, w) = norminput().unwrap();
    assert_eq!(x.getshape().getdims(), &[NORMROWS, NORMCOLS]);
    assert_eq!(w.getshape().getdims(), &[NORMCOLS]);

    let r = ropeinput().unwrap();
    assert_eq!(r.getshape().getdims(), &[SEQLEN, HEADS * HEADDIM]);

    let (table, indices) = embeddinginput().unwrap();
    assert_eq!(
        table.getshape().getdims(),
        &[EMBEDDINGVOCAB, ::bench::EMBEDDIM]
    );
    assert_eq!(indices.len(), GATHER);

    let (q, k, v) = attentioninput().unwrap();
    assert_eq!(q.getshape(), k.getshape());
    assert_eq!(q.getshape(), v.getshape());
}

#[test]
fn executor_admits_the_workloads() {
    let exec = cpuexec();
    let view = elementwiseinput().unwrap();
    assert!(exec.apply(::cpu::cpuop::relu, &view.getview()).is_ok());
}
