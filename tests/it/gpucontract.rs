//! Device-free contract vectors shared by the backend suites.
//!
//! Every case here settles on the host before the backend opens a device or
//! allocates an output, so one list of vectors can be run against the CPU, the
//! Metal, and the CUDA executor on any machine — with or without a GPU — and the
//! three observed outcomes must be identical. That is the mechanical form of the
//! cross-backend requirement in `docs/design.md` §21.1: the same operand must
//! produce the same shape, index, and error classification on every device.
//!
//! Keeping the vectors in one place is deliberate. A backend-local copy of
//! these cases would let one backend's error classification drift while its own
//! suite still passed.

#![allow(dead_code, clippy::unwrap_used, clippy::expect_used)]

use crate::support::build;
use api::{backenderror, compute};
use lib::limits;

/// One observed outcome: `None` means the call returned successfully.
pub type observation = (&'static str, Option<backenderror>);

/// Budget bounds used where a case must not be rejected by an unrelated bound.
const WIDE_ELEMENTS: usize = 1 << 24;
const WIDE_BYTES: usize = 1 << 28;
const WIDE_SEQ: usize = 1 << 20;
const WIDE_BATCH: usize = 1 << 16;

/// Returns the limits a permissive case uses, with `elements` and `bytes` varied.
fn budget(elements: usize, bytes: usize) -> limits {
    limits::new(elements, bytes, WIDE_SEQ, WIDE_BATCH)
}

/// Runs `case` against an executor built with `lim` and reports the outcome.
fn observe<E: compute, F>(make: &impl Fn(limits) -> E, lim: limits, case: F) -> Option<backenderror>
where
    F: FnOnce(&E) -> Result<(), backenderror>,
{
    let exec = make(lim);
    case(&exec).err()
}

/// Runs every vector against `make` and returns the observed outcome per label.
///
/// A case that must succeed asserts its own documented degenerate result inside
/// the closure, because "accepted" alone would not pin the value the contract
/// promises for a zero contraction (the additive identity, not a launch).
pub fn run<E: compute>(make: &impl Fn(limits) -> E) -> Vec<observation> {
    let wide = || budget(WIDE_ELEMENTS, WIDE_BYTES);
    let mut observed: Vec<observation> = Vec::new();

    // Rank above the declared maximum is a declared-bound rejection.
    let toprank = build(vec![1; 9], vec![1.0]);
    observed.push((
        "silu/rank",
        observe(make, wide(), |e| e.silu(&toprank.getview()).map(|_| ())),
    ));

    // Element and byte budgets, each one element/byte past its inclusive bound.
    let five = build(vec![5], vec![1.0; 5]);
    observed.push((
        "silu/elements",
        observe(make, budget(4, WIDE_BYTES), |e| {
            e.silu(&five.getview()).map(|_| ())
        }),
    ));
    let two = build(vec![2], vec![1.0; 2]);
    observed.push((
        "silu/bytes",
        observe(make, budget(WIDE_ELEMENTS, 4), |e| {
            e.silu(&two.getview()).map(|_| ())
        }),
    ));

    observed.extend(matmulvectors(make));
    observed.extend(elementwisevectors(make));
    observed.extend(normalizationvectors(make));
    observed.extend(positionvectors(make));
    observed.extend(gathervectors(make));
    observed.extend(attentionvectors(make));
    observed
}

/// `matmul` shape, budget, and zero-contraction vectors.
fn matmulvectors<E: compute>(make: &impl Fn(limits) -> E) -> Vec<observation> {
    let wide = budget(WIDE_ELEMENTS, WIDE_BYTES);
    let mut observed = Vec::new();

    let flat = build(vec![3], vec![0.0; 3]);
    observed.push((
        "matmul/rank",
        observe(make, wide, |e| {
            e.matmul(&flat.getview(), &flat.getview()).map(|_| ())
        }),
    ));
    let mismatched = build(vec![2, 3], vec![0.0; 6]);
    let other = build(vec![4, 2], vec![0.0; 8]);
    observed.push((
        "matmul/contraction",
        observe(make, wide, |e| {
            e.matmul(&mismatched.getview(), &other.getview())
                .map(|_| ())
        }),
    ));
    let batched = build(vec![2, 2, 3], vec![0.0; 12]);
    let plain = build(vec![3, 3], vec![0.0; 9]);
    observed.push((
        "matmul/batchdims",
        observe(make, wide, |e| {
            e.matmul(&batched.getview(), &plain.getview()).map(|_| ())
        }),
    ));

    // Operands fit; the result does not. `[8, 1] * [1, 9]` is 8 and 9 elements
    // in, 72 elements (288 bytes) out, so the result is the only bound exceeded.
    let column = build(vec![8, 1], vec![1.0; 8]);
    let row = build(vec![1, 9], vec![1.0; 9]);
    observed.push((
        "matmul/outelements",
        observe(make, budget(64, WIDE_BYTES), |e| {
            e.matmul(&column.getview(), &row.getview()).map(|_| ())
        }),
    ));
    observed.push((
        "matmul/outbytes",
        observe(make, budget(WIDE_ELEMENTS, 64), |e| {
            e.matmul(&column.getview(), &row.getview()).map(|_| ())
        }),
    ));

    // A zero contraction is answered on the host, so the result budget can be
    // probed at its exact inclusive bound without touching a device.
    let emptyrows = build(vec![8, 0], vec![]);
    let emptycols = build(vec![0, 8], vec![]);
    observed.push((
        "matmul/k0/elements-at-bound",
        observe(make, budget(64, WIDE_BYTES), |e| {
            let out = e.matmul(&emptyrows.getview(), &emptycols.getview())?;
            assert_eq!(out.getshape().getdims(), &[8, 8]);
            assert!(out.getdata().iter().all(|&x| x == 0.0));
            Ok(())
        }),
    ));
    observed.push((
        "matmul/k0/bytes-at-bound",
        observe(make, budget(WIDE_ELEMENTS, 256), |e| {
            let out = e.matmul(&emptyrows.getview(), &emptycols.getview())?;
            assert_eq!(out.getshape().getdims(), &[8, 8]);
            Ok(())
        }),
    ));
    observed.push((
        "matmul/k0/elements-over",
        observe(make, budget(63, WIDE_BYTES), |e| {
            e.matmul(&emptyrows.getview(), &emptycols.getview())
                .map(|_| ())
        }),
    ));
    observed.push((
        "matmul/k0/bytes-over",
        observe(make, budget(WIDE_ELEMENTS, 255), |e| {
            e.matmul(&emptyrows.getview(), &emptycols.getview())
                .map(|_| ())
        }),
    ));
    observed
}

/// `add` and `mul` vectors.
fn elementwisevectors<E: compute>(make: &impl Fn(limits) -> E) -> Vec<observation> {
    let wide = budget(WIDE_ELEMENTS, WIDE_BYTES);
    let mut observed = Vec::new();
    let lhs = build(vec![2, 3], vec![0.0; 6]);
    let rhs = build(vec![3, 2], vec![0.0; 6]);
    observed.push((
        "add/shape",
        observe(make, wide, |e| {
            e.add(&lhs.getview(), &rhs.getview()).map(|_| ())
        }),
    ));
    observed.push((
        "mul/shape",
        observe(make, wide, |e| {
            e.mul(&lhs.getview(), &rhs.getview()).map(|_| ())
        }),
    ));
    observed.push((
        "mul/elements",
        observe(make, budget(4, WIDE_BYTES), |e| {
            e.mul(&lhs.getview(), &rhs.getview()).map(|_| ())
        }),
    ));
    observed
}

/// `rmsnorm` and `softmax` vectors.
fn normalizationvectors<E: compute>(make: &impl Fn(limits) -> E) -> Vec<observation> {
    let wide = budget(WIDE_ELEMENTS, WIDE_BYTES);
    let mut observed = Vec::new();
    let emptyaxis = build(vec![2, 0], vec![]);
    let emptyweight = build(vec![0], vec![]);
    observed.push((
        "rmsnorm/emptylastaxis",
        observe(make, wide, |e| {
            e.rmsnorm(&emptyaxis.getview(), &emptyweight.getview(), 1.0e-5)
                .map(|_| ())
        }),
    ));
    let row = build(vec![2, 3], vec![0.0; 6]);
    let shortweight = build(vec![2], vec![1.0; 2]);
    observed.push((
        "rmsnorm/weightlength",
        observe(make, wide, |e| {
            e.rmsnorm(&row.getview(), &shortweight.getview(), 1.0e-5)
                .map(|_| ())
        }),
    ));
    observed.push((
        "softmax/emptylastaxis",
        observe(make, wide, |e| e.softmax(&emptyaxis.getview()).map(|_| ())),
    ));
    observed
}

/// `rope` vectors.
fn positionvectors<E: compute>(make: &impl Fn(limits) -> E) -> Vec<observation> {
    let wide = budget(WIDE_ELEMENTS, WIDE_BYTES);
    let mut observed = Vec::new();
    let even = build(vec![2, 4], vec![0.0; 8]);
    observed.push((
        "rope/heads0",
        observe(make, wide, |e| {
            e.rope(&even.getview(), 0, 10_000.0).map(|_| ())
        }),
    ));
    let odd = build(vec![2, 3], vec![0.0; 6]);
    observed.push((
        "rope/oddhead",
        observe(make, wide, |e| {
            e.rope(&odd.getview(), 1, 10_000.0).map(|_| ())
        }),
    ));
    let ragged = build(vec![2, 6], vec![0.0; 12]);
    observed.push((
        "rope/raggedheads",
        observe(make, wide, |e| {
            e.rope(&ragged.getview(), 4, 10_000.0).map(|_| ())
        }),
    ));
    let flat = build(vec![4], vec![0.0; 4]);
    observed.push((
        "rope/rank1",
        observe(make, wide, |e| {
            e.rope(&flat.getview(), 1, 10_000.0).map(|_| ())
        }),
    ));
    observed
}

/// `embedding` vectors.
fn gathervectors<E: compute>(make: &impl Fn(limits) -> E) -> Vec<observation> {
    let wide = budget(WIDE_ELEMENTS, WIDE_BYTES);
    let mut observed = Vec::new();
    let table = build(vec![3, 2], vec![0.0; 6]);
    observed.push((
        "embedding/index",
        observe(make, wide, |e| {
            e.embedding(&table.getview(), &[3]).map(|_| ())
        }),
    ));

    // The gather output is 8 rows of width 2, past a budget of 8 elements,
    // while the table itself holds only 4 elements.
    let small = build(vec![2, 2], vec![1.0; 4]);
    observed.push((
        "embedding/gatherbudget",
        observe(make, budget(8, WIDE_BYTES), |e| {
            e.embedding(&small.getview(), &[0; 8]).map(|_| ())
        }),
    ));
    observed
}

/// `attention` vectors.
fn attentionvectors<E: compute>(make: &impl Fn(limits) -> E) -> Vec<observation> {
    let wide = budget(WIDE_ELEMENTS, WIDE_BYTES);
    let mut observed = Vec::new();
    let query = build(vec![2, 4], vec![0.0; 8]);
    let key = build(vec![2, 2], vec![0.0; 4]);
    let value = build(vec![2, 2], vec![0.0; 4]);

    let flat = build(vec![4], vec![0.0; 4]);
    observed.push((
        "attention/rank1",
        observe(make, wide, |e| {
            e.attention(
                &flat.getview(),
                &key.getview(),
                &value.getview(),
                1,
                1,
                true,
            )
            .map(|_| ())
        }),
    ));
    let shorter = build(vec![3, 4], vec![0.0; 12]);
    observed.push((
        "attention/seq",
        observe(make, wide, |e| {
            e.attention(
                &shorter.getview(),
                &key.getview(),
                &value.getview(),
                1,
                1,
                true,
            )
            .map(|_| ())
        }),
    ));
    let widekey = build(vec![2, 3], vec![0.0; 6]);
    observed.push((
        "attention/kvshape",
        observe(make, wide, |e| {
            e.attention(
                &query.getview(),
                &widekey.getview(),
                &value.getview(),
                1,
                1,
                true,
            )
            .map(|_| ())
        }),
    ));
    observed.push((
        "attention/headsplit",
        observe(make, wide, |e| {
            e.attention(
                &query.getview(),
                &key.getview(),
                &value.getview(),
                3,
                1,
                true,
            )
            .map(|_| ())
        }),
    ));
    observed.push((
        "attention/kvheads0",
        observe(make, wide, |e| {
            e.attention(
                &query.getview(),
                &key.getview(),
                &value.getview(),
                2,
                0,
                true,
            )
            .map(|_| ())
        }),
    ));
    observed.push((
        "attention/kvwidth",
        observe(make, wide, |e| {
            e.attention(
                &query.getview(),
                &key.getview(),
                &value.getview(),
                2,
                2,
                true,
            )
            .map(|_| ())
        }),
    ));
    observed
}

/// Returns the outcome every backend MUST produce for [`run`]'s case order.
pub fn expected() -> Vec<observation> {
    use backenderror::{indexoutofrange, resourceexhausted, shapemismatch, unsupportedoperation};
    let shape = |op: &'static str| Some(shapemismatch { op });
    vec![
        ("silu/rank", Some(resourceexhausted)),
        ("silu/elements", Some(resourceexhausted)),
        ("silu/bytes", Some(resourceexhausted)),
        ("matmul/rank", shape("matmul")),
        ("matmul/contraction", shape("matmul")),
        ("matmul/batchdims", shape("matmul")),
        ("matmul/outelements", Some(resourceexhausted)),
        ("matmul/outbytes", Some(resourceexhausted)),
        ("matmul/k0/elements-at-bound", None),
        ("matmul/k0/bytes-at-bound", None),
        ("matmul/k0/elements-over", Some(resourceexhausted)),
        ("matmul/k0/bytes-over", Some(resourceexhausted)),
        ("add/shape", shape("add")),
        ("mul/shape", shape("mul")),
        ("mul/elements", Some(resourceexhausted)),
        (
            "rmsnorm/emptylastaxis",
            Some(unsupportedoperation { op: "rmsnorm" }),
        ),
        ("rmsnorm/weightlength", shape("rmsnorm")),
        (
            "softmax/emptylastaxis",
            Some(unsupportedoperation { op: "softmax" }),
        ),
        ("rope/heads0", shape("rope")),
        ("rope/oddhead", shape("rope")),
        ("rope/raggedheads", shape("rope")),
        ("rope/rank1", shape("rope")),
        (
            "embedding/index",
            Some(indexoutofrange { index: 3, len: 3 }),
        ),
        ("embedding/gatherbudget", Some(resourceexhausted)),
        ("attention/rank1", shape("attention")),
        ("attention/seq", shape("attention")),
        ("attention/kvshape", shape("attention")),
        ("attention/headsplit", shape("attention")),
        ("attention/kvheads0", shape("attention")),
        ("attention/kvwidth", shape("attention")),
    ]
}

/// Asserts that `make` reproduces [`expected`] exactly, naming `backend` in the
/// failure so a diverging backend is identified without reading the diff.
pub fn assertconforms<E: compute>(backend: &str, make: impl Fn(limits) -> E) {
    let got = run(&make);
    let want = expected();
    assert_eq!(
        got.len(),
        want.len(),
        "{backend}: vector count changed without updating `expected`"
    );
    for ((label, observed), (expectedlabel, expected)) in got.iter().zip(want.iter()) {
        assert_eq!(
            label, expectedlabel,
            "{backend}: vector order changed without updating `expected`"
        );
        assert_eq!(
            observed, expected,
            "{backend}: case `{label}` returned {observed:?}, the contract requires {expected:?}"
        );
    }
}
