//! Incremental-decode attention: the query may be shorter than the key/value
//! operands. The single-token query attends to every cached key under the
//! causal mask, and must reproduce the last row of the full-sequence attention.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use crate::support::{build, wide};
use api::compute;
use cpu::cpuexecutor;

#[test]
fn a_shorter_query_reproduces_the_last_full_attention_row() {
    let exec = cpuexecutor::new(wide());
    let heads = 2;
    let kvheads = 1;
    let headdim = 4;

    // One new query token at absolute position 2; three cached key/value rows.
    let q = build(
        vec![1, heads * headdim],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    );
    let k = build(
        vec![3, kvheads * headdim],
        vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
    );
    let v = build(
        vec![3, kvheads * headdim],
        vec![
            2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
        ],
    );

    let incremental = exec
        .attention(
            &q.getview(),
            &k.getview(),
            &v.getview(),
            heads,
            kvheads,
            true,
        )
        .unwrap();

    // Full-sequence equivalent: the same query as the last row, arbitrary
    // (zero) rows before it, over the same keys and values.
    let mut fullq = vec![0.0_f32; 3 * heads * headdim];
    fullq[2 * heads * headdim..].copy_from_slice(q.getdata());
    let full = build(vec![3, heads * headdim], fullq);
    let full_out = exec
        .attention(
            &full.getview(),
            &k.getview(),
            &v.getview(),
            heads,
            kvheads,
            true,
        )
        .unwrap();

    let last = &full_out.getdata()[full_out.getdata().len() - heads * headdim..];
    assert_eq!(incremental.getshape().getdims(), &[1, heads * headdim]);
    assert_eq!(incremental.getdata(), last);
}

#[test]
fn a_longer_query_than_the_keys_is_rejected() {
    let exec = cpuexecutor::new(wide());
    let heads = 1;
    let kvheads = 1;
    let headdim = 2;

    let q = build(vec![3, heads * headdim], vec![1.0; 6]);
    let k = build(vec![2, kvheads * headdim], vec![1.0; 4]);
    let v = build(vec![2, kvheads * headdim], vec![1.0; 4]);

    let err = exec
        .attention(
            &q.getview(),
            &k.getview(),
            &v.getview(),
            heads,
            kvheads,
            true,
        )
        .unwrap_err();
    assert_eq!(err, api::backenderror::shapemismatch { op: "attention" });
}
