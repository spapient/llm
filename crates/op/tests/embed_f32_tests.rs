use super::*;

#[test]
fn embed_matches_the_row_reference() {
    let sample = sample(6, 4);
    let expected = reference(&sample).expect("参考可计算");
    let mut out = vec![0.0; 6 * 4];
    forward(
        sample.f32("idx").expect("idx"),
        sample.f32("table").expect("table"),
        &mut out,
    )
    .expect("几何自洽");
    assert_eq!(out, expected);
}

#[test]
fn embed_rejects_an_out_of_range_index() {
    let sample = sample(4, 3);
    let table = sample.f32("table").expect("table");
    let mut out = vec![0.0; 4 * 3];
    let idx = vec![100.0; 4];
    assert!(
        forward(&idx, table, &mut out).is_err(),
        "下标越界词表应被拒绝"
    );
}

#[test]
fn embed_rejects_a_table_wrong_length() {
    let sample = sample(4, 3);
    let idx = sample.f32("idx").expect("idx");
    let mut out = vec![0.0; 4 * 3];
    let table = vec![0.0; 1];
    assert!(
        forward(idx, &table, &mut out).is_err(),
        "表长不符行宽应被拒绝"
    );
}
