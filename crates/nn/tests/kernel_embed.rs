    use super::*;

    #[test]
    fn embed_matches_the_row_reference() {
        let sample = sample(6, 4);
        let expected = reference(&sample).expect("参考可计算");
        let mut out = vec![0.0; 6 * 4];
        forward(
            tensor::View::one(sample.f32("idx").expect("idx")),
            tensor::View::one(sample.f32("table").expect("table")),
            &mut out,
        )
        .expect("几何自洽");
        assert_eq!(out, expected);
    }

    #[test]
    fn embed_rejects_an_out_of_range_idx() {
        let sample = sample(4, 3);
        let table = sample.f32("table").expect("table");
        let mut out = vec![0.0; 4 * 3];
        let idx = vec![100.0; 4];
        assert!(
            forward(tensor::View::one(idx.as_slice()), tensor::View::one(table), &mut out).is_err(),
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
            forward(tensor::View::one(idx), tensor::View::one(table.as_slice()), &mut out).is_err(),
            "表长不符行宽应被拒绝"
        );
    }

    /// 退化与重复边界：空索引拒绝；行重复取值时逐行 gather 与参考一致。
    #[test]
    fn embed_handles_empty_and_repeated_idxs() {
        let sample = sample(4, 3);
        let table = sample.f32("table").expect("table");
        let mut out = vec![];
        assert!(forward(tensor::View::one(&[]), tensor::View::one(table), &mut out).is_err(), "空索引必须拒绝");
        let repeated = vec![0.0, 0.0, 1.0, 1.0];
        let mut out = vec![0.0; repeated.len() * 3];
        let mut repeat_sample = crate::embed_f32::sample(repeated.len(), 3);
        repeat_sample.bufs[0].values = crate::model::Values::F32(repeated.clone());
        let expected = reference(&repeat_sample).expect("参考可计算");
        forward(tensor::View::one(repeated.as_slice()), tensor::View::one(table), &mut out).expect("重复索引不越界");
        assert_eq!(out, expected);
    }
