    use super::*;

    #[test]
    fn rmsnorm_matches_the_scalar_reference() {
        let sample = sample(128, 1);
        let expected = reference(&sample).expect("参考可计算");
        let mut out = vec![0.0; 128];
        forward(
            sample.f32("x").expect("x"),
            sample.f32("gamma").expect("gamma"),
            &mut out,
        )
        .expect("等长");
        assert_eq!(out, expected);
    }

    #[test]
    fn rmsnorm_rejects_empty_input() {
        let mut out = vec![];
        assert!(
            forward(&[], &[], &mut out).is_err(),
            "空行没有可统计的均方根"
        );
        assert!(reference(&sample(0, 1)).is_err(), "参考同样拒绝空行");
    }

    /// 三缓冲非对齐长度各自拒绝：x/gamma 不等、x/out 不等都不能静默截位。
    #[test]
    fn rmsnorm_rejects_each_mismatched_buffer_length() {
        let mut out = vec![0.0; 3];
        assert!(
            forward(&[1.0, 2.0], &[1.0], &mut out).is_err(),
            "x 与 gamma 不等"
        );
        assert!(
            forward(&[1.0, 2.0], &[1.0, 2.0], &mut out).is_err(),
            "x 与 out 不等"
        );
        assert!(reference(&sample(0, 1)).is_err(), "参考路径同样拒绝空行");
    }

    /// 零范数（全零行）：分母由 `EPS` 托底，输出为零而不是除零；权重按原样乘回。
    #[test]
    fn rmsnorm_zero_row_is_guided_by_epsilon() {
        let x = vec![0.0; 4];
        let gamma = vec![0.5, 0.5, 0.5, 0.5];
        let mut out = vec![0.0; 4];
        forward(&x, &gamma, &mut out).expect("EPS 托底后全零行可归一化");
        assert!(out.iter().all(|v| v.abs() < 1e-6), "全零行输出趋近零");
    }

    /// 饱和与重复边界：大值、极值、重复元素行，结果有限且与参考一致。
    #[test]
    fn rmsnorm_saturating_and_repeated_inputs_match_reference() {
        let cases: &[&[f32]] = &[
            &[1.0, 1.0, 1.0],
            &[-1.0, 2.0, -3.0],
            &[f32::MAX; 3],
            &[f32::MIN; 3],
            &[0.25, -0.5, 0.75, -1.0],
        ];
        for (index, x) in cases.iter().enumerate() {
            let gamma: Vec<f32> = x.iter().map(|_| 0.75).collect();
            let sample = Sample {
                m: x.len(),
                n: x.len(),
                scalars: vec![],
                buffers: vec![
                    Buffer {
                        name: "x",
                        values: crate::model::Values::F32(x.to_vec()),
                    },
                    Buffer {
                        name: "gamma",
                        values: crate::model::Values::F32(gamma.clone()),
                    },
                    Buffer {
                        name: "y",
                        values: crate::model::Values::F32(vec![0.0; x.len()]),
                    },
                ],
            };
            let expected = reference(&sample).expect("参考可计算");
            let mut out = vec![0.0; x.len()];
            forward(x, &gamma, &mut out).expect("等长非空");
            assert_eq!(out, expected, "用例 {index}");
            assert!(out.iter().all(|v| v.is_finite()), "用例 {index}：结果有限");
        }
    }

    #[test]
    fn batch_normalizes_each_segment_independently() {
        let row_dim = 3;
        let batch = 2;
        let x: Vec<f32> = (0..batch * row_dim).map(|i| i as f32 * 0.5 - 2.0).collect();
        let gamma: Vec<f32> = (0..row_dim).map(|i| 0.6 + i as f32 * 0.1).collect();

        let mut out = vec![0.0; x.len()];
        forward_batch(&x, &gamma, &mut out, row_dim).expect("批量归一化");

        // 参考：每段独立走单行入口，gamma 逐段复用，段间不混均值。
        let mut expected = vec![0.0; x.len()];
        for (b, seg) in x.chunks(row_dim).enumerate() {
            let span = b * row_dim..(b + 1) * row_dim;
            forward(seg, &gamma, &mut expected[span]).expect("段内参考");
        }
        assert_eq!(out, expected);
    }

    #[test]
    fn batch_one_entry_matches_the_single_row_kernel() {
        let x = vec![1.0, -2.0, 0.5, 3.0];
        let gamma = vec![0.7, 0.6, 0.8, 0.5];
        let mut single = vec![0.0; x.len()];
        forward(&x, &gamma, &mut single).expect("单行");
        let mut batch = vec![0.0; x.len()];
        forward_batch(&x, &gamma, &mut batch, x.len()).expect("单段经批量入口");
        assert_eq!(single, batch, "batch=1 必须与单行入口逐元素等价");
    }

    #[test]
    fn batch_rejects_bad_row_dimension_and_length() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![0.5; 2];
        let mut out = vec![0.0; 4];
        // 行宽为 0。
        assert!(forward_batch(&x, &gamma, &mut out, 0).is_err());
        // gamma 长度与行宽不符。
        let wrong_gamma = vec![0.5; 3];
        assert!(forward_batch(&x, &wrong_gamma, &mut out, 2).is_err());
        // x 与 out 不等长。
        let mut short = vec![0.0; 3];
        assert!(forward_batch(&x, &gamma, &mut short, 2).is_err());
        // 缓冲不是行宽整数倍。
        assert!(forward_batch(&x[..3], &gamma, &mut short, 2).is_err());
    }
