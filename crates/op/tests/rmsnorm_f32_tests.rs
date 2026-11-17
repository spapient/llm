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
