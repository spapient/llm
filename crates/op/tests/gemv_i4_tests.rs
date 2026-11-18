use super::*;
use crate::gemv::packed_test_support::{at, with};

crate::gemv::packed_test_support::packed_gemv_tests!(I4, MODEL, GROUP, forward, 2, 32);

fn packed(values: &[i8]) -> Vec<u8> {
    values
        .chunks(2)
        .map(|pair| {
            let low = (pair[0] + 8) as u8 & 0x0F;
            let high = (pair.get(1).copied().unwrap_or(0) + 8) as u8 & 0x0F;
            low | (high << 4)
        })
        .collect()
}

#[test]
fn reference_dequantizes_and_dots() {
    // 1 行、1 组、4 个权重：[1, 2, 3, 4]，scale = 0.5，x = [1, 1, 1, 1]
    let w = packed(&[1, 2, 3, 4]);
    let sample = with(4, 1, vec![1.0; 4], w, vec![0.5]);
    let y = gemv::reference::<I4>(&sample).expect("参考可算");
    assert_eq!(y.len(), 1);
    assert!((y[0] - 5.0).abs() < 1e-6, "0.5*(1+2+3+4)=5，实际 {}", y[0]);
}

#[test]
fn reference_handles_multiple_groups_and_rows() {
    let m = 8;
    let n = 2;
    let w = packed(&[1, 2, 3, 4, 5, 6, 7, 8, 1, 1, 1, 1, 2, 2, 2, 2]);
    let x: Vec<f32> = (0..m).map(|index| index as f32 + 1.0).collect();
    let sample = with(m, n, x, w, vec![0.25, 0.5, 1.0, 1.0]);
    let y = gemv::reference::<I4>(&sample).expect("参考可算");
    assert_eq!(y.len(), n);
    assert!(y[0] > 0.0);
    assert!(y[1] > y[0], "第二行权重更大");
}

#[test]
fn forward_writes_into_the_caller_buffer_and_matches_the_reference() {
    let m = 8;
    let n = 2;
    let w = packed(&[1, 2, 3, 4, 5, 6, 7, 8, 1, 1, 1, 1, 2, 2, 2, 2]);
    let x: Vec<f32> = (0..m).map(|index| index as f32 + 1.0).collect();
    let scales = vec![0.25, 0.5, 1.0, 1.0];
    let expected =
        gemv::reference::<I4>(&with(m, n, x.clone(), w.clone(), scales.clone())).expect("参考可算");

    // 复用一块脏缓冲：结果必须是覆盖写，读到的不能是上一次的残留。
    let mut out = vec![f32::NAN; n];
    forward(&x, &w, &scales, &mut out).expect("内核可算");
    assert_eq!(out, expected, "内核与参考必须是同一段算术");
}

/// 手写一遍求和，不经过本文件的任何循环：分组边界、`scale` 落在哪、行偏移都由测试自己算。
///
/// 这是替换内层循环（查表 + 按组切片 + 每组一次乘法）之后仍然钉住算术的判据——
/// `reference` 与 `forward` 是同一段代码，只对它俩对拍等于自己跟自己对。
#[test]
fn the_kernel_agrees_with_a_hand_written_sum() {
    let m = 8;
    let n = 2;
    // 打包只支持 [-8, 7]：`value + 8` 必须落在 4 位里，所以 8 会溢出成 -8。
    let w = packed(&[1, 2, 3, 4, 5, 6, 7, -8, -1, 0, 1, 2, 3, -4, 5, -6]);
    let x: Vec<f32> = (0..m).map(|index| index as f32 * 0.5 - 1.0).collect();
    // 行 0：组 0 = 0.25、组 1 = 0.5；行 1：组 0 = 1.0、组 1 = -1.0。
    let scales = vec![0.25, 0.5, 1.0, -1.0];

    let mut out = vec![f32::NAN; n];
    forward(&x, &w, &scales, &mut out).expect("内核可算");

    let row = |weights: [f32; 8], scale: [f32; 2]| {
        scale[0] * (0..4).map(|k| weights[k] * x[k]).sum::<f32>()
            + scale[1] * (0..4).map(|k| weights[4 + k] * x[4 + k]).sum::<f32>()
    };
    let expected = [
        row([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, -8.0], [0.25, 0.5]),
        row([-1.0, 0.0, 1.0, 2.0, 3.0, -4.0, 5.0, -6.0], [1.0, -1.0]),
    ];
    assert!((out[0] - expected[0]).abs() < 1e-5, "{:?}", out);
    assert!((out[1] - expected[1]).abs() < 1e-5, "{:?}", out);
}

#[test]
fn shape_mismatch_is_an_error() {
    // 输入太短。
    assert!(gemv::reference::<I4>(&with(4, 1, vec![1.0], vec![0u8; 2], vec![1.0])).is_err());
    // 权重太短。
    assert!(gemv::reference::<I4>(&with(4, 1, vec![1.0; 4], vec![0u8; 1], vec![1.0])).is_err());
    // 没有分组。
    assert!(gemv::reference::<I4>(&with(4, 1, vec![1.0; 4], vec![0u8; 2], vec![])).is_err());
    // 分组数为奇数：m=3、1 组 → 组大小 3。
    assert!(gemv::reference::<I4>(&with(3, 1, vec![1.0; 3], vec![0u8; 2], vec![1.0])).is_err());
    // m 为奇数、分组为偶数但不整除：先被"不能整除"拦下，解包循环从不碰到奇数尾。
    assert!(
        gemv::reference::<I4>(&with(5, 1, vec![1.0; 5], vec![0u8; 3], vec![1.0, 1.0])).is_err()
    );
    // 奇数宽度无论分组奇偶都进不到解包：奇组被偶校验拦，偶组被整除校验拦。
    assert!(gemv::reference::<I4>(&with(7, 1, vec![1.0; 7], vec![0u8; 4], vec![1.0])).is_err());
    assert!(
        gemv::reference::<I4>(&with(7, 1, vec![1.0; 7], vec![0u8; 4], vec![1.0, 1.0])).is_err()
    );
}

/// 标度负例：非有限标度在分组守卫后、解包前拦下，不允许把整组乘积带进 NaN/Inf。
#[test]
fn gemv_i4_rejects_non_finite_scales() {
    let m = 4;
    let w = vec![0u8; m / 2];
    let x = vec![1.0; m];
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let error = forward(&x, &w, &[bad], &mut [0.0; 1]).expect_err("非有限标度必须拒绝");
        assert_eq!(error.code(), diagnostic::Code::Invalid);
    }
}

#[test]
fn cost_comes_from_the_declared_expression_without_hand_written_constants() {
    let cost = at::<I4>(4096, 4096).expect("可估");
    assert_eq!(cost.flops, 2.0 * 4096.0 * 4096.0);
    // 三个输入都是读：int4 权重、输入、缩放因子；只有输出是写。
    assert_eq!(
        cost.read,
        4096 * 4096 / 2 + 4096 * 4 + 4096 * 128 * 4,
        "权重 8 MiB + 输入 16 KiB + 缩放 2 MiB"
    );
    assert_eq!(cost.write, 4096 * 4, "写方向只有输出那一份");
    assert_eq!(
        cost.bytes(),
        4096 * 4096 / 2 + 4096 * 4 + 4096 * 128 * 4 + 4096 * 4,
        "合计与改造前逐字节相同"
    );
}

#[test]
fn a_small_shape_pins_every_term_so_a_factor_cannot_hide() {
    // 大形状里 `m/32` 之类的因子很容易互相抵消，小形状才能逐项钉住：
    // w = 4·64 个 int4（128 B）、x = 64·4、scales = 4·2·4、y = 4·4。
    let cost = at::<I4>(64, 4).expect("可估");
    assert_eq!(cost.flops, 2.0 * 64.0 * 4.0);
    assert_eq!(cost.read, 128 + 256 + 32);
    assert_eq!(cost.write, 16);
    assert_eq!(cost.bytes(), 128 + 256 + 32 + 16);
}

#[test]
fn call_arguments_follow_the_template_signature() {
    let signature = crate::sig::Signature::parse("gemv_i4", include_str!("../ops/gemv_i4.k"))
        .expect("签名可解析");
    assert_eq!(
        signature
            .call_args(&gemv::sample::<I4>(64, 4))
            .expect("实参"),
        "x, w, scales, y, 64, 4, 32"
    );
    assert_eq!(signature.output(), Some("y"));
}
