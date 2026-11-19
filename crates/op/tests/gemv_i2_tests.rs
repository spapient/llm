use super::*;
use crate::gemv::packed_test_support::{at, pack, with};

crate::gemv::packed_test_support::packed_gemv_tests!(I2, MODEL, GROUP, forward, 1, 16);
crate::gemv::packed_test_support::packed_gemv_config_tests!(
    I2,
    "gemv_i2",
    2,
    "权重 4 MiB + 输入 16 KiB + 缩放 2 MiB",
    [1, -1, 0, 1],
    0.5
);

#[test]
fn reference_handles_multiple_groups_and_rows() {
    let m = 8;
    let n = 2;
    let w = pack(2, &[1, -1, 0, 1, 1, -1, 0, 1, -2, -1, 0, 1, -2, -1, 0, 1]);
    let x: Vec<f32> = (0..m).map(|index| index as f32 + 1.0).collect();
    let sample = with(m, n, x, w, vec![0.25, 0.5, 1.0, -1.0]);
    let y = crate::gemv::reference::<I2>(&sample).expect("参考可算");
    assert_eq!(y.len(), n);
    assert!(y[0].is_finite());
    assert!(y[1].is_finite());
}

#[test]
fn forward_writes_into_the_caller_buffer_and_matches_the_reference() {
    let m = 8;
    let n = 2;
    let w = pack(2, &[1, -1, 0, 1, 1, -1, 0, 1, -2, -1, 0, 1, -2, -1, 0, 1]);
    let x: Vec<f32> = (0..m).map(|index| index as f32 + 1.0).collect();
    let scales = vec![0.25, 0.5, 1.0, -1.0];
    let expected =
        crate::gemv::reference::<I2>(&with(m, n, x.clone(), w.clone(), scales.clone())).expect("参考可算");

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
    // 打包只支持 [-2, 1]：码值由 value + 2 落到 2 位里。
    let w = pack(2, &[1, -1, 0, 1, 1, -1, 0, 1, -2, -1, 0, 1, -2, -1, 0, 1]);
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
        row([1.0, -1.0, 0.0, 1.0, 1.0, -1.0, 0.0, 1.0], [0.25, 0.5]),
        row([-2.0, -1.0, 0.0, 1.0, -2.0, -1.0, 0.0, 1.0], [1.0, -1.0]),
    ];
    assert!((out[0] - expected[0]).abs() < 1e-5, "{:?}", out);
    assert!((out[1] - expected[1]).abs() < 1e-5, "{:?}", out);
}

#[test]
fn shape_mismatch_is_an_error() {
    // 输入太短。
    assert!(crate::gemv::reference::<I2>(&with(4, 1, vec![1.0], vec![0u8; 1], vec![1.0])).is_err());
    // 权重太短。
    assert!(crate::gemv::reference::<I2>(&with(4, 1, vec![1.0; 4], vec![0u8; 0], vec![1.0])).is_err());
    // 没有分组。
    assert!(crate::gemv::reference::<I2>(&with(4, 1, vec![1.0; 4], vec![0u8; 1], vec![])).is_err());
    // 分组不是 4 的倍数：m=6、1 组 → 组大小 6。
    assert!(crate::gemv::reference::<I2>(&with(6, 1, vec![1.0; 6], vec![0u8; 2], vec![1.0])).is_err());
    // m 不是 4 的倍数、分组为 4 倍数但不整除：先被"不能整除"拦下，解包循环从不碰到残缺尾码。
    assert!(
        crate::gemv::reference::<I2>(&with(5, 1, vec![1.0; 5], vec![0u8; 2], vec![1.0, 1.0])).is_err()
    );
    // 非 4 倍数宽度无论分组奇偶都进不到解包：非四倍数组被四倍数校验拦，其余被整除校验拦。
    assert!(crate::gemv::reference::<I2>(&with(7, 1, vec![1.0; 7], vec![0u8; 2], vec![1.0])).is_err());
    assert!(
        crate::gemv::reference::<I2>(&with(7, 1, vec![1.0; 7], vec![0u8; 2], vec![1.0, 1.0])).is_err()
    );
}
