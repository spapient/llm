use super::*;
use crate::model::{Buffer, Cost, Sample, ShapeContext, Values};
use diagnostic::Error;
use prof::atom::DType;

/// 把 `-2..=1` 的整数码按四个一组打包成字节：低 2 位在前。
fn packed(values: &[i8]) -> Vec<u8> {
    values
        .chunks(4)
        .map(|quad| {
            let code = |i: usize| (quad.get(i).copied().unwrap_or(0) + 2) as u8 & 0x03;
            code(0) | (code(1) << 2) | (code(2) << 4) | (code(3) << 6)
        })
        .collect()
}

fn with(m: usize, n: usize, x: Vec<f32>, w: Vec<u8>, scales: Vec<f32>) -> Sample {
    Sample {
        m,
        n,
        scalars: Vec::new(),
        buffers: vec![
            Buffer {
                name: "x",
                values: Values::F32(x),
            },
            Buffer {
                name: "w",
                values: Values::U8(w),
            },
            Buffer {
                name: "scales",
                values: Values::F32(scales),
            },
        ],
    }
}

/// 测试里只关心 `(m, n)`：它们走的是层那条路径——把宽度填进算子声明的具名轴。
fn at(m: usize, n: usize) -> Result<Cost, Error> {
    gemv::cost::<I2>(&ShapeContext::new([('m', m), ('n', n)])?)
}

#[test]
fn reference_dequantizes_and_dots() {
    // 1 行、1 组、4 个权重：[1, -1, 0, 1]，scale = 0.5，x = [1, 1, 1, 1]
    let w = packed(&[1, -1, 0, 1]);
    let sample = with(4, 1, vec![1.0; 4], w, vec![0.5]);
    let y = gemv::reference::<I2>(&sample).expect("参考可算");
    assert_eq!(y.len(), 1);
    assert!(
        (y[0] - 0.5).abs() < 1e-6,
        "0.5*(1-1+0+1)=0.5，实际 {}",
        y[0]
    );
}

#[test]
fn reference_handles_multiple_groups_and_rows() {
    let m = 8;
    let n = 2;
    let w = packed(&[1, -1, 0, 1, 1, -1, 0, 1, -2, -1, 0, 1, -2, -1, 0, 1]);
    let x: Vec<f32> = (0..m).map(|index| index as f32 + 1.0).collect();
    let sample = with(m, n, x, w, vec![0.25, 0.5, 1.0, -1.0]);
    let y = gemv::reference::<I2>(&sample).expect("参考可算");
    assert_eq!(y.len(), n);
    assert!(y[0].is_finite());
    assert!(y[1].is_finite());
}

#[test]
fn forward_writes_into_the_caller_buffer_and_matches_the_reference() {
    let m = 8;
    let n = 2;
    let w = packed(&[1, -1, 0, 1, 1, -1, 0, 1, -2, -1, 0, 1, -2, -1, 0, 1]);
    let x: Vec<f32> = (0..m).map(|index| index as f32 + 1.0).collect();
    let scales = vec![0.25, 0.5, 1.0, -1.0];
    let expected =
        gemv::reference::<I2>(&with(m, n, x.clone(), w.clone(), scales.clone())).expect("参考可算");

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
    let w = packed(&[1, -1, 0, 1, 1, -1, 0, 1, -2, -1, 0, 1, -2, -1, 0, 1]);
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
fn forward_rejects_a_buffer_that_does_not_fit_the_shapes() {
    let error = forward(&[1.0; 4], &[], &[1.0], &mut [0.0; 1]).expect_err("权重太短");
    assert_eq!(error.code(), diagnostic::Code::Invalid);
    assert!(
        forward(&[1.0; 4], &[0u8; 1], &[1.0], &mut []).is_err(),
        "空输出"
    );
}

#[test]
fn shape_mismatch_is_an_error() {
    // 输入太短。
    assert!(gemv::reference::<I2>(&with(4, 1, vec![1.0], vec![0u8; 1], vec![1.0])).is_err());
    // 权重太短。
    assert!(gemv::reference::<I2>(&with(4, 1, vec![1.0; 4], vec![0u8; 0], vec![1.0])).is_err());
    // 没有分组。
    assert!(gemv::reference::<I2>(&with(4, 1, vec![1.0; 4], vec![0u8; 1], vec![])).is_err());
    // 分组不是 4 的倍数：m=6、1 组 → 组大小 6。
    assert!(gemv::reference::<I2>(&with(6, 1, vec![1.0; 6], vec![0u8; 2], vec![1.0])).is_err());
    // m 不是 4 的倍数、分组为 4 倍数但不整除：先被"不能整除"拦下，解包循环从不碰到残缺尾码。
    assert!(
        gemv::reference::<I2>(&with(5, 1, vec![1.0; 5], vec![0u8; 2], vec![1.0, 1.0])).is_err()
    );
    // 非 4 倍数宽度无论分组奇偶都进不到解包：非四倍数组被四倍数校验拦，其余被整除校验拦。
    assert!(gemv::reference::<I2>(&with(7, 1, vec![1.0; 7], vec![0u8; 2], vec![1.0])).is_err());
    assert!(
        gemv::reference::<I2>(&with(7, 1, vec![1.0; 7], vec![0u8; 2], vec![1.0, 1.0])).is_err()
    );
}

/// 标度负例：非有限标度在分组守卫后、解包前拦下，不允许把整组乘积带进 NaN/Inf。
#[test]
fn gemv_i2_rejects_non_finite_scales() {
    let m = 4;
    let w = vec![0u8; m / 4];
    let x = vec![1.0; m];
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let error = forward(&x, &w, &[bad], &mut [0.0; 1]).expect_err("非有限标度必须拒绝");
        assert_eq!(error.code(), diagnostic::Code::Invalid);
    }
}

#[test]
fn the_production_shape_is_a_legal_input() {
    let (m, n) = MODEL.shape;
    assert!(m.is_multiple_of(GROUP), "生产形状的 M 必须是组大小的整数倍");
    assert!(at(m, n).is_ok(), "生产形状必须能算出代价");
    assert_eq!(MODEL.dtype, DType::F32, "样本的输出缓冲是 f32");
    assert_eq!(MODEL.group as usize, GROUP, "注册的粒度就是注解的粒度");
}

#[test]
fn cost_comes_from_the_declared_expression_without_hand_written_constants() {
    let cost = at(4096, 4096).expect("可估");
    assert_eq!(cost.flops, 2.0 * 4096.0 * 4096.0);
    // 三个输入都是读：int2 权重、输入、缩放因子；只有输出是写。
    assert_eq!(
        cost.read,
        4096 * 4096 / 4 + 4096 * 4 + 4096 * 128 * 4,
        "权重 4 MiB + 输入 16 KiB + 缩放 2 MiB"
    );
    assert_eq!(cost.write, 4096 * 4, "写方向只有输出那一份");
    assert_eq!(
        cost.bytes(),
        4096 * 4096 / 4 + 4096 * 4 + 4096 * 128 * 4 + 4096 * 4,
        "合计与改造前逐字节相同"
    );
}

#[test]
fn a_small_shape_pins_every_term_so_a_factor_cannot_hide() {
    // 大形状里 `m/32` 之类的因子很容易互相抵消，小形状才能逐项钉住：
    // w = 4·64 个 int2（64 B）、x = 64·4、scales = 4·2·4、y = 4·4。
    let cost = at(64, 4).expect("可估");
    assert_eq!(cost.flops, 2.0 * 64.0 * 4.0);
    assert_eq!(cost.read, 64 + 256 + 32);
    assert_eq!(cost.write, 16);
    assert_eq!(cost.bytes(), 64 + 256 + 32 + 16);
}

#[test]
fn a_shape_that_does_not_fit_the_grouping_is_an_error() {
    let error = at(33, 4).expect_err("33 不是 32 的整数倍");
    assert!(error.to_string().contains("整数倍"), "{error}");
    assert_eq!(error.code(), diagnostic::Code::Invalid);
}

#[test]
fn sample_is_deterministic_and_sized_to_the_group() {
    let first = gemv::sample::<I2>(64, 4);
    assert_eq!(first, gemv::sample::<I2>(64, 4));
    assert_eq!(first.u8("w").expect("w").len(), 4 * 16);
    assert_eq!(first.f32("scales").expect("scales").len(), 4 * (64 / GROUP));
    assert_eq!(first.f32("y").expect("y").len(), 4);
}

#[test]
fn call_arguments_follow_the_template_signature() {
    let signature = crate::sig::Signature::parse("gemv_i2", include_str!("../ops/gemv_i2.k"))
        .expect("签名可解析");
    assert_eq!(
        signature
            .call_args(&gemv::sample::<I2>(64, 4))
            .expect("实参"),
        "x, w, scales, y, 64, 4, 32"
    );
    assert_eq!(signature.output(), Some("y"));
}
