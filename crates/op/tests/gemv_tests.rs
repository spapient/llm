use super::*;
use crate::gemv_f32::F32;
use crate::gemv_i4::I4;

fn at<F: Format>(m: usize, n: usize) -> Result<Cost, Error> {
    cost::<F>(&ShapeContext::new([('m', m), ('n', n)])?)
}

/// 两条格式的表达式必须各自与操作数配对，且派生轴只出现在量化格式里。
#[test]
fn the_derived_axis_belongs_to_the_quantized_format_only() {
    let quantized = ShapeContext::new([('m', 64), ('n', 4)])
        .expect("轴")
        .extents_with_group('g', 'm', 32)
        .expect("可分组");
    let axes = <I4 as Format>::formula()
        .expect("配对")
        .axes(&quantized)
        .expect("可分解");
    let reduced: Vec<char> = axes.reduced().map(|axis| axis.label()).collect();
    let grouped: Vec<char> = axes.grouped().map(|axis| axis.label()).collect();
    assert_eq!(reduced, ['m'], "只有输入维参与求和");
    assert_eq!(grouped, ['g'], "量化分组是派生轴");
    assert_eq!(axes.output_shape(), [4]);

    let dense = ShapeContext::new([('m', 64), ('n', 4)]).expect("轴");
    let axes = <F32 as Format>::formula()
        .expect("配对")
        .axes(dense.extents())
        .expect("可分解");
    assert_eq!(axes.grouped().count(), 0, "稠密格式没有派生轴");
}

/// 同一形状下两种格式的流量差 8 倍以上：f32 权重是 int4 的 8 倍，标度只属于量化格式。
#[test]
fn the_two_formats_price_the_same_shape_differently() {
    let dense = at::<F32>(64, 4).expect("可估");
    let quantized = at::<I4>(64, 4).expect("可估");
    assert_eq!(dense.read, (64 * 4 + 64) * 4, "w 与 x 都是 f32");
    assert_eq!(quantized.read, 4 * 32 + 64 * 4 + 4 * 2 * 4);
    assert!(dense.read > quantized.read, "稠密权重的读下界更高");
}

/// 样本的缓冲集合由格式决定：量化格式有 `scales`，稠密格式没有，`gsz` 取各自的步长。
#[test]
fn the_sample_follows_the_format_instead_of_a_fixed_buffer_list() {
    let dense = sample::<F32>(64, 4);
    assert!(dense.buffer("scales").is_none(), "稠密格式没有标度");
    assert_eq!(dense.scalar("gsz"), Some(32));

    let quantized = sample::<I4>(64, 4);
    assert_eq!(quantized.f32("scales").expect("scales").len(), 4 * 2);
    assert_eq!(quantized.scalar("gsz"), Some(32));
    assert_eq!(quantized, sample::<I4>(64, 4), "样本必须可复现");
}

/// 权重元素类型不符时报的是"这份格式期望什么"，而不是笼统的"不是 u8"。
#[test]
fn a_weight_buffer_of_the_wrong_element_type_names_the_expected_one() {
    let mut dense = sample::<F32>(8, 2);
    dense
        .set("w", Values::U8(vec![0; 16]))
        .expect("长度相同即可换");
    let error = reference::<F32>(&dense).expect_err("u8 不是 f32 格式的权重");
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("f32"), "{error}");
}

/// 测试用的小粒度实例：`Quantized<4>` 与生产实例共用同一份 [`Packed`] 实现，只换声明粒度 const。
#[test]
fn a_smaller_grain_is_the_same_implementation() {
    let sample = sample::<Quantized<4>>(16, 2);
    assert_eq!(sample.f32("scales").expect("scales").len(), 2 * 4);
    assert_eq!(
        at::<Quantized<4>>(16, 2).expect("可估").flops,
        2.0 * 16.0 * 2.0
    );
    assert!(at::<Quantized<8>>(16, 2).is_ok());
    assert!(
        at::<Quantized<32>>(16, 2).is_err(),
        "16 不是 32 的整数倍：声明粒度决定形状合法性"
    );
}

/// 派生轴的除数只能是形状粒度：写歪一个数时，代价与样本算的是两种分组。
#[test]
fn the_derived_divisor_must_be_the_grain() {
    assert!(derived_matches_grain(32, None), "没有派生轴时不受约束");
    assert!(derived_matches_grain(32, Some(('g', 'm', 32))));
    assert!(
        !derived_matches_grain(32, Some(('g', 'm', 8))),
        "8 元素一组的代价配 32 元素一组的标度，是两份语义"
    );

    // 两条生产格式的自证真的被求值过：这两行编译得过，说明 `CHECK` 对它们成立。
    let _: () = <I4 as Format>::CHECK;
    let _: () = <F32 as Format>::CHECK;
}

/// 骨架的检查对两种格式都成立：输出为空与输入过短由同一段代码拒绝。
#[test]
fn the_shared_checks_apply_to_every_format() {
    for error in [
        reference::<F32>(&Sample {
            m: 4,
            n: 0,
            scalars: Vec::new(),
            buffers: vec![],
        })
        .expect_err("输出维度为 0"),
        reference::<I4>(&Sample {
            m: 4,
            n: 1,
            scalars: Vec::new(),
            buffers: vec![],
        })
        .expect_err("缺缓冲"),
    ] {
        assert!(
            matches!(error.code(), Code::Invalid | Code::Missing),
            "{error}"
        );
    }
    assert!(<F32 as Format>::EXPR.contains('m'), "常量在类型上可取");
    assert_eq!(<Quantized<8> as Format>::GRAIN, 8);
}
