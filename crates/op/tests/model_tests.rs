use diagnostic::{Code, Error};
use llm_spec::model::{AxisRole, AxisSpec, Cost, DType, Model, Sample, ShapeCtx, Values, bind};

const fn zero_cost(_: &ShapeCtx) -> Result<Cost, Error> {
    Ok(Cost::ZERO)
}

fn sample(m: usize, n: usize) -> Sample {
    Sample {
        m,
        n,
        scalars: Vec::new(),
        bufs: Vec::new(),
    }
}

fn reference(_: &Sample) -> diagnostic::Result<Vec<f32>> {
    Ok(Vec::new())
}

fn test_shape(m: usize, n: usize) -> Result<ShapeCtx, Error> {
    ShapeCtx::new([('k', m), ('j', n)])
}

/// 只声明输入轴：它不是一个可投影的算子，`projection` 必须说出来，而不是猜一对数字。
const HALF_PROJECTION: Model = Model {
    group: 1,
    shape: (4, 4),
    dtype: DType::F32,
    axes: &[AxisSpec {
        role: AxisRole::Input,
        label: 'k',
    }],
    shape_ctx: test_shape,
    cost: zero_cost,
    sample,
    reference,
};

#[test]
fn a_projection_ctx_fills_the_axes_by_role_not_by_position() {
    let model = Model {
        axes: &[
            AxisSpec {
                role: AxisRole::Input,
                label: 'k',
            },
            AxisSpec {
                role: AxisRole::Output,
                label: 'j',
            },
        ],
        ..HALF_PROJECTION
    };
    let shape = model.projection(64, 4).expect("两个轴都声明了");
    assert_eq!(shape.get('k'), Some(64), "输入宽度填的是 input 轴");
    assert_eq!(shape.get('j'), Some(4), "输出宽度填的是 output 轴");
    assert_eq!(shape.get('m'), None, "没有把 `m` 当成约定");
}

#[test]
fn a_model_without_both_projection_axes_is_not_a_projection() {
    let error = HALF_PROJECTION.projection(4, 4).expect_err("缺输出轴");
    assert_eq!(error.code(), Code::Unsupported);
    assert!(error.to_string().contains("输出轴"), "{error}");
}

#[test]
fn a_derived_axis_is_declared_by_the_operator() {
    let shape = ShapeCtx::new([('m', 64), ('n', 4)]).expect("合法");
    let extents = shape
        .extents_with_group('g', 'm', 32)
        .expect("64 能被 32 整除");
    assert_eq!(
        extents.get('g'),
        Some(2),
        "调用方没有给 `g`，算子自己切出来"
    );
    assert!(shape.extents_with_group('g', 'm', 32).is_ok());
    let error = shape
        .extents_with_group('g', 'm', 48)
        .expect_err("不是整数倍");
    assert!(error.to_string().contains("整数倍"), "{error}");
}

#[test]
fn a_ctx_rejects_an_empty_axis_instead_of_pricing_zero() {
    let error = ShapeCtx::new([('m', 0)]).expect_err("空轴");
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("长度为 0"), "{error}");
}

#[test]
fn bind_puts_real_numbers_where_the_reference_reads_them() {
    let model = crate::model("gemv_f32").expect("稠密语义");
    let x: Vec<f32> = (0..8).map(|index| index as f32).collect();
    let w: Vec<f32> = (0..16).map(|index| index as f32 * 0.5).collect();
    let sample = bind(&model, 8, 2, &x, &w).expect("形状合法");

    assert_eq!(sample.f32("x").expect("输入"), x.as_slice());
    assert_eq!(sample.f32("w").expect("权重"), w.as_slice());
    // 输出缓冲与标量仍来自算子样本：真实数据只替换 x / w，不顺手改 ABI。
    assert_eq!(sample.f32("y").expect("输出"), [0.0f32, 0.0]);
    assert_eq!(sample.scalar("M"), Some(8));
    assert_eq!(sample.scalar("N"), Some(2));
    // 参考实现读的就是绑定进来的数，所以"绑定"与"对拍"不会各看一份数据。
    let y = (model.reference)(&sample).expect("参考");
    let want: f32 = x.iter().zip(&w[..8]).map(|(a, b)| a * b).sum();
    assert_eq!(y[0], want);
}

#[test]
fn bind_refuses_packed_weights_instead_of_reading_them_as_f32() {
    let model = crate::model("gemv_i4").expect("量化语义");
    let error =
        bind(&model, 64, 2, &[0.0; 64], &[0.0; 128]).expect_err("量化算子的权重缓冲是打包字节");
    assert_eq!(error.code(), Code::Unsupported);
    assert!(error.to_string().contains("打包字节"), "{error}");
}

#[test]
fn set_refuses_a_pointer_type_change_even_when_the_length_matches() {
    let model = crate::model("gemv_f32").expect("稠密语义");
    let mut sample = (model.sample)(8, 2);
    let want = sample.f32("x").expect("输入").len();
    // 长度对得上、类型对不上：旧口径会放它过去，下游 reference 便按错误变体读同一块数据。
    let error = sample
        .set("x", Values::U8(vec![0; want]))
        .expect_err("float 槽位不能换成 unsigned char");
    assert_eq!(error.code(), Code::Invalid);
    let text = error.to_string();
    assert!(text.contains("`x`"), "{text}");
    assert!(text.contains("float"), "{text}");
    assert!(text.contains("unsigned char"), "{text}");
    // 换回同类型才算换成：类型检查不是"一律拒绝"。
    sample
        .set("x", Values::F32(vec![1.0; want]))
        .expect("同类型换数据");
    assert_eq!(sample.f32("x").expect("输入"), [1.0f32; 8]);
    // 长度仍单独判：类型对了但个数不对，还是 Invalid。
    assert_eq!(
        sample
            .set("x", Values::F32(vec![0.0; want + 1]))
            .expect_err("长度不符")
            .code(),
        Code::Invalid
    );
}

#[test]
fn bind_checks_lengths_and_the_shape_granularity() {
    let dense = crate::model("gemv_f32").expect("稠密语义");
    assert_eq!(
        bind(&dense, 8, 2, &[0.0; 7], &[0.0; 16])
            .expect_err("输入长度不符")
            .code(),
        Code::Invalid
    );
    assert_eq!(
        bind(&dense, 8, 2, &[0.0; 8], &[0.0; 15])
            .expect_err("权重长度不符")
            .code(),
        Code::Invalid
    );
    // 量化算子的粒度是 32：形状不自洽时先被挡在这里，而不是进内核后读到半个分组。
    let quantized = crate::model("gemv_i4").expect("量化语义");
    let error = bind(&quantized, 8, 2, &[0.0; 8], &[0.0; 16]).expect_err("粒度不符");
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("形状粒度"), "{error}");
}
