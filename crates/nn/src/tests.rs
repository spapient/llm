use super::*;

/// 只声明了输出、没有输入宽度：层合法，但没有可计价的形状。
#[crate::spec(name = "no_axes", out = [8], ops = ["gemv_i4"])]
struct NoAxes;

#[test]
fn an_unknown_operator_is_rejected_at_definition_time() {
    let error = ContractSpec::new(
        "bad",
        Shape::new([4]),
        vec!["gemv_i4".to_string(), "flash_attention".to_string()],
    )
    .expect_err("未声明的算子");
    assert_eq!(error.code(), Code::Missing);
    assert!(error.to_string().contains("flash_attention"), "{error}");
}

#[test]
fn a_linear_layer_points_at_a_declared_operator() {
    let layer = Linear::new("blk.0.attn_q", 8, 4).expect("合法");
    assert_eq!(layer.name(), "blk.0.attn_q");
    assert_eq!(layer.output().dims(), [4]);
    assert_eq!(layer.spca().ops(), ["gemv_i4"]);
    assert_eq!(layer.out_features(), 4);
    assert_eq!(layer.format(), WeightFormat::INT4);
}

#[test]
fn a_zero_width_layer_is_rejected() {
    assert_eq!(Linear::new("x", 0, 4).unwrap_err().code(), Code::Invalid);
    assert_eq!(Linear::new("x", 4, 0).unwrap_err().code(), Code::Invalid);
}

#[test]
fn a_linear_layer_declares_its_floor() {
    let layer = Linear::new("blk.0.attn_q", 4096, 4096).expect("合法");
    let floor = layer.bounds().expect("生产形状").expect("有算子");
    // 层自己不算代价：它必须与算子语义逐个数字一致，否则就是第二份公式。
    assert_eq!(
        floor,
        Bounds::of_op(Linear::OP, 4096, 4096).expect("同一份语义")
    );
    assert!(
        floor.read() > 8 * 1024 * 1024,
        "光 int4 权重就 8 MiB：{}",
        floor.read()
    );
}

#[test]
fn a_layer_can_exist_without_a_legal_kernel_shape() {
    // 宽度非零，所以层能构造；但 8 不是量化分组的整数倍，它没有合法内核，也就没有下界。
    let layer = Linear::new("odd", 8, 4).expect("宽度非零即可构造");
    assert_eq!(
        layer.bounds().unwrap_err().code(),
        Code::Invalid,
        "形状非法仍然是错误，不是「无法计价」"
    );
}

#[test]
fn a_format_without_an_operator_is_declared_but_unpriced() {
    // 同一份权重，文件里写的是 f16：层可以声明，但还没有 16 位元素的算子给它计价。
    let layer = Linear::with_format("blk.0.attn_q", WeightFormat::F16, 4096, 4096)
        .expect("宽度非零即可构造");
    assert!(layer.spca().ops().is_empty(), "没有算子可声明");
    let error = layer.bounds().unwrap_err();
    assert_eq!(error.code(), Code::Unsupported);
    assert!(error.to_string().contains("f16"), "{error}");
}

/// f32 权重现在有算子，而且报出的是它自己的代价（64 MiB 读），不是 int4 的 10.5 MB。
#[test]
fn a_dense_weight_is_priced_by_the_dense_operator() {
    let layer = Linear::with_format("blk.0.attn_q", WeightFormat::F32, 4096, 4096)
        .expect("宽度非零即可构造");
    assert_eq!(layer.spca().ops(), [Linear::DENSE]);
    let floor = layer.bounds().expect("合法").expect("有算子");
    assert_eq!(
        floor,
        Bounds::of_op(Linear::DENSE, 4096, 4096).expect("同一份语义")
    );
    // 读侧 = 权重（4096² 个 f32）+ 输入向量（4096 个 f32）：下界按方向分账，输入不是免费的。
    assert_eq!(
        floor.read(),
        64 * 1024 * 1024 + 4096 * 4,
        "f32 权重 + 一次输入向量的读下界"
    );
}

#[test]
fn a_float_weight_is_not_priced_as_int4() {
    // 这就是改造前的隐性错误：同一个 [4096, 4096] 条目，f32 的存储本身是 64 MiB，
    // 而 int4 的读下界 ≈ 10.5 MB —— 差 6.4 倍，且旧代码会把前者报成后者。
    let int4 = Bounds::of_op(Linear::OP, 4096, 4096).expect("合法");
    let float_bytes = Shape::new([4096, 4096]).element_bytes(tensor::DType::F32);
    assert_eq!(float_bytes, 64 * 1024 * 1024);
    let ratio = float_bytes as f64 / int4.read() as f64;
    assert!((6.3..6.5).contains(&ratio), "f32 / int4 读侧 = {ratio}");
    // 两条算子的差不能被静默抹平：稠密那条报出自己的代价，量化那条报自己的。
    let dense = Linear::with_format("w", WeightFormat::F32, 4096, 4096).expect("可声明");
    let dense = dense.bounds().expect("合法").expect("有算子");
    assert_eq!(
        dense.read(),
        float_bytes as u64 + 4096 * 4,
        "f32 读下界 = 权重字节数 + 输入向量"
    );
    assert!(dense.read() > int4.read());
}

#[test]
fn an_annotated_layer_is_a_checked_declaration() {
    let layer = LmHead;
    assert_eq!(layer.name(), "lm_head");
    assert_eq!(layer.output().dims(), [4096]);
    assert_eq!(layer.spca().ops(), ["gemv_i4"]);
}

#[test]
fn an_annotated_layer_prices_its_floor_from_the_same_semantics() {
    let floor = LmHead.bounds().expect("形状合法").expect("有算子");
    assert_eq!(
        floor,
        Bounds::of_op(Linear::OP, 4096, 4096).expect("同一份语义"),
        "注解层的下界与 Linear 同源"
    );
    // 与手写的同宽度投影逐项相等：注解只是另一种声明方式，不是另一条公式。
    let linear = Linear::new(LmHead.name(), 4096, 4096).expect("合法");
    assert_eq!(floor, linear.bounds().expect("合法").expect("有算子"));
}

#[test]
fn an_annotated_layer_without_an_input_width_has_no_floor() {
    // 没有 `in`：层合法，但规约轴宽度未知，算子语义表无法被调用。
    assert_eq!(NoAxes.bounds().expect("不是错误"), None);
}

#[test]
fn the_annotated_implementation_is_shared_by_every_instance() {
    let first = LmHead.spca();
    let second = LmHead.spca();
    assert!(core::ptr::eq(first, second), "规格只构造一次（OnceLock）");
}

#[test]
fn a_spec_works_as_a_trait_object() {
    let layer = Linear::new("blk.0.attn_k", 32, 6).expect("合法");
    let dynamic: &dyn Contract = &layer;
    assert_eq!(dynamic.spca().name(), "blk.0.attn_k");
    assert!(dynamic.bounds().expect("合法").is_some());
}

#[test]
fn a_spec_without_an_explicit_role_defaults_to_trainable() {
    let spec = ContractSpec::new("legacy", Shape::new([4]), vec!["gemv_i4".to_string()])
        .expect("算子已声明");
    assert_eq!(spec.role(), ParamRole::Trainable);
}

#[test]
fn an_explicit_role_is_round_trip_declared() {
    let spec = ContractSpec::new_with_role(
        "blk.0.norm",
        Shape::new([128]),
        vec!["rmsnorm_f32".to_string()],
        ParamRole::Norm,
    )
    .expect("算子已声明");
    assert_eq!(spec.role(), ParamRole::Norm);
    assert!(spec.role().wd_exempt());
}

#[test]
fn a_projection_layer_declares_the_trainable_role() {
    let layer = Linear::new("blk.0.attn_q", 8, 4).expect("合法");
    assert_eq!(layer.spca().role(), ParamRole::Trainable);
    assert!(!layer.spca().role().wd_exempt());
}
