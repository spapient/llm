use super::*;

/// 一整条写回链：六个步骤各归其位，参数被正确装箱。
#[test]
fn a_full_chain_parses_into_six_steps() {
    let steps = parse("bias; act:gelu; residual; scale:0.5; quant:i8; store").expect("合法链");
    assert_eq!(
        steps,
        [
            Step::Bias,
            Step::Act(ActKind::Gelu),
            Step::Residual,
            Step::Scale(0.5),
            Step::Quant(DType::I8),
            Step::Store,
        ],
    );
}

/// `parse(render(s)) == s`：渲染不丢任何参数，是可逆的规范形。
#[test]
fn render_is_the_parse_inverse() {
    let cases: &[&[Step]] = &[
        &[],
        &[Step::Store],
        &[Step::Bias, Step::Act(ActKind::Silu), Step::Residual],
        &[Step::Scale(2.5), Step::Quant(DType::I4), Step::Act(ActKind::LeakyRelu)],
        &[
            Step::Bias,
            Step::Act(ActKind::Gelu),
            Step::Residual,
            Step::Scale(0.5),
            Step::Quant(DType::I2),
            Step::Store,
        ],
    ];
    for steps in cases {
        assert_eq!(parse(&render(steps)).expect("回环可解析"), steps.to_vec());
    }
}

/// 渲染的规范形固定：分号分隔、无参步骤不拼冒号。
#[test]
fn render_is_canonical() {
    let steps = [Step::Bias, Step::Act(ActKind::Relu), Step::Scale(1.5), Step::Store];
    assert_eq!(render(&steps), "bias; act:relu; scale:1.5; store");
}

/// 闭集之外的步骤名必须被拒，错误指向行号。
#[test]
fn an_unknown_step_is_rejected_with_its_line() {
    let err = parse("bias; norm; store").expect_err("`norm` 不在闭集");
    assert_eq!(err.code(), Code::Invalid);
    assert!(err.to_string().contains("未知写回步骤 `norm`"), "{err}");
    assert!(err.to_string().contains("第 1 行"), "{err}");
}

/// 未知激活被拒。
#[test]
fn an_unknown_activation_is_rejected() {
    let err = parse("act:swish").expect_err("`swish` 不在激活闭集");
    assert_eq!(err.code(), Code::Invalid);
    assert!(err.to_string().contains("未知激活 `swish`"), "{err}");
}

/// `quant` 只收 `i8 / i4 / i2`，浮点与更宽整数都被拒。
#[test]
fn quant_accepts_only_i8_i4_i2() {
    assert!(parse("quant:i8").is_ok());
    assert!(parse("quant:i4").is_ok());
    assert!(parse("quant:i2").is_ok());
    for bad in ["quant:f32", "quant:i16", "quant:bf16"] {
        let err = parse(bad).expect_err(bad);
        assert_eq!(err.code(), Code::Invalid);
    }
}

/// `scale` 只收有限十进制数。
#[test]
fn scale_accepts_only_finite_decimals() {
    assert_eq!(parse("scale:0.25").expect("有限"), [Step::Scale(0.25)]);
    for bad in ["scale:nan", "scale:inf", "scale:-inf", "scale:abc"] {
        let err = parse(bad).expect_err(bad);
        assert_eq!(err.code(), Code::Invalid);
    }
}

/// 无参步骤带参数即错；有参步骤缺参数即错。
#[test]
fn arguments_must_match_the_step() {
    assert!(parse("bias:1").is_err(), "bias 无参");
    assert!(parse("residual:x").is_err(), "residual 无参");
    assert!(parse("store:y").is_err(), "store 无参");
    assert!(parse("act").is_err(), "act 缺参");
    assert!(parse("scale").is_err(), "scale 缺参");
    assert!(parse("quant").is_err(), "quant 缺参");
}

/// 分号与逗号都可作分隔；空段被跳过；行号随换行前进。
#[test]
fn separators_whitespace_and_lines() {
    let steps = parse("  bias ,  act:relu  ;\n  scale:2  , store ").expect("混合分隔");
    assert_eq!(
        steps,
        [
            Step::Bias,
            Step::Act(ActKind::Relu),
            Step::Scale(2.0),
            Step::Store,
        ],
    );
    assert!(parse("").expect("空输入").is_empty());
    assert!(parse(";;;,").expect("全空段").is_empty());
    let err = parse("bias;\n\nnorm").expect_err("第三行的坏步骤");
    assert!(err.to_string().contains("第 3 行"), "{err}");
}

/// `apply`：bias + residual + scale 依次施加，与手算逐位一致。
#[test]
fn apply_bias_residual_scale_matches_hand() {
    let steps = parse("bias; residual; scale:0.5").expect("合法");
    let mut tile = [1.0, 2.0, 3.0, 4.0];
    let ctx = Ctx {
        n: 2,
        bias: Some(&[10.0, 20.0]),
        residual: Some(&[1.0, 1.0, 1.0, 1.0]),
    };
    apply(&steps, &mut tile, ctx).expect("形状匹配");
    // 行主序、两列：(v + bias[col] + residual) * 0.5
    assert_eq!(tile, [6.0, 11.5, 7.0, 12.5]);
}

/// `apply`：quant 的往返是四舍五入 + 饱和。
#[test]
fn apply_quant_rounds_and_saturates() {
    let steps = [Step::Quant(DType::I8)];
    let mut tile = [1.5, -130.0, 200.0, -0.5];
    apply(&steps, &mut tile, Ctx { n: 4, bias: None, residual: None }).expect("无缓冲");
    // round-half-away-from-zero：1.5→2、-0.5→-1；两侧饱和到 i8 边界。
    assert_eq!(tile, [2.0, -128.0, 127.0, -1.0]);
}

/// `apply`：五种激活的锚点（逐点精确的锚 + 超越函数锚带容差）。
#[test]
fn apply_activations_match_anchors() {
    let relu = [Step::Act(ActKind::Relu)];
    let mut tile = [-2.0, 0.0, 3.0];
    apply(&relu, &mut tile, Ctx { n: 3, bias: None, residual: None }).expect("relu");
    assert_eq!(tile, [0.0, 0.0, 3.0]);

    let leaky = [Step::Act(ActKind::LeakyRelu)];
    let mut tile = [-2.0, 0.0, 2.0];
    apply(&leaky, &mut tile, Ctx { n: 3, bias: None, residual: None }).expect("leaky");
    assert_eq!(tile, [-0.02, 0.0, 2.0]);

    let hard = [Step::Act(ActKind::HardSigmoid)];
    let mut tile = [-6.0, 0.0, 6.0];
    apply(&hard, &mut tile, Ctx { n: 3, bias: None, residual: None }).expect("hard");
    assert_eq!(tile, [0.0, 0.5, 1.0]);

    let silu = [Step::Act(ActKind::Silu)];
    let mut tile = [1.0];
    apply(&silu, &mut tile, Ctx { n: 1, bias: None, residual: None }).expect("silu");
    assert!((tile[0] - 0.731_058_6).abs() < 1e-6, "silu(1) = {}", tile[0]);

    let gelu = [Step::Act(ActKind::Gelu)];
    let mut tile = [0.0, 1.0];
    apply(&gelu, &mut tile, Ctx { n: 2, bias: None, residual: None }).expect("gelu");
    assert_eq!(tile[0], 0.0, "gelu(0) = 0");
    assert!((tile[1] - 0.841_344_7).abs() < 1e-5, "gelu(1) = {}", tile[1]);
}

/// `apply`：形状不符必须显式失败，而不是静默越界或截断。
#[test]
fn apply_rejects_shape_mismatch() {
    let steps = [Step::Bias];
    let mut tile = [0.0; 4];
    let bad_bias = Ctx { n: 3, bias: Some(&[1.0, 2.0]), residual: None };
    let err = apply(&steps, &mut tile, bad_bias).expect_err("bias 长度 ≠ 列数");
    assert_eq!(err.code(), Code::Invalid);

    let bad_residual = Ctx { n: 2, bias: None, residual: Some(&[1.0; 3]) };
    let err = apply(&steps, &mut tile, bad_residual).expect_err("residual 长度 ≠ 瓦片");
    assert_eq!(err.code(), Code::Invalid);
}

/// `store` 在标量参考里是空操作：施加前后瓦片不变。
#[test]
fn store_is_a_no_op_in_the_reference() {
    let steps = [Step::Store];
    let mut tile = [1.0, 2.0, 3.0];
    let before = tile;
    apply(&steps, &mut tile, Ctx { n: 3, bias: None, residual: None }).expect("store");
    assert_eq!(tile, before);
}
