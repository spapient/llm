use super::*;
use diagnostic::Code;

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
        &[
            Step::Scale(2.5),
            Step::Quant(DType::I4),
            Step::Act(ActKind::LeakyRelu),
        ],
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
    let steps = [
        Step::Bias,
        Step::Act(ActKind::Relu),
        Step::Scale(1.5),
        Step::Store,
    ];
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
