// `epilogue::apply` 的执行面：几何步、激活步、量化步、顺序与错误。

use super::apply;
use prof::epilogue::parse;

/// 空步骤是恒等：`out` 不变。
#[test]
fn empty_steps_are_identity() {
    let mut out = [1.0, 2.0, 3.0];
    assert!(apply(&[], &mut out, 3, None, None).is_ok());
    assert_eq!(out, [1.0, 2.0, 3.0]);
}

/// `bias` 按列广播：`out[r][c] += bias[c]`。
#[test]
fn bias_broadcasts_across_rows() {
    let steps = parse("bias").unwrap();
    let mut out = [1.0, 2.0, 3.0, 4.0];
    assert!(apply(&steps, &mut out, 2, Some(&[10.0, 20.0]), None).is_ok());
    assert_eq!(out, [11.0, 22.0, 13.0, 24.0]);
}

/// `bias` 列宽不符：返回错误而不是静默截断。
#[test]
fn bias_wrong_width_is_rejected() {
    let steps = parse("bias").unwrap();
    let mut out = [0.0; 4];
    assert!(apply(&steps, &mut out, 2, Some(&[10.0, 20.0, 30.0]), None).is_err());
}

/// `residual` 逐元素加：`out[i] += residual[i]`。
#[test]
fn residual_adds_elementwise() {
    let steps = parse("residual").unwrap();
    let mut out = [1.0, 2.0, 3.0];
    assert!(apply(&steps, &mut out, 3, None, Some(&[10.0, 20.0, 30.0])).is_ok());
    assert_eq!(out, [11.0, 22.0, 33.0]);
}

/// `residual` 长度不符：返回错误。
#[test]
fn residual_wrong_len_is_rejected() {
    let steps = parse("residual").unwrap();
    let mut out = [0.0; 4];
    assert!(apply(&steps, &mut out, 4, None, Some(&[1.0, 2.0])).is_err());
}

/// `scale:<α>` 逐元素乘。
#[test]
fn scale_multiplies() {
    let steps = parse("scale:0.5").unwrap();
    let mut out = [2.0, 4.0, 6.0];
    assert!(apply(&steps, &mut out, 3, None, None).is_ok());
    assert_eq!(out, [1.0, 2.0, 3.0]);
}

/// `act:relu` 负区归零、正区恒等。
#[test]
fn relu_applies() {
    let steps = parse("act:relu").unwrap();
    let mut out = [-2.0, 0.0, 2.0];
    assert!(apply(&steps, &mut out, 3, None, None).is_ok());
    assert_eq!(out, [0.0, 0.0, 2.0]);
}

/// `act:gelu` 与 [`crate::gelu_f32::forward`] 逐位一致：执行层只转发、不写第二份算术。
#[test]
fn gelu_delegates_to_unary_kernel() {
    let steps = parse("act:gelu").unwrap();
    let input = [-2.0, -1.0, 0.0, 1.0, 2.0];
    let mut out = input;
    apply(&steps, &mut out, 5, None, None).unwrap();
    let mut reference = input;
    crate::gelu_f32::forward(&input, &mut reference).unwrap();
    assert_eq!(out, reference);
}

/// `quant:i8` 对称 round-trip：步长 0.25，最大误差 0.125，越界饱和。
#[test]
fn quant_i8_rounds_to_grid() {
    let steps = parse("quant:i8").unwrap();
    let mut out = [0.4, 0.13, -0.4, 100.0];
    apply(&steps, &mut out, 4, None, None).unwrap();
    assert!((out[0] - 0.5).abs() < 1e-6, "0.4 -> {}", out[0]);
    assert!((out[1] - 0.25).abs() < 1e-6, "0.13 -> {}", out[1]);
    assert!((out[2] + 0.5).abs() < 1e-6, "-0.4 -> {}", out[2]);
    assert!((out[3] - 31.75).abs() < 1e-6, "饱和 -> {}", out[3]);
}

/// `store` 是无副作用标记：`out` 不变。
#[test]
fn store_is_noop() {
    let steps = parse("store").unwrap();
    let mut out = [1.0, 2.0];
    assert!(apply(&steps, &mut out, 2, None, None).is_ok());
    assert_eq!(out, [1.0, 2.0]);
}

/// 顺序语义：`bias; scale:2; quant:i8` 严格按声明顺序执行。
#[test]
fn pipeline_runs_in_declared_order() {
    let steps = parse("bias; scale:2; quant:i8").unwrap();
    let mut out = [0.5, 1.5, 2.5, 3.5];
    assert!(apply(&steps, &mut out, 2, Some(&[1.0, 2.0]), None).is_ok());
    // bias → [1.5, 3.5, 3.5, 5.5]；scale:2 → [3, 7, 7, 11]；quant:i8 已在 0.25 网格上，恒等。
    assert_eq!(out, [3.0, 7.0, 7.0, 11.0]);
}
