//! 测试内核共享的操作数数量、dtype 与等长校验；实际数值闭包保持独立。

use crate::invalid;
use crate::queue::{View, ViewMut};
use diagnostic::Result;

pub(crate) fn map_f32(
    inputs: &[View<'_>],
    outputs: &mut [ViewMut<'_>],
    arity: usize,
    site: &'static str,
    apply: impl FnOnce(&[&[f32]], &mut [f32]),
) -> Result<()> {
    if inputs.len() != arity || outputs.len() != 1 {
        return Err(invalid(site, "操作数数量不符"));
    }
    let typed: Vec<&[f32]> = inputs
        .iter()
        .map(|input| input.f32().ok_or_else(|| invalid(site, "输入不是 f32")))
        .collect::<Result<_>>()?;
    let output = outputs[0]
        .f32_mut()
        .ok_or_else(|| invalid(site, "输出不是 f32"))?;
    if typed.iter().any(|input| input.len() != output.len()) {
        return Err(invalid(site, "形状不一致"));
    }
    apply(&typed, output);
    Ok(())
}
