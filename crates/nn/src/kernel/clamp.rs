//! 逐元素裁剪：`y[n] = clamp(x[n], lo, hi)`；`lo` / `hi` 取模块常量（f32 标量入参通道未贯通，同 `dequant_i8` 口径）。

use crate::model::{Buf, Cost, Sample, ShapeCtx, Values, cost_n, x_buf};
use diagnostic::{Code, Error};
use tensor::View;

/// 单轴定义与形状投影复用 unary 骨架：本文件只声明 clamp 特有的代价、样本与遍历。
pub use crate::unary::{AXES, project_shape};

/// 样本固定的下界（内核模板里写同一常量）。
pub const LO: f32 = -1.0;
/// 样本固定的上界（内核模板里写同一常量）。
pub const HI: f32 = 1.0;

/// 标量代价：每元素两次比较（下界、上界），读一次写一次。
///
/// # Errors
///
/// 形状缺轴时返回说明。
pub fn cost(shape: &ShapeCtx) -> Result<Cost, Error> {
    cost_n(shape, "clamp", 2.0, 1)
}

/// 确定性样本：`x` 夹具、`y` 零缓冲；`lo` / `hi` 取模块常量（签名通道只支持整数标量）。
pub fn sample(len: usize, _: usize) -> Sample {
    Sample {
        m: len,
        n: len,
        scalars: vec![("N", len as i64)],
        bufs: vec![
            x_buf(len),
            Buf {
                name: "y",
                values: Values::F32(vec![0.0; len]),
            },
        ],
    }
}

/// 标量界前置：两界有限且下界不大于上界。
///
/// # Errors
///
/// 任一界非有限，或下界大于上界时返回说明。
fn check_bounds(lo: f32, hi: f32) -> diagnostic::Result<()> {
    if !lo.is_finite() || !hi.is_finite() {
        return Err(Error::new(Code::Invalid, "clamp 界必须是有限值"));
    }
    if lo > hi {
        return Err(Error::new(
            Code::Invalid,
            format!("clamp 下界 {lo} 大于上界 {hi}"),
        ));
    }
    Ok(())
}

/// 标量参考：逐元素裁剪，与内核同一公式，界取模块常量。
///
/// # Errors
///
/// 样本缺 `x`，或界非法时返回说明。
pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
    let x = sample.f32("x")?;
    check_bounds(LO, HI)?;
    Ok(x.iter().map(|value| value.clamp(LO, HI)).collect())
}

/// 内核本体：结果写进调用方给的缓冲。
///
/// # Errors
///
/// `x` 与 `out` 不等长，或界非法时返回说明。
pub fn forward(lo: f32, hi: f32, x: View<'_, f32>, out: &mut [f32]) -> diagnostic::Result<()> {
    let x = x.as_slice();
    check_bounds(lo, hi)?;
    if x.len() != out.len() {
        return Err(Error::new(
            Code::Invalid,
            format!("缓冲长度不一致：x {}、out {}", x.len(), out.len()),
        ));
    }
    for (place, value) in out.iter_mut().zip(x) {
        *place = value.clamp(lo, hi);
    }
    Ok(())
}

/// 标记类型：承载 `#[model]` 语义。
pub struct Clamp;

/// clamp 的具体语义别名。
#[annotation::model(name = "clamp_f32", shape = [4096, 4096], dtype = "f32", skeleton = "elementwise")]
pub type ClampF32 = Clamp;

// 基座契约（ADR 0050）：本模块自带具体入口，直接转发；图输入只有 `x`，`lo` / `hi` 取模块常量。
crate::operator_here!(Clamp, "clamp_f32", 1, (4096, 4096), false, reference);

#[cfg(test)]
mod tests {
    include!("../../tests/kernel_clamp.rs");
}
