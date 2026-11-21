//! RoPE（旋转位置编码）：按位置 `p` 对一对对分量做二维旋转。
//!
//! 对偶数下标 `2i` 与奇数下标 `2i+1`，频率 `freq_i = base^(-2i/d)`，角度 `angle = p*freq_i`：
//!
//! ```text
//! y[2i]   = x[2i] cos(angle) - x[2i+1] sin(angle)
//! y[2i+1] = x[2i] sin(angle) + x[2i+1] cos(angle)
//! ```
//!
//! 常量 `base = 10000`。本模块自带形状、代价、样本、标量参考与遍历，`#[annotation::model]` 装配。

use crate::model::{Buffer, Cost, Sample, ShapeContext, Values, cost_n, x_buffer};
use diagnostic::{Code, Error};

/// 频率底数。
pub const BASE: f32 = 10000.0;

// 单轴定义与形状投影复用 unary 骨架。
pub use crate::unary::{AXES, project_shape};

/// 标量代价：每对分量约 6 次运算（两对乘加 + 一次 sin/cos 视为共享），按元素约 4 次估算。
///
/// # Errors
///
/// 形状缺轴时返回说明。
pub fn cost(shape: &ShapeContext) -> Result<Cost, Error> {
    cost_n(shape, "rope", 4.0, 1)
}

/// 确定性样本：`x`、`y` 两个等长（偶数）缓冲，位置固定为 7。
pub fn sample(len: usize, _: usize) -> Sample {
    Sample {
        m: len,
        n: len,
        scalars: vec![("N", len as i64), ("pos", 7)],
        buffers: vec![
            x_buffer(len),
            Buffer {
                name: "y",
                values: Values::F32(vec![0.0; len]),
            },
        ],
    }
}

/// 标量参考：与内核同一公式，位置取样本标量 `P`。
pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
    let x = sample.f32("x")?;
    let pos = sample
        .scalar("pos")
        .ok_or_else(|| Error::new(Code::Invalid, "rope 样本缺标量 `pos`"))? as f32;
    rotate(x, pos)
}

/// 共享旋转：参考与内核的同一套数学。末尾若有无法配对的元素，原样透传。
fn rotate(x: &[f32], pos: f32) -> diagnostic::Result<Vec<f32>> {
    let mut out = vec![0.0; x.len()];
    rotate_into(x, pos, &mut out)?;
    Ok(out)
}

/// 旋转写进给定缓冲：参考与内核共用的算术本体。内核路径由此直写调用方缓冲，
/// 不再先落一份临时 `Vec` 再整块拷贝；末尾不成对的元素原样透传。
fn rotate_into(x: &[f32], pos: f32, out: &mut [f32]) -> diagnostic::Result<()> {
    if x.len() != out.len() {
        return Err(Error::new(
            Code::Invalid,
            format!("缓冲长度不一致：x {}、out {}", x.len(), out.len()),
        ));
    }
    let d = x.len();
    let pairs = d / 2;
    for i in 0..pairs {
        let freq = BASE.powf(-((2 * i) as f32) / d as f32);
        let angle = pos * freq;
        let (sin, cos) = angle.sin_cos();
        let (e, o) = (x[2 * i], x[2 * i + 1]);
        out[2 * i] = e * cos - o * sin;
        out[2 * i + 1] = e * sin + o * cos;
    }
    if d % 2 == 1 {
        out[d - 1] = x[d - 1];
    }
    Ok(())
}

/// 内核本体：结果写进调用方给的缓冲。
///
/// # Errors
///
/// 输入与输出长度不一致时返回说明；末尾无法配对的奇数元素原样透传（不报错）。
pub fn forward(pos: usize, x: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
    rotate_into(x, pos as f32, out)
}

/// 校验多头旋转长度并派生头数；非法头宽、不等长或非头宽倍数在此一次性拦下。
fn rope_heads(x: &[f32], out: &[f32], head_dim: usize) -> diagnostic::Result<usize> {
    if head_dim == 0 || x.len() != out.len() || !x.len().is_multiple_of(head_dim) {
        return Err(Error::new(
            Code::Invalid,
            format!(
                "RoPE 头宽 {head_dim} 非法，或长度 {} 与其/输出长度不符",
                x.len()
            ),
        ));
    }
    Ok(x.len() / head_dim)
}

/// 多头旋转：头优先块布局 `[heads, head_dim]`，每段独立按头宽重计频率（段内下标从 0 起），
/// 不把整段长度当一个头宽；段内旋转算术复用 `rotate_into`，这里只做头分段编排。
///
/// # Errors
///
/// 头宽为 0、缓冲不是头宽的整数倍，或输入输出长度不一致时返回说明。
pub fn forward_heads(
    pos: usize,
    x: &[f32],
    out: &mut [f32],
    head_dim: usize,
) -> diagnostic::Result<()> {
    let heads = rope_heads(x, out, head_dim)?;
    for head in 0..heads {
        let start = head * head_dim;
        let span = start..start + head_dim;
        rotate_into(&x[span.clone()], pos as f32, &mut out[span])?;
    }
    Ok(())
}

/// 批量校验：返回单序列段长 `heads·head_dim`；批数取 `positions.len()`，段内头宽/倍数守卫
/// 交给 [`forward_heads`]。
fn rope_batch_segments(
    positions: &[usize],
    x: &[f32],
    out: &[f32],
    head_dim: usize,
) -> diagnostic::Result<usize> {
    let batch = positions.len();
    if batch == 0 || head_dim == 0 {
        return Err(Error::new(
            Code::Invalid,
            format!("批量序列数 {batch} 与头宽 {head_dim} 都必须非零"),
        ));
    }
    if x.len() != out.len() || !x.len().is_multiple_of(batch * head_dim) {
        return Err(Error::new(
            Code::Invalid,
            format!(
                "x 长度 {} 不是 batch·head_dim（{}）的整数倍，或与输出不等长",
                x.len(),
                batch * head_dim
            ),
        ));
    }
    Ok(x.len() / batch)
}

/// 批量旋转：`[batch, heads, head_dim]` 头优先块，第 `b` 段连续 `heads·head_dim`，每序列
/// 独立位置 `positions[b]`；段内复用 [`forward_heads`] 头分段编排，段间互不串段。
///
/// # Errors
///
/// `positions` 为空、头宽为 0、输入输出不等长或长度不是批段整数倍时返回说明。
pub fn forward_batch(
    positions: &[usize],
    x: &[f32],
    out: &mut [f32],
    head_dim: usize,
) -> diagnostic::Result<()> {
    let seg = rope_batch_segments(positions, x, out, head_dim)?;
    for (seq, &pos) in positions.iter().enumerate() {
        let start = seq * seg;
        forward_heads(
            pos,
            &x[start..start + seg],
            &mut out[start..start + seg],
            head_dim,
        )?;
    }
    Ok(())
}

/// 标记类型：承载 `#[model]` 语义。
pub struct Rope;

/// RoPE 的具体语义别名。
#[annotation::model(name = "rope_f32", shape = [4096, 4096], dtype = "f32", skeleton = "rope")]
pub type RopeF32 = Rope;

crate::operator_here!(Rope, "rope_f32", 1, (4096, 4096), true, reference);

#[cfg(test)]
mod tests {
    include!("../../tests/rope.rs");
}
