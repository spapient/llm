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

use crate::model::{Buffer, Cost, Sample, ShapeContext, Values};
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
    let n = shape
        .get('n')
        .ok_or_else(|| Error::new(Code::Invalid, "rope 形状缺轴 `n`"))?;
    let bytes = (n as u64) * 4;
    Ok(Cost {
        flops: (n as f64) * 4.0,
        read: bytes,
        write: bytes,
    })
}

/// 确定性样本：`x`、`y` 两个等长（偶数）缓冲，位置固定为 7。
pub fn sample(len: usize, _: usize) -> Sample {
    Sample {
        m: len,
        n: len,
        scalars: vec![("N", len as i64), ("pos", 7)],
        buffers: vec![
            Buffer {
                name: "x",
                values: Values::F32((0..len).map(|i| ((i as f32) * 0.25) - 2.0).collect()),
            },
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
    let d = x.len();
    let mut out = vec![0.0; d];
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
    Ok(out)
}

/// 内核本体：结果写进调用方给的缓冲。
///
/// # Errors
///
/// 输入与输出等长且为偶数，否则返回说明。
pub fn forward(pos: usize, x: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
    if x.len() != out.len() {
        return Err(Error::new(
            Code::Invalid,
            format!("缓冲长度不一致：x {}、out {}", x.len(), out.len()),
        ));
    }
    let rotated = rotate(x, pos as f32)?;
    out.copy_from_slice(&rotated);
    Ok(())
}

/// 标记类型：承载 `#[model]` 语义。
pub struct Rope;

/// RoPE 的具体语义别名。
#[annotation::model(name = "rope_f32", shape = [4096, 4096], dtype = "f32", skeleton = "rope")]
pub type RopeF32 = Rope;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_matches_the_scalar_reference() {
        let sample = sample(128, 1);
        let expected = reference(&sample).expect("参考可计算");
        let mut out = vec![0.0; 128];
        forward(7, sample.f32("x").expect("x"), &mut out).expect("等长");
        assert_eq!(out, expected);
    }

    #[test]
    fn an_odd_length_passes_its_tail_through() {
        let out = rotate(&[1.0, 2.0, 3.0], 0.0).expect("奇数长度不报错");
        assert_eq!(out[2], 3.0, "末尾无法配对的元素原样透传");
    }
}
