//! 2 位反量化：打包字节 `q`（每字节四个 2 位码），`y = scale * (code - 2)`。
//!
//! 与 [`crate::dequant_i4`] 的差别是每字节四个码：`n` 个输出元素对应 `n / 4` 个字节。
//! 低 2 位在前、高位在后；码是无符号 0..3，按对称零点 2 转成 -2..=1。本模块自带形状、
//! 代价、样本、参考与遍历，`#[annotation::model]` 装配。

use crate::model::{Buffer, Cost, Sample, ShapeContext, Values};
use diagnostic::{Code, Error};

/// 反量化标度（2 的负幂网格，f32 精确）。
pub const SCALE: f32 = 0.25;
/// 2 位对称零点：码 0..3 减去 2 得到 -2..=1。
pub const ZERO: f32 = 2.0;

// 单轴定义与形状投影复用 unary 骨架。
pub use crate::unary::{AXES, project_shape};

/// 标量代价：每元素一次减法一次乘法；读打包字节（n/4 个），写 n 个 f32。
///
/// # Errors
///
/// 形状缺轴时返回说明。
pub fn cost(shape: &ShapeContext) -> Result<Cost, Error> {
    let n = shape
        .get('n')
        .ok_or_else(|| Error::new(Code::Invalid, "dequant 形状缺轴 `n`"))?;
    Ok(Cost {
        flops: (n as f64) * 2.0,
        read: (n as u64) / 4,
        write: (n as u64) * 4,
    })
}

/// 确定性样本：`q`（打包字节，n/4 个）、`y`（n 个 f32）。
pub fn sample(len: usize, _: usize) -> Sample {
    let bytes = len / 4;
    Sample {
        m: len,
        n: len,
        scalars: vec![("N", len as i64)],
        buffers: vec![
            Buffer {
                name: "q",
                values: Values::U8((0..bytes).map(|i| (i * 37 % 251) as u8).collect()),
            },
            Buffer {
                name: "y",
                values: Values::F32(vec![0.0; len]),
            },
        ],
    }
}

/// 解包一个字节的第 `slot` 个 2 位段（0 最低位），按零点转成有符号值。
fn code(byte: u8, slot: usize) -> f32 {
    ((byte >> (2 * slot)) & 0x03) as f32 - ZERO
}

/// 标量参考：逐字节解四个码，与内核同一公式。
pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
    let q = sample.u8("q")?;
    let mut out = Vec::with_capacity(q.len() * 4);
    for byte in q {
        out.push(SCALE * code(*byte, 0));
        out.push(SCALE * code(*byte, 1));
        out.push(SCALE * code(*byte, 2));
        out.push(SCALE * code(*byte, 3));
    }
    Ok(out)
}

/// 内核本体：结果写进调用方给的缓冲。
///
/// # Errors
///
/// 输出长度不是输入字节数的 4 倍时返回说明。
pub fn forward(scale: f32, zero: f32, q: &[u8], out: &mut [f32]) -> diagnostic::Result<()> {
    if out.len() != q.len() * 4 {
        return Err(Error::new(
            Code::Invalid,
            format!("缓冲长度不符：q {} 字节、out {} 元素", q.len(), out.len()),
        ));
    }
    // 固定 4 输出：as_chunks_mut 直接给出数组型切片，编译器展开四个码位的计算。
    // 上面的长度检查保证 remainder 为空（out 长度恰为 q 的 4 倍）。
    let (quads, _remainder) = out.as_chunks_mut::<4>();
    for (byte, quad) in q.iter().zip(quads) {
        quad[0] = scale * (((*byte) & 0x03) as f32 - zero);
        quad[1] = scale * (((*byte >> 2) & 0x03) as f32 - zero);
        quad[2] = scale * (((*byte >> 4) & 0x03) as f32 - zero);
        quad[3] = scale * ((*byte >> 6) as f32 - zero);
    }
    Ok(())
}

/// 标记类型：承载 `#[model]` 语义。
pub struct DequantI2;

/// 2 位反量化的具体语义别名。
#[annotation::model(name = "dequant_i2", shape = [4096, 4096], dtype = "f32", skeleton = "dequant_i2")]
pub type DequantI2F32 = DequantI2;

#[cfg(test)]
#[path = "../tests/dequant_i2_tests.rs"]
mod tests;
