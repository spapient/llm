//! 4 位反量化：打包字节 `q`（每字节两个 nibble），`y = scale * (nibble - 8)`。
//!
//! 与 [`crate::dequant_i8`] 的差别是输入为打包字节：`n` 个输出元素对应 `n / 2` 个字节。
//! 低 4 位在前、高 4 位在后；nibble 是无符号 0..15，按对称零点 8 转有符号。本模块自带形状、
//! 代价、样本、参考与遍历，`#[annotation::model]` 装配。

use crate::model::{Buffer, Cost, Sample, ShapeContext, Values};
use diagnostic::{Code, Error};

/// 反量化标度（2 的负幂网格，f32 精确）。
pub const SCALE: f32 = 0.25;
/// 4 位对称零点。
pub const ZERO: f32 = 8.0;

// 单轴定义与形状投影复用 unary 骨架。
pub use crate::unary::{AXES, project_shape};

/// 标量代价：每元素一次减法一次乘法；读打包字节（n/2 个），写 n 个 f32。
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
        read: (n as u64) / 2,
        write: (n as u64) * 4,
    })
}

/// 确定性样本：`q`（打包字节，n/2 个）、`y`（n 个 f32）。
pub fn sample(len: usize, _: usize) -> Sample {
    let bytes = len / 2;
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

/// 标量参考：逐字节解两个 nibble，与内核同一公式。
pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
    let q = sample.u8("q")?;
    let mut out = Vec::with_capacity(q.len() * 2);
    for byte in q {
        out.push(SCALE * ((byte & 0x0F) as f32 - ZERO));
        out.push(SCALE * ((byte >> 4) as f32 - ZERO));
    }
    Ok(out)
}

/// 内核本体：结果写进调用方给的缓冲。
///
/// # Errors
///
/// 输出长度不是输入字节数的 2 倍时返回说明。
pub fn forward(scale: f32, zero: f32, q: &[u8], out: &mut [f32]) -> diagnostic::Result<()> {
    if out.len() != q.len() * 2 {
        return Err(Error::new(
            Code::Invalid,
            format!("缓冲长度不符：q {} 字节、out {} 元素", q.len(), out.len()),
        ));
    }
    // 固定 2 输出：as_chunks_mut 直接给出数组型切片，编译器展开低/高 nibble 两条计算。
    // 上面的长度检查保证 remainder 为空（out 长度恰为 q 的 2 倍）。
    let (pairs, _remainder) = out.as_chunks_mut::<2>();
    for (byte, pair) in q.iter().zip(pairs) {
        pair[0] = scale * ((byte & 0x0F) as f32 - zero);
        pair[1] = scale * ((byte >> 4) as f32 - zero);
    }
    Ok(())
}

/// 标记类型：承载 `#[model]` 语义。
pub struct DequantI4;

/// 4 位反量化的具体语义别名。
#[annotation::model(name = "dequant_i4", shape = [4096, 4096], dtype = "f32", skeleton = "dequant_i4")]
pub type DequantI4F32 = DequantI4;

#[cfg(test)]
mod tests;
