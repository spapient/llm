//! 打包反量化骨架：每字节 `CPB` 个码，`y = scale · (code − zero)`。
//!
//! `dequant_i2`（4 码/字节）与 `dequant_i4`（2 码/字节）共用形状投影、确定性样本、标量参考与
//! 打包内核；位宽、掩码、零点作为参数传入，解包与长度校验只写一份。`dequant_i8` 不打包
//! （f32 承载量化值），不走本骨架。

use crate::model::{Buffer, Cost, Sample, ShapeContext, Values};
use diagnostic::{Code, Error};

// 单轴定义与形状投影复用 unary 骨架。
pub use crate::unary::{AXES, project_shape};

/// 反量化标度（2 的负幂网格，f32 精确）。
pub const SCALE: f32 = 0.25;

/// 打包格式的前置约束：输出长度必须是输入字节数的 `cpb` 倍。
fn check_len(q: &[u8], out: &[f32], cpb: usize) -> Result<(), Error> {
    if out.len() != q.len() * cpb {
        return Err(Error::new(
            Code::Invalid,
            format!("缓冲长度不符：q {} 字节、out {} 元素", q.len(), out.len()),
        ));
    }
    Ok(())
}

/// 仿射参数校验：标度不能为零（否则整组塌成常数、失去量化口径），零点必须落在码域内。
///
/// 码域是 `0..2^width`：零点是码点到有符号值的平移量，越出码域意味着存在无法表示的码点，
/// 属于配置错误而不是数值极端。两条判据与位宽同源：`width = 8 / cpb`，打包格式各自传 cpb。
///
/// # Errors
///
/// 标度为零、零点不是有限值，或零点不在 `0..2^width` 内时返回 [`Code::Invalid`]。
pub fn check_affine(scale: f32, zero: f32, cpb: usize) -> Result<(), Error> {
    if scale == 0.0 || !scale.is_finite() {
        return Err(Error::new(Code::Invalid, "反量化标度必须是非零有限值"));
    }
    if !zero.is_finite() {
        return Err(Error::new(Code::Invalid, "反量化零点必须是有限值"));
    }
    let width = 8 / cpb;
    let domain = 1u32 << width;
    if !(0.0..=domain as f32).contains(&zero) {
        return Err(Error::new(
            Code::Invalid,
            format!("零点 {zero} 越出码域 0..{domain}（{width} 位码）"),
        ));
    }
    Ok(())
}

/// 标量代价：每元素一次减法一次乘法；读打包字节（`n / cpb` 个），写 `n` 个 f32。
///
/// # Errors
///
/// 形状缺轴时返回说明。
pub fn cost_packed(shape: &ShapeContext, cpb: usize) -> Result<Cost, Error> {
    let n = shape
        .get('n')
        .ok_or_else(|| Error::new(Code::Invalid, "dequant 形状缺轴 `n`"))?;
    Ok(Cost {
        flops: (n as f64) * 2.0,
        read: (n / cpb) as u64,
        write: (n as u64) * 4,
    })
}

/// 确定性样本：`q`（打包字节，`len / cpb` 个）、`y`（`len` 个 f32）。
pub fn sample_packed(len: usize, cpb: usize) -> Sample {
    let bytes = len / cpb;
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

/// 解包字节内第 `slot` 个码，按对称零点回退成有符号值。
pub fn unpack(byte: u8, slot: usize, cpb: usize, zero: f32) -> f32 {
    let width = 8 / cpb;
    let mask = (1u8 << width) - 1;
    ((byte >> (slot * width)) & mask) as f32 - zero
}

/// 标量参考：逐字节解 `cpb` 个码，与内核同一公式。
pub fn reference_packed(q: &[u8], cpb: usize, zero: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(q.len() * cpb);
    for byte in q {
        for slot in 0..cpb {
            out.push(SCALE * unpack(*byte, slot, cpb, zero));
        }
    }
    out
}

/// 内核本体：结果写进调用方给的缓冲。
///
/// # Errors
///
/// 输出长度不是输入字节数的 `cpb` 倍时返回说明。
pub fn forward_packed(
    scale: f32,
    zero: f32,
    q: &[u8],
    out: &mut [f32],
    cpb: usize,
) -> diagnostic::Result<()> {
    check_len(q, out, cpb)?;
    check_affine(scale, zero, cpb)?;
    for (byte, chunk) in q.iter().zip(out.chunks_mut(cpb)) {
        for (slot, place) in chunk.iter_mut().enumerate() {
            *place = scale * unpack(*byte, slot, cpb, zero);
        }
    }
    Ok(())
}
