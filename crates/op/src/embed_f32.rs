//! Embedding 查表：`y[n] = table[idx[n]]`（gather）。
//!
//! 索引以 f32 承载（当前 `Values` 只有 f32 变体），内核取下标时转成整数。权重表长度固定为
//! `V = 64`，由样本标量给出；索引落在 `[0, V)`。本模块自带形状、代价、样本、参考与遍历。

use crate::model::{Buffer, Cost, Sample, ShapeContext, Values};
use diagnostic::{Code, Error};

/// 权重表大小（闭集内的固定档）。
pub const VOCAB: usize = 64;

// 单轴定义与形状投影复用 unary 骨架。
pub use crate::unary::{AXES, project_shape};

/// 标量代价：每元素一次 gather 读 + 一次写，运算按 1 次（下标转换）估算。
///
/// # Errors
///
/// 形状缺轴时返回说明。
pub fn cost(shape: &ShapeContext) -> Result<Cost, Error> {
    let n = shape
        .get('n')
        .ok_or_else(|| Error::new(Code::Invalid, "embed 形状缺轴 `n`"))?;
    Ok(Cost {
        flops: n as f64,
        read: (n as u64) * 4,
        write: (n as u64) * 4,
    })
}

/// 确定性样本：`idx`（f32 承载的整数下标）、`table`（长度 V）、`y`（长度 n）。
pub fn sample(len: usize, _: usize) -> Sample {
    Sample {
        m: len,
        n: len,
        scalars: vec![("N", len as i64), ("V", VOCAB as i64)],
        buffers: vec![
            Buffer {
                name: "idx",
                values: Values::F32((0..len).map(|i| (i % VOCAB) as f32).collect()),
            },
            Buffer {
                name: "table",
                values: Values::F32((0..VOCAB).map(|i| (i as f32) * 0.5).collect()),
            },
            Buffer {
                name: "y",
                values: Values::F32(vec![0.0; len]),
            },
        ],
    }
}

/// 标量参考：与内核同一 gather 公式。
pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
    let idx = sample.f32("idx")?;
    let table = sample.f32("table")?;
    gather(idx, table)
}

/// 共享查表：参考与内核同一套。
fn gather(idx: &[f32], table: &[f32]) -> diagnostic::Result<Vec<f32>> {
    let mut out = Vec::with_capacity(idx.len());
    for value in idx {
        let k = *value as usize;
        if k >= table.len() {
            return Err(Error::new(
                Code::Invalid,
                format!("embedding 下标 {} 越界（表长 {}）", k, table.len()),
            ));
        }
        out.push(table[k]);
    }
    Ok(out)
}

/// 内核本体：结果写进调用方给的缓冲。
///
/// # Errors
///
/// 索引与输出等长，否则返回说明。
pub fn forward(idx: &[f32], table: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
    if idx.len() != out.len() {
        return Err(Error::new(
            Code::Invalid,
            format!("缓冲长度不一致：idx {}、out {}", idx.len(), out.len()),
        ));
    }
    let gathered = gather(idx, table)?;
    out.copy_from_slice(&gathered);
    Ok(())
}

/// 标记类型：承载 `#[model]` 语义。
pub struct Embed;

/// Embedding 的具体语义别名。
#[annotation::model(name = "embed_f32", shape = [4096, 4096], dtype = "f32", skeleton = "embed")]
pub type EmbedF32 = Embed;

#[cfg(test)]
#[path = "embed_f32_tests.rs"]
mod tests;
