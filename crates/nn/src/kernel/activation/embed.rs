//! Embedding 行查表：`y[n, :] = table[idx[n], :]`（行向量 gather）。
use crate::model::{AxisRole, AxisSpec, Buf, Cost, Sample, ShapeCtx, Values};
use diagnostic::{Code, Error};

/// 权重表行数（闭集内的固定档）。
pub const VOCAB: usize = 64;

/// 两条基本轴：token 轴 `n`（输入）、行宽轴 `d`（输出）。
pub const AXES: &[AxisSpec] = &[
    AxisSpec {
        role: AxisRole::Input,
        label: 'n',
    },
    AxisSpec {
        role: AxisRole::Output,
        label: 'd',
    },
];

/// `(m, n)` 即 `(token 数 n, 行宽 d)`。
///
/// # Errors
///
/// 当前实现不会失败。
pub fn project_shape(tokens: usize, dim: usize) -> Result<ShapeCtx, Error> {
    ShapeCtx::new([('n', tokens), ('d', dim)])
}

/// 标量代价：每个 token 搬一行 `D`（读表一行 + 写输出一行），下标转换按 1 次/token 估算。
///
/// 常驻表按被访问的行数计价，不把整张词表读一遍（表流量口径属于代价模型，层不重复推）。
///
/// # Errors
///
/// 形状缺轴时返回说明。
pub fn cost(shape: &ShapeCtx) -> Result<Cost, Error> {
    let n = shape
        .get('n')
        .ok_or_else(|| Error::new(Code::Invalid, "embed 形状缺轴 `n`"))?;
    let d = shape
        .get('d')
        .ok_or_else(|| Error::new(Code::Invalid, "embed 形状缺轴 `d`"))?;
    let elements = (n as u64) * (d as u64);
    Ok(Cost {
        flops: n as f64,
        read: elements * 4,
        write: elements * 4,
    })
}

/// 确定性样本：`idx`（f32 承载的整数下标，长度 n）、`table`（长度 `V·D`）、`y`（长度 `n·D`）。
pub fn sample(tokens: usize, dim: usize) -> Sample {
    let table = (0..VOCAB * dim)
        .map(|idx| (idx % VOCAB) as f32 * 0.5)
        .collect();
    Sample {
        m: tokens,
        n: dim,
        scalars: vec![("N", tokens as i64), ("V", VOCAB as i64), ("D", dim as i64)],
        bufs: vec![
            Buf {
                name: "idx",
                values: Values::F32((0..tokens).map(|i| (i % VOCAB) as f32).collect()),
            },
            Buf {
                name: "table",
                values: Values::F32(table),
            },
            Buf {
                name: "y",
                values: Values::F32(vec![0.0; tokens * dim]),
            },
        ],
    }
}

/// 标量参考：与内核同一行 gather 公式。
pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
    let idx = sample.f32("idx")?;
    let table = sample.f32("table")?;
    gather(idx, table, sample.n)
}

/// 共享行查表：参考与内核同一套；`dim` 为行宽，表长必须是 `dim` 的整数倍，行数由此推出。
fn gather(idx: &[f32], table: &[f32], dim: usize) -> diagnostic::Result<Vec<f32>> {
    if dim == 0 {
        return Err(Error::new(Code::Invalid, "embed 行宽不能为零"));
    }
    if !table.len().is_multiple_of(dim) {
        return Err(Error::new(
            Code::Invalid,
            format!("embedding 表长 {} 不是行宽 {dim} 的整数倍", table.len()),
        ));
    }
    let rows = table.len() / dim;
    let mut out = Vec::with_capacity(idx.len() * dim);
    for value in idx {
        let k = *value as usize;
        if k >= rows {
            return Err(Error::new(
                Code::Invalid,
                format!("embedding 下标 {k} 越界（词表 {rows}）"),
            ));
        }
        let row = k * dim;
        out.extend_from_slice(&table[row..row + dim]);
    }
    Ok(out)
}

/// 内核本体：结果写进调用方给的缓冲。
///
/// 行宽 `D` 由输出长度与 token 数推出：`out.len() == idx.len()·D`；表长必须是 `D` 的整数倍。
///
/// # Errors
///
/// 缓冲长度不自洽、表长不符或下标越界时返回说明。
pub fn forward(idx: &[f32], table: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
    let tokens = idx.len();
    if tokens == 0 {
        return Err(Error::new(Code::Invalid, "embed 索引为空"));
    }
    if !out.len().is_multiple_of(tokens) {
        return Err(Error::new(
            Code::Invalid,
            format!("输出长度 {} 不能按 token 数 {tokens} 整除出行宽", out.len()),
        ));
    }
    let dim = out.len() / tokens;
    let gathered = gather(idx, table, dim)?;
    out.copy_from_slice(&gathered);
    Ok(())
}

/// 标记类型：承载 `#[model]` 语义。
pub struct Embed;

/// Embedding 的具体语义别名。
#[annotation::model(name = "embed_f32", shape = [8, 64], dtype = "f32", skeleton = "embed")]
pub type EmbedF32 = Embed;

crate::operator_here!(Embed, "embed_f32", 2, (8, 64), true, reference);

#[cfg(test)]
mod tests {
    include!("../../../tests/kernel/embed.rs");
}
