//! 注意力：`O = softmax(QKᵀ/√d)·V`。
//!
//! 同一个形状（`s` 条序列、头宽 `d`）下给出两套实现，互相对拍：
//!
//! - `reference` / `forward`：朴素三阶段——先算全部分数 `QKᵀ/√d`，再对每行 softmax，最后乘 `V`。
//!   它要一块 `s×s` 的完整分数矩阵，是数值参考；
//! - [`flash_reference`] / [`flash_forward`]：online softmax 分块，维护 running max 与归一化和，
//!   单遍扫描 `K/V`，不落地完整分数矩阵。两套结果必须逐元素相等（误差内）。
//!
//! 样本、形状与代价两算子共用；`#[annotation::model]` 按注册名挑对应的参考函数。

use crate::model::{AxisRole, AxisSpec, Buffer, Cost, Sample, ShapeContext, Values};
use diagnostic::{Code, Error};

/// 两条基本轴：序列轴 `s`、头宽轴 `d`，都既是输入也是输出。
pub const AXES: &[AxisSpec] = &[
    AxisSpec {
        role: AxisRole::Input,
        label: 's',
    },
    AxisSpec {
        role: AxisRole::Output,
        label: 'd',
    },
];

/// `(m, n)` 即 `(序列长 s, 头宽 d)`。
///
/// # Errors
///
/// 当前实现不会失败。
pub fn project_shape(s: usize, d: usize) -> Result<ShapeContext, Error> {
    ShapeContext::new([('s', s), ('d', d)])
}

/// 标量代价：分数矩阵 `s²` 个点（每个 d 维内积）+ softmax + 对 `V` 的加权。
///
/// # Errors
///
/// 缺轴时返回说明。
pub fn cost(shape: &ShapeContext) -> Result<Cost, Error> {
    let s = shape
        .get('s')
        .ok_or_else(|| Error::new(Code::Invalid, "attention 形状缺轴 `s`"))?;
    let d = shape
        .get('d')
        .ok_or_else(|| Error::new(Code::Invalid, "attention 形状缺轴 `d`"))?;
    Ok(Cost {
        flops: (s as f64) * (s as f64) * ((d as f64) * 2.0 + 3.0),
        read: (s as u64) * (d as u64) * 4 * 3,
        write: (s as u64) * (d as u64) * 4,
    })
}

/// 确定性样本：`q`/`k`/`v` 三个输入与输出 `o`，均为 `s*d`。
pub fn sample(s: usize, d: usize) -> Sample {
    let det = |base: f32, step: f32| move |index: usize| base + (index as f32) * step;
    let q = det(-0.5, 0.13);
    let k = det(0.4, -0.09);
    let v = det(-0.2, 0.11);
    let len = s * d;
    Sample {
        m: s,
        n: d,
        scalars: vec![("S", s as i64), ("D", d as i64)],
        buffers: vec![
            Buffer {
                name: "q",
                values: Values::F32((0..len).map(q).collect()),
            },
            Buffer {
                name: "k",
                values: Values::F32((0..len).map(k).collect()),
            },
            Buffer {
                name: "v",
                values: Values::F32((0..len).map(v).collect()),
            },
            Buffer {
                name: "y",
                values: Values::F32(vec![0.0; len]),
            },
        ],
    }
}

fn sqrt_d(sample: &Sample) -> f32 {
    (sample.n as f32).sqrt()
}

/// 三个输入缓冲：朴素与 flash 路径读同一份 `q/k/v`，缺失时的文案因此只写一次。
fn qkv(sample: &Sample) -> diagnostic::Result<(&[f32], &[f32], &[f32])> {
    Ok((sample.f32("q")?, sample.f32("k")?, sample.f32("v")?))
}

/// 行 `row` 与行 `col` 的点积：四个朴素/flash 实现共用，免得每处重写 `d` 长的累加循环。
fn dot(x: &[f32], y: &[f32], row: usize, col: usize, d: usize) -> f32 {
    let mut sum = 0.0f32;
    for h in 0..d {
        sum += x[row * d + h] * y[col * d + h];
    }
    sum
}

/// 四个缓冲长度都等于 `s*d` 才放行：朴素与 flash 内核共用这条校验。
fn check_len(q: &[f32], k: &[f32], v: &[f32], out: &[f32], len: usize) -> diagnostic::Result<()> {
    if q.len() != len || k.len() != len || v.len() != len || out.len() != len {
        return Err(Error::new(Code::Invalid, "缓冲长度与 s*d 不一致"));
    }
    Ok(())
}

/// 注意力的一行机器：三个操作数与行几何（`s`、`d`、`√d`），四个入口共用同一份算式。
struct Rows<'a> {
    q: &'a [f32],
    k: &'a [f32],
    v: &'a [f32],
    s: usize,
    d: usize,
    scale: f32,
}

impl<'a> Rows<'a> {
    /// 从样本取三个操作数。
    fn of(sample: &'a Sample) -> diagnostic::Result<Self> {
        let (q, k, v) = qkv(sample)?;
        Ok(Self {
            q,
            k,
            v,
            s: sample.m,
            d: sample.n,
            scale: sqrt_d(sample),
        })
    }

    /// 朴素路径的一行：分数 p[j] = dot(q[row], k[j]) / √d → 数值稳定 softmax → 对 `V` 加权。
    fn naive(&self, row: usize, out: &mut [f32]) {
        let scores: Vec<f32> = (0..self.s)
            .map(|col| dot(self.q, self.k, row, col, self.d) / self.scale)
            .collect();
        let max = scores.iter().copied().fold(f32::MIN, f32::max);
        let exps: Vec<f32> = scores.iter().map(|score| (score - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        for h in 0..self.d {
            let mut acc = 0.0f32;
            for j in 0..self.s {
                acc += (exps[j] / sum) * self.v[j * self.d + h];
            }
            out[row * self.d + h] = acc;
        }
    }

    /// online（flash）路径的一行：跑一遍 `j` 维护最大值、归一化分母与累加向量，返回 `(acc, sum)`。
    fn online(&self, row: usize) -> (Vec<f32>, f32) {
        let mut max = f32::MIN;
        let mut sum = 0.0f32;
        let mut acc = vec![0.0f32; self.d];
        for col in 0..self.s {
            let score = dot(self.q, self.k, row, col, self.d) / self.scale;
            let new_max = max.max(score);
            let reweight = (max - new_max).exp();
            let prob = (score - new_max).exp();
            sum = sum * reweight + prob;
            for h in 0..self.d {
                acc[h] = acc[h] * reweight + prob * self.v[col * self.d + h];
            }
            max = new_max;
        }
        (acc, sum)
    }
}

/// 朴素参考：落地完整分数矩阵，逐行 softmax，再乘 `V`。
pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
    let rows = Rows::of(sample)?;
    let mut out = vec![0.0f32; rows.s * rows.d];
    for row in 0..rows.s {
        rows.naive(row, &mut out);
    }
    Ok(out)
}

/// Flash 参考：online softmax 分块（这里块取整条序列，语义与分块等价），单遍 K/V。
pub fn flash_reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
    let rows = Rows::of(sample)?;
    let mut out = vec![0.0f32; rows.s * rows.d];
    for row in 0..rows.s {
        let (acc, sum) = rows.online(row);
        for h in 0..rows.d {
            out[row * rows.d + h] = acc[h] / sum;
        }
    }
    Ok(out)
}

/// 朴素内核本体：结果写进 `out`。
///
/// # Errors
///
/// 长度不符时返回说明。
pub fn forward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    s: usize,
    d: usize,
) -> diagnostic::Result<()> {
    check_len(q, k, v, out, s * d)?;
    let rows = Rows {
        q,
        k,
        v,
        s,
        d,
        scale: (d as f32).sqrt(),
    };
    for row in 0..s {
        rows.naive(row, out);
    }
    Ok(())
}

/// Flash 内核本体：online softmax，单遍 K/V，写进 `out`。
///
/// # Errors
///
/// 长度不符时返回说明。
pub fn flash_forward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    s: usize,
    d: usize,
) -> diagnostic::Result<()> {
    check_len(q, k, v, out, s * d)?;
    let rows = Rows {
        q,
        k,
        v,
        s,
        d,
        scale: (d as f32).sqrt(),
    };
    for row in 0..s {
        let (acc, sum) = rows.online(row);
        for h in 0..d {
            out[row * d + h] = acc[h] / sum;
        }
    }
    Ok(())
}

/// 标记类型：承载 `#[model]` 语义。
pub struct Attention;

/// 朴素注意力语义别名。
#[annotation::model(name = "attention_f32", shape = [8, 8], dtype = "f32", skeleton = "attention")]
pub type AttentionF32 = Attention;

#[cfg(test)]
#[path = "../tests/attention_tests.rs"]
mod tests;
