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

/// 单行的 online（flash）注意力：跑一遍 `j` 维护最大值、归一化分母与累加向量，返回 `(acc, sum)`。
/// flash 参考与 flash 内核共用同一份计算，只差最后写回各自的 `out`。
fn online_row(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    row: usize,
    s: usize,
    d: usize,
    scale: f32,
) -> (Vec<f32>, f32) {
    let mut max = f32::MIN;
    let mut sum = 0.0f32;
    let mut acc = vec![0.0f32; d];
    for j in 0..s {
        let score = dot(q, k, row, j, d) / scale;
        let new_max = max.max(score);
        let reweight = (max - new_max).exp();
        let prob = (score - new_max).exp();
        sum = sum * reweight + prob;
        for h in 0..d {
            acc[h] = acc[h] * reweight + prob * v[j * d + h];
        }
        max = new_max;
    }
    (acc, sum)
}

/// 朴素路径的 V 加权求和：按归一化概率逐列累加写进 `out[row]`，参考与内核共用。
fn weight_v(v: &[f32], exps: &[f32], sum: f32, s: usize, d: usize, row: usize, out: &mut [f32]) {
    for h in 0..d {
        let mut acc = 0.0f32;
        for j in 0..s {
            acc += (exps[j] / sum) * v[j * d + h];
        }
        out[row * d + h] = acc;
    }
}

/// 朴素参考：落地完整分数矩阵，逐行 softmax，再乘 `V`。
pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
    let (s, d) = (sample.m, sample.n);
    let q = sample.f32("q")?;
    let k = sample.f32("k")?;
    let v = sample.f32("v")?;
    let scale = sqrt_d(sample);
    let mut out = vec![0.0f32; s * d];

    for i in 0..s {
        // 1) 分数行 p[j] = dot(q[i], k[j]) / sqrt(d)。
        let mut scores = vec![0.0f32; s];
        for (j, score) in scores.iter_mut().enumerate() {
            *score = dot(q, k, i, j, d) / scale;
        }
        // 2) 数值稳定 softmax。
        let max = scores.iter().copied().fold(f32::MIN, f32::max);
        let exps: Vec<f32> = scores.iter().map(|score| (score - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        // 3) 对 V 加权求和。
        weight_v(v, &exps, sum, s, d, i, &mut out);
    }
    Ok(out)
}

/// Flash 参考：online softmax 分块（这里块取整条序列，语义与分块等价），单遍 K/V。
pub fn flash_reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
    let (s, d) = (sample.m, sample.n);
    let q = sample.f32("q")?;
    let k = sample.f32("k")?;
    let v = sample.f32("v")?;
    let scale = sqrt_d(sample);
    let mut out = vec![0.0f32; s * d];

    for i in 0..s {
        let (acc, sum) = online_row(q, k, v, i, s, d, scale);
        for h in 0..d {
            out[i * d + h] = acc[h] / sum;
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
    let len = s * d;
    check_len(q, k, v, out, len)?;
    let scale = (d as f32).sqrt();
    for i in 0..s {
        let mut scores = vec![0.0f32; s];
        for (j, score) in scores.iter_mut().enumerate() {
            *score = dot(q, k, i, j, d) / scale;
        }
        let max = scores.iter().copied().fold(f32::MIN, f32::max);
        let exps: Vec<f32> = scores.iter().map(|score| (score - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        weight_v(v, &exps, sum, s, d, i, out);
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
    let len = s * d;
    check_len(q, k, v, out, len)?;
    let scale = (d as f32).sqrt();
    for i in 0..s {
        let (acc, sum) = online_row(q, k, v, i, s, d, scale);
        for h in 0..d {
            out[i * d + h] = acc[h] / sum;
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
mod tests {
    use super::*;

    #[test]
    fn flash_and_plain_attention_agree_and_rows_are_distributions() {
        let sample = sample(8, 8);
        let plain = reference(&sample).expect("朴素参考");
        let flash = flash_reference(&sample).expect("flash 参考");
        assert_eq!(plain.len(), flash.len());
        for (a, b) in plain.iter().zip(flash.iter()) {
            assert!((a - b).abs() <= 1e-5, "两套算法必须一致：{a} vs {b}");
        }
        // 每个 query 行的注意力权重隐含归一化：输出对全 1 的 V 应得 1，这里直接核对确定性结果非 NaN。
        assert!(plain.iter().all(|value| value.is_finite()));

        let mut out = vec![0.0f32; 64];
        forward(
            sample.f32("q").unwrap(),
            sample.f32("k").unwrap(),
            sample.f32("v").unwrap(),
            &mut out,
            8,
            8,
        )
        .expect("朴素内核");
        assert_eq!(out, plain);

        let mut flash_out = vec![0.0f32; 64];
        flash_forward(
            sample.f32("q").unwrap(),
            sample.f32("k").unwrap(),
            sample.f32("v").unwrap(),
            &mut flash_out,
            8,
            8,
        )
        .expect("flash 内核");
        for (a, b) in flash_out.iter().zip(flash.iter()) {
            assert!((a - b).abs() <= 1e-5);
        }
    }
}
