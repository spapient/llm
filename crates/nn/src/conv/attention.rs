//! 注意力：`O = softmax(QKᵀ/√d)·V`。

use crate::model::{AxisRole, AxisSpec, Buf, Cost, Sample, ShapeCtx, Values};
use diagnostic::{Code, Error};

pub mod tiled;
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
pub fn project_shape(s: usize, d: usize) -> Result<ShapeCtx, Error> {
    ShapeCtx::new([('s', s), ('d', d)])
}

/// 取形状轴；缺失时按模块前缀与轴名报错，避免各调用点各写一遍分支。
fn axis(shape: &ShapeCtx, what: &str, label: char) -> Result<usize, Error> {
    shape
        .get(label)
        .ok_or_else(|| Error::new(Code::Invalid, format!("{what} 形状缺轴 `{label}`")))
}

/// 标量代价：分数矩阵 `s²` 个点（每个 d 维内积）+ softmax + 对 `V` 的加权。
///
/// # Errors
///
/// 缺轴时返回说明。
pub fn cost(shape: &ShapeCtx) -> Result<Cost, Error> {
    let s = axis(shape, "attention", 's')?;
    let d = axis(shape, "attention", 'd')?;
    Ok(Cost {
        flops: (s as f64) * (s as f64) * ((d as f64) * 2.0 + 3.0),
        read: (s as u64) * (d as u64) * 4 * 3,
        write: (s as u64) * (d as u64) * 4,
    })
}

/// 确定性 q/k/v/y 缓冲：`q` 与 `y` 取 `q_len`，`k`/`v` 取 `kv_len`。
fn sample_bufs(q_len: usize, kv_len: usize) -> Vec<Buf> {
    let det = |base: f32, step: f32, len: usize| -> Vec<f32> {
        (0..len).map(|idx| base + idx as f32 * step).collect()
    };
    let buf = |name: &'static str, values: Vec<f32>| Buf {
        name,
        values: Values::F32(values),
    };
    vec![
        buf("q", det(-0.5, 0.13, q_len)),
        buf("k", det(0.4, -0.09, kv_len)),
        buf("v", det(-0.2, 0.11, kv_len)),
        buf("y", vec![0.0; q_len]),
    ]
}

/// 确定性样本：`q`/`k`/`v` 三个输入与输出 `o`，均为 `s*d`。
pub fn sample(s: usize, d: usize) -> Sample {
    let len = s * d;
    Sample {
        m: s,
        n: d,
        scalars: vec![("S", s as i64), ("D", d as i64)],
        bufs: sample_bufs(len, len),
    }
}

/// 三个输入缓冲：朴素与 flash 路径读同一份 `q/k/v`，缺失时的文案因此只写一次。
fn qkv(sample: &Sample) -> diagnostic::Result<(&[f32], &[f32], &[f32])> {
    Ok((sample.f32("q")?, sample.f32("k")?, sample.f32("v")?))
}

/// 行 `row` 与行 `col` 的点积：四个朴素/flash 实现共用，免得每处重写 `d` 长的累加循环。
fn dot(x: &[f32], y: &[f32], row: usize, col: usize, d: usize) -> f32 {
    crate::kernel::matmul::dot_lanes(&x[row * d..][..d], &y[col * d..][..d])
}

/// 头宽不能为 0：`√d` 作分数除数，`d=0` 会让每条分数除零成 NaN。
fn check_width(d: usize) -> diagnostic::Result<()> {
    if d == 0 {
        return Err(Error::new(Code::Invalid, "注意力头宽不能为 0"));
    }
    Ok(())
}

/// 注意力的一行机器：三个操作数与 K/V 列几何（`cols`、读宽 `d`、写宽 `write`、`√d`）——
/// 稠密注意力与潜空间注意力共用同一份算式，差只差在写宽可以比读宽窄。
pub(crate) struct Rows<'a> {
    q: &'a [f32],
    k: &'a [f32],
    v: &'a [f32],
    cols: usize,
    d: usize,
    write: usize,
    scale: f32,
}

impl<'a> Rows<'a> {
    /// 以操作数与 K/V 列几何建行机器：`scale = √d`。
    fn new(q: &'a [f32], k: &'a [f32], v: &'a [f32], cols: usize, d: usize) -> Self {
        Self {
            q,
            k,
            v,
            cols,
            d,
            write: d,
            scale: (d as f32).sqrt(),
        }
    }

    /// 窄写版本：读宽 `d`、写宽 `write`（`v` 的列宽与输出列宽都是 `write`）。
    fn narrow(
        q: &'a [f32],
        k: &'a [f32],
        v: &'a [f32],
        cols: usize,
        d: usize,
        write: usize,
    ) -> Self {
        Self {
            write,
            ..Self::new(q, k, v, cols, d)
        }
    }

    /// 从样本取三个操作数。
    fn of(sample: &'a Sample) -> diagnostic::Result<Self> {
        let (q, k, v) = qkv(sample)?;
        check_width(sample.n)?;
        Ok(Self::new(q, k, v, sample.m, sample.n))
    }

    /// 朴素路径的一行：分数 p[j] = dot(q[row], k[j]) / √d → 数值稳定 softmax → 对 `V` 加权。
    fn naive(&self, row: usize, out: &mut [f32]) {
        let scores: Vec<f32> = (0..self.cols)
            .map(|col| dot(self.q, self.k, row, col, self.d) / self.scale)
            .collect();
        let max = scores.iter().copied().fold(f32::MIN, f32::max);
        let exps: Vec<f32> = scores.iter().map(|score| (score - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        // `V` 按 `col` 分块，每块是 `write` 个输出列；按块求和让行列关系写在类型系统里，
        // 而不是靠 `j * write + h` 手算偏移。
        let targets = &mut out[row * self.write..][..self.write];
        for (h, slot) in targets.iter_mut().enumerate() {
            *slot = exps
                .iter()
                .zip(self.v.chunks_exact(self.write))
                .map(|(prob, vrow)| (prob / sum) * vrow[h])
                .sum();
        }
    }

    /// online（flash）路径的一行：跑一遍 `col` 维护最大值、归一化分母与累加向量。
    fn online(&self, row: usize, acc: &mut [f32]) -> f32 {
        let mut max = f32::MIN;
        let mut sum = 0.0f32;
        acc.fill(0.0);
        for col in 0..self.cols {
            let score = dot(self.q, self.k, row, col, self.d) / self.scale;
            let new_max = max.max(score);
            let reweight = (max - new_max).exp();
            let prob = (score - new_max).exp();
            sum = sum * reweight + prob;
            let vrow = &self.v[col * self.write..][..self.write];
            for (slot, value) in acc.iter_mut().zip(vrow) {
                *slot = *slot * reweight + prob * value;
            }
            max = new_max;
        }
        sum
    }
}

/// 一行机器两种实现共用的执行开关：朴素直接写 `out`，online 先累加再除归一化和。
#[derive(Clone, Copy)]
pub(crate) enum Mode {
    /// 三阶段朴素：落地全部分数再 softmax。
    Naive,
    /// online softmax：单遍 K/V，不落地分数矩阵。
    Online,
}

/// 按执行模式跑 `rows_count` 行，结果写进 `out`。
pub(crate) fn run_rows(rows: &Rows<'_>, rows_count: usize, out: &mut [f32], mode: Mode) {
    match mode {
        Mode::Naive => {
            for row in 0..rows_count {
                rows.naive(row, out);
            }
        }
        Mode::Online => {
            // 预分配一条累加缓冲，所有行复用；online 每行 fill 清零后写回。
            // 消除热路径上「每行一次堆分配」，prefill（rows_count = s）收益随序列长增长。
            let mut acc = vec![0.0f32; rows.write];
            for row in 0..rows_count {
                let sum = rows.online(row, &mut acc);
                for h in 0..rows.write {
                    out[row * rows.write + h] = acc[h] / sum;
                }
            }
        }
    }
}

/// 朴素参考：落地完整分数矩阵，逐行 softmax，再乘 `V`。
pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
    let rows = Rows::of(sample)?;
    let mut out = vec![0.0f32; rows.cols * rows.d];
    run_rows(&rows, rows.cols, &mut out, Mode::Naive);
    Ok(out)
}

/// Flash 参考：online softmax 分块（这里块取整条序列，语义与分块等价），单遍 K/V。
pub fn flash_reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
    let rows = Rows::of(sample)?;
    let mut out = vec![0.0f32; rows.cols * rows.d];
    run_rows(&rows, rows.cols, &mut out, Mode::Online);
    Ok(out)
}

/// 朴素内核本体：prefill 自注意力是 `decode` 的 `q_rows = hist = s` 特例，结果写进 `out`。
///
/// # Errors
///
/// 长度不符、头宽为 0，或序列长为 0 时返回说明。
pub fn forward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    s: usize,
    d: usize,
) -> diagnostic::Result<()> {
    decode(q, k, v, out, s, s, d, Mode::Naive)
}

/// Flash 内核本体：online softmax，单遍 K/V，写进 `out`。prefill 口径同 [`forward`]。
///
/// # Errors
///
/// 长度不符、头宽为 0，或序列长为 0 时返回说明。
pub fn flash_forward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    s: usize,
    d: usize,
) -> diagnostic::Result<()> {
    decode(q, k, v, out, s, s, d, Mode::Online)
}

mod decode;
pub mod latent;
pub use decode::{
    decode_flash_forward, decode_flash_forward_batch, decode_flash_forward_batch_var,
    decode_flash_forward_heads, decode_forward, decode_forward_batch, decode_forward_batch_var,
    decode_forward_heads,
};
pub use latent::{LatentAttention, LatentAttentionF32, forward as latent_forward};

pub(crate) use decode::check_decode_len;
use decode::decode;

// 语义表对外的名字保持 `attention::decode_attention`（`nn::conv::decode_attention` 由 lib.rs 再导出）。
pub use decode::semantics as decode_attention;

/// 标记类型：承载 `#[model]` 语义。
pub struct Attention;

/// 朴素注意力语义别名。
#[annotation::model(name = "attention_f32", shape = [8, 8], dtype = "f32", skeleton = "attention")]
pub type AttentionF32 = Attention;

/// Flash 注意力标记类型：形状、样本、代价与朴素 [`Attention`] 完全共用，只把标量参考换成
/// online softmax 版本（[`flash_reference`]）。`const = "FLASH_MODEL"` 让两个算子同模块各持一份语义表。
pub struct FlashAttention;

/// Flash 注意力语义别名；codegen 按注册名选 [`flash_reference`]。
#[annotation::model(name = "flash_attention_f32", shape = [8, 8], dtype = "f32", skeleton = "attention", const = "FLASH_MODEL")]
pub type FlashAttentionF32 = FlashAttention;

// 基座契约（ADR 0050）：两条算子共用形状/代价/样本，只有参考入口分叉。
crate::operator_here!(Attention, "attention_f32", 3, (8, 8), true, reference);
crate::operator_here!(
    FlashAttention,
    "flash_attention_f32",
    3,
    (8, 8),
    true,
    flash_reference
);

#[cfg(test)]
#[path = "../../tests/conv/latent_attention_tests.rs"]
mod latent_attention_tests;
#[cfg(test)]
#[path = "../../tests/conv/attention_tests.rs"]
mod tests;
