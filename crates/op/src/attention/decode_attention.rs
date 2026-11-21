//! decode 注意力语义：query 行数与 K/V 历史行数解耦。
//!
//! 典型 decode 一步 `q_rows = 1`，`hist` 是这条序列已经缓存的 K/V 行数，随生成增长。
//! 样本几何因此与 prefill [`crate::attention`] 不同：`q`/`y` 是 `q_rows·d`，`k`/`v` 是
//! `hist·d`，标量为 `Q_ROWS` / `HIST` / `D`。两套参考都复用 [`crate::attention`] 里已经
//! 校验过的 decode 算术，本模块只补形状上下文、确定性样本与代价。

use crate::model::{AxisRole, AxisSpec, Cost, Sample, ShapeContext};
use diagnostic::{Code, Error};

/// 样本固定历史长度：真实 `hist` 由执行侧按已缓存页数推出，语义表样本用一份确定长度。
const HIST: usize = 4;

/// 两条基本轴：query 行轴 `q`、头宽轴 `d`；历史行轴 `h` 是样本派生轴。
pub const AXES: &[AxisSpec] = &[
    AxisSpec {
        role: AxisRole::Input,
        label: 'q',
    },
    AxisSpec {
        role: AxisRole::Output,
        label: 'd',
    },
];

/// `(m, n)` 即 `(q_rows, d)`；历史长度固定进形状上下文作为派生轴。
///
/// # Errors
///
/// 轴长为 0 或字母重复时由 [`ShapeContext::new`] 返回说明。
pub fn project_shape(q_rows: usize, d: usize) -> Result<ShapeContext, Error> {
    ShapeContext::new([('q', q_rows), ('d', d), ('h', HIST)])
}

/// 标量代价：每行 query 扫 `hist` 条历史（每个 d 维内积）+ softmax + 对 `V` 加权。
///
/// # Errors
///
/// 缺轴时返回说明。
pub fn cost(shape: &ShapeContext) -> Result<Cost, Error> {
    let q = super::axis(shape, "decode attention", 'q')?;
    let d = super::axis(shape, "decode attention", 'd')?;
    let h = super::axis(shape, "decode attention", 'h')?;
    Ok(Cost {
        flops: (q as f64) * (h as f64) * ((d as f64) * 2.0 + 3.0),
        read: (q as u64) * (d as u64) * 4 + (h as u64) * (d as u64) * 4 * 2,
        write: (q as u64) * (d as u64) * 4,
    })
}

/// 确定性样本：`q` 与输出 `y` 为 `q_rows·d`，`k`/`v` 为 `hist·d`。
pub fn sample(q_rows: usize, d: usize) -> Sample {
    let hist = HIST;
    Sample {
        m: q_rows,
        n: d,
        scalars: vec![
            ("Q_ROWS", q_rows as i64),
            ("HIST", hist as i64),
            ("D", d as i64),
        ],
        buffers: super::sample_buffers(q_rows * d, hist * d),
    }
}

/// 取样本标量并转行数；缺标量时按名字报错。
fn scalar_or(sample: &Sample, name: &str) -> Result<usize, Error> {
    sample
        .scalar(name)
        .ok_or_else(|| Error::new(Code::Invalid, format!("decode 样本缺标量 `{name}`")))
        .map(|value| value as usize)
}

/// 从样本读出 decode 三元几何 `(q_rows, hist, d)`。
fn geometry(sample: &Sample) -> diagnostic::Result<(usize, usize, usize)> {
    let q_rows = scalar_or(sample, "Q_ROWS")?;
    let hist = scalar_or(sample, "HIST")?;
    let d = scalar_or(sample, "D")?;
    Ok((q_rows, hist, d))
}

/// 朴素参考：query 对历史 K/V 做一遍数值稳定 softmax，再对 `V` 加权。
pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
    let (q_rows, hist, d) = geometry(sample)?;
    let (q, k, v) = super::qkv(sample)?;
    let mut out = vec![0.0; q_rows * d];
    crate::attention::decode_forward(q, k, v, &mut out, q_rows, hist, d)?;
    Ok(out)
}

/// Flash 参考：online softmax 单遍扫历史 K/V；flash 别名模块的参考入口指向这里。
pub fn decode_flash_reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
    let (q_rows, hist, d) = geometry(sample)?;
    let (q, k, v) = super::qkv(sample)?;
    let mut out = vec![0.0; q_rows * d];
    crate::attention::decode_flash_forward(q, k, v, &mut out, q_rows, hist, d)?;
    Ok(out)
}

/// 标记类型：承载 `#[model]` 语义。
pub struct DecodeAttention;

/// 朴素 decode 注意力语义别名。
#[annotation::model(name = "decode_attention_f32", shape = [1, 8], dtype = "f32", skeleton = "attention")]
pub type DecodeAttentionF32 = DecodeAttention;

/// Flash decode 注意力标记类型：形状、样本、代价与朴素 [`DecodeAttention`] 共用，只把标量参考
/// 换成 online softmax 版本（[`decode_flash_reference`]）。`const = "DECODE_FLASH_MODEL"` 让
/// 两个算子同模块各持一份语义表。
pub struct DecodeFlashAttention;

/// Flash decode 注意力语义别名；codegen 按注册名选 [`decode_flash_reference`]。
#[annotation::model(name = "decode_flash_attention_f32", shape = [1, 8], dtype = "f32", skeleton = "attention", const = "DECODE_FLASH_MODEL")]
pub type DecodeFlashAttentionF32 = DecodeFlashAttention;

// 基座契约（ADR 0050）：同上一对，只把参考入口换成 flash 版本。
crate::operator_here!(
    DecodeAttention,
    "decode_attention_f32",
    3,
    (1, 8),
    true,
    reference
);
crate::operator_here!(
    DecodeFlashAttention,
    "decode_flash_attention_f32",
    3,
    (1, 8),
    true,
    decode_flash_reference
);
