//! decode 注意力语义：query 行数与 K/V 历史行数解耦。
//!
//! 典型 decode 一步 `q_rows = 1`，`hist` 是这条序列已经缓存的 K/V 行数，随生成增长。
//! 样本几何因此与 prefill [`crate::attention`] 不同：`q`/`y` 是 `q_rows·d`，`k`/`v` 是
//! `hist·d`，标量为 `Q_ROWS` / `HIST` / `D`。两套参考都复用 [`crate::attention`] 里已经
//! 校验过的 decode 算术，本模块只补形状上下文、确定性样本与代价。

use crate::model::{AxisRole, AxisSpec, Buffer, Cost, Sample, ShapeContext, Values};
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

/// 取形状轴；缺失时按轴名报错，避免三个调用点各写一遍分支。
fn axis(shape: &ShapeContext, label: char) -> Result<usize, Error> {
    shape.get(label).ok_or_else(|| {
        Error::new(
            Code::Invalid,
            format!("decode attention 形状缺轴 `{label}`"),
        )
    })
}

/// 标量代价：每行 query 扫 `hist` 条历史（每个 d 维内积）+ softmax + 对 `V` 加权。
///
/// # Errors
///
/// 缺轴时返回说明。
pub fn cost(shape: &ShapeContext) -> Result<Cost, Error> {
    let q = axis(shape, 'q')?;
    let d = axis(shape, 'd')?;
    let h = axis(shape, 'h')?;
    Ok(Cost {
        flops: (q as f64) * (h as f64) * ((d as f64) * 2.0 + 3.0),
        read: (q as u64) * (d as u64) * 4 + (h as u64) * (d as u64) * 4 * 2,
        write: (q as u64) * (d as u64) * 4,
    })
}

/// 确定性样本：`q` 与输出 `y` 为 `q_rows·d`，`k`/`v` 为 `hist·d`。
pub fn sample(q_rows: usize, d: usize) -> Sample {
    let hist = HIST;
    let det = |base: f32, step: f32| move |index: usize| base + (index as f32) * step;
    let q = det(-0.5, 0.13);
    let k = det(0.4, -0.09);
    let v = det(-0.2, 0.11);
    let q_len = q_rows * d;
    let kv_len = hist * d;
    Sample {
        m: q_rows,
        n: d,
        scalars: vec![
            ("Q_ROWS", q_rows as i64),
            ("HIST", hist as i64),
            ("D", d as i64),
        ],
        buffers: vec![
            Buffer {
                name: "q",
                values: Values::F32((0..q_len).map(q).collect()),
            },
            Buffer {
                name: "k",
                values: Values::F32((0..kv_len).map(k).collect()),
            },
            Buffer {
                name: "v",
                values: Values::F32((0..kv_len).map(v).collect()),
            },
            Buffer {
                name: "y",
                values: Values::F32(vec![0.0; q_len]),
            },
        ],
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

/// 样本三元 `q/k/v` 切片；两条参考实现共用。
fn qkv(sample: &Sample) -> diagnostic::Result<(&[f32], &[f32], &[f32])> {
    Ok((sample.f32("q")?, sample.f32("k")?, sample.f32("v")?))
}

/// 朴素参考：query 对历史 K/V 做一遍数值稳定 softmax，再对 `V` 加权。
pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
    let (q_rows, hist, d) = geometry(sample)?;
    let (q, k, v) = qkv(sample)?;
    let mut out = vec![0.0; q_rows * d];
    crate::attention::decode_forward(q, k, v, &mut out, q_rows, hist, d)?;
    Ok(out)
}

/// Flash 参考：online softmax 单遍扫历史 K/V；flash 别名模块的参考入口指向这里。
pub fn decode_flash_reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
    let (q_rows, hist, d) = geometry(sample)?;
    let (q, k, v) = qkv(sample)?;
    let mut out = vec![0.0; q_rows * d];
    crate::attention::decode_flash_forward(q, k, v, &mut out, q_rows, hist, d)?;
    Ok(out)
}

/// 标记类型：承载 `#[model]` 语义。
pub struct DecodeAttention;

/// 朴素 decode 注意力语义别名。
#[annotation::model(name = "decode_attention_f32", shape = [1, 8], dtype = "f32", skeleton = "attention")]
pub type DecodeAttentionF32 = DecodeAttention;
