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

/// 头宽不能为 0：`√d` 作分数除数，`d=0` 会让每条分数除零成 NaN。
fn check_width(d: usize) -> diagnostic::Result<()> {
    if d == 0 {
        return Err(Error::new(Code::Invalid, "注意力头宽不能为 0"));
    }
    Ok(())
}

/// 注意力的一行机器：三个操作数与 K/V 列几何（`cols`、`d`、`√d`），十个入口共用同一份算式。
///
/// `cols` 是 K/V 的列数：自注意力时等于序列长 `s`，decode 时等于缓存历史 `hist`。查询行数
/// 由各入口的循环上界单独给出，不落进这里。
struct Rows<'a> {
    q: &'a [f32],
    k: &'a [f32],
    v: &'a [f32],
    cols: usize,
    d: usize,
    scale: f32,
}

impl<'a> Rows<'a> {
    /// 从样本取三个操作数。
    fn of(sample: &'a Sample) -> diagnostic::Result<Self> {
        let (q, k, v) = qkv(sample)?;
        check_width(sample.n)?;
        Ok(Self {
            q,
            k,
            v,
            cols: sample.m,
            d: sample.n,
            scale: sqrt_d(sample),
        })
    }

    /// 朴素路径的一行：分数 p[j] = dot(q[row], k[j]) / √d → 数值稳定 softmax → 对 `V` 加权。
    fn naive(&self, row: usize, out: &mut [f32]) {
        let scores: Vec<f32> = (0..self.cols)
            .map(|col| dot(self.q, self.k, row, col, self.d) / self.scale)
            .collect();
        let max = scores.iter().copied().fold(f32::MIN, f32::max);
        let exps: Vec<f32> = scores.iter().map(|score| (score - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        for h in 0..self.d {
            let mut acc = 0.0f32;
            for j in 0..self.cols {
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
        for col in 0..self.cols {
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

/// 一行机器两种实现共用的执行开关：朴素直接写 `out`，online 先累加再除归一化和。
#[derive(Clone, Copy)]
enum Mode {
    /// 三阶段朴素：落地全部分数再 softmax。
    Naive,
    /// online softmax：单遍 K/V，不落地分数矩阵。
    Online,
}

/// 按执行模式跑 `rows_count` 行，结果写进 `out`。
fn run_rows(rows: &Rows<'_>, rows_count: usize, out: &mut [f32], mode: Mode) {
    match mode {
        Mode::Naive => {
            for row in 0..rows_count {
                rows.naive(row, out);
            }
        }
        Mode::Online => {
            for row in 0..rows_count {
                let (acc, sum) = rows.online(row);
                for h in 0..rows.d {
                    out[row * rows.d + h] = acc[h] / sum;
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

/// 朴素内核本体：prefill 自注意力是 [`decode`] 的 `q_rows = hist = s` 特例，结果写进 `out`。
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

/// decode 缓冲长度校验：`q`/`out` 是 `q_rows·d`，`k`/`v` 是 `hist·d`；两边行数可以不同。
/// decode 查询与历史行数都必须非零。
fn check_rows(q_rows: usize, hist: usize) -> diagnostic::Result<()> {
    if q_rows == 0 || hist == 0 {
        return Err(Error::new(
            Code::Invalid,
            format!("decode 查询行数 {q_rows} 与历史行数 {hist} 都必须非零"),
        ));
    }
    Ok(())
}

/// 一对切片都必须等于预期长度。
fn check_pair(first: &[f32], second: &[f32], expect: usize, what: &str) -> diagnostic::Result<()> {
    if first.len() != expect || second.len() != expect {
        return Err(Error::new(
            Code::Invalid,
            format!("{what} 长度与预期 {expect} 不一致"),
        ));
    }
    Ok(())
}

fn check_decode_len(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &[f32],
    q_rows: usize,
    hist: usize,
    d: usize,
) -> diagnostic::Result<()> {
    check_width(d)?;
    check_rows(q_rows, hist)?;
    let q_len = q_rows * d;
    let kv_len = hist * d;
    check_pair(q, out, q_len, "q/out")?;
    check_pair(k, v, kv_len, "k/v")?;
    Ok(())
}

/// decode 内核共享：`decode_forward` 与 `decode_flash_forward` 只差 softmax 策略（[`Mode`]）。
#[allow(clippy::too_many_arguments)]
fn decode(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    q_rows: usize,
    hist: usize,
    d: usize,
    mode: Mode,
) -> diagnostic::Result<()> {
    check_decode_len(q, k, v, out, q_rows, hist, d)?;
    let rows = Rows {
        q,
        k,
        v,
        cols: hist,
        d,
        scale: (d as f32).sqrt(),
    };
    run_rows(&rows, q_rows, out, mode);
    Ok(())
}

/// decode 朴素内核：`q`（`q_rows·d`）对历史 K/V（`hist·d`）做注意力，结果写进 `out`（`q_rows·d`）。
///
/// 与 [`forward`] 的 prefill 同长自注意力不同，本入口不要求 `q_rows == hist`；典型 decode
/// 一步 `q_rows = 1`，`hist` 随生成增长。
///
/// # Errors
///
/// 长度不符、头宽为 0，或查询/历史行数为 0 时返回说明。
pub fn decode_forward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    q_rows: usize,
    hist: usize,
    d: usize,
) -> diagnostic::Result<()> {
    decode(q, k, v, out, q_rows, hist, d, Mode::Naive)
}

/// decode flash 内核：online softmax 单遍扫历史 K/V，写进 `out`。形状口径同 [`decode_forward`]。
///
/// # Errors
///
/// 同 [`decode_forward`]。
pub fn decode_flash_forward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    q_rows: usize,
    hist: usize,
    d: usize,
) -> diagnostic::Result<()> {
    decode(q, k, v, out, q_rows, hist, d, Mode::Online)
}

/// 单头 decode 内核的函数指针：朴素与 flash 包装共用同一段头分段编排。
type SingleHead =
    fn(&[f32], &[f32], &[f32], &mut [f32], usize, usize, usize) -> diagnostic::Result<()>;

/// 多头 decode：头优先块布局——`q`/`out` 为 `heads · q_rows · d`，`k`/`v` 为 `heads · hist · d`，
/// 第 `h` 个头是一段连续的 `q_rows · d` / `hist · d`。头数由 `q` 长度按 `q_rows · d` 推出，
/// 头间互不耦合、每个头独立 softmax；头内算术复用传入的单头内核，这里只做头分段编排。
///
/// # Errors
///
/// `q` 不是 `q_rows · d` 的整数倍、缓冲长度不成头块倍数，或任一头段长度/宽度/行数不符时返回说明。
/// 校验多头块长度并派生头数：`q`/`out` 是 `heads·q_rows·d`，`k`/`v` 是 `heads·hist·d`；
/// 头内宽度与行数守卫交给逐段调用的单头内核。
fn headed_heads(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &[f32],
    q_rows: usize,
    hist: usize,
    d: usize,
) -> diagnostic::Result<usize> {
    let q_stride = q_rows * d;
    let kv_stride = hist * d;
    if q_stride == 0 || !q.len().is_multiple_of(q_stride) {
        return Err(Error::new(
            Code::Invalid,
            format!("q 长度 {} 不是 q_rows·d（{q_stride}）的整数倍", q.len()),
        ));
    }
    let heads = q.len() / q_stride;
    check_pair(q, out, heads * q_stride, "q/out")?;
    check_pair(k, v, heads * kv_stride, "k/v")?;
    Ok(heads)
}

#[allow(clippy::too_many_arguments)]
fn decode_heads(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    q_rows: usize,
    hist: usize,
    d: usize,
    single: SingleHead,
) -> diagnostic::Result<()> {
    let heads = headed_heads(q, k, v, out, q_rows, hist, d)?;
    let q_stride = q_rows * d;
    let kv_stride = hist * d;
    for head in 0..heads {
        let q_start = head * q_stride;
        let kv_start = head * kv_stride;
        let qh = &q[q_start..q_start + q_stride];
        let kh = &k[kv_start..kv_start + kv_stride];
        let vh = &v[kv_start..kv_start + kv_stride];
        let oh = &mut out[q_start..q_start + q_stride];
        single(qh, kh, vh, oh, q_rows, hist, d)?;
    }
    Ok(())
}

/// 多头朴素 decode 内核：头数由 `q` 长度派生，布局同 `decode_heads`，头内走 [`decode_forward`]。
///
/// # Errors
///
/// 同 `decode_heads`。
pub fn decode_forward_heads(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    q_rows: usize,
    hist: usize,
    d: usize,
) -> diagnostic::Result<()> {
    decode_heads(q, k, v, out, q_rows, hist, d, decode_forward)
}

/// 多头 flash decode 内核：头数由 `q` 长度派生，布局同 `decode_heads`，头内走 [`decode_flash_forward`]。
///
/// # Errors
///
/// 同 `decode_heads`。
pub fn decode_flash_forward_heads(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    q_rows: usize,
    hist: usize,
    d: usize,
) -> diagnostic::Result<()> {
    decode_heads(q, k, v, out, q_rows, hist, d, decode_flash_forward)
}

/// 批量校验：批内各序列等长历史（统一 `hist`）。`heads` 是显式几何参数，不按总长反推，
/// 避免批数不足时把头数压成 1 而放过错长缓冲；段内宽度/行数守卫交给 `decode_heads`。
#[allow(clippy::too_many_arguments)]
fn headed_batch(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &[f32],
    q_rows: usize,
    hist: usize,
    d: usize,
    batch: usize,
    heads: usize,
) -> diagnostic::Result<()> {
    if batch == 0 || heads == 0 {
        return Err(Error::new(
            Code::Invalid,
            format!("批量序列数 {batch} 与头数 {heads} 都必须非零"),
        ));
    }
    let q_seq = heads * q_rows * d;
    let kv_seq = heads * hist * d;
    check_batch_pairs(q, k, v, out, q_seq, kv_seq, batch)
}

/// 批块长度守卫：`q`/`out` 是 `batch·q_seq`，`k`/`v` 是 `batch·kv_seq`；宽度与查询/历史
/// 行数守卫交给逐段调用的 `decode_heads`。
#[allow(clippy::too_many_arguments)]
fn check_batch_pairs(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &[f32],
    q_seq: usize,
    kv_seq: usize,
    batch: usize,
) -> diagnostic::Result<()> {
    check_pair(q, out, batch * q_seq, "q/out")?;
    check_pair(k, v, batch * kv_seq, "k/v")?;
    Ok(())
}

/// 批量 decode：`q`/`out` 按批分段，第 `b` 段是 `heads·q_rows·d`，`k`/`v` 第 `b` 段是
/// `heads·hist·d`；批内段间独立 softmax、互不串段。段内复用 `decode_heads` 头分段编排，
/// 这里只做批切片。变长历史（各段独立 `hist_b`）不在本入口口径内。
#[allow(clippy::too_many_arguments)]
fn decode_batch(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    q_rows: usize,
    hist: usize,
    d: usize,
    heads: usize,
    batch: usize,
    single: SingleHead,
) -> diagnostic::Result<()> {
    headed_batch(q, k, v, out, q_rows, hist, d, batch, heads)?;
    let q_seq = heads * q_rows * d;
    let kv_seq = heads * hist * d;
    for seq in 0..batch {
        let q_start = seq * q_seq;
        let kv_start = seq * kv_seq;
        decode_heads(
            &q[q_start..q_start + q_seq],
            &k[kv_start..kv_start + kv_seq],
            &v[kv_start..kv_start + kv_seq],
            &mut out[q_start..q_start + q_seq],
            q_rows,
            hist,
            d,
            single,
        )?;
    }
    Ok(())
}

/// 批量朴素 decode 内核：段内走 [`decode_forward`]，布局同 `decode_batch`。
///
/// # Errors
///
/// 同 `decode_batch`。
#[allow(clippy::too_many_arguments)]
pub fn decode_forward_batch(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    q_rows: usize,
    hist: usize,
    d: usize,
    heads: usize,
    batch: usize,
) -> diagnostic::Result<()> {
    decode_batch(q, k, v, out, q_rows, hist, d, heads, batch, decode_forward)
}

/// 批量 flash decode 内核：段内走 [`decode_flash_forward`]，布局同 `decode_batch`。
///
/// # Errors
///
/// 同 `decode_batch`。
#[allow(clippy::too_many_arguments)]
pub fn decode_flash_forward_batch(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    q_rows: usize,
    hist: usize,
    d: usize,
    heads: usize,
    batch: usize,
) -> diagnostic::Result<()> {
    decode_batch(
        q,
        k,
        v,
        out,
        q_rows,
        hist,
        d,
        heads,
        batch,
        decode_flash_forward,
    )
}

/// 非零几何守卫：批数 / 头数 / 查询行数 / 头宽任一为零即拒绝。
fn batch_geometry_nonzero(
    batch: usize,
    heads: usize,
    q_rows: usize,
    d: usize,
) -> diagnostic::Result<()> {
    if batch == 0 || heads == 0 || q_rows == 0 || d == 0 {
        return Err(Error::new(
            Code::Invalid,
            format!("批量序列数 {batch}、头数 {heads}、查询行数 {q_rows}、头宽 {d} 都必须非零"),
        ));
    }
    Ok(())
}

/// 变长批量校验：`hist` 逐序列历史行数；`q`/`out` 等长段 `heads·q_rows·d`，`k`/`v` 总长
/// 是各段 `heads·hist_b·d` 之和。头数 / 批数显式传入，不从总长反推，避免批数不足时把头数压成 1。
#[allow(clippy::too_many_arguments)]
fn headed_batch_var(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &[f32],
    q_rows: usize,
    hist: &[usize],
    d: usize,
    heads: usize,
) -> diagnostic::Result<()> {
    let batch = hist.len();
    batch_geometry_nonzero(batch, heads, q_rows, d)?;
    let q_seq = heads * q_rows * d;
    check_pair(q, out, batch * q_seq, "q/out")?;
    let kv_total: usize = hist.iter().map(|h| heads * h * d).sum();
    check_pair(k, v, kv_total, "k/v")?;
    Ok(())
}

/// 变长批量 decode：`q`/`out` 等长段 `heads·q_rows·d`；`k`/`v` 第 `b` 段是
/// `heads·hist_b·d`，段间紧排、长度随 `hist[b]` 不同。段内复用 `decode_heads` 头分段编排，
/// 这里只按累计历史游标切批段；各段独立 softmax、互不串段。段内行数 / 宽度守卫交给 `decode_heads`。
#[allow(clippy::too_many_arguments)]
fn decode_batch_var(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    q_rows: usize,
    hist: &[usize],
    d: usize,
    heads: usize,
    single: SingleHead,
) -> diagnostic::Result<()> {
    headed_batch_var(q, k, v, out, q_rows, hist, d, heads)?;
    let q_seq = heads * q_rows * d;
    let mut kv_cursor = 0;
    for (seq, &hist_b) in hist.iter().enumerate() {
        let kv_seq = heads * hist_b * d;
        let q_start = seq * q_seq;
        decode_heads(
            &q[q_start..q_start + q_seq],
            &k[kv_cursor..kv_cursor + kv_seq],
            &v[kv_cursor..kv_cursor + kv_seq],
            &mut out[q_start..q_start + q_seq],
            q_rows,
            hist_b,
            d,
            single,
        )?;
        kv_cursor += kv_seq;
    }
    Ok(())
}

/// 变长批量朴素 decode：段内走 [`decode_forward`]，布局同 `decode_batch_var`。
///
/// # Errors
///
/// 同 `decode_batch_var`。
#[allow(clippy::too_many_arguments)]
pub fn decode_forward_batch_var(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    q_rows: usize,
    hist: &[usize],
    d: usize,
    heads: usize,
) -> diagnostic::Result<()> {
    decode_batch_var(q, k, v, out, q_rows, hist, d, heads, decode_forward)
}

/// 变长批量 flash decode：段内走 [`decode_flash_forward`]，布局同 `decode_batch_var`。
///
/// # Errors
///
/// 同 `decode_batch_var`。
#[allow(clippy::too_many_arguments)]
pub fn decode_flash_forward_batch_var(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    q_rows: usize,
    hist: &[usize],
    d: usize,
    heads: usize,
) -> diagnostic::Result<()> {
    decode_batch_var(q, k, v, out, q_rows, hist, d, heads, decode_flash_forward)
}

/// 标记类型：承载 `#[model]` 语义。
pub struct Attention;

/// 朴素注意力语义别名。
#[annotation::model(name = "attention_f32", shape = [8, 8], dtype = "f32", skeleton = "attention")]
pub type AttentionF32 = Attention;

#[cfg(test)]
#[path = "../tests/attention_tests.rs"]
mod tests;
