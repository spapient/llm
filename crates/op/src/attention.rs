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

/// 生成一组 `*_forward_*` / `*_flash_forward_*` 公开包装对：朴素与 flash 只差单头内核选择，
/// 签名与布局随形状函数保持一致。
macro_rules! decode_shape_pair {
    (
        $naive:ident, $flash:ident, $target:ident,
        $naive_doc:literal, $flash_doc:literal,
        $( $arg:ident : $ty:ty ),* $(,)?
    ) => {
        #[doc = $naive_doc]
        #[allow(clippy::too_many_arguments)]
        pub fn $naive($( $arg: $ty ),*) -> diagnostic::Result<()> {
            $target($( $arg ),*, decode_forward)
        }

        #[doc = $flash_doc]
        #[allow(clippy::too_many_arguments)]
        pub fn $flash($( $arg: $ty ),*) -> diagnostic::Result<()> {
            $target($( $arg ),*, decode_flash_forward)
        }
    };
}

decode_shape_pair!(
    decode_forward_heads,
    decode_flash_forward_heads,
    decode_heads,
    "多头朴素 decode 内核：头数由 `q` 长度派生，布局同 `decode_heads`，头内走 [`decode_forward`]。\n\n# Errors\n\n同 `decode_heads`。",
    "多头 flash decode 内核：头数由 `q` 长度派生，布局同 `decode_heads`，头内走 [`decode_flash_forward`]。\n\n# Errors\n\n同 `decode_heads`。",
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    q_rows: usize,
    hist: usize,
    d: usize
);

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

decode_shape_pair!(
    decode_forward_batch,
    decode_flash_forward_batch,
    decode_batch,
    "批量朴素 decode 内核：段内走 [`decode_forward`]，布局同 `decode_batch`。\n\n# Errors\n\n同 `decode_batch`。",
    "批量 flash decode 内核：段内走 [`decode_flash_forward`]，布局同 `decode_batch`。\n\n# Errors\n\n同 `decode_batch`。",
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    q_rows: usize,
    hist: usize,
    d: usize,
    heads: usize,
    batch: usize
);

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

decode_shape_pair!(
    decode_forward_batch_var,
    decode_flash_forward_batch_var,
    decode_batch_var,
    "变长批量朴素 decode：段内走 [`decode_forward`]，布局同 `decode_batch_var`。\n\n# Errors\n\n同 `decode_batch_var`。",
    "变长批量 flash decode：段内走 [`decode_flash_forward`]，布局同 `decode_batch_var`。\n\n# Errors\n\n同 `decode_batch_var`。",
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    q_rows: usize,
    hist: &[usize],
    d: usize,
    heads: usize
);

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

#[cfg(test)]
mod tests {
use super::*;

#[test]
fn flash_and_plain_attention_agree_and_rows_are_distributions() {
    let sample = sample(8, 8);
    let plain = reference(&sample).expect("朴素参考");
    let flash = flash_reference(&sample).expect("flash 参考");
    assert!(
        testkit::approx_eq(&plain, &flash, 1e-5),
        "两套算法必须一致：逐元素绝对容差 1e-5"
    );
    // 每个 query 行的注意力权重隐含归一化：输出对全 1 的 V 应得 1，这里直接核对确定性结果非 NaN。
    assert!(testkit::all_finite(&plain));

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
    assert!(testkit::approx_eq(&flash_out, &flash, 1e-5));
}

/// 行主序确定性向量：`rows × d`，第 `i` 个元素取 `base + i·step`，便于逐行核对。
fn det_rows(rows: usize, d: usize, base: f32, step: f32) -> Vec<f32> {
    (0..rows * d)
        .map(|index| base + index as f32 * step)
        .collect()
}

#[test]
fn decode_flash_and_plain_agree_with_single_query_row() {
    let (q_rows, hist, d) = (1, 5, 4);
    let (q, k, v) = decode_qkv(q_rows, hist, d);

    let mut plain = vec![0.0f32; q_rows * d];
    decode_forward(&q, &k, &v, &mut plain, q_rows, hist, d).expect("朴素 decode");
    assert!(testkit::all_finite(&plain));

    let mut flash = vec![0.0f32; q_rows * d];
    decode_flash_forward(&q, &k, &v, &mut flash, q_rows, hist, d).expect("flash decode");
    assert!(
        testkit::approx_eq(&flash, &plain, 1e-5),
        "decode 朴素与 flash 必须一致：逐元素绝对容差 1e-5"
    );
}

#[test]
fn decode_supports_multiple_query_rows_against_history() {
    let (q_rows, hist, d) = (2, 4, 4);
    let (q, k, v) = decode_qkv(q_rows, hist, d);

    let mut plain = vec![0.0f32; q_rows * d];
    decode_forward(&q, &k, &v, &mut plain, q_rows, hist, d).expect("朴素 decode");
    let mut flash = vec![0.0f32; q_rows * d];
    decode_flash_forward(&q, &k, &v, &mut flash, q_rows, hist, d).expect("flash decode");

    assert_plain_flash(&plain, &flash);
}

#[test]
fn decode_rejects_buffer_length_mismatch() {
    let (q_rows, hist, d) = (1, 5, 4);
    let (q, k, v) = decode_qkv(q_rows, hist, d);

    // k 少一行：历史长度对不上。
    let mut out = vec![0.0f32; q_rows * d];
    assert!(decode_forward(&q, &k[..(hist - 1) * d], &v, &mut out, q_rows, hist, d).is_err());
    // 输出缓冲行数对不上。
    let mut short_out = vec![0.0f32; d - 1];
    assert!(decode_forward(&q, &k, &v, &mut short_out, q_rows, hist, d).is_err());
}

#[test]
fn decode_rejects_zero_head_width_or_empty_rows() {
    let empty: &[f32] = &[];
    let mut out = vec![];
    // d=0：√d 除零。
    assert!(decode_forward(empty, empty, empty, &mut out, 1, 5, 0).is_err());
    assert!(decode_flash_forward(empty, empty, empty, &mut out, 1, 5, 0).is_err());
    // 历史为空：softmax 分母为 0。
    let q = det_rows(1, 4, -0.5, 0.13);
    let mut hist_out = vec![0.0f32; 4];
    assert!(decode_forward(&q, empty, empty, &mut hist_out, 1, 0, 4).is_err());
    // 查询行为空。
    assert!(decode_forward(empty, &[0.0; 4], &[0.0; 4], &mut out, 0, 1, 4).is_err());
}

#[test]
fn attention_rejects_zero_head_width() {
    // d=0：三个输入与输出都是空缓冲，长度校验放行后由头宽守卫拦下，避免 √d 除零成 NaN。
    let empty: &[f32] = &[];
    let mut out = vec![];
    assert!(forward(empty, empty, empty, &mut out, 8, 0).is_err());
    assert!(flash_forward(empty, empty, empty, &mut out, 8, 0).is_err());
    assert!(reference(&sample(8, 0)).is_err(), "朴素参考同样守卫头宽");
    assert!(
        flash_reference(&sample(8, 0)).is_err(),
        "flash 参考同样守卫头宽"
    );
}

/// 头优先块布局：第 `h` 头的 `q`/`out` 是一段连续 `rows·d`，`k`/`v` 是一段连续 `hist·d`。
fn det_headed(heads: usize, rows: usize, d: usize, base: f32, step: f32) -> Vec<f32> {
    (0..heads * rows * d)
        .map(|index| base + index as f32 * step)
        .collect()
}

/// 无头 decode 的确定性 q/k/v：`q` 取 `q_rows` 行、`k`/`v` 各取 `hist` 行，同一条线性序列。
fn decode_qkv(q_rows: usize, hist: usize, d: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    (
        det_rows(q_rows, d, -0.5, 0.13),
        det_rows(hist, d, 0.4, -0.09),
        det_rows(hist, d, -0.2, 0.11),
    )
}

/// 多头 decode 的确定性 q/k/v：头优先块布局。
fn headed_qkv(
    heads: usize,
    q_rows: usize,
    hist: usize,
    d: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    (
        det_headed(heads, q_rows, d, -0.5, 0.13),
        det_headed(heads, hist, d, 0.4, -0.09),
        det_headed(heads, hist, d, -0.2, 0.11),
    )
}

/// plain/flash 两套 decode 的公共对拍：全有限 + 逐元素绝对容差 1e-5。
fn assert_plain_flash(plain: &[f32], flash: &[f32]) {
    assert!(testkit::all_finite(plain));
    assert!(testkit::approx_eq(flash, plain, 1e-5));
}

/// 常数 V 段校验：段 `seq` 全取 `1 + seq`，softmax 权重和为 1，输出应恒等于该段常数。
fn assert_constant_v(plain: &[f32], q_seq: usize, batch: usize) {
    for seq in 0..batch {
        let constant = 1.0 + seq as f32;
        let span = &plain[seq * q_seq..(seq + 1) * q_seq];
        assert!(
            span.iter().all(|value| (value - constant).abs() < 1e-5),
            "序列 {seq} 输出应恒等于本段 V 常数 {constant}，得到 {span:?}"
        );
    }
}

#[test]
fn multi_head_decode_keeps_each_head_independent_and_plain_matches_flash() {
    let (q_rows, hist, heads, d) = (1, 4, 2, 3);
    let q = det_headed(heads, q_rows, d, -0.5, 0.13);
    let k = det_headed(heads, hist, d, 0.4, -0.09);
    // V 按头取常数：头 0 全 1、头 1 全 2。softmax 权重和为 1，常数 V 下头内输出恒等于该常数，
    // 借此核对两头结果互不串段——头 0 段不应出现 2，头 1 段不应出现 1。
    let mut v = vec![0.0; heads * hist * d];
    for head in 0..heads {
        let constant = 1.0 + head as f32;
        for value in v[head * hist * d..(head + 1) * hist * d].iter_mut() {
            *value = constant;
        }
    }

    let mut plain = vec![0.0; heads * q_rows * d];
    decode_forward_heads(&q, &k, &v, &mut plain, q_rows, hist, d).expect("朴素多头 decode");
    let mut flash = vec![0.0; heads * q_rows * d];
    decode_flash_forward_heads(&q, &k, &v, &mut flash, q_rows, hist, d).expect("flash 多头 decode");

    assert_plain_flash(&plain, &flash);
    for head in 0..heads {
        let constant = 1.0 + head as f32;
        let span = &plain[head * q_rows * d..(head + 1) * q_rows * d];
        assert!(
            span.iter().all(|value| (value - constant).abs() < 1e-5),
            "头 {head} 输出应恒等于本头 V 常数 {constant}，得到 {span:?}"
        );
    }
}

#[test]
fn one_head_entry_matches_the_single_head_kernel() {
    let (q_rows, hist, d) = (2, 4, 4);
    let (q, k, v) = decode_qkv(q_rows, hist, d);

    let mut single = vec![0.0; q_rows * d];
    decode_forward(&q, &k, &v, &mut single, q_rows, hist, d).expect("单头朴素");
    let mut multi = vec![0.0; q_rows * d];
    decode_forward_heads(&q, &k, &v, &mut multi, q_rows, hist, d).expect("单头经多头入口");
    assert_eq!(single, multi, "heads=1 必须与单头内核逐元素等价");
}

#[test]
fn multi_head_rejects_non_multiple_buffer_length() {
    let (q_rows, hist, heads, d) = (1, 4, 2, 3);
    let (q, k, v) = headed_qkv(heads, q_rows, hist, d);
    let mut out = vec![0.0; heads * q_rows * d];
    // q 只给一个头的长度，而 k/v 是两头：头块倍数校验应拦下。
    assert!(decode_forward_heads(&q[..q_rows * d], &k, &v, &mut out, q_rows, hist, d).is_err());
    // q 长度不是 q_rows·d 的整数倍：无法推出整头，直接拒绝。
    let mut ragged = vec![0.0; q.len() - 1];
    assert!(decode_forward_heads(&q[..q.len() - 1], &k, &v, &mut ragged, q_rows, hist, d).is_err());
}

#[test]
fn batch_decode_keeps_each_sequence_independent_and_plain_matches_flash() {
    let (q_rows, hist, heads, d, batch) = (1, 4, 2, 3, 2);
    let q_seq = heads * q_rows * d;
    let kv_seq = heads * hist * d;
    let mut q = Vec::new();
    let mut k = Vec::new();
    for seq in 0..batch {
        q.extend(det_headed(heads, q_rows, d, -0.5 + seq as f32, 0.13));
        k.extend(det_headed(heads, hist, d, 0.4 + seq as f32, -0.09));
    }
    // V 按序列段取常数：段 0 全 1、段 1 全 2。softmax 权重和为 1，常数 V 下段输出恒等于
    // 该段常数，借此核对两条序列结果互不串段。
    let mut v = vec![0.0; batch * kv_seq];
    for seq in 0..batch {
        let constant = 1.0 + seq as f32;
        for value in v[seq * kv_seq..(seq + 1) * kv_seq].iter_mut() {
            *value = constant;
        }
    }

    let mut plain = vec![0.0; batch * q_seq];
    decode_forward_batch(&q, &k, &v, &mut plain, q_rows, hist, d, heads, batch)
        .expect("朴素批量 decode");
    let mut flash = vec![0.0; batch * q_seq];
    decode_flash_forward_batch(&q, &k, &v, &mut flash, q_rows, hist, d, heads, batch)
        .expect("flash 批量 decode");

    assert_plain_flash(&plain, &flash);
    assert_constant_v(&plain, q_seq, batch);
}

#[test]
fn batch_one_entry_matches_the_multi_head_kernel() {
    let (q_rows, hist, heads, d) = (2, 4, 2, 3);
    let (q, k, v) = headed_qkv(heads, q_rows, hist, d);

    let mut multi = vec![0.0; heads * q_rows * d];
    decode_forward_heads(&q, &k, &v, &mut multi, q_rows, hist, d).expect("多头朴素");
    let mut batch = vec![0.0; heads * q_rows * d];
    decode_forward_batch(&q, &k, &v, &mut batch, q_rows, hist, d, heads, 1)
        .expect("单段经批量入口");
    assert_eq!(multi, batch, "batch=1 必须与多头入口逐元素等价");
}

#[test]
fn batch_rejects_non_multiple_length_and_zero_batch() {
    let (q_rows, hist, heads, d) = (1, 4, 2, 3);
    let (q, k, v) = headed_qkv(heads, q_rows, hist, d);
    let mut out = vec![0.0; heads * q_rows * d];
    // batch=2 但 q/k/v 只给一段：批块倍数校验应拦下。
    assert!(decode_forward_batch(&q, &k, &v, &mut out, q_rows, hist, d, heads, 2).is_err());
    // batch=0：序列数必须非零。
    assert!(decode_forward_batch(&q, &k, &v, &mut out, q_rows, hist, d, heads, 0).is_err());
    // q 长度不是批段整数倍：无法推出整段，直接拒绝。
    let mut ragged = vec![0.0; q.len() - 1];
    assert!(
        decode_forward_batch(
            &q[..q.len() - 1],
            &k,
            &v,
            &mut ragged,
            q_rows,
            hist,
            d,
            heads,
            1
        )
        .is_err()
    );
}

#[test]
fn batch_var_keeps_mixed_history_lengths_independent_and_plain_matches_flash() {
    let (q_rows, heads, d) = (1, 2, 3);
    let hist = [3usize, 5];
    let batch = hist.len();
    let q_seq = heads * q_rows * d;
    let mut q = Vec::new();
    let mut k = Vec::new();
    for (seq, &h) in hist.iter().enumerate() {
        q.extend(det_headed(heads, q_rows, d, -0.5 + seq as f32, 0.13));
        k.extend(det_headed(heads, h, d, 0.4 + seq as f32, -0.09));
    }
    // V 按序列段取常数：段 0 全 1、段 1 全 2。softmax 权重和为 1，常数 V 下段输出恒等于
    // 该段常数，借此核对两条不同历史长度的序列结果互不串段。
    let mut v = Vec::new();
    for (seq, &h) in hist.iter().enumerate() {
        let constant = 1.0 + seq as f32;
        v.extend(std::iter::repeat_n(constant, heads * h * d));
    }

    let mut plain = vec![0.0; batch * q_seq];
    decode_forward_batch_var(&q, &k, &v, &mut plain, q_rows, &hist, d, heads)
        .expect("朴素变长批量 decode");
    let mut flash = vec![0.0; batch * q_seq];
    decode_flash_forward_batch_var(&q, &k, &v, &mut flash, q_rows, &hist, d, heads)
        .expect("flash 变长批量 decode");

    assert_plain_flash(&plain, &flash);
    assert_constant_v(&plain, q_seq, batch);
}

#[test]
fn batch_var_one_entry_matches_the_multi_head_kernel() {
    let (q_rows, hist, heads, d) = (1, 4, 2, 3);
    let (q, k, v) = headed_qkv(heads, q_rows, hist, d);

    let mut multi = vec![0.0; heads * q_rows * d];
    decode_forward_heads(&q, &k, &v, &mut multi, q_rows, hist, d).expect("多头朴素");
    let mut batch = vec![0.0; heads * q_rows * d];
    decode_forward_batch_var(&q, &k, &v, &mut batch, q_rows, &[hist], d, heads)
        .expect("变长入口单段");
    assert_eq!(multi, batch, "batch=1 变长入口必须与多头入口逐元素等价");
}

#[test]
fn batch_var_rejects_mismatched_kv_length_empty_or_zero_history() {
    let (q_rows, heads, d) = (1, 2, 3);
    let hist = [3usize, 5];
    let q: Vec<_> = (0..heads * q_rows * d * hist.len()).map(|_| 0.1).collect();
    let k: Vec<_> = (0..heads * hist.iter().sum::<usize>() * d)
        .map(|_| 0.2)
        .collect();
    let v: Vec<_> = k.clone();
    let mut out = vec![0.0; q.len()];

    // K/V 总长与逐段历史之和不符：游标切段守不住，直接拒绝。
    assert!(
        decode_forward_batch_var(&q, &k[..k.len() - 1], &v, &mut out, q_rows, &hist, d, heads)
            .is_err()
    );
    // 空历史表：序列数必须非零。
    assert!(decode_forward_batch_var(&q, &k, &v, &mut out, q_rows, &[], d, heads).is_err());
    // 某段历史为 0：段内行数守卫拦下。
    assert!(decode_forward_batch_var(&q, &k, &v, &mut out, q_rows, &[3, 0], d, heads).is_err());
}
}


pub mod decode_attention {
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

    /// Flash decode 注意力标记类型：形状、样本、代价与朴素 [`DecodeAttention`] 共用，只把标量参考
    /// 换成 online softmax 版本（[`decode_flash_reference`]）。`const = "DECODE_FLASH_MODEL"` 让
    /// 两个算子同模块各持一份语义表。
    pub struct DecodeFlashAttention;

    /// Flash decode 注意力语义别名；codegen 按注册名选 [`decode_flash_reference`]。
    #[annotation::model(name = "decode_flash_attention_f32", shape = [1, 8], dtype = "f32", skeleton = "attention", const = "DECODE_FLASH_MODEL")]
    pub type DecodeFlashAttentionF32 = DecodeFlashAttention;
}
