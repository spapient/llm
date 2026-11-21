//! 注意力：`O = softmax(QKᵀ/√d)·V`。
//!
//! 同一个形状（`s` 条序列、头宽 `d`）下给出两套实现，互相对拍：
//!
//! - `reference` / `forward`：朴素三阶段——先算全部分数 `QKᵀ/√d`，再对每行 softmax，最后乘 `V`。
//!   它要一块 `s×s` 的完整分数矩阵，是数值参考；
//! - [`flash_reference`] / [`flash_forward`]：online softmax 分块，维护 running max 与归一化和，
//!   单遍扫描 `K/V`，不落地完整分数矩阵。两套结果必须逐元素相等（误差内）；
//! - [`crate::attention::tiled`]：上面两条的分块 tile 版本，**执行侧走的那条**（`serve` / `train` 的适配
//!   都指向它），与 `forward` / `flash_forward` 分别逐位一致，只是 K/V 块在行块内被复用。
//!
//! 样本、形状与代价两算子共用；`#[annotation::model]` 按注册名挑对应的参考函数。

use crate::model::{AxisRole, AxisSpec, Buffer, Cost, Sample, ShapeContext, Values};
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
pub fn project_shape(s: usize, d: usize) -> Result<ShapeContext, Error> {
    ShapeContext::new([('s', s), ('d', d)])
}

/// 取形状轴；缺失时按模块前缀与轴名报错，避免各调用点各写一遍分支。
fn axis(shape: &ShapeContext, what: &str, label: char) -> Result<usize, Error> {
    shape
        .get(label)
        .ok_or_else(|| Error::new(Code::Invalid, format!("{what} 形状缺轴 `{label}`")))
}

/// 标量代价：分数矩阵 `s²` 个点（每个 d 维内积）+ softmax + 对 `V` 的加权。
///
/// # Errors
///
/// 缺轴时返回说明。
pub fn cost(shape: &ShapeContext) -> Result<Cost, Error> {
    let s = axis(shape, "attention", 's')?;
    let d = axis(shape, "attention", 'd')?;
    Ok(Cost {
        flops: (s as f64) * (s as f64) * ((d as f64) * 2.0 + 3.0),
        read: (s as u64) * (d as u64) * 4 * 3,
        write: (s as u64) * (d as u64) * 4,
    })
}

/// 确定性 q/k/v/y 缓冲：`q` 与 `y` 取 `q_len`，`k`/`v` 取 `kv_len`。
fn sample_buffers(q_len: usize, kv_len: usize) -> Vec<Buffer> {
    let det = |base: f32, step: f32, len: usize| -> Vec<f32> {
        (0..len).map(|index| base + index as f32 * step).collect()
    };
    let buffer = |name: &'static str, values: Vec<f32>| Buffer {
        name,
        values: Values::F32(values),
    };
    vec![
        buffer("q", det(-0.5, 0.13, q_len)),
        buffer("k", det(0.4, -0.09, kv_len)),
        buffer("v", det(-0.2, 0.11, kv_len)),
        buffer("y", vec![0.0; q_len]),
    ]
}

/// 确定性样本：`q`/`k`/`v` 三个输入与输出 `o`，均为 `s*d`。
pub fn sample(s: usize, d: usize) -> Sample {
    let len = s * d;
    Sample {
        m: s,
        n: d,
        scalars: vec![("S", s as i64), ("D", d as i64)],
        buffers: sample_buffers(len, len),
    }
}

/// 三个输入缓冲：朴素与 flash 路径读同一份 `q/k/v`，缺失时的文案因此只写一次。
fn qkv(sample: &Sample) -> diagnostic::Result<(&[f32], &[f32], &[f32])> {
    Ok((sample.f32("q")?, sample.f32("k")?, sample.f32("v")?))
}

/// 行 `row` 与行 `col` 的点积：四个朴素/flash 实现共用，免得每处重写 `d` 长的累加循环。
///
/// 归约本身走 [`crate::matmul::dot_lanes`]——同一个多链点积在 `op` 里只有一份，否则注意力与
/// 矩阵乘的向量化程度会各自漂移（实测：单链标量归约在 1024×64 注意力上比多链慢 1.4×）。
fn dot(x: &[f32], y: &[f32], row: usize, col: usize, d: usize) -> f32 {
    crate::matmul::dot_lanes(&x[row * d..][..d], &y[col * d..][..d])
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
    /// 以操作数与 K/V 列几何建行机器：`scale = √d`。
    fn new(q: &'a [f32], k: &'a [f32], v: &'a [f32], cols: usize, d: usize) -> Self {
        Self {
            q,
            k,
            v,
            cols,
            d,
            scale: (d as f32).sqrt(),
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
        for h in 0..self.d {
            let mut acc = 0.0f32;
            for j in 0..self.cols {
                acc += (exps[j] / sum) * self.v[j * self.d + h];
            }
            out[row * self.d + h] = acc;
        }
    }

    /// online（flash）路径的一行：跑一遍 `col` 维护最大值、归一化分母与累加向量。
    ///
    /// 累加向量写进调用方传入的 `acc`（先清零再写回），返回归一化分母 `sum`。缓冲由
    /// [`run_rows`] 预分配复用，避免热路径每行一次堆分配。
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
            for h in 0..self.d {
                acc[h] = acc[h] * reweight + prob * self.v[col * self.d + h];
            }
            max = new_max;
        }
        sum
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
            // 预分配一条累加缓冲，所有行复用；online 每行 fill 清零后写回。
            // 消除热路径上「每行一次堆分配」，prefill（rows_count = s）收益随序列长增长。
            let mut acc = vec![0.0f32; rows.d];
            for row in 0..rows_count {
                let sum = rows.online(row, &mut acc);
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

pub(crate) fn check_decode_len(
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
    let rows = Rows::new(q, k, v, hist, d);
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
#[path = "../tests/attention_tests.rs"]
mod tests;

pub mod decode_attention {
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
}
