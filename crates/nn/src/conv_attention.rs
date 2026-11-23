//! 注意力：`O = softmax(QKᵀ/√d)·V`。

use crate::model::{AxisRole, AxisSpec, Buf, Cost, Sample, ShapeCtx, Values};
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

mod decode {
    //! decode 注意力的两个面：内核算术（[`kernel`]）与语义表（[`semantics`]）。

    mod kernel {
        //! decode 内核族：query 行数（`q_rows`）与 K/V 历史行数（`hist`）解耦的通用路径。
        use diagnostic::{Code, Error};

        use crate::conv::attention::{Mode, Rows, check_width, run_rows};

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
        fn check_pair(
            first: &[f32],
            second: &[f32],
            expect: usize,
            what: &str,
        ) -> diagnostic::Result<()> {
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
        pub(crate) fn decode(
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
                    format!(
                        "批量序列数 {batch}、头数 {heads}、查询行数 {q_rows}、头宽 {d} 都必须非零"
                    ),
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
    }

    pub mod semantics {
        //! decode 注意力语义：query 行数与 K/V 历史行数解耦。
        use crate::model::{AxisRole, AxisSpec, Cost, Sample, ShapeCtx};
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
        /// 轴长为 0 或字母重复时由 [`ShapeCtx::new`] 返回说明。
        pub fn project_shape(q_rows: usize, d: usize) -> Result<ShapeCtx, Error> {
            ShapeCtx::new([('q', q_rows), ('d', d), ('h', HIST)])
        }

        /// 标量代价：每行 query 扫 `hist` 条历史（每个 d 维内积）+ softmax + 对 `V` 加权。
        ///
        /// # Errors
        ///
        /// 缺轴时返回说明。
        pub fn cost(shape: &ShapeCtx) -> Result<Cost, Error> {
            let q = crate::conv::attention::axis(shape, "decode attention", 'q')?;
            let d = crate::conv::attention::axis(shape, "decode attention", 'd')?;
            let h = crate::conv::attention::axis(shape, "decode attention", 'h')?;
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
                bufs: crate::conv::attention::sample_bufs(q_rows * d, hist * d),
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
            let (q, k, v) = crate::conv::attention::qkv(sample)?;
            let mut out = vec![0.0; q_rows * d];
            crate::conv::attention::decode_forward(q, k, v, &mut out, q_rows, hist, d)?;
            Ok(out)
        }

        /// Flash 参考：flash 只是同一数学的**优化**（online softmax 单遍扫 K/V），语义锚仍取朴素定义。
        /// 指回 [`decode_flash_forward`] 会让"内核对不对"与"参考对不对"落在同一行代码上（F6 自我对拍）。
        pub fn decode_flash_reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
            reference(sample)
        }

        /// 标记类型：承载 `#[model]` 语义。
        pub struct DecodeAttention;

        /// 朴素 decode 注意力语义别名。
        #[annotation::model(name = "decode_attention_f32", shape = [1, 8], dtype = "f32", skeleton = "attention")]
        pub type DecodeAttentionF32 = DecodeAttention;

        /// Flash decode 注意力标记类型：形状、样本、代价与朴素 [`DecodeAttention`] 共用，语义锚也共用朴素
        /// 定义（[`decode_flash_reference`] 转调 [`reference`]）；两者差在**内核**，不在数学。
        /// `const = "DECODE_FLASH_MODEL"` 让两个算子同模块各持一份语义表。
        pub struct DecodeFlashAttention;

        /// Flash decode 注意力语义别名；注册名 `decode_flash_attention_f32` 指向同一套数学。
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
    pub use kernel::{
        decode_flash_forward, decode_flash_forward_batch, decode_flash_forward_batch_var,
        decode_flash_forward_heads, decode_forward, decode_forward_batch, decode_forward_batch_var,
        decode_forward_heads,
    };

    pub(super) use kernel::decode;
}
pub mod latent {
    //! 潜空间注意力（MLA 吸收式单步）：`O[r,c] = Σ_j softmax_j(q[r]·k[j]/√W) · v[j,c]`。
    //!
    //! 行 = `(查询行 × 头)` 压平，这与 [`attention`](super) 的几何完全同构；差别只有一处：**输出宽度
    //! `dc` 可以小于读入宽度 `W`**。压缩存的是 `W`（潜向量 ‖ 解耦 RoPE 段），头内乘积写出的是 `dc`，
    //! 上投影 `W^{UV}` 就折在这个口径差里。实测几何（`d_c = 512`、`d_h^R = 64`）是 `W = 576 > dc`，
    //! 所以不能反过来把 `dc` 当声明轴——那样真实几何根本喂不进来。
    //!
    //! RoPE 段必须对同一查询行的各头相同：这是 MLA 的解耦键只有一个、被所有头共享的直接后果。构造
    //! `q` 时把这段复制到每个头，本算子照常按整行做内积即可，不需要知道"哪几列是 RoPE"。

    use crate::model::{AxisRole, AxisSpec, Buf, Cost, Sample, ShapeCtx, Values};
    use diagnostic::{Code, Error};

    use super::axis;

    /// 两个轴：压平后的查询行 `r`、读入行宽 `w`。
    pub const AXES: &[AxisSpec] = &[
        AxisSpec {
            role: AxisRole::Input,
            label: 'r',
        },
        AxisSpec {
            role: AxisRole::Output,
            label: 'w',
        },
    ];

    /// 输出缓冲名：只用来在样本里量出 `dc`。
    const OUT: &str = "y";

    /// `(m, n)` 即 `(压平后的查询行数 r, 读入行宽 W)`。
    ///
    /// # Errors
    ///
    /// 当前实现不会失败。
    pub fn project_shape(r: usize, w: usize) -> Result<ShapeCtx, Error> {
        ShapeCtx::new([('r', r), ('w', w)])
    }

    /// 标量代价：每个查询行对 `t` 条键各做一次宽 `W` 的内积，再按 `dc` 列加权。
    ///
    /// 计价口径取 `dc = W / 2`：声明只有两个轴，`dc` 不进去，只能按同口径的半宽估；真实 `dc` 更小
    /// 时实际代价更低，所以这是上界而不是虚报。
    ///
    /// # Errors
    ///
    /// 缺轴时返回说明。
    pub fn cost(shape: &ShapeCtx) -> Result<Cost, Error> {
        let r = axis(shape, "latent_attention", 'r')?;
        let w = axis(shape, "latent_attention", 'w')?;
        let (dot, weighted) = pricing(r, w);
        Ok(Cost {
            flops: dot + weighted,
            read: (rows_of(r, w) * 4) as u64,
            write: (r * w.div_ceil(2) * 4) as u64,
        })
    }

    /// 分数段与加权段的浮点计数。
    fn pricing(r: usize, w: usize) -> (f64, f64) {
        let dc = w.div_ceil(2);
        let rows = r as f64;
        let t = r as f64;
        (rows * t * w as f64 * 2.0, rows * t * dc as f64 * 2.0)
    }

    /// 读侧元素数：`q`/`k` 是宽 `W` 的行，`v` 是宽 `W/2` 的行。
    fn rows_of(r: usize, w: usize) -> usize {
        let dc = w.div_ceil(2);
        r * w + r * w + r * dc
    }

    /// 确定性样本：`q`/`k` 是 `r × W`，`v` 是 `r × W/2`（前视同构，键行数取 `r`），`y` 是 `r × W/2`。
    ///
    /// 四个标量是 `.k` 的签名实参（`R` 行数、`T` 键行数、`W` 读宽、`DC` 写宽）——它们是同一份几何的
    /// 另一处投影，不是额外的自由度：调用方按长度算出后传进去，与算子内部的折法一致。
    pub fn sample(m: usize, n: usize) -> Sample {
        let (r, w) = (m.max(1), n.max(1));
        let dc = w.div_ceil(2);
        let det = |base: f32, step: f32, len: usize| -> Vec<f32> {
            (0..len).map(|idx| base + idx as f32 * step).collect()
        };
        let buf = |name: &'static str, values: Vec<f32>| Buf {
            name,
            values: Values::F32(values),
        };
        Sample {
            m: r,
            n: w,
            scalars: vec![
                ("R", r as i64),
                ("T", r as i64),
                ("W", w as i64),
                ("DC", dc as i64),
            ],
            bufs: vec![
                buf("q", det(-0.5, 0.13, r * w)),
                buf("k", det(0.17, -0.05, r * w)),
                buf("v", det(-0.23, 0.09, r * dc)),
                buf(OUT, vec![0.0; r * dc]),
            ],
        }
    }

    /// 一次调用的几何：压平行数、键行数、读宽、写宽——四个数都由长度直接折出，没有标量可依赖。
    type Geometry = (usize, usize, usize, usize);

    /// 由样本声明与输出长度定几何（前视同构：键行数取查询行数）。
    fn sample_geometry(sample: &Sample) -> diagnostic::Result<Geometry> {
        let (r, w) = (sample.m, sample.n);
        Ok((r, r, w, sample.f32(OUT)?.len() / r.max(1)))
    }

    /// 由运行期张量定几何。
    ///
    /// # Errors
    ///
    /// 长度不是读宽的整数倍时返回说明。
    fn tensor_geometry(q: usize, k: usize, out: usize, w: usize) -> diagnostic::Result<Geometry> {
        let site = "latent_attention_f32";
        let rows = fold(site, q, w, "q")?;
        Ok((rows, fold(site, k, w, "k")?, w, out / rows.max(1)))
    }

    /// 按读宽把长度折成行数。
    fn fold(site: &str, len: usize, w: usize, what: &str) -> diagnostic::Result<usize> {
        if len.is_multiple_of(w) {
            return Ok(len / w);
        }
        Err(Error::new(
            Code::Invalid,
            format!("{site} 的 `{what}` 长度 {len} 不是行宽 {w} 的整数倍"),
        ))
    }

    /// 参考与内核共用父模块的行机器，只是写宽比读宽窄；两条入口分别落在两种算法上，
    /// 于是对拍仍然是在比算法，而不是在比同一段代码。
    fn machine<'a>(q: &'a [f32], k: &'a [f32], v: &'a [f32], g: Geometry) -> super::Rows<'a> {
        super::Rows::narrow(q, k, v, g.1, g.2, g.3)
    }

    /// 参考：`online softmax` 单遍 K/V，逐行归一化后写进压缩宽度。
    ///
    /// # Errors
    ///
    /// 操作数缺失或长度不符时返回说明。
    pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
        let g = sample_geometry(sample)?;
        let (q, k, v) = super::qkv(sample)?;
        let mut out = vec![0.0f32; g.0 * g.3];
        super::run_rows(&machine(q, k, v, g), g.0, &mut out, super::Mode::Online);
        Ok(out)
    }

    /// 可执行内核：`naive` 三阶段（落地分数再归一化），几何只从张量长度取，不借道样本结构。
    ///
    /// # Errors
    ///
    /// 长度不整除时返回说明。
    pub fn forward(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        out: &mut [f32],
        w: usize,
    ) -> diagnostic::Result<()> {
        let g = tensor_geometry(q.len(), k.len(), out.len(), w)?;
        super::run_rows(&machine(q, k, v, g), g.0, out, super::Mode::Naive);
        Ok(())
    }

    /// 标记类型：承载 `#[model]` 语义。
    pub struct LatentAttention;

    /// 潜空间注意力的语义别名；codegen 与 `nn` 都按 `latent_attention_f32` 这个名字取它。
    #[annotation::model(name = "latent_attention_f32", shape = [8, 8], dtype = "f32", skeleton = "attention")]
    pub type LatentAttentionF32 = LatentAttention;

    // 基座契约（ADR 0050）：三个操作数（q、k、v），输出 y；写宽比读宽窄，反向按两口径散射
    // （dV 走压缩宽度、dQ/dK 走读入宽度），公式在 `crates/train` 的公式目录里。
    crate::operator_here!(
        LatentAttention,
        "latent_attention_f32",
        3,
        (8, 8),
        true,
        reference
    );
}
pub use decode::{
    decode_flash_forward, decode_flash_forward_batch, decode_flash_forward_batch_var,
    decode_flash_forward_heads, decode_forward, decode_forward_batch, decode_forward_batch_var,
    decode_forward_heads,
};
pub use latent::{LatentAttention, LatentAttentionF32, forward as latent_forward};

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
mod latent_attention_tests {
    include!("../tests/conv_latent_attention_tests.rs");
}
#[cfg(test)]
mod tests {
    include!("../tests/conv_attention_tests.rs");
}
