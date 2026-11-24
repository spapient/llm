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
/// `q`/`k`/`v` 都是 `s × d` 连续二维视图，序列长与头宽从视图形状取；长度不符仍由内部守卫拦下。
///
/// # Errors
///
/// 秩不是二维、长度不符、头宽为 0，或序列长为 0 时返回说明。
pub fn forward(
    q: tensor::View<'_, f32>,
    k: tensor::View<'_, f32>,
    v: tensor::View<'_, f32>,
    out: &mut [f32],
) -> diagnostic::Result<()> {
    let (s, d) = crate::kernel::matmul::dim2(q.dims(), "attention_f32")?;
    crate::kernel::matmul::dim2(k.dims(), "attention_f32")?;
    crate::kernel::matmul::dim2(v.dims(), "attention_f32")?;
    decode(
        q.as_slice(),
        k.as_slice(),
        v.as_slice(),
        out,
        s,
        s,
        d,
        Mode::Naive,
    )
}

/// Flash 内核本体：online softmax，单遍 K/V，写进 `out`。prefill 口径同 [`forward`]。
///
/// # Errors
///
/// 秩不是二维、长度不符、头宽为 0，或序列长为 0 时返回说明。
pub fn flash_forward(
    q: tensor::View<'_, f32>,
    k: tensor::View<'_, f32>,
    v: tensor::View<'_, f32>,
    out: &mut [f32],
) -> diagnostic::Result<()> {
    let (s, d) = crate::kernel::matmul::dim2(q.dims(), "flash_attention_f32")?;
    crate::kernel::matmul::dim2(k.dims(), "flash_attention_f32")?;
    crate::kernel::matmul::dim2(v.dims(), "flash_attention_f32")?;
    decode(
        q.as_slice(),
        k.as_slice(),
        v.as_slice(),
        out,
        s,
        s,
        d,
        Mode::Online,
    )
}

mod decode;
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
    /// `q`/`k` 是 `r × W` / `t × W` 连续二维视图，读宽从视图形状取；`v` 与输出是压缩宽度 `dc`，
    /// 仍保留裸切片（写宽与读宽不同，不进同一张量视图）。
    ///
    /// # Errors
    ///
    /// 秩不是二维、宽度不匹配，或长度不整除时返回说明。
    pub fn forward(
        q: tensor::View<'_, f32>,
        k: tensor::View<'_, f32>,
        v: &[f32],
        out: &mut [f32],
    ) -> diagnostic::Result<()> {
        let (_, w) = crate::kernel::matmul::dim2(q.dims(), "latent_attention_f32")?;
        let (_, kw) = crate::kernel::matmul::dim2(k.dims(), "latent_attention_f32")?;
        if kw != w {
            return Err(Error::new(
                Code::Invalid,
                format!("latent_attention_f32 读宽不符：q·w={w}，k·w={kw}"),
            ));
        }
        let g = tensor_geometry(q.as_slice().len(), k.as_slice().len(), out.len(), w)?;
        super::run_rows(
            &machine(q.as_slice(), k.as_slice(), v, g),
            g.0,
            out,
            super::Mode::Naive,
        );
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
    include!("../../tests/conv_latent_attention_tests.rs");
}
#[cfg(test)]
mod tests {
    include!("../../tests/conv_attention_tests.rs");
}
