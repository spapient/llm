//! 潜空间注意力（MLA 吸收式单步）：`O[r,c] = Σ_j softmax_j(q[r]·k[j]/√W) · v[j,c]`。
//!
//! 行 = `(查询行 × 头)` 压平，这与 [`attention`](super) 的几何完全同构；差别只有一处：**输出宽度
//! `dc` 可以小于读入宽度 `W`**。压缩存的是 `W`（潜向量 ‖ 解耦 RoPE 段），头内乘积写出的是 `dc`，
//! 上投影 `W^{UV}` 就折在这个口径差里。实测几何（`d_c = 512`、`d_h^R = 64`）是 `W = 576 > dc`，
//! 所以不能反过来把 `dc` 当声明轴——那样真实几何根本喂不进来。
//!
//! RoPE 段必须对同一查询行的各头相同：这是 MLA 的解耦键只有一个、被所有头共享的直接后果。构造
//! `q` 时把这段复制到每个头，本算子照常按整行做内积即可，不需要知道"哪几列是 RoPE"。

use crate::model::{AxisRole, AxisSpec, Buffer, Cost, Sample, ShapeContext, Values};
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
pub fn project_shape(r: usize, w: usize) -> Result<ShapeContext, Error> {
    ShapeContext::new([('r', r), ('w', w)])
}

/// 标量代价：每个查询行对 `t` 条键各做一次宽 `W` 的内积，再按 `dc` 列加权。
///
/// 计价口径取 `dc = W / 2`：声明只有两个轴，`dc` 不进去，只能按同口径的半宽估；真实 `dc` 更小
/// 时实际代价更低，所以这是上界而不是虚报。
///
/// # Errors
///
/// 缺轴时返回说明。
pub fn cost(shape: &ShapeContext) -> Result<Cost, Error> {
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
        (0..len).map(|index| base + index as f32 * step).collect()
    };
    let buffer = |name: &'static str, values: Vec<f32>| Buffer {
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
        buffers: vec![
            buffer("q", det(-0.5, 0.13, r * w)),
            buffer("k", det(0.17, -0.05, r * w)),
            buffer("v", det(-0.23, 0.09, r * dc)),
            buffer(OUT, vec![0.0; r * dc]),
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
