//! 大 K 稠密 GEMM（可变收缩轴）：`Y[m,n] = Σ_k A[m,k] · B[k,n]`。
//!
//! 与 [`crate::gemm`] 的 `K=8` 量化骨架互补：本模块的收缩轴长度 `k` 由调用方给定，覆盖 LLM
//! prefill 主链路的 GEMM（如 `k=4096`）。契约四件套同源：
//! - 声明：`Y[m,n] = Σ_k A[m,k]·B[k,n]`，行主序，输出 `f32`；
//! - 语义：浮点代价 `2·m·n·k` FLOP，读入 `(m·k + k·n)·4` 字节；
//! - 标量参考：[`matmul_scalar`]，朴素 `i→j→p` 三重循环，每个输出元素独立按 `p` 顺序累加；
//! - tile 实现：[`matmul_tiled`]，分块 rank-1 更新（GotoBLAS 风格的简化 CPU tile）。
//!
//! 两条路径累加顺序不同，`f32` 加法不结合，因此对拍按 IEEE 容差（[`close_enough`]，对齐
//! lld §10.4 的 IEEE `rtol/atol` 口径），不做逐位相等。
//!
//! # 两种权重布局，两个核
//!
//! [`matmul_tiled`] 吃 `B [k, n]`（收缩轴在外），适合转置过的中间矩阵；线性层（`nn::MatMul` 与
//! `serve` 的投影）的权重是 `w [n, k]`——每个输出通道一整行。那份布局下按 `[k, n]` 分块等于每次
//! 都跨行取数，所以另有一个核 [`matmul_tn_tiled`]：输出列块 × 输出行块 × 收缩块三级分块，块内
//! 是「一整行权重 × 连续行块」的点积，两段都连续。行批投影必须走这个核——逐行调 `gemv` 会把整份
//! 权重按行数重复读一遍。
//!
//! # 一副骨架，两档精度
//!
//! 收缩轴可变的 GEMM 有两档：[`Precision for f32`] 稠密（累加器 `f32`，对拍按 IEEE 容差）与
//! [`Precision for i8`] 对称 int8（累加器 `i32`，对拍逐位相等，见 [`matmul_i8`]）。两者的分块遍历、
//! 尺寸校验、轴表与生产形状全同，只差「元素类型 + 累加器 + 一次乘加」，因此骨架只写这一份、
//! 按元素类型单态化——[`Precision`] 是那份差异的唯一出处。

use std::mem::size_of;
use std::ops::Range;

use diagnostic::{Code, Error, Result};

use crate::model::{AxisRole, AxisSpec, Buf, Cost, Sample, ShapeCtx, Values};

/// 输出 tile 行数。
pub const TILE_M: usize = 64;

/// 输出 tile 列数。
pub const TILE_N: usize = 64;

/// 收缩轴 tile 宽度。
pub const TILE_K: usize = 32;

/// 线性层布局（`x [rows, k]` × `w [n, k]`）的 tile 行数。
pub const TN_ROWS: usize = 16;

/// 线性层 tile 的输出列数：一列权重块要同时在累加器里驻留，不能大于 `TN_ROWS × TN_COLS` 的 L1 预算。
pub const TN_COLS: usize = 32;

/// 线性层 tile 的收缩块宽度。
pub const TN_K: usize = 256;

/// 收缩轴可变 GEMM 的精度契约：两档只差元素类型、累加器与一次乘加。
pub trait Precision: Copy {
    /// 累加器与输出元素：`f32` 档就是 [`f32`]，`i8` 档是 [`i32`]（转回 f32 由注册侧做）。
    type Acc: Copy + Default;

    /// 注册名：尺寸校验与代价标签都取它，入口不必再传一遍名字。
    const NAME: &'static str;

    /// 一次乘加：`*acc += a·b`。
    fn mac(acc: &mut Self::Acc, a: Self, b: Self);
}

impl Precision for f32 {
    type Acc = f32;
    const NAME: &'static str = "matmul_f32";

    fn mac(acc: &mut f32, a: f32, b: f32) {
        *acc += a * b;
    }
}

impl Precision for i8 {
    type Acc = i32;
    const NAME: &'static str = "matmul_i8";

    fn mac(acc: &mut i32, a: i8, b: i8) {
        *acc += i32::from(a) * i32::from(b);
    }
}

/// 标量参考：`out[i,j] = Σ_p a[i,p]·b[p,j]`，每个元素按 `p` 顺序独立累加；两档精度共用。
pub fn matmul_scalar<P: Precision>(
    a: &[P],
    b: &[P],
    m: usize,
    n: usize,
    k: usize,
    out: &mut [P::Acc],
) {
    for i in 0..m {
        let arow = &a[i * k..][..k];
        let orow = &mut out[i * n..][..n];
        for j in 0..n {
            let mut acc = P::Acc::default();
            for (p, &x) in arow.iter().enumerate() {
                P::mac(&mut acc, x, b[p * n + j]);
            }
            orow[j] = acc;
        }
    }
}

/// 分块 tile：按 [`TILE_M`]×[`TILE_N`] 切输出、[`TILE_K`] 切收缩轴，块内 rank-1 更新。
pub fn matmul_tiled<P: Precision>(
    a: &[P],
    b: &[P],
    m: usize,
    n: usize,
    k: usize,
    out: &mut [P::Acc],
) {
    out.fill(P::Acc::default());
    let mut bi = 0;
    while bi < m {
        let i_end = (bi + TILE_M).min(m);
        let mut bk = 0;
        while bk < k {
            let k_end = (bk + TILE_K).min(k);
            let mut bj = 0;
            while bj < n {
                let j_end = (bj + TILE_N).min(n);
                let tile = Tile {
                    rows: bi..i_end,
                    contraction: bk..k_end,
                    cols: bj..j_end,
                };
                accumulate_tile(a, b, out, &tile, n, k);
                bj += TILE_N;
            }
            bk += TILE_K;
        }
        bi += TILE_M;
    }
}

/// 一个 tile 的三条区间：输出行、收缩轴、输出列。
struct Tile {
    rows: Range<usize>,
    contraction: Range<usize>,
    cols: Range<usize>,
}

/// 块内 rank-1 更新：`out[i, j] += Σ_p a[i, p]·b[p, j]`，三条下标都取 `tile` 给的区间。
fn accumulate_tile<P: Precision>(
    a: &[P],
    b: &[P],
    out: &mut [P::Acc],
    tile: &Tile,
    n: usize,
    k: usize,
) {
    for i in tile.rows.clone() {
        let start = i * k + tile.contraction.start;
        let arow = &a[start..i * k + tile.contraction.end];
        for (p, &x) in arow.iter().enumerate() {
            let base = (tile.contraction.start + p) * n;
            let brow = &b[base + tile.cols.start..base + tile.cols.end];
            let orow = &mut out[i * n + tile.cols.start..i * n + tile.cols.end];
            for (o, &y) in orow.iter_mut().zip(brow) {
                P::mac(o, x, y);
            }
        }
    }
}

/// 三条缓冲的尺寸校验，两档精度的四个 `forward` 共用；算子名取自 [`Precision::NAME`]。
fn check_dims<P: Precision>(
    a: &[P],
    b: &[P],
    m: usize,
    n: usize,
    k: usize,
    out: &[P::Acc],
) -> Result<()> {
    if a.len() != m * k || b.len() != k * n || out.len() != m * n {
        return Err(Error::new(
            Code::Invalid,
            format!(
                "{} 缓冲尺寸不符：a={}，b={}，out={}，期望 m·k={}，k·n={}，m·n={}",
                P::NAME,
                a.len(),
                b.len(),
                out.len(),
                m * k,
                k * n,
                m * n
            ),
        ));
    }
    Ok(())
}

/// 标量参考入口：校验尺寸后写 `out`。
///
/// # Errors
///
/// 三条缓冲长度与 `(m, n, k)` 不符时返回说明。
pub fn forward_scalar<P: Precision>(
    a: &[P],
    b: &[P],
    m: usize,
    n: usize,
    k: usize,
    out: &mut [P::Acc],
) -> Result<()> {
    check_dims(a, b, m, n, k, out)?;
    matmul_scalar(a, b, m, n, k, out);
    Ok(())
}

/// tile 实现入口：校验尺寸后写 `out`。
///
/// # Errors
///
/// 三条缓冲长度与 `(m, n, k)` 不符时返回说明。
pub fn forward_tiled<P: Precision>(
    a: &[P],
    b: &[P],
    m: usize,
    n: usize,
    k: usize,
    out: &mut [P::Acc],
) -> Result<()> {
    check_dims(a, b, m, n, k, out)?;
    matmul_tiled(a, b, m, n, k, out);
    Ok(())
}

mod tn;

pub use tn::dot_lanes;
pub use tn::{forward_tn_scalar, forward_tn_tiled, matmul_tn_scalar, matmul_tn_tiled};

/// IEEE 容差对拍：`|a - e| ≤ atol + rtol·|e|`。`rtol/atol` 与 lld §10.4 的 IEEE 口径同源。
pub fn close_enough(actual: &[f32], expected: &[f32], rtol: f32, atol: f32) -> Result<()> {
    if actual.len() != expected.len() {
        return Err(Error::new(
            Code::Invalid,
            format!(
                "对拍长度不符：实际 {}，参考 {}",
                actual.len(),
                expected.len()
            ),
        ));
    }
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        let tol = atol + rtol * e.abs();
        if (a - e).abs() > tol {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "第 {i} 个元素 {a} 与参考 {e} 差 {}，超过容差 {tol}",
                    (a - e).abs()
                ),
            ));
        }
    }
    Ok(())
}

/// 收缩轴长度（生产档）：prefill 主链路 GEMM 的 K。
pub const K: usize = 4096;

/// 三轴：`m`/`k` 输入，`n` 输出（角色与 [`crate::gemm::AXES`] 一致，收缩轴长度由 [`K`] 给定）。
pub const AXES: &[AxisSpec] = &[
    AxisSpec {
        role: AxisRole::Input,
        label: 'm',
    },
    AxisSpec {
        role: AxisRole::Input,
        label: 'k',
    },
    AxisSpec {
        role: AxisRole::Output,
        label: 'n',
    },
];

/// 形状投影：输出 `m×n`，收缩轴 `k = K`（生产档）。
///
/// # Errors
///
/// 轴表不合法时由 [`ShapeCtx::new`] 报出。
pub fn project_shape(m: usize, n: usize) -> Result<ShapeCtx> {
    ShapeCtx::new([('m', m), ('k', K), ('n', n)])
}

/// 标量代价：`2·m·K·n` FLOP，读 `(m·K + K·n)·size_of::<P>()` 字节，写 `m·n·4` 字节。
///
/// 读数由精度定（`f32` 每元素 4 字节、`i8` 1 字节），写出恒为 f32；轴名与算子名取自
/// [`Precision::NAME`]。
///
/// # Errors
///
/// 形状上下文缺轴时返回带算子名的说明。
fn cost_of<P: Precision>(shape: &ShapeCtx) -> Result<Cost> {
    let axis = |label: char| {
        shape
            .get(label)
            .ok_or_else(|| Error::new(Code::Invalid, format!("{} 形状缺轴 `{label}`", P::NAME)))
    };
    let m = axis('m')?;
    let k = axis('k')?;
    let n = axis('n')?;
    Ok(Cost {
        flops: (2 * m * k * n) as f64,
        read: ((m * k + k * n) * size_of::<P>()) as u64,
        write: (m * n * 4) as u64,
    })
}

/// 稠密档的代价：输入每元素 4 字节，口径见 `cost_of`。
///
/// # Errors
///
/// 形状上下文缺轴时返回带算子名的说明。
pub fn cost(shape: &ShapeCtx) -> Result<Cost> {
    cost_of::<f32>(shape)
}

/// 三缓冲样本的公共骨架：`a` / `b` 的取值由精度给，`y` 恒为 f32 零缓冲，标量 `M`/`K`/`N`
/// 与 `.k` 签名一致。
fn sample_of(m: usize, n: usize, k: usize, a: Values, b: Values) -> Sample {
    Sample {
        m,
        n,
        scalars: vec![("M", m as i64), ("K", k as i64), ("N", n as i64)],
        bufs: vec![
            Buf {
                name: "a",
                values: a,
            },
            Buf {
                name: "b",
                values: b,
            },
            Buf {
                name: "y",
                values: Values::F32(vec![0.0; m * n]),
            },
        ],
    }
}

/// 确定性样本：`a` / `b` 由同一份采样序列给出，`y` 为零缓冲，收缩轴取生产形状 [`K`]。
pub fn sample(m: usize, n: usize) -> Sample {
    sample_k(m, n, K)
}

/// 显式收缩轴长度的确定性样本：与 [`sample`] 同构，只有 `K` 由调用方给定。
///
/// 形状前提（尾块、不整除）只有把 `K` 拿在手里才验得了——`K` 不是 `(m, n)` 的函数，
/// [`sample`] 把它写死成生产形状，`3×5×7` 这种形状就无从表达。
pub fn sample_k(m: usize, n: usize, k: usize) -> Sample {
    sample_of(
        m,
        n,
        k,
        Values::F32(deterministic(m * k, 1)),
        Values::F32(deterministic(k * n, 2)),
    )
}

/// 标量参考：读 `a` / `b`，按 [`matmul_scalar`] 计算输出。
///
/// # Errors
///
/// 样本缺 `a` / `b` 时返回说明。
pub fn reference(sample: &Sample) -> Result<Vec<f32>> {
    let a = sample.f32("a")?;
    let b = sample.f32("b")?;
    let mut out = vec![0.0f32; sample.m * sample.n];
    matmul_scalar(a, b, sample.m, sample.n, K, &mut out);
    Ok(out)
}

/// 稠密矩阵乘标记类型：只用来承载语义表，没有实例。
pub struct Matmul;

/// 语义表由注解生成：生产形状 `256×256`，收缩轴 [`K`]，输出 f32。
#[annotation::model(name = "matmul_f32", shape = [256, 256], dtype = "f32", skeleton = "matmul")]
pub type MatmulF32 = Matmul;

crate::operator_here!(Matmul, "matmul_f32", 2, (256, 256), false, reference);

/// 确定性采样（与测试同源）：`(len, salt)` 决定序列，不引入外部依赖。
fn deterministic(len: usize, salt: u32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = (i as u32)
                .wrapping_mul(2654435761)
                .wrapping_add(salt.wrapping_mul(40503));
            ((x >> 8) as f32) / 256.0 - 32.0
        })
        .collect()
}

#[cfg(test)]
#[path = "../../tests/kernel/matmul_tests.rs"]
mod tests;

/// 大 K 对称 int8 GEMM 的语义与精度档：`Y[m, n] = Σ_k A[m, k] · B[k, n]`，累加器 `i32`。
pub mod i8;
