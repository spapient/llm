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
//! [`Precision for i8`] 对称 int8（累加器 `i32`，对拍逐位相等，见 `matmul_i8`）。两者的分块遍历、
//! 尺寸校验、轴表与生产形状全同，只差「元素类型 + 累加器 + 一次乘加」，因此骨架只写这一份、
//! 按元素类型单态化——[`Precision`] 是那份差异的唯一出处。

use std::mem::size_of;
use std::ops::Range;

use diagnostic::{Code, Error, Result};

use crate::model::{Buf, Cost, Sample, ShapeCtx, Values};
use tensor::View;

/// 取二维视图的 `(rows, cols)`；秩不符按算子名报错。矩阵乘族各出口共用这一处几何解包。
pub(crate) fn dim2(dims: &[usize], op: &str) -> Result<(usize, usize)> {
    let &[rows, cols] = dims else {
        return Err(Error::new(
            Code::Invalid,
            format!("{op} 需要二维张量，实际秩 {}", dims.len()),
        ));
    };
    Ok((rows, cols))
}

/// 两条二维视图的收缩轴对齐：返回 `(m, k, n)`，`a` 的列与 `b` 的行不符时说明。
pub(crate) fn dim2_pair(a: &[usize], b: &[usize], op: &str) -> Result<(usize, usize, usize)> {
    let (m, ka) = dim2(a, op)?;
    let (kb, n) = dim2(b, op)?;
    if ka != kb {
        return Err(Error::new(
            Code::Invalid,
            format!("{op} 收缩轴不符：a·k={ka}，b·k={kb}"),
        ));
    }
    Ok((m, ka, n))
}

/// 转置布局（TN）几何：`x[rows,k]` 配 `w[n,k]`，收缩轴都是末列；返回 `(rows, n, k)`。
pub(crate) fn tn_geom(x: &[usize], w: &[usize]) -> Result<(usize, usize, usize)> {
    let (rows, k) = dim2(x, "matmul_tn")?;
    let (n, k2) = dim2(w, "matmul_tn")?;
    if k != k2 {
        return Err(Error::new(
            Code::Invalid,
            format!("matmul_tn 收缩轴不符：x·k={k}，w·k={k2}"),
        ));
    }
    Ok((rows, n, k))
}

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

/// 全算子族共享的精度根（累加器 + 乘累加），定义在 [`crate::scalar`]，此处转出供 GEMM 入口使用。
pub use crate::scalar::Precision;

/// GEMM 的精度档：在共享 [`Precision`] 之上再加一个注册名（尺寸校验与代价标签取它）。
///
/// 注册名是「每档对应的 matmul 算子名」，属 GEMM 局部而非元素精度的固有属性，
/// 故与共享根分离：unary / binary / reduce 泛型到 `Precision` / `Scalar` 即可，不需要名字。
pub trait Tier: Precision {
    /// 注册名：尺寸校验与代价标签都取它，入口不必再传一遍名字。
    const NAME: &'static str;
}

impl Tier for f32 {
    const NAME: &'static str = "matmul_f32";
}

impl Precision for i8 {
    type Acc = i32;

    fn mac(acc: &mut i32, a: i8, b: i8) {
        *acc += i32::from(a) * i32::from(b);
    }
}

impl Tier for i8 {
    const NAME: &'static str = "matmul_i8";
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

/// 三条缓冲的尺寸校验，两档精度的四个 `forward` 共用；算子名取自 [`Tier::NAME`]。
fn check_dims<P: Tier>(
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

/// 裸切片骨架函数指针：标量与 tile 两条公共入口各自传入对应的微内核。
type MatmulKernel<P> = fn(&[P], &[P], usize, usize, usize, &mut [<P as Precision>::Acc]);

/// 校验两条二维视图与输出缓冲，再把裸切片交给指定骨架；标量与 tile 两条入口共用这套装配。
fn run<P: Tier>(
    a: View<'_, P>,
    b: View<'_, P>,
    out: &mut [P::Acc],
    kernel: MatmulKernel<P>,
) -> Result<()> {
    let (m, k, n) = dim2_pair(a.dims(), b.dims(), P::NAME)?;
    check_dims(a.as_slice(), b.as_slice(), m, n, k, out)?;
    kernel(a.as_slice(), b.as_slice(), m, n, k, out);
    Ok(())
}

/// 标量参考入口：校验尺寸后写 `out`。
///
/// # Errors
///
/// 三条缓冲长度与 `(m, n, k)` 不符时返回说明。
pub fn forward_scalar<P: Tier>(a: View<'_, P>, b: View<'_, P>, out: &mut [P::Acc]) -> Result<()> {
    run(a, b, out, matmul_scalar)
}

/// tile 实现入口：校验尺寸后写 `out`。
///
/// # Errors
///
/// 三条缓冲长度与 `(m, n, k)` 不符时返回说明。
pub fn forward_tiled<P: Tier>(a: View<'_, P>, b: View<'_, P>, out: &mut [P::Acc]) -> Result<()> {
    run(a, b, out, matmul_tiled)
}

/// 线性层布局（TN）的标量参考：`out[r, j] = Σ_p x[r, p]·w[j, p]`，每个元素按 `p` 顺序独立累加。
pub fn matmul_tn_scalar(x: &[f32], w: &[f32], rows: usize, n: usize, k: usize, out: &mut [f32]) {
    for r in 0..rows {
        let xrow = &x[r * k..][..k];
        let orow = &mut out[r * n..][..n];
        for (j, slot) in orow.iter_mut().enumerate() {
            let wrow = &w[j * k..][..k];
            let mut acc = 0.0f32;
            for (&a, &b) in xrow.iter().zip(wrow) {
                acc += a * b;
            }
            *slot = acc;
        }
    }
}

/// 线性层布局的分块 tile：[`TN_ROWS`] 行 × [`TN_COLS`] 列的输出块，收缩轴按 [`TN_K`] 切。
pub fn matmul_tn_tiled(x: &[f32], w: &[f32], rows: usize, n: usize, k: usize, out: &mut [f32]) {
    let mut block = [0.0f32; TN_ROWS * TN_COLS];
    let mut j0 = 0;
    while j0 < n {
        let j1 = (j0 + TN_COLS).min(n);
        let mut r0 = 0;
        while r0 < rows {
            let r1 = (r0 + TN_ROWS).min(rows);
            let acc = &mut block[..(r1 - r0) * TN_COLS];
            acc.fill(0.0);
            let mut p0 = 0;
            while p0 < k {
                let p1 = (p0 + TN_K).min(k);
                let tile = Tile {
                    rows: r0..r1,
                    contraction: p0..p1,
                    cols: j0..j1,
                };
                accumulate_tn_tile(x, w, acc, &tile, k);
                p0 = p1;
            }
            for row in r0..r1 {
                let start = (row - r0) * TN_COLS;
                let len = j1 - j0;
                out[row * n + j0..row * n + j1].copy_from_slice(&acc[start..start + len]);
            }
            r0 = r1;
        }
        j0 = j1;
    }
}

/// 块内累加：`acc[row, j] += Σ_p x[row, p]·w[j, p]`，与 NN 布局共用 [`Tile`] 的三条区间。
fn accumulate_tn_tile(x: &[f32], w: &[f32], acc: &mut [f32], tile: &Tile, k: usize) {
    for j in tile.cols.clone() {
        let wchunk = &w[j * k + tile.contraction.start..j * k + tile.contraction.end];
        let slot = j - tile.cols.start;
        for row in tile.rows.clone() {
            let xchunk = &x[row * k + tile.contraction.start..row * k + tile.contraction.end];
            acc[(row - tile.rows.start) * TN_COLS + slot] += dot_lanes(xchunk, wchunk);
        }
    }
}

/// 累加链数：8 条独立的部分和，乘加不排成一条依赖链（编译后是一整条 SIMD 乘加指令）。
const LANES: usize = 8;

/// 多点积：`LANES` 条独立累加链。
pub fn dot_lanes(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let (x_chunks, x_tail) = a[..len].as_chunks::<LANES>();
    let (y_chunks, y_tail) = b[..len].as_chunks::<LANES>();
    let mut lanes = [0.0f32; LANES];
    for (x, y) in x_chunks.iter().zip(y_chunks) {
        let mut lane = 0;
        while lane < LANES {
            lanes[lane] += x[lane] * y[lane];
            lane += 1;
        }
    }
    let mut sum = lanes.iter().sum::<f32>();
    for (x, y) in x_tail.iter().zip(y_tail) {
        sum += x * y;
    }
    sum
}

/// 线性层布局的尺寸校验：输入 `rows × k`、权重 `n × k`、输出 `rows × n`。
fn check_tn_dims(x: &[f32], w: &[f32], rows: usize, n: usize, k: usize, out: &[f32]) -> Result<()> {
    if x.len() != rows * k || w.len() != n * k || out.len() != rows * n {
        return Err(Error::new(
            Code::Invalid,
            format!(
                "线性层 GEMM 缓冲尺寸不符：x={}，w={}，out={}，期望 rows·k={}，n·k={}，rows·n={}",
                x.len(),
                w.len(),
                out.len(),
                rows * k,
                n * k,
                rows * n
            ),
        ));
    }
    Ok(())
}

/// 线性层布局的标量入口：校验尺寸后写 `out`。
///
/// # Errors
///
/// 三条缓冲长度与 `(rows, n, k)` 不符时返回说明。
pub fn forward_tn_scalar(x: View<'_, f32>, w: View<'_, f32>, out: &mut [f32]) -> Result<()> {
    let (rows, n, k) = tn_geom(x.dims(), w.dims())?;
    check_tn_dims(x.as_slice(), w.as_slice(), rows, n, k, out)?;
    matmul_tn_scalar(x.as_slice(), w.as_slice(), rows, n, k, out);
    Ok(())
}

/// 线性层布局的 tile 入口：校验尺寸后写 `out`。这是服务侧行批投影的唯一生产路径。
///
/// # Errors
///
/// 三条缓冲长度与 `(rows, n, k)` 不符时返回说明。
pub fn forward_tn_tiled(x: View<'_, f32>, w: View<'_, f32>, out: &mut [f32]) -> Result<()> {
    let (rows, n, k) = tn_geom(x.dims(), w.dims())?;
    check_tn_dims(x.as_slice(), w.as_slice(), rows, n, k, out)?;
    matmul_tn_tiled(x.as_slice(), w.as_slice(), rows, n, k, out);
    Ok(())
}

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

/// 对拍夹具：把 `[m,k]` 与 `[k,n]` 两份样本装配成两个只读视图，f32 / i8 两档共用。
#[cfg(test)]
pub(crate) fn pair_views<'a, K: Precision>(
    m: usize,
    n: usize,
    k: usize,
    a: &'a [K],
    b: &'a [K],
) -> (View<'a, K>, View<'a, K>) {
    let av = View::matrix(m, k, a).expect("a 二维");
    let bv = View::matrix(k, n, b).expect("b 二维");
    (av, bv)
}

/// 收缩轴长度（生产档）：prefill 主链路 GEMM 的 K。
pub const K: usize = 4096;

/// 三轴角色与固定档 GEMM 完全同源；只有本模块的 [`K`] 取生产档。
pub use crate::gemm::AXES;

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
/// [`Tier::NAME`]。
///
/// # Errors
///
/// 形状上下文缺轴时返回带算子名的说明。
fn cost_of<P: Tier>(shape: &ShapeCtx) -> Result<Cost> {
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
            Buf::f32("y", vec![0.0; m * n]),
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
/// 样本缺 `a` / `b` / `K`，或 `K` 不是非负 usize 时返回说明。
pub fn reference(sample: &Sample) -> Result<Vec<f32>> {
    let a = sample.f32("a")?;
    let b = sample.f32("b")?;
    let k = contraction_axis(sample)?;
    let mut out = vec![0.0f32; sample.m * sample.n];
    matmul_scalar(a, b, sample.m, sample.n, k, &mut out);
    Ok(out)
}

/// 样本的收缩轴单独解析，缺失与非负/宽度错误保持不同诊断。
fn contraction_axis(sample: &Sample) -> Result<usize> {
    let k = sample
        .scalar("K")
        .ok_or_else(|| Error::new(Code::Missing, "样本缺收缩轴标量 `K`"))?;
    usize::try_from(k)
        .map_err(|_| Error::new(Code::Invalid, format!("收缩轴 `K` 不是非负 usize：{k}")))
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
mod tests {
    include!("../tests/kernel_matmul_test_support.rs");
    include!("../tests/kernel_matmul_tests.rs");
}

/// 对称 int8 GEMM 变体：标量/分块出口与形状投影，骨架算术在本模块。
///
/// 大 K 对称 int8 GEMM：`Y[m, n] = Σ_k A[m, k] · B[k, n]`，累加器 `i32`。
pub mod i8 {
    use diagnostic::Result;
    use tensor::View;

    use crate::model::{Cost, Sample, ShapeCtx, Values};

    pub use super::{
        AXES, K, TILE_K, TILE_M, TILE_N, forward_scalar, matmul_scalar, matmul_tiled, project_shape,
    };

    const BLOCK_ROWS: usize = 4;
    const BLOCK_COLS: usize = 16;

    /// 分块入口：公共 `run` 完成视图与尺寸校验，4×16 瓦片在编译期单态化。
    ///
    /// # Errors
    ///
    /// 三条缓冲长度与 `(m, n, k)` 不符时返回说明。
    pub fn forward_tiled(a: View<'_, i8>, b: View<'_, i8>, out: &mut [i32]) -> Result<()> {
        super::run(a, b, out, blocked::<BLOCK_ROWS, BLOCK_COLS>)
    }

    /// const 泛型瓦片：完整块走寄存器累加，两块残区共用同一标量区域骨架。
    fn blocked<const R: usize, const C: usize>(
        a: &[i8],
        b: &[i8],
        m: usize,
        n: usize,
        k: usize,
        out: &mut [i32],
    ) {
        let mut i0 = 0;
        while i0 + R <= m {
            let mut j0 = 0;
            while j0 + C <= n {
                block::<R, C>(a, b, out, (i0, j0), n, k);
                j0 += C;
            }
            scalar_region(a, b, out, i0..i0 + R, j0..n, n, k);
            i0 += R;
        }
        scalar_region(a, b, out, i0..m, 0..n, n, k);
    }

    fn block<const R: usize, const C: usize>(
        a: &[i8],
        b: &[i8],
        out: &mut [i32],
        at: (usize, usize),
        n: usize,
        k: usize,
    ) {
        let (i0, j0) = at;
        let mut acc = [[0i32; C]; R];
        for p in 0..k {
            let wide = widen::<C>(&b[p * n + j0..][..C]);
            for r in 0..R {
                let x = i32::from(a[(i0 + r) * k + p]);
                for j in 0..C {
                    acc[r][j] += x * wide[j];
                }
            }
        }
        for r in 0..R {
            out[(i0 + r) * n + j0..][..C].copy_from_slice(&acc[r]);
        }
    }

    fn widen<const C: usize>(row: &[i8]) -> [i32; C] {
        let mut wide = [0i32; C];
        for (slot, &value) in wide.iter_mut().zip(row) {
            *slot = i32::from(value);
        }
        wide
    }

    /// 标量残区：行列边界均由区间声明，不再分别维护“残行”和“残列”两份循环。
    fn scalar_region(
        a: &[i8],
        b: &[i8],
        out: &mut [i32],
        rows: std::ops::Range<usize>,
        cols: std::ops::Range<usize>,
        n: usize,
        k: usize,
    ) {
        for i in rows {
            for j in cols.clone() {
                out[i * n + j] = dot(a, b, i, j, n, k);
            }
        }
    }

    fn dot(a: &[i8], b: &[i8], i: usize, j: usize, n: usize, k: usize) -> i32 {
        let mut acc = 0i32;
        for p in 0..k {
            acc += i32::from(a[i * k + p]) * i32::from(b[p * n + j]);
        }
        acc
    }

    /// 标量代价：`2·m·K·n` FLOP；输入为 i8，输出 f32。
    ///
    /// # Errors
    ///
    /// 形状上下文缺轴时返回带算子名的说明。
    pub fn cost(shape: &ShapeCtx) -> Result<Cost> {
        super::cost_of::<i8>(shape)
    }

    /// 确定性样本：`a` / `b` 为对称 i8（u8 字节承载有符号码），`y` 为 f32 零缓冲。
    pub fn sample(m: usize, n: usize) -> Sample {
        sample_k(m, n, K)
    }

    /// 显式收缩轴长度的确定性样本。
    pub fn sample_k(m: usize, n: usize, k: usize) -> Sample {
        super::sample_of(
            m,
            n,
            k,
            Values::U8(deterministic_i8(m * k, 1)),
            Values::U8(deterministic_i8(k * n, 2)),
        )
    }

    /// 标量参考：读 `a` / `b`，i32 精确累加后转 f32 输出。
    ///
    /// # Errors
    ///
    /// 样本缺 `a` / `b` 时返回说明。
    pub fn reference(sample: &Sample) -> Result<Vec<f32>> {
        let a: Vec<i8> = sample.u8("a")?.iter().map(|&x| x as i8).collect();
        let b: Vec<i8> = sample.u8("b")?.iter().map(|&x| x as i8).collect();
        let mut acc = vec![0i32; sample.m * sample.n];
        matmul_scalar(&a, &b, sample.m, sample.n, K, &mut acc);
        Ok(acc.into_iter().map(|x| x as f32).collect())
    }

    /// 对称 int8 矩阵乘标记类型：只用来承载语义表，没有实例。
    pub struct MatmulI8Op;

    /// 语义表由注解生成：生产形状 `256×256`，收缩轴 [`K`]。
    #[annotation::model(name = "matmul_i8", shape = [256, 256], dtype = "f32", skeleton = "matmul")]
    pub type MatmulI8 = MatmulI8Op;

    crate::operator_here!(MatmulI8Op, "matmul_i8", 2, (256, 256), false, reference);

    fn deterministic_i8(len: usize, salt: u32) -> Vec<u8> {
        (0..len)
            .map(|i| {
                let x = (i as u32)
                    .wrapping_mul(2654435761)
                    .wrapping_add(salt.wrapping_mul(40503));
                ((x % 251) as i32 - 125) as i8 as u8
            })
            .collect()
    }

    #[cfg(test)]
    mod tests {
        include!("../tests/kernel_matmul_test_support.rs");
        include!("../tests/kernel_matmul_i8_tests.rs");
    }
}
