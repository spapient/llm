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

use std::ops::Range;

use diagnostic::{Code, Error, Result};

use crate::model::{AxisRole, AxisSpec, Buffer, Cost, Sample, ShapeContext, Values};

/// 输出 tile 行数。
pub const TILE_M: usize = 64;

/// 输出 tile 列数。
pub const TILE_N: usize = 64;

/// 收缩轴 tile 宽度。
pub const TILE_K: usize = 32;

/// 线性层布局（`x [rows, k]` × `w [n, k]`）的 tile 行数。
///
/// 与 [`TILE_M`] 分开是因为两个核的复用轴不同：`[m, k] × [k, n]` 里 B 的列块沿 `k` 走，而线性层把
/// 权重整行 `[n, k]` 放在一处，行块才是一次驻留的单位。
pub const TN_ROWS: usize = 16;

/// 线性层 tile 的输出列数：一列权重块要同时在累加器里驻留，不能大于 `TN_ROWS × TN_COLS` 的 L1 预算。
pub const TN_COLS: usize = 32;

/// 线性层 tile 的收缩块宽度。
///
/// 收缩块越宽，累加器"读出→加→写回"的次数越少（次数是 `k / TN_K`），而块内全是连续点积；
/// 256 既让循环开销摊到 256 次乘加上，又让一行权重块（`TN_COLS × TN_K`）留在 L1。
pub const TN_K: usize = 256;

/// 标量参考：`out[i,j] = Σ_p a[i,p]·b[p,j]`，每个元素按 `p` 顺序独立累加。
///
/// 不读 `out` 的初值（直接赋值），是 tile 实现与后续 kernel 的对拍真值。
pub fn matmul_scalar(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, out: &mut [f32]) {
    for i in 0..m {
        let arow = &a[i * k..][..k];
        let orow = &mut out[i * n..][..n];
        for j in 0..n {
            let mut acc = 0.0f32;
            for (p, &x) in arow.iter().enumerate() {
                acc += x * b[p * n + j];
            }
            orow[j] = acc;
        }
    }
}

/// 分块 tile：按 [`TILE_M`]×[`TILE_N`] 切输出、[`TILE_K`] 切收缩轴，块内 rank-1 更新。
///
/// 先清空 `out` 再逐块累加。A 行块（TILE_M×TILE_K）在 `bj` 循环内复用、B 列块
/// （TILE_K×TILE_N）在 `i` 循环内复用，把跨块 cache miss 压到 tile 边界。
pub fn matmul_tiled(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, out: &mut [f32]) {
    out.fill(0.0);
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
///
/// 三条区间一起决定「乘哪一块」，分开放会让调用点出现三对含义相近的 `usize` 实参；包成一个值之后
/// 传参只剩「哪个 tile」这一件事，块内累加也就不必知道它是第几块。
struct Tile {
    rows: Range<usize>,
    contraction: Range<usize>,
    cols: Range<usize>,
}

/// 块内 rank-1 更新：`out[i, j] += Σ_p a[i, p]·b[p, j]`，三条下标都取 `tile` 给的区间。
///
/// `n` / `k` 是整块的行宽与收缩轴长度：区间是绝对下标，行主序下 `[i, j]` 仍是 `i * n + j`。
fn accumulate_tile(a: &[f32], b: &[f32], out: &mut [f32], tile: &Tile, n: usize, k: usize) {
    for i in tile.rows.clone() {
        let start = i * k + tile.contraction.start;
        let arow = &a[start..i * k + tile.contraction.end];
        for (p, &x) in arow.iter().enumerate() {
            let base = (tile.contraction.start + p) * n;
            let brow = &b[base + tile.cols.start..base + tile.cols.end];
            let orow = &mut out[i * n + tile.cols.start..i * n + tile.cols.end];
            for (o, &y) in orow.iter_mut().zip(brow) {
                *o += x * y;
            }
        }
    }
}

/// 三条缓冲的尺寸校验，两版 `forward` 共用。
fn check_dims(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, out: &[f32]) -> Result<()> {
    if a.len() != m * k || b.len() != k * n || out.len() != m * n {
        return Err(Error::new(
            Code::Invalid,
            format!(
                "GEMM 缓冲尺寸不符：a={}，b={}，out={}，期望 m·k={}，k·n={}，m·n={}",
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
pub fn forward_scalar(
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
    out: &mut [f32],
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
pub fn forward_tiled(
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
    out: &mut [f32],
) -> Result<()> {
    check_dims(a, b, m, n, k, out)?;
    matmul_tiled(a, b, m, n, k, out);
    Ok(())
}

/// 标量参考（线性层布局）：`out[r, j] = Σ_p x[r, p]·w[j, p]`，每个元素按 `p` 顺序独立累加。
///
/// 不读 `out` 的初值（直接赋值），是 [`matmul_tn_tiled`] 的对拍真值。
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

/// 分块 tile（线性层布局）：[`TN_ROWS`] 行 × [`TN_COLS`] 列的输出块，收缩轴按 [`TN_K`] 切。
///
/// 输出列块在最外层，于是每个权重列块（`TN_COLS × k`，prefill 下约 1 MiB）在一次外循环里被行块反复
/// 复用、只从内存读一遍；行块在内层，块内累加器是 [`TN_ROWS`]×[`TN_COLS`] 的定长栈缓冲。逐行调
/// `gemv` 的做法把同一份权重按行数重复读，那个倍数正好是本核省掉的量。
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
                let tile = TnTile {
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

/// 线性层布局的一个 tile：输出行、收缩轴、输出列三条区间（与 [`Tile`] 同构，含义不同）。
struct TnTile {
    rows: Range<usize>,
    contraction: Range<usize>,
    cols: Range<usize>,
}

/// 块内累加：`acc[row, j] += Σ_p x[row, p]·w[j, p]`，`acc` 的行宽是 [`TN_COLS`]。
///
/// `k` 是权重与输入的行宽：区间是绝对下标，行主序下 `[row, p]` 仍是 `row * k + p`。
///
/// 收缩块取 [`TN_K`] 是这一层的关键：累加器每块只被读改写一次（整段 `k` 上共 `k / TN_K` 次），
/// 块内全是连续点积；块再窄就变成「累加器往返」而不是「乘加」。
fn accumulate_tn_tile(x: &[f32], w: &[f32], acc: &mut [f32], tile: &TnTile, k: usize) {
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
///
/// 尾数（`len` 不是 `LANES` 的倍数）并入标量尾部——链内顺序与标量路径不同，只会带来末位舍入差，
/// 而公开对拍容差（1e-3 相对误差）足以吸收它；换来的是内层 `LANES` 倍指令级并行，这一项才是
/// 分块内核相对逐行 `gemv` 不落下风的前提（实测 1.8×，见 `crates/benches/criterion/matmul_f32.rs`）。
///
/// 它是 `op` 里唯一的多链点积：矩阵乘 tile 与两条注意力内核（`attention` / `attention_tiled`）
/// 都调这一份，否则同一个归约循环会被写成四份、向量化程度也四份不一样。
pub(crate) fn dot_lanes(a: &[f32], b: &[f32]) -> f32 {
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
pub fn forward_tn_scalar(
    x: &[f32],
    w: &[f32],
    rows: usize,
    n: usize,
    k: usize,
    out: &mut [f32],
) -> Result<()> {
    check_tn_dims(x, w, rows, n, k, out)?;
    matmul_tn_scalar(x, w, rows, n, k, out);
    Ok(())
}

/// 线性层布局的 tile 入口：校验尺寸后写 `out`。这是服务侧行批投影的唯一生产路径。
///
/// # Errors
///
/// 三条缓冲长度与 `(rows, n, k)` 不符时返回说明。
pub fn forward_tn_tiled(
    x: &[f32],
    w: &[f32],
    rows: usize,
    n: usize,
    k: usize,
    out: &mut [f32],
) -> Result<()> {
    check_tn_dims(x, w, rows, n, k, out)?;
    matmul_tn_tiled(x, w, rows, n, k, out);
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

/// 收缩轴长度（生产档）：prefill 主链路 GEMM 的 K。
///
/// 与 [`crate::gemm`] 的 `K=8` 固定档同构：把大 K 作为算子语义的生产形状写入 [`MODEL`]，
/// 使 `nn::Bounds` / roofline 对 256×256×4096 的代价分子取 2·m·K·n 而不是 gemm 的 2·m·8·n。
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
/// 轴表不合法时由 [`ShapeContext::new`] 报出。
pub fn project_shape(m: usize, n: usize) -> Result<ShapeContext> {
    ShapeContext::new([('m', m), ('k', K), ('n', n)])
}

/// 标量代价：`2·m·K·n` FLOP，读 `(m·K + K·n)·4` 字节，写 `m·n·4` 字节。
///
/// # Errors
///
/// 形状上下文缺轴时返回带算子名的说明。
pub fn cost(shape: &ShapeContext) -> Result<Cost> {
    let axis = |label: char| {
        shape
            .get(label)
            .ok_or_else(|| Error::new(Code::Invalid, format!("matmul_f32 形状缺轴 `{label}`")))
    };
    let m = axis('m')?;
    let k = axis('k')?;
    let n = axis('n')?;
    Ok(Cost {
        flops: (2 * m * k * n) as f64,
        read: ((m * k + k * n) * 4) as u64,
        write: (m * n * 4) as u64,
    })
}

/// 确定性样本：`a` / `b` 由同一份采样序列给出，`y` 为零缓冲，标量 `M`/`K`/`N` 与 `.k` 签名一致。
pub fn sample(m: usize, n: usize) -> Sample {
    Sample {
        m,
        n,
        scalars: vec![("M", m as i64), ("K", K as i64), ("N", n as i64)],
        buffers: vec![
            Buffer {
                name: "a",
                values: Values::F32(deterministic(m * K, 1)),
            },
            Buffer {
                name: "b",
                values: Values::F32(deterministic(K * n, 2)),
            },
            Buffer {
                name: "y",
                values: Values::F32(vec![0.0; m * n]),
            },
        ],
    }
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
///
/// 手写常量的版本已删：`#[annotation::model]` 是它与注册表之间唯一的绑定点，两处各写一份
/// 就会出现「注册名改了、语义表没改」这种编译得过却绑错算子的状态。
#[annotation::model(name = "matmul_f32", shape = [256, 256], dtype = "f32", skeleton = "matmul")]
pub type MatmulF32 = Matmul;

crate::operator_here!(Matmul, "matmul_f32", (256, 256), false, reference);

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
    use super::*;

    /// 确定性采样：伪随机但无外部依赖，同一个 `(len, salt)` 永远同一个序列。
    fn sample(len: usize, salt: u32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let x = (i as u32)
                    .wrapping_mul(2654435761)
                    .wrapping_add(salt.wrapping_mul(40503));
                ((x >> 8) as f32) / 256.0 - 32.0
            })
            .collect()
    }

    fn assert_tiled_matches(m: usize, n: usize, k: usize) {
        let a = sample(m * k, 1);
        let b = sample(k * n, 2);
        let mut expected = vec![0.0f32; m * n];
        let mut actual = vec![0.0f32; m * n];
        forward_scalar(&a, &b, m, n, k, &mut expected).expect("参考可计算");
        forward_tiled(&a, &b, m, n, k, &mut actual).expect("tile 可计算");
        // K=4096 级 f32 累加的顺序误差在 1e-3 量级，rtol 取 1e-3、atol 取 1e-3。
        close_enough(&actual, &expected, 1e-3, 1e-3).expect("tile 与标量参考在容差内一致");
    }

    #[test]
    fn tiled_matches_scalar_on_standard_shape() {
        assert_tiled_matches(256, 256, 4096);
    }

    #[test]
    fn tiled_matches_scalar_on_tile_multiples() {
        assert_tiled_matches(128, 64, 96);
    }

    #[test]
    fn tiled_matches_scalar_on_ragged_edges() {
        // 非 tile 整数倍：末块带守卫（TILE_M/N/K 都切出残边）。
        assert_tiled_matches(100, 70, 4093);
    }

    #[test]
    fn tiled_matches_scalar_on_trivial_shapes() {
        assert_tiled_matches(1, 1, 1);
        assert_tiled_matches(1, 8, 3);
        assert_tiled_matches(8, 1, 3);
    }

    /// 线性层布局对拍：同一份输入与权重，标量参考与 tile 在 IEEE 容差内一致。
    fn assert_tn_matches(rows: usize, n: usize, k: usize) {
        let x = sample(rows * k, 3);
        let w = sample(n * k, 4);
        let mut expected = vec![0.0f32; rows * n];
        let mut actual = vec![0.0f32; rows * n];
        forward_tn_scalar(&x, &w, rows, n, k, &mut expected).expect("参考可计算");
        forward_tn_tiled(&x, &w, rows, n, k, &mut actual).expect("tile 可计算");
        close_enough(&actual, &expected, 1e-3, 1e-3).expect("线性层 tile 与标量参考一致");
    }

    #[test]
    fn tn_tiled_matches_scalar_on_prefill_shape() {
        // 行批 256、输出 128、收缩轴 1024：prefill 投影的代表形状。
        assert_tn_matches(256, 128, 1024);
    }

    #[test]
    fn tn_tiled_matches_scalar_on_tile_multiples() {
        assert_tn_matches(TN_ROWS, TN_COLS, TN_K);
    }

    #[test]
    fn tn_tiled_matches_scalar_on_ragged_edges() {
        // 非 tile 整数倍：行块、列块、收缩块三处都切出残边。
        assert_tn_matches(TN_ROWS + 3, TN_COLS + 5, TN_K + 7);
    }

    #[test]
    fn tn_tiled_matches_scalar_on_single_row() {
        // decode 口径：一行输入也必须与 `gemv_f32` 同一算式（同一份点积顺序）。
        assert_tn_matches(1, 8, 13);
    }

    #[test]
    fn tn_empty_contraction_yields_zeros() {
        let mut out = vec![1.0f32; 6];
        forward_tn_tiled(&[], &[], 2, 3, 0, &mut out).expect("k=0 合法");
        assert_eq!(out, vec![0.0f32; 6], "空收缩轴输出必须全零");
    }

    #[test]
    fn tn_forward_rejects_mismatched_buffers() {
        let x = vec![0.0f32; 8];
        let w = vec![0.0f32; 8];
        let mut out = vec![0.0f32; 8];
        assert!(
            forward_tn_tiled(&x, &w, 3, 3, 3, &mut out).is_err(),
            "尺寸不符必须报错"
        );
        assert!(
            forward_tn_scalar(&x, &w, 2, 2, 2, &mut out).is_err(),
            "标量入口也必须报错"
        );
    }

    #[test]
    fn empty_contraction_yields_zeros() {
        let mut out = vec![1.0f32; 4];
        forward_tiled(&[], &[], 2, 2, 0, &mut out).expect("k=0 合法");
        assert_eq!(out, vec![0.0f32; 4], "空收缩轴输出必须全零");
    }

    #[test]
    fn forward_rejects_mismatched_buffers() {
        let a = vec![0.0f32; 8];
        let b = vec![0.0f32; 8];
        let mut out = vec![0.0f32; 8];
        assert!(
            forward_scalar(&a, &b, 3, 3, 3, &mut out).is_err(),
            "尺寸不符必须报错"
        );
    }

    /// E10 计算可预测性（`#[ignore]`：
    /// `cargo test -p llm-op --lib -- e10_predictability --release --ignored --nocapture`）。
    ///
    /// 口径与 §10.10 E10 卡一致：预热后对固定窗口（20 次 512×512×512 tile 矩阵乘）计 120 个
    /// 窗口墙钟，`CV_t = σ_t/μ_t`，并报 P50/P95/P99。热节流项本机无传感器轨迹，单列不混入。
    #[test]
    #[ignore = "E10 可预测性基准：cargo test -p llm-op --lib -- e10_predictability --release --ignored --nocapture"]
    fn e10_predictability() {
        let (m, n, k) = (512usize, 512usize, 512usize);
        let a = sample(m * k, 11);
        let b = sample(k * n, 12);
        let mut out = vec![0.0f32; m * n];

        for _ in 0..20 {
            matmul_tiled(&a, &b, m, n, k, &mut out);
        }

        let iters_per_window = 20usize;
        let windows = 120usize;
        let mut times = Vec::with_capacity(windows);
        for _ in 0..windows {
            let t = std::time::Instant::now();
            for _ in 0..iters_per_window {
                matmul_tiled(&a, &b, m, n, k, &mut out);
                std::hint::black_box(&out);
            }
            times.push(t.elapsed().as_secs_f64());
        }

        let count = times.len() as f64;
        let mean = times.iter().sum::<f64>() / count;
        let var = times.iter().map(|t| (t - mean) * (t - mean)).sum::<f64>() / (count - 1.0);
        let std = var.sqrt();
        let cv = std / mean;
        let mut sorted = times.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = |p: f64| -> f64 { sorted[((p * (sorted.len() as f64 - 1.0)).round()) as usize] };
        println!(
            "E10 CV={:.4}%（mean={:.3} ms/window std={:.3} ms P50={:.3} P95={:.3} P99={:.3} windows={windows} iters/window={iters_per_window} shape={m}x{n}x{k}）",
            cv * 100.0,
            mean * 1e3,
            std * 1e3,
            q(0.50) * 1e3,
            q(0.95) * 1e3,
            q(0.99) * 1e3,
        );
    }
}
