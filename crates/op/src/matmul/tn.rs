use std::ops::Range;

use diagnostic::{Code, Error, Result};

use super::{TN_COLS, TN_K, TN_ROWS};

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
/// 分块内核相对逐行 `gemv` 不落下风的前提（实测 1.8×，见 `crates/benchmarks/src/matmul.rs`）。
///
/// 它是 `op` 里唯一的多链点积：矩阵乘 tile 与两条注意力内核（`attention` / `attention::tiled`）
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
