//! 大 K 对称 int8 GEMM 的语义与精度档：`Y[m, n] = Σ_k A[m, k] · B[k, n]`，累加器 `i32`。
//! 职责：i8 矩阵乘的标量/分块出口与形状投影。骨架算术在 [`crate::matmul`]。
use diagnostic::Result;
use tensor::View;

use crate::model::{Cost, Sample, ShapeCtx, Values};

pub use super::{
    AXES, K, TILE_K, TILE_M, TILE_N, forward_scalar, matmul_scalar, matmul_tiled, project_shape,
};

/// 输出块行数：4 行共用每一条 `B` 行块的读取。
const BLOCK_ROWS: usize = 4;

/// 输出块列数：一行的累加器是 16 个 `i32`，4 行共 8 个 NEON 向量寄存器。
const BLOCK_COLS: usize = 16;

/// 分块入口：校验尺寸后走 4×16 寄存器分块。
///
/// 整数累加与遍历次序无关，因此结果与 [`matmul_scalar`] 逐位一致；收益来自 `B` 行块
/// 一次读取被 4 行复用（原 rank-1 更新每行都要重读一遍 `B`）。
///
/// # Errors
///
/// 三条缓冲长度与 `(m, n, k)` 不符时返回说明。
pub fn forward_tiled(a: View<'_, i8>, b: View<'_, i8>, out: &mut [i32]) -> Result<()> {
    let (m, k) = super::dim2(a.dims(), "matmul_i8")?;
    let (k2, n) = super::dim2(b.dims(), "matmul_i8")?;
    if k != k2 {
        return Err(diagnostic::Error::new(
            diagnostic::Code::Invalid,
            format!("matmul_i8 收缩轴不符：a·k={k}，b·k={k2}"),
        ));
    }
    super::check_dims(a.as_slice(), b.as_slice(), m, n, k, out)?;
    blocked(a.as_slice(), b.as_slice(), m, n, k, out);
    Ok(())
}

/// 4×16 块 + 行尾/列尾：块内累加器留在寄存器，尾块退回标量但结果不变。
fn blocked(a: &[i8], b: &[i8], m: usize, n: usize, k: usize, out: &mut [i32]) {
    let mut i0 = 0;
    while i0 + BLOCK_ROWS <= m {
        let mut j0 = 0;
        while j0 + BLOCK_COLS <= n {
            block(a, b, out, (i0, j0), n, k);
            j0 += BLOCK_COLS;
        }
        edge_cols(a, b, out, i0, j0, n, k);
        i0 += BLOCK_ROWS;
    }
    edge_rows(a, b, out, i0, m, n, k);
}

/// 一个 `4×16` 输出块：`B` 的每个 p 行块只读一次、只拓宽一次，再广播给 4 行。
fn block(a: &[i8], b: &[i8], out: &mut [i32], at: (usize, usize), n: usize, k: usize) {
    let (i0, j0) = at;
    let mut acc = [[0i32; BLOCK_COLS]; BLOCK_ROWS];
    for p in 0..k {
        let wide = widen(&b[p * n + j0..][..BLOCK_COLS]);
        for r in 0..BLOCK_ROWS {
            let x = i32::from(a[(i0 + r) * k + p]);
            for j in 0..BLOCK_COLS {
                acc[r][j] += x * wide[j];
            }
        }
    }
    for r in 0..BLOCK_ROWS {
        out[(i0 + r) * n + j0..][..BLOCK_COLS].copy_from_slice(&acc[r]);
    }
}

/// `B` 的一行块拓宽一次：这里就是「只拓宽一次」的唯一位置，逐行重做会让向量化只剩一半。
fn widen(row: &[i8]) -> [i32; BLOCK_COLS] {
    let mut wide = [0i32; BLOCK_COLS];
    for (slot, &value) in wide.iter_mut().zip(row) {
        *slot = i32::from(value);
    }
    wide
}

/// 列尾（`j0..n`）：块内按行标量累加，结果与分块逐位一致。
fn edge_cols(a: &[i8], b: &[i8], out: &mut [i32], i0: usize, mut j0: usize, n: usize, k: usize) {
    while j0 < n {
        for r in 0..BLOCK_ROWS {
            let mut acc = 0i32;
            for p in 0..k {
                acc += i32::from(a[(i0 + r) * k + p]) * i32::from(b[p * n + j0]);
            }
            out[(i0 + r) * n + j0] = acc;
        }
        j0 += 1;
    }
}

/// 行尾（`i0..m`）：不足 [`BLOCK_ROWS`] 的行逐元素累加。
fn edge_rows(a: &[i8], b: &[i8], out: &mut [i32], mut i0: usize, m: usize, n: usize, k: usize) {
    while i0 < m {
        for j in 0..n {
            let mut acc = 0i32;
            for p in 0..k {
                acc += i32::from(a[i0 * k + p]) * i32::from(b[p * n + j]);
            }
            out[i0 * n + j] = acc;
        }
        i0 += 1;
    }
}

/// 标量代价：`2·m·K·n` FLOP；输入为 i8（每元素 1 字节），输出 f32（每元素 4 字节）。
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

/// 显式收缩轴长度的确定性样本：与 [`sample`] 同构，只有 `K` 由调用方给定。
pub fn sample_k(m: usize, n: usize, k: usize) -> Sample {
    super::sample_of(
        m,
        n,
        k,
        Values::U8(deterministic_i8(m * k, 1)),
        Values::U8(deterministic_i8(k * n, 2)),
    )
}

/// 标量参考：读 `a` / `b`（u8 承载的有符号码），i32 精确累加后转 f32 输出。
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

/// 语义表由注解生成：生产形状 `256×256`，收缩轴 [`K`]，`i32` 精确累加后输出 f32。
#[annotation::model(name = "matmul_i8", shape = [256, 256], dtype = "f32", skeleton = "matmul")]
pub type MatmulI8 = MatmulI8Op;

crate::operator_here!(MatmulI8Op, "matmul_i8", 2, (256, 256), false, reference);

/// 确定性 i8 采样：覆盖正负与零，`(len, salt)` 固定即序列固定。
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
    include!("../../../tests/kernel_matmul_i8_tests.rs");
}
