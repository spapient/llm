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

use diagnostic::{Code, Error, Result};

/// 输出 tile 行数。
pub const TILE_M: usize = 64;

/// 输出 tile 列数。
pub const TILE_N: usize = 64;

/// 收缩轴 tile 宽度。
pub const TILE_K: usize = 32;

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
                for i in bi..i_end {
                    let arow = &a[i * k + bk..i * k + k_end];
                    for (p, &x) in arow.iter().enumerate() {
                        let brow = &b[(bk + p) * n + bj..(bk + p) * n + j_end];
                        let orow = &mut out[i * n + bj..i * n + j_end];
                        for (o, &y) in orow.iter_mut().zip(brow) {
                            *o += x * y;
                        }
                    }
                }
                bj += TILE_N;
            }
            bk += TILE_K;
        }
        bi += TILE_M;
    }
}

/// 三条缓冲的尺寸校验，两版 `forward` 共用。
fn check_dims(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, out: &[f32]) -> Result<()> {
    if a.len() != m * k || b.len() != k * n || out.len() != m * n {
        return Err(Error::new(
            Code::Invalid,
            format!("GEMM 缓冲尺寸不符：a={}，b={}，out={}，期望 m·k={}，k·n={}，m·n={}",
                a.len(), b.len(), out.len(), m * k, k * n, m * n),
        ));
    }
    Ok(())
}

/// 标量参考入口：校验尺寸后写 `out`。
///
/// # Errors
///
/// 三条缓冲长度与 `(m, n, k)` 不符时返回说明。
pub fn forward_scalar(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, out: &mut [f32]) -> Result<()> {
    check_dims(a, b, m, n, k, out)?;
    matmul_scalar(a, b, m, n, k, out);
    Ok(())
}

/// tile 实现入口：校验尺寸后写 `out`。
///
/// # Errors
///
/// 三条缓冲长度与 `(m, n, k)` 不符时返回说明。
pub fn forward_tiled(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, out: &mut [f32]) -> Result<()> {
    check_dims(a, b, m, n, k, out)?;
    matmul_tiled(a, b, m, n, k, out);
    Ok(())
}

/// IEEE 容差对拍：`|a - e| ≤ atol + rtol·|e|`。`rtol/atol` 与 lld §10.4 的 IEEE 口径同源。
pub fn close_enough(actual: &[f32], expected: &[f32], rtol: f32, atol: f32) -> Result<()> {
    if actual.len() != expected.len() {
        return Err(Error::new(
            Code::Invalid,
            format!("对拍长度不符：实际 {}，参考 {}", actual.len(), expected.len()),
        ));
    }
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        let tol = atol + rtol * e.abs();
        if (a - e).abs() > tol {
            return Err(Error::new(
                Code::Invalid,
                format!("第 {i} 个元素 {a} 与参考 {e} 差 {}，超过容差 {tol}", (a - e).abs()),
            ));
        }
    }
    Ok(())
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
        assert!(forward_scalar(&a, &b, 3, 3, 3, &mut out).is_err(), "尺寸不符必须报错");
    }
}
