//! 大 K 对称 int8 GEMM（可变收缩轴）：Y[m,n] = sum_k A[m,k] * B[k,n]。
//!
//! 与 crate::gemm 的 K=8 量化骨架互补：收缩轴 k 由调用方给定。累加器用 i32 精确累加
//! （对应 CUDA dp4a），tile 化只改遍历次序、不改数值结果（整数加法交换律），因此
//! matmul_tiled 与 matmul_scalar 对拍逐位相等，不做容差。

use diagnostic::{Code, Error, Result};

/// 输出 tile 行数。
pub const TILE_M: usize = 64;

/// 输出 tile 列数。
pub const TILE_N: usize = 64;

/// 收缩轴 tile 宽度。
pub const TILE_K: usize = 32;

/// 标量参考：out[i,j] = sum_p a[i,p]*b[p,j]，每个元素按 p 顺序独立累加。
///
/// 不读 out 的初值（直接赋值），是 tile 实现的对拍真值。
pub fn matmul_scalar(a: &[i8], b: &[i8], m: usize, n: usize, k: usize, out: &mut [i32]) {
    for i in 0..m {
        let arow = &a[i * k..][..k];
        let orow = &mut out[i * n..][..n];
        for j in 0..n {
            let mut acc = 0i32;
            for (p, &x) in arow.iter().enumerate() {
                acc += x as i32 * b[p * n + j] as i32;
            }
            orow[j] = acc;
        }
    }
}

/// 分块 tile：按 TILE_M x TILE_N 切输出、TILE_K 切收缩轴，块内 rank-1 更新。
///
/// 先清空 out 再逐块累加；A 行块在 bj 循环内复用、B 列块在 i 循环内复用。
pub fn matmul_tiled(a: &[i8], b: &[i8], m: usize, n: usize, k: usize, out: &mut [i32]) {
    out.fill(0);
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
    rows: std::ops::Range<usize>,
    contraction: std::ops::Range<usize>,
    cols: std::ops::Range<usize>,
}

/// 块内 rank-1 更新：out[i,j] += sum_p a[i,p]*b[p,j]，下标都取 tile 给的区间。
///
/// n / k 是整块的行宽与收缩轴长度；行主序下 [i,j] 仍是 i * n + j。
fn accumulate_tile(a: &[i8], b: &[i8], out: &mut [i32], tile: &Tile, n: usize, k: usize) {
    for i in tile.rows.clone() {
        let start = i * k + tile.contraction.start;
        let arow = &a[start..i * k + tile.contraction.end];
        for (p, &x) in arow.iter().enumerate() {
            let base = (tile.contraction.start + p) * n;
            let brow = &b[base + tile.cols.start..base + tile.cols.end];
            let orow = &mut out[i * n + tile.cols.start..i * n + tile.cols.end];
            for (o, &y) in orow.iter_mut().zip(brow) {
                *o += x as i32 * y as i32;
            }
        }
    }
}

/// 三条缓冲的尺寸校验，两版 forward 共用。
fn check_dims(a: &[i8], b: &[i8], m: usize, n: usize, k: usize, out: &[i32]) -> Result<()> {
    if a.len() != m * k || b.len() != k * n || out.len() != m * n {
        return Err(Error::new(
            Code::Invalid,
            format!(
                "i8 GEMM 缓冲尺寸不符：a={}，b={}，out={}，期望 m*k={}，k*n={}，m*n={}",
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

/// 标量参考入口：校验尺寸后写 out。
///
/// # Errors
///
/// 三条缓冲长度与 (m, n, k) 不符时返回说明。
pub fn forward_scalar(
    a: &[i8],
    b: &[i8],
    m: usize,
    n: usize,
    k: usize,
    out: &mut [i32],
) -> Result<()> {
    check_dims(a, b, m, n, k, out)?;
    matmul_scalar(a, b, m, n, k, out);
    Ok(())
}

/// tile 实现入口：校验尺寸后写 out。
///
/// # Errors
///
/// 三条缓冲长度与 (m, n, k) 不符时返回说明。
pub fn forward_tiled(
    a: &[i8],
    b: &[i8],
    m: usize,
    n: usize,
    k: usize,
    out: &mut [i32],
) -> Result<()> {
    check_dims(a, b, m, n, k, out)?;
    matmul_tiled(a, b, m, n, k, out);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 确定性 i8 样本：覆盖正负与零，(len, salt) 固定即序列固定。
    fn sample(len: usize, salt: i32) -> Vec<i8> {
        (0..len)
            .map(|i| {
                let x = (i as i32).wrapping_mul(31).wrapping_add(salt);
                (x % 64 - 32) as i8
            })
            .collect()
    }

    fn assert_tiled_matches(m: usize, n: usize, k: usize) {
        let a = sample(m * k, 7);
        let b = sample(k * n, 11);
        let mut expected = vec![0i32; m * n];
        let mut actual = vec![0i32; m * n];
        forward_scalar(&a, &b, m, n, k, &mut expected).expect("参考可计算");
        forward_tiled(&a, &b, m, n, k, &mut actual).expect("tile 可计算");
        assert_eq!(actual, expected, "i8 tile 必须与标量参考逐位一致");
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
        let mut out = vec![1i32; 4];
        forward_tiled(&[], &[], 2, 2, 0, &mut out).expect("k=0 合法");
        assert_eq!(out, vec![0i32; 4], "空收缩轴输出必须全零");
    }

    #[test]
    fn forward_rejects_mismatched_buffers() {
        let a = vec![0i8; 8];
        let b = vec![0i8; 8];
        let mut out = vec![0i32; 8];
        assert!(
            forward_scalar(&a, &b, 3, 3, 3, &mut out).is_err(),
            "尺寸不符必须报错"
        );
    }
}
