//! 矩阵乘：`Y[m,n] = sum_k A[m,k] * B[k,n]`。
//!
//! CPU 片段没有 MMA 原语，本实现退化成标量三重循环（语义正确，SIMD/tile 化在 P4 后端补），
//! 收缩轴固定为 `K = 8` 以控制标量成本。本模块自带形状、代价、样本、参考与遍历。

use crate::model::{Buffer, Cost, Sample, ShapeContext, Values};
use diagnostic::{Code, Error};

// 收缩轴、三轴契约、形状投影与取轴辅助复用 gemm 家族共享模块。
pub use crate::gemm::{AXES, K, axis, project_shape};

/// 标量代价：`2*m*k*n` 次浮点运算（乘加各一），读 A/B、写 Y。
///
/// # Errors
///
/// 形状缺轴时返回说明。
pub fn cost(shape: &ShapeContext) -> Result<Cost, Error> {
    let m = axis(shape, 'm', "gemm")?;
    let k = axis(shape, 'k', "gemm")?;
    let n = axis(shape, 'n', "gemm")?;
    Ok(Cost {
        flops: (2 * m * k * n) as f64,
        read: ((m * k + k * n) as u64) * 4,
        write: ((m * n) as u64) * 4,
    })
}

/// 确定性样本：A（m×k）、B（k×n）、Y（m×n）。
pub fn sample(m: usize, n: usize) -> Sample {
    Sample {
        m,
        n,
        scalars: vec![("M", m as i64), ("K", K as i64), ("N", n as i64)],
        buffers: vec![
            Buffer {
                name: "a",
                values: Values::F32((0..m * K).map(|i| ((i % 5) as f32) * 0.25 - 0.5).collect()),
            },
            Buffer {
                name: "b",
                values: Values::F32((0..K * n).map(|i| ((i % 7) as f32) * 0.2 - 0.4).collect()),
            },
            Buffer {
                name: "y",
                values: Values::F32(vec![0.0; m * n]),
            },
        ],
    }
}

/// 标量参考：三重循环，与内核同一公式。
pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
    let m = sample.m;
    let n = sample.n;
    let a = sample.f32("a")?;
    let b = sample.f32("b")?;
    let mut out = vec![0.0; m * n];
    matmul(a, b, m, n, &mut out);
    Ok(out)
}

/// 内核本体：结果写进调用方给的缓冲。
///
/// # Errors
///
/// 缓冲尺寸不符时返回说明。
pub fn forward(
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    out: &mut [f32],
) -> diagnostic::Result<()> {
    if [a.len(), b.len(), out.len()] != [m * K, K * n, m * n] {
        return Err(Error::new(Code::Invalid, "gemm 缓冲尺寸不符"));
    }
    matmul(a, b, m, n, out);
    Ok(())
}

/// 三重循环骨架：`reference` 与 `forward` 共用，外层遍历只写一份；固定 K 的点积落在
/// [`dot_k`]。
fn matmul(a: &[f32], b: &[f32], m: usize, n: usize, out: &mut [f32]) {
    for i in 0..m {
        let row = &a[i * K..i * K + K];
        for j in 0..n {
            out[i * n + j] = dot_k(row, b, n, j);
        }
    }
}

/// 固定 K=8 点积：a 的一行（连续）与 b 的一列（按 n 跨步）。
///
/// 走切片迭代且长度固定，编译器把它展开成 8 个标量乘加，没有运行时循环控制开销，
/// 也消除了 reference 与 forward 之间重复的内层循环。
fn dot_k(a_row: &[f32], b: &[f32], n: usize, j: usize) -> f32 {
    a_row
        .iter()
        .enumerate()
        .map(|(p, x)| x * b[p * n + j])
        .sum()
}

/// 标记类型：承载 `#[model]` 语义。
pub struct Gemm;

/// 矩阵乘的具体语义别名。
#[annotation::model(name = "gemm_f32", shape = [16, 16], dtype = "f32", skeleton = "gemm")]
pub type GemmF32 = Gemm;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemm_matches_the_scalar_reference() {
        let sample = sample(16, 12);
        let expected = reference(&sample).expect("参考可计算");
        let mut out = vec![0.0; 16 * 12];
        forward(
            sample.f32("a").expect("a"),
            sample.f32("b").expect("b"),
            16,
            12,
            &mut out,
        )
        .expect("尺寸符");
        assert_eq!(out, expected);
    }
}
