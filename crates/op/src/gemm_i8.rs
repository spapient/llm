//! 对称 int8 矩阵乘：`Y[m,n] = Σ_k i8(A[m,k]) · i8(B[k,n])`。
//!
//! 收缩轴固定 [`K`]（8），点积先在 `i32` 上精确累加（对应 CUDA 的 `dp4a`：每 4 对 int8
//! 乘积累加进一个 int32），再收敛成 f32。CPU 没有 dp4a，退化成标量整数点积，但因为是整数
//! 累加，结果与设备指令逐位一致，不存在浮点舍入分歧。

use crate::model::{Buffer, Cost, Sample, ShapeContext, Values};
use diagnostic::{Code, Error};

// 收缩轴、三轴契约、形状投影与取轴辅助复用 gemm 家族共享模块（dp4a 一次 4 个，K=8 即两次）。
pub use crate::gemm::{axis, AXES, K, project_shape};

/// 代价：整数乘加按 `2*m*k*n` 计；A/B 各占 1 字节，Y 每元素 4 字节。
///
/// # Errors
///
/// 形状缺轴时返回说明。
pub fn cost(shape: &ShapeContext) -> Result<Cost, Error> {
    let m = axis(shape, 'm', "gemm_i8")?;
    let k = axis(shape, 'k', "gemm_i8")?;
    let n = axis(shape, 'n', "gemm_i8")?;
    Ok(Cost {
        flops: (2 * m * k * n) as f64,
        read: (m * k + k * n) as u64,
        write: (m * n) as u64 * 4,
    })
}

/// 确定性样本：A、B 取对称小整数（-8..7）的 int8 位模式，Y 为 f32。
pub fn sample(m: usize, n: usize) -> Sample {
    Sample {
        m,
        n,
        scalars: vec![("M", m as i64), ("K", K as i64), ("N", n as i64)],
        buffers: vec![
            Buffer {
                name: "a",
                values: Values::U8(
                    (0..m * K)
                        .map(|i| (((i * 37) % 16) as i8 - 8) as u8)
                        .collect(),
                ),
            },
            Buffer {
                name: "b",
                values: Values::U8(
                    (0..K * n)
                        .map(|i| (((i * 53) % 16) as i8 - 8) as u8)
                        .collect(),
                ),
            },
            Buffer {
                name: "y",
                values: Values::F32(vec![0.0; m * n]),
            },
        ],
    }
}

/// 固定 K=8 的整数点积：A 一行（连续）与 B 一列（按 n 跨步），i32 精确累加。
///
/// 走切片迭代且长度固定，编译器展开成 8 个标量整数乘加，无运行时循环控制。
fn dot_k(a_row: &[u8], b: &[u8], n: usize, j: usize) -> i32 {
    a_row
        .iter()
        .enumerate()
        .map(|(p, packed)| (*packed as i8) as i32 * (b[p * n + j] as i8) as i32)
        .sum()
}

/// 整数矩阵乘：逐行点积写进 `out`，参考与内核共用同一份分派。
fn matmul(a: &[u8], b: &[u8], m: usize, n: usize, out: &mut [f32]) {
    for i in 0..m {
        let row = &a[i * K..][..K];
        for j in 0..n {
            out[i * n + j] = dot_k(row, b, n, j) as f32;
        }
    }
}

/// 标量参考：整数点积后转 f32。
pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
    let m = sample.m;
    let n = sample.n;
    let a = sample.u8("a")?;
    let b = sample.u8("b")?;
    let mut out = vec![0.0; m * n];
    matmul(a, b, m, n, &mut out);
    Ok(out)
}

/// 内核本体：结果写进调用方给的缓冲。
///
/// # Errors
///
/// 缓冲尺寸不符时返回说明。
pub fn forward(a: &[u8], b: &[u8], m: usize, n: usize, out: &mut [f32]) -> diagnostic::Result<()> {
    if a.len() != m * K || b.len() != K * n || out.len() != m * n {
        return Err(Error::new(Code::Invalid, "gemm_i8 缓冲尺寸不符"));
    }
    matmul(a, b, m, n, out);
    Ok(())
}

/// 标记类型：承载 `#[model]` 语义。
pub struct GemmI8;

/// 对称 int8 矩阵乘的语义别名。
#[annotation::model(name = "gemm_i8", shape = [16, 16], dtype = "f32", skeleton = "gemm")]
pub type GemmI8Model = GemmI8;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemm_i8_matches_the_scalar_reference() {
        let sample = sample(16, 12);
        let expected = reference(&sample).expect("参考可计算");
        let mut out = vec![0.0; 16 * 12];
        forward(
            sample.u8("a").expect("a"),
            sample.u8("b").expect("b"),
            16,
            12,
            &mut out,
        )
        .expect("尺寸符");
        assert_eq!(out, expected);
    }
}
