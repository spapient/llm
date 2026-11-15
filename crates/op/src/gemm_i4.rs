//! 对称 int4 矩阵乘：`Y[m,n] = Σ_k i4(A[m,k]) · i4(B[k,n])`。
//!
//! A、B 都是打包 nibble（每字节 2 个，低 4 位在前、高 4 位在后），对称零点 8。先把 nibble
//! 解包成有符号整数，再在 `i32` 上精确累加（对应设备侧 `mma` 的整数路径），最后转 f32。
//! CPU 没有 mma，退化为标量整数点积；整数累加逐位精确，与设备指令无舍入分歧。

use crate::model::{Buffer, Cost, Sample, ShapeContext, Values};
use diagnostic::{Code, Error};

// 收缩轴长度复用 gemm 家族共享模块。
pub use crate::gemm::K;

/// 对称零点：nibble 0..15 减去 8 得到 -8..7。
pub const ZERO: i32 = 8;

// 三轴契约、形状投影与取轴辅助复用 gemm 家族共享模块。
pub use crate::gemm::{AXES, axis, project_shape};

/// 代价：整数乘加按 `2*m*k*n` 计；A/B 为打包 nibble（每 2 元素 1 字节），Y 每元素 4 字节。
///
/// # Errors
///
/// 形状缺轴时返回说明。
pub fn cost(shape: &ShapeContext) -> Result<Cost, Error> {
    let m = axis(shape, 'm', "gemm_i4")?;
    let k = axis(shape, 'k', "gemm_i4")?;
    let n = axis(shape, 'n', "gemm_i4")?;
    Ok(Cost {
        flops: (2 * m * k * n) as f64,
        read: ((m * k + k * n) / 2) as u64,
        write: (m * n) as u64 * 4,
    })
}

/// 把打包字节解成有符号 nibble（低 4 位在前、高 4 位在后），输出长度是字节数的 2 倍。
fn unpack(packed: &[u8], out: &mut Vec<i8>) {
    out.clear();
    out.reserve(packed.len() * 2);
    for byte in packed {
        out.push((byte & 0x0F) as i8 - ZERO as i8);
        out.push(((byte >> 4) & 0x0F) as i8 - ZERO as i8);
    }
}

/// 确定性样本：A、B 为打包 nibble（确定性位模式），Y 为 f32。
pub fn sample(m: usize, n: usize) -> Sample {
    let pack = |len: usize, seed: usize| -> Vec<u8> {
        (0..len / 2)
            .map(|i| {
                let lo = ((i * seed + 1) % 16) as u8;
                let hi = ((i * seed + 9) % 16) as u8;
                lo | (hi << 4)
            })
            .collect()
    };
    Sample {
        m,
        n,
        scalars: vec![("M", m as i64), ("K", K as i64), ("N", n as i64)],
        buffers: vec![
            Buffer {
                name: "a",
                values: Values::U8(pack(m * K, 37)),
            },
            Buffer {
                name: "b",
                values: Values::U8(pack(K * n, 53)),
            },
            Buffer {
                name: "y",
                values: Values::F32(vec![0.0; m * n]),
            },
        ],
    }
}

/// 固定 K=8 的整数点积：解包后 A 一行（连续）与 B 一列（按 n 跨步），i32 精确累加。
///
/// 长度固定，编译器展开成 8 个标量整数乘加，无运行时循环控制。
fn dot_k(a_row: &[i8], b: &[i8], n: usize, j: usize) -> i32 {
    a_row
        .iter()
        .enumerate()
        .map(|(p, x)| *x as i32 * b[p * n + j] as i32)
        .sum()
}

/// 解包后的整数矩阵乘：逐行点积写进 `out`，参考与内核共用同一份分派。
fn matmul(a: &[i8], b: &[i8], m: usize, n: usize, out: &mut [f32]) {
    for i in 0..m {
        let row = &a[i * K..][..K];
        for j in 0..n {
            out[i * n + j] = dot_k(row, b, n, j) as f32;
        }
    }
}

/// 标量参考：解包后整数点积，转 f32。
pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
    let m = sample.m;
    let n = sample.n;
    let mut a = Vec::new();
    let mut b = Vec::new();
    unpack(sample.u8("a")?, &mut a);
    unpack(sample.u8("b")?, &mut b);
    let mut out = vec![0.0; m * n];
    matmul(&a, &b, m, n, &mut out);
    Ok(out)
}

/// 内核本体：结果写进调用方给的缓冲。
///
/// # Errors
///
/// 缓冲尺寸不符时返回说明。
pub fn forward(a: &[u8], b: &[u8], m: usize, n: usize, out: &mut [f32]) -> diagnostic::Result<()> {
    if a.len() != m * K / 2 || b.len() != K * n / 2 || out.len() != m * n {
        return Err(Error::new(Code::Invalid, "gemm_i4 缓冲尺寸不符"));
    }
    let mut ad = Vec::new();
    let mut bd = Vec::new();
    unpack(a, &mut ad);
    unpack(b, &mut bd);
    matmul(&ad, &bd, m, n, out);
    Ok(())
}

/// 标记类型：承载 `#[model]` 语义。
pub struct GemmI4;

/// 对称 int4 矩阵乘的语义别名。
#[annotation::model(name = "gemm_i4", shape = [16, 16], dtype = "f32", skeleton = "gemm")]
pub type GemmI4Model = GemmI4;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemm_i4_matches_the_scalar_reference() {
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
