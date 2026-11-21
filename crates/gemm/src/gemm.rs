//! GEMM 家族共享的轴契约、形状投影与格式无关的矩阵乘骨架。

use crate::einsum::Bits;
use crate::gemv::Element;
use crate::model::{AxisRole, AxisSpec, Buffer, Cost, Sample, ShapeContext, Values};
use crate::quant::Row;
use diagnostic::{Code, Error};

/// 收缩轴长度（固定档）。
pub const K: usize = 8;

/// 输出列块宽度：B 的一个列块（`K × TILE_N` 个码）解包一次后被整段行复用。
pub const TILE_N: usize = 64;

/// 每字节打包码数：位宽 ≤ 8 时按 `8 / bits` 算；8 位逐元素为 1。
pub const fn codes_per_byte(bits: Bits) -> usize {
    8 / bits.bits() as usize
}

/// 三轴：`m`/`k` 输入，`n` 输出。
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

/// 形状：输出 `m×n`，收缩轴 `k` 固定。
///
/// # Errors
///
/// 当前实现不会失败。
pub fn project_shape(m: usize, n: usize) -> Result<ShapeContext, Error> {
    ShapeContext::new([('m', m), ('k', K), ('n', n)])
}

/// 从形状上下文取一条轴；缺轴时按 `op` 标签报 [`Code::Invalid`]。
pub fn axis(shape: &ShapeContext, label: char, op: &str) -> Result<usize, Error> {
    shape
        .get(label)
        .ok_or_else(|| Error::new(Code::Invalid, format!("{op} 形状缺轴 `{label}`")))
}

/// 矩阵乘格式：一条 GEMM 里随精度变化的那部分。
pub trait Gemm {
    /// 样本缓冲里的元素类型（[`Values`] 的两个闭集变体）。
    type Elem: Element;

    /// 诊断里点名的算子名：缺轴、尺寸不符时进错误文案。
    const OP: &'static str;

    /// A / B 输入元素位宽；输出恒为 f32。
    const BITS: Bits;

    /// 打包格式的静态配置（[`Row`]）：每字节码数、零点、样本偏移、对称码表单点收敛。
    fn config() -> Option<Row> {
        None
    }

    /// A（`m×K`）确定性取值：打包格式走 `packed_sample_a`；逐元素格式自行实现本函数。
    fn sample_a(m: usize) -> Vec<Self::Elem> {
        let Some(row) = Self::config() else {
            return Vec::new();
        };
        let conf = packed::Conf::from(&row);
        Self::Elem::from_bytes(packed::packed_sample_a(m, &conf))
    }

    /// B（`K×n`）确定性取值（口径同 [`sample_a`](Self::sample_a)）。
    fn sample_b(n: usize) -> Vec<Self::Elem> {
        let Some(row) = Self::config() else {
            return Vec::new();
        };
        let conf = packed::Conf::from(&row);
        Self::Elem::from_bytes(packed::packed_sample_b(n, &conf))
    }

    /// A 缓冲应有的元素个数（打包后按 [`Elem`](Self::Elem) 计）。
    fn a_len(m: usize) -> usize {
        let Some(row) = Self::config() else { return 0 };
        packed::packed_a_len(m, &packed::Conf::from(&row))
    }

    /// B 缓冲应有的元素个数。
    fn b_len(n: usize) -> usize {
        let Some(row) = Self::config() else { return 0 };
        packed::packed_b_len(n, &packed::Conf::from(&row))
    }

    /// 输出位置 `(i, j)` 的点积：A 第 `i` 行（连续 `K` 个元素）与 B 第 `j` 列（跨步 `n`）。
    fn dot(a: &[Self::Elem], b: &[Self::Elem], n: usize, i: usize, j: usize) -> f32 {
        let Some(row) = Self::config() else {
            return 0.0;
        };
        let Some(a) = Element::borrow_u8(a) else {
            return 0.0;
        };
        let Some(b) = Element::borrow_u8(b) else {
            return 0.0;
        };
        packed::dot(a, b, n, i, j, &packed::Conf::from(&row))
    }

    /// 内核本体：这份格式的 tile 矩阵乘，结果写满 `out`（`m × n`）。
    fn mul(a: &[Self::Elem], b: &[Self::Elem], m: usize, n: usize, out: &mut [f32]);

    /// 逐元素格式（不打包）的码值：测试侧闭式重算用；打包格式由骨架按位流解包，返回空切片。
    fn codes(values: &[Self::Elem]) -> Vec<i32> {
        let _ = values;
        Vec::new()
    }
}

/// 从样本里取输入缓冲：名字固定 `a` / `b`，元素类型由格式给出。
fn take<'a, G: Gemm>(sample: &'a Sample, name: &str) -> Result<&'a [G::Elem], Error> {
    let buffer = sample
        .buffer(name)
        .ok_or_else(|| Error::new(Code::Missing, format!("样本里没有缓冲 `{name}`")))?;
    G::Elem::borrow(&buffer.values).ok_or_else(|| {
        Error::new(
            Code::Invalid,
            format!(
                "缓冲 `{name}` 不是 {}：这份格式的输入元素类型",
                G::Elem::NAME
            ),
        )
    })
}

/// 标量代价：`flops` 固定 `2·m·k·n`；输入按 [`Gemm::BITS`] 计费，输出每元素 4 字节。
///
/// # Errors
///
/// 形状缺轴时返回说明（带 [`Gemm::OP`] 算子名）。
pub fn cost<G: Gemm>(shape: &ShapeContext) -> Result<Cost, Error> {
    let m = axis(shape, 'm', G::OP)?;
    let k = axis(shape, 'k', G::OP)?;
    let n = axis(shape, 'n', G::OP)?;
    let read = G::BITS.bytes_for((m * k) as u128) + G::BITS.bytes_for((k * n) as u128);
    Ok(Cost {
        flops: (2 * m * k * n) as f64,
        read: read as u64,
        write: (m * n) as u64 * 4,
    })
}

/// 确定性样本：A、B 由格式取值，Y 为 f32 零缓冲。
pub fn sample<G: Gemm>(m: usize, n: usize) -> Sample {
    Sample {
        m,
        n,
        scalars: vec![("M", m as i64), ("K", K as i64), ("N", n as i64)],
        buffers: vec![
            G::Elem::buffer("a", G::sample_a(m)),
            G::Elem::buffer("b", G::sample_b(n)),
            Buffer {
                name: "y",
                values: Values::F32(vec![0.0; m * n]),
            },
        ],
    }
}

/// 打包量化格式（每字节 `CPB` 个码）的公共解包、取样与点积助手。
pub mod packed;
/// 矩阵遍历：外层只写一份，内层点积落在 [`Gemm::dot`]。
pub fn matmul<G: Gemm>(a: &[G::Elem], b: &[G::Elem], m: usize, n: usize, out: &mut [f32]) {
    for i in 0..m {
        for j in 0..n {
            out[i * n + j] = G::dot(a, b, n, i, j);
        }
    }
}

pub use packed::{accumulate_tiled, decode_stream, packed_tiled, signed_codes};

/// 标量参考：逐 `(i, j)` 走 [`Gemm::dot`]，是与 tile 内核**相互独立**的对拍真值。
pub fn reference<G: Gemm>(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
    let m = sample.m;
    let n = sample.n;
    let a = take::<G>(sample, "a")?;
    let b = take::<G>(sample, "b")?;
    let mut out = vec![0.0; m * n];
    matmul::<G>(a, b, m, n, &mut out);
    Ok(out)
}

/// 内核本体：先校验打包长度，再把结果写进调用方给的缓冲。
///
/// # Errors
///
/// 三项缓冲长度不符时返回说明（带 [`Gemm::OP`] 算子名）。
pub fn forward<G: Gemm>(
    a: &[G::Elem],
    b: &[G::Elem],
    m: usize,
    n: usize,
    out: &mut [f32],
) -> diagnostic::Result<()> {
    if a.len() != G::a_len(m) || b.len() != G::b_len(n) || out.len() != m * n {
        return Err(Error::new(Code::Invalid, format!("{} 缓冲尺寸不符", G::OP)));
    }
    G::mul(a, b, m, n, out);
    Ok(())
}

/// 各打包 / 不打包格式共用的对拍：内核算出的结果必须与标量参考逐位一致。
#[cfg(test)]
pub(crate) fn matches_reference<G: Gemm>() {
    let sample = sample::<G>(16, 12);
    let expected = reference::<G>(&sample).expect("参考可计算");
    let a = take::<G>(&sample, "a").expect("a");
    let b = take::<G>(&sample, "b").expect("b");
    let mut out = vec![0.0; 16 * 12];
    forward::<G>(a, b, 16, 12, &mut out).expect("尺寸符");
    assert_eq!(out, expected);
}

/// 多形状对拍矩阵：每条格式在一档 `(m, n)` 配置上重放参考与内核，结果逐位一致。
#[cfg(test)]
pub(crate) fn matches_reference_over_shapes<G: Gemm>(shapes: &[(usize, usize)]) {
    for &(m, n) in shapes {
        let sample = sample::<G>(m, n);
        let expected = reference::<G>(&sample).expect("参考可计算");
        let a = take::<G>(&sample, "a").expect("a");
        let b = take::<G>(&sample, "b").expect("b");
        let mut out = vec![0.0; m * n];
        forward::<G>(a, b, m, n, &mut out).expect("尺寸符");
        assert_eq!(out, expected, "m={m}、n={n} 必须与参考逐位一致");
    }
}

/// 整数点积的独立闭式校验：不经过 `G::dot` 与 `matmul`，测试侧自己按收缩轴重算。
#[cfg(test)]
pub(crate) fn hand_closed<G: Gemm>() {
    const M: usize = 5;
    const N: usize = 7;
    let sample = sample::<G>(M, N);
    let a = take::<G>(&sample, "a").expect("a");
    let b = take::<G>(&sample, "b").expect("b");
    let mut out = vec![0.0; M * N];
    forward::<G>(a, b, M, N, &mut out).expect("尺寸符");

    let unpacked: Vec<i32> = sample_unpacked::<G>(a, b, M, N);
    for i in 0..M {
        for j in 0..N {
            let dot: i32 = (0..K)
                .map(|p| unpacked[i * K + p] * unpacked[M * K + p * N + j])
                .sum();
            assert_eq!(out[i * N + j], dot as f32, "({i}, {j}) 闭式点积不符");
        }
    }
}

/// 把 A、B 解包成有符号整数码；测试侧独立读数，供 [`hand_closed`] 闭式重算。
#[cfg(test)]
fn sample_unpacked<G: Gemm>(a: &[G::Elem], b: &[G::Elem], m: usize, n: usize) -> Vec<i32> {
    let mut values = Vec::with_capacity(m * K + K * n);
    match G::config() {
        Some(row) => {
            let a: &[u8] = Element::borrow_u8(a).expect("打包格式元素为 u8");
            let b: &[u8] = Element::borrow_u8(b).expect("打包格式元素为 u8");
            values.extend(packed::unpack_stream(a, row.cpb, row.zero));
            values.extend(packed::unpack_stream(b, row.cpb, row.zero));
        }
        None => {
            values.extend(G::codes(a));
            values.extend(G::codes(b));
        }
    }
    values
}

/// 各格式共用的测试生成：打包格式（i2/i4/i8）跑参考、形状矩阵与独立闭式点积三则；
/// 稠密 f32 只跑参考对拍（整数闭式点积不适用）。形状矩阵只定义一次。
#[cfg(test)]
#[path = "../tests/gemm_test_support.rs"]
pub(crate) mod test_support;

/// 生成一份 GEMM 精度变体的完整公开面：标记类型、[`Gemm`] 实现、`forward` 入口、
/// [`annotation::model`] 注解与测试挂载。差异只随后面的关联项（取样 / 打包 / 点积契约）变化；
/// 新增一档精度只需加一行调用，不另写样板。
#[macro_export]
macro_rules! gemm_variant {
    (
        $name:ident, $alias:ident,
        $op:tt, $kind:ident, $differentiable:literal,
        $elem:ty, $bits:expr,
        $( $methods:item ),* $(,)?
    ) => {
        #[doc = concat!("`", $op, "` 的标记类型：格式契约（取样 / 打包 / 点积）见 `crate::gemm` 的 `Gemm` 骨架。")]
        pub struct $name;

        impl $crate::gemm::Gemm for $name {
            type Elem = $elem;

            const OP: &'static str = $op;

            const BITS: $crate::einsum::Bits = $bits;

            $( $methods )*
        }

        // 基座契约（ADR 0050）：三条行为从 `Gemm` 骨架转发，粒度由注解的骨架默认值给出。
        $crate::operator_via!($name, $op, 2, (16, 16), $differentiable, super, $crate::gemm::AXES);

        #[doc = concat!("`", $op, "` 内核本体：结果写进调用方给的缓冲。")]
        pub fn forward(
            a: &[$elem],
            b: &[$elem],
            m: usize,
            n: usize,
            out: &mut [f32],
        ) -> diagnostic::Result<()> {
            $crate::gemm::forward::<$name>(a, b, m, n, out)
        }

        #[annotation::model(name = $op, shape = [16, 16], dtype = "f32", skeleton = "gemm")]
        #[doc = concat!("`", $op, "` 矩阵乘的语义别名。")]
        pub type $alias = $name;

        #[cfg(test)]
        mod tests {
            use super::*;
            $crate::gemm::test_support::$kind!($name);
        }
    };
}

pub mod i2 {
    //! `gemm_i2`：对称 int2 矩阵乘。骨架在 [`crate::gemm`]，本模块只有"对称 int2"这一份格式
    //! 契约：A、B 是打包 2 位码（每字节 4 个，低 2 位在前），对称零点取配置表 [`I2`]；取样、
    //! 打包长度与整数点积由骨架按 [`I2`] 一行配置读出，点积在 `i32` 上精确累加后转 f32。
    pub use crate::gemm::{AXES, K, project_shape};

    use crate::quant::I2;

    crate::gemm_variant!(
        GemmI2,
        GemmI2Model,
        "gemm_i2",
        gemm_packed_tests,
        false,
        u8,
        I2.bits,
        fn config() -> Option<crate::quant::Row> {
            Some(I2)
        },
        fn mul(a: &[u8], b: &[u8], m: usize, n: usize, out: &mut [f32]) {
            crate::gemm::packed_tiled(a, b, m, n, &crate::gemm::packed::Conf::from(&I2), out)
        },
    );
}

pub mod i4 {
    //! `gemm_i4`：对称 int4 矩阵乘。骨架在 [`crate::gemm`]，本模块只有"对称 int4"这一份格式
    //! 契约：A、B 是打包 nibble（每字节 2 个，低 4 位在前、高 4 位在后），对称零点取配置表
    //! [`I4`]；取样、打包长度与整数点积由骨架按 [`I4`] 一行配置读出，点积在 `i32` 上精确累加。
    pub use crate::gemm::{AXES, K, project_shape};

    use crate::quant::I4;

    crate::gemm_variant!(
        GemmI4,
        GemmI4Model,
        "gemm_i4",
        gemm_packed_tests,
        false,
        u8,
        I4.bits,
        fn config() -> Option<crate::quant::Row> {
            Some(I4)
        },
        fn mul(a: &[u8], b: &[u8], m: usize, n: usize, out: &mut [f32]) {
            crate::gemm::packed_tiled(a, b, m, n, &crate::gemm::packed::Conf::from(&I4), out)
        },
    );
}

pub mod i8 {
    //! `gemm_i8`：对称 int8 矩阵乘。骨架在 [`crate::gemm`]，本模块只有"对称 int8"这一份格式
    //! 契约：`u8` 承载 int8 位模式，点积先在 `i32` 上精确累加（对应 CUDA `dp4a`：每 4 对 int8
    //! 乘积累加进一个 int32），再收敛成 f32。CPU 没有 dp4a，退化为标量整数点积。
    pub use crate::gemm::{AXES, K, project_shape};

    use crate::quant::I8;

    crate::gemm_variant!(
        GemmI8,
        GemmI8Model,
        "gemm_i8",
        gemm_packed_tests,
        false,
        u8,
        I8.bits,
        fn sample_a(m: usize) -> Vec<u8> {
            (0..m * K)
                .map(|i| (((i * 37) % 16) as i8 - 8) as u8)
                .collect()
        },
        fn sample_b(n: usize) -> Vec<u8> {
            (0..K * n)
                .map(|i| (((i * 53) % 16) as i8 - 8) as u8)
                .collect()
        },
        fn a_len(m: usize) -> usize {
            m * K
        },
        fn b_len(n: usize) -> usize {
            K * n
        },
        fn dot(a: &[u8], b: &[u8], n: usize, i: usize, j: usize) -> f32 {
            let row = &a[i * K..][..K];
            let acc: i32 = row
                .iter()
                .enumerate()
                .map(|(p, packed)| (*packed as i8) as i32 * (b[p * n + j] as i8) as i32)
                .sum();
            acc as f32
        },
        fn codes(values: &[u8]) -> Vec<i32> {
            crate::gemm::signed_codes(values)
        },
        fn mul(a: &[u8], b: &[u8], m: usize, n: usize, out: &mut [f32]) {
            let codes_a = crate::gemm::signed_codes(a);
            let codes_b = crate::gemm::signed_codes(b);
            crate::gemm::accumulate_tiled(&codes_a, &codes_b, m, n, out)
        },
    );
}

pub mod f32 {
    //! `gemm_f32`：稠密 f32 矩阵乘。骨架在 [`crate::gemm`]，本模块只有"稠密 f32"这一份格式
    //! 契约：元素类型、位宽、样本取值与点积算术。CPU 片段没有 MMA 原语，算术走通用稠密 tile
    //! （[`crate::matmul::matmul_tiled`]），与标量双重循环逐位一致。
    pub use crate::gemm::{AXES, K, project_shape};

    crate::gemm_variant!(
        Dense,
        GemmF32,
        "gemm_f32",
        gemm_dense_tests,
        true,
        f32,
        crate::einsum::Bits::ThirtyTwo,
        fn sample_a(m: usize) -> Vec<f32> {
            (0..m * K).map(|i| ((i % 5) as f32) * 0.25 - 0.5).collect()
        },
        fn sample_b(n: usize) -> Vec<f32> {
            (0..K * n).map(|i| ((i % 7) as f32) * 0.2 - 0.4).collect()
        },
        fn a_len(m: usize) -> usize {
            m * K
        },
        fn b_len(n: usize) -> usize {
            K * n
        },
        fn dot(a: &[f32], b: &[f32], n: usize, i: usize, j: usize) -> f32 {
            let row = &a[i * K..][..K];
            row.iter().enumerate().map(|(p, x)| x * b[p * n + j]).sum()
        },
        fn mul(a: &[f32], b: &[f32], m: usize, n: usize, out: &mut [f32]) {
            // 稠密 f32 不分块：`K = 8` 时一行 A 的 8 个权重就在寄存器里，分块（64 列块或
            // `matmul_tiled` 的三级块）只会多出累加器驻留与写回，实测反而慢 2~3 倍
            // （`gemm_f32` 基准的三档对照）。顺序累加同时保证与标量参考逐位相等。
            crate::gemm::matmul::<Self>(a, b, m, n, out)
        },
    );
}
