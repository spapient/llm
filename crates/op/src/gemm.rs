//! GEMM 家族共享的轴契约、形状投影与格式无关的矩阵乘骨架。
//!
//! `gemm_f32`、`gemm_i8`、`gemm_i4` 是同一副三轴骨架（`m`/`k` 输入、`n` 输出）的三种
//! 权重格式：轴的角色、收缩轴长度、形状投影、代价口径、样本装配、矩阵遍历与尺寸校验完全
//! 相同，差异只在元素类型、位宽、样本取值、打包长度与点积算术。本模块通过 [`Gemm`] trait
//! 收口这部分共享事实；各格式只写一份格式契约，并按格式单态化，不在热路径上引入运行时分发。
//! 与 [`crate::gemv`] 的两轴骨架对称，只是这里多一条收缩轴。
//!
//! 算术有两条：执行路径 [`forward`] 按格式调 [`Gemm::mul`]——打包格式走 [`packed_tiled`]（位流
//! 解一遍、再按 [`TILE_N`] 列块做整数累加，解包次数从 `m·n·K` 降到 `m·K + K·n`），稠密 f32 则
//! 保持 [`matmul`] 的顺序遍历（`K = 8` 时分块实测更慢，见该处注释）；标量 [`matmul`] 同时保留为
//! 与内核**相互独立**的对拍真值，四条路径逐位相等。

use crate::einsum::Bits;
use crate::gemv::Element;
use crate::model::{AxisRole, AxisSpec, Buffer, Cost, Sample, ShapeContext, Values};
use crate::quant::Row;
use diagnostic::{Code, Error};

/// 收缩轴长度（固定档）。
pub const K: usize = 8;

/// 输出列块宽度：B 的一个列块（`K × TILE_N` 个码）解包一次后被整段行复用。
///
/// 与 [`crate::matmul::TILE_N`] 不是同一个块：那条路径要切 `m`/`n`/`k` 三级、收缩轴由调用方
/// 给定；这里收缩轴恒为 [`K`]，块只决定「B 的哪一段列驻留」，块宽取 64 使 `K` 段码与一行
/// 累加器合起来仍在一级缓存里。
pub const TILE_N: usize = 64;

/// 每字节打包码数：位宽 ≤ 8 时按 `8 / bits` 算；8 位逐元素为 1。
///
/// 打包布局与 [`packed`] 同一口径：i2=4、i4=2、i8=1。桥接层几何反推用，避免各 lane 各抄
/// 一遍除法。
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
///
/// 三份格式对轴的判读逐字相同，只有诊断里点名的算子名不同——标签作为参数传入，避免三份
/// 平行的取轴辅助各抄一遍。
pub fn axis(shape: &ShapeContext, label: char, op: &str) -> Result<usize, Error> {
    shape
        .get(label)
        .ok_or_else(|| Error::new(Code::Invalid, format!("{op} 形状缺轴 `{label}`")))
}

/// 矩阵乘格式：一条 GEMM 里随精度变化的那部分。
///
/// 实现者只写"这份格式用什么元素、什么位宽、怎么取样、怎么打包、怎么点积"；代价、样本装配、
/// 矩阵遍历与尺寸校验由本模块骨架（[`cost`] / [`sample`] / [`reference()`] / [`forward`]）
/// 收口。所有关联项都是编译期常量：没有一件需要运行期再判断一次。
pub trait Gemm {
    /// 样本缓冲里的元素类型（[`Values`] 的两个闭集变体）。
    type Elem: Element;

    /// 诊断里点名的算子名：缺轴、尺寸不符时进错误文案。
    const OP: &'static str;

    /// A / B 输入元素位宽；输出恒为 f32。
    const BITS: Bits;

    /// 打包格式的静态配置（[`Row`]）：每字节码数、零点、样本偏移、对称码表单点收敛。
    ///
    /// 打包 lane（i2/i4）返回自身那一行；逐元素 lane（i8）返回 `None`，走 `codes` 按位模式
    /// 解释。新增一档打包精度只在 `quant` 配置表加一行并把这里指过去，不在三条 lane 各抄除法、
    /// 零点与偏移表。
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
    ///
    /// 打包格式走 [`packed::dot`]；逐元素格式自行实现本函数。
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
    ///
    /// 打包格式把 A、B 的位流各解包一次再累加（[`packed_tiled`]），逐元素整数格式直接取码
    /// （[`accumulate_tiled`]），稠密 f32 走 [`crate::matmul::matmul_tiled`]。三种形态都保持
    /// 「每个输出元素按 `p` 升序累加」，于是与 [`matmul`] 的标量口径逐位相等。
    ///
    /// 没有默认实现：默认返回值会让「忘了覆盖」变成静默全零，而调用方拿不到任何信号。
    fn mul(a: &[Self::Elem], b: &[Self::Elem], m: usize, n: usize, out: &mut [f32]);

    /// 逐元素格式（不打包）的码值：测试侧闭式重算用；打包格式由骨架按位流解包，返回空切片。
    ///
    /// 与 [`dot`](Self::dot) 同一组码值口径，只是按元素铺开，不做点积。
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
///
/// int2（4 码/字节）与 int4（2 码/字节）只在码宽、掩码、零点与取样偏移上不同；这些事实作为
/// 参数传入，打包位域的字节布局、码位拆解与整数累加只写一份。int8 不打包，不走这里。
pub mod packed {
    use super::K;

    /// 打包配置：取样/长度/点积共用一行 [`Row`](crate::quant::Row) 的位宽事实。
    pub struct Conf {
        /// 每字节码数。
        pub cpb: usize,
        /// 对称零点（无符号码点回退有符号整数的平移量）。
        pub zero: i32,
        /// 样本码值偏移，长度等于 `cpb`。
        pub offsets: &'static [u32],
    }

    impl From<&crate::quant::Row> for Conf {
        fn from(row: &crate::quant::Row) -> Self {
            Self {
                cpb: row.cpb,
                zero: row.zero,
                offsets: row.offsets,
            }
        }
    }

    /// 解包字节内第 `slot` 个码，按对称零点回退成有符号整数。
    pub fn unpack(byte: u8, slot: usize, cpb: usize, zero: i32) -> i8 {
        let width = crate::quant::width_cpb(cpb);
        let mask = (1u8 << width) - 1;
        let value = (byte >> (slot * width)) & mask;
        (value as i32 - zero) as i8
    }

    /// 确定性打包位模式：`len` 个码按 `cpb` 每字节装一行；码值取
    /// `(字节下标 · seed + offsets[c]) mod 2^width`。
    pub fn pack(len: usize, seed: usize, cpb: usize, offsets: &[u32]) -> Vec<u8> {
        let width = crate::quant::width_cpb(cpb);
        let modulus = 1usize << width;
        (0..len / cpb)
            .map(|byte| {
                (0..cpb).fold(0u8, |acc, code| {
                    let value = ((byte * seed + offsets[code] as usize) % modulus) as u8;
                    acc | (value << (code * width))
                })
            })
            .collect()
    }

    /// 打包矩阵点积：A 第 `i` 行与 B 第 `j` 列，收缩轴固定 [`K`]，整数累加后转 f32。
    pub fn dot(a: &[u8], b: &[u8], n: usize, i: usize, j: usize, conf: &Conf) -> f32 {
        let bytes_per_row = K / conf.cpb;
        let mut acc: i32 = 0;
        for p in 0..K {
            let x = unpack(
                a[i * bytes_per_row + p / conf.cpb],
                p % conf.cpb,
                conf.cpb,
                conf.zero,
            ) as i32;
            let y_pos = p * n + j;
            let y = unpack(b[y_pos / conf.cpb], y_pos % conf.cpb, conf.cpb, conf.zero) as i32;
            acc += x * y;
        }
        acc as f32
    }

    /// A（`m×K`）确定性打包取样；seed 固定，同一形状取值可复现。
    pub fn packed_sample_a(m: usize, conf: &Conf) -> Vec<u8> {
        pack(m * K, 37, conf.cpb, conf.offsets)
    }

    /// B（`K×n`）确定性打包取样。
    pub fn packed_sample_b(n: usize, conf: &Conf) -> Vec<u8> {
        pack(K * n, 53, conf.cpb, conf.offsets)
    }

    /// A 打包后字节数：`m·K/cpb`。
    pub fn packed_a_len(m: usize, conf: &Conf) -> usize {
        m * K / conf.cpb
    }

    /// B 打包后字节数：`K·n/cpb`。
    pub fn packed_b_len(n: usize, conf: &Conf) -> usize {
        K * n / conf.cpb
    }

    /// 按元素位流解包一串打包字节：每字节 `cpb` 个码，低槽在前，码值减 `zero`。
    pub fn unpack_stream(bytes: &[u8], cpb: usize, zero: i32) -> Vec<i32> {
        bytes
            .iter()
            .flat_map(|&byte| (0..cpb).map(move |slot| unpack(byte, slot, cpb, zero) as i32))
            .collect()
    }
}

/// 矩阵遍历：外层只写一份，内层点积落在 [`Gemm::dot`]。
pub fn matmul<G: Gemm>(a: &[G::Elem], b: &[G::Elem], m: usize, n: usize, out: &mut [f32]) {
    for i in 0..m {
        for j in 0..n {
            out[i * n + j] = G::dot(a, b, n, i, j);
        }
    }
}

/// 位流 → 整数码：每个码一个 `i32`，按行主序连续铺开（A 行宽 [`K`]、B 行宽 `n`）。
///
/// 解包次数从此只与 `m·K + K·n` 有关，而不是与 `m·n·K` 有关——移位、掩码与零点平移正是量化
/// 格式的主要成本，旧做法在每个输出元素上把它们重做一遍。
pub fn decode_stream(codes: &mut [i32], bytes: &[u8], conf: &packed::Conf) {
    for (index, code) in codes.iter_mut().enumerate() {
        let byte = bytes[index / conf.cpb];
        *code = packed::unpack(byte, index % conf.cpb, conf.cpb, conf.zero) as i32;
    }
}

/// 整数码的 tile 累加：外层按 [`TILE_N`] 走列块，块内 `p` 在外、列在内。
///
/// 每个输出元素仍按 `p` 升序做 `i32` 加法，于是与 [`matmul`] / [`packed::dot`] 逐位相等；
/// 列在内让 B 的列块段与累加器都是连续读写，内层是一次可向量化的 AXPY。
///
/// `codes_a` 至少 `m·K` 个码、`codes_b` 至少 `K·n` 个，`out` 恰为 `m·n`；长度由
/// [`forward`] 在入口校验，这里不再重复判——tile 内层每多一次分支就多一次整块的成本。
pub fn accumulate_tiled(codes_a: &[i32], codes_b: &[i32], m: usize, n: usize, out: &mut [f32]) {
    let mut acc = vec![0i32; TILE_N.min(n)];
    let mut start = 0;
    while start < n {
        let width = TILE_N.min(n - start);
        let acc = &mut acc[..width];
        let (rows, _) = codes_a.as_chunks::<K>();
        for (row, arow) in rows.iter().take(m).enumerate() {
            acc.fill(0);
            for (p, &x) in arow.iter().enumerate() {
                let segment = &codes_b[p * n + start..][..width];
                rank1_update(acc, x, segment);
            }
            write_back(out, row, n, start, width, acc);
        }
        start += width;
    }
}

/// 块内 rank-1 更新：`acc[j] += x · segment[j]`。
///
/// 独立成函数是为把 [`accumulate_tiled`] 的圈复杂度压在门禁阈值内：内层乘加是纯算术，
/// 没有分支，抽出来不损失任何语义。
fn rank1_update(acc: &mut [i32], x: i32, segment: &[i32]) {
    for (slot, &y) in acc.iter_mut().zip(segment) {
        *slot += x * y;
    }
}

/// 把 `acc` 写回输出行：`out[row, start..start+width] = acc as f32`。
///
/// 与 [`rank1_update`] 同理：写回循环没有分支，抽出来只为压低调用方的圈复杂度。
fn write_back(out: &mut [f32], row: usize, n: usize, start: usize, width: usize, acc: &[i32]) {
    let target = &mut out[row * n + start..][..width];
    for (slot, &value) in target.iter_mut().zip(acc.iter()) {
        *slot = value as f32;
    }
}

/// 逐元素格式的整数码：`u8` 位模式按 `i8` 解释，与 [`Gemm::dot`] 的取码口径同一份。
///
/// 打包格式不用它——位流解包还带零点平移（[`decode_stream`]）。
pub fn signed_codes(values: &[u8]) -> Vec<i32> {
    values.iter().map(|&byte| byte as i8 as i32).collect()
}

/// 打包格式的本体：两条位流各解一遍，再走 [`accumulate_tiled`]。
///
/// A 是 `m × K` 个码、B 是 `K × n` 个码，解包结果只在这一个调用里存活；块内不再碰位流。
pub fn packed_tiled(a: &[u8], b: &[u8], m: usize, n: usize, conf: &packed::Conf, out: &mut [f32]) {
    let mut codes_a = vec![0i32; m * K];
    let mut codes_b = vec![0i32; K * n];
    decode_stream(&mut codes_a, a, conf);
    decode_stream(&mut codes_b, b, conf);
    accumulate_tiled(&codes_a, &codes_b, m, n, out);
}

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
///
/// 三条格式的测试体只有算子名不同，断言与尺寸口径收口到这里，避免三份平行用例各抄一遍。
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
///
/// 形状由配置表给出（单行单列、多行单列、单行多列、常规矩阵），覆盖收缩轴 `K` 之外的两类
/// 边界；内核与参考是同一段算术，逐位相等是最低限度的口径，真正的独立闭式校验见
/// [`hand_closed<G>`]。
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
///
/// 参考与生产同一份代码时，对它俩对拍等于自己跟自己对；这条用例把每个输出元素的 `K` 项
/// 乘积和在测试里重算一遍（`i32` 精确累加，与内核整数路径同口径），逐元素钉住算术。
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
///
/// 打包格式按 [`Gemm::config`] 一行配置解包（码数、零点随配置表），逐元素格式走 [`Gemm::codes`]；
/// 不再按位宽 `match` 硬编码两套字节布局。
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
pub(crate) mod test_support {
    /// 对拍形状矩阵：单行单列、多行单列、单行多列、常规矩阵，以及跨列块的三档。
    ///
    /// 后三档是 tile 的接缝：`n` 恰为一整块、多一列、两块多两列，用来钉住
    /// [`super::TILE_N`] 的边界收口（块宽不是序列长的倍数时最容易被漏写或重复写）。
    pub(crate) const SHAPES: &[(usize, usize)] = &[
        (1, 1),
        (5, 1),
        (1, 7),
        (16, 12),
        (16, super::TILE_N),
        (16, super::TILE_N + 1),
        (3, 2 * super::TILE_N + 2),
    ];

    /// 打包量化格式共有的三则对拍。
    macro_rules! gemm_packed_tests {
        ($ty:ty) => {
            #[test]
            fn matches_the_scalar_reference() {
                crate::gemm::matches_reference::<$ty>();
            }

            #[test]
            fn matches_over_the_shape_matrix() {
                crate::gemm::matches_reference_over_shapes::<$ty>(
                    $crate::gemm::test_support::SHAPES,
                );
            }

            #[test]
            fn matches_the_hand_closed_dot() {
                crate::gemm::hand_closed::<$ty>();
            }
        };
    }

    /// 稠密 f32 格式：整数闭式点积不适用，只做标量参考对拍。
    macro_rules! gemm_dense_tests {
        ($ty:ty) => {
            #[test]
            fn matches_the_scalar_reference() {
                crate::gemm::matches_reference::<$ty>();
            }
        };
    }
    pub(crate) use gemm_dense_tests;
    pub(crate) use gemm_packed_tests;
}

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
        impl $crate::operator::Operator for $name {
            const NAME: &'static str = $op;
            const SHAPE: (usize, usize) = (16, 16);
            const DTYPE: $crate::model::DType = $crate::model::DType::F32;
            const AXES: &'static [$crate::model::AxisSpec] = $crate::gemm::AXES;
            const DIFFERENTIABLE: bool = $differentiable;

            fn shape_ctx(
                m: usize,
                n: usize,
            ) -> Result<$crate::model::ShapeContext, ::diagnostic::Error> {
                $crate::gemm::project_shape(m, n)
            }

            fn cost(
                shape: &$crate::model::ShapeContext,
            ) -> Result<$crate::model::Cost, ::diagnostic::Error> {
                $crate::gemm::cost::<Self>(shape)
            }

            fn sample(m: usize, n: usize) -> $crate::model::Sample {
                $crate::gemm::sample::<Self>(m, n)
            }

            fn reference(sample: &$crate::model::Sample) -> ::diagnostic::Result<Vec<f32>> {
                $crate::gemm::reference::<Self>(sample)
            }
        }

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

pub mod gemm_i2 {
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

pub mod gemm_i4 {
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

pub mod gemm_i8 {
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

pub mod gemm_f32 {
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
