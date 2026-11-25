//! GEMM 家族共享的轴契约、形状投影与格式无关的矩阵乘骨架。

use crate::einsum::Bits;
use crate::gemv::Element;
#[cfg(test)]
use crate::model::Values;
use crate::model::{AxisRole, AxisSpec, Buf, Cost, Sample, ShapeCtx};
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
pub fn project_shape(m: usize, n: usize) -> Result<ShapeCtx, Error> {
    ShapeCtx::new([('m', m), ('k', K), ('n', n)])
}

/// 从形状上下文取一条轴；缺轴时按 `op` 标签报 [`Code::Invalid`]。
pub fn axis(shape: &ShapeCtx, label: char, op: &str) -> Result<usize, Error> {
    shape
        .get(label)
        .ok_or_else(|| Error::new(Code::Invalid, format!("{op} 形状缺轴 `{label}`")))
}

/// 矩阵乘格式：一条 GEMM 里随精度变化的那部分。
pub trait Gemm {
    /// 样本缓冲里的元素类型（[`Values`](crate::model::Values) 的两个闭集变体）。
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
        let conf = Conf::from(&row);
        Self::Elem::from_bytes(packed_sample_a(m, &conf))
    }

    /// B（`K×n`）确定性取值（口径同 [`sample_a`](Self::sample_a)）。
    fn sample_b(n: usize) -> Vec<Self::Elem> {
        let Some(row) = Self::config() else {
            return Vec::new();
        };
        let conf = Conf::from(&row);
        Self::Elem::from_bytes(packed_sample_b(n, &conf))
    }

    /// A 缓冲应有的元素个数（打包后按 [`Elem`](Self::Elem) 计）。
    fn a_len(m: usize) -> usize {
        let Some(row) = Self::config() else { return 0 };
        packed_a_len(m, &Conf::from(&row))
    }

    /// B 缓冲应有的元素个数。
    fn b_len(n: usize) -> usize {
        let Some(row) = Self::config() else { return 0 };
        packed_b_len(n, &Conf::from(&row))
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
        dot(a, b, n, i, j, &Conf::from(&row))
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
    let buf = sample
        .buf(name)
        .ok_or_else(|| Error::new(Code::Missing, format!("样本里没有缓冲 `{name}`")))?;
    G::Elem::borrow(&buf.values).ok_or_else(|| {
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
pub fn cost<G: Gemm>(shape: &ShapeCtx) -> Result<Cost, Error> {
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
        bufs: vec![
            G::Elem::buf("a", G::sample_a(m)),
            G::Elem::buf("b", G::sample_b(n)),
            Buf::f32("y", vec![0.0; m * n]),
        ],
    }
}

/// 打包配置：取样/长度/点积共用一行 `Row` 的位宽事实。
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
    (i32::from(value) - zero) as i8
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
        let x = i32::from(unpack(
            a[i * bytes_per_row + p / conf.cpb],
            p % conf.cpb,
            conf.cpb,
            conf.zero,
        ));
        let y_pos = p * n + j;
        let y = i32::from(unpack(
            b[y_pos / conf.cpb],
            y_pos % conf.cpb,
            conf.cpb,
            conf.zero,
        ));
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
        .flat_map(|&byte| (0..cpb).map(move |slot| i32::from(unpack(byte, slot, cpb, zero))))
        .collect()
}

/// 位流 → 整数码：每个码一个 `i32`，按行主序连续铺开（A 行宽 [`K`]、B 行宽 `n`）。
pub fn decode_stream(codes: &mut [i32], bytes: &[u8], conf: &Conf) {
    for (idx, code) in codes.iter_mut().enumerate() {
        let byte = bytes[idx / conf.cpb];
        *code = i32::from(unpack(byte, idx % conf.cpb, conf.cpb, conf.zero));
    }
}

/// 整数码的 tile 累加：外层按 [`TILE_N`] 走列块，块内 `p` 在外、列在内。
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
fn rank1_update(acc: &mut [i32], x: i32, segment: &[i32]) {
    for (slot, &y) in acc.iter_mut().zip(segment) {
        *slot += x * y;
    }
}

/// 把 `acc` 写回输出行：`out[row, start..start+width] = acc as f32`。
fn write_back(out: &mut [f32], row: usize, n: usize, start: usize, width: usize, acc: &[i32]) {
    let target = &mut out[row * n + start..][..width];
    for (slot, &value) in target.iter_mut().zip(acc.iter()) {
        *slot = value as f32;
    }
}

/// 逐元素格式的整数码：`u8` 位模式按 `i8` 解释，与 [`Gemm::dot`] 的取码口径同一份。
pub fn signed_codes(values: &[u8]) -> Vec<i32> {
    values.iter().map(|&byte| i32::from(byte as i8)).collect()
}

/// 打包格式的本体：两条位流各解一遍，再走 [`accumulate_tiled`]。
pub fn packed_tiled(a: &[u8], b: &[u8], m: usize, n: usize, conf: &Conf, out: &mut [f32]) {
    let mut codes_a = vec![0i32; m * K];
    let mut codes_b = vec![0i32; K * n];
    decode_stream(&mut codes_a, a, conf);
    decode_stream(&mut codes_b, b, conf);
    accumulate_tiled(&codes_a, &codes_b, m, n, out);
}

/// 矩阵遍历：外层只写一份，内层点积落在 [`Gemm::dot`]。
pub fn matmul<G: Gemm>(a: &[G::Elem], b: &[G::Elem], m: usize, n: usize, out: &mut [f32]) {
    for i in 0..m {
        for j in 0..n {
            out[i * n + j] = G::dot(a, b, n, i, j);
        }
    }
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
#[cfg(test)]
pub(crate) fn matches_reference<G: Gemm>() {
    matches_reference_over_shapes::<G>(&[(16, 12)]);
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

/// 取样本 `a`/`b` 并按 `m×n` 跑一遍生产 [`forward`]：两条闭式校验的公共装配，
/// 这一段必须逐字一致，否则 [`hand_closed`] 与 [`dense_closed`] 比的不是同一次调用。
#[cfg(test)]
fn closed_pair<'a, G: Gemm>(
    sample: &'a Sample,
    m: usize,
    n: usize,
    out: &mut [f32],
) -> (&'a [G::Elem], &'a [G::Elem]) {
    let a = take::<G>(sample, "a").expect("a");
    let b = take::<G>(sample, "b").expect("b");
    forward::<G>(a, b, m, n, out).expect("尺寸符");
    (a, b)
}

/// 整数点积的独立闭式校验：不经过 `G::dot` 与 `matmul`，测试侧自己按收缩轴重算。
#[cfg(test)]
pub(crate) fn hand_closed<G: Gemm>() {
    const M: usize = 5;
    const N: usize = 7;
    let sample = sample::<G>(M, N);
    let mut out = vec![0.0; M * N];
    let (a, b) = closed_pair::<G>(&sample, M, N, &mut out);

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

/// 稠密格式的独立闭式校验：**不经过** `G::dot` / `matmul` / 生产 `mul`，测试侧在 f64 里
/// 自己按收缩轴重算，再用前向误差界 `γ_k·Σ|a·b|`（`γ_k = k·u/(1−k·u)`、`u = 2^-24`）判。
///
/// 这条校验之所以必要：`gemm_f32` 的 `mul` 与 [`reference`] 走的是**同一个** `matmul`，
/// 两者逐位相等是被构造出来的，`matches_the_scalar_reference` 对稠密格式的证据为零（F6）。
/// f64 重算与生产路径不共享任何一行算术，因此它能抓到 `mul` 自身的错。
///
/// 界取**半个** `γ_k·Σ|a·b|`（`MARGIN = 0.5`）而不是整个：整个 `γ_k` 对任何重排求和几乎恒真，
/// 断言它等于没断言。实测最坏比值 **0.1683**（`k=8` 样本、全 `γ_k` 口径），半个界即 3.0 倍余量。
#[cfg(test)]
pub(crate) fn dense_closed<G: Gemm>() {
    const M: usize = 16;
    const N: usize = 12;
    /// 实测界的收紧系数：留余量但不留到与 `γ_k` 同宽。
    const MARGIN: f64 = 0.5;
    let sample = sample::<G>(M, N);
    let mut out = vec![0.0; M * N];
    let (a, b) = closed_pair::<G>(&sample, M, N, &mut out);

    let (af, bf) = (f32_values::<G>(a), f32_values::<G>(b));
    let k = af.len() / M;
    assert_eq!(bf.len(), k * N, "B 的元素数不是 k×n");
    for i in 0..M {
        for j in 0..N {
            let (sum, mass) = exact(&af, &bf, k, N, i, j);
            let bound = MARGIN * test_support::gamma(k) * mass;
            let err = (f64::from(out[i * N + j]) - sum).abs();
            // 样本全零时界与误差同时为 0，断言恒真——那是最容易被忽略的空转，先把它钉掉。
            assert!(bound > 0.0, "({i}, {j}) 闭式界退化为 0：样本无判别力");
            assert!(
                err <= bound,
                "({i}, {j}) 前向误差 {err:e} 超过 {MARGIN}·γ_{k}·Σ|a·b| = {bound:e}"
            );
        }
    }
}

/// f64 里的精确点积与 `|a·b|` 的质量和——重排误差界的分子与分母。
#[cfg(test)]
fn exact(af: &[f32], bf: &[f32], k: usize, n: usize, i: usize, j: usize) -> (f64, f64) {
    (0..k).fold((0.0, 0.0), |(sum, mass), p| {
        let prod = f64::from(af[i * k + p]) * f64::from(bf[p * n + j]);
        (sum + prod, mass + prod.abs())
    })
}

/// 把元素读成 f32；打包格式（`u8` 元素）返回空，由调用方按 `k = 0` 自行失败。
#[cfg(test)]
fn f32_values<G: Gemm>(values: &[G::Elem]) -> Vec<f32> {
    match G::Elem::buf("v", values.to_vec()).values {
        Values::F32(v) => v,
        _ => Vec::new(),
    }
}

/// 把 A、B 解包成有符号整数码；测试侧独立读数，供 [`hand_closed`] 闭式重算。
#[cfg(test)]
fn sample_unpacked<G: Gemm>(a: &[G::Elem], b: &[G::Elem], m: usize, n: usize) -> Vec<i32> {
    let mut values = Vec::with_capacity(m * K + K * n);
    if let Some(row) = G::config() {
        let a: &[u8] = Element::borrow_u8(a).expect("打包格式元素为 u8");
        let b: &[u8] = Element::borrow_u8(b).expect("打包格式元素为 u8");
        values.extend(unpack_stream(a, row.cpb, row.zero));
        values.extend(unpack_stream(b, row.cpb, row.zero));
    } else {
        values.extend(G::codes(a));
        values.extend(G::codes(b));
    }
    values
}

/// 各格式共用的测试生成：打包格式（i2/i4/i8）跑参考、形状矩阵与独立闭式点积三则；
/// 稠密 f32 跑独立 f64 闭式校验（整数闭式不适用，但那不等于只能自我对拍）。形状矩阵只定义一次。
#[cfg(test)]
mod test_support {
    include!("../tests/kernel_gemm_test_support.rs");
}

/// 生成一份 GEMM 精度变体的完整公开面：标记类型、[`Gemm`] 实现、`forward` 入口、
/// [`annotation::model`] 注解与测试挂载。差异只随后面的关联项（取样 / 打包 / 点积契约）变化；
/// 新增一档精度只需加一行调用，不另写样板。
///
/// 出口按元素类型分两路：稠密 `f32` 的两路激活是二维只读视图，几何从 `dims()` 取；打包档
/// `u8` 是位流，行宽按字节折算，保留裸切片与调用方给出的 `m`/`n`（裁决：打包位模式不进 View）。
/// 变体公共样板：标记类型、`Gemm` 契约、基座转发、语义别名与测试挂载。两路只在元素类型与
/// `forward` 形态上分叉，其余装配收敛到本宏，避免两份逐字相同的展开模板。
#[macro_export]
macro_rules! gemm_variant_shared {
    (
        $name:ident, $alias:ident,
        $op:tt, $kind:ident, $differentiable:literal,
        $bits:expr, $elem:ty,
        $( $methods:tt ),* $(,)?
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
        $crate::operator_via!($name, $op, 2, (16, 16), $differentiable, skel, $crate::gemm::AXES);

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

/// 装配一档 GEMM 精度变体的完整公开面：标记类型、`Gemm` 实现、`forward` 入口、注解别名与测试挂载。
#[macro_export]
macro_rules! gemm_variant {
    (
        $name:ident, $alias:ident,
        $op:tt, $kind:ident, $differentiable:literal,
        f32, $bits:expr,
        $( $methods:item ),* $(,)?
    ) => {
        $crate::gemm_variant_shared!(
            $name, $alias, $op, $kind, $differentiable, $bits, f32,
            $( $methods ),*
        );

        #[doc = concat!("`", $op, "` 内核本体：结果写进调用方给的缓冲。")]
        pub fn forward(
            a: tensor::View<'_, f32>,
            b: tensor::View<'_, f32>,
            out: &mut [f32],
        ) -> diagnostic::Result<()> {
            let (m, _, n) = $crate::matmul::dim2_pair(a.dims(), b.dims(), $op)?;
            $crate::gemm::forward::<$name>(a.as_slice(), b.as_slice(), m, n, out)
        }
    };

    (
        $name:ident, $alias:ident,
        $op:tt, $kind:ident, $differentiable:literal,
        u8, $bits:expr,
        $( $methods:item ),* $(,)?
    ) => {
        $crate::gemm_variant_shared!(
            $name, $alias, $op, $kind, $differentiable, $bits, u8,
            $( $methods ),*
        );

        #[doc = concat!("`", $op, "` 内核本体：结果写进调用方给的缓冲。")]
        pub fn forward(
            a: &[u8],
            b: &[u8],
            m: usize,
            n: usize,
            out: &mut [f32],
        ) -> diagnostic::Result<()> {
            $crate::gemm::forward::<$name>(a, b, m, n, out)
        }
    };
}

/// 四档精度变体只声明格式契约；遍历、形状与计价共享上方骨架。
mod variants {
    /// 打包档声明：模块名、标记、语义别名、算子名与量化配置同行绑定。
    macro_rules! packed_variant {
        ($module:ident, $name:ident, $alias:ident, $op:literal, $row:ident) => {
            #[doc = concat!("`", $op, "`：对称打包矩阵乘，由 `", stringify!($row), "` 配置驱动。")]
            pub mod $module {
                use crate::gemm as skel;
                pub use crate::gemm::{AXES, K, project_shape};
                use crate::quant::$row;

                crate::gemm_variant!(
                    $name,
                    $alias,
                    $op,
                    gemm_packed_tests,
                    false,
                    u8,
                    $row.bits,
                    fn config() -> Option<crate::quant::Row> {
                        Some($row)
                    },
                    fn mul(a: &[u8], b: &[u8], m: usize, n: usize, out: &mut [f32]) {
                        crate::gemm::packed_tiled(a, b, m, n, &crate::gemm::Conf::from(&$row), out)
                    },
                );
            }
        };
    }

    packed_variant!(i2, GemmI2, GemmI2Model, "gemm_i2", I2);
    packed_variant!(i4, GemmI4, GemmI4Model, "gemm_i4", I4);

    pub mod i8 {
        //! `gemm_i8`：对称 int8 矩阵乘，`i32` 精确累加后收敛为 f32。
        use crate::gemm as skel;
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
                    .map(|(p, packed)| i32::from(*packed as i8) * i32::from(b[p * n + j] as i8))
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
        //! `gemm_f32`：稠密 f32 矩阵乘，与标量参考保持同一累加顺序。
        use crate::gemm as skel;
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
                crate::gemm::matmul::<Self>(a, b, m, n, out)
            },
        );
    }
}
pub use variants::{f32, i2, i4, i8};
