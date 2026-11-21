use super::{K, TILE_N};

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

/// 位流 → 整数码：每个码一个 `i32`，按行主序连续铺开（A 行宽 [`super::K`]、B 行宽 `n`）。
///
/// 解包次数从此只与 `m·K + K·n` 有关，而不是与 `m·n·K` 有关——移位、掩码与零点平移正是量化
/// 格式的主要成本，旧做法在每个输出元素上把它们重做一遍。
pub fn decode_stream(codes: &mut [i32], bytes: &[u8], conf: &Conf) {
    for (index, code) in codes.iter_mut().enumerate() {
        let byte = bytes[index / conf.cpb];
        *code = unpack(byte, index % conf.cpb, conf.cpb, conf.zero) as i32;
    }
}

/// 整数码的 tile 累加：外层按 [`super::TILE_N`] 走列块，块内 `p` 在外、列在内。
///
/// 每个输出元素仍按 `p` 升序做 `i32` 加法，于是与 [`super::matmul`] / [`dot`] 逐位相等；
/// 列在内让 B 的列块段与累加器都是连续读写，内层是一次可向量化的 AXPY。
///
/// `codes_a` 至少 `m·K` 个码、`codes_b` 至少 `K·n` 个，`out` 恰为 `m·n`；长度由
/// [`super::forward`] 在入口校验，这里不再重复判——tile 内层每多一次分支就多一次整块的成本。
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

/// 逐元素格式的整数码：`u8` 位模式按 `i8` 解释，与 [`super::Gemm::dot`] 的取码口径同一份。
///
/// 打包格式不用它——位流解包还带零点平移（[`decode_stream`]）。
pub fn signed_codes(values: &[u8]) -> Vec<i32> {
    values.iter().map(|&byte| byte as i8 as i32).collect()
}

/// 打包格式的本体：两条位流各解一遍，再走 [`accumulate_tiled`]。
///
/// A 是 `m × K` 个码、B 是 `K × n` 个码，解包结果只在这一个调用里存活；块内不再碰位流。
pub fn packed_tiled(a: &[u8], b: &[u8], m: usize, n: usize, conf: &Conf, out: &mut [f32]) {
    let mut codes_a = vec![0i32; m * K];
    let mut codes_b = vec![0i32; K * n];
    decode_stream(&mut codes_a, a, conf);
    decode_stream(&mut codes_b, b, conf);
    accumulate_tiled(&codes_a, &codes_b, m, n, out);
}
