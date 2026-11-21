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
