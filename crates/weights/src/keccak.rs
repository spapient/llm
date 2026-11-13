//! `Keccak-f[1600]` 与 SHAKE 可扩展输出函数（FIPS 202），零依赖、纯安全 u64 算术。
//!
//! ML-DSA（FIPS 204）的所有哈希都走 SHAKE，不使用定长 SHA-2/3：本模块是格签名的地基。
//! 状态是 5×5 个 u64 lane（小端映射字节），24 轮置换 θ/ρ/π/χ/ι。

/// 24 个轮常数。
const RC: [u64; 24] = [
    0x0000_0000_0000_0001,
    0x0000_0000_0000_8082,
    0x8000_0000_0000_808a,
    0x8000_0000_8000_8000,
    0x0000_0000_0000_808b,
    0x0000_0000_8000_0001,
    0x8000_0000_8000_8081,
    0x8000_0000_0000_8009,
    0x0000_0000_0000_008a,
    0x0000_0000_0000_0088,
    0x0000_0000_8000_8009,
    0x0000_0000_8000_000a,
    0x0000_0000_8000_808b,
    0x8000_0000_0000_008b,
    0x8000_0000_0000_8089,
    0x8000_0000_0000_8003,
    0x8000_0000_0000_8002,
    0x8000_0000_0000_0080,
    0x0000_0000_0000_800a,
    0x8000_0000_8000_000a,
    0x8000_0000_8000_8081,
    0x8000_0000_0000_8080,
    0x0000_0000_8000_0001,
    0x8000_0000_8000_8008,
];

/// ρ 旋转偏移表，按 `[x][y]` 索引。
const ROT: [[u32; 5]; 5] = [
    [0, 36, 3, 41, 18],
    [1, 44, 10, 45, 2],
    [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39, 8, 14],
];

/// `Keccak-f[1600]`：24 轮原地置换。
fn keccak_f(state: &mut [u64; 25]) {
    for &rc in RC.iter() {
        // θ
        let mut c = [0u64; 5];
        for x in 0..5 {
            c[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
        }
        let mut d = [0u64; 5];
        for x in 0..5 {
            d[x] = c[(x + 4) % 5] ^ c[(x + 1) % 5].rotate_left(1);
        }
        for x in 0..5 {
            for y in 0..5 {
                state[x + 5 * y] ^= d[x];
            }
        }

        // ρ + π：B[y, 2x+3y] = rot(A[x,y])
        let mut b = [0u64; 25];
        for x in 0..5 {
            for y in 0..5 {
                b[y + 5 * ((2 * x + 3 * y) % 5)] = state[x + 5 * y].rotate_left(ROT[x][y]);
            }
        }

        // χ
        for x in 0..5 {
            for y in 0..5 {
                state[x + 5 * y] =
                    b[x + 5 * y] ^ ((!b[((x + 1) % 5) + 5 * y]) & b[((x + 2) % 5) + 5 * y]);
            }
        }

        // ι
        state[0] ^= rc;
    }
}

/// 海绵构造：吸收输入后挤出 `out_len` 字节。`rate` 是字节率，`suffix` 是域分隔后缀。
fn sponge(rate: usize, suffix: u8, input: &[u8], out_len: usize) -> Vec<u8> {
    let mut state = [0u64; 25];

    // 多速率填充 pad10*1：先放域分隔后缀，补零到块对齐，末字节置终止位 1。
    let mut msg = input.to_vec();
    msg.push(suffix);
    while !msg.len().is_multiple_of(rate) {
        msg.push(0);
    }
    let last = msg.len() - 1;
    msg[last] |= 0x80;

    // 吸收
    for block in msg.chunks(rate) {
        for (i, &byte) in block.iter().enumerate() {
            state[i / 8] ^= u64::from(byte) << (8 * (i % 8));
        }
        keccak_f(&mut state);
    }

    // 挤出
    let mut out = Vec::with_capacity(out_len);
    while out.len() < out_len {
        let mut block = Vec::with_capacity(rate);
        for lane in &state {
            block.extend_from_slice(&lane.to_le_bytes());
        }
        let take = rate.min(out_len - out.len());
        out.extend_from_slice(&block[..take]);
        keccak_f(&mut state);
    }
    out
}

/// SHAKE128：率 168 字节（1344 bit），域分隔后缀 0x1F。
pub fn shake128(input: &[u8], out_len: usize) -> Vec<u8> {
    sponge(168, 0x1f, input, out_len)
}

/// SHAKE256：率 136 字节（1088 bit），域分隔后缀 0x1F。
pub fn shake256(input: &[u8], out_len: usize) -> Vec<u8> {
    sponge(136, 0x1f, input, out_len)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn shake128_empty_matches_known_vector() {
        // 空输入 SHAKE128 前 32 字节，标准向量。
        assert_eq!(
            hex(&shake128(&[], 32)),
            "7f9c2ba4e88f827d616045507605853ed73b8093f6efbc88eb1a6eacfa66ef26"
        );
    }

    #[test]
    fn shake256_empty_matches_known_vector() {
        assert_eq!(
            hex(&shake256(&[], 64)),
            "46b9dd2b0ba88d13233b3feb743eeb243fcd52ea62b81b82b50c27646ed5762f\
             d75dc4ddd8c0f200cb05019d67b592f6fc821c49479ab48640292eacb3b7c4be"
        );
    }

    #[test]
    fn output_can_span_many_blocks() {
        // 跨多个挤出块的长输出，长度精确且确定。
        let out = shake256(b"abc", 1000);
        assert_eq!(out.len(), 1000);
        assert_eq!(out, shake256(b"abc", 1000));
    }
}
