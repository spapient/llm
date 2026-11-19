//! CRC-32（IEEE 802.3，反射多项式 `0xEDB8_8320`）：八路查表的权威实现。
//!
//! 表在编译期由逐位定义展开；一次吃 8 字节，把一条依赖链拆成三条并行查表。正确性另有逐字节
//! 定义的对拍测试钉住，因此"快"不是靠换算法换来的。容器与检查点的尾部校验直接复用本模块，
//! 两条字节路径之间不存在第二份生产实现。

/// 位反转多项式：IEEE 802.3 的 `0x04C1_1DB7` 反射形式。
const POLY: u32 = 0xEDB8_8320;

static SLICE: [[u32; 256]; 8] = slices();

const fn slices() -> [[u32; 256]; 8] {
    let mut slice = [[0u32; 256]; 8];
    let mut index = 0;
    while index < 256 {
        let mut value = index as u32;
        let mut bit = 0;
        while bit < 8 {
            // 无分支：最低位为 1 时掩码全 1，等价于 `(value >> 1) ^ POLY`。
            value = (value >> 1) ^ (POLY & 0u32.wrapping_sub(value & 1));
            bit += 1;
        }
        slice[0][index] = value;
        index += 1;
    }
    // 第 k 张表 = 第 k-1 张表再"过一整字节"。
    let mut level = 1;
    while level < 8 {
        let mut index = 0;
        while index < 256 {
            let previous = slice[level - 1][index];
            slice[level][index] = (previous >> 8) ^ slice[0][(previous & 0xFF) as usize];
            index += 1;
        }
        level += 1;
    }
    slice
}

/// 计算一段字节的 CRC-32。
#[must_use]
#[inline]
pub fn crc32(bytes: &[u8]) -> u32 {
    !update(!0, bytes)
}

/// 从已有状态继续更新，支持分段喂字节。
#[must_use]
#[inline]
pub fn update(mut crc: u32, bytes: &[u8]) -> u32 {
    let (words, rest) = bytes.as_chunks::<8>();
    // 跨块的 CRC 状态是串行链（下一块依赖上一块结果），不能并行；块内 8 位已用八路查表展开。
    for word in words {
        let head = crc ^ u32::from_le_bytes([word[0], word[1], word[2], word[3]]);
        let tail = u32::from_le_bytes([word[4], word[5], word[6], word[7]]);
        crc = SLICE[7][(head & 0xFF) as usize]
            ^ SLICE[6][((head >> 8) & 0xFF) as usize]
            ^ SLICE[5][((head >> 16) & 0xFF) as usize]
            ^ SLICE[4][(head >> 24) as usize]
            ^ SLICE[3][(tail & 0xFF) as usize]
            ^ SLICE[2][((tail >> 8) & 0xFF) as usize]
            ^ SLICE[1][((tail >> 16) & 0xFF) as usize]
            ^ SLICE[0][(tail >> 24) as usize];
    }
    // 尾部不足 8 字节：状态仍逐字节串行，无法再按块展开。
    for byte in rest {
        crc = SLICE[0][((crc ^ u32::from(*byte)) & 0xFF) as usize] ^ (crc >> 8);
    }
    crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_agrees_with_the_bitwise_definition() {
        testkit::assert_table_matches_bitwise(crc32);
    }

    #[test]
    fn chunked_feeds_match_one_shot() {
        let data: Vec<u8> = (0u8..=250).collect();
        let whole = crc32(&data);
        let mut state = !0u32;
        for chunk in data.chunks(7) {
            state = update(state, chunk);
        }
        assert_eq!(!state, whole);
    }
}
