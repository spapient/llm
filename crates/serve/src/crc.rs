//! CRC-32（IEEE 802.3，反射多项式 `0xEDB8_8320`）：帧的完整性校验。
//!
//! 表在编译期由逐位定义展开。正确性由逐位定义的对拍与标准检验值钉住，因此"查表"不是
//! "换了一套算法"。
//!
//! 与 `weights` 各持一份实现是有意的：本 crate 只依赖 `diagnostic`，线协议需要保持依赖最小化
//! （便于 no-std / wasm 等不同 profile）；帧只覆盖"帧头 + 载荷"这一小段，单字节表已经足够，
//! 让协议层为这二十行算术拖入权重容器的整条依赖链比重复这张表更贵。

/// 位反转多项式：IEEE 802.3 的 `0x04C1_1DB7` 反射形式。
const POLY: u32 = 0xEDB8_8320;

static TABLE: [u32; 256] = table();

const fn table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut index = 0;
    while index < 256 {
        let mut value = index as u32;
        let mut bit = 0;
        while bit < 8 {
            value = if value & 1 == 1 {
                (value >> 1) ^ POLY
            } else {
                value >> 1
            };
            bit += 1;
        }
        table[index] = value;
        index += 1;
    }
    table
}

/// 计算一段字节的 CRC-32。
pub fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for byte in bytes {
        crc = TABLE[((crc ^ u32::from(*byte)) & 0xFF) as usize] ^ (crc >> 8);
    }
    !crc
}

#[cfg(test)]
#[path = "../tests/crc_tests.rs"]
mod tests;
