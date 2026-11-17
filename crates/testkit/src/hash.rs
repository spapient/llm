//! 位级 CRC-32 参考定义：与各 crate 的查表实现独立，供对拍共享，免得每份测试重写一遍。

/// CRC-32 反射多项式（IEEE 802.3 的 `0x04C1_1DB7` 反射形式）。
const CRC_POLY: u32 = 0xEDB8_8320;

/// CRC-32 的逐字节参考定义。
///
/// 逐位循环是 CRC 移位寄存器的固有串行依赖：每一位的余数由上一位移出的比特决定，
/// 无法整块查表；这一份刻意保留位级写法，作为查表实现的独立对拍基准。
#[must_use]
pub fn crc32_bitwise(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for byte in bytes {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ CRC_POLY
            } else {
                crc >> 1
            };
        }
    }
    !crc
}
