//! 位级 CRC-32 参考定义：与各 crate 的查表实现独立，供对拍共享，免得每份测试重写一遍。

/// CRC-32 反射多项式（IEEE 802.3 的 `0x04C1_1DB7` 反射形式）。
const CRC_POLY: u32 = 0xEDB8_8320;

/// CRC-32 的逐字节参考定义。
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

/// 把某个 crate 的查表 CRC-32 实现与位级参考对拍：`0..=200` 递增序列、空串与标准 check 值。
pub fn assert_table_matches_bitwise(crc32: impl Fn(&[u8]) -> u32) {
    let data: Vec<u8> = (0u8..=200).collect();
    assert_eq!(crc32(&data), crc32_bitwise(&data));
    assert_eq!(crc32(b""), crc32_bitwise(b""));
    assert_eq!(crc32(b"123456789"), 0xCBF4_3926, "标准 check 值");
}
