//! CRC-32（IEEE 802.3，反射多项式 `0xEDB8_8320`）：帧的完整性校验。
//!
//! 表在编译期由逐位定义展开。正确性由逐位定义的对拍与标准检验值钉住，因此"查表"不是
//! "换了一套算法"。
//!
//! 与 `pack` 各持一份实现是有意的：容器覆盖的是整份文件（段头、对齐填充与全局段），帧覆盖的
//! 是"帧头 + 载荷"这一小段；为二十行算术让协议层依赖权重容器，比重复这张表更贵。

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
mod tests {
    use super::*;

    /// 逐字节参考定义：与查表实现独立，用来对拍。
    fn bitwise(bytes: &[u8]) -> u32 {
        let mut crc = !0u32;
        for byte in bytes {
            crc ^= u32::from(*byte);
            for _ in 0..8 {
                crc = if crc & 1 == 1 {
                    (crc >> 1) ^ POLY
                } else {
                    crc >> 1
                };
            }
        }
        !crc
    }

    #[test]
    fn the_table_agrees_with_the_bitwise_definition() {
        let data: Vec<u8> = (0u8..=200).collect();
        assert_eq!(crc32(&data), bitwise(&data));
        assert_eq!(crc32(b""), bitwise(b""));
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926, "标准 check 值");
    }
}
