//! 量化格式的编译期配置表：位宽、打包密度、对称零点、对称码表与样本偏移单点收敛。
//!
//! GEMV / GEMM / 反量化三族打包格式（`i2` / `i4` / `i8`）过去各抄一遍"每字节几个码、零点是
//! 几、码表怎么排、样本码值怎么取"。这些是**格式事实**，与设备、遍历、校验无关；收口到这里
//! 之后，新增一档精度只填一行 [`Row`]，三条骨架都按这张表读常量，不再各写一份除法与码表。
//!
//! 表内全部是 `const`：运行期一条指令都不留；逐元素格式（`i8`）不打包，码表退化为按
//! `i8` 解释位模式，由骨架分支处理，不在打包表里塞第二套抽象。

use crate::einsum::Bits;

/// 一档对称量化格式的静态配置。
///
/// 字段都能在编译期求值：`bits` 决定流量计费与位域宽度，`cpb`（codes per byte）决定打包
/// 长度折算，`zero` 是无符号码点回退成有符号整数的平移量，`table` 是对称码表（打包格式）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Row {
    /// 元素位宽（闭集 [`Bits`]）。
    pub bits: Bits,
    /// 每字节多少个码：`8 / 位宽`；逐元素为 1。
    pub cpb: usize,
    /// 对称零点（无符号码域 `0..2^bits` 减去它得到有符号整数码）。
    pub zero: i32,
    /// 对称解包表：码点下标 → 有符号整数值；打包格式长度 `2^bits`。
    pub table: &'static [f32],
    /// 样本码值偏移：打包取样时按 `(byte · seed + offsets[code]) mod 2^width` 填位。
    pub offsets: &'static [u32],
}

/// 2 位打包：4 码/字节，零点 2，表 `{-2,-1,0,1}`。
pub const I2: Row = Row {
    bits: Bits::Two,
    cpb: 4,
    zero: 2,
    table: &[-2.0, -1.0, 0.0, 1.0],
    offsets: &[1, 2, 3, 4],
};

/// 4 位打包：2 码/字节，零点 8，表 `{-8..=7}`。
pub const I4: Row = Row {
    bits: Bits::Four,
    cpb: 2,
    zero: 8,
    table: &[
        -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
    ],
    offsets: &[1, 9],
};

/// 8 位逐元素：1 码/字节，零点 0；不打包，`table` 留空由骨架按 `i8` 解释。
pub const I8: Row = Row {
    bits: Bits::Eight,
    cpb: 1,
    zero: 0,
    table: &[],
    offsets: &[],
};

impl Row {
    /// 码域大小：`2^bits`；零点必须落在 `0..domain` 内。
    pub const fn domain(self) -> u32 {
        1 << self.bits.bits()
    }

    /// 打包宽度（位）：`8 / cpb`；逐元素格式为 8。
    pub const fn width(self) -> usize {
        8 / self.cpb
    }

    /// 是否打包（每字节多于一个码）。
    pub const fn packed(self) -> bool {
        self.cpb > 1
    }
}

/// 由每字节码数 `cpb` 推位宽：`8 / cpb`。
///
/// 打包骨架只持有 `cpb`（从格式表传入），位域宽度由此单点派生；`gemv` 打包点积与未来设备
/// 片段生成都读这一份，不在骨架里再写一遍 `8 / cpb`。`cpb` 必须是 8 的正约数。
pub const fn width_cpb(cpb: usize) -> usize {
    8 / cpb
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 三条打包/逐元素配置自洽：每字节码数 = 8 / 位宽；打包表长度等于码域；零点落在码域内。
    #[test]
    fn rows_are_self_consistent() {
        for row in [I2, I4, I8] {
            assert_eq!(row.cpb, 8 / row.bits.bits() as usize, "cpb 必须由位宽推出");
            assert!(
                row.zero >= 0 && row.zero < row.domain() as i32,
                "零点必须在码域内"
            );
            if row.packed() {
                assert_eq!(
                    row.table.len(),
                    row.domain() as usize,
                    "打包表必须覆盖整个码域"
                );
                assert_eq!(row.offsets.len(), row.cpb, "样本偏移与每字节码槽同长");
            }
        }
    }

    /// 对称表逐位回退：`table[code] = code − zero`，与骨架解包口径一致。
    #[test]
    fn packed_tables_are_symmetric() {
        for row in [I2, I4] {
            for (code, &value) in row.table.iter().enumerate() {
                assert_eq!(
                    value,
                    (code as i32 - row.zero) as f32,
                    "code {code} 表值不符"
                );
            }
        }
    }
}
