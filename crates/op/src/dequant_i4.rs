//! 4 位反量化：打包字节 `q`（每字节两个 nibble），`y = scale * (nibble - 8)`。
//!
//! 低 4 位在前、高 4 位在后；nibble 是无符号 0..15，按对称零点 8 转有符号。打包骨架在
//! [`crate::dequant`]，本文件只写位宽与零点，`#[annotation::model]` 装配。

use crate::quant::I4;

crate::dequant_packed!(
    DequantI4,
    DequantI4F32,
    "dequant_i4",
    "../tests/dequant_i4_tests.rs",
    I4
);
