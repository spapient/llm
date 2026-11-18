//! 2 位反量化：打包字节 `q`（每字节四个 2 位码），`y = scale * (code - 2)`。
//!
//! 低 2 位在前、高位在后；码是无符号 0..3，按对称零点 2 转成 -2..=1。打包骨架在
//! [`crate::dequant`]，本文件只写位宽与零点，`#[annotation::model]` 装配。

use crate::quant::I2;

crate::dequant_packed!(
    DequantI2,
    DequantI2F32,
    "dequant_i2",
    "../tests/dequant_i2_tests.rs",
    I2
);
