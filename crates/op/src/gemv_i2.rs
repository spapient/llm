//! `gemv_i2` 的算子语义：2 位打包权重 + 每 [`GROUP`] 个元素一条标度。
//!
//! 骨架在 [`crate::gemv`]，本文件只有"2 位打包"这一份配置：位宽、别名与测试挂载。
//! 四个 2 位码共用一个字节（低 2 位在前），因此输入宽度 `m` 必须是 4 的倍数且整除分组，
//! 非 4 倍数尾码在解包前拦下；解包与分组推进的算术都在 [`crate::gemv::Packed`]。

use crate::quant::I2;

crate::gemv_packed!(I2Conf, I2, "gemv_i2", "../tests/gemv_i2_tests.rs", I2);
