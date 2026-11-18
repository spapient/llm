//! `gemv_i4` 的算子语义：4 位打包权重 + 每 [`GROUP`] 个元素一条标度。
//!
//! 骨架在 [`crate::gemv`]，本文件只有"4 位打包"这一份配置：位宽、别名与测试挂载。
//! 两个 4 位码共用一个字节（低 4 位在前），因此输入宽度 `m` 必须是 2 的倍数且整除分组，
//! 奇数尾码在解包前拦下；解包与分组推进的算术都在 [`crate::gemv::Packed`]。

use crate::quant::I4;

crate::gemv_packed!(I4Conf, I4, "gemv_i4", "../tests/gemv_i4_tests.rs", I4);
