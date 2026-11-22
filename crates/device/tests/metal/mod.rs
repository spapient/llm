//! 职责：Metal 执行器与内核的真机测试装配。
//!
//! 子模块按被测面切开：契约冒烟 / GEMM 基准 / 注意力 / 张量核路径。
//! 本文件只做装配与共享夹具，不含任何断言。

use super::harness::*;
use super::*;
use llm_optim::backend::TensorGeometry;
use llm_optim::execution::Backend::Metal;
use llm_optim::execution::{Mode, program};

mod attention;
mod contract;
mod gemm;
mod tensor;

/// 立方 GEMM 的自建样本：只带三个标量与空缓冲，形状由 `m/n/k` 直接给出。
fn tensor_sample(m: usize, n: usize, k: usize) -> op::model::Sample {
    op::model::Sample {
        m,
        n,
        scalars: vec![("M", m as i64), ("N", n as i64), ("K", k as i64)],
        buffers: Vec::new(),
    }
}
