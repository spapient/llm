//! L1 确定性随机基础件：计数器型 Philox4x32-10 引擎与自包含的分布闭集。

#![deny(unsafe_code)]

pub mod dist;
pub mod philox;

pub use dist::{Bernoulli, Normal, Uniform};
pub use philox::{Rng, block};

#[cfg(test)]
#[path = "../tests/prop_numeric.rs"]
mod prop_numeric;
