//! libfuzzer harness：分布闭集与 Philox 流的语义不变量。
#![cfg_attr(unix, no_main)]

#[allow(dead_code)]
#[path = "drivers.rs"]
mod drivers;

include!("harness.rs");

fuzz_harness!(drivers::Dist);
