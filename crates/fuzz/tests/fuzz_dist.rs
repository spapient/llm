//! libfuzzer harness：分布闭集与 Philox 流的语义不变量。
#![cfg_attr(unix, no_main)]

include!("support.rs");

fuzz_harness!(drivers::DistSampler);
