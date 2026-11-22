//! libfuzzer harness：自有 `.u` 容器 v2（段表 + 对齐数据 + 两级 CRC 尾部）。
#![cfg_attr(unix, no_main)]

#[allow(dead_code)]
#[path = "drivers.rs"]
mod drivers;

include!("harness.rs");

fuzz_harness!(drivers::WeightsArchive);
