//! libfuzzer harness：自有 `.u` 容器 v2（描述符表 + 对齐数据段 + 两级 CRC 尾部）。
#![cfg_attr(unix, no_main)]

#[path = "support/drivers.rs"]
mod drivers;
include!("support/harness.rs");

fuzz_harness!(drivers::UContainer);
