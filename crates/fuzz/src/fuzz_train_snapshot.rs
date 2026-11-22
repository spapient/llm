//! libfuzzer harness：训练检查点（魔数 + 版本 + 事实 + 条目表 + CRC-32 尾部）。
#![cfg_attr(unix, no_main)]

#[allow(dead_code)]
#[path = "drivers.rs"]
mod drivers;

include!("harness.rs");

fuzz_harness!(drivers::Snapshot);
