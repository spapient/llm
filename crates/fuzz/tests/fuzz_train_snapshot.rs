//! libfuzzer harness：训练检查点字节格式（魔数 + 版本 + 事实 + 条目表 + CRC-32）。
#![cfg_attr(unix, no_main)]

#[path = "support/drivers.rs"]
mod drivers;
include!("support/harness.rs");

fuzz_harness!(drivers::Snapshot);
