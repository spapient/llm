//! libfuzzer harness：safetensors 整体归档（8 字节头长 + JSON 头 + 数据段）。
#![no_main]

#[path = "support/drivers.rs"]
mod drivers;
include!("support/harness.rs");

fuzz_harness!(drivers::Safetensors);
