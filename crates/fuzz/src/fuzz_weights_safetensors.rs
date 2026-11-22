//! libfuzzer harness：safetensors 整体（8 字节头长 + JSON 头 + 数据段）。
#![cfg_attr(unix, no_main)]

#[allow(dead_code)]
#[path = "drivers.rs"]
mod drivers;

include!("harness.rs");

fuzz_harness!(drivers::Safetensors);
