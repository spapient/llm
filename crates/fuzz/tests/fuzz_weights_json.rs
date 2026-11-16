//! libfuzzer harness：safetensors 头部 JSON。
#![no_main]

#[path = "support/drivers.rs"]
mod drivers;
include!("support/harness.rs");

fuzz_harness!(drivers::Json);
