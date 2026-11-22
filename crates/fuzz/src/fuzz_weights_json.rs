//! libfuzzer harness：safetensors 头部的 JSON（任意文本，含截断与重复键）。
#![cfg_attr(unix, no_main)]

#[allow(dead_code)]
#[path = "drivers.rs"]
mod drivers;

include!("harness.rs");

fuzz_harness!(drivers::Json);
