//! libfuzzer harness：服务端帧解码（半帧、超限长度、坏尾部）。
#![no_main]

#[path = "support/drivers.rs"]
mod drivers;
include!("support/harness.rs");

fuzz_harness!(drivers::Frames);
