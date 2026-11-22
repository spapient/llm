//! libfuzzer harness：服务端帧解码（任意字节流里能切出几帧、半帧断开怎么算）。
#![cfg_attr(unix, no_main)]

#[allow(dead_code)]
#[path = "drivers.rs"]
mod drivers;

include!("harness.rs");

fuzz_harness!(drivers::ServeFrames);
