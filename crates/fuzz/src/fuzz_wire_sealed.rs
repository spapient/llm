//! libfuzzer harness：线格式三种接收器的一致性（校验和 = crc32、计量 = 写出长度、切片不越界）。
#![cfg_attr(unix, no_main)]

#[allow(dead_code)]
#[path = "drivers.rs"]
mod drivers;

include!("harness.rs");

fuzz_harness!(drivers::Sealed);
