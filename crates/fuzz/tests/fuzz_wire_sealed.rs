//! libfuzzer harness：线格式密封、计量与切片接收器（校验和、长度、内容一致性）。
#![no_main]

#[path = "support/drivers.rs"]
mod drivers;
include!("support/harness.rs");

fuzz_harness!(drivers::Sealed);
