//! libfuzzer harness：线格式密封、计量与切片接收器（校验和、长度、内容一致性）。
#![cfg_attr(unix, no_main)]

include!("support.rs");

fuzz_harness!(drivers::Sealed);
