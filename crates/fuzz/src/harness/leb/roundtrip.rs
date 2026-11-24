//! libfuzzer harness：LEB128 规范往返（任意 u64 与任意原始字节前缀）。
#![cfg_attr(unix, no_main)]

#[allow(dead_code)]
#[path = "../../drivers.rs"]
mod drivers;

include!("../../harness.rs");

fuzz_harness!(drivers::LebRoundtrip);
