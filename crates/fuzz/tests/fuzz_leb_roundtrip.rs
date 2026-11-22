//! libfuzzer harness：LEB128 规范往返（任意 u64 与任意原始字节前缀）。
#![cfg_attr(unix, no_main)]

include!("support.rs");

fuzz_harness!(drivers::LebRoundtrip);
