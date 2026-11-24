//! libfuzzer harness：LEB128 游标（非规范编码、容量越界、字节不足都必须稳定拒绝）。
#![cfg_attr(unix, no_main)]

#[allow(dead_code)]
#[path = "../../drivers.rs"]
mod drivers;

include!("../../harness.rs");

fuzz_harness!(drivers::WireLeb);
