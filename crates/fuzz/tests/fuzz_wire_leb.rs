//! libfuzzer harness：LEB128 变长整数游标（非规范编码、超宽序列、截断字节）。
#![cfg_attr(unix, no_main)]

#[path = "support/drivers.rs"]
mod drivers;
include!("support/harness.rs");

fuzz_harness!(drivers::Leb);
