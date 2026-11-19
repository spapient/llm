//! libfuzzer harness：LEB128 变长整数游标（非规范编码、超宽序列、截断字节）。
#![cfg_attr(unix, no_main)]

include!("support.rs");

fuzz_harness!(drivers::Leb);
