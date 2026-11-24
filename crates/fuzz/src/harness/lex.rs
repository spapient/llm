//! libfuzzer harness：词法解析（未闭合分组/注释/字符串与饱和标识符）。
#![cfg_attr(unix, no_main)]

#[allow(dead_code)]
#[path = "../drivers.rs"]
mod drivers;

include!("../harness.rs");

fuzz_harness!(drivers::Lex);
