//! libfuzzer harness：命令行文法（任意 argv 文本，含空串与非法 UTF-8）。
#![cfg_attr(unix, no_main)]

#[allow(dead_code)]
#[path = "drivers.rs"]
mod drivers;

include!("harness.rs");

fuzz_harness!(drivers::Cli);
