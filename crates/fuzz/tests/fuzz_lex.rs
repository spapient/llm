//! libfuzzer harness：工具词法解析。
#![cfg_attr(unix, no_main)]

#[path = "support/drivers.rs"]
mod drivers;
include!("support/harness.rs");

fuzz_harness!(drivers::Lex);
