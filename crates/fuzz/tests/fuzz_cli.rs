//! libfuzzer harness：命令行参数文法。
#![no_main]

#[path = "support/drivers.rs"]
mod drivers;
include!("support/harness.rs");

fuzz_harness!(drivers::Invocation);
