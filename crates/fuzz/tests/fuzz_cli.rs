//! libfuzzer harness：命令行参数文法。
#![cfg_attr(unix, no_main)]

include!("support.rs");

fuzz_harness!(drivers::Invocation);
