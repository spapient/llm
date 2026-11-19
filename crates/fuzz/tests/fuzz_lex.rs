//! libfuzzer harness：工具词法解析。
#![cfg_attr(unix, no_main)]

include!("support.rs");

fuzz_harness!(drivers::Lex);
