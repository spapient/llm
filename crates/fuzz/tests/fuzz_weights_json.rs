//! libfuzzer harness：safetensors 头部 JSON。
#![cfg_attr(unix, no_main)]

include!("support.rs");

fuzz_harness!(drivers::Json);
