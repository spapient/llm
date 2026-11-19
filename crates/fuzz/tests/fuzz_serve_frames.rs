//! libfuzzer harness：服务端帧解码（半帧、超限长度、坏尾部）。
#![cfg_attr(unix, no_main)]

include!("support.rs");

fuzz_harness!(drivers::Frames);
