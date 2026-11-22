//! libfuzzer harness：HTTP/1.1 P0 请求头（任意字节，只能稳定拒绝或给出请求行与长度）。
#![cfg_attr(unix, no_main)]

#[allow(dead_code)]
#[path = "drivers.rs"]
mod drivers;

include!("harness.rs");

fuzz_harness!(drivers::HttpHead);
