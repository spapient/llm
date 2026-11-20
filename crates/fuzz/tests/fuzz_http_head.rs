//! libfuzzer harness：HTTP/1.1 P0 请求头（请求行、Content-Length、截断与冲突）。
#![cfg_attr(unix, no_main)]

include!("support.rs");

fuzz_harness!(drivers::HttpHead);
