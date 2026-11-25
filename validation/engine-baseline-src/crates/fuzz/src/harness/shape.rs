//! libfuzzer harness：形状算术（饱和乘积与亚字节位级取整）。
#![cfg_attr(unix, no_main)]

#[allow(dead_code)]
#[path = "../drivers.rs"]
mod drivers;

include!("../harness.rs");

fuzz_harness!(drivers::Shape);
