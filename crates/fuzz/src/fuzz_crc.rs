//! libfuzzer harness：CRC-32 分块续算与逐位参考对拍（长文、空串、任意切点）。
#![cfg_attr(unix, no_main)]

#[allow(dead_code)]
#[path = "drivers.rs"]
mod drivers;

include!("harness.rs");

fuzz_harness!(drivers::Crc);
