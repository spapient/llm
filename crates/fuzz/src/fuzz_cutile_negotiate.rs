//! libfuzzer harness：内核能力协商的产物契约（命中变体可回指到降级链）。
#![cfg_attr(unix, no_main)]

#[allow(dead_code)]
#[path = "drivers.rs"]
mod drivers;

include!("harness.rs");

fuzz_harness!(drivers::CutileNegotiate);
