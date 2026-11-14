//! libfuzzer harness：safetensors 整体归档（8 字节头长 + JSON 头 + 数据段）。
//!
//! 入口定义与确定性 sweep 用例同一份 [`drivers::Safetensors`]；解析拒绝是合法结果，只有
//! panic 会被 libFuzzer 记为崩溃。
#![no_main]

#[path = "../drivers.rs"]
mod drivers;

use libfuzzer_sys::fuzz_target;
use llm_fuzz::Target;

fuzz_target!(|data: &[u8]| {
    let _ = drivers::Safetensors.run(data);
});
