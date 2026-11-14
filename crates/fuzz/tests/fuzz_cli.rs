//! libfuzzer harness：命令行参数文法。
//!
//! 入口定义与确定性 sweep 用例同一份 [`drivers::Invocation`]；这里只把 libFuzzer 喂来的
//! 字节交给它。解析拒绝是合法结果，只有 panic 会被 libFuzzer 记为崩溃。
#![no_main]

#[path = "../drivers.rs"]
mod drivers;

use libfuzzer_sys::fuzz_target;
use llm_fuzz::Target;

fuzz_target!(|data: &[u8]| {
    let _ = drivers::Invocation.run(data);
});
