//! libfuzzer harness：线格式的密封、计量与切片接收器。
//!
//! 入口定义与确定性 sweep 用例同一份 [`drivers::Sealed`]。这一条验的是不变量而不是接受/拒绝：
//! 校验和、长度、缓冲区内容三者对同一批字节必须互相对得上。
#![no_main]

#[path = "../drivers.rs"]
mod drivers;

use libfuzzer_sys::fuzz_target;
use llm_fuzz::Target;

fuzz_target!(|data: &[u8]| {
    let _ = drivers::Sealed.run(data);
});
