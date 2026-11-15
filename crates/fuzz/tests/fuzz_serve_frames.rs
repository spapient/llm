//! libfuzzer harness：服务端帧解码（半帧、超限长度、坏尾部）。
//!
//! 入口定义与确定性 sweep 用例同一份 [`drivers::Frames`]。解码器是有状态的（缓冲 +
//! poison 标志），随机字节流下的"取完再关"最容易暴露吞字节的问题。
#![no_main]

#[path = "../drivers.rs"]
mod drivers;

use libfuzzer_sys::fuzz_target;
use llm_fuzz::Target;

fuzz_target!(|data: &[u8]| {
    let _ = drivers::Frames.run(data);
});
