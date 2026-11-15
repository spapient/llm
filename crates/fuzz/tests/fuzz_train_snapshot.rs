//! libfuzzer harness：训练检查点的字节格式（魔数 + 版本 + 事实 + 条目表 + CRC-32）。
//!
//! 入口定义与确定性 sweep 用例同一份 [`drivers::Snapshot`]。检查点的错误路径长在
//! "校验和对不上"与"校验和恰好对上、正文却坏"两处，后者只有模糊测试能走到。
#![no_main]

#[path = "../drivers.rs"]
mod drivers;

use libfuzzer_sys::fuzz_target;
use llm_fuzz::Target;

fuzz_target!(|data: &[u8]| {
    let _ = drivers::Snapshot.run(data);
});
