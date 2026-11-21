//! 驱动方：把每个"接受不受信任输入"的解析器灌进 [`sweep`]。
//!
//! 种子是**一份合法输入**，派生用例是它的确定性扰动：于是遍历走的是"边角上的一条错误路径"，
//! 而不是与真实输入毫无关系的噪声——真实缺陷几乎总在这些边角上（多一个字节的长度、写错的
//! 分组数、半个转义序列）。同一 `(seed, case)` 在任何机器上都得到同一串字节，失败可复现。
//!
//! 覆盖文本与字节两类边界：`cli` 的参数文法、`tools` 的词法解析、`weights` 的 JSON /
//! safetensors / 自有 `.u` 容器、`train` 的检查点、`serve` 的帧解码、`wire` 的 LEB128 游标，以及
//! `wire` 三种接收器的
//! 不变量。它们都是"外部来的字节"，也都把错误当返回值，因此"不 panic / 结论稳定 / 拒绝有
//! 话说"三条契约在这里是同一件事；最后一个不拒绝任何输入，它验的是"三条承诺互相一致"。
//!
//! 适配定义统一来自 [`drivers`]，合法范例统一来自 [`seeds`]，与 libfuzzer harness、语料
//! 生成器共用，避免判据与种子两份漂移。
//!
//! **不再覆盖 TOML**：`ops/`、`devices/`、`lints/vocabulary.toml` 在构建期就被解析成表
//! （ADR 0022），运行期没有 TOML 入口可灌；`op` 的声明入口要一份模板正文作上下文，由
//! `op` 自己的单测逐条覆盖。GGUF 与 pickle 容器在仓内没有编码器，种子只能手抄字节，格式
//! 一演进就会悄悄变成"合法种子其实不合法"，因此不进遍历。

#[path = "support.rs"]
mod support;

use support::{drivers, seeds};

use drivers::{
    Frames, HttpHead, Invocation, Json, Leb, Lex, Safetensors, Sealed, Snapshot, UContainer,
};
use llm_fuzz::{Seed, Target, sweep};

/// 每个目标跑多少条派生用例。
///
/// 512 条 × 9 个目标 = 4608 次解析（每条还跑两遍验稳定），边际成本是毫秒级；数量取到"能
/// 覆盖每个字节位置被扰动一次"就够——加密强度的随机源在这里没有价值，可复现才有。
const CASES: u64 = 512;

/// 全部被测入口及其合法种子；适配器顺序与 [`seeds::exemplars`] 一一对应，种子字节只在
/// 范例清单里构造一次，遍历与种子校验两条测试共用这份配对。
fn targets() -> [(&'static dyn Target, Seed); 10] {
    let adapters: [&'static dyn Target; 10] = [
        &Invocation,
        &Json,
        &Safetensors,
        &UContainer,
        &Snapshot,
        &Frames,
        &HttpHead,
        &Sealed,
        &Leb,
        &Lex,
    ];
    let exemplars = seeds::exemplars();
    std::array::from_fn(|i| (adapters[i], Seed::new(exemplars[i].1.clone())))
}

#[test]
fn every_parser_turns_arbitrary_bytes_into_a_stable_verdict() {
    for (target, seed) in targets() {
        if let Err(finding) = sweep(target, &seed, CASES) {
            panic!("{finding}");
        }
    }
}

/// 种子本身必须是**被接受**的：否则整个遍历只是在错误路径上打转，
/// "合法输入也只许有一种结论"这条就没人验。
#[test]
fn every_seed_is_a_valid_input() {
    for (target, seed) in targets() {
        assert!(
            target.run(seed.bytes()).is_ok(),
            "{} 的种子应当被接受：{}",
            target.name(),
            llm_fuzz::repro(seed.bytes())
        );
    }
}
