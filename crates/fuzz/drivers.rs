//! 驱动侧适配：把真实解析器接成 [`llm_fuzz::Target`]。
//!
//! 这份定义是**唯一出处**：确定性遍历（`tests/parsers.rs` 的 [`llm_fuzz::sweep`]）与
//! libfuzzer 模糊 harness（`tests/fuzz_*.rs`）都从这里取同一组入口。两边不各写一份判据，避免
//! 文本/字节边界的处理在两处悄悄漂移。
//!
//! 运行时只把"任意字节 → 接受/拒绝"的结果交回契约层；解析器本身在 `cli` / `weights`。
//!
//! 每个 harness 例子只取其中一个适配，未被该消费方引用的项是有意保留的整组定义，不报 dead_code。
#![allow(dead_code)]

use diagnostic::{Code, Error};
use llm_fuzz::Target;

/// 文本解析器的公共外壳：非 UTF-8 不是"解析失败"，而是"这份输入不适用于文本入口"。
///
/// 仍然是一个合法的拒绝（带类别的非空消息），因此不违反 [`Target`] 三契约；把它区分出来，
/// 是为了之后加字节入口时不必再想一遍。
pub fn text(input: &[u8]) -> Result<&str, Error> {
    std::str::from_utf8(input)
        .map_err(|_| Error::new(Code::Invalid, "输入不是 UTF-8").with("fuzz::drivers"))
}

/// 命令行：空白切词，再交给唯一的文法。
pub struct Invocation;

impl Target for Invocation {
    fn name(&self) -> &str {
        "cli::Invocation"
    }

    fn run(&self, input: &[u8]) -> Result<(), Error> {
        let words: Vec<String> = text(input)?
            .split_whitespace()
            .map(str::to_string)
            .collect();
        cli::Invocation::parse(&words).map(|_| ())
    }
}

/// safetensors 头部里的 JSON。
pub struct Json;

impl Target for Json {
    fn name(&self) -> &str {
        "weights::json"
    }

    fn run(&self, input: &[u8]) -> Result<(), Error> {
        weights::json::parse(text(input)?).map(|_| ())
    }
}

/// safetensors 整体：8 字节头长 + JSON 头 + 数据段。
pub struct Safetensors;

impl Target for Safetensors {
    fn name(&self) -> &str {
        "weights::safetensors::Archive"
    }

    fn run(&self, input: &[u8]) -> Result<(), Error> {
        weights::safetensors::Archive::parse(input).map(|_| ())
    }
}
