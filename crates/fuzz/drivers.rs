//! 驱动侧适配：把真实解析器接成 [`llm_fuzz::Target`]。
//!
//! 这份定义是**唯一出处**：确定性遍历（`tests/parsers.rs` 的 [`llm_fuzz::sweep`]）与
//! libfuzzer 模糊 harness（`tests/fuzz_*.rs`）都从这里取同一组入口。两边不各写一份判据，避免
//! 文本/字节边界的处理在两处悄悄漂移。
//!
//! 运行时只把"任意字节 → 接受/拒绝"的结果交回契约层；解析器本身在 `cli` / `weights` /
//! `train::snapshot` / `serve::proto`，不变量本身在 `wire`。
//!
//! 每个 harness 例子只取其中一个适配，未被该消费方引用的项是有意保留的整组定义，不报 dead_code。
#![allow(dead_code)]

use diagnostic::{Code, Error};
use llm_fuzz::Target;
use wire::sink::Sink;

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

/// 训练检查点：魔数 + 版本 + 事实 + 条目表 + CRC-32 尾部。
///
/// `decode` 内部先跑 `verify`，所以这一条同时覆盖"校验和在错误路径上不 panic"与
/// "正文解析只在校验通过后才开始"两件事。
pub struct Snapshot;

impl Target for Snapshot {
    fn name(&self) -> &str {
        "train::snapshot::Snapshot"
    }

    fn run(&self, input: &[u8]) -> Result<(), Error> {
        train::snapshot::Snapshot::decode(input).map(|_| ())
    }
}

/// 服务端帧解码：任意字节流的"能切出几帧、剩下的算不算坏"。
///
/// 逐帧取尽再 `finish`：只调 `push` 会漏掉"半帧断开"那条路径，而它正是协议最容易犯的
/// 安静错误（把半个请求当成完整的）。
pub struct Frames;

impl Target for Frames {
    fn name(&self) -> &str {
        "serve::proto::Decoder"
    }

    fn run(&self, input: &[u8]) -> Result<(), Error> {
        let mut decoder = serve::proto::Decoder::new();
        decoder.push(input)?;
        while decoder.next_frame()?.is_some() {}
        decoder.finish()
    }
}

/// 线格式的密封与计量：同一批字节经三种 `Sink` 必须得到互相一致的结论。
///
/// 这一条不拒绝任何输入（`Sink` 没有"格式错"这种状态），它验的是**不变量**：密封算出的
/// 校验和等于 `crc32`、计量长度等于实际写出长度、切片接收器在容量够时置位标志且不越界。
/// 三条里任何一条在某个字节位置翻转，都会让下游的长度承诺与校验承诺对不上。
pub struct Sealed;

impl Target for Sealed {
    fn name(&self) -> &str {
        "wire::sink::Sealed"
    }

    fn run(&self, input: &[u8]) -> Result<(), Error> {
        let mut written: Vec<u8> = Vec::new();
        let checksum = {
            let mut sealed = wire::sink::Sealed::new(&mut written);
            sealed.put(input);
            sealed.checksum()
        };
        let mut counted = wire::sink::Count::default();
        counted.put(input);
        let mut slice = vec![0u8; input.len()];
        let mut buffer = wire::sink::SliceSink::new(&mut slice);
        buffer.put(input);
        let mismatches = [
            checksum != wire::crc::crc32(input),
            counted.0 != input.len(),
            written.as_slice() != input,
            buffer.overflow(),
            buffer.written() != input.len(),
        ];
        if mismatches.iter().any(|bad| *bad) {
            return Err(
                Error::new(Code::Internal, "三种接收器对同一批字节给出了不一致的结论")
                    .with("fuzz::drivers::Sealed"),
            );
        }
        Ok(())
    }
}
