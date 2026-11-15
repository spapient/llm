//! 驱动方：把每个"接受不受信任输入"的解析器灌进 [`sweep`]。
//!
//! 种子是**一份合法输入**，派生用例是它的逐字节扰动：于是遍历走的是"边角上的一条错误路径"，
//! 而不是与真实输入毫无关系的噪声——真实缺陷几乎总在这些边角上（多一个字节的长度、写错的
//! 分组数、半个转义序列）。同一 `(seed, case)` 在任何机器上都得到同一串字节，失败可复现。
//!
//! 覆盖的是文本与字节两类边界：`cli` 的参数文法、`weights` 的 JSON 与 safetensors 头部、
//! `train` 的检查点、`serve` 的帧解码，以及 `wire` 三种接收器的不变量。它们都是"外部来的
//! 字节"，也都把错误当返回值，因此"不 panic / 结论稳定 / 拒绝有话说"三条契约在这里是
//! 同一件事；最后一个不拒绝任何输入，它验的是"三条承诺互相一致"。
//!
//! 适配定义统一来自 [`drivers`]，与 libfuzzer harness 共用，避免两份判据漂移。
//!
//! **不再覆盖 TOML**：`ops/`、`devices/`、`lints/vocabulary.toml` 在构建期就被解析成表
//! （ADR 0022），运行期没有 TOML 入口可灌；`op` 的声明入口要一份模板正文作上下文，由
//! `op` 自己的单测逐条覆盖。

#[path = "../drivers.rs"]
mod drivers;

use drivers::{Frames, Invocation, Json, Safetensors, Sealed, Snapshot};
use llm_fuzz::{Seed, Target, sweep};

/// 每个目标跑多少条派生用例。
///
/// 512 条 × 6 个目标 = 3072 次解析（每条还跑两遍验稳定），边际成本是毫秒级；数量取到"能覆盖
/// 每个字节位置被扰动一次"就够——加密强度的随机源在这里没有价值，可复现才有。
const CASES: u64 = 512;

/// 一份合法的命令行：默认命令 + 一个带取值的选项。
const ARGS: &[u8] = b"status --weights model.safetensors";

/// 一份合法的 safetensors 头 JSON：一个 2×3 的 f32 条目，正好 24 字节数据。
const HEADER: &str = r#"{"w": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]}}"#;

/// 由 [`HEADER`] 拼出完整文件字节（数据段全零，`parse` 不看内容只认长度）。
fn safetensors() -> Vec<u8> {
    let mut file = (HEADER.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(HEADER.as_bytes());
    file.resize(file.len() + 24, 0);
    file
}

/// 一份合法的检查点：两条条目、事实四个数都取最小合法值。
///
/// 种子必须是**能解码**的，否则遍历只在错误路径上打转；因此这里直接经库编码一份，
/// 而不是照字节格式手抄（手抄的版本一旦格式演进就会悄悄变成"合法种子其实不合法"）。
fn snapshot() -> Vec<u8> {
    let facts = train::snapshot::Facts::new(1, 1, 1, 1).expect("事实合法");
    let entry = weights::Entry::new("w", tensor::DType::F32, tensor::Shape::new([2, 3])).at(8);
    let header = weights::Header::new(vec![entry], 8).expect("条目表合法");
    train::snapshot::Snapshot::new(facts, header)
        .encode()
        .expect("编码合法检查点")
}

/// 一条合法的帧：`Request` + 载荷 `hi`，含头尾校验。
fn frame() -> Vec<u8> {
    serve::proto::Envelope::new(
        serve::proto::Version::current(),
        serve::proto::Kind::Request,
        b"hi".to_vec(),
    )
    .frame()
    .expect("编码合法帧")
    .to_vec()
}

#[test]
fn every_parser_turns_arbitrary_bytes_into_a_stable_verdict() {
    let targets: [(&dyn Target, Seed); 6] = [
        (&Invocation, Seed::new(ARGS.to_vec())),
        (&Json, Seed::new(HEADER.as_bytes().to_vec())),
        (&Safetensors, Seed::new(safetensors())),
        (&Snapshot, Seed::new(snapshot())),
        (&Frames, Seed::new(frame())),
        (&Sealed, Seed::new(frame())),
    ];
    for (target, seed) in targets {
        if let Err(finding) = sweep(target, &seed, CASES) {
            panic!(
                "{} 在第 {} 个用例上违约（{}）；输入 {}",
                target.name(),
                finding.case(),
                finding.contract(),
                Escape(finding.input())
            );
        }
    }
}

/// 种子本身必须是**被接受**的：否则整个遍历只是在错误路径上打转，
/// "合法输入也只许有一种结论"这条就没人验。
#[test]
fn every_seed_is_a_valid_input() {
    let targets: [(&dyn Target, Seed); 6] = [
        (&Invocation, Seed::new(ARGS.to_vec())),
        (&Json, Seed::new(HEADER.as_bytes().to_vec())),
        (&Safetensors, Seed::new(safetensors())),
        (&Snapshot, Seed::new(snapshot())),
        (&Frames, Seed::new(frame())),
        (&Sealed, Seed::new(frame())),
    ];
    for (target, seed) in targets {
        assert!(
            target.run(seed.bytes()).is_ok(),
            "{} 的种子应当被接受：{}",
            target.name(),
            Escape(seed.bytes())
        );
    }
}

/// 失败输入的可读形式：能当 UTF-8 读就原样给，否则给转义后的字节——贴进 issue 就能复现。
struct Escape<'a>(&'a [u8]);

impl std::fmt::Display for Escape<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match std::str::from_utf8(self.0) {
            Ok(text) => write!(f, "{text:?}"),
            Err(_) => {
                f.write_str("0x")?;
                for byte in self.0 {
                    write!(f, "{byte:02x}")?;
                }
                Ok(())
            }
        }
    }
}
