//! 驱动方：把每个"接受不受信任输入"的解析器灌进 [`sweep`]。
//!
//! 种子是**一份合法输入**，派生用例是它的逐字节扰动：于是遍历走的是"边角上的一条错误路径"，
//! 而不是与真实输入毫无关系的噪声——真实缺陷几乎总在这些边角上（多一个字节的长度、写错的
//! 分组数、半个转义序列）。同一 `(seed, case)` 在任何机器上都得到同一串字节，失败可复现。
//!
//! 覆盖的是文本与字节两类边界：`config` 的 TOML 子集、`prof` 的档案、`cli` 的参数文法、
//! `weights` 的 JSON 与 safetensors 头部。它们都是"外部来的字节"，也都把错误当返回值，
//! 因此"不 panic / 结论稳定 / 拒绝有话说"三条契约在这里是同一件事。
//!
//! `op` 的声明与模板同样在闭集里，但它们的入口要一份模板正文作上下文（`OpDecl::parse`
//! 同时吃声明与模板），已经由 `op` 自己的单测逐条覆盖，这里不重复构造第二份模板。

use diagnostic::{Code, Error};
use llm_fuzz::{Seed, Target, sweep};

/// 每个目标跑多少条派生用例。
///
/// 512 条 × 5 个目标 = 2560 次解析（每条还跑两遍验稳定），边际成本是毫秒级；数量取到"能覆盖
/// 每个字节位置被扰动一次"就够——加密强度的随机源在这里没有价值，可复现才有。
const CASES: u64 = 512;

/// 文本解析器的公共外壳：非 UTF-8 不是"解析失败"，而是"这份输入不适用于文本入口"。
///
/// 仍然是一个合法的拒绝（带类别的非空消息），因此不影响三条契约；把它区分出来只是为了让
/// 之后加字节入口时不必再想一遍。
fn text(input: &[u8]) -> Result<&str, Error> {
    std::str::from_utf8(input)
        .map_err(|_| Error::new(Code::Invalid, "输入不是 UTF-8").with("fuzz::parsers"))
}

/// 配置文件的 TOML 子集。
struct Toml;

impl Target for Toml {
    fn name(&self) -> &str {
        "config::toml"
    }

    fn run(&self, input: &[u8]) -> Result<(), Error> {
        // `ParseError` 只带行号与说明，类别在调用方由"这是文本语法"决定。
        config::toml::parse(text(input)?)
            .map(|_| ())
            .map_err(|error| Error::new(Code::Invalid, error.to_string()).with("config::toml"))
    }
}

/// 设备档案：同一份 TOML 子集，但在 `prof` 里还要满足闭集与数字的一致性。
struct Profile;

impl Target for Profile {
    fn name(&self) -> &str {
        "prof::Profile"
    }

    fn run(&self, input: &[u8]) -> Result<(), Error> {
        prof::prof::Profile::parse(text(input)?)
            .map(|_| ())
            .map_err(|error| {
                Error::new(error.code(), error.to_string()).with("prof::Profile::parse")
            })
    }
}

/// 命令行：空白切词，再交给唯一的文法。
struct Invocation;

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
struct Json;

impl Target for Json {
    fn name(&self) -> &str {
        "weights::json"
    }

    fn run(&self, input: &[u8]) -> Result<(), Error> {
        weights::json::parse(text(input)?).map(|_| ())
    }
}

/// safetensors 整体：8 字节头长 + JSON 头 + 数据段。
struct Safetensors;

impl Target for Safetensors {
    fn name(&self) -> &str {
        "weights::safetensors::Archive"
    }

    fn run(&self, input: &[u8]) -> Result<(), Error> {
        weights::safetensors::Archive::parse(input).map(|_| ())
    }
}

/// 一份**合法**的 TOML 子集：表、键值、整数、字符串、布尔与表数组各一个。
const TOML: &[u8] = b"[a]\nb = 1\nc = \"x\"\nd = true\n[[e]]\nf = [1, 2]\n";

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

/// 种子的取值在这里定：档案直接取磁盘上真实装载的那一份（`prof` 的格式变一次，这里跟着变）。
fn profile() -> Vec<u8> {
    include_bytes!("../../prof/devices/cpu.toml").to_vec()
}

#[test]
fn every_parser_turns_arbitrary_bytes_into_a_stable_verdict() {
    let targets: [(&dyn Target, Seed); 5] = [
        (&Toml, Seed::new(TOML.to_vec())),
        (&Profile, Seed::new(profile())),
        (&Invocation, Seed::new(ARGS.to_vec())),
        (&Json, Seed::new(HEADER.as_bytes().to_vec())),
        (&Safetensors, Seed::new(safetensors())),
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
    let targets: [(&dyn Target, Seed); 5] = [
        (&Toml, Seed::new(TOML.to_vec())),
        (&Profile, Seed::new(profile())),
        (&Invocation, Seed::new(ARGS.to_vec())),
        (&Json, Seed::new(HEADER.as_bytes().to_vec())),
        (&Safetensors, Seed::new(safetensors())),
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
