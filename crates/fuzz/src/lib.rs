//! 模糊测试的**契约层**：定义"被测入口"、确定性用例生成与"什么算违约"，不绑定任何具体解析器。
//!
//! **拥有**：[`Target`]、[`Seed`]、[`Behaviour`]、[`Finding`]、[`check_target`] / [`sweep`] /
//! [`sweep_all`]。**不拥有**：具体解析器（`cutile` 的模板与片段名）、语料库与进程管理
//! （属于 CI 的 fuzz job）。**依赖**：`diagnostic`（运行时只此一个）。**被依赖**：
//! `tests/parsers.rs`（本 crate 自己的集成测试，以 `+dev` 依赖接上 `cli` / `weights` 等
//! 真实解析器）与 `src/fuzz_*.rs`（libfuzzer harness 入口；判据与种子在 `src/drivers.rs` /
//! `src/seeds.rs`）。
//!
//! 为什么表单独立在一个 crate：真正需要它的不是一个解析器，而是**每一个**解析器。把
//! "什么算失败"写在一处，才能保证命令行文法与 safetensors 头部被同一套判据检查，而不是各写
//! 一份 `#[test]` 里再手抄一遍。
//!
//! # 三条契约
//!
//! 一次运行是否"符合预期"不是感觉，而是下面三条之一被违反：
//!
//! 1. **不 panic**：任何输入都不该打死宿主进程；
//! 2. **结论稳定**：同一输入跑两次必须给出同样的接受/拒绝、同样的类别与消息；
//! 3. **拒绝要有话说**：拒绝时消息为空，等于"失败了但不知道为什么"，这是诊断缺陷。
//!
//! 三条都由 [`check_target`] 核对。第 1 条靠 [`std::panic::catch_unwind`]：panic 被记成一条带
//! 目标名、用例号与输入的 [`Behaviour::Panicked`] 证据，而不是让整次遍历连同已跑过的用例
//! 一起消失。这并不与"不吞掉崩溃"矛盾——崩溃仍然被报告成 [`Finding`]，只是报告里带着"哪条
//! 输入打死了它"。若交给驱动方自己 `catch_unwind`，每个驱动都会重写一遍证据格式，还得自己
//! 保住用例号。
//!
//! panic 自身的消息仍会按宿主默认行为打到 stderr；[`Behaviour::Panicked`] 里带的是从
//! payload 取回的那一份，便于写进报告。本 crate 不装 `unsafe`。
//!
//! # 模块
//!
//! - [`seed`]：确定性种子与变异调度（SplitMix64 流 + 位/字节翻转、边界字节、长度增减）；
//! - [`finding`]：结论、证据与可复现的输入呈现；
//! - [`contract`]：三条契约的核对入口，sweep 与 harness 共用。
//!
//! # 例子
//!
//! ```
//! use diagnostic::{Code, Error};
//! use llm_fuzz::{Seed, Target, sweep};
//!
//! struct Bytes;
//!
//! impl Target for Bytes {
//!     fn name(&self) -> &str {
//!         "bytes"
//!     }
//!
//!     fn run(&self, input: &[u8]) -> Result<(), Error> {
//!         if input.is_empty() {
//!             return Err(Error::new(Code::Invalid, "输入不能为空"));
//!         }
//!         Ok(())
//!     }
//! }
//!
//! let seed = Seed::new(b"abc");
//! assert!(sweep(&Bytes, &seed, 4).is_ok(), "拒绝本身是合法结果");
//! ```

#![deny(unsafe_code)]

use diagnostic::Error;

// 让本 crate 以库名自引用：`drivers.rs` 被三个外部目标 `#[path]` 包含（`tests/parsers.rs`、
// `src/fuzz_*.rs`、`tests/gen_corpus.rs`）时，它们通过 `llm_fuzz::Target` 取判据 trait；
// 库自身的测试模块也走同一条路径，避免再要求每个包含方在 crate 根各补一个 `Target`。
// 自引用只在 `#[cfg(test)]` 包含的 `drivers` 里被消费，纯 lib 构建不引用，故放开未用告警。
#[allow(unused_extern_crates)]
extern crate self as llm_fuzz;

pub mod contract {
    //! 三条契约的核对入口：确定性遍历 [`sweep`] 与 libfuzzer harness 从这里取同一份判据，
    //! 拒绝判据不会在两处悄悄漂移。

    use crate::Target;
    use crate::finding::{Behaviour, Finding};
    use crate::seed::Seed;

    /// 契约一：任何输入都必须被解析成结论，而不是打死宿主。
    const PANIC: &str = "不 panic：任何输入都必须被解析成结论，而不是打死宿主";
    /// 契约三：拒绝时必须给出非空消息。
    const EMPTY_MESSAGE: &str = "拒绝时必须给出非空消息";
    /// 契约二：同一输入跑两次必须给出同一个结论。
    const STABILITY: &str = "同一输入必须给出同一个结论";

    /// 跑一次并把 `Result` 归一成 [`Behaviour`]，让"接受/拒绝"的比较不依赖 `Error` 的内部结构。
    pub fn judge(target: &dyn Target, input: &[u8]) -> Behaviour {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| target.run(input))) {
            Ok(Ok(())) => Behaviour::Accepted,
            Ok(Err(error)) => Behaviour::Rejected {
                code: error.code(),
                message: error.message().to_string(),
            },
            Err(payload) => Behaviour::Panicked {
                message: panic_message(&payload),
            },
        }
    }

    /// 核对单条用例的三条契约；违约返回第一条证据。
    ///
    /// 热路径不落任何分配：`name` 与 `input` 只在确实违约时才克隆进 [`Finding`]，libFuzzer
    /// 每秒百万次的「通过」迭代不为此付堆分配。
    pub fn check_target(target: &dyn Target, case: u64, input: &[u8]) -> Result<(), Finding> {
        let first = judge(target, input);
        if let Behaviour::Panicked { .. } = first {
            return Err(Finding::new(target.name(), case, input, first, PANIC));
        }
        if let Behaviour::Rejected { message, .. } = &first
            && message.trim().is_empty()
        {
            return Err(Finding::new(
                target.name(),
                case,
                input,
                first,
                EMPTY_MESSAGE,
            ));
        }
        let second = judge(target, input);
        if first != second {
            return Err(Finding::new(target.name(), case, input, second, STABILITY));
        }
        Ok(())
    }

    /// 跑 `cases` 个确定性用例；全部符合契约返回 `Ok(())`，否则返回第一条证据。
    pub fn sweep(target: &dyn Target, seed: &Seed, cases: u64) -> Result<(), Finding> {
        (0..cases).try_for_each(|case| {
            let input = seed.derive(case);
            check_target(target, case, &input)
        })
    }

    /// 跑 `cases` 个确定性用例，收齐**全部**违约证据。
    pub fn sweep_all(target: &dyn Target, seed: &Seed, cases: u64) -> Vec<Finding> {
        (0..cases)
            .filter_map(|case| {
                let input = seed.derive(case);
                check_target(target, case, &input).err()
            })
            .collect()
    }

    /// 从 panic payload 取回说明：`panic!("…")` 给 `&str`，`panic!("{x}")` 给 `String`，
    /// `panic_any` 可能给任意值。常见数值/字节载荷按 `Debug` 渲染；都取不到时留占位说明，
    /// 不留空白。
    fn panic_message(payload: &Box<dyn std::any::Any + Send>) -> String {
        if let Some(message) = payload.downcast_ref::<&str>() {
            return (*message).to_string();
        }
        if let Some(message) = payload.downcast_ref::<String>() {
            return message.clone();
        }
        if let Some(bytes) = payload.downcast_ref::<&[u8]>() {
            return format!("{bytes:?}");
        }
        if let Some(value) = [
            debug_payload::<u8>(payload),
            debug_payload::<u32>(payload),
            debug_payload::<u64>(payload),
            debug_payload::<usize>(payload),
            debug_payload::<bool>(payload),
        ]
        .into_iter()
        .flatten()
        .next()
        {
            return value;
        }
        "panic payload 没有携带字符串说明".to_string()
    }

    /// 常见数值载荷按 `Debug` 取回；类型不符返回 `None`。
    fn debug_payload<T: std::fmt::Debug + 'static>(
        payload: &Box<dyn std::any::Any + Send>,
    ) -> Option<String> {
        payload
            .downcast_ref::<T>()
            .map(|value| format!("{value:?}"))
    }
}

pub mod finding {
    //! 一次运行的结论与失败证据：判据违约定型成可复现、可贴进 issue 的值。

    use core::fmt;

    use diagnostic::Code;

    /// 一次运行的结论。
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum Behaviour {
        /// 输入被接受。
        Accepted,
        /// 输入被拒绝，附错误类别与消息。
        Rejected {
            /// 错误类别（日志与退出码的对外契约）。
            code: Code,
            /// 一句话说明；空消息属于违约。
            message: String,
        },
        /// 入口用 panic 结束——违约，附从 payload 取回的消息。
        Panicked {
            /// payload 里的说明；取不到时是占位文本。
            message: String,
        },
    }

    impl fmt::Display for Behaviour {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::Accepted => formatter.write_str("accepted"),
                Self::Rejected { code, message } => write!(formatter, "rejected {code}: {message}"),
                Self::Panicked { message } => write!(formatter, "panicked: {message}"),
            }
        }
    }

    /// 失败证据：目标名、用例序号、输入是什么、违反了哪条契约。
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct Finding {
        target: String,
        case: u64,
        input: Vec<u8>,
        behaviour: Behaviour,
        contract: &'static str,
    }

    impl Finding {
        /// 一条证据：哪条用例、什么输入、什么结论、违反了哪条契约。
        pub(crate) fn new(
            target: &str,
            case: u64,
            input: &[u8],
            behaviour: Behaviour,
            contract: &'static str,
        ) -> Self {
            Self {
                target: target.to_string(),
                case,
                input: input.to_vec(),
                behaviour,
                contract,
            }
        }

        /// 被测目标名：把问题定位到具体解析器。
        pub fn target(&self) -> &str {
            &self.target
        }

        /// 用例序号（与 [`crate::Seed::derive`] 的参数一致；libfuzzer 喂来的输入记 0）。
        pub fn case(&self) -> u64 {
            self.case
        }

        /// 触发问题的输入。
        pub fn input(&self) -> &[u8] {
            &self.input
        }

        /// 这条输入得到的结论。
        pub fn behaviour(&self) -> &Behaviour {
            &self.behaviour
        }

        /// 被违反的契约，直接写进报告。
        pub fn contract(&self) -> &'static str {
            self.contract
        }
    }

    impl fmt::Display for Finding {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            writeln!(
                formatter,
                "fuzz::{} case #{}: {}",
                self.target, self.case, self.contract
            )?;
            writeln!(formatter, "  verdict: {}", self.behaviour)?;
            write!(formatter, "  input: {}", repro(&self.input))
        }
    }

    /// 输入的可读复现形式：能当 UTF-8 读就给 Rust 字符串字面量（带转义），否则给紧凑十六进制。
    #[derive(Debug, Clone, Copy)]
    pub struct Repro<'a>(&'a [u8]);

    /// 输入的可读复现形式（见 [`Repro`]）。
    pub fn repro(input: &[u8]) -> Repro<'_> {
        Repro(input)
    }

    impl fmt::Display for Repro<'_> {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            if let Ok(text) = std::str::from_utf8(self.0) {
                write!(formatter, "{text:?}")
            } else {
                const NIBBLE: [u8; 16] = *b"0123456789abcdef";
                let mut hex = String::with_capacity(self.0.len() * 2);
                for &byte in self.0 {
                    hex.push(NIBBLE[(byte >> 4) as usize] as char);
                    hex.push(NIBBLE[(byte & 0x0F) as usize] as char);
                }
                write!(formatter, "0x{hex}")
            }
        }
    }
}

pub mod seed {
    //! 确定性用例生成：一份合法范例 + 用例序号 → 跨平台可复现的扰动字节串。

    /// 确定性种子：同一个种子永远派生同一批用例，失败才能复现。
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct Seed(Vec<u8>);

    impl Seed {
        /// 由初始字节构造；空种子是合法输入（"空输入"本身就是一道用例）。
        pub fn new(bytes: impl Into<Vec<u8>>) -> Self {
            Self(bytes.into())
        }

        /// 初始字节。
        pub fn bytes(&self) -> &[u8] {
            &self.0
        }

        /// 初始字节数。
        pub fn len(&self) -> usize {
            self.0.len()
        }

        /// 是否为空种子。
        pub fn is_empty(&self) -> bool {
            self.0.is_empty()
        }
    }

    /// 64 位确定性流（SplitMix64）：状态回卷后做两次乘加终混，低 32 位与高 32 位都有充分
    /// 雪崩。类型不对外公开：外部只通过 [`Seed::derive`] 取字节，不暴露流的形状。
    #[derive(Debug, Clone)]
    struct Rng {
        state: u64,
    }

    impl Rng {
        /// 由初始状态构造；状态为 0 也合法（常量增量会把它推离零点）。
        fn new(state: u64) -> Self {
            Self { state }
        }

        /// 取下一个 64 位值。
        fn next_u64(&mut self) -> u64 {
            self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ z >> 31
        }

        /// 取 `[0, bound)` 内的确定性下标；`bound` 为 0 时返回 0（调用方据此判空）。
        fn below(&mut self, bound: usize) -> usize {
            match bound {
                0 => 0,
                bound => ((u128::from(self.next_u64()) * bound as u128) >> 64) as usize,
            }
        }

        /// 取一个随机字节。
        fn byte(&mut self) -> u8 {
            self.next_u64() as u8
        }
    }

    /// 范例字节的 64 位指纹：FNV-1a 逐字节折叠，再过 SplitMix 终混。
    fn fingerprint(bytes: &[u8]) -> u64 {
        let mut hash: u64 = 0xCBF2_9CE4_8422_2325;
        for &byte in bytes {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0100_0000_01B3);
        }
        let mut z = hash;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ z >> 31
    }

    /// 格式边角上反复出现的危险字节：零、上界、符号位边界、ASCII 空白与 UTF-8 续节/首节。
    const BOUNDARY: [u8; 12] = [
        0x00, 0x7F, 0x80, 0xFF, 0x01, 0x7E, 0xFE, 0x20, 0x0A, 0x0D, 0xBF, 0xC0,
    ];

    /// 单步变异数量；用例内每一步都从 [`MUTATIONS`] 表里抽。新增一步只改表与这个计数。
    const STEPS: usize = 9;

    /// 变异步表：下标即 [`Rng::below`] 抽中的编号；表分派把调度分支移出 [`mutate`]，
    /// 每一步仍是独立小函数，空/单字节边界各自退化安全。
    const MUTATIONS: [fn(&mut Vec<u8>, &mut Rng); STEPS] = [
        flip_bit,
        substitute_boundary,
        replace_byte,
        nudge_byte,
        insert_byte,
        delete_byte,
        truncate,
        append_bytes,
        duplicate_region,
    ];

    impl Seed {
        /// 派生第 `idx` 个用例：范例的一份确定性扰动。
        pub fn derive(&self, idx: u64) -> Vec<u8> {
            let mut rng = Rng::new(fingerprint(&self.0) ^ idx.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let mut out = self.0.clone();
            if out.is_empty() {
                // 空范例没有可删可改的位置：从流里造 0..=2 个边界字节，仍然覆盖空串、单字节与
                // 双字节这三个最短边界。
                let len = rng.below(3);
                return (0..len)
                    .map(|_| BOUNDARY[rng.below(BOUNDARY.len())])
                    .collect();
            }
            mutate(&mut out, &mut rng);
            out
        }
    }

    /// 对 `out` 做 1..=3 步复合变异；约一成用例原样保留，保住合法路径覆盖。
    fn mutate(out: &mut Vec<u8>, rng: &mut Rng) {
        if rng.below(10) == 0 {
            return;
        }
        let steps = 1 + rng.below(3);
        for _ in 0..steps {
            MUTATIONS[rng.below(STEPS)](out, rng);
        }
    }

    /// 翻转一个随机位；空串无可选位置，跳过。
    // 分派表要求统一成 `&mut Vec<u8>`；这几步只按下标改写，不改长度。
    #[allow(clippy::ptr_arg)]
    fn flip_bit(out: &mut Vec<u8>, rng: &mut Rng) {
        if out.is_empty() {
            return;
        }
        let pos = rng.below(out.len());
        out[pos] ^= 1 << rng.below(8);
    }

    /// 把一个位置换成流里的随机字节；空串跳过。
    #[allow(clippy::ptr_arg)]
    fn replace_byte(out: &mut Vec<u8>, rng: &mut Rng) {
        if out.is_empty() {
            return;
        }
        let pos = rng.below(out.len());
        out[pos] = rng.byte();
    }

    /// 把一个位置换成表格里的边界字节；空串跳过。
    #[allow(clippy::ptr_arg)]
    fn substitute_boundary(out: &mut Vec<u8>, rng: &mut Rng) {
        if out.is_empty() {
            return;
        }
        let pos = rng.below(out.len());
        out[pos] = BOUNDARY[rng.below(BOUNDARY.len())];
    }

    /// 一个字节做环绕 +1 / -1：长度、偏移、计数值这类整数字段的差一错误走这一步；空串跳过。
    #[allow(clippy::ptr_arg)]
    fn nudge_byte(out: &mut Vec<u8>, rng: &mut Rng) {
        if out.is_empty() {
            return;
        }
        let pos = rng.below(out.len());
        // -1 用环绕加 255 表达，保持单字节运算。
        let delta = if rng.below(2) == 0 { 1u8 } else { u8::MAX };
        out[pos] = out[pos].wrapping_add(delta);
    }

    /// 在随机位置插入一个字节（边界字节与随机字节各半）。
    fn insert_byte(out: &mut Vec<u8>, rng: &mut Rng) {
        let pos = rng.below(out.len() + 1);
        let byte = pick(rng);
        out.insert(pos, byte);
    }

    /// 删除一个位置；长度 ≤ 1 时保留，不做没有意义的删除。
    fn delete_byte(out: &mut Vec<u8>, rng: &mut Rng) {
        if out.len() <= 1 {
            return;
        }
        let pos = rng.below(out.len());
        out.remove(pos);
    }

    /// 截断到随机前缀长度（含 0：整条变空输入）。
    fn truncate(out: &mut Vec<u8>, rng: &mut Rng) {
        let keep = rng.below(out.len() + 1);
        out.truncate(keep);
    }

    /// 尾部追加 1..=4 个字节；先在小数组里填好，再整块 `extend_from_slice`，不逐个 push。
    fn append_bytes(out: &mut Vec<u8>, rng: &mut Rng) {
        let count = 1 + rng.below(4);
        let mut bytes = [0u8; 4];
        for slot in &mut bytes[..count] {
            *slot = pick(rng);
        }
        out.extend_from_slice(&bytes[..count]);
    }

    /// 复制一段既有字节插到随机位置：长串重复与跨区拼接走这一步；空串跳过。
    fn duplicate_region(out: &mut Vec<u8>, rng: &mut Rng) {
        if out.is_empty() {
            return;
        }
        let start = rng.below(out.len());
        let rest = out.len() - start;
        let end = start + 1 + rng.below(rest);
        let region = out[start..end].to_vec();
        let pos = rng.below(out.len() + 1);
        out.splice(pos..pos, region);
    }

    /// 一半取边界字节，一半取流随机字节。
    fn pick(rng: &mut Rng) -> u8 {
        match rng.below(2) {
            0 => BOUNDARY[rng.below(BOUNDARY.len())],
            _ => rng.byte(),
        }
    }
}

pub use contract::{check_target, sweep, sweep_all};
pub use finding::{Behaviour, Finding, Repro, repro};
pub use seed::Seed;

/// 被测入口：把任意字节变成"接受 / 拒绝"。
pub trait Target {
    /// 目标名：写进失败证据，便于把问题定位到具体解析器。
    fn name(&self) -> &str;

    /// 跑一次输入。
    fn run(&self, input: &[u8]) -> Result<(), Error>;

    /// 单条输入跑完整三条契约核对。
    fn exercise(&self, input: &[u8]) -> Result<(), Finding>
    where
        Self: Sized,
    {
        check_target(self, 0, input)
    }
}

// 契约单测与共享定义：`tests/parsers.rs`、`src/fuzz_*.rs` 与库内单测取同一份驱动，
// 不再各留一份。`drivers` 在测试构建里把 `seeds` 一起编进来（`#[path]`），并给出
// 「名称/适配器/种子」三合一的登记表，供遍历与语料生成共用。
#[cfg(test)]
#[allow(dead_code)]
mod drivers;
#[cfg(test)]
mod fuzz_tests;
