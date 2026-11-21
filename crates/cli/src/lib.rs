//! 组合根（L7）与命令行文法层：把 `argv` 与环境变量解析成 [`Command`] 与 [`settings::Settings`]，
//! 再由 [`entry`] 分派到 `run` / `status` / `train` 编排入口；两个二进制 `llm` 与 `cutile` 都在本包
//! （入口见 `src/main.rs` 与 `src/bin/cutile.rs`）。
//!
//! **拥有**：一级命令文法 [`argv`]（`Command`、`Invocation`、`USAGE`）、内核层审计的子命令文法
//! [`kernel::CutileCommand`]、两层共用的选项解析器 [`flag`]、三个目录开关的解析策略 [`settings`]
//! （默认值、`CUTILE_*`、`--set`、来源追踪），以及编排分派 [`entry`]。
//! **不拥有**：内核业务实现（各层库自己）、输出排版（`tools`）。编排入口只装配、取值解析与人可读排版，不定义新类型；
//! `main` 把 [`entry`] 的 [`Result`] 转成退出码。
//! **依赖**：各内核层与 `diagnostic`；本包即入口，不被更高层库依赖。
//!
//! 配置为什么在这一层：默认目录是**部署策略**（面向发布后的二进制），`--set` 是**命令行文法**；
//! 两者都是"这次调用怎么来的"，而不是"内核怎么实现"。内核只收 `cutile::dirs::Dirs` 这份纯数据，
//! 于是它可以被测试、服务或别的宿主按各自的约定接管（ADR 0023）。
//!
//! 解析与执行分开的理由：本包是组装点，它的 `main` 只该做"解析 → 分派 → 退出码"三件事。
//! 把参数文法写进组装点或写进被驱动的层（`cutile` 曾经自己认 `doctor` / `verify` / `bench`），
//! 等于多出一套文法：两个入口对同一个选项的解释会慢慢分叉。
//!
//! # 例子
//!
//! ```
//! use llm_cli::{Command, Invocation};
//!
//! let invocation = Invocation::parse(&[]).expect("空参数即默认命令");
//! assert_eq!(invocation.command(), &Command::Status { weights: None });
//!
//! let verify = Invocation::parse(&["verify".to_string(), "--op".to_string(), "gemv_i4".to_string()])
//!     .expect("合法参数");
//! assert_eq!(verify.command(), &Command::Verify { op: Some("gemv_i4".to_string()) });
//!
//! assert!(Invocation::parse(&["frobnicate".to_string()]).is_err(), "未知命令不猜");
//! ```

#![deny(unsafe_code)]

mod argv;
mod config;
mod flag;
mod key;
mod platform;
mod root;
mod run;
mod status;
mod train;

pub mod kernel;
pub mod settings;

pub(crate) use self::train::archive;
pub use argv::{Command, Invocation, USAGE};

use diagnostic::{Code, Error, Result};

/// 命令分派：解析后的命令接到对应编排；输出由各编排函数落到 stdout，
/// 错误交给二进制入口汇总成退出码。
///
/// 这里是一次穷尽的匹配，不设 `other =>` 兜底臂：新增 `Command` 变体时编译器会指名道姓地
/// 要求在这里接上编排，「加了命令但没接线」因此不可能编译通过。
pub fn entry(invocation: &Invocation) -> Result<()> {
    // 设备后端是进程级注册表，装配点必须在产品路径上（ADR 0025 §1、ADR 0035）：
    // 只在测试里注册会让 `llm run --device cuda` 把“没装配”误报成“没有该后端”。
    engine::backends::install();
    match invocation.command() {
        Command::Help => {
            print!("{USAGE}");
            Ok(())
        }
        Command::Platform => platform::report(),
        Command::Status { weights } => status::status(weights.as_deref()),
        Command::Run {
            weights,
            input,
            device,
        } => run::run(weights, input.as_deref(), device.as_deref()),
        Command::Import { input, output } => archive::import(input, output),
        Command::Inspect { input } => archive::inspect(input),
        Command::Check { input } => archive::check(input),
        Command::Settle { input, output } => archive::settle(input, output.as_deref()),
        Command::Audit { input, ledger } => archive::audit(input, ledger),
        Command::Train {
            lr,
            warmup,
            steps,
            clip,
            accumulation,
            load,
        } => train::train_run(
            lr.as_deref(),
            warmup.as_deref(),
            steps.as_deref(),
            clip.as_deref(),
            accumulation.as_deref(),
            load.as_deref(),
        ),
        // 这两个命令的实现在 `cutile` 二进制里：组装点不复制审计实现。
        Command::Verify { .. } | Command::Bench { .. } => Err(Error::new(
            Code::Invalid,
            "该命令当前由 `cutile` 二进制提供（`llm` 只负责组装）",
        )
        .with("cli::entry")),
    }
}
