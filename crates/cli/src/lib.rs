//! 组合根（L7）与命令行文法层：把 `argv` 与环境变量解析成 [`Command`] 与 [`settings::Settings`]，
//! 再由 [`entry`] 分派到 `run` / `status` / `train` 编排入口；两个二进制 `llm` 与 `cutile` 都在本包
//! （入口见 `src/main.rs` 与 `src/cutile.rs`）。
//!
//! **拥有**：一级命令文法（[`Command`]、[`Invocation`]、[`USAGE`]）、内核层审计的子命令文法
//! [`kernel::CutileCommand`]、两层共用的选项解析器 `flag`、三个目录开关的解析策略 [`settings`]
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
//! let argv = ["inspect", "--in", "model.u"].map(str::to_string);
//! let inspect = Invocation::parse(&argv).expect("合法参数");
//!
//! // 取值借用 `argv`，解析不复制：`matches!` 这一句里没有任何分配。
//! assert_eq!(inspect.command(), &Command::Inspect { input: "model.u" });
//!
//! assert!(Invocation::parse(&["frobnicate".to_string()]).is_err(), "未知命令不猜");
//! ```

#![deny(unsafe_code)]

mod archive;
mod argv;
mod config;
mod flag;
mod key;
mod platform;
mod run;
mod status;
mod train;

pub mod kernel;
pub mod settings;

pub use argv::{Command, Invocation, USAGE};

use diagnostic::{Code, Error, Result};

/// 命令分派：解析后的命令接到对应编排；输出由各编排函数落到 stdout，
/// 错误交给二进制入口汇总成退出码。
pub fn entry(invocation: &Invocation<'_>) -> Result<()> {
    let settings = settings::Settings::load()?;
    entry_with_settings(invocation, &settings)
}

/// 使用已加载的来源配置分派；二进制入口在解析 argv 前剥离 `--set` / `--config`。
pub fn entry_with_settings(
    invocation: &Invocation<'_>,
    settings: &settings::Settings,
) -> Result<()> {
    shard::backends::install();
    route(invocation.command(), settings)
}

use crate::settings::Settings;

/// 一级分派：自处理的命令直接落地，其余按类别下推。
pub(crate) fn route(command: &Command<'_>, settings: &Settings) -> Result<()> {
    match command {
        Command::Help => usage(),
        Command::Platform => crate::platform::report(),
        Command::Status { .. } | Command::Run { .. } | Command::Train { .. } => {
            run_command(command, settings)
        }
        other => objects(other),
    }
}

/// 打印用法。
fn usage() -> Result<()> {
    print!("{}", crate::USAGE);
    Ok(())
}

/// 读权重、跑闭环：三者都在算，都归这里。
fn run_command(command: &Command<'_>, settings: &Settings) -> Result<()> {
    match command {
        Command::Status { weights } => crate::status::status(weights.as_deref()),
        Command::Run {
            weights,
            input,
            device,
        } => crate::run::run(weights, input.as_deref(), device.as_deref()),
        Command::Train {
            lr,
            warmup,
            steps,
            clip,
            accumulation,
            load,
        } => crate::train::train_run(
            settings,
            lr.as_deref(),
            warmup.as_deref(),
            steps.as_deref(),
            clip.as_deref(),
            accumulation.as_deref(),
            load.as_deref(),
        ),
        other => stray(other),
    }
}

/// 容器类命令：读、写、沉降一个 `.u` 文件。
fn objects(command: &Command<'_>) -> Result<()> {
    match command {
        Command::Import { input, output } => crate::archive::import(input, output),
        Command::Settle { input, output } => crate::archive::settle(input, output.as_deref()),
        Command::Audit { input, ledger } => crate::archive::audit(input, ledger),
        other => examine(other),
    }
}

/// `inspect` / `check` / `deploy` 只差一步验码：同一个入参，同一层。
fn examine(command: &Command<'_>) -> Result<()> {
    match command {
        Command::Inspect { input } => crate::archive::inspect(input),
        Command::Check { input } => crate::archive::check_archive(input),
        Command::Deploy { input } => crate::archive::deploy(input),
        other => stray(other),
    }
}

/// 阶梯末端：上面的编译期证明说这里走不到。留一条诊断而不是 `unreachable!`——真漏了也该
/// 是内部错误码，不该让用户看到 panic。
fn stray(command: &Command<'_>) -> Result<()> {
    Err(Error::new(Code::Internal, format!("命令 `{command:?}` 没有落点")).with("cli::entry"))
}
