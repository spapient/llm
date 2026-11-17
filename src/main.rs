//! 组装点：把各层连起来，**不定义任何新类型**（`docs/ARCHITECTURE.md` 的 L7）。
//!
//! 它做三件事：`status` 构造每一层的最小实例并打印摘要（"依赖图真的能编译、各层公共 API 真的
//! 可用"的端到端证据），`run` 用权重文件里的真实数值把各层的算子真的执行一遍（协商 → 拼接 →
//! emit → 编译 → 执行 → 与标量参考对拍），`train` 解析训练配置并接入会话闭环——依赖方向只是
//! 评审约定，能不能跑只能靠这里真正组装。
//!
//! 三条命令的装配与排版分别在 [`status`]、[`run`]、[`train`] 三个 bin 内模块；本文件只做
//! "解析 → 分派 → 退出码"。任何新类型都应该长在它所属的层里，而不是长在这里。

mod run;
mod status;
mod train;

use diagnostic::Result;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let invocation = match cli::Invocation::parse(&args) {
        Ok(invocation) => invocation,
        Err(error) => {
            eprintln!("✗ {error}");
            eprint!("{}", cli::USAGE);
            return ExitCode::from(2);
        }
    };
    dispatch(invocation)
}

/// 命令分派：每个命令只接一个薄壳函数，退出码汇总统一走 [`finish`]。
fn dispatch(invocation: cli::Invocation) -> ExitCode {
    match invocation.command() {
        cli::Command::Help => {
            print!("{}", cli::USAGE);
            ExitCode::SUCCESS
        }
        cli::Command::Status { weights } => finish(status::status(weights.as_deref())),
        cli::Command::Run {
            weights,
            input,
            device,
        } => finish(run::run(weights, input.as_deref(), device.as_deref())),
        cli::Command::Verify { .. } | cli::Command::Bench { .. } => audit_note(),
        cli::Command::Train {
            lr,
            warmup,
            steps,
            clip,
            accumulation,
            load,
        } => finish(train::train_run(
            lr.as_deref(),
            warmup.as_deref(),
            steps.as_deref(),
            clip.as_deref(),
            accumulation.as_deref(),
            load.as_deref(),
        )),
    }
}

/// 命令结果 → 退出码：成功走 0，失败按诊断码分级打印。
fn finish(result: Result<()>) -> ExitCode {
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("✗ {error}");
            ExitCode::from(error.code().exit_code())
        }
    }
}

/// 组装点不复制审计实现：这两个命令今天由 `cutile` 二进制提供。
fn audit_note() -> ExitCode {
    eprintln!("✗ 该命令当前由 `cargo run --bin cutile -- ...` 提供（`llm` 只负责组装）");
    ExitCode::from(2)
}
