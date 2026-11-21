//! `llm` 二进制入口：只做"解析 → 分派 → 退出码"，装配与排版在 [`llm_cli::entry`]。

use diagnostic::Result;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let invocation = match llm_cli::Invocation::parse(&args) {
        Ok(invocation) => invocation,
        Err(error) => {
            eprintln!("✗ {error}");
            eprint!("{}", llm_cli::USAGE);
            return ExitCode::from(2);
        }
    };
    finish(llm_cli::entry(&invocation))
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
