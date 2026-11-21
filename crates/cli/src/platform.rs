//! `llm platform`：把六平台执行合同打成一张表（[ADR 0052](../../../docs/adr/0052-execution-platform-contract.md)）。
//!
//! 六行恒出，缺证据的行照实写成 `open` 并带上理由——少一行会让"未覆盖"与"根本没问"变成同一种输出。
//! 本模块只排版：判定全在 [`device::platform::audit`]，这里不重算任何一件证据。

use diagnostic::Result;
use device::platform::{Machine, Row, Standing, audit};
use std::fmt::Write as _;

/// 打印一台机器上的六行裁决；返回 `()`，因为"这台机器没有 CUDA"不是错误。
pub fn report() -> Result<()> {
    print!("{}", render());
    Ok(())
}

/// 整张表一次写屏：逐行 `println!` 会让同一份报告分七次抢锁，行之间也可能被别的输出插进来。
fn render() -> String {
    let host = Machine;
    let mut text = String::from("六平台执行合同（设备 + 编译器两件证据，ADR 0052）\n");
    for (row, standing) in Row::ALL.iter().zip(audit(&host).iter()) {
        // 写 `String` 不会失败（`fmt::Write` 只在载体报错时失败），忽略返回值的错误位。
        let _ = writeln!(
            text,
            "  {:<14} {:<8} {}",
            row.as_str(),
            standing.as_str(),
            detail(standing)
        );
    }
    text
}

/// 一格里的第二列：覆盖给证据，未覆盖给理由的短名——两种状态各有一条唯一来源。
fn detail(standing: &Standing) -> &str {
    match standing {
        Standing::Covered { evidence } => evidence,
        Standing::Open { why } => why.as_str(),
    }
}

/// 契约表的单元测试。
#[cfg(test)]
#[path = "../tests/platform.rs"]
mod tests;
