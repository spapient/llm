//! `llm platform`：把六平台执行合同打成一张表（[ADR 0052](../../../docs/adr/0052-execution-platform-contract.md)）。
//!
//! 六行恒出，缺证据的行照实写成 `open` 并带上理由——少一行会让"未覆盖"与"根本没问"变成同一种输出。
//! 本模块只排版：判定全在 [`engine::platform::audit`]，这里不重算任何一件证据。

use diagnostic::Result;
use engine::platform::{Machine, Row, Standing, audit};

/// 打印一台机器上的六行裁决；返回 `()`，因为"这台机器没有 CUDA"不是错误。
pub fn report() -> Result<()> {
    let host = Machine;
    println!("六平台执行合同（设备 + 编译器两件证据，ADR 0052）");
    for (row, standing) in Row::ALL.iter().zip(audit(&host).iter()) {
        println!(
            "  {:<14} {:<8} {}",
            row.as_str(),
            standing.as_str(),
            detail(standing)
        );
    }
    Ok(())
}

/// 一格里的第二列：覆盖给证据，未覆盖给理由的短名——两种状态各有一条唯一来源。
fn detail(standing: &Standing) -> &str {
    match standing {
        Standing::Covered { evidence } => evidence,
        Standing::Open { why } => why.as_str(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_row_renders_a_non_empty_detail() {
        let host = Machine;
        for (row, standing) in Row::ALL.iter().zip(audit(&host).iter()) {
            assert!(
                !detail(standing).is_empty(),
                "`{}` 的那一格必须写明证据或理由",
                row.as_str()
            );
        }
    }

    #[test]
    fn a_covered_row_shows_its_evidence_and_an_open_row_shows_why() {
        let covered = Standing::Covered {
            evidence: "aarch64 · NEON/SME 有序点积".to_string(),
        };
        assert_eq!(detail(&covered), "aarch64 · NEON/SME 有序点积");
        let open = Standing::Open {
            why: engine::platform::Why::NoToolchain,
        };
        assert_eq!(detail(&open), "no-toolchain");
        assert_eq!(detail(&open), engine::platform::Why::NoToolchain.as_str());
    }
}
