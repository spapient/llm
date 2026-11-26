//! `platform` 的单元测试：六行恒出、每格有话说、整表一次写屏。
//!
//! 经 `src/platform.rs` 的 `#[path]` 挂进 lib 单测（N12：`src/` 里不放测试）。

use super::*;

testkit::test_cases! {
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

fn a_covered_row_shows_its_evidence_and_an_open_row_shows_why() {
    let covered = Standing::Covered {
        evidence: "aarch64 · NEON/SME 有序点积".to_string(),
    };
    assert_eq!(detail(&covered), "aarch64 · NEON/SME 有序点积");
    let open = Standing::Open {
        why: shard::platform::Why::NoToolchain,
    };
    assert_eq!(detail(&open), "no-toolchain");
    assert_eq!(detail(&open), shard::platform::Why::NoToolchain.as_str());
}

fn the_table_is_rendered_in_one_piece() {
    let text = render();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 1 + Row::ALL.len(), "表头 + 每平台一行");
    assert!(lines[0].starts_with("六平台执行合同"), "{}", lines[0]);
    assert!(text.ends_with('\n'), "一次写屏的文本自带收尾换行");
}
}
