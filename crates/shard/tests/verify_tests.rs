//! `cutile::verify` 的用例：校验判据与失败口径。
//!
//! 生产文件只留执行与判据，用例集中于此。

use super::*;
use crate::gate::LanguageGate;

fn failed(label: &str) -> VerifyEntry {
    VerifyEntry::Failed {
        label: label.to_string(),
        error: Error::new(Code::Invalid, "测试失败"),
    }
}

#[test]
fn witness_note_is_absent_when_nothing_ran() {
    assert!(witness_note(0, 0, &LanguageGate::Covered).is_none());
    // Rejected 不追加句子：拒绝已写进 Evidence::failure，这里再印一次只是重复。
    let rejected = LanguageGate::Rejected {
        reason: "第一行报错".to_string(),
    };
    assert!(witness_note(0, 0, &rejected).is_none());
}

/// 非 C 前端的生产态 verify 走 `prof::runtime` FFI 后端岛，shard 只保留 C 参考通道。
#[test]
fn the_ascend_emit_reports_the_prof_runtime_lane() {
    let profile = cutile::builtin::Catalogue::try_builtin()
        .expect("内置数据")
        .profile("ascend")
        .cloned()
        .expect("ascend 档案");
    let error = Lane::ready(&profile).expect_err("shard 不再生成设备 verify 程序");
    assert_eq!(error.code(), Code::Unsupported);
    assert!(error.message().contains("prof::runtime"), "{error}");
}

/// 再往外一层：整条 [`verify_one`] 的结论也必须落在 prof::runtime 后端岛上。
#[test]
fn verifying_ascend_reports_the_prof_runtime_lane() {
    let catalogue = cutile::builtin::Catalogue::try_builtin().expect("内置数据");
    let profile = catalogue.profile("ascend").cloned().expect("ascend 档案");
    let op = catalogue.op("gemv_i4").expect("gemv_i4 声明");
    let dirs = Dirs::under(std::env::temp_dir().join("cutile-verify-ascend"));
    let error = verify_one(op, &profile, &dirs, &cutile::cap::Proof::none())
        .expect_err("shard 不再生成设备 verify 程序");
    assert_eq!(error.code(), Code::Unsupported);
    assert!(error.message().contains("prof::runtime"), "{error}");
}

#[test]
fn witness_note_counts_device_and_host_separately() {
    let note = witness_note(2, 3, &LanguageGate::Covered).expect("有结论");
    assert!(note.contains("见证设备执行 2 条通过"), "{note}");
    assert!(note.contains("见证自检 3 条通过"), "{note}");
    // Covered 表示设备执行已覆盖，不再重复语言门。
    assert!(!note.contains("语言门"), "{note}");
}

#[test]
fn witness_note_renders_the_language_gate_branches() {
    let passed = LanguageGate::Passed {
        front: "clang@21".to_string(),
        witnesses: 4,
    };
    assert!(
        witness_note(0, 0, &passed)
            .expect("有结论")
            .contains("语言门 clang@21 通过（4 条）")
    );
    let absent = LanguageGate::Absent {
        reason: "缺 clang".to_string(),
    };
    assert!(
        witness_note(0, 0, &absent)
            .expect("有结论")
            .contains("语言门不可用：缺 clang")
    );
}

#[test]
fn a_negotiation_classifies_head_and_path() {
    let head = Negotiation::Head {
        variant: "v0".to_string(),
        of: 3,
    };
    assert!(head.head_hit() && head.has_path());
    let fallback = Negotiation::Fallback {
        variant: "v1".to_string(),
        step: 2,
        of: 3,
    };
    assert!(!fallback.head_hit() && fallback.has_path());
    let no_path = Negotiation::NoPath {
        of: 3,
        unmet: vec!["wmma".to_string()],
    };
    assert!(!no_path.head_hit() && !no_path.has_path());
    let unknown = Negotiation::Unknown {
        reason: "无档案".to_string(),
    };
    assert!(!unknown.head_hit() && !unknown.has_path());
}

#[test]
fn the_report_collects_failures_and_head_hits() {
    let report = VerifyReport {
        op: "gemv".to_string(),
        devices: vec!["a".to_string(), "b".to_string()],
        entries: vec![failed("a"), failed("b")],
        negotiations: vec![
            Negotiation::Head {
                variant: "v0".to_string(),
                of: 1,
            },
            Negotiation::NoPath {
                of: 1,
                unmet: Vec::new(),
            },
        ],
    };
    assert_eq!(report.failures().len(), 2, "两个 Failed 条目各给出一条错误");
    assert_eq!(report.head_hits(), (1, 2), "两台设备里一台命中首档");
}
