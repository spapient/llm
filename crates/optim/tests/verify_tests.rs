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

/// 昇腾的 `emit` 走的是**设备通道**（与 `cuda` / `metal` 同构），不是“这条 emit 还没接通”那一句：
/// 登记 `Lane::Ascend` 之后，`verify --devices ascend` 的结论只由“本机有没有这台设备”决定。
///
/// 本用例在 `cutile` 里跑，引擎后端没有注册，所以期望的就是 [`NO_DEVICE`]——如果 `Lane::ready`
/// 又退回 `unsupported_emit`，那条分支会打印“内核执行路径尚未接通”，把这台设备说成连通道都
/// 没有，而实际缺的只是机器。
#[test]
fn the_ascend_emit_takes_the_device_lane() {
    let profile = cutile::builtin::Catalogue::try_builtin()
        .expect("内置数据")
        .profile("ascend")
        .cloned()
        .expect("ascend 档案");
    let error = Lane::ready(&profile).expect_err("本机没有昇腾执行前端");
    assert_eq!(error.code(), Code::Unsupported);
    assert_eq!(error.message(), NO_DEVICE);
}

/// 再往外一层：整条 [`verify_one`] 的结论也必须落在"本机没有这台设备"上。次序有讲究——通道判据
/// 在协商之前（`Lane::ready` 先于 `Run::negotiate`），否则 `L0 < L2` 会先报"无可行变体"，把
/// "缺一台机器"说成"这台设备连声明都不够"，而后者是要改数据才能解决的。
#[test]
fn verifying_ascend_without_the_device_reports_the_missing_device() {
    let catalogue = cutile::builtin::Catalogue::try_builtin().expect("内置数据");
    let profile = catalogue.profile("ascend").cloned().expect("ascend 档案");
    let op = catalogue.op("gemv_i4").expect("gemv_i4 声明");
    let dirs = Dirs::under(std::env::temp_dir().join("cutile-verify-ascend"));
    let error =
        verify_one(op, &profile, &dirs, &cutile::cap::Proof::none()).expect_err("本机没有昇腾设备");
    assert_eq!(error.code(), Code::Unsupported);
    assert_eq!(error.message(), NO_DEVICE);
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
