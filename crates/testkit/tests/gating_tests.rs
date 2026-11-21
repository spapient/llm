use super::*;

#[test]
fn gating_accepts_all_lists_and_ignores_blank_entries() {
    assert!(matches("all", "nvidia"));
    assert!(matches("cpu, nvidia", "nvidia"));
    assert!(matches(" cpu ", "cpu"));
    assert!(!matches("", "cpu"));
    assert!(!matches("cpu", "nvidia"));
}

#[test]
fn skip_registers_reason_and_count() {
    let before = skip_count();
    assert!(skip("测试用临时跳过原因"));
    assert!(skip_count() > before, "跳过计数必须单调递增");
    assert!(
        skipped_reasons()
            .iter()
            .any(|reason| reason == "测试用临时跳过原因")
    );
}

#[test]
fn device_or_skip_registers_when_disabled() {
    let before = skip_count();
    assert!(device_or_skip("nvidia", "测试：未启用 nvidia"));
    assert!(skip_count() > before, "跳过计数必须单调递增");
}
