use diagnostic::data::Value;

/// 生成表按文件名升序（旧的 `prof::load(dir)` 也这样排序）。
#[test]
fn embedded_names_are_sorted() {
    let generated = crate::names();
    assert!(
        generated.windows(2).all(|pair| pair[0] < pair[1]),
        "{generated:?}"
    );
    for name in generated {
        assert!(crate::embedded(name).is_some(), "{name} 有名字却没有表");
    }
}

/// 每份生成表都是一份设备档案：有 `[meta]`，也有能力区间。
#[test]
fn every_embedded_entry_is_a_device_archive() {
    for name in crate::names() {
        let table = crate::embedded(name).expect("名字都能取到表");
        assert!(table.table("meta").is_some(), "{name} 缺 `[meta]`");
        let provides = table.array("provides").expect("档案必须有 `[[provides]]`");
        assert!(!provides.is_empty(), "{name} 没有任何能力区间");
        assert!(
            provides
                .iter()
                .all(|entry| matches!(entry, Value::Table(_))),
            "{name} 的 `provides` 元素必须是表"
        );
    }
}

#[test]
fn embedded_reports_a_missing_archive() {
    assert!(crate::embedded("no-such-device").is_none());
}

#[test]
fn select_picks_the_profile_matching_local_arch() {
    let profiles = crate::prof::builtin_profiles();
    let detected = crate::prof::Detected::new().with("arch", std::env::consts::ARCH);
    let chosen = crate::prof::select(profiles, &detected).expect("本机架构应命中一份档案");
    assert_eq!(chosen.meta.vendor, "cpu");
}

#[test]
fn select_reports_unsupported_when_nothing_matches() {
    let profiles = crate::prof::builtin_profiles();
    let detected = crate::prof::Detected::new().with("arch", "no-such-arch");
    let error = crate::prof::select(profiles, &detected).expect_err("无命中必须报错");
    assert_eq!(error.code(), diagnostic::Code::Unsupported);
}
