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

/// 六种写回形式是装配的必经之路：每份档案都要六条齐全、各有一条见证。
///
/// 缺一条在构建期就会炸（`codegen` 按形式找段），但那时报的是"片段里没有这一段"；
/// 在档案层拦下来，诊断才指向写错的档案。
#[test]
fn every_archive_declares_the_six_epilogue_forms() {
    for name in crate::names() {
        let table = crate::embedded(name).expect("名字都能取到表");
        let profile = crate::prof::Profile::parse(table).expect("内置档案可解析");
        let forms = profile
            .epilogue
            .iter()
            .map(|epi| epi.form)
            .collect::<Vec<_>>();
        assert_eq!(
            forms,
            crate::atom::Epilogue::ALL,
            "`{name}` 的 `[[epilogue]]` 必须六种形式各一条"
        );
        assert!(
            profile.epilogue.iter().all(|epi| epi.witness.is_some()),
            "`{name}` 的每条写回供给都要有见证"
        );
    }
}
