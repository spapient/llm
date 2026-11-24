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

/// 每个后端都实现完整算子矩阵，且只替换正文、不改变共享 ABI。
#[test]
fn every_device_kernel_matches_the_shared_operator_contract() {
    for vendor in crate::names() {
        let kernels = crate::catalogue::all_for(vendor);
        assert_eq!(
            kernels.len(),
            crate::OPS.len(),
            "`{vendor}` 的内核矩阵不完整"
        );
        for view in crate::OPS {
            let reference = crate::catalogue::named(view.name)
                .expect("CPU 参考算子")
                .to_decl()
                .expect("CPU 参考声明自洽");
            let specialized = crate::catalogue::named_for(vendor, view.name)
                .unwrap_or_else(|| panic!("`{vendor}` 缺 `{}`", view.name))
                .to_decl()
                .unwrap_or_else(|error| panic!("`{vendor}`/`{}`：{error}", view.name));
            assert_eq!(
                specialized.signature, reference.signature,
                "`{vendor}` ABI 漂移"
            );
            assert_eq!(
                specialized.needs, reference.needs,
                "`{vendor}` 能力需求漂移"
            );
            assert_eq!(
                specialized.layouts, reference.layouts,
                "`{vendor}` 布局漂移"
            );
        }
    }
}

/// 后端表按 `(vendor, op)` 严格升序，二分查找不会遇到重复键或乱序。
#[test]
fn device_templates_are_sorted_and_complete() {
    assert_eq!(
        crate::DEVICE_TEMPLATES.len(),
        crate::OPS.len() * crate::names().len()
    );
    assert!(
        crate::DEVICE_TEMPLATES
            .windows(2)
            .all(|pair| { (pair[0].0, pair[0].1) < (pair[1].0, pair[1].1) })
    );
    assert!(crate::catalogue::template_for("no-such-device", "add_f32").is_none());
    assert!(crate::catalogue::template_for("cpu", "no-such-op").is_none());
}

/// 发射选择真正读取后端正文；CPU 相同实现保持零拷贝，其他实现拥有替换后的声明。
#[test]
fn device_specialization_selects_the_backend_without_changing_abi() {
    let op = crate::catalogue::named("add_f32")
        .expect("逐元素参考算子")
        .to_decl()
        .expect("参考声明");
    let cpu = crate::catalogue::for_device(&op, "cpu").expect("CPU 内核存在");
    assert!(matches!(cpu, std::borrow::Cow::Borrowed(_)));

    let nvidia = crate::catalogue::for_device(&op, "nvidia").expect("NVIDIA 内核存在");
    assert!(matches!(nvidia, std::borrow::Cow::Owned(_)));
    assert_eq!(nvidia.signature, op.signature);
    assert_eq!(
        nvidia.template,
        crate::catalogue::template_for("nvidia", &op.name).expect("NVIDIA 正文")
    );

    let error = crate::catalogue::for_device(&op, "no-such-device").expect_err("未知后端必须失败");
    assert_eq!(error.code(), diagnostic::Code::Missing);
}
