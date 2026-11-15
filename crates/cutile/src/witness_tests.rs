use super::*;
use crate::builtin::Catalogue;
use crate::toolchain;
use testkit;

fn dirs() -> Dirs {
    Dirs::under(std::env::temp_dir().join("cutile-witness-tests"))
}

fn cpu() -> Profile {
    Catalogue::try_builtin()
        .expect("内置数据")
        .profile("cpu")
        .cloned()
        .expect("cpu 档案")
}

fn device(vendor: &str) -> Profile {
    Catalogue::try_builtin()
        .expect("内置数据")
        .profile(vendor)
        .cloned()
        .unwrap_or_else(|| panic!("{vendor} 档案"))
}

#[test]
fn every_declared_witness_executes_on_the_device() {
    if toolchain::skip_without_host_compiler() {
        return;
    }
    let profile = cpu();
    let declared = profile.provides.len();
    let runs = run_all(&profile, &dirs()).expect("见证必须全部通过");
    assert_eq!(runs.len(), declared, "每条已声明的见证都要有一条执行结果");
    assert!(
        runs.iter().all(|run| run.scope == Scope::Device),
        "cpu 就是本机，它的见证拿到的必须是设备结论"
    );
    assert!(runs.iter().all(|run| run.max_rel_err <= 1e-5), "{runs:?}");
}

#[test]
fn a_wrong_expectation_is_a_failure() {
    if toolchain::skip_without_host_compiler() {
        return;
    }
    let mut profile = cpu();
    profile.provides[0].witness.as_mut().expect("有见证").expect[0] += 1.0;
    let error = run_all(&profile, &dirs()).expect_err("期望值不对必须失败");
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("quant"), "{error}");
}

#[test]
fn a_witness_must_write_exactly_the_declared_number_of_values() {
    if toolchain::skip_without_host_compiler() {
        return;
    }
    let mut profile = cpu();
    profile.provides[0]
        .witness
        .as_mut()
        .expect("有见证")
        .expect
        .truncate(1);
    let error = run_all(&profile, &dirs()).expect_err("个数不符必须失败");
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("写了 2 个值"), "{error}");
}

#[test]
fn a_missing_function_is_a_failure_without_compiling() {
    let mut profile = cpu();
    profile.provides[0].witness.as_mut().expect("有见证").func = "witness_absent".to_string();
    let error = run_all(&profile, &dirs()).expect_err("函数不存在必须失败");
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("witness_absent"), "{error}");
}

/// 每台设备的每条已声明见证恰好被执行一次，且由能给出更强结论的范围执行。
#[test]
fn every_declared_witness_is_executed_once_in_the_strongest_scope() {
    if toolchain::skip_without_host_compiler() {
        return;
    }
    let catalogue = Catalogue::try_builtin().expect("内置数据");
    for profile in catalogue.profiles() {
        let vendor = profile.meta.vendor.as_str();
        let declared = profile
            .provides
            .iter()
            .filter(|provide| provide.witness.is_some())
            .count();
        let runs = self_check(profile, &dirs())
            .unwrap_or_else(|error| panic!("{vendor} 的见证自检失败：{error}"));
        assert!(
            runs.iter().all(|run| run.scope == Scope::Host),
            "{vendor}: 自检的结论不能被当成设备证据"
        );
        let executed = if device_available(profile) {
            run_all(profile, &dirs())
                .unwrap_or_else(|error| panic!("{vendor} 的设备执行失败：{error}"))
                .len()
        } else {
            0
        };
        assert_eq!(
            runs.len() + executed,
            declared,
            "{vendor}: 自检 {} + 设备执行 {executed} ≠ 声明的 {declared}",
            runs.len()
        );
        assert!(
            device_available(profile) == (runs.is_empty()),
            "{vendor}: 设备在本机时不该再跑一遍自检（同一份见证只执行一次）"
        );
    }
}

/// 低等级声明同样在本机跑一遍，`grade` 才不会两头都不成立。
#[test]
fn a_cpu_l0_witness_is_executed_on_the_device_too() {
    if toolchain::skip_without_host_compiler() {
        return;
    }
    let profile = cpu();
    let unverified = profile
        .provides
        .iter()
        .filter(|provide| !provide.grade.is_verified())
        .map(|provide| provide.prim)
        .collect::<Vec<_>>();
    assert!(!unverified.is_empty(), "前提：cpu 至少有一条 L0 声明");
    let runs = run_all(&profile, &dirs()).expect("L0 见证也必须能跑");
    let checked = runs.iter().map(|run| run.prim).collect::<Vec<_>>();
    for prim in unverified {
        assert!(
            checked.contains(&prim),
            "`{prim}` 的 L0 见证没有被执行：{checked:?}"
        );
    }
}

#[test]
fn every_metal_witness_runs_on_the_gpu() {
    let profile = device("metal");
    if !device_available(&profile) {
        testkit::skip("cutile: Metal 设备不在本机，GPU 见证用例未运行");
        return;
    }
    let runs = run_all(&profile, &dirs()).expect("Metal 见证必须全部通过");
    assert_eq!(runs.len(), profile.provides.len(), "每条声明都要有结果");
    assert!(
        runs.iter().all(|run| run.scope == Scope::Device),
        "GPU 上跑出来的结论不能被降级成自检"
    );
    assert!(runs.iter().all(|run| run.max_rel_err <= 1e-5), "{runs:?}");
}

#[test]
fn a_plain_c_signature_does_not_survive_the_metal_front_end() {
    if !device_available(&device("metal")) {
        testkit::skip("cutile: Metal 设备不在本机，MSL 前端用例未运行");
        return;
    }
    let program = "#include <metal_stdlib>\nusing namespace metal;\n\
                   int witness_bad(float *out) { out[0] = 1.0f; return 1; }\n";
    let profile = device("metal");
    let metal = device::spec(Emit::Metal).expect("metal 前端");
    let error = (metal.run)(&profile, &dirs(), program, 1, Scope::Device)
        .expect_err("MSL 必须要求显式地址空间");
    let text = error.to_string();
    assert!(text.contains("xcrun metal"), "{text}");
    assert!(text.contains("address space"), "{text}");
}

#[test]
fn the_self_check_ignores_the_device_preamble_and_flags() {
    if toolchain::skip_without_host_compiler() {
        return;
    }
    let mut profile = device("nvidia");
    profile.preamble = "#error 设备专属前置声明\n".to_string();
    profile.flags = vec!["-arch=sm_100a".to_string(), "-nocudainc".to_string()];
    let runs = self_check(&profile, &dirs()).expect("自检不得使用设备 preamble/flags");
    assert!(!runs.is_empty(), "nvidia 有已声明的见证");
}

#[test]
fn a_wrong_expectation_fails_even_the_host_self_check() {
    if toolchain::skip_without_host_compiler() {
        return;
    }
    let mut profile = device("ascend");
    profile.provides[0].witness.as_mut().expect("有见证").expect[0] += 1.0;
    let error = self_check(&profile, &dirs()).expect_err("期望值不对必须失败");
    assert!(error.to_string().contains("见证自检不成立"), "{error}");
    assert!(error.to_string().contains("ascend"), "{error}");
}

#[test]
fn every_accelerator_witness_parses_in_its_own_language() {
    for (vendor, selector) in [("nvidia", "-x cuda"), ("intel", "-x cl")] {
        let profile = device(vendor);
        let declared = profile
            .provides
            .iter()
            .filter(|provide| provide.witness.is_some())
            .count();
        match language_gate(&profile, &dirs()) {
            LanguageGate::Passed { front, witnesses } => {
                assert!(front.contains(selector), "{vendor}: {front}");
                assert_eq!(witnesses, declared, "{vendor}: 门要盖住每一条声明");
            }
            LanguageGate::Absent { reason } if reason.contains("不在本机") => {}
            other => panic!("{vendor} 的见证没有在 `{selector}` 里过一遍：{other}"),
        }
    }
    match language_gate(&device("ascend"), &dirs()) {
        LanguageGate::Absent { reason } => assert!(
            reason.contains("ascendc"),
            "缺席也要说清是哪门语言：{reason}"
        ),
        other => panic!("AscendC 没有开源前端，门不该开成 `{other}`"),
    }
}

#[test]
fn a_witness_that_runs_on_the_device_needs_no_language_gate() {
    assert_eq!(language_gate(&cpu(), &dirs()), LanguageGate::Covered);
    let metal = device("metal");
    assert_eq!(
        matches!(language_gate(&metal, &dirs()), LanguageGate::Covered),
        device_available(&metal),
        "只有能在设备上真跑时才免开语言门"
    );
}

#[test]
fn counts_are_read_from_stderr_in_order() {
    assert_eq!(
        crate::device::parse_counts("witness 1\nwitness 2\n"),
        vec![1, 2]
    );
    assert_eq!(crate::device::parse_counts("noise\n"), Vec::<i64>::new());
}
