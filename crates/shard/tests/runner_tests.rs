use super::*;

#[test]
fn every_timed_run_is_its_own_sample() {
    let samples = parse_samples("noise\nns 120\nns 100\nns 140\n");
    assert_eq!(samples, vec![120.0, 100.0, 140.0]);
    assert!(parse_samples("").is_empty());
}

#[test]
fn min_and_median_are_computed_from_all_samples() {
    let timing = Timing {
        samples_ns: vec![30.0, 10.0, 20.0, 40.0],
    };
    assert_eq!(timing.min_ns(), Some(10.0));
    assert_eq!(timing.p50_ns(), Some(25.0));
    assert_eq!(timing.p95_ns(), Some(40.0));
    assert_eq!(Timing::default().min_ns(), None);
    assert_eq!(Timing::default().p50_ns(), None);
    assert_eq!(Timing::default().p95_ns(), None);
}

#[test]
fn bit_patterns_decode_back_to_floats() {
    let bits = f32::to_bits(1.5);
    assert_eq!(parse_bits(&format!("{bits}\n")), vec![1.5]);
}

#[test]
fn bit_patterns_round_trip_for_random_bits() {
    let mut rng = testkit::Rng::new(0xB17);
    for _ in 0..64 {
        let bits = rng.below(u32::MAX);
        let decoded = parse_bits(&format!("{bits}\n"));
        // 按位比较：NaN 的位模式也要能往返，而 `f32 == f32` 对 NaN 不成立。
        assert_eq!(decoded[0].to_bits(), bits);
    }
}

#[test]
fn quantiles_are_ordered_for_random_samples() {
    let mut rng = testkit::Rng::new(0xA17E);
    for _ in 0..32 {
        let count = rng.below(64) + 1;
        let samples: Vec<f64> = (0..count)
            .map(|_| f64::from(rng.below(1_000_000)))
            .collect();
        let timing = Timing {
            samples_ns: samples,
        };
        let min = timing.min_ns().expect("非空");
        let p50 = timing.p50_ns().expect("非空");
        let p95 = timing.p95_ns().expect("非空");
        assert!(min <= p50 && p50 <= p95, "分位必须有序：{min} {p50} {p95}");
    }
}

#[test]
fn compiler_gets_the_flags_it_was_given() {
    // 真机用例：默认跳过，`LLM_TEST_DEVICE=cpu` 才真的编译一次。
    if testkit::device_or_skip(
        "cpu",
        "cutile: 未显式启用 CPU 真机编译用例（LLM_TEST_DEVICE=cpu）",
    ) {
        return;
    }
    let workdir = std::env::temp_dir().join(format!("shard-runner-{}", std::process::id()));
    let source = "#include <stdio.h>\n\
int main(void) { printf(\"%u\\n\", (unsigned)FLAG_MARK); return 0; }\n";
    let exe =
        compile("cc", &["-DFLAG_MARK=7".to_string()], source, &workdir).expect("带档案参数编译");
    let outcome = run(&exe).expect("运行");
    assert_eq!(outcome.bits, vec![f32::from_bits(7)]);
    let _ = std::fs::remove_dir_all(&workdir);
}

#[test]
fn one_summary_agrees_with_every_individual_statistic() {
    let timing = Timing {
        samples_ns: vec![40.0, 10.0, 30.0, 20.0],
    };
    let summary = timing.summary().expect("非空样本");
    assert_eq!(summary.min_ns, timing.min_ns().expect("min"));
    assert_eq!(summary.p50_ns, timing.p50_ns().expect("p50"));
    assert_eq!(summary.p95_ns, timing.p95_ns().expect("p95"));
    assert_eq!(
        summary,
        TimingSummary {
            min_ns: 10.0,
            p50_ns: 25.0,
            p95_ns: 40.0
        }
    );
    assert_eq!(Timing::default().summary(), None);
}

// ---- toolchain_tests.rs ----

#[test]
fn a_portable_profile_may_declare_any_version() {
    let tool = Toolchain {
        command: "cc".to_string(),
        version: Some("11.4.0".to_string()),
        banner: "cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0".to_string(),
    };
    assert!(tool.agrees_with("*"), "便携档案的 `*` 不能被报成不一致");
    assert!(tool.agrees_with("cc-11"), "同主版本相容");
    assert!(!tool.agrees_with("clang-17"), "跨主版本仍要说出来");
}

#[test]
fn versions_are_read_from_what_the_compilers_actually_print() {
    // 三家真实输出：找的是同一段东西，而不是各自的字段名。
    assert_eq!(
        parse_version("Apple clang version 21.0.0 (clang-2100.4.1.1)\nTarget: arm64-apple-darwin"),
        Some("21.0.0".to_string())
    );
    assert_eq!(
        parse_version("gcc (Ubuntu 13.2.0-23ubuntu4) 13.2.0\nCopyright (C) 2023"),
        Some("13.2.0".to_string())
    );
    assert_eq!(
        parse_version(
            "nvcc: NVIDIA (R) Cuda compiler driver\nCuda compilation tools, release 13.3, V13.3.44"
        ),
        Some("13.3".to_string())
    );
    // 没有版本串时不猜：`-` 会出现在身份里，报告同时打印原文。
    assert_eq!(parse_version("cc (some vendor toolchain)\n"), None);
    // 孤立数字不是版本号（`-O2`、`sm_100` 常见于参数与设备名）。
    assert_eq!(parse_version("flags: -O2 sm_100"), None);
}

#[test]
fn the_identity_carries_the_measured_version() {
    let tool = Toolchain {
        command: "cc".to_string(),
        version: Some("21.0.0".to_string()),
        banner: "Apple clang version 21.0.0".to_string(),
    };
    assert_eq!(tool.id(), "cc@21.0.0");
    assert!(tool.agrees_with("clang-21"), "登记值带族名也算一致");
    assert!(tool.agrees_with("21.4"), "只比主版本号");
    assert!(!tool.agrees_with("clang-17"), "换主版本就是另一条工具链");
    assert!(!tool.agrees_with(""), "空登记不能算一致");

    let unknown = Toolchain {
        command: "cc".to_string(),
        version: None,
        banner: "cc (some vendor toolchain)".to_string(),
    };
    assert_eq!(unknown.id(), "cc@-");
    assert!(!unknown.agrees_with("clang-21"), "不知道版本就不能声称一致");
}

#[cfg(unix)]
#[test]
fn a_probe_reads_the_version_of_the_command_it_was_given() {
    use std::os::unix::fs::PermissionsExt;
    let root = std::env::temp_dir().join(format!("cutile-probe-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).expect("临时目录");
    let fake = root.join("fake-cc.sh");
    std::fs::write(&fake, "#!/bin/sh\necho 'Fake clang version 99.1.2'\n").expect("写脚本");
    let mut perms = std::fs::metadata(&fake).expect("脚本").permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(&fake, perms).expect("可执行");
    let tool = probe(&fake.to_string_lossy()).expect("脚本可探测");
    assert_eq!(tool.version.as_deref(), Some("99.1.2"));
    assert_eq!(tool.id(), format!("{}@99.1.2", fake.to_string_lossy()));
    let _ = std::fs::remove_dir_all(&root);
}

/// 档案里可以写带前缀参数的命令（`xcrun metal`）：探测要把参数一起交给程序，身份保留声明原文。
#[cfg(unix)]
#[test]
fn a_declared_command_may_carry_prefix_arguments() {
    use std::os::unix::fs::PermissionsExt;
    let root = std::env::temp_dir().join(format!("cutile-declared-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).expect("临时目录");
    let fake = root.join("fake-xcrun.sh");
    // 只有前缀参数真的被传进来才自报版本：探测丢掉参数时这里就只剩 `None`。
    std::fs::write(
        &fake,
        "#!/bin/sh\nif [ \"$1\" = metal ]; then echo 'Apple metal version 32023.922'; fi\n",
    )
    .expect("写脚本");
    let mut perms = std::fs::metadata(&fake).expect("脚本").permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(&fake, perms).expect("可执行");
    let declared = format!("{} metal", fake.display());
    let tool = probe_declared(&declared).expect("带前缀参数的命令可探测");
    assert_eq!(tool.version.as_deref(), Some("32023.922"));
    assert_eq!(
        tool.id(),
        format!("{declared}@32023.922"),
        "身份保留声明原文"
    );
    assert!(
        probe_declared("cutile-no-such-compiler-0x").is_none(),
        "装不上的声明如实说没有，不退回合主编译器"
    );
    assert!(probe_declared("   ").is_none(), "空声明不是命令");
    let _ = std::fs::remove_dir_all(&root);
}

/// 设备档案只能探测它自己声明的编译器：本机没装就不能报成宿主的 `cc`。
#[test]
fn a_device_profile_probes_the_tool_it_declares() {
    let ascend = cutile::prof::Profile::parse(prof::embedded("ascend").expect("ascend 档案"))
        .expect("档案可解析");
    match for_profile(&ascend) {
        Some(tool) => assert_eq!(tool.command, "ccec", "只认档案声明的编译器"),
        None => assert!(
            probe_declared("ccec").is_none(),
            "`ccec` 探得到就不能报成没有"
        ),
    }
}

#[test]
fn missing_compiler_is_not_an_error_by_itself() {
    // 只是探测：返回 None 也算合法结果，调用方负责报"缺设备即失败"。
    let _ = detect();
}

#[test]
fn parse_version_only_returns_dotted_tokens_and_never_panics() {
    let mut rng = testkit::Rng::new(0x7E57);
    for _ in 0..64 {
        let line = format!(
            "tools {} release {}.{} build {}",
            rng.below(999),
            rng.below(999),
            rng.below(999),
            rng.below(99999)
        );
        if let Some(version) = parse_version(&line) {
            assert!(version.contains('.'), "版本串必须带点：{version}");
        }
    }
}
