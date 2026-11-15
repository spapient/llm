use super::*;

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
            "tool {} release {}.{} build {}",
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

#[test]
fn paths_without_an_msvc_segment_parse_as_empty() {
    let mut rng = testkit::Rng::new(0xC12);
    for _ in 0..32 {
        let path = format!(r"C:\Tools\{}\bin\cl.exe", rng.below(9999));
        assert!(msvc_version(Path::new(&path)).is_empty());
    }
}

#[test]
fn the_newest_msvc_wins_by_version_not_by_string_order() {
    let candidates = [
        r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\9.29.30133\bin\Hostx64\x64\cl.exe",
        r"D:\BuildTools\VC\Tools\MSVC\17.10.0\bin\Hostx64\x64\cl.exe",
    ];
    let newest = candidates
        .into_iter()
        .max_by_key(|cl| msvc_version(Path::new(cl)))
        .expect("候选非空");
    assert!(
        newest.contains("17.10.0"),
        "字符串序会把 9.29 排到 17.10 之后：{newest}"
    );
}

#[test]
fn a_path_without_a_version_loses_to_one_with_it() {
    assert!(msvc_version(Path::new(r"C:\msvc\cl.exe")).is_empty());
    assert_eq!(
        msvc_version(Path::new(
            r"C:\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64\cl.exe"
        )),
        vec![14, 29, 30133]
    );
}
