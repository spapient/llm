//! `cutile` 驱动方的单元测试：语法分派、退出码、排版落到 stdout。
//!
//! 经 `src/cutile.rs` 的 `#[path]` 挂进该 bin 的单测（N12：`src/` 里不放测试）。

use super::*;
use cutile::emit::Fragment;

fn settings() -> Settings {
    Settings::defaults(&llm_cli::settings::Roots::new(
        std::env::temp_dir().join("cutile-driver-tests"),
    ))
}

/// nvidia 在本机是否可执行（nvcc 在场）。T600 上为真，此时「缺设备即失败」的场景不成立，
/// 用例主动让位——设备在场时跑的是真机对拍（更强的证据），不是「缺设备」文案。
fn nvidia_present() -> bool {
    let catalogue = Catalogue::load(&settings().dirs()).expect("内置数据");
    catalogue
        .profile("nvidia")
        .is_some_and(llm_optim::witness::device_available)
}

/// 性能判据每轮都要有一句结论：没声明峰值时说“未测量”，不能什么都不打印。
#[test]
fn the_performance_line_is_printed_even_when_nothing_was_measured() {
    let peak = llm_optim::bench::Peak {
        value: 68.0e9,
        src: "2*8*2666e6".to_string(),
    };

    let reached = section::performance_note(
        llm_optim::bench::Performance::new(Some(0.72), false),
        Some(&peak),
    );
    assert!(reached.contains("72.0%"), "{reached}");
    assert!(reached.contains("达到 70% 门槛"), "{reached}");
    assert!(reached.contains("不是“相对原生库的 85%”"), "{reached}");

    let below = section::performance_note(
        llm_optim::bench::Performance::new(Some(0.0069), false),
        Some(&peak),
    );
    assert!(below.contains("0.7%"), "{below}");
    assert!(below.contains("低于 70% 门槛"), "{below}");
    assert!(below.contains("特化代码"), "{below}");

    let cached = section::performance_note(
        llm_optim::bench::Performance::new(Some(0.19), true),
        Some(&peak),
    );
    assert!(cached.contains("口径无效"), "{cached}");
    assert!(cached.contains("不参与 §16 判定"), "{cached}");

    // 没有峰值：判据没有尺子，仍要说清“未测量”，而不是静默通过。
    let unknown = section::performance_note(llm_optim::bench::Performance::new(None, false), None);
    assert!(unknown.contains("未测量"), "{unknown}");
    assert!(unknown.contains("不参与 Go/No-Go"), "{unknown}");
    assert!(!unknown.contains("门槛"), "{unknown}");
}

#[test]
fn doctor_prints_every_stage() {
    let text = doctor(&settings(), "gemv_i4", "cpu", None).expect("doctor 应成功");
    assert!(text.contains("path: pipe2"), "{text}");
    assert!(text.contains("[布局]"), "{text}");
    assert!(text.contains("[存储]"), "{text}");
    assert!(text.contains("[emit]"), "{text}");
    assert!(text.contains("cache key"), "{text}");
    assert!(text.contains("[配置]"), "{text}");
}

#[test]
fn doctor_prints_the_capacity_next_to_the_declared_traffic() {
    // 打印的必须是档案容量（判定的分母）与声明用量（分子的来源），不是中转缓冲的字节数。
    let text = doctor(&settings(), "gemv_i4", "cpu", None).expect("doctor 应成功");
    assert!(text.contains("容量 524288 B"), "{text}");
    assert!(text.contains("声明 load<=4096 B"), "{text}");
    assert!(text.contains("GM=main 容量 不限"), "{text}");
}

#[test]
fn doctor_explains_a_rejected_device_without_failing() {
    if nvidia_present() {
        return; // nvidia 可执行时，缺设备被拒场景不成立
    }
    let text = doctor(&settings(), "gemv_i4", "nvidia", None).expect("仍应打印诊断");
    assert!(text.contains("没有可行变体"), "{text}");
    assert!(text.contains("L2"), "{text}");
}

#[test]
fn doctor_explains_the_key_it_just_wrote() {
    // `--why` 要回答的是“这份缓存为什么没命中”：同一份键只能说一致，不能编差异。
    let settings = settings();
    let first = doctor(&settings, "gemv_i4", "cpu", None).expect("doctor 应成功");
    let hex = first
        .split("cache key ")
        .nth(1)
        .and_then(|rest| rest.split_whitespace().next())
        .expect("打印了键")
        .to_string();
    let text = doctor(&settings, "gemv_i4", "cpu", Some(&hex)).expect("doctor 应成功");
    assert!(text.contains("[缓存]"), "{text}");
    assert!(text.contains("与本次构建的键一致"), "{text}");
}

#[test]
fn doctor_explains_a_key_that_never_landed() {
    let text =
        doctor(&settings(), "gemv_i4", "cpu", Some("0000000000000000")).expect("doctor 应成功");
    assert!(text.contains("[缓存]"), "{text}");
    assert!(text.contains("从未落盘"), "{text}");
}

#[test]
fn doctor_reports_a_half_written_key_as_half_written() {
    // `doctor` 会先把本次的键补齐，所以“半截落盘”只能对另一个键观察到——
    // 那正是真实场景：另一个进程崩在 `put` 中间，留下 `meta.txt` 与 `source.tmp`。
    let settings = settings();
    let first = doctor(&settings, "gemv_i4", "cpu", None).expect("doctor 应成功");
    let hex = first
        .split("cache key ")
        .nth(1)
        .and_then(|rest| rest.split_whitespace().next())
        .expect("打印了键")
        .to_string();
    let source = settings.cache_dir().join(&hex[..2]).join(&hex);
    let half = settings.cache_dir().join("00").join("0000000000000001");
    std::fs::create_dir_all(&half).expect("建目录");
    std::fs::copy(source.join("meta.txt"), half.join("meta.txt")).expect("复制键快照");

    let text = doctor(&settings, "gemv_i4", "cpu", Some("0000000000000001")).expect("doctor");
    assert!(text.contains("只有半截落盘"), "{text}");
    assert!(
        !text.contains("与本次构建的键一致"),
        "没有 source 不能算命中：{text}"
    );
    let _ = std::fs::remove_dir_all(&half);
}

#[test]
fn verify_reports_a_missing_device_as_a_failure() {
    if nvidia_present() {
        return; // nvidia 可执行时，缺设备即失败场景不成立
    }
    let devices = vec!["nvidia".to_string()];
    let error = verify(&settings(), "gemv_i4", &devices, false).unwrap_err();
    assert_eq!(error.code(), Code::Unsupported);
    assert!(error.to_string().contains("缺设备即失败"), "{error}");
}

#[test]
fn the_ledger_lists_every_declared_primitive() {
    let catalogue = Catalogue::load(&settings().dirs()).expect("内置数据");
    let profile = catalogue.resolve("cpu").expect("cpu 档案");
    let verified = Verified {
        device: "cpu".to_string(),
        channel: llm_optim::verify::Channel::Parallel,
        vendor: profile.meta.vendor.to_string(),
        id: profile.meta.id.to_string(),
        variant: "pipe2".to_string(),
        tile: "GMAX=32".to_string(),
        shapes: vec![(32, 1), (4096, 4096)],
        witnesses: 9,
        self_checked: 0,
        max_rel_err: 0.0,
        fragment: Fragment::new(53),
        tool_ver: profile.meta.tool_ver.clone(),
        toolchain: Some("cc@21.0.0".to_string()),
        toolchain_agrees: true,
        flags: "-O3".to_string(),
        key: "0".to_string(),
        artifacts: Vec::new(),
    };
    let note = write_ledger(
        &catalogue,
        &verified,
        &section::passed_line(&verified),
        &settings().dirs(),
    )
    .expect("应写出台账");
    let path = note.trim().rsplit_once(' ').expect("路径").1;
    let text = std::fs::read_to_string(path).expect("台账可读");
    assert!(text.contains("| 原语 | 实现 | 等级 | 见证 |"), "{text}");
    assert!(text.contains("pipe2"), "{text}");
}

/// `cache gc` 的上限来源：`--limit` 最高，其次是配置键 `cache.limit_bytes`，内核常量
/// 只在两者都缺时兜底；默认值与内核缺省同值（行为不变）。
#[test]
fn cache_gc_prefers_the_flag_then_the_settings_key() {
    let roots = llm_cli::settings::Roots::new(
        std::env::temp_dir().join(format!("cutile-gc-settings-{}", std::process::id())),
    );
    let mut settings = Settings::defaults(&roots);
    settings
        .apply_override("cache.limit_bytes=1024")
        .expect("合法覆盖");

    let text = cache_gc(&settings, None).expect("空缓存可回收");
    assert!(
        text.starts_with(&format!(
            "cache gc | 上限 {}",
            ::diagnostic::render::human_bytes(1024)
        )),
        "{text}"
    );

    let text = cache_gc(&settings, Some(2048)).expect("命令行覆盖");
    assert!(
        text.starts_with(&format!(
            "cache gc | 上限 {}",
            ::diagnostic::render::human_bytes(2048)
        )),
        "{text}"
    );

    let text = cache_gc(&Settings::defaults(&roots), None).expect("空缓存可回收");
    assert!(
        text.starts_with(&format!(
            "cache gc | 上限 {}",
            ::diagnostic::render::human_bytes(cutile::cache::CACHE_LIMIT)
        )),
        "{text}"
    );
}
