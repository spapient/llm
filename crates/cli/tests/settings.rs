//! `settings` 的单元测试：默认目录、优先级、来源追踪、配置文件层。
//!
//! 经 `src/settings.rs` 的 `#[path]` 挂进 lib 单测（N12：`src/` 里不放测试）。

use super::*;
use crate::key::APP;
use crate::key::Key;
use diagnostic::Code;
use std::path::PathBuf;

fn nothing(_: &str) -> Option<String> {
    None
}

/// 测试用的用户根：必须是当前平台认可的绝对路径，否则 `is_absolute` 在 Windows 上为假。
#[cfg(unix)]
fn test_home() -> PathBuf {
    PathBuf::from("/home/u")
}

#[cfg(windows)]
fn test_home() -> PathBuf {
    PathBuf::from(r"C:\Users\u")
}

testkit::test_cases! {
fn defaults_are_absolute_user_level_paths() {
    let roots = Roots::new(test_home());
    let settings = Settings::defaults(&roots);
    assert_eq!(settings.cache_dir(), roots.default_cache_dir(APP));
    assert_eq!(settings.work_dir(), roots.default_work_dir(APP));
    assert!(settings.cache_dir().is_absolute());
    assert_eq!(settings.data_root(), None);
}

fn an_empty_environment_uses_defaults() {
    let roots = Roots::new("/home/u");
    let settings = Settings::load_with(&roots, nothing).expect("没有环境变量不是错误");
    assert_eq!(settings.cache_dir(), roots.default_cache_dir(APP));
    assert_eq!(settings.work_dir(), roots.default_work_dir(APP));
    assert_eq!(settings.data_root(), None);
}

fn env_wins_over_defaults_and_rebases_relative_paths() {
    let roots = Roots::new("/home/u").with_base("/cwd");
    let settings = Settings::load_with(&roots, |name| match name {
        "CUTILE_CACHE_DIR" => Some("/env/cache".to_string()),
        "CUTILE_WORK_DIR" => Some("env-work".to_string()),
        "CUTILE_DATA_DIR" => Some("data".to_string()),
        _ => None,
    })
    .expect("配置可加载");

    assert_eq!(settings.cache_dir(), PathBuf::from("/env/cache"));
    // 环境变量里的相对路径按启动目录解析，不按缓存根。
    assert_eq!(settings.work_dir(), PathBuf::from("/cwd/env-work"));
    assert_eq!(settings.data_root(), Some(Path::new("/cwd/data")));
}

fn the_legacy_cache_variable_is_still_honoured() {
    let roots = Roots::new("/home/u");
    let settings = Settings::load_with(&roots, |name| {
        (name == "CUTILE_CACHE").then(|| "legacy/cache".to_string())
    })
    .expect("配置可加载");
    assert_eq!(settings.cache_dir(), PathBuf::from("/home/u/legacy/cache"));
    assert!(
        settings.describe().contains("CUTILE_CACHE（env）"),
        "来源要写出兼容名：{}",
        settings.describe()
    );
}

fn the_formal_variable_beats_the_legacy_one() {
    let roots = Roots::new("/home/u");
    let settings = Settings::load_with(&roots, |name| match name {
        "CUTILE_CACHE_DIR" => Some("/formal".to_string()),
        "CUTILE_CACHE" => Some("/legacy".to_string()),
        _ => None,
    })
    .expect("配置可加载");
    assert_eq!(settings.cache_dir(), PathBuf::from("/formal"));
}

fn cli_overrides_beat_everything() {
    let roots = Roots::new("/home/u");
    let mut settings = Settings::load_with(&roots, |name| {
        (name == "CUTILE_CACHE_DIR").then(|| "/env/cache".to_string())
    })
    .expect("配置可加载");
    assert_eq!(settings.cache_dir(), PathBuf::from("/env/cache"));
    settings
        .apply_override("cache.dir=/cli/cache")
        .expect("覆盖");
    assert_eq!(settings.cache_dir(), PathBuf::from("/cli/cache"));
    assert!(settings.apply_override("cache.dir").is_err());
}

fn arguments_are_split_into_overrides_and_payload() {
    let roots = Roots::new("/home/u");
    let mut settings = Settings::defaults(&roots);
    let rest = settings
        .apply_arguments(&[
            "--set".to_string(),
            "cache.dir=/cli/cache".to_string(),
            "doctor".to_string(),
            "--device".to_string(),
            "cpu".to_string(),
        ])
        .expect("合法覆盖");
    assert_eq!(rest, ["doctor", "--device", "cpu"]);
    assert_eq!(settings.cache_dir(), PathBuf::from("/cli/cache"));
}

fn an_override_without_a_value_is_an_error() {
    let roots = Roots::new("/home/u");
    let mut settings = Settings::defaults(&roots);
    let error = settings
        .apply_arguments(&["--set".to_string()])
        .expect_err("`--set` 后面什么都没有");
    assert_eq!(error.code(), Code::Invalid);
}

fn empty_values_and_unknown_keys_are_rejected() {
    let roots = Roots::new("/home/u");
    let mut settings = Settings::defaults(&roots);
    let unknown = settings
        .apply_override("cache.dirr=/tmp")
        .expect_err("键未登记");
    assert!(
        format!("{unknown}").contains("cache.dir、work.dir、data.root"),
        "报错要列出登记过的键：{unknown}"
    );
    let empty = settings.apply_override("cache.dir=  ").expect_err("空值");
    assert!(format!("{empty}").contains("值为空"), "{empty}");
}

fn describe_lists_every_key_with_its_origin() {
    let roots = Roots::new("/home/u");
    let text = Settings::defaults(&roots).describe();
    assert!(
        text.starts_with("[配置] 默认 < 配置文件 < 环境变量 < 命令行"),
        "{text}"
    );
    for key in Key::KEYS {
        assert!(text.contains(key.dotted()), "{text}");
    }
    assert!(text.contains("内置默认值"), "{text}");
    assert!(text.contains("未设置"), "没有默认值的键要如实写出：{text}");
}

fn describe_marks_the_layer_of_every_override() {
    let roots = Roots::new("/home/u");
    let mut settings = Settings::load_with(&roots, |name| {
        (name == "CUTILE_WORK_DIR").then(|| "/env/work".to_string())
    })
    .expect("配置可加载");
    settings
        .apply_override("cache.dir=/cli/cache")
        .expect("覆盖");
    let text = settings.describe();
    assert!(text.contains("CUTILE_WORK_DIR（env）"), "{text}");
    assert!(text.contains("--set cache.dir=/cli/cache（cli）"), "{text}");
}
}

/// 临时配置文件路径：pid + 原子序号保证并行测试不撞名。
fn temp_config() -> PathBuf {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    std::env::temp_dir().join(format!(
        "cutile-settings-{}-{}.toml",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ))
}

testkit::test_cases! {
fn a_config_file_sets_values_and_reports_its_origin() {
    let path = temp_config();
    std::fs::write(
        &path,
        "cache.dir = \"/file/cache\"\ncache.limit_bytes = 1073741824\n",
    )
    .expect("写临时配置");
    let roots = Roots::new("/home/u");
    let mut settings = Settings::load_with(&roots, nothing).expect("配置可加载");
    settings.load_config(&path).expect("合法配置文件");
    assert_eq!(settings.cache_dir(), PathBuf::from("/file/cache"));
    assert_eq!(settings.cache_limit_bytes(), Some(1024 * 1024 * 1024));
    let text = settings.describe();
    assert!(
        text.contains("（file）") && text.contains(&path.display().to_string()),
        "来源要写出文件路径：{text}"
    );
    std::fs::remove_file(&path).ok();
}

fn a_missing_config_file_is_a_missing_error() {
    let roots = Roots::new("/home/u");
    let mut settings = Settings::load_with(&roots, nothing).expect("配置可加载");
    let error = settings
        .load_config(Path::new("/no/such/cutile-config.toml"))
        .expect_err("显式指定但不存在");
    assert_eq!(error.code(), Code::Missing);
}

fn a_malformed_config_file_is_rejected() {
    let path = temp_config();
    std::fs::write(&path, "cache.dir = ").expect("写坏配置");
    let roots = Roots::new("/home/u");
    let mut settings = Settings::load_with(&roots, nothing).expect("配置可加载");
    let error = settings.load_config(&path).expect_err("坏 TOML 必须报错");
    assert_eq!(error.code(), Code::Invalid);
    std::fs::remove_file(&path).ok();
}

fn unknown_keys_and_wrong_types_in_a_config_file_are_rejected() {
    let roots = Roots::new("/home/u");
    let unknown = temp_config();
    std::fs::write(&unknown, "no.such.key = 1").expect("写未知键配置");
    let mut settings = Settings::load_with(&roots, nothing).expect("配置可加载");
    let error = settings.load_config(&unknown).expect_err("未知键必须报错");
    assert!(format!("{error}").contains("cache.dir"), "{error}");
    std::fs::remove_file(&unknown).ok();

    let mistyped = temp_config();
    std::fs::write(&mistyped, "cache.limit_bytes = \"abc\"").expect("写类型错配置");
    let error = settings.load_config(&mistyped).expect_err("类型错必须报错");
    assert!(format!("{error}").contains("非负整数"), "{error}");
    std::fs::remove_file(&mistyped).ok();
}

fn env_beats_file_and_cli_beats_env() {
    let path = temp_config();
    std::fs::write(&path, "cache.dir = \"/file/cache\"").expect("写临时配置");
    let roots = Roots::new("/home/u");
    let mut settings = Settings::load_with(&roots, |name| {
        (name == "CUTILE_CACHE_DIR").then(|| "/env/cache".to_string())
    })
    .expect("配置可加载");
    settings.load_config(&path).expect("合法配置文件");
    assert_eq!(settings.cache_dir(), PathBuf::from("/env/cache"));
    settings
        .apply_override("cache.dir=/cli/cache")
        .expect("覆盖");
    assert_eq!(settings.cache_dir(), PathBuf::from("/cli/cache"));
    std::fs::remove_file(&path).ok();
}

fn arguments_are_split_into_config_overrides_and_payload() {
    let path = temp_config();
    std::fs::write(&path, "cache.dir = \"/file/cache\"").expect("写临时配置");
    let roots = Roots::new("/home/u");
    let mut settings = Settings::load_with(&roots, nothing).expect("配置可加载");
    let rest = settings
        .apply_arguments(&[
            "--config".to_string(),
            path.display().to_string(),
            "doctor".to_string(),
        ])
        .expect("合法覆盖");
    assert_eq!(rest, ["doctor"]);
    assert_eq!(settings.cache_dir(), PathBuf::from("/file/cache"));
    std::fs::remove_file(&path).ok();
}

fn the_numeric_key_has_its_current_default() {
    let roots = Roots::new("/home/u");
    let settings = Settings::load_with(&roots, nothing).expect("配置可加载");
    assert_eq!(settings.cache_limit_bytes(), Some(2 * 1024 * 1024 * 1024));
    assert!(settings.describe().contains("cache.limit_bytes"));
}

fn the_relative_base_can_be_changed_without_moving_the_directories() {
    let roots = Roots::new("/home/u").with_base("/work/demo");
    assert_eq!(roots.base(), Path::new("/work/demo"));
    assert_eq!(
        roots.default_cache_dir(APP),
        PathBuf::from("/home/u/cutile/cache")
    );
}

fn discovery_never_anchors_to_the_repository_checkout() {
    let roots = Roots::discover();
    let cache = roots.default_cache_dir(APP);
    assert!(cache.is_absolute(), "{}", cache.display());
    assert!(
        !cache.starts_with(env!("CARGO_MANIFEST_DIR")),
        "{}",
        cache.display()
    );
}

fn train_spec_defaults_come_from_the_train_types() {
    let roots = Roots::new(test_home());
    let settings = Settings::defaults(&roots);
    assert_eq!(
        settings.train_spec().expect("默认训练配置合法"),
        train::TrainSpec::default()
    );
}

fn train_settings_project_environment_and_cli_overrides() {
    let roots = Roots::new(test_home());
    let mut settings = Settings::load_with(&roots, |name| match name {
        "CUTILE_TRAIN_LEARNING_RATE" => Some("0.2".to_string()),
        "CUTILE_TRAIN_WARMUP_STEPS" => Some("2".to_string()),
        "CUTILE_TRAIN_MAX_STEPS" => Some("8".to_string()),
        "CUTILE_TRAIN_BATCH_ROWS" => Some("2".to_string()),
        "CUTILE_TRAIN_BATCH_COLS" => Some("4".to_string()),
        _ => None,
    })
    .expect("训练环境配置合法");
    settings
        .apply_override("train.learning_rate=0.5")
        .expect("CLI 覆盖学习率");
    settings
        .apply_override("train.accumulation=3")
        .expect("CLI 覆盖累积窗口");
    settings
        .apply_override("train.clip_norm=0.25")
        .expect("CLI 覆盖裁剪");

    let spec = settings.train_spec().expect("投影合法");
    assert_eq!(spec.session.learning_rate, 0.5);
    assert_eq!(spec.session.warmup_steps, 2);
    assert_eq!(spec.session.max_steps, 8);
    assert_eq!(spec.session.accumulation, 3);
    assert_eq!(spec.session.clip_norm, Some(0.25));
    assert_eq!((spec.loader.rows, spec.loader.cols), (2, 4));
}

fn train_objectives_project_to_typed_config() {
    let roots = Roots::new(test_home());
    let mut settings = Settings::defaults(&roots);
    settings
        .apply_override("train.objective=distill")
        .expect("蒸馏目标");
    settings
        .apply_override("train.distill_temperature=3.5")
        .expect("温度");
    assert_eq!(
        settings.train_spec().expect("蒸馏配置").objective,
        train::ObjectiveConfig::Distill(train::Distill::new(3.5, 0.5))
    );
    settings
        .apply_override("train.objective=grpo")
        .expect("GRPO 目标");
    settings
        .apply_override("train.grpo_beta=0.2")
        .expect("KL 系数");
    assert_eq!(
        settings.train_spec().expect("GRPO 配置").objective,
        train::ObjectiveConfig::Grpo(train::Grpo::new(0.2, 0.2))
    );
    settings
        .apply_override("train.objective=unknown")
        .expect("文本解析成功");
    assert_eq!(settings.train_spec().unwrap_err().code(), Code::Invalid);
}

fn invalid_train_values_are_rejected_at_parse_or_validation() {
    let roots = Roots::new(test_home());
    let mut settings = Settings::defaults(&roots);
    for (key, value) in [
        ("train.learning_rate", "NaN"),
        ("train.clip_norm", "inf"),
        ("train.max_steps", "-1"),
    ] {
        let error = settings
            .apply_override(&format!("{key}={value}"))
            .expect_err("非法配置值必须拒绝");
        assert_eq!(error.code(), Code::Invalid);
    }

    settings
        .apply_override("train.clip_norm=0")
        .expect("数值解析成功，语义校验在投影边界");
    assert_eq!(settings.train_spec().unwrap_err().code(), Code::Invalid);
}
}
