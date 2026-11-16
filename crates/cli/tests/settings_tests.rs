use super::*;

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

#[test]
fn defaults_are_absolute_user_level_paths() {
    let roots = Roots::new(test_home());
    let settings = Settings::defaults(&roots);
    assert_eq!(settings.cache_dir(), roots.default_cache_dir(APP));
    assert_eq!(settings.work_dir(), roots.default_work_dir(APP));
    assert!(settings.cache_dir().is_absolute());
    assert_eq!(settings.data_root(), None);
}

#[test]
fn an_empty_environment_uses_defaults() {
    let roots = Roots::new("/home/u");
    let settings = Settings::load_with(&roots, nothing).expect("没有环境变量不是错误");
    assert_eq!(settings.cache_dir(), roots.default_cache_dir(APP));
    assert_eq!(settings.work_dir(), roots.default_work_dir(APP));
    assert_eq!(settings.data_root(), None);
}

#[test]
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

#[test]
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

#[test]
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

#[test]
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

#[test]
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

#[test]
fn an_override_without_a_value_is_an_error() {
    let roots = Roots::new("/home/u");
    let mut settings = Settings::defaults(&roots);
    let error = settings
        .apply_arguments(&["--set".to_string()])
        .expect_err("`--set` 后面什么都没有");
    assert_eq!(error.code(), Code::Invalid);
}

#[test]
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
    assert!(format!("{empty}").contains("路径为空"), "{empty}");
}

#[test]
fn describe_lists_every_key_with_its_origin() {
    let roots = Roots::new("/home/u");
    let text = Settings::defaults(&roots).describe();
    assert!(
        text.starts_with("[配置] 默认 < 环境变量 < 命令行"),
        "{text}"
    );
    for key in Key::KEYS {
        assert!(text.contains(key.dotted()), "{text}");
    }
    assert!(text.contains("内置默认值"), "{text}");
    assert!(text.contains("未设置"), "没有默认值的键要如实写出：{text}");
    assert!(!text.contains("文件"), "不再有配置文件这一层：{text}");
}

#[test]
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

#[test]
fn the_relative_base_can_be_changed_without_moving_the_directories() {
    let roots = Roots::new("/home/u").with_base("/work/demo");
    assert_eq!(roots.base(), Path::new("/work/demo"));
    assert_eq!(
        roots.default_cache_dir(APP),
        PathBuf::from("/home/u/cutile/cache")
    );
}

#[test]
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
