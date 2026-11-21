use super::*;
use diagnostic::Code;

fn args(list: &[&str]) -> Vec<String> {
    list.iter().map(|item| item.to_string()).collect()
}

#[test]
fn the_default_command_is_status() {
    assert_eq!(
        Invocation::parse(&[]).expect("默认").command(),
        &Command::Status { weights: None }
    );
    assert_eq!(
        Invocation::parse(&args(&["status"]))
            .expect("显式")
            .command(),
        &Command::Status { weights: None }
    );
}

#[test]
fn platform_is_a_zero_option_command() {
    assert_eq!(
        Invocation::parse(&args(&["platform"]))
            .expect("零选项")
            .command(),
        &Command::Platform
    );
    let extra = Invocation::parse(&args(&["platform", "--op", "gemv_i4"])).expect_err("不接受参数");
    assert!(extra.to_string().contains("platform"), "{extra}");
}

#[test]
fn a_weights_file_is_carried_through_without_being_opened() {
    assert_eq!(
        Invocation::parse(&args(&["status", "--weights", "model.safetensors"]))
            .expect("合法")
            .command(),
        &Command::Status {
            weights: Some("model.safetensors".to_string())
        }
    );
    let missing = Invocation::parse(&args(&["status", "--weights"])).expect_err("缺取值");
    assert!(missing.to_string().contains("缺少取值"), "{missing}");
    let wrong = Invocation::parse(&args(&["status", "--op", "gemv_i4"]))
        .expect_err("status 只接受 --weights");
    assert!(wrong.to_string().contains("--weights"), "{wrong}");
}

#[test]
fn run_carries_its_options_in_any_order_and_refuses_to_guess_the_file() {
    assert_eq!(
        Invocation::parse(&args(&[
            "run",
            "--device",
            "cpu",
            "--weights",
            "model.safetensors",
            "--input",
            "1,2,3",
        ]))
        .expect("合法")
        .command(),
        &Command::Run {
            weights: "model.safetensors".to_string(),
            input: Some("1,2,3".to_string()),
            device: Some("cpu".to_string()),
        }
    );
    let missing = Invocation::parse(&args(&["run", "--input", "1,2"])).expect_err("缺 --weights");
    assert!(missing.to_string().contains("--weights"), "{missing}");
    let unknown = Invocation::parse(&args(&["run", "--weights", "m", "--op", "gemv_i4"]))
        .expect_err("未知选项");
    assert!(unknown.to_string().contains("--op"), "{unknown}");
}

#[test]
fn train_carries_its_options_without_parsing_numbers() {
    assert_eq!(
        Invocation::parse(&args(&[
            "train",
            "--lr",
            "0.01",
            "--warmup",
            "10",
            "--steps",
            "100",
            "--clip",
            "1.0",
            "--accumulation",
            "2",
            "--load",
            "ckpt.snap",
        ]))
        .expect("合法")
        .command(),
        &Command::Train {
            lr: Some("0.01".to_string()),
            warmup: Some("10".to_string()),
            steps: Some("100".to_string()),
            clip: Some("1.0".to_string()),
            accumulation: Some("2".to_string()),
            load: Some("ckpt.snap".to_string()),
        }
    );
    // 不带任何选项也是合法命令：数值缺省由组装点的训练配置决定。
    assert_eq!(
        Invocation::parse(&args(&["train"]))
            .expect("无选项")
            .command(),
        &Command::Train {
            lr: None,
            warmup: None,
            steps: None,
            clip: None,
            accumulation: None,
            load: None,
        }
    );
    let unknown = Invocation::parse(&args(&["train", "--op", "gemv_i4"])).expect_err("未知选项");
    assert!(unknown.to_string().contains("--op"), "{unknown}");
    let missing = Invocation::parse(&args(&["train", "--lr"])).expect_err("缺取值");
    assert!(missing.to_string().contains("缺少取值"), "{missing}");
}

#[test]
fn help_has_three_spellings() {
    for spelling in ["help", "-h", "--help"] {
        assert_eq!(
            Invocation::parse(&args(&[spelling]))
                .expect("帮助")
                .command(),
            &Command::Help,
            "{spelling}"
        );
    }
}

#[test]
fn archive_commands_carry_paths_through_without_opening_here() {
    assert_eq!(
        Invocation::parse(&args(&[
            "import",
            "--in",
            "model.safetensors",
            "--out",
            "model.u"
        ]))
        .expect("合法")
        .command(),
        &Command::Import {
            input: "model.safetensors".to_string(),
            output: "model.u".to_string(),
        }
    );
    assert_eq!(
        Invocation::parse(&args(&["inspect", "--in", "model.u"]))
            .expect("合法")
            .command(),
        &Command::Inspect {
            input: "model.u".to_string()
        }
    );
    assert_eq!(
        Invocation::parse(&args(&["check", "--in", "model.u"]))
            .expect("合法")
            .command(),
        &Command::Check {
            input: "model.u".to_string()
        }
    );

    assert_eq!(
        Invocation::parse(&args(&["settle", "--in", "model.u"]))
            .expect("合法")
            .command(),
        &Command::Settle {
            input: "model.u".to_string(),
            output: None,
        },
        "不给 --out 就是打到 stdout，不猜默认路径"
    );
    assert_eq!(
        Invocation::parse(&args(&[
            "settle",
            "--out",
            "config.json",
            "--in",
            "model.u"
        ]))
        .expect("顺序自由")
        .command(),
        &Command::Settle {
            input: "model.u".to_string(),
            output: Some("config.json".to_string()),
        }
    );
    assert_eq!(
        Invocation::parse(&args(&[
            "audit",
            "--in",
            "model.u",
            "--ledger",
            "ledger.json"
        ]))
        .expect("合法")
        .command(),
        &Command::Audit {
            input: "model.u".to_string(),
            ledger: "ledger.json".to_string(),
        }
    );
    assert_eq!(
        Invocation::parse(&args(&[
            "audit",
            "--ledger",
            "ledger.json",
            "--in",
            "model.u"
        ]))
        .expect("顺序自由")
        .command(),
        &Command::Audit {
            input: "model.u".to_string(),
            ledger: "ledger.json".to_string(),
        }
    );

    let no_ledger =
        Invocation::parse(&args(&["audit", "--in", "model.u"])).expect_err("缺 --ledger");
    assert!(no_ledger.to_string().contains("--ledger"), "{no_ledger}");
    let missing_out =
        Invocation::parse(&args(&["import", "--in", "model.safetensors"])).expect_err("缺 --out");
    assert!(missing_out.to_string().contains("--out"), "{missing_out}");
    let missing_in = Invocation::parse(&args(&["check"])).expect_err("缺 --in");
    assert!(missing_in.to_string().contains("--in"), "{missing_in}");
    let unknown = Invocation::parse(&args(&["inspect", "--op", "gemv_i4"])).expect_err("未知选项");
    assert!(unknown.to_string().contains("--op"), "{unknown}");
}

#[test]
fn unknown_words_and_flags_are_rejected_with_the_usage_hint() {
    let unknown = Invocation::parse(&args(&["frobnicate"])).expect_err("未知命令");
    assert_eq!(unknown.code(), Code::Invalid);
    assert!(unknown.to_string().contains("frobnicate"), "{unknown}");

    let flag = Invocation::parse(&args(&["run", "--weights"])).expect_err("缺取值");
    assert!(flag.to_string().contains("缺少取值"), "{flag}");

    let extra = Invocation::parse(&args(&["status", "now"])).expect_err("多余参数");
    assert!(extra.to_string().contains("now"), "{extra}");
}

#[test]
fn the_usage_text_lists_every_command() {
    for command in [
        "status", "platform", "run", "train", "import", "inspect", "check", "settle", "audit",
        "help",
    ] {
        assert!(USAGE.contains(command), "{command}");
    }
}

/// 编译期自证：`Command` 的每个变体都必须在这里有落点，且这里**不写 `_` 兜底臂**。
///
/// `lib.rs` 的分派为了守住 `Ccn ≤ 5` 分成四层，末端各有一个 `_` 臂；那些臂走不走得到不由
/// 文档声明，而由这张平表证明：新增变体时这里编译失败，`cargo check --all-targets` 因此点名
/// 要求接线。名字同时充当"这条命令确实出现在用法里"的清单。
#[test]
fn every_command_variant_has_a_destination() {
    fn destination(command: &Command) -> &'static str {
        match command {
            Command::Status { .. } => "status",
            Command::Run { .. } => "run",
            Command::Train { .. } => "train",
            Command::Import { .. } => "import",
            Command::Inspect { .. } => "inspect",
            Command::Check { .. } => "check",
            Command::Settle { .. } => "settle",
            Command::Audit { .. } => "audit",
            Command::Platform => "platform",
            Command::Help => "help",
        }
    }
    let all = [
        Command::Status { weights: None },
        Command::Platform,
        Command::Help,
    ];
    for command in &all {
        assert!(USAGE.contains(destination(command)), "{command:?}");
    }
    let mut names: Vec<&str> = all.iter().map(destination).collect();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), all.len(), "落点名重复：{names:?}");
}
