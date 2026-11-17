use diagnostic::Code;
use llm_cli::{Command, Invocation, USAGE};

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
fn an_operator_is_carried_through_without_defaulting_here() {
    assert_eq!(
        Invocation::parse(&args(&["verify", "--op", "gemv_i4"]))
            .expect("合法")
            .command(),
        &Command::Verify {
            op: Some("gemv_i4".to_string())
        }
    );
    assert_eq!(
        Invocation::parse(&args(&["bench"]))
            .expect("无选项")
            .command(),
        &Command::Bench { op: None }
    );
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

    let flag = Invocation::parse(&args(&["verify", "--op"])).expect_err("缺取值");
    assert!(flag.to_string().contains("缺少取值"), "{flag}");

    let extra = Invocation::parse(&args(&["status", "now"])).expect_err("多余参数");
    assert!(extra.to_string().contains("now"), "{extra}");
}

#[test]
fn the_usage_text_lists_every_command() {
    for command in [
        "status", "verify", "bench", "train", "import", "inspect", "check", "help",
    ] {
        assert!(USAGE.contains(command), "{command}");
    }
}
