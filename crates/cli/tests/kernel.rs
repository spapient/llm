use super::*;

fn args(list: &[&str]) -> Vec<String> {
    list.iter().map(|item| item.to_string()).collect()
}

#[test]
fn parses_each_subcommand_with_its_defaults() {
    assert_eq!(
        CutileCommand::parse(&args(&["doctor"])).unwrap(),
        CutileCommand::Doctor {
            op: DEFAULT_OP.to_string(),
            device: "auto".to_string(),
            why: None
        }
    );
    assert_eq!(
        CutileCommand::parse(&args(&[
            "doctor",
            "--device",
            "cpu",
            "--why",
            "83a3ecdb31cd1225"
        ]))
        .unwrap(),
        CutileCommand::Doctor {
            op: DEFAULT_OP.to_string(),
            device: "cpu".to_string(),
            why: Some("83a3ecdb31cd1225".to_string())
        }
    );
    assert_eq!(
        CutileCommand::parse(&args(&["cache", "gc"])).unwrap(),
        CutileCommand::Cache { limit: None }
    );
    assert_eq!(
        CutileCommand::parse(&args(&["cache", "gc", "--limit", "1048576"])).unwrap(),
        CutileCommand::Cache {
            limit: Some(1_048_576)
        }
    );
    assert_eq!(
        CutileCommand::parse(&args(&["verify", "--devices", "cpu,cuda", "--emit-ledger"])).unwrap(),
        CutileCommand::Verify {
            op: DEFAULT_OP.to_string(),
            devices: vec!["cpu".to_string(), "cuda".to_string()],
            ledger: true
        }
    );
    assert_eq!(
        CutileCommand::parse(&args(&["bench", "--shape", "1024x64", "--repeat", "3"])).unwrap(),
        CutileCommand::Bench {
            op: DEFAULT_OP.to_string(),
            shape: Some("1024x64".to_string()),
            device: None,
            repeat: 3
        }
    );
}

#[test]
fn bench_without_shape_leaves_it_to_the_operator() {
    // 缺省形状是算子的生产形状：文法里不再有一份方阵。
    assert_eq!(
        CutileCommand::parse(&args(&["bench"])).unwrap(),
        CutileCommand::Bench {
            op: DEFAULT_OP.to_string(),
            shape: None,
            device: None,
            repeat: 50
        }
    );
}

#[test]
fn unknown_arguments_are_rejected() {
    let error = CutileCommand::parse(&args(&["doctor", "--nope"])).unwrap_err();
    assert!(error.to_string().contains("未知参数"), "{error}");
    let error = CutileCommand::parse(&args(&["nope"])).unwrap_err();
    assert!(error.to_string().contains("未知子命令"), "{error}");
    assert!(CutileCommand::parse(&[]).is_err(), "缺少子命令不猜");
}

#[test]
fn options_belong_to_their_subcommand() {
    // `--shape` 只对 bench 有意义：写错子命令时报错，而不是静默忽略。
    let error = CutileCommand::parse(&args(&["verify", "--shape", "1024x64"])).unwrap_err();
    assert!(error.to_string().contains("未知参数"), "{error}");
}

#[test]
fn a_flag_without_a_value_is_rejected() {
    let error = CutileCommand::parse(&args(&["doctor", "--device"])).unwrap_err();
    assert!(error.to_string().contains("缺少取值"), "{error}");
    let error = CutileCommand::parse(&args(&["bench", "--repeat", "many"])).unwrap_err();
    assert!(error.to_string().contains("正整数"), "{error}");
    let error = CutileCommand::parse(&args(&["cache", "gc", "--limit", "2GiB"])).unwrap_err();
    assert!(error.to_string().contains("字节数"), "{error}");
}

#[test]
fn cache_needs_a_maintenance_action() {
    // `cache` 自己不是动作：缺子命令与未实现的子命令都报错，不静默当成 gc。
    let error = CutileCommand::parse(&args(&["cache"])).unwrap_err();
    assert!(error.to_string().contains("缺少子命令"), "{error}");
    let error = CutileCommand::parse(&args(&["cache", "rm"])).unwrap_err();
    assert!(error.to_string().contains("未知子命令"), "{error}");
}
