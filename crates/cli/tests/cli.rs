//! 进程级端到端：真的派生二进制，断言"解析 → 分派 → 执行 → 退出码 → 输出"这条链。
//!
//! `cli::Invocation::parse` 与 `USAGE` 的单元测试与生产类型同文件（`src/lib.rs`）；本文件补的是
//! 它们之间那段没人测过的路：`main` 把解析结果分派给哪一层、错误落成哪个退出码、输出去了 stdout 还是 stderr。
//! 用例不读真实模型：权重文件都在临时目录里现造（合成 safetensors、被截断的一份、内容坏的一份）。
//!
//! 退出码是**对外契约**（`diagnostic::Code::exit_code`）：`invalid=2`、`unsupported=3`、
//! `missing=4`、`io=5`、`internal=70`。任何一个改动都必须先改这里。

use assert_cmd::Command;
use std::path::PathBuf;

/// 用法文本里必须出现的那几行：命令表是 `llm help` 的唯一职责。
const USAGE_COMMANDS: [&str; 10] = [
    "status", "run", "verify", "bench", "train", "import", "inspect", "check", "settle", "audit",
];

fn llm() -> Command {
    Command::cargo_bin("llm").expect("二进制在测试构建里必须存在")
}

/// 写一份合成 safetensors：两个秩 2 条目 + 一条秩 1 条目（故意留一条未覆盖）。
///
/// 数据段全零：示例只验证"链路取到了字节"，数值含义由库层用例负责。
fn write_weights(tag: &str, header: &str, data_bytes: usize) -> PathBuf {
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.resize(bytes.len() + data_bytes, 0);
    let path =
        std::env::temp_dir().join(format!("llm-cli-{}-{tag}.safetensors", std::process::id()));
    std::fs::write(&path, &bytes).expect("临时权重可写");
    path
}

fn good_weights(tag: &str) -> (PathBuf, String) {
    let header = r#"{"blk.0.attn_q.weight": {"dtype": "F32", "shape": [4, 4], "data_offsets": [0, 64]},
                    "blk.0.norm.weight": {"dtype": "F32", "shape": [4], "data_offsets": [64, 80]}}"#;
    let path = write_weights(tag, header, 80);
    (path, header.to_string())
}

#[test]
fn help_prints_the_usage_and_succeeds() {
    let output = llm().arg("help").output().expect("进程可派生");
    assert_eq!(output.status.code(), Some(0), "`help` 是正常结束，不是错误");
    let stdout = String::from_utf8_lossy(&output.stdout);
    for command in USAGE_COMMANDS {
        assert!(
            stdout.contains(command),
            "用法文本必须列出 `{command}`：命令表只有一处定义"
        );
    }
    assert!(
        output.stderr.is_empty(),
        "成功路径不该往 stderr 写东西：{:?}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn the_default_command_reports_every_layer() {
    let output = llm().output().expect("进程可派生");
    assert_eq!(output.status.code(), Some(0), "不给参数就是 `status`");
    let stdout = String::from_utf8_lossy(&output.stdout);
    for layer in [
        "声明层  op",
        "档案层  prof",
        "片段层  prim",
        "层定义  nn",
        "下界    模型",
    ] {
        assert!(stdout.contains(layer), "status 必须报出 `{layer}` 这一行");
    }
}

#[test]
fn an_unknown_command_exits_with_the_usage_error_code() {
    let output = llm().arg("frobnicate").output().expect("进程可派生");
    assert_eq!(output.status.code(), Some(2), "未知命令是 invalid=2");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("✗"), "失败必须带统一前缀：{stderr:?}");
    assert!(
        stderr.contains("用法"),
        "失败后要跟着用法文本，否则用户得再查一次文档"
    );
    assert!(output.stdout.is_empty(), "错误信息不该同时写到 stdout");
}

#[test]
fn a_missing_option_value_is_a_parse_error_not_a_default() {
    let output = llm()
        .args(["run", "--weights"])
        .output()
        .expect("进程可派生");
    assert_eq!(output.status.code(), Some(2), "缺取值是文法错误");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--weights"),
        "错误里要点名缺取值的选项：{stderr:?}"
    );
}

#[test]
fn a_weights_file_that_does_not_exist_exits_with_the_io_code() {
    let missing = std::env::temp_dir().join("llm-cli-definitely-absent.safetensors");
    let output = llm()
        .args(["run", "--weights"])
        .arg(&missing)
        .output()
        .expect("进程可派生");
    assert_eq!(output.status.code(), Some(5), "打不开文件是 io=5");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("llm-cli-definitely-absent"),
        "错误里要带上是哪个路径：{stderr:?}"
    );
}

#[test]
fn a_truncated_weights_file_exits_with_the_invalid_code() {
    let (path, header) = good_weights("truncated");
    // 砍掉最后 16 字节：头部声明的数据段比文件长，属"文件被截断"。
    let bytes = std::fs::read(&path).expect("读回");
    std::fs::write(&path, &bytes[..bytes.len() - 16]).expect("写入截断版");
    let output = llm()
        .args(["run", "--weights"])
        .arg(&path)
        .output()
        .expect("进程可派生");
    assert_eq!(output.status.code(), Some(2), "自证失败是 invalid=2");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("截断"),
        "要说清是截断还是多余字节：{stderr:?}"
    );
    assert!(header.contains("blk.0.attn_q.weight"), "夹具本身没被改动");
}

#[test]
fn status_reads_headers_from_a_weights_file() {
    let (path, _) = good_weights("status");
    let output = llm()
        .args(["status", "--weights"])
        .arg(&path)
        .output()
        .expect("进程可派生");
    assert_eq!(
        output.status.code(),
        Some(0),
        "合法权重文件下 status 必须成功"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("safetensors"),
        "权重那一行要说明读的是哪种格式：{stdout}"
    );
    assert!(
        stdout.contains("未覆盖 1 条"),
        "秩 1 的条目必须如实列进未覆盖清单，而不是被静默忽略"
    );
}

/// 写一份合成快照：3 槽累积梯度、3 个 micro-batch、可选绑定对；权重段为空。
fn write_snapshot(tag: &str, ties: Vec<(usize, usize)>) -> PathBuf {
    use train::snapshot::{Facts, GradAccum, Moments, Snapshot, Ties};
    let facts = Facts::new(5, 1, 2, 1, 4).expect("恢复事实");
    let header = weights::Header::new(Vec::new(), 0).expect("空头部");
    let accum = GradAccum::new(vec![vec![1.0; 4], vec![2.0; 4], vec![3.0; 4]], 3);
    let snapshot = Snapshot::new_with_all(facts, header, Moments::empty(), accum, Ties::new(ties));
    let path = std::env::temp_dir().join(format!("llm-cli-{}-{tag}.snap", std::process::id()));
    snapshot.write(&path).expect("快照可写");
    path
}

#[test]
fn train_load_reports_accumulation_window_and_tie_pairs() {
    let tied = write_snapshot("tied", vec![(0, 1)]);
    let output = llm()
        .args(["train", "--load"])
        .arg(&tied)
        .args(["--accumulation", "4"])
        .output()
        .expect("进程可派生");
    assert_eq!(
        output.status.code(),
        Some(0),
        "合法快照下 `--load` 必须成功"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("3 / 4 micro-batch"),
        "累积窗口要报已汇总数 / 窗口：{stdout}"
    );
    assert!(stdout.contains("1 对"), "绑定对数要随快照报出：{stdout}");

    let loose = write_snapshot("loose", Vec::new());
    let output = llm()
        .args(["train", "--load"])
        .arg(&loose)
        .output()
        .expect("进程可派生");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("0 对"), "无绑定时报 0 对：{stdout}");
}

/// 合成一份 `.u` 容器：复用现有 safetensors 样板经 `import` 落盘，再交给 `inspect` / `check`。
fn imported_container(tag: &str) -> PathBuf {
    let source = good_weights(tag).0;
    let container = std::env::temp_dir().join(format!("llm-cli-{}-{tag}.u", std::process::id()));
    let output = llm()
        .args(["import", "--in"])
        .arg(&source)
        .args(["--out"])
        .arg(&container)
        .output()
        .expect("进程可派生");
    assert_eq!(
        output.status.code(),
        Some(0),
        "合法 safetensors 下 `import` 必须成功：{:?}",
        String::from_utf8_lossy(&output.stderr)
    );
    // 导入要自证：报告里的体积比与无损是**落盘后重新读出来**的账，而不是"我写成功了"。
    let stdout = String::from_utf8_lossy(&output.stdout);
    for field in ["体积比", "lossless"] {
        assert!(
            stdout.contains(field),
            "`import` 必须报出 `{field}`：{stdout}"
        );
    }
    container
}

#[test]
fn import_then_inspect_reports_live_metrics() {
    let container = imported_container("inspect");
    let output = llm()
        .args(["inspect", "--in"])
        .arg(&container)
        .output()
        .expect("进程可派生");
    assert_eq!(
        output.status.code(),
        Some(0),
        "合法容器下 `inspect` 必须成功：{:?}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    for field in [
        "条目数",
        "参数总数",
        "V_artifact",
        "V_FP16",
        "ratio",
        "lossless",
        "REF 命中",
    ] {
        assert!(
            stdout.contains(field),
            "`inspect` 必须报出 `{field}`：{stdout}"
        );
    }
    assert!(
        stdout.contains("blk.0.attn_q.weight"),
        "条目清单要列出名字：{stdout}"
    );
    assert!(stdout.contains("f32"), "条目清单要列出 dtype：{stdout}");
}

#[test]
fn check_decodes_every_tensor_of_a_valid_container() {
    let container = imported_container("check");
    let output = llm()
        .args(["check", "--in"])
        .arg(&container)
        .output()
        .expect("进程可派生");
    assert_eq!(
        output.status.code(),
        Some(0),
        "合法容器下 `check` 必须成功：{:?}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("2 tensors decoded"),
        "解码口径要报出条目数：{stdout}"
    );
}

#[test]
fn settle_writes_the_same_ledger_to_stdout_and_to_a_file() {
    let container = imported_container("settle");
    let output = llm()
        .args(["settle", "--in"])
        .arg(&container)
        .output()
        .expect("进程可派生");
    assert_eq!(
        output.status.code(),
        Some(0),
        "合法容器下 `settle` 必须成功：{:?}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    for field in [
        "\"schema\"",
        "\"file_len\"",
        "blk.0.attn_q.weight",
        "\"is_lossless\"",
        "\"codec_reason\"",
        "\"decision\"",
    ] {
        assert!(stdout.contains(field), "账本必须报出 `{field}`：{stdout}");
    }

    // `--out` 只换落点，不换文本：写出来的字节要与 stdout 上一模一样。
    let ledger = container.with_extension("u.json");
    let output = llm()
        .args(["settle", "--in"])
        .arg(&container)
        .args(["--out"])
        .arg(&ledger)
        .output()
        .expect("进程可派生");
    assert_eq!(output.status.code(), Some(0));
    let written = std::fs::read_to_string(&ledger).expect("账本可回读");
    assert_eq!(written, stdout, "写盘与 stdout 必须同一份文本");
    let stdout = String::from_utf8_lossy(&output.stdout);
    for field in ["条目数", "体积比", "lossless"] {
        assert!(stdout.contains(field), "落盘后要报 `{field}`：{stdout}");
    }
}

#[test]
fn audit_accepts_the_ledger_it_wrote_and_rejects_a_tampered_cell() {
    let container = imported_container("audit");
    let ledger = container.with_extension("u.json");
    let output = llm()
        .args(["settle", "--in"])
        .arg(&container)
        .args(["--out"])
        .arg(&ledger)
        .output()
        .expect("进程可派生");
    assert_eq!(
        output.status.code(),
        Some(0),
        "先得有一份账本：{:?}",
        String::from_utf8_lossy(&output.stderr)
    );

    // 同一份字节 + 同一份账：逐格复核通过，报告说出结论而不只是不报错。
    let output = llm()
        .args(["audit", "--in"])
        .arg(&container)
        .args(["--ledger"])
        .arg(&ledger)
        .output()
        .expect("进程可派生");
    assert_eq!(
        output.status.code(),
        Some(0),
        "自己的账本必须复核通过：{:?}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    for field in ["条目数", "体积比", "lossless", "一致"] {
        assert!(stdout.contains(field), "复核要报 `{field}`：{stdout}");
    }

    // 改一格体积比：账本与字节不再同源，必须以 invalid=2 失败并点名那一格。
    let text = std::fs::read_to_string(&ledger).expect("账本可回读");
    let forged = text.replacen("\"volume_ratio\": ", "\"volume_ratio\": 9", 1);
    assert_ne!(forged, text, "篡改必须真的改到一格：{text}");
    std::fs::write(&ledger, forged).expect("写入篡改版");
    let output = llm()
        .args(["audit", "--in"])
        .arg(&container)
        .args(["--ledger"])
        .arg(&ledger)
        .output()
        .expect("进程可派生");
    assert_eq!(output.status.code(), Some(2), "账本不符是 invalid=2");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("volume_ratio"),
        "要能定位到那一格：{stderr}"
    );
}

#[test]
fn check_rejects_a_truncated_or_tampered_container() {
    let container = imported_container("tampered");
    let mut bytes = std::fs::read(&container).expect("读回容器");

    // 截断尾部：全局 CRC 与末段帧被砍掉，打开期自证失败。
    let truncated = container.with_extension("u.truncated");
    std::fs::write(&truncated, &bytes[..bytes.len() - 16]).expect("写入截断版");
    let output = llm()
        .args(["check", "--in"])
        .arg(&truncated)
        .output()
        .expect("进程可派生");
    assert_ne!(
        output.status.code(),
        Some(0),
        "截断容器必须以非零退出码失败"
    );

    // 翻转正文一个比特：段级 / 全局 CRC 不符，解码路径不会走到。
    let tampered = container.with_extension("u.tampered");
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0xff;
    std::fs::write(&tampered, &bytes).expect("写入篡改版");
    let output = llm()
        .args(["check", "--in"])
        .arg(&tampered)
        .output()
        .expect("进程可派生");
    assert_ne!(
        output.status.code(),
        Some(0),
        "篡改容器必须以非零退出码失败"
    );
}

#[test]
fn archive_commands_reject_missing_or_unknown_grammar() {
    // `import` 缺 `--out`：两个选项都必需，不猜默认输出路径。
    let source = good_weights("grammar").0;
    let output = llm()
        .args(["import", "--in"])
        .arg(&source)
        .output()
        .expect("进程可派生");
    assert_eq!(output.status.code(), Some(2), "缺 `--out` 是 invalid=2");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("--out"), "错误要点名缺哪个选项：{stderr}");

    // `inspect` 缺 `--in`；`check` 给了未知选项；多余参数同样是文法错误。
    let output = llm().args(["inspect"]).output().expect("进程可派生");
    assert_eq!(output.status.code(), Some(2), "缺 `--in` 是 invalid=2");

    let output = llm()
        .args(["check", "--in", "model.u", "--op", "gemv_i4"])
        .output()
        .expect("进程可派生");
    assert_eq!(output.status.code(), Some(2), "未知选项是 invalid=2");

    let output = llm()
        .args(["inspect", "--in", "model.u", "extra"])
        .output()
        .expect("进程可派生");
    assert_eq!(output.status.code(), Some(2), "多余参数是 invalid=2");
}

#[test]
fn audit_commands_are_dispatched_elsewhere_and_say_so() {
    for command in ["verify", "bench"] {
        let output = llm().arg(command).output().expect("进程可派生");
        assert_eq!(
            output.status.code(),
            Some(2),
            "`{command}` 在组装点不是可用命令"
        );
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("cutile"),
            "必须指出去哪里跑，而不是只说「不支持」：{stderr:?}"
        );
    }
}

#[test]
fn a_run_never_ends_in_a_panic() {
    // 这台机器上可能没有编译器、也没有设备：`run` 允许失败，但**只能**以诊断结束。
    // 退出码 101 是 Rust 的 panic 约定，出现即说明错误路径把结论变成了崩溃。
    let (path, _) = good_weights("run");
    let output = llm()
        .args(["run", "--weights"])
        .arg(&path)
        .output()
        .expect("进程可派生");
    let code = output.status.code().expect("被信号杀死也算违约");
    assert_ne!(
        code,
        101,
        "`run` 不许 panic：{:?}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        [0, 2, 3, 4, 5, 70].contains(&code),
        "退出码必须落在 Code::exit_code 的闭集里，实际 {code}"
    );
    if code == 0 {
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("层"), "成功路径要报出每层的结果：{stdout}");
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("✗"), "失败路径必须有诊断前缀：{stderr:?}");
        assert!(
            !stderr.trim_end().ends_with("✗"),
            "诊断前缀后面必须跟一句说明：{stderr:?}"
        );
    }
}
