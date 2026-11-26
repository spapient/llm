//! 端到端：容器与台账这条链——`import` 写出容器、`inspect`/`check` 读回、
//! `settle` 与 `audit` 在台账上互相校验。
//!
//! 与 [`cli`](../cli/idx.html) 目标同一套夹具（`support.rs`），分文件只是为了让每个端到端
//! 目标保持在一屏之内；两者跑的仍是同一个真的 `llm` 二进制。

#[path = "support.rs"]
mod support;

use std::path::PathBuf;
use support::{good_weights, llm};

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

testkit::test_cases! {
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
}
