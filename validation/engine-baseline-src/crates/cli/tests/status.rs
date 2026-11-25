//! `status` 的单元测试：最小实例装配、真实条目读取与下界行。
//!
//! 经 `src/status.rs` 的 `#[path]` 挂进 lib 单测（N12：`src/` 里不放测试）。

use super::*;

/// 写一个最小的 safetensors：两个矩阵条目 + 一个范数条目（故意留一条未覆盖）。
fn write_safetensors(name: &str) -> std::path::PathBuf {
    let json = r#"{"blk.0.attn_q.weight": {"dtype": "F32", "shape": [64, 32], "data_offsets": [0, 8192]},
                   "blk.0.attn_o.weight": {"dtype": "F32", "shape": [64, 64], "data_offsets": [8192, 24576]},
                   "blk.0.norm.weight": {"dtype": "F32", "shape": [64], "data_offsets": [24576, 24832]}}"#;
    let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(json.as_bytes());
    bytes.resize(bytes.len() + 24832, 0);
    let path = std::env::temp_dir().join(format!("{}-{name}", std::process::id()));
    std::fs::write(&path, &bytes).expect("临时文件可写");
    // 读回确认：某些环境里临时目录被并行进程清理过，空文件会让断言变得难以解释。
    let written = std::fs::read(&path).expect("临时文件可读");
    assert_eq!(written, bytes, "落盘内容必须与待写内容一致");
    path
}

#[test]
fn without_a_weights_file_the_model_is_the_declared_stack() {
    let (model, row) = load_model(None).expect("声明层合法");
    assert_eq!(model.layers().len(), 2);
    assert!(model.uncovered().is_empty());
    assert!(row.contains("只用声明层"), "{row}");
    let floor = model.floor();
    assert!(floor.total().read() > 2 * 8 * 1024 * 1024);
    assert!(floor.unpriced().is_empty(), "声明的 int4 层都能计价");
}

#[test]
fn a_real_file_supplies_the_layers_and_reports_what_it_could_not_use() {
    let path = write_safetensors("llm-status-test.safetensors");
    let (model, row) = load_model(Some(&path.display().to_string())).expect("文件合法");
    let names: Vec<&str> = model.layers().iter().map(|layer| layer.name()).collect();
    assert_eq!(names, ["blk.0.attn_q", "blk.0.attn_o"], "条目的偏移顺序");
    assert_eq!(model.uncovered(), ["blk.0.norm.weight"]);
    // 条目写的是 F32：层被列出来，而且由稠密算子计价——读的是权重本身的字节数，
    // 不是按 int4 折算出来的零头。
    let floor = model.floor();
    assert!(floor.unpriced().is_empty(), "{:?}", floor.unpriced());
    assert_eq!(
        floor.total(),
        nn::Bounds::of_op(nn::Linear::DENSE, 32, 64)
            .expect("稠密语义")
            .plus(nn::Bounds::of_op(nn::Linear::DENSE, 64, 64).expect("稠密语义")),
        "栈的总下界 = 两层稠密投影之和"
    );
    assert_eq!(
        floor.total().read(),
        ((64 * 32 + 32) + (64 * 64 + 64)) as u64 * 4,
        "读侧 = 权重 + 输入向量"
    );
    assert_eq!(
        floor.total().write(),
        (64 + 64) as u64 * 4,
        "写侧 = 两段输出"
    );
    assert!(row.contains("3 条条目"), "{row}");
    assert!(row.contains("24832 字节"), "{row}");
    assert!(
        row.contains("按需读 `blk.0.norm.weight` 256 字节成功"),
        "样例读的是最小的那条：{row}"
    );
    let header = row
        .split("只读头部 ")
        .nth(1)
        .and_then(|rest| rest.split(' ').next())
        .expect("行里要有头部读取量");
    let header: usize = header.parse().expect("头部读取量是数字");
    assert!(
        header < 1024,
        "数据段 24832 字节，但只读了 {header} 字节头部：{row}"
    );
    let _ = std::fs::remove_file(path);
}

#[test]
fn a_missing_file_is_an_io_error_not_a_panic() {
    let error = load_model(Some("/nonexistent/llm.safetensors")).expect_err("读不到");
    assert_eq!(error.code(), diagnostic::Code::Io);
    assert!(error.to_string().contains("打开文件"), "{error}");
}

#[test]
fn a_truncated_file_fails_before_any_layer_is_used() {
    let path = std::env::temp_dir().join(format!("{}-truncated.safetensors", std::process::id()));
    let json = r#"{"w.weight": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]}}"#;
    let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(json.as_bytes());
    bytes.resize(bytes.len() + 8, 0); // 头部说了 16 字节，只给了 8 字节
    std::fs::write(&path, &bytes).expect("临时文件可写");

    let error = load_model(Some(&path.display().to_string())).expect_err("被截断");
    assert_eq!(error.code(), diagnostic::Code::Invalid);
    assert!(error.to_string().contains("文件被截断"), "{error}");
    let _ = std::fs::remove_file(path);
}
