//! 端到端契约与跨层对拍：公开数据树→命令→CLI，以及真实权重→L3 事实。

use crate::architecture::{source_text, workspace_root};
use llm_cli::commands::Commands;
use llm_cli::run;

/// 独立“使用者”装配命令；不借用 `Commands` 的任何私有转换。
fn command_for(id: &str, dump: Option<&str>) -> Commands {
    match dump {
        Some(name) => Commands::View {
            id: id.to_string(),
            dims: Vec::new(),
            elem: None,
            dump: Some(name.to_string()),
        },
        None => Commands::Info { id: id.to_string() },
    }
}

/// 与外部用户同一路径：命令字符串经 clap 解析，再交给运行入口。
fn command_from(args: &[&str]) -> Result<String, String> {
    use clap::Parser;
    let cli = llm_cli::Cli::try_parse_from(args).map_err(|error| error.to_string())?;
    run(cli.command).map_err(|error| error.to_string())
}

/// `ops` 的示例 JSON 保留一个数值端口和两个字节端口，作为独立使用者的数据树。
fn ops_json() -> serde_json::Value {
    serde_json::json!({
        "version": 1,
        "id": "ops-demo",
        "meta": {"dims": [1], "total": 1, "desc": "demo"},
        "covers": {"full": {"name": "full", "elems": 1}},
        "extents": [],
        "items": {"main": {"elems": 1, "start": 0, "delta": [1]}},
        "rows": [{
            "item": "main", "lo": [0], "hi": [1], "elems": 1, "start": 0, "cover": "full",
            "tile": {"dense": {"dims": [1]}}, "elem": {"num": 1.5, "words": [1, 2]}
        }],
        "bindings": {"last": {"item": "main", "elem": {"num": 1.5, "words": [1, 2]}}}
    })
}

/// 与外部用户同一路径：JSON 进入公开数据树，再交给 CLI 入口。
fn run_tree(id: &str, value: &serde_json::Value, dump: Option<&str>) -> Result<String, String> {
    let view = llm_tree::from_value(value).map_err(|error| error.to_string())?;
    run(command_for(id, dump), &view).map_err(|error| error.to_string())
}

/// 一行数字的容差对拍，不把浮点值放进字符串相等断言。
fn close_line(actual: &str, expected: &str, tol: f64, label: &str) {
    let parse = |line: &str| -> Result<Vec<f64>, String> {
        line.split_whitespace()
            .map(|word| word.parse::<f64>().map_err(|error| error.to_string()))
            .collect()
    };
    let left = parse(actual).unwrap_or_else(|error| panic!("{label} 左值不是数字：{error}"));
    let right = parse(expected).unwrap_or_else(|error| panic!("{label} 右值不是数字：{error}"));
    crate::compare::assert_close(&left, &right, tol, 0.0);
}

#[test]
fn ops_default_info_and_dump_use_the_same_public_binding() {
    let info = run_tree("ops", &ops_json(), None).expect("ops info 应成功");
    assert!(info.contains("last: [\"main\",\"num\"]"));
    let dump = run_tree("ops", &ops_json(), Some("num")).expect("ops dump 应成功");
    close_line(&dump, "1.5\n", 0.0, "ops 数值");
}

#[test]
fn tree_source_resolves_meta_dims_without_private_reader_hooks() {
    let mut value = ops_json();
    value["id"] = serde_json::Value::String("tree".into());
    value["covers"]["full"]["elem"] = serde_json::Value::String("num".into());
    let info = run_tree("tree", &value, None).expect("tree info 应成功");
    assert!(info.contains("dims: [1]"));
    assert!(info.contains("total: 1"));
    let dump = run_tree("tree", &value, Some("num")).expect("tree dump 应成功");
    close_line(&dump, "1.5\n", 0.0, "tree 数值");
}

#[test]
fn cli_parser_and_runner_reject_an_unknown_port_with_useful_context() {
    let error = command_from(&["llm", "view", "ops", "--dump", "missing"])
        .expect_err("未知端口必须拒绝");
    assert!(error.contains("端口") || error.contains("数据端口"), "{error}");
    assert!(error.contains("ops"), "{error}");
}

#[test]
fn command_construction_is_name_safe_for_missing_operands() {
    let command = command_for("tensor", None);
    assert!(matches!(command, Commands::Info { .. }));
    let error = run(command).expect_err("不存在的公开输入必须报错");
    assert!(error.contains("tensor"), "{error}");
}

/// 生产侧文档示例也走同一公开命令；数值部分仍用浮点对拍。
#[test]
fn documented_public_examples_remain_valid() {
    let path = workspace_root().join("crates/cli/src/lib.rs");
    let source = source_text(&path).expect("CLI 源码可读");
    let mut block = String::new();
    for line in source.lines() {
        if line.contains("let mut cmd") {
            block.push_str(line.trim_start_matches("//! "));
            block.push('\n');
        }
    }
    assert!(block.contains("Commands::Info"), "文档示例必须覆盖公开命令");
    let info = command_from(&["llm", "info", "ops"]).expect("文档 info 示例应成功");
    assert!(info.contains("id: ops"));
}

use weights::chain::{Chain, Ruling, Rung};
use weights::{Archive, I4, Segment, Stored, Wire, Writer, dequantize, quantize};

/// 量化分组：32 元素一组、一个标度，与 `nn`/`serve` 的 4bit 权重口径一致。
const GROUP: usize = 32;

/// 确定性样本：`sample` 落在 `[-1, 1)`，乘一个小尺度让 `ρ < 1`。
fn scaled(len: usize, salt: u32, scale: f32) -> Vec<f32> {
    crate::sample(len, salt)
        .iter()
        .map(|value| value * scale)
        .collect()
}

/// `‖W‖_F`：`‖W‖₂ ≤ ‖W‖_F` 对任何矩阵成立，拿它当 `ρ` 无需先分解。
fn frobenius(w: &[f32]) -> f64 {
    w.iter()
        .map(|&v| f64::from(v) * f64::from(v))
        .sum::<f64>()
        .sqrt()
}

/// 两列之差的欧氏范数：同一算式既当 `ε`，也当两条输出的实测差。
fn distance(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&p, &q)| f64::from(p - q).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// 行主序 `[rows, k] × [n, k]ᵀ`：复用 `serve`/`train` 的 TN 分块内核。
fn forward(x: &[f32], w: &[f32], rows: usize, n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * n];
    nn::matmul::forward_tn_tiled(
        tensor::View::matrix(rows, k, x).expect("x 二维"),
        tensor::View::matrix(n, k, w).expect("w 二维"),
        &mut out,
    )
    .expect("几何由用例自己保证");
    out
}

/// 两层 ReLU 网络；第二层权重由调用方给（可以是量化后的那份）。
fn two_layer(x: &[f32], w1: &[f32], w2: &[f32], rows: usize, k: usize, n: usize) -> Vec<f32> {
    let hidden: Vec<f32> = forward(x, w1, rows, n, k)
        .into_iter()
        .map(|value| value.max(0.0))
        .collect();
    forward(&hidden, w2, rows, n, n)
}

/// 4bit 逐组量化后再解回的权重：`ε` 量的是将要使用的产物。
fn quantized(w: &[f32]) -> Vec<f32> {
    let (packed, scales) = quantize(I4, GROUP, w).expect("量化");
    let mut out = vec![0.0f32; w.len()];
    dequantize(I4, &packed, &scales, GROUP, &mut out).expect("反量化");
    out
}

/// 两版权重 → 一段 L3 事实。
fn rung(w: &[f32], w_hat: &[f32], lipschitz: f64, offset: f64) -> Rung {
    Rung::new(
        frobenius(w),
        frobenius(w_hat),
        distance(w, w_hat),
        lipschitz,
        offset,
    )
    .expect("有限样本")
}

#[test]
fn a_quantized_layer_stays_inside_the_l3_bound() {
    let (k, n, rows) = (32usize, 32usize, 1usize);
    let x = scaled(rows * k, 1, 1.0);
    let w1 = scaled(n * k, 2, 0.25);
    let w2 = scaled(n * n, 3, 0.25);
    let w2_hat = quantized(&w2);
    let exact = two_layer(&x, &w1, &w2, rows, k, n);
    let compressed = two_layer(&x, &w1, &w2_hat, rows, k, n);
    let relu = Rung::new(1.0, 1.0, 0.0, 1.0, 0.0).expect("ReLU 段");
    let chain = Chain::new(
        frobenius(&x),
        vec![rung(&w1, &w1, 1.0, 0.0), relu, rung(&w2, &w2_hat, 1.0, 0.0)],
    )
    .expect("链");
    let bound = chain.bracket().expect("界");
    let observed = distance(&exact, &compressed);
    assert!(observed > 0.0, "量化没造成任何差异，这条用例什么也没测到");
    assert!(observed <= bound.error(), "实测 {observed:.6e} 超出网络级上界 {:.6e}", bound.error());
    assert!(bound.error().is_finite() && bound.error() > observed);
    assert_eq!(chain.uncovered(), 0);
}

#[test]
fn a_batch_scales_with_the_square_root_of_its_rows() {
    let (k, n, rows) = (16usize, 16usize, 4usize);
    let x = scaled(rows * k, 4, 1.0);
    let w = scaled(n * k, 5, 0.25);
    let w_hat = quantized(&w);
    let exact = forward(&x, &w, rows, n, k);
    let compressed = forward(&x, &w_hat, rows, n, k);
    let widest = (0..rows)
        .map(|row| frobenius(&x[row * k..(row + 1) * k]))
        .fold(0.0f64, f64::max);
    let chain = Chain::new(widest, vec![rung(&w, &w_hat, 1.0, 0.0)]).expect("链");
    let bound = chain.bracket().expect("界").error() * (rows as f64).sqrt();
    let observed = distance(&exact, &compressed);
    assert!(observed > 0.0, "量化没造成任何差异，这条用例什么也没测到");
    assert!(observed <= bound, "实测 {observed:.6e} 超出整批界 {bound:.6e}");
}

#[test]
fn a_bitwise_artifact_round_trip_completes_the_lossless_ruling() {
    let w1 = scaled(64, 6, 0.25);
    let w2 = scaled(64, 7, 0.25);
    let mut writer = Writer::new();
    for (name, values) in [("layer.0.w", &w1), ("layer.1.w", &w2)] {
        writer.push(segment(name, values)).expect("段名与载荷自洽");
    }
    let bytes = writer.finish(6).expect("落盘");
    let archive = Archive::parse(&bytes).expect("解析");
    let decoded: Vec<Vec<f32>> = archive
        .segments()
        .iter()
        .map(|stored| decode(&archive, stored))
        .collect();
    let bitwise = decoded == [w1.clone(), w2.clone()];
    assert!(bitwise, "无损容器的往返必须逐位相等");
    let chain = Chain::new(
        frobenius(&w1),
        vec![rung(&w1, &w1, 1.0, 0.0), rung(&w2, &w2, 1.0, 0.0)],
    )
    .expect("链");
    assert_eq!(chain.bracket().expect("界").ruling(bitwise), Ruling::Lossless);
}

/// 一段 F32 权重：载荷就是小端字面量，没有第二套编码。
fn segment(name: &str, values: &[f32]) -> Segment {
    let bytes: Vec<u8> = values.iter().flat_map(|value| value.to_le_bytes()).collect();
    let shape = vec![values.len() as u64];
    Segment::new(name, Wire::F32, shape, None, bytes).expect("段")
}

/// 逐段解码；`F32` 段的字节数必须是 4 的整数倍。
fn decode(archive: &Archive<'_>, stored: &Stored) -> Vec<f32> {
    let raw = archive.load(stored).expect("解码");
    let (words, rest) = raw.as_chunks::<4>();
    assert!(rest.is_empty(), "F32 段的载荷长度必须是 4 的倍数");
    words.iter().map(|word| f32::from_le_bytes(*word)).collect()
}
