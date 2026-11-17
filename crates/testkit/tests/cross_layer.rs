//! 跨层对拍：同一份权重字节，从容器（`weights`）→ 层声明（`nn`）→ 训练缓冲（`train`）三段走，
//! 每一段都必须说同一件事。
//!
//! 单层测试各自只在自己的坐标系里自洽：`weights` 只知道偏移与形状，`nn` 只知道
//! (in, out) 与算子名，`train` 只知道 `Var` 与缓冲长度。三段拼起来才会暴露"形状差一次转置""算
//! 子名在注册表里拼错""字节序反了"这类跨层错位——它们都能让三层各自的用例全绿。
//!
//! 数据是现造的 2×4 权重：手算参考用整数与 0.5 的精确组合，因此对拍容差取 f32 的
//! 「同尺度逐元素」档，而不是量化的余弦档。

use autograd::{Sizes, Store, Tape};
use nn::{Contract, Stack};
use tensor::{DType, Shape};
use llm_testkit::approx_eq;
use train::{Buffers, Data, Reference};
use weights::safetensors::Archive;

/// 夹具：两条条目，一条秩 2（会归因为线性层），一条秩 1（必须落进未覆盖清单）。
const NAMES: [&str; 2] = ["blk.0.attn_q.weight", "blk.0.norm.weight"];
/// 层声明把条目的方位后缀去掉：层名 + 后缀才是容器里的键。
const WEIGHT_SUFFIX: &str = ".weight";
const WEIGHT: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 0.5, -1.0, 0.0, 2.0];
const NORM: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
const INPUT: [f32; 4] = [1.0, -1.0, 2.0, 0.5];

/// 把 `f32` 切片编码成安全张量（little-endian）并拼出完整文件。
///
/// 逐标量重组是必须的：`f32` ↔ 4 字节是格式定义的串行映射，没有"整块 slice 直接当浮点用"
/// 的合法路径（`unsafe_code = deny` 关着 `transmute`）。
fn encode(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn decode(bytes: &[u8]) -> Vec<f32> {
    let (chunks, _) = bytes.as_chunks::<4>();
    chunks
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

fn fixture() -> Vec<u8> {
    let weight = encode(&WEIGHT);
    let norm = encode(&NORM);
    let header = format!(
        r#"{{"{w}": {{"dtype": "F32", "shape": [2, 4], "data_offsets": [0, {nw}]}},
            "{nm}": {{"dtype": "F32", "shape": [4], "data_offsets": [{nw}, {total}]}}}}"#,
        w = NAMES[0],
        nm = NAMES[1],
        nw = weight.len(),
        total = weight.len() + norm.len(),
    );
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&weight);
    bytes.extend_from_slice(&norm);
    bytes
}

#[test]
fn the_container_reports_the_shape_the_fixture_declared() {
    let bytes = fixture();
    let archive = Archive::parse(&bytes).expect("夹具必须是合法 safetensors");
    let entry = archive.header().get(NAMES[0]).expect("秩 2 条目");
    assert_eq!(entry.shape().dims(), &[2, 4], "形状是 (out, in) 的顺序");
    assert_eq!(entry.dtype(), DType::F32);
    assert_eq!(
        entry.len(),
        WEIGHT.len() * 4,
        "条目字节数由形状与精度推出（8 个 f32）"
    );
}

#[test]
fn the_layer_declaration_agrees_with_the_container() {
    let bytes = fixture();
    let archive = Archive::parse(&bytes).expect("解析");
    let stack = Stack::from_header(archive.header()).expect("头部可归因");
    assert_eq!(stack.layers().len(), 1, "只有秩 2 的 `.weight` 条目是投影");

    let layer = stack.layers()[0]
        .linear()
        .expect("秩 2 条目必须归因为线性层");
    let entry = archive
        .header()
        .get(&format!("{}{WEIGHT_SUFFIX}", stack.layers()[0].name()))
        .expect("同一条目");
    let dims = entry.shape().dims();
    assert_eq!(
        format!("{}{WEIGHT_SUFFIX}", layer.name()),
        NAMES[0],
        "层名是条目名去掉方位后缀：两个名字必须互为逆运算"
    );
    assert_eq!(
        (layer.in_features(), layer.out_features()),
        (dims[1], dims[0]),
        "容器存的是 (out, in)，层声明必须把两轴对调回来，否则整条链路差一次转置"
    );
    assert_eq!(
        layer.format().operator(),
        Some("gemv_f32"),
        "精度决定算子名：这个名字要与训练缓冲里 `compute` 的实参一致"
    );
    assert_eq!(
        stack.uncovered(),
        [NAMES[1]],
        "秩 1 的条目必须如实列进未覆盖清单，而不是被静默当成投影"
    );
}

#[test]
fn the_training_buffer_computes_the_scalar_reference_of_the_same_weights() {
    let bytes = fixture();
    let archive = Archive::parse(&bytes).expect("解析");
    let entry = archive.header().get(NAMES[0]).expect("秩 2 条目");
    let weight = decode(<Archive<'_> as weights::Reader>::read(&archive, entry).expect("取到字节"));

    // 三层共用同一份形状：容器 → 层声明 → 图。任何一处不一致，`set` 与 `compute` 会有一处报错。
    let stack = Stack::from_header(archive.header()).expect("归因");
    let layer = stack.layers()[0].linear().expect("线性层");
    let mut tape = Tape::new();
    let (x, w) = (tape.leaf(), tape.leaf());
    let y = tape.record("gemv_f32", &[x, w]).expect("记录");
    let mut sizes = Sizes::new(tape.vars());
    sizes.set(x, Shape::new([layer.in_features()]), DType::F32);
    sizes.set(
        w,
        Shape::new([layer.out_features(), layer.in_features()]),
        DType::F32,
    );
    sizes.set(y, Shape::new([layer.out_features()]), DType::F32);

    let mut store = Buffers::new(sizes, Reference);
    store.load(x, Data::F32(INPUT.to_vec())).expect("输入");
    store.load(w, Data::F32(weight.clone())).expect("权重");
    store.compute("gemv_f32", &[x, w], y).expect("算");
    let observed = store.value(y).expect("有值");
    let observed = observed.f32().expect("f32 视图");

    // 手算参考：不经过任何库代码，只按 (out, in) 的行主序点乘。
    let expected: Vec<f32> = (0..layer.out_features())
        .map(|row| {
            let offset = row * layer.in_features();
            INPUT
                .iter()
                .zip(&weight[offset..offset + INPUT.len()])
                .map(|(a, b)| a * b)
                .sum()
        })
        .collect();
    assert_eq!(observed.len(), expected.len());
    assert!(
        approx_eq(observed, &expected, 1e-6),
        "训练缓冲的 gemv 必须与标量参考一致：实测 {observed:?}，参考 {expected:?}"
    );
}

#[test]
fn a_weight_entry_that_is_not_a_projection_never_produces_a_layer() {
    let bytes = fixture();
    let archive = Archive::parse(&bytes).expect("解析");
    let stack = Stack::from_header(archive.header()).expect("归因");
    let names: Vec<&str> = stack.layers().iter().map(|layer| layer.name()).collect();
    assert!(
        !names.contains(&NAMES[1]),
        "秩 1 条目出现在层清单里：{names:?}"
    );
    assert!(
        stack.layers().iter().all(|layer| {
            archive
                .header()
                .get(&format!("{}{WEIGHT_SUFFIX}", layer.name()))
                .is_some_and(|entry| entry.shape().dims().len() == 2)
        }),
        "每一层都必须对回一条秩 2 条目"
    );
}
