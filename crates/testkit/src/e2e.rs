//! 跨层端到端对拍：容器 → 层声明 → 训练缓冲，以及 PTQ 量化 → 融合反量化推理。
//!
//! 单层测试各自只在自己的坐标系里自洽；这里把真实字节串起来，暴露跨层错位与量化/反量化
//! 口径漂移。两段各自的说明与断言边界见对应小节的注释。

use crate::{Rng, approx_eq, cosine_similarity};
use autograd::{Sizes, Store, Tape};
use nn::{Contract, Stack};
use nn::{gemv_f32, gemv_i2, gemv_i4, gemv_i8};
use tensor::{DType, Shape, View};
use train::{Bufs, Data, Reference};
use weights::ptq::{self, I2, I4, I8};
use weights::safetensors::Archive;

// ---------------------------------------------------------------------------
// 跨层对拍：容器 → 层声明 → 训练缓冲
// ---------------------------------------------------------------------------
//
// 同一份权重字节，从容器（`weights`）→ 层声明（`nn`）→ 训练缓冲（`train`）三段走，
// 每一段都必须说同一件事。
//
// 单层测试各自只在自己的坐标系里自洽：`weights` 只知道偏移与形状，`nn` 只知道
// (in, out) 与算子名，`train` 只知道 `Var` 与缓冲长度。三段拼起来才会暴露"形状差一次转置""算
// 子名在注册表里拼错""字节序反了"这类跨层错位——它们都能让三层各自的用例全绿。
//
// 数据是现造的 2×4 权重：手算参考用整数与 0.5 的精确组合，因此对拍容差取 f32 的
// 「同尺度逐元素」档，而不是量化的余弦档。

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
    crate::safetensors_bytes(&header, &[(NAMES[0], weight), (NAMES[1], norm)])
}

#[cfg(test)]
crate::test_cases! {
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
        nn::Linear::operator_for(layer.format()),
        Some("gemv_f32"),
        "精度决定算子名：这个名字要与训练缓冲里 `compute` 的实参一致"
    );
    assert_eq!(
        stack.uncovered(),
        [NAMES[1]],
        "秩 1 的条目必须如实列进未覆盖清单，而不是被静默当成投影"
    );
}

fn the_training_buf_computes_the_scalar_reference_of_the_same_weights() {
    let bytes = fixture();
    let archive = Archive::parse(&bytes).expect("解析");
    let entry = archive.header().get(NAMES[0]).expect("秩 2 条目");
    let weight = decode(<Archive<'_> as weights::Reader>::read(&archive, entry).expect("取到字节"));

    // 三层共用同一份形状：容器 → 层声明 → 图。任何一处不一致，`set` 与 `compute` 会有一处报错。
    let stack = Stack::from_header(archive.header()).expect("归因");
    let layer = stack.layers()[0].linear().expect("线性层");
    let mut tape = Tape::new();
    let (x, w) = (tape.leaf(), tape.leaf());
    let y = tape.record("gemv_f32", &[w, x]).expect("记录");
    let mut sizes = Sizes::new(tape.vars());
    sizes.set(x, Shape::new([layer.in_features()]), DType::F32);
    sizes.set(
        w,
        Shape::new([layer.out_features(), layer.in_features()]),
        DType::F32,
    );
    sizes.set(y, Shape::new([layer.out_features()]), DType::F32);

    let mut store = Bufs::new(sizes, Reference);
    store.load(x, Data::F32(INPUT.to_vec())).expect("输入");
    store.load(w, Data::F32(weight.clone())).expect("权重");
    store.compute_step("gemv_f32", &[w, x], y).expect("算");
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
}

// ---------------------------------------------------------------------------
// PTQ 量化 → 融合反量化推理
// ---------------------------------------------------------------------------
//
// `weights::ptq` 把 f32 权重按组量化成打包码 + 每组件标度（ADR 0018：换算算法留在 `weights`，
// 离线算），`nn::gemv_i2/i4/i8` 的融合内核在同一个 kernel 内解包 + 点积。两层各自在自己的坐标
// 系里自洽，只有把真实字节串起来才能证明「量化」与「融合反量化」说的是同一件事——字节序、零点、
// 标度分组三处任一漂移都会让本测试红掉。
//
// 断言只看方向（余弦，尺度不变）：PTQ 是有界舍入，不要求逐元素相等；i2 只有三档码，阈值逐级放宽。

/// 合成基准形状：输入宽、输出行数、量化分组。gemv 标度按行分、行内按组，分组须是输入宽的公因数。
const M: usize = 64;
const N: usize = 4;
const GROUP: usize = 32;

/// 合成基准：f32 权重 → 逐行 PTQ 打包 → 融合反量化内核 → logits，方向必须与 f32 参考一致。
#[test]
fn ptq_quantized_gemv_keeps_cosine_with_the_f32_reference() {
    let mut rng = Rng::new(2026_1118);
    let x: Vec<f32> = (0..M).map(|_| rng.signed_unit()).collect();
    let weights: Vec<f32> = (0..N * M).map(|_| rng.signed_unit()).collect();

    let mut reference = vec![0.0f32; N];
    gemv_f32::forward(View::one(&x), &weights, &mut reference).expect("稠密参考可算");

    // i2 只有三档码，余弦阈值逐级放宽；i4/i8 随位宽收紧。
    assert_quantized("i8", I8, gemv_i8::forward, 0.999, &x, &weights, &reference);
    assert_quantized("i4", I4, gemv_i4::forward, 0.98, &x, &weights, &reference);
    assert_quantized("i2", I2, gemv_i2::forward, 0.80, &x, &weights, &reference);
}

/// 逐行量化再按 `w[nm]` 布局拼接，喂给融合内核对拍。
///
/// gemv 的标度按行分、行内按组，所以一行必须独立量化（跨行共享标度会让内核语义错位）。
/// 误差类型写成泛型 `E` 而不点名 `diagnostic`：testkit 的测试不直接依赖诊断层，退耦得更干净。
fn assert_quantized<E: std::fmt::Debug>(
    name: &str,
    scheme: ptq::Scheme,
    run: impl FnOnce(View<'_, f32>, &[u8], View<'_, f32>, &mut [f32]) -> Result<(), E>,
    min_cosine: f32,
    x: &[f32],
    weights: &[f32],
    reference: &[f32],
) {
    // 容量按 `Scheme::packed_len` 一次算清：字节数算式只留 `Scheme` 那一份（`cpb` 已收成 `bits`）。
    let mut packed = Vec::with_capacity(scheme.packed_len(N * M));
    let mut scales = Vec::with_capacity(N * M.div_ceil(GROUP));
    for row in 0..N {
        let (p, s) =
            ptq::quantize(scheme, GROUP, &weights[row * M..(row + 1) * M]).expect("行内 PTQ 可算");
        packed.extend(p);
        scales.extend(s);
    }

    let mut out = vec![0.0f32; N];
    run(View::one(x), &packed, View::one(&scales), &mut out).expect("融合内核可算");
    let cosine = cosine_similarity(reference, &out).expect("两条向量非零");
    assert!(
        cosine >= min_cosine,
        "{name} 位 PTQ 融合反量化 logits 余弦 {cosine:.5} < {min_cosine}"
    );
}

use weights::chain::{Chain, Ruling, Rung};
use weights::{Archive as WeightArchive, Segment, Stored, Wire, Writer};

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
    let (packed, scales) = weights::quantize(weights::I4, GROUP, w).expect("量化");
    let mut out = vec![0.0f32; w.len()];
    weights::dequantize(weights::I4, &packed, &scales, GROUP, &mut out).expect("反量化");
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

#[cfg(test)]
crate::test_cases! {
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
    assert!(
        observed <= bound.error(),
        "实测 {observed:.6e} 超出网络级上界 {:.6e}",
        bound.error()
    );
    assert!(bound.error().is_finite() && bound.error() > observed);
    assert_eq!(chain.uncovered(), 0);
}

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
    assert!(
        observed <= bound,
        "实测 {observed:.6e} 超出整批界 {bound:.6e}"
    );
}

fn a_bitwise_artifact_round_trip_completes_the_lossless_ruling() {
    let w1 = scaled(64, 6, 0.25);
    let w2 = scaled(64, 7, 0.25);
    let mut writer = Writer::new();
    for (name, values) in [("layer.0.w", &w1), ("layer.1.w", &w2)] {
        writer.push(segment(name, values)).expect("段名与载荷自洽");
    }
    let bytes = writer.finish(6).expect("落盘");
    let archive = WeightArchive::parse(&bytes).expect("解析");
    let decoded: Vec<Vec<f32>> = archive
        .segments()
        .iter()
        .map(|stored| decode_stored(&archive, stored))
        .collect();
    let bitwise = decoded == [w1.clone(), w2.clone()];
    assert!(bitwise, "无损容器的往返必须逐位相等");
    let chain = Chain::new(
        frobenius(&w1),
        vec![rung(&w1, &w1, 1.0, 0.0), rung(&w2, &w2, 1.0, 0.0)],
    )
    .expect("链");
    assert_eq!(
        chain.bracket().expect("界").ruling(bitwise),
        Ruling::Lossless
    );
}
}

/// 一段 F32 权重：载荷就是小端字面量，没有第二套编码。
fn segment(name: &str, values: &[f32]) -> Segment {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let shape = vec![values.len() as u64];
    Segment::new(name, Wire::F32, shape, None, bytes).expect("段")
}

/// 逐段解码；`F32` 段的字节数必须是 4 的整数倍。
fn decode_stored(archive: &WeightArchive<'_>, stored: &Stored) -> Vec<f32> {
    let raw = archive.load(stored).expect("解码");
    let (words, rest) = raw.as_chunks::<4>();
    assert!(rest.is_empty(), "F32 段的载荷长度必须是 4 的倍数");
    words.iter().map(|word| f32::from_le_bytes(*word)).collect()
}

// ---- 层边界与装配契约 ----

use std::collections::{BTreeMap, BTreeSet};

use crate::architecture::{
    DOC, backticks, contract, rust_files, source_text, src_root, workspace, workspace_root,
};
use crate::ownership::{is_public_definition, public_item_name};

/// 允许用 `String` 做**消息片段**的文件（不是"允许用 `String` 报错"）。
pub(crate) const MESSAGE_FRAGMENTS: &[&str] = &[];

/// 错误边界不得退回字符串（`docs/ARCHITECTURE.md` 第三节）：凡是被 L7 直接调用的层——数据层
/// （`prof`）、内核层（`cutile`，含 `report`）——失败类型必须是
/// `diagnostic::Error`，否则类别与退出码就失去了来源。
#[test]
pub(crate) fn the_error_boundary_does_not_fall_back_to_strings() {
    let contract = contract();
    let mut paths = Vec::new();
    for member in contract.layers.keys() {
        let root = src_root(member);
        let files = rust_files(&root);
        assert!(
            !files.is_empty(),
            "层表成员 `{member}` 在 {} 下一个源文件都没有：扫描集由层表派生，这里必须扫得到东西，\
         否则合并/删除成员会让本测试静默通过",
            root.display()
        );
        paths.extend(files.iter().cloned());
    }
    let mut offences = Vec::new();
    for path in paths {
        if MESSAGE_FRAGMENTS
            .iter()
            .any(|exempt| path == std::path::Path::new(exempt))
        {
            continue;
        }
        let text = std::fs::read_to_string(&path).expect("源文件可读");
        for (idx, line) in text.lines().enumerate() {
            if line.trim_start().starts_with("//") {
                continue;
            }
            // 只认错误位是 `String` 的形态；`Result<Vec<String>, Error>` 这类是合法的。
            if line.contains("Result<") && line.contains(", String>") {
                offences.push(format!("{}:{}", path.display(), idx + 1));
            }
        }
    }
    assert!(
        offences.is_empty(),
        "这些位置仍以 `String` 报错，应改为 `diagnostic::Error`：{offences:?}"
    );
}

/// 厂商名基线：机制层已无按厂商命名的公开入口（`cutile` 的 17 个 `*_program` 收成了
/// `Backend` + `Mode` 一个泛型入口），因此这里保持为空——再有人加按厂商起名的公开项，门禁
/// 直接报越界而不是记进基线。
pub(crate) const VENDOR_PUBLIC_BASELINE: &[(&str, &str)] = &[];

/// 允许公开项带厂商名的成员：后端实现层，`cutile` 的 `Backend` 缝由它填。
/// `prof` 收设备档案与 FFI 后端岛；`shard` 承接过渡执行面；`device` crate 已删除。
pub(crate) const VENDOR_OWNERS: [&str; 2] = ["prof", "shard"];

/// 厂商名清单；判据是"公开项以 `厂商名_` 起头"。
pub(crate) const VENDORS: [&str; 8] = [
    "ascend", "cuda", "hip", "metal", "musa", "rocm", "sycl", "vulkan",
];

/// 越界岛白名单：`docs/adr/0035-unsafe-ffi-boundary.md` §6 的表格，路径 → `allow(unsafe_code)` 条数。
pub(crate) const UNSAFE_ISLANDS: &[(&str, usize)] = &[
    ("crates/prof/src/metal/runtime.rs", 1),
    ("crates/prof/src/opencl.rs", 1),
    // `fast` 段为 `fast.rs`（i8 包装 4）与 `pool.rs`（常驻池 3），总数不变。
    ("crates/nn/src/fast.rs", 4),
    ("crates/nn/src/pool.rs", 3),
    ("crates/prof/src/runtime.rs", 1),
    ("crates/prof/src/gemm.rs", 1),
    ("crates/storage/src/mmap.rs", 1),
];

/// 文档正文点名的 `crate::名字` 必须在源码里真的存在。
#[test]
pub(crate) fn the_document_names_only_existing_items() {
    let (members, _) = workspace();
    let mut missing = BTreeSet::new();
    for span in backticks(DOC) {
        let Some((crate_name, name)) = reference(&span) else {
            continue;
        };
        if members.contains(crate_name) && !declares_any(crate_name, name) {
            missing.insert(format!("{crate_name}::{name}"));
        }
    }
    assert!(
        missing.is_empty(),
        "文档点名的这些路径在源码里不存在（改名或删模块后没有同步）：{missing:?}"
    );
}

/// 厂商名不得出现在机制层的公开面上：`cutile` 只持 `Backend` 缝，厂商实现归 `shard` 执行面。
#[test]
pub(crate) fn the_mechanism_layer_exposes_no_vendor_names() {
    let (members, _) = workspace();
    let mut found = BTreeSet::new();
    for member in members {
        if VENDOR_OWNERS.contains(&member.as_str()) {
            continue;
        }
        let root = src_root(member);
        for path in rust_files(&root).iter() {
            let Some(text) = source_text(path) else {
                continue;
            };
            for name in text.lines().filter_map(public_item_name) {
                if VENDORS.iter().any(|vendor| vendor_prefix(name, vendor)) {
                    found.insert((member.clone(), name.to_string()));
                }
            }
        }
    }
    let baseline = VENDOR_PUBLIC_BASELINE
        .iter()
        .map(|(crate_name, name)| ((*crate_name).to_string(), (*name).to_string()))
        .collect::<BTreeSet<_>>();
    let added = found.difference(&baseline).collect::<Vec<_>>();
    let stale = baseline.difference(&found).collect::<Vec<_>>();
    assert!(
        added.is_empty(),
        "机制层的公开项出现厂商名（厂商名只该出现在 `{VENDOR_OWNERS:?}`）：{added:?}"
    );
    assert!(
        stale.is_empty(),
        "厂商名基线过期（这些项已经不在了，从 `VENDOR_PUBLIC_BASELINE` 删掉）：{stale:?}"
    );
}

/// `allow(unsafe_code)` 只允许出现在 ADR 0035 §6 列举的越界岛里。
#[test]
pub(crate) fn unsafe_islands_are_the_ones_the_adr_lists() {
    let mut found = BTreeMap::new();
    for member in &workspace().0 {
        let root = src_root(member);
        for path in rust_files(&root).iter() {
            let Some(text) = source_text(path) else {
                continue;
            };
            let count = text
                .lines()
                .filter(|line| {
                    let trimmed = line.trim_start();
                    // 模块级 `#![allow]` 与条目级 `#[allow]` 都算；文档注释里的提法不算。
                    trimmed.starts_with('#') && trimmed.contains("allow(unsafe_code)")
                })
                .count();
            if count > 0 {
                let relative = path
                    .strip_prefix(workspace_root())
                    .expect("扫描路径在仓内")
                    .display()
                    .to_string();
                found.insert(relative, count);
            }
        }
    }
    let expected = UNSAFE_ISLANDS
        .iter()
        .map(|(path, count)| ((*path).to_string(), *count))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(
        found, expected,
        "越界岛与 ADR 0035 §6 的白名单不一致：多一处是新岛（先改 ADR 说清不变量），少一处是基线过期"
    );
}

/// 设备后端必须在**产品路径**上装配，不能只活在测试里。
#[test]
pub(crate) fn the_device_backends_are_installed_on_the_product_path() {
    let mut entries = 0;
    let mut missing = Vec::new();
    for path in rust_files(&workspace_root().join("crates/cli/src")).iter() {
        let Ok(body) = std::fs::read_to_string(path) else {
            continue;
        };
        // 库入口：`pub fn entry(`。二进制入口：`fn main(` 且不转调库入口（转调的由库那处装着）。
        let library = body.contains("pub fn entry(");
        let standalone = body.contains("fn main(") && !body.contains("llm_cli::entry");
        if !library && !standalone {
            continue;
        }
        entries += 1;
        if !body.contains("backends::install()") {
            missing.push(path.display().to_string());
        }
    }
    assert!(
        entries >= 2,
        "只扫到 {entries} 个产品入口：`llm` 与 `cutile` 两个二进制各自要有一条产品路径"
    );
    assert!(
        missing.is_empty(),
        "这些产品入口没有装配设备后端（`shard::backends::install()`）：{missing:?}——\
     装配点只留在 `tests/` 时，用户拿到的是“本机没有该后端”这条错误信息"
    );
}

fn reference(span: &str) -> Option<(&str, &str)> {
    let (head, tail) = span.split_once("::")?;
    let is_crate = !head.is_empty()
        && head
            .chars()
            .all(|value| value.is_ascii_lowercase() || value.is_ascii_digit() || value == '_');
    if !is_crate {
        return None;
    }
    let end = tail
        .find(|value: char| !(value.is_alphanumeric() || value == '_'))
        .unwrap_or(tail.len());
    let name = &tail[..end];
    (!name.is_empty()).then_some((head, name))
}

fn declares_any(crate_name: &str, name: &str) -> bool {
    const KINDS: [&str; 8] = [
        "struct", "enum", "trait", "type", "fn", "const", "union", "mod",
    ];
    let root = src_root(crate_name);
    rust_files(&root).iter().any(|path| {
        source_text(path).is_some_and(|text| {
            text.lines().any(|line| {
                is_public_definition(line, name, &KINDS) || is_module_or_macro(line, name)
            })
        })
    })
}

fn is_module_or_macro(line: &str, name: &str) -> bool {
    let trimmed = line.trim_start();
    let rest = trimmed
        .strip_prefix("pub(crate) ")
        .or_else(|| trimmed.strip_prefix("pub "))
        .unwrap_or(trimmed);
    ["mod ", "macro_rules! "].iter().any(|prefix| {
        rest.strip_prefix(prefix)
            .is_some_and(|tail| ident_is(tail.trim_start(), name))
    })
}

fn ident_is(text: &str, name: &str) -> bool {
    text.strip_prefix(name).is_some_and(|after| {
        after
            .chars()
            .next()
            .is_none_or(|next| !(next.is_alphanumeric() || next == '_'))
    })
}

fn vendor_prefix(name: &str, vendor: &str) -> bool {
    name.strip_prefix(vendor)
        .is_some_and(|after| after.starts_with('_'))
}
