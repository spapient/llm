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

#[test]
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
    assert!(
        observed <= bound.error(),
        "实测 {observed:.6e} 超出网络级上界 {:.6e}",
        bound.error()
    );
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
    assert!(
        observed <= bound,
        "实测 {observed:.6e} 超出整批界 {bound:.6e}"
    );
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
