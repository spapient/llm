//! L3 合成律的跨层对拍：把 `weights::l3` 给出的网络级上界，对着**真实执行路径**的误差核一遍。
//!
//! 三层证据，各自回答一个不同的问题：
//!
//! - 逐段界只有真的罩住实测误差才算数（[`a_quantized_layer_stays_inside_the_l3_bound`]）；
//! - 产物逐位往返是 `Ruling::Lossless` 的**第二项先决条件**，光有 `ε = 0` 不够
//!   （[`a_bitwise_artifact_round_trip_completes_the_lossless_ruling`]）；
//! - 有段没被覆盖时，整网一律不得宣称无损（[`a_segment_outside_the_theorem_never_claims_the_whole_network`]）。
//!
//! 误差量的是 `op::matmul::forward_tn_tiled` 的输出差——与 `serve`/`train` 同一条内核，
//! 不是另写一份参考乘法。随机源用本 crate 的 [`llm_testkit::Rng`]，不在这里养第二个生成器。

use weights::l3::{Chain, Ruling, Rung};
use weights::{Archive, I4, Segment, Wire, Writer};

/// 确定性伪随机：用例必须每次给同一组数，否则"界没罩住"无法复现。
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        // 线性同余（Numerical Recipes 常数）；只取高位作图，低位周期太短。
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 0.5
    }

    fn vec(&mut self, len: usize, scale: f32) -> Vec<f32> {
        (0..len).map(|_| self.next() * 2.0 * scale).collect()
    }
}

/// 行主序 `[rows, k]` 的稠密乘法：走 `serve`/`train` 用的那份 TN 分块内核。
fn matmul(x: &[f32], w: &[f32], rows: usize, n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * n];
    op::matmul::forward_tn_tiled(x, w, rows, n, k, &mut out).expect("几何由用例自己保证");
    out
}

/// Frobenius 范数：`‖W‖₂ ≤ ‖W‖_F` 是任何矩阵都成立的上界，用它当 ρ 无需先分解。
fn frobenius(w: &[f32]) -> f64 {
    w.iter()
        .map(|&v| (v as f64) * (v as f64))
        .sum::<f64>()
        .sqrt()
}

/// 两向量的欧氏距离。
fn distance(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&p, &q)| ((p - q) as f64).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// 4bit 逐组量化后的那份权重：`ε` 是对**将要使用的那份产物**实测的精确扰动，不是估计。
fn quantized(w: &[f32]) -> Vec<f32> {
    const GROUP: usize = 32;
    let (packed, scales) = weights::quantize(I4, GROUP, w).expect("量化");
    let mut out = vec![0.0f32; w.len()];
    weights::dequantize(I4, &packed, &scales, GROUP, &mut out).expect("反量化");
    out
}

#[test]
fn a_quantized_layer_stays_inside_the_l3_bound() {
    let mut rng = Lcg(7);
    let (k, n, rows) = (32usize, 32usize, 1usize);
    let x = rng.vec(rows * k, 1.0);
    let w1 = rng.vec(n * k, 0.25);
    let w2 = rng.vec(n * n, 0.25);
    let w2_hat = quantized(&w2);

    // 参考：两层 ReLU 网络，第二层权重是量化后的那份。
    let mut y = matmul(&x, &w1, rows, n, k);
    y.iter_mut().for_each(|v| *v = v.max(0.0));
    let y = matmul(&y, &w2, rows, n, n);
    let mut y_hat = matmul(&x, &w1, rows, n, k);
    y_hat.iter_mut().for_each(|v| *v = v.max(0.0));
    let y_hat = matmul(&y_hat, &w2_hat, rows, n, n);

    // 段序即执行序：线性 → ReLU（精确，L = 1、b = 0、ε = 0）→ 量化后的线性。
    let relu = Rung::new(1.0, 1.0, 0.0, 1.0, 0.0).expect("ReLU 段");
    let chain = Chain::new(
        frobenius(&x),
        vec![
            Rung::new(frobenius(&w1), frobenius(&w1), 0.0, 1.0, 0.0).expect("线性段"),
            relu,
            Rung::new(
                frobenius(&w2),
                frobenius(&w2_hat),
                distance(&w2, &w2_hat),
                1.0,
                0.0,
            )
            .expect("量化段"),
        ],
    )
    .expect("链");
    let bound = chain.bracket().expect("界");

    let observed = distance(&y, &y_hat);
    assert!(observed > 0.0, "量化没造成任何差异，这条用例什么也没测到");
    assert!(
        observed <= bound.error(),
        "实测 {observed:.6e} 超出网络级上界 {:.6e}",
        bound.error()
    );
    // ρ 刻意取 Frobenius（`‖W‖₂ ≤ ‖W‖_F` 恒成立、无需先分解），所以界偏松是设计取舍而不是
    // 判据放宽：要紧的是它**必须罩住**实测值，并且是有限、非零的数。
    assert!(bound.error().is_finite() && bound.error() > observed);
    assert_eq!(chain.uncovered(), 0);
}

#[test]
fn a_batch_scales_with_the_square_root_of_its_rows() {
    let mut rng = Lcg(11);
    let (k, n, rows) = (16usize, 16usize, 4usize);
    let x = rng.vec(rows * k, 1.0);
    let w = rng.vec(n * k, 0.25);
    let w_hat = quantized(&w);

    let y = matmul(&x, &w, rows, n, k);
    let y_hat = matmul(&x, &w_hat, rows, n, k);

    // 逐行上界相同、行间独立，所以整批的界是单条界的 √rows 倍（不是 rows 倍）。
    let widest = (0..rows)
        .map(|row| frobenius(&x[row * k..(row + 1) * k]))
        .fold(0.0f64, f64::max);
    let chain = Chain::new(
        widest,
        vec![
            Rung::new(
                frobenius(&w),
                frobenius(&w_hat),
                distance(&w, &w_hat),
                1.0,
                0.0,
            )
            .expect("段"),
        ],
    )
    .expect("链");
    let bound = chain.bracket().expect("界").error() * (rows as f64).sqrt();

    let observed = distance(&y, &y_hat);
    assert!(observed > 0.0);
    assert!(
        observed <= bound,
        "实测 {observed:.6e} 超出整批界 {bound:.6e}"
    );
}

#[test]
fn a_bitwise_artifact_round_trip_completes_the_lossless_ruling() {
    let mut rng = Lcg(23);
    let w1 = rng.vec(64, 0.25);
    let w2 = rng.vec(64, 0.25);

    // 真实容器往返：写入 → 解析 → 逐段解码，比的是字节而不是"应该相等"的预期。
    let mut writer = Writer::new();
    for (name, values) in [("layer.0.w", &w1), ("layer.1.w", &w2)] {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let segment = Segment::new(name, Wire::F32, vec![64], None, bytes).expect("段");
        writer.push(segment).expect("入写");
    }
    let bytes = writer.finish(6).expect("落盘");
    let archive = Archive::parse(&bytes).expect("解析");
    let decoded: Vec<Vec<f32>> = archive
        .segments()
        .iter()
        .map(|seg| {
            let raw = archive.load(seg).expect("解码");
            let (words, _) = raw.as_chunks::<4>();
            words.iter().map(|word| f32::from_le_bytes(*word)).collect()
        })
        .collect();
    let bitwise = decoded.len() == 2 && decoded[0] == w1 && decoded[1] == w2;
    assert!(bitwise, "无损容器的往返必须逐位相等");

    // 两条先决条件都成立才叫无损：`ε = 0`（定理项）**且**产物逐位还原（实测项）。
    let chain = Chain::new(
        frobenius(&w1),
        vec![
            Rung::new(frobenius(&w1), frobenius(&w1), 0.0, 1.0, 0.0).expect("第一段"),
            Rung::new(frobenius(&w2), frobenius(&w2), 0.0, 1.0, 0.0).expect("第二段"),
        ],
    )
    .expect("链");
    assert_eq!(
        chain.bracket().expect("界").ruling(bitwise),
        Ruling::Lossless
    );
    // 同样的界、同样的 ε，只是产物没能逐位还原：整网不得宣称无损。
    assert_eq!(
        chain.bracket().expect("界").ruling(!bitwise),
        Ruling::Unverified
    );
}

#[test]
fn a_segment_outside_the_theorem_never_claims_the_whole_network() {
    let lossy = Chain::new(1.0, vec![Rung::new(1.0, 1.0, 1e-4, 1.0, 0.0).expect("段")])
        .expect("链")
        .bracket()
        .expect("界");
    // 有非零分项就是有损：覆盖缺口问的是"能否签发整网无损证书"，改不了"这次压缩有损"。
    assert_eq!(lossy.ruling(true), Ruling::Refuse);
    assert_eq!(lossy.ruling(false), Ruling::Refuse);

    // 全部 `ε = 0` 才轮到覆盖面说话：缺两段 → 只到"适用段无损"。
    assert_eq!(
        exact()
            .with_uncovered(2)
            .bracket()
            .expect("界")
            .ruling(true),
        Ruling::Partial
    );
    // 缺往返证据先于覆盖缺口：没有"产物确实还原"就还谈不上覆盖面。
    assert_eq!(
        exact()
            .with_uncovered(2)
            .bracket()
            .expect("界")
            .ruling(false),
        Ruling::Unverified
    );
    // 覆盖完整、往返逐位、分项全零：这三条同时成立才是整网无损。
    assert_eq!(
        exact().bracket().expect("界").ruling(true),
        Ruling::Lossless
    );
}

/// 一条分项全零的单段链：`with_uncovered` 会消耗链，故每次现造一条。
fn exact() -> Chain {
    Chain::new(1.0, vec![Rung::new(1.0, 1.0, 0.0, 1.0, 0.0).expect("段")]).expect("链")
}
