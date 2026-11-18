//! PTQ 量化 → 融合反量化推理的端到端对拍。
//!
//! `weights::ptq` 把 f32 权重按组量化成打包码 + 每组件标度（ADR 0018：换算算法留在 `weights`，
//! 离线算），`op::gemv_i2/i4/i8` 的融合内核在同一个 kernel 内解包 + 点积。两层各自在自己的坐标
//! 系里自洽，只有把真实字节串起来才能证明「量化」与「融合反量化」说的是同一件事——字节序、零点、
//! 标度分组三处任一漂移都会让本测试红掉。
//!
//! 断言只看方向（余弦，尺度不变）：PTQ 是有界舍入，不要求逐元素相等；i2 只有三档码，阈值逐级放宽。

use llm_testkit::{Rng, cosine_similarity};
use op::{gemv_f32, gemv_i2, gemv_i4, gemv_i8};
use weights::ptq::{self, I2, I4, I8};

/// 合成基准形状：输入宽、输出行数、量化分组。gemv 标度按行分、行内按组，分组须是输入宽的公因数。
const M: usize = 64;
const N: usize = 4;
const GROUP: usize = 32;

/// 合成基准：f32 权重 → 逐行 PTQ 打包 → 融合反量化内核 → logits，方向必须与 f32 参考一致。
#[test]
fn ptq_quantized_gemv_keeps_cosine_with_the_f32_reference() {
    let mut rng = Rng::new(2026_1118);
    let between = |rng: &mut Rng| (rng.below(2001) as f32 / 1000.0) - 1.0;
    let x: Vec<f32> = (0..M).map(|_| between(&mut rng)).collect();
    let weights: Vec<f32> = (0..N * M).map(|_| between(&mut rng)).collect();

    let mut reference = vec![0.0f32; N];
    gemv_f32::forward(&x, &weights, &mut reference).expect("稠密参考可算");

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
    run: impl FnOnce(&[f32], &[u8], &[f32], &mut [f32]) -> Result<(), E>,
    min_cosine: f32,
    x: &[f32],
    weights: &[f32],
    reference: &[f32],
) {
    let mut packed = Vec::with_capacity(N * M.div_ceil(scheme.cpb));
    let mut scales = Vec::with_capacity(N * M.div_ceil(GROUP));
    for row in 0..N {
        let (p, s) =
            ptq::quantize(scheme, GROUP, &weights[row * M..(row + 1) * M]).expect("行内 PTQ 可算");
        packed.extend(p);
        scales.extend(s);
    }

    let mut out = vec![0.0f32; N];
    run(x, &packed, &scales, &mut out).expect("融合内核可算");
    let cosine = cosine_similarity(reference, &out).expect("两条向量非零");
    assert!(
        cosine >= min_cosine,
        "{name} 位 PTQ 融合反量化 logits 余弦 {cosine:.5} < {min_cosine}"
    );
}
