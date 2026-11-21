//! `ptq` 的单元测试：量化、反解与自适应补偿。

use super::{
    I2, I3, I4, I8, adaptive_quantize, compensate, compensate_adaptive, dequantize, quantize,
};

/// 打包位序：i4 低 nibble 在前，`[7, -7]` → 码 `[15, 1]` → 字节 `0x1F`。
#[test]
fn i4_packs_low_nibble_first() {
    let (packed, scales) = quantize(I4, 32, &[7.0, -7.0]).expect("合法输入");
    assert_eq!(packed, vec![0x1F]);
    assert_eq!(scales, vec![1.0]);
    let mut out = [0.0; 2];
    dequantize(I4, &packed, &scales, 32, &mut out).expect("合法反解");
    assert_eq!(out, [7.0, -7.0]);
}

/// 打包位序：i2 每字节 4 码，`[1, -1, 0, 0]` → 码 `[3, 1, 2, 2]` → 字节 `0xA7`。
#[test]
fn i2_packs_four_codes_per_byte() {
    let (packed, scales) = quantize(I2, 32, &[1.0, -1.0, 0.0, 0.0]).expect("合法输入");
    assert_eq!(packed, vec![0xA7]);
    assert_eq!(scales, vec![1.0]);
    let mut out = [0.0; 4];
    dequantize(I2, &packed, &scales, 32, &mut out).expect("合法反解");
    assert_eq!(out, [1.0, -1.0, 0.0, 0.0]);
}

/// 打包位序：i3 每字节不整除，`[3, -3, 1, 0]` → 码 `[6, 0, 4, 3]` → 三个字节的位流。
#[test]
fn i3_packs_three_bits_per_code() {
    let (packed, scales) = quantize(I3, 32, &[3.0, -3.0, 1.0, 0.0]).expect("合法输入");
    assert_eq!(scales, vec![1.0]);
    let mut out = [0.0; 4];
    dequantize(I3, &packed, &scales, 32, &mut out).expect("合法反解");
    assert_eq!(out, [3.0, -3.0, 1.0, 0.0]);
    // 位宽不整除：4 码 × 3 bit = 12 bit → 2 字节。
    assert_eq!(packed.len(), 2);
    // 码 [6,0,4,3] 低 3 位在前：位置 0..2 码 6、3..5 码 0、6..8 码 4（跨字节）、9..11 码 3。
    assert_eq!(packed[0], 6, "码 6 在 0..2，码 4 的低 2 位 0 在 6..7");
    assert_eq!(
        packed[1],
        1 | (3 << 1),
        "码 4 的第 3 位在 byte1 位 0，码 3 在 1..3"
    );
}

/// 自适应分配按误差预算选位宽：极紧预算停在全 8bit 且不达标，极松预算一路降到 2bit。
#[test]
fn adaptive_quantize_assigns_bits_by_error_budget() {
    let x: Vec<f32> = (0..256).map(|i| ((i as f32).sin()) * 3.7).collect();
    let norm: f64 = x
        .iter()
        .map(|v| (*v as f64) * (*v as f64))
        .sum::<f64>()
        .sqrt();

    let tight = adaptive_quantize(32, &x, 1e-9).expect("合法");
    assert!(tight.widths.iter().all(|&w| w == 8));
    assert!(!tight.met, "预算低于 8bit 可达误差，必须如实报未达标");

    let loose = adaptive_quantize(32, &x, norm * 2.0).expect("合法");
    assert!(
        loose.widths.iter().all(|&w| w == 2),
        "松预算应降到最低档：{:?}",
        loose.widths
    );
    assert!(loose.met);
    assert!((loose.bpw - 2.0).abs() < 1e-9);
}

/// i8 逐元素二补码：`[127, -127]` → 字节 `[127, 129]`，反解归位。
#[test]
fn i8_stores_twos_complement() {
    let (packed, scales) = quantize(I8, 32, &[127.0, -127.0]).expect("合法输入");
    assert_eq!(packed, vec![127, 129]);
    assert_eq!(scales, vec![1.0]);
    let mut out = [0.0; 2];
    dequantize(I8, &packed, &scales, 32, &mut out).expect("合法反解");
    assert_eq!(out, [127.0, -127.0]);
}

/// 量化误差有界：每元素误差不超过本组半码距（`scale / 2`），且余弦相似度随位宽提升。
#[test]
fn round_trip_error_is_bounded() {
    let x: Vec<f32> = (0..256)
        .map(|i| ((i as f32).sin() * 3.7) + (i as f32 % 7.0) - 3.0)
        .collect();
    // i2 只有 {-1,0,1} 三档码，余弦阈值放低；i4/i8 随位宽逐级收紧。
    for (scheme, min_cosine) in [(I2, 0.80), (I4, 0.98), (I8, 0.999)] {
        let (packed, scales) = quantize(scheme, 32, &x).expect("合法输入");
        let mut out = vec![0.0; x.len()];
        dequantize(scheme, &packed, &scales, 32, &mut out).expect("合法反解");

        let mut cosine_num = 0.0f64;
        let mut x2 = 0.0f64;
        let mut o2 = 0.0f64;
        for (a, b) in x.iter().zip(&out) {
            cosine_num += (*a as f64) * (*b as f64);
            x2 += (*a as f64) * (*a as f64);
            o2 += (*b as f64) * (*b as f64);
        }
        let cosine = cosine_num / (x2.sqrt() * o2.sqrt());
        assert!(
            cosine > min_cosine,
            "scheme {} 余弦 {cosine} 低于阈值 {min_cosine}",
            scheme.bits
        );

        // 误差界用本组 scale：round 到最近码，|x - scale*signed| <= scale/2。
        for (i, (a, b)) in x.iter().zip(&out).enumerate() {
            let group_scale = scales[i / 32];
            let err = (a - b).abs();
            assert!(
                err <= group_scale / 2.0 + 1e-4,
                "scheme {} 第 {i} 项误差 {err} 超过半码距 {}",
                scheme.bits,
                group_scale / 2.0
            );
        }
    }
}

/// 全零组不产生零 scale：scale 取 1，码全为对称零点（i4 为 8），反解精确归零。
#[test]
fn all_zero_group_is_invertible() {
    let x = [0.0f32; 64];
    let (packed, scales) = quantize(I4, 32, &x).expect("合法输入");
    assert_eq!(scales, vec![1.0, 1.0]);
    assert!(packed.iter().all(|&b| b == 0x88), "码应为零点 8");
    let mut out = [0.0; 64];
    dequantize(I4, &packed, &scales, 32, &mut out).expect("合法反解");
    assert!(out.iter().all(|&v| v == 0.0));
}

/// 非法输入：零分组、非有限值、长度不匹配都显式报错，不静默。
#[test]
fn rejects_bad_inputs() {
    assert!(quantize(I4, 0, &[1.0]).is_err());
    assert!(quantize(I4, 32, &[f32::NAN]).is_err());
    assert!(quantize(I4, 32, &[f32::INFINITY]).is_err());

    let (packed, scales) = quantize(I4, 32, &[1.0, -1.0]).expect("合法输入");
    let mut out = [0.0; 3];
    assert!(dequantize(I4, &packed, &scales, 32, &mut out).is_err());
}

/// M4/E5 输出空间 SNR 扫描（`#[ignore]` 实测基准：
/// `cargo test -p llm-weights --lib -- output_space_snr_scan --ignored --nocapture`）。
///
/// 口径与 §10.10 M4/E5 卡一致：W4A16、g=32；SNR = 20·log10(‖Wx‖ / ‖Wx − Q(W)x‖)，
/// 跨层取 P50。本测试只出数字、不设 ≥44 dB 断言——44 dB 是门禁而非通过条件；
/// 当前 `quantize` 是无补偿的对称分组量化，必须如实报告它离门禁的距离，而不是伪造达标。
#[test]
#[ignore = "量化精度基准：cargo test -p llm-weights --lib -- output_space_snr_scan --ignored --nocapture"]
fn output_space_snr_scan() {
    let (rows, cols, g) = (2048usize, 2048usize, 32usize);
    let layers = 4usize;

    // 确定性 LCG + Box–Muller 高斯权重（无外部 RNG 依赖，可复现）。
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut next_f64 = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut next_gauss = move || {
        let u1 = next_f64().max(1e-12);
        let u2 = next_f64();
        ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
    };

    for (scheme, name) in [(I2, "i2"), (I4, "i4"), (I8, "i8")] {
        let mut snrs = Vec::with_capacity(layers);
        for _ in 0..layers {
            let w: Vec<f32> = (0..rows * cols).map(|_| next_gauss()).collect();
            let x: Vec<f32> = (0..cols).map(|_| next_gauss()).collect();
            let (packed, scales) = quantize(scheme, g, &w).expect("合法输入");
            let mut qw = vec![0.0f32; rows * cols];
            dequantize(scheme, &packed, &scales, g, &mut qw).expect("合法反解");
            let (num, den) = output_snr(&w, &qw, &x, rows, cols);
            snrs.push(20.0 * (num / den).log10());
        }
        snrs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = snrs[layers / 2];
        println!(
            "M4/E5 SNR[{name}] g={g} layers={layers} rows={rows} cols={cols} P50={p50:.3} dB 样本={:?}",
            snrs.iter().map(|v| format!("{v:.3}")).collect::<Vec<_>>()
        );
    }
}

/// `y = Wx` 与 `‖y − Q(W)x‖` 的 f64 累积：SNR = 20·log10(‖y‖ / ‖y − ŷ‖)。
fn output_snr(w: &[f32], qw: &[f32], x: &[f32], rows: usize, cols: usize) -> (f64, f64) {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for r in 0..rows {
        let (mut y, mut yq) = (0.0f64, 0.0f64);
        for c in 0..cols {
            let xv = x[c] as f64;
            y += w[r * cols + c] as f64 * xv;
            yq += qw[r * cols + c] as f64 * xv;
        }
        num += y * y;
        let d = y - yq;
        den += d * d;
    }
    (num.sqrt(), den.sqrt())
}

/// 确定性高斯矩阵（Box–Muller，无外部 RNG 依赖）。
fn gaussian(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    let mut next_f64 = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    (0..rows * cols)
        .map(|_| {
            let u1 = next_f64().max(1e-12);
            let u2 = next_f64();
            ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
        })
        .collect()
}

/// 结构近似低秩权重：`W = U·Vᵀ + σ_n·N`，`U∈m×r`、`V∈n×r`、`N` 高斯同形。
fn low_rank_plus_noise(m: usize, n: usize, r: usize, sigma_n: f32, seed: u64) -> Vec<f32> {
    let u = gaussian(m, r, seed);
    let v = gaussian(n, r, seed ^ 0x9e37_79b9);
    let mut w = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f64;
            for k in 0..r {
                s += u[i * r + k] as f64 * v[j * r + k] as f64;
            }
            w[i * n + j] = s as f32;
        }
    }
    let noise = gaussian(m, n, seed ^ 0xdead_beef);
    for (slot, &nval) in w.iter_mut().zip(&noise) {
        *slot += sigma_n * nval;
    }
    w
}

/// M4 谱隙证书自动拒压 + 秩-10 残差补偿（`#[ignore]`：
/// `cargo test -p llm-weights --lib -- m4_residual_compensation_snr --release --ignored --nocapture`）。
///
/// 谱隙证书（`svd::signal_rank`，相对阈值 0.2）自动选秩：结构近似低秩权重在秩 10 处有谱隙
/// → 补偿；满秩高斯无谱隙 → 拒压。σn∈{0.1,0.2,0.3} 扫描补偿 SNR 的 44 dB 分界。
#[test]
#[ignore = "M4 补偿基准：cargo test -p llm-weights --lib -- m4_residual_compensation_snr --release --ignored --nocapture"]
fn m4_residual_compensation_snr() {
    let (m, n, r) = (512usize, 512usize, 10usize);
    for sigma_n in [0.1f32, 0.2, 0.3] {
        let w = low_rank_plus_noise(m, n, r, sigma_n, 0xfeed_0001);
        match compensate(I4, 32, &w, m, n).expect("补偿") {
            Some(comp) => {
                let x = gaussian(n, 1, 0xace_0002);
                let (num, den) = output_snr(&w, &comp.recon, &x, m, n);
                let snr = 20.0 * (num / den).log10();
                let frob = frob_snr(&w, &comp.recon);
                println!(
                    "M4 结构化 σn={sigma_n} 补偿 秩={} 相对谱隙γ/σ_1={:.3} 输出SNR={snr:.3} dB FrobeniusSNR={frob:.3} dB",
                    comp.rank,
                    comp.gap / comp.sigma1,
                );
            }
            None => println!("M4 结构化 σn={sigma_n} 拒压（无谱隙）"),
        }
    }

    let (m, n) = (256usize, 256usize);
    let w = gaussian(m, n, 0xcafe_0003);
    match compensate(I4, 32, &w, m, n).expect("补偿") {
        Some(comp) => println!(
            "M4 满秩高斯 补偿 秩={} 相对谱隙γ/σ_1={:.3e}（不应发生）",
            comp.rank,
            comp.gap / comp.sigma1,
        ),
        None => println!("M4 满秩高斯 拒压（无谱隙，不冒充 4bit）"),
    }
}

/// E5 2–8bit 自适应分配（`#[ignore]`：
/// `cargo test -p llm-weights --lib -- e5_adaptive_allocation_snr --release --ignored --nocapture`）。
///
/// 谱隙选秩 + 残差 `{2,3,4,8}` 自适应量化，目标 44 dB：σn 越大残差越难压，分配器自动多给
/// 8bit 组，平均码位随 σn 上升——σn=0.3 也达到 ≥44 dB（代价是 bpw 从 4 升到 ~6）。
#[test]
#[ignore = "E5 自适应基准：cargo test -p llm-weights --lib -- e5_adaptive_allocation_snr --release --ignored --nocapture"]
fn e5_adaptive_allocation_snr() {
    let (m, n, r) = (512usize, 512usize, 10usize);
    for sigma_n in [0.1f32, 0.2, 0.3] {
        let w = low_rank_plus_noise(m, n, r, sigma_n, 0xfeed_0001);
        // 目标取 46 dB：输出空间 SNR 对随机 x 有 ~±1–2 dB 的投影波动，Frobenius 46 dB
        // 给 44 dB 门禁留出裕量，保证配对输出 SNR 的下界也 ≥44。
        match compensate_adaptive(32, &w, m, n, 46.0).expect("自适应补偿") {
            Some(comp) => {
                let mut snrs = Vec::new();
                for seed in [0xace_0005u64, 0xace_0006, 0xace_0007] {
                    let x = gaussian(n, 1, seed);
                    let (num, den) = output_snr(&w, &comp.recon, &x, m, n);
                    snrs.push(20.0 * (num / den).log10());
                }
                snrs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let frob = frob_snr(&w, &comp.recon);
                let (c2, c3, c4, c8) = width_histogram(&comp.widths);
                println!(
                    "E5 结构化 σn={sigma_n} 自适应 秩={} bpw={:.3} met={} 位宽[2:{c2} 3:{c3} 4:{c4} 8:{c8}] 输出SNR(P50={:.3},min={:.3}) dB FrobeniusSNR={frob:.3} dB",
                    comp.rank, comp.bpw, comp.met, snrs[1], snrs[0],
                );
            }
            None => println!("E5 结构化 σn={sigma_n} 拒压（无谱隙）"),
        }
    }
}

/// 位宽直方图：`(2bit 组数, 3bit 组数, 4bit 组数, 8bit 组数)`。
fn width_histogram(widths: &[u8]) -> (usize, usize, usize, usize) {
    let count = |w: u8| widths.iter().filter(|&&x| x == w).count();
    (count(2), count(3), count(4), count(8))
}

/// Frobenius 口径 SNR：`20·log10(‖W‖_F / ‖W − Ŵ‖_F)`。
fn frob_snr(w: &[f32], recon: &[f32]) -> f64 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (a, b) in w.iter().zip(recon) {
        let a = *a as f64;
        let d = a - *b as f64;
        num += a * a;
        den += d * d;
    }
    20.0 * (num.sqrt() / den.sqrt()).log10()
}
