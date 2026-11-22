//! 职责：真机正确性判据——独立参考、可证误差界、边界形状、写满保证、布局前置。
//!
//! 与 `gemm.rs` / `tensor.rs` / `attention.rs` 的分工：那三份用例主要出吞吐，对拍用的是夹具的
//! [`matmul_values`] / [`attention_inputs`]，取值是 `0.25·k` 这类**可精确表示**的格点。取这样的值
//! 时每个乘积在 f32 里都是精确的，最佳累加顺序下误差恒为 `0.00e0`——**任何累加顺序错误都会被
//! 掩盖**，`误差 0.00e0` 并不是「算对了」的证据。本文件的输入一律用满尾数伪随机值，判据换成
//! 本身可证的误差界，并补上三处此前完全未验证的面：
//!
//! - 大形状（`M·N·K > 1024³` 时夹具不算参考，只打印 `误差 NaN`）；
//! - 张量核读的 `Bt`（B 的 half 转置）——布局前置本身从未被对拍过，它错则整批 GEMM 全错；
//! - 输出是否**被写满**、同一派发是否**可逐位复现**（漏写与竞争只在初值上露头）。

use super::*;
use op::model::{Buf, Sample, Values};

/// SplitMix64：确定性伪随机位模式。免依赖、跨平台一致，用例失败时可照序号复算。
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn bits(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// 满尾数 f32，落在 `[-0.5, 0.5)`：23 位尾数全随机，不落在任何 2 的幂格点上，
    /// 因而乘积必须真正舍入，累加顺序的错误会显现。
    fn f32(&mut self) -> f32 {
        let bits = (self.bits() >> 40) as u32;
        f32::from_bits(0x3f80_0000 | (bits & 0x007f_ffff)) - 1.5
    }

    /// 落在 `[0.25, 0.5)` 的满尾数 f32（随机符号）：half 的正规区间，避开次正规与溢出的舍入差异。
    fn half_normal(&mut self) -> f32 {
        let bits = (self.bits() >> 40) as u32;
        let magnitude = 0.25 + 0.25 * ((bits & 0x007f_ffff) as f32 / 8_388_608.0);
        if bits & 0x0080_0000 != 0 {
            -magnitude
        } else {
            magnitude
        }
    }

    fn values(&mut self, len: usize) -> Vec<f32> {
        (0..len).map(|_| self.f32()).collect()
    }
}

/// `γ_n = n·u / (1 − n·u)`，`u = 2^-24`（f32 就近取偶的单位舍入误差）。
///
/// 内积 `Σ a·b` 有 `k` 个乘积与 `k` 次加法，且每个乘积自身也要舍入一次，故取 `n = 2k`。
fn gamma(n: usize) -> f64 {
    let nu = n as f64 * 2f64.powi(-24);
    nu / (1.0 - nu)
}

/// 单个输出元素的 f64 参考与其前向误差界：返回 `(Σa·b, γ_{2k}·Σ|a·b|)`。
///
/// 参考在 f64 里独立重算，与设备的任何一条路径不共享一行算术（对照见算子侧 F6 自我对拍）。
fn reference_and_bound(a: &[f32], b: &[f32], n: usize, k: usize, i: usize, j: usize) -> (f64, f64) {
    let mut want = 0.0f64;
    let mut magnitude = 0.0f64;
    for p in 0..k {
        let x = f64::from(a[i * k + p]);
        let y = f64::from(b[p * n + j]);
        want += x * y;
        magnitude += x.abs() * y.abs();
    }
    (want, gamma(2 * k) * magnitude)
}

/// 逐元素按可证误差界判定：`|fl(Σa·b) − Σa·b| ≤ γ_{2k}·Σ|a·b|`。
///
/// 界随数据缩放——`Σ|a·b|` 大就放宽、接近零就收紧。这正是不需要手调常数的原因：
/// 定值阈值（如 `1e-3`）在真值量级跨十几个数量级时必然要么太松、要么误报。
fn assert_within_bound(
    label: &str,
    got: &[f32],
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
) {
    assert_eq!(got.len(), m * n, "{label} 的输出元素数应与 M·N 一致");
    let mut worst = (0.0f64, 0usize, 0usize);
    for i in 0..m {
        for j in 0..n {
            let (want, bound) = reference_and_bound(a, b, n, k, i, j);
            let error = (f64::from(got[i * n + j]) - want).abs();
            assert!(
                error <= bound,
                "{label} 的元素 ({i},{j})：|差|={error:.3e} 超出误差界 {bound:.3e}\
                 （f64 参考 {want:.6e}、设备 {:.6e}，Σ|a·b| 决定界宽）",
                got[i * n + j]
            );
            let used = if bound > 0.0 { error / bound } else { 0.0 };
            if used > worst.0 {
                worst = (used, i, j);
            }
        }
    }
    println!(
        "{label}：最差元素 ({},{}) 用掉误差界的 {:.0}%",
        worst.1,
        worst.2,
        worst.0 * 100.0
    );
}

/// 快照设备缓冲（内核已同步完成，共享内存可直接读）。
fn snapshot(buf: &mut MetalBuf) -> Vec<u8> {
    buf.as_mut_bytes().to_vec()
}

/// 把输出缓冲填成 NaN：内核若漏写任何一块，NaN 会原样留下，不会被「旧值是 0」蒙混过去。
fn fill_nan(buf: &mut MetalBuf) {
    for slot in buf.as_mut_bytes().chunks_exact_mut(4) {
        slot.copy_from_slice(&f32::NAN.to_ne_bytes());
    }
}

/// half 位模式还原成 f32（本文件只喂正常区间的值，不涉及次正规与溢出）。
fn half_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 1;
    let exponent = i32::from((bits >> 10) & 0x1f);
    let mantissa = u32::from(bits & 0x03ff);
    let magnitude = match exponent {
        0 => mantissa as f32 * 2f32.powi(-24),
        31 if mantissa == 0 => f32::INFINITY,
        31 => f32::NAN,
        _ => (1.0 + mantissa as f32 / 1024.0) * 2f32.powi(exponent - 15),
    };
    if sign == 1 { -magnitude } else { magnitude }
}

/// 显式收缩轴的稠密 GEMM 样本：`K` 不是 `(m, n)` 的函数，模型自带的 [`op::matmul::sample`]
/// 把它钉在生产形状 4096 上，用它生成 MSL 就等于让内核拿 4096 去算 96×80×96 的题。
fn gemm_sample(m: usize, n: usize, k: usize) -> op::model::Sample {
    op::matmul::sample_k(m, n, k)
}

/// 注意力输出的相对 Frobenius 误差 `‖got − want‖ / ‖want‖`。
///
/// 分母取整批范数：逐元素相对误差在真值接近 0 时会被灾难性抵消放大成 O(1)，
/// 那不是「算错了」而是「分母选错了」。
fn frobenius(got: &[f32], want: &[f64]) -> f64 {
    let mut numerator = 0.0f64;
    let mut denominator = 0.0f64;
    for (got, want) in got.iter().zip(want) {
        let difference = f64::from(*got) - want;
        numerator += difference * difference;
        denominator += want * want;
    }
    (numerator / denominator).sqrt()
}

/// 注意力 f64 参考：与内核同一数学（`softmax(QKᵀ/√D)·V`），最大项在 f64 里扣除。
fn attention_reference(q: &[f32], k: &[f32], v: &[f32], s: usize, d: usize) -> Vec<f64> {
    let scale = (d as f64).sqrt();
    let mut out = vec![0.0f64; s * d];
    let mut scores = vec![0.0f64; s];
    for i in 0..s {
        let mut row_max = f64::NEG_INFINITY;
        for j in 0..s {
            let mut dot = 0.0f64;
            for p in 0..d {
                dot += f64::from(q[i * d + p]) * f64::from(k[j * d + p]);
            }
            scores[j] = dot / scale;
            row_max = row_max.max(scores[j]);
        }
        let mut sum = 0.0f64;
        for score in scores.iter_mut() {
            *score = (*score - row_max).exp();
            sum += *score;
        }
        for h in 0..d {
            let mut acc = 0.0f64;
            for j in 0..s {
                acc += scores[j] * f64::from(v[j * d + h]);
            }
            out[i * d + h] = acc / sum;
        }
    }
    out
}

/// 上传一路 Q/K/V 并开一路输出缓冲：注意力四路缓冲的固定装配。
fn attention_bufs(
    device: &MetalDevice,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    len: usize,
) -> (MetalBuf, MetalBuf, MetalBuf, MetalBuf) {
    (
        upload(device, q, "Q"),
        upload(device, k, "K"),
        upload(device, v, "V"),
        output_buf(device, len, "Y"),
    )
}

mod attention;
mod bound;
mod exact;
mod layout;
mod scan;
mod spot;
