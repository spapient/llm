//! 块 scale 的紧凑编码：把"每块一个 `f32`"压缩为"基准 + 动态范围 + 每块一个对数定点字节"。
//!
//! 旧版 E8 每 8 维一个裸 `f32` scale，是端到端体积劣势的主因（每块 4 字节）。本模块取所有
//! scale 的最大值为基准 `base`，在对数域把每块相对基准的比值 `r = scale/base ∈ (0,1]` 量化到
//! 一个字节：动态范围 `D = -log2(min/base)`，码 `code = 255·(1 + log2(r)/D)`。
//!
//! 对数域（而非线性）的理由是相对误差在整个动态范围均匀，小值处不会像线性定点那样相对误差
//! 放大；`n` 块的 scale 区从 `4n` 字节降到 `8 + n` 字节（两个 f32 基准 + 每块一字节），块数
//! 越多越省。`log2`/`exp2` 用标准库浮点方法，零第三方依赖。是否启用由上层在端到端 MSE/体积
//! 对拍后决定，本模块不改变格内最近点算法。

use diagnostic::{Code, Error, Result};

/// 定点最大码：相对基准 1.0 对应 255。
const MAX_CODE: u8 = 255;

/// 基准区长度：base 与动态范围 D 两个 f32。
const HEAD_LEN: usize = 8;

/// 把一组正 scale 编码为紧凑字节：base(f32) + D(f32) + 每块一个 u8。
pub fn encode(scales: &[f32]) -> Result<Vec<u8>> {
    if scales.is_empty() {
        return Ok(Vec::new());
    }
    check_scales(scales)?;
    let base = scales.iter().copied().fold(f32::MIN, f32::max);
    let min = scales.iter().copied().fold(f32::MAX, f32::min);
    let range = (-(min / base).log2()).max(0.0);

    let mut out = Vec::with_capacity(HEAD_LEN + scales.len());
    out.extend_from_slice(&base.to_le_bytes());
    out.extend_from_slice(&range.to_le_bytes());
    out.extend(scales.iter().map(|scale| code(*scale, base, range)));
    Ok(out)
}

/// 每个 scale 必须是正有限值；这条判据不回答块数或布局。
fn check_scales(scales: &[f32]) -> Result<()> {
    for &scale in scales {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(
                Error::new(Code::Invalid, format!("scale 必须为正有限值，实际 {scale}"))
                    .with("scale::encode"),
            );
        }
    }
    Ok(())
}

/// 一个 scale 的定点码：相对基准 1.0 对应 255，动态范围为 0 时全码。
fn code(scale: f32, base: f32, range: f32) -> u8 {
    let raw = if range > 0.0 {
        255.0 * (1.0 + (scale / base).log2() / range)
    } else {
        255.0
    };
    (raw.round() as i64).clamp(1, 255) as u8
}

/// 紧凑字节 → 各块 scale；截断或字节不足显式报错。
pub fn decode(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.is_empty() {
        return Ok(Vec::new());
    }
    let (base, range) = head(bytes)?;
    let mut scales = Vec::with_capacity(bytes.len() - HEAD_LEN);
    for &code in &bytes[HEAD_LEN..] {
        if code == 0 {
            return Err(
                Error::new(Code::Invalid, "scale 定点码不能为 0（会产生零 scale）")
                    .with("scale::decode"),
            );
        }
        let t = range * (f32::from(code) / f32::from(MAX_CODE) - 1.0);
        scales.push(base * 2.0f32.powf(t));
    }
    Ok(scales)
}

/// 基准区：base 与动态范围 D，并校验两者合法。
fn head(bytes: &[u8]) -> Result<(f32, f32)> {
    if bytes.len() < HEAD_LEN {
        return Err(Error::new(
            Code::Invalid,
            format!("scale 区至少 {HEAD_LEN} 字节，实际 {}", bytes.len()),
        )
        .with("scale::decode"));
    }
    let base = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let range = f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    if !valid_head(base, range) {
        return Err(Error::new(Code::Invalid, "scale 基准/动态范围不合法").with("scale::decode"));
    }
    Ok((base, range))
}

/// base 必须正、有限；D 必须非负、有限。
fn valid_head(base: f32, range: f32) -> bool {
    base.is_finite() && base > 0.0 && range.is_finite() && range >= 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 跨越多个数量级的正 scale，专门检验对数域在小值处的相对误差是否均匀。
    fn wide_scales(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let exponent = -6.0 * i as f32 / n as f32;
                2.0f32.powf(exponent)
            })
            .collect()
    }

    #[test]
    fn compact_scales_round_trip_with_uniform_relative_error() {
        let raw = wide_scales(512);
        let bytes = encode(&raw).expect("编码");
        assert_eq!(bytes.len(), HEAD_LEN + raw.len());
        assert!(bytes.len() < raw.len() * 4, "必须比裸 f32 省");
        let back = decode(&bytes).expect("解码");
        assert_eq!(back.len(), raw.len());
        // 对数域：全范围相对误差均匀且有界（一个对数步长 2^(D/255)）。
        for (a, b) in raw.iter().zip(&back) {
            assert!((a - b).abs() / a <= 0.02, "相对误差 < 2%");
        }
    }

    #[test]
    fn empty_and_single_block_are_handled() {
        assert_eq!(encode(&[]).expect("空"), Vec::<u8>::new());
        let one = encode(&[0.5]).expect("单块");
        assert_eq!(one.len(), HEAD_LEN + 1);
        let back = decode(&one).expect("解码");
        assert!((back[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn equal_scales_have_zero_range() {
        let raw = vec![0.3f32; 64];
        let bytes = encode(&raw).expect("编码");
        let back = decode(&bytes).expect("解码");
        for value in back {
            assert!((value - 0.3).abs() < 1e-6);
        }
    }

    #[test]
    fn bad_values_and_truncation_are_errors() {
        assert!(encode(&[0.0]).is_err());
        assert!(encode(&[-1.0]).is_err());
        assert!(encode(&[f32::NAN]).is_err());
        assert!(decode(&[0u8, 1]).is_err());
        let mut bad = encode(&[0.5, 0.25]).expect("先编码");
        let last = bad.len() - 1;
        bad[last] = 0;
        assert!(decode(&bad).is_err());
    }

    /// 真实权重风格的数据：块间能量慢变（动态范围约 1.6 个数量级），块内随机。
    fn slow_data(n: usize) -> Vec<f32> {
        let mut s = 11u64;
        let nblocks = n / 8;
        let mut out = Vec::with_capacity(n);
        for b in 0..nblocks {
            let energy = 1.0 + 0.5 * (b as f32 * 0.01).sin();
            for _ in 0..8 {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let x = ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * energy;
                out.push(x);
            }
        }
        out
    }

    /// 端到端对拍：紧凑 scale 与裸 f32 scale 各自重建，MSE 基本持平，体积严格更小。
    #[test]
    fn end_to_end_mse_is_kept_while_size_shrinks() {
        use crate::lattice;
        let data = slow_data(8 * 1024);
        let bits = 4u8;
        let (packed, exact_scales) = lattice::quantize8(&data, bits).expect("格量化");

        let compact = encode(&exact_scales).expect("紧凑 scale");
        let compact_scales = decode(&compact).expect("解码 scale");

        let mut exact_back = vec![0.0f32; data.len()];
        let mut compact_back = vec![0.0f32; data.len()];
        lattice::dequantize8(&packed, &exact_scales, bits, &mut exact_back).expect("精确解码");
        lattice::dequantize8(&packed, &compact_scales, bits, &mut compact_back).expect("紧凑解码");

        let mse = |back: &[f32]| {
            back.iter()
                .zip(&data)
                .map(|(a, b)| f64::powi((a - b) as f64, 2))
                .sum::<f64>()
                / data.len() as f64
        };
        let exact_mse = mse(&exact_back);
        let compact_mse = mse(&compact_back);
        assert!(
            compact_mse <= exact_mse * 1.05,
            "exact={exact_mse} compact={compact_mse}"
        );
        assert!(compact.len() < exact_scales.len() * 4);
    }
}
