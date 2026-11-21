//! Dropout：训练期按 `p` 置零、按 `1/(1-p)` 缩放；推理期为恒等。
//!
//! 抽样走 [`crate::rnd::SplitMix64`]，同一种子必得同一掩码——这让"训练/推理两条路径
//! 的差异"可以被逐位对拍，而不是只能看统计。

use diagnostic::{Code, Error, Result};

use crate::rnd::SplitMix64;

/// 就地对 `x` 施加 dropout。
///
/// `train == false` 时原样返回（推理期不缩放）；训练期要求 `0 ≤ p < 1`。
///
/// # Errors
///
/// `p` 越界时返回说明。
pub fn dropout(x: &mut [f32], p: f32, seed: u64, train: bool) -> Result<()> {
    if !train || p == 0.0 {
        return Ok(());
    }
    if !(0.0..1.0).contains(&p) {
        return Err(Error::new(
            Code::Invalid,
            format!("dropout 概率 {p} 应在 [0,1)"),
        ));
    }
    apply_mask(x, p, seed);
    Ok(())
}

/// 用 [`SplitMix64`] 抽同一掩码：低于 `p` 置零，否则乘 `1/(1-p)` 保均值。
fn apply_mask(x: &mut [f32], p: f32, seed: u64) {
    let keep = 1.0 / (1.0 - p);
    let mut rng = SplitMix64::new(seed);
    for v in x.iter_mut() {
        if rng.next_f32() < p {
            *v = 0.0;
        } else {
            *v *= keep;
        }
    }
}

#[cfg(test)]
#[path = "../tests/dropout_tests.rs"]
mod tests;
